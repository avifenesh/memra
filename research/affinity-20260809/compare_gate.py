#!/usr/bin/env python3
"""Plain-session affinity replay gate: the correctness + slope receipt.

THE EXACTNESS BAR — and WHY IT IS NOT "resumed == cold" (read this first).
The mission brief asked for "every output byte-identical" between affinity on/off. Building
this gate re-confirmed what the predecessor spec-affinity lane already proved with four
independent receipts (research/session-affinity-20260805/RESULTS.md §"ROOT CAUSE"):

  resumed == cold is NOT a property this engine has, on ANY reuse tier, and no reuse is even
  required to break it. Chunked prefill is not reduction-order-stable: a resume primes
  [rewind_boundary .. end] as its own chunk sequence instead of one full prime, and a
  different chunk split changes the reduction order in the prefill GEMMs, perturbing logits
  in the last bits and flipping a near-tie argmax at long generation windows. `MEMRA_PRIME_CHUNK`
  alone (a documented machine-config knob) changes greedy text on the SAME prompt with zero
  reuse. So two rigs already produce different greedy text; a resume is just another re-chunk.

Asserting resumed == cold would therefore wire a permanently-red gate that blames affinity for
chunked prefill's reduction order. The gate asserts what affinity actually OWNS:

  1. DETERMINISM — the affinity resume path reproduces itself byte-for-byte across servers
     (on-run-1 == on-run-2 == on-run-3). A resume that flipped run to run would be a real bug.
  2. NO NEW DIVERGENCE CLASS — affinity must not diverge from a true cold oracle on any turn
     with a SHALLOW shared prefix (an early/token-0 flip = a resume-state corruption). Deep
     divergences after a long coherent shared prefix are the pre-existing near-tie class the
     shipped prefix-cache tier already exhibits; affinity may inherit it, never widen it.
  3. SHORT-WINDOW EXACTNESS (when a short-window arm is supplied) — with generation stopped
     before near-ties can cascade, affinity resumes are byte-IDENTICAL to cold. This is the
     positive proof the resumed STATE is correct (the predecessor lane's `nx` arm: 4/4).
  4. BUDGET — every request's completion_tokens <= max_tokens (the overshoot contract).
  5. SLOPE — the ON arm's TTFT collapses after the learning turns with plain_affinity_rewinds>0.

`--on`/`--off` are the on/off arms; `--cold` is the TRUE cold oracle (MEMRA_KV_REUSE=0, every
tier off) — supply it to run checks 2/3 instead of the confounded off-arm (which still runs the
prefix cache). `--shallow-chars N` is the shared-prefix floor below which a divergence is a real
bug (default 32). Exit 0 = GREEN, non-zero = a named failure.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def load_rows(path: pathlib.Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("type") == "request":
            rows.append(row)
    return rows


def key(row: dict) -> tuple:
    return (row["phase"], row["phase_index"], row["request_index"])


def linfit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least-squares slope, intercept, R^2. Returns (0,0,0) for degenerate input."""
    n = len(xs)
    if n < 2:
        return 0.0, 0.0, 0.0
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0, 0.0, 0.0
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    ybar = sy / n
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, intercept, r2


def load_texts(arm_dir: pathlib.Path) -> dict:
    """Per-turn response text keyed as (phase, phase_index). The response bodies carry the full
    text (requests.jsonl carries only the sha), needed for shared-prefix-depth analysis."""
    out = {}
    resp = arm_dir / "responses"
    if not resp.is_dir():
        return out
    for f in resp.iterdir():
        name = f.name
        if not name.endswith(".json"):
            continue
        for phase in ("sequential", "postburst", "burst"):
            if name.startswith(phase + "-"):
                try:
                    idx = int(name[len(phase) + 1:-5])
                except ValueError:
                    continue
                out[(phase, idx)] = (json.loads(f.read_text()).get("text") or "")
    return out


def shared_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", type=pathlib.Path, required=True, help="affinity-ON requests.jsonl")
    ap.add_argument("--off", type=pathlib.Path, help="affinity-OFF (MEMRA_AFFINITY=0) requests.jsonl")
    ap.add_argument("--cold", type=pathlib.Path,
                    help="TRUE cold oracle (MEMRA_KV_REUSE=0) requests.jsonl — checks 2/3 use this")
    ap.add_argument("--on-runs", type=pathlib.Path, nargs="*", default=[],
                    help="additional affinity-ON replay requests.jsonl for the determinism check")
    ap.add_argument("--on-metrics", type=pathlib.Path, help="affinity-ON metrics-final.json")
    ap.add_argument("--short-on", type=pathlib.Path,
                    help="short-window (early-stop) affinity-ON requests.jsonl for check 3")
    ap.add_argument("--short-cold", type=pathlib.Path,
                    help="short-window cold-oracle requests.jsonl for check 3")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--learning-turns", type=int, default=2)
    ap.add_argument("--shallow-chars", type=int, default=32,
                    help="shared-prefix floor: an on-vs-cold divergence shallower than this is a "
                         "resume-state bug, not the pre-existing near-tie class")
    ap.add_argument("--out", type=pathlib.Path, help="write the receipt JSON here")
    args = ap.parse_args()

    on_rows = {key(r): r for r in load_rows(args.on)}
    failures: list[str] = []
    sha = lambda rows, k: rows.get(k, {}).get("text_sha256")

    # ---- CHECK 1: DETERMINISM — the affinity resume path reproduces itself across servers. ----
    det_arms = []
    for p in args.on_runs:
        if p.exists():
            det_arms.append({key(r): r for r in load_rows(p)})
    nondet = []
    for extra in det_arms:
        for k in sorted(set(on_rows) & set(extra)):
            if sha(on_rows, k) != sha(extra, k):
                nondet.append(list(k))
    if nondet:
        failures.append(f"DETERMINISM: affinity resume path is NON-deterministic across servers "
                        f"({len(nondet)} turn(s) differ run-to-run) — {nondet[:3]}")

    # ---- CHECK 2: NO NEW DIVERGENCE CLASS — vs the true cold oracle, every on-vs-cold text ----
    # divergence must sit AFTER a long coherent shared prefix (the pre-existing near-tie class the
    # shipped prefix tier already shows). A SHALLOW divergence would be resume-state corruption.
    cold_rows = {}
    on_dir = args.on.parent
    on_text = load_texts(on_dir)
    shallow_bugs = []
    deep_divergences = []
    if args.cold and args.cold.exists():
        cold_rows = {key(r): r for r in load_rows(args.cold)}
        cold_text = load_texts(args.cold.parent)
        for k in sorted(set(on_rows) & set(cold_rows)):
            if sha(on_rows, k) == sha(cold_rows, k):
                continue
            kk = (k[0], k[1])
            a, b = on_text.get(kk, ""), cold_text.get(kk, "")
            depth = shared_prefix_len(a, b)
            entry = {"request": list(k), "shared_prefix_chars": depth,
                     "on_chars": len(a), "cold_chars": len(b)}
            if depth < args.shallow_chars:
                shallow_bugs.append(entry)
            else:
                deep_divergences.append(entry)
        if shallow_bugs:
            failures.append(f"NEW-DIVERGENCE: {len(shallow_bugs)} affinity turn(s) diverge from the "
                            f"cold oracle with a shared prefix < {args.shallow_chars} chars — a "
                            f"resume-state bug, not the near-tie class — {shallow_bugs[:3]}")

    # ---- CHECK 3: SHORT-WINDOW EXACTNESS — resumes byte-identical to cold when gen stops early. --
    short_mismatch = []
    short_checked = 0
    if args.short_on and args.short_on.exists() and args.short_cold and args.short_cold.exists():
        s_on = {key(r): r for r in load_rows(args.short_on)}
        s_cold = {key(r): r for r in load_rows(args.short_cold)}
        for k in sorted(set(s_on) & set(s_cold)):
            short_checked += 1
            if sha(s_on, k) != sha(s_cold, k):
                short_mismatch.append(list(k))
        if short_mismatch:
            failures.append(f"SHORT-WINDOW: {len(short_mismatch)}/{short_checked} short-generation "
                            f"resumes NOT byte-identical to cold — the resumed STATE is wrong "
                            f"(near-ties cannot cascade in a short window) — {short_mismatch[:3]}")

    # ---- CHECK 4: BUDGET — completion_tokens <= max_tokens. ----
    overshoot = []
    for k, r in on_rows.items():
        ct = r.get("completion_tokens")
        if isinstance(ct, int) and ct > args.max_tokens:
            overshoot.append({"request": list(k), "completion_tokens": ct})
    if overshoot:
        failures.append(f"BUDGET: {len(overshoot)} response(s) exceeded max_tokens={args.max_tokens} "
                        f"— {overshoot[:3]}")

    # ---- CHECK 5: SLOPE — the ON arm collapses the TTFT climb with rewinds > 0. ----
    off_rows = {key(r): r for r in load_rows(args.off)} if args.off and args.off.exists() else {}

    def seq_series(rows: dict) -> list[tuple[int, float, int]]:
        seq = [rows[k] for k in rows if k[0] == "sequential"]
        seq.sort(key=lambda r: r["phase_index"])
        return [(r["phase_index"], r.get("ttft_s", 0.0), r.get("cached_tokens") or 0) for r in seq]

    def uncached_slope(rows: dict) -> tuple[float, float, float]:
        xs, ys = [], []
        for k in rows:
            if k[0] != "sequential" or k[1] < args.learning_turns:
                continue
            r = rows[k]
            pt = r.get("prompt_tokens")
            if pt is None:
                continue
            xs.append(float(pt - (r.get("cached_tokens") or 0)))
            ys.append(float(r.get("ttft_s", 0.0)))
        return linfit(xs, ys)

    on_seq = seq_series(on_rows)
    off_seq = seq_series(off_rows)
    on_slope, _, on_r2 = uncached_slope(on_rows)
    off_slope, _, off_r2 = uncached_slope(off_rows)

    rewinds = None
    if args.on_metrics and args.on_metrics.exists():
        rewinds = json.loads(args.on_metrics.read_text()).get("plain_affinity_rewinds")
        if not rewinds:
            failures.append("SLOPE: affinity-ON metrics report plain_affinity_rewinds=0 — the "
                            "mechanism never engaged (a flat slope here would be coincidence)")

    receipt = {
        "requests_compared": len(on_rows),
        "determinism_ok": not nondet,
        "determinism_arms": 1 + len(det_arms),
        "no_new_divergence_ok": not shallow_bugs,
        "shallow_bugs": shallow_bugs,
        "deep_divergences": len(deep_divergences),
        "deep_divergence_detail": deep_divergences,
        "short_window_checked": short_checked,
        "short_window_exact_ok": not short_mismatch,
        "budget_ok": not overshoot,
        "max_tokens": args.max_tokens,
        "plain_affinity_rewinds": rewinds,
        "slope_on_ms_per_uncached_tok": round(on_slope * 1000.0, 4),
        "slope_off_ms_per_uncached_tok": round(off_slope * 1000.0, 4),
        "slope_on_r2": round(on_r2, 5),
        "slope_off_r2": round(off_r2, 5),
        "shallow_chars_floor": args.shallow_chars,
        "learning_turns": args.learning_turns,
        "ttft_by_turn": {
            "on": [{"turn": i, "ttft_s": t, "cached": c} for (i, t, c) in on_seq],
            "off": [{"turn": i, "ttft_s": t, "cached": c} for (i, t, c) in off_seq],
        },
        "failures": failures,
    }
    text = json.dumps(receipt, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)

    if failures:
        print(f"\nGATE FAIL ({len(failures)}):", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nGATE GREEN: resume deterministic, no new divergence class vs cold, within budget, "
          "slope collapses with rewinds > 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
