#!/usr/bin/env python3
"""The frozen mixed hit/cold c=4 cell (memra#1233; from the memra#1223 lane's harness).

The sellgate shape of research/sellgate-20260812 and research/coldfix-20260812: 4,860-token
prompts, exactly 60 completion tokens, temperature 0, nine cache hits and one cold miss per ten
requests (mixed90), four closed-loop clients, 20 requests per repetition, against a hot set seeded
once per boot. The harness is the frozen sellgate replay itself (imported, not copied), so the
request shapes, seeding and per-request rows are the receipts' own.

usage: mixed-hitcold-cell.py --repo REPO --base URL --model NAME --reps N --out OUT.jsonl [--label L]

Writes every seed row, request row and cell row to OUT.jsonl, then one summary row: hit TTFT p50
and p95 pooled over the repetitions' hits (the coldfix figure), the per-repetition hit TTFT p95,
the output tok/s median, and the token check (every request exactly 60 tokens, finish length).
"""
import argparse
import importlib.util
import json
import statistics
import sys
import urllib.request
from pathlib import Path


def load(path):
    spec = importlib.util.spec_from_file_location("memra_sellgate_replay", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="arm")
    ap.add_argument("--timeout", type=float, default=900.0)
    a = ap.parse_args()
    h = load(a.repo / "research/sellgate-20260812/sellgate_replay.py")
    workload = h.load_workload(a.repo / "research/sellgate-20260812/workload.lock.json")
    base = a.base.rstrip("/")
    models = json.loads(urllib.request.urlopen(base + "/v1/models", timeout=30).read())
    ids = [m.get("id") for m in models.get("data", [])]
    if a.model not in ids:
        raise SystemExit(f"refusing: {base} serves {ids}, not {a.model}")
    grid = h.gdn_grid()
    ep = h.Endpoint(label=a.label, base=base, model=a.model)
    goldens = {}
    out = open(a.out, "a")
    seeds, seed_fail = h.seed_hot_set([ep], workload, f"hitcold-{a.label}", a.timeout, goldens, grid)
    for r in seeds:
        out.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, sort_keys=True) + "\n")
    hits, rep_p95, outs, bad = [], [], [], 0
    colds, hit_cached = [], {}
    for rep in range(1, a.reps + 1):
        reqs, samples, cells = h.run_cell([ep], workload, f"hitcold-{a.label}", "mixed90", rep, 4,
                                          a.timeout, goldens, grid)
        for r in [*samples, *reqs, *cells]:
            out.write(json.dumps(r, sort_keys=True) + "\n")
        rh = [float(r["ttft_ms"]) for r in reqs if r.get("cache_role") == "hit" and r.get("ttft_ms") is not None]
        hits += rh
        colds += [float(r["ttft_ms"]) for r in reqs if r.get("cache_role") == "miss" and r.get("ttft_ms") is not None]
        for r in reqs:
            if r.get("cache_role") == "hit":
                hit_cached[str(r.get("cached_tokens"))] = hit_cached.get(str(r.get("cached_tokens")), 0) + 1
        rep_p95.append(round(pct(rh, 0.95), 3) if rh else None)
        outs += [c.get("output_tok_s") for c in cells if c.get("output_tok_s")]
        bad += sum(1 for r in reqs if r.get("completion_tokens") != workload["completion_tokens"]
                   or r.get("finish_reason") != "length")
    summary = {"kind": "summary", "label": a.label, "reps": a.reps, "hits": len(hits),
               "hit_ttft_p50_ms": round(pct(hits, 0.5), 3) if hits else None,
               "hit_ttft_p95_ms": round(pct(hits, 0.95), 3) if hits else None,
               "hit_ttft_max_ms": round(max(hits), 3) if hits else None,
               "rep_hit_ttft_p95_ms": rep_p95,
               "output_tok_s_median": round(statistics.median(outs), 3) if outs else None,
               "requests_not_exact": bad, "seed_failures": seed_fail,
               "cold_ttft_p50_ms": round(pct(colds, 0.5), 3) if colds else None,
               "cold_ttft_p95_ms": round(pct(colds, 0.95), 3) if colds else None,
               "colds": len(colds), "hit_cached_tokens": hit_cached}
    try:
        m = json.loads(urllib.request.urlopen(base + "/metrics", timeout=30).read())
        summary["metrics"] = {k: m.get(k) for k in m if k.startswith("prefix_cache") or k in (
            "prefix_hits", "prefix_misses", "prefix_evictions", "prefix_inserts")}
    except Exception as e:  # noqa: BLE001
        summary["metrics_error"] = str(e)
    out.write(json.dumps(summary, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if bad == 0 and not seed_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
