#!/usr/bin/env python3
"""prefix-newest-turn-fits-gate.py: a growing conversation's newest turn fits the prefix cache.

The defect (memra#523 items 1 and 3): under the default policy (`MEMRA_PREFIX_CACHE_POLICY=slru`)
a newly published entry could be its own eviction victim, or the snapshot preflight could refuse
it outright, whenever the free share beside a PROMOTED cohort was smaller than the entry. A
growing long-context conversation then ran cold on every turn (`cached_tokens=0`, a full
re-prefill, the incident's 90-130 s ticks and 408s) while other tenants' entries sat protected,
and the only trace was one once-announced `snapshot skipped` line. `cached_tokens` decides the
customer's bill and whether the cache engaged at all, so the gate reads it from the response.

Serving shape, one card, one boot of the real `memra-server` per cell, plain path
(`MEMRA_SERVE_SPEC=0`), a SMALL explicit prefix budget, default policy:

  1. cohort: a second tenant (`cache_salt=cohort`) sends three prompts of different lengths, each
     twice; the second send is a whole-entry hit whose lease promotes the entry, so the cohort
     ends PROTECTED (asserted: `cached_tokens == prompt_tokens` on the second send, protected
     share not exceeded, so nothing is demoted).
  2. twin: the growing tenant (`cache_salt=grow`) replays an 8-turn conversation, turn k+1 =
     turn k's prompt ids plus a fixed number of new ids (a real conversation appends the answer
     and the next message; the cache mechanics are the same, and exact ids make
     `cached_tokens` a closed-form expectation).

  The pressure arithmetic is checked from the server's own lines, not assumed: the cohort's
  `insert probation` lines give a bytes(tokens) fit for this artifact, and the gate REFUSES
  (exit 2) unless cohort <= protected share, cohort + turn-1 entry > budget, and every turn's
  entry <= budget. That is exactly the incident's shape scaled to a small budget.

Assertions (bytes from the server's `[prefix-cache]` lines and `/metrics`):
  V1 cached:      every turn k >= 2 reports `usage.prompt_tokens_details.cached_tokens` >= turn
                  k-1's `prompt_tokens` (the previous turn's entry was restored).
  V2 lines:       turn 1 publishes (`insert probation`, no `insert refused`, no
                  `snapshot skipped`); every turn k >= 2 has exactly one `[prefix-cache] hit:`
                  line of at least turn k-1's tokens AND publishes its own entry; no turn of the
                  growing tenant is refused or skipped.
  V3 effective:   for every turn whose window carries an `evict` line, the bytes the cache grew
                  by (`prefix_cache_bytes` after minus before) equal the effective free
                  (`cuda_driver_free_bytes + cuda_pool_cached_bytes`) the turn consumed, within
                  a slack: an eviction that did not return its bytes to effective free would show
                  the turn consuming the whole inserted entry. This is #523 item 4's honesty
                  (the day-13 gate's V3) under the capacity-eviction shape.
  V4 protected:   at least one `evict (... Protected LRU)` line in the growing tenant's turns:
                  the room came from the protected cohort, never from the entry itself.
  Identity is recorded per turn (sha256 of each completion text) for the runner to compare
  across binaries; it is not a verdict inside one cell (restored-vs-cold identity has its
  own gates).

Exit 0 = every assertion held; 1 = a verdict failed (red on `main` today: `cached_tokens=0` on
every turn after the once-announced `snapshot skipped` line); 2 = REFUSED (lock, port, shape).

usage: prefix-newest-turn-fits-gate.py [--external-lock FD] --model GGUF --bin memra-server \
           --out NEW_DIR [--port N] [--budget-mib 1024] [--cohort-tokens 2800,3000,3200] \
           [--turns 8] [--start-tokens 9200] [--grow-tokens 300]
Lock: the canonical rig lock only (`/tmp/memra-gpu.lock` or `/tmp/memra-5090.lock`), held for
the whole cell; `--external-lock FD` inherits the collector's FD (lead ruling 5) and is verified
with tools/tier-lock-proof.py before anything binds a port or boots a server.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCKS = ("/tmp/memra-gpu.lock", "/tmp/memra-5090.lock")
MIB = 1 << 20
V3_SLACK = 64 * MIB

RE_ON = re.compile(r"\[prefix-cache\] on: budget \d+MB \((\d+) B,.*policy ([^,]+),")
RE_INSERT = re.compile(r"\[prefix-cache\] insert probation \(([^)]+)\): (\d+) tokens, ([\d.]+)MB")
RE_HIT = re.compile(r"\[prefix-cache\] hit: (\d+) of (\d+) prompt tokens from cache")
# fix: `evict (Protected LRU)`, `evict (snapshot preflight, Protected LRU)`; main: `evict (snapshot preflight)`
RE_EVICT = re.compile(
    r"\[prefix-cache\] evict \((?:snapshot preflight(?:, )?)?(Probation|Protected)? ?(?:LRU)?\): (\d+) tokens, ([\d.]+)MB"
)
RE_REFUSED = re.compile(r"\[prefix-cache\] insert refused: (.*)$")
RE_SKIPPED = re.compile(r"\[prefix-cache\] snapshot skipped: (.*)$")
RE_DEMOTE = re.compile(r"\[prefix-cache\] demote \(protected bytes\): ([\d.]+)MB")
RE_RECLAIM = re.compile(r"\[admit-oom\] reclaim-on-defer: ")
# The per-request route receipt (`[glm5-spec] route=plain ... cold=1 restored=0 reason=...`) is
# printed by draft-capable models on every request; it is recorded per turn so a cold turn can
# be quoted from the receipt, never a verdict input (V1 reads `cached_tokens` from the response).
RE_ROUTE = re.compile(r"\[glm5-spec\] route=\S+ .*cold=(\d) restored=(\d)")


def refuse(msg: str) -> None:
    print(f"REFUSED: {msg}", flush=True)
    sys.exit(2)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Server:
    def __init__(self, binary: str, model: str, port: int, log: Path, budget_mib: int):
        self.binary, self.model, self.port, self.log = binary, model, port, log
        self.budget_mib = budget_mib
        self.proc: subprocess.Popen | None = None
        self.base = f"http://127.0.0.1:{port}"
        self.offset = 0

    def boot(self) -> None:
        if self.probe():
            refuse(f"port {self.port} already serving; refusing to boot over it")
        env = dict(os.environ)
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                "MEMRA_COMPAT": "openai",
                "MEMRA_MODELS": f"gate={self.model}",
                "MEMRA_ADDR": f"127.0.0.1:{self.port}",
                "MEMRA_CTX": "16384",
                "MEMRA_MAX_SESSIONS": "4",
                "MEMRA_PREFIX_CACHE_MB": str(self.budget_mib),
                "MEMRA_SERVE_SPEC": "0",
                "MEMRA_TIMEOUT_MS_MAX": "240000",
            }
        )
        # The subject is the DEFAULT policy: never pre-set it, never inherit a launcher's value.
        env.pop("MEMRA_PREFIX_CACHE_POLICY", None)
        env.pop("MEMRA_PREFIX_CACHE_PROTECTED_PCT", None)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("wb") as log:
            # Same process group as the gate: the collector's timeout reaps everything.
            self.proc = subprocess.Popen([self.binary], stdout=log, stderr=subprocess.STDOUT, env=env)
        for _ in range(300):
            if self.probe():
                return
            if self.proc.poll() is not None:
                refuse(f"server died during boot (exit {self.proc.returncode}); see {self.log}")
            time.sleep(2)
        self.stop()
        refuse("server never became ready")

    def probe(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base}/v1/models", timeout=2):
                return True
        except Exception:
            return False

    def stop(self) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.proc = None

    def metrics(self) -> dict:
        with urllib.request.urlopen(f"{self.base}/metrics", timeout=10) as f:
            return json.load(f)

    def settled_metrics(self) -> dict | None:
        """/metrics after the last retire has published and two reads agree (bounded wait)."""
        last = None
        for _ in range(40):
            try:
                m = self.metrics()
            except Exception:
                time.sleep(0.25)
                continue
            row = mem_row(m)
            if row["active_sessions"] == 0 and row["queued_requests"] == 0 and row == last:
                return row
            last = row
            time.sleep(0.25)
        return last

    def new_log_lines(self) -> list[str]:
        data = self.log.read_bytes()
        chunk = data[self.offset :]
        self.offset = len(data)
        return chunk.decode(errors="replace").splitlines()

    def complete(self, ids: list[int], salt: str, max_tokens: int = 8) -> dict:
        body = {
            "model": "gate",
            "prompt_ids": ids,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
            "cache_salt": salt,
            "timeout_ms": 240000,
        }
        req = urllib.request.Request(
            f"{self.base}/v1/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=300) as f:
                status, payload = f.status, json.load(f)
        except urllib.error.HTTPError as e:
            status, raw = e.code, e.read().decode(errors="replace")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"raw": raw}
        elapsed = time.monotonic() - t0
        text_out, finish = "", None
        if status == 200:
            choice = payload["choices"][0]
            text_out = choice.get("text") if "text" in choice else json.dumps(choice.get("message"), sort_keys=True)
            finish = choice.get("finish_reason")
        # compat mode carries the OpenAI usage object; the native shape carries the same
        # worker truth as flat prompt_tokens/cached_tokens fields.
        usage = payload.get("usage")
        if usage is None and status == 200:
            usage = {
                "prompt_tokens": payload.get("prompt_tokens"),
                "prompt_tokens_details": {"cached_tokens": payload.get("cached_tokens")},
            }
        usage = usage or {}
        return {
            "status": status,
            "elapsed_s": round(elapsed, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "finish_reason": finish,
            "text_sha256": sha(text_out or ""),
            "error": None if status == 200 else payload,
        }


def mem_row(m: dict) -> dict:
    return {
        k: int(m.get(k, 0))
        for k in (
            "cuda_driver_free_bytes",
            "cuda_pool_reserved_bytes",
            "cuda_pool_used_bytes",
            "cuda_pool_cached_bytes",
            "prefix_cache_bytes",
            "prefix_cache_entries",
            "prefix_cache_evictions",
            "prefix_cache_skips_budget",
            "prefix_cache_skips_pinned",
            "active_sessions",
            "queued_requests",
        )
    }


def effective_free(row: dict) -> int:
    return row["cuda_driver_free_bytes"] + row["cuda_pool_cached_bytes"]


def parse_window(lines: list[str]) -> dict:
    ev: dict = {"inserts": [], "hits": [], "evicts": [], "refused": [], "skipped": [], "demotes": 0, "reclaims": 0, "lines": [], "route": []}
    for ln in lines:
        if "[prefix-cache]" in ln or "[admit-oom]" in ln or "[glm5-spec] route=" in ln:
            ev["lines"].append(ln.strip())
        m = RE_ROUTE.search(ln)
        if m:
            ev["route"].append({"cold": int(m.group(1)), "restored": int(m.group(2)), "line": ln.strip()})
            continue
        m = RE_INSERT.search(ln)
        if m:
            ev["inserts"].append({"why": m.group(1), "tokens": int(m.group(2)), "mb": float(m.group(3))})
            continue
        m = RE_HIT.search(ln)
        if m:
            ev["hits"].append({"hit": int(m.group(1)), "prompt": int(m.group(2))})
            continue
        m = RE_EVICT.search(ln)
        if m:
            ev["evicts"].append({"segment": m.group(1) or "unstated", "tokens": int(m.group(2)), "mb": float(m.group(3))})
            continue
        m = RE_REFUSED.search(ln)
        if m:
            ev["refused"].append(ln.strip())
            continue
        m = RE_SKIPPED.search(ln)
        if m:
            ev["skipped"].append(ln.strip())
            continue
        if RE_DEMOTE.search(ln):
            ev["demotes"] += 1
        if RE_RECLAIM.search(ln):
            ev["reclaims"] += 1
    return ev


def fit_bytes_per_token(points: list[tuple[int, float]]) -> tuple[float, float]:
    """Least squares bytes = fixed + per_token * tokens over (tokens, bytes) points."""
    n = len(points)
    sx = sum(t for t, _ in points)
    sy = sum(b for _, b in points)
    sxx = sum(t * t for t, _ in points)
    sxy = sum(t * b for t, b in points)
    den = n * sxx - sx * sx
    if den == 0:
        return 0.0, sy / n
    per_token = (n * sxy - sx * sy) / den
    fixed = (sy - per_token * sx) / n
    return per_token, fixed


def ids_for(tokens: int) -> list[int]:
    # Stable, well inside the vocab, non-repeating over the longest prompt used here.
    return [3000 + (i % 7000) for i in range(tokens)]


def main() -> None:
    argv = sys.argv[1:]
    lock_fd = None
    lock_owner = "internal-canonical"
    if argv[:1] == ["--external-lock"]:
        if len(argv) < 2 or not argv[1].isdigit():
            refuse("inherited lock FD required after --external-lock")
        lock_fd, lock_owner = int(argv[1]), "collector"
        argv = argv[2:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--port", type=int, default=18114)
    ap.add_argument("--budget-mib", type=int, default=1024)
    ap.add_argument("--cohort-tokens", default="2800,3000,3200")
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--start-tokens", type=int, default=9200)
    ap.add_argument("--grow-tokens", type=int, default=300)
    ap.add_argument("--gpu-lock", default=os.environ.get("MEMRA_GPU_LOCK", "/tmp/memra-gpu.lock"))
    args = ap.parse_args(argv)

    if args.gpu_lock not in LOCKS:
        refuse("noncanonical GPU lock")
    if lock_owner == "internal-canonical":
        fh = open(args.gpu_lock, "a")  # noqa: SIM115 (held for the process lifetime)
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            refuse("canonical GPU lock busy")
        lock_fd = fh.fileno()
    proof = subprocess.run(
        [sys.executable, str(HERE / "tier-lock-proof.py"), "--fd", str(lock_fd), "--lock", args.gpu_lock, "--owner", lock_owner],
        capture_output=True, text=True, pass_fds=(lock_fd,),
    )
    if proof.returncode != 0:
        refuse(f"lock proof failed: {proof.stdout.strip()} {proof.stderr.strip()}")
    if not Path(args.model).is_file():
        refuse(f"model missing: {args.model}")
    if not os.access(args.bin, os.X_OK):
        refuse(f"server binary missing or not executable: {args.bin}")
    if args.out.exists():
        refuse(f"--out must be a new directory: {args.out}")
    cohort_tokens = [int(t) for t in args.cohort_tokens.split(",") if t]
    if len(cohort_tokens) < 2 or len(set(cohort_tokens)) != len(cohort_tokens):
        refuse("--cohort-tokens needs at least two distinct lengths (the bytes(tokens) fit)")
    if args.turns < 2:
        refuse("--turns must be at least 2")
    last_tokens = args.start_tokens + (args.turns - 1) * args.grow_tokens
    if last_tokens + 8 + 64 >= 16384:
        refuse(f"the last turn ({last_tokens} tokens) does not fit MEMRA_CTX=16384")
    args.out.mkdir(parents=True)
    (args.out / "LOCK.json").write_text(proof.stdout)
    rig = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,power.limit,power.max_limit,driver_version",
         "--format=csv,noheader"], capture_output=True, text=True,
    ).stdout.strip()
    binsha = hashlib.sha256(Path(args.bin).read_bytes()).hexdigest()
    (args.out / "rig.json").write_text(json.dumps({"nvidia_smi": rig, "binary_sha256": binsha, "binary": args.bin, "model": args.model}, indent=2) + "\n")

    srv = Server(args.bin, args.model, args.port, args.out / "server.log", args.budget_mib)
    srv.boot()
    rec: dict = {"boot": {}, "cohort": [], "turns": []}
    try:
        boot_lines = srv.new_log_lines()
        on = next((RE_ON.search(ln) for ln in boot_lines if RE_ON.search(ln)), None)
        if on is None:
            refuse("no `[prefix-cache] on:` boot line; the cache did not arm")
        budget = int(on.group(1))
        policy = on.group(2).strip()
        rec["boot"] = {"budget_bytes": budget, "policy": policy, "line": next(ln.strip() for ln in boot_lines if RE_ON.search(ln))}
        if "SLRU" not in policy.upper():
            refuse(f"the boot did not report the SLRU default (policy {policy!r}); the gate's subject is the default policy")
        if budget != args.budget_mib * MIB:
            refuse(f"budget {budget} B differs from the requested {args.budget_mib} MiB")
        protected_share = budget * 80 // 100

        # ---- 1. the protected cohort -----------------------------------------------------
        points: list[tuple[int, float]] = []
        for n in cohort_tokens:
            ids = ids_for(n)
            ids = [i + 60000 for i in ids]  # a distinct id range: the cohort shares no prefix with the twin
            first = srv.complete(ids, "cohort")
            time.sleep(0.5)
            w1 = parse_window(srv.new_log_lines())
            second = srv.complete(ids, "cohort")
            time.sleep(0.5)
            w2 = parse_window(srv.new_log_lines())
            row = srv.settled_metrics()
            rec["cohort"].append({"tokens": n, "first": first, "second": second, "window_first": w1, "window_second": w2, "metrics_after": row})
            if first["status"] != 200 or second["status"] != 200:
                refuse(f"cohort prompt of {n} tokens was not served: {first['error'] or second['error']}")
            ins = [i for i in w1["inserts"] if i["tokens"] == first["prompt_tokens"]]
            if not ins:
                refuse(f"cohort prompt of {n} tokens published no entry (window: {w1['lines']})")
            points.append((ins[0]["tokens"], ins[0]["mb"] * 1e6))
            if second["cached_tokens"] != second["prompt_tokens"] or not w2["hits"]:
                refuse(f"cohort promotion did not happen for {n} tokens: second send cached={second['cached_tokens']} of {second['prompt_tokens']}")
        cohort_row = rec["cohort"][-1]["metrics_after"]
        if cohort_row is None:
            refuse("/metrics never settled after the cohort")
        cohort_bytes = cohort_row["prefix_cache_bytes"]
        if cohort_row["prefix_cache_entries"] != len(cohort_tokens) or cohort_row["prefix_cache_evictions"] != 0:
            refuse(f"the cohort is not intact: {cohort_row}")
        if any(w["demotes"] for c in rec["cohort"] for w in (c["window_first"], c["window_second"])):
            refuse("a cohort entry was demoted during seeding; the cohort exceeds the protected share")
        per_token, fixed = fit_bytes_per_token(points)
        est = lambda tokens: fixed + per_token * tokens  # noqa: E731
        e1 = est(args.start_tokens)
        e_last = est(last_tokens)
        shape = {
            "budget_bytes": budget,
            "protected_share_bytes": protected_share,
            "cohort_bytes": cohort_bytes,
            "cohort_points": points,
            "fit_bytes_per_token": per_token,
            "fit_fixed_bytes": fixed,
            "turn1_entry_estimate_bytes": e1,
            "last_turn_entry_estimate_bytes": e_last,
            "cohort_within_protected_share": cohort_bytes <= protected_share,
            "turn1_exceeds_free_share": cohort_bytes + e1 > budget,
            "every_turn_fits_budget": e_last <= budget,
            "two_turns_fit_budget": est(last_tokens - args.grow_tokens) + e_last <= budget,
        }
        (args.out / "shape.json").write_text(json.dumps(shape, indent=2) + "\n")
        if not shape["cohort_within_protected_share"]:
            refuse(f"cohort {cohort_bytes} B exceeds the protected share {protected_share} B: the cohort would be demoted, not the incident's shape")
        if not shape["turn1_exceeds_free_share"]:
            refuse(f"no pressure at turn 1: cohort {cohort_bytes} + turn-1 entry {e1:.0f} <= budget {budget}; raise --start-tokens or lower --budget-mib")
        if not shape["every_turn_fits_budget"]:
            refuse(f"the last turn's entry {e_last:.0f} B exceeds the budget {budget} B; every turn must be insertable")

        # ---- 2. the 8-turn twin ------------------------------------------------------------
        ids = ids_for(args.start_tokens)
        prev_prompt_tokens = None
        for k in range(1, args.turns + 1):
            if k > 1:
                # The appended ids stand for the previous answer plus the next message.
                ids = ids + [20000 + ((k * 1000 + j) % 9000) for j in range(args.grow_tokens)]
            before = srv.settled_metrics()
            r = srv.complete(ids, "grow")
            time.sleep(0.5)
            window = parse_window(srv.new_log_lines())
            after = srv.settled_metrics()
            if before is None or after is None:
                refuse(f"turn {k}: /metrics never settled")
            eff_before, eff_after = effective_free(before), effective_free(after)
            grew = after["prefix_cache_bytes"] - before["prefix_cache_bytes"]
            consumed = eff_before - eff_after
            turn = {
                "turn": k,
                "prompt_ids": len(ids),
                "status": r["status"],
                "prompt_tokens": r["prompt_tokens"],
                "cached_tokens": r["cached_tokens"],
                "prev_prompt_tokens": prev_prompt_tokens,
                "elapsed_s": r["elapsed_s"],
                "finish_reason": r["finish_reason"],
                "text_sha256": r["text_sha256"],
                "error": r["error"],
                "window": window,
                "metrics_before": before,
                "metrics_after": after,
                "effective_free_before": eff_before,
                "effective_free_after": eff_after,
                "effective_free_consumed": consumed,
                "prefix_bytes_grew": grew,
                "inserted_bytes_from_line": int(sum(i["mb"] for i in window["inserts"]) * 1e6),
                "evicted_bytes_from_lines": int(sum(e["mb"] for e in window["evicts"]) * 1e6),
                "v3_credit_error_bytes": (consumed - grew) if window["evicts"] else None,
            }
            rec["turns"].append(turn)
            if r["status"] != 200:
                refuse(f"turn {k} was not served (HTTP {r['status']}): {r['error']}")
            prev_prompt_tokens = r["prompt_tokens"]
        rec["final_metrics"] = srv.settled_metrics()
    finally:
        srv.stop()

    # ---- verdicts -------------------------------------------------------------------------
    turns = rec["turns"]
    v1_rows = [t["cached_tokens"] is not None and t["prev_prompt_tokens"] is not None and t["cached_tokens"] >= t["prev_prompt_tokens"] for t in turns[1:]]
    v1 = all(v1_rows)
    cold_after_1 = sum(1 for t in turns[1:] if (t["cached_tokens"] or 0) == 0)

    def publishes(t: dict) -> bool:
        return any(i["tokens"] == t["prompt_tokens"] for i in t["window"]["inserts"])

    v2_rows = []
    for t in turns:
        w = t["window"]
        clean = not w["refused"] and not w["skipped"]
        if t["turn"] == 1:
            v2_rows.append(clean and publishes(t))
        else:
            hit_ok = len(w["hits"]) == 1 and w["hits"][0]["hit"] >= (t["prev_prompt_tokens"] or 0)
            v2_rows.append(clean and hit_ok and publishes(t))
    v2 = all(v2_rows)
    v3_rows = [abs(t["v3_credit_error_bytes"]) <= V3_SLACK for t in turns if t["v3_credit_error_bytes"] is not None]
    v3 = bool(v3_rows) and all(v3_rows)
    protected_evictions = sum(1 for t in turns for e in t["window"]["evicts"] if e["segment"] == "Protected")
    total_evictions = sum(len(t["window"]["evicts"]) for t in turns)
    refused_lines = [ln for t in turns for ln in t["window"]["refused"] + t["window"]["skipped"]]
    v4 = protected_evictions >= 1
    ok = v1 and v2 and v3 and v4
    verdict = (
        f"PREFIX-NEWEST-TURN-FITS: budget_bytes={rec['boot']['budget_bytes']} cohort_bytes={shape['cohort_bytes']} "
        f"turns={len(turns)} cold_turns_after_1={cold_after_1} cached_ok={sum(v1_rows)}/{len(v1_rows)} "
        f"lines_ok={sum(v2_rows)}/{len(v2_rows)} evictions={total_evictions} protected_evictions={protected_evictions} "
        f"refused_or_skipped={len(refused_lines)} effective_free_ok={sum(v3_rows)}/{len(v3_rows)} "
        f"V1={'ok' if v1 else 'FAIL'} V2={'ok' if v2 else 'FAIL'} V3={'ok' if v3 else 'FAIL'} V4={'ok' if v4 else 'FAIL'} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    table = ["| turn | prompt_tokens | cached_tokens | prev prompt_tokens | route cold/restored | hit line | insert | evict (segment) | refused/skipped | effective free consumed | prefix bytes grew | V3 error | elapsed s | text sha256[:16] |",
             "| ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for t in turns:
        w = t["window"]
        hits = ", ".join("{hit} of {prompt}".format(**h) for h in w["hits"]) or "none"
        inserts = ", ".join("{tokens} tok {mb}MB ({why})".format(**i) for i in w["inserts"]) or "none"
        evicts = ", ".join("{tokens} tok {mb}MB ({segment})".format(**e) for e in w["evicts"]) or "none"
        prev = t["prev_prompt_tokens"] if t["prev_prompt_tokens"] is not None else "-"
        v3err = t["v3_credit_error_bytes"] if t["v3_credit_error_bytes"] is not None else "-"
        route = ", ".join("cold={cold} restored={restored}".format(**r) for r in w["route"]) or "none"
        table.append(
            f"| {t['turn']} | {t['prompt_tokens']} | {t['cached_tokens']} | {prev} | {route} | {hits} | {inserts} | {evicts} | "
            f"{len(w['refused']) + len(w['skipped'])} | {t['effective_free_consumed']} | {t['prefix_bytes_grew']} | "
            f"{v3err} | {t['elapsed_s']} | {t['text_sha256'][:16]} |"
        )
    summary = {
        "verdict": verdict,
        "pass": ok,
        "binary_sha256": binsha,
        "rig": rig,
        "boot": rec["boot"],
        "shape": shape,
        "cohort": rec["cohort"],
        "turns": turns,
        "final_metrics": rec.get("final_metrics"),
        "refused_or_skipped_lines": refused_lines,
        "assertions": {"V1_cached": v1, "V2_lines": v2, "V3_effective_free": v3, "V4_protected_evicted": v4},
        "v3_slack_bytes": V3_SLACK,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "TURNS.md").write_text("\n".join(table) + "\n")
    (args.out / "VERDICT.txt").write_text(verdict + "\n")
    print(rec["boot"]["line"])
    for ln in refused_lines:
        print("refusal:", ln)
    for t in turns:
        for ln in t["window"]["lines"]:
            print(f"turn {t['turn']}:", ln)
    print("\n".join(table))
    print(verdict, flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
