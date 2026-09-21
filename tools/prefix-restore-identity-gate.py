#!/usr/bin/env python3
"""prefix-restore-identity-gate.py: one prompt restored from entries of chosen lengths is the cold prompt.

The defect (memra#602; research/spill-b-20260919/DAY17.md, probe E): the prefix cache published the
prompt-end seed wherever the prompt ended, so a hit restored an entry whose end sat OFF the 32-token
GDN WY-chunk grid and primed the queued suffix as one `prime_cache` call starting there. Under the
chunked GDN scan an off-grid call start shifts the fold grid (`align_prime_ranges_to_gdn`,
hybrid_forward.rs), so the restored render was a second numeric program and greedy near-ties flipped:
on the local RTX 5090 the 12,350-token day-16 prompt restored from ON-grid entries (12,288 and 12,320)
reproduced the cold bytes and restored from OFF-grid entries (12,250 and 12,300) produced the other
stream (`RESTORE-POINTS ... identical=3/5`). Since day 18 every seed lands on the grid
(`seed_capture_boundary`, worker.rs): an entry seeded from a prompt of p tokens has `capture_len(p)`
tokens (the twin gate's helper), a hit restores exactly that many, and the suffix primes from a grid
start. Owner law: one numeric program per request; a restored suffix produces the cold prime's bytes.

Serving shape, one card, two boots of the real `memra-server`, plain path (`MEMRA_SERVE_SPEC=0`),
greedy `prompt_ids` requests, `max_tokens` 8, the twin gate's own id generators (so the target prompt
is turn `--turns` of the day-16 chain byte for byte: `twin_ids(11000, 150, 10)[-1]`, the 12,350-id turn
10 that flipped; the chain's turn 12, 12,650 ids, is `--turns 12`):
  1. cache OFF (`MEMRA_PREFIX_CACHE_MB=0`): the cold completion of the target prompt, text kept.
  2. cache ON (`--budget-mib`): for every restore point p, under its own `cache_salt` so no point can
     hit another's entry, a SEED request of `target[:p]` (cold; publishes the entry) and then a HIT
     request of the whole target (a restore of that entry plus a suffix prime).

Clauses, every one a verdict:
  V1 identity:  every hit's completion text (sha256 of the text) equals the cold completion's.
  V2 grid:      per point, the seed's `insert (seed): N tokens` line has N == capture_len(p), the
                hit's `cached_tokens` and its `[prefix-cache] hit: N of M` line both equal that N,
                N is a multiple of `--grid`, and every `[primeseg] call start=..` of the hit
                (MEMRA_DEBUG_PRIMESEG=1, an existing diagnostic) has grid_off=0: the restore point is
                on the grid and the suffix primes from a grid start. Red on `main` before day 18 for
                every off-grid point even where V1 happens to hold (12,200 held on day 17 from a
                single cold lineage and flipped eight restores deep: a near-tie that did not flip is
                not identity).
Verdict line:
  PREFIX-RESTORE-IDENTITY: target=T points=N identical=K/N grid_ok=M/N grid=G
  p(seed S, restored R, off O, suffix Q):yes|NO ... -> PASS|FAIL
Exit 0 = PASS; 1 = a clause failed; 2 = REFUSED (lock, port, shape, an unserved request).

usage: prefix-restore-identity-gate.py [--external-lock FD] --model GGUF --bin memra-server --out NEW_DIR \
           [--port N] [--budget-mib 1024] [--turns 10] [--start-tokens 11000] [--grow-tokens 150] \
           [--points 12288,12320,12200,12250,12300] [--grid 32] [--max-tokens 8]
Lock: the canonical rig lock only (`/tmp/memra-gpu.lock` or `/tmp/memra-5090.lock`), held for the
whole cell; `--external-lock FD` inherits the collector's FD (lead ruling 5) and is verified with
tools/tier-lock-proof.py before anything binds a port or boots a server.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("twin_gate", HERE / "prefix-newest-turn-fits-gate.py")
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

LOCKS = ("/tmp/memra-gpu.lock", "/tmp/memra-5090.lock")
KEEP = ("[prefix-cache]", "[admit-oom]", "[spec-k]", "[primeseg] call", "[suffix-prime]")


def refuse(msg: str) -> None:
    print(f"REFUSED: {msg}", flush=True)
    sys.exit(2)


class Server(gate.Server):
    """The twin gate's server (same env, same default-policy discipline) plus the completion text."""

    def complete_text(self, ids: list[int], salt: str, max_tokens: int) -> dict:
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
            f"{self.base}/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
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
        usage = payload.get("usage") or {}
        return {
            "status": status,
            "elapsed_s": round(elapsed, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "finish_reason": finish,
            "text": text_out,
            "text_sha256": hashlib.sha256((text_out or "").encode()).hexdigest(),
            "error": None if status == 200 else payload,
        }


def kept(lines: list[str]) -> list[str]:
    return [ln.strip() for ln in lines if any(k in ln for k in KEEP)]


def first_diff(a: str, b: str) -> int | None:
    if a == b:
        return None
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), n)


def main() -> None:
    argv = sys.argv[1:]
    lock_fd, lock_owner = None, "internal-canonical"
    if argv[:1] == ["--external-lock"]:
        if len(argv) < 2 or not argv[1].isdigit():
            refuse("inherited lock FD required after --external-lock")
        lock_fd, lock_owner = int(argv[1]), "collector"
        argv = argv[2:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--port", type=int, default=18117)
    ap.add_argument("--budget-mib", type=int, default=1024)
    ap.add_argument("--turns", type=int, default=10, help="the chain turn whose prompt is the target (10 = the day-17 12,350-id near-tie)")
    ap.add_argument("--start-tokens", type=int, default=11000)
    ap.add_argument("--grow-tokens", type=int, default=150)
    ap.add_argument("--points", default="12288,12320,12200,12250,12300", help="entry lengths p to seed and restore from (the day-17 five)")
    ap.add_argument("--grid", type=int, default=32, help="the GDN prime grid (Engine::gdn_chunk_size(), 32 shipped)")
    ap.add_argument("--max-tokens", type=int, default=8)
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
    if args.grid < gate.PRIME_MIN_T or args.grid % gate.PRIME_MIN_T:
        refuse(f"--grid {args.grid} is not a multiple of PRIME_MIN_T={gate.PRIME_MIN_T}")
    target = gate.twin_ids(args.start_tokens, args.grow_tokens, args.turns)[-1]
    if len(target) + args.max_tokens + 64 >= 16384:
        refuse(f"the target prompt ({len(target)} tokens) does not fit MEMRA_CTX=16384")
    points = [int(p) for p in args.points.split(",") if p]
    if not points or len(set(points)) != len(points):
        refuse("--points needs at least one distinct entry length")
    bad = [p for p in points if p < gate.PREFIX_CACHE_MIN_TOKENS or p >= len(target) or gate.capture_len(p, args.grid) is None]
    if bad:
        refuse(f"--points must lie in {gate.PREFIX_CACHE_MIN_TOKENS}..{len(target) - 1} with a grid-aligned entry: {bad}")

    args.out.mkdir(parents=True)
    (args.out / "LOCK.json").write_text(proof.stdout)
    rig = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,power.limit,power.max_limit,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    binsha = hashlib.sha256(Path(args.bin).read_bytes()).hexdigest()
    plan = {
        "rig": rig, "binary_sha256": binsha, "binary": args.bin, "model": args.model, "budget_mib": args.budget_mib,
        "turns": args.turns, "start_tokens": args.start_tokens, "grow_tokens": args.grow_tokens, "target_tokens": len(target),
        "target_ids_sha256": hashlib.sha256(",".join(map(str, target)).encode()).hexdigest(),
        "points": points, "grid": args.grid, "max_tokens": args.max_tokens,
        "expected_capture": {p: gate.capture_len(p, args.grid) for p in points},
    }
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    def run(srv: Server, ids: list[int], salt: str) -> dict:
        r = srv.complete_text(ids, salt, args.max_tokens)
        time.sleep(0.5)
        lines = srv.new_log_lines()
        r["lines"] = kept(lines)
        w = gate.parse_window(lines)
        r["inserts"], r["hits"], r["primeseg"], r["refused"] = w["inserts"], w["hits"], w["primeseg"], w["refused"] + w["skipped"]
        return r

    # ---- 1. cold boot ------------------------------------------------------------------------
    srv = Server(args.bin, args.model, args.port, args.out / "cold" / "server.log", 0)
    srv.boot()
    try:
        boot = srv.new_log_lines()
        if any(gate.RE_ON.search(ln) for ln in boot):
            refuse("the cold boot armed the prefix cache; MEMRA_PREFIX_CACHE_MB=0 must disable it")
        cold = run(srv, target, "grow")
        if cold["status"] != 200:
            refuse(f"the cold target was not served (HTTP {cold['status']}): {cold['error']}")
        if cold["cached_tokens"]:
            refuse(f"cold target: cached_tokens={cold['cached_tokens']} in a cache-off boot")
    finally:
        srv.stop()
    (args.out / "cold" / "row.json").write_text(json.dumps(cold, indent=2) + "\n")
    print(f"cold: prompt={cold['prompt_tokens']} completion={cold['completion_tokens']} finish={cold['finish_reason']} sha={cold['text_sha256'][:16]} "
          f"off_grid_calls={sum(1 for c in cold['primeseg'] if c['grid_off'])}/{len(cold['primeseg'])}", flush=True)

    # ---- 2. cache-on boot: seed and hit per point ----------------------------------------------
    srv = Server(args.bin, args.model, args.port, args.out / "measured" / "server.log", args.budget_mib)
    srv.boot()
    rows: list[dict] = []
    try:
        boot = srv.new_log_lines()
        on = next((ln.strip() for ln in boot if gate.RE_ON.search(ln)), None)
        if on is None:
            refuse("no `[prefix-cache] on:` boot line; the cache did not arm")
        for p in points:
            salt = f"rp-{p}"
            seed = run(srv, target[:p], salt)
            if seed["status"] != 200:
                refuse(f"restore point {p}: the seed was not served: {seed['error']}")
            hit = run(srv, target, salt)
            if hit["status"] != 200:
                refuse(f"restore point {p}: the hit was not served: {hit['error']}")
            seed_inserts = [i for i in seed["inserts"] if i["why"] == "seed"]
            published = seed_inserts[0]["tokens"] if len(seed_inserts) == 1 else None
            expected = gate.capture_len(p, args.grid)
            hit_line = hit["hits"][0]["hit"] if len(hit["hits"]) == 1 else None
            off_grid = [c for c in hit["primeseg"] if c["grid_off"] != 0]
            row = {
                "point": p, "salt": salt, "expected_capture": expected, "published": published, "seed_inserts": len(seed_inserts),
                "restored": hit["cached_tokens"], "hit_line": hit_line, "suffix": (hit["prompt_tokens"] or 0) - (hit["cached_tokens"] or 0),
                "restored_grid_off": (hit["cached_tokens"] or 0) % args.grid,
                "hit_calls": len(hit["primeseg"]), "off_grid_calls": len(off_grid),
                "seed": seed, "hit": hit,
                "identical_to_cold": hit["text"] == cold["text"], "first_diff_char": first_diff(hit["text"], cold["text"]),
                "refused": seed["refused"] + hit["refused"],
            }
            row["grid_ok"] = (
                published is not None and published == expected
                and hit["cached_tokens"] == published and hit_line == published
                and published % args.grid == 0
                and len(hit["primeseg"]) >= 1 and not off_grid
                and not row["refused"]
            )
            rows.append(row)
            print(f"point {p}: seed published={published} (expected {expected}) hit restored={hit['cached_tokens']} suffix={row['suffix']} "
                  f"calls={row['hit_calls']} off_grid={row['off_grid_calls']} completion={hit['completion_tokens']} finish={hit['finish_reason']} "
                  f"sha={hit['text_sha256'][:16]} {'== cold' if row['identical_to_cold'] else '!= cold (first diff char ' + str(row['first_diff_char']) + ')'} "
                  f"grid_ok={'yes' if row['grid_ok'] else 'NO'}", flush=True)
        final = srv.settled_metrics()
    finally:
        srv.stop()

    v1_rows = [r["identical_to_cold"] for r in rows]
    v2_rows = [r["grid_ok"] for r in rows]
    v1, v2 = all(v1_rows), all(v2_rows)
    ok = v1 and v2
    cells = " ".join(
        f"{r['point']}(seed{r['published']},restored{r['restored']},off{r['restored_grid_off']},suffix{r['suffix']}):{'yes' if r['identical_to_cold'] else 'NO'}"
        for r in rows
    )
    verdict = (
        f"PREFIX-RESTORE-IDENTITY: target={len(target)} points={len(rows)} identical={sum(v1_rows)}/{len(rows)} "
        f"grid_ok={sum(v2_rows)}/{len(rows)} grid={args.grid} {cells} V1={'ok' if v1 else 'FAIL'} V2={'ok' if v2 else 'FAIL'} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    table = ["| point | expected capture | published | restored (cached_tokens) | hit line | restored % grid | suffix | hit prime calls | off-grid calls | completion | finish | sha[:16] | cold sha[:16] | == cold | first diff char | grid ok | restored text | cold text |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | --- | --- | --- |"]
    for r in rows:
        h = r["hit"]
        table.append(
            f"| {r['point']} | {r['expected_capture']} | {r['published']} | {r['restored']} | {r['hit_line']} | {r['restored_grid_off']} | {r['suffix']} | "
            f"{r['hit_calls']} | {r['off_grid_calls']} | {h['completion_tokens']} | {h['finish_reason']} | {h['text_sha256'][:16]} | {cold['text_sha256'][:16]} | "
            f"{'yes' if r['identical_to_cold'] else 'NO'} | {r['first_diff_char'] if r['first_diff_char'] is not None else '-'} | "
            f"{'yes' if r['grid_ok'] else 'NO'} | {json.dumps(h['text'])} | {json.dumps(cold['text'])} |"
        )
    summary = {
        "verdict": verdict, "pass": ok, "plan": plan, "boot_line": on, "cold": cold, "points": rows, "final_metrics": final,
        "assertions": {"V1_identity": v1, "V2_grid": v2},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "TURNS.md").write_text("\n".join(table) + "\n")
    (args.out / "VERDICT.txt").write_text(verdict + "\n")
    print(on)
    for ln in cold["lines"]:
        print("cold:", ln)
    for r in rows:
        for ln in r["seed"]["lines"]:
            print(f"point {r['point']} seed:", ln)
        for ln in r["hit"]["lines"]:
            print(f"point {r['point']} hit:", ln)
    print("\n".join(table))
    print(verdict, flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
