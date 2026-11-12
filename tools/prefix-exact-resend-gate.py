#!/usr/bin/env python3
"""prefix-exact-resend-gate.py: an exact re-send restores the whole prompt and is the cold prompt.

The defect (memra#1233): since memra#602 the prompt-end seed publishes at the largest GDN-grid
length under an off-grid prompt end, so a growing conversation primes its suffix from a grid start.
An exact re-send of the same prompt then restored only the grid entry and primed the off-grid
remainder before its first token (a 4,860-token prompt restored 4,832 and primed 28). Since the fix
the grid entry carries the prompt-end tail the publishing prime ended in (`PromptEndTail`,
crates/memra-server/src/worker.rs): an exact re-send restores every prompt token and decodes its
first token from the restored logits, while a longer prompt still restores the grid entry.

Serving shape, one card, two boots of the real `memra-server`, plain path (`MEMRA_SERVE_SPEC=0`),
greedy `prompt_ids` requests, the twin gate's id generators: P1 = turn 1 of the day-16 chain
(11,000 ids, 24 past the grid), P2 = turn 2 (11,150 ids, P1 plus 150), P3 = P1's grid entry
followed by P1's remainder reversed (the same length, another prompt end on the same entry).
  1. cache OFF: the cold completions of P1, P2 and P3.
  2. cache ON, one `cache_salt`: P1 (cold, publishes), P1 again (the exact re-send), P2 (growth),
     P1 again, P2 again; then P3 (a covered hit: it restores P1's grid entry, primes its own
     remainder and attaches its tail in place of P1's), P3 again, P1 (its tail is gone: the grid
     entry and a primed remainder, which attaches P1's tail again), P1 again.

Clauses, every one a verdict:
  V1 resend:    each P1 re-send reports `cached_tokens` == len(P1), prints `hit: len(P1) of len(P1)`
                and makes no prime call (`[primeseg] call` lines, MEMRA_DEBUG_PRIMESEG=1).
  V2 growth:    the first P2 restores exactly capture_len(P1), the grid entry, and every prime call
                it makes starts on the grid (memra#602's law is unchanged); the P2 re-send restores
                len(P2).
  V3 identity:  every completion's text equals the cold completion of the same prompt.
  V4 tail:      the cache-on boot printed `[prefix-cache] insert tail:` for P1 and for P2.
  V5 covered:   a tail attached by a covered hit serves its own prompt and no other. The first
                P3 and the P1 after it restore capture_len(P1) with every prime call on the grid;
                each attaches its tail (the first P3 after `evict tail (replaced)`); the P3 re-send
                and the last P1 restore the whole prompt with no prime call.
Red on main before memra#1233: V1 (cached capture_len(P1), one 24-token prime call), V2's re-send
clause, V4 and V5.
Verdict line:
  PREFIX-EXACT-RESEND: p1=N p2=M grid=G resend_ok=a/b growth_ok=c/d identical=e/f tails=t covered_ok=g/h V1.. -> PASS|FAIL
Exit 0 = PASS; 1 = a clause failed; 2 = REFUSED.

usage: prefix-exact-resend-gate.py [--external-lock FD] --model GGUF --bin memra-server --out NEW_DIR \
           [--port N] [--budget-mib 2048] [--start-tokens 11000] [--grow-tokens 150] [--grid 32] \
           [--max-tokens 16]
Lock: the canonical rig lock only, as the twin gate.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("restore_gate", HERE / "prefix-restore-identity-gate.py")
rgate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rgate)
gate = rgate.gate

LOCKS = ("/tmp/memra-gpu.lock", "/tmp/memra-5090.lock")
RE_HIT = re.compile(r"\[prefix-cache\] hit: (\d+) of (\d+) prompt tokens")
RE_TAIL = re.compile(r"\[prefix-cache\] insert tail: (\d+) prompt-end tokens past the (\d+)-token entry")
RE_TAIL_REPLACED = re.compile(r"\[prefix-cache\] evict tail \(replaced\): (\d+) prompt-end tokens past the (\d+)-token entry")


def refuse(msg: str) -> None:
    print(f"REFUSED: {msg}", flush=True)
    sys.exit(2)


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
    ap.add_argument("--port", type=int, default=18118)
    ap.add_argument("--budget-mib", type=int, default=2048)
    ap.add_argument("--start-tokens", type=int, default=11000)
    ap.add_argument("--grow-tokens", type=int, default=150)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=16)
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
    p1, p2 = gate.twin_ids(args.start_tokens, args.grow_tokens, 2)
    if len(p1) % args.grid == 0:
        refuse(f"P1 ({len(p1)} ids) ends on the {args.grid}-token grid; the gate needs an off-grid prompt end")
    cap1 = gate.capture_len(len(p1), args.grid)
    if cap1 is None or len(p2) - cap1 < gate.PRIME_MIN_T:
        refuse("P1 has no grid entry P2 can restore")
    p3 = p1[:cap1] + p1[cap1:][::-1]
    if p3 == p1:
        refuse("P1's remainder past the grid reads the same reversed; P3 would be P1")

    args.out.mkdir(parents=True)
    (args.out / "LOCK.json").write_text(proof.stdout)
    rig = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,power.limit,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    plan = {
        "rig": rig, "binary": args.bin, "binary_sha256": hashlib.sha256(Path(args.bin).read_bytes()).hexdigest(),
        "model": args.model, "budget_mib": args.budget_mib, "p1": len(p1), "p2": len(p2), "grid": args.grid,
        "p3": len(p3), "capture_len_p1": cap1, "max_tokens": args.max_tokens,
    }
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    def run(srv, ids: list[int], salt: str) -> dict:
        r = srv.complete_text(ids, salt, args.max_tokens)
        time.sleep(0.5)
        lines = srv.new_log_lines()
        r["lines"] = rgate.kept(lines)
        w = gate.parse_window(lines)
        r["primeseg"] = w["primeseg"]
        r["hit_lines"] = [(int(m.group(1)), int(m.group(2))) for ln in lines if (m := RE_HIT.search(ln))]
        r["tails"] = [(int(m.group(1)), int(m.group(2))) for ln in lines if (m := RE_TAIL.search(ln))]
        r["tails_replaced"] = [(int(m.group(1)), int(m.group(2))) for ln in lines if (m := RE_TAIL_REPLACED.search(ln))]
        if r["status"] != 200:
            refuse(f"a request was not served (HTTP {r['status']}): {r['error']}")
        return r

    srv = rgate.Server(args.bin, args.model, args.port, args.out / "cold" / "server.log", 0)
    srv.boot()
    try:
        cold = {"p1": run(srv, p1, "cold"), "p2": run(srv, p2, "cold"), "p3": run(srv, p3, "cold")}
    finally:
        srv.stop()
    (args.out / "cold" / "rows.json").write_text(json.dumps(cold, indent=2) + "\n")

    srv = rgate.Server(args.bin, args.model, args.port, args.out / "cached" / "server.log", args.budget_mib)
    srv.boot()
    steps = [("p1-cold", "p1", p1), ("p1-resend", "p1", p1), ("p2-growth", "p2", p2), ("p1-resend2", "p1", p1),
             ("p2-resend", "p2", p2), ("p3-covered", "p3", p3), ("p3-resend", "p3", p3), ("p1-covered", "p1", p1),
             ("p1-resend3", "p1", p1)]
    rows = []
    try:
        for name, which, ids in steps:
            r = run(srv, ids, "resend")
            r["step"], r["prompt"], r["which"] = name, len(ids), which
            rows.append(r)
        final = srv.settled_metrics()
    finally:
        srv.stop()

    by = {r["step"]: r for r in rows}
    resend = [by["p1-resend"], by["p1-resend2"]]
    tail1 = (len(p1) - cap1, cap1)

    def covered(r: dict, replaced: bool) -> bool:
        return (r["cached_tokens"] == cap1 and r["hit_lines"] == [(cap1, r["prompt"])] and len(r["primeseg"]) >= 1
                and all(c["grid_off"] == 0 for c in r["primeseg"]) and r["tails"] == [tail1]
                and (r["tails_replaced"] == [tail1]) == replaced)

    def whole(r: dict) -> bool:
        return r["cached_tokens"] == r["prompt"] and r["hit_lines"] == [(r["prompt"], r["prompt"])] and not r["primeseg"]

    covered_ok = [covered(by["p3-covered"], True), whole(by["p3-resend"]), covered(by["p1-covered"], True),
                  whole(by["p1-resend3"])]
    resend_ok = [
        r["cached_tokens"] == len(p1) and r["hit_lines"] == [(len(p1), len(p1))] and not r["primeseg"] for r in resend
    ]
    g = by["p2-growth"]
    growth_ok = [
        g["cached_tokens"] == cap1 and all(c["grid_off"] == 0 for c in g["primeseg"]) and len(g["primeseg"]) >= 1,
        by["p2-resend"]["cached_tokens"] == len(p2),
    ]
    ident = [r["text_sha256"] == cold[r["which"]]["text_sha256"] for r in rows]
    tails = [t for r in rows for t in r["tails"]]
    v4 = (len(p1) - cap1, cap1) in tails and any(t[1] + t[0] == len(p2) for t in tails)
    v1, v2, v3, v5 = all(resend_ok), all(growth_ok), all(ident), all(covered_ok)
    ok = v1 and v2 and v3 and v4 and v5
    verdict = (
        f"PREFIX-EXACT-RESEND: p1={len(p1)} p2={len(p2)} grid={args.grid} capture_p1={cap1} "
        f"resend_ok={sum(resend_ok)}/{len(resend_ok)} growth_ok={sum(growth_ok)}/{len(growth_ok)} "
        f"identical={sum(ident)}/{len(ident)} tails={len(tails)} covered_ok={sum(covered_ok)}/{len(covered_ok)} "
        f"V1={'ok' if v1 else 'FAIL'} V2={'ok' if v2 else 'FAIL'} V3={'ok' if v3 else 'FAIL'} V4={'ok' if v4 else 'FAIL'} "
        f"V5={'ok' if v5 else 'FAIL'} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    table = ["| step | prompt | cached_tokens | hit line | prime calls | off-grid calls | tails inserted | elapsed s | sha[:16] | cold sha[:16] | == cold |",
             "| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | --- | --- | --- |"]
    for r, same in zip(rows, ident):
        c = cold[r["which"]]
        table.append(
            f"| {r['step']} | {r['prompt']} | {r['cached_tokens']} | {r['hit_lines']} | {len(r['primeseg'])} | "
            f"{sum(1 for x in r['primeseg'] if x['grid_off'])} | {r['tails']} | {r['elapsed_s']} | {r['text_sha256'][:16]} | "
            f"{c['text_sha256'][:16]} | {'yes' if same else 'NO'} |"
        )
    summary = {"verdict": verdict, "pass": ok, "plan": plan, "cold": cold, "rows": rows, "final_metrics": final,
               "assertions": {"V1_resend": v1, "V2_growth": v2, "V3_identity": v3, "V4_tail": v4, "V5_covered": v5}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "STEPS.md").write_text("\n".join(table) + "\n")
    (args.out / "VERDICT.txt").write_text(verdict + "\n")
    for r in rows:
        for ln in r["lines"]:
            print(f"{r['step']}:", ln)
    print("\n".join(table))
    print(verdict, flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
