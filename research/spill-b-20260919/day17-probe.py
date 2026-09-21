#!/usr/bin/env python3
"""day17-probe.py: restored-suffix completion versus the cold completion of the same prompt, texts kept.

WP-B day 17 (memra#523 thread, the day-16 digest precondition failure on the local RTX 5090). The twin
gate (`tools/prefix-newest-turn-fits-gate.py`) records only a sha256 per completion; this probe replays
the same request shape on the same binary and keeps the completion TEXT of both sides, the server's own
per-request lines (`[primeseg]` under MEMRA_DEBUG_PRIMESEG, `[prefix-cache]`, `[admit-oom]`, `[spec-k]`,
`[ttft]`) and the first differing character between the restored completion and the cold one. It is a
classification instrument, not a gate: exit 0 when every compared turn is identical, 1 when any differs,
2 REFUSED. It changes no engine or server code and sets no new MEMRA_* name: the arm environment is
passed through `--env KEY=VAL` (existing documented reads only) into BOTH boots.

Two boots of the real `memra-server`, plain path (`MEMRA_SERVE_SPEC=0`), `MEMRA_CTX=16384`, greedy,
`prompt_ids` requests, the gate's own id generators (`ids_for`, `twin_ids`, `cohort_ids` imported from
the gate, so the prompts are byte-identical to the gate's and to day 16's loop):
  1. cache OFF (`MEMRA_PREFIX_CACHE_MB=0`): the cold completion of every turn in `--cold-turns`.
  2. cache ON (`--budget-mib`): optional cohort seeding (each cohort prompt twice, as the gate and the
     day-16 harness do), then the growing tenant's turns 1..N, turn k+1 = turn k's ids + grow ids, so
     every turn k >= 2 is a restore of turn k-1's entry plus a suffix of `--grow-tokens` ids.
Per compared turn: restored text == cold text?, first differing char, completion token counts, finish
reasons, `cached_tokens`, the prime receipts. Verdict line:
  RESTORE-VS-COLD: arm=<label> turns=N compared=M identical=K/M first_divergent_turn=T prompt=P
  restored=R suffix=S first_diff_char=C -> IDENTICAL|DIVERGENT

Lock: the canonical rig lock only, inherited from the collector (`--external-lock FD`) and verified with
tools/tier-lock-proof.py before any port binds; or taken here when run bare (never done on this lane).
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
REPO = HERE.parent.parent
TOOLS = REPO / "tools"
_spec = importlib.util.spec_from_file_location("twin_gate", TOOLS / "prefix-newest-turn-fits-gate.py")
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

LOCKS = ("/tmp/memra-gpu.lock", "/tmp/memra-5090.lock")
KEEP = ("[prefix-cache]", "[admit-oom]", "[admit-trim]", "[spec-k]", "[primeseg]", "[ttft]", "[suffix-prime]")


def refuse(msg: str) -> None:
    print(f"REFUSED: {msg}", flush=True)
    sys.exit(2)


class Server(gate.Server):
    def __init__(self, binary, model, port, log, budget_mib, extra_env: dict, request_timeout_s: int):
        super().__init__(binary, model, port, log, budget_mib)
        self.extra_env = extra_env
        self.request_timeout_s = request_timeout_s

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
                "MEMRA_TIMEOUT_MS_MAX": str(self.request_timeout_s * 1000),
            }
        )
        env.pop("MEMRA_PREFIX_CACHE_POLICY", None)
        env.pop("MEMRA_PREFIX_CACHE_PROTECTED_PCT", None)
        env.update(self.extra_env)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("wb") as log:
            self.proc = subprocess.Popen([self.binary], stdout=log, stderr=subprocess.STDOUT, env=env)
        for _ in range(300):
            if self.probe():
                return
            if self.proc.poll() is not None:
                refuse(f"server died during boot (exit {self.proc.returncode}); see {self.log}")
            time.sleep(2)
        self.stop()
        refuse("server never became ready")

    def complete_text(self, ids: list[int], salt: str, max_tokens: int = 8) -> dict:
        body = {
            "model": "gate",
            "prompt_ids": ids,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
            "cache_salt": salt,
            "timeout_ms": self.request_timeout_s * 1000,
        }
        req = urllib.request.Request(
            f"{self.base}/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout_s + 60) as f:
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


def lines_of_interest(lines: list[str]) -> list[str]:
    return [ln.strip() for ln in lines if any(k in ln for k in KEEP)]


def first_diff(a: str, b: str) -> int | None:
    if a == b:
        return None
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


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
    ap.add_argument("--port", type=int, default=18116)
    ap.add_argument("--budget-mib", type=int, default=1024)
    ap.add_argument("--cohort-tokens", default="1250,1350,1450,1550")
    ap.add_argument("--no-cohort", action="store_true")
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--start-tokens", type=int, default=11000)
    ap.add_argument("--grow-tokens", type=int, default=150)
    ap.add_argument("--cold-turns", default="all", help="comma list of turns to prime cold, or all")
    ap.add_argument("--env", action="append", default=[], help="KEY=VAL for both boots (existing documented reads only)")
    ap.add_argument("--label", default="default")
    ap.add_argument("--request-timeout-s", type=int, default=240)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--gpu-lock", default=os.environ.get("MEMRA_GPU_LOCK", "/tmp/memra-gpu.lock"))
    args = ap.parse_args(argv)

    if args.gpu_lock not in LOCKS:
        refuse("noncanonical GPU lock")
    if lock_owner == "internal-canonical":
        fh = open(args.gpu_lock, "a")  # noqa: SIM115
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            refuse("canonical GPU lock busy")
        lock_fd = fh.fileno()
    proof = subprocess.run(
        [sys.executable, str(TOOLS / "tier-lock-proof.py"), "--fd", str(lock_fd), "--lock", args.gpu_lock, "--owner", lock_owner],
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
    extra_env = {}
    for kv in args.env:
        if "=" not in kv:
            refuse(f"--env wants KEY=VAL, got {kv!r}")
        k, v = kv.split("=", 1)
        if not k.startswith("MEMRA_"):
            refuse(f"--env only passes MEMRA_* reads through, got {k!r}")
        extra_env[k] = v
    cohort_tokens = [] if args.no_cohort else [int(t) for t in args.cohort_tokens.split(",") if t]
    prompts = gate.twin_ids(args.start_tokens, args.grow_tokens, args.turns)
    last_tokens = len(prompts[-1])
    if last_tokens + args.max_tokens + 64 >= 16384:
        refuse(f"the last turn ({last_tokens} tokens) does not fit MEMRA_CTX=16384")
    cold_turns = list(range(1, args.turns + 1)) if args.cold_turns == "all" else sorted({int(t) for t in args.cold_turns.split(",") if t})
    if any(t < 1 or t > args.turns for t in cold_turns):
        refuse(f"--cold-turns outside 1..{args.turns}")

    args.out.mkdir(parents=True)
    (args.out / "LOCK.json").write_text(proof.stdout)
    rig = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,power.limit,power.max_limit,driver_version,temperature.gpu", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    binsha = hashlib.sha256(Path(args.bin).read_bytes()).hexdigest()
    plan = {
        "label": args.label, "env": extra_env, "rig": rig, "binary_sha256": binsha, "binary": args.bin, "model": args.model,
        "budget_mib": args.budget_mib, "cohort_tokens": cohort_tokens, "turns": args.turns, "start_tokens": args.start_tokens,
        "grow_tokens": args.grow_tokens, "cold_turns": cold_turns, "max_tokens": args.max_tokens,
        "request_timeout_s": args.request_timeout_s,
        "prompt_ids_sha256": [hashlib.sha256(",".join(map(str, p)).encode()).hexdigest() for p in prompts],
        "prompt_lengths": [len(p) for p in prompts],
    }
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    def run_request(srv: Server, ids: list[int], salt: str) -> dict:
        r = srv.complete_text(ids, salt, args.max_tokens)
        time.sleep(0.5)
        r["lines"] = lines_of_interest(srv.new_log_lines())
        return r

    # ---- 1. cold boot ------------------------------------------------------------------------
    cold: dict[int, dict] = {}
    srv = Server(args.bin, args.model, args.port, args.out / "cold" / "server.log", 0, extra_env, args.request_timeout_s)
    srv.boot()
    try:
        boot = srv.new_log_lines()
        if any(gate.RE_ON.search(ln) for ln in boot):
            refuse("the cold boot armed the prefix cache; MEMRA_PREFIX_CACHE_MB=0 must disable it")
        (args.out / "cold" / "boot.txt").write_text("\n".join(lines_of_interest(boot)) + "\n")
        for k in cold_turns:
            r = run_request(srv, prompts[k - 1], "grow")
            if r["status"] != 200:
                refuse(f"cold turn {k} was not served (HTTP {r['status']}): {r['error']}")
            if r["cached_tokens"]:
                refuse(f"cold turn {k}: cached_tokens={r['cached_tokens']} in a cache-off boot")
            r["turn"], r["prompt_ids"] = k, len(prompts[k - 1])
            cold[k] = r
            print(f"cold turn {k}: prompt={r['prompt_tokens']} completion={r['completion_tokens']} finish={r['finish_reason']} sha={r['text_sha256'][:16]} {r['elapsed_s']}s", flush=True)
    finally:
        srv.stop()
    (args.out / "cold" / "rows.json").write_text(json.dumps([cold[k] for k in cold_turns], indent=2) + "\n")

    # ---- 2. cache-on boot: cohort, then the chain ----------------------------------------------
    srv = Server(args.bin, args.model, args.port, args.out / "chain" / "server.log", args.budget_mib, extra_env, args.request_timeout_s)
    srv.boot()
    rows: list[dict] = []
    cohort_rows: list[dict] = []
    try:
        boot = srv.new_log_lines()
        on = next((ln.strip() for ln in boot if gate.RE_ON.search(ln)), None)
        if on is None:
            refuse("no `[prefix-cache] on:` boot line; the cache did not arm")
        (args.out / "chain" / "boot.txt").write_text("\n".join(lines_of_interest(boot)) + "\n")
        for i, n in enumerate(cohort_tokens, start=1):
            ids = gate.cohort_ids(n)
            for send in (1, 2):
                r = run_request(srv, ids, f"cohort-{i}")
                if r["status"] != 200:
                    refuse(f"cohort {n} send {send} was not served: {r['error']}")
                r.update({"role": f"seed{send}", "tenant": i, "prompt_ids": n})
                cohort_rows.append(r)
        prev = None
        for k, ids in enumerate(prompts, start=1):
            r = run_request(srv, ids, "grow")
            if r["status"] != 200:
                refuse(f"turn {k} was not served (HTTP {r['status']}): {r['error']}")
            r.update({"turn": k, "prompt_ids": len(ids), "prev_prompt_tokens": prev})
            c = cold.get(k)
            if c is not None:
                r["cold_text_sha256"] = c["text_sha256"]
                r["identical_to_cold"] = r["text"] == c["text"]
                r["first_diff_char"] = first_diff(r["text"], c["text"])
            rows.append(r)
            prev = r["prompt_tokens"]
            tag = "" if c is None else (" == cold" if r["identical_to_cold"] else f" != cold (first diff char {r['first_diff_char']})")
            print(f"turn {k}: prompt={r['prompt_tokens']} cached={r['cached_tokens']} completion={r['completion_tokens']} finish={r['finish_reason']} sha={r['text_sha256'][:16]}{tag} {r['elapsed_s']}s", flush=True)
    finally:
        srv.stop()
    (args.out / "chain" / "cohort.json").write_text(json.dumps(cohort_rows, indent=2) + "\n")
    (args.out / "chain" / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")

    compared = [r for r in rows if "identical_to_cold" in r]
    identical = [r for r in compared if r["identical_to_cold"]]
    divergent = [r for r in compared if not r["identical_to_cold"]]
    first = divergent[0] if divergent else None
    verdict = (
        f"RESTORE-VS-COLD: arm={args.label} turns={len(rows)} compared={len(compared)} identical={len(identical)}/{len(compared)} "
        + (f"first_divergent_turn={first['turn']} prompt={first['prompt_tokens']} restored={first['cached_tokens']} "
           f"suffix={first['prompt_tokens'] - (first['cached_tokens'] or 0)} first_diff_char={first['first_diff_char']} "
           if first else "first_divergent_turn=none ")
        + f"-> {'DIVERGENT' if divergent else 'IDENTICAL'}"
    )
    table = ["| turn | prompt | cached | suffix | completion | finish | sha[:16] | cold completion | cold finish | cold sha[:16] | == cold | first diff char | restored text | cold text |",
             "| ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | --- | ---: | --- | --- |"]
    for r in rows:
        c = cold.get(r["turn"])
        suffix = r["prompt_tokens"] - (r["cached_tokens"] or 0)
        if c is None:
            table.append(f"| {r['turn']} | {r['prompt_tokens']} | {r['cached_tokens']} | {suffix} | {r['completion_tokens']} | {r['finish_reason']} | {r['text_sha256'][:16]} | - | - | - | - | - | {json.dumps(r['text'])} | - |")
        else:
            table.append(
                f"| {r['turn']} | {r['prompt_tokens']} | {r['cached_tokens']} | {suffix} | {r['completion_tokens']} | {r['finish_reason']} | {r['text_sha256'][:16]} | "
                f"{c['completion_tokens']} | {c['finish_reason']} | {c['text_sha256'][:16]} | {'yes' if r['identical_to_cold'] else 'NO'} | "
                f"{r['first_diff_char'] if r['first_diff_char'] is not None else '-'} | {json.dumps(r['text'])} | {json.dumps(c['text'])} |"
            )
    summary = {"verdict": verdict, "divergent": bool(divergent), "plan": plan, "cold": [cold[k] for k in cold_turns], "cohort": cohort_rows, "turns": rows}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "TURNS.md").write_text("\n".join(table) + "\n")
    (args.out / "VERDICT.txt").write_text(verdict + "\n")
    print(on)
    for r in rows:
        for ln in r["lines"]:
            print(f"turn {r['turn']}:", ln)
    print("\n".join(table))
    print(verdict, flush=True)
    sys.exit(1 if divergent else 0)


if __name__ == "__main__":
    main()
