#!/usr/bin/env python3
"""M1 B4: serving shape of the spill arms (M1-PREREG.md B4 and its amendment).

  run --server memra-server --artifact FILE --arms-lock b3-arms.lock.json --arms worker16[,X]
      --regime cold|warm --proof PRIVATE_PROOF.json --out DIR --rig pro-single --lock-fd N
      [--pairs 5] [--concurrency 1,4] [--port 18129] [--stub-no-lock]

One visit = one fresh memra-server boot with the B3 common env plus the arm env, the regime
applied to the artifact before the boot, then the fixed 32-request set (m1-b4-prompts, derived
from the B3 prompt) streamed through /v1/completions with `c` client workers, greedy,
max_tokens 128, stream_options.include_usage. With two arms: for each concurrency, 5 pairs in
AB order and 5 in BA. With one arm (no B3 winner): 5 visits per concurrency, descriptive. Per
request: TTFT, E2E, TPOT, every inter-token gap. Per visit: p50/p95/p99 of each, request and
token throughput over the visit's request window, the server's [spill-pread] snapshot, the
250 ms sampler on the proof's leaves. Server output goes straight to a file; parsing reads files.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "tools/tier-battery.py").exists())


def load(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = load("battery", ROOT / "tools/tier-battery.py")
CACHE = load("regime", HERE / "m1-cache-regime.py")
RUNNER = load("runner", HERE / "m1-spill-runner.py")
SAMPLE = load("sampler", HERE / "m1-host-sampler.py")
SAMPLER = HERE / "m1-host-sampler.py"


def prompts(lock):
    base = (ROOT / lock["prompt"]["file"]).read_text().strip()
    return [f"{base} Answer variant {i + 1} of 32 in your own order." for i in range(32)]


def pct(values, q):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def stream_one(port, prompt, max_tokens):
    body = {"model": "gate", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    stamps, usage, text = [], {}, []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                piece = ch.get("text") or ""
                if piece:
                    stamps.append(time.monotonic() - t0)
                    text.append(piece)
    e2e = time.monotonic() - t0
    n = usage.get("completion_tokens", len(stamps))
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    return {"ttft_s": stamps[0] if stamps else None, "e2e_s": e2e, "completion_tokens": n,
            "tpot_s": (e2e - stamps[0]) / (n - 1) if stamps and n > 1 else None, "itl_s": gaps,
            "t_start": t0, "t_end": t0 + e2e, "text_len": len("".join(text))}


def run_requests(port, items, concurrency, max_tokens):
    work, results, errors = queue.Queue(), [None] * len(items), []
    for i, p in enumerate(items):
        work.put((i, p))

    def worker():
        while True:
            try:
                i, p = work.get_nowait()
            except queue.Empty:
                return
            try:
                results[i] = stream_one(port, p, max_tokens)
            except Exception as e:  # recorded, never retried
                errors.append(f"request {i}: {type(e).__name__}: {e}")
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, errors


def visit_summary(results, window_s):
    ok = [r for r in results if r]
    fields = {}
    for key in ("ttft_s", "e2e_s", "tpot_s"):
        vals = [r[key] for r in ok if r[key] is not None]
        fields[key] = {f"p{q}": pct(vals, q) for q in (50, 95, 99)}
    gaps = [g for r in ok for g in r["itl_s"]]
    fields["itl_s"] = {f"p{q}": pct(gaps, q) for q in (50, 95, 99)}
    tokens = sum(r["completion_tokens"] for r in ok)
    fields.update(requests=len(ok), tokens=tokens, window_s=window_s,
                  req_per_s=len(ok) / window_s if window_s else None,
                  tok_per_s=tokens / window_s if window_s else None)
    return fields


def boot(args, env, log):
    with open(log, "xb") as out:
        proc = subprocess.Popen([str(args.server)], stdout=out, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        B.require(proc.poll() is None, f"server exited during boot; see {Path(log).name}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=2)
            return proc
        except OSError:
            time.sleep(0.5)
    proc.kill()
    raise ValueError("server never became ready")


def run(args):
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=False)
    if args.stub_no_lock:
        B.require(Path(args.server).name.startswith("m1-stub"), "--stub-no-lock is only for the stub server")
        lock_rec = {"stub": True}
    else:
        proc = subprocess.run([sys.executable, str(ROOT / "tools/tier-lock-proof.py"), "--fd", str(args.lock_fd),
                               "--lock", B.LOCKS[args.rig]], pass_fds=(args.lock_fd,), capture_output=True, text=True)
        B.require(proc.returncode == 0, f"lock proof failed: {proc.stderr.strip()}")
        lock_rec = json.loads(proc.stdout)
    lock = json.loads(Path(args.arms_lock).read_text())
    arms = [next(a for a in lock["arms"] if a["name"] == n) for n in args.arms.split(",")]
    proof, identity, leaves, top = RUNNER.proof_view(args.proof)
    B.require(B.filesystem_identity(Path(args.artifact).parent) == identity, "artifact is not on the proven filesystem")
    items = prompts(lock)
    (args.out / "identity.json").write_text(json.dumps({
        "script_sha256": RUNNER.sha(__file__), "server_sha256": RUNNER.sha(args.server), "arms": args.arms,
        "regime": args.regime, "prompts_sha256": hashlib.sha256("\n".join(items).encode()).hexdigest(),
        "lock": lock_rec, "leaves": leaves, "qualified": False}, indent=1) + "\n")
    concurrencies = [int(c) for c in args.concurrency.split(",")]
    visits = []
    schedule = []
    for c in concurrencies:
        if len(arms) == 1:
            schedule += [(c, p + 1, "single", [arms[0]]) for p in range(args.pairs)]
        else:
            # Round-robin as in B3: odd rounds forward, even rounds reversed, so every pair of
            # arms meets `pairs` times in each relative order (for two arms: 5 AB plus 5 BA).
            for p in range(args.pairs):
                schedule.append((c, p + 1, "forward", arms))
                schedule.append((c, p + 1, "reverse", arms[::-1]))
    for c, pair, order, arm_order in schedule:
        for arm in arm_order:
            name = f"c{c}-p{pair}-{order}-{arm['name']}"
            vdir = args.out / name
            vdir.mkdir()
            B.require(B.filesystem_identity(Path(args.artifact).parent) == identity, "identity changed")
            regime_ok = CACHE.cold([args.artifact]) if args.regime == "cold" else CACHE.warm([args.artifact])
            env = RUNNER.arm_env(lock, arm)
            env.pop("MEMRA_NGEN", None)
            env.update({"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"gate={args.artifact}",
                        "MEMRA_ADDR": f"127.0.0.1:{args.port}", "MEMRA_CTX": "8192",
                        "MEMRA_MAX_SESSIONS": str(max(4, c))})
            env.pop("MEMRA_PROMPT_FILE", None)
            env.pop("MEMRA_CHAT", None)
            server = boot(args, env, vdir / "server.log")
            with (vdir / "sampler.err").open("xb") as serr:
                sampler = subprocess.Popen([sys.executable, str(SAMPLER), "sample", "--devices", ",".join(leaves + [top]),
                                            "--pid", str(server.pid), "--out", str(vdir / "host.jsonl")],
                                           stdout=subprocess.DEVNULL, stderr=serr)
            time.sleep(0.5)
            t0 = time.monotonic()
            results, errors = run_requests(args.port, items, c, 128)
            window = time.monotonic() - t0
            sampler.send_signal(signal.SIGTERM)
            sampler.wait(timeout=10)
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=120)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            (vdir / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results if r))
            tel = SAMPLE.validate(vdir / "host.jsonl", leaves + [top], 500)
            rec = {"name": name, "arm": arm["name"], "concurrency": c, "pair": pair, "order": order,
                   "regime_ok": regime_ok, "errors": errors, "telemetry_ok": not tel,
                   "server_exit": server.returncode, "summary": visit_summary(results, window)}
            rec["scored"] = bool(regime_ok and not errors and rec["telemetry_ok"])
            (vdir / "visit.json").write_text(json.dumps(rec, indent=1) + "\n")
            visits.append(rec)
            s = rec["summary"]
            print(f"M1-B4 {name} scored={rec['scored']} tok/s={s['tok_per_s']:.2f} ttft_p50={s['ttft_s']['p50']} "
                  f"itl_p99={s['itl_s']['p99']} errors={len(errors)}", flush=True)
    summary = {"visits": len(visits), "arms": args.arms, "regime": args.regime, "qualified": False, "cells": {},
               "verdicts": {}}
    base = arms[0]["name"]
    for c in concurrencies:
        for arm in arms[1:]:
            ratios = {"forward": [], "reverse": []}
            for v in visits:
                if v["arm"] != arm["name"] or v["concurrency"] != c or not v["scored"]:
                    continue
                b = next((x for x in visits if x["arm"] == base and x["concurrency"] == c and x["pair"] == v["pair"]
                          and x["order"] == v["order"] and x["scored"]), None)
                if b:
                    ratios[v["order"]].append(v["summary"]["tok_per_s"] / b["summary"]["tok_per_s"])
            fw, rv = ratios["forward"], ratios["reverse"]
            allr = fw + rv
            if min(len(fw), len(rv)) < 4:
                verdict = "insufficient"
            else:
                med = statistics.median(allr)
                up = min(sum(r > 1 for r in fw), sum(r > 1 for r in rv))
                down = min(sum(r < 1 for r in fw), sum(r < 1 for r in rv))
                verdict = "winner" if med >= 1.05 and up >= 4 else "loser" if med <= 0.95 and down >= 4 else "flat"
            summary["verdicts"][f"c{c}/{arm['name']}"] = {"verdict": verdict, "median_tok_ratio": statistics.median(allr) if allr else None,
                                                          "forward": fw, "reverse": rv}
    for c in concurrencies:
        for arm in arms:
            vs = [v for v in visits if v["concurrency"] == c and v["arm"] == arm["name"] and v["scored"]]
            if not vs:
                continue
            summary["cells"][f"c{c}/{arm['name']}"] = {
                "n": len(vs),
                "median_tok_per_s": statistics.median(v["summary"]["tok_per_s"] for v in vs),
                "median_req_per_s": statistics.median(v["summary"]["req_per_s"] for v in vs),
                **{f"median_{k}_{q}": statistics.median(v["summary"][k][q] for v in vs if v["summary"][k][q] is not None)
                   for k in ("ttft_s", "e2e_s", "tpot_s", "itl_s") for q in ("p50", "p95", "p99")}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("M1-B4-SUMMARY " + json.dumps(summary["cells"]), flush=True)
    for k, v in summary["verdicts"].items():
        print(f"M1-B4-VERDICT {k} vs {base}: {v['verdict']} median_tok_ratio={v['median_tok_ratio']}", flush=True)
    return 0 if all(v["scored"] for v in visits) else 3


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    for name in ("--server", "--artifact", "--arms-lock", "--arms", "--proof", "--out"):
        r.add_argument(name, required=True)
    r.add_argument("--regime", choices=["cold", "warm"], required=True)
    r.add_argument("--rig", choices=["pro-single", "rtx5090"], default="pro-single")
    r.add_argument("--lock-fd", type=int)
    r.add_argument("--pairs", type=int, default=5)
    r.add_argument("--concurrency", default="1,4")
    r.add_argument("--port", type=int, default=18129)
    r.add_argument("--stub-no-lock", action="store_true")
    args = ap.parse_args(argv)
    B.require(args.stub_no_lock or args.lock_fd is not None, "--lock-fd (inherited canonical lock) required")
    B.require(args.stub_no_lock or (args.pairs == 5 and args.concurrency == "1,4"), "registered: 5 pairs, c=1 and 4")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
