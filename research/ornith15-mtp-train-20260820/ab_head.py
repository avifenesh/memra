#!/usr/bin/env python3
"""Serve-level A/B: vendor MTP head vs continued-trained head, ST path.

Each arm = memra-server over an ST dir (spec default ON for MTP checkpoints),
identical greedy 256-token probes, acceptance read programmatically from the
response's `usage.spec` extension ({rounds, drafted, accepted}) plus wall time.
Arms run interleaved in BOTH orders (V,T / T,V pairs) with a fresh server per
run; a final spec-off run anchors the plain-decode reference. Contended box —
ratios only, absolutes are not bankable.
"""
import argparse
import json
import os
import pathlib
import signal
import subprocess
import time
import urllib.request

PROBES = [
    ("code", "Write a Python function that parses an ISO-8601 timestamp string and "
             "returns seconds since the Unix epoch, handling timezone offsets. Include tests."),
    ("agentic", "You are a coding agent in a Rust repository. The test "
                "`test_kv_append_wrap` fails with an off-by-one at the ring boundary. "
                "Describe, step by step, how you would locate and fix the bug, then give the patch."),
]


def start_server(bin_path, model_dir, addr, log_path, spec_on):
    env = dict(os.environ)
    env["MEMRA_MODELS"] = f"ornith15={model_dir}"
    env["MEMRA_ADDR"] = addr
    if not spec_on:
        env["MEMRA_SERVE_SPEC"] = "0"
    log = open(log_path, "ab")
    proc = subprocess.Popen([bin_path], env=env, stdout=log, stderr=log)
    base = f"http://{addr}"
    for _ in range(180):
        try:
            urllib.request.urlopen(base + "/v1/models", timeout=2)
            return proc
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError(f"server died, see {log_path}")
            time.sleep(2)
    proc.send_signal(signal.SIGKILL)
    raise RuntimeError("server not healthy after 360s")


def run_probes(addr, max_tokens=256):
    out = []
    for name, prompt in PROBES:
        body = {"model": "ornith15",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": 0}
        req = urllib.request.Request(f"http://{addr}/v1/chat/completions",
                                     json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=900) as resp:
            j = json.loads(resp.read())
        dt = time.time() - t0
        usage = j.get("usage", {})
        spec = usage.get("spec") or {}
        out.append({"probe": name, "elapsed_s": round(dt, 3),
                    "completion_tokens": usage.get("completion_tokens"),
                    "tok_s": round((usage.get("completion_tokens") or 0) / dt, 2),
                    "spec": spec,
                    "acc_rate": round(spec["accepted"] / spec["drafted"], 4)
                    if spec.get("drafted") else None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True,
                    help="comma list name=dir[,name=dir...]; interleaved round-robin, order rotated per round")
    ap.add_argument("--server-bin", default=os.path.expanduser("~/memra-src/target/release/memra-server"))
    ap.add_argument("--addr", default="127.0.0.1:8095")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--pairs", type=int, default=3)
    args = ap.parse_args()

    log_dir = args.out.parent
    results = []

    def one_run(tag, model_dir, spec_on=True):
        proc = start_server(args.server_bin, model_dir, args.addr,
                            log_dir / f"ab-server-{tag}.log", spec_on)
        try:
            rows = run_probes(args.addr)
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        for r in rows:
            r.update({"arm": tag.rsplit("-", 1)[0], "run": tag})
            results.append(r)
            print(json.dumps(r), flush=True)

    arms = [(kv.split("=", 1)[0], kv.split("=", 1)[1]) for kv in args.arms.split(",")]
    for rnd in range(args.pairs):
        order = arms[rnd % len(arms):] + arms[:rnd % len(arms)]  # rotate start per round
        for arm, d in order:
            one_run(f"{arm}-{rnd}", d)
    one_run("plain-0", arms[0][1], spec_on=False)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary = {}
    for arm in [a for a, _ in arms] + ["plain"]:
        rows = [r for r in results if r["arm"] == arm]
        if not rows:
            continue
        accs = [r["acc_rate"] for r in rows if r["acc_rate"] is not None]
        summary[arm] = {
            "n": len(rows),
            "median_tok_s": sorted(r["tok_s"] for r in rows)[len(rows) // 2],
            "mean_acc": round(sum(accs) / len(accs), 4) if accs else None,
        }
    print("SUMMARY", json.dumps(summary))
    (log_dir / "ab-summary.json").write_text(json.dumps(summary, indent=1))
    print("AB DONE")


if __name__ == "__main__":
    main()
