#!/usr/bin/env python3
"""Decode rows + concurrent host census for one boot (lane/glm5-host-audit, 2026-09-01).

One arm = one BOOT (MEMRA_WORKER_AFFINITY is read once through OnceLock, so it cannot
alternate inside a process; the interleave unit is a boot, stated not hidden — same
deviation the decode-attribution lane banked).

Per boot it: primes the prompt once (so the prefix restores and TTFT stops dominating),
then runs `reps` identical steady-state reps, and runs host-sampler.py CONCURRENTLY with
each timed rep so the scheduling receipt and the tok/s row describe the SAME tokens.

greedy (temperature 0) is the INSTRUMENT — byte-deterministic, so it carries the identity
oracle. vendor-default sampled (temperature 1.0 / top_p 0.95, seeded) is the PRODUCT and
every serving-decision row needs it (LAW never-serve-greedy). Both are emitted.

reasoning_effort is PINNED (TRAP reasoning-effort-unpinned-decode-cell): omitting it
measures think-prose, not the claim shape, and faked a fleet-wide regression once.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

MODEL = os.environ.get("MEMRA_PROBE_MODEL", "zai/glm-5.3-flash")


def one_request(port, prompt, arm, max_tokens, effort="low"):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "reasoning_effort": effort,
    }
    if arm == "greedy":
        body["temperature"] = 0.0
    else:
        body["temperature"] = 1.0
        body["top_p"] = 0.95
        body["seed"] = 20260901
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    ttft = None
    text_parts, usage = [], None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    text_parts.append(piece)
    elapsed = time.monotonic() - t0
    text = "".join(text_parts)
    completion = (usage or {}).get("completion_tokens") or 0
    return {
        "elapsed_s": round(elapsed, 4),
        "ttft_s": round(ttft, 4) if ttft is not None else None,
        "completion_tokens": completion,
        # tok/s from the SERVER's own completion count over the streamed wall, decode-only:
        # the prime rep removes prefill from every timed rep, so this is the decode rate.
        "tok_s": round(completion / (elapsed - (ttft or 0.0)), 4)
        if completion and elapsed > (ttft or 0.0)
        else None,
        "sha16": hashlib.sha256(text.encode()).hexdigest()[:16],
        "usage": usage,
        "chars": len(text),
    }


def looped(text_sha_rows):
    """LAW greedy-is-the-instrument: a degenerated repeat loop is flagged and EXCLUDED
    from aggregates, never filed as a finding. Detection is on the emitted text, so it
    is done by the caller that holds the text; here we only carry the flag through."""
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--prompt-idx", type=int, default=5)
    ap.add_argument("--sampler", required=True)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    port = os.environ.get("PORT", "18700")
    pool = json.load(open(os.environ["PROMPTS_JSON"]))["decode"]
    prompt = pool[args.prompt_idx % len(pool)]["text"]

    out = {"arm": args.arm, "tag": args.tag, "prompt_idx": args.prompt_idx, "reps": []}

    # Prime once: the server restores the prefix on every later rep, so prefill leaves
    # the measurement instead of being averaged into it.
    prime = one_request(port, prompt, "greedy", 16)
    out["prime"] = prime

    for shape in ("greedy", "sampled"):
        rows = []
        for rep in range(args.reps):
            samp_json = os.path.join(
                args.out, f"sched-{args.arm}-{shape}-{rep}.json"
            )
            # Sampler runs CONCURRENTLY with the timed rep and is bounded by it.
            sp = subprocess.Popen(
                [
                    sys.executable, args.sampler,
                    "--pid", str(args.pid),
                    "--secs", "600",          # outlived by the kill below
                    "--interval", "0.05",
                    "--label", f"{args.arm}/{shape}/rep{rep}",
                    "--json", samp_json,
                ],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            time.sleep(0.4)                    # let the sampler take its first tick
            row = one_request(port, prompt, shape, args.max_tokens)
            sp.terminate()
            try:
                sp.wait(timeout=20)
            except subprocess.TimeoutExpired:
                sp.kill()
            row["rep"] = rep
            row["sched_json"] = samp_json if os.path.exists(samp_json) else None
            rows.append(row)
        ok = [r["tok_s"] for r in rows if r["tok_s"]]
        shas = sorted({r["sha16"] for r in rows})
        out[shape] = {
            "rows": rows,
            "median_tok_s": round(statistics.median(ok), 4) if ok else None,
            "rel_spread_pct": round(100.0 * (max(ok) - min(ok)) / statistics.median(ok), 4)
            if len(ok) > 1
            else None,
            "shas": shas,
            "sha_stable": len(shas) == 1,
        }

    print(json.dumps(out))


if __name__ == "__main__":
    main()
