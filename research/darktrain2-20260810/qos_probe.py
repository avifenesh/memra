#!/usr/bin/env python3
"""Barrier-released streaming QoS and byte-exactness probe for one serving cell."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import threading
import time
import urllib.request


FILLER = (
    "The operator measures latency, allocator state, checkpoint durability, and exact output "
    "while a lower-priority optimizer yields to interactive traffic. "
)
PROMPT = (
    "In exactly four concise bullets, explain why a GPU background job must yield to an "
    "interactive inference request. Include one point about memory. Context: " + FILLER * 5
)


def proc_state(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return "gone"
    return raw[raw.rfind(")") + 1 :].split()[0]


def nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))]


def one_request(base: str, model: str, max_tokens: int, timeout: float,
                barrier: threading.Barrier, index: int) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 3407,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    barrier.wait()
    t0 = time.monotonic()
    ttft = None
    pieces: list[str] = []
    usage: dict = {}
    rid = None
    finish = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            for raw in response:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                rid = event.get("id") or rid
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = (delta.get("content") or "") + (delta.get("reasoning") or "")
                    if piece:
                        if ttft is None:
                            ttft = time.monotonic() - t0
                        pieces.append(piece)
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        text = "".join(pieces)
        return {
            "index": index,
            "ok": True,
            "rid": rid,
            "ttft_s": ttft,
            "latency_s": time.monotonic() - t0,
            "finish_reason": finish,
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "text_utf8_b64": base64.b64encode(text.encode()).decode(),
            "text_bytes": len(text.encode()),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    except Exception as exc:  # the raw row is the receipt; fail the cell below
        return {
            "index": index,
            "ok": False,
            "ttft_s": ttft,
            "latency_s": time.monotonic() - t0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--watch-pid", type=int)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--golden", type=Path)
    ap.add_argument("--create-golden", action="store_true")
    ap.add_argument("--skip-exactness", action="store_true")
    args = ap.parse_args()

    args.rows.parent.mkdir(parents=True, exist_ok=True)
    barrier = threading.Barrier(args.requests + 1)
    rows: list[dict | None] = [None] * args.requests

    def worker(index: int) -> None:
        rows[index] = one_request(
            args.base, args.model, args.max_tokens, args.timeout, barrier, index
        )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.requests)]
    for thread in threads:
        thread.start()
    watched_before = proc_state(args.watch_pid) if args.watch_pid else None
    release = time.monotonic()
    barrier.wait()
    watched_terminal = None
    watched_latency_s = None
    if args.watch_pid:
        deadline = release + 10
        while time.monotonic() < deadline:
            state = proc_state(args.watch_pid)
            if state == "T" or state == "gone":
                watched_terminal = state
                watched_latency_s = time.monotonic() - release
                break
            time.sleep(0.001)
    for thread in threads:
        thread.join()
    wall_s = time.monotonic() - release
    final_rows = [row for row in rows if row is not None]
    with args.rows.open("w", encoding="utf-8") as fh:
        for row in final_rows:
            fh.write(json.dumps({"label": args.label, **row}, sort_keys=True) + "\n")

    oks = [row for row in final_rows if row.get("ok")]
    errors = [row for row in final_rows if not row.get("ok")]
    ttfts = [float(row["ttft_s"]) for row in oks if row.get("ttft_s") is not None]
    lats = [float(row["latency_s"]) for row in oks]
    hashes = sorted({str(row["text_sha256"]) for row in oks})
    exactness = "skipped"
    expected_sha = None
    if not args.skip_exactness:
        if len(hashes) != 1:
            exactness = "within-cell-mismatch"
        elif args.golden is None:
            exactness = "missing-golden-argument"
        else:
            current = base64.b64decode(oks[0]["text_utf8_b64"]) if oks else b""
            if args.create_golden:
                args.golden.parent.mkdir(parents=True, exist_ok=True)
                args.golden.write_bytes(current)
            if not args.golden.exists():
                exactness = "missing-golden-file"
            else:
                expected = args.golden.read_bytes()
                expected_sha = hashlib.sha256(expected).hexdigest()
                exactness = "match" if current == expected and hashes == [expected_sha] else "mismatch"

    summary = {
        "label": args.label,
        "requests": args.requests,
        "n_ok": len(oks),
        "n_error": len(errors),
        "wall_s": round(wall_s, 6),
        "ttft_p50_s": round(statistics.median(ttfts), 6) if ttfts else None,
        "ttft_p99_s": round(nearest_rank(ttfts, 0.99), 6) if ttfts else None,
        "latency_p50_s": round(statistics.median(lats), 6) if lats else None,
        "latency_p99_s": round(nearest_rank(lats, 0.99), 6) if lats else None,
        "text_sha256": hashes,
        "expected_sha256": expected_sha,
        "exactness": exactness,
        "watch_pid": args.watch_pid,
        "watch_state_before": watched_before,
        "watch_terminal": watched_terminal,
        "watch_latency_ms": round(watched_latency_s * 1000, 3)
        if watched_latency_s is not None
        else None,
        "errors": [row.get("error") for row in errors[:3]],
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    if errors or len(oks) != args.requests:
        return 1
    if not args.skip_exactness and exactness != "match":
        print(f"P0 exactness failure: {exactness}", flush=True)
        return 86
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
