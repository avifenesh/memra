#!/usr/bin/env python3
"""Drive the forward and inverse mixed-context admission receipts."""

from __future__ import annotations

import argparse
import json
import pathlib
import threading
import time
import urllib.error
import urllib.request


def append(path: pathlib.Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


def metrics(base: str) -> dict:
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=10) as response:
            return json.load(response)
    except Exception as error:
        return {"metrics_error": f"{type(error).__name__}: {error}"}


def request(
    base: str,
    order: str,
    phase: str,
    index: int,
    max_ctx: int,
    max_tokens: int,
    barrier: threading.Barrier | None = None,
) -> dict:
    if barrier is not None:
        barrier.wait()
    started = time.time()
    body = {
        "model": "step",
        "messages": [
            {
                "role": "user",
                "content": (
                    f"val256 admission {order}/{phase}/{index}, nonce {time.time_ns()}. "
                    "In one short sentence, distinguish a context cap from prompt length."
                ),
            }
        ],
        "max_ctx": max_ctx,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "cache_salt": f"val256-{order}",
        "session_id": f"val256-{order}-{phase}-{index}",
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    row = {
        "kind": "request",
        "order": order,
        "phase": phase,
        "index": index,
        "max_ctx": max_ctx,
        "max_tokens": max_tokens,
        "started_unix_s": round(started, 6),
        "ok": False,
        "chunks": 0,
        "done": False,
        "finish_reason": None,
    }
    try:
        with urllib.request.urlopen(req, timeout=14400) as response:
            row["http_status"] = response.status
            first_byte = None
            for raw in response:
                if first_byte is None:
                    first_byte = time.time()
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    row["done"] = True
                    break
                row["chunks"] += 1
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    row["bad_json"] = payload[:300]
                    continue
                if "error" in event:
                    error = event["error"]
                    row["server_error"] = (
                        error.get("message", str(error))
                        if isinstance(error, dict)
                        else str(error)
                    )[:500]
                    continue
                choices = event.get("choices") or [{}]
                if choices[0].get("finish_reason"):
                    row["finish_reason"] = choices[0]["finish_reason"]
            if first_byte is not None:
                row["ttfb_s"] = round(first_byte - started, 3)
    except urllib.error.HTTPError as error:
        row["http_status"] = error.code
        row["error"] = error.read().decode("utf-8", "replace")[:500]
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"[:500]
    row["wall_s"] = round(time.time() - started, 3)
    row["ok"] = (
        row.get("http_status") == 200
        and row["chunks"] > 0
        and row["done"]
        and row["finish_reason"] in ("stop", "length")
        and not any(name in row for name in ("bad_json", "server_error", "error"))
    )
    return row


def burst(
    base: str,
    rows: pathlib.Path,
    order: str,
    phase: str,
    count: int,
    max_ctx: int,
    max_tokens: int,
) -> list[dict]:
    release = threading.Barrier(count)
    result: list[dict] = []
    lock = threading.Lock()

    def one(index: int) -> None:
        row = request(base, order, phase, index, max_ctx, max_tokens, release)
        with lock:
            result.append(row)
            append(rows, row)

    threads = [threading.Thread(target=one, args=(index,)) for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("rows", type=pathlib.Path)
    parser.add_argument("order", choices=("forward", "inverse"))
    args = parser.parse_args()
    if args.rows.exists():
        raise SystemExit(f"refusing to append a second run to {args.rows}")

    append(
        args.rows,
        {
            "kind": "run",
            "order": args.order,
            "n": 1,
            "server_ctx": 262144,
            "sequence": (
                "8k calibrator; two 256k parks; c=4 128k burst"
                if args.order == "forward"
                else "256k first; c=4 8k burst"
            ),
        },
    )

    if args.order == "forward":
        calibrator = request(args.base, args.order, "calibrator", 0, 8192, 16)
        append(args.rows, calibrator)
        if not calibrator["ok"]:
            return 2
        for index in range(2):
            parked = request(args.base, args.order, "park", index, 262144, 16)
            append(args.rows, parked)
            if not parked["ok"]:
                return 3
        append(args.rows, {"kind": "metrics", "phase": "after_park", "value": metrics(args.base)})
        result = burst(args.base, args.rows, args.order, "burst128k", 4, 131072, 64)
    else:
        first = request(args.base, args.order, "first256k", 0, 262144, 16)
        append(args.rows, first)
        if not first["ok"]:
            return 4
        append(args.rows, {"kind": "metrics", "phase": "after_first", "value": metrics(args.base)})
        result = burst(args.base, args.rows, args.order, "small8k", 4, 8192, 64)

    append(args.rows, {"kind": "metrics", "phase": "final", "value": metrics(args.base)})
    ok = sum(bool(row["ok"]) for row in result)
    append(args.rows, {"kind": "summary", "order": args.order, "burst_ok": ok, "burst_n": len(result)})
    return 0 if ok == len(result) else 5


if __name__ == "__main__":
    raise SystemExit(main())
