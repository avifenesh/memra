#!/usr/bin/env python3
"""Mixed-context admission receipt client.

The server is started at 262k by run-5090-mixed-ctx.sh. This client sends a finite 8k
prompt with no max_ctx, an explicit 128k probe, parks two 256k sessions, then releases a
four-client 128k burst together. Every response is appended as one JSONL row so a
partial/failing run remains inspectable.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import threading
import time
import urllib.error
import urllib.request


def append_row(path: pathlib.Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def fetch_metrics(base: str) -> dict:
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=10) as response:
            payload = response.read().decode("utf-8", "replace")
        return json.loads(payload)
    except Exception as exc:  # Receipt, not a reason to discard the request run.
        return {"metrics_error": f"{type(exc).__name__}: {exc}"}


def stream_request(
    base: str,
    phase: str,
    index: int,
    max_ctx: int | None,
    max_tokens: int,
    release: threading.Barrier | None = None,
    prompt_ids: list[int] | None = None,
) -> dict:
    if release is not None:
        release.wait()
    started = time.time()
    body: dict = {
        "model": "cap",
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "cache_salt": "cap256k",
    }
    if max_ctx is not None:
        body["max_ctx"] = max_ctx
    if prompt_ids is None:
        body["messages"] = [
            {
                "role": "user",
                "content": (
                    f"Admission receipt {phase}-{index}, nonce {time.time_ns()}. "
                    "Explain in one short paragraph why a request-specific KV-cache "
                    "estimate must scale with its context cap."
                ),
            }
        ]
        endpoint = "/v1/chat/completions"
    else:
        body["prompt_ids"] = prompt_ids
        endpoint = "/v1/completions"
    request = urllib.request.Request(
        base + endpoint,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    row = {
        "kind": "request",
        "phase": phase,
        "index": index,
        "max_ctx": max_ctx,
        "max_tokens": max_tokens,
        "requested_prompt_tokens": len(prompt_ids) if prompt_ids is not None else None,
        "started_unix_s": round(started, 6),
        "ok": False,
        "chunks": 0,
        "done": False,
        "finish_reason": None,
    }
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
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
                    err = event["error"]
                    row["server_error"] = (
                        err.get("message", str(err)) if isinstance(err, dict) else str(err)
                    )[:500]
                    continue
                choices = event.get("choices") or [{}]
                finish = choices[0].get("finish_reason")
                if finish:
                    row["finish_reason"] = finish
            if first_byte is not None:
                row["ttfb_s"] = round(first_byte - started, 3)
    except urllib.error.HTTPError as exc:
        row["http_status"] = exc.code
        row["error"] = exc.read().decode("utf-8", "replace")[:500]
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
    row["wall_s"] = round(time.time() - started, 3)
    row["ok"] = (
        row.get("http_status") == 200
        and row["chunks"] > 0
        and row["done"]
        and row["finish_reason"] in ("stop", "length")
        and "bad_json" not in row
        and "server_error" not in row
        and "error" not in row
    )
    return row


def request_charge_contexts(server_log: pathlib.Path) -> list[int]:
    pattern = re.compile(r"\[admission\] request cost: .*? ctx=(\d+) ")
    return [
        int(match.group(1))
        for line in server_log.read_text(encoding="utf-8", errors="replace").splitlines()
        if (match := pattern.search(line)) is not None
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("rows", type=pathlib.Path)
    parser.add_argument("arm", choices=("before", "after"))
    parser.add_argument("server_log", type=pathlib.Path)
    args = parser.parse_args()
    if args.rows.exists():
        raise SystemExit(f"refusing to append a second run to {args.rows}")

    append_row(
        args.rows,
        {
            "kind": "run",
            "arm": args.arm,
            "started_unix_s": round(time.time(), 6),
            "n": 1,
            "sequence": (
                "8k finite prompt without max_ctx; explicit 128k probe; "
                "two 256k parks; c=4 128k burst"
            ),
        },
    )

    prompt_ids = [5_000 + (index % 1_024) for index in range(8_120)]
    prompt8k = stream_request(
        args.base,
        "prompt8k_no_max_ctx",
        0,
        None,
        64,
        prompt_ids=prompt_ids,
    )
    append_row(args.rows, prompt8k)
    if not prompt8k["ok"]:
        print(json.dumps({"phase": "prompt8k_no_max_ctx", "row": prompt8k}, sort_keys=True))
        return 2

    explicit = stream_request(args.base, "explicit128k", 0, 131072, 16)
    append_row(args.rows, explicit)
    if not explicit["ok"]:
        print(json.dumps({"phase": "explicit128k", "row": explicit}, sort_keys=True))
        return 3

    for index in range(2):
        parked = stream_request(args.base, "park", index, 262144, 16)
        append_row(args.rows, parked)
        if not parked["ok"]:
            print(json.dumps({"phase": "park", "row": parked}, sort_keys=True))
            return 4
        time.sleep(0.5)

    append_row(
        args.rows,
        {"kind": "metrics", "phase": "after_park", "value": fetch_metrics(args.base)},
    )

    barrier = threading.Barrier(4)
    burst: list[dict] = []
    burst_lock = threading.Lock()

    def run_burst(index: int) -> None:
        row = stream_request(args.base, "burst", index, 131072, 64, barrier)
        with burst_lock:
            burst.append(row)
            append_row(args.rows, row)

    threads = [threading.Thread(target=run_burst, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    append_row(
        args.rows,
        {"kind": "metrics", "phase": "after_burst", "value": fetch_metrics(args.base)},
    )

    ordered = sorted(burst, key=lambda row: row.get("ttfb_s", float("inf")))
    ttfb = [row["ttfb_s"] for row in burst if "ttfb_s" in row]
    summary = {
        "arm": args.arm,
        "burst_ok": sum(bool(row["ok"]) for row in burst),
        "burst_n": len(burst),
        "ttfb_service_order_s": [row.get("ttfb_s") for row in ordered],
        "ttfb_span_s": round(max(ttfb) - min(ttfb), 3) if len(ttfb) == 4 else None,
        "wall_median_s": round(statistics.median(row["wall_s"] for row in burst), 3),
    }
    charge_contexts = request_charge_contexts(args.server_log)
    expected_contexts = [8_192, 131_072, 262_144]
    missing_contexts = sorted(set(expected_contexts) - set(charge_contexts))
    summary["charge_contexts_in_log"] = sorted(set(charge_contexts))
    summary["expected_charge_contexts"] = expected_contexts
    summary["missing_charge_contexts"] = missing_contexts
    append_row(args.rows, {"kind": "gate", "phase": "charge_contexts", "value": summary})
    print(json.dumps(summary, sort_keys=True))
    if summary["burst_ok"] != 4:
        return 5
    return 0 if not missing_contexts else 6


if __name__ == "__main__":
    raise SystemExit(main())
