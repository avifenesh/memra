#!/usr/bin/env python3
"""Freeze one deterministic serve golden and require repeated byte identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time
import urllib.error
import urllib.request


PROMPT = (
    "Write a deterministic five-bullet checklist for bringing up a GPU inference "
    "server. Use exactly one sentence per bullet."
)


def one_request(args: argparse.Namespace, repeat: int) -> tuple[dict, bytes]:
    body = {
        "model": args.model,
        "temperature": 0,
        "seed": 3407,
        "max_tokens": args.max_tokens,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": f"{args.label}-golden-{repeat}",
    }
    request = urllib.request.Request(
        args.base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    first_visible = None
    pieces: list[str] = []
    usage: dict = {}
    finish_reason = None
    request_id = None
    done = False
    status = None
    error_text = None
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            status = response.status
            for raw_line in response:
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    done = True
                    break
                event = json.loads(payload)
                if event.get("error"):
                    raise RuntimeError(json.dumps(event["error"], sort_keys=True))
                request_id = event.get("id") or request_id
                usage = event.get("usage") or usage
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = (
                        (choice.get("text") or "")
                        + (delta.get("content") or "")
                        + (delta.get("reasoning") or "")
                        + (delta.get("reasoning_content") or "")
                    )
                    if piece:
                        first_visible = first_visible or time.monotonic()
                        pieces.append(piece)
                    finish_reason = choice.get("finish_reason") or finish_reason
    except urllib.error.HTTPError as error:
        status = error.code
        error_text = error.read().decode(errors="replace")[:500]
    except Exception as error:
        error_text = f"{type(error).__name__}: {error}"[:500]
    ended = time.monotonic()
    visible = "".join(pieces).encode()
    cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    row = {
        "repeat": repeat,
        "http_status": status,
        "request_id": request_id,
        "done": done,
        "ttft_s": first_visible - started if first_visible is not None else None,
        "latency_s": ended - started,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": cached_tokens,
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": finish_reason,
        "visible_bytes": len(visible),
        "visible_sha256": hashlib.sha256(visible).hexdigest(),
        "error": error_text,
    }
    row["ok"] = bool(
        status == 200
        and done
        and request_id
        and first_visible is not None
        and visible
        and cached_tokens in (None, 0)
        and not error_text
    )
    return row, visible


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--golden", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.max_tokens < 1:
        parser.error("repeats and max-tokens must be positive")
    if args.out.exists() or args.golden.exists():
        parser.error("refusing to overwrite an existing output")

    rows = []
    outputs = []
    for repeat in range(1, args.repeats + 1):
        row, visible = one_request(args, repeat)
        rows.append(row)
        outputs.append(visible)
        print(json.dumps(row, sort_keys=True), flush=True)

    hashes = sorted({row["visible_sha256"] for row in rows})
    golden = outputs[0] if outputs else b""
    clean = bool(
        len(rows) == args.repeats
        and all(row["ok"] for row in rows)
        and len(hashes) == 1
        and all(output == golden for output in outputs)
    )
    summary = {
        "schema": "memra.percard.serve-golden.v1",
        "label": args.label,
        "model": args.model,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "repeats": args.repeats,
        "matches": sum(output == golden for output in outputs),
        "unique_output_hashes": hashes,
        "golden_sha256": hashlib.sha256(golden).hexdigest(),
        "golden_bytes": len(golden),
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "seed": 3407,
        "cache_namespace": "unique per repeat",
        "verdict": "PASS" if clean else "FAIL",
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.golden.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.golden.write_bytes(golden)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
