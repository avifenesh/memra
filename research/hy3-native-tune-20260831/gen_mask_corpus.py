#!/usr/bin/env python3
"""Generate a resumable HY3 own-output corpus through Memra's serving API.

Only text emitted by the served HY3 process is later counted for the FR-Spec rank
map. Source rows provide prompts, never the ranked distribution. The request has
no sampling override, so it exercises the model configuration actually served by
this Memra process.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18087")
    parser.add_argument("--model", default="hy3")
    parser.add_argument("--target-tokens", type=int, default=131_072)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=1_200.0)
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


def load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = next(
            (
                str(message.get("content", "")).strip()
                for message in row.get("messages", [])
                if message.get("role") == "user"
                and str(message.get("content", "")).strip()
            ),
            None,
        )
        if prompt is None:
            raise ValueError(f"{path}:{line_number}: no non-empty user message")
        prompts.append(prompt)
    return prompts


def successful_rows(path: Path) -> tuple[set[int], int]:
    done: set[int] = set()
    tokens = 0
    if not path.exists():
        return done, tokens
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "ok":
            continue
        done.add(int(row["index"]))
        tokens += int(row["completion_tokens"])
    return done, tokens


def request_one(
    *,
    endpoint: str,
    model: str,
    index: int,
    prompt: str,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload, separators=(",", ":")).encode()
    url = f"{endpoint.rstrip('/')}/v1/chat/completions"
    started = time.monotonic()
    last_error = ""
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url,
                data=body,
                headers={"content-type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                parsed = json.load(response)
            choice = parsed["choices"][0]
            usage = parsed.get("usage", {})
            text = str(choice["message"].get("content", ""))
            completion_tokens = int(usage.get("completion_tokens", 0))
            if completion_tokens <= 0 or not text:
                raise ValueError("empty completion or missing completion_tokens")
            return {
                "status": "ok",
                "index": index,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "completion_tokens": completion_tokens,
                "server_elapsed_s": usage.get("elapsed_s"),
                "wall_elapsed_s": time.monotonic() - started,
                "finish_reason": choice.get("finish_reason"),
                "text": text,
            }
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            last_error = f"{type(error).__name__}: {error}"
            if attempt < retries:
                time.sleep(1.0 + attempt)
    return {
        "status": "error",
        "index": index,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "completion_tokens": 0,
        "wall_elapsed_s": time.monotonic() - started,
        "error": last_error,
    }


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    prompt_count: int,
    completed: set[int],
    completion_tokens: int,
    run_started: float,
) -> None:
    summary = {
        "format": "memra-hy3-frspec-corpus-v1",
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "endpoint": args.endpoint,
        "model": args.model,
        "sampling_fields": [],
        "rank_distribution_source": "served-model-completions",
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "target_tokens": args.target_tokens,
        "available_prompts": prompt_count,
        "completed_prompts": len(completed),
        "completion_tokens": completion_tokens,
        "target_met": completion_tokens >= args.target_tokens,
        "invocation_wall_s": time.monotonic() - run_started,
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.target_tokens <= 0 or args.max_tokens <= 0 or args.concurrency <= 0:
        raise SystemExit("target-tokens, max-tokens, and concurrency must be positive")
    prompts = load_prompts(args.dataset)
    if args.limit:
        prompts = prompts[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    done, completion_tokens = successful_rows(args.output)
    pending = [(index, prompt) for index, prompt in enumerate(prompts) if index not in done]
    run_started = time.monotonic()
    print(
        f"[hy3-mask-corpus] resume prompts={len(done)}/{len(prompts)} "
        f"tokens={completion_tokens}/{args.target_tokens}",
        flush=True,
    )

    with args.output.open("a", encoding="utf-8") as output:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as executor:
            for offset in range(0, len(pending), args.concurrency):
                if completion_tokens >= args.target_tokens:
                    break
                batch = pending[offset : offset + args.concurrency]
                futures = [
                    executor.submit(
                        request_one,
                        endpoint=args.endpoint,
                        model=args.model,
                        index=index,
                        prompt=prompt,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                        retries=args.retries,
                    )
                    for index, prompt in batch
                ]
                rows = sorted((future.result() for future in futures), key=lambda row: row["index"])
                for row in rows:
                    output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    if row["status"] == "ok":
                        done.add(int(row["index"]))
                        completion_tokens += int(row["completion_tokens"])
                output.flush()
                os.fsync(output.fileno())
                print(
                    f"[hy3-mask-corpus] prompts={len(done)}/{len(prompts)} "
                    f"tokens={completion_tokens}/{args.target_tokens}",
                    flush=True,
                )
                write_summary(
                    args.summary,
                    args=args,
                    prompt_count=len(prompts),
                    completed=done,
                    completion_tokens=completion_tokens,
                    run_started=run_started,
                )

    write_summary(
        args.summary,
        args=args,
        prompt_count=len(prompts),
        completed=done,
        completion_tokens=completion_tokens,
        run_started=run_started,
    )
    if completion_tokens < args.target_tokens:
        raise SystemExit(
            f"prompt set exhausted at {completion_tokens} tokens; "
            f"target is {args.target_tokens}"
        )


if __name__ == "__main__":
    main()
