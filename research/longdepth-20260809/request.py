#!/usr/bin/env python3
"""Issue one deterministic native memra completion and preserve its exact response."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", default="step35")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--max-tokens", required=True, type=int)
    parser.add_argument("--temperature", required=True, choices=("0", "0.7"))
    parser.add_argument("--top-p", required=True, choices=("0.9", "1"))
    parser.add_argument("--cell", required=True)
    parser.add_argument("--rep", required=True, type=int)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    prompt = pathlib.Path(args.prompt_file).read_text()
    body = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": 0,
        "min_p": 0.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        # Two different but frozen seeds make sampled repetitions independent and replayable.
        # The seed is immaterial in the greedy cells.
        "seed": 2026080900 + args.rep,
        # prompt-file is already rendered through the artifact's Step35 chat template with
        # Reasoning: low. The native completion surface preserves exact generated token ids.
        "chat": False,
        "stream": False,
        # Each repetition is a fresh cache trajectory. The server also has all reuse pools off,
        # but the namespace makes that property explicit in the request receipt.
        "cache_salt": f"longdepth-{args.cell}-rep{args.rep}",
    }
    request_bytes = canonical_bytes(body)
    (out / "request.json").write_bytes(request_bytes)

    request = urllib.request.Request(
        args.base.rstrip("/") + "/v1/completions",
        data=request_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_wall = time.time()
    started_mono = time.monotonic()
    status = None
    response_headers: dict[str, str] = {}
    raw = b""
    error = None
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            status = response.status
            response_headers = dict(response.headers.items())
            raw = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        response_headers = dict(exc.headers.items())
        raw = exc.read()
        error = f"HTTPError: {exc}"
    except Exception as exc:  # Preserve the exact client exception; the server log is separate.
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - started_mono

    (out / "response.raw").write_bytes(raw)
    parsed = None
    if raw:
        try:
            parsed = json.loads(raw)
        except Exception as exc:
            error = error or f"response JSON decode failed: {type(exc).__name__}: {exc}"

    meta: dict[str, object] = {
        "cell": args.cell,
        "rep": args.rep,
        "started_unix_s": started_wall,
        "elapsed_client_s": elapsed,
        "http_status": status,
        "response_headers": response_headers,
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "temperature": body["temperature"],
        "top_p": body["top_p"],
        "seed": body["seed"],
        "error": error,
    }

    if isinstance(parsed, dict):
        (out / "response.json").write_bytes(canonical_bytes(parsed))
        text = parsed.get("text")
        tokens = parsed.get("tokens")
        if isinstance(text, str):
            (out / "completion.txt").write_text(text)
        else:
            error = error or "native response missing string field 'text'"
        if isinstance(tokens, list) and all(isinstance(token, int) for token in tokens):
            (out / "tokens.txt").write_text(" ".join(map(str, tokens)) + "\n")
        else:
            error = error or "native response missing integer array field 'tokens'"
        meta.update({
            "model": parsed.get("model"),
            "stop_reason": parsed.get("stop_reason"),
            "n_tokens": parsed.get("n_tokens"),
            "prompt_tokens": parsed.get("prompt_tokens"),
            "cached_tokens": parsed.get("cached_tokens"),
            "elapsed_server_s": parsed.get("elapsed_s"),
            "text_bytes": len(text.encode()) if isinstance(text, str) else None,
            "token_array_len": len(tokens) if isinstance(tokens, list) else None,
        })

    meta["error"] = error
    (out / "request-meta.json").write_bytes(canonical_bytes(meta))
    print(json.dumps(meta, sort_keys=True, ensure_ascii=False), flush=True)
    if error or status != 200:
        return 1
    if meta.get("n_tokens") != meta.get("token_array_len"):
        print("native response n_tokens does not match tokens array", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
