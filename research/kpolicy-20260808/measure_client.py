#!/usr/bin/env python3
"""Measure one request-conditioned K-policy cell through the serving API."""

import argparse
import hashlib
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request


def request(base, model, prompt, cache_salt, max_tokens, raw_path):
    body = {
        "model": model,
        "prompt": prompt,
        "cache_salt": cache_salt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    req = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=1200) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raw_path.with_suffix(raw_path.suffix + ".error").write_text(detail)
        raise RuntimeError(f"HTTP {exc.code}: {detail[:500]}") from exc
    elapsed = time.monotonic() - started
    raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload, elapsed


def completion_text(payload):
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0].get("text"), str):
        raise RuntimeError("response has no choices[0].text")
    return choices[0]["text"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--class", dest="prompt_class", required=True,
                        choices=("cold-short", "cold-long", "cached-long"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--rep", required=True, type=int)
    parser.add_argument("--out", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--setup-tokens", type=int, default=64)
    args = parser.parse_args()

    prompt = pathlib.Path(args.prompt).read_text()
    raw_dir = pathlib.Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    label = f"{args.model}-{args.prompt_class}-k{args.k}-r{args.rep}"
    cache_salt = f"kpolicy-{label}"
    setup = None

    if args.prompt_class == "cached-long":
        setup_path = raw_dir / f"{label}-setup.json"
        setup, setup_wall = request(
            args.base,
            args.model,
            prompt,
            cache_salt,
            args.setup_tokens,
            setup_path,
        )
        setup_text = completion_text(setup)
        if not setup_text:
            raise RuntimeError("cached-long setup returned empty text")
        prompt = (
            prompt
            + setup_text
            + "\n\nContinue from the prior answer. State the most important remaining "
              "regression test in one paragraph.\n"
        )
    else:
        setup_wall = None

    response_path = raw_dir / f"{label}.json"
    payload, wall_s = request(
        args.base,
        args.model,
        prompt,
        cache_salt,
        args.max_tokens,
        response_path,
    )
    usage = payload.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    spec = usage.get("spec")
    completion_tokens = int(usage.get("completion_tokens") or 0)
    server_elapsed_s = float(usage.get("elapsed_s") or 0.0)
    if completion_tokens <= 0:
        raise RuntimeError("measured request returned zero completion tokens")
    if args.k == 0 and spec is not None:
        raise RuntimeError("K=0 response unexpectedly carried spec telemetry")
    if args.k > 0 and spec is None:
        raise RuntimeError(f"K={args.k} response carried no spec telemetry")

    cached_tokens = int(details.get("cached_tokens") or 0)
    if args.prompt_class == "cached-long" and cached_tokens < 1024:
        raise RuntimeError(
            f"cached-long resumed only {cached_tokens} tokens; expected at least 1024"
        )
    if args.prompt_class != "cached-long" and cached_tokens != 0:
        raise RuntimeError(
            f"{args.prompt_class} unexpectedly resumed {cached_tokens} tokens"
        )

    text = completion_text(payload)
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": label,
        "model": args.model,
        "class": args.prompt_class,
        "k": args.k,
        "rep": args.rep,
        "cache_salt": cache_salt,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "wall_s": wall_s,
        "server_elapsed_s": server_elapsed_s,
        "net_tok_s": completion_tokens / wall_s,
        "server_tok_s": (
            completion_tokens / server_elapsed_s if server_elapsed_s > 0 else None
        ),
        "spec": spec,
        "acceptance_rate": spec.get("acceptance_rate") if spec else None,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "setup_wall_s": setup_wall,
        "setup_prompt_tokens": (
            (setup.get("usage") or {}).get("prompt_tokens") if setup else None
        ),
        "setup_completion_tokens": (
            (setup.get("usage") or {}).get("completion_tokens") if setup else None
        ),
        "raw_response": str(response_path),
    }
    with open(args.out, "a") as output:
        output.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"MEASURE-FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
