#!/usr/bin/env python3
"""Run one mixed prompt/cache/concurrency workload against memra-server."""

import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import threading
import time
import urllib.error
import urllib.request


def post(base, model, name, prompt, cache_salt, max_tokens, raw_dir):
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
        (raw_dir / f"{name}.error").write_text(detail)
        raise RuntimeError(f"{name}: HTTP {exc.code}: {detail[:500]}") from exc
    wall_s = time.monotonic() - started
    (raw_dir / f"{name}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload, wall_s


def record(name, payload, wall_s):
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0].get("text"), str):
        raise RuntimeError(f"{name}: response has no choices[0].text")
    usage = payload.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    if completion_tokens <= 0:
        raise RuntimeError(f"{name}: zero completion tokens")
    server_elapsed_s = float(usage.get("elapsed_s") or 0.0)
    spec = usage.get("spec")
    text = choices[0]["text"]
    return {
        "name": name,
        "wall_s": wall_s,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "cached_tokens": int(details.get("cached_tokens") or 0),
        "completion_tokens": completion_tokens,
        "server_elapsed_s": server_elapsed_s,
        "mode": "spec" if spec else "plain",
        "acceptance_rate": spec.get("acceptance_rate") if spec else None,
        "spec": spec,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def require_cache(row):
    if row["cached_tokens"] < 1024:
        raise RuntimeError(
            f"{row['name']}: cached_tokens={row['cached_tokens']}, expected >=1024"
        )


def require_cold(row):
    if row["cached_tokens"] != 0:
        raise RuntimeError(
            f"{row['name']}: cached_tokens={row['cached_tokens']}, expected 0"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", default="q9")
    parser.add_argument("--short", required=True)
    parser.add_argument("--long", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--raw-dir", required=True)
    args = parser.parse_args()

    short_prompt = pathlib.Path(args.short).read_text()
    long_prompt = pathlib.Path(args.long).read_text()
    raw_dir = pathlib.Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.arm}-r{args.rep}"

    # One unmeasured warmup per independent server boot. It uses a private namespace and
    # finishes before the workload clock starts.
    post(
        args.base,
        args.model,
        f"{prefix}-warmup",
        short_prompt,
        f"kpolicy-mixed-{prefix}-warmup",
        32,
        raw_dir,
    )

    workload_started = time.monotonic()
    rows = []

    payload, wall = post(
        args.base,
        args.model,
        f"{prefix}-seq-short",
        short_prompt,
        f"kpolicy-mixed-{prefix}-seq-short",
        128,
        raw_dir,
    )
    seq_short = record("seq-short", payload, wall)
    require_cold(seq_short)
    rows.append(seq_short)

    payload, wall = post(
        args.base,
        args.model,
        f"{prefix}-seq-long",
        long_prompt,
        f"kpolicy-mixed-{prefix}-seq-long",
        128,
        raw_dir,
    )
    seq_long = record("seq-long", payload, wall)
    require_cold(seq_long)
    rows.append(seq_long)

    cached_salt = f"kpolicy-mixed-{prefix}-cached"
    payload, wall = post(
        args.base,
        args.model,
        f"{prefix}-cached-setup",
        long_prompt,
        cached_salt,
        64,
        raw_dir,
    )
    cached_setup = record("cached-setup", payload, wall)
    require_cold(cached_setup)
    rows.append(cached_setup)
    if not cached_setup["text"]:
        raise RuntimeError("cached-setup: empty text")

    cached_prompt = (
        long_prompt
        + cached_setup["text"]
        + "\n\nContinue the analysis with the most important regression test.\n"
    )
    payload, wall = post(
        args.base,
        args.model,
        f"{prefix}-seq-cached",
        cached_prompt,
        cached_salt,
        128,
        raw_dir,
    )
    seq_cached = record("seq-cached", payload, wall)
    require_cache(seq_cached)
    rows.append(seq_cached)
    if not seq_cached["text"]:
        raise RuntimeError("seq-cached: empty text")

    wave_cached_prompt = (
        cached_prompt
        + seq_cached["text"]
        + "\n\nNow give the minimal implementation order in five bullets.\n"
    )
    wave_specs = [
        (
            "wave-short-a",
            short_prompt,
            f"kpolicy-mixed-{prefix}-wave-short-a",
            0.0,
        ),
        ("wave-cached", wave_cached_prompt, cached_salt, 0.01),
        (
            "wave-long",
            long_prompt,
            f"kpolicy-mixed-{prefix}-wave-long",
            0.02,
        ),
        (
            "wave-short-b",
            short_prompt,
            f"kpolicy-mixed-{prefix}-wave-short-b",
            0.02,
        ),
    ]
    barrier = threading.Barrier(len(wave_specs))

    def wave_request(spec):
        name, prompt, cache_salt, delay = spec
        barrier.wait()
        time.sleep(delay)
        payload, wall_s = post(
            args.base,
            args.model,
            f"{prefix}-{name}",
            prompt,
            cache_salt,
            256,
            raw_dir,
        )
        return record(name, payload, wall_s)

    wave_started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        wave_rows = list(executor.map(wave_request, wave_specs))
    wave_wall_s = time.monotonic() - wave_started
    for row in wave_rows:
        if row["name"] == "wave-cached":
            require_cache(row)
        else:
            require_cold(row)
    rows.extend(wave_rows)

    workload_wall_s = time.monotonic() - workload_started
    completion_total = sum(row["completion_tokens"] for row in rows)
    result = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arm": args.arm,
        "rep": args.rep,
        "requests": len(rows),
        "completion_tokens_total": completion_total,
        "prompt_tokens_total": sum(row["prompt_tokens"] for row in rows),
        "cached_tokens_total": sum(row["cached_tokens"] for row in rows),
        "workload_wall_s": workload_wall_s,
        "wave_wall_s": wave_wall_s,
        "agg_tok_s": completion_total / workload_wall_s,
        "wave_tok_s": (
            sum(row["completion_tokens"] for row in wave_rows) / wave_wall_s
        ),
        "rows": [{key: value for key, value in row.items() if key != "text"} for row in rows],
    }
    with open(args.out, "a") as output:
        output.write(json.dumps(result, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
