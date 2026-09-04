#!/usr/bin/env python3
"""Eight-turn DSV4 sampled host-cache transparency gate."""

import hashlib
import json
import os
import time
import urllib.request


URL = os.environ.get("DSV4_URL", "http://127.0.0.1:18080/v1/completions")
SOURCE = os.environ.get(
    "DSV4_SOURCE",
    "/root/wt-dsv4f-2card-1m-20260904/crates/memra-engine/src/dsv4_gpu.rs",
)
SEED = int(os.environ.get("DSV4_SEED", "20260904"))


def request(prompt: str, salt: str) -> tuple[dict, str, float]:
    body = {
        "model": "dsv4",
        "prompt": prompt,
        "max_tokens": 32,
        "seed": SEED,
        "cache_salt": salt,
    }
    req = urllib.request.Request(
        URL,
        json.dumps(body).encode(),
        {"content-type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as response:
        data = json.load(response)
    return data, data["choices"][0]["text"], time.perf_counter() - started


def main() -> None:
    with open(SOURCE, encoding="utf-8") as handle:
        excerpt = handle.read(1200)
    prompt = (
        "We are reviewing a native Rust and CUDA inference engine. "
        "Here is a real source excerpt:\n\n"
        + excerpt
        + "\n\nTurn 1: State the most important invariant you see."
    )
    rows = []
    for turn in range(1, 9):
        warm, warm_text, warm_wall = request(prompt, "sampled-seeded-8turn-warm")
        cold, cold_text, cold_wall = request(
            prompt, f"sampled-seeded-8turn-cold-{turn}"
        )
        warm_usage = warm["usage"]
        cold_usage = cold["usage"]
        row = {
            "turn": turn,
            "prompt_tokens": warm_usage["prompt_tokens"],
            "cached_tokens": warm_usage["prompt_tokens_details"]["cached_tokens"],
            "completion_tokens": warm_usage["completion_tokens"],
            "warm_wall_s": round(warm_wall, 6),
            "cold_wall_s": round(cold_wall, 6),
            "warm_engine_s": warm_usage["elapsed_s"],
            "cold_engine_s": cold_usage["elapsed_s"],
            "warm_spec": warm_usage.get("spec"),
            "cold_spec": cold_usage.get("spec"),
            "output_sha256": hashlib.sha256(warm_text.encode()).hexdigest(),
            "identity": warm_text == cold_text,
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
        assert row["identity"], f"turn {turn}: fixed-seed warm/cold output mismatch"
        assert row["warm_spec"] == row["cold_spec"], (
            f"turn {turn}: fixed-seed warm/cold spec telemetry mismatch"
        )
        prompt += (
            warm_text
            + f"\n\nTurn {turn + 1}: Continue the review with one new concrete observation."
        )
    assert all(row["cached_tokens"] > 0 for row in rows[1:])
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "surface": "/v1/completions",
                "sampling": "generation defaults plus fixed identity-gate seed only",
                "seed": SEED,
                "turns": len(rows),
                "rows": rows,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
