#!/usr/bin/env python3
"""Served DSV4 cold-vs-restored identity gate, plain and bundled DSpark."""

import json
import os
import time
import urllib.request


URL = os.environ.get("DSV4_URL", "http://127.0.0.1:18080/v1/completions")
FIXTURE = os.environ.get(
    "DSV4_FIXTURE",
    "/root/ds4f-fixtures/fixtures-0731/dsv4_0731_fixtures_ref.json",
)


def request(body: dict) -> tuple[dict, float]:
    req = urllib.request.Request(
        URL,
        json.dumps(body).encode(),
        {"content-type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response), time.perf_counter() - started


def response_identity(response: dict) -> tuple:
    choice = response["choices"][0]
    return (
        choice.get("text"),
        choice.get("finish_reason"),
        response["usage"]["completion_tokens"],
    )


def run_arm(name: str, prompt: list[int], suffix: list[int], extra: dict) -> dict:
    common = {
        "model": "dsv4",
        "temperature": 0,
        "ignore_eos": True,
        **extra,
    }
    leader, leader_wall = request(
        {
            **common,
            "prompt_ids": prompt,
            "max_tokens": 1,
            "cache_salt": f"{name}-restored",
        }
    )
    extended = prompt + suffix
    restored, restored_wall = request(
        {
            **common,
            "prompt_ids": extended,
            "max_tokens": 64,
            "cache_salt": f"{name}-restored",
        }
    )
    cold, cold_wall = request(
        {
            **common,
            "prompt_ids": extended,
            "max_tokens": 64,
            "cache_salt": f"{name}-cold",
        }
    )
    leader_cached = leader["usage"]["prompt_tokens_details"]["cached_tokens"]
    restored_cached = restored["usage"]["prompt_tokens_details"]["cached_tokens"]
    cold_cached = cold["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert leader_cached == 0, (name, "leader cached", leader_cached)
    assert restored_cached == len(prompt), (
        name,
        "restored cached",
        restored_cached,
        len(prompt),
    )
    assert cold_cached == 0, (name, "cross-salt cold", cold_cached)
    assert response_identity(restored) == response_identity(cold), (
        name,
        "restored output differs from cold",
    )
    restored_spec = restored["usage"].get("spec")
    cold_spec = cold["usage"].get("spec")
    if name == "dspark":
        assert restored_spec and restored_spec["rounds"] > 0
        assert cold_spec and cold_spec["rounds"] > 0
        assert restored_spec == cold_spec, (name, "spec telemetry differs")
    else:
        assert restored_spec is None and cold_spec is None, (
            name,
            "penalized greedy must use plain",
        )
    return {
        "arm": name,
        "prompt": len(prompt),
        "suffix": len(suffix),
        "completion": restored["usage"]["completion_tokens"],
        "cached": [leader_cached, restored_cached, cold_cached],
        "wall_s": {
            "leader": round(leader_wall, 6),
            "restored": round(restored_wall, 6),
            "cold": round(cold_wall, 6),
        },
        "engine_elapsed_s": {
            "restored": restored["usage"]["elapsed_s"],
            "cold": cold["usage"]["elapsed_s"],
        },
        "spec": restored_spec,
        "identity": "byte-equal response text/finish/count",
    }


def main() -> None:
    with open(FIXTURE, encoding="utf-8") as handle:
        fixture = json.load(handle)
    prompt = fixture["tokens_160"]
    suffix = fixture["tokens_32"][1:12]
    rows = [
        run_arm("dspark", prompt, suffix, {}),
        run_arm("plain", prompt, suffix, {"repetition_penalty": 1.01}),
    ]
    print(json.dumps({"verdict": "PASS", "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
