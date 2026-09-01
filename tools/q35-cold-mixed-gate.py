#!/usr/bin/env python3
"""Q35 cold-prefill regression gate: one frozen mixed90 c=4 sellgate cell."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any


def load_sellgate(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("memra_sellgate_replay", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import sellgate harness from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def public(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def assert_identity(base: str, model: str, timeout: float) -> None:
    """Refuse to measure whatever happens to answer on this port.

    GATE-INTEGRITY-20260819 section 10. `--base` defaults to 127.0.0.1:8177, which is
    tools/serve-smoke.sh's port. This gate is a client, not a binder, so tools/port-guard.sh does
    not apply to it — but the failure mode it was written for is worse for a client: an occupied
    port makes a BINDER fail to bind and die loudly, while a client cheerfully measures the
    stranger and prints numbers. tools/accept-gate.sh:143 records the live incident: the rig's
    idle llama-server held the port and "had that foreign process instead answered 200 with a
    plausible body, the gate would have measured SOMEONE ELSE'S MODEL and pinned it."

    So the first request this gate makes is an identity probe, and a port that cannot name the
    model under test is a refusal — not a warning, and not a retry against a different port.
    """
    url = f"{base}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=min(timeout, 30.0)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            f"identity probe failed: {url} did not answer a model list ({error}). "
            "Boot the server under test, or pass --base."
        ) from error
    served = [
        entry.get("id")
        for entry in (payload.get("data") or [])
        if isinstance(entry, dict)
    ]
    if not served:
        raise ValueError(
            f"identity probe failed: {url} answered with no model ids ({payload!r}). "
            "An empty list is not agreement — something is listening that is not the server "
            "under test."
        )
    if model not in served:
        raise ValueError(
            f"identity probe FAILED: {base} serves {served} and this gate measures {model!r}. "
            "Refusing to run: every number below would come from a different program. Free the "
            "port, boot the right model, or pass --base/--model deliberately."
        )
    print(
        json.dumps(
            {"identity_probe": "ok", "base": base, "model": model, "served": served},
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8177")
    parser.add_argument("--model", default="q35-coldfix")
    parser.add_argument("--namespace", default="serve-smoke-q35-coldfix")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    # BEFORE anything is measured: prove the port belongs to the model under test.
    assert_identity(args.base.rstrip("/"), args.model, args.timeout)

    harness = load_sellgate(repo / "research/sellgate-20260812/sellgate_replay.py")
    workload = harness.load_workload(repo / "research/sellgate-20260812/workload.lock.json")
    expected_shape = {
        "prompt_tokens": 4860,
        "completion_tokens": 60,
        "hit_requests_per_cycle": 9,
        "miss_requests_per_cycle": 1,
        "minimum_requests_per_cell": 20,
    }
    actual_shape = {key: workload.get(key) for key in expected_shape}
    if actual_shape != expected_shape:
        raise ValueError(
            f"frozen workload shape moved: expected {expected_shape}, got {actual_shape}"
        )

    endpoint = harness.Endpoint(label="q35", base=args.base.rstrip("/"), model=args.model)
    goldens: dict[tuple[str, int], str] = {}
    seed_rows, seed_failures = harness.seed_hot_set(
        [endpoint], workload, args.namespace, args.timeout, goldens
    )
    for row in seed_rows:
        print(json.dumps(public(row), sort_keys=True), flush=True)

    requests, samples, cells = harness.run_cell(
        [endpoint], workload, args.namespace, "mixed90", 1, 4, args.timeout, goldens
    )
    for row in [*samples, *requests, *cells]:
        print(json.dumps(row, sort_keys=True), flush=True)

    hits = [row for row in requests if row.get("cache_role") == "hit"]
    misses = [row for row in requests if row.get("cache_role") == "miss"]
    short = [
        {
            "index": row.get("index"),
            "cache_role": row.get("cache_role"),
            "completion_tokens": row.get("completion_tokens"),
            "finish_reason": row.get("finish_reason"),
            "request_id": row.get("request_id"),
            "text_sha256": row.get("text_sha256"),
        }
        for row in requests
        if row.get("completion_tokens") != 60 or row.get("finish_reason") != "length"
    ]
    shape_ok = len(requests) == 20 and len(hits) == 18 and len(misses) == 2
    cell_ok = len(cells) == 1 and bool(cells[0].get("clean"))
    verdict = "PASS" if shape_ok and cell_ok and not seed_failures and not short else "FAIL"
    summary = {
        "kind": "q35_cold_mixed_gate",
        "schema": "memra.q35-cold-mixed-gate.v1",
        "concurrency": 4,
        "requests": len(requests),
        "hit_requests": len(hits),
        "cold_misses": len(misses),
        "expected_completion_tokens": 60,
        "short_or_non_length": short,
        "seed_failures": seed_failures,
        "cell_clean": cell_ok,
        "verdict": verdict,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
