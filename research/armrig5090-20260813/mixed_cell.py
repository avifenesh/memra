#!/usr/bin/env python3
"""Run one local-5090 frozen sellgate mixed90 cell.

This imports the byte-frozen sellgate replay and calls its seed/run primitives
for the one local endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("cx_sellgate_replay", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def public(row: dict[str, Any], label: str, policy_arm: str) -> dict[str, Any]:
    return {
        **{key: value for key, value in row.items() if not key.startswith("_")},
        "label": label,
        "policy_arm": policy_arm,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target", choices=("q27",), required=True)
    parser.add_argument("--policy-arm", choices=("repaired", "eager"), required=True)
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--workload-lock", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()

    if args.rep < 1 or args.concurrency < 1:
        parser.error("rep and concurrency must be positive")
    if not args.module.is_file() or not args.workload_lock.is_file():
        parser.error("module and workload lock must exist")

    replay = load_module(args.module)
    workload = replay.load_workload(args.workload_lock)
    endpoint = replay.Endpoint(args.target, args.base.rstrip("/"), args.model)
    goldens: dict[tuple[str, int], str] = {}

    protocol = {
        "kind": "single_mixed_protocol",
        "schema": "memra.gscost.mixed.v1",
        "label": args.label,
        "target": args.target,
        "policy_arm": args.policy_arm,
        "rep": args.rep,
        "concurrency": args.concurrency,
        "module_sha256": sha256_file(args.module),
        "workload_lock_sha256": sha256_file(args.workload_lock),
        "shape": (
            "one endpoint on GPU0; frozen 4860+60 sellgate prompt; eight hot entries; "
            "nine full-prefix hits plus one real miss per ten requests"
        ),
    }

    seed_rows, seed_failures = replay.seed_hot_set(
        [endpoint], workload, args.namespace, args.timeout, goldens
    )
    request_rows, samples, cells = replay.run_cell(
        [endpoint],
        workload,
        args.namespace,
        "mixed90",
        args.rep,
        args.concurrency,
        args.timeout,
        goldens,
    )
    if len(cells) != 1:
        raise RuntimeError(f"expected one cell, got {len(cells)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as output:
        output.write(json.dumps(protocol, sort_keys=True) + "\n")
        for row in seed_rows:
            output.write(json.dumps(public(row, args.label, args.policy_arm), sort_keys=True) + "\n")
        for row in [*samples, *request_rows, *cells]:
            output.write(json.dumps(public(row, args.label, args.policy_arm), sort_keys=True) + "\n")

    result = public(cells[0], args.label, args.policy_arm)
    result["seed_failures"] = seed_failures
    result["seed_rows"] = len(seed_rows)
    print(json.dumps(result, sort_keys=True), flush=True)
    if seed_failures:
        for failure in seed_failures:
            print(json.dumps({"kind": "seed_failure", "error": failure}), flush=True)
    return 0 if not seed_failures and bool(cells[0]["clean"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
