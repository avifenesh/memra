#!/usr/bin/env python3
"""Reduce the single-card requalification and diff the sold envelopes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


FROZEN_REDUCER_SHA256 = "eb231926686c3f97a69f8b023bff2f1ea19ed41b2e0b76b9ca5283aa69822d09"
WORKLOAD_SHA256 = "85597a0a28ed874f440b4a966c0b43fd3e31b94fe868266de9e299decc208c34"
LEVELS_BY_TARGET = {
    "q27": [1, 2, 4, 8, 12, 16, 20],
    "q35": [1, 2, 4, 8, 16, 32, 40, 48],
}
KNEE_GRID_BY_TARGET = {
    "q27": [4, 8, 12, 16, 20],
    "q35": [4, 16, 32, 40, 48],
}
SOLD_CAP = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: expected JSON object")
        rows.append(value)
    return rows


def load_frozen_reducer(path: Path) -> ModuleType:
    if sha256_file(path) != FROZEN_REDUCER_SHA256:
        raise ValueError(f"{path}: frozen reducer hash mismatch")
    spec = importlib.util.spec_from_file_location("frozen_sellgate_reduce", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import frozen reducer from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_manifest(root: Path) -> dict[str, Any]:
    manifest = root / "MANIFEST.sha256"
    entries = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        path = root / relative.removeprefix("./")
        if sha256_file(path) != digest:
            raise ValueError(f"manifest mismatch: {relative}")
        entries += 1
    return {
        "entries": entries,
        "manifest_sha256": sha256_file(manifest),
    }


def parse_hashes(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        result[name] = digest
    return result


def exactness_summary(raw: Path, target: str) -> dict[str, Any]:
    rows = read_jsonl(raw / "exactness" / f"{target}.jsonl")
    summaries = [row for row in rows if row.get("kind") == "summary"]
    if len(summaries) != 1:
        raise ValueError(f"{target}: expected one prefix-exactness summary")
    return summaries[0]


def correctness_summary(raw: Path) -> dict[str, Any]:
    gates = raw / "gates"
    kernel = (gates / "kernel-gpu0.log").read_text(encoding="utf-8")
    result: dict[str, Any] = {
        "kernel_gpu0": {
            "all_green": "ALL GREEN" in kernel,
            "failure_marker": "MISMATCH" in kernel or "FAIL" in kernel,
        },
        "targets": {},
    }
    for target in LEVELS_BY_TARGET:
        gen = (gates / f"run-gen-{target}.log").read_text(encoding="utf-8")
        spec = (gates / f"run-spec-{target}.log").read_text(encoding="utf-8")
        row = {
            "run_gen_argmax_match": "argmax=" in gen and "MATCH" in gen and "MISMATCH" not in gen,
            "run_spec_k1_k8_pass_count": spec.count("self-consistency: PASS"),
            "run_spec_overall_pass": "=== SELF-CONSISTENCY PASS ===" in spec
            and "SELF-CONSISTENCY FAIL" not in spec,
        }
        row["pass"] = bool(
            result["kernel_gpu0"]["all_green"]
            and not result["kernel_gpu0"]["failure_marker"]
            and row["run_gen_argmax_match"]
            and row["run_spec_k1_k8_pass_count"] == 8
            and row["run_spec_overall_pass"]
        )
        result["targets"][target] = row
    return result


def comparison(old: float, new: float, higher_is_better: bool) -> dict[str, Any]:
    delta = new - old
    percent = delta / abs(old) * 100.0 if old else None
    regressed = delta < 0 if higher_is_better else delta > 0
    return {
        "old": old,
        "new": new,
        "delta": delta,
        "delta_percent": percent,
        "higher_is_better": higher_is_better,
        "regressed": regressed,
    }


def thermal_summary(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 11:
            continue
        try:
            temperature = float(fields[3])
            power = float(fields[4])
            clock = float(fields[6])
            memory = float(fields[8])
            utilization = float(fields[10])
        except ValueError:
            continue
        for key, value in (
            ("max_temperature_c", temperature),
            ("max_power_w", power),
            ("max_clock_mhz", clock),
            ("max_memory_used_mib", memory),
            ("max_utilization_percent", utilization),
        ):
            result[key] = max(result.get(key, value), value)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--old-summary", type=Path, required=True)
    parser.add_argument("--frozen-reducer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    if not (args.raw / "campaign" / "campaign.complete").is_file():
        raise ValueError("campaign completion sentinel is absent")

    frozen = load_frozen_reducer(args.frozen_reducer)
    old = read_json(args.old_summary)
    manifest = verify_manifest(args.raw)
    replay_paths = sorted((args.raw / "campaign").glob("r??-*/replay.jsonl"))
    if len(replay_paths) != 10:
        raise ValueError(f"expected ten model/repetition replay files, got {len(replay_paths)}")

    protocols: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for path in replay_paths:
        rows = read_jsonl(path)
        protocol = [row for row in rows if row.get("kind") == "protocol"]
        summary = [row for row in rows if row.get("kind") == "summary"]
        if len(protocol) != 1 or len(summary) != 1:
            raise ValueError(f"{path}: expected one protocol and one summary")
        protocols.extend(protocol)
        run_summaries.extend(summary)
        requests.extend(row for row in rows if row.get("kind") == "request")
        cells.extend(row for row in rows if row.get("kind") == "cell")

    expected_cells = sum(len(levels) * 2 * 5 for levels in LEVELS_BY_TARGET.values())
    if len(cells) != expected_cells:
        raise ValueError(f"expected {expected_cells} cells, got {len(cells)}")
    if any(row.get("workload_lock_sha256") != WORKLOAD_SHA256 for row in protocols):
        raise ValueError("protocol workload hash drift")
    for row in run_summaries:
        target = str(row["target"])
        if int(row.get("cells") or 0) != len(LEVELS_BY_TARGET[target]) * 2:
            raise ValueError(f"{target} repetition {row.get('global_repetition')}: incomplete")

    for target, levels in LEVELS_BY_TARGET.items():
        target_protocols = [row for row in protocols if row["target"] == target]
        if sorted(int(row["global_repetition"]) for row in target_protocols) != [1, 2, 3, 4, 5]:
            raise ValueError(f"{target}: repetitions are incomplete")
        if any([int(value) for value in row["levels"]] != levels for row in target_protocols):
            raise ValueError(f"{target}: level grid drift")

    groups = {
        target: {
            arm: {
                str(level): frozen.summarize_cell_group(
                    target, arm, level, requests, cells
                )
                for level in levels
            }
            for arm in ("cold", "mixed90")
        }
        for target, levels in LEVELS_BY_TARGET.items()
    }
    correctness = correctness_summary(args.raw)
    targets: dict[str, Any] = {}
    p0_regressions: list[dict[str, Any]] = [
        {
            "kind": "cell_integrity",
            "target": row["target"],
            "arm": row["arm"],
            "repetition": row["rep"],
            "concurrency": row["concurrency"],
            "requests_ok": row["requests_ok"],
            "requests_n": row["requests_n"],
            "completion_tokens": row["completion_tokens"],
            "expected_completion_tokens": int(row["requests_n"]) * 60,
            "integrity_failures": row["integrity_failures"],
        }
        for row in cells
        if not row["clean"]
    ]
    for target in LEVELS_BY_TARGET:
        mixed = groups[target]["mixed90"]
        knee_grid = KNEE_GRID_BY_TARGET[target]
        knee = SOLD_CAP
        path = []
        previous = SOLD_CAP
        for level in knee_grid[1:]:
            prior = mixed[str(previous)]
            current = mixed[str(level)]
            rose = bool(
                prior["all_clean"]
                and current["all_clean"]
                and float(current["output_tok_s_median"])
                > float(prior["output_tok_s_median"])
            )
            path.append(
                {
                    "from": previous,
                    "to": level,
                    "from_output_tok_s": prior["output_tok_s_median"],
                    "to_output_tok_s": current["output_tok_s_median"],
                    "clean_rise": rose,
                }
            )
            if not rose:
                break
            knee = level
            previous = level
        headroom = (knee / SOLD_CAP - 1.0) * 100.0
        c4 = mixed[str(SOLD_CAP)]
        old_target = old["targets"][target]
        old_c4 = old_target["c4_mixed90"]
        comparisons = {
            "hit_ttft_p50_ms": comparison(
                float(old_c4["ttft_hit"]["p50_ms"]),
                float(c4["ttft_hit"]["p50_ms"]),
                False,
            ),
            "hit_ttft_p95_ms": comparison(
                float(old_c4["ttft_hit"]["p95_ms"]),
                float(c4["ttft_hit"]["p95_ms"]),
                False,
            ),
            "mixed_output_tok_s": comparison(
                float(old_c4["output_tok_s_median"]),
                float(c4["output_tok_s_median"]),
                True,
            ),
            "knee": comparison(float(old_target["capacity_width"]), float(knee), True),
            "headroom_percent": comparison(
                float(old_target["capacity_headroom_percent"]), headroom, True
            ),
        }
        for metric, row in comparisons.items():
            if row["regressed"]:
                p0_regressions.append(
                    {"kind": "envelope_regression", "target": target, "metric": metric, **row}
                )
        base_cells = [
            row
            for row in cells
            if row["target"] == target and int(row["concurrency"]) in (1, 2, 4, 8)
        ]
        exactness = exactness_summary(args.raw, target)
        criteria = {
            "standard_correctness": correctness["targets"][target]["pass"],
            "serial_cache_exactness": exactness.get("verdict") == "PASS",
            "required_base_cells_40_of_40_clean": len(base_cells) == 40
            and all(row["clean"] for row in base_cells),
            "all_scored_cells_clean": all(
                row["clean"] for row in cells if row["target"] == target
            ),
            "c4_hit_ttft_p95_lt_2s": float(c4["ttft_hit"]["p95_ms"]) < 2000.0,
            "c4_all_traffic_ttft_p50_lt_2s": float(c4["ttft_all"]["p50_ms"]) < 2000.0,
            "c4_cached_token_accounting_zero_drift": c4["cached_tokens_in_drift"] == 0
            and c4["prefix_cache_hit_tokens_drift"] == 0,
            "capacity_headroom_ge_25_percent": headroom >= 25.0,
        }
        targets[target] = {
            "verdict": "SELLABLE" if all(criteria.values()) else "NOT at c=4",
            "criteria": criteria,
            "correctness": correctness["targets"][target],
            "cache_exactness": exactness,
            "sold_cap": SOLD_CAP,
            "knee": knee,
            "capacity_headroom_percent": headroom,
            "capacity_path": path,
            "c4_mixed90": c4,
            "c4_cold": groups[target]["cold"][str(SOLD_CAP)],
            "comparisons_to_old_envelope": comparisons,
            "prime_batch_calls": sum(
                int(
                    (args.raw / "campaign" / f"r{rep:02d}-{target}" / "prime-batch-count.txt")
                    .read_text(encoding="utf-8")
                    .strip()
                )
                for rep in range(1, 6)
            ),
        }

    orchestrator = (args.raw / "orchestrator.log").read_text(encoding="utf-8")
    lock_match = re.search(r"REQUAL2_LOCK_ACQUIRED ts=([^ ]+)", orchestrator)
    pass_match = re.search(r"REQUAL2_COMPLETE ts=([^ ]+)", orchestrator)
    result = {
        "schema": "memra.requal2.analysis.v1",
        "runtime_source": parse_hashes(args.raw / "SHA256SUMS.input"),
        "manifest": manifest,
        "protocol": {
            "physical_gpu": 0,
            "one_model_resident_at_a_time": True,
            "model_boot_order": "odd repetitions q27,q35; even repetitions q35,q27",
            "repetitions": 5,
            "workload_lock_sha256": WORKLOAD_SHA256,
            "lock_acquired_at": lock_match.group(1) if lock_match else None,
            "lock_released_at": pass_match.group(1) if pass_match else None,
        },
        "counts": {
            "replay_files": len(replay_paths),
            "cells": len(cells),
            "requests": len(requests),
            "base_cells_by_target": {
                target: sum(
                    row["target"] == target and int(row["concurrency"]) in (1, 2, 4, 8)
                    for row in cells
                )
                for target in LEVELS_BY_TARGET
            },
            "clean_cells_by_target": {
                target: sum(row["target"] == target and bool(row["clean"]) for row in cells)
                for target in LEVELS_BY_TARGET
            },
            "short_completions_by_target": {
                target: sum(
                    int(row.get("completion_tokens") or 0) != 60
                    for row in requests
                    if row["target"] == target
                )
                for target in LEVELS_BY_TARGET
            },
        },
        "correctness": correctness,
        "targets": targets,
        "groups": groups,
        "thermal": thermal_summary(args.raw / "gpu-250ms.csv"),
        "p0_regressions": p0_regressions,
        "p0": bool(p0_regressions),
        "overall_verdict": (
            "P0_REGRESSION"
            if p0_regressions
            else "PASS"
            if all(row["verdict"] == "SELLABLE" for row in targets.values())
            else "FAIL"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "overall_verdict": result["overall_verdict"],
                "p0_regressions": len(p0_regressions),
                "targets": {target: row["verdict"] for target, row in targets.items()},
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
