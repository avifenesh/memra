#!/usr/bin/env python3
"""Validate and summarize the frozen N=5 interleaved serving measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "raw" / "measurement"
DEFAULT_OUTPUT = HERE / "measurement-summary.json"
MODELS = ("q27", "q35")
ARMS = ("baseline", "candidate")
METRICS = ("prime_ms", "prefill_tok_s", "cold_ttft_ms_client")
EXPECTED_REPETITIONS = set(range(1, 6))


def stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(input_dir: Path) -> None:
    manifest = input_dir / "MANIFEST.sha256"
    entries = []
    for path in sorted(candidate for candidate in input_dir.rglob("*") if candidate.is_file()):
        if path == manifest:
            continue
        entries.append(f"{sha256(path)}  {path.relative_to(input_dir).as_posix()}")
    manifest.write_text("\n".join(entries) + "\n")


def read_rows(input_dir: Path) -> list[dict[str, object]]:
    path = input_dir / "measurements.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 20:
        raise ValueError(f"expected 20 measurement rows, found {len(rows)} in {path}")
    if [row["sequence"] for row in rows] != list(range(1, 21)):
        raise ValueError("measurement sequence is not exactly 1..20")
    return rows


def validate(input_dir: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    provenance = json.loads((input_dir / "provenance.json").read_text())
    if provenance.get("actual_shape_text_identity") != "PASS":
        raise ValueError("provenance does not record actual-shape text identity PASS")
    if provenance.get("repetitions") != 5:
        raise ValueError("provenance does not record exactly five repetitions")

    for model in MODELS:
        model_rows = [row for row in rows if row.get("model") == model]
        if len(model_rows) != 10:
            raise ValueError(f"expected 10 {model} rows, found {len(model_rows)}")
        hashes = {row.get("text_sha256") for row in model_rows}
        if len(hashes) != 1:
            raise ValueError(f"actual-shape text differs for {model}: {sorted(hashes)}")
        for arm in ARMS:
            arm_rows = [row for row in model_rows if row.get("arm") == arm]
            repetitions = {row.get("repetition") for row in arm_rows}
            if repetitions != EXPECTED_REPETITIONS:
                raise ValueError(f"{model}/{arm} repetitions are {sorted(repetitions)}")
            for row in arm_rows:
                if row.get("prompt_tokens") != 4860 or row.get("cached_tokens") != 0:
                    raise ValueError(f"request contract changed in sequence {row.get('sequence')}")
                if row.get("thermal_cap_mhz") != [210, 1200]:
                    raise ValueError(f"thermal-cap record changed in sequence {row.get('sequence')}")
                expected_server = provenance["servers"][arm]["sha256"]
                expected_model = provenance["models"][model]["sha256"]
                if row.get("server_sha256") != expected_server:
                    raise ValueError(f"server hash changed in sequence {row.get('sequence')}")
                if row.get("model_sha256") != expected_model:
                    raise ValueError(f"model hash changed in sequence {row.get('sequence')}")
    return provenance


def telemetry_bounds(input_dir: Path) -> dict[str, object]:
    active: list[tuple[float, float]] = []
    for path in sorted(input_dir.glob("??-*/telemetry.csv")):
        for raw in path.read_text(errors="replace").splitlines():
            fields = [field.strip() for field in raw.split(",")]
            if len(fields) != 7:
                continue
            try:
                temperature = float(fields[2])
                clock = float(fields[3])
                utilization = float(fields[6])
            except ValueError:
                continue
            if utilization > 0:
                active.append((clock, temperature))
    if not active:
        raise ValueError("no active GPU telemetry samples found")
    return {
        "definition": "telemetry samples with utilization.gpu > 0",
        "samples": len(active),
        "clock_mhz": [min(clock for clock, _ in active), max(clock for clock, _ in active)],
        "temperature_c": [
            min(temperature for _, temperature in active),
            max(temperature for _, temperature in active),
        ],
    }


def summarize(
    input_dir: Path, rows: list[dict[str, object]], provenance: dict[str, object], verdict: str
) -> dict[str, object]:
    models: dict[str, object] = {}
    for model in MODELS:
        grouped: dict[str, list[dict[str, object]]] = {
            arm: sorted(
                (row for row in rows if row["model"] == model and row["arm"] == arm),
                key=lambda row: int(row["repetition"]),
            )
            for arm in ARMS
        }
        model_summary: dict[str, object] = {}
        for arm, arm_rows in grouped.items():
            model_summary[arm] = {
                metric: stats([float(row[metric]) for row in arm_rows]) for metric in METRICS
            }

        paired: dict[str, list[float]] = {}
        paired_medians: dict[str, float] = {}
        separation: dict[str, object] = {}
        for metric in METRICS:
            deltas = [
                (float(candidate[metric]) - float(baseline[metric]))
                / float(baseline[metric])
                * 100.0
                for baseline, candidate in zip(grouped["baseline"], grouped["candidate"], strict=True)
            ]
            paired[metric] = deltas
            paired_medians[metric] = statistics.median(deltas)
            baseline_values = [float(row[metric]) for row in grouped["baseline"]]
            candidate_values = [float(row[metric]) for row in grouped["candidate"]]
            separation[metric] = {
                "candidate_minus_baseline_median": (
                    statistics.median(candidate_values) - statistics.median(baseline_values)
                ),
                "baseline_range": max(baseline_values) - min(baseline_values),
                "candidate_range": max(candidate_values) - min(candidate_values),
            }
        model_summary["paired_candidate_delta_percent"] = paired
        model_summary["paired_candidate_delta_median_percent"] = paired_medians
        model_summary["median_separation_and_arm_ranges"] = separation
        models[model] = model_summary

    return {
        "schema": "memra.shmconflict.measurement-summary.v1",
        "verdict": verdict,
        "thermal_cap_mhz": [210, 1200],
        "relative_only": True,
        "prompt_tokens": 4860,
        "repetitions_per_arm": 5,
        "actual_shape_text_identity": provenance["actual_shape_text_identity"],
        "models": models,
        "active_telemetry": telemetry_bounds(input_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--verdict", choices=("PENDING", "GO", "PARTIAL", "NO-GO"), default="PENDING"
    )
    args = parser.parse_args()
    input_dir = args.input.resolve()
    rows = read_rows(input_dir)
    provenance = validate(input_dir, rows)
    summary = summarize(input_dir, rows, provenance, args.verdict)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_manifest(input_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
