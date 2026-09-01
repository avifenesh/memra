#!/usr/bin/env python3
"""Extract reviewable shared-wavefront attribution from NCU source/SASS CSV.

The NCU report stays outside the repository.  This script consumes only the
exported source/SASS CSV and emits relative-PC evidence.  Classification is
structural for fa_prefill_qw_db: the one-time plain M88 group is Q, the first
per-tile plain M88 group is K, the final two per-tile plain M88 PCs are P, and
all MT88 PCs are V.  P stores lie between the K and P-load regions.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO


METRICS = (
    "L1 Conflicts Shared N-Way",
    "L1 Wavefronts Shared Excessive",
    "L1 Wavefronts Shared",
    "L1 Wavefronts Shared Ideal",
    "Instructions Executed",
)


@dataclass(frozen=True)
class Row:
    address: int
    source: str
    conflicts: int
    excess: int
    actual: int
    ideal: int
    executed: int


@dataclass(frozen=True)
class InputSpec:
    arm: str
    model: str
    path: Path


def as_int(value: str) -> int:
    if value in ("", "-"):
        return 0
    return int(float(value))


def parse_datasets(path: Path) -> list[list[Row]]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    starts = [i for i, row in enumerate(rows) if row and row[0] == "Address"]
    if not starts:
        raise ValueError(f"no NCU source datasets in {path}")

    datasets: list[list[Row]] = []
    for dataset_index, start in enumerate(starts):
        end = starts[dataset_index + 1] - 1 if dataset_index + 1 < len(starts) else len(rows)
        header = rows[start]
        index = {name: header.index(name) for name in ("Address", "Source", *METRICS)}
        parsed: list[Row] = []
        for raw in rows[start + 1 : end]:
            if not raw or not raw[index["Address"]].startswith("0x"):
                continue
            parsed.append(
                Row(
                    address=int(raw[index["Address"]], 16),
                    source=raw[index["Source"]].strip(),
                    conflicts=as_int(raw[index["L1 Conflicts Shared N-Way"]]),
                    excess=as_int(raw[index["L1 Wavefronts Shared Excessive"]]),
                    actual=as_int(raw[index["L1 Wavefronts Shared"]]),
                    ideal=as_int(raw[index["L1 Wavefronts Shared Ideal"]]),
                    executed=as_int(raw[index["Instructions Executed"]]),
                )
            )
        datasets.append(parsed)
    return datasets


def classify(rows: list[Row]) -> dict[int, str]:
    plain = [row for row in rows if "LDSM.16.M88.4" in row.source]
    transposed = [row for row in rows if "LDSM.16.MT88.4" in row.source]
    positive_exec = sorted({row.executed for row in plain if row.executed > 0})
    if len(positive_exec) != 2:
        raise ValueError(f"expected Q and per-tile M88 execution classes, got {positive_exec}")
    q_exec, tile_exec = positive_exec
    q_rows = [row for row in plain if row.executed == q_exec]
    tiled_plain = sorted((row for row in plain if row.executed == tile_exec), key=lambda row: row.address)
    if len(q_rows) != 16 or len(tiled_plain) != 34 or len(transposed) != 32:
        raise ValueError(
            "unexpected fa_prefill_qw_db SASS groups: "
            f"Q={len(q_rows)} tiled-M88={len(tiled_plain)} MT88={len(transposed)}"
        )
    k_rows, p_load_rows = tiled_plain[:-2], tiled_plain[-2:]
    k_end = max(row.address for row in k_rows)
    p_load_end = max(row.address for row in p_load_rows)

    phases: dict[int, str] = {}
    phases.update((row.address, "q_ldmatrix") for row in q_rows)
    phases.update((row.address, "k_ldmatrix") for row in k_rows)
    phases.update((row.address, "p_ldmatrix") for row in p_load_rows)
    phases.update((row.address, "v_ldmatrix_trans") for row in transposed)
    for row in rows:
        if row.source.startswith("STS") and row.executed == tile_exec and k_end < row.address < p_load_end:
            phases[row.address] = "p_store"
    return phases


PHASE_META = {
    "q_ldmatrix": ("Q tile", "16x256 row-major; no padding or swizzle", "LDSM.16.M88.4"),
    "k_ldmatrix": ("K double-buffer tiles", "32x256 row-major; no padding or swizzle", "LDSM.16.M88.4"),
    "p_store": ("P restage", "64x32 row-major; no padding or swizzle", "STS"),
    "p_ldmatrix": ("P restage", "64x32 row-major; no padding or swizzle", "LDSM.16.M88.4"),
    "v_ldmatrix_trans": ("V double-buffer tiles", "32x256 row-major; no padding or swizzle", "LDSM.16.MT88.4"),
    "other_shared": ("Other shared traffic", "mixed", "other"),
    "total": ("All shared traffic", "mixed", "all"),
}

CANDIDATE_LAYOUT = {
    "q_ldmatrix": "16x256 bf16; 16B chunk XOR row&7; no padding",
    "k_ldmatrix": "32x256 bf16; 16B chunk XOR row&7; no padding",
    "p_store": "64x32 bf16; four 16B chunks XOR row&3; no padding",
    "p_ldmatrix": "64x32 bf16; four 16B chunks XOR row&3; no padding",
    "v_ldmatrix_trans": "32x256 bf16; 16B chunk XOR row&7; no padding",
}


def sum_rows(rows: Iterable[Row]) -> tuple[int, int, int, int, set[int]]:
    material = list(rows)
    return (
        sum(row.executed for row in material),
        sum(row.actual for row in material),
        sum(row.ideal for row in material),
        sum(row.excess for row in material),
        {row.conflicts for row in material if row.conflicts > 0},
    )


def parse_spec(value: str) -> InputSpec:
    try:
        arm, model, raw_path = value.split(":", 2)
    except ValueError as error:
        raise argparse.ArgumentTypeError("input must be ARM:MODEL:PATH") from error
    return InputSpec(arm=arm, model=model, path=Path(raw_path))


def open_output(path: Path | None) -> tuple[TextIO, bool]:
    if path is None:
        import sys

        return sys.stdout, False
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", newline=""), True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=parse_spec, required=True, metavar="ARM:MODEL:PATH")
    parser.add_argument("--shapes", default="4096,764", help="dataset-order query-token labels")
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--pcs-out", type=Path)
    args = parser.parse_args()
    shapes = [value.strip() for value in args.shapes.split(",") if value.strip()]

    summary_handle, close_summary = open_output(args.summary_out)
    pcs_handle, close_pcs = open_output(args.pcs_out) if args.pcs_out else (None, False)
    try:
        summary = csv.writer(summary_handle, lineterminator="\n")
        summary.writerow(
            (
                "arm", "model", "shape", "phase", "array", "physical_layout", "sass_pattern",
                "pc_count", "instructions_executed", "conflict_degrees", "actual_wavefronts",
                "ideal_wavefronts", "excess_wavefronts", "actual_over_ideal", "excess_share_pct",
            )
        )
        pcs = csv.writer(pcs_handle, lineterminator="\n") if pcs_handle else None
        if pcs:
            pcs.writerow(
                (
                    "arm", "model", "shape", "phase", "relative_pc", "sass", "instructions_executed",
                    "conflict_degree", "actual_wavefronts", "ideal_wavefronts", "excess_wavefronts",
                )
            )

        for spec in args.input:
            datasets = parse_datasets(spec.path)
            if len(datasets) != len(shapes):
                raise ValueError(f"{spec.path}: {len(datasets)} datasets, but {len(shapes)} shapes supplied")
            for shape, rows in zip(shapes, datasets, strict=True):
                phases = classify(rows)
                shared_rows = [row for row in rows if row.actual or row.ideal or row.excess]
                grouped: dict[str, list[Row]] = {key: [] for key in PHASE_META if key != "total"}
                for row in shared_rows:
                    grouped[phases.get(row.address, "other_shared")].append(row)
                total_excess = sum(row.excess for row in shared_rows)
                base = min(row.address for row in rows)
                for phase in ("q_ldmatrix", "k_ldmatrix", "p_store", "p_ldmatrix", "v_ldmatrix_trans", "other_shared", "total"):
                    selected = shared_rows if phase == "total" else grouped[phase]
                    executed, actual, ideal, excess, conflicts = sum_rows(selected)
                    array, layout, pattern = PHASE_META[phase]
                    if spec.arm.startswith("candidate"):
                        layout = CANDIDATE_LAYOUT.get(phase, layout)
                    ratio = actual / ideal if ideal else 0.0
                    share = 100.0 * excess / total_excess if total_excess else 0.0
                    summary.writerow(
                        (
                            spec.arm, spec.model, shape, phase, array, layout, pattern, len(selected), executed,
                            ";".join(str(value) for value in sorted(conflicts)), actual, ideal, excess,
                            f"{ratio:.6f}", f"{share:.6f}",
                        )
                    )
                    if pcs and phase != "total":
                        for row in selected:
                            pcs.writerow(
                                (
                                    spec.arm, spec.model, shape, phase, f"0x{row.address - base:06x}", row.source,
                                    row.executed, row.conflicts, row.actual, row.ideal, row.excess,
                                )
                            )
    finally:
        if close_summary:
            summary_handle.close()
        if close_pcs and pcs_handle:
            pcs_handle.close()


if __name__ == "__main__":
    main()
