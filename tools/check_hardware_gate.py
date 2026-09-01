#!/usr/bin/env python3
"""Validate a hardware-specific release receipt against the pushed source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


STEP_GATE = "step-pro"
STEP_MODEL = "Step-3.7-Flash-FP8"
STEP_HARDWARE = "RTX PRO 6000 Blackwell"
STEP_EVIDENCE_KINDS = {
    "kernel-exactness",
    "topology-exactness",
    "model-exactness",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class GateError(ValueError):
    """The receipt does not prove the requested hardware gate."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise GateError(f"{key} must be a non-empty string")
    return value


def _safe_repo_path(repo_root: Path, raw_path: str) -> Path:
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise GateError(f"source path must stay inside the repository: {raw_path}")
    path = repo_root / relative
    if not path.is_file():
        raise GateError(f"source file is missing: {raw_path}")
    return path


def verify_manifest(evidence_dir: Path) -> None:
    manifest = evidence_dir / "manifest.sha256"
    if not manifest.is_file():
        raise GateError(f"evidence manifest is missing: {manifest}")

    entries = 0
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        try:
            expected, raw_name = line.split(maxsplit=1)
        except ValueError as error:
            raise GateError(
                f"invalid manifest line {line_number} in {manifest}"
            ) from error
        if not SHA256_RE.fullmatch(expected):
            raise GateError(
                f"invalid SHA-256 on line {line_number} in {manifest}"
            )
        raw_name = raw_name.lstrip("*")
        relative = Path(raw_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise GateError(
                f"manifest entry escapes evidence directory: {raw_name}"
            )
        artifact = evidence_dir / relative
        if not artifact.is_file():
            raise GateError(f"manifest artifact is missing: {artifact}")
        actual = sha256_file(artifact)
        if actual != expected:
            raise GateError(
                f"manifest mismatch for {artifact}: expected {expected}, got {actual}"
            )
        entries += 1

    if entries == 0:
        raise GateError(f"evidence manifest is empty: {manifest}")


def validate_receipt(
    receipt_path: Path, repo_root: Path, changed_files: list[str]
) -> None:
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"cannot read receipt {receipt_path}: {error}") from error

    if not isinstance(receipt, dict):
        raise GateError("receipt root must be an object")
    if receipt.get("schema") != 1:
        raise GateError("receipt schema must be 1")
    if _require_string(receipt, "gate") != STEP_GATE:
        raise GateError(f"gate must be {STEP_GATE}")
    if _require_string(receipt, "model") != STEP_MODEL:
        raise GateError(f"model must be {STEP_MODEL}")
    if _require_string(receipt, "hardware") != STEP_HARDWARE:
        raise GateError(f"hardware must be {STEP_HARDWARE}")
    if _require_string(receipt, "status") != "PASS":
        raise GateError("receipt status must be PASS")

    source_files = receipt.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise GateError("source_files must be a non-empty object")
    for raw_path, expected in source_files.items():
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            raise GateError("source_files keys and values must be strings")
        if not SHA256_RE.fullmatch(expected):
            raise GateError(f"invalid source SHA-256 for {raw_path}")
        actual = sha256_file(_safe_repo_path(repo_root, raw_path))
        if actual != expected:
            raise GateError(
                f"source mismatch for {raw_path}: expected {expected}, got {actual}"
            )

    uncovered = sorted(set(changed_files) - set(source_files))
    if uncovered:
        raise GateError(
            "receipt does not bind changed engine files: " + ", ".join(uncovered)
        )

    evidence = receipt.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise GateError("evidence must be a non-empty array")

    kinds: set[str] = set()
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            raise GateError(f"evidence[{index}] must be an object")
        kind = _require_string(item, "kind")
        evidence_dir = Path(_require_string(item, "directory"))
        if not evidence_dir.is_absolute() or not evidence_dir.is_dir():
            raise GateError(
                f"evidence[{index}] directory must be an existing absolute path"
            )
        verdict = evidence_dir / "verdict.txt"
        if not verdict.is_file():
            raise GateError(f"evidence verdict is missing: {verdict}")
        if not verdict.read_text(encoding="utf-8").startswith("PASS"):
            raise GateError(f"evidence verdict is not PASS: {verdict}")

        expected_manifest = _require_string(item, "manifest_sha256")
        expected_verdict = _require_string(item, "verdict_sha256")
        if not SHA256_RE.fullmatch(expected_manifest) or not SHA256_RE.fullmatch(
            expected_verdict
        ):
            raise GateError(f"evidence[{index}] has an invalid SHA-256")
        actual_manifest = sha256_file(evidence_dir / "manifest.sha256")
        actual_verdict = sha256_file(verdict)
        if actual_manifest != expected_manifest:
            raise GateError(
                f"evidence manifest changed for {evidence_dir}: "
                f"expected {expected_manifest}, got {actual_manifest}"
            )
        if actual_verdict != expected_verdict:
            raise GateError(
                f"evidence verdict changed for {evidence_dir}: "
                f"expected {expected_verdict}, got {actual_verdict}"
            )
        verify_manifest(evidence_dir)
        kinds.add(kind)

    missing_kinds = sorted(STEP_EVIDENCE_KINDS - kinds)
    if missing_kinds:
        raise GateError(
            "Step PRO receipt is missing evidence kinds: " + ", ".join(missing_kinds)
        )


def changed_engine_files(repo_root: Path, base: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base}..HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    changed: list[str] = []
    for path in result.stdout.splitlines():
        if path.startswith("crates/memra-engine/cu/") or (
            path.startswith("crates/memra-engine/src/")
            and path.endswith(".rs")
        ):
            changed.append(path)
    return sorted(set(changed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path.cwd(), type=Path)
    parser.add_argument("--base")
    parser.add_argument("--changed-file", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    changed_files = list(args.changed_file)
    if args.base:
        changed_files.extend(changed_engine_files(repo_root, args.base))
    changed_files = sorted(set(changed_files))
    if not changed_files:
        print("hardware-gate: no changed engine files were supplied", file=sys.stderr)
        return 2

    try:
        validate_receipt(args.receipt.resolve(), repo_root, changed_files)
    except GateError as error:
        print(f"hardware-gate: FAIL: {error}", file=sys.stderr)
        return 1

    print(
        "hardware-gate: PASS "
        f"gate={STEP_GATE} model={STEP_MODEL} hardware={STEP_HARDWARE} "
        f"files={len(changed_files)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
