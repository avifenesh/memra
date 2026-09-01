#!/usr/bin/env python3

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.check_hardware_gate import GateError, sha256_file, validate_receipt


class HardwareGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.source = self.repo / "crates/memra-engine/src/parallel.rs"
        self.source.parent.mkdir(parents=True)
        self.source.write_text("pub const STEP: bool = true;\n", encoding="utf-8")
        self.changed_file = "crates/memra-engine/src/parallel.rs"

        self.evidence: list[dict[str, str]] = []
        for kind in (
            "kernel-exactness",
            "topology-exactness",
            "model-exactness",
        ):
            evidence_dir = self.root / kind
            evidence_dir.mkdir()
            verdict = evidence_dir / "verdict.txt"
            verdict.write_text(f"PASS kind={kind}\n", encoding="utf-8")
            payload = evidence_dir / "payload.log"
            payload.write_text(f"{kind}\n", encoding="utf-8")
            manifest = evidence_dir / "manifest.sha256"
            manifest.write_text(
                f"{sha256_file(verdict)}  ./verdict.txt\n"
                f"{sha256_file(payload)}  ./payload.log\n",
                encoding="utf-8",
            )
            self.evidence.append(
                {
                    "kind": kind,
                    "directory": str(evidence_dir),
                    "manifest_sha256": sha256_file(manifest),
                    "verdict_sha256": sha256_file(verdict),
                }
            )

        self.receipt_path = self.root / "receipt.json"
        self.write_receipt()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def write_receipt(self, **updates: object) -> None:
        receipt: dict[str, object] = {
            "schema": 1,
            "gate": "step-pro",
            "model": "Step-3.7-Flash-FP8",
            "hardware": "RTX PRO 6000 Blackwell",
            "status": "PASS",
            "source_files": {self.changed_file: sha256_file(self.source)},
            "evidence": self.evidence,
        }
        receipt.update(updates)
        self.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    def validate(self) -> None:
        validate_receipt(
            self.receipt_path, self.repo, changed_files=[self.changed_file]
        )

    def test_valid_receipt(self) -> None:
        self.validate()

    def test_wrong_target_is_rejected(self) -> None:
        self.write_receipt(hardware="RTX 5090")
        with self.assertRaisesRegex(GateError, "hardware must be"):
            self.validate()

    def test_source_change_is_rejected(self) -> None:
        self.source.write_text("pub const STEP: bool = false;\n", encoding="utf-8")
        with self.assertRaisesRegex(GateError, "source mismatch"):
            self.validate()

    def test_unbound_changed_file_is_rejected(self) -> None:
        extra = self.repo / "crates/memra-engine/src/other.rs"
        extra.write_text("pub fn other() {}\n", encoding="utf-8")
        with self.assertRaisesRegex(GateError, "does not bind"):
            validate_receipt(
                self.receipt_path,
                self.repo,
                changed_files=[self.changed_file, "crates/memra-engine/src/other.rs"],
            )

    def test_corrupt_evidence_is_rejected(self) -> None:
        Path(self.evidence[0]["directory"], "payload.log").write_text(
            "corrupt\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(GateError, "manifest mismatch"):
            self.validate()

    def test_missing_evidence_class_is_rejected(self) -> None:
        self.write_receipt(evidence=self.evidence[:-1])
        with self.assertRaisesRegex(GateError, "missing evidence kinds"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
