#!/usr/bin/env python3

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.check_hardware_gate import GateError, changed_engine_files, sha256_file, validate_receipt


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
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "CPU fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.git("config", "core.hooksPath", "/dev/null")
        self.git("add", "."); self.git("commit", "-qm", "base")

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

    def git(self, *args: str) -> str:
        return subprocess.check_output(["git", "-C", str(self.repo), *args], text=True).strip()

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

    def test_runtime_dependency_selector_includes_workspace_and_build_inputs(self) -> None:
        paths = ["crates/memra-engine/Cargo.toml", "crates/memra-engine/build.rs",
                 "crates/memra-kv/src/lib.rs", "crates/memra-tier/src/transfers.rs",
                 "crates/memra-tokenizer/src/lib.rs", "crates/memra-server/src/worker.rs",
                 "crates/memra-gguf/src/source.rs", "Cargo.lock", ".cargo/config.toml",
                 "rust-toolchain.toml", "tools/release-battery.sh"]
        with patch("tools.check_hardware_gate.subprocess.run",
                   return_value=SimpleNamespace(stdout="\n".join(paths + ["docs/README.md"]))):
            self.assertEqual(changed_engine_files(self.repo, "base"), sorted(paths))
        for path in paths:
            with self.subTest(path=path):
                with self.assertRaisesRegex(GateError, "does not bind"):
                    validate_receipt(self.receipt_path, self.repo, changed_files=[path])

    def test_removed_input_needs_an_explicit_absence_tombstone(self) -> None:
        self.write_receipt(source_files={self.changed_file: None})
        self.source.unlink()
        self.git("commit", "-qam", "actual candidate deletion")
        self.validate()
        self.source.write_text("replacement\n")
        with self.assertRaisesRegex(GateError, "deleted source is still present"):
            self.validate()

    def test_worktree_only_deletion_cannot_cover_a_present_candidate_input(self) -> None:
        base = self.git("rev-parse", "HEAD")
        self.source.write_text("pub const STEP: bool = false;\n")
        self.git("commit", "-qam", "candidate modifies source")
        self.write_receipt(source_files={self.changed_file: None})
        self.source.unlink()
        self.assertEqual(changed_engine_files(self.repo, base), [self.changed_file])
        self.assertIn("false", self.git("show", "HEAD:" + self.changed_file))
        with self.assertRaisesRegex(GateError, "still present in candidate Git tree"):
            self.validate()

    def test_rename_and_mixed_changes_keep_old_and_new_obligations(self) -> None:
        base = self.git("rev-parse", "HEAD")
        destination = self.source.with_name("renamed.rs")
        self.source.rename(destination)
        extra = destination.with_name("new.rs"); extra.write_text("pub fn new() {}\n")
        self.git("add", "."); self.git("commit", "-qm", "rename and add")
        names = [str(p.relative_to(self.repo)) for p in (destination, extra)]
        changed = changed_engine_files(self.repo, base)
        self.assertEqual(changed, sorted([self.changed_file, *names]))
        # Construct after rename without trying to hash the now-absent old path.
        receipt = json.loads(self.receipt_path.read_text())
        receipt["source_files"] = {self.changed_file: None, names[0]: sha256_file(destination), names[1]: sha256_file(extra)}
        self.receipt_path.write_text(json.dumps(receipt))
        validate_receipt(self.receipt_path, self.repo, changed)
        receipt["source_files"][names[0]] = None
        self.receipt_path.write_text(json.dumps(receipt)); destination.unlink()
        with self.assertRaisesRegex(GateError, "still present in candidate Git tree"):
            validate_receipt(self.receipt_path, self.repo, changed)

    def test_tombstone_refuses_without_a_candidate_git_tree(self) -> None:
        self.write_receipt(source_files={self.changed_file: None})
        self.source.unlink()
        outside = self.root / "not-a-repository"; outside.mkdir()
        with self.assertRaisesRegex(GateError, "readable candidate Git tree"):
            validate_receipt(self.receipt_path, outside, [self.changed_file])

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
