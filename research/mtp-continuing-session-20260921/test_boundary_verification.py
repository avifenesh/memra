import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import verify_archives


class BoundaryEvidenceTests(unittest.TestCase):
    def test_check_preserves_exact_committed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            original = b'{ "archives": [], "policy": "fixed" }\n'
            path.write_bytes(original)
            verify_archives.store_result(path, {"policy": "fixed", "archives": []}, True)
            self.assertEqual(path.read_bytes(), original)

    def test_stale_check_fails_without_overwriting_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            original = b'{"archives": ["both-models"]}\n'
            path.write_bytes(original)
            with self.assertRaisesRegex(ValueError, "differs"):
                verify_archives.store_result(path, {"archives": []}, True)
            self.assertEqual(path.read_bytes(), original)

    def test_missing_committed_receipt_is_not_created_in_check_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.json"
            with self.assertRaises(FileNotFoundError):
                verify_archives.store_result(path, {"archives": []}, True)
            self.assertFalse(path.exists())

    def test_missing_archive_declaration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cold = root / "research/mtp-calibrated-depth-20260920/receipts"
            warm = root / "research/mtp-continuing-session-20260921/receipts"
            cold.mkdir(parents=True)
            warm.mkdir(parents=True)
            cold_data = {"archives": [{"file": name} for name in ["qwen-records.tar.gz", "gemma-records.tar.gz"]]}
            (cold / "manifest.json").write_text(json.dumps(cold_data))
            (warm / "manifest.json").write_text(json.dumps({
                "archives": [{"file": name} for name in ["qwen-records.tar.gz", "gemma-records.tar.gz", "common-records.tar.gz"]],
                "runtime_source": {"runtime_archive_sha256": "fixed", "archive_bytes": 1},
            }))
            with patch.object(verify_archives, "ROOT", root):
                self.assertEqual(len(verify_archives.archive_metadata()[0]), 6)
                cold_data["archives"].pop()
                (cold / "manifest.json").write_text(json.dumps(cold_data))
                with self.assertRaisesRegex(ValueError, "archive declaration"):
                    verify_archives.archive_metadata()

    def test_smaller_member_corpus_is_rejected_even_with_new_manifest_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "qwen-records.tar.gz"
            checks = archive.with_name("qwen-records.sha256")
            checks.write_text("a" * 64 + "  first\n" + "b" * 64 + "  second\n")
            row = {"file_manifest_sha256": hashlib.sha256(checks.read_bytes()).hexdigest(), "files": 2}
            self.assertEqual(len(verify_archives.member_hashes(archive, row, 2)), 2)
            checks.write_text("a" * 64 + "  first\n")
            row["file_manifest_sha256"] = hashlib.sha256(checks.read_bytes()).hexdigest()
            row["files"] = 1
            with self.assertRaisesRegex(ValueError, "count changed"):
                verify_archives.member_hashes(archive, row, 2)


if __name__ == "__main__":
    unittest.main()
