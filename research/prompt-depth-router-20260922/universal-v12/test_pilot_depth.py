"""Verify exact sealed pilot-model extraction and resume checks."""

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

import pilot_depth


class PilotModelSourceTest(unittest.TestCase):
    def test_sealed_weights_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "parent.tar.gz"
            manifest_path = root / "manifest.json"
            expected = {}
            with tarfile.open(archive, "w:gz") as tar:
                for k in (3, 10, 20):
                    for kind in ("depth", "confidence"):
                        name = (
                            "training/cd-models/augmented/"
                            f"topk{k}/{kind}-history.tsv"
                        )
                        content = f"synthetic-{k}-{kind}\n".encode()
                        member = tarfile.TarInfo(name)
                        member.size = len(content)
                        tar.addfile(member, io.BytesIO(content))
                        expected[name] = {
                            "bytes": len(content),
                            "sha256":
                            hashlib.sha256(content).hexdigest(),
                        }
            archive_sha = pilot_depth.sha(archive)
            manifest_path.write_text(json.dumps({
                "archive_sha256": archive_sha,
                "members": expected,
            }))
            output = root / "models"
            with mock.patch.object(
                pilot_depth, "V9_SHA", archive_sha,
            ):
                self.assertEqual(
                    pilot_depth.model_files(
                        archive, manifest_path, output,
                    ),
                    {
                        name: entry["sha256"]
                        for name, entry in expected.items()
                    },
                )
                changed = output / "topk3/depth-history.tsv"
                changed.write_text("changed\n")
                with self.assertRaises(ValueError):
                    pilot_depth.model_files(
                        archive, manifest_path, output,
                    )

    def test_saved_action_keys_roundtrip(self):
        self.assertEqual(
            pilot_depth.canonical({"k_actions": {10: 8}}),
            {"k_actions": {"10": 8}},
        )


if __name__ == "__main__":
    unittest.main()
