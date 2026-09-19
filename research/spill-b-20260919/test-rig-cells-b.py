#!/usr/bin/env python3
"""Executable runner teeth. All child commands are explicitly marked CPU stubs."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent

class RunnerTests(unittest.TestCase):
    def exercise(self, fail=False):
        with tempfile.TemporaryDirectory(prefix="spill-b-runner-test-") as scratch:
            out = Path(scratch) / "receipt"
            args = ["bash", str(HERE / "rig-cells-b.sh"), "--dry-run", "--out", str(out)]
            if fail:
                args.append("--stub-fail")
            result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            self.assertEqual(result.returncode, 1 if fail else 0, result.stdout)
            rows = [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1 if fail else 20)
            for row in rows:
                self.assertFalse(row["gpu_executed"])
                self.assertEqual(row["status"], "FAIL" if fail else "DRY_RUN")
                self.assertEqual(hashlib.sha256((out / row["raw"]).read_bytes()).hexdigest(), row["raw_sha256"])
            if fail:
                self.assertEqual(rows[0]["exit"], 7)
                self.assertIn(b"injected stub failure", result.stdout)
            else:
                self.assertEqual(sum("PENDING" in row["cell"] for row in rows), 4)
                for row in rows:
                    cmd = row["command"]
                    if "baseline" in row["cell"]:
                        self.assertEqual(cmd[:4], ["flock", "-w", "300", "/tmp/memra-5090.lock"])
                    if row["cell"].endswith(("identity", "teeth", "failures")):
                        self.assertNotIn("flock", cmd)  # existing scripts lock internally
                cells = [row["cell"] for row in rows]
                self.assertLess(cells.index("before-qwen-32768-baseline"), cells.index("before-identity"))
                manifest = json.loads((out / "manifest.json").read_text())
                self.assertEqual(manifest["contexts"], [8192, 32768])
                self.assertEqual(manifest["excluded_pro_contexts"], [131072, 262144])
                self.assertFalse(manifest["gpu_claim"])
            # Idempotent re-entry must refuse overwriting an immutable receipt.
            again = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            self.assertEqual(again.returncode, 1)
    def test_order_raw_hashes_and_no_gpu_claim(self):
        self.exercise()
    def test_injected_command_failure_is_not_swallowed(self):
        self.exercise(True)

if __name__ == "__main__":
    unittest.main()
