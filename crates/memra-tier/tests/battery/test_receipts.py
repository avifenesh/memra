"""CPU-only adversarial tests of the actual receipt validator/CLI; no model evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location("tier_battery", ROOT / "tools/tier-battery.py")
BATTERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BATTERY)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="memra-tier-cpu-")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        def blob(name, data):
            (self.root / name).write_bytes(data)
            return {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        row = {"schema_version": 1, "cell": "boundary", "pair_id": "fixture-pair", "run_id": "fixture-off", "arm": "off", "kind": "cpu-fixture", "rig": "cpu", "lock": None, "route": "local", "direct_path_proven": False, "migrated_bytes": 0, "runtime_commit": "a" * 40, "binary_sha256": "b" * 64, "artifact_sha256": "c" * 64, "plan_sha256": "d" * 64, "layout_sha256": "e" * 64, "prompt_sha256": "f" * 64, "numeric_class": "opaque-fixture-no-executor", "context_tokens": 3, "requests": 1, "state": blob("state.bin", b"\x01\x02\x03"), "logits": blob("logits.bin", b"\0\0\x80?"), "tokens": blob("tokens.bin", b"\1\0\0\0"), "raw_log": blob("raw.log", b"fixture only\n"), "telemetry": None, "telemetry_interval_ms": None, "status": "pass", "exit_code": 0}
        self.rows = [row, copy.deepcopy(row)]
        self.rows[1].update(arm="on", run_id="fixture-on", route="host", migrated_bytes=3)

    def rejects(self, rows):
        with self.assertRaises((ValueError, TypeError, KeyError)):
            BATTERY.validate_rows(rows, self.root)

    def test_real_cli_positive_and_corrupted_evidence_red(self):
        path = self.root / "runs.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in self.rows) + "\n")
        cmd = [sys.executable, str(ROOT / "tools/tier-battery.py"), "--validate", str(path)]
        result = subprocess.run(cmd, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("2 records / 1 forced pairs", result.stdout)
        (self.root / "state.bin").write_bytes(b"bad")
        result = subprocess.run(cmd, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("evidence hash mismatch", result.stderr)

    def test_missing_empty_duplicate_and_partial_batches(self):
        self.rejects([])
        self.rejects(self.rows[:1])
        self.rejects([self.rows[0], self.rows[0]])
        (self.root / "state.bin").unlink()
        self.rejects(self.rows)

    def test_program_epoch_boundary_identity_and_nonvacuity(self):
        for field, value in [("numeric_class", "other-program"), ("layout_sha256", "0" * 64), ("migrated_bytes", 0), ("context_tokens", 0), ("status", "skip"), ("exit_code", 77), ("telemetry", {})]:
            with self.subTest(field=field):
                rows = copy.deepcopy(self.rows)
                rows[1][field] = value
                self.rejects(rows)

    def test_third_lock_fake_p2p_and_host_bounce_mislabel(self):
        for changes in [{"lock": "/tmp/not-a-valid-lock"}, {"route": "pcie-p2p", "direct_path_proven": True}, {"route": "host-bounce", "direct_path_proven": True}, {"rig": "pro-four"}]:
            rows = copy.deepcopy(self.rows)
            rows[1].update(changes)
            self.rejects(rows)

    def test_mismatched_outputs_rejected_even_with_valid_files(self):
        (self.root / "other.bin").write_bytes(b"\2\0\0\0")
        self.rows[1]["tokens"] = {"path": "other.bin", "sha256": hashlib.sha256(b"\2\0\0\0").hexdigest(), "bytes": 4}
        self.rejects(self.rows)

    def test_schema_and_validator_agree_on_fields(self):
        schema = json.loads((ROOT / "research/spill-d-20260919/runs.schema.json").read_text())
        self.assertEqual(set(schema["required"]), set(self.rows[0]))
        self.assertEqual(set(schema["properties"]), set(self.rows[0]))
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["properties"]["cell"]["enum"]), set(sum(BATTERY.CASES.values(), [])))
        self.rows[0]["schema_version"] = True
        self.rejects(self.rows)

    def test_plan_is_pending_and_requires_standard_gates(self):
        result = subprocess.run([sys.executable, str(ROOT / "tools/tier-battery.py"), "--plan"], text=True, capture_output=True, check=True)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["status"], "pending-gpu-adapters")
        self.assertIn("step-pro", plan["standard_gates"])
        self.assertEqual(len(plan["cases"]["D1"]), 8)
        self.assertEqual(plan["performance"], {"AB_pairs": 5, "BA_pairs": 5, "telemetry_interval_ms": 250})


if __name__ == "__main__":
    unittest.main(verbosity=2)
