"""OWED 14: the collector labels storage NVMe only through a bound M1 proof (CPU only).

Runs against the collector at tools/tier-battery.py with owed14/tier-battery-storage-proof.patch
applied. Green cases use a synthetic PASS receipt bound to the live filesystem identity, so the
suite passes on any host, including CI runners that are VMs (a real proof fails A1 there by
design). `M1_LIVE_PROOF=1` adds one case that runs the real proof tool on the repo filesystem.
No lock is taken, no GPU is touched, no nvidia-smi runs.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "tools/tier-battery.py").exists())
spec = importlib.util.spec_from_file_location("battery_owed14", ROOT / "tools/tier-battery.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
TOOL = ROOT / "research/spill-f-20260919/m1-nvme-proof.py"


def synthetic_proof(root, **override):
    live = B.filesystem_identity(Path(root))
    proof = {"schema": "m1-nvme-proof-v1", "verdict": "PASS", "class": "nvme-local-direct",
             "reasons": [], "utc": "2026-09-25T00:00:00+00:00",
             "tool_sha256": hashlib.sha256(TOOL.read_bytes()).hexdigest(),
             "A8_identity": {"device": live["device"], "mount_id": live.get("mount_id"),
                             "filesystem_id_sha256_16": hashlib.sha256(
                                 str(live["filesystem_id"]).encode()).hexdigest()[:16]}}
    for key, value in override.items():
        if key == "identity":
            proof["A8_identity"].update(value)
        else:
            proof[key] = value
    return proof


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        base = Path(self.tmp.name)
        self.root = base / "scratch"
        self.root.mkdir()
        self.out = base / "out"
        self.out.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def write_proof(self, proof, name="PROOF.json"):
        path = Path(self.tmp.name) / name
        path.write_text(json.dumps(proof))
        return path

    def storage_json(self):
        return json.loads((self.out / "STORAGE.json").read_text())


class Green(Case):
    def test_bound_pass_receipt_is_the_only_nvme_label(self):
        proof = self.write_proof(synthetic_proof(self.root))
        record = B.capture_storage(self.root, self.out, proof_path=proof)
        self.assertEqual(record["class"], "nvme-local-direct")
        self.assertTrue(record["nvme_proven"])
        self.assertEqual(record["label"], B.M1_PROVEN_STORAGE)
        self.assertEqual(record["m1_proof"]["receipt"]["path"], "STORAGE-PROOF.json")
        self.assertEqual((self.out / "STORAGE-PROOF.json").read_bytes(), proof.read_bytes())
        self.assertEqual(self.storage_json()["class"], "nvme-local-direct")
        B.validate_storage_record(record, self.out, ["true"])

    def test_private_receipt_with_raw_filesystem_id_binds(self):
        live = B.filesystem_identity(self.root)
        proof = synthetic_proof(self.root)
        proof["A8_identity"] = dict(live)
        record = B.capture_storage(self.root, self.out, proof_path=self.write_proof(proof))
        self.assertTrue(record["nvme_proven"])

    def test_subdirectory_of_the_proven_mount_binds(self):
        sub = self.root / "objects"
        sub.mkdir()
        record = B.capture_storage(sub, self.out, proof_path=self.write_proof(synthetic_proof(self.root)))
        self.assertTrue(record["nvme_proven"])

    @unittest.skipUnless(os.environ.get("M1_LIVE_PROOF") == "1", "set M1_LIVE_PROOF=1 on bare metal")
    def test_live_proof_tool_receipt(self):
        pub = Path(self.tmp.name) / "live.public.json"
        priv = Path(self.tmp.name) / "live.private.json"
        proc = subprocess.run([sys.executable, str(TOOL), "--path", str(self.root),
                               "--bind-bytes", str(64 << 20), "--private-out", str(priv),
                               "--public-out", str(pub)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        record = B.capture_storage(self.root, self.out, proof_path=pub)
        self.assertEqual(record["class"], "nvme-local-direct")


class Red(Case):
    def refused(self, proof, needle, allow_unproven=False):
        path = self.write_proof(proof) if isinstance(proof, dict) else proof
        with self.assertRaisesRegex(ValueError, "storage proof refused"):
            B.capture_storage(self.root, self.out, allow_unproven, proof_path=path)
        saved = self.storage_json()
        self.assertEqual(saved["class"], "m1-proof-refused")
        self.assertFalse(saved["nvme_proven"])
        self.assertIn(needle, saved["m1_proof_refusal"])
        self.assertFalse((self.out / "STORAGE-PROOF.json").exists())

    def test_fail_verdict(self):
        self.refused(synthetic_proof(self.root, verdict="FAIL", **{"class": None}), "did not PASS")

    def test_reasons_present(self):
        self.refused(synthetic_proof(self.root, reasons=["A1: guest kernel"]), "did not PASS")

    def test_wrong_schema(self):
        self.refused(synthetic_proof(self.root, schema="something-else"), "not an M1 proof")

    def test_different_tool(self):
        self.refused(synthetic_proof(self.root, tool_sha256="0" * 64), "different proof tool")

    def test_other_device(self):
        live = B.filesystem_identity(self.root)
        self.refused(synthetic_proof(self.root, identity={"device": live["device"] + 1}),
                     "not this mount")

    def test_other_mount(self):
        live = B.filesystem_identity(self.root)
        self.refused(synthetic_proof(self.root, identity={"mount_id": live["mount_id"] + 1}),
                     "not this mount")

    def test_other_filesystem_id(self):
        self.refused(synthetic_proof(self.root, identity={"filesystem_id_sha256_16": "0" * 16}),
                     "filesystem id differs")

    def test_missing_receipt(self):
        self.refused(Path(self.tmp.name) / "absent.json", "No such file")

    def test_malformed_receipt(self):
        bad = Path(self.tmp.name) / "bad.json"
        bad.write_text("{not json")
        self.refused(bad, "Expecting property name")

    def test_failing_proof_never_downgrades_under_opt_in(self):
        self.refused(synthetic_proof(self.root, verdict="FAIL"), "did not PASS", allow_unproven=True)


class Unproven(Case):
    def test_name_match_alone_is_not_proof(self):
        with self.assertRaisesRegex(ValueError, "no M1 proof"):
            B.capture_storage(self.root, self.out)
        saved = self.storage_json()
        self.assertFalse(saved["nvme_proven"])
        self.assertIn(saved["class"], {"nvme-name-only-unproven", "overlay-unproven"})
        record = B.capture_storage(self.root, Path(tempfile.mkdtemp(dir=self.tmp.name)), True)
        self.assertEqual(record["label"], B.UNPROVEN_STORAGE)
        self.assertFalse(record["nvme_proven"])

    def test_bind_mount_source_suffix_is_stripped_and_still_unproven(self):
        calls = []

        def fake(command, raw, timeout=30, echo=True, **_):
            calls.append(command)
            text = {"storage-source": "/dev/md0[/var/lib/volumes/v1]\n",
                    "storage-ancestry": "md0\nnvme0n1\nnvme1n1\n"}.get(raw.stem, "{}\n")
            raw.write_text(text)
            return 0, False

        with patch.object(B, "tee_run", side_effect=fake):
            record = B.capture_storage(self.root, self.out, True)
        self.assertEqual(calls[-1][-1], "/dev/md0")
        self.assertTrue(record["nvme_name_hint"])
        self.assertEqual(record["class"], "nvme-name-only-unproven")
        self.assertFalse(record["nvme_proven"])


class Validator(Case):
    def proven_record(self):
        proof = self.write_proof(synthetic_proof(self.root))
        return B.capture_storage(self.root, self.out, proof_path=proof)

    def test_tampered_archived_proof_is_refused(self):
        record = self.proven_record()
        (self.out / "STORAGE-PROOF.json").write_text("{}")
        with self.assertRaises(ValueError):
            B.validate_storage_record(record, self.out, ["true"])

    def test_nvme_label_without_binding_is_refused(self):
        record = self.proven_record()
        stripped = copy.deepcopy(record)
        del stripped["m1_proof"]
        with self.assertRaisesRegex(ValueError, "without an M1 proof binding"):
            B.validate_storage_record(stripped, self.out, ["true"])

    def test_legacy_name_match_label_is_refused(self):
        record = self.proven_record()
        legacy = copy.deepcopy(record)
        legacy.update({"class": "nvme-ancestry", "label": "NVMe ancestry only; not measured spill speed"})
        del legacy["m1_proof"]
        with self.assertRaisesRegex(ValueError, "without an M1 proof binding"):
            B.validate_storage_record(legacy, self.out, ["true"])


class Cli(unittest.TestCase):
    def test_storage_proof_requires_storage_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            proc = subprocess.run([sys.executable, str(ROOT / "tools/tier-battery.py"),
                                   "--out", str(out), "--storage-proof", str(Path(tmp) / "p.json"),
                                   "--execute", "true"], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertIn("--storage-proof requires --storage-root", proc.stderr)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
