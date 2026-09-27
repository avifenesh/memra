#!/usr/bin/env python3
"""CPU admission and actual OS isolation tests. No CUDA, model loads or engine build."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import component as c


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.contract = {"min_tests": 1}
        self.result = {"tests": ["one"], "run": 1, "pass": True, "skipped": 0,
                       "failures": 0, "errors": 0, "expected_failures": 0, "unexpected_successes": 0}

    def test_vacuity_skip_failure_and_duplicate_roster_refuse(self):
        self.assertTrue(c.valid_result(self.result, self.contract))
        for mutation in ({"run": 0, "tests": []}, {"run": 2, "tests": ["one", "one"]},
                         {"pass": False}, {"skipped": 1}, {"failures": 1}, {"errors": 1},
                         {"expected_failures": 1}, {"unexpected_successes": 1}, {"run": True}):
            with self.subTest(mutation=mutation):
                self.assertFalse(c.valid_result(self.result | mutation, self.contract))
        self.assertFalse(c.valid_result(self.result | {"run": 0, "tests": []}, {"min_tests": 0}))
        self.assertFalse(c.valid_result(self.result | {"run": 3, "tests": "abc"}, self.contract))

    def test_actual_bytes_modes_and_symlinks(self):
        file = self.root / "input"
        file.write_text("one")
        first = c.file_id(file)
        file.write_text("two")
        self.assertNotEqual(c.file_id(file), first)
        second = c.file_id(file)
        file.chmod(0o755)
        self.assertNotEqual(c.file_id(file), second)
        file.unlink(); file.symlink_to(self.root / "absent")
        with self.assertRaises(ValueError):
            c.file_id(file)

    def test_corrupt_stale_and_missing_evidence_refuse(self):
        identity = {"fixture": "one"}
        (self.root / "run.log").write_text("ran one test")
        (self.root / "result.json").write_text(json.dumps(self.result))
        receipt = {"key": c.digest(identity), "identity": identity, "exit_code": 0,
                   "evidence": {p: c.file_id(self.root / p) for p in ("run.log", "result.json")}}
        (self.root / "receipt.json").write_text(json.dumps(receipt))
        c.verify(self.root, identity, self.contract)
        with self.assertRaises(ValueError):
            c.verify(self.root, {"fixture": "changed"}, self.contract)
        (self.root / "run.log").write_text("changed")
        with self.assertRaises(ValueError):
            c.verify(self.root, identity, self.contract)
        (self.root / "run.log").unlink()
        with self.assertRaises(OSError):
            c.verify(self.root, identity, self.contract)


class IsolationTests(unittest.TestCase):
    def test_actual_isolation_reuse_and_planted_fixture_failure(self):
        # This is deliberately a real bubblewrap execution, not a mock sandbox.
        # Unsupported runners fail closed instead of silently skipping this check.
        with tempfile.TemporaryDirectory(prefix="component-controls-") as tmp:
            root = Path(tmp)
            repo, cache = root / "repo", root / "cache"
            tools = repo / "tools/fast-gate"
            tools.mkdir(parents=True)
            for name in ("component.py", "component_worker.py"):
                shutil.copy2(c.HERE / name, tools / name)
            (repo / "oracle.txt").write_text("expected\n")
            (root / "owner-secret.txt").write_text("must be invisible")
            (repo / "tools/test_fixture.py").write_text('''import os
from pathlib import Path
import unittest
class Checks(unittest.TestCase):
    def test_oracle(self):
        self.assertEqual(Path("/source/oracle.txt").read_text(), "expected\\n")
    def test_hidden_inputs(self):
        self.assertFalse(Path("/source/hidden.txt").exists())
        self.assertFalse(Path("/source/.cargo/config.toml").exists())
    def test_environment_and_devices(self):
        self.assertNotIn("COMPONENT_UNDECLARED_INPUT", os.environ)
        self.assertFalse(Path("/dev/nvidia0").exists())
    def test_host_home(self):
        self.assertEqual(os.environ["HOME"], "/tmp/home")
        self.assertEqual(list(Path("/tmp/home").iterdir()), [])
''')
            spec = {"modules": ["test_fixture"], "inputs": ["tools/test_fixture.py", "oracle.txt"],
                    "min_tests": 4, "scope": "CPU isolation fixture only"}
            config = {"schema": "memra-dev-dependencies-v1", "components": {"fixture": spec}}
            (tools / "dependencies.json").write_text(json.dumps(config))
            with patch.dict(os.environ, {"COMPONENT_UNDECLARED_INPUT": "must be absent"}):
                first = c.run(repo, cache, "fixture")
                self.assertEqual(first["decision"], "run")
                repeat = c.run(repo, cache, "fixture")
                self.assertEqual(repeat["decision"], "reuse")
                self.assertEqual(c.lookup(repo, cache, "fixture")["decision"], "reuse")
                # Neither file is visible. Reuse is valid only for this isolated component.
                (repo / "hidden.txt").write_text("new source outside admitted closure")
                (repo / ".cargo").mkdir()
                (repo / ".cargo/config.toml").write_text("[env]\nHIDDEN='value'\n")
                self.assertEqual(c.run(repo, cache, "fixture")["decision"], "reuse")
                shadow = c.run(repo, cache, "fixture", fresh=True)
                self.assertEqual(shadow["decision"], "run")
                self.assertNotEqual(shadow["receipt"], first["receipt"])
                self.assertTrue(Path(shadow["receipt"]).is_file())
                # A changed consumed fixture must rerun and fail, never reuse green.
                (repo / "oracle.txt").write_text("planted regression\n")
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(ValueError, "component failed"):
                    c.run(repo, cache, "fixture")
                (repo / "oracle.txt").unlink()
                with self.assertRaises(OSError):
                    c.run(repo, cache, "fixture")
                (repo / "oracle.txt").write_text("expected\n")
                # A failing fresh check of the same key quarantines the older pass.
                with patch.object(c, "sandbox", return_value=["/usr/bin/false"]), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(ValueError, "component failed"):
                    c.run(repo, cache, "fixture", fresh=True)
                for action in (lambda: c.lookup(repo, cache, "fixture"), lambda: c.run(repo, cache, "fixture")):
                    with self.assertRaisesRegex(ValueError, "shadow contradicted"):
                        action()


if __name__ == "__main__":
    unittest.main()
