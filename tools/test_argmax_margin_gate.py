#!/usr/bin/env python3
"""CPU-only wrapper controls; stubs exercise the real parsing and verdict branches."""
import pathlib
import shutil
import subprocess
import tempfile
import unittest


class GateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="memra-margin-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        (self.root / "tools").mkdir()
        self.gate = self.root / "tools/argmax-margin-gate.sh"
        shutil.copyfile(pathlib.Path(__file__).with_name(self.gate.name), self.gate)
        (self.root / "target/release").mkdir(parents=True)
        self.probe = self.root / "target/release/argmax-margin-probe"
        self.prompt = self.root / "prompt.txt"
        self.prompt.write_text("A real prompt fixture for the wrapper control.\n")
        self.gguf = self.root / "model.gguf"
        self.gguf.touch()
        self.hf = self.root / "hf-model"
        self.hf.mkdir()

    def run_gate(self, output, model=None, extra=()):
        self.probe.write_text("#!/bin/sh\nprintf '%s\\n' '" + output + "'\n")
        self.probe.chmod(0o700)
        return subprocess.run(["bash", str(self.gate), str(model or self.gguf),
            "--prompt", str(self.prompt), "--window", "2", "--logdir", str(self.root / "logs"),
            *extra], capture_output=True, text=True, timeout=5)

    def test_valid_formats_and_canary(self):
        rows = "10 1 1.0000 1 1.0000 0.1000 yes\n11 2 1.0000 2 1.0000 0.1000 yes"
        for model in (self.gguf, self.hf):
            for extra in ((), ("--canary",)):
                with self.subTest(model=model.name, extra=extra):
                    r = self.run_gate(rows, model, extra)
                    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                    self.assertIn("CANARY PASS:" if extra else "  PASS:", r.stdout)

    def test_invalid_tables_fail_even_with_canary(self):
        for rows in ("", "10 1 1.0 1 1.0 0.1 yes", "10 1 nan 1 1.0 0.1 yes\n11 2 1.0 2 1.0 0.1 yes",
                     "10 1 1e999 1 1.0 0.1 yes\n11 2 1.0 2 1.0 0.1 yes",
                     "10 1 1.0 1 1.0 0.1 yes\n10 2 1.0 2 1.0 0.1 yes"):
            for extra in ((), ("--canary",)):
                with self.subTest(rows=rows, extra=extra):
                    r = self.run_gate(rows, extra=extra)
                    self.assertNotEqual(r.returncode, 0, r.stdout)
                    self.assertIn("probe table", r.stdout)

    def test_wide_margin_flip_fails(self):
        r = self.run_gate("10 1 5.0 2 5.0 0.1 NO\n11 2 1.0 2 1.0 0.1 yes")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("UNEXPLAINED", r.stdout)

    def test_small_explained_flip_passes_without_rounding(self):
        r = self.run_gate("10 1 2.000000000e-5 2 2.000000000e-5 3.000000000e-5 NO\n11 2 1.0 2 1.0 0.1 yes")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("SUMMARY flips=1 bad=0", r.stdout)

    def test_explicit_missing_model_fails(self):
        r = self.run_gate("", self.root / "missing")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("requested model does not exist", r.stdout)

    def test_bad_window_and_missing_value_fail(self):
        for extra in (("--window", "0"), ("--window", "bad"), ("--window",)):
            with self.subTest(extra=extra):
                self.assertEqual(self.run_gate("", extra=extra).returncode, 2)

    def test_bad_thresholds_fail(self):
        for extra in (("--max-flips", "bad"), ("--max-flips", "-1"),
                      ("--margin-floor", "nan"), ("--margin-floor", "1e999"),
                      ("--margin-floor", "-0.1")):
            with self.subTest(extra=extra):
                self.assertEqual(self.run_gate("", extra=extra).returncode, 2)

    def test_missing_prompt_fails(self):
        self.prompt.unlink()
        r = self.run_gate("")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("prompt", r.stdout)

    def test_missing_probe_for_explicit_model_fails(self):
        r = subprocess.run(["bash", str(self.gate), str(self.hf),
            "--prompt", str(self.prompt)], capture_output=True, text=True, timeout=5)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("build target/release/argmax-margin-probe", r.stdout)


if __name__ == "__main__":
    unittest.main()
