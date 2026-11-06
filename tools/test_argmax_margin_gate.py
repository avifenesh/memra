#!/usr/bin/env python3
"""CPU-only wrapper controls; stubs exercise the real parsing and verdict branches."""
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

PRIME_MATCH = ("PRIME-GATE: class=MATCH bp_argmax=7 tw_argmax=7 tw_margin=1.5e0 "
               "bp_margin=1.25e0 maxdiff=3.0e-2")
ROWS = "10 1 1.0000 1 1.0000 0.1000 yes\n11 2 1.0000 2 1.0000 0.1000 yes"
SERVED_MATCH = "SERVED-ARGMAX: rows=2 device_mismatch=0"
SERVED_EAGER_ROUTED = "SERVED-ARGMAX: n/a reason=eager-routed"
BATCH = ("--decode", "batch")


# The gate runs "${CARGO_TARGET_DIR:-target}"/release/argmax-margin-probe; the fixture stubs the
# copy's own target/, so a caller's exported CARGO_TARGET_DIR must not leak in.
FIXTURE_ENV = {k: v for k, v in os.environ.items() if k != "CARGO_TARGET_DIR"}

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

    def run_gate(self, output, model=None, extra=(), env=None, prime=PRIME_MATCH):
        if prime is not None:
            output = output + "\n" + prime if output else prime
        self.probe.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > '" + str(self.root / "probe.args")
                              + "'\nprintf '%s\\n' '" + output + "'\n")
        self.probe.chmod(0o700)
        return subprocess.run(["bash", str(self.gate), str(model or self.gguf),
            "--prompt", str(self.prompt), "--window", "2", "--logdir", str(self.root / "logs"),
            *extra], capture_output=True, text=True, timeout=5,
            env={**FIXTURE_ENV, **(env or {})})

    def test_raw_log_path_is_relative_to_the_evidence_root(self):
        rows = "10 1 1.0000 1 1.0000 0.1000 yes\n11 2 1.0000 2 1.0000 0.1000 yes"
        r = self.run_gate(rows, env={"MEMRA_GATE_LOG_ROOT": str(self.root)})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("  raw: logs/gate-model-prompt.log", r.stdout)
        self.assertNotIn(str(self.root), r.stdout)
        r = self.run_gate(rows)
        self.assertIn(f"  raw: {self.root}/logs/gate-model-prompt.log", r.stdout)

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

    def test_structured_batched_prime_fails(self):
        prime = PRIME_MATCH.replace("class=MATCH", "class=MISMATCH-STRUCTURED")
        r = self.run_gate(ROWS, prime=prime)
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("PRIME-SUMMARY prime=MISMATCH-STRUCTURED", r.stdout)
        self.assertIn("batched serving prime (prime_cache) diverges structurally", r.stdout)

    def test_neartie_prime_flip_and_named_skip_pass_and_are_reported(self):
        for prime, summary in (
            (PRIME_MATCH.replace("class=MATCH", "class=FLIP-NEARTIE"), "prime=FLIP-NEARTIE"),
            ("PRIME-GATE: SKIP reason=tokenwise-served", "prime=SKIP:tokenwise-served"),
        ):
            with self.subTest(prime=prime):
                r = self.run_gate(ROWS, prime=prime)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertIn("PRIME-SUMMARY " + summary, r.stdout)

    def test_missing_or_malformed_prime_line_fails_even_with_canary(self):
        for prime, word in ((None, "MISSING"), (PRIME_MATCH + "\n" + PRIME_MATCH, "MISSING"),
                            ("PRIME-GATE: class=MAYBE x=1", "MALFORMED"),
                            ("PRIME-GATE: SKIP", "MALFORMED")):
            for extra in ((), ("--canary",)):
                with self.subTest(prime=prime, extra=extra):
                    r = self.run_gate(ROWS, extra=extra, prime=prime)
                    self.assertNotEqual(r.returncode, 0, r.stdout)
                    self.assertIn("PRIME-GATE line is " + word, r.stdout)

    def test_canary_injects_a_structured_prime_line(self):
        r = self.run_gate(ROWS, extra=("--canary",))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("the reader saw: MISMATCH-STRUCTURED", r.stdout)

    def probe_args(self):
        return (self.root / "probe.args").read_text().splitlines()

    def test_default_cell_calls_the_probe_as_before(self):
        r = self.run_gate(ROWS)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.probe_args(), [str(self.gguf), str(self.prompt), "2"])
        self.assertIn("decode=eager", r.stdout)
        self.assertNotIn("SERVED-SUMMARY", r.stdout)

    def test_batch_cell_runs_the_served_program_and_reports_it(self):
        for extra in (BATCH, (*BATCH, "--canary")):
            with self.subTest(extra=extra):
                r = self.run_gate(ROWS, extra=extra, prime=SERVED_MATCH)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertEqual(self.probe_args(),
                                 [str(self.gguf), str(self.prompt), "2", "--decode", "batch"])
                self.assertIn("decode=batch", r.stdout)
                self.assertIn("SERVED-SUMMARY served=MATCH", r.stdout)
                self.assertNotIn("PRIME-SUMMARY", r.stdout)
                self.assertIn("CANARY PASS:" if "--canary" in extra else "  PASS:", r.stdout)
                self.assertTrue(list((self.root / "logs").glob("*.batch*.log")), r.stdout)

    def test_batch_cell_device_token_mismatch_fails(self):
        r = self.run_gate(ROWS, extra=BATCH, prime=SERVED_MATCH.replace("=0", "=1"))
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("SERVED-SUMMARY served=MISMATCH", r.stdout)
        self.assertIn("device-sampled token differs", r.stdout)

    def test_batch_cell_missing_or_malformed_served_line_fails_even_with_canary(self):
        # A probe that ignores --decode runs the eager program and prints the prime line instead.
        for served, word in ((None, "MISSING"), (PRIME_MATCH, "MISSING"),
                             (SERVED_MATCH + "\n" + SERVED_MATCH, "MISSING"),
                             ("SERVED-ARGMAX: rows=3 device_mismatch=0", "MALFORMED"),
                             ("SERVED-ARGMAX: rows=2 device_mismatch=x", "MALFORMED"),
                             ("SERVED-ARGMAX: rows=2", "MALFORMED"),
                             ("SERVED-ARGMAX: n/a reason=other", "MALFORMED"),
                             ("SERVED-ARGMAX: n/a", "MALFORMED"),
                             (SERVED_EAGER_ROUTED + "\n" + SERVED_MATCH, "MISSING")):
            for extra in (BATCH, (*BATCH, "--canary")):
                with self.subTest(served=served, extra=extra):
                    r = self.run_gate(ROWS, extra=extra, prime=served)
                    self.assertNotEqual(r.returncode, 0, r.stdout)
                    self.assertIn("SERVED-ARGMAX line is " + word, r.stdout)

    def test_batch_canary_injects_a_served_mismatch_line(self):
        r = self.run_gate(ROWS, extra=(*BATCH, "--canary"), prime=SERVED_MATCH)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("served-token mismatch line; the reader saw: MISMATCH", r.stdout)

    def test_eager_routed_model_passes_on_the_table_and_says_so(self):
        # The probe's routing read says the scheduler decodes this model eagerly: the table then
        # measured decode_step, and the cell reports EAGER-ROUTED instead of a served-token check.
        for extra in (BATCH, (*BATCH, "--canary")):
            with self.subTest(extra=extra):
                r = self.run_gate(ROWS, extra=extra, prime=SERVED_EAGER_ROUTED)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertIn("SERVED-SUMMARY served=EAGER-ROUTED", r.stdout)
                self.assertIn("CANARY PASS:" if "--canary" in extra else "  PASS:", r.stdout)
        r = self.run_gate(ROWS, extra=(*BATCH, "--canary"), prime=SERVED_EAGER_ROUTED)
        self.assertIn("served-token mismatch line; the reader saw: MISMATCH", r.stdout)
        # EAGER-ROUTED keeps the margin rules: an unexplained flip still fails.
        r = self.run_gate("10 1 5.0 2 5.0 0.1 NO\n11 2 1.0 2 1.0 0.1 yes", extra=BATCH,
                          prime=SERVED_EAGER_ROUTED)
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("UNEXPLAINED", r.stdout)

    def test_batch_cell_keeps_the_margin_rules(self):
        r = self.run_gate("10 1 5.0 2 5.0 0.1 NO\n11 2 1.0 2 1.0 0.1 yes", extra=BATCH, prime=SERVED_MATCH)
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("UNEXPLAINED", r.stdout)
        r = self.run_gate("10 1 2.0e-5 2 2.0e-5 3.0e-5 NO\n11 1 2.0e-5 2 2.0e-5 3.0e-5 NO", extra=BATCH,
                          prime=SERVED_MATCH)
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("TOO-MANY-FLIPS 2 > 1", r.stdout)

    def test_bad_decode_value_fails(self):
        for extra in (("--decode", "dc"), ("--decode",)):
            with self.subTest(extra=extra):
                self.assertEqual(self.run_gate("", extra=extra).returncode, 2)

    def test_wide_margin_flip_fails(self):
        r = self.run_gate("10 1 5.0 2 5.0 0.1 NO\n11 2 1.0 2 1.0 0.1 yes")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("UNEXPLAINED", r.stdout)

    def test_small_explained_flip_passes_without_rounding(self):
        r = self.run_gate("10 1 2.000000000e-5 2 2.000000000e-5 3.000000000e-5 NO\n11 2 1.0 2 1.0 0.1 yes")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("SUMMARY flips=1 bad=0", r.stdout)

    def test_per_prompt_budget_covers_only_its_model_and_prompt(self):
        def table(flips, window=12):
            return "\n".join(f"{10 + i} 1 0.0100 {2 if i < flips else 1} 0.0100 0.0500 "
                             + ("NO" if i < flips else "yes") for i in range(window))
        qwen = self.root / "Qwen3.8-27B-NVFP4-test.gguf"
        qwen.touch()
        other_qwen = self.root / "Qwen3.8-27B-Q8_0-test.gguf"
        other_qwen.touch()
        long_prompt = self.root / "p4-16k.txt"
        long_prompt.write_text("A long prompt fixture.\n")

        def gate(model, prompt, flips, window=12, extra=()):
            self.probe.write_text("#!/bin/sh\nprintf '%s\\n' '" + table(flips, window) + "\n" + PRIME_MATCH + "'\n")
            self.probe.chmod(0o700)
            return subprocess.run(["bash", str(self.gate), str(model), "--prompt", str(prompt),
                "--window", str(window), "--logdir", str(self.root / "logs"), *extra],
                capture_output=True, text=True, timeout=5, env=FIXTURE_ENV)

        r = gate(qwen, long_prompt, 3)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("SUMMARY flips=3 bad=0", r.stdout)
        self.assertIn("flip budget: 3 (row: Qwen3.8-27B NVFP4 on p4-16k", r.stdout)
        r = gate(qwen, long_prompt, 4)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("TOO-MANY-FLIPS 4 > 3", r.stdout)
        r = gate(qwen, long_prompt, 6, window=24)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        r = gate(qwen, long_prompt, 7, window=24)
        self.assertIn("TOO-MANY-FLIPS 7 > 6", r.stdout)
        r = gate(qwen, long_prompt, 2, extra=("--max-flips", "1"))
        self.assertIn("TOO-MANY-FLIPS 2 > 1", r.stdout)
        self.assertIn("(row: --max-flips)", r.stdout)
        for model, prompt in ((qwen, self.prompt), (self.gguf, long_prompt), (other_qwen, long_prompt)):
            with self.subTest(model=model.name, prompt=prompt.name):
                r = gate(model, prompt, 3)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("TOO-MANY-FLIPS 3 > 1", r.stdout)
                self.assertIn("(row: default)", r.stdout)

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
            "--prompt", str(self.prompt)], capture_output=True, text=True, timeout=5, env=FIXTURE_ENV)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("build argmax-margin-probe", r.stdout)


if __name__ == "__main__":
    unittest.main()
