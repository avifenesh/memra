#!/usr/bin/env python3
"""CPU-only wrapper controls. The measurement is a stub; the verdict is the real one.

The stub prints a canned probe log. Every `--judge` call goes to the built
`argmax-margin-probe`, so the wrapper's verdict branches run on
`memra_engine::argmax_margin::judge`, the function the gate uses on a GPU. Build it first:
`cargo build --release -p memra-engine --bin argmax-margin-probe`.
"""
import hashlib
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

PRIME_MATCH = ("PRIME-GATE: class=MATCH bp_argmax=7 tw_argmax=7 tw_margin=1.5e0 "
               "bp_margin=1.25e0 maxdiff=3.0e-2")
ROWS = "10 1 1.0000 1 1.0000 0.1000 yes\n11 2 1.0000 2 1.0000 0.1000 yes"
# Prefill holds token 1 by 14.4; decode picks token 2 by 2.5. The spread is what such a flip
# needs at least: the mean of the two margins.
WIDE_FLIP = "10 1 1.44e1 2 2.5e0 1.67e1 NO\n11 2 1.0 2 1.0 0.1 yes"
SERVED_MATCH = "SERVED-ARGMAX: rows=2 device_mismatch=0"
SERVED_EAGER_ROUTED = "SERVED-ARGMAX: n/a reason=eager-routed"
BATCH = ("--decode", "batch")


# The gate runs "${CARGO_TARGET_DIR:-target}"/release/argmax-margin-probe; the fixture stubs the
# copy's own target/, so a caller's exported CARGO_TARGET_DIR must not leak in.
FIXTURE_ENV = {k: v for k, v in os.environ.items() if k != "CARGO_TARGET_DIR"}
REPO = pathlib.Path(__file__).resolve().parent.parent
REAL_PROBE = pathlib.Path(os.environ.get("CARGO_TARGET_DIR", REPO / "target")) / "release/argmax-margin-probe"
DEVICE = "CPU fixture device"
FLOOR = "4.000000000e-1"

class GateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # No skip: a missing judge would turn every verdict case into a pass by absence.
        if not REAL_PROBE.is_file():
            raise AssertionError(f"{REAL_PROBE} is missing: build it with "
                                 "`cargo build --release -p memra-engine --bin argmax-margin-probe`")

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
        self.floors = self.root / "tools/argmax-margin-floors.tsv"
        self.floors.write_text("# model\tprompt\tprompt_sha256\tdecode\tdevice\tfloor\trows\tset_by_row\tp50\tp99"
                               "\tp999\tknown_rows\treceipt\n")
        for model in ("model", "hf-model"):
            for decode in ("eager", "batch"):
                self.add_floor(model, self.prompt, decode)

    def add_floor(self, model, prompt, decode, floor=FLOOR, device=DEVICE, known="-"):
        sha = hashlib.sha256(prompt.read_bytes()).hexdigest()
        with self.floors.open("a") as out:
            out.write("\t".join((model, prompt.stem, sha, decode, device, floor, "2040", "2047",
                                 "1.000000000e-1", "2.000000000e-1", "3.000000000e-1", known,
                                 "research/fixture/gate.log")) + "\n")

    def stub_probe(self, output):
        """The stub: `--judge` goes to the real probe, a measurement prints `output`."""
        self.probe.write_text(
            "#!/bin/sh\nif [ \"$1\" = --judge ]; then exec '" + str(REAL_PROBE) + "' \"$@\"; fi\n"
            "printf '%s\\n' \"$@\" > '" + str(self.root / "probe.args")
            + "'\nprintf '%s\\n' '" + output + "'\n")
        self.probe.chmod(0o700)

    def run_gate(self, output, model=None, extra=(), env=None, prime=PRIME_MATCH, device=DEVICE):
        if prime is not None:
            output = output + "\n" + prime if output else prime
        self.stub_probe(output)
        return subprocess.run(["bash", str(self.gate), str(model or self.gguf),
            "--prompt", str(self.prompt), "--window", "2", "--logdir", str(self.root / "logs"),
            "--device", device, *extra], capture_output=True, text=True, timeout=20,
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
        # EAGER-ROUTED keeps the margin rules: a wide flip still fails.
        r = self.run_gate(WIDE_FLIP, extra=BATCH, prime=SERVED_EAGER_ROUTED)
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("WIDE-FLIP pos=10 ", r.stdout)

    def test_batch_cell_keeps_the_margin_rules(self):
        r = self.run_gate(WIDE_FLIP, extra=BATCH, prime=SERVED_MATCH)
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("WIDE-FLIP pos=10 ", r.stdout)
        r = self.run_gate("10 1 2.0e-5 2 2.0e-5 3.0e-5 NO\n11 1 2.0e-5 2 2.0e-5 3.0e-5 NO", extra=BATCH,
                          prime=SERVED_MATCH)
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("TOO-MANY-NEAR-TIES 2 > budget 1", r.stdout)

    def test_bad_decode_value_fails(self):
        for extra in (("--decode", "dc"), ("--decode",)):
            with self.subTest(extra=extra):
                self.assertEqual(self.run_gate("", extra=extra).returncode, 2)

    def test_wide_margin_flip_fails_and_the_row_can_be_examined(self):
        # The red arm: prefill holds its token by 14.4, decode picks another by 2.5. The rule
        # before memra#1199 passed this row (its spread, 16.7, exceeds the smaller margin).
        detail = "detail pos=10 prefill: top1=1 the examination line"
        r = self.run_gate(WIDE_FLIP + "\n" + detail)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("  FLOOR: 0.4000 (largest gap shift over 2040 agreeing rows, set by row 2047; "
                      "p50 0.1000, p99 0.2000, p99.9 0.3000; no calibration flip above it; "
                      "research/fixture/gate.log)", r.stdout)
        self.assertIn("  WIDE-FLIP pos=10 prefill_top1=1 margin_p=14.4000 decode_top1=2 margin_d=2.5000 "
                      "spread=16.7000 gap_shift=16.9000 > floor=0.4000", r.stdout)
        self.assertIn("    " + detail, r.stdout)
        self.assertIn("SUMMARY flips=1 bad=1 near_ties=0 wide_flips=1 ", r.stdout)
        self.assertIn("  FAIL: a wide flip", r.stdout)

    def test_near_tie_inside_the_floor_passes_without_rounding(self):
        r = self.run_gate("10 1 2.000000000e-5 2 2.000000000e-5 3.000000000e-5 NO\n11 2 1.0 2 1.0 0.1 yes")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("near-tie pos=10 ", r.stdout)
        self.assertIn("SUMMARY flips=1 bad=0 near_ties=1 ", r.stdout)
        self.assertIn("  PASS: no wide flip", r.stdout)
        # A flip whose two margins add up to more than the floor is not a near tie.
        r = self.run_gate("10 1 2.5e-1 2 2.5e-1 3.0e-1 NO\n11 2 1.0 2 1.0 0.1 yes")
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("WIDE-FLIP pos=10 ", r.stdout)

    def test_wide_row_fails_without_a_flip(self):
        # Both programs pick token 1, at margins 14.4 and 7.2: no flip, and not rounding.
        r = self.run_gate("10 1 1.44e1 1 7.2e0 7.2e0 yes\n11 2 1.0 2 1.0 0.1 yes")
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("WIDE-ROW pos=10 ", r.stdout)
        self.assertIn("SUMMARY flips=0 bad=1 ", r.stdout)

    def test_cell_without_a_floor_row_is_uncalibrated(self):
        other = "another device"
        # No flip: passes, says so, and does not judge the wide row.
        r = self.run_gate("10 1 1.44e1 1 7.2e0 7.2e0 yes\n11 2 1.0 2 1.0 0.1 yes", device=other)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("  FLOOR: none (UNCALIBRATED: no floor row for model=model prompt=prompt ", r.stdout)
        self.assertIn("  PASS: UNCALIBRATED, no flip.", r.stdout)
        # A flip cannot be classified: it fails, however small.
        tie = "10 1 2.0e-5 2 2.0e-5 3.0e-5 NO\n11 2 1.0 2 1.0 0.1 yes"
        r = self.run_gate(tie, device=other)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("UNCALIBRATED-FLIP pos=10 ", r.stdout)
        # The release battery's switch refuses the cell outright.
        r = self.run_gate(ROWS, device=other, extra=("--require-floor",))
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("argmax-margin-gate: UNCALIBRATED (no floor row", r.stdout)
        self.assertNotIn("  PASS:", r.stdout)
        self.assertEqual(self.run_gate(ROWS, extra=("--require-floor",)).returncode, 0)
        # The canary still has teeth there: the injected flip fails closed.
        r = self.run_gate(ROWS, device=other, extra=("--canary",))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("UNCALIBRATED-FLIP pos=9999 ", r.stdout)
        # A floor belongs to the prompt's bytes: the same name with other bytes has none.
        self.prompt.write_text("Another prompt under the same name.\n")
        r = self.run_gate(tie)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("UNCALIBRATED-FLIP pos=10 ", r.stdout)

    def test_known_row_is_named_and_passes_unless_the_candidate_widens_it(self):
        # Main's calibration flipped row 10 at margins 1.5 and 2.5, above its floor of 0.4: a
        # known row, kept in the floor row with its shift (4.0).
        self.floors.write_text("")
        self.add_floor("model", self.prompt, "eager", known="10:18:1.5e0:15:2.5e0")
        same = "10 18 1.5e0 15 2.5e0 2.6e0 NO\n11 2 1.0 2 1.0 0.1 yes"
        r = self.run_gate(same + "\ndetail pos=10 the examination line")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("known rows (calibration flips above it, memra#1203): 10 (shift 4.0000)", r.stdout)
        self.assertIn("  KNOWN-ROW pos=10 prefill_top1=18 margin_p=1.5000 decode_top1=15 margin_d=2.5000 "
                      "spread=2.6000 gap_shift=4.0000 <= recorded=4.0000 (calibration: prefill_top1=18 "
                      "margin_p=1.5000 decode_top1=15 margin_d=2.5000; memra#1203) floor=0.4000", r.stdout)
        self.assertIn("    detail pos=10 the examination line", r.stdout)
        self.assertIn("SUMMARY flips=1 bad=0 near_ties=0 ", r.stdout)
        self.assertIn("  PASS: no wide flip", r.stdout)
        # A candidate whose gap shift at that row exceeds main's recorded shift fails.
        r = self.run_gate("10 18 1.5e0 15 2.75e0 2.9e0 NO\n11 2 1.0 2 1.0 0.1 yes")
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("  KNOWN-ROW-WIDER pos=10 ", r.stdout)
        self.assertIn("gap_shift=4.2500 > recorded=4.0000 ", r.stdout)
        # The known row covers its own position only.
        r = self.run_gate("10 2 1.0 2 1.0 0.1 yes\n11 18 1.5e0 15 2.5e0 2.6e0 NO")
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("  WIDE-FLIP pos=11 ", r.stdout)

    def test_floors_table_must_exist_and_parse(self):
        self.floors.write_text("model\tprompt\n")
        r = self.run_gate(ROWS)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("MALFORMED: floors line 1", r.stdout)
        self.floors.unlink()
        r = self.run_gate(ROWS)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("floors table", r.stdout)

    def test_a_lane_passes_its_own_floors_table(self):
        lane = self.root / "lane-floors.tsv"
        lane.write_text(self.floors.read_text().replace(FLOOR, "2.000000000e1"))
        r = self.run_gate(WIDE_FLIP, extra=("--floors", str(lane)))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("near-tie pos=10 ", r.stdout)
        self.assertIn("  FLOOR: 20.0000 ", r.stdout)

    def test_canary_names_the_injected_row(self):
        r = self.run_gate(ROWS, extra=("--canary",))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("WIDE-FLIP pos=9999 prefill_top1=111111 margin_p=1000.0000 decode_top1=222222 "
                      "margin_d=1000.0000 spread=1000.0000 gap_shift=2000.0000 > floor=0.4000", r.stdout)
        self.assertIn("CANARY PASS:", r.stdout)

    def test_calibrate_measures_the_whole_prompt_and_prints_the_floors_row(self):
        rows = "10 1 2.0 1 2.25 0.3 yes\n11 2 5.0 2 5.75 0.8 yes\n12 3 2.0e-2 4 3.0e-2 4.0e-2 NO"
        sha = hashlib.sha256(self.prompt.read_bytes()).hexdigest()
        for extra, decode, prime in (((), "eager", PRIME_MATCH),
                                     (BATCH, "batch", "SERVED-ARGMAX: rows=3 device_mismatch=0")):
            with self.subTest(decode=decode):
                r = self.run_gate(rows, extra=(*extra, "--calibrate", "--receipt", "research/lane/cal.log"),
                                  prime=prime)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertEqual(self.probe_args()[:3], [str(self.gguf), str(self.prompt), "all"])
                self.assertIn("  CALIBRATION: rows=3 agreeing=2 flips=1 floor=7.500000000e-1 set_by_row=11 "
                              "p50=7.500000000e-1 p99=7.500000000e-1 p99.9=7.500000000e-1 flips_above_floor=0 "
                              "max_flips_in_12=1", r.stdout)
                self.assertIn(f"  FLOOR-ROW: model\tprompt\t{sha}\t{decode}\t{DEVICE}\t7.500000000e-1\t2\t11\t"
                              "7.500000000e-1\t7.500000000e-1\t7.500000000e-1\t-\tresearch/lane/cal.log", r.stdout)
                # A calibration judges nothing: no verdict line of any kind.
                self.assertNotIn("PASS", r.stdout)
                self.assertNotIn("SUMMARY flips=", r.stdout)
                self.assertIn("  CALIBRATED: add the FLOOR-ROW line", r.stdout)
        self.assertEqual(self.run_gate(rows, extra=("--calibrate", "--canary")).returncode, 2)

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

        for model in (qwen, other_qwen, self.gguf):
            for prompt in (self.prompt, long_prompt):
                if (model, prompt) != (self.gguf, self.prompt):
                    self.add_floor(model.stem, prompt, "eager")

        def gate(model, prompt, flips, window=12, extra=()):
            self.stub_probe(table(flips, window) + "\n" + PRIME_MATCH)
            return subprocess.run(["bash", str(self.gate), str(model), "--prompt", str(prompt),
                "--window", str(window), "--logdir", str(self.root / "logs"), "--device", DEVICE, *extra],
                capture_output=True, text=True, timeout=20, env=FIXTURE_ENV)

        r = gate(qwen, long_prompt, 3)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("SUMMARY flips=3 bad=0", r.stdout)
        self.assertIn("flip budget: 3 (row: Qwen3.8-27B NVFP4 on p4-16k", r.stdout)
        r = gate(qwen, long_prompt, 4)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("TOO-MANY-NEAR-TIES 4 > budget 3", r.stdout)
        r = gate(qwen, long_prompt, 6, window=24)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        r = gate(qwen, long_prompt, 7, window=24)
        self.assertIn("TOO-MANY-NEAR-TIES 7 > budget 6", r.stdout)
        r = gate(qwen, long_prompt, 2, extra=("--max-flips", "1"))
        self.assertIn("TOO-MANY-NEAR-TIES 2 > budget 1", r.stdout)
        self.assertIn("(row: --max-flips)", r.stdout)
        for model, prompt in ((qwen, self.prompt), (self.gguf, long_prompt), (other_qwen, long_prompt)):
            with self.subTest(model=model.name, prompt=prompt.name):
                r = gate(model, prompt, 3)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("TOO-MANY-NEAR-TIES 3 > budget 1", r.stdout)
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
        # --margin-floor went with the rule it patched (memra#1199): it is refused, not ignored.
        for extra in (("--max-flips", "bad"), ("--max-flips", "-1"), ("--margin-floor", "0.5"),
                      ("--floors",), ("--device",)):
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
