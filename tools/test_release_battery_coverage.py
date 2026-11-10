#!/usr/bin/env python3
"""CPU control-flow fixtures for the real release battery; no GPU qualification."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ("kernel-check-27b.cells", "kernel-check-step35.cells")
REQUIRED = [
    line.split("#", 1)[0].strip()
    for name in MANIFESTS
    for line in (ROOT / "tools" / name).read_text().splitlines()
    if line.split("#", 1)[0].strip()
]


STEP_IDENTITY = ("STEP-IDENTITY: 4 batched plain steps against one 4-column verify, 2 rounds after 24 primed "
                 "tokens: 0 of 1986560 logits differ (max |diff| 0), 0 argmax differ -> MATCH\n")


def spec_output(ks=range(1, 9), identity=STEP_IDENTITY):
    return identity + "".join(
        f"[generate_spec K={k}] 32 tok in 1.000s = 31.00 tok/s\n"
        "  acceptance: 8/16 = 50.0%   self-consistency: PASS (identical to plain target)\n"
        for k in ks
    ) + "=== SELF-CONSISTENCY PASS ===\n"


def kernel_output(required=REQUIRED, skips=()):
    return (
        "".join(f"{name} fixture OK\n" for name in required)
        + "".join(f"SKIP {name} (missing fixture artifact)\n" for name in skips)
        + f"\nALL GREEN ({len(required) + len(skips)} cells, {len(skips)} skipped)\n"
    )


def argmax_output(prime="MATCH"):
    return ("  SUMMARY flips=0 bad=0\n"
            + (f"  PRIME-SUMMARY prime={prime}\n" if prime else "")
            + "  PASS: fixture calibrated margin gate\n")


def argmax_batch_output(served="MATCH"):
    return ("  SUMMARY flips=0 bad=0\n"
            + (f"  SERVED-SUMMARY served={served}\n" if served else "")
            + "  PASS: fixture calibrated margin gate\n")


class ReleaseBatteryCoverageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="release-coverage-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tools = self.root / "tools"
        self.bin = self.root / "target" / "release"
        self.fake_path = self.root / "bin"
        for path in (self.tools, self.bin, self.fake_path):
            path.mkdir(parents=True)
        for name in ("release-battery.sh", "release-coverage.py", "resolve-physical-gpu.py", *MANIFESTS):
            source = ROOT / "tools" / name
            if source.exists():
                shutil.copy2(source, self.tools / name)
        self.model = self.root / "own.gguf"
        self.model.write_bytes(b"fixture")
        self.roster = self.root / "roster.tsv"
        self.roster.write_text(f"own\tfixture\t{self.model}\n")
        self.kernel_log = self.root / "kernel.log"
        self.kernel_log.write_text(kernel_output())
        self.spec_log = self.root / "spec.log"
        self.spec_log.write_text(spec_output())
        self.write_executable(self.bin / "kernel-check", '''#!/usr/bin/env bash
printf '%s\n' "$@" > "$FIXTURE_ROOT/kernel.args"
env > "$FIXTURE_ROOT/kernel.env"
cat "$FIXTURE_ROOT/kernel.log"
exit "${FIXTURE_KC_RC:-0}"
''')
        self.write_executable(self.bin / "run-spec", '''#!/usr/bin/env bash
env > "$FIXTURE_ROOT/spec.env"
if [ "${FIXTURE_ENV_SENSITIVE:-0}" = 1 ] && [ -n "${MEMRA_SPEC_K:-}" ]; then
    sed -n '1,3p' "$FIXTURE_ROOT/spec.log"
    echo '=== SELF-CONSISTENCY PASS ==='
else
    cat "$FIXTURE_ROOT/spec.log"
fi
exit "${FIXTURE_SPEC_RC:-0}"
''')
        self.write_executable(self.bin / "argmax-margin-probe", "#!/bin/sh\nexit 0\n")
        self.argmax_log = self.root / "argmax.log"
        self.argmax_log.write_text(argmax_output())
        self.argmax_batch_log = self.root / "argmax-batch.log"
        self.argmax_batch_log.write_text(argmax_batch_output())
        # The served-batch cell calls the same gate with `--decode batch`; the stub answers each
        # cell from its own log so the battery's per-cell reading is what the fixture tests.
        self.write_executable(self.tools / "argmax-margin-gate.sh", """#!/bin/sh
case " $* " in
  *" --decode batch "*)
    env > "$FIXTURE_ROOT/argmax-batch.env"
    printf '%s\\n' "$@" > "$FIXTURE_ROOT/argmax-batch.args"
    cat "$FIXTURE_ROOT/argmax-batch.log"
    exit "${FIXTURE_ARGMAX_BATCH_RC:-0}" ;;
esac
env > "$FIXTURE_ROOT/argmax.env"
printf '%s\\n' "$@" > "$FIXTURE_ROOT/argmax.args"
cat "$FIXTURE_ROOT/argmax.log"
exit "${FIXTURE_ARGMAX_RC:-0}"
""")
        # A fake nvidia-smi must answer THREE distinct query shapes now (memra#264 follow-up,
        # PR #898): the physical-GPU resolver's `index,uuid,memory.free` device list, the
        # per-UUID `memory.free` headroom read, and the per-UUID compute-apps tenant read.
        # One card, ample free memory, no tenants; the fixture is about CONTROL FLOW, not
        # about exercising a real multi-GPU host (which needs an actual box run).
        self.write_executable(self.fake_path / "nvidia-smi", '''#!/bin/sh
case "$*" in
  *"query-gpu=index,uuid,memory.free"*) echo "0, GPU-fixture, 999999999" ;;
  *"query-compute-apps"*) exit 0 ;;
  *) echo 999999999 ;;
esac
''')
        # Exercise the actual battery on macOS too, without changing its Linux sizing arm.
        self.write_executable(self.fake_path / "stat", "#!/bin/sh\necho 1048576\n")
        self.env = dict(
            PATH=f"{self.fake_path}:{os.environ['PATH']}",
            FIXTURE_ROOT=str(self.root), CARD_WAIT_S="0", LC_ALL="C",
        )

    @staticmethod
    def write_executable(path, text):
        path.write_text(text)
        path.chmod(0o755)

    def run_battery(self, expected=0, diagnostic=None, **env):
        result = subprocess.run(
            ["bash", str(self.tools / "release-battery.sh"), "--roster", str(self.roster)],
            cwd=self.root, env={**self.env, **env}, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60,
        )
        if expected == 0:
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("RELEASE BATTERY PASS", result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertNotIn("RELEASE BATTERY PASS", result.stdout)
        if diagnostic:
            self.assertIn(diagnostic, result.stdout)
        return result.stdout

    def test_complete(self):
        self.run_battery()

    def test_card_receipt_names_the_resolved_physical_gpu(self):
        # memra#264 follow-up (PR #898): the card line must name WHICH physical GPU the
        # headroom number came from, not just the number. The fake nvidia-smi's device list
        # names one card, GPU-fixture; that UUID must appear in the receipt.
        stdout = self.run_battery()
        import hashlib
        tag = "sha256:" + hashlib.sha256(b"GPU-fixture").hexdigest()[:12]
        self.assertIn(f"gpu={tag}", stdout)
        # The banked battery.log is public: it names the card by the same digest the cell
        # record seals (uuid_sha256), never by its raw UUID, and the roster by a repo path.
        self.assertNotIn("GPU-fixture", stdout)
        self.assertIn("roster: roster.tsv (", stdout)
        self.assertNotIn(str(self.root), stdout)

    def test_evidence_records_each_actual_producer_before_parsing(self):
        for code in ("0", "1"):
            with self.subTest(kernel_exit=code):
                destination = self.root / f"evidence-{code}"
                result = subprocess.run(
                    ["bash", str(self.tools / "release-battery.sh"), "--roster", str(self.roster),
                     "--evidence-dir", str(destination)], cwd=self.root,
                    env={**self.env, "FIXTURE_KC_RC": code}, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
                self.assertEqual(result.returncode == 0, code == "0", result.stdout)
                rows = [line.split("\t") for line in (destination / "runs.tsv").read_text().splitlines()]
                self.assertEqual([(row[0], row[1]) for row in rows],
                                 [("kernel", "kernel"), ("argmax", "fixture"),
                                  ("argmax-batch", "fixture"), ("spec", "fixture")])
                self.assertEqual(rows[0][2], code)
                self.assertEqual((destination / rows[0][3]).read_text(), self.kernel_log.read_text().rstrip() + "\n")
                self.assertEqual((destination / rows[2][3]).read_text(),
                                 self.argmax_batch_log.read_text().rstrip() + "\n")
                self.assertEqual((destination / rows[3][3]).read_text(), self.spec_log.read_text().rstrip() + "\n")
                if code != "0": self.assertNotIn("RELEASE BATTERY PASS", result.stdout)

    def test_complete_in_reverse_order(self):
        self.spec_log.write_text(spec_output(range(8, 0, -1)))
        self.run_battery()

    def test_step_identity_not_applicable_is_accepted(self):
        self.spec_log.write_text(spec_output(
            identity="STEP-IDENTITY: n/a (no MTP head or no generic verify forward)\n"))
        self.run_battery()

    def test_spec_incomplete_or_invalid(self):
        cases = {
            "narrowed": spec_output([4]),
            "duplicated": spec_output([1] * 8),
            "missing": spec_output(range(1, 8)),
            "out-of-range": spec_output(range(2, 10)),
            "summary-only": "=== SELF-CONSISTENCY PASS ===\n",
            "no-step-identity": spec_output(identity=""),
            "step-identity-mismatch": spec_output(identity=STEP_IDENTITY.replace("-> MATCH", "-> MISMATCH")),
            "step-identity-twin-prime": spec_output(identity=STEP_IDENTITY.replace("-> MATCH", "-> TWIN-PRIME-MISMATCH")),
            "step-identity-twice": spec_output(identity=STEP_IDENTITY * 2),
            "step-identity-after-the-sweep": spec_output(identity="").replace(
                "=== SELF-CONSISTENCY PASS ===", STEP_IDENTITY + "=== SELF-CONSISTENCY PASS ==="),
            "failed": spec_output().replace("self-consistency: PASS", "self-consistency: FAIL", 1),
            "missing-verdict": spec_output().replace("self-consistency: PASS", "unrelated: PASS", 1),
            "duplicate-verdict": spec_output().replace(
                "[generate_spec K=2]", "  self-consistency: PASS (identical to plain target)\n[generate_spec K=2]", 1),
            "sampled": spec_output().replace("identical to plain target", "seeded rerun identical"),
            "missing-summary": spec_output().replace("=== SELF-CONSISTENCY PASS ===", ""),
            "duplicate-summary": spec_output() + "=== SELF-CONSISTENCY PASS ===\n",
            "failed-summary": spec_output() + "=== SELF-CONSISTENCY FAIL ===\n",
            "early-summary": "=== SELF-CONSISTENCY PASS ===\n" + spec_output().replace("=== SELF-CONSISTENCY PASS ===", ""),
            "orphan-verdict": "self-consistency: PASS (identical to plain target)\n" + spec_output(),
            "empty": "",
        }
        for case, output in cases.items():
            with self.subTest(case=case):
                self.spec_log.write_text(output)
                self.run_battery(expected=1, diagnostic="run-spec  FAIL")

    def test_argmax_arm_requires_the_batched_prime_verdict(self):
        # The battery runs argmax-margin-gate.sh and not run-gen, and run-spec primes both its
        # reference and its candidate through prime_cache, so the gate's PRIME-SUMMARY is the
        # only prime-vs-tokenwise check a release gets. A gate PASS without it cannot pass.
        for prime in ("MATCH", "FLIP-NEARTIE", "SKIP:tokenwise-served"):
            with self.subTest(prime=prime):
                self.argmax_log.write_text(argmax_output(prime))
                self.assertIn(f"prime={prime}", self.run_battery())
        for prime in ("", "MISMATCH-STRUCTURED", "SKIP:gate-off", "SKIP:tokenwise-forced", "SKIP:short-prompt"):
            with self.subTest(prime=prime):
                self.argmax_log.write_text(argmax_output(prime))
                self.run_battery(expected=1, diagnostic="batched-prime arm reported prime=")

    def test_argmax_arm_runs_the_default_prime_program(self):
        self.run_battery(MEMRA_PRIME_GATE="0", MEMRA_PRIME_TOKENWISE="1")
        argmax_env = dict(line.split("=", 1) for line in (self.root / "argmax.env").read_text().splitlines()
                          if "=" in line)
        for name in ("MEMRA_PRIME_GATE", "MEMRA_PRIME_TOKENWISE"):
            self.assertNotIn(name, argmax_env)

    def test_served_batch_arm_requires_its_own_pass_and_served_verdict(self):
        # The eager cell cannot stand in for the served batched decode program: the battery runs
        # the gate a second time with --decode batch and accepts only served=MATCH from it.
        stdout = self.run_battery()
        self.assertIn("fixture  argmax-margin-batch  PASS  SUMMARY flips=0 bad=0 served=MATCH", stdout)
        self.assertIn("fixture  argmax-margin  PASS", stdout)
        for served in ("", "MISMATCH", "MISSING", "EAGER", "MATCHED"):
            with self.subTest(served=served):
                self.argmax_batch_log.write_text(argmax_batch_output(served))
                self.run_battery(expected=1, diagnostic="served-batch arm reported served=")
        # The probe's routing read (HybridModel::served_decode_is_eager) names a model the
        # scheduler decodes eagerly; its table measured decode_step, the program that serves it.
        self.argmax_batch_log.write_text(argmax_batch_output("EAGER-ROUTED"))
        stdout = self.run_battery()
        self.assertIn("fixture  argmax-margin-batch  PASS  SUMMARY flips=0 bad=0 served=EAGER-ROUTED", stdout)
        self.argmax_batch_log.write_text(argmax_batch_output())
        self.run_battery(expected=1, diagnostic="argmax-margin-batch  FAIL", FIXTURE_ARGMAX_BATCH_RC="1")
        self.argmax_batch_log.write_text("argmax-margin-gate: SKIP (no probe model on this rig)\n")
        self.run_battery(expected=1, diagnostic="argmax-margin-batch  REFUSED   gate skipped itself")

    def test_served_batch_arm_runs_the_default_served_program(self):
        self.run_battery(MEMRA_SERVE_B1FAST="1")
        batch_env = dict(line.split("=", 1) for line in (self.root / "argmax-batch.env").read_text().splitlines()
                         if "=" in line)
        self.assertNotIn("MEMRA_SERVE_B1FAST", batch_env)
        args = (self.root / "argmax-batch.args").read_text().splitlines()
        self.assertEqual(args[:4], [str(self.model), "--decode", "batch", "--require-floor"])

    def test_cell_on_an_unmaintained_device_class_is_not_judged(self):
        # The gate's NOT-JUDGED outcome (a device class with no push-gated target and no floor
        # row): the battery names the cell, counts it, and neither passes nor fails on it.
        line = ("argmax-margin-gate: NOT-JUDGED (not judged: device class h100 not maintained, no floor row; "
                "device=NVIDIA H100 80GB HBM3)\n")
        self.argmax_log.write_text(line)
        self.argmax_batch_log.write_text(line)
        stdout = self.run_battery()
        self.assertIn("fixture  argmax-margin  NOT-JUDGED  (not judged: device class h100 not maintained", stdout)
        self.assertIn("fixture  argmax-margin-batch  NOT-JUDGED  (not judged: device class h100", stdout)
        self.assertIn("argmax cells NOT JUDGED: 2 (device class not maintained, no floor row or no stored rows)", stdout)
        self.assertNotIn("argmax-margin  PASS", stdout)
        self.assertNotIn("argmax-margin-batch  PASS", stdout)
        # The line with a failing exit code is a failure, not a quiet skip.
        self.run_battery(expected=1, diagnostic="argmax-margin  FAIL", FIXTURE_ARGMAX_RC="1")

    def test_listed_argmax_cells_are_the_gate_calls_the_battery_makes(self):
        # tools/check-argmax-floors.py reads --list-argmax-cells to demand a floor row per cell.
        # The list is honest only if it names exactly the calls the loop makes: one eager and one
        # --decode batch call per roster model, both on the listed prompt.
        listed = subprocess.run(
            ["bash", str(self.tools / "release-battery.sh"), "--roster", str(self.roster), "--list-argmax-cells"],
            cwd=self.root, env=self.env, text=True, capture_output=True, timeout=60)
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertEqual(listed.stdout.splitlines(),
                         ["fixture\town\tboard-2048\teager", "fixture\town\tboard-2048\tbatch"])
        self.run_battery()
        eager = (self.root / "argmax.args").read_text().splitlines()
        batch = (self.root / "argmax-batch.args").read_text().splitlines()
        self.assertNotIn("--decode", eager)
        self.assertEqual(batch[1:3], ["--decode", "batch"])
        for args in (eager, batch):
            self.assertEqual(args[0], str(self.model))
            prompt = args[args.index("--prompt") + 1]
            self.assertTrue(prompt.endswith("/research/e2e/prompts/board-2048.txt"), prompt)

    def test_cell_without_a_calibrated_floor_is_refused(self):
        # A release never passes an argmax cell as UNCALIBRATED: both cells call the gate with
        # --require-floor, and its refusal is reported as one, not as a gate failure.
        self.run_battery()
        args = (self.root / "argmax.args").read_text().splitlines()
        self.assertEqual(args[:2], [str(self.model), "--require-floor"])
        refusal = "argmax-margin-gate: UNCALIBRATED (no floor row in tools/argmax-margin-floors.tsv for model=m)\n"
        passed = self.argmax_log.read_text()
        self.argmax_log.write_text(refusal)
        self.run_battery(expected=1, diagnostic="argmax-margin  REFUSED   no calibrated floor for this cell",
                         FIXTURE_ARGMAX_RC="1")
        self.argmax_log.write_text(passed)
        self.argmax_batch_log.write_text(refusal)
        self.run_battery(expected=1, diagnostic="argmax-margin-batch  REFUSED   no calibrated floor for this cell",
                         FIXTURE_ARGMAX_BATCH_RC="1")

    def test_cell_without_main_stored_rows_is_refused(self):
        # memra#1204: a cell with a floor row but no stored rows of main is refused, not failed.
        refusal = ("argmax-margin-gate: NO-STORED-ROWS (no stored rows in tools/argmax-margin-rows for "
                   "model=m model_sha256=aa prompt=board-2048)\n")
        passed = self.argmax_log.read_text()
        self.argmax_log.write_text(refusal)
        self.run_battery(expected=1, diagnostic="argmax-margin  REFUSED   no stored rows of main for this cell: "
                         "NO-STORED-ROWS (no stored rows in tools/argmax-margin-rows", FIXTURE_ARGMAX_RC="1")
        self.argmax_log.write_text(passed)
        self.argmax_batch_log.write_text(refusal)
        self.run_battery(expected=1, diagnostic="argmax-margin-batch  REFUSED   no stored rows of main for this cell",
                         FIXTURE_ARGMAX_BATCH_RC="1")

    def test_nonzero_exit_cannot_pass(self):
        self.run_battery(expected=1, FIXTURE_SPEC_RC="1")
        self.run_battery(expected=1, FIXTURE_KC_RC="1")

    def test_sanitizes_coverage_environment(self):
        self.run_battery(
            MEMRA_SPEC_K="4", MEMRA_PROMPT_DIR="alternate-mode", MEMRA_GEN_ONLY="1",
            MEMRA_SPEC_TEMP="0.8", MEMRA_NGEN="1", MEMRA_KC_FAST="1", MEMRA_KC_ONLY="dtype5",
            FIXTURE_ENV_SENSITIVE="1",
        )
        spec_env = dict(line.split("=", 1) for line in (self.root / "spec.env").read_text().splitlines())
        for name in ("MEMRA_SPEC_K", "MEMRA_PROMPT_DIR", "MEMRA_GEN_ONLY"):
            self.assertNotIn(name, spec_env)
        self.assertEqual(spec_env["MEMRA_SPEC_TEMP"], "0")
        self.assertEqual(spec_env["MEMRA_NGEN"], "32")
        kc_env = dict(line.split("=", 1) for line in (self.root / "kernel.env").read_text().splitlines())
        for name in ("MEMRA_KC_FAST", "MEMRA_KC_ONLY"):
            self.assertNotIn(name, kc_env)

    def test_passes_required_manifests_to_kernel_check(self):
        self.run_battery()
        args = (self.root / "kernel.args").read_text().splitlines()
        self.assertEqual(args, [str(self.model), "--require-manifest", str(self.tools / MANIFESTS[0]),
                                "--require-manifest", str(self.tools / MANIFESTS[1])])

    def test_kernel_incomplete_or_invalid(self):
        cases = {
            "summary-only": "ALL GREEN (100 cells, 0 skipped)\n",
            "missing-required": kernel_output(REQUIRED[1:]),
            "skipped-required": kernel_output(REQUIRED[1:], [REQUIRED[0]]),
            "failed-required": kernel_output().replace("fixture OK", "fixture FAIL", 1),
            "excess-skips": kernel_output(skips=[f"optional-{n}" for n in range(12)]),
            "unaccounted-skip": kernel_output(skips=["optional"]).replace("1 skipped", "0 skipped"),
            "unnamed-skip": kernel_output().replace("0 skipped", "1 skipped"),
            "duplicate-skip": "SKIP optional (missing fixture artifact)\n" + kernel_output(skips=["optional"]),
            "filtered": kernel_output(skips=["optional"]).replace("missing fixture artifact", "capability filtered by MEMRA_KC_ONLY=dtype5"),
            "zero-executed": kernel_output().replace(f"{len(REQUIRED)} cells", "0 cells"),
            "duplicate-summary": kernel_output() + f"ALL GREEN ({len(REQUIRED)} cells, 0 skipped)\n",
            "malformed-summary": kernel_output().replace("0 skipped", "unknown skipped"),
            "trailing-failure": kernel_output() + "late failure\n",
        }
        for case, output in cases.items():
            with self.subTest(case=case):
                self.kernel_log.write_text(output)
                self.run_battery(expected=1, diagnostic="kernel-check           FAIL", MEMRA_CI_KC_SKIP_BUDGET="999")

    def test_named_skips_within_budget_are_recorded(self):
        self.kernel_log.write_text(kernel_output(skips=[f"optional-{n}" for n in range(11)]))
        output = self.run_battery()
        self.assertIn("11 skipped", output)
        self.assertIn("SKIP optional-10 (missing fixture artifact)", output)

    def test_missing_manifest_refuses(self):
        (self.tools / MANIFESTS[0]).unlink()
        self.run_battery(expected=1)

    def test_repeated_required_subcases_and_recovered_skip(self):
        self.kernel_log.write_text(
            f"SKIP {REQUIRED[0]} (first artifact unavailable)\n"
            f"{REQUIRED[0]}[extra-shape] fixture OK\n" + kernel_output()
        )
        self.run_battery()

    def test_manifest_changes_are_enforced(self):
        with (self.tools / MANIFESTS[0]).open("a") as manifest:
            manifest.write("\nnew-required-cell\n")
        self.run_battery(expected=1, diagnostic="new-required-cell")
        self.kernel_log.write_text(kernel_output([*REQUIRED, "new-required-cell"]))
        self.run_battery()


if __name__ == "__main__":
    unittest.main()
