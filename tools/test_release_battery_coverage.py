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


def spec_output(ks=range(1, 9)):
    return "".join(
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
        for name in ("release-battery.sh", "release-coverage.py", *MANIFESTS):
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
    sed -n '1,2p' "$FIXTURE_ROOT/spec.log"
    echo '=== SELF-CONSISTENCY PASS ==='
else
    cat "$FIXTURE_ROOT/spec.log"
fi
exit "${FIXTURE_SPEC_RC:-0}"
''')
        self.write_executable(self.bin / "argmax-margin-probe", "#!/bin/sh\nexit 0\n")
        self.write_executable(self.tools / "argmax-margin-gate.sh", """#!/bin/sh
echo '  SUMMARY flips=0 bad=0'
echo '  PASS: fixture calibrated margin gate'
""")
        self.write_executable(self.fake_path / "nvidia-smi", "#!/bin/sh\necho 999999999\n")
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
            ["bash", str(self.tools / "release-battery.sh"), "--generic-only",
             "--roster", str(self.roster)],
            cwd=self.root, env={**self.env, **env}, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60,
        )
        if expected == 0:
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("GENERIC BATTERY PASS", result.stdout)
            self.assertNotIn("RELEASE BATTERY PASS", result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertNotIn("GENERIC BATTERY PASS", result.stdout)
            self.assertNotIn("RELEASE BATTERY PASS", result.stdout)
        if diagnostic:
            self.assertIn(diagnostic, result.stdout)
        return result.stdout

    def test_complete(self):
        self.run_battery()

    def test_full_release_without_serving_record_refuses_before_producers(self):
        result = subprocess.run(
            ["bash", str(self.tools / "release-battery.sh"), "--roster", str(self.roster)],
            cwd=self.root, env=self.env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=60,
        )
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("required --serving-record is missing", result.stdout)
        self.assertNotIn("BATTERY PASS", result.stdout)
        self.assertFalse((self.root / "kernel.args").exists())

    def test_evidence_records_each_actual_producer_before_parsing(self):
        for code in ("0", "1"):
            with self.subTest(kernel_exit=code):
                destination = self.root / f"evidence-{code}"
                result = subprocess.run(
                    ["bash", str(self.tools / "release-battery.sh"), "--generic-only",
                     "--roster", str(self.roster),
                     "--evidence-dir", str(destination)], cwd=self.root,
                    env={**self.env, "FIXTURE_KC_RC": code}, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
                self.assertEqual(result.returncode == 0, code == "0", result.stdout)
                rows = [line.split("\t") for line in (destination / "runs.tsv").read_text().splitlines()]
                self.assertEqual([(row[0], row[1]) for row in rows],
                                 [("kernel", "kernel"), ("argmax", "fixture"), ("spec", "fixture")])
                self.assertEqual(rows[0][2], code)
                self.assertEqual((destination / rows[0][3]).read_text(), self.kernel_log.read_text().rstrip() + "\n")
                self.assertEqual((destination / rows[2][3]).read_text(), self.spec_log.read_text().rstrip() + "\n")
                if code != "0": self.assertNotIn("RELEASE BATTERY PASS", result.stdout)

    def test_complete_in_reverse_order(self):
        self.spec_log.write_text(spec_output(range(8, 0, -1)))
        self.run_battery()

    def test_spec_incomplete_or_invalid(self):
        cases = {
            "narrowed": spec_output([4]),
            "duplicated": spec_output([1] * 8),
            "missing": spec_output(range(1, 8)),
            "out-of-range": spec_output(range(2, 10)),
            "summary-only": "=== SELF-CONSISTENCY PASS ===\n",
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
