"""Adversarial CPU tests. Wrapper execution uses only temporary shell stubs."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("fast_gate_plan_tests", HERE / "plan.py")
plan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plan)

CONTEXTS = {"compiler", "build", "runtime", "numeric", "harness", "oracle", "model",
            "binary", "hardware", "topology", "request"}
MIMO_GATES = {"MiMo text checkpoint parity", "MiMo vision", "MiMo audio", "MiMo MTP",
              "MiMo serving and topology"}


class PlannerTests(unittest.TestCase):
    def assert_development_only(self, value):
        self.assertIs(value["qualification"], False)
        self.assertEqual(value["gpu_selection"], "shadow-only")
        self.assertEqual(set(value["context_contract"]), CONTEXTS)

    def test_declared_cpu_input_does_not_compile_or_load_models(self):
        p = plan.make_plan(["tools/q35-cold-mixed-gate.py"])
        validation = {"mode": "scoped", "packages": [], "jobs": {},
                      "cpu_contracts": [{"id": "q35-cache", "inputs": ["tools/q35-cold-mixed-gate.py"]}],
                      "native": {"requirements": ["live evidence still required for the issue"]}}
        plan.attach_validation(p, validation)
        self.assertEqual(p["decision"], "cpu-only")
        self.assertEqual(p["kernel_scope"], "none")
        self.assertEqual(p["probes"], [])
        self.assertEqual(p["cpu_contracts"], ["q35-cache"])
        self.assertTrue(p["native_evidence_obligations"])
        self.assert_development_only(p)

    def test_research_oracle_inputs_keep_native_expansion(self):
        validation = {"mode": "scoped", "packages": [], "cpu_contracts": [], "native": {"requirements": []}}
        for path in ("research/gemma4-bringup/e4b-chat-watercycle-ids.txt",
                     "research/gemma4-bringup/depth-prompt-1736-ids.txt",
                     "research/chunk-invariance-20260805/prompt-pp6257.txt",
                     "research/e2e/prompts/short.txt", "research/campaign/fixture.md"):
            with self.subTest(path=path):
                p = plan.make_plan([path])
                original = {key: p[key] for key in ("decision", "expansion", "required_unregistered", "probes")}
                plan.attach_validation(p, validation)
                self.assertEqual(p['decision'], 'expand')
                self.assertEqual({key: p[key] for key in original}, original)

    def test_plain_docs_can_use_cpu_content_checks(self):
        p = plan.make_plan(["docs/TESTING.md"])
        plan.attach_validation(p, {"mode": "scoped", "packages": [], "cpu_contracts": [], "native": {"requirements": []}})
        self.assertEqual(p['decision'], 'cpu-only')

    def test_mixed_cpu_contract_and_oracle_cannot_hide_native_work(self):
        p = plan.make_plan(["tools/q35-cold-mixed-gate.py", "research/e2e/prompts/short.txt"])
        plan.attach_validation(p, {"mode": "scoped", "packages": [],
                                  "cpu_contracts": [{"id": "q35-cache", "inputs": ["tools/q35-cold-mixed-gate.py"]}],
                                  "native": {"requirements": ["live evidence"]}})
        self.assertEqual(p['decision'], 'expand')
        self.assertTrue(p['required_unregistered'])

    def test_explicit_native_probe_or_context_prevents_cpu_shortcut(self):
        validation = {"mode": "scoped", "packages": [], "cpu_contracts": [], "native": {"requirements": []}}
        for explicit, context in [(True, ()), (False, ("compiler",))]:
            p = plan.make_plan(["tools/q35-cold-mixed-gate.py"])
            plan.attach_validation(p, validation, explicit, context)
            self.assertEqual(p["decision"], "expand")

    def test_sampled_cuda_selects_device_sampler_and_acceptance_oracles(self):
        value = plan.make_plan(["crates/memra-engine/cu/spec_sample.cu"])
        self.assertTrue({"samp", "accept"} <= set(value["probes"]))
        self.assertTrue({"q35spec", "g31spec"} <= set(value["spec_probes"]))
        self.assertIn("cuda", value["transitive_impact"])
        self.assertIn("native-checkpoint", value["transitive_impact"])
        self.assert_development_only(value)

    def test_prefill_numeric_change_keeps_measured_acceptance_blind_spot(self):
        value = plan.make_plan(["crates/memra-engine/cu/mmq_nvfp4_w4a8.cu"])
        self.assertIn("accept", value["probes"])
        self.assertIn("nvfp4-gemm", value["kernel_sections"])
        self.assertEqual(value["decision"], "expand")

    def test_spill_changes_require_forced_cache_regime(self):
        for name in ("moe_cache", "spill", "spill_pread", "cpu_experts", "hybrid_forward"):
            with self.subTest(name=name):
                value = plan.make_plan([f"crates/memra-engine/src/{name}.rs"])
                self.assertIn("q35slru", value["probes"])
        _, _, models = plan.registry(HERE)
        environment = models["q35slru"][-1].split()
        self.assertIn("MEMRA_MOE_RESIDENT=0", environment)
        self.assertIn("MEMRA_MOE_SLOTS=1024", environment)

    def test_shared_scheduler_expands_and_retains_serving_probes(self):
        value = plan.make_plan(["crates/memra-server/src/worker.rs"])
        self.assertTrue({"sstress", "accept", "tickinv35", "b2geo35"} <= set(value["probes"]))
        self.assertEqual(value["decision"], "expand")
        self.assertEqual(value["kernel_scope"], "all")
        self.assertTrue({"shared-runtime", "model-plans", "mimo", "native-checkpoint"}
                        <= value["transitive_impact"].keys())
        self.assertIn("dependency:shared-runtime", value["transitive_impact"]["model-plans"])

    def test_mimo_retains_text_vision_audio_mtp_and_serving_scope(self):
        for path in ("crates/memra-engine/src/mimo.rs",
                     "crates/memra-gguf/src/model_packs/mimo_v2_5/mod.rs"):
            with self.subTest(path=path):
                value = plan.make_plan([path])
                self.assertEqual(set(value["required_unregistered"]), MIMO_GATES)
                self.assertEqual(value["decision"], "expand")
                self.assert_development_only(value)

    def test_included_research_fixture_cannot_hide_behind_no_gate_mapping(self):
        value = plan.make_plan(["research/census/expected-plan.json"])
        self.assertEqual(value["decision"], "expand")
        self.assertTrue({"generated-and-oracle-inputs", "harness", "native-checkpoint"}
                        <= value["transitive_impact"].keys())
        self.assertTrue(value["reasons"][0]["map_lines"])

    def test_harness_build_config_and_unknown_inputs_expand(self):
        cases = {
            "tools/prime-batch-exact-gate.sh": "harness",
            "Cargo.lock": "build-inputs",
            ".cargo/config.toml": "build-inputs",
            "crates/memra-engine/build.rs": "build-inputs",
            "rust-toolchain.toml": "build-inputs",
            "unknown/generated-input.bin": None,
        }
        for path, node in cases.items():
            with self.subTest(path=path):
                value = plan.make_plan([path])
                self.assertEqual(value["decision"], "expand")
                self.assertEqual(value["kernel_scope"], "all")
                if node:
                    self.assertIn(node, value["transitive_impact"])
                else:
                    self.assertTrue(any("unmodelled input" in e for e in value["expansion"]))

    def test_every_native_context_category_invalidates_native_reuse(self):
        for context in sorted(CONTEXTS):
            with self.subTest(context=context):
                value = plan.make_plan([], context_changes=[context])
                self.assertEqual(value["decision"], "expand")
                self.assertIn("context:" + context, value["transitive_impact"]["native-checkpoint"])
                self.assert_development_only(value)
        with self.assertRaisesRegex(ValueError, "unknown context category"):
            plan.make_plan([], context_changes=["invented-context"])

    def test_empty_input_reports_no_change_without_qualification(self):
        value = plan.make_plan([])
        self.assertEqual(value["decision"], "no-change")
        self.assertEqual(value["probes"], [])
        self.assertEqual(value["components"], [])
        self.assert_development_only(value)

    def test_explicit_probes_cannot_erase_expansion_or_required_model_gates(self):
        value = plan.make_plan(["crates/memra-server/src/worker.rs"], overrides=["g12", "q35spec"])
        self.assertEqual(value["probes"], ["g12"])
        self.assertEqual(value["spec_probes"], ["q35spec"])
        self.assertEqual(value["decision"], "expand")
        self.assertEqual(set(value["required_unregistered"]), MIMO_GATES)
        with self.assertRaisesRegex(ValueError, "unknown requested probes"):
            plan.make_plan([], overrides=["nonexistent-probe"])

    def test_graph_cycle_and_missing_dependency_refuse(self):
        with self.assertRaisesRegex(ValueError, "dependency cycle"):
            plan.graph_order({"a": {"needs": ["b"]}, "b": {"needs": ["a"]}})
        with self.assertRaisesRegex(ValueError, "unknown dependency"):
            plan.graph_order({"a": {"needs": ["missing"]}})

    def test_dependency_order_is_transitive_even_when_nodes_are_unsorted(self):
        self.assertEqual(plan.graph_order({"third": {"needs": ["second"]},
                                          "first": {}, "second": {"needs": ["first"]}}),
                         ["first", "second", "third"])

    def test_path_escape_or_shell_field_injection_refuses(self):
        for path in ("/outside", "../outside", "tools/../../outside", "line\nbreak", "field\tbreak"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "noncanonical changed path"):
                plan.make_plan([path])


class CheckoutAndWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="memra-plan-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.fg = self.repo / "tools/fast-gate"
        self.fg.mkdir(parents=True)
        for name in ("fast-gate.sh", "plan.py", "map.tsv", "models.tsv", "dependencies.json"):
            shutil.copy2(HERE / name, self.fg / name)
        registry = self.fg / "models.tsv"
        lines = []
        for line in registry.read_text().splitlines():
            fields = line.split("\t")
            if len(fields) == 6 and not line.startswith("#") and fields[1] != "cmd":
                fields[2] = str(self.root / ("absent-" + fields[0] + ".gguf"))
                line = "\t".join(fields)
            lines.append(line)
        registry.write_text("\n".join(lines) + "\n")
        (self.repo / "crates").mkdir()
        (self.repo / "crates/tracked.rs").write_text("// tracked fixture\n")
        (self.repo / ".gitignore").write_text("crates/ignored.inc\ntools/__pycache__/\n")
        self.fake_bin = self.root / "fake-bin"
        self.fake_bin.mkdir()
        # Every executable that could compile, use a GPU lock, or run a model is
        # replaced. The wrapper and planner remain unchanged copies under test.
        self.script(self.fake_bin / "cargo", 'printf "cargo stub\\n" >> "$CALLS"\nexit 0\n')
        self.script(self.fake_bin / "flock", 'printf "flock stub\\n" >> "$CALLS"\nshift 3\nexec "$@"\n')
        self.script(self.repo / "target/release/kernel-check", 'printf "kernel stub\\n" >> "$CALLS"\nexit 0\n')
        self.script(self.repo / "tools/local-ci.sh", 'printf "forbidden full battery stub\\n" >> "$CALLS"\nexit 97\n')
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "CPU fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.git("config", "core.hooksPath", "/dev/null")
        self.commit()
        self.calls = self.root / "calls.log"
        self.env = {**os.environ, "PATH": str(self.fake_bin) + os.pathsep + os.environ["PATH"],
                    "CALLS": str(self.calls), "MEMRA_GATE_LOGDIR": str(self.root / "logs"),
                    "MEMRA_GPU_LOCK": str(self.root / "unused-gpu.lock"),
                    "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}

    def script(self, path, body):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nset -eu\n" + body)
        path.chmod(0o755)

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], stderr=subprocess.PIPE)

    def commit(self):
        self.git("add", ".")
        self.git("commit", "-qm", "CPU fixture")

    def wrapper(self, *args):
        result = subprocess.run(["bash", str(self.fg / "fast-gate.sh"), *args],
                                env=self.env, text=True, capture_output=True, timeout=15)
        self.assertFalse((self.root / "unused-gpu.lock").exists())
        if self.calls.exists():
            self.assertNotIn("forbidden full battery", self.calls.read_text())
        return result

    def replace_probe(self, probe, fields):
        path = self.fg / "models.tsv"
        lines = path.read_text().splitlines()
        self.assertTrue(any(line.startswith(probe + "\t") for line in lines))
        path.write_text("\n".join("\t".join([probe, *fields]) if line.startswith(probe + "\t")
                                  else line for line in lines) + "\n")

    def test_changed_paths_include_tracked_and_new_inputs(self):
        (self.repo / "crates/tracked.rs").write_text("// edit\n")
        (self.repo / "crates/new.rs").write_text("// new input\n")
        paths, hidden = plan.changed_paths(self.repo, "HEAD")
        self.assertEqual(paths, ["crates/new.rs", "crates/tracked.rs"])
        self.assertEqual(hidden, [])

    def test_assume_unchanged_and_skip_worktree_expand_even_without_git_diff(self):
        path = "crates/tracked.rs"
        for flag, undo in (("--assume-unchanged", "--no-assume-unchanged"),
                           ("--skip-worktree", "--no-skip-worktree")):
            with self.subTest(flag=flag):
                self.git("update-index", flag, path)
                paths, hidden = plan.changed_paths(self.repo, "HEAD")
                self.assertIn(path, hidden)
                value = plan.make_plan(paths, hidden=hidden)
                self.assertEqual(value["decision"], "expand")
                self.assertTrue(any("hidden/index-masked" in e for e in value["expansion"]))
                self.git("update-index", undo, path)

    def test_ignored_build_inputs_expand_but_python_cache_is_excluded(self):
        (self.repo / "crates/ignored.inc").write_text("generated build input\n")
        bytecode = self.repo / "tools/__pycache__/fixture.pyc"
        bytecode.parent.mkdir()
        bytecode.write_bytes(b"CPU synthetic bytecode")
        paths, hidden = plan.changed_paths(self.repo, "HEAD")
        self.assertEqual(paths, [])
        self.assertEqual(hidden, ["crates/ignored.inc"])
        self.assertEqual(plan.make_plan(paths, hidden=hidden)["decision"], "expand")

    def test_plan_only_returns_explanation_without_build_or_lock(self):
        (self.repo / "crates/tracked.rs").write_text("// edit\n")
        result = self.wrapper("--plan")
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual(value["decision"], "expand")
        self.assertIs(value["qualification"], False)
        self.assertFalse(self.calls.exists())

    def test_declared_cpu_contract_runs_without_compiler_or_gpu(self):
        tools = self.repo / 'tools'
        for name in ('validation_plan.py', 'validation_inputs.json', 'skip-census.py',
                     'resolve-physical-gpu.py', 'test_resolve_physical_gpu.py', 'unittest-floor.sh'):
            shutil.copy2(HERE.parent / name, tools / name)
        package = self.repo / 'crates/memra-server'
        package.mkdir()
        (package / 'Cargo.toml').write_text('[package]\nname="memra-server"\n')
        (self.repo / 'Cargo.toml').write_text('[workspace]\nmembers=["crates/memra-server"]\n')
        self.commit()
        source = tools / 'resolve-physical-gpu.py'
        canonical = source.read_bytes()
        source.write_text(source.read_text() + '\n# changed CPU contract fixture\n')
        incomplete = self.wrapper()
        self.assertNotEqual(incomplete.returncode, 0, incomplete.stdout + incomplete.stderr)
        self.assertIn('support_record_inputs.py', incomplete.stdout + incomplete.stderr)
        self.assertIn('incomplete coverage', incomplete.stdout)
        self.assertFalse(self.calls.exists(), 'missing selector dependency must expand without native work')
        source.write_bytes(canonical)
        shutil.copy2(HERE.parent / 'support_record_inputs.py', tools / 'support_record_inputs.py')
        self.commit()
        source.write_text(source.read_text() + '\n# changed CPU contract fixture\n')
        result = self.wrapper()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('CPU/content contracts PASS', result.stdout)
        self.assertFalse(self.calls.exists(), 'CPU contract must not call cargo, a model, or flock')
        # The same entry point must go red when its actual CPU contract fails.
        source.write_text('raise RuntimeError("planted resolver failure")\n')
        result = self.wrapper()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.calls.exists())

    def test_unknown_diff_ref_refuses_without_build(self):
        for args in (("--plan", "--diff", "missing-ref"), ("--diff", "missing-ref")):
            with self.subTest(args=args):
                result = self.wrapper(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("REFUSED plan", result.stderr)
                self.assertFalse(self.calls.exists())

    def test_invalid_tier_and_unknown_probe_refuse_without_build(self):
        for args in (("--tier", "-1"), ("--tier", "arbitrary"),
                     ("--plan", "--probes", "missing-probe"), ("--probes", "missing-probe")):
            with self.subTest(args=args):
                result = self.wrapper(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.calls.exists())

    def test_unknown_input_requires_expansion_before_build(self):
        (self.repo / "unmodelled.input").write_text("unknown dependency\n")
        result = self.wrapper()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("incomplete coverage", result.stdout)
        self.assertFalse(self.calls.exists())

    def test_clean_checkout_does_not_claim_validation(self):
        result = self.wrapper()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("NO VALIDATION performed", result.stdout)
        self.assertFalse(self.calls.exists())

    def test_missing_model_cannot_report_success(self):
        self.replace_probe("g12", ["argmax", str(self.root / "absent.gguf"), "@prompt.txt", "20", "-"])
        result = self.wrapper("--probes", "g12")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("SKIP (no model", result.stdout)
        self.assertNotIn("g12: PASS", result.stdout)
        self.assertFalse(self.calls.exists(), "missing data must refuse before compilation")

    def test_silent_command_exit_zero_cannot_report_success(self):
        self.script(self.repo / "tools/fake-check.sh", 'exit 0\n')
        self.replace_probe("samp", ["cmd", "tools/fake-check.sh", "-", "-", "-"])
        result = self.wrapper("--probes", "samp")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("empty command evidence", result.stdout)

    def test_missing_prompt_refuses_before_compilation(self):
        self.setup_model_probe("g31spec", "gspec", "stream agreement 8/8")
        (self.repo / "prompt.txt").unlink()
        result = self.wrapper("--probes", "g31spec")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("missing/empty prompt", result.stdout)
        self.assertFalse(self.calls.exists())

    def test_self_gating_skip_exit_zero_cannot_report_success(self):
        self.script(self.repo / "tools/fake-check.sh", 'printf "fake-check: SKIP (artifact missing)\\n"\n')
        self.replace_probe("samp", ["cmd", "tools/fake-check.sh", "-", "-", "-"])
        result = self.wrapper("--probes", "samp")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("samp: SKIP", result.stdout)
        self.assertNotIn("samp: PASS", result.stdout)

    def setup_model_probe(self, probe, kind, log):
        model = self.repo / "model.gguf"
        model.write_text("CPU fixture, not a real model")
        (self.repo / "prompt.txt").write_text("CPU fixture prompt")
        self.replace_probe(probe, [kind, str(model), "@prompt.txt", "8", "-"])
        binary = "gemma-gate" if kind == "gspec" else "run-gen"
        self.script(self.repo / "target/release" / binary, "cat <<'CPU_LOG'\n" + log + "\nCPU_LOG\n")

    def test_zero_stream_agreement_is_vacuous_and_fails(self):
        self.setup_model_probe("g31spec", "gspec", "stream agreement 0/0")
        result = self.wrapper("--probes", "g31spec")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("g31spec: FAIL", result.stdout)

    def test_nonzero_stream_agreement_exercises_positive_classifier(self):
        self.setup_model_probe("g31spec", "gspec", "stream agreement 8/8")
        result = self.wrapper("--probes", "g31spec")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("g31spec: PASS", result.stdout)

    def test_missing_golden_cannot_report_success(self):
        self.setup_model_probe("g12", "argmax", "argmax=1 decode argmax=1 CPU MATCH\ntokens: [1, 2]")
        result = self.wrapper("--probes", "g12")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("NO GOLDEN pinned", result.stdout)
        self.assertFalse(self.calls.exists(), "missing golden must refuse before compilation")

    def test_empty_generated_tokens_cannot_match_an_empty_golden(self):
        self.setup_model_probe("g12", "argmax", "argmax=1 decode argmax=1 CPU MATCH\ntokens: []")
        (self.fg / "goldens").mkdir(exist_ok=True)
        (self.fg / "goldens/g12.tokens").write_text("# CPU empty golden\ntokens: []\n")
        result = self.wrapper("--probes", "g12")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("zero generated token IDs", result.stdout)

    def test_explicit_probes_work_without_git_but_claim_unknown_coverage(self):
        self.setup_model_probe("g31spec", "gspec", "stream agreement 8/8")
        shutil.rmtree(self.repo / ".git")
        result = self.wrapper("--probes", "g31spec")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("change coverage unknown", result.stdout)
        self.assertIn("g31spec: PASS", result.stdout)

    def test_default_refresh_pins_available_goldens_and_names_missing_ones(self):
        self.setup_model_probe("g12", "argmax", "argmax=1 decode argmax=1 CPU MATCH\ntokens: [1, 2]")
        result = self.wrapper("--refresh-goldens", "--force")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("1 written", result.stdout)
        self.assertIn("unavailable", result.stdout)
        self.assertTrue((self.fg / "goldens/g12.tokens").is_file())
        self.assertNotIn("cargo stub", self.calls.read_text() if self.calls.exists() else "")

    def test_refresh_of_only_missing_explicit_model_fails(self):
        self.replace_probe("g12", ["argmax", str(self.root / "absent.gguf"), "@prompt.txt", "20", "-"])
        result = self.wrapper("--refresh-goldens", "--probes", "g12", "--force")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("0 written", result.stdout)

    def test_refresh_without_git_cannot_pin_unidentified_goldens(self):
        self.setup_model_probe("g12", "argmax", "argmax=1 decode argmax=1 CPU MATCH\ntokens: [1, 2]")
        shutil.rmtree(self.repo / ".git")
        result = self.wrapper("--refresh-goldens", "--probes", "g12", "--force")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("REFUSED golden refresh without Git provenance", result.stdout)


if __name__ == "__main__":
    unittest.main()
