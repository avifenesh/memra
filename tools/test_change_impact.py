#!/usr/bin/env python3
"""Change-impact classifier: a table of example diffs -> (build target, scope) cells.

Pure logic: the rules are the tracked tools/change-impact.toml, the arch tables are
literal. The two arch-table parse tests read the real build.rs shapes."""
from __future__ import annotations

from pathlib import Path
import subprocess
import re
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import change_impact as ci  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RULES_TEXT = (ROOT / ci.RULES_PATH).read_text()
# Today's table: 12.1 builds the 120a target.
MAIN = ci.ArchTable(accepted=("120a", "100a", "90a", "89"),
                    caps={"12.0": "120a", "12.1": "120a", "10.0": "100a", "9.0": "90a", "8.9": "89"})
# A table with a separate GB10 target in the 120a policy family (the shape of memra#1025).
SPLIT = ci.ArchTable(accepted=("120a", "121a", "100a", "90a", "89"),
                     caps={"12.0": "120a", "12.1": "121a", "10.0": "100a", "9.0": "90a", "8.9": "89"},
                     families={"121a": "120a"})
# The tracked rules carry the sm_121a row; NO_121A_RULES is the same table without it.
SPLIT_RULES = RULES_TEXT
NO_121A_RULES = re.sub(r'\[target\."121a"\]\n(?:[^\[\n].*\n)*\n', '', RULES_TEXT, count=1)
assert '[target."121a"]' in SPLIT_RULES and '[target."121a"]' not in NO_121A_RULES
ALL = ("120a", "100a", "90a", "89")
CODE = ("    let x = launch(a, b);",)
COMMENT = ("    // explains the launch",)
DEVICE = ('    if BUILT_CUDA_ARCH == "120a" { tile = 128 }',)


def change(path, added=CODE, removed=(), old="100644", new="100644"):
    return ci.Change(path, old, new, tuple(added), tuple(removed))


def cells(*specs):
    """cells(("kernel", ALL), ("perf", ("120a",))) -> {(target, scope)}"""
    return {(t, scope) for scope, targets in specs for t in targets}


# (name, changes, expected cells, expected uncovered packs, arch, crate inputs)
TABLE = [
    ("docs only", [change("docs/RELEASE-QUALIFICATION.md"), change("AGENTS.md"), change("README.md")],
     set(), set(), MAIN, None),
    ("integration test only", [change("crates/memra-engine/tests/exact.rs")], set(), set(), MAIN, None),
    ("tool test only", [change("tools/test_plan.py"), change("tools/fast-gate/test_plan.py")],
     set(), set(), MAIN, None),
    ("research receipt", [change("research/lane-20261102/run.log")], set(), set(), MAIN, None),
    ("qualification metadata", [change("research/release-qualification/cells/x/record.json"),
                                change("research/INDEX.md")], set(), set(), MAIN, None),
    ("CPU tooling and CI", [change("tools/fast-gate/plan.py"), change(".github/workflows/ci.yml"),
                            change("docs/support-records.toml")], set(), set(), MAIN, None),
    ("CPU-only crates", [change("crates/memra-cli/src/main.rs"), change("crates/memra-net-guard/src/lib.rs")],
     set(), set(), MAIN, None),
    ("host crates linked into the served binary", [change("crates/memra-tokenizer/src/chat.rs"),
                                                   change("crates/memra-sampling/src/lib.rs"),
                                                   change("crates/memra-reference/src/lib.rs")],
     cells(("numeric", ALL)), set(), MAIN, None),
    ("HTTP-side server bookkeeping", [change("crates/memra-server/src/auth.rs")], set(), set(), MAIN, None),
    ("diagnostic binaries", [change("crates/memra-engine/src/bin/decode_bench.rs"),
                             change("crates/memra-probe/src/main.rs")], set(), set(), MAIN, None),
    ("comment-only kernel and engine edits", [change("crates/memra-engine/cu/qmatvec.cu", COMMENT, COMMENT),
                                              change("crates/memra-engine/src/lib.rs", COMMENT, ())],
     set(), set(), MAIN, None),
    ("shared kernel", [change("crates/memra-engine/cu/qmatvec.cu")],
     cells(("kernel", ALL), ("perf", ALL)), set(), MAIN, None),
    ("FFI shim", [change("crates/memra-engine/src/mmq_ffi.rs")],
     cells(("kernel", ALL), ("perf", ALL)), set(), MAIN, None),
    ("fail-closed stub reaches only the stub families", [change("crates/memra-engine/cu/mmq_fp4_stub.cu")],
     cells(("kernel", ("90a", "89"))), set(), MAIN, None),
    ("block-scale MMA unit reaches 120a and 100a", [change("crates/memra-engine/cu/mmq_fp4.cu")],
     cells(("kernel", ("120a", "100a")), ("perf", ("120a", "100a"))), set(), MAIN, None),
    ("block-scale MMA unit reaches the GB10 target through its family",
     [change("crates/memra-engine/cu/mmq_nvfp4_w4a8.cu")],
     cells(("kernel", ("120a", "121a", "100a")), ("perf", ("120a", "121a", "100a"))), set(), SPLIT, None),
    ("CUTLASS unit is the 120a target only, not its family", [change("crates/memra-engine/cu/cutlass_fp4_sm120.cu")],
     cells(("kernel", ("120a",)), ("perf", ("120a",))), set(), SPLIT, None),
    ("wgmma header is Hopper only", [change("crates/memra-engine/cu/wgmma_common.cuh")],
     cells(("kernel", ("90a",)), ("perf", ("90a",))), set(), MAIN, None),
    ("shared engine code", [change("crates/memra-engine/src/decode.rs")],
     cells(("numeric", ALL), ("perf", ALL)), set(), MAIN, None),
    ("device-keyed default", [change("crates/memra-engine/src/lib.rs", DEVICE)],
     cells(("numeric", ALL), ("defaults", ALL), ("perf", ALL)), set(), MAIN, None),
    ("engine bookkeeping is not perf", [change("crates/memra-engine/src/env_audit.rs")],
     cells(("numeric", ALL)), set(), MAIN, None),
    ("covered model pack", [change("crates/memra-gguf/src/model_packs/qwen35/mod.rs")],
     cells(("numeric", ALL)), set(), MAIN, None),
    ("uncovered model pack has no device cell", [change("crates/memra-gguf/src/model_packs/gemma4_dense/mod.rs")],
     set(), {"gemma4_dense"}, MAIN, None),
    ("a family-named engine file is shared engine code", [change("crates/memra-engine/src/glm5_tp.rs")],
     cells(("numeric", ALL), ("perf", ALL)), set(), MAIN, None),
    ("pack kernel is never pack-local", [change("crates/memra-engine/cu/dsv4_gpu.cu")],
     cells(("kernel", ALL), ("perf", ALL)), set(), MAIN, None),
    ("serving policy", [change("crates/memra-server/src/worker.rs")],
     cells(("serving", ALL), ("perf", ALL)), set(), MAIN, None),
    ("lock file", [change("Cargo.lock")], cells(("build", ALL), ("perf", ALL)), set(), MAIN, None),
    ("engine build script", [change("crates/memra-engine/build.rs", ("    let y = 1;",))],
     cells(("build", ALL), ("perf", ALL)), set(), MAIN, None),
    ("battery coverage and the rules are the gate", [change("tools/release-roster.tsv"),
                                                       change("tools/change-impact.toml")],
     cells(("gate", ALL)), set(), MAIN, None),
    ("battery binary is the gate", [change("crates/memra-engine/src/bin/kernel_check.rs")],
     cells(("gate", ALL)), set(), MAIN, None),
    ("a docs file a crate reads", [change("docs/FLAGS.md")],
     cells(("numeric", ALL), ("perf", ALL)), set(), MAIN, {"docs/FLAGS.md": {"memra-engine"}}),
    ("a research fixture a CPU crate reads", [change("research/x/goldens/a.txt")],
     set(), set(), MAIN, {"research/x/goldens/*.txt": {"memra-cli"}}),
    ("a research fixture a linked crate reads", [change("research/x/goldens/a.txt")],
     cells(("numeric", ALL)), set(), MAIN, {"research/x/goldens/*.txt": {"memra-tokenizer"}}),
    ("unknown path reaches everything", [change("weird/new-input.bin")],
     cells(*((s, ALL) for s in ci.UNKNOWN_SCOPES)), set(), MAIN, None),
    ("research symlink is research", [change("research/alias", old="000000", new="120000")],
     set(), set(), MAIN, None),
    ("crate symlink reaches everything", [change("crates/memra-engine/src/alias.rs", old="000000", new="120000")],
     cells(*((s, ALL) for s in ci.UNKNOWN_SCOPES)), set(), MAIN, None),
    ("serve-smoke helper is the gate", [change("tools/q35-cold-mixed-gate.py")],
     cells(("gate", ALL)), set(), MAIN, None),
    ("CI input census is CPU tooling", [change("tools/validation_plan.py")], set(), set(), MAIN, None),
    ("unregistered workspace crate reaches everything", [change("crates/memra-newthing/src/lib.rs")],
     cells(*((s, ALL) for s in ci.UNKNOWN_SCOPES)), set(), MAIN, None),
]


class ClassifierTable(unittest.TestCase):
    def test_table(self):
        for name, changes, expected, uncovered, arch, inputs in TABLE:
            with self.subTest(name):
                rules = ci.load_rules(SPLIT_RULES if arch is SPLIT else RULES_TEXT, arch)
                impact = ci.classify(changes, rules, arch, crate_inputs=inputs)
                got = {(c["target"], c["scope"]) for c in impact["cells"]}
                self.assertEqual(got, expected, ci.render(impact))
                self.assertEqual(set(impact["uncovered"]), uncovered)

    def test_requirements_and_push_gate(self):
        rules = ci.load_rules(RULES_TEXT, MAIN)
        impact = ci.classify([change("crates/memra-engine/cu/qmatvec.cu"), change("crates/memra-server/src/worker.rs")],
                             rules, MAIN)
        needs = ci.requirements(impact)
        self.assertEqual(needs["120a"]["kinds"], {"battery", "serving", "perf"})
        self.assertTrue(needs["120a"]["push_gate"])
        self.assertFalse(any(needs[t]["push_gate"] for t in ("100a", "90a", "89")))
        stub = ci.requirements(ci.classify([change("crates/memra-engine/cu/mmq_fp8_blk_stub.cu")], rules, MAIN))
        self.assertEqual(set(stub), {"90a", "89"})
        self.assertEqual(stub["90a"]["kinds"], {"battery"})

    def test_gate_change_needs_every_kind_and_card_keyed_default_names_its_device(self):
        rules = ci.load_rules(RULES_TEXT, MAIN)
        gate = ci.requirements(ci.classify([change("tools/serve-smoke.sh")], rules, MAIN))
        self.assertEqual(gate["120a"]["kinds"], {"battery", "serving", "perf"})
        keyed = ci.classify([change("crates/memra-server/src/kv_vmm.rs",
                                    ('    if name.contains("RTX PRO 6000 Blackwell") { vmm = true }',))], rules, MAIN)
        needs = ci.requirements(keyed)
        self.assertEqual(needs["120a"]["devices"], {"rtxpro6000"})
        self.assertEqual(needs["100a"]["devices"], set())
        generic = ci.requirements(ci.classify([change("crates/memra-engine/src/lib.rs", DEVICE)], rules, MAIN))
        self.assertEqual(generic["120a"]["devices"], set())

    def test_comment_lines_that_are_not_comments(self):
        rules = ci.load_rules(RULES_TEXT, MAIN)
        for lines in ((r"    // continued \\",), ("",), ("    ",)):
            with self.subTest(lines=lines):
                impact = ci.classify([change("crates/memra-engine/cu/qmatvec.cu", lines)], rules, MAIN)
                self.assertTrue(impact["cells"])

    def test_impact_targets_trailer(self):
        rules = ci.load_rules(SPLIT_RULES, SPLIT)
        kernel = [change("crates/memra-engine/cu/qmatvec.cu")]
        declared, reasons, _ = ci.parse_trailers(
            [("a" * 40, "fix\n\nImpact-Targets: sm_121a\nImpact-Reason: GB10 arm only\n"),
             ("b" * 40, "more\n\nImpact-Targets: 121a\nImpact-Reason: same arm\n")], SPLIT)
        self.assertEqual(declared, {"121a"})
        self.assertEqual(len(reasons), 2)
        impact = ci.classify(kernel, rules, SPLIT, declared=declared)
        self.assertEqual({c["target"] for c in impact["cells"]}, {"121a"})
        # Build, gate and unknown files are never narrowed.
        for path in ("crates/memra-engine/build.rs", "tools/release-roster.tsv", "weird/file"):
            impact = ci.classify([change(path)], rules, SPLIT, declared=declared)
            self.assertIn("120a", {c["target"] for c in impact["cells"]}, path)
        # One undeclared commit keeps the full target set.
        declared, _, missing = ci.parse_trailers(
            [("a" * 40, "x\n\nImpact-Targets: 121a\nImpact-Reason: r\n"), ("b" * 40, "plain\n")], SPLIT)
        self.assertIsNone(declared)
        self.assertEqual(missing, ["b" * 40])
        # A declared target that excludes the path's own targets leaves no cell.
        impact = ci.classify([change("crates/memra-engine/cu/wgmma_common.cuh")], rules, SPLIT, declared={"121a"})
        self.assertEqual(impact["cells"], [])
        with self.assertRaisesRegex(ci.ImpactError, "without an Impact-Reason"):
            ci.parse_trailers([("a" * 40, "x\n\nImpact-Targets: 121a\n")], SPLIT)
        with self.assertRaisesRegex(ci.ImpactError, "unknown target"):
            ci.parse_trailers([("a" * 40, "x\n\nImpact-Targets: 75\nImpact-Reason: r\n")], SPLIT)

    def test_new_target_needs_a_policy_row(self):
        with self.assertRaisesRegex(ci.ImpactError, "no \\[target\\] row .*121a"):
            ci.load_rules(NO_121A_RULES, SPLIT)

    def test_comment_only_needs_both_sides_unchanged_mode(self):
        rules = ci.load_rules(RULES_TEXT, MAIN)
        new_file = ci.classify([change("crates/memra-engine/src/x.rs", COMMENT, (), old="000000")], rules, MAIN)
        self.assertTrue(new_file["cells"], "a new file is a new module, not a comment edit")
        mixed = ci.classify([change("crates/memra-engine/src/x.rs", COMMENT + CODE)], rules, MAIN)
        self.assertTrue(mixed["cells"])


class LinkClosure(unittest.TestCase):
    """Evidence follows the binaries a test runs: the battery and the perf A/B run memra-engine
    binaries, so code outside memra-engine's link closure needs serve-smoke instead."""

    def setUp(self):
        self.rules = ci.load_rules(RULES_TEXT, MAIN)
        self.linked = ci.link_closure(lambda p: (ROOT / p).read_text() if (ROOT / p).exists() else None)

    def needs(self, changes, **kw):
        impact = ci.classify(changes, self.rules, MAIN, linked=self.linked, **kw)
        return {t: (n["kinds"], n["device_kinds"]) for t, n in ci.requirements(impact).items()}

    def test_closure_of_this_tree(self):
        self.assertIn("memra-engine", self.linked)
        self.assertIn("memra-tokenizer", self.linked)
        for outside in ("memra-server", "memra-lanes", "memra-cli", "memra-probe"):
            self.assertNotIn(outside, self.linked)

    def test_server_change_needs_serve_smoke_only(self):
        needs = self.needs([change("crates/memra-server/src/worker.rs")])
        self.assertEqual(needs, {t: ({"serving"}, {}) for t in ALL})

    def test_engine_change_still_needs_the_battery(self):
        needs = self.needs([change("crates/memra-engine/src/decode.rs")])
        self.assertEqual(needs, {t: ({"battery", "perf"}, {}) for t in ALL})

    def test_tier_is_linked_even_though_its_class_is_serving(self):
        needs = self.needs([change("crates/memra-tier/src/lib.rs")])
        self.assertEqual(needs["120a"][0], {"battery", "serving", "perf"})

    def test_server_default_keyed_on_a_card(self):
        line = ('    if name.contains("RTX 5090") { depth = 4 }',)
        needs = self.needs([change("crates/memra-server/src/worker.rs", line)])
        self.assertEqual(needs["120a"], ({"serving"}, {"rtx5090": {"serving"}}))

    def test_unknown_closure_counts_everything_linked(self):
        impact = ci.classify([change("crates/memra-server/src/worker.rs")], self.rules, MAIN, linked=None)
        self.assertIn("battery", ci.requirements(impact)["120a"]["kinds"])

    def test_a_lock_file_a_server_reads_is_still_a_build_input(self):
        # memra-server hashes Cargo.lock into its build id; the lock still versions every crate
        needs = self.needs([change("Cargo.lock")], crate_inputs={"Cargo.lock": {"memra-server"}})
        self.assertEqual(needs["120a"][0], {"battery", "perf"})

    def test_literal_census_reads_compiled_strings_only(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run = lambda *a: subprocess.run(["git", "-C", tmp, *a], check=True, capture_output=True)
            run("init", "-q")
            files = {"crates/memra-gguf/src/model_packs/w/fixtures/source.lock.json": '{"path": "README.md"}\n',
                     "crates/memra-tier/tests/battery/test_collector.py": "# lock names (AGENTS.md)\nx = 'AGENTS.md'\n",
                     "crates/memra-server/Cargo.toml": "# built by .github/workflows/release.yml\n",
                     "crates/memra-engine/src/doc.rs": "/// see docs/KERNELS.md\n// docs/TESTING.md\nfn f() {}\n",
                     "crates/memra-engine/build.rs": 'let flags = manifest.join("../../docs/FLAGS.md");\n',
                     "crates/memra-engine/tests/t.rs": 'let _ = include_str!("../../../docs/MODELS.md");\n'}
            for path, text in files.items():
                Path(tmp, path).parent.mkdir(parents=True, exist_ok=True)
                Path(tmp, path).write_text(text)
            run("add", "."); run("-c", "user.name=f", "-c", "user.email=f@x", "commit", "-qm", "a")
            wanted = ["README.md", "AGENTS.md", ".github/workflows/release.yml", "docs/KERNELS.md",
                      "docs/TESTING.md", "docs/FLAGS.md", "docs/MODELS.md"]
            self.assertEqual(ci.literal_inputs(tmp, "HEAD", wanted), {"docs/FLAGS.md": {"memra-engine"}})


FLAGS_FIXTURE = """## Fixture doors

| `MEMRA_FX_LIVE` | on | a live row |
| `MEMRA_FX_GONE` | **REMOVED 2026-11-01** | marked in its row |
| `MEMRA_FX_FAM_*` | | a prefix row |

### Removed fixture doors

- `MEMRA_FX_OLD`: deleted with its kernel
  and `MEMRA_FX_CONT` on a continuation line

  `MEMRA_FX_AFTER_BLANK` after a blank line is not in the ledger
- `MEMRA_FX_LIVE` is still defined above, so it stays legal
"""


class FlagsRegistry(unittest.TestCase):
    """docs/FLAGS.md reaches the engine only through the env registry build.rs generates."""

    def test_port_matches_the_generator_on_a_fixture(self):
        # expected values come from running the real build.rs generator on this text
        legal, prefixes, retired = ci.flags_registry(FLAGS_FIXTURE)
        self.assertEqual(legal, ["MEMRA_FX_LIVE"])
        self.assertEqual(prefixes, ["MEMRA_FX_FAM_"])
        self.assertEqual(retired, [("MEMRA_FX_CONT", "Removed fixture doors"),
                                   ("MEMRA_FX_GONE", "row marked REMOVED 2026-11-01"),
                                   ("MEMRA_FX_OLD", "Removed fixture doors")])

    def test_the_pinned_generator_is_this_trees(self):
        self.assertEqual(ci.registry_generator_sha((ROOT / ci.BUILD_RS).read_text()), ci.REGISTRY_GENERATOR_SHA,
                         "build.rs generate_env_registry changed: re-verify flags_registry against it, then re-pin")

    def run_repo(self, tmp, flags_v2, build_rs_v2=None):
        run = lambda *a: subprocess.run(["git", "-C", tmp, *a], check=True, capture_output=True)
        run("init", "-q")
        build_rs = (ROOT / ci.BUILD_RS).read_text()
        for path, text in {ci.BUILD_RS: build_rs, ci.FLAGS_MD: FLAGS_FIXTURE}.items():
            Path(tmp, path).parent.mkdir(parents=True, exist_ok=True)
            Path(tmp, path).write_text(text)
        run("add", "."); run("-c", "user.name=f", "-c", "user.email=f@x", "commit", "-qm", "a")
        Path(tmp, ci.FLAGS_MD).write_text(flags_v2)
        if build_rs_v2 is not None:
            Path(tmp, ci.BUILD_RS).write_text(build_rs_v2)
        run("add", "."); run("-c", "user.name=f", "-c", "user.email=f@x", "commit", "-qm", "b")
        return ci.same_flags_registry(tmp, "HEAD~1", "HEAD")

    def test_prose_edit_is_docs_and_a_new_name_is_not(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(self.run_repo(tmp, FLAGS_FIXTURE.replace("a live row", "the live row, reworded")))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(self.run_repo(tmp, FLAGS_FIXTURE + "| `MEMRA_FX_NEW` | off | new |\n"))
        with tempfile.TemporaryDirectory() as tmp:  # a blank line moves a name out of the ledger
            self.assertFalse(self.run_repo(tmp, FLAGS_FIXTURE.replace("kernel\n  and", "kernel\n\n  and")))
        rules = ci.load_rules(RULES_TEXT, MAIN)
        result = ci.classify_file(change(ci.FLAGS_MD), rules, MAIN, crate_inputs={ci.FLAGS_MD: {"memra-engine"}},
                                  same_registry=True)
        self.assertEqual(result["class"], "docs")

    def test_a_changed_generator_turns_the_shortcut_off(self):
        import tempfile
        build_rs = (ROOT / ci.BUILD_RS).read_text().replace("fn generate_env_registry(out: &Path) {",
                                                           "fn generate_env_registry(out: &Path) {\n    let _ = 1;", 1)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(self.run_repo(tmp, FLAGS_FIXTURE.replace("a live row", "reworded"), build_rs))


class DiffParse(unittest.TestCase):
    def test_header_lookalike_content_is_still_content(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run = lambda *a: subprocess.run(["git", "-C", tmp, *a], check=True, capture_output=True)
            run("init", "-q")
            path = Path(tmp, "crates/memra-engine/src/x.rs")
            path.parent.mkdir(parents=True)
            path.write_text("// a\n-- a/y\nlet z = 1;\n")
            run("add", "."); run("-c", "user.name=f", "-c", "user.email=f@x", "commit", "-qm", "a")
            # Removing a code line that reads like a file header, plus a comment edit.
            path.write_text("// b\nlet z = 1;\n")
            run("add", "."); run("-c", "user.name=f", "-c", "user.email=f@x", "commit", "-qm", "b")
            (change,) = ci.diff_changes(tmp, "HEAD~1", "HEAD")
            self.assertIn("-- a/y", change.removed)
            rules = ci.load_rules(RULES_TEXT, MAIN)
            self.assertEqual(ci.classify_file(change, rules, MAIN)["class"], "numeric")


class ArchTableParse(unittest.TestCase):
    def show(self, ref, path):
        result = subprocess.run(["git", "-C", str(ROOT), "show", f"{ref}:{path}"], capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else None

    def test_parses_this_tree(self):
        arch = ci.parse_arch_table((ROOT / ci.BUILD_RS).read_text(),
                                   (ROOT / ci.CUDA_ARCH_RS).read_text() if (ROOT / ci.CUDA_ARCH_RS).exists() else None)
        self.assertIn("120a", arch.accepted)
        self.assertEqual(arch.target_for_cap("12.0"), "120a")
        rules = ci.load_rules(RULES_TEXT, arch)
        self.assertEqual(ci.device_target(rules, arch, "rtx5090"), "120a")
        self.assertEqual(ci.device_target(rules, arch, "rtxpro6000"), "120a")

    def test_parses_both_table_shapes(self):
        legacy = '''
fn detect_arch() -> String {
    let arch = match cap.as_deref() {
        Some("12.0") | Some("12.1") => "120a",
        Some("10.0") => "100a",
        _ => "120a",
    };
}
fn main() {
    assert!(
        matches!(cuda_arch.as_str(), "120a" | "100a"),
        "MEMRA_CUDA_ARCH must be 120a (default) or 100a"
    );
}'''
        arch = ci.parse_arch_table(legacy)
        self.assertEqual(arch.accepted, ("120a", "100a"))
        self.assertEqual(arch.caps, {"12.0": "120a", "12.1": "120a", "10.0": "100a"})
        split_build = legacy.replace('"120a" | "100a"),', '"120a" | "121a" | "100a"),').replace("cuda_arch.as_str", "target.as_str")
        table = '''
pub fn target_for_compute_cap(cap: &str) -> Option<&'static str> {
    match cap.trim() { "12.0" => Some("120a"), "12.1" => Some("121a"), "10.0" => Some("100a"), _ => None }
}
pub fn family(target: &str) -> &str { match target { "121a" => "120a", t => t } }'''
        arch = ci.parse_arch_table(split_build, table)
        self.assertEqual(arch.target_for_cap("12.1"), "121a")
        self.assertEqual(arch.family("121a"), "120a")
        with self.assertRaisesRegex(ci.ImpactError, "refuses"):
            ci.parse_arch_table(legacy.replace('=> "100a"', '=> "103a"'))
        with self.assertRaisesRegex(ci.ImpactError, "missing"):
            ci.parse_arch_table(None)


if __name__ == "__main__":
    unittest.main()
