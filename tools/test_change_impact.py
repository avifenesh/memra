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
    ("CUTLASS unit is the 120a family, against the vendored tree", [change("crates/memra-engine/cu/cutlass_fp4_sm120.cu")],
     cells(("kernel", ("120a", "121a")), ("perf", ("120a", "121a"))), set(), SPLIT, None),
    ("vendored CUTLASS headers reach the 120a family only",
     [change("crates/memra-engine/third_party/cutlass/include/cute/tensor.hpp")],
     cells(("kernel", ("120a", "121a")), ("perf", ("120a", "121a"))), set(), SPLIT, None),
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


class GateToolResearchInputs(unittest.TestCase):
    """A research file a gate tool reads when it runs is the gate's input, not a receipt: an edit to
    it changes what the battery or serve-smoke measures. Every research/ path a gate-class tool
    spells on a code line must therefore classify as something other than research, or the push
    gate would let it change with no evidence and a cell record would outlive the change
    (cell_evidence.research_drift)."""

    def test_every_research_path_a_gate_tool_reads_is_not_classed_research(self):
        rules = ci.load_rules(RULES_TEXT, MAIN)
        gate = [r["pattern"] for r in rules["rule"] if r.get("class") == "gate"]
        tools = subprocess.run(["git", "-C", str(ROOT), "ls-files", "tools"], capture_output=True,
                               text=True, check=True).stdout.split()
        missed = []
        for path in tools:
            if not any(re.search(p, path) for p in gate) or path.endswith(".toml"):
                continue
            text = (ROOT / path).read_text(errors="replace")
            for line in text.splitlines():
                code = line.strip()
                if not code or code.startswith(("#", "//", "///", "*")):
                    continue
                for literal in re.findall(r"research/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", line):
                    literal = literal.rstrip(".")
                    if literal.startswith(("research/release-qualification", "research/INDEX")):
                        continue  # the qualification metadata namespace, excluded from inputs
                    probe = literal if "." in literal.rsplit("/", 1)[-1] else literal + "/x.txt"
                    item = ci.classify_file(change(probe), rules, MAIN)
                    if item["class"] == "research":
                        missed.append(f"{path}: {literal}")
        self.assertEqual(missed, [], "add a gate rule in tools/change-impact.toml for these")


DOCS_LITERAL = re.compile(r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.md|\b(?:LICENSE[A-Za-z0-9_.-]*|NOTICE)\b|"
                          r"(?<![A-Za-z0-9_.-])\.(?:gitignore|ignore|gitattributes)\b")


def docs_reads(path, text, docs):
    """The tracked docs-class files a tool's code lines name: a literal equal to one, or a trailing
    path of one (`Path("docs") / "FLAGS.md"` names `docs/FLAGS.md`). Comment lines and Python
    docstrings are left out."""
    found, in_doc = [], False
    for line in text.splitlines():
        code = line.strip()
        if path.endswith(".py") and code.count('"""') % 2:
            in_doc = not in_doc
            continue
        if in_doc or not code or code.startswith(("#", "//", "///", "*")):
            continue
        for literal in DOCS_LITERAL.findall(line):
            literal = literal.lstrip("./") if literal.startswith("./") else literal
            hits = [d for d in docs if d == literal or d.endswith("/" + literal)]
            found.extend(f"{path}: {d}" for d in hits)
    return found


class GateToolDocsInputs(unittest.TestCase):
    """The docs counterpart (cell_evidence.research_drift admits docs drift too): a tracked docs-class
    file that a gate tool reads when a device record is produced would change what the record
    measured. Left out: the pre-push hook (its docs reads are the flags and registry censuses, run on
    the pushed tree at every push), the tag-only qualify-release.py and the Step PRO gate."""

    PUSH_TIME = ("tools/hooks/pre-push", "tools/qualify-release.py", "tools/check_hardware_gate.py")
    # Literals that name a docs file without reading it.
    NOT_READS = {"tools/cell_evidence.py: .gitattributes": "the DRIFT_NEVER entry",
                 "tools/check-argmax-floors.py: docs/TESTING.md": "the census's hint names the doc"}

    def test_the_scan_sees_the_read_forms(self):
        docs = {"docs/FLAGS.md", "docs/decisions/PINNED.md", "README.md", "LICENSE", ".gitattributes"}
        lines = ['cat docs/FLAGS.md', 'p = Path("docs") / "FLAGS.md"', "cat docs/decisions/PINNED.md",
                 "open('README.md')", "cat LICENSE", "git check-attr -a -- x  # .gitattributes",
                 "x = open('.gitattributes')", "# cat docs/FLAGS.md", "echo VERDICT.md"]
        found = docs_reads("tools/x.sh", "\n".join(lines), docs)
        self.assertEqual(sorted(set(found)), sorted({
            "tools/x.sh: docs/FLAGS.md", "tools/x.sh: docs/decisions/PINNED.md", "tools/x.sh: README.md",
            "tools/x.sh: LICENSE", "tools/x.sh: .gitattributes"}))

    def test_no_device_phase_gate_tool_reads_a_tracked_docs_file(self):
        rules = ci.load_rules(RULES_TEXT, MAIN)
        gate = [r["pattern"] for r in rules["rule"] if r.get("class") == "gate"]
        tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True,
                                 text=True, check=True).stdout.split()
        docs = {p for p in tracked if ci.classify_file(change(p), rules, MAIN)["class"] == "docs"
                or re.search(r"^(docs/.*|[^/]+)\.md$|^(LICENSE[^/]*|NOTICE|\.gitignore|\.ignore|\.gitattributes)$", p)}
        missed = []
        for path in sorted(p for p in tracked if p.startswith("tools/")):
            if path in self.PUSH_TIME or path.endswith(".toml") or not any(re.search(g, path) for g in gate):
                continue
            missed += [m for m in docs_reads(path, (ROOT / path).read_text(errors="replace"), docs)
                       if m not in self.NOT_READS]
        self.assertEqual(sorted(set(missed)), [], "add a gate rule in tools/change-impact.toml for these")


class LiteralInputs(unittest.TestCase):
    """A crate line that spells a docs path is a read unless every spelling sits in a comment or a
    string literal holding whitespace (a message that names the file). 2026-11-12: error texts
    citing docs/FLAGS.md and docs/SERVING.md made every edit of those docs a battery-and-perf
    change on every target."""

    PATH = "docs/FLAGS.md"

    def mention(self, text):
        k = text.index(self.PATH)
        start = text.rfind("\n", 0, k) + 1
        end = text.find("\n", k)
        return ci._rust_prose_mention(text, ci.rust_text_spans(text), start, len(text) if end < 0 else end,
                                      self.PATH)

    def test_messages_and_comments_are_mentions(self):
        for text in ('fn f() { eprintln!("see docs/FLAGS.md now"); }\n',
                     "let x = 1; // docs/FLAGS.md\n",
                     "/* see\n docs/FLAGS.md */ let x = 1;\n",
                     'let m = "first line \\\n docs/FLAGS.md: second";\n',
                     'let m = r#"a "quoted" docs/FLAGS.md mention"#;\n'):
            self.assertTrue(self.mention(text), text)

    def test_path_literals_and_code_stay_reads(self):
        for text in ('fn f() { let _ = include_str!("../../docs/FLAGS.md"); }\n',
                     'let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/FLAGS.md");\n',
                     'let m = include_str!(r"../../docs/FLAGS.md");\n',
                     "fn f<'a>(x: &'a str) { let c = '\\''; let q = \"../../docs/FLAGS.md\"; }\n",
                     'f("../../docs/FLAGS.md", "see docs/FLAGS.md");\n',
                     "let p = docs/FLAGS.md;\n",
                     # a command line reads a path inside a string with spaces
                     'Command::new("sh").args(["-c", "grep -c x ../../docs/FLAGS.md"]);\n',
                     # a C raw string, then a glob whose `/*` must not swallow the read below
                     'let a = cr"C:\\"; let g = "crates/*/src";\nlet t = include_str!("../../docs/FLAGS.md");\n',
                     # an unterminated block comment: the lexer lost its place, so it is a read
                     "/* unterminated\nlet t = docs/FLAGS.md;\n"):
            self.assertFalse(self.mention(text), text)

    def test_the_census_drops_only_mentions(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run = lambda *a: subprocess.run(["git", "-C", tmp, *a], check=True, capture_output=True)
            run("init", "-q")
            src = Path(tmp, "crates/memra-server/src/lib.rs")
            src.parent.mkdir(parents=True)
            src.write_text('pub fn f() { eprintln!("ledger: docs/FLAGS.md \'Removed\'"); }\n'
                           'pub fn g() -> &\'static str { include_str!("../../../docs/SERVING.md") }\n')
            run("add", "."); run("-c", "user.name=f", "-c", "user.email=f@x", "commit", "-qm", "a")
            found = ci.literal_inputs(tmp, "HEAD", ["docs/FLAGS.md", "docs/SERVING.md"])
            self.assertEqual(found, {"docs/SERVING.md": {"memra-server"}})


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
