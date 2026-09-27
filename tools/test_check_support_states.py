#!/usr/bin/env python3
"""Red arms for tools/check-support-states.py (memra#551).

Each arm copies the real inputs into a scratch root, plants one overclaim, and requires the
census to fail for that reason. The clean copy must pass first, so a census that passes
everything, or fails everything, is caught here.
"""
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
CHECK = REPO / "tools/check-support-states.py"
HY3 = "crates/memra-gguf/src/model_packs/hy3/mod.rs"
GLM = "crates/memra-gguf/src/model_packs/glm5_next/mod.rs"
CLI = "crates/memra-cli/src/lib.rs"
RECORDS = "docs/support-records.toml"
HY3_CARD = "docs/models/hy3.md"


class SupportStateCensus(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="memra-support-states-")
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        for rel in ("crates/memra-gguf/src/model_packs", "docs"):
            shutil.copytree(REPO / rel, self.root / rel)
        for rel in (CLI, "README.md", "STATUS.md", "CLAUDE.md"):
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / rel, self.root / rel)
        records = tomllib.loads((REPO / RECORDS).read_text())["record"]
        cited = {p for r in records for ps in r.get("evidence", {}).values() for p in ps}
        cited |= {"research/modelplan-onboarding-hy3-20260830/tiny/gates.txt"}
        for rel in cited:
            if rel.startswith("ci:"):
                continue
            src = (REPO / rel).parent
            shutil.copytree(src, self.root / pathlib.Path(rel).parent, dirs_exist_ok=True)

    def run_check(self):
        return subprocess.run(
            [sys.executable, str(CHECK), "--root", str(self.root)],
            capture_output=True,
            text=True,
        )

    def edit(self, rel, old, new, count=1):
        path = self.root / rel
        text = path.read_text()
        self.assertEqual(text.count(old), count, f"fixture anchor moved in {rel}: {old!r}")
        path.write_text(text.replace(old, new))

    def assert_fails(self, needle, code=1):
        result = self.run_check()
        self.assertEqual(result.returncode, code, result.stdout + result.stderr)
        self.assertIn(needle, result.stderr)

    def test_clean_copy_passes(self):
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("all backed", result.stdout)

    def test_pack_promoted_without_a_record(self):
        self.edit(
            HY3,
            "(memra#551).\n    support: Some(NativeSupport::NativeReference),",
            "(memra#551).\n    support: Some(NativeSupport::NativeQualified),",
        )
        self.assert_fails("hy3_nvfp4 declares support NativeQualified but its records prove NativeReference")

    def test_record_promoted_without_gates(self):
        self.edit(RECORDS, 'pack = "hy3_nvfp4"\nstate = "NativeReference"', 'pack = "hy3_nvfp4"\nstate = "NativeQualified"')
        self.assert_fails("NativeQualified requires CheckpointParity passed")

    def test_tiny_fixture_never_qualifies(self):
        self.edit(
            RECORDS,
            'pack = "qwen3"\nstate = "NativeReference"',
            'pack = "qwen3"\nstate = "NativeQualified"\nmemra_commit = "' + "a" * 40 + '"',
        )
        self.assert_fails("a tiny fixture never qualifies a pack")

    def test_sibling_family_evidence_is_refused(self):
        self.edit(
            RECORDS,
            'Config = ["ci:verify-tiny"]\nTinyParity = ["ci:verify-tiny"]\n',
            'Config = ["ci:verify-tiny"]\nTinyParity = ["research/modelplan-onboarding-hy3-20260830/tiny/gates.txt"]\n',
            count=len(re.findall(r'Config = \["ci:verify-tiny"\]\nTinyParity = \["ci:verify-tiny"\]\n', (REPO / RECORDS).read_text())),
        )
        self.assert_fails("is for family 'hy3', not 'hy3_nvfp4'")

    def test_ci_tiny_must_reach_the_profile(self):
        self.edit(CLI, "            .chain(model_packs::ONBOARDING_PROFILES)\n", "")
        self.assert_fails("ci:verify-tiny does not reach hy3_nvfp4")

    def test_ci_tiny_needs_a_tiny_plan(self):
        self.edit(HY3, "    tiny_plan: Some(tiny_plan),\n};\n\nfn plan_builder", "    tiny_plan: None,\n};\n\nfn plan_builder")
        self.assert_fails("ci:verify-tiny skips hy3_nvfp4")

    def test_none_pack_with_a_positive_record(self):
        self.edit(GLM, "support: Some(NativeSupport::NativeReference),", "support: None,")
        self.assert_fails("glm5_next declares support None but its records prove NativeReference")

    def test_record_for_unknown_pack(self):
        self.edit(RECORDS, 'pack = "llama_dense"', 'pack = "llama_dense_v2"')
        self.assert_fails("pack 'llama_dense_v2' is not a declared pack family")

    def test_doc_claims_more_than_its_record(self):
        self.edit(
            HY3_CARD,
            "and **NativeReference** for the exact all-expert",
            "and **NativeQualified** for the exact all-expert",
        )
        self.assert_fails("says NativeQualified but the named records prove NativeReference")

    def test_doc_state_without_marker(self):
        (self.root / "docs/NEW.md").write_text("Foo is NativeQualified on B200.\n")
        self.assert_fails("docs/NEW.md:1: NativeQualified needs one")

    def test_none_marker_on_a_claim(self):
        (self.root / "docs/NEW.md").write_text("Foo is NativeQualified on B200. <!-- support: none -->\n")
        self.assert_fails("support: none on a line that reads as a claim")

    def test_cue_words_do_not_excuse_a_none_claim(self):
        # revuto on #891: each line holds a word ("state", "no", "plus") a cue list would accept.
        for claim in (
            "Support state: **NativeQualified** on B200.",
            "Hy3 NVFP4 is NativeQualified on four cards with no fallback.",
            "Qwen3.5-9B NVFP4 is NativeQualified, plus MTP.",
        ):
            with self.subTest(claim=claim):
                (self.root / "docs/NEW.md").write_text(claim + " <!-- support: none -->\n")
                self.assert_fails("docs/NEW.md:1: support: none on a line that reads as a claim of NativeQualified")

    def test_none_must_deny_every_state_it_names(self):
        (self.root / "docs/NEW.md").write_text(
            "Pending NativeQualified; Foo is NativeTuned. <!-- support: none; not NativeQualified -->\n"
        )
        self.assert_fails("reads as a claim of NativeTuned")

    def test_denied_none_line_passes(self):
        (self.root / "docs/NEW.md").write_text(
            "Foo still needs the NativeQualified gate set. <!-- support: none; not NativeQualified -->\n"
        )
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_denying_a_proven_state(self):
        (self.root / "docs/NEW.md").write_text(
            "Hy3 is NativeReference. <!-- support: hy3-bf16-reference; not NativeReference -->\n"
        )
        self.assert_fails("denies NativeReference but the named records prove NativeReference")

    def test_unknown_record_in_marker(self):
        (self.root / "docs/NEW.md").write_text("Hy3 is NativeReference. <!-- support: hy3-made-up -->\n")
        self.assert_fails("unknown support record hy3-made-up")

    def test_unparseable_support_is_an_input_error(self):
        self.edit(GLM, "support: Some(NativeSupport::NativeReference),", "support: glm_support(),")
        self.assert_fails("unparseable support expression", code=2)

    def test_empty_records_is_an_input_error(self):
        (self.root / RECORDS).write_text("# nothing\n")
        self.assert_fails("has no [[record]] entries", code=2)


if __name__ == "__main__":
    unittest.main()
