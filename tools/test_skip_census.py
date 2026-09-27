#!/usr/bin/env python3
"""Negative controls for tools/skip-census.py (memra #484).

Each case builds a throwaway tree (tools/skip-census.py copied in, a crate or two, a manifest)
and runs the census against it, so every arm is decisive without a Rust build. The run arms use
a stub `cargo` on PATH that prints canned libtest output.

The skip shapes planted here are the ones the census used to miss, copied from the tree:
  - `skipping: {HEBREW_ARCHIVE} is not on this machine` (memra-gguf nemotron_rnnt)
  - `skipping: whisper checkpoint is not on this machine` (memra-tokenizer detokenize,
    memra-reference speech::text)
  - `skip: {path} not staged` inside a per-case loop (memra-tokenizer hf_tests)
  - a multi-line `eprintln!(\n "SKIP {fixture} (...)")` (memra-reference glm5_vision_upstream)
  - a `file:line` manifest anchor that no longer points at the test

`--tool <path>` runs the same controls against another copy of the census, which is how the
PR shows the controls fail on the tool from before the fix:
    git show origin/main~N:tools/skip-census.py > /tmp/old.py
    python3 tools/test_skip_census.py --tool /tmp/old.py
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = Path(__file__).resolve().parent
TOOL = HERE / "skip-census.py"
for i, arg in enumerate(list(sys.argv)):
    if arg == "--tool" and i + 1 < len(sys.argv):
        TOOL = Path(sys.argv[i + 1]).resolve()
        del sys.argv[i : i + 2]
        break

HEADER = "# test manifest\n"


class Tree:
    """A throwaway repo root with the census tool, crates and a manifest."""

    def __init__(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="skip-census-test-"))
        (self.dir / "tools").mkdir()
        shutil.copy(TOOL, self.dir / "tools" / "skip-census.py")
        self.bin = self.dir / "stubbin"
        self.bin.mkdir()

    def close(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, rel: str, text: str) -> None:
        path = self.dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")

    def manifest(self, *rows: tuple[str, str, str, str]) -> None:
        body = HEADER + "".join("\t".join(r) + "\n" for r in rows)
        (self.dir / "tools" / "skip-census.tsv").write_text(body, encoding="utf-8")

    def census(self, *args: str, env: dict[str, str] | None = None) -> tuple[int, str]:
        full_env = dict(os.environ)
        full_env["PATH"] = f"{self.bin}{os.pathsep}{full_env.get('PATH', '')}"
        full_env.update(env or {})
        proc = subprocess.run(
            [sys.executable, str(self.dir / "tools" / "skip-census.py"), *args],
            cwd=self.dir,
            env=full_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
        )
        return proc.returncode, proc.stdout

    def stub_cargo(self, output: str) -> None:
        stub = self.bin / "cargo"
        stub.write_text(
            "#!/usr/bin/env python3\nimport sys\nsys.stdout.write(" + repr(output) + ")\n",
            encoding="utf-8",
        )
        stub.chmod(0o755)

    def run(self, output: str, budget: int) -> tuple[int, str]:
        self.stub_cargo(output)
        return self.census(
            "run", "--budget-var", "T_SKIP_BUDGET", "--min-passed", "1",
            "--", "cargo", "test", "-p", "memra-x", "--lib",
            env={"T_SKIP_BUDGET": str(budget)},
        )


def lib(body: str) -> str:
    return "#[cfg(test)]\nmod tests {\n" + textwrap.indent(textwrap.dedent(body).strip("\n"), "    ") + "\n}\n"


PROTOCOL_TEST = lib(
    """
    #[test]
    fn gated() {
        if !std::path::Path::new("/nope").exists() {
            eprintln!("SKIP[/nope]: gated assertions not run");
            return;
        }
    }
    """
)
PROTOCOL_ROW = ("memra-x", "tests::gated", "crates/memra-x/src/lib.rs", "SKIP[/nope]: gated assertions not run")


def result_line(passed: int) -> str:
    return (
        f"test result: ok. {passed} passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; "
        "finished in 0.01s\n"
    )


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.t = Tree()
        self.addCleanup(self.t.close)

    def assertFails(self, rc: int, out: str, *needles: str) -> None:
        self.assertNotEqual(rc, 0, f"census passed but must fail:\n{out}")
        for needle in needles:
            self.assertIn(needle, out, f"diagnosis does not name {needle!r}:\n{out}")


class Verify(Base):
    def test_clean_tree_passes(self) -> None:
        self.t.write("crates/memra-x/src/lib.rs", PROTOCOL_TEST)
        self.t.manifest(PROTOCOL_ROW)
        rc, out = self.t.census("verify")
        self.assertEqual(rc, 0, out)

    def test_lowercase_skipping_is_unstructured(self) -> None:
        # The RNNT / Whisper shape: printed `skipping:`, counted by nothing before #484.
        self.t.write("crates/memra-x/src/lib.rs", PROTOCOL_TEST + lib(
            """
            #[test]
            fn real_archive() {
                if !std::path::Path::new(HEBREW_ARCHIVE).exists() {
                    eprintln!("skipping: {HEBREW_ARCHIVE} is not on this machine");
                    return;
                }
            }
            """
        ).replace("mod tests", "mod rnnt"))
        self.t.manifest(PROTOCOL_ROW)
        rc, out = self.t.census("verify")
        self.assertFails(rc, out, "rnnt::real_archive")

    def test_per_case_lowercase_skip_is_unstructured(self) -> None:
        # The staged-tokenizer shape: a loop that `continue`s past each missing case.
        self.t.write("crates/memra-x/src/lib.rs", PROTOCOL_TEST + lib(
            """
            #[test]
            fn staged_cases() {
                for path in ["/a", "/b", "/c"] {
                    if !std::path::Path::new(path).exists() {
                        eprintln!("skip: {path} not staged");
                        continue;
                    }
                }
            }
            """
        ).replace("mod tests", "mod hf"))
        self.t.manifest(PROTOCOL_ROW)
        rc, out = self.t.census("verify")
        self.assertFails(rc, out, "hf::staged_cases")

    def test_multiline_literal_is_seen(self) -> None:
        # The upstream-fixture shape: the literal sits on the line after `eprintln!(`.
        self.t.write("crates/memra-x/tests/fixture.rs", """
            #[test]
            fn banked_fixture() {
                for fixture in ["a", "b"] {
                    eprintln!(
                        "SKIP {fixture} (regenerable fixture not present)"
                    );
                    continue;
                }
            }
            """)
        self.t.write("crates/memra-x/src/lib.rs", PROTOCOL_TEST)
        self.t.manifest(PROTOCOL_ROW)
        rc, out = self.t.census("verify")
        self.assertFails(rc, out, "banked_fixture")

    def test_undeclared_protocol_skip_fails(self) -> None:
        self.t.write("crates/memra-x/src/lib.rs", PROTOCOL_TEST)
        self.t.write("crates/memra-y/src/lib.rs", lib(
            """
            #[tokio::test]
            async fn new_blind_spot() {
                eprintln!("SKIP[/nope]: brand new blind spot");
            }
            """
        ))
        self.t.manifest(PROTOCOL_ROW)
        rc, out = self.t.census("verify")
        self.assertFails(rc, out, "memra-y tests::new_blind_spot (")

    def test_stale_row_fails(self) -> None:
        self.t.write("crates/memra-x/src/lib.rs", PROTOCOL_TEST)
        self.t.manifest(PROTOCOL_ROW, ("memra-x", "tests::gone", "crates/memra-x/src/lib.rs",
                                       "SKIP[/gone]: removed test not run"))
        rc, out = self.t.census("verify")
        self.assertFails(rc, out, "tests::gone", "no longer exists in the source")

    def test_line_anchor_is_refused(self) -> None:
        # The stale-anchor shape: the row still says :900 after the test moved to line 5.
        self.t.write("crates/memra-x/src/lib.rs", PROTOCOL_TEST)
        crate, test, where, message = PROTOCOL_ROW
        self.t.manifest((crate, test, where + ":900", message))
        rc, out = self.t.census("verify")
        self.assertFails(rc, out, "line anchor")

    def test_wrong_file_fails(self) -> None:
        self.t.write("crates/memra-x/src/lib.rs", PROTOCOL_TEST)
        self.t.write("crates/memra-x/src/other.rs", "\n")
        crate, test, _, message = PROTOCOL_ROW
        self.t.manifest((crate, test, "crates/memra-x/src/other.rs", message))
        rc, out = self.t.census("verify")
        self.assertFails(rc, out, "tests::gated")

    def test_unregistered_helper_print_fails(self) -> None:
        self.t.write("crates/memra-x/src/lib.rs", PROTOCOL_TEST + lib(
            """
            fn needs_the_box() -> bool {
                eprintln!("SKIP[the box]: helper-driven test not run");
                false
            }

            #[test]
            fn uses_helper() {
                if !needs_the_box() {
                    return;
                }
            }
            """
        ).replace("mod tests", "mod helpers"))
        self.t.manifest(PROTOCOL_ROW)
        rc, out = self.t.census("verify")
        self.assertFails(rc, out, "needs_the_box")


class Run(Base):
    def setUp(self) -> None:
        super().setUp()
        self.t.write("crates/memra-x/src/lib.rs", PROTOCOL_TEST + lib(
            """
            #[test]
            fn staged_cases() {
                for path in ["/a", "/b", "/c"] {
                    if !std::path::Path::new(path).exists() {
                        eprintln!("SKIP[{path}]: staged case not run");
                        continue;
                    }
                }
            }
            """
        ).replace("mod tests", "mod hf"))
        self.t.manifest(
            PROTOCOL_ROW,
            ("memra-x", "hf::staged_cases", "crates/memra-x/src/lib.rs", "SKIP[{path}]: staged case not run"),
        )
        rc, out = self.t.census("verify")
        self.assertEqual(rc, 0, out)

    def test_declared_skip_within_budget_passes(self) -> None:
        rc, out = self.t.run(
            "running 2 tests\n"
            "test tests::gated ... SKIP[/nope]: gated assertions not run\nok\n"
            "test tests::other ... ok\n" + result_line(2),
            budget=1,
        )
        self.assertEqual(rc, 0, out)
        self.assertIn("1 skipped (budget 1)", out)

    def test_zero_of_n_cases_counts_every_case(self) -> None:
        # All three staged cases missing is three skips, not one green test.
        output = (
            "running 2 tests\n"
            "test hf::staged_cases ... SKIP[/a]: staged case not run\n"
            "SKIP[/b]: staged case not run\n"
            "SKIP[/c]: staged case not run\nok\n"
            "test tests::other ... ok\n" + result_line(2)
        )
        rc, out = self.t.run(output, budget=2)
        self.assertFails(rc, out, "3 skip(s), budget 2 (T_SKIP_BUDGET)")
        rc, out = self.t.run(output, budget=3)
        self.assertEqual(rc, 0, out)

    def test_unstructured_run_line_fails(self) -> None:
        rc, out = self.t.run(
            "running 2 tests\n"
            "test tests::gated ... skipping: whisper checkpoint is not on this machine\nok\n"
            "test tests::other ... ok\n" + result_line(2),
            budget=99,
        )
        self.assertFails(rc, out, "tests::gated", "not in the protocol")

    def test_undeclared_run_line_fails(self) -> None:
        rc, out = self.t.run(
            "running 1 tests\n"
            "test tests::brand_new ... SKIP[/nope]: something nobody declared\nok\n" + result_line(1),
            budget=99,
        )
        self.assertFails(rc, out, "tests::brand_new")

    def test_template_matches_in_full_not_by_prefix(self) -> None:
        # `SKIP[{path}]: staged case not run` must not accept a different reason with the
        # same prefix.
        rc, out = self.t.run(
            "running 1 tests\n"
            "test hf::staged_cases ... SKIP[/a]: an entirely different reason\nok\n" + result_line(1),
            budget=99,
        )
        self.assertFails(rc, out, "hf::staged_cases")


if __name__ == "__main__":
    unittest.main()
