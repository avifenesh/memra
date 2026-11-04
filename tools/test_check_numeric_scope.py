#!/usr/bin/env python3
"""Red arms for tools/check-numeric-scope.py.

Each fixture arm writes a small Rust source with one entry shape and requires the census to pass
it with the install and refuse it without. The real-tree arm requires the engine sources to pass
as committed and to fail once one real method loses its install, so a census that passes
everything, or that stopped seeing the real entry points, is caught here.
"""
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
CHECK = REPO / "tools/check-numeric-scope.py"
SRC = REPO / "crates/memra-engine/src"

# Every fixture carries one compliant method and one compliant walker, so a fixture that removes
# its own subject still leaves a census that sees entry points (exit 1, not the broken-census 2).
BASE = """
impl HybridModel {
    pub fn decode_step(&self, e: &Engine, token: u32) -> Result<(), E> {
        let _numeric = self.enter_numeric();
        self.inner(e, token)
    }
}
impl PrimeWalker for W<'_> {
    type Output = ();
    fn advance_chunk(&mut self) -> Result<PrimeChunk, PrimeError> {
        let _numeric = self.model.enter_numeric();
        Ok(PrimeChunk::default())
    }
    fn remaining_chunks(&self) -> usize { 0 }
    fn finish(self) -> Result<(), PrimeError> {
        let _numeric = self.model.enter_numeric();
        Ok(())
    }
}
"""


class NumericScopeCensus(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="memra-numeric-scope-")
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)

    def run_check(self, root):
        return subprocess.run([sys.executable, str(CHECK), str(root)], capture_output=True, text=True)

    def verdict(self, body):
        (self.root / "lib.rs").write_text(BASE + textwrap.dedent(body))
        return self.run_check(self.root)

    def assert_pass(self, body):
        r = self.verdict(body)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def assert_refuse(self, body, needle):
        r = self.verdict(body)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn(needle, r.stdout)

    def test_base_fixture_passes(self):
        self.assert_pass("")

    def test_public_method_with_engine_must_enter_first(self):
        method = """
        impl crate::hybrid::HybridModel {
            pub(crate) fn verify_rows(&self, e: &crate::Engine, rows: &[u32]) -> Result<(), E> {
                // a comment before the install is fine
                %s
                self.launch(e, rows)
            }
        }
        """
        self.assert_pass(method % "let _numeric = self.enter_numeric();")
        self.assert_refuse(method % "", "HybridModel::verify_rows")
        # The install must be the first statement, not merely present.
        self.assert_refuse(
            method % "let n = rows.len();\n                let _numeric = self.enter_numeric();",
            "HybridModel::verify_rows",
        )

    def test_private_methods_and_methods_without_engine_are_not_entries(self):
        self.assert_pass(
            """
            impl HybridModel {
                fn inner(&self, e: &Engine, token: u32) -> Result<(), E> { self.go(e, token) }
                pub fn n_layers(&self) -> usize { self.layers.len() }
                pub fn load(e: &Engine, src: &dyn TensorSource) -> Result<Self, E> { todo!() }
            }
            """
        )

    def test_walker_steps_must_enter_the_model(self):
        walker = """
        impl crate::prime_walker::PrimeWalker for Other<'_> {
            type Output = Vec<f32>;
            fn advance_chunk(&mut self) -> Result<PrimeChunk, PrimeError> {
                %s
                self.step()
            }
            fn remaining_chunks(&self) -> usize { self.left }
            fn finish(self) -> Result<Vec<f32>, PrimeError> {
                let _numeric = self.target.enter_numeric();
                self.done()
            }
        }
        """
        self.assert_pass(walker % "let _numeric = self.target.enter_numeric();")
        self.assert_refuse(walker % "", "PrimeWalker::advance_chunk")

    def test_model_threads_must_enter_or_say_host_only(self):
        spawn = """
        impl HybridModel {
            fn stages(&self, e: &Engine) -> Result<(), E> {
                std::thread::scope(|scope| {
                    %s
                    let h = scope.spawn(move || -> Result<usize, String> {
                        %s
                        self.stage0(e)
                    });
                    h.join()
                })
            }
        }
        """
        self.assert_pass(spawn % ("", "let _numeric = self.enter_numeric();"))
        self.assert_pass(spawn % ("", "let _numeric = profile.enter();"))
        self.assert_pass(spawn % ("// numeric-scope: host-only (queues CPU jobs)", ""))
        self.assert_refuse(spawn % ("", ""), "a thread spawned for a model")

    def test_threads_in_free_fns_that_take_the_model_count(self):
        free = """
        fn pipeline_chunk(m: &crate::hybrid::HybridModel, e: &Engine) {
            std::thread::scope(|s| {
                s.spawn(move || {
                    %s
                    m.stage0(e)
                });
            });
        }
        """
        self.assert_pass(free % "let _numeric = m.enter_numeric();")
        self.assert_refuse(free % "", "a thread spawned for a model")

    def test_strings_comments_and_test_modules_do_not_confuse_it(self):
        self.assert_pass(
            r'''
            impl HybridModel {
                pub fn tricky(&self, e: &Engine) -> Result<[u8; 4], E> {
                    let _numeric = self.enter_numeric();
                    let a = "}{ not a brace"; let b = '}'; let c = r#"impl HybridModel {"#;
                    /* } nested /* { */ } */
                    Ok([0; 4])
                }
            }
            #[cfg(test)]
            mod tests {
                impl HybridModel {
                    pub fn fixture(&self, e: &Engine) {}
                }
            }
            '''
        )

    def test_a_tree_without_entry_points_is_a_broken_census(self):
        (self.root / "lib.rs").write_text("fn main() {}\n")
        r = self.run_check(self.root)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)

    def test_real_tree_passes_and_refuses_one_dropped_install(self):
        tree = self.root / "src"
        shutil.copytree(SRC, tree)
        clean = self.run_check(tree)
        self.assertEqual(clean.returncode, 0, clean.stdout + clean.stderr)
        counts = re.search(r"\((\d+) HybridModel methods, (\d+) walker steps, (\d+) model threads\)", clean.stdout)
        self.assertIsNotNone(counts, clean.stdout)
        methods, walkers, threads = map(int, counts.groups())
        # Floors, not exact counts: they catch a census that silently stopped seeing the entries.
        self.assertGreaterEqual(methods, 200)
        self.assertGreaterEqual(walkers, 10)
        self.assertGreaterEqual(threads, 6)
        target = tree / "decode.rs"
        text = target.read_text()
        dropped = re.sub(r"let _numeric = self\.enter_numeric\(\);", "", text, count=1)
        self.assertNotEqual(dropped, text)
        target.write_text(dropped)
        red = self.run_check(tree)
        self.assertEqual(red.returncode, 1, red.stdout + red.stderr)
        self.assertIn("decode.rs", red.stdout)


if __name__ == "__main__":
    unittest.main()
