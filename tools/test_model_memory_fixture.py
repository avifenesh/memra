"""Cold-cache and failure-propagation controls for the standalone memory fixture."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class MemoryFixtureRunnerTests(unittest.TestCase):
    def exercise(self, failure="", *, omit_fetch=False):
        with tempfile.TemporaryDirectory(prefix="memra-memory-runner-") as directory:
            root = Path(directory)
            repo, cache, binaries = root / "repo", root / "cache", root / "bin"
            (repo / "tools").mkdir(parents=True)
            cache.mkdir()
            binaries.mkdir()
            (repo / "Cargo.toml").write_text("[workspace]\n")
            (repo / "Cargo.lock").write_text("# pinned fixture lock\n")
            source = Path(__file__).with_name("test-model-memory-fixture.sh").read_text()
            if omit_fetch:
                source = source.replace('cargo fetch --manifest-path "$repo/Cargo.toml" --locked', ':')
            runner = repo / "tools/test-model-memory-fixture.sh"
            runner.write_text(source)
            cargo = binaries / "cargo"
            cargo.write_text('''#!/usr/bin/env python3
import os
from pathlib import Path
import sys
args = sys.argv[1:]
cache = Path(os.environ["CARGO_HOME"])
with (cache / "calls").open("a") as log:
    log.write(args[0] + "\\n")
if args[0] == "fetch":
    assert args == ["fetch", "--manifest-path", os.environ["EXPECTED_MANIFEST"], "--locked"]
    if os.environ["INJECT_FAILURE"] == "fetch":
        sys.exit(23)
    (cache / "ready").touch()
elif args[0] == "test":
    if not (cache / "ready").exists():
        sys.exit(17)
    assert "--offline" in args and "--lib" in args
    manifest = Path(args[args.index("--manifest-path") + 1])
    assert (manifest.parent / "Cargo.lock").read_text() == "# pinned fixture lock\\n"
    assert "model_memory_plan.rs" in (manifest.parent / "lib.rs").read_text()
    if os.environ["INJECT_FAILURE"] == "test":
        sys.exit(42)
else:
    sys.exit(99)
''')
            cargo.chmod(0o755)
            env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"],
                       CARGO_HOME=str(cache), EXPECTED_MANIFEST=str(repo / "Cargo.toml"),
                       INJECT_FAILURE=failure)
            # A caller outside the repository must still fetch the intended lockfile.
            result = subprocess.run([shutil.which("sh"), str(runner)], cwd=root,
                                    env=env, capture_output=True, text=True)
            calls = (cache / "calls").read_text().splitlines()
            self.assertFalse(list((repo / "target").glob("model-memory-fixture-src.*")))
            return result.returncode, calls

    def test_cold_cache_does_not_need_another_suite(self):
        self.assertEqual(self.exercise(), (0, ["fetch", "test"]))
        self.assertEqual(self.exercise(omit_fetch=True), (17, ["test"]))

    def test_fetch_failure_stops_before_compilation(self):
        self.assertEqual(self.exercise("fetch"), (23, ["fetch"]))

    def test_test_failure_is_preserved_and_scratch_is_removed(self):
        self.assertEqual(self.exercise("test"), (42, ["fetch", "test"]))


if __name__ == "__main__":
    unittest.main()
