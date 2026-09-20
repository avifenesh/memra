#!/usr/bin/env python3
"""CPU-only input-view controls. No CUDA compiler, GPU or native proof."""
import hashlib
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import release_input_view as view
import release_qualification as q
from test_release_qualification import Fixture

SPEC = importlib.util.spec_from_file_location("capture_view", Path(__file__).with_name("qualify-release.py"))
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


class InputViewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="memra-input-view-")
        self.addCleanup(self.tmp.cleanup)
        # rustc records the canonical cwd; macOS /var is an alias of /private/var.
        self.root = Path(self.tmp.name).resolve()
        self.f = Fixture(self.root)
        self.payload = self.f.repo / q.PUBLICATION / "direct/value.txt"
        self.payload.parent.mkdir(parents=True)
        self.payload.write_text("EXCLUDED-PAYLOAD\n")
        self.f.commit("metadata fixture")
        self.rustc = Path(subprocess.check_output(
            ["rustup", "which", "--toolchain", "1.97.1", "rustc"], text=True).strip())

    def stage(self, name):
        source = capture.clean_source(self.f.repo)
        out = self.root / name
        out.mkdir()
        staged = capture.prepare_build_source(self.f.repo, source, out)
        self.assertEqual(view.verify(staged, source), view.identity(source))
        return source, out, staged

    def compile(self, staged, out):
        return subprocess.run([str(self.rustc), "--edition=2024", "--crate-name", "view_fixture",
            "--crate-type=lib", "--emit=metadata", "--remap-path-prefix", str(out) + "=/build",
            "--remap-path-prefix", str(staged) + "=/source",
            str(staged / "crates/memra-engine/src/lib.rs"), "-o", str(out / "fixture.rmeta")],
            cwd=staged, capture_output=True, text=True)

    def test_metadata_absent_provenance_intact_and_git_identity_has_no_objects(self):
        source, out, staged = self.stage("identity")
        self.assertEqual((out / "provenance" / self.payload.relative_to(self.f.repo)).read_text(), "EXCLUDED-PAYLOAD\n")
        for name in (q.PUBLICATION.rstrip("/"), "research/INDEX.md"):
            self.assertFalse((staged / name).exists())
        sha = subprocess.check_output(["git", "-C", str(staged), "rev-parse", "--short=12", "HEAD"], text=True).strip()
        self.assertEqual(sha, source["commit"][:12])
        result = subprocess.run(["git", "-C", str(staged), "show", "HEAD:research/INDEX.md"], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(list((staged / ".git/objects").iterdir()), [])
        self.assertEqual(capture.clean_source(out / "provenance"), source)

    def test_direct_string_and_byte_includes_cannot_read_excluded_files(self):
        rust = self.f.repo / "crates/memra-engine/src/lib.rs"
        for macro in ("include_str", "include_bytes"):
            for dependency in (str(self.payload.relative_to(self.f.repo)), "research/INDEX.md"):
                with self.subTest(macro=macro, dependency=dependency):
                    value_type = "str" if macro == "include_str" else "[u8]"
                    rust.write_text(f'pub const PAYLOAD: &{value_type} = {macro}!("../../../{dependency}");\n')
                    self.f.commit("direct compiler read")
                    source, out, staged = self.stage(macro + "-" + Path(dependency).name)
                    result = self.compile(staged, out)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("couldn't read", result.stderr)
                    self.assertIn(Path(dependency).name, result.stderr)
                    self.assertEqual(view.verify(staged, source), view.identity(source))

    def test_metadata_only_publication_preserves_compiler_inputs_and_output(self):
        rust = self.f.repo / "crates/memra-engine/src/lib.rs"
        rust.write_text('pub const PAYLOAD: &str = "ordinary source input";\n')
        self.f.commit("ordinary compiled program")
        before, out_a, a = self.stage("before")
        self.assertEqual(self.compile(a, out_a).returncode, 0)
        self.payload.write_text("REPLACED-METADATA\n")
        index = self.f.repo / "research/INDEX.md"
        index.write_text(index.read_text() + "metadata append\n")
        self.f.commit("metadata-only publication")
        after, out_b, b = self.stage("after")
        self.assertEqual(self.compile(b, out_b).returncode, 0)
        self.assertEqual(before["inputs_sha256"], after["inputs_sha256"])
        self.assertEqual(hashlib.sha256((out_a / "fixture.rmeta").read_bytes()).digest(),
                         hashlib.sha256((out_b / "fixture.rmeta").read_bytes()).digest())
        self.assertTrue(q.verify_source(before, self.f.repo, "HEAD")["publication_equivalent"])
        # Revision metadata remains separately pinned to each real source commit.
        self.assertNotEqual(view.identity(before), view.identity(after))

    def test_unrecorded_files_modes_metadata_and_git_object_routes_refuse(self):
        source, out, staged = self.stage("tamper")
        file = staged / "crates/memra-engine/src/lib.rs"
        original = file.read_bytes()
        file.write_text("changed compiler bytes\n")
        with self.assertRaisesRegex(q.GateError, "bytes/mode changed"):
            view.verify(staged, source)
        file.write_bytes(original)
        file.chmod(0o755)
        with self.assertRaises(q.GateError):
            view.verify(staged, source)
        file.chmod(0o644)
        for name in ("research/INDEX.md", q.PUBLICATION + "payload", ".git/objects/info/alternates"):
            path = staged / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(out / "provenance"))
            with self.assertRaises(q.GateError):
                view.verify(staged, source)
            path.unlink()
            while path.parent != staged and not any(path.parent.iterdir()):
                parent = path.parent
                if str(parent.relative_to(staged)) in ("research", ".git/objects"):
                    break
                parent.rmdir()
                path = parent
        self.assertEqual(view.verify(staged, source), view.identity(source))

    def test_ci_setup_refuses_nonhosted_machines(self):
        setup = Path(__file__).with_name("install-release-sandbox-ci.sh")
        for actions, runner in (("false", "github-hosted"), ("true", "self-hosted")):
            with self.subTest(actions=actions, runner=runner):
                result = subprocess.run(["bash", str(setup)], capture_output=True, text=True,
                    env={**os.environ, "GITHUB_ACTIONS": actions, "RUNNER_ENVIRONMENT": runner})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("only for disposable GitHub-hosted runners", result.stderr)

    @unittest.skipUnless(sys.platform == "linux", "Linux compiler filesystem isolation")
    def test_linux_namespace_refuses_build_script_provenance_reads(self):
        bwrap = shutil.which("bwrap")
        self.assertIsNotNone(bwrap, "Linux controls require installed bubblewrap; do not skip them")
        source, out, staged = self.stage("sandbox")
        (out / "cargo-home").mkdir()
        toolkit = out / "fixture-cuda"
        (toolkit / "bin").mkdir(parents=True)
        hidden_store = toolkit / "provenance-objects"
        hidden_store.mkdir()
        (hidden_store / "payload").write_text("must remain outside the compiler boundary")
        env = capture.build_environment(out, toolkit / "bin/nvcc", self.rustc)
        prefix, _ = view.sandbox(staged, out, self.rustc.parent.parent, toolkit, env, Path(bwrap),
                                 [self.f.repo, out / "provenance", hidden_store])
        probe = r'''set -eu
test ! -e /source/research/INDEX.md
test ! -e /source/research/release-qualification
test ! -e "$1/research/INDEX.md"
test ! -e "$2/research/INDEX.md"
test ! -e /source/.git/objects/info/alternates
test ! -e /cuda/provenance-objects/payload
test ! -e /dev/nvidia0
test "$(git rev-parse --short=12 HEAD)" = "$3"
if git show HEAD:research/INDEX.md 2>/dev/null; then exit 1; fi
if touch /source/research/INDEX.md 2>/dev/null; then exit 1; fi
printf 'filesystem-boundary-pass\n'
'''
        result = subprocess.run(prefix + ["/bin/sh", "-c", probe, "probe", str(self.f.repo),
                                          str(out / "provenance"), source["commit"][:12]], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("filesystem-boundary-pass", result.stdout)
        # These must run inside a successfully created sandbox: a startup failure
        # is not evidence that a read, capability or network route was denied.
        probe = r'''
import pathlib, socket, sys
status = dict(line.split(":", 1) for line in pathlib.Path("/proc/self/status").read_text().splitlines() if ":" in line)
for name in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"):
    assert int(status[name].strip(), 16) == 0, (name, status[name])
assert pathlib.Path("/proc/self/ns/net").readlink().as_posix() != sys.argv[1]
interfaces = {line.split(":", 1)[0].strip() for line in pathlib.Path("/proc/net/dev").read_text().splitlines()[2:]}
assert interfaces == {"lo"}, interfaces
with socket.socket() as client:
    client.settimeout(1)
    assert client.connect_ex(("127.0.0.1", int(sys.argv[2]))) != 0, "host loopback escaped network namespace"
print("capability-network-boundary-pass")
'''
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            with socket.create_connection(listener.getsockname(), timeout=1):
                pass  # Positive control: the host listener really is reachable.
            result = subprocess.run(prefix + ["/usr/bin/python3", "-c", probe,
                os.readlink("/proc/self/ns/net"), str(listener.getsockname()[1])],
                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("capability-network-boundary-pass", result.stdout)
        # Execute an actual build-script-shaped shell read, not a scan of macro text.
        for path in ("/source/research/INDEX.md", "/source/" + q.PUBLICATION + "direct/value.txt",
                     str(out / "provenance/research/INDEX.md")):
            result = subprocess.run(prefix + ["/bin/sh", "-c", 'cat "$1"', "build-script", path],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("No such file", result.stderr)
        result = subprocess.run(prefix + ["/toolchain/bin/rustc", "--crate-type=lib", "--emit=metadata",
            "/source/crates/memra-engine/src/lib.rs", "-o", "/target/positive.rmeta"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(view.verify(staged, source), view.identity(source))

    @unittest.skipUnless(sys.platform == "linux", "Linux controlled Cargo producer")
    def test_linux_controlled_producer_metadata_and_build_script_boundary(self):
        # These are six tiny std-only fixture binaries, never the engine or GPU proof.
        bwrap = shutil.which("bwrap")
        self.assertIsNotNone(bwrap, "Linux controls require bubblewrap")
        repo = self.f.repo
        (repo / "Cargo.toml").write_text('[workspace]\nmembers=["crates/memra-engine","crates/memra-server","crates/memra-tokenizer"]\nresolver="2"\n')
        groups = {"engine": q.BINARIES[:4], "server": ("memra-server",), "tokenizer": ("tok-parity",)}
        for crate, binaries in groups.items():
            directory = repo / ("crates/memra-" + crate)
            (directory / "Cargo.toml").write_text(f'[package]\nname="memra-{crate}"\nversion="0.0.0"\nedition="2024"\n')
            (directory / "src/bin").mkdir()
            for binary in binaries:
                (directory / "src/bin" / (binary + ".rs")).write_text('fn main() { println!("CPU fixture only"); }\n')
        cargo = self.rustc.with_name("cargo")
        subprocess.run([str(cargo), "generate-lockfile", "--offline"], cwd=repo, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.f.commit("tiny CPU workspace")
        toolkit = self.root / "fixture-toolkit"
        (toolkit / "bin").mkdir(parents=True)
        nvcc = toolkit / "bin/nvcc"
        nvcc.write_text('#!/bin/sh\nprintf "CPU fixture nvcc version only; no CUDA compilation\\n"\n')
        nvcc.chmod(0o755)
        empty_cache = self.root / "empty-cargo"
        empty_cache.mkdir()

        def build(name):
            source = capture.clean_source(repo)
            out = self.root / name
            args = argparse.Namespace(repo=repo, expected_head=source["commit"], out=out,
                                      nvcc=nvcc, bwrap=Path(bwrap), jobs=1)
            with patch.dict(os.environ, {"CARGO_HOME": str(empty_cache), "CUDA_VISIBLE_DEVICES": ""}):
                try:
                    capture.build(args)
                except q.GateError:
                    # unittest cleanup removes fixtures, so preserve the actual
                    # compiler/sandbox failure in the CI log before re-raising.
                    for name in ("fetch.log", "build.log"):
                        log = out / name
                        if log.is_file():
                            print(f"CPU fixture {log}:\n{log.read_text()}", file=sys.stderr)
                    raise
            record = q.json_bytes((out / "build.json").read_bytes())
            q.validate_build(record, source, record["source"], q.Evidence(out))
            return source, record

        before, build_a = build("cargo-before")
        self.payload.write_text("unused metadata replacement\n")
        index = repo / "research/INDEX.md"
        index.write_text(index.read_text() + "unused index append\n")
        self.f.commit("metadata-only publication")
        after, build_b = build("cargo-after")
        self.assertEqual(before["inputs_sha256"], after["inputs_sha256"])
        self.assertEqual(build_a["binaries"], build_b["binaries"])
        self.assertTrue(q.verify_source(before, repo, "HEAD")["publication_equivalent"])
        # Actual Cargo executes this script inside the namespace. The full caller
        # still contains INDEX, but the compiler cannot escape back to that checkout.
        script = repo / "crates/memra-engine/build.rs"
        script.write_text('fn main() { std::fs::read_to_string(' + json.dumps(str(repo / "research/INDEX.md"), ensure_ascii=False)
                          + ').expect("CPU fixture forbidden provenance read"); }\n')
        self.f.commit("build script provenance read")
        with self.assertRaisesRegex(q.GateError, "native build failed"):
            build("cargo-forbidden")
        log = (self.root / "cargo-forbidden/build.log").read_text()
        self.assertIn("CPU fixture forbidden provenance read", log)
        self.assertIn("No such file", log)
        self.assertFalse((self.root / "cargo-forbidden/build.json").exists())


if __name__ == "__main__":
    unittest.main()
