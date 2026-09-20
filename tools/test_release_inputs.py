#!/usr/bin/env python3
"""CPU-only regressions for SEC-547-1/2 and PERF-547-1; no native execution."""
from __future__ import annotations
import copy
import importlib.util
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import release_qualification as q
from test_release_qualification import Fixture

spec = importlib.util.spec_from_file_location("release_capture_inputs", Path(__file__).with_name("qualify-release.py"))
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def gguf(path, split_count=None, split_no=0):
    # Valid GGUF v3 header, no tensor payload; split count/no use the spec's UINT16.
    entries = [] if split_count is None else [("split.count", split_count), ("split.no", split_no)]
    data = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, 0, len(entries)))
    for key, value in entries:
        raw = key.encode()
        data += struct.pack("<Q", len(raw)) + raw + struct.pack("<IH", 2, value)
    data += b"\0" * (-len(data) % 32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class ReleaseInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="memra-547-inputs-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.f = Fixture(self.root)

    def test_index_flags_cannot_hide_changed_tracked_bytes(self):
        file = self.f.repo / "crates/memra-engine/src/lib.rs"
        original = file.read_bytes()
        capture.clean_source(self.f.repo)
        for flag in ("assume-unchanged", "skip-worktree"):
            with self.subTest(flag=flag):
                self.f.git("update-index", "--" + flag, str(file.relative_to(self.f.repo)))
                file.write_bytes(b"// compiler reads different bytes\n")
                self.assertEqual(self.f.git("status", "--porcelain"), b"")
                with self.assertRaisesRegex(q.GateError, "actual tracked input differs"):
                    capture.clean_source(self.f.repo)
                file.write_bytes(original)
                self.f.git("update-index", "--no-" + flag, str(file.relative_to(self.f.repo)))
        capture.clean_source(self.f.repo)

    def test_hidden_mode_and_symlink_changes_refuse(self):
        file = self.f.repo / "crates/memra-engine/src/lib.rs"
        self.f.git("config", "core.filemode", "false")
        file.chmod(0o755)
        with self.assertRaisesRegex(q.GateError, "executable mode changed"):
            capture.clean_source(self.f.repo)
        file.chmod(0o644)
        link = self.f.repo / "source-link"
        link.symlink_to("Cargo.lock")
        self.f.commit("tracked link")
        self.f.git("update-index", "--assume-unchanged", "source-link")
        link.unlink(); link.symlink_to("tools/release-roster.tsv")
        with self.assertRaisesRegex(q.GateError, "actual tracked input differs"):
            capture.clean_source(self.f.repo)

    def test_ignored_build_inputs_and_forced_cargo_environment_refuse(self):
        (self.f.repo / ".gitignore").write_text(".cargo/config.toml\ncrates/**/hidden.inc\n")
        self.f.commit("ignore rules")
        config = self.f.repo / ".cargo/config.toml"
        config.parent.mkdir()
        config.write_text('[env]\nDOCS_RS = {value="1", force=true}\n')
        self.assertEqual(self.f.git("status", "--porcelain"), b"")
        with self.assertRaisesRegex(q.GateError, "ignored build/gate input"):
            capture.clean_source(self.f.repo)
        config.unlink()
        hidden = self.f.repo / "crates/memra-engine/src/hidden.inc"
        hidden.write_bytes(b"hidden compiler input")
        with self.assertRaisesRegex(q.GateError, "ignored build/gate input"):
            capture.clean_source(self.f.repo)
        hidden.unlink()
        for content in ('[env]\nDOCS_RS={value="1",force=true}\n',
                        '[env]\nMEMRA_CUDA_ARCH={value="89",force=true}\n',
                        '[build]\nrustc-wrapper="untracked-wrapper"\n'):
            config.write_text(content)
            self.f.git("add", "-f", ".cargo/config.toml"); self.f.git("commit", "-qm", "tracked unsafe config")
            with self.assertRaisesRegex(q.GateError, "unsupported effective Cargo configuration"):
                capture.clean_source(self.f.repo)
        config.write_text("[build]\njobs = 6\n")
        self.f.git("add", "-f", ".cargo/config.toml"); self.f.git("commit", "-qm", "safe jobs config")
        capture.clean_source(self.f.repo)

    def test_ancestor_config_and_compiler_wrapper_controls(self):
        ancestor = self.root / ".cargo/config.toml"
        ancestor.parent.mkdir(); ancestor.write_text('[env]\nDOCS_RS={value="1",force=true}\n')
        with self.assertRaisesRegex(q.GateError, "unrecorded ancestor/Cargo configuration"):
            capture.clean_source(self.f.repo)
        ancestor.unlink()
        with patch.dict(os.environ, {"PATH": "/usr/bin", "DOCS_RS": "1", "CARGO_HOME": "/untrusted",
                "RUSTC_WRAPPER": "wrapper", "RUSTC_WORKSPACE_WRAPPER": "wrapper2",
                "CARGO_BUILD_RUSTC_WRAPPER": "wrapper3", "CARGO_BUILD_TARGET": "other-target",
                "RUSTFLAGS": "--cfg docsrs", "CARGO_ENCODED_RUSTFLAGS": "bad",
                "CC": "other-cc", "MEMRA_CUDA_ARCH": "89"}, clear=True):
            env = capture.build_environment(self.root / "owned", Path("/cuda/bin/nvcc"), Path("/rust/bin/rustc"))
        for key in ("DOCS_RS", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "CARGO_BUILD_RUSTC_WRAPPER",
                    "CARGO_BUILD_TARGET", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CC"):
            self.assertNotIn(key, env)
        self.assertEqual(env["CARGO_HOME"], str(self.root / "owned/cargo-home"))
        self.assertEqual(env["MEMRA_CUDA_ARCH"], "120a")
        self.assertEqual(env["RUSTC"], "/rust/bin/rustc")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")

    def test_owned_source_contains_git_bytes_and_omits_ignored_caller_files(self):
        (self.f.repo / ".gitignore").write_text("hidden.bin\n")
        self.f.commit("ignore caller-only input")
        (self.f.repo / "hidden.bin").write_bytes(b"not a Git compiler input")
        expected = capture.clean_source(self.f.repo)
        out = self.root / "owned-build"; out.mkdir()
        staged = capture.prepare_build_source(self.f.repo, expected, out)
        self.assertFalse((staged / "hidden.bin").exists())
        self.assertEqual(capture.clean_source(staged), expected)
        original = self.f.repo / "crates/memra-engine/src/lib.rs"
        original.write_text("// caller changed after staging\n")
        self.assertNotEqual(original.read_bytes(), (staged / original.relative_to(self.f.repo)).read_bytes())
        self.assertEqual(capture.clean_source(staged), expected)

    def test_owned_cargo_home_copies_no_config_credentials_or_extracted_source(self):
        old, out = self.root / "old-cargo", self.root / "build"
        old.mkdir(); out.mkdir()
        (old / "config.toml").write_text("CPU fixture unsafe config")
        (old / "credentials.toml").write_text("CPU fixture, no credential")
        for name in ("registry/cache/archive.crate", "registry/index/entry", "registry/src/modified.rs"):
            p = old / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("CPU fixture")
        with patch.dict(os.environ, {"CARGO_HOME": str(old)}):
            home = capture.prepare_cargo_home(out)
        self.assertFalse((home / "config.toml").exists())
        self.assertFalse((home / "credentials.toml").exists())
        self.assertFalse((home / "registry/src").exists())
        self.assertTrue((home / "registry/cache/archive.crate").is_file())

    def test_legacy_or_uncontrolled_build_cannot_be_relabelled_qualified(self):
        original = copy.deepcopy(self.f.build)
        mutations = (
            lambda build: build.update(schema="memra-native-build-v1"),
            lambda build: build["recipe"].update(cargo_home="ambient"),
            lambda build: build["recipe"].update(build_source="caller-checkout"),
            lambda build: build["recipe"]["compilers"].pop("nvcc"),
        )
        for mutate in mutations:
            self.f.build = copy.deepcopy(original)
            mutate(self.f.build)
            self.f.refresh()
            with self.assertRaises(q.GateError):
                self.f.verify()
        self.f.build = original
        self.f.refresh()
        self.f.verify()

    def test_absolute_relative_and_symlinked_oracle_directories_are_equivalent(self):
        for name in self.f.models:
            gguf(Path(name))
        oracle = Path(self.f.oracle_dir)
        oracle_name = "Qwen3.5-9B-NVFP4-MTP-GGUF.gguf"
        elsewhere = self.root / "retained-oracle" / oracle_name; gguf(elsewhere)
        (oracle / oracle_name).symlink_to(elsewhere)
        directory_alias = self.root / "oracle-directory-link"; directory_alias.symlink_to(oracle, target_is_directory=True)
        expected = capture.model_inventory(self.f.repo, oracle)
        for spelling in (oracle, Path(os.path.relpath(oracle, Path.cwd())), directory_alias):
            with self.subTest(spelling=str(spelling)):
                inventory = capture.model_inventory(self.f.repo, spelling)
                self.assertEqual(inventory, expected)
                self.assertIn(str(oracle.resolve() / oracle_name), inventory)
                self.f.run["models_before"] = inventory
                self.f.run["models_after"] = copy.deepcopy(inventory)
                self.f.run["oracle_directory"] = str(oracle.resolve())
                self.f.run["numeric_environment"]["MEMRA_KC_MODELS_DIR"] = q.digest(str(oracle.resolve()).encode())
                self.f.run["numeric_environment_after"] = dict(self.f.run["numeric_environment"])
                self.f.refresh()
                self.f.verify()

    def test_split_count_type_duplicate_and_extent_refusals(self):
        model = self.root / "header.gguf"
        for count in (None, 0, 1):
            gguf(model, count)
            capture.release_inputs.single_file_gguf(model)
        def metadata(kind, payload, duplicate=False):
            key = b"split.count"
            entry = struct.pack("<Q", len(key)) + key + struct.pack("<I", kind) + payload
            data = struct.pack("<4sIQQ", b"GGUF", 3, 0, 2 if duplicate else 1) + entry
            if duplicate: data += entry
            model.write_bytes(data + b"\0" * (-len(data) % 32))
        for kind, payload in ((6, struct.pack("<f", 1.0)), (5, struct.pack("<i", -1)),
                              (8, struct.pack("<Q", 1) + b"2")):
            metadata(kind, payload)
            with self.assertRaises(q.GateError): capture.release_inputs.single_file_gguf(model)
        metadata(2, struct.pack("<H", 1), duplicate=True)
        with self.assertRaisesRegex(q.GateError, "duplicated split.count"):
            capture.release_inputs.single_file_gguf(model)
        model.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1))
        with self.assertRaisesRegex(q.GateError, "truncated GGUF"):
            capture.release_inputs.single_file_gguf(model)

    def test_valid_split_gguf_is_explicitly_unsupported_even_if_renamed(self):
        for name in self.f.models:
            gguf(Path(name))
        first = self.root / "split/model-00001-of-00002.gguf"
        second = self.root / "split/model-00002-of-00002.gguf"
        gguf(first, 2, 0); gguf(second, 2, 1)
        roster = self.f.repo / "tools/release-roster.tsv"
        roster.write_text(f"own\tsplit\t{first}\n")
        for state in ("present", "changed", "missing"):
            if state == "changed": second.write_bytes(second.read_bytes() + b"changed bytes")
            if state == "missing": second.unlink()
            with self.assertRaisesRegex(q.GateError, "split GGUF qualification is unsupported"):
                capture.model_inventory(self.f.repo, Path(self.f.oracle_dir))
        renamed = first.with_name("renamed.gguf"); first.rename(renamed)
        roster.write_text(f"own\tsplit\t{renamed}\n")
        with self.assertRaisesRegex(q.GateError, "split GGUF qualification is unsupported"):
            capture.model_inventory(self.f.repo, Path(self.f.oracle_dir))


if __name__ == "__main__":
    unittest.main()
