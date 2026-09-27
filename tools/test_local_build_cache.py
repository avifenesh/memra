"""CPU admission controls using synthetic capsules, never native/GPU evidence."""
import copy
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

import local_build_cache as cache
import release_qualification as q
import release_input_view as view
from test_release_qualification import Fixture, ROOT


class BuildCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="memra-cache-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = Fixture(self.root)
        self.repo, self.build = self.fixture.repo, self.fixture.out
        for name in (*cache.ci.REQUIRED_TOOLS, "check_hardware_gate.py"):
            shutil.copy2(ROOT / "tools" / name, self.repo / "tools" / name)
        (self.repo / ".gitignore").write_text("crates/ignored.inc\n")
        self.fixture.commit("cache prerequisites")
        source = q.source_snapshot(self.repo)
        self.fixture.put("source.json", source)
        record = self.fixture.build
        record.update(source=self.fixture.ref("source.json"),
                      source_before=source["inputs_sha256"], source_after=source["inputs_sha256"],
                      input_view_before=view.identity(source), input_view_after=view.identity(source))
        self.fixture.put("build.json", record)
        for name in cache.ci.BINARIES:
            destination = self.build / "target/release" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.fixture.binary_dir / name, destination)
            destination.chmod(0o755)
        def artifact(name):
            return {"name": name, "url": "https://example.invalid/" + name,
                    "sha256": "a" * 64, "size": 20}
        self.fixture.put("oracle-manifest.json", {
            "schema": "memra-gpu-ci-inputs-v1", "lease_wrapper": artifact("memra-gpu-run"),
            "oracles": [artifact("oracle.gguf")]})
        self.expected = cache.describe(self.repo, self.build)
        self.storage = self.root / "cache"
        self.out = self.root / "restored"

    def store(self):
        return cache.store(self.repo, self.build, self.storage, self.expected)

    def load(self):
        return cache.load(self.repo, self.storage, self.expected, self.out)

    def assert_refused(self, action, message=None):
        with self.assertRaises((q.GateError, cache.ci.Refused, OSError, tarfile.TarError)) as error:
            action()
        if message:
            self.assertIn(message, str(error.exception))
        self.assertFalse(self.out.exists())
        self.assertEqual(list(self.root.glob(".build-reader-*")), [])
        if self.storage.exists():
            self.assertEqual(list(self.storage.glob(".build-writer-*")), [])

    def mutate_archive(self, change):
        path = cache.cache_entry(self.storage, self.expected)
        with tarfile.open(path, "r:") as archive:
            files = {member.name: archive.extractfile(member).read() for member in archive}
        change(files)
        path.chmod(0o644)
        with tarfile.open(path, "w") as archive:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))

    def test_roundtrip_uses_real_verifiers_and_private_executable_files(self):
        self.assertEqual(self.store()["status"], "stored")
        self.assertEqual(self.load()["status"], "restored")
        self.assertEqual(cache.describe(self.repo, self.out), self.expected)
        for name in cache.ci.BINARIES:
            binary = self.out / "target/release" / name
            self.assertEqual(binary.stat().st_mode & 0o777, 0o755)
            self.assertNotEqual(binary.stat().st_ino, (self.build / "target/release" / name).stat().st_ino)

    def test_second_store_does_not_replace_immutable_entry(self):
        self.store()
        path = cache.cache_entry(self.storage, self.expected)
        before = path.stat()
        self.assertEqual(self.store()["status"], "already-present")
        self.assertEqual(before.st_ino, path.stat().st_ino)
        self.assertEqual(path.stat().st_mode & 0o777, 0o444)

    def test_empty_success_log_retains_native_verifier_admission(self):
        (self.build / "fetch.log").write_bytes(b"")
        record = cache.read_json(self.build / "build.json")
        record["fetch_log"] = self.fixture.ref("fetch.log")
        self.fixture.put("build.json", record)
        self.expected = cache.describe(self.repo, self.build)
        self.store()
        self.load()
        self.assertEqual((self.out / "fetch.log").read_bytes(), b"")

    def test_duplicate_metadata_keys_refuse_export(self):
        path = self.build / "build.json"
        value = path.read_text().rstrip()
        path.write_text(value[:-1] + ', "schema": "memra-native-build-v3"}')
        self.assert_refused(lambda: cache.describe(self.repo, self.build), "duplicate JSON key")

    def test_every_required_bundle_member_is_mandatory(self):
        for name in sorted(cache.capsule_files(self.expected["scope"])):
            with self.subTest(member=name):
                path = self.build / name
                content = path.read_bytes()
                path.unlink()
                # Only the oracle attachment is optional on the producer's output.
                # Removing it changes scope and cannot satisfy the old expectation.
                self.assert_refused(lambda: cache.verify_capsule(self.repo, self.build, self.expected))
                path.write_bytes(content)

    def test_unexpected_duplicate_and_link_archive_members_refuse(self):
        self.store()
        archive_path = cache.cache_entry(self.storage, self.expected)
        original = archive_path.read_bytes()
        archive_path.chmod(0o644)
        for name, kind in (("unexpected.txt", tarfile.REGTYPE), ("build.log", tarfile.REGTYPE),
                           ("linked", tarfile.SYMTYPE), ("../escape", tarfile.REGTYPE)):
            with self.subTest(name=name, kind=kind):
                archive_path.write_bytes(original)
                with tarfile.open(archive_path, "a") as archive:
                    member = tarfile.TarInfo(name)
                    member.type = kind
                    member.linkname = "/etc/passwd" if kind == tarfile.SYMTYPE else ""
                    member.size = 0
                    archive.addfile(member)
                self.assert_refused(self.load, "unexpected member")

    def test_actual_source_bytes_override_git_assume_unchanged(self):
        self.store()
        path = "crates/memra-engine/src/lib.rs"
        self.fixture.git("update-index", "--assume-unchanged", path)
        (self.repo / path).write_text("// hidden source mutation\n")
        self.assert_refused(self.load, "actual tracked input differs")

    def test_actual_source_executable_mode_is_an_input(self):
        self.store()
        (self.repo / "crates/memra-engine/src/lib.rs").chmod(0o755)
        self.assert_refused(self.load, "executable mode changed")

    def test_included_research_fixture_is_not_metadata(self):
        self.store()
        (self.repo / "research/runtime-input.json").write_text('{"changed":true}')
        self.assert_refused(self.load, "actual tracked input differs")

    def test_untracked_and_ignored_build_inputs_refuse(self):
        self.store()
        for name in ("crates/new.inc", "crates/ignored.inc", "tools/new-oracle.json"):
            with self.subTest(name=name):
                path = self.repo / name
                path.write_text("hidden compiler or oracle input")
                self.assert_refused(self.load, "untracked or ignored")
                path.unlink()

    def test_ancestor_cargo_configuration_refuses(self):
        self.store()
        config = self.root / ".cargo/config.toml"
        config.parent.mkdir()
        config.write_text('[build]\nrustflags = ["--cfg", "changed"]\n')
        self.assert_refused(self.load, "unrecorded ancestor/Cargo configuration")

    def test_exact_candidate_cannot_use_publication_equivalence(self):
        self.store()
        (self.repo / "research/INDEX.md").write_text("# Research\nnew publication\n")
        self.fixture.commit("publication append")
        self.assert_refused(self.load, "checkout does not match")

    def test_compiler_platform_numeric_and_oracle_context_changes_miss(self):
        self.store()
        mutations = [
            lambda e: e["build"]["recipe"]["compilers"]["nvcc"].update(sha256="b" * 64),
            lambda e: e["build"]["platform"].update(glibc="different"),
            lambda e: e["build"]["compiler_environment"].update(MEMRA_CUDA_ARCH="b" * 64),
            lambda e: e["build"].update(command=["different", "build"]),
        ]
        if self.expected["oracle_manifest"] is not None:
            mutations.append(lambda e: e["oracle_manifest"]["oracles"][0].update(sha256="b" * 64))
        for change in mutations:
            expected = copy.deepcopy(self.expected)
            change(expected)
            with self.subTest(expected_key=cache.expectation_key(expected)):
                self.assertNotEqual(cache.expectation_key(expected), cache.expectation_key(self.expected))
                self.assert_refused(lambda: cache.load(self.repo, self.storage, expected, self.out), "entry missing")

    def test_binary_tamper_is_rejected_after_unpack(self):
        self.store()
        self.mutate_archive(lambda files: files.update({"target/release/run-gen": b"changed binary"}))
        self.assert_refused(self.load, "caller's expectation")

    def test_rewritten_self_consistent_build_record_cannot_replace_expected_output(self):
        self.store()
        def change(files):
            files["target/release/run-gen"] = b"changed binary"
            record = json.loads(files["build.json"])
            record["binaries"]["run-gen"] = {"bytes": 14, "sha256": q.digest(b"changed binary"), "format": "ELF-x86_64"}
            files["build.json"] = q.canonical(record)
        self.mutate_archive(change)
        self.assert_refused(self.load, "caller's expectation")

    def test_missing_payload_and_truncated_archive_refuse(self):
        self.store()
        self.mutate_archive(lambda files: files.pop("fetch.log"))
        self.assert_refused(self.load, "missing required")
        path = cache.cache_entry(self.storage, self.expected)
        path.write_bytes(b"not a tar file")
        self.assert_refused(self.load)

    def test_missing_or_symlinked_cache_entry_refuses(self):
        self.assert_refused(self.load, "entry missing")
        self.storage.mkdir()
        cache.cache_entry(self.storage, self.expected).symlink_to(self.build / "build.log")
        self.assert_refused(self.load, "entry missing")

    def test_changed_bound_log_and_uncontrolled_recipe_cannot_be_described(self):
        (self.build / "build.log").write_text("replaced log")
        self.assert_refused(lambda: cache.describe(self.repo, self.build), "evidence changed")
        record = cache.read_json(self.build / "build.json")
        record["log"] = self.fixture.ref("build.log")
        record["recipe"]["policy"] = "uncontrolled"
        self.fixture.put("build.json", record)
        self.assert_refused(lambda: cache.describe(self.repo, self.build), "uncontrolled native build recipe")

    def test_failed_store_does_not_publish_or_replace_an_entry(self):
        self.store()
        path = cache.cache_entry(self.storage, self.expected)
        original = path.read_bytes()
        (self.build / "target/release/run-spec").write_bytes(b"changed after descriptor")
        self.assert_refused(self.store, "caller's expectation")
        self.assertEqual(path.read_bytes(), original)

    def test_existing_output_is_preserved(self):
        self.store()
        self.out.mkdir()
        marker = self.out / "owned"
        marker.write_text("do not replace")
        with self.assertRaisesRegex(q.GateError, "output already exists"):
            self.load()
        self.assertEqual(marker.read_text(), "do not replace")

    def test_cli_exports_stores_and_restores_with_elapsed_time(self):
        descriptor = self.root / "expected.json"
        commands = (
            ["describe", "--build", str(self.build), "--out", str(descriptor)],
            ["store", "--build", str(self.build), "--expect", str(descriptor)],
            ["load", "--expect", str(descriptor), "--out", str(self.out)],
        )
        rows = []
        for arguments in commands:
            result = subprocess.run([sys.executable, str(ROOT / "tools/local_build_cache.py"), *arguments,
                                     "--repo", str(self.repo), "--cache", str(self.storage)],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            row = json.loads(result.stdout)
            self.assertGreater(row["seconds"], 0)
            self.assertEqual(row["scope"], self.expected["scope"])
            if arguments[0] == "load":
                self.assertEqual(row["oracle_manifest_attached"], self.expected["scope"] == cache.GPU_CAPSULE)
            rows.append({"operation": arguments[0], "seconds": row["seconds"]})
        print("synthetic_capsule_timings=" + json.dumps({"scope": self.expected["scope"], "rows": rows}), flush=True)


class BuildOnlyCacheTests(BuildCacheTests):
    """Run every admission control again before any oracle attachment exists."""
    def setUp(self):
        super().setUp()
        (self.build / "oracle-manifest.json").unlink()
        self.expected = cache.describe(self.repo, self.build)

    def test_absent_oracle_is_explicit_and_not_a_gpu_capsule(self):
        self.assertEqual(self.expected["scope"], cache.BUILD_ONLY)
        self.assertIsNone(self.expected["oracle_manifest"])
        self.assertEqual(set(self.expected["payloads"]), cache.BUILD_FILES)
        self.store()
        result = self.load()
        self.assertFalse(result["oracle_manifest_attached"])
        self.assertFalse((self.out / "oracle-manifest.json").exists())
        with self.assertRaises(FileNotFoundError):
            cache.ci.verify_build(self.repo, self.out, self.expected["source"]["commit"])

    def test_oracle_attachment_cannot_be_smuggled_into_build_only_archive(self):
        self.store()
        self.mutate_archive(lambda files: files.update({"oracle-manifest.json": b"{}"}))
        self.assert_refused(self.load, "unexpected member")

    def test_scope_and_oracle_claim_must_agree(self):
        for mutation in (lambda e: e.update(scope=cache.GPU_CAPSULE),
                         lambda e: e.update(oracle_manifest={}),
                         lambda e: e.update(schema="memra-local-build-expectation-v1")):
            value = copy.deepcopy(self.expected)
            mutation(value)
            self.assert_refused(lambda: cache.expectation_key(value))

    def test_missing_native_provenance_cannot_be_described(self):
        original = cache.read_json(self.build / "build.json")
        for field in ("binaries", "source_before", "input_view_after", "compiler_environment", "recipe"):
            value = copy.deepcopy(original)
            value.pop(field)
            self.fixture.put("build.json", value)
            with self.subTest(field=field), self.assertRaises((q.GateError, KeyError)):
                cache.describe(self.repo, self.build)


if __name__ == "__main__":
    unittest.main()
