"""CPU failure controls for GPU CI plumbing; these do not produce native evidence."""

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("gpu_ci", Path(__file__).with_name("gpu-ci.py"))
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


def inputs():
    def item(name, data):
        return {"name": name, "url": "https://example.invalid/" + name,
                "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    return {
        "schema": "memra-gpu-ci-inputs-v1",
        "lease_wrapper": item("memra-gpu-run", b"wrapper"),
        "oracles": [item("oracle.gguf", b"weights")],
    }


class Response(io.BytesIO):
    url = "https://example.invalid/artifact"
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()


class GpuCiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_manifest_keeps_exact_names_hashes_and_sizes(self):
        value = inputs()
        self.assertEqual(ci.manifest(json.dumps(value)), value)

    def test_unpinned_ambiguous_or_excessive_inputs_refuse(self):
        controls = []
        for mutation in (
            lambda m: m.update(schema="other"),
            lambda m: m["oracles"].clear(),
            lambda m: m["oracles"].append(copy.deepcopy(m["oracles"][0])),
            lambda m: m["oracles"][0].update(name="../weights.gguf"),
            lambda m: m["oracles"][0].update(url="file:///etc/passwd"),
            lambda m: m["oracles"][0].update(url="https://token:secret@example.invalid/file"),
            lambda m: m["oracles"][0].update(sha256="latest"),
            lambda m: m["oracles"][0].update(size=True),
            lambda m: m["oracles"][0].update(size=ci.MAX_ORACLE_BYTES + 1),
            lambda m: m["lease_wrapper"].update(name="different-wrapper"),
        ):
            value = inputs()
            mutation(value)
            controls.append(value)
        for value in controls:
            with self.subTest(value=value), self.assertRaises(ci.Refused):
                ci.manifest(json.dumps(value))

    def test_download_validates_size_and_hash_before_publication(self):
        item = inputs()["oracles"][0]
        with patch.object(ci.urllib.request, "urlopen", return_value=Response(b"weights")):
            self.assertEqual(ci.download(item, self.root).read_bytes(), b"weights")

    def test_corrupt_truncated_and_oversized_downloads_leave_no_artifact(self):
        item = inputs()["oracles"][0]
        for data in (b"weightX", b"small", b"weights extra"):
            with patch.object(ci.urllib.request, "urlopen", return_value=Response(data)):
                with self.assertRaises(ci.Refused):
                    ci.download(item, self.root)
            self.assertEqual(list(self.root.iterdir()), [])

    def test_manifest_digest_mismatch_refuses(self):
        data = json.dumps(inputs()).encode()
        with patch.object(ci.urllib.request, "urlopen", return_value=Response(data)):
            with self.assertRaises(ci.Refused):
                ci.fetch_manifest("https://example.invalid/manifest", "a" * 64, self.root / "manifest.json")
        self.assertFalse((self.root / "manifest.json").exists())

    def test_reported_commit_comes_from_hash_bound_source_evidence(self):
        source = json.dumps({"commit": "b" * 40}).encode()
        (self.root / "source.json").write_bytes(source)
        record = {"status": "qualified", "source": {
            "path": "source.json", "sha256": hashlib.sha256(source).hexdigest(),
        }}
        (self.root / "record.json").write_text(json.dumps(record))
        self.assertEqual(ci.sealed_commit(self.root), "b" * 40)
        (self.root / "source.json").write_text(json.dumps({"commit": "a" * 40}))
        with self.assertRaises(ci.Refused):
            ci.sealed_commit(self.root)

    def test_unsealed_or_noncanonical_source_cannot_supply_a_commit(self):
        (self.root / "record.json").write_text(json.dumps({
            "status": "pending", "source": {"path": "../source.json", "sha256": "a" * 64},
        }))
        with self.assertRaises(ci.Refused):
            ci.sealed_commit(self.root)

    def archive(self, members):
        path = self.root / "capsule.tar"
        with tarfile.open(path, "w") as archive:
            for name, kind in members:
                member = tarfile.TarInfo(name)
                member.type = kind
                member.linkname = "/etc/passwd" if kind == tarfile.SYMTYPE else ""
                member.size = 1 if kind == tarfile.REGTYPE else 0
                archive.addfile(member, io.BytesIO(b"x"))
        return path

    def test_missing_extra_duplicate_and_link_capsule_members_refuse(self):
        normal = [(name, tarfile.REGTYPE) for name in sorted(ci.CAPSULE_FILES)]
        for members in (
            normal[:-1], normal + [("../escape", tarfile.REGTYPE)], normal + [normal[0]],
            [(normal[0][0], tarfile.SYMTYPE), *normal[1:]],
        ):
            with self.assertRaises(ci.Refused):
                ci.unpack_build(self.archive(members), self.root / "out")
            self.assertFalse((self.root / "out").exists())

    def test_capsule_roundtrip_preserves_binary_executability(self):
        normal = [(name, tarfile.REGTYPE) for name in sorted(ci.CAPSULE_FILES)]
        ci.unpack_build(self.archive(normal), self.root / "out")
        binary = self.root / "out/target/release/kernel-check"
        self.assertEqual(binary.read_bytes(), b"x")
        self.assertEqual(binary.stat().st_mode & 0o777, 0o755)

    def test_candidate_mismatch_and_missing_native_contract_refuse(self):
        with self.assertRaises(ci.Refused):
            ci.candidate(self.root, "main")
        with patch.object(ci.subprocess, "check_output", return_value="a" * 40):
            with self.assertRaises(ci.Refused):
                ci.candidate(self.root, "b" * 40)
            with self.assertRaises(ci.Refused):
                ci.candidate(self.root, "a" * 40)


if __name__ == "__main__":
    unittest.main()
