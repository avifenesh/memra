import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from reproduce_native import unpack


class NativeArchiveTests(unittest.TestCase):
    def bundle(self, root, name="native/status.json", *, duplicate=False, symlink=False, omit=False):
        receipts = root / "receipts"
        receipts.mkdir()
        payload = b'{"status":"completed"}\n'
        record = {"file": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        (receipts / "native-data.members.jsonl").write_text(json.dumps(record) + "\n")
        with tarfile.open(receipts / "native-data.tar.gz", "w:gz") as archive:
            if not omit:
                for _ in range(2 if duplicate else 1):
                    member = tarfile.TarInfo(name)
                    member.size = len(payload)
                    if symlink:
                        member.type = tarfile.SYMTYPE
                        member.linkname = "../../outside"
                    archive.addfile(member, io.BytesIO(payload) if not symlink else None)
        for filename in ("runtime-source.tar.gz", "harness-source.tar.gz"):
            (receipts / filename).write_bytes(b"unused by data unpacking")
        files = {
            p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in receipts.iterdir()
        }
        manifest = {"files": files, "member_count": 1, "expanded_data_bytes": len(payload)}
        (receipts / "manifest.json").write_text(json.dumps(manifest))
        pin = hashlib.sha256((receipts / "manifest.json").read_bytes()).hexdigest()
        (receipts / "manifest.sha256").write_text(pin + "  manifest.json\n")
        return receipts, pin

    def test_verified_regular_member_is_recovered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipts, pin = self.bundle(root)
            destination = root / "out"
            unpack(receipts, pin, destination)
            self.assertEqual((destination / "native/status.json").read_text(), '{"status":"completed"}\n')

    def test_valid_outer_hash_does_not_authorize_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipts, pin = self.bundle(root, "../outside")
            with self.assertRaises(ValueError):
                unpack(receipts, pin, root / "out")
            self.assertFalse((root / "outside").exists())

    def test_links_and_duplicate_members_are_refused(self):
        for options in ({"symlink": True}, {"duplicate": True}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                receipts, pin = self.bundle(root, **options)
                with self.assertRaises(ValueError):
                    unpack(receipts, pin, root / "out")

    def test_missing_members_are_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipts, pin = self.bundle(root, omit=True)
            with self.assertRaises(ValueError):
                unpack(receipts, pin, root / "out")


if __name__ == "__main__":
    unittest.main()
