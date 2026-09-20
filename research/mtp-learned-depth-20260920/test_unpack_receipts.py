import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from unpack_receipts import unpack


class ArchiveTests(unittest.TestCase):
    def fixture(self, root, member='fixture/output.txt', link=False):
        path = root / 'fixture.tar.gz'
        payload = b'captured whitespace \r\n\n'
        with tarfile.open(path, 'w:gz') as archive:
            info = tarfile.TarInfo(member)
            if link:
                info.type = tarfile.SYMTYPE
                info.linkname = '../../outside'
                archive.addfile(info)
            else:
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        data = path.read_bytes()
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps({'archives': [{'file': path.name, 'bytes': len(data),
                                                     'sha256': hashlib.sha256(data).hexdigest()}]}))
        return manifest, path, payload

    def test_preserves_capture_bytes_and_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _, payload = self.fixture(root)
            dest = root / 'unpacked'
            self.assertEqual(unpack(manifest, root, dest), 1)
            self.assertEqual((dest / 'fixture/output.txt').read_bytes(), payload)
            with self.assertRaisesRegex(ValueError, 'already exist'):
                unpack(manifest, root, dest)

    def test_corruption_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, archive, _ = self.fixture(root)
            archive.write_bytes(archive.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                unpack(manifest, root, root / 'unpacked')
            self.assertFalse((root / 'unpacked').exists())

    def test_extracts_the_validated_snapshot_if_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, archive, payload = self.fixture(root)
            destination = root / 'unpacked'
            original_mkdir = Path.mkdir

            def change_source(path, *args, **kwargs):
                archive.write_bytes(b'changed after validation')
                return original_mkdir(path, *args, **kwargs)

            with patch.object(Path, 'mkdir', change_source):
                unpack(manifest, root, destination)
            self.assertEqual((destination / 'fixture/output.txt').read_bytes(), payload)

    def test_traversal_and_links_fail_before_writing(self):
        for member, link in [('fixture/../../outside', False), ('/outside', False),
                             ('fixture/link', True)]:
            with self.subTest(member=member), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, _, _ = self.fixture(root, member, link)
                with self.assertRaisesRegex(ValueError, 'unsafe'):
                    unpack(manifest, root, root / 'unpacked')
                self.assertFalse((root / 'unpacked').exists())


if __name__ == '__main__':
    unittest.main()
