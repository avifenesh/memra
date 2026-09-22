import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from unpack_receipts import unpack


class UnpackTests(unittest.TestCase):
    def bundle(self, root, names=('qwen-case/record.json',), symlink=False):
        lane = root / 'lane'
        receipts = lane / 'receipts'
        receipts.mkdir(parents=True)
        archive = receipts / 'qwen-records.tar.gz'
        data = b'{"tokens": 10}\n'
        with tarfile.open(archive, 'w:gz') as stream:
            for name in names:
                member = tarfile.TarInfo(name)
                member.size = len(data)
                if symlink:
                    member.type = tarfile.SYMTYPE
                    member.linkname = '../../outside'
                    member.size = 0
                    stream.addfile(member)
                else:
                    stream.addfile(member, io.BytesIO(data))
        hashes = receipts / 'qwen-records.sha256'
        hashes.write_text(''.join(hashlib.sha256(data).hexdigest() + '  ' + name + '\n' for name in names))
        manifest = {'archives': [{'file': archive.name, 'bytes': archive.stat().st_size,
                                 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                                 'file_manifest_sha256': hashlib.sha256(hashes.read_bytes()).hexdigest()}],
                    'shared': []}
        (receipts / 'manifest.json').write_text(json.dumps(manifest))
        return lane, archive

    def test_valid_records_extract_and_existing_destination_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lane, _ = self.bundle(root)
            out = root / 'out'
            self.assertEqual(unpack(lane, out), 1)
            self.assertEqual((out / 'qwen-case/record.json').read_bytes(), b'{"tokens": 10}\n')
            with self.assertRaisesRegex(ValueError, 'destination must be new'):
                unpack(lane, out)

    def test_archive_corruption_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lane, archive = self.bundle(root)
            archive.write_bytes(archive.read_bytes() + b'corrupt')
            with self.assertRaisesRegex(ValueError, 'archive hash/size'):
                unpack(lane, root / 'out')
            self.assertFalse((root / 'out').exists())

    def test_traversal_links_and_duplicate_members_are_rejected(self):
        for names, symlink in [(('qwen-case/../../outside',), False),
                               (('qwen-case/link',), True),
                               (('qwen-case/same', 'qwen-case/same'), False)]:
            with self.subTest(names=names, symlink=symlink), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                lane, _ = self.bundle(root, names, symlink)
                with self.assertRaisesRegex(ValueError, 'unsafe or duplicate'):
                    unpack(lane, root / 'out')
                self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
