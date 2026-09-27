import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unpack_receipts import unpack


def sha(data): return hashlib.sha256(data).hexdigest()


class ArchiveTests(unittest.TestCase):
    def fixture(self, root, evil=None):
        rows=[]
        for family in ('qwen','gemma','common'):
            name=f'{family}-records.tar.gz'
            member=f'experiment/{family}-example/tape.txt' if family!='common' else 'source.json'
            if evil and family=='qwen': member=evil
            payload=b'example\n';raw=io.BytesIO()
            with tarfile.open(fileobj=raw,mode='w:gz') as archive:
                info=tarfile.TarInfo(member);info.size=len(payload)
                archive.addfile(info,io.BytesIO(payload))
            data=raw.getvalue();(root/name).write_bytes(data)
            checks=f'{sha(payload)}  {member}\n'.encode()
            (root/name.replace('.tar.gz','.sha256')).write_bytes(checks)
            rows.append({'file':name,'bytes':len(data),'sha256':sha(data),'files':1,'file_manifest_sha256':sha(checks)})
        manifest={'archives':rows}
        (root/'manifest.json').write_text(json.dumps(manifest))
        return manifest

    def test_valid_inventory(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);self.fixture(r)
            unpack(r,r/'out')
            self.assertEqual((r/'out/source.json').read_bytes(),b'example\n')

    def test_traversal_never_creates_destination(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);self.fixture(r,'../escape')
            with self.assertRaisesRegex(ValueError,'Unsafe'):
                unpack(r,r/'out')
            self.assertFalse((r/'out').exists())

    def test_archive_change_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);self.fixture(r)
            with (r/'gemma-records.tar.gz').open('ab') as f:f.write(b'altered')
            with self.assertRaisesRegex(ValueError,'hash or size'):
                unpack(r,r/'out')
            self.assertFalse((r/'out').exists())

    def test_path_aliases_never_create_destination(self):
        for member in ('experiment/qwen-example/./tape.txt',
                       'experiment/qwen-example//tape.txt'):
            with self.subTest(member=member), tempfile.TemporaryDirectory() as d:
                r=Path(d);self.fixture(r,member)
                with self.assertRaisesRegex(ValueError,'Unsafe'):
                    unpack(r,r/'out')
                self.assertFalse((r/'out').exists())

    def test_dropped_archive_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);m=self.fixture(r);m['archives'].pop()
            (r/'manifest.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'three-archive'):
                unpack(r,r/'out')

    def test_duplicate_member_declaration_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);m=self.fixture(r)
            p=r/'qwen-records.sha256';data=p.read_bytes()*2;p.write_bytes(data)
            m['archives'][0].update(files=2,file_manifest_sha256=sha(data))
            (r/'manifest.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                unpack(r,r/'out')

    def test_reviewed_manifest_pin_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);self.fixture(r)
            with self.assertRaisesRegex(ValueError,'reviewed checksum'):
                unpack(r,r/'out','0'*64)


if __name__=='__main__':unittest.main()
