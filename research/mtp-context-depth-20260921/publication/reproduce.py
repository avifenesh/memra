"""Reproduce both reports using hash-verified data and the exact measured audit code."""
import argparse
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from unpack_receipts import unpack

RUNTIME_SHA256 = '7f5a5d5446e3af717085d0258a6993743e892a32badc79af94366523a6144328'
RUNTIME_COMMIT = 'cb2f1783a0818705c5528f5b11cae0b0bc176deb'
PREFIX = 'research/mtp-context-depth-20260921/'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipts', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    a=p.parse_args()
    source=a.receipts/'runtime-source.tar.gz'
    source_bytes=source.read_bytes()
    sha=hashlib.sha256(source_bytes).hexdigest()
    if source.is_symlink() or sha != RUNTIME_SHA256:
        raise ValueError('Unrecognized runtime archive; no archived code was loaded')
    with tempfile.TemporaryDirectory(prefix='mtp-context-reproduction-') as directory:
        root=Path(directory)
        manifest=unpack(a.receipts, root/'receipts', a.manifest_sha256)
        if manifest['runtime_source']['runtime_archive_sha256'] != RUNTIME_SHA256:
            raise ValueError('Runtime declaration differs from the measured archive')
        if manifest['runtime_source']['source_commit'] != RUNTIME_COMMIT:
            raise ValueError('Runtime commit differs from the measured source')
        if json.loads((root/'receipts/source.json').read_text()) != manifest['runtime_source']:
            raise ValueError('Runtime metadata differs from the sealed source receipt')
        shutil.copyfile(root/'receipts/source.json',root/'source.json')
        lane=root/'repo'/PREFIX
        lane.mkdir(parents=True)
        copied=set()
        with tarfile.open(fileobj=io.BytesIO(source_bytes)) as archive:
            for member in archive:
                if member.name.startswith(PREFIX) and member.isfile():
                    name=member.name[len(PREFIX):]
                    if '/' not in name and (name.endswith('.py') or name.endswith('-artifacts.lock.json')):
                        (lane/name).write_bytes(archive.extractfile(member).read());copied.add(name)
        if not {'audit_context.py','audit_reuse.py','controller.py','calibrate.py','run_study.py','qwen-artifacts.lock.json','gemma-artifacts.lock.json'} <= copied:
            raise ValueError('Measured source is missing a required audit input')
        tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(lane),
                              '-p','test_*.py'],capture_output=True,text=True)
        test_output=tests.stdout+tests.stderr
        print(test_output,end='')
        count=re.search(r'\bRan\s+(\d+)\s+tests?\b',test_output)
        if tests.returncode or not count or int(count.group(1))<14:
            raise ValueError('The measured-source regression suite did not pass its 14-test floor')
        subprocess.run([sys.executable,str(root/'receipts/final-audit-tools/finalize.py'),
                        '--root',str(root),'--portable','--check'],check=True)
    print('Both complete reports and the final audit reproduce from the sealed archives.')


if __name__=='__main__':main()
