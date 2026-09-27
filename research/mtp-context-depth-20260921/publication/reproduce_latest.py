"""Reproduce the interleaved old/latest engine comparison from verified archives."""
import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from unpack_receipts import unpack

SOURCE='5450580fe2e5eb5c34cd17452f472ed368034f41'
ARCHIVE='943d165b80f268ffacc63e78191c669ed4bda562b3151d45758876321e3f6dc8'
PREFIX='research/mtp-context-depth-20260921/'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipts',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    args=parser.parse_args()
    payload=(args.receipts/'runtime-source.tar.gz').read_bytes()
    if hashlib.sha256(payload).hexdigest()!=ARCHIVE:
        raise ValueError('Unrecognized latest runtime archive')
    with tempfile.TemporaryDirectory(prefix='latest-mtp-reproduction-') as directory:
        root=Path(directory);data=root/'receipts/latest-followup'
        manifest=unpack(args.receipts,data,args.manifest_sha256)
        if (manifest.get('kind')!='interleaved-engine-followup'
                or manifest['runtime_source']['source_commit']!=SOURCE
                or manifest['runtime_source']['runtime_archive_sha256']!=ARCHIVE):
            raise ValueError('Latest comparison source declaration changed')
        experiment=data/'experiment'
        for source in experiment.iterdir():
            target=data/source.name
            if target.exists():raise ValueError('Comparison archive layout collides')
            source.rename(target)
        experiment.rmdir()
        shutil.copy2(args.receipts/'runtime-source.tar.gz',data/'runtime-source.tar.gz')
        lane=root/'latest/repo'/PREFIX
        lane.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
            for member in archive:
                if member.name.startswith(PREFIX) and member.isfile():
                    name=member.name[len(PREFIX):]
                    if '/' not in name and (name.endswith('.py') or name.endswith('-artifacts.lock.json')):
                        (lane/name).write_bytes(archive.extractfile(member).read())
        subprocess.run([sys.executable,str(data/'audit_compare.py'),'--root',str(root),'--check'],check=True)
    print('The old/latest source comparison and cross-version token-identity report reproduce.')


if __name__=='__main__':main()
