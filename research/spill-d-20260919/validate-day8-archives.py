#!/usr/bin/env python3
"""Replay immutable peer CELL archives with the current collector, never CUDA."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
ARCHIVES = [
    ('A', '06fda471d38a1e34920ff30948dc5f9effd25fae', 'research/spill-a-20260919/day7'),
    ('B', 'd5bf6ce7a676e1a15fba15b82c32016e75158f1d', 'research/spill-b-20260919/rented-5090-20260919/day8-active-8192'),
    ('C', '0b52d6888504996b1661e7ebeaaa196226a2a540', 'research/spill-c-20260919/rented-5090-20260919/day8'),
    ('F', '7644c41f', 'research/spill-f-20260919/h2d-copies'),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = []
    with tempfile.TemporaryDirectory(prefix='memra-d-archive-') as temp:
        for lane, revision, path in ARCHIVES:
            sha = subprocess.check_output(['git', 'rev-parse', revision], cwd=ROOT, text=True).strip()
            archive = subprocess.check_output(['git', 'archive', sha, path], cwd=ROOT)
            with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
                stream.extractall(temp, filter='data')
            metadata.append({'lane': lane, 'commit': sha, 'path': path,
                             'archive_sha256': hashlib.sha256(archive).hexdigest()})
        command = [sys.executable, str(ROOT/'tools/tier-battery.py'), '--validate', temp]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        (args.out/'validate.stdout.json').write_text(result.stdout)
        (args.out/'validate.stderr.log').write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(result.stderr)
        summary = json.loads(result.stdout)
        assert summary['cells'] == 14, summary['cells']
        assert summary['failed_commands'] == 1 and summary['refused_commands'] == 1
        assert summary['qualification'] is False
    receipt = {'archives': metadata, 'collector_sha256': hashlib.sha256((ROOT/'tools/tier-battery.py').read_bytes()).hexdigest(),
               'exit_code': result.returncode, 'qualification': False,
               'cells': summary['cells'], 'failed_commands': summary['failed_commands'],
               'refused_commands': summary['refused_commands']}
    (args.out/'identity.json').write_text(json.dumps(receipt, indent=2)+'\n')
    line = ('CAPTURE ARCHIVES MATCH: 14 cells; 12 executed-not-qualified; '
            '1 failed command; 1 refused command; qualification=false')
    (args.out/'summary.txt').write_text(line+'\n')
    print(line)


if __name__ == '__main__':
    main()
