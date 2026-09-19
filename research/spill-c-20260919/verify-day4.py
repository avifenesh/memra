#!/usr/bin/env python3
"""Exact-source CPU receipt; stdout+stderr saved losslessly before summarizing.
No engine build, GPU command, network call or env/credential capture.
"""
from pathlib import Path
import datetime
import gzip
import hashlib
import json
import subprocess

root = Path(__file__).resolve().parents[2]
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
now = datetime.datetime.now(datetime.timezone.utc)
out = root / 'research/spill-c-20260919/day4-checks' / (now.strftime('%Y%m%dT%H%M%S%fZ')+'-'+head[:8])
out.mkdir(parents=True)
commands = [
    ['cargo', 'fmt', '--all', '--', '--check'],
    ['cargo', 'check', '-p', 'memra-tier', '--offline', '--all-targets'],
    ['cargo', 'check', '-p', 'memra-tier', '--offline', '--all-targets', '--target', 'x86_64-unknown-linux-gnu'],
    ['cargo', 'test', '-p', 'memra-tier', '--offline', '--no-fail-fast'],
    ['cargo', 'test', '-p', 'memra-tier', '--offline', '--test', 'bank', 'day4', '--', '--nocapture'],
    ['cargo', 'clippy', '-p', 'memra-tier', '--offline', '--all-targets', '--', '-D', 'warnings'],
    ['git', 'diff', '--check'],
    ['bash', 'tools/check-flags.sh'],
    ['git', 'apply', '--check', 'research/spill-c-20260919/HY3-DISPATCH-PATCH.diff'],
    ['python3', 'research/spill-c-20260919/slru-trace.py', '--check'],
    ['bash', '-n', 'research/spill-c-20260919/rig-cells-c.sh'],
    ['python3', 'research/spill-c-20260919/test-rig-cells.py'],
    ['git', 'diff', '--exit-code', '020d2047', '--', 'crates/memra-tier/src/contracts.rs', 'crates/memra-tier/tests/contracts'],
    ['git', 'diff', '--exit-code', '020d2047', '--', 'crates/memra-engine/src'],
]
rows = []
for n, argv in enumerate(commands):
    path = out / f'{n:02d}.log'
    with path.open('wb') as log:
        run = subprocess.run(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    data = path.read_bytes()
    compressed = path.with_suffix('.log.gz')
    compressed.write_bytes(gzip.compress(data, mtime=0))
    path.unlink()
    rows.append(dict(argv=argv, exit=run.returncode, log=compressed.name,
                     stdout_stderr_sha256=hashlib.sha256(data).hexdigest()))
    print(f'exit={run.returncode} '+ ' '.join(argv))
files = list((root/'crates/memra-tier/src/bank').glob('*.rs'))
files += list((root/'crates/memra-tier/tests/bank').glob('*.rs'))
files += list((root/'research/spill-c-20260919').glob('*.py'))
files += list((root/'research/spill-c-20260919').glob('*.sh'))
files += list((root/'research/spill-c-20260919').glob('*.diff'))
files += list((root/'research/spill-c-20260919/fixtures').glob('*.json'))
receipt = dict(utc=now.isoformat(), source_head=head,
               scope='CPU only; Linux is compile-check only; no native engine/server or CUDA qualification',
               commands=rows, files_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
(out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print('receipt: '+str(out.relative_to(root)))
raise SystemExit(any(r['exit'] for r in rows))
