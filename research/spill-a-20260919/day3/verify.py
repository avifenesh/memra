#!/usr/bin/env python3
"""Bounded CPU/cross-target checks, retaining exact raw output before parsing."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'research/spill-a-20260919/day3'
RAW = OUT / 'raw'
RAW.mkdir(exist_ok=True)
commands = [
    ('fmt', ['cargo', 'fmt', '--all', '--', '--check']),
    ('check-macos', ['cargo', 'check', '-p', 'memra-tier', '-p', 'memra-kv', '--offline', '--all-targets']),
    ('check-linux', ['cargo', 'check', '-p', 'memra-tier', '-p', 'memra-kv', '--offline', '--all-targets', '--target', 'x86_64-unknown-linux-gnu']),
    ('test', ['cargo', 'test', '-p', 'memra-tier', '--offline']),
    ('test-kv', ['cargo', 'test', '-p', 'memra-kv', '--offline']),
    ('clippy', ['cargo', 'clippy', '-p', 'memra-tier', '--offline', '--all-targets', '--', '-D', 'warnings']),
    ('diff', ['git', 'diff', '--check']),
    ('flags', ['bash', 'tools/check-flags.sh']),
    ('runner-syntax', ['bash', '-n', 'research/spill-a-20260919/rig-cells-a.sh']),
    ('runner-dry', ['python3', 'research/spill-a-20260919/day3/test_runner.py']),
    ('fixtures', ['python3', 'crates/memra-tier/tests/contracts/fixture_reference.py', '--check']),
]
results = []
for name, argv in commands:
    path = RAW / (name + '.log')
    with path.open('wb') as log:
        result = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=180)
    data = path.read_bytes()
    (RAW / (name + '.log.gz')).write_bytes(gzip.compress(data, mtime=0))
    path.unlink()
    results.append(dict(name=name, command=argv, exit=result.returncode))
    (OUT/'commands.json').write_text(json.dumps(results, indent=2)+'\n')
    print(f'{name}: exit={result.returncode}')
    if result.returncode:
        print(data.decode(errors='replace'))
        raise SystemExit(result.returncode)
paths = list((ROOT/'crates/memra-tier/src/io').rglob('*.rs'))
paths += list((ROOT/'crates/memra-tier/src/object_store').rglob('*.rs'))
paths += list((ROOT/'crates/memra-tier/src/pool').rglob('*.rs'))
paths += list((ROOT/'crates/memra-tier/tests/storage').rglob('*.rs'))
paths += [ROOT/'crates/memra-engine/src/bin/storage_bench.rs', ROOT/'crates/memra-tier/src/contracts.rs']
source = dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              utc=datetime.now(timezone.utc).isoformat(),
              files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
(OUT/'source-manifest.json').write_text(json.dumps(source, indent=2)+'\n')
print('DAY3_CPU_CHECKS_PASS: Linux type-check only, no Linux execution/GPU/NVMe qualification')
