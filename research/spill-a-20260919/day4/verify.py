#!/usr/bin/env python3
"""CPU-only milestone receipts. Never substitutes for native hardware gates."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
commands = [
    ('fmt', ['cargo', 'fmt', '--all', '--', '--check']),
    ('check-mac', ['cargo', 'check', '-p', 'memra-tier', '-p', 'memra-kv', '--offline', '--all-targets']),
    ('check-linux', ['cargo', 'check', '-p', 'memra-tier', '-p', 'memra-kv', '--offline', '--all-targets', '--target', 'x86_64-unknown-linux-gnu']),
    ('test-tier', ['cargo', 'test', '-p', 'memra-tier', '--offline', '--no-fail-fast']),
    ('test-kv', ['cargo', 'test', '-p', 'memra-kv', '--offline', '--no-fail-fast']),
    ('clippy', ['cargo', 'clippy', '-p', 'memra-tier', '--offline', '--all-targets', '--', '-D', 'warnings']),
    ('telemetry', ['cargo', 'test', '-p', 'memra-tier', '--offline', '--test', 'storage', 'telemetry::', '--', '--nocapture']),
    ('runner', ['python3', 'research/spill-a-20260919/day3/test_runner.py']),
    ('dry-run', ['bash', 'research/spill-a-20260919/rig-cells-a.sh', '/never-created nvme', '--dry-run']),
    ('shell', ['bash', '-n', 'research/spill-a-20260919/rig-cells-a.sh']),
    ('fixtures', ['python3', 'crates/memra-tier/tests/contracts/fixture_reference.py', '--check']),
    ('diff', ['git', 'diff', '--check']),
    ('flags', ['bash', 'tools/check-flags.sh']),
]
rows = []
for name, argv in commands:
    start = time.monotonic()
    result = subprocess.run(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = OUT / (name + '.log.gz')
    log.write_bytes(gzip.compress(result.stdout, mtime=0))
    rows.append(dict(name=name, argv=argv, exit=result.returncode,
                     seconds=round(time.monotonic()-start, 3), log=log.name,
                     raw_sha256=hashlib.sha256(result.stdout).hexdigest()))
    print(f'{name}: exit={result.returncode}', flush=True)
    if name == 'telemetry':
        lines = [line.removeprefix('TELEMETRY_ROW ') for line in result.stdout.decode().splitlines() if line.startswith('TELEMETRY_ROW ')]
        (OUT/'telemetry-fixture.jsonl').write_text(''.join(line+'\n' for line in lines))
(OUT/'commands.json').write_text(json.dumps(rows, indent=2)+'\n')
files = [p for base in ('crates/memra-tier/src/object_store', 'crates/memra-tier/src/telemetry', 'crates/memra-tier/tests/storage') for p in (ROOT/base).rglob('*') if p.is_file() and p.suffix in ('.rs', '.py')]
(OUT/'source-manifest.json').write_text(json.dumps(dict(
    source_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}), indent=2)+'\n')
sys.exit(any(row['exit'] for row in rows))
