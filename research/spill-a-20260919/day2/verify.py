#!/usr/bin/env python3
"""CPU-only reproducer. Never runs a GPU/model gate or makes a spill-speed claim."""
import gzip
import hashlib
import json
from pathlib import Path
import plistlib
import shutil
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'research/spill-a-20260919/day2'
RAW = OUT / 'raw'
RAW.mkdir(exist_ok=True)
commands = []


def run(name, command, expected=0, timeout=180):
    path = RAW / (name + '.log')
    with path.open('wb') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    data = path.read_bytes()  # Parse only after raw stdout/stderr is retained.
    (RAW / (name + '.log.gz')).write_bytes(gzip.compress(data, mtime=0))
    path.unlink()
    commands.append({'name': name, 'command': command, 'exit': result.returncode, 'expected': expected})
    (OUT / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    print(f'{name}: exit={result.returncode}; expected={expected}')
    if result.returncode != expected:
        raise RuntimeError(data.decode(errors='replace'))
    return data


checks = [
    ('fmt', ['cargo', 'fmt', '--all', '--', '--check']),
    ('check', ['cargo', 'check', '-p', 'memra-tier', '--offline', '--all-targets']),
    ('test', ['cargo', 'test', '-p', 'memra-tier', '--offline']),
    ('diff', ['git', 'diff', '--check']),
    ('flags', ['bash', 'tools/check-flags.sh']),
    ('clippy', ['cargo', 'clippy', '-p', 'memra-tier', '--offline', '--all-targets', '--', '-D', 'warnings']),
    ('fixtures', ['python3', 'crates/memra-tier/tests/contracts/fixture_reference.py', '--check']),
    ('build-cpu', ['cargo', 'build', '-p', 'memra-tier', '--offline']),
]
for name, command in checks:
    run(name, command)
sha = sorted((ROOT/'target/debug/deps').glob('libsha2-*.rlib'), key=lambda p: p.stat().st_mtime)[-1]
run('build-cli', ['rustc', '--edition=2024', 'crates/memra-engine/src/bin/storage_bench.rs',
                  '--extern', 'memra_tier=target/debug/libmemra_tier.rlib',
                  '--extern', 'sha2=' + str(sha.relative_to(ROOT)),
                  '-L', 'dependency=target/debug/deps', '-o', 'target/debug/storage-bench'])
# Only selected non-identifying storage-shape fields enter the receipt.
info = plistlib.loads(subprocess.run(['diskutil', 'info', '-plist', '/'], capture_output=True, check=True).stdout)
shape = {k: info.get(k) for k in ['FilesystemType', 'SolidState', 'BusProtocol', 'Internal']}
assert shape['SolidState'] is True
(OUT/'storage-shape.json').write_text(json.dumps(shape, indent=2)+'\n')
rows = []
scratch = OUT/'owned-characterization-scratch'
scratch.mkdir()  # Refuse any prior scratch, never delete unrelated files.
try:
    for size in [264, 1048576, 4194568]:
        for backend in ['buffered', 'uncached']:
            path = str((scratch/f'{size}-{backend}').relative_to(ROOT))
            for mode in ['roundtrip', 'restore']:
                name = f'{mode}-{size}-{backend}'
                data = run(name, ['target/debug/storage-bench', mode, path, str(size), backend])
                row = json.loads(data)
                payload = bytes((17*i+3)%251 for i in range(size))
                domain = b'valid-bytes'
                expected = hashlib.sha256(b'memra-tier\0v1\0'+len(domain).to_bytes(8,'little')+domain+size.to_bytes(8,'little')+payload).digest()
                assert bytes(row['payload_checksum']) == expected
                assert row['version'] == 1 and row['valid_bytes'] == size
                assert row['physical_bytes'] is None and row['pinned_bytes'] == 0
                assert row['h2d_ns'] is None and row['d2h_ns'] is None and row['p2p_ns'] is None
                assert row['fallbacks'] == (1 if backend == 'uncached' else 0)
                rows.append(row)
                if mode == 'roundtrip':
                    run(name+'-refuse-reuse', ['target/debug/storage-bench', mode, path, str(size), backend], expected=1)
    run('unsupported-gpu-mode', ['target/debug/storage-bench', 'h2d', str(scratch.relative_to(ROOT))], expected=1)
    (OUT/'runs.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
finally:
    shutil.rmtree(scratch)
source = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
paths = list((ROOT/'crates/memra-tier/src').rglob('*.rs')) + list((ROOT/'crates/memra-tier/tests/storage').rglob('*.rs'))
paths += [ROOT/'crates/memra-engine/src/bin/storage_bench.rs', ROOT/'target/debug/storage-bench']
manifest = {'source_commit':source,'utc':datetime.now(timezone.utc).isoformat(),'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(OUT/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('CHARACTERIZATION_PASS: 12 single-run byte-exact rows; development-Mac I/O only; no spill-speed claim')
