#!/usr/bin/env python3
"""Replay day-9 receipts; no GPU execution or qualification is implied."""
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from verify_day8 import descriptors

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = (
    'PASS v1.3 transfer_source_retirement native CUDA',
    'PASS v1.3 device_hand_back native CUDA',
)
SIZES = (4096, 65536, 1048576, 16777216, 67108864, 268435456)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def roundtrips(text):
    lines = text.splitlines()
    assert len(lines) == len(SIZES), 'six roundtrips required'
    for size, line in zip(SIZES, lines):
        match = re.fullmatch(r'PASS native D2H-H2D roundtrip bytes=(\d+) N=1 expected_sha256=([a-f0-9]{64}) actual_sha256=([a-f0-9]{64}) byte_exact=true source_freed_host_live=true handback_no_copy=true governor_zero=true', line)
        assert match and int(match[1]) == size and match[2] == match[3], line


def verify(root):
    manifest = json.loads((root/'remote-hashes.json').read_text())
    files = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name != 'remote-hashes.json'}
    assert set(manifest) == files, 'manifest must cover every raw file'
    for name, digest in manifest.items():
        path = (root/name).resolve()
        assert path.is_relative_to(root.resolve()), 'manifest escaped receipt'
        assert sha(path.read_bytes()) == digest, name
    assert (root/'build.exit').read_text().strip() == '0'
    identity = json.loads((root/'identity.json').read_text())
    commit = identity['source_commit']
    assert re.fullmatch('[a-f0-9]{40}', commit)
    for field, path in (
        ('runner_sha256', 'research/spill-a-20260919/day9_cells.py'),
        ('collector_sha256', 'tools/tier-battery.py'),
        ('contract_sha256', 'crates/memra-tier/src/conformance/revision_v13.rs'),
    ):
        source = subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT)
        assert sha(source) == identity[field], field
    captures = sorted(root.glob('attempt-*/command.capture.json'))
    assert len(captures) == 1, 'expected exactly one executed campaign, no selection'
    run = captures[0].parent
    capture = json.loads(captures[0].read_text())
    descriptors(capture, run)
    assert capture['exit_code'] == 0 and not capture['timed_out']
    assert capture['status'] == 'executed-not-qualified'
    assert capture['qualification'] is False
    for case in ('conformance', 'roundtrip', 'lock-proof'):
        assert (run/(case+'.exit')).read_text().strip() == '0'
    lock = json.loads((run/'lock.json').read_text())
    worker = json.loads((run/'lock-proof.log').read_text())
    assert lock['rig'] == 'pro-single' and lock['acquired']
    for record in (lock, worker):
        assert record['lock'] == '/tmp/memra-gpu.lock'
        assert record['owner'] == 'collector'
        assert record['mechanism'] == 'inherited-flock-same-open-description'
    assert (lock['device'], lock['inode']) == (worker['device'], worker['inode'])
    for side in ('before', 'after'):
        assert capture['compute_apps'][side]['exit_code'] == 0
        assert (run/f'command.{side}.log').read_text().strip() == 'pid, process_name, used_gpu_memory [MiB]'
    binary = json.loads((run/'binary.json').read_text())
    assert binary['sha256'] == identity['binary_sha256'] and not binary['qualification']
    verdicts = (run/'conformance.log').read_text().splitlines()
    assert len(verdicts) == 11 and all(line.startswith('PASS ') for line in verdicts)
    for line in CANONICAL:
        assert verdicts.count(line) == 1
    assert verdicts[-1] == 'PASS native governor zero after controlled drain'
    roundtrips((run/'roundtrip.log').read_text())
    # Missing telemetry is incomplete evidence, not malformed correctness data.
    telemetry = capture.get('gpu_telemetry', {})
    power = capture.get('gpu_power_limits', [])
    if power:
        assert power == [{'device': '0', 'power.limit': '600.00 W', 'power.max_limit': '600.00 W'}]
    samples = []
    if telemetry.get('raw_csv'):
        samples = list(csv.DictReader((run/telemetry['raw_csv']['path']).read_text().splitlines(), skipinitialspace=True))
        assert telemetry.get('interval_ms') == 250
        for sample in samples:
            assert sample['power.limit [W]'] == '600.00 W'
            assert sample['power.max_limit [W]'] == '600.00 W'
    assert 'RTX PRO 6000 Blackwell' in (root/'hardware-after.log').read_text()
    return dict(status='raw-replay-pass', qualification=False, source_commit=commit,
        binary_sha256=binary['sha256'], canonical_verdicts=list(CANONICAL),
        conformance_verbatim=verdicts, roundtrip_sizes=list(SIZES),
        governor_drain=True, files_hash_checked=len(manifest),
        telemetry_interval_ms=telemetry.get('interval_ms'), power_limits=power,
        telemetry_samples=len(samples),
        telemetry_complete=bool(samples and power))


if __name__ == '__main__':
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name('day9')/'native/final'
    print(json.dumps(verify(root), indent=2))
