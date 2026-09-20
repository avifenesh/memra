#!/usr/bin/env python3
"""Offline, fail-closed replay of day-8 raw receipts; never native qualification."""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent / 'day8/native'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def descriptors(value, root):
    if isinstance(value, dict):
        if {'path', 'bytes', 'sha256'} <= value.keys():
            path = (root/value['path']).resolve()
            assert path.is_relative_to(root.resolve()), 'descriptor escaped receipt'
            assert path.stat().st_size == value['bytes'], str(path)
            assert digest(path) == value['sha256'], str(path)
        for item in value.values():
            descriptors(item, root)
    elif isinstance(value, list):
        for item in value:
            descriptors(item, root)


def capture(root):
    row = json.loads((root/'command.capture.json').read_text())
    descriptors(row, root)
    assert row['status'] == 'executed-not-qualified' and row['exit_code'] == 0
    assert row['qualification'] is False and row['timed_out'] is False
    assert row['gpu_power_limits'] == [{'device':'0', 'power.limit':'600.00 W', 'power.max_limit':'600.00 W'}]
    for field in ('before','after'):
        assert row['compute_apps'][field]['exit_code'] == 0
        assert (root/f'command.{field}.log').read_text().strip() == 'pid, process_name, used_gpu_memory [MiB]'
    lock = json.loads((root/'lock.json').read_text())
    assert lock['rig'] == 'pro-single' and lock['lock'] == '/tmp/memra-gpu.lock' and lock['acquired']
    return row


def main():
    capture(ROOT/'conformance-retry1')
    verdicts = (ROOT/'conformance-retry1/command.log').read_text().splitlines()
    assert len(verdicts) == 8 and all(x.startswith('PASS ') for x in verdicts)
    capture(ROOT/'roundtrip')
    roundtrips = (ROOT/'roundtrip/command.log').read_text().splitlines()
    assert len(roundtrips) == 6
    for line, size in zip(roundtrips, (4096,65536,1048576,16777216,67108864,268435456)):
        match = re.fullmatch(r'PASS native D2H-H2D roundtrip bytes=(\d+) N=1 expected_sha256=([a-f0-9]{64}) actual_sha256=([a-f0-9]{64}) byte_exact=true source_freed_host_live=true handback_no_copy=true governor_zero=true', line)
        assert match and int(match[1]) == size and match[2] == match[3]
    candidates = sorted((ROOT/'storage-batch').glob('attempt-*/subcells.jsonl'))
    assert len(candidates) == 1, 'expected one executed batch, no selection between runs'
    batch = candidates[0].parent
    capture(batch)
    rows = [json.loads(line) for line in candidates[0].read_text().splitlines()]
    assert len(rows) == 9
    seen = {}
    for row in rows:
        raw = batch/row['raw_log']
        assert row['exit_code'] == 0 and row['qualification'] is False
        assert digest(raw) == row['sha256'] and raw.stat().st_size == row['bytes']
        assert row['cell'] not in seen
        seen[row['cell']] = raw
    storage = []
    for size in (264,4097,1048576,4194568):
        pair = []
        for operation in ('roundtrip','restore'):
            row = json.loads(seen[f'direct-{operation}-{size}'].read_text())
            assert row['status'] == 'byte-exact' and row['valid_bytes'] == size
            assert row['backend_requested'] == 'direct'
            assert row['backend_actual'] == 'linux-o-direct-read-write' and row['fallbacks'] == 0
            storage.append(dict(operation=operation, **row))
            pair.append(row['payload_checksum'])
        assert pair[0] == pair[1]
    pread = [json.loads(line) for line in seen['pread-baseline'].read_text().splitlines()]
    assert len(pread) == 8
    keys = set()
    for row in pread:
        key = (row['chunk_bytes'],row['mode'],row['cache_before'])
        assert key not in keys
        keys.add(key)
        assert row['N'] == 1 and row['byte_exact'] and not row['qualification']
        assert row['expected_sha256'] == row['actual_sha256']
        assert row['bytes'] == 64 << 20 and row['calls']*row['chunk_bytes'] == row['bytes']
        assert row['resident_pages_before'] == (0 if row['cache_before']=='cold' else row['total_pages'])
        assert row['read_ns'] > 0 and row['wall_with_hash_ns'] >= row['read_ns']
    assert keys == {(size,mode,cache) for size in (1<<20,16<<20)
                    for mode in ('buffered','O_DIRECT') for cache in ('cold','warm')}
    print(json.dumps(dict(status='raw-replay-pass', qualification=False,
        canonical_v13='HELD: native schedules not bound', conformance_verbatim=verdicts,
        roundtrip_sizes=[4096,65536,1048576,16777216,67108864,268435456],
        storage=storage, pread=pread, batch=str(batch.relative_to(ROOT))),indent=2))


if __name__ == '__main__':
    main()
