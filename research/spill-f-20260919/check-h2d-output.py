#!/usr/bin/env python3
"""Check probe plumbing output (not timing quality or G2 qualification)."""
import json
from pathlib import Path
import sys


def check(text, size, copies, order, dry=False):
    # D's tier-envelope.py at 7f7bf547 consumes bare JSON, not RESULT tokens.
    rows = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
    samples = [r for r in rows if r.get('record') == 'sample']
    assert len(samples) == 4, 'incomplete both-direction matrix'
    arms = ['pageable', 'pinned-cacheable']
    if order == 'ba':
        arms.reverse()
    for row, (direction, arm) in zip(samples, [(d, a) for d in ['h2d', 'd2h'] for a in arms]):
        for key, value in dict(bytes=size, copies=copies, order=order, n=1,
                               direction=direction, arm=arm,
                               event_timing='sum-per-operation-owner-stream',
                               completed_bytes=0 if dry else size*copies,
                               verified_bytes=0 if dry else size).items():
            assert row[key] == value, (key, row[key], value)
        assert row['pinned_flags'] == (0 if arm == 'pinned-cacheable' else None)
        if dry:
            for key in ['identity', 'wall_ns', 'event_ms', 'expected_sha256', 'actual_sha256',
                        'power_before', 'power_after']:
                assert row[key] is None, key
        else:
            assert row['identity'] is True
            assert row['expected_sha256'] == row['actual_sha256']
            assert len(row['actual_sha256']) == 64
            assert row['power_before'] == row['power_after']
            assert row['wall_ns'] > 0 and row['event_ms'] >= 0
            assert row['mono_end_ns'] >= row['mono_start_ns']
            assert row['unix_end_ns'] >= row['unix_start_ns']
    summaries = [r for r in rows if r.get('record') == 'RESULT']
    assert len(summaries) == 1
    assert summaries[0]['qualified'] is False
    assert summaries[0]['samples'] == 4
    assert summaries[0]['comparator_red_rejected'] is True
    controls = [r for r in rows if r.get('record') == 'control']
    assert len(controls) == (0 if dry else 8)
    assert all(r['identity'] is True for r in controls)
    return samples


if __name__ == '__main__':
    check(Path(sys.argv[1]).read_text(), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4],
          dry='--dry' in sys.argv[5:])
    print('PASS: complete N=1 visit output; not G2 qualification')
