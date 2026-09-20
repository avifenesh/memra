#!/usr/bin/env python3
"""Replay day-nine raw native rehearsal receipts; no GPU and no medians."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT/'research/spill-d-20260919/day9/native'
SPEC = importlib.util.spec_from_file_location('native_replay', ROOT/'tools/tier-envelope.py')
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)
# The native collector's CSV clock is UTC; replay independently of the reader's zone.
os.environ['TZ'] = 'UTC'
time.tzset()
manifest = json.loads((EV/'export-manifest.json').read_text())
assert all(E.B.digest(EV/path) == sha for path, sha in manifest['export_sha256'].items())
assert (EV/'build/exit').read_text().strip() == '0'
binary_hash = (EV/'build/binary.sha256').read_text().split()[0]
build_commit = (EV/'build/source.txt').read_text().strip()
probe_path = 'crates/memra-engine/src/bin/h2d_probe.rs'
built_source = subprocess.check_output(['git', 'show', build_commit+':'+probe_path], cwd=ROOT)
assert built_source == subprocess.check_output(['git', 'show', '7644c41f:'+probe_path], cwd=ROOT)
rows_out = []
for size in (4096, 16777216):
    for direction in ('h2d', 'd2h'):
        path = EV/'rehearsal'/f'g2-{size}-{direction}'
        capture = E.B.validate_cell(path/'collector/CELL.jsonl')
        assert capture['status'] == 'executed-not-qualified'
        identity = json.loads((path/'identity.json').read_text())
        assert identity['binary_sha256'] == binary_hash
        for field, file in [('collector_sha256', 'tools/tier-battery.py'), ('runner_sha256', 'tools/tier-envelope.py')]:
            source = subprocess.check_output(['git', 'show', identity['source_commit']+':'+file], cwd=ROOT)
            assert hashlib.sha256(source).hexdigest() == identity[field]
        assert built_source == subprocess.check_output(['git', 'show', identity['source_commit']+':'+probe_path], cwd=ROOT)
        rows = [json.loads(line) for line in (path/'envelope.jsonl').read_text().splitlines()]
        samples = [row for row in rows if row['phase'] != 'calibration']
        assert [(row['round'], row['order'], row['arm']) for row in samples] == [
            (0, 'AB', 'pageable'), (0, 'AB', 'pinned-cacheable'),
            (0, 'BA', 'pinned-cacheable'), (0, 'BA', 'pageable')]
        calibration = json.loads((path/'calibration.json').read_text())
        copies = calibration['copies']
        assert calibration['discarded'] and all(r['sample']['wall_ns'] >= 250_000_000 for r in samples)
        for row in rows:
            E.validate_visit(row['sample'], size, direction, row['arm'], row['sample']['copies'])
            raw = E.B.evidence(path, row['raw_log']).read_text()
            assert row['sample'] in [json.loads(line) for line in raw.splitlines() if line.startswith('{')]
            assert row['sample']['copies'] == copies or row['phase'] == 'calibration'
        cap = rows[0]['power_before']
        assert all(r['power_before'] == r['power_after'] == cap for r in rows)
        telemetry = E.telemetry_check(path/'collector/command.gpu.csv', rows[0]['started_unix_seconds'],
                                     rows[-1]['ended_unix_seconds'], cap)
        summary = json.loads((path/'summary.json').read_text())
        assert summary['telemetry'] == telemetry
        assert summary['scoring_eligible'] is False and summary['medians_published'] is False
        rows_out.append({'bytes': size, 'direction': direction, 'copies': copies, 'visits': len(samples),
                         'minimum_visit_ns': min(r['sample']['wall_ns'] for r in samples),
                         'telemetry': telemetry, 'qualification': False})
failed = EV/'g2-4096-h2d'
assert E.B.validate_cell(failed/'collector/CELL.jsonl')['status'] == 'failed'
assert all(json.loads(line)['phase'] == 'calibration' for line in (failed/'envelope.jsonl').read_text().splitlines())
print(json.dumps({'status': 'executed-not-qualified', 'cells': rows_out, 'visits': 16,
                  'retained_failed_calibration_cells': 1, 'qualification': False,
                  'binary_sha256': binary_hash, 'build_commit': build_commit}, indent=2))
