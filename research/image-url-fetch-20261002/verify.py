#!/usr/bin/env python3
"""Verify the retained native HTTP fixture verdicts without model execution."""
import json
from pathlib import Path
root = Path(__file__).resolve().parent / 'raw' / 'qwen-pass'
on = json.loads((root / 'on/summary.json').read_text())
off = json.loads((root / 'off/summary.json').read_text())
assert on['pass'] and off['pass']
assert on['scope'] == off['scope'] == 'native_vision'
assert on['vision_acceptance'] == off['vision_acceptance'] == 'passed'
rows = {r['case']: r for r in map(json.loads, (root / 'on/http.jsonl').read_text().splitlines())}
for key, row in rows.items():
    positive = key.startswith(('inline-', 'remote-', 'five-redirect-control'))
    assert row['status'] == (200 if positive else 400), key
assert rows['whole-byte-budget-False']['request_bytes'] < 192 * 1024 * 1024
assert rows['whole-byte-budget-True']['request_bytes'] < 192 * 1024 * 1024
for key in ('whole-byte-budget-False', 'whole-byte-budget-True', 'grow', 'large-header'):
    assert rows[key]['body']['error']['code'] == 'image_url_too_large'
assert rows['redirect-scheme']['body']['error']['code'] == 'image_url_blocked'
for key, low, high in [('per-image-timeout',9,14), ('caller-deadline',0.7,4), ('whole-pass-timeout',19,24)]:
    assert low <= rows[key]['elapsed_s'] <= high
report = (root / 'on/transport-unit.log').read_text()
assert 'IMAGE_FETCH_FIXTURE_PASS' in report and '1 passed' in report
print(json.dumps({'pass':True, 'native_vision':True, 'on_http_cases':len(rows), 'off_no_fetch':off['door_off_no_fetch']['pass'], 'exact_production_fetch_bytes':True, 'timeout_seconds':{key:rows[key]['elapsed_s'] for key in ('per-image-timeout','caller-deadline','whole-pass-timeout')}},indent=2))
