#!/usr/bin/env python3
"""Dependency-free validation against D's schema subset, not a schema copy."""
import json
from pathlib import Path
import sys
SCHEMA = Path(__file__).resolve().parents[4] / 'research/spill-d-20260919/telemetry.schema.json'

def check(value, schema):
    if 'anyOf' in schema:
        for candidate in schema['anyOf']:
            try:
                check(value, candidate)
                return
            except AssertionError:
                pass
        raise AssertionError(f'no matching type: {value!r}')
    if 'const' in schema:
        assert value == schema['const']
    if 'enum' in schema:
        assert value in schema['enum']
    kind = schema.get('type')
    if kind == 'object':
        assert isinstance(value, dict)
        assert set(schema.get('required', [])) <= value.keys()
        if schema.get('additionalProperties') is False:
            assert value.keys() <= schema['properties'].keys()
        for key, item in value.items():
            check(item, schema['properties'][key])
    elif kind == 'array':
        assert isinstance(value, list) and len(value) >= schema.get('minItems', 0)
        for item in value:
            check(item, schema['items'])
    elif kind == 'integer':
        assert type(value) is int and value >= schema.get('minimum', float('-inf'))
    elif kind == 'number':
        assert type(value) in (int, float) and value >= schema.get('minimum', float('-inf'))
    elif kind == 'null':
        assert value is None

schema = json.loads(SCHEMA.read_text())
rows = [json.loads(line) for line in sys.stdin if line.strip()]
assert rows
for row in rows:
    check(row, schema)
    for wait in row['wait_ns'].values():
        assert wait['p50'] <= wait['p95'] <= wait['p99']
for prev, row in zip(rows, rows[1:]):
    assert 0 < row['monotonic_ns'] - prev['monotonic_ns'] <= 500_000_000
    for field in ['read_bytes', 'write_bytes']:
        assert row['nvme'][field] >= prev['nvme'][field]
# Red arm: the exact schema must refuse missing required fields and wrong labels.
for field in schema['required']:
    bad = dict(rows[0]); del bad[field]
    try:
        check(bad, schema)
    except AssertionError:
        continue
    raise AssertionError(f'validator accepted omitted {field}')
print(f'TELEMETRY_SCHEMA_PASS: {len(rows)} rows; required-field red arms passed')
