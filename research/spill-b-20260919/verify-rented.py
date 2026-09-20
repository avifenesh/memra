#!/usr/bin/env python3
"""Verify lossless archived logs and collector descriptors without GPU execution."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'rented-5090-20260919'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def raw(path):
    if path.exists():
        return path.read_bytes()
    return gzip.decompress(path.with_suffix(path.suffix + '.gz').read_bytes())

rows = json.loads((ROOT / 'raw-manifest.json').read_text())
for row in rows:
    archive = (ROOT / row['archive']).read_bytes()
    assert digest(archive) == row['sha256'], row['archive']
    assert digest(gzip.decompress(archive)) == row['raw_sha256'], row['archive']

def check_descriptors(value, directory):
    if isinstance(value, dict):
        if {'path', 'bytes', 'sha256'} <= value.keys():
            data = raw(directory / value['path'])
            assert len(data) == value['bytes'], value['path']
            assert digest(data) == value['sha256'], value['path']
        for child in value.values():
            check_descriptors(child, directory)
    elif isinstance(value, list):
        for child in value:
            check_descriptors(child, directory)

captures = list(ROOT.rglob('command.capture.json'))
for path in captures:
    check_descriptors(json.loads(path.read_text()), path.parent)
base = ROOT / 'baseline-8192'
receipt = base / 'receipt'
identity = dict(line.split('=', 1) for line in (receipt / 'identity.txt').read_text().splitlines())
assert digest((receipt / 'prompt.u32le').read_bytes()) == identity['prompt_sha256']
assert len((receipt / 'prompt.u32le').read_bytes()) == 8064 * 4
assert digest((receipt / 'plan.debug').read_bytes()) == identity['plan_debug_sha256']
assert identity['binary_sha256'] == (ROOT / 'gate-build/binary.sha256').read_text().split()[0]
assert not (receipt / 'BASELINE.txt').exists()
assert json.loads((base / 'command.capture.json').read_text())['exit_code'] == 2
assert 'REFUSED: capture requires' in (receipt / 'REFUSED.txt').read_text()
print(f'PASS: {len(rows)} archived raw hashes, {len(captures)} collector descriptor trees; baseline refusal and artifact-bound prompt/plan/binary linkage intact (NOT qualification)')
