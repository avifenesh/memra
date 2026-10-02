#!/usr/bin/env python3
"""Check the captured numerical contracts and deduplicated raw f32 receipts."""
import hashlib
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parent

def rows(name):
    return [json.loads(line) for line in (ROOT / name).read_text().splitlines()]

log_index = json.loads((ROOT / 'log-index.json').read_text())
with tarfile.open(ROOT / 'logs.tar.gz', 'r:gz') as archive:
    members = [member for member in archive if member.isfile()]
    assert {member.name for member in members} == set(log_index)
    for member in members:
        assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == log_index[member.name]

index = json.loads((ROOT / 'f32-index.json').read_text())
blobs = {}
with tarfile.open(ROOT / 'f32-receipts.tar.gz', 'r:gz') as archive:
    for member in archive:
        assert member.isfile() and re.fullmatch(r'[0-9a-f]{64}\.f32', member.name)
        data = archive.extractfile(member).read()
        digest = hashlib.sha256(data).hexdigest()
        assert member.name == digest + '.f32'
        assert digest not in blobs
        blobs[digest] = len(data)
for path, item in index.items():
    assert not Path(path).is_absolute() and '..' not in Path(path).parts
    assert item['bytes'] > 0 and item['bytes'] % 4 == 0
    assert blobs[item['sha256']] == item['bytes']
assert set(blobs) == {item['sha256'] for item in index.values()}

def equal(left, right):
    return index[left] == index[right]

inputs = rows('inputs.jsonl')
assert [x['i'] for x in inputs] == [288, 0, 1, 4]
for item in inputs:
    assert hashlib.sha256(item['request']['prompt'].encode()).hexdigest() == item['prompt_sha256']
for http_arm, api_arm in [('http', 'split4'), ('http-unsplit', 'aligned')]:
    http = rows(f'cell1/{http_arm}.jsonl')
    logits = rows(f'cell1/{http_arm}-logits.jsonl')
    api = rows(f'cell1/{api_arm}.jsonl')
    assert len(http) == len(logits) == len(api) == 4
    for h, l, d, expected in zip(http, logits, api, inputs):
        assert h['i'] == d['i'] == expected['i']
        assert h['tokens']['tokens'] == d['tokens']
        assert d['tokens'][0] == 2
        assert h['response']['tokens'] == [l['top5'][0]['id']]
        assert h['response']['tokens_predicted'] == 1
        assert l['top5'] == d['top5']
for arm in ['aligned', 'split4']:
    for item in inputs:
        i = item['i']
        plain = f'cell1/{arm}-full/{i}-logits.f32'
        assert not equal(plain, f'traced-{arm}-full/{i}-logits.f32')
        unfused = f'nofusion-{arm}-full/{i}-logits.f32'
        assert equal(unfused, f'nofusion-traced-{arm}-full/{i}-logits.f32')
        assert equal(plain, f'fusion-family/none-{arm}-full/{i}-logits.f32')
        for family in ['matmul', 'other']:
            assert equal(plain, f'fusion-family/{family}-{arm}-full/{i}-logits.f32')
        assert equal(unfused, f'fusion-family/rms-{arm}-full/{i}-logits.f32')
        assert not equal(plain, unfused)
for line, item in zip((ROOT / 'native-pin.tsv').read_text().splitlines(), inputs):
    i, tokens, top = line.split('\t')
    assert int(i) == item['i']
    new = [entry.split(':') for entry in top.split(',')]
    old = item['prior']['memra_top5']
    assert len(new) == len(old) == 5
    for (token, value), prior in zip(new, old):
        assert int(token) == prior['id']
        assert f'{float(value):.9f}' == f"{prior['logit']:.9f}"
print(json.dumps({'raw_paths':len(index), 'unique_f32_blobs':len(blobs),
    'matched_http_api_pairs':8, 'intrusive_default_trace_pairs':8,
    'transparent_unfused_trace_pairs':8, 'instrumented_default_identity_pairs':8,
    'rms_only_equals_unfused_pairs':8, 'other_family_no_change_pairs':16,
    'historical_native_top5_rows':4}, indent=2))
