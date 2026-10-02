#!/usr/bin/env python3
"""Verify the retained native HTTP and vendor-default receipts without a GPU."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent / 'raw'

def answer(body):
    if body.get('object') == 'response':
        assert body['status'] == 'completed'
        return ''.join(part['text'] for item in body['output'] if item.get('type') == 'message' for part in item['content'] if part.get('type') == 'output_text')
    assert body['choices'][0]['finish_reason'] == 'stop'
    return body['choices'][0]['message']['content']

reports = []
for name in ('qwen-pass', 'qwen-vendor-pass', 'qwen-combined'):
    raw = root / name
    on = json.loads((raw / 'on/summary.json').read_text())
    off = json.loads((raw / 'off/summary.json').read_text())
    assert on['pass'] and off['pass']
    assert on['scope'] == off['scope'] == 'native_vision'
    assert on['vision_acceptance'] == off['vision_acceptance'] == 'passed'
    rows = {r['case']: r for r in map(json.loads, (raw / 'on/http.jsonl').read_text().splitlines())}
    for key, row in rows.items():
        positive = key.startswith(('inline-', 'remote-', 'five-redirect-control', 'vendor-default-', 'background-remote-submit', 'background-poll-'))
        assert row['status'] == (200 if positive else 400), key
    red = answer(rows['inline-red-control']['body'])
    blue = answer(rows['inline-blue-control']['body'])
    assert red.strip().lower() == 'red' and blue.strip().lower() == 'blue'
    for key in ('remote-red-False', 'remote-red-True', 'five-redirect-control'):
        assert answer(rows[key]['body']) == red
    assert answer(rows['remote-blue-control']['body']) == blue
    for key in ('whole-byte-budget-False', 'whole-byte-budget-True'):
        assert rows[key]['request_bytes'] < 192 * 1024 * 1024
    for key in ('whole-byte-budget-False', 'whole-byte-budget-True', 'grow', 'large-header'):
        assert rows[key]['body']['error']['code'] == 'image_url_too_large'
    assert rows['redirect-scheme']['body']['error']['code'] == 'image_url_blocked'
    for key, low, high in [('per-image-timeout', 9, 14), ('caller-deadline', 0.7, 4), ('whole-pass-timeout', 19, 24)]:
        assert low <= rows[key]['elapsed_s'] <= high
    text = (raw / 'on/transport-unit.log').read_text()
    assert 'IMAGE_FETCH_FIXTURE_PASS' in text and '1 passed' in text
    if name in ('qwen-vendor-pass', 'qwen-combined'):
        for api, keys, case in [('chat', {'model', 'messages'}, 'vendor-default-False'), ('responses', {'model', 'input'}, 'vendor-default-True')]:
            request = json.loads((raw / 'on' / f'vendor-default-{api}-request.json').read_text())
            assert set(request) == keys, request.keys()
            assert answer(rows[case]['body']).strip().lower() == 'red'
            assert on['vendor_default_' + api]['decode_fields'] == []
    if name == 'qwen-combined':
        assert on['background_remote_image_and_preflight']['pass']
        request = json.loads((raw / 'on/background-remote-request.json').read_text())
        assert request['background'] is True
        assert rows['background-remote-submit']['body']['status'] == 'queued'
        polls = [row for key, row in rows.items() if key.startswith('background-poll-')]
        assert polls and answer(polls[-1]['body']) == red
        assert rows['background-literal-blocked']['body']['error']['code'] == 'image_url_blocked'
        assert rows['background-caller-deadline']['body']['error']['code'] == 'image_url_unreachable'
        assert 0.7 <= rows['background-caller-deadline']['elapsed_s'] <= 4
    reports.append({'dataset': name, 'pass': True, 'on_http_cases': len(rows), 'off_no_fetch': off['door_off_no_fetch']['pass'], 'native_colour_and_identity': True, 'exact_fetch_bytes': True, 'bare_requests': name in ('qwen-vendor-pass', 'qwen-combined')})
print(json.dumps({'pass': True, 'receipts': reports}, indent=2))
