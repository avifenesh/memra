#!/usr/bin/env python3
"""Recheck the persisted native endpoint and accounting receipts without a GPU."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
raw = root / 'raw' / 'attempt2'
summary = json.loads((raw / 'on' / 'summary.json').read_text())
http = [json.loads(line) for line in (raw / 'on' / 'http.jsonl').read_text().splitlines()]
rows = [json.loads(line) for arm in ('off', 'on') for line in (raw / arm / 'receipts.jsonl').read_text().splitlines()]
assert len({row['id'] for row in rows}) == len(rows) == 5
bodies = {}
for key in ('complete_id', 'cancel_id'):
    ident = summary[key]
    body = next(item['body'] for item in reversed(http) if item['status'] == 200 and item['body'].get('id') == ident and item['body'].get('status') in ('completed', 'cancelled'))
    row = next(row for row in rows if row['id'] == ident)
    usage = body['usage']
    assert row['prompt_tokens'] == usage['input_tokens']
    assert row['cached_tokens'] == usage['input_tokens_details']['cached_tokens']
    assert row['completion_tokens'] == usage['output_tokens']
    bodies[key] = body

def text(body):
    return ''.join(part['text'] for item in body['output'] if item['type'] == 'message' for part in item['content'] if part['type'] == 'output_text')

complete = text(bodies['complete_id'])
partial = text(bodies['cancel_id'])
assert partial and complete.startswith(partial)
assert len(json.loads(complete)) == 600
assert summary['complete_native_s'] > 90
assert summary['pass'] is True
print(json.dumps({'pass': True, 'accepted_requests': 5, 'terminal_callbacks_each': 1, 'all_usage_fields_match': True, 'cancel_text_is_original_prefix': True, 'native_duration_s': summary['complete_native_s'], 'complete_tokens': summary['complete_tokens'], 'cancel_tokens': summary['cancel_tokens']}, indent=2))
