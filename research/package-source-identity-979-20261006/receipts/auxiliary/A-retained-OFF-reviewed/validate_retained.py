"""Validate completed retained OFF bytes only. This launches no server or GPU work."""
import hashlib
import json
import re
from pathlib import Path

INPUTS = Path('/home/evidence-user/.cache/evidence-manager-20261006/A-serving-v3-reviewed/ROOT-SERVING-INPUTS.json')
inputs = json.loads(INPUTS.read_bytes())
root = Path(inputs['record_root'])
records = root / 'cache-off'
readset = {}

def raw(path):
    value = path.read_bytes()
    readset[str(path)] = {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}
    return value

def obj(path):
    return json.loads(raw(path))

requests = obj(Path(inputs['request_file']))
assert readset[inputs['request_file']]['sha256'] == inputs['request_sha256']
assert len(requests['turns']) == 8
version = raw(root / 'version.stdout').decode()
assert 'build_id_src package-source-v1' in version
fingerprint = next(line.removeprefix('system_fingerprint ') for line in version.splitlines()
                   if line.startswith('system_fingerprint '))
assert not raw(root / 'version.stderr')
forbidden = {'temperature', 'top_p', 'top_k', 'min_p', 'seed', 'frequency_penalty',
             'presence_penalty', 'repetition_penalty', 'reasoning_effort'}
assert all(not forbidden.intersection(row) for row in requests['turns'] + [requests['stream']])
assert obj(records / 'ready.request.json') is None
ready = obj(records / 'ready.response')
models = obj(records / 'models.response')
assert any(row['id'] == 'package-probe' for row in models['data'])
rows = []
for i, payload in enumerate(requests['turns'], 1):
    assert obj(records / f'turn{i}.request.json') == payload
    value = obj(records / f'turn{i}.response')
    assert value['system_fingerprint'] == fingerprint
    assert value['model'] == 'package-probe'
    assert isinstance(value['id'], str) and value['id']
    assert isinstance(value['created'], int)
    assert value['usage']['completion_tokens'] > 0
    assert value['choices'][0]['finish_reason'] in ('stop', 'length')
    message = value['choices'][0]['message']
    assert ((message.get('reasoning') or '') + (message.get('content') or '')).strip()
    assert value['usage']['spec']['rounds'] > 0 and value['usage']['spec']['drafted'] > 0
    rows.append({'turn': i, 'cached_tokens': value['usage']['prompt_tokens_details']['cached_tokens'],
                 'spec': value['usage']['spec']})
assert obj(records / 'stream.request.json') == requests['stream']
frames = [line[6:] for line in raw(records / 'stream.response').decode().splitlines()
          if line.startswith('data: ')]
assert frames and frames[-1] == '[DONE]'
decoded = [json.loads(frame) for frame in frames[:-1]]
assert decoded and all(frame['system_fingerprint'] == fingerprint for frame in decoded)
assert len({frame['id'] for frame in decoded}) == 1
assert all(isinstance(frame['created'], int) and frame['model'] == 'package-probe' for frame in decoded)
log = raw(records / 'server.log').decode()
assert re.search(r'\[spec-k\] model="package-probe".*K=[1-9][0-9]*', log)
assert '[env-audit] REFUSED' not in log and 'registry absent' not in log
attach = [line for line in log.splitlines() if
          '[mtp-draft] loading external MTP draft:' in line or 'regime draft attached (' in line]
assert attach and any(inputs['draft']['path'] in line for line in attach)
refusals = re.findall(r'\[prefix-cache\] insert refused: entry \d+ exceeds budget (\d+)', log)
assert refusals and set(refusals) == {'0'}
prefix_lines = [line for line in log.splitlines() if '[prefix-cache]' in line]
assert not any(re.search(r'\bhit\b|\brestored?\b', line) for line in prefix_lines)
affinity = [int(x) for x in re.findall(r'spec-affinity: rewound to (\d+) of', log)]
assert [row['cached_tokens'] for row in rows if row['cached_tokens'] > 0] == affinity
teardown = obj(records / 'TEARDOWN.json')
assert teardown['owned_process_retired'] and teardown['reader_joined'] and teardown['reader_pipe_closed']
assert not teardown['cleanup_errors'] and teardown['exit'] == 0
assert teardown['primary_error'] == 'AssertionError: '
for retired in teardown['owned_session_retirement']:
    assert not retired['live_members_after'] and retired['leader_reaped_after_session_retired']
proofs = teardown['canonical_lock_before_after']
assert len(proofs) == 2
assert all(proof['lock'] == '/tmp/memra-5090.lock' and
           proof['mechanism'] == 'inherited-flock-same-open-description' for proof in proofs)
assert (proofs[0]['device'], proofs[0]['inode']) == (proofs[1]['device'], proofs[1]['inode'])
result = {'status': 'Retained OFF predicates inspected and passed in a separate CPU validation',
          'original_GPU_job': 'd6f7265ceab8', 'original_GPU_job_status': 'FAILED and unchanged',
          'original_failure': 'private cached_tokens-zero predicate conflated affinity with prefix storage',
          'source': inputs['source'], 'binary_sha256': inputs['binary_sha256'],
          'fingerprint': fingerprint, 'turns': rows, 'SSE_frames': len(decoded),
          'prefix_budget_zero_and_insert_refusals': True, 'affinity_reuse_distinct': True,
          'retained_teardown_and_lock_proofs': True, 'new_server_or_GPU_or_artifact_hash_run': False,
          'qualification': False, 'readset': readset}
out = Path(__file__).parent / 'RESULT.json'
assert not out.exists()
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('readset', 'turns')}))
