"""Bank compact, hash-bound receipts without copying server logs or request bodies."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ap = argparse.ArgumentParser()
ap.add_argument('lane', type=Path)
ap.add_argument('output', type=Path)
a = ap.parse_args()

def digest(path):
    return {'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest(), 'bytes': path.stat().st_size}

def small_row(row):
    return {k: row.get(k) for k in ['verdict', 'prompt_tokens', 'cached_tokens', 'completion_tokens', 'finish', 'ttft_s', 'wall_s', 'spec', 'error', 'out_sha16']}

report = {'issue': 372, 'base': '72aa777c3763a21281fb9c1c55b2499f2c14d1ee', 'binaries': {}, 'cells': {}, 'checks': {}}
for name in ['candidate', 'candidate-final', 'candidate-faults', 'main-control', 'candidate-pressure']:
    path = a.lane / name
    if path.exists():
        report['binaries'][name] = digest(path)
for path in sorted(a.lane.glob('*/receipt.json')):
    raw = json.loads(path.read_text())
    result = {'binary_sha256': raw['binary_sha256'], 'harness_passed': raw['passed'], 'error': raw.get('error'), 'server_exit': raw.get('server_exit'), 'gpu_after': raw.get('gpu_after'), 'rows': []}
    for cell in raw['rows']:
        lines = cell['log']
        result['rows'].append({'cell': cell['cell'], 'row': small_row(cell['row']), 'faults': cell['faults'], 'log': [l for l in lines if '[dspark-acc]' not in l and '[dflash-oracle]' not in l] + [l for l in lines if '[dspark-acc]' in l][-1:], 'before': {k:v for k,v in cell['before'].items() if k in ['admitted', 'completed', 'tokens_out', 'cuda_pool_used_bytes', 'cuda_pool_reserved_bytes', 'prefix_cache_bytes', 'plain_reuse_parked', 'spec_reuse_parked', 'dspark_reuse_parked']}, 'after': {k:v for k,v in cell['after'].items() if k in ['admitted', 'completed', 'tokens_out', 'cuda_pool_used_bytes', 'cuda_pool_reserved_bytes', 'prefix_cache_bytes', 'plain_reuse_parked', 'spec_reuse_parked', 'dspark_reuse_parked']}})
    result['direct_pool_readback'] = [l for l in (path.parent / 'server.log').read_text(errors='replace').splitlines() if '[retained-pool]' in l][-8:]
    result['references'] = {p.name: digest(p) for p in path.parent.iterdir() if p.name in ['server.log', 'profile.json', 'models.toml', 'receipt.json'] or p.name.endswith('-request.json')}
    report['cells'][path.parent.name] = result
for path in sorted(a.lane.glob('*.log')):
    if not path.name.startswith(('tests-', 'fmt', 'clippy', 'build')):
        continue
    lines = path.read_text(errors='replace').splitlines()
    report['checks'][path.name] = {**digest(path), 'evidence': [l for l in lines if re.search(r'test result: (ok\. [1-9]|FAILED)|Finished |error:|^error\[|retained-plan-test|a consumed carrier|Some\(Prime\(15\)\)', l)]}
comparison = a.lane / 'oracle-comparison.json'
if comparison.exists():
    report['oracle'] = json.loads(comparison.read_text())
    old = Path('/tmp/qwen-ornith-5090-capacity-20260908/qwen-32k-old-tap-oracle')
    report['greedy_tapes'] = {}
    for cell in ['a-seed', 'a-restore']:
        got = json.loads((a.lane / 'oracle-final' / (cell + '.json')).read_text())['choices']
        expected = json.loads((old / (cell + '.json')).read_text())['choices']
        report['greedy_tapes'][cell] = {'choices_equal': got == expected, 'choices_sha256': hashlib.sha256(json.dumps(got, sort_keys=True).encode()).hexdigest()}
report['source_sha256'] = {name: digest(a.lane / 'source' / name) for name in ['crates/memra-server/src/worker.rs', 'crates/memra-engine/src/dflash.rs']}
report['verdict'] = 'DRAFT: required 128k cold/1024-output plus warm continuation is blocked by cold decode OOM before warm admission; no merge or capacity claim'
a.output.write_text(json.dumps(report, indent=2) + '\n')
