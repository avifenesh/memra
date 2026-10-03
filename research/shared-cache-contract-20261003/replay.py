#!/usr/bin/env python3
"""CPU-only proof of actual collector imports and declared planner obligations."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BASE = '75bac45b8acc0cb96190689a2e11a882310ebdbd'
NAMES = ('background-chat-text', 'serving-qualification', 'sampled-mtp')
sys.path.insert(0, str(ROOT / 'tools'))
import validation_plan as current

READER = '''import importlib.util,json,sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
def load(name):
    s=importlib.util.spec_from_file_location(name,root/(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
serving=load('collect-serving-qualification')
sampled=load('collect-sampled-mtp')
print(json.dumps([serving.server_capture_len(241),sampled.base.server_capture_len(241)]))
'''


def replay():
    with tempfile.TemporaryDirectory(prefix='memra-shared-cache-proof-') as directory:
        scratch = Path(directory)
        hashes = {}
        for name in ('cache_qualification.py', 'collect-serving-qualification.py',
                     'collect-sampled-mtp.py', 'skip-census.py', 'validation_plan.py',
                     'validation_inputs.json'):
            data = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{BASE}:tools/{name}'])
            (scratch / name).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        before = json.loads(subprocess.check_output([sys.executable, '-B', '-c', READER, directory]))
        helper = scratch / 'cache_qualification.py'
        source = helper.read_text()
        expression = 'return boundary if boundary >= PREFIX_CACHE_MIN_TOKENS else None'
        assert source.count(expression) == 1
        helper.write_text(source.replace(expression, 'return prompt_tokens'))
        after = json.loads(subprocess.check_output([sys.executable, '-B', '-c', READER, directory]))
        assert before == [224, 224] and after == [241, 241]
        spec = importlib.util.spec_from_file_location('baseline_plan', scratch / 'validation_plan.py')
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        tree = current.Tree(ROOT, BASE)
        old = baseline.make_plan(['tools/cache_qualification.py'], tree, tree)
        new = current.make_plan(['tools/cache_qualification.py'], tree, tree)
        assert [c['id'] for c in old['cpu_contracts']] == ['background-chat-text'], old['reason']
        assert {c['id'] for c in new['cpu_contracts']} == set(NAMES)
        assert not any(old['jobs'].values()) and not any(new['jobs'].values())
        assert not old['native']['qualification'] and not new['native']['qualification']
        for name in NAMES:
            for field in ('cpu', 'native', 'python_requirements'):
                assert baseline.TOOL_CONTRACTS[name].get(field) == current.TOOL_CONTRACTS[name].get(field)
        return {'baseline': BASE, 'original_sources_sha256': hashes,
                'candidate_registry_sha256': hashlib.sha256((ROOT / 'tools/validation_plan.py').read_bytes()).hexdigest(),
                'actual_serving_and_transitive_sampled_before': before,
                'actual_serving_and_transitive_sampled_after': after,
                'baseline_contracts': [c['id'] for c in old['cpu_contracts']],
                'candidate_contracts': [c['id'] for c in new['cpu_contracts']],
                'candidate_native_requirements': new['native']['requirements'],
                'unchanged_cpu_commands_and_native_obligations': True,
                'unrelated_compilation_selected': any(new['jobs'].values()),
                'qualification': False}


if __name__ == '__main__':
    print(json.dumps(replay(), indent=2))
