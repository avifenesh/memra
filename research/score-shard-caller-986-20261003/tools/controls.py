"""Real self-test before/wrong/restore and focused compiling admission mutants."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

if not __debug__:
    raise RuntimeError('score-shard proof requires enabled assertions')
root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=True)
merger = root / 'tools/merge_expert_score_shards.py'
runner = root / 'tools/run_score_shard_contract.py'
original = Path(__file__).with_name('original_merge_expert_score_shards.py')
original_sha = 'dcdefb0b1a0337c1083dfb9512725c5d232e130e9d895301f5e724d9112a6d95'
assert hashlib.sha256(original.read_bytes()).hexdigest() == original_sha
base, current = original.read_text(), merger.read_text()
def functions(source):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name != 'self_test'}
assert functions(base) == functions(current), 'ordinary functions changed'
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
env.pop('PYTHONOPTIMIZE', None)
rows = []
def command(name, path, optimized=False):
    result = subprocess.run([sys.executable, *(['-O'] if optimized else []), str(path), '--self-test'],
                            capture_output=True, env=env, timeout=15)
    raw = result.stdout + result.stderr
    (out / (name + '.log')).write_bytes(raw)
    row = {'control': name, 'exit': result.returncode, 'optimized': optimized,
           'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
           'raw_sha256': hashlib.sha256(raw).hexdigest()}
    rows.append(row)
    return result, raw
with tempfile.TemporaryDirectory(prefix='score-shard-proof-') as owned:
    folder = Path(owned)
    for version, source in (('original50f', base), ('current', current)):
        path = folder / (version + '.py')
        path.write_text(source)
        result, raw = command(version + '-before', path)
        assert result.returncode == 0 and b'self-test: PASS' in raw
        for field in ('layers', 'scores', 'shards'):
            expression = 'result["model"]["moe_layers"]' if field == 'layers' else f'result["{field}"]'
            changed = source.replace('    return result\n', f'    {expression} = []\n    return result\n', 1)
            assert changed != source
            compile(changed, str(path), 'exec')
            path.write_text(changed)
            result, raw = command(version + '-wrong-' + field, path)
            assert result.returncode == 1 and b'AssertionError' in raw and b'self-test: PASS' not in raw
            result, raw = command(version + '-wrong-' + field + '-optimized', path, True)
            expected = 0 if version == 'original50f' else 1
            assert result.returncode == expected
            if expected:
                assert b'requires enabled assertions' in raw and b'self-test: PASS' not in raw
        path.write_text(source)
        result, raw = command(version + '-restored', path)
        assert result.returncode == 0 and b'self-test: PASS' in raw
    (out / 'actual-before-wrong-restore.json').write_text(json.dumps(rows, indent=2) + '\n')

    source = (root / 'tools/test_score_shard_contract.py').read_text()
    runner_source = runner.read_text()
    tree = ast.parse(current)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'self_test')
    function.body = [ast.Pass()]
    noop = ast.unparse(ast.fix_missing_locations(tree))
    tree = ast.parse(current)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'self_test')
    body = next(n for n in function.body if isinstance(n, ast.With)).body
    del body[next(i for i, n in enumerate(body) if isinstance(n, ast.Assert))]
    missing = ast.unparse(tree)
    mutations = [
        ('noop-selftest', 'merger', noop, 'test_original_two_predicates_execute_with_real_merge'),
        ('remove-original-assertion', 'merger', missing, 'test_original_two_predicates_execute_with_real_merge'),
        ('disable-observation', 'test', source.replace("if seen != lines or calls != [['score-1.json', 'score-2.json']]:", 'if False:'),
         'test_noop_selftest_refuses'),
        ('ignore-predicate-identity', 'test', source.replace('if [ast.dump(n.test) for n in assertions] != expected:', 'if False:'),
         'test_missing_or_replaced_assertion_refuses'),
        ('disabled-optimization-refusal', 'merger', current.replace('    if not __debug__:', '    if False:', 1),
         'test_optimization_refuses_before_fixture'),
        ('remove-caller', 'workflow', (root / '.github/workflows/ci.yml').read_text().replace(
         'python3 tools/run_score_shard_contract.py', 'true'), 'test_mandatory_ci_caller'),
        ('mask-caller', 'workflow', (root / '.github/workflows/ci.yml').read_text().replace(
         '      - name: Score-shard self-test assertion admission (CPU-only)\n',
         '      - name: Score-shard self-test assertion admission (CPU-only)\n        if: false\n'),
         'test_mandatory_ci_caller'),
        ('ignore-discovery', 'runner', runner_source.replace('        discovered = discovered_ids(suite)',
          '        discovered = sorted(REQUIRED_IDS)'), 'test_missing_replaced_unrelated_duplicate_discovery_refuses'),
        ('ignore-execution', 'runner', runner_source.replace('        return 1\n    fields =',
          '        pass\n    fields =').replace('        return 1\n    original =', '        pass\n    original ='),
         'test_missing_duplicate_or_no_success_execution_refuses'),
        ('ignore-outcome', 'runner', runner_source.replace('        return 1\n    original =', '        pass\n    original ='),
         'test_skipped_failed_expected_failure_and_unexpected_success_refuse'),
    ]
    reds = []
    for name, kind, changed, method in mutations:
        originals = {'test': source, 'runner': runner_source, 'merger': current,
                     'workflow': (root / '.github/workflows/ci.yml').read_text()}
        assert changed != originals[kind], name
        path = folder / (name + ('.yml' if kind == 'workflow' else '.py'))
        path.write_text(changed)
        if kind != 'workflow':
            compile(changed, str(path), 'exec')
        code = '''import importlib.util,pathlib,sys,unittest
sys.path.insert(0,sys.argv[1])
import test_score_shard_contract as fixtures
kind,path,method=sys.argv[2:]
if kind in ('test','runner'):
    name='test_score_shard_contract' if kind=='test' else 'run_score_shard_contract'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    if kind=='test':
        fixtures=module
        fixtures.ROOT=pathlib.Path(sys.argv[1]).parent
        fixtures.MERGER=fixtures.ROOT/'tools/merge_expert_score_shards.py'
    else: fixtures.runner=module
elif kind=='merger': fixtures.MERGER=pathlib.Path(path)
else:
    class Root:
        def __truediv__(self,child):
            if child=='.github/workflows/ci.yml': return pathlib.Path(path)
            raise ValueError('unexpected fixture path')
    fixtures.ROOT=Root()
result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([fixtures.ScoreShardContractTests(method)]))
sys.exit(0 if result.wasSuccessful() else 1)
'''
        result = subprocess.run([sys.executable, '-c', code, str(root / 'tools'), kind, str(path), method],
                                env=env, capture_output=True, timeout=20)
        raw = result.stdout + result.stderr
        (out / (name + '.log')).write_bytes(raw)
        assert result.returncode == 1 and b'FAIL:' in raw and b'ERROR:' not in raw, (name, raw.decode())
        reds.append({'control': name, 'method': method, 'exit': result.returncode,
                     'source_sha256': hashlib.sha256(changed.encode()).hexdigest(),
                     'raw_sha256': hashlib.sha256(raw).hexdigest()})
    (out / 'coherent-reds.json').write_text(json.dumps(reds, indent=2) + '\n')
pins = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (merger, runner, root/'tools/test_score_shard_contract.py',
                     root/'.github/workflows/ci.yml', original, Path(__file__).resolve())}
(out / 'source-bound-results.json').write_text(json.dumps({'source_pins': pins,
      'ordinary_functions_ast_equal': True, 'original_sha256': original_sha,
      'before_wrong_restore_runs': len(rows), 'coherent_red_groups': len(reds),
      'red_fixture_errors': 0, 'owned_roots_removed': True, 'qualification': False,
      'scope': 'CPU self-test/caller admission only; no model scoring or native execution'}, indent=2) + '\n')
print(json.dumps({'before_wrong_restore_runs': len(rows), 'coherent_red_groups': len(reds),
                  'red_fixture_errors': 0, 'ordinary_functions_ast_equal': True, 'qualification': False}))
