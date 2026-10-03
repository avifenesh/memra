"""Run source-bound CPU controls and admit their mandatory behavioral coverage."""
if not __debug__:
    raise RuntimeError('planning IO proof assertions must be enabled')
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=True)
here = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'tools'))
import test_validation_local_input_io as suite
import validation_coverage as coverage

names = [
    'test_descriptor_cleanup_on_success_and_failure',
    'test_fifo_and_directory_readers_refuse_before_content',
    'test_leaf_and_parent_links_are_refused_before_reading_target',
    'test_missing_and_noncanonical_paths_remain_refusals',
    'test_opened_parent_stays_anchored_when_its_path_is_replaced',
    'test_own_registry_fifo_refuses_real_include_reader',
    'test_real_consumers_refuse_leaf_aliases_and_replacement_races',
    'test_real_descendant_consumers_refuse_ancestor_aliases_and_anchor_races',
    'test_regular_text_and_raw_bytes_preserve_original_semantics',
    'test_root_cargo_fifo_expands_actual_planning_before_read',
    'test_snapshot_reads_remain_git_bytes_not_local_filesystem',
    'test_stat_to_fifo_replacement_is_nonblocking_and_refused',
    'test_stat_to_symlink_replacement_refuses_actual_target_content',
    'test_tracked_census_reader_fifo_refuses_raw_byte_adapter',
    'test_unrelated_cargo_manifest_fifo_refuses_command_planning',
]
assert unittest.defaultTestLoader.getTestCaseNames(suite.LocalInputIO) == names
inputs = ['tools/validation_plan.py', 'tools/support_record_inputs.py',
          'tools/validation_inputs.json', 'tools/test_validation_local_input_io.py',
          'tools/test_validation_plan.py', 'tools/validation_coverage.py', 'tools/skip-census.py']
inputs += [p.relative_to(root).as_posix() for p in here.glob('*.py')]
pins = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in inputs}
context = {'lane': 'local-planning-io-972', 'program': 'CPU planning input reads',
           'source': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD']).decode().strip()}
tests = []; results = {}; edges = []
for name in names:
    buffer = io.StringIO()
    result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(unittest.TestSuite([suite.LocalInputIO(name)]))
    (out / (name + '.log')).write_text(buffer.getvalue())
    assert result.wasSuccessful() and result.testsRun == 1 and not result.skipped, name
    edge = 'input/' + name
    edges.append(edge)
    tests.append({'id': name, 'cost': 1, 'covers': [edge], 'inputs': pins, 'scope': context, 'mandatory': True})
    results[name] = {'status': 'passed', 'context': context, 'executed': 1, 'skipped': 0, 'edges': {edge: 'passed'}}

for helper, receipt, expected in (
        ('coherent_controls.py', 'coherent-controls.json', 7),
        ('consumer_controls.py', 'executed-consumers.json', 4),
        ('cli_controls.py', 'actual-cli-controls.json', 12)):
    run = subprocess.run([sys.executable, str(here / helper), str(root), str(out)], capture_output=True, timeout=120)
    (out / (helper + '.log')).write_bytes(run.stdout + run.stderr)
    assert run.returncode == 0, (helper, run.stderr.decode())
    data = json.loads((out / receipt).read_text())
    rows = data if isinstance(data, list) else data.get('controls', data.get('cases'))
    assert len(rows) == expected, helper
    if helper == 'consumer_controls.py':
        aliases = json.loads((out / 'executed-alias-consumers.json').read_text())
        assert aliases['owned_roots_removed'] and len(aliases['controls']) == 6
        rows += aliases['controls']; expected += 6
    group_edges = []
    for index, row in enumerate(rows):
        if helper == 'coherent_controls.py':
            raw = (out / (row['control'] + '.log')).read_bytes()
            assert row['exit'] == 1 and hashlib.sha256(raw).hexdigest() == row['log_sha256']
            assert b'FAIL:' in raw and b'ERROR:' not in raw
        elif helper == 'consumer_controls.py':
            if 'alias' in row:
                assert row['original_observed_read'] in ('text', 'bytes')
                assert row['current_outcome'] in ('full planning expansion', 'refused before content')
            else:
                assert row['actual_fifo'] and row['no_fifo_io'] and row['old_observed_read'] in ('text', 'bytes')
        else:
            if row['case'].startswith('regular-dry-'): assert row['baseline_identical']
            if row['case'] == 'unknown-reader-full': assert row['baseline_obligations_identical']
        edge = helper + '/' + str(index)
        group_edges.append(edge)
    edges += group_edges
    tests.append({'id': helper, 'cost': 1, 'covers': group_edges, 'inputs': pins, 'scope': context, 'mandatory': True})
    results[helper] = {'status': 'passed', 'context': context, 'executed': expected, 'skipped': 0,
                       'edges': {edge: 'passed' for edge in group_edges}}

plan = coverage.select(edges, tests, context, root)
for result in results.values(): result['contract_id'] = plan['contract_id']
verdict = coverage.validate_results(plan, results, context, root)
negatives = []
for kind in ('missing-edge', 'skipped-mandatory'):
    bad = copy.deepcopy(results)
    if kind == 'missing-edge': bad[names[0]]['edges'] = {}
    else: bad[names[0]]['skipped'] = 1
    try: coverage.validate_results(plan, bad, context, root)
    except ValueError as error: negatives.append({'control': kind, 'refused': str(error)})
    else: raise AssertionError(kind)
for name, value in [('source-pins.json', pins), ('coverage-plan.json', plan),
                    ('coverage-results.json', results), ('coverage-verdict.json', verdict),
                    ('coverage-negative.json', negatives)]:
    (out / name).write_text(json.dumps(value, indent=2) + '\n')
print(json.dumps({'test_methods': len(names), 'admitted_edges': len(edges), 'verdict': verdict,
                  'admission_negatives': len(negatives), 'native_qualification': False}))
