"""Actual owned reader calls; result rows are synthetic CPU metadata only."""
if not __debug__:
    raise RuntimeError('coverage IO proof assertions must be enabled')
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

root = Path(sys.argv[1]).resolve(); out = Path(sys.argv[2]).resolve(); out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root / 'tools'))
import validation_coverage as current
import test_validation_coverage_input_io as fixtures

baseline = Path(__file__).with_name('original_validation_coverage.py')
spec = importlib.util.spec_from_file_location('original_coverage', baseline)
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
observations = []
for consumer in ('select', 'validate_results'):
    f = fixtures.CoverageInputIO(); f.setUp()
    try:
        real_read = Path.read_bytes; swapped = []
        def swap(path):
            if path == f.source and not swapped:
                path.unlink(); path.symlink_to(f.external); swapped.append(True)
            return real_read(path)
        with mock.patch.object(Path, 'read_bytes', swap):
            if consumer == 'select': result = old.select(['io'], [f.case], f.context, f.root)
            else: result = old.validate_results(f.plan, f.results, f.context, f.root)
        assert swapped == [True] and f.source.resolve() == f.external
        assert result.get('decision', result.get('status')) in ('scoped', 'passed')
        f.source.unlink(); f.source.write_bytes(f.raw); real_open = os.open; swapped = []
        def replacement(name, flags, *args, **kwargs):
            if name == 'harness.py' and not swapped:
                f.source.unlink(); f.source.symlink_to(f.external); swapped.append(True)
            return real_open(name, flags, *args, **kwargs)
        with mock.patch.object(current.os, 'open', side_effect=replacement):
            if consumer == 'select':
                repaired = current.select(['io'], [f.case], f.context, f.root); assert repaired['decision'] == 'expand'
            else:
                try: current.validate_results(f.plan, f.results, f.context, f.root)
                except ValueError: repaired = {'status': 'refused'}
                else: raise AssertionError('escaping source admitted')
        assert swapped == [True]
        observations.append({'consumer': consumer, 'original_actual_swap': True, 'original_outcome': result,
                             'current_actual_swap': True, 'current_outcome': repaired,
                             'synthetic_result_metadata_only': True, 'actual_test_execution': False})
    finally: f.doCleanups()

class ContentObserved(Exception): pass
for consumer in ('manifest', 'results'):
    f = fixtures.CoverageInputIO(); f.setUp()
    try:
        manifest, results = f.documents(); target = manifest if consumer == 'manifest' else results
        target.unlink(); os.mkfifo(target); real_text = Path.read_text; engaged = []
        def observe(path, *args, **kwargs):
            if path == target: engaged.append(True); raise ContentObserved()
            return real_text(path, *args, **kwargs)
        argv = [old.__file__, str(manifest), '--root', str(f.root), '--results', str(results)]
        with mock.patch.object(sys, 'argv', argv), mock.patch.object(Path, 'read_text', observe):
            try: old.main()
            except ContentObserved: pass
            else: raise AssertionError('original CLI did not engage FIFO observer')
        assert engaged == [True]
        with mock.patch.object(sys, 'argv', argv):
            try: current.main()
            except ValueError: pass
            else: raise AssertionError('current CLI engaged FIFO')
        observations.append({'consumer': consumer, 'original_owned_fifo_read_observed': True,
                             'fifo_content_io_prevented_by_observer': True, 'current_outcome': 'refused',
                             'actual_test_execution': False})
    finally: f.doCleanups()
(out / 'actual-before-after.json').write_text(json.dumps(observations, indent=2) + '\n')

source = (root / 'tools/validation_coverage.py').read_text()
prefix = 'test_validation_coverage_input_io.CoverageInputIO.'
mutations = [
    ('remove-prestat', source.replace('if not stat.S_ISREG(info.st_mode):', 'if False:'), 'test_known_fifo_directory_socket_refuse_before_special_open'),
    ('remove-leaf-nofollow', source.replace('os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK', 'os.O_RDONLY | os.O_NONBLOCK'), 'test_leaf_replacement_alias_refuses_both_real_consumers'),
    ('remove-parent-nofollow', source.replace('os.O_DIRECTORY | os.O_NOFOLLOW', 'os.O_DIRECTORY'), 'test_resolved_ancestor_replacement_refuses_opened_escape'),
    ('remove-nonblock', source.replace(' | os.O_NONBLOCK', ''), 'test_leaf_replacement_fifo_is_nonblocking_and_refused'),
    ('remove-opened-type', source.replace('if not stat.S_ISREG(os.fstat(leaf).st_mode):', 'if False:'), 'test_leaf_replacement_fifo_is_nonblocking_and_refused'),
    ('remove-root-bound', source.replace('base = Path(root).resolve() if root is not None else target.parent', 'base = target.parent'), 'test_missing_escaping_and_broken_sources_expand_or_refuse'),
    ('normalize-bytes', source.replace('return source.read()', "return source.read().replace(b'\\r\\n', b'\\n') if binary else source.read()"), 'test_regular_bytes_text_and_context_preserve_semantics'),
    ('leak-parent', source.replace('            os.close(parent)\n\n\ndef contract_digest', '            pass\n\n\ndef contract_digest'), 'test_descriptors_close_on_success_type_and_open_failure'),
]
reds = []
with tempfile.TemporaryDirectory(prefix='memra-coverage-mutations-') as owned:
    for name, changed, test in mutations:
        assert changed != source, name
        path = Path(owned) / (name + '.py'); path.write_text(changed)
        runner = "import importlib.util,sys,unittest;sys.path.insert(0,sys.argv[1]);s=importlib.util.spec_from_file_location('validation_coverage',sys.argv[2]);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(sys.argv[3:]));sys.exit(0 if r.wasSuccessful() else 1)"
        run = subprocess.run([sys.executable, '-c', runner, str(root / 'tools'), str(path), prefix + test], capture_output=True, timeout=15)
        raw = run.stdout + run.stderr; (out / (name + '.log')).write_bytes(raw)
        assert run.returncode == 1 and b'FAIL:' in raw and b'ERROR:' not in raw, (name, raw.decode())
        reds.append({'control': name, 'exit': run.returncode, 'source_sha256': hashlib.sha256(changed.encode()).hexdigest(),
                     'raw_sha256': hashlib.sha256(raw).hexdigest(), 'assertion_method': test})
(out / 'coherent-reds.json').write_text(json.dumps(reds, indent=2) + '\n')

required = (
    'test_cli_fifo_documents_refuse_without_blocking_normal_and_optimized',
    'test_cli_open_replacement_fifo_refuses_both_document_readers',
    'test_contained_leaf_parent_and_absolute_source_aliases_remain_valid',
    'test_descriptors_close_on_success_type_and_open_failure',
    'test_external_cli_documents_and_aliases_work_normal_and_optimized',
    'test_known_fifo_directory_socket_refuse_before_special_open',
    'test_leaf_replacement_alias_refuses_both_real_consumers',
    'test_leaf_replacement_fifo_is_nonblocking_and_refused',
    'test_missing_escaping_and_broken_sources_expand_or_refuse',
    'test_opened_parent_keeps_original_bytes_after_path_replacement',
    'test_regular_bytes_text_and_context_preserve_semantics',
    'test_resolved_ancestor_replacement_refuses_opened_escape',
)
assert tuple(unittest.defaultTestLoader.getTestCaseNames(fixtures.CoverageInputIO)) == required
results = []
for name in required:
    buffer = io.StringIO(); result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(unittest.TestSuite([fixtures.CoverageInputIO(name)]))
    assert result.wasSuccessful() and result.testsRun == 1 and not (result.skipped or result.expectedFailures or result.unexpectedSuccesses), name
    raw = buffer.getvalue().encode(); (out / (name + '.log')).write_bytes(raw)
    results.append({'method': name, 'executed': 1, 'skipped': 0, 'expected_failures': 0, 'unexpected_successes': 0,
                    'raw_sha256': hashlib.sha256(raw).hexdigest()})
pins = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [root/'tools/validation_coverage.py', root/'tools/test_validation_coverage_input_io.py', root/'tools/test_validation_coverage.py', Path(__file__).resolve(), baseline]}
(out / 'source-bound-results.json').write_text(json.dumps({'inputs': pins, 'results': results,
      'original_source_sha256': hashlib.sha256(baseline.read_bytes()).hexdigest(), 'before_after_consumers': len(observations),
      'coherent_red_groups': len(reds), 'red_fixture_errors': 0, 'owned_roots_removed': True,
      'qualification': False, 'scope': 'CPU input reader behavior; synthetic row metadata is not test/native execution'}, indent=2) + '\n')
print(json.dumps({'actual_methods': len(results), 'before_after_consumers': len(observations), 'coherent_red_groups': len(reds), 'red_fixture_errors': 0, 'qualification': False}))
