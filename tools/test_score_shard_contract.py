"""Exercise the original score-shard assertions and the mandatory CPU admission."""

import ast
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import run_score_shard_contract as runner
import validation_plan as vp


ROOT = Path(__file__).resolve().parent.parent
MERGER = ROOT / 'tools/merge_expert_score_shards.py'
REFUSAL = 'expert score shard self-test requires enabled assertions'
PREDICATES = ('result["model"]["moe_layers"] == [1, 2]',
              'len(result["scores"]) == 4 and len(result["shards"]) == 2')
COMMANDS = ('python3 tools/merge_expert_score_shards.py --self-test',
            'python3 tools/run_score_shard_contract.py')
STEP = '      - name: Score-shard self-test assertion admission (CPU-only)\n'


def observe(namespace, source):
    """Require the original predicates and real merge call to run in this function."""
    if not __debug__:
        raise RuntimeError('score-shard observation requires enabled assertions')
    function = namespace['self_test']
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == 'self_test')
    assertions = sorted((n for n in ast.walk(node) if isinstance(n, ast.Assert)),
                        key=lambda n: n.lineno)
    expected = [ast.dump(ast.parse(p, mode='eval').body) for p in PREDICATES]
    if [ast.dump(n.test) for n in assertions] != expected:
        raise ValueError('self-test must retain both original assertion predicates')
    lines = {n.lineno for n in assertions}
    seen, calls = set(), []
    original = namespace['merge']

    def merge(paths):
        result = original(paths)
        calls.append([path.name for path in paths])
        return result

    def trace(frame, event, arg):
        if frame.f_code is function.__code__ and event == 'line' and frame.f_lineno in lines:
            seen.add(frame.f_lineno)
        return trace

    previous = sys.gettrace()
    namespace['merge'] = merge
    try:
        sys.settrace(trace)
        function()
    finally:
        sys.settrace(previous)
        namespace['merge'] = original
    if seen != lines or calls != [['score-1.json', 'score-2.json']]:
        raise ValueError('self-test did not execute both original assertions and real merge')
    return len(seen), calls


def check_caller(text):
    text = '\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('#'))
    job = re.search(r'(?ms)^  gates:\n(.*?)(?=^  [a-z][a-z_-]*:|\Z)', text)
    header = job[1].split('    steps:', 1)[0] if job else ''
    if ('    if: ${{ !cancelled() }}' not in header
            or re.search(r'(?m)^\s+continue-on-error:', header)):
        raise ValueError('mandatory gates job is masked or missing')
    if text.count(STEP) != 1:
        raise ValueError('mandatory score-shard caller is missing or duplicated')
    step = re.search(re.escape(STEP) + r'(.*?)(?=^      - |\Z)', job[1], re.S | re.M)
    required = ['run: |', *COMMANDS]
    if not step or [line.strip() for line in step[1].splitlines() if line.strip()] != required:
        raise ValueError('mandatory score-shard caller changed or failure swallowed')


def namespace():
    return runpy.run_path(str(MERGER))['self_test'].__globals__


def cases():
    cls = sys.modules[runner.MODULE].ScoreShardContractTests
    return [cls(name) for name in runner.REQUIRED_METHODS]


class ScoreShardContractTests(unittest.TestCase):
    def command(self, args, *, optimize=None):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        env.pop('PYTHONOPTIMIZE', None)
        if optimize is not None:
            env['PYTHONOPTIMIZE'] = optimize
        return subprocess.run([sys.executable, *args], cwd=ROOT, env=env,
                              text=True, capture_output=True, timeout=15)

    def test_ordinary_selftest_cli(self):
        result = self.command([str(MERGER), '--self-test'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'expert score shard merge self-test: PASS')

    def test_original_two_predicates_execute_with_real_merge(self):
        try:
            count, calls = observe(namespace(), MERGER.read_text())
        except ValueError as error:
            self.fail(str(error))
        self.assertEqual(count, 2)
        self.assertEqual(calls, [['score-1.json', 'score-2.json']])

    def test_wrong_merge_each_original_predicate_fails_and_restores(self):
        ns = namespace()
        original = ns['merge']
        for field in ('layers', 'scores', 'shards'):
            with self.subTest(field=field):
                def wrong(paths):
                    result = original(paths)
                    if field == 'layers':
                        result['model']['moe_layers'] = []
                    else:
                        result[field] = []
                    return result
                # Each wrong result is coherent and the ordinary merge executes first.
                with mock.patch.dict(ns, merge=wrong):
                    with self.assertRaises(AssertionError):
                        ns['self_test']()
                self.assertEqual(observe(ns, MERGER.read_text())[0], 2)

    def test_noop_selftest_refuses(self):
        ns = namespace()
        with mock.patch.dict(ns, self_test=lambda: None):
            with self.assertRaisesRegex(ValueError, 'did not execute both'):
                observe(ns, MERGER.read_text())

    def test_missing_or_replaced_assertion_refuses(self):
        for index in (0, 1):
            for replace in (False, True):
                with self.subTest(index=index, replace=replace):
                    tree = ast.parse(MERGER.read_text())
                    node = next(n for n in tree.body
                                if isinstance(n, ast.FunctionDef) and n.name == 'self_test')
                    body = next(n for n in node.body if isinstance(n, ast.With)).body
                    positions = [i for i, n in enumerate(body) if isinstance(n, ast.Assert)]
                    if replace:
                        body[positions[index]].test = ast.Constant(True)
                    else:
                        del body[positions[index]]
                    source = ast.unparse(tree)
                    ns = {'__name__': 'score_assertion_control'}
                    exec(compile(source, str(MERGER), 'exec'), ns)
                    with self.assertRaisesRegex(ValueError, 'both original assertion predicates'):
                        observe(ns, source)

    def test_optimization_refuses_before_fixture(self):
        for script, function, refusal in (
                (MERGER, 'self_test', REFUSAL),
                (ROOT / 'tools/run_score_shard_contract.py', 'main',
                 'score-shard admission controls require enabled assertions')):
            code = (f'import runpy,tempfile,unittest; m=runpy.run_path({str(script)!r}); '
                    'tempfile.TemporaryDirectory=lambda **kw: (_ for _ in ()).throw('
                    'RuntimeError("FIXTURE_ENTERED")); '
                    'unittest.defaultTestLoader.discover=lambda *a,**kw: (_ for _ in ()).throw('
                    f'RuntimeError("DISCOVERY_ENTERED")); m[{function!r}]()')
            args = ['--self-test'] if script == MERGER else []
            for flag in ('-O', '-OO'):
                with self.subTest(script=script.name, flag=flag):
                    for invocation in ([flag, str(script), *args], [flag, '-c', code]):
                        result = self.command(invocation)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn(refusal, result.stderr)
                        self.assertNotIn('PASS', result.stdout)
                        self.assertEqual(result.stderr.splitlines()[-1], 'RuntimeError: ' + refusal)
            for value in ('1', '2'):
                with self.subTest(script=script.name, env=value):
                    result = self.command([str(script), *args], optimize=value)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(refusal, result.stderr)

    def test_mandatory_ci_caller(self):
        try:
            check_caller((ROOT / '.github/workflows/ci.yml').read_text())
        except ValueError as error:
            self.fail(str(error))

    def test_removed_masked_swallowed_and_optimized_caller_refuses(self):
        text = (ROOT / '.github/workflows/ci.yml').read_text()
        changes = [text.replace(command, replacement) for command in COMMANDS
                   for replacement in ('true', command + ' || true', command.replace('python3', 'python3 -O'))]
        changes += [text.replace(STEP, STEP + line) for line in
                    ('        if: false\n', '        continue-on-error: true\n')]
        changes += [text.replace('    if: ${{ !cancelled() }}', '    if: false', 1),
                    text.replace('  gates:\n', '  gates:\n    continue-on-error: true\n', 1),
                    text.replace(STEP, '# removed step\n')]
        for changed in changes:
            with self.subTest(change=changed != text):
                self.assertNotEqual(changed, text)
                with self.assertRaises(ValueError):
                    check_caller(changed)

    def test_ordinary_merge_cli_bytes_unchanged_under_optimization(self):
        with tempfile.TemporaryDirectory(prefix='score-shard-cli-') as folder:
            root = Path(folder)
            paths, rows, receipts = [], [], []
            for layer in (2, 1):
                path = root / f'score-{layer}.json'
                shard = {'format': 'memra-expert-retention-scores-v1',
                         'model': {'moe_layers': [layer], 'complete_moe_layers': [1, 2], 'expert_count': 2},
                         'rank_metric': 'fixture', 'policy': {'x': 1}, 'calibration': {'private': True},
                         'source': {'sha256': 'fixture'}, 'teacher_targets': {},
                         'scores': [{'layer': layer, 'expert': expert, 'retain_score': layer + expert}
                                    for expert in (1, 0)]}
                path.write_text(json.dumps(shard))
                paths.append(path)
                rows.extend(shard['scores'])
                receipts.append({'path': str(path.resolve()),
                                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'layers': [layer]})
            expected = dict(shard)
            expected['model'] = dict(shard['model'], moe_layers=[1, 2])
            expected['scores'] = sorted(rows, key=lambda row: (row['layer'], row['expert']))
            expected['shards'] = receipts
            expected_bytes = (json.dumps(expected, indent=2, sort_keys=True) + '\n').encode()
            output = root / 'output.json'
            args = [str(MERGER), '--out', str(output)]
            for path in paths:
                args += ['--shard', str(path)]
            for flags in ([], ['-O'], ['-OO']):
                with self.subTest(flags=flags):
                    result = self.command([*flags, *args])
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(output.read_bytes(), expected_bytes)

    def test_merger_change_retains_full_native_selection(self):
        tree = vp.Tree(ROOT, 'HEAD')
        plan = vp.make_plan(['tools/merge_expert_score_shards.py'], tree, tree)
        self.assertEqual(plan['mode'], 'full')
        self.assertTrue(all(plan['jobs'].values()))
        self.assertEqual(plan['native']['scope'], 'full')
        self.assertFalse(plan['native']['qualification'])

    def test_exact_control_inventory(self):
        names = unittest.defaultTestLoader.getTestCaseNames(type(self))
        self.assertEqual(set(names), set(runner.REQUIRED_METHODS))
        self.assertEqual(len(names), 14)

    def test_missing_replaced_unrelated_duplicate_discovery_refuses(self):
        class Other(unittest.TestCase):
            def runTest(self):
                raise AssertionError('unrelated must not run')
        probes = [[], cases()[:-1] + [Other()], cases()[:-1] + [cases()[0]],
                  cases() + [cases()[0]], [Other() for _ in range(14)]]
        probes += [[case for case in cases() if case.id() != missing.id()] for missing in cases()]
        with ExitStack() as stack:
            for name in runner.REQUIRED_METHODS:
                stack.enter_context(mock.patch.object(type(self), name, lambda case: None))
            for probe in probes:
                output = io.StringIO()
                self.assertEqual(runner.run(unittest.TestSuite(probe), stream=output), 1)
                self.assertIn('FAIL: discovery:', output.getvalue())
                self.assertNotIn('Ran ', output.getvalue())

    def test_missing_duplicate_or_no_success_execution_refuses(self):
        class Empty(unittest.TestSuite):
            def run(self, result, debug=False):
                return result
        class Duplicate(unittest.TestSuite):
            def run(self, result, debug=False):
                self._tests[0](result)
                self._tests[0](result)
                for case in self._tests[2:]:
                    case(result)
                return result
        class NoOutcome(unittest.TestSuite):
            def run(self, result, debug=False):
                for case in self:
                    result.startTest(case)
                    result.stopTest(case)
                return result
        class FloatCount(unittest.TestSuite):
            def run(self, result, debug=False):
                super().run(result, debug)
                result.testsRun = float(result.testsRun)
                return result
        class Malformed(unittest.TestSuite):
            def run(self, result, debug=False):
                super().run(result, debug)
                result.skipped = None
                return result
        with ExitStack() as stack:
            for name in runner.REQUIRED_METHODS:
                stack.enter_context(mock.patch.object(type(self), name, lambda case: None))
            for cls in (Empty, Duplicate, NoOutcome, FloatCount, Malformed):
                with self.subTest(suite=cls.__name__):
                    self.assertEqual(runner.run(cls(cases()), stream=io.StringIO()), 1)

    def test_skipped_failed_expected_failure_and_unexpected_success_refuse(self):
        for mode, expected in (('healthy', 0), ('skip', 1), ('failure', 1), ('error', 1),
                               ('expected-failure', 1), ('unexpected-success', 1)):
            with self.subTest(mode=mode), ExitStack() as stack:
                for name in runner.REQUIRED_METHODS:
                    stack.enter_context(mock.patch.object(type(self), name, lambda case: None))
                def control(case):
                    if mode == 'skip':
                        case.skipTest('planted skip')
                    if mode in ('failure', 'expected-failure'):
                        case.fail('planted failure')
                    if mode == 'error':
                        raise RuntimeError('planted error')
                if mode in ('expected-failure', 'unexpected-success'):
                    control = unittest.expectedFailure(control)
                stack.enter_context(mock.patch.object(type(self), runner.REQUIRED_METHODS[0], control))
                self.assertEqual(runner.run(unittest.TestSuite(cases()), stream=io.StringIO()), expected)


if __name__ == '__main__':
    sys.exit(runner.main())
