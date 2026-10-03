"""Exercise the actual self-test, its optimization refusal, and its mandatory CI caller."""

import ast
from contextlib import redirect_stdout
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

import validation_plan as vp
import run_expert_tier_contract as runner


ROOT = Path(__file__).resolve().parent.parent
BUILDER = ROOT / 'tools/build_expert_tier_plan.py'
REFUSAL = 'expert tier plan self-test requires enabled assertions'
CASES = ['usage-pyramid', 'uniform-nvfp4', 'reap50-plus25', 'uniform-nvfp4',
         'quartile-prune', 'traffic-ladder', 'traffic-ladder', 'traffic-ladder', 'traffic-ladder']
COMMANDS = ['python3 tools/build_expert_tier_plan.py --self-test',
            'python3 tools/run_expert_tier_contract.py']


def observe(namespace, source):
    """Observe production function calls and assertion lines, with independent floors."""
    function = namespace['self_test']
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == 'self_test')
    expected = {n.lineno for n in ast.walk(node) if isinstance(n, ast.Assert)}
    seen, calls = set(), []
    original = namespace['build_plan']

    def build(args):
        calls.append(args.recipe)
        return original(args)

    def trace(frame, event, arg):
        if frame.f_code is function.__code__ and event == 'line' and frame.f_lineno in expected:
            seen.add(frame.f_lineno)
        return trace

    previous = sys.gettrace()
    namespace['build_plan'] = build
    try:
        sys.settrace(trace)
        with redirect_stdout(io.StringIO()):
            function()
    finally:
        sys.settrace(previous)
        namespace['build_plan'] = original
    if len(expected) < 26 or seen != expected or len(seen) < 26 or calls != CASES:
        raise ValueError('self-test did not execute all 26 assertion sites and nine recipe cases')
    return len(seen), calls


def check_wiring(text):
    text = '\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('#'))
    job = re.search(r'(?ms)^  gates:\n(.*?)(?=^  [a-z][a-z_-]*:|\Z)', text)
    if not job or "    if: ${{ !cancelled() }}" not in job[1].split('    steps:', 1)[0]:
        raise ValueError('mandatory gates job is masked or missing')
    step = re.search(r'(?ms)^      - name: Expert-tier plan self-test assertion admission \(CPU-only\)\n'
                     r'(.*?)(?=^      - |\Z)', job[1])
    if not step or re.search(r'(?m)^\s+(?:if|continue-on-error):', step[1]):
        raise ValueError('expert-tier caller is masked or missing')
    parts = step[1].split('        run: |\n')
    if len(parts) != 2 or [line.strip() for line in parts[1].splitlines() if line.strip()] != COMMANDS:
        raise ValueError('expert-tier caller commands or non-vacuity floor changed')


class ExpertTierContractTests(unittest.TestCase):
    def command(self, args, *, optimize=None):
        env = dict(os.environ)
        env.pop('PYTHONOPTIMIZE', None)
        if optimize is not None:
            env['PYTHONOPTIMIZE'] = optimize
        return subprocess.run([sys.executable, *args], cwd=ROOT, env=env,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)

    def test_ordinary_selftest_cli(self):
        result = self.command([str(BUILDER), '--self-test'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'expert tier plan self-test: PASS')

    def test_real_selftest_executes_nine_cases_and_26_assertions(self):
        namespace = runpy.run_path(str(BUILDER))
        count, cases = observe(namespace['self_test'].__globals__, BUILDER.read_text())
        self.assertGreaterEqual(count, 26)
        self.assertEqual(cases, CASES)

    def test_cli_optimization_refuses(self):
        for flag in ('-O', '-OO'):
            with self.subTest(flag=flag):
                result = self.command([flag, str(BUILDER), '--self-test'])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(REFUSAL, result.stderr)
                self.assertNotIn('PASS', result.stdout)

    def test_environment_optimization_refuses(self):
        for value in ('1', '2'):
            with self.subTest(value=value):
                result = self.command([str(BUILDER), '--self-test'], optimize=value)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(REFUSAL, result.stderr)
                self.assertNotIn('PASS', result.stdout)

    def test_import_optimization_refuses_before_fixture(self):
        code = ('import runpy; m=runpy.run_path(' + repr(str(BUILDER)) + '); '
                'g=m["self_test"].__globals__; '
                'g["tempfile"].TemporaryDirectory=lambda **kw: (_ for _ in ()).throw('
                'AssertionError("fixture entered before refusal")); m["self_test"]()')
        for flag in ('-O', '-OO'):
            with self.subTest(flag=flag):
                result = self.command([flag, '-c', code])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(REFUSAL, result.stderr)
                self.assertNotIn('AssertionError: fixture entered', result.stderr)

    def test_planted_incorrect_result_fails(self):
        namespace = runpy.run_path(str(BUILDER))['self_test'].__globals__
        original = namespace['build_plan']
        with mock.patch.dict(namespace, build_plan=lambda args: dict(original(args), pruned_experts={})):
            with self.assertRaises(AssertionError):
                namespace['self_test']()

    def test_noop_selftest_refuses_observation(self):
        namespace = runpy.run_path(str(BUILDER))['self_test'].__globals__
        with mock.patch.dict(namespace, self_test=lambda: None):
            with self.assertRaisesRegex(ValueError, '26 assertion sites and nine recipe cases'):
                observe(namespace, BUILDER.read_text())

    def test_removed_assertion_refuses_observation(self):
        tree = ast.parse(BUILDER.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'self_test')
        body = next(n for n in function.body if isinstance(n, ast.With)).body
        index = next(i for i, node in enumerate(body) if isinstance(node, ast.Assert))
        del body[index]
        source = ast.unparse(tree)
        namespace = {'__name__': 'assertion_removed_control'}
        exec(compile(source, str(BUILDER), 'exec'), namespace)
        with self.assertRaisesRegex(ValueError, '26 assertion sites and nine recipe cases'):
            observe(namespace, source)

    def test_ci_wiring(self):
        check_wiring((ROOT / '.github/workflows/ci.yml').read_text())

    def test_removed_or_masked_ci_caller_refuses(self):
        text = (ROOT / '.github/workflows/ci.yml').read_text()
        for changed in (text.replace(COMMANDS[0], '# removed caller'),
                        text.replace(COMMANDS[1], 'true'),
                        text.replace(COMMANDS[0], COMMANDS[0] + ' || true'),
                        text.replace('      - name: Expert-tier plan self-test assertion admission (CPU-only)\n',
                                     '      - name: Expert-tier plan self-test assertion admission (CPU-only)\n        if: false\n'),
                        text.replace('    if: ${{ !cancelled() }}', '    if: false', 1)):
            with self.subTest(change=changed != text):
                with self.assertRaises(ValueError):
                    check_wiring(changed)

    def test_ordinary_plan_generation_unchanged_under_optimization(self):
        with tempfile.TemporaryDirectory(prefix='expert-tier-cli-') as folder:
            output = Path(folder) / 'plan.json'
            args = [str(BUILDER), '--recipe', 'uniform-nvfp4', '--expert-count', '4',
                    '--original-expert-count', '4', '--layers', '1', '--top-k', '2', '--out', str(output)]
            normal = self.command(args)
            self.assertEqual(normal.returncode, 0, normal.stderr)
            expected = output.read_bytes()
            self.assertEqual(json.loads(expected)['layer_summary']['1']['nvfp4'], 4)
            for flag in ('-O', '-OO'):
                with self.subTest(flag=flag):
                    optimized = self.command([flag, *args])
                    self.assertEqual(optimized.returncode, 0, optimized.stderr)
                    self.assertEqual(output.read_bytes(), expected)

    def test_builder_change_retains_full_selection(self):
        tree = vp.Tree(ROOT, 'HEAD')
        plan = vp.make_plan(['tools/build_expert_tier_plan.py'], tree, tree)
        self.assertEqual(plan['mode'], 'full')
        self.assertTrue(all(plan['jobs'].values()))
        self.assertEqual(plan['native']['scope'], 'full')
        self.assertFalse(plan['native']['qualification'])

    def test_runner_refuses_empty_short_skipped_failed_and_expected_failures(self):
        for count, skip, fail, expected_failure, expected in (
                (13, False, False, False, 0), (0, False, False, False, 1),
                (12, False, False, False, 1), (13, True, False, False, 1),
                (13, False, True, False, 1), (13, False, True, True, 1),
                (13, False, False, True, 1)):
            with self.subTest(count=count, skip=skip, fail=fail, expected_failure=expected_failure):
                class Case(unittest.TestCase):
                    def runTest(self):
                        if skip:
                            self.skipTest('planted skip')
                        if fail:
                            self.fail('planted assertion failure')
                if expected_failure:
                    Case.runTest = unittest.expectedFailure(Case.runTest)
                self.assertEqual(runner.run(unittest.TestSuite(Case() for _ in range(count)),
                                            stream=io.StringIO()), expected)


if __name__ == '__main__':
    unittest.main()
