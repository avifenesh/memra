"""Caller and identity refusal controls; the original nine methods remain unchanged."""

import ast
from contextlib import ExitStack, contextmanager
import io
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import run_sft_gen_contract as runner
import test_sft_gen as original
import validation_plan as vp


ROOT = Path(__file__).resolve().parent.parent
EXPECTED_ORIGINAL_METHODS = {
    'test_battery_has_six_shapes_and_four_task_kinds',
    'test_all_fixture_sources_compile_and_scan_clean',
    'test_secret_scanner_detects_assigned_and_prefixed_keys',
    'test_content_hash_ignores_ids_times_and_usage',
    'test_event_parser_requires_json_objects_and_one_session',
    'test_prepare_workspace_accepts_precreated_empty_directory',
    'test_file_stdout_capture_preserves_large_single_write',
    'test_opencode_config_requires_exact_provider_pin',
    'test_tool_audit_rejects_rust_and_accelerator_commands',
}
CALLER = 'python3 tools/run_sft_gen_contract.py'


def cases():
    return [getattr(sys.modules[module], name)(method)
            for (module, name), methods in runner.GROUPS.items() for method in methods]


@contextmanager
def protocol_fixture():
    # Stub only nested policy fixtures. The top-level run executes the real nine
    # original predicates and real admission predicates with recorded identities.
    with ExitStack() as stack:
        for (module, name), methods in runner.GROUPS.items():
            cls = getattr(sys.modules[module], name)
            stack.enter_context(mock.patch.object(cls, 'setUpClass', classmethod(lambda cls: None)))
            for method in methods:
                stack.enter_context(mock.patch.object(cls, method, lambda case: None))
        yield


def check_caller(text):
    text = '\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('#'))
    job = re.search(r'(?ms)^  gates:\n(.*?)(?=^  [a-z][a-z_-]*:|\Z)', text)
    if not job or '    if: ${{ !cancelled() }}' not in job[1].split('    steps:', 1)[0]:
        raise ValueError('mandatory gates job is masked or missing')
    step = re.search(r'(?ms)^      - name: SFT generator CPU control admission\n'
                     r'(.*?)(?=^      - |\Z)', job[1])
    if not step or re.search(r'(?m)^\s+(?:if|continue-on-error):', step[1]):
        raise ValueError('mandatory SFT caller is masked or missing')
    if [line.strip() for line in step[1].splitlines() if line.strip()] != ['run: ' + CALLER]:
        raise ValueError('SFT caller command changed or failure swallowed')


class SftGenContractTests(unittest.TestCase):
    def test_original_control_inventory(self):
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(original.SftGeneratorTests)
        actual = {case._testMethodName for case in suite}
        self.assertEqual(actual, EXPECTED_ORIGINAL_METHODS)
        self.assertEqual(set(runner.ORIGINAL_METHODS), EXPECTED_ORIGINAL_METHODS)
        self.assertEqual(len(runner.ORIGINAL_METHODS), 9)

    def test_empty_missing_replaced_unrelated_duplicate_discovery(self):
        class Unrelated(unittest.TestCase):
            def runTest(self):
                raise AssertionError('unrelated case must not execute')
        all_cases = cases()
        probes = [('empty', unittest.TestSuite()),
                  ('unrelated', unittest.TestSuite(Unrelated() for _ in range(18))),
                  ('replaced', unittest.TestSuite(all_cases[:-1] + [Unrelated()])),
                  ('duplicate', unittest.TestSuite(all_cases[:-1] + [all_cases[0]])),
                  ('extra-duplicate', unittest.TestSuite(all_cases + [all_cases[0]]))]
        probes.extend(('missing-' + missing.id(), unittest.TestSuite(case for case in cases()
                      if case.id() != missing.id())) for missing in all_cases)
        for label, suite in probes:
            with self.subTest(label=label), protocol_fixture():
                output = io.StringIO()
                self.assertEqual(runner.run(suite, stream=output), 1)
                self.assertIn('FAIL: discovery:', output.getvalue())
                self.assertNotIn('Ran ', output.getvalue())

    def test_missing_and_equal_count_duplicate_execution(self):
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
        for cls in (Empty, Duplicate):
            with self.subTest(kind=cls.__name__), protocol_fixture():
                output = io.StringIO()
                self.assertEqual(runner.run(cls(cases()), stream=output), 1)
                self.assertIn('executed identities', output.getvalue())

    def test_non_successful_results_refuse(self):
        for mode, expected in (('healthy', 0), ('skip', 1), ('failure', 1), ('error', 1),
                               ('expected-failure', 1), ('unexpected-success', 1)):
            with self.subTest(mode=mode), protocol_fixture():
                def control(case):
                    if mode == 'skip':
                        case.skipTest('planted skip')
                    if mode in ('failure', 'expected-failure'):
                        case.fail('planted failure')
                    if mode == 'error':
                        raise RuntimeError('planted error')
                if mode in ('expected-failure', 'unexpected-success'):
                    control = unittest.expectedFailure(control)
                with mock.patch.object(original.SftGeneratorTests, runner.ORIGINAL_METHODS[0], control):
                    self.assertEqual(runner.run(unittest.TestSuite(cases()), stream=io.StringIO()), expected)

    def test_optimization_refuses_before_fixture_and_import(self):
        code = ('import sys;sys.path.insert(0,' + repr(str(ROOT / 'tools')) + ');'
                'import run_sft_gen_contract as r;'
                'r.owned_fixture=lambda: (_ for _ in ()).throw(AssertionError("fixture touched"));'
                'r.discover=lambda: (_ for _ in ()).throw(AssertionError("controls imported"));r.main()')
        for flag, optimize in (('-O', None), ('-OO', None), (None, '1'), (None, '2')):
            env = dict(os.environ)
            env.pop('PYTHONOPTIMIZE', None)
            if optimize:
                env['PYTHONOPTIMIZE'] = optimize
            with self.subTest(flag=flag, optimize=optimize):
                args = [sys.executable] + ([flag] if flag else []) + ['-c', code]
                result = subprocess.run(args, cwd=ROOT, env=env, text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('SFT generator admission requires enabled assertions', result.stderr)
                self.assertNotIn('AssertionError:', result.stderr)

    def test_mandatory_caller_and_masking_refusals(self):
        text = (ROOT / '.github/workflows/ci.yml').read_text()
        try:
            check_caller(text)
        except ValueError as error:
            self.fail(str(error))
        for changed in (text.replace(CALLER, 'true'), text.replace(CALLER, CALLER + ' || true'),
                        text.replace('      - name: SFT generator CPU control admission\n',
                                     '      - name: SFT generator CPU control admission\n        if: false\n'),
                        text.replace('    if: ${{ !cancelled() }}', '    if: false', 1)):
            with self.subTest(change=changed != text):
                with self.assertRaises(ValueError):
                    check_caller(changed)

    def test_original_control_call_scope_and_generation_guard(self):
        tree = ast.parse((ROOT / 'tools/test_sft_gen.py').read_text())
        calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Attribute)
                 and isinstance(node.func.value.value, ast.Name) and node.func.value.value.id == 'self'
                 and node.func.value.attr == 'tool'}
        self.assertEqual(calls, {'build_templates', 'scan_templates', 'scan_text', 'content_hash',
                                'parse_json_events', 'session_id_from_events', 'prepare_workspace',
                                'run_with_file_stdout', 'validate_opencode_config', 'audit_tool_commands'})
        module = original.load_tool()
        runner.block_generation(module)
        for name in ('generate_one', 'opencode_version', 'main'):
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError, 'outside CPU control scope'):
                    getattr(module, name)()

    def test_private_fixture_boundary_and_cleanup(self):
        previous = {key: os.environ.get(key) for key in ('TMPDIR', 'GIT_CONFIG_GLOBAL', 'GIT_CONFIG_COUNT')}
        old_tempdir = tempfile.tempdir
        with runner.owned_fixture() as root:
            self.assertEqual(Path(tempfile.gettempdir()), root)
            self.assertEqual(Path(os.environ['GIT_CONFIG_GLOBAL']), root / 'gitconfig')
            self.assertEqual((root / 'gitconfig').read_text(), '')
            self.assertTrue((root / 'hooks').is_dir())
            self.assertEqual(os.environ['GIT_CONFIG_COUNT'], '1')
            retained = root
        self.assertFalse(retained.exists())
        self.assertEqual(tempfile.tempdir, old_tempdir)
        self.assertEqual({key: os.environ.get(key) for key in previous}, previous)

    def test_generator_selection_remains_full(self):
        tree = vp.Tree(ROOT, 'HEAD')
        for path in ('tools/sft-gen.py', 'tools/test_sft_gen.py'):
            with self.subTest(path=path):
                plan = vp.make_plan([path], tree, tree)
                self.assertEqual(plan['mode'], 'full')
                self.assertTrue(all(plan['jobs'].values()))
                self.assertEqual(plan['native']['scope'], 'full')
                self.assertFalse(plan['native']['qualification'])


if __name__ == '__main__':
    sys.exit(runner.main())
