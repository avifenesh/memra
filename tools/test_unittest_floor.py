"""Required execution, outcome shape, and CLI controls for unittest-floor."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

import unittest_floor as floor

ROOT = Path(__file__).resolve().parent.parent


class Admission(unittest.TestCase):
    def evidence(self, **changes):
        data = dict(discovered=2, run=2, passed=2, skipped=0, failures=0, errors=0,
                    expected_failures=0, unexpected_successes=0)
        data.update(changes)
        return data

    def call(self, body, minimum=2, *, optimize='', extra=()):
        with tempfile.TemporaryDirectory(prefix='unittest-admission-') as tmp:
            path = Path(tmp)
            (path / 'test_fixture.py').write_text('import unittest\n' + body)
            env = dict(os.environ, PYTHONOPTIMIZE=optimize, PYTHONDONTWRITEBYTECODE='1')
            result = subprocess.run([str(ROOT / 'tools/unittest-floor.sh'), str(path),
                                     'test_*.py', str(minimum), *extra], env=env,
                                    capture_output=True, text=True, timeout=15)
            return result

    def test_exact_successful_execution(self):
        self.assertTrue(floor.admit(self.evidence(), 2))
        self.assertFalse(floor.admit(self.evidence(), 3))

    def test_missing_unexpected_and_non_object_evidence(self):
        for data in (None, [], {}, self.evidence(extra=0)):
            self.assertFalse(floor.admit(data, 1))

    def test_boolean_negative_string_and_missing_counts(self):
        for name in floor.FIELDS:
            for value in (True, False, -1, '2', None):
                self.assertFalse(floor.admit(self.evidence(**{name: value}), 1))
            data = self.evidence()
            del data[name]
            self.assertFalse(floor.admit(data, 1))

    def test_empty_and_incomplete_execution(self):
        self.assertFalse(floor.admit(self.evidence(discovered=0, run=0, passed=0), 0))
        self.assertFalse(floor.admit(self.evidence(discovered=3), 1))
        self.assertFalse(floor.admit(self.evidence(passed=1), 1))

    def test_all_non_success_outcomes_refuse(self):
        for name in floor.FIELDS - {'discovered', 'run', 'passed'}:
            self.assertFalse(floor.admit(self.evidence(**{name: 1}), 1))

    def test_healthy_cli_extra_arguments_and_import_root(self):
        result = self.call('class A(unittest.TestCase):\n'
                           ' def test_one(self): self.assertTrue(True)\n'
                           ' def test_two(self): self.assertEqual(1,1)\n', extra=('-v',))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('ran 2 tests (floor 2)', result.stdout)

    def test_skipped_required_predicate_refuses(self):
        result = self.call('class A(unittest.TestCase):\n'
                           ' def test_one(self): self.assertTrue(True)\n'
                           ' @unittest.skip("planted")\n'
                           ' def test_two(self): self.fail("not executed")\n')
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('skipped=1', result.stderr)

    def test_expected_failure_refuses(self):
        result = self.call('class A(unittest.TestCase):\n'
                           ' def test_one(self): self.assertTrue(True)\n'
                           ' @unittest.expectedFailure\n'
                           ' def test_two(self): self.fail("planted")\n')
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('expected_failures=1', result.stderr)

    def test_error_and_unexpected_success_refuse(self):
        for body in (' def test_two(self): raise RuntimeError("planted")\n',
                     ' @unittest.expectedFailure\n def test_two(self): pass\n'):
            result = self.call('class A(unittest.TestCase):\n def test_one(self): pass\n' + body)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)

    def test_optimized_refusal_precedes_fixture_import(self):
        result = self.call('raise RuntimeError("fixture imported before refusal")\n', optimize='1')
        self.assertEqual(result.returncode, 1)
        self.assertIn('assertions must be enabled before test imports', result.stderr)
        self.assertNotIn('fixture imported before refusal', result.stderr)

    def test_console_count_forgery_does_not_admit(self):
        result = self.call('import atexit\n'
                           'atexit.register(lambda: print("Ran 999 tests"))\n'
                           'class A(unittest.TestCase):\n def test_one(self): pass\n', minimum=2)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('Ran 999 tests', result.stdout)
        self.assertIn('run=1', result.stderr)

    def test_discovery_import_error_does_not_admit(self):
        result = self.call('raise RuntimeError("planted import failure")\n', minimum=1)
        self.assertEqual(result.returncode, 1)
        self.assertIn('planted import failure', result.stderr)

    def test_selected_suite_after_cli_filter_is_complete(self):
        result = self.call('class A(unittest.TestCase):\n def test_one(self): pass\n'
                           ' def test_two(self): pass\n', minimum=1, extra=('-k', 'one'))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('ran 1 tests (floor 1)', result.stdout)

    def test_actual_early_stop_refuses_even_above_low_floor(self):
        result = self.call('class A(unittest.TestCase):\n def test_one(self): pass\n'
                           ' def test_two(self): pass\n'
                           'class Early(unittest.TestSuite):\n'
                           ' def run(self, result, debug=False):\n'
                           '  self._tests[0].run(result)\n  return result\n'
                           'def load_tests(loader, tests, pattern):\n'
                           ' return Early([A("test_one"), A("test_two")])\n', minimum=1)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('discovered=2', result.stderr)
        self.assertIn('run=1', result.stderr)

    def test_repeating_one_case_cannot_replace_another_selected_case(self):
        result = self.call('class A(unittest.TestCase):\n def test_one(self): pass\n'
                           ' def test_two(self): pass\n'
                           'class Repeat(unittest.TestSuite):\n'
                           ' def run(self, result, debug=False):\n'
                           '  self._tests[0].run(result)\n  self._tests[0].run(result)\n  return result\n'
                           'def load_tests(loader, tests, pattern):\n'
                           ' return Repeat([A("test_one"), A("test_two")])\n')
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('test identities differ', result.stderr)

    def test_help_prints_text_but_never_admits_zero_execution(self):
        for option in ('-h', '--help'):
            result = self.call('raise RuntimeError("help must not import fixture")\n', extra=(option,))
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn('usage:', result.stdout)
            self.assertIn('without execution evidence', result.stderr)
            self.assertNotIn('help must not import fixture', result.stderr)


if __name__ == '__main__':
    unittest.main()
