#!/usr/bin/env python3
"""CPU controls for native caller-runner refusal and owned-child cleanup."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('qualify_callers', Path(__file__).with_name('qualify-callers.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class CallerRunnerTests(unittest.TestCase):
    def test_complete_schedule_preserves_scope_and_requires_each_native_case(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, out, owned = base / 'repo', base / 'evidence', base / 'owned'
            root.mkdir()
            owned.mkdir()
            record = {'tests': {key: {'artifact': {'path': f'test-target/release/deps/{key}'}}
                                for key in ('worker', 'repack', 'gemma-prime')}, 'binaries': {}}
            launched, controlled = [], []

            def fake(command, environment, target, _timeout):
                target.mkdir(parents=True)
                launched.append((command, environment))
                if 'inspect' in command:
                    bundle = Path(command[-1])
                    bundle.mkdir()
                    (bundle / 'artifact.lock').write_text('synthetic scheduling fixture')
                    text = 'CPU inspect fixture\n'
                elif command[1] == 'retained-capture':
                    text = 'RETAINED_CAPTURE_PASS\n'
                elif Path(command[0]).name == 'worker':
                    text = 'NATIVE_WORKER_BOUNDARY_PASS\ntest result: ok. 1 passed; 0 failed; 0 ignored;\n'
                elif Path(command[0]).name == 'repack':
                    text = 'test result: ok. 1 passed; 0 failed; 0 ignored;\n'
                else:
                    text = 'RETAINED_CALLER_PASS\n'
                (target / 'stdout.log').write_text(text)
                (target / 'stderr.log').write_text('')
                return 0

            def control(command, target, timeout, *, environment, check_invariants):
                check_invariants()
                controlled.append(command)
                return {'controller_returncode': fake(command, environment, target, timeout)}

            args = SimpleNamespace(phase='callers', model=base / 'model', out=out,
                                   build_record=owned / 'build.json', timeout_seconds=5)
            with patch.object(runner, 'ROOT', root), \
                    patch.object(runner.build, 'verify_build_record', return_value=(record, 'a' * 64)), \
                    patch.object(runner.admission, 'verify_source', return_value={'fixture': True}), \
                    patch.object(runner.admission, 'verify_lease', return_value={'requested_uuids': ['owned']}), \
                    patch.object(runner, 'plain', side_effect=fake), \
                    patch.object(runner.controller, 'run_controlled', side_effect=control), \
                    patch.object(runner.subprocess, 'check_output', return_value='gpu_uuid,pid\n'), \
                    patch.object(runner.subprocess, 'Popen'), \
                    patch.dict(os.environ, {'MEMRA_GPU_LEASE_FILE': 'fixture'}), patch('builtins.print'):
                runner.run(args)
            self.assertEqual(len(launched), 26)  # one inspect, one capture, 24 independent cases
            self.assertEqual(len(controlled), 11)  # five graph/prime plus six worker env cases
            cases = json.loads((out / 'cases.json').read_text())
            self.assertTrue(all(case['passed'] for case in cases))
            self.assertEqual(sum(case['kind'] == 'cpu-inspect' for case in cases), 1)
            workers = [(command, env) for command, env in launched if Path(command[0]).name == 'worker']
            self.assertEqual({(env['REWRITE_PROBE_CASE'], env['REWRITE_PROBE_DRIFT']) for _, env in workers},
                             {(case, drift) for case in runner.WORKER_CASES for drift in ('library', 'env')})
            self.assertTrue(all('MEMRA_REWRITE_BUNDLE' not in env for _, env in workers))
            self.assertEqual(json.loads((out / 'result.json').read_text())['serving_binary_qualification'], False)

    def test_native_unit_result_cannot_be_vacuous_ignored_or_failed(self):
        self.assertTrue(runner.test_success('test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 400 filtered out;\n'))
        for text in ('', 'test result: ok. 0 passed; 0 failed; 0 ignored;',
                     'test result: ok. 0 passed; 0 failed; 1 ignored;',
                     'test result: FAILED. 0 passed; 1 failed; 0 ignored;'):
            self.assertFalse(runner.test_success(text))

    def test_invalid_timeout_refuses_before_lease_or_gpu_work(self):
        with patch.object(runner.admission, 'verify_lease') as lease:
            for timeout in (0, -1, float('nan'), float('inf'), 86401):
                with self.assertRaises(ValueError):
                    runner.run(SimpleNamespace(timeout_seconds=timeout))
            lease.assert_not_called()

    def test_plain_child_exit_and_raw_output_preserved(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner.admission, 'verify_lease'):
            out = Path(directory) / 'case'
            code = runner.plain([sys.executable, '-c', 'import sys; print("raw witness"); sys.exit(7)'],
                                os.environ.copy(), out, 5)
            self.assertEqual(code, 7)
            self.assertEqual((out / 'stdout.log').read_text(), 'raw witness\n')

    def test_lost_lease_kills_and_reaps_the_owned_child(self):
        calls = 0
        def lease():
            nonlocal calls
            calls += 1
            if calls > 3:
                raise RuntimeError('deliberate lost lease')
        with tempfile.TemporaryDirectory() as directory, patch.object(runner.admission, 'verify_lease', side_effect=lease):
            out = Path(directory) / 'case'
            with self.assertRaisesRegex(RuntimeError, 'lost lease'):
                runner.plain([sys.executable, '-c', 'import os,time; print(os.getpid(),flush=True); time.sleep(30)'],
                             os.environ.copy(), out, 5)
            pid = int((out / 'stdout.log').read_text().strip())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_controller_invariant_failure_reaps_without_memory_access(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner.controller, 'require_linux'), \
                patch.object(runner.controller, 'validate_stopped_group') as group:
            result = runner.controller.run_controlled(
                [sys.executable, '-c', 'import time; time.sleep(30)'], Path(directory) / 'case', 5,
                environment=os.environ.copy(),
                check_invariants=lambda: (_ for _ in ()).throw(RuntimeError('lost lease')))
            self.assertEqual(result['status'], 'failed')
            self.assertIn('lost lease', result['error'])
            self.assertIsNotNone(result['child_returncode'])
            group.assert_not_called()


if __name__ == '__main__':
    unittest.main()
