#!/usr/bin/env python3
"""Real SIGTERM at final verification and PASS publication; native commands mocked."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

BASE = Path(__file__).resolve().parent
CHILD = r'''
import importlib.util,sys,tempfile
from pathlib import Path
file,kind,fault,tmp=sys.argv[1:]
Path(tmp).mkdir();tempfile.tempdir=tmp
spec=importlib.util.spec_from_file_location('signal_fixture',file)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
if kind=='baseline':m.NativeFinalizationTests().exercise(fault)
else:m.FinalizationTests().exercise(kind,fault)
'''


class RealCancellationTests(unittest.TestCase):
    def check_case(self, kind, fault):
        file = BASE / ('test_native_finalization.py' if kind == 'baseline' else 'test_qualify_callers.py')
        env = os.environ.copy()
        old = env.get('REWRITE_FINALIZATION_OLD_ROOT')
        if old:
            env['REWRITE_NATIVE_RUNNER_UNDER_TEST'] = str(Path(old) / 'qualify-native.py')
            env['REWRITE_RUNNER_UNDER_TEST'] = str(Path(old) / 'qualify-callers.py')
        else:
            env.pop('REWRITE_NATIVE_RUNNER_UNDER_TEST', None)
            env.pop('REWRITE_RUNNER_UNDER_TEST', None)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-B', '-c', CHILD, str(file), kind, fault,
                                     str(Path(directory) / 'owned-child-temp')], env=env,
                                    capture_output=True, text=True, timeout=20)
            records = [json.loads(line) for line in result.stdout.splitlines()
                       if line.startswith('{"signal_control"')]
            print(f'REAL_SIGNAL_CONTROL kind={kind} fault={fault} exit={result.returncode} records={records}', flush=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['signals_sent'], 1)
            self.assertTrue(records[0]['had_error'])
            self.assertIn(records[0]['status'], ('failed', 'incomplete'))

    def test_sigterm_during_final_verify(self):
        for kind in ('baseline', 'callers', 'supported-callers', 'transfer', 'battery'):
            with self.subTest(kind=kind): self.check_case(kind, 'sig_verify')

    def test_sigterm_during_pending_result_write(self):
        for kind in ('baseline', 'callers', 'supported-callers', 'transfer', 'battery'):
            with self.subTest(kind=kind): self.check_case(kind, 'sig_pending')

    def test_sigterm_before_atomic_replace(self):
        for kind in ('baseline', 'callers', 'supported-callers', 'transfer', 'battery'):
            with self.subTest(kind=kind): self.check_case(kind, 'sig_replace')


if __name__ == '__main__': unittest.main()
