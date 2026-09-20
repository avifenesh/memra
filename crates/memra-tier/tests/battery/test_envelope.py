"""G2 protocol plumbing tests. No CUDA, telemetry fabrication, or performance claim."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location('envelope', ROOT/'tools/tier-envelope.py')
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)


def sample():
    return {'schema_version': 1, 'record': 'sample', 'bytes': 4096, 'direction': 'h2d',
            'arm': 'pageable', 'copies': 1, 'completed_bytes': 4096, 'verified_bytes': 4096,
            'identity': True, 'wall_ns': 1000, 'event_ms': .001, 'setup_ns': 10, 'verify_ns': 10,
            'expected_sha256': 'a'*64, 'actual_sha256': 'a'*64, 'unix_start_ns': 100,
            'unix_end_ns': 1100, 'power_before': {'power.limit': '400.00 W'},
            'power_after': {'power.limit': '400.00 W'}}


class EnvelopeTests(unittest.TestCase):
    def test_visit_preserves_collector_group_and_lock_fd(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(out=Path(temp), probe=Path('/probe'), bytes=4096,
                                   direction='h2d', worker=42)
            with patch.object(E, 'power', return_value={}), \
                 patch.object(E.B, 'tee_run', side_effect=RuntimeError('launch sentinel')) as launch:
                with self.assertRaisesRegex(RuntimeError, 'launch sentinel'):
                    E.visit(args, 'test', 0, 'AB', 1)
            self.assertTrue(launch.call_args.kwargs['shared_group'])
            self.assertEqual(launch.call_args.kwargs['pass_fds'], (42,))

    def test_calibration_covers_fast_arm_and_does_not_increase_n(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(out=Path(temp), correctness_only=True)
            def samples(_args, phase, attempt, order, copies):
                self.assertEqual((phase, order), ('calibration', 'AB'))
                return [{'sample': {'wall_ns': 1_000_000*copies}},
                        {'sample': {'wall_ns': 2_000_000*copies}}]
            with patch.object(E, 'visit', side_effect=samples):
                self.assertEqual(E.calibrate(args), 438)
            receipt = json.loads((args.out/'calibration.json').read_text())
            self.assertTrue(receipt['discarded'])
            self.assertEqual(len(receipt['attempts']), 2)

    def test_calibration_converges_with_fixed_launch_overhead(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(out=Path(temp), correctness_only=True)
            def samples(_args, phase, attempt, order, copies):
                return [{'sample': {'wall_ns': 20_000_000 + copies*1_000_000}}]
            with patch.object(E, 'visit', side_effect=samples):
                self.assertGreaterEqual(E.calibrate(args), 330)

    def test_calibration_refuses_insufficient_copy_cap(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(out=Path(temp), correctness_only=True)
            with patch.object(E, 'visit', return_value=[{'sample': {'wall_ns': 1}}]):
                with self.assertRaisesRegex(ValueError, 'capped'):
                    E.calibrate(args)

    def test_registered_matrix_and_both_orders(self):
        self.assertEqual(E.SIZES, [4096, 16384, 65536, 262144, 1048576, 4194304,
                                  16777216, 67108864, 268435456, 1073741824])
        rows = E.orders(5)
        self.assertEqual(len(rows), 20)
        for i in range(5):
            self.assertEqual(rows[4*i:4*i+4], [(i, 'AB', 'pageable'), (i, 'AB', 'pinned'),
                                             (i, 'BA', 'pinned'), (i, 'BA', 'pageable')])

    def test_invalid_sample_refuses(self):
        E.validate_visit(sample(), 4096, 'h2d', 'pageable', 1)
        for key, value in [('verified_bytes', 0), ('identity', False), ('wall_ns', 0),
                           ('event_ms', float('nan')), ('actual_sha256', 'b'*64),
                           ('copies', 2), ('power_after', {'power.limit': '600.00 W'})]:
            row = copy.deepcopy(sample()); row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                E.validate_visit(row, 4096, 'h2d', 'pageable', 1)

    def test_n1_is_explicitly_not_scored(self):
        command = [sys.executable, str(ROOT/'tools/tier-envelope.py'), '--plan', '--rounds', '1']
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        result = subprocess.run(command+['--correctness-only'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(len(plan['cells']), 20)
        self.assertEqual(plan['timeout_seconds'], 300)
        self.assertFalse(plan['medians_published'])
        self.assertFalse(plan['qualification'])

    def test_telemetry_spacing_power_and_coverage(self):
        import datetime
        start = datetime.datetime(2026, 9, 20).timestamp()
        cap = {'power_limit_w': 400., 'power_max_limit_w': 600.}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'gpu.csv'
            header = 'timestamp,index,power.limit [W],power.max_limit [W]\n'
            good = ['2026/09/20 00:00:00.000,0,400 W,600 W\n',
                    '2026/09/20 00:00:00.250,0,400 W,600 W\n',
                    '2026/09/20 00:00:00.500,0,400 W,600 W\n']
            path.write_text(header+''.join(good))
            self.assertEqual(E.telemetry_check(path, start+.1, start+.4, cap)['status'], 'pass')
            with self.assertRaises(ValueError): E.telemetry_check(path, start-.1, start+.4, cap)
            for rows in [good[:1], [good[0], good[2].replace('.500', '.900')],
                         [good[0], good[1].replace('400 W', '600 W'), good[2]]]:
                path.write_text(header+''.join(rows))
                with self.assertRaises(ValueError): E.telemetry_check(path, start+.1, start+.4, cap)

    def test_worker_refuses_missing_owning_fd_before_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(ROOT/'tools/tier-envelope.py'), '--worker', '12345',
                                     '--out', tmp, '--probe', sys.executable, '--correctness-only', '--rounds', '1'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertFalse((Path(tmp)/'identity.json').exists())


if __name__ == '__main__':
    unittest.main()
