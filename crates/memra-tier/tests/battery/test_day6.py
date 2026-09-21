"""Collector day-six CPU/stub regressions; never hardware qualification."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('battery_day6', ROOT/'tools/tier-battery.py')
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
import private_lock


class CollectorTests(private_lock.PrivateLockMixin, unittest.TestCase):
    BATTERY = B
    def test_execute_preserves_literal_separator_and_child_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out = root/'out'
            child = [sys.executable, '-c', 'import json,sys; print(json.dumps(sys.argv[1:]))',
                     '--', '--ignored', '--exact', '--nocapture', '--execute', 'a b']
            proc = subprocess.run([sys.executable, str(ROOT/'tools/tier-battery.py'),
                '--rig', 'rtx5090', private_lock.FLAG, '--out', str(out), '--execute', *child],
                env={**os.environ, 'PATH': str(root/'absent')}, capture_output=True, timeout=10)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(json.loads((out/'command.log').read_text()), child[3:])
            capture = json.loads((out/'command.capture.json').read_text())
            self.assertEqual(capture['command'], child)
            B.validate_cell(out/'CELL.jsonl')

    def test_refusals_require_terminal_diagnostic_and_exit_two(self):
        for text, code, expected in [
            ('REFUSED: unsupported route\n', 2, 'refused'),
            ('noise\nError: operation not supported\n', 2, 'failed'),
            ('kv-tier-gate: REFUSED: unsupported route\n', 2, 'refused'),
            ('REFUSED: unsupported route\n', 9, 'failed'),
            ('REFUSED: unsupported route\ntrailer\n', 2, 'failed'),
            ('Error: diagnostic only\n', 0, 'executed-not-qualified'),
            ('unknown\n', 2, 'failed'),
        ]:
            with self.subTest(text=text, code=code), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                record = B.SubprocessRunner(str(root/'absent')).run(
                    [sys.executable, '-c', f'import sys; sys.stdout.write({text!r}); sys.exit({code})'],
                    root/'command.log', echo=False)
                self.assertEqual(record['status'], expected)
                B.validate_capture(record, root)
                if expected == 'refused':
                    self.assertEqual(record['failure_quote'], text.splitlines()[-1])
                    bad = copy.deepcopy(record); bad['exit_code'] = 9
                    with self.assertRaises(ValueError): B.validate_capture(bad, root)
        self.assertIsNone(B.explicit_refusal('REFUSED: timed out', 2, True))

    def test_recovery_and_per_cell_power_runbook(self):
        doc = (ROOT/'research/spill-d-20260919/RIG-DAY1.md').read_text()
        for phrase in ('Keep ≤30 min before replacing, by receipt state; sync after every cell',
                       'two stops today', '400 W restricted-power development',
                       'gpu_power_limits', 'CAPTURE-CONTRACT.md', 'not had that fragment applied'):
            self.assertIn(phrase, doc)

    def test_second_rented_topology_is_gen5_capable_not_bandwidth_proof(self):
        spec = importlib.util.spec_from_file_location('topology_day6', ROOT/'tools/tier-topology.py')
        topology = importlib.util.module_from_spec(spec); spec.loader.exec_module(topology)
        fixture = json.loads((Path(__file__).parent/'topology.rented-5090-second.fixture.json').read_text())
        report = topology.probe(fixture)
        self.assertFalse(report['route_qualification'])
        self.assertEqual(report['pcie_links'], [{'device_ordinal': 0, 'generation_current': 1,
            'generation_max': 5, 'device_generation_max': 5, 'host_generation_max': 5,
            'width_current': 16, 'width_max': 16, 'bandwidth_measured': False}])

    def test_power_limits_are_captured_per_cell_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); smi = root/'nvidia-smi'
            smi.write_text('#!'+sys.executable+'\nimport sys,time\n'
                'if "-lms" in sys.argv:\n'
                ' assert "power.limit,power.max_limit" in sys.argv[1]\n'
                ' print("timestamp, index, power.limit [W], power.max_limit [W]", flush=True)\n'
                ' print("STUB, 0, 400.00, 600.00", flush=True)\n'
                ' print("STUB, 1, N/A, N/A", flush=True)\n'
                ' time.sleep(10)\n'
                'else: print("pid, process_name, used_memory")\n')
            smi.chmod(0o755)
            record = B.SubprocessRunner(str(smi)).run(
                [sys.executable, '-c', 'import time; time.sleep(.3)'], root/'command.log', echo=False)
            expected = [{'device': '0', 'power.limit': '400.00', 'power.max_limit': '600.00'},
                        {'device': '1', 'power.limit': 'N/A', 'power.max_limit': 'N/A'}]
            self.assertEqual(record['gpu_power_limits'], expected)
            end = json.loads((root/'CELL.jsonl').read_text().splitlines()[-1])
            self.assertEqual(end['gpu_power_limits'], expected)
            B.validate_capture(record, root)
            bad = copy.deepcopy(record); bad['gpu_power_limits'][0]['power.limit'] = '600.00'
            with self.assertRaises(ValueError): B.validate_capture(bad, root)
            (root/'command.gpu.csv').write_text('tampered')
            with self.assertRaises(ValueError): B.validate_capture(record, root)


if __name__ == '__main__':
    unittest.main()
