#!/usr/bin/env python3
"""Red arms for target-card receipt replay; temporary copies never alter receipts."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

LANE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('day9', LANE / 'verify-day9.py')
DAY9 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DAY9)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.original = DAY9.RAW
        self.addCleanup(setattr, DAY9, 'RAW', self.original)
        DAY9.RAW = Path(self.temp.name)
        self.case = 'default-gen-off'
        self.path = DAY9.RAW / self.case
        shutil.copytree(self.original / self.case, self.path)
        self.capture = json.loads((self.path / 'command.capture.json').read_text())

    def save(self, capture):
        (self.path / 'command.capture.json').write_text(json.dumps(capture))

    def test_green(self):
        DAY9.replay(self.case)

    def test_capture_reds(self):
        for field, value in [('exit_code', 2), ('timed_out', True),
                             ('parse_error', 'bad'), ('qualification', True),
                             ('status', 'qualified')]:
            with self.subTest(field=field):
                capture = copy.deepcopy(self.capture)
                capture[field] = value
                self.save(capture)
                with self.assertRaises(ValueError):
                    DAY9.replay(self.case)

    def test_wrong_command_power_and_cadence(self):
        for kind in ['command', 'power', 'cadence']:
            with self.subTest(kind=kind):
                capture = copy.deepcopy(self.capture)
                if kind == 'command':
                    capture['command'][1] = 'MEMRA_MOE_RESIDENT=1'
                elif kind == 'power':
                    capture['gpu_power_limits'][0]['power.limit'] = '400.00 W'
                else:
                    capture['gpu_telemetry']['interval_ms'] = 1000
                self.save(capture)
                with self.assertRaises(ValueError):
                    DAY9.replay(self.case)

    def test_wrong_lock(self):
        (self.path / 'lock.json').write_text(json.dumps(
            {'rig': 'rtx5090', 'lock': '/tmp/memra-5090.lock', 'acquired': True}))
        with self.assertRaises(ValueError):
            DAY9.replay(self.case)

    def test_changed_log_hash(self):
        with (self.path / 'command.log').open('a') as log:
            log.write('tampered\n')
        with self.assertRaises(ValueError):
            DAY9.replay(self.case)

    def test_rehashed_missing_tape(self):
        path = self.path / 'command.log'
        text = '\n'.join(line for line in path.read_text().splitlines()
                         if not line.startswith('tokens:')) + '\n'
        path.write_text(text)
        self.capture['raw_log']['bytes'] = len(path.read_bytes())
        self.capture['raw_log']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.save(self.capture)
        with self.assertRaises(ValueError):
            DAY9.replay(self.case)


if __name__ == '__main__':
    unittest.main()
