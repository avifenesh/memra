"""CPU regressions for wrapper ownership validation. These never execute a GPU command."""
import copy
import importlib.util
import os
from pathlib import Path
import stat
import tempfile
from unittest.mock import patch
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('qualify_native', Path(__file__).with_name('qualify-native.py'))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.gpu = 'GPU-12345678-abcd-abcd-abcd-123456789abc'
        self.second = 'GPU-22345678-abcd-abcd-abcd-123456789abc'
        self.lease = {'wrapper_pid': 4101, 'child_pid': 4102, 'requested_uuids': [self.gpu],
                      'lock_order': [self.gpu], 'lock_files': {self.gpu: f'/tmp/memra-gpu-locks/{self.gpu}.lock'}}
        self.info = SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_dev=os.makedev(8, 1), st_ino=731)
        self.rows = '4: FLOCK ADVISORY WRITE 4101 08:01:731 0 EOF\n'

    def verify(self, lease=None, visible=None, ancestors=None, rows=None):
        gate.verify_lock_records(lease or self.lease, visible or self.gpu,
                                 {4101, 4102, os.getpid()} if ancestors is None else ancestors,
                                 self.rows if rows is None else rows, lambda _: self.info)

    def test_digest_does_not_require_python311_file_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bytes'
            for payload in (b'', b'\x00\xffcheckpoint\n' * 100000):
                path.write_bytes(payload)
                expected = gate.hashlib.sha256(payload).hexdigest()
                with patch.object(gate.hashlib, 'file_digest', None, create=True):
                    self.assertEqual(gate.digest(path), expected)

    def test_real_wrapper_shape_accepts_exact_exclusive_card(self):
        self.verify()

    def test_marker_without_ancestor_owned_flock_refuses(self):
        for rows in ['', self.rows.replace('4101', '4103'), self.rows.replace('FLOCK', 'POSIX'),
                     self.rows.replace('WRITE', 'READ'), self.rows.replace('731', '732'),
                     self.rows.replace('0 EOF', '0 4095')]:
            with self.subTest(rows=rows), self.assertRaises(RuntimeError):
                self.verify(rows=rows)
        with self.assertRaises(RuntimeError):
            self.verify(ancestors={4102, os.getpid()})

    def test_visible_device_order_and_exact_set_are_mandatory(self):
        for visible in ['0', self.second, f'{self.gpu},{self.second}']:
            with self.subTest(visible=visible), self.assertRaises(RuntimeError):
                self.verify(visible=visible)
        extra = copy.deepcopy(self.lease)
        extra['lock_files'][self.second] = f'/tmp/memra-gpu-locks/{self.second}.lock'
        with self.assertRaises(RuntimeError):
            self.verify(lease=extra)

    def test_interrupted_negative_control_cannot_pass(self):
        text = 'REWRITE_IDENTITY_GATE_FAIL: MODEL_LOAD: does not bind numeric_program_sha256=abc'
        refusal = 'does not bind numeric_program_sha256='
        self.assertTrue(gate.case_passed(1, refusal, text))
        for code in [-9, -15, 137, 143, 2, 0]:
            with self.subTest(code=code):
                self.assertFalse(gate.case_passed(code, refusal, text))
        self.assertFalse(gate.case_passed(1, refusal, refusal))

    def test_output_hashes_require_every_prompt_exactly_once(self):
        lines = [f'OUTPUT stage=installed-eager prompt={i} values=1 sha256={str(i) * 64}' for i in range(3)]
        self.assertEqual(set(gate.output_hashes('\n'.join(lines), 'installed-eager')), {'0', '1', '2'})
        for bad in ['\n'.join(lines[:2]), '\n'.join(lines + lines[:1]), '']:
            with self.subTest(text=bad), self.assertRaises(RuntimeError):
                gate.output_hashes(bad, 'installed-eager')

    def test_noncanonical_or_partial_lockset_refuses(self):
        wrong = copy.deepcopy(self.lease)
        wrong['lock_files'][self.gpu] = '/tmp/memra-gpu.lock'
        with self.assertRaises(RuntimeError):
            self.verify(lease=wrong)
        partial = copy.deepcopy(self.lease)
        partial['requested_uuids'] = [self.second, self.gpu]
        partial['lock_order'] = sorted(partial['requested_uuids'])
        partial['lock_files'][self.second] = f'/tmp/memra-gpu-locks/{self.second}.lock'
        with self.assertRaises(RuntimeError):
            self.verify(lease=partial, visible=','.join(partial['requested_uuids']), rows=self.rows.replace('731', '799'))
        duplicate = copy.deepcopy(self.lease)
        duplicate['requested_uuids'] *= 2
        with self.assertRaises(RuntimeError):
            self.verify(lease=duplicate)


if __name__ == '__main__':
    unittest.main()
