#!/usr/bin/env python3
"""Red arms for the budget-cell verifier and the runner's collector classifier."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

LANE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, LANE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load('runner', 'run-day10-budget.py')
VERIFIER = load('verifier', 'verify-day10-budget.py')


class Classifier(unittest.TestCase):
    def test_outcomes(self):
        with tempfile.TemporaryDirectory() as temp:
            capture = Path(temp) / 'command.capture.json'
            lock = RUNNER.LOCK_REFUSED + '\n'
            self.assertEqual(RUNNER.classify(2, lock, capture), 'lock-refused')
            self.assertEqual(RUNNER.classify(2, 'REFUSED: something else\n', capture), 'failed')
            self.assertEqual(RUNNER.classify(1, lock, capture), 'failed')
            capture.write_text(json.dumps({'status': 'refused'}))
            self.assertEqual(RUNNER.classify(2, lock, capture), 'refused')
            self.assertEqual(RUNNER.classify(2, '{...}', capture), 'refused')
            self.assertEqual(RUNNER.classify(0, '{...}', capture), 'failed')
            capture.write_text(json.dumps({'status': 'executed-not-qualified'}))
            self.assertEqual(RUNNER.classify(0, '{...}', capture), 'complete')
            self.assertEqual(RUNNER.classify(2, '{...}', capture), 'failed')
            capture.write_text(json.dumps({'status': 'failed'}))
            self.assertEqual(RUNNER.classify(1, '{...}', capture), 'failed')

    def test_budgets(self):
        self.assertEqual([(f, b) for f, b, _ in RUNNER.CELLS],
                         [('--expert-bank-gpu-bytes', 6021176), ('--expert-bank-gpu-bytes', 6881344),
                          ('--expert-bank-host-bytes', 1)])
        self.assertEqual(sorted(RUNNER.NAMES.values()), ['budget-exact-8', 'budget-host-refuse-1', 'budget-refuse-7'])


class Receipts(unittest.TestCase):
    def setUp(self):
        status = json.loads((VERIFIER.RAW / 'budget-status.json').read_text())
        self.cells = {row['case']: row['directory'] for row in status['completed']}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for directory in self.cells.values():
            shutil.copytree(VERIFIER.RAW / directory, self.root / directory)

    def capture(self, case):
        return self.root / self.cells[case] / 'command.capture.json'

    def rehash(self, case):
        path = self.root / self.cells[case]
        capture = json.loads((path / 'command.capture.json').read_text())
        raw = (path / 'command.log').read_bytes()
        capture['raw_log']['bytes'] = len(raw)
        capture['raw_log']['sha256'] = hashlib.sha256(raw).hexdigest()
        (path / 'command.capture.json').write_text(json.dumps(capture))

    def test_green(self):
        refusal = VERIFIER.replay_refusal(self.root / self.cells['budget-refuse-7'])
        self.assertEqual((refusal['requested_bytes'], refusal['minimum_bytes']), (6021176, 6881344))
        exact, tape = VERIFIER.replay_exact(self.root / self.cells['budget-exact-8'])
        self.assertEqual(exact['gpu_slots'], 8)
        self.assertGreater(exact['gpu_evictions'], 0)
        self.assertEqual(len(tape), 32)

    def test_host_refusal_green_and_reds(self):
        path = self.root / self.cells['budget-host-refuse-1']
        result = VERIFIER.replay_host_refusal(path)
        self.assertEqual((result['requested_bytes'], result['minimum_bytes'], result['ceiling_bytes']),
                         (1, 860160, 268435456))
        log = path / 'command.log'
        text = log.read_text()
        with self.subTest(red='legacy error shape'):
            lines = text.splitlines()
            lines[-3] = 'Error: "experts-via-tier host bank budget cannot hold one expert record"'
            lines[-2] = 'native_exit_code=1'
            log.write_text('\n'.join(lines) + '\n')
            self.rehash('budget-host-refuse-1')
            with self.assertRaisesRegex(ValueError, 'wrapper did not pass the native token'):
                VERIFIER.replay_host_refusal(path)
        with self.subTest(red='gpu token in the host cell'):
            lines = text.splitlines()
            gpu = 'REFUSED: experts-via-tier GPU bank budget cannot hold the eight-slot minimum (requested 1, minimum 6881344, ceiling 9)'
            lines[-3] = lines[-1] = gpu
            log.write_text('\n'.join(lines) + '\n')
            self.rehash('budget-host-refuse-1')
            capture = json.loads((path / 'command.capture.json').read_text())
            capture['failure_quote'] = gpu
            (path / 'command.capture.json').write_text(json.dumps(capture))
            with self.assertRaisesRegex(ValueError, 'refusal text changed'):
                VERIFIER.replay_host_refusal(path)

    def test_refusal_reds(self):
        path = self.root / self.cells['budget-refuse-7']
        original = (path / 'command.capture.json').read_text()
        for field, value in [('exit_code', 1), ('status', 'failed'), ('timed_out', True),
                             ('command', ['env', 'run-gen'])]:
            with self.subTest(field=field):
                capture = json.loads(original)
                capture[field] = value
                (path / 'command.capture.json').write_text(json.dumps(capture))
                with self.assertRaises(ValueError):
                    VERIFIER.replay_refusal(path)
        (path / 'command.capture.json').write_text(original)
        log = path / 'command.log'
        text = log.read_text()
        with self.subTest(red='tampered log hash'):
            log.write_text(text + 'trailing\n')
            with self.assertRaisesRegex(ValueError, 'receipt hash mismatch'):
                VERIFIER.replay_refusal(path)
        with self.subTest(red='work after refusal'):
            log.write_text('[expert-host-slru] key=0:0:0 bytes=1 slot=0 hit=false victim=-\n' + text)
            self.rehash('budget-refuse-7')
            with self.assertRaisesRegex(ValueError, 'work after refusal'):
                VERIFIER.replay_refusal(path)
        with self.subTest(red='refusal not final'):
            log.write_text(text + 'loaded qwen35moe\n')
            self.rehash('budget-refuse-7')
            with self.assertRaisesRegex(ValueError, 'refusal is not the final line'):
                VERIFIER.replay_refusal(path)
        with self.subTest(red='wrong arithmetic'):
            log.write_text(text.replace('minimum 6881344', 'minimum 6881343'))
            self.rehash('budget-refuse-7')
            capture = json.loads((path / 'command.capture.json').read_text())
            capture['failure_quote'] = log.read_text().splitlines()[-1]
            (path / 'command.capture.json').write_text(json.dumps(capture))
            with self.assertRaisesRegex(ValueError, 'refusal arithmetic'):
                VERIFIER.replay_refusal(path)

    def test_exact_reds(self):
        path = self.root / self.cells['budget-exact-8']
        log = path / 'command.log'
        text = log.read_text()
        with self.subTest(red='slot count'):
            log.write_text(text.replace('[expert-gpu-slru] slots=8 ', '[expert-gpu-slru] slots=9 '))
            self.rehash('budget-exact-8')
            with self.assertRaisesRegex(ValueError, 'GPU extents'):
                VERIFIER.replay_exact(path)
        with self.subTest(red='missing tape'):
            log.write_text('\n'.join(l for l in text.splitlines() if not l.startswith('tokens:')) + '\n')
            self.rehash('budget-exact-8')
            with self.assertRaises(ValueError):
                VERIFIER.replay_exact(path)
        with self.subTest(red='budget line after install'):
            lines = text.splitlines()
            budget = next(l for l in lines if l.startswith('[experts-via-tier] gpu_bank_budget'))
            lines.remove(budget)
            lines.insert(lines.index(next(l for l in lines if l.startswith('loaded '))) + 1, budget)
            log.write_text('\n'.join(lines) + '\n')
            self.rehash('budget-exact-8')
            with self.assertRaisesRegex(ValueError, 'budget line missing or after install'):
                VERIFIER.replay_exact(path)
        with self.subTest(red='tampered hash'):
            log.write_text(text.replace('prefill argmax=198', 'prefill argmax=199', 1))
            with self.assertRaisesRegex(ValueError, 'receipt hash mismatch'):
                VERIFIER.replay_exact(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
