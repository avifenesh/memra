#!/usr/bin/env python3
"""Replay real, archived overlay bytes and refusal arms; no new box execution."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
LANE = ROOT / 'research/spill-a-20260919'
SOURCE = ROOT / 'research/spill-lead-20260919/rented-5090-20260919/receipts/first-hour-20260919T130057Z/a-restore-264'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


battery = load('a_battery', ROOT / 'tools/tier-battery.py')
join = load('a_capture', LANE / 'storage_capture.py')
runner = load('a_box2', LANE / 'box2_cells.py')


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='capture-test-', dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'cell'
        shutil.copytree(SOURCE, self.root)
        self.journal = self.root / 'CELL.jsonl'

    def validate(self, samples=None):
        return join.validate_storage_cell(self.journal, battery, samples)

    def rewrite_capture(self, update):
        path = self.root / 'command.capture.json'
        capture = json.loads(path.read_text())
        update(capture)
        path.write_text(json.dumps(capture))
        rows = [json.loads(line) for line in self.journal.read_text().splitlines()]
        rows[-1]['capture'] = battery.descriptor(self.root, path)
        self.journal.write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def test_archived_real_capture_and_nulls(self):
        result = self.validate()
        self.assertFalse(result['qualification'])
        self.assertEqual(result['samples'][0]['status'], 'byte-exact')
        self.assertIsNone(result['samples'][0]['physical_bytes'])
        self.assertEqual(result['gpu_telemetry_status'], 'empty')
        self.assertEqual(result['samples'][0]['valid_bytes'], 264)
        self.validate([{'run_id': result['run_id'], 'sample': result['samples'][0]}])

    def test_raw_tamper(self):
        with (self.root / 'command.log').open('ab') as out:
            out.write(b'\n')
        with self.assertRaisesRegex(ValueError, 'size mismatch'):
            self.validate()

    def test_truncated_journal(self):
        with self.journal.open('a') as out:
            out.write('{')
        with self.assertRaisesRegex(ValueError, 'complete start/end'):
            self.validate()

    def test_noncanonical_lock(self):
        (self.root / 'lock.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'canonical'):
            self.validate()

    def test_failed_capture_even_when_resealed(self):
        self.rewrite_capture(lambda c: c.update(exit_code=1))
        with self.assertRaisesRegex(ValueError, 'failed storage capture'):
            self.validate()

    def test_changed_command_even_when_resealed(self):
        self.rewrite_capture(lambda c: c['command'].__setitem__(3, '4097'))
        with self.assertRaisesRegex(ValueError, 'command/class'):
            self.validate()

    def test_orphan_duplicate_and_modified_envelopes(self):
        result = self.validate()
        good = {'run_id': result['run_id'], 'sample': result['samples'][0]}
        for envelope in ([dict(good, run_id='foreign')], [good, good], []):
            with self.assertRaises(ValueError):
                self.validate(envelope)
        bad = copy.deepcopy(good)
        bad['sample']['io_bytes'] += 1
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.validate([bad])

    def test_escaped_diagnostic_even_when_resealed(self):
        self.rewrite_capture(lambda c: c['gpu_telemetry']['raw_csv'].update(path='../elsewhere'))
        with self.assertRaisesRegex(ValueError, 'diagnostic hash/path'):
            self.validate()

    def test_capture_window(self):
        self.rewrite_capture(lambda c: c.update(ended_monotonic_ns=c['started_monotonic_ns']))
        with self.assertRaisesRegex(ValueError, 'capture window'):
            self.validate()

    def test_box2_routing(self):
        scratch = Path('/never-created-a')
        for cell in ('build', 'build-worker', 'storage-tests', 'gc', 'gc-upgrade', 'catalog'):
            command = runner.command_for(cell, scratch, scratch)
            self.assertEqual(command[0], 'cargo')
            self.assertEqual(command[command.index('-j')+1], '16')
        for cell in ('pinned', 'direct-roundtrip-264', 'direct-restore-4097', 'buffered-restore-1048576'):
            command = runner.command_for(cell, scratch, scratch)
            self.assertEqual(command[command.index('--rig')+1], 'rtx5090')
            executable = command[command.index('--execute')+1]
            self.assertEqual(executable, 'bash')
            self.assertEqual(command[-2], '-c')
            self.assertNotEqual(executable, '--')
        with self.assertRaises(ValueError):
            runner.plan('direct-roundtrip-0', scratch)
        command = ['bash', str(LANE/'rig-cells-a.sh'), '--box2', str(scratch),
                   '--out', str(scratch/'receipt'), '--cell', 'pinned', '--dry-run']
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        row = json.loads(result.stdout)
        self.assertFalse(row['executed'])
        self.assertFalse(scratch.exists())
        self.assertEqual(row['storage_label'], 'overlay, development, not spill speed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
