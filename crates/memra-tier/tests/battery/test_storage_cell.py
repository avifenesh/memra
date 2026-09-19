"""A's explicit storage-cell dispatcher on archived bytes, not fresh GPU evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
REAL = ROOT/'research/spill-lead-20260919/rented-5090-20260919/receipts/first-hour-20260919T130057Z'
spec = importlib.util.spec_from_file_location('storage_cell_battery', ROOT/'tools/tier-battery.py')
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)


class StorageCellTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'cell'
        shutil.copytree(REAL/'a-restore-264', self.root)
        self.journal = self.root/'CELL.jsonl'
        self.output = self.root/'joined.jsonl'

    def validate(self, envelopes=None):
        argv = [sys.executable, str(ROOT/'tools/tier-battery.py'), '--validate', str(self.journal),
                '--schema', 'storage-cell', '--out', str(self.output)]
        if envelopes is not None:
            path = self.root/'envelopes.jsonl'
            path.write_text(''.join(json.dumps(e)+'\n' for e in envelopes))
            argv += ['--storage-samples', str(path)]
        return subprocess.run(argv, capture_output=True, text=True, timeout=10)

    def rewrite_capture(self, edit):
        path = self.root/'command.capture.json'
        capture = json.loads(path.read_text()); edit(capture)
        path.write_text(json.dumps(capture)+'\n')
        rows = [json.loads(line) for line in self.journal.read_text().splitlines()]
        rows[-1]['capture'] = B.descriptor(self.root, path)
        self.journal.write_text(''.join(json.dumps(row)+'\n' for row in rows))

    def test_real_capture_explicit_join_nulls_and_immutable_output(self):
        result = self.validate()
        self.assertEqual(result.returncode, 0, result.stderr)
        joined = json.loads(self.output.read_text())
        self.assertEqual(joined['kind'], 'storage-cell-join')
        self.assertFalse(joined['qualification'])
        self.assertIsNone(joined['samples'][0]['physical_bytes'])
        self.assertEqual(joined['gpu_telemetry_status'], 'empty')
        self.assertEqual(joined['samples'][0]['valid_bytes'], 264)
        self.assertEqual(self.validate().returncode, 2)
        self.output.unlink()
        envelope = {'run_id': joined['run_id'], 'sample': joined['samples'][0]}
        self.assertEqual(self.validate([envelope]).returncode, 0)
        self.output.unlink()
        for bad in ([dict(envelope, run_id='foreign')], [envelope, envelope], []):
            result = self.validate(bad)
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertFalse(self.output.exists())
        bad = copy.deepcopy(envelope); bad['sample']['io_bytes'] += 1
        self.assertEqual(self.validate([bad]).returncode, 2)

    def test_raw_hash_and_capture_command_and_timestamp_refusals(self):
        mutations = [
            lambda c: c.update(exit_code=2),
            lambda c: c.update(command=['storage-bench', 'restore', '.', '4097', 'buffered']),
            lambda c: c.update(ended_monotonic_ns=c['started_monotonic_ns']),
            lambda c: c['gpu_telemetry']['raw_csv'].update(path='../escape'),
        ]
        original = (self.root/'command.capture.json').read_bytes()
        journal = self.journal.read_bytes()
        for edit in mutations:
            self.rewrite_capture(edit)
            self.assertEqual(self.validate().returncode, 2)
            self.assertFalse(self.output.exists())
            (self.root/'command.capture.json').write_bytes(original)
            self.journal.write_bytes(journal)
        rows = [json.loads(line) for line in self.journal.read_text().splitlines()]
        rows[-1]['ended_utc'] = '2000-01-01T00:00:00+00:00'
        self.journal.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        self.assertEqual(self.validate().returncode, 2)
        self.journal.write_bytes(journal)
        with (self.root/'command.log').open('ab') as stream: stream.write(b'\n')
        self.assertEqual(self.validate().returncode, 2)

    def test_start_only_and_wrong_schema_never_promote(self):
        result = subprocess.run([sys.executable, str(ROOT/'tools/tier-battery.py'), '--validate',
            str(self.journal), '--schema', 'runs'], capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.journal.write_text(self.journal.read_text().splitlines()[0]+'\n')
        self.assertEqual(self.validate().returncode, 2)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
