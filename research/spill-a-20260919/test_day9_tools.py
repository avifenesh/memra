"""Offline replay/tamper controls for native day-9 receipts."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import verify_day9 as replay

SOURCE = Path(__file__).with_name('day9')/'native/final'


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='spill-a-day9-replay-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'receipt'
        shutil.copytree(SOURCE, self.root)
        self.run_dir = next(self.root.glob('attempt-*/conformance.log')).parent

    def reseal(self):
        manifest = {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(self.root.rglob('*'))
                    if p.is_file() and p.name != 'remote-hashes.json'}
        (self.root/'remote-hashes.json').write_text(json.dumps(manifest))

    def test_replay(self):
        result = replay.verify(self.root)
        self.assertEqual(result['status'], 'raw-replay-pass')
        self.assertFalse(result['qualification'])

    def test_raw_tamper(self):
        path = self.run_dir/'conformance.log'
        path.write_text(path.read_text()+'forged\n')
        with self.assertRaises(AssertionError):
            replay.verify(self.root)

    def test_canonical_verdict_cannot_be_relabelled(self):
        path = self.run_dir/'conformance.log'
        path.write_text(path.read_text().replace(replay.CANONICAL[0], 'PASS additive source retirement'))
        self.reseal()
        with self.assertRaises(AssertionError):
            replay.verify(self.root)

    def test_roundtrip_hash_mismatch(self):
        path = self.run_dir/'roundtrip.log'
        text = path.read_text()
        start = text.index('actual_sha256=')+len('actual_sha256=')
        path.write_text(text[:start]+('0' if text[start] != '0' else '1')+text[start+1:])
        self.reseal()
        with self.assertRaises(AssertionError):
            replay.verify(self.root)

    def test_missing_power_is_incomplete_not_false_success(self):
        path = self.run_dir/'command.capture.json'
        row = json.loads(path.read_text())
        row.pop('gpu_power_limits', None)
        path.write_text(json.dumps(row))
        self.reseal()
        result = replay.verify(self.root)
        self.assertFalse(result['telemetry_complete'])
        self.assertFalse(result['qualification'])

    def test_manifest_omission(self):
        path = self.root/'remote-hashes.json'
        row = json.loads(path.read_text())
        row.pop(str((self.run_dir/'conformance.log').relative_to(self.root)))
        path.write_text(json.dumps(row))
        with self.assertRaises(AssertionError):
            replay.verify(self.root)


if __name__ == '__main__':
    unittest.main()
