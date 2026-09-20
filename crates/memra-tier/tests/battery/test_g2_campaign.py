"""G2 orchestration parser red arms; archived 5090 rows are inputs, not PRO evidence."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('g2', ROOT/'research/spill-d-20260919/run-g2.py')
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)


class G2Tests(unittest.TestCase):
    def fixture(self):
        raw = ROOT/'research/spill-f-20260919/h2d-copies/native-n1-attempt2/visits/4096-1000-ab.log'
        rows = [json.loads(s) for s in raw.read_text().splitlines() if s.startswith('{')]
        for row in rows:
            if row.get('record') == 'sample':
                row['wall_ns'] = 500_000_000
                for when in ['power_before', 'power_after']:
                    for key in ['power.limit', 'power.max_limit']:
                        row[when][key] = '600.00 W'
        return rows

    def test_calibration_cap_and_target(self):
        self.assertEqual(G.next_copies(1000, [{'wall_ns': 100_000_000}, {'wall_ns': 200_000_000}]), 5000)
        with self.assertRaises(ValueError):
            G.next_copies(100000, [{'wall_ns': 10}])
        with self.assertRaises(ValueError):
            G.next_copies(1000, [{'wall_ns': 0}])

    def check(self, rows):
        return G.check_samples('\n'.join(json.dumps(r) for r in rows), 4096, 1000, 'ab', scored=True)

    def test_complete_scored_matrix_and_red_arms(self):
        rows = self.fixture()
        self.assertEqual(len(self.check(rows)), 4)
        for key, value in [('wall_ns', 249999999), ('identity', False), ('actual_sha256', '0'*64),
                           ('copies', 1), ('order', 'ba'), ('evidence_class', 'dry-run-no-cuda')]:
            rows = self.fixture()
            next(r for r in rows if r.get('record') == 'sample')[key] = value
            with self.assertRaises((AssertionError, ValueError)):
                self.check(rows)
        rows = self.fixture()
        sample = next(r for r in rows if r.get('record') == 'sample')
        for when in ['power_before', 'power_after']:
            sample[when]['power.limit'] = '575.00 W'
        with self.assertRaises(ValueError):
            self.check(rows)
        rows.remove(sample)
        with self.assertRaises(AssertionError):
            self.check(rows)


if __name__ == '__main__':
    unittest.main()
