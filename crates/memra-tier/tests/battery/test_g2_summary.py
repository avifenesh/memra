"""Synthetic outer-aggregation coverage; never native performance evidence."""
import datetime
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import test_g2_campaign

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('summary', ROOT/'research/spill-d-20260919/summarize-g2.py')
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


class SummaryTests(unittest.TestCase):
    def fixture(self, root):
        visits = root/'visits'; visits.mkdir()
        collector = root/'collector'; collector.mkdir()
        recorded = []
        for size in S.G.SIZES:
            for pair in range(5):
                for order in ('ab', 'ba'):
                    name = f'score-{size}-{pair}-{order}'
                    rows = test_g2_campaign.G2Tests().fixture()
                    for r in rows:
                        if 'bytes' in r:
                            r['bytes'] = size
                        if r.get('record') == 'sample':
                            r.update(order=order, completed_bytes=size*1000, verified_bytes=size)
                    if order == 'ba':
                        positions = [i for i,r in enumerate(rows) if r.get('record') == 'sample']
                        for i,j in zip(positions[::2],positions[1::2]):
                            rows[i], rows[j] = rows[j], rows[i]
                    raw = visits/(name+'.log')
                    raw.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
                    (visits/(name+'.compute.log')).write_text('')
                    recorded.extend({'phase':'scored', 'visit':name, 'raw_log':raw.name,
                                     'raw_sha256':S.G.B.digest(raw), **r}
                                    for r in rows if r.get('record') == 'sample')
        (visits/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in recorded))
        (visits/'calibration.json').write_text(json.dumps({size:1000 for size in S.G.SIZES}))
        capture = {'exit_code':0, 'result':{'samples':200}, 'elapsed_seconds':4.75,
                   'gpu_telemetry':{'status':'captured-unvalidated', 'interval_ms':250,
                                    'raw_csv':{'path':'command.gpu.csv'}}}
        (collector/'command.capture.json').write_text(json.dumps(capture))
        gpu = 'timestamp, power.limit [W], power.max_limit [W], temperature.gpu, clocks.current.sm [MHz], clocks.current.memory [MHz], power.draw [W], pcie.link.gen.current, pcie.link.width.current\n'
        start = datetime.datetime(2026,9,20)
        for i in range(20):
            stamp = (start+datetime.timedelta(milliseconds=i*250)).strftime('%Y/%m/%d %H:%M:%S.%f')[:-3]
            gpu += stamp+', 600.00 W, 600.00 W, 50, 2000 MHz, 14000 MHz, 150.00 W, 5, 16\n'
        (collector/'command.gpu.csv').write_text(gpu)

    def test_all_observations_required_and_raw_replayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root)
            # Only the outer collector fixture is synthetic; native-visit parsing is real.
            with patch.object(S.G.B, 'validate_cell'):
                result = S.summarize(root)
                self.assertEqual(result['scored_samples'], 200)
                self.assertEqual(len(result['medians']), 20)
                self.assertTrue(all(r['n'] == 10 and r['ab_n'] == r['ba_n'] == 5 for r in result['medians']))
                self.assertEqual(result['thermal']['interval_max_ms'], 250)
                raw = root/'visits/score-4096-0-ab.log'
                raw.write_text(raw.read_text()+'\n')
                with self.assertRaisesRegex(ValueError, 'raw sample/hash mismatch'):
                    S.summarize(root)

    def test_missing_observation_and_changed_power_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root)
            with patch.object(S.G.B, 'validate_cell'):
                p = root/'collector/command.gpu.csv'
                saved = p.read_text(); p.write_text(saved.replace('600.00 W', '575.00 W', 1))
                with self.assertRaisesRegex(ValueError, 'power envelope changed'):
                    S.summarize(root)
                p.write_text(saved)
                p = root/'visits/samples.jsonl'
                p.write_text('\n'.join(p.read_text().splitlines()[:-1])+'\n')
                with self.assertRaisesRegex(ValueError, 'incomplete/reordered'):
                    S.summarize(root)


if __name__ == '__main__':
    unittest.main()
