"""Offline container/interruption/schema regressions; no hardware qualification."""
import copy
import importlib.util
import json
import os
import signal
import time
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('battery4', ROOT/'tools/tier-battery.py')
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)


class Day4Tests(unittest.TestCase):
    def bootstrap(self, out, extra=(), source=None):
        script = source or (ROOT/'tools/tier-rig-bootstrap.sh').read_text()
        return subprocess.run(['bash', '-s', '--', '--dry-run', '--out', str(out), *extra],
                              input=script, text=True, cwd=ROOT, capture_output=True,
                              env={**os.environ, 'BRANCH': 'lane/spill-d-test'}, timeout=30)

    def test_container_power_refusal_records_and_accepts_max(self):
        source = (ROOT/'tools/tier-rig-bootstrap.sh').read_text()
        source = source.replace('32607, 600.00, 600.00', '32607, 575.00, 600.00')
        source = source.replace('code = 0', "code = 4 if label == 'set-power' else 0")
        source = source.replace("'600.00, 600.00\\n'", "'575.00, 600.00\\n'")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'out'
            proc = self.bootstrap(out, ('--set-power', '--provider', 'runpod'), source)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            r = json.loads((out/'BOOTSTRAP.json').read_text())
            self.assertEqual(r['power_set_exit'], 4)
            self.assertTrue(r['power_limit_restricted'])
            self.assertEqual(r['paths']['persistent_root'], '/workspace')
            self.assertIsNone(r['paths']['nvme'])
            self.assertFalse(r['qualification'])

    def test_resume_revalidates_and_preserves_old_failure(self):
        source = (ROOT/'tools/tier-rig-bootstrap.sh').read_text()
        bad = source.replace('code = 0', "code = 9 if label == 'build-1' else 0")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'out'
            self.assertEqual(self.bootstrap(out, source=bad).returncode, 2)
            old = (out/'BOOTSTRAP.json').read_bytes()
            events = [json.loads(x) for x in (out/'BOOTSTRAP.jsonl').read_text().splitlines()]
            self.assertTrue(any(x['event'] == 'cell-start' for x in events))
            self.assertTrue((out/'build-1.log').exists())
            self.assertEqual(self.bootstrap(out, ('--resume',)).returncode, 0)
            self.assertEqual((out/'BOOTSTRAP.json').read_bytes(), old)
            resumed = json.loads(next((out/'attempts').glob('*/BOOTSTRAP.json')).read_text())
            self.assertEqual(resumed['resume_from']['last_step'], 'build-1')
            self.assertIn('accept-1', {s['name'] for s in resumed['steps']})

    def test_sigkill_keeps_active_bootstrap_log_and_journal(self):
        source = (ROOT/'tools/tier-rig-bootstrap.sh').read_text()
        source = source.replace('code = 0', "time.sleep(20) if label == 'build-1' else None; code = 0")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out = root/'out'; script = root/'kill.sh'
            script.write_text(source)
            proc = subprocess.Popen(['bash', str(script), '--dry-run', '--out', str(out)],
                                    cwd=ROOT, env={**os.environ, 'BRANCH': 'lane/spill-d-test'},
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    start_new_session=True)
            reached = False
            try:
                for _ in range(500):
                    receipt = out/'BOOTSTRAP.json'
                    if receipt.exists():
                        r = json.loads(receipt.read_text())
                        if r.get('active_step', {}).get('name') == 'build-1' and (out/'build-1.log').exists():
                            reached = True
                            break
                    time.sleep(.01)
                self.assertTrue(reached, 'did not reach interruption point')
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
            receipt = json.loads((out/'BOOTSTRAP.json').read_text())
            self.assertEqual(receipt['status'], 'incomplete')
            self.assertEqual(receipt['active_step']['name'], 'build-1')
            events = [json.loads(line) for line in (out/'BOOTSTRAP.jsonl').read_text().splitlines()]
            self.assertEqual(events[-1]['event'], 'cell-start')
            self.assertTrue((out/'build-1.log').read_text().startswith('STUB build'))
            self.assertEqual(self.bootstrap(out, ('--resume',)).returncode, 0)

    def test_vast_requires_persistence_and_nvme_is_not_assumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertNotEqual(self.bootstrap(root/'bad', ('--provider', 'vast')).returncode, 0)
            proc = self.bootstrap(root/'good', ('--provider', 'vast', '--persistent-root', '/workspace', '--nvme-root', '/scratch'))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            r = json.loads((root/'good/BOOTSTRAP.json').read_text())
            self.assertEqual(r['paths']['nvme'], '/scratch')
            names = [s['name'] for s in r['steps']]
            self.assertLess(names.index('storage-lsblk'), names.index('nvme-mount'))
            source = (ROOT/'tools/tier-rig-bootstrap.sh').read_text().replace("'nvme0n1\\n'", "'overlay\\n'")
            self.assertEqual(self.bootstrap(root/'network', ('--nvme-root', '/scratch'), source).returncode, 2)

    def test_both_schema_cli_and_type_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'telemetry.jsonl'
            samples = [B.sample_fake(i*B.INTERVAL_NS, i) for i in range(2)]
            path.write_text(''.join(json.dumps(r)+'\n' for r in samples))
            result = subprocess.run([sys.executable, str(ROOT/'tools/tier-battery.py'), '--validate', str(path)], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            schema = json.loads((ROOT/'research/spill-d-20260919/telemetry.schema.json').read_text())
            samples[0]['devices'][0]['vram_bytes'] = 1.5
            with self.assertRaises(ValueError): B.validate_schema(samples[0], schema)
            result = subprocess.run([sys.executable, str(ROOT/'tools/tier-battery.py'), '--validate', str(ROOT/'research/spill-d-20260919/day2-dry-run/runs.jsonl')], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_storage_join_no_orphans_missing_or_relabeling(self):
        sample = dict(version=1, fixture='opaque', backend_requested='direct', backend_actual='buffered',
                      status='failed', valid_bytes=3, padded_bytes=4, io_bytes=0, physical_bytes=None,
                      queue_ns=None, io_ns=None, h2d_ns=None, d2h_ns=None, p2p_ns=None, total_ns=5,
                      inflight=0, pinned_bytes=0, pageable_bytes=None, fallbacks=1, payload_checksum=[0]*32)
        wrapped = {'run_id': 'a', 'sample': sample}
        joined = B.join_storage([{'run_id': 'a'}], [wrapped])
        self.assertEqual(joined[0]['samples'][0], sample)
        self.assertFalse(joined[0]['qualification'])
        for rows, samples in [([{'run_id': 'b'}], [wrapped]), ([{'run_id': 'a'}], []),
                              ([{'run_id': 'a'}, {'run_id': 'a'}], [wrapped])]:
            with self.assertRaises(ValueError): B.join_storage(rows, samples)
        bad = copy.deepcopy(wrapped); bad['sample']['io_bytes'] = True
        with self.assertRaises(ValueError): B.join_storage([{'run_id': 'a'}], [bad])

    def test_cell_start_survives_unexpected_interruption_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out = root/'cell'
            command = [sys.executable, '-c', "print('baseline')"]
            argv = [sys.executable, str(ROOT/'tools/tier-battery.py'), '--rig', 'rtx5090', '--out', str(out)]
            env = {**os.environ, 'PATH': str(root/'missing')}
            first = subprocess.run([*argv, '--execute', *command], env=env, capture_output=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            records = [json.loads(line) for line in (out/'CELL.jsonl').read_text().splitlines()]
            self.assertEqual([r['event'] for r in records], ['start', 'end'])
            self.assertIn('ended_utc', records[-1])
            # Simulate an interrupted cell: only its durable start survives.
            (out/'CELL.jsonl').write_text(json.dumps(records[0])+'\n')
            second = subprocess.run([*argv, '--resume', '--execute', *command], env=env, capture_output=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            resumed = next((out/'attempts').glob('*/CELL.jsonl'))
            rows = [json.loads(line) for line in resumed.read_text().splitlines()]
            self.assertEqual(rows[0]['resume_from']['last_event'], 'start')
            self.assertFalse(rows[-1]['qualification'])

    def test_first_hour_is_correctness_only_both_orders(self):
        plan = B.first_hour_plan()
        self.assertEqual(sum(s['budget_minutes'] for s in plan['stages']), 60)
        self.assertEqual([r['arms'] for r in plan['correctness_orders']], [['off', 'on'], ['on', 'off']])
        self.assertEqual(plan['performance_cells'], [])
        self.assertFalse(plan['performance_medians_allowed'])


if __name__ == '__main__':
    unittest.main()
