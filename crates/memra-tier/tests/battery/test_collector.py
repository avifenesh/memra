"""Collector protocol tests: synthetic only, no hardware qualification."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
def load(name):
    spec=importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name+'.py'))
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
B=load('tier-battery'); P=load('tier-placement'); T=load('tier-topology')
import private_lock

class CollectorTests(private_lock.PrivateLockMixin, unittest.TestCase):
    BATTERY = B
    def test_interleaved_both_orders_and_minimum(self):
        self.assertEqual(B.paired_orders(5),[(i,o) for i in range(5) for o in ('AB','BA')])
        for n in [0,4,True]:
            with self.assertRaises(ValueError): B.paired_orders(n)

    def test_canonical_lock_exclusion_no_third_name(self):
        # The production table is exactly two names (CLAUDE.md, lock names are a correctness
        # surface); the ruling-12 seam puts the SAME two names in this test's private directory.
        self.assertEqual(B.CANONICAL_LOCKS, {'rtx5090': '/tmp/memra-5090.lock', 'pro-single': '/tmp/memra-gpu.lock',
                                             'pro-pair': '/tmp/memra-gpu.lock', 'pro-four': '/tmp/memra-gpu.lock', 'cpu': None})
        self.assertEqual(B.lock_table(), B.CANONICAL_LOCKS)
        self.assertEqual(B.LOCKS['pro-pair'], self.private('/tmp/memra-gpu.lock'))
        self.assertEqual({Path(v).name for v in B.LOCKS.values() if v}, {'memra-5090.lock', 'memra-gpu.lock'})
        with B.campaign_lock('pro-pair') as lock:
            self.assertEqual(lock, B.LOCKS['pro-pair'])
            self.assertEqual(Path(lock).name, 'memra-gpu.lock')
            with self.assertRaises(BlockingIOError):
                with B.campaign_lock('pro-four'): pass
        with self.assertRaises(ValueError):
            with B.campaign_lock('invented'): pass
        with self.assertRaises(ValueError):
            B.lock_table(self.lock_dir/'absent')

    def test_private_lock_seam_receipts_never_validate_against_the_canonical_table(self):
        # A cell captured under MEMRA_TIER_BATTERY_LOCK_DIR records the private path, and the
        # same receipt REFUSES to validate in a process without the seam: the seam can never
        # produce a receipt that reads as a rig-locked campaign.
        with tempfile.TemporaryDirectory(prefix='tier-seam-') as tmp:
            out = Path(tmp)/'cell'
            argv = [sys.executable, str(ROOT/'tools/tier-battery.py'), '--rig', 'rtx5090']
            env = {**os.environ, 'PATH': str(Path(tmp)/'no-tools')}
            # The seam alone refuses to run a campaign: the explicit flag is the test's statement.
            unflagged = subprocess.run([*argv, '--out', str(out), '--execute', sys.executable, '-c', 'pass'],
                                       env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(unflagged.returncode, 2, unflagged.stderr)
            self.assertIn('REFUSED: MEMRA_TIER_BATTERY_LOCK_DIR is set but --private-lock-dir-for-tests was not passed', unflagged.stderr)
            self.assertFalse(out.exists())
            proc = subprocess.run([*argv, private_lock.FLAG, '--out', str(out), '--execute', sys.executable, '-c', "print('seam cell')"],
                                  env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn(f'tier-battery: MEMRA_TIER_BATTERY_LOCK_DIR={self.lock_dir}: PRIVATE lock directory (test seam); the rig lock is NOT held by this process', proc.stderr)
            self.assertEqual(json.loads((out/'lock.json').read_text()),
                             {'rig': 'rtx5090', 'lock': self.private('/tmp/memra-5090.lock'), 'acquired': True,
                              'seam': str(self.lock_dir)})
            validate = [sys.executable, str(ROOT/'tools/tier-battery.py'), '--validate', str(out)]
            seamed = subprocess.run(validate, capture_output=True, text=True, timeout=30)
            self.assertEqual(seamed.returncode, 0, seamed.stderr)
            refused = subprocess.run(validate, env=self.canonical_env(), capture_output=True, text=True, timeout=30)
            self.assertEqual(refused.returncode, 2, refused.stderr)
            self.assertIn(f"REFUSED: lock.json seam={str(self.lock_dir)!r} is not this process's MEMRA_TIER_BATTERY_LOCK_DIR=None", refused.stderr)
            # A different seam is not this seam either.
            with tempfile.TemporaryDirectory(prefix='tier-other-seam-') as other:
                foreign = subprocess.run(validate, env={**os.environ, private_lock.SEAM: other}, capture_output=True, text=True, timeout=30)
                self.assertEqual(foreign.returncode, 2, foreign.stderr)
                self.assertIn('lock.json seam=', foreign.stderr)
            # The in-process dry run records the seam in its manifest and validates only under it.
            bundle = Path(tmp)/'dry'
            B.run_dry_campaign(bundle, echo=False)
            self.assertEqual(json.loads((bundle/'manifest.json').read_text())['seam'], str(self.lock_dir))
            self.assertEqual(B.validate_campaign(bundle), 22)
            with self.canonical_locks():
                with self.assertRaises(ValueError): B.validate_campaign(bundle)

    def test_dry_runner_full_campaign_raw_hashes_failure_and_n(self):
        with tempfile.TemporaryDirectory(prefix='tier-collector-') as tmp:
            out=Path(tmp)/'bundle'
            summary=B.run_dry_campaign(out,echo=False)
            self.assertEqual(summary['status'],'dry-run-not-qualification')
            self.assertEqual(B.validate_campaign(out),22)
            self.assertEqual(summary['arms']['on']['N'],10)
            self.assertEqual(summary['arms']['off']['N'],10)
            self.assertEqual(summary['thermal_regime'],'synthetic-no-thermal-measurement')
            rows=[json.loads(l) for l in (out/'runs.jsonl').read_text().splitlines()]
            self.assertEqual(B.validate_rows(rows,out),11)
            runs=[json.loads(l) for l in (out/'collector.jsonl').read_text().splitlines()]
            self.assertEqual(len(runs),22)
            for r in runs:
                path=B.evidence(out,r['telemetry'])
                B.validate_telemetry([json.loads(l) for l in path.read_text().splitlines()],'cpu-fixture')
                B.evidence(out,r['raw_log'])
            f=json.loads((out/'failures.jsonl').read_text())
            self.assertEqual(f['exit_code'],9)
            self.assertEqual(f['failure_quote'],'ERROR: injected read failure (CPU fixture)')
            self.assertIn(f['failure_quote'],B.evidence(out,f['raw_log']).read_text())
            manifest=json.loads((out/'manifest.json').read_text())
            self.assertTrue(manifest['lock_acquired'])
            self.assertEqual(manifest['clock'],'virtual-monotonic')
            # Every simulated arm must remain the same program/control.
            self.assertEqual(len({r['state']['sha256'] for r in runs}),1)
            altered=copy.deepcopy(runs); altered[2],altered[3]=altered[3],altered[2]
            (out/'collector.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in altered))
            with self.assertRaises(ValueError):B.validate_campaign(out)
            (out/'collector.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in runs))
            summary['arms']['on']['N']=9
            (out/'summary.json').write_text(json.dumps(summary))
            with self.assertRaises(ValueError):B.validate_campaign(out)

    def test_telemetry_schema_and_red_controls(self):
        samples=[B.sample_fake(i*B.INTERVAL_NS,3*i) for i in range(3)]
        B.validate_telemetry(samples,'cpu-fixture')
        schema=json.loads((ROOT/'research/spill-d-20260919/telemetry.schema.json').read_text())
        self.assertEqual(set(schema['required']),set(samples[0]))
        for mutation in [lambda s:s.pop('host'),lambda s:s['devices'][0]['routes'].pop('pcie-p2p'),lambda s:s.update(monotonic_ns=-1),lambda s:s.update(monotonic_ns=2_000_000_000),lambda s:s['wait_ns']['queue'].update(p50=900),lambda s:s.update(kind='gpu'),lambda s:s['nvme'].update(read_bytes=-1),lambda s:s['nvme'].update(read_bytes=100)]:
            bad=copy.deepcopy(samples);mutation(bad[1])
            with self.assertRaises((ValueError,KeyError)):B.validate_telemetry(bad,'cpu-fixture')
        with self.assertRaises(ValueError):B.percentiles([])

    def test_sampler_calls_real_cadence_reader_and_stops(self):
        def read(ns):
            sample=B.sample_fake(ns,0)
            if len(sampler.samples)==2:sampler.stop.set()
            return sample
        sampler=B.Sampler(read)
        thread=threading.Thread(target=sampler.run);thread.start();thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertIsNone(sampler.error)
        self.assertEqual(len(sampler.samples),3)
        B.validate_telemetry(sampler.samples,'cpu-fixture')

    def test_raw_timeout_and_unknown_cause_preserved(self):
        with tempfile.TemporaryDirectory(prefix='tier-tee-') as tmp:
            log=Path(tmp)/'raw.log'
            # Allow interpreter startup under concurrent CPU builds before testing
            # the timeout/drain contract; 100 ms can expire before the first print.
            code,timeout=B.tee_run([sys.executable,'-c',"import time; print('before timeout',flush=True); time.sleep(10)"],log,timeout=1,echo=False)
            self.assertTrue(timeout); self.assertNotEqual(code,0)
            self.assertIn('before timeout',log.read_text())

    def test_descendant_stdout_cannot_hang_collector(self):
        with tempfile.TemporaryDirectory(prefix='tier-tee-child-') as tmp:
            code,timeout=B.tee_run([sys.executable,'-c',"import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)']); print('parent exiting',flush=True)"],Path(tmp)/'raw.log',timeout=2,echo=False)
            self.assertTrue(timeout)
            self.assertEqual(code,0) # parent success is NOT whole-run success

    def test_topology_fixture_and_missing_tools(self):
        fixture=json.loads((Path(__file__).parent/'topology.fixture.json').read_text())
        result=T.probe(fixture)
        self.assertFalse(result['route_qualification'])
        self.assertEqual(len(result['commands']),6)
        self.assertEqual(result['commands'][0]['stdout'],fixture['nvidia-smi topo -m']['stdout'])
        with mock.patch.object(T.subprocess,'run',side_effect=FileNotFoundError('absent')):
            result=T.probe()
            self.assertTrue(all(c['exit_code'] is None for c in result['commands']))
        with self.assertRaises(ValueError):T.probe({})

    def test_placement_both_census_tables_frontiers_and_override(self):
        tables=P.tables()
        a,b=tables['candidates']
        self.assertEqual(a['weights'],[76_367_322_992,74_062_760_176,73_925_697_008,83_172_210_424])
        self.assertEqual(b['weights'],[68_977_822_104,81_452_261_064,66_536_196_120,90_561_711_312])
        for c in (a,b):self.assertEqual(sum(c['weights']),307_527_990_600)
        self.assertEqual(P.frontier(P.SPLITS[0],652),{'first_N':29,'device':0,'first_context':1_037_290,'first_8192_step':1_040_384})
        self.assertEqual(P.frontier(P.SPLITS[1],652)['first_N'],40)
        self.assertEqual(P.frontier(P.SPLITS[1],652)['first_context'],1_035_194)
        self.assertEqual(P.frontier(P.SPLITS[1],356)['first_N'],73)
        self.assertEqual(P.frontier(P.SPLITS[1],652,4_000_000_000)['first_N'],31)
        self.assertEqual(P.row(P.SPLITS[0],652,3,1)['global_bytes'],3912)
        self.assertEqual(len(P.tables(264)['candidates'][0]['grid']),6)
        with self.assertRaises(ValueError):P.tables(0)
        p=subprocess.run([sys.executable,str(ROOT/'tools/tier-placement.py'),'--record-bytes','264','--json'],capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(p.stdout)['candidates'][0]['grid'][0]['record_bytes'],264)

if __name__=='__main__':unittest.main(verbosity=2)
