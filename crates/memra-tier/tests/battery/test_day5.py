"""First-rental regressions. Stub runs/parser checks are not GPU qualification."""
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]

def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

B = load('tier-battery')
T = load('tier-topology')
REAL = ROOT / 'research/spill-lead-20260919/rented-5090-20260919/receipts'


class Day5Tests(unittest.TestCase):
    def bootstrap(self, out, extra=(), source=None):
        return subprocess.run(['bash', '-s', '--', '--dry-run', '--out', str(out), *extra],
                              input=source or (ROOT/'tools/tier-rig-bootstrap.sh').read_text(),
                              text=True, cwd=ROOT, capture_output=True,
                              env={**os.environ, 'BRANCH': 'lane/spill-d-test'}, timeout=30)

    def test_nvcc_base_arch_and_real_compile_are_both_required(self):
        original = (ROOT/'tools/tier-rig-bootstrap.sh').read_text()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(self.bootstrap(root/'base').returncode, 0)
            r = json.loads((root/'base/BOOTSTRAP.json').read_text())
            self.assertEqual((root/'base/nvcc-arches.log').read_text(), 'compute_120\n')
            step = next(s for s in r['steps'] if s['name'] == 'cuda-compile')
            self.assertIn('-arch=sm_120a', step['command'])
            bad = original.replace('code = 0', "code = 1 if label == 'cuda-compile' else 0")
            self.assertEqual(self.bootstrap(root/'bad', source=bad).returncode, 2)
            r = json.loads((root/'bad/BOOTSTRAP.json').read_text())
            self.assertNotIn('accept-1', [s['name'] for s in r['steps']])

    def test_bootstrap_overlay_default_refuses_opt_in_preserves_raw(self):
        source = (ROOT/'tools/tier-rig-bootstrap.sh').read_text()
        source = source.replace("'/dev/nvme0n1\\n'", "'overlay\\n'")
        source = source.replace("'nvme0n1\\n'", "'lsblk: overlay: not a block device\\n'")
        source = source.replace('code = 0', "code = 32 if label == 'nvme-ancestry' else 0")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mode, extra, code in [('default', (), 2), ('allowed', ('--allow-unproven-storage',), 0)]:
                out = root/mode
                proc = self.bootstrap(out, ('--nvme-root', str(root), *extra), source)
                self.assertEqual(proc.returncode, code, proc.stderr)
                r = json.loads((out/'BOOTSTRAP.json').read_text())
                self.assertIsNone(r['paths']['nvme'])
                self.assertFalse(r['storage']['nvme_proven'])
                self.assertEqual(r['storage']['label'], B.UNPROVEN_STORAGE)
                self.assertEqual((out/'nvme-ancestry.log').read_text(), 'lsblk: overlay: not a block device\n')
                self.assertTrue((out/'storage-findmnt.log').is_file())
                self.assertTrue((out/'storage-lsblk.log').is_file())
                self.assertEqual(r['storage']['allow_unproven_storage'], bool(extra))
            self.assertNotEqual(self.bootstrap(root/'missing-path', ('--allow-unproven-storage',)).returncode, 0)

    def test_pidfile_status_no_self_match_stale_pid_or_new_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); pidfile = root/'bootstrap.pid'
            argv = ['bash', str(ROOT/'tools/tier-rig-bootstrap.sh'), '--status', '--pidfile', str(pidfile)]
            # The shell command contains the script name twice, as an SSH launcher does.
            def status():
                return subprocess.run(['bash', '-c', '"$@" # tier-rig-bootstrap.sh', '_', *argv],
                                      capture_output=True, text=True)
            self.assertEqual(status().returncode, 1)
            self.assertFalse(pidfile.exists())
            pidfile.write_text(json.dumps({'pid': os.getpid()}))
            self.assertEqual(status().returncode, 1)  # live but unlocked/recycled PID
            with pidfile.open('r+') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual(status().returncode, 0)
                duplicate = self.bootstrap(root/'duplicate', ('--pidfile', str(pidfile)))
                self.assertNotEqual(duplicate.returncode, 0)
                self.assertFalse((root/'duplicate').exists())
                self.assertIn('already running', duplicate.stderr)
            self.assertEqual(status().returncode, 1)
            proc = self.bootstrap(root/'new', ('--pidfile', str(pidfile)))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(status().returncode, 1)
            argv[2] = '--already-running'
            self.assertEqual(status().returncode, 1)

    def test_jobs_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertNotEqual(self.bootstrap(Path(tmp)/'out', ('--jobs', '32')).returncode, 0)

    def stub_storage(self, command, raw, **kwargs):
        if command[0] == 'findmnt':
            raw.write_text('overlay\n' if 'SOURCE' in command else '{"filesystems":[{"fstype":"overlay"}]}\n')
            return 0, False
        if '-s' in command:
            raw.write_text('lsblk: overlay: not a block device\n')
            return 32, False
        raw.write_text('{"blockdevices":[]}\n')
        return 0, False

    def test_storage_collector_opt_in_and_timing_reach_both_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            no = root/'no'; no.mkdir()
            with patch.object(B, 'tee_run', self.stub_storage):
                with self.assertRaises(ValueError): B.capture_storage(root, no)
            self.assertTrue((no/'STORAGE.json').exists())
            out = root/'yes'; out.mkdir()
            with patch.object(B, 'tee_run', self.stub_storage):
                storage = B.capture_storage(root, out, True)
            record = B.SubprocessRunner(str(root/'no-smi')).run(
                [sys.executable, '-c', "print('Error: deliberate failure'); raise SystemExit(9)"],
                out/'command.log', echo=False, storage=storage)
            self.assertEqual(record['failure_quote'], 'Error: deliberate failure')
            rows = [json.loads(line) for line in (out/'CELL.jsonl').read_text().splitlines()]
            for row in [record, *rows]:
                self.assertEqual(row['storage']['label'], B.UNPROVEN_STORAGE)
                self.assertFalse(row['storage']['nvme_proven'])
            self.assertEqual(rows[1]['elapsed_seconds'], record['elapsed_seconds'])
            self.assertEqual(record['elapsed_seconds'], rows[1]['duration_ns']/1e9)
            self.assertEqual(rows[1]['ended_utc'], record['ended_utc'])
            self.assertEqual(rows[0]['started_utc'], record['started_utc'])
            (out/'lock.json').write_text(json.dumps({'rig':'rtx5090','lock':B.LOCKS['rtx5090'],'acquired':True}))
            result = B.validate_cell(out/'CELL.jsonl')
            self.assertEqual(result['status'], 'failed')
            self.assertFalse(result['qualification'])
            bad = copy.deepcopy(record); bad['elapsed_seconds'] = -1
            with self.assertRaises(ValueError): B.validate_capture(bad, out)
            (out/'storage-ancestry.log').write_text('tampered')
            with self.assertRaises(ValueError): B.validate_capture(record, out)

    def test_storage_cli_default_refuses_then_explicit_mode_is_labeled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root/'storage-bench'
            fake.write_text('#!' + sys.executable + '\nprint("CPU stub, no storage performance")\n')
            fake.chmod(0o755)
            argv = [sys.executable, str(ROOT/'tools/tier-battery.py'), '--rig', 'rtx5090']
            env = {**os.environ, 'PATH': str(root/'no-tools')}
            for mode, options, expected in [
                    ('unspecified', [], 2),
                    ('unproven', ['--storage-root', str(root)], 2),
                    ('development', ['--storage-root', str(root), '--allow-unproven-storage'], 0)]:
                out = root/mode
                proc = subprocess.run([*argv, '--out', str(out), *options, '--execute', str(fake),
                                      'roundtrip', str(root/'object'), '264', 'buffered'],
                                      env=env, text=True, capture_output=True, timeout=10)
                self.assertEqual(proc.returncode, expected, proc.stderr)
                if expected:
                    self.assertFalse((out/'command.log').exists())
                else:
                    result = B.validate_cell(out/'CELL.jsonl')
                    self.assertEqual(result['storage_label'], B.UNPROVEN_STORAGE)
                    self.assertFalse(result['qualification'])
                    self.assertFalse(result['legacy_timing'])

    def test_real_first_hour_integrity_keeps_failed_cell_and_empty_telemetry(self):
        cells = sorted(REAL.rglob('CELL.jsonl'))
        self.assertEqual(len(cells), 9)
        results = [B.validate_cell(p) for p in cells]
        self.assertEqual(sum(r['status'] == 'failed' for r in results), 1)
        self.assertTrue(all(r['legacy_timing'] and not r['qualification'] for r in results))
        proc = subprocess.run([sys.executable, str(ROOT/'tools/tier-battery.py'), '--validate', str(REAL)],
                              text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)['failed_commands'], 1)
        explicit = subprocess.run([sys.executable, str(ROOT/'tools/tier-battery.py'), '--schema', 'runs', '--validate', str(cells[0])], capture_output=True)
        self.assertNotEqual(explicit.returncode, 0)
        # The strict byte gate must still refuse these capture rows.
        rows = [json.loads(s) for s in cells[0].read_text().splitlines()]
        with self.assertRaises(ValueError): B.validate_rows(rows, cells[0].parent)

    def test_capture_validation_rejects_corruption_and_missing_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            import shutil
            source = next(REAL.glob('first-hour*/d1-local'))
            shutil.copytree(source, root/'cell'); out = root/'cell'
            path = out/'CELL.jsonl'
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            rows[1]['duration_ns'] += 1
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError): B.validate_cell(path)
            path.write_text(json.dumps(rows[0])+'\n')
            with self.assertRaises(ValueError): B.validate_cell(path)
            record = json.loads((out/'command.capture.json').read_text())
            record['raw_log']['path'] = '../command.log'
            with self.assertRaises(ValueError): B.validate_capture(record, out)

    def test_real_bootstrap_contract_and_archived_binary_manifest(self):
        r = json.loads(next(REAL.glob('*run3/BOOTSTRAP.json')).read_text())
        first = next(REAL.glob('first-hour*'))
        self.assertEqual(r['status'], 'bootstrap-complete-not-tier-qualified')
        self.assertEqual(r['cuda_acceptance'], 'two-full-readbacks')
        self.assertFalse(r['qualification'])
        self.assertEqual(r['source_commit'], (first/'source.commit').read_text().strip())
        pins = {Path(line.split()[1]).name: line.split()[0]
                for line in (first/'binaries.sha256').read_text().splitlines()}
        self.assertEqual(pins, {name: v['sha256'] for name, v in r['release_binaries'].items()})
        self.assertEqual(len(pins), 6)
        self.assertTrue(all(step['exit_code'] == 0 for step in r['steps']))

    def test_real_topology_fixture_current_is_not_max(self):
        fixture = json.loads((Path(__file__).parent/'topology.rented-5090.fixture.json').read_text())
        report = T.probe(fixture)
        self.assertEqual(report['kind'], 'cpu-fixture')
        self.assertFalse(report['route_qualification'])
        self.assertEqual(report['pcie_links'], [{'device_ordinal':0, 'generation_current':1,
            'generation_max':4, 'device_generation_max':5, 'host_generation_max':4,
            'width_current':16, 'width_max':16, 'bandwidth_measured':False}])
        self.assertEqual(T.pcie_links('unparseable/missing'), [])

    def test_spot_recovery_requires_per_cell_sync_and_pinned_redownload(self):
        doc = (ROOT/'research/spill-d-20260919/RIG-DAY1.md').read_text()
        for phrase in ('after EVERY cell', 'Required resources are currently unavailable',
                       'immutable pinned locator', 'pushed\n  branch/commit', 'receipt state'):
            self.assertIn(phrase, doc)

    def test_c_runbook_stages_existing_goldens_not_a_new_oracle(self):
        doc = (ROOT/'research/spill-d-20260919/RIG-DAY1.md').read_text()
        self.assertIn('cp research/qwen4exp-bringup-20260829/gpu-eager/bank-bytes-goldens.tsv', doc)
        self.assertIn('sha256sum "$EV/c-ple-tiny/receipt/bank-bytes-goldens.tsv"', doc)
        self.assertIn('target/release/qwen4exp_gpu_gate "$EV/c-ple-tiny/receipt/ple-tiny.tsv"', doc)


if __name__ == '__main__':
    unittest.main()
