"""Single target-class card profile; CPU tests do not qualify hardware."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
LOCK = '/tmp/memra-gpu.lock'
import private_lock


class ProSingleTests(private_lock.PrivateLockMixin, unittest.TestCase):
    def collector(self, *args, env=None, flag=True):
        # The explicit flag accompanies the seam (and only the seam): a child that sees the
        # canonical environment gets neither.
        seamed = private_lock.SEAM in (os.environ if env is None else env)
        opt = [private_lock.FLAG] if flag and seamed else []
        return subprocess.run([sys.executable, str(ROOT/'tools/tier-battery.py'),
                               '--rig', 'pro-single', *opt, *args], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=30)

    def test_plan_and_schema(self):
        # The plan is a production surface: without the seam it names the canonical lock.
        result = self.collector('--plan', env=self.canonical_env())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['locks']['pro-single'], LOCK)
        seamed = self.collector('--plan')
        self.assertEqual(json.loads(seamed.stdout)['locks']['pro-single'], self.private(LOCK))
        schema = json.loads((ROOT/'research/spill-d-20260919/runs.schema.json').read_text())
        self.assertIn('pro-single', schema['properties']['rig']['enum'])

    def test_collector_acquires_canonical_lock_and_refuses_contention(self):
        # The canonical NAME under this test's private directory (ruling 12): the collector
        # takes the table's lock for the rig and refuses at once when it is held.
        lock = self.private(LOCK)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with open(lock, 'a') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = self.collector('--out', str(root/'refused'), '--execute',
                                        sys.executable, '-c', 'raise AssertionError("must not run")')
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse((root/'refused/command.log').exists())
            child = ('import fcntl; h=open(' + repr(lock) + ',"a"); '
                     '\ntry: fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)'
                     '\nexcept BlockingIOError: print("CANONICAL LOCK HELD")'
                     '\nelse: raise AssertionError("collector did not hold lock")')
            result = self.collector('--out', str(root/'accepted'), '--execute', sys.executable, '-c', child)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('CANONICAL LOCK HELD', (root/'accepted/command.log').read_text())
            self.assertEqual(json.loads((root/'accepted/lock.json').read_text()),
                             {'rig': 'pro-single', 'lock': lock, 'acquired': True, 'seam': str(self.lock_dir)})
            self.assertEqual(self.collector('--validate', str(root/'accepted')).returncode, 0)

    def bootstrap(self, out, source, flag=True, env=None):
        opt = [private_lock.FLAG] if flag else []
        return subprocess.run(['bash', '-s', '--', '--dry-run', *opt, '--rig', 'pro-single', '--out', str(out)],
                              input=source, cwd=ROOT, capture_output=True, text=True, timeout=30,
                              env={**(os.environ if env is None else env), 'BRANCH': 'lane/spill-d-test'})

    def test_seam_without_the_explicit_flag_refuses_before_any_lock_or_receipt(self):
        # An inherited MEMRA_TIER_BATTERY_LOCK_DIR alone never moves a campaign or a bootstrap
        # off the canonical lock: without --private-lock-dir-for-tests both refuse before they
        # create a directory or open a lock, and the flag without the seam refuses too.
        lock = self.private(LOCK)
        source = (ROOT/'tools/tier-rig-bootstrap.sh').read_text()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refused = self.collector('--out', str(root/'cell'), '--execute', sys.executable, '-c',
                                     'raise AssertionError("must not run")', flag=False)
            self.assertEqual(refused.returncode, 2, refused.stderr)
            self.assertIn('REFUSED: MEMRA_TIER_BATTERY_LOCK_DIR is set but --private-lock-dir-for-tests was not passed', refused.stderr)
            self.assertIn('PRIVATE lock directory (test seam); the rig lock is NOT held', refused.stderr)
            self.assertFalse((root/'cell').exists())
            dry = self.collector('--dry-run', '--out', str(root/'dry'), flag=False)
            self.assertEqual(dry.returncode, 2, dry.stderr)
            self.assertFalse((root/'dry').exists())
            with open(lock, 'a') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)  # nothing above took it
            canonical = self.canonical_env()
            flag_only = subprocess.run([sys.executable, str(ROOT/'tools/tier-battery.py'), '--rig', 'pro-single',
                                        private_lock.FLAG, '--plan'], env=canonical, capture_output=True, text=True, timeout=30)
            self.assertEqual(flag_only.returncode, 2, flag_only.stderr)
            self.assertIn('REFUSED: --private-lock-dir-for-tests without MEMRA_TIER_BATTERY_LOCK_DIR', flag_only.stderr)
            # --plan and --validate take no lock: they run under the seam without the flag.
            self.assertEqual(self.collector('--plan', flag=False).returncode, 0)
            boot = self.bootstrap(root/'boot', source, flag=False)
            self.assertEqual(boot.returncode, 2, boot.stderr)
            self.assertIn('REFUSED: MEMRA_TIER_BATTERY_LOCK_DIR is set but --private-lock-dir-for-tests was not passed', boot.stderr)
            self.assertFalse((root/'boot').exists())
            boot_flag_only = self.bootstrap(root/'boot-flag', source, env=self.canonical_env())
            self.assertEqual(boot_flag_only.returncode, 2, boot_flag_only.stderr)
            self.assertIn('REFUSED: --private-lock-dir-for-tests without MEMRA_TIER_BATTERY_LOCK_DIR', boot_flag_only.stderr)
            self.assertFalse((root/'boot-flag').exists())

    def test_bootstrap_stub_memory_match_and_wrapper(self):
        source = (ROOT/'tools/tier-rig-bootstrap.sh').read_text()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.bootstrap(root/'good', source)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root/'good/BOOTSTRAP.json').read_text())
            self.assertEqual(report['rig'], 'pro-single')
            # The tracked script names the canonical lock; under the seam the report and the
            # wrapper carry the same name in this test's private directory.
            self.assertIn("'lock': '/tmp/memra-gpu.lock'", source)
            self.assertEqual(report['lock'], self.private(LOCK))
            self.assertEqual(report['lock_seam'], str(self.lock_dir))
            self.assertIn('PRIVATE lock directory (test seam); the rig lock is NOT held', result.stderr)
            self.assertEqual(Path(report['lock']).name, 'memra-gpu.lock')
            self.assertFalse(report['qualification'])
            self.assertEqual(report['status'], 'dry-run-complete-not-qualified')
            self.assertIn('RTX PRO 6000 Blackwell', report['power_verbatim'])
            self.assertIn('exec flock --close -n -x '+self.private(LOCK), (root/'good/locked-run.sh').read_text())
            for label, old, new in [('memory', '97887, 600.00, 600.00', '89999, 600.00, 600.00'),
                                    ('card', 'NVIDIA RTX PRO 6000 Blackwell Server Edition', 'NVIDIA GeForce RTX 5090'),
                                    ('power', '97887, 600.00, 600.00', '97887, 600.00, 599.00')]:
                result = self.bootstrap(root/label, source.replace(old, new))
                self.assertEqual(result.returncode, 2, result.stderr)
                report = json.loads((root/label/'BOOTSTRAP.json').read_text())
                self.assertNotIn('accept-1', [s['name'] for s in report['steps']])


if __name__ == '__main__':
    unittest.main()
