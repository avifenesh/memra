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
    def collector(self, *args, env=None):
        return subprocess.run([sys.executable, str(ROOT/'tools/tier-battery.py'),
                               '--rig', 'pro-single', *args], cwd=ROOT, env=env,
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
                             {'rig': 'pro-single', 'lock': lock, 'acquired': True})
            self.assertEqual(self.collector('--validate', str(root/'accepted')).returncode, 0)

    def bootstrap(self, out, source):
        return subprocess.run(['bash', '-s', '--', '--dry-run', '--rig', 'pro-single', '--out', str(out)],
                              input=source, cwd=ROOT, capture_output=True, text=True, timeout=30,
                              env={**os.environ, 'BRANCH': 'lane/spill-d-test'})

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
