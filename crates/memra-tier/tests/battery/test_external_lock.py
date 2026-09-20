"""Inherited-flock lifetime and scoped process teardown, CPU-only."""
import fcntl
import importlib.util
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('external_battery', ROOT/'tools/tier-battery.py')
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
PROOF = ROOT/'tools/tier-lock-proof.py'


class ExternalLockTests(unittest.TestCase):
    def check_proof(self, fd, inherited=(), path='/tmp/memra-5090.lock'):
        return subprocess.run([sys.executable, str(PROOF), '--fd', str(fd), '--lock', path],
                              pass_fds=inherited, text=True, capture_output=True, timeout=5)

    def test_owning_fd_only_not_same_inode_foreign_closed_missing_or_unlocked(self):
        with B.campaign_lock('rtx5090', inherit=True) as owner:
            good = self.check_proof(owner.fileno(), (owner.fileno(),))
            self.assertEqual(good.returncode, 0, good.stderr)
            self.assertEqual(json.loads(good.stdout)['owner'], 'collector')
            with open(B.LOCKS['rtx5090'], 'a') as same_inode:
                result = self.check_proof(same_inode.fileno(), (same_inode.fileno(),))
                self.assertEqual(result.returncode, 2)
                self.assertIn('does not own', result.stderr)
            with tempfile.TemporaryFile() as foreign:
                self.assertEqual(self.check_proof(foreign.fileno(), (foreign.fileno(),)).returncode, 2)
            self.assertEqual(self.check_proof(owner.fileno()).returncode, 2)
            self.assertEqual(self.check_proof(-1).returncode, 2)
            with self.assertRaises(BlockingIOError):
                with B.campaign_lock('rtx5090'): pass
        with open(B.LOCKS['rtx5090'], 'a') as unlocked:
            result = self.check_proof(unlocked.fileno(), (unlocked.fileno(),))
            self.assertEqual(result.returncode, 2)
            self.assertIn('not locked', result.stderr)
        missing = subprocess.run([sys.executable, str(PROOF), '--lock', B.LOCKS['rtx5090']],
                                 capture_output=True, timeout=5)
        self.assertEqual(missing.returncode, 2)

    def test_parent_close_does_not_unlock_live_descendant(self):
        child = None
        try:
            with B.campaign_lock('rtx5090', inherit=True) as owner:
                child = subprocess.Popen([sys.executable, '-c',
                    'import sys; print("holding",flush=True); sys.stdin.read()'],
                    pass_fds=(owner.fileno(),), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                self.assertEqual(child.stdout.readline().strip(), 'holding')
            with self.assertRaises(BlockingIOError):
                with B.campaign_lock('rtx5090'): pass
            child.communicate('', timeout=5)
            with B.campaign_lock('rtx5090'): pass
        finally:
            if child is not None:
                if child.poll() is None: child.kill()
                child.communicate(timeout=5)

    def collect(self, root, child, timeout=5):
        return subprocess.run([sys.executable, str(ROOT/'tools/tier-battery.py'),
            '--rig', 'rtx5090', '--timeout', str(timeout), '--out', str(root/'cell'),
            '--external-lock', '--execute', *child], capture_output=True, text=True,
            env={**os.environ, 'PATH': str(root/'no-tools')}, timeout=timeout+10)

    def test_collector_inheritance_integrity_and_foreign_launch_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child = [sys.executable, str(PROOF), '--fd', '@COLLECTOR_LOCK_FD@',
                     '--lock', B.LOCKS['rtx5090']]
            with B.campaign_lock('rtx5090'):
                result = self.collect(root, child)
                self.assertEqual(result.returncode, 2)
                self.assertFalse((root/'cell/command.log').exists())
            (root/'cell').rmdir()
            result = self.collect(root, child)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((root/'cell/command.log').read_text())['owner'], 'collector')
            self.assertFalse(B.validate_cell(root/'cell/CELL.jsonl')['qualification'])
            (root/'cell/lock.json').write_text('{}\n')
            with self.assertRaises(ValueError): B.validate_cell(root/'cell/CELL.jsonl')

    def test_legacy_fragment_applies_preserves_assertions_and_internal_external_proofs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); tools = root/'tools'; tools.mkdir()
            names = ['kv-host-spill-identity-gate.sh', 'kv-host-spill-failure-gate.sh']
            for name in [*names, 'tier-lock-proof.py', 'port-guard.sh']:
                shutil.copy(ROOT/'tools'/name, tools/name)
            patch = ROOT/'research/spill-d-20260919/LEGACY-EXTERNAL-LOCK.diff'
            # The authorized fragment is now applied in-tree. Reverse/reapply in
            # the disposable tree to keep its historical assertion-preservation teeth.
            subprocess.run(['git', 'apply', '--reverse', str(patch)], cwd=root, check=True, capture_output=True)
            originals = {name: (tools/name).read_text() for name in names}
            subprocess.run(['git', 'apply', str(patch)], cwd=root, check=True, capture_output=True)
            for i, name in enumerate(names):
                script = tools/name
                source = script.read_text()
                marker = '# Two DISJOINT' if i == 0 else '# Same fixtures'
                original = originals[name]
                self.assertEqual(source[source.index(marker):], original[original.index(marker):])
                self.assertNotIn('pkill', source)
                self.assertNotIn('flock -w', source)
                subprocess.run(['bash', '-n', str(script)], check=True, capture_output=True)
                if shutil.which('shellcheck'):
                    checked = subprocess.run(['shellcheck', '-x', str(script)], cwd=tools, text=True, capture_output=True)
                    self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
                # Exercise exact patched prologue; no server or model command runs.
                script.write_text(source[:source.index('SERVER_PID=""')]+'echo "LOCK PROOF ACCEPTED"\n')
                internal = root/(str(i)+'-internal')
                result = subprocess.run(['bash', str(script), 'unused-model', 'unused-binary', str(internal)],
                                        capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads((internal/'LOCK.json').read_text())['owner'], 'internal-canonical')
                external = root/(str(i)+'-external')
                cellroot = root/(str(i)+'-capture'); cellroot.mkdir()
                # bash is absolute because the collector test PATH hides nvidia-smi,
                # while the script needs ordinary CPU utilities for proof checking.
                command = [sys.executable, str(ROOT/'tools/tier-battery.py'), '--rig', 'rtx5090',
                    '--out', str(cellroot/'cell'), '--external-lock', '--execute',
                    shutil.which('bash'), str(script), '--external-lock', '@COLLECTOR_LOCK_FD@',
                    'unused-model', 'unused-binary', str(external)]
                result = subprocess.run(command, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads((external/'LOCK.json').read_text())['owner'], 'collector')
                refused = root/(str(i)+'-refused')
                result = subprocess.run(['bash', str(script), '--external-lock', '12345',
                    'unused-model', 'unused-binary', str(refused)], capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(refused.exists())

    def test_failure_success_and_timeout_kill_only_command_group(self):
        # The descendant closes stdout; it must not evade teardown by letting the
        # ordinary raw-log drain finish. No process-name/global pkill is involved.
        for mode in ('success', 'failure', 'timeout'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); marker = root/'survived'
                child_program = 'import time; from pathlib import Path; time.sleep(3); Path('+repr(str(marker))+').write_text("bad")'
                script = ('import subprocess,sys,time\n'
                    'fd=int(sys.argv[1])\n'
                    'subprocess.Popen([sys.executable,"-c",'+repr(child_program)+'], '
                    'stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,pass_fds=(fd,))\n'
                    + ('time.sleep(20)\n' if mode == 'timeout' else '')
                    + ('sys.exit(7)\n' if mode == 'failure' else ''))
                sentinel = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'])
                try:
                    result = self.collect(root, [sys.executable, '-c', script, '@COLLECTOR_LOCK_FD@'], timeout=1)
                    self.assertEqual(result.returncode, {'success':0,'failure':7,'timeout':2}[mode], result.stderr)
                    self.assertIsNone(sentinel.poll())
                    # Killed descendants may retain an FD until the kernel schedules
                    # final release; bounded retries test eventual inode release.
                    acquired = False
                    for _ in range(50):
                        try:
                            with B.campaign_lock('rtx5090'): acquired = True
                            break
                        except BlockingIOError: time.sleep(.02)
                    self.assertTrue(acquired)
                    self.assertFalse(marker.exists())
                finally:
                    sentinel.kill(); sentinel.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
