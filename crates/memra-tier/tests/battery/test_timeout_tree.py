"""CPU-only nested timeout/lock tests; anonymous inode, never a third rig lock."""
import fcntl
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location('timeout_battery', ROOT/'tools/tier-battery.py')
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


class TimeoutTreeTests(unittest.TestCase):
    def run_tree(self, detached=False, inner_timeout=30, outer_timeout=.6):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inode = root/'fixture.data'
            lock = inode.open('w+b')
            pidfile = root/'grandchild.pid'
            marker = root/'survived'
            # An independent open description lets us check the inherited flock lifetime.
            independent = os.open(inode, os.O_RDWR)
            inode.unlink()
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                grandchild = (f'import os,time; from pathlib import Path; '
                              f'Path({str(pidfile)!r}).write_text(str(os.getpid())); '
                              f'time.sleep(1.2); Path({str(marker)!r}).write_text("escaped"); time.sleep(30)')
                child = ('import subprocess,sys,time; '
                         f'p=subprocess.Popen([sys.executable,"-c",{grandchild!r}],pass_fds=({lock.fileno()},)); '
                         'p.wait()')
                worker = ('import runpy,sys; from pathlib import Path; '
                          f'b=runpy.run_path({str(ROOT/"tools/tier-battery.py")!r}); '
                          f'b["tee_run"]([sys.executable,"-c",{child!r}],Path({str(root/"inner.log")!r}),'
                          f'timeout={inner_timeout},echo=False,pass_fds=({lock.fileno()},),'
                          f'shared_group={not detached})')
                # The worker, not this test process, owns the flock after launch.
                command = [sys.executable, '-c', worker]
                outer = ('import runpy,sys; from pathlib import Path; '
                         f'b=runpy.run_path({str(ROOT/"tools/tier-battery.py")!r}); '
                         f'print(b["tee_run"]({command!r},Path({str(root/"outer.log")!r}),'
                         f'timeout={outer_timeout},echo=False,pass_fds=({lock.fileno()},)))')
                p = subprocess.Popen([sys.executable, '-c', outer], pass_fds=(lock.fileno(),),
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                lock.close()
                stdout, stderr = p.communicate(timeout=8)
                self.assertEqual(p.returncode, 0, stderr)
                self.assertTrue(pidfile.exists(), 'grandchild never launched')
                time.sleep(1.3)
                self.assertEqual(marker.exists(), detached, stdout)
                if detached:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(independent, fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    state = subprocess.run(['/bin/ps', '-o', 'stat=', '-p', pidfile.read_text()],
                                           capture_output=True, text=True, check=False).stdout.strip()
                    self.assertTrue(not state or state.startswith('Z'), 'grandchild still running: '+state)
                    fcntl.flock(independent, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                if pidfile.exists():
                    try:
                        os.kill(int(pidfile.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                lock.close()
                os.close(independent)

    def test_nested_success_does_not_kill_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            worker = ('import runpy,sys; from pathlib import Path; '
                      f'b=runpy.run_path({str(ROOT/"tools/tier-battery.py")!r}); '
                      f'b["tee_run"]([sys.executable,"-c","print(123)"],Path({str(root/"inner")!r}),'
                      'echo=False,shared_group=True); print("worker-alive")')
            self.assertEqual(B.tee_run([sys.executable, '-c', worker], root/'outer', echo=False), (0, False))
            self.assertIn('worker-alive', (root/'outer').read_text())

    def test_outer_timeout_kills_grandchild_and_releases_lock(self):
        self.run_tree()

    def test_inner_timeout_kills_shared_group_and_releases_lock(self):
        self.run_tree(inner_timeout=.3, outer_timeout=3)

    def test_red_detached_grandchild_escapes_and_holds_lock(self):
        self.run_tree(detached=True)


if __name__ == '__main__':
    unittest.main()
