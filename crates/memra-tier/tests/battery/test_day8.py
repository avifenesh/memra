"""Day-eight token and actual-filesystem binding regressions (CPU only)."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('battery_day8', ROOT/'tools/tier-battery.py')
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)


class StorageBindingTests(unittest.TestCase):
    def test_wrapped_storage_cannot_bypass_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'out'
            cmd = ['storage-bench', 'restore', str(Path(tmp)/'object'), '264', 'buffered']
            wrapped = ['bash', '-c', shlex.join(cmd)]
            self.assertEqual(B.storage_command(wrapped), cmd)
            self.assertEqual(B.storage_command(cmd[:3]), cmd[:3])
            self.assertEqual(B.storage_command(cmd[:4]), cmd[:4])
            proc = subprocess.run([sys.executable, str(ROOT/'tools/tier-battery.py'),
                '--out', str(out), '--execute', *wrapped], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2)
            self.assertIn('requires --storage-root', proc.stderr)
            self.assertFalse(out.exists())
            for argv in [['bash', '-c', 'exec '+shlex.join(cmd)],
                         ['bash', '-c', shlex.join(cmd)+'; true'],
                         ['sh', '-ec', shlex.join(cmd)],
                         ['bash', '-c', 'storage-bench restore "$ROOT/o" 264 buffered']]:
                with self.subTest(argv=argv), self.assertRaises(ValueError):
                    B.storage_command(argv)
            self.assertEqual(B.storage_command(['env', '-u', 'VAR', 'MODE=x', *wrapped]), cmd)

    def test_real_stat_binding_rejects_escape_and_different_mount(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp); root = parent/'root'; root.mkdir()
            other = parent/'other'; other.mkdir()
            obj = root/'object'
            binding = B.storage_binding(root, obj)
            self.assertEqual(binding['root_filesystem']['device'], root.stat().st_dev)
            self.assertEqual(binding['root_filesystem']['filesystem_id'], os.statvfs(root).f_fsid)
            obj.write_bytes(b'fixture')
            self.assertEqual(B.storage_binding(root, obj), binding)
            (root/'escape').symlink_to(other, target_is_directory=True)
            self.assertEqual(B.storage_binding(root, root)['object'], str(root.resolve()))
            for bad in (other/'object', root/'escape'/'object'):
                with self.subTest(path=bad), self.assertRaises(ValueError):
                    B.storage_binding(root, bad)
            with patch.object(B, 'filesystem_identity', side_effect=[
                    {'device': 1, 'filesystem_id': 2, 'mount_id': 3},
                    {'device': 1, 'filesystem_id': 2, 'mount_id': 4}]):
                with self.assertRaisesRegex(ValueError, 'filesystem differ'):
                    B.storage_binding(root, obj)
            storage = {'root': str(root), 'object_binding': binding}
            B.verify_storage_binding(storage)
            changed = copy.deepcopy(binding); changed['object_filesystem']['device'] += 1
            with patch.object(B, 'storage_binding', return_value=changed):
                with self.assertRaisesRegex(ValueError, 'changed during'):
                    B.verify_storage_binding(storage)

    def test_capture_binds_command_and_both_cell_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out = root/'out'; out.mkdir()
            obj = root/'object'; binary = root/'storage-bench'
            binary.write_text('#!'+sys.executable+'\nimport pathlib,sys\n'
                              'pathlib.Path(sys.argv[2]).write_bytes(b"x")\nprint("CPU stub")\n')
            binary.chmod(0o755)
            cmd = [str(binary), 'roundtrip', str(obj), '1', 'buffered']
            def stub(command, raw, **kwargs):
                raw.write_text('overlay\n'); return 0, False
            with patch.object(B, 'tee_run', stub):
                storage = B.capture_storage(root, out, True, obj)
            result = B.SubprocessRunner(str(root/'no-smi')).run(cmd, out/'command.log',
                                                               echo=False, storage=storage)
            B.validate_capture(result, out)
            rows = [json.loads(line) for line in (out/'CELL.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['storage'], rows[1]['storage'])
            self.assertEqual(result['storage'], rows[0]['storage'])
            bad = copy.deepcopy(result); bad['command'][2] += '-wrong'
            with self.assertRaisesRegex(ValueError, 'command mismatch'):
                B.validate_capture(bad, out)
            bad = copy.deepcopy(result)
            bad['storage']['object_binding']['object_filesystem']['device'] += 1
            with self.assertRaisesRegex(ValueError, 'filesystem mismatch'):
                B.validate_capture(bad, out)


if __name__ == '__main__':
    unittest.main()
