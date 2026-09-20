#!/usr/bin/env python3
"""CPU fault controls for baseline native-runner result finalization; no GPU calls."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

path = Path(os.environ.get('REWRITE_NATIVE_RUNNER_UNDER_TEST', Path(__file__).with_name('qualify-native.py')))
spec = importlib.util.spec_from_file_location('native_runner_under_test', path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class NativeFinalizationTests(unittest.TestCase):
    def exercise(self, fault):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root, out, binary = base / 'repo', base / 'evidence', base / 'owned/target/release'
            root.mkdir(); binary.mkdir(parents=True)
            record = {'source': {'commit': 'a' * 40}, 'binaries': {}}
            for name in runner.build_record.BINARIES:
                file = binary / name; file.write_bytes(b'CPU executable fixture')
                record['binaries'][name] = {'sha256': runner.digest(file)}
            finished = False
            waits = 0
            checked_final = False
            telemetry = Mock()
            def wait(**kwargs):
                nonlocal finished, waits
                finished = True; waits += 1
                if fault == 'timeout' and waits == 1:
                    raise subprocess.TimeoutExpired(['nvidia-smi'], kwargs['timeout'])
                return 0
            telemetry.wait.side_effect = wait
            if fault == 'terminate':
                telemetry.terminate.side_effect = OSError('deliberate telemetry error')
            def verify(*_args):
                nonlocal checked_final
                if finished:
                    checked_final = True
                    if fault == 'verify': raise RuntimeError('deliberate final identity failure')
                return record, 'b' * 64
            def source_mutation(_model, variant):
                variant.mkdir(); (variant / 'weights').write_bytes(b'CPU wrong-weight fixture')
                return {'fixture': True}
            original_rmtree = runner.shutil.rmtree
            def remove(path, *args, **kwargs):
                if fault == 'variant' and Path(path) == out / 'wrong-weights':
                    raise OSError('deliberate variant cleanup failure')
                return original_rmtree(path, *args, **kwargs)
            original_write = runner.write_json
            def write(path, value):
                if fault == 'manifest' and path.name == 'files-sha256.json':
                    raise OSError('deliberate manifest failure')
                if path.name == 'result.json.pending' and value.get('status') == 'passed':
                    self.assertTrue(finished and checked_final)
                    self.assertFalse((out / 'wrong-weights').exists())
                    self.assertFalse((out / 'changed-executable').exists())
                    if fault == 'publish': raise OSError('deliberate publication failure')
                original_write(path, value)
            def run(command, *args, **kwargs):
                self.assertEqual(command[0], 'nvidia-smi')
                kwargs['stdout'].write(b'gpu_uuid,pid\n')
                return subprocess.CompletedProcess(command, 0)
            def popen(command, *args, **kwargs):
                if command[0] == 'nvidia-smi': return telemetry
                name = Path(kwargs['stdout'].name).stem
                if name == 'inspect':
                    bundle = out / 'bundle'; bundle.mkdir()
                    (bundle / 'artifact.lock').write_text('CPU fixture')
                refusals = {'missing-bundle': 'read artifact.lock',
                            'different-numerical-program': 'does not bind numeric_program_sha256=',
                            'different-weights-same-geometry': 'does not bind artifact_sha256=',
                            'different-executable-same-source': 'does not bind implementation_sha256='}
                code = 1 if name in refusals else 0
                text = (f'REWRITE_IDENTITY_GATE_FAIL: {refusals[name]}\n' if code else
                        ''.join(f'OUTPUT stage={stage} prompt={i} values=1 sha256={str(i)*64}\n'
                                for stage in ('installed-eager', 'check-eager', 'fresh-kv-diagnostic') for i in range(3)))
                kwargs['stdout'].write(text.encode())
                return SimpleNamespace(returncode=code, poll=lambda: code, wait=lambda **_: code, terminate=lambda: None)
            args = SimpleNamespace(model=base / 'model', out=out, binary_dir=binary,
                                   build_record=base / 'owned/build.json', mtp_model=None, mtp_sha256=None)
            failure = None
            with patch.object(runner, 'ROOT', root), \
                    patch.object(runner.build_record, 'verify_build_record', side_effect=verify), \
                    patch.object(runner, 'verify_source', return_value={'fixture': True}), \
                    patch.object(runner, 'verify_lease', return_value={'requested_uuids': []}), \
                    patch.object(runner, 'mutate_weight', side_effect=source_mutation), \
                    patch.object(runner.subprocess, 'run', side_effect=run), \
                    patch.object(runner.subprocess, 'Popen', side_effect=popen), \
                    patch.object(runner.shutil, 'rmtree', side_effect=remove), \
                    patch.object(runner, 'write_json', side_effect=write), \
                    patch.dict(os.environ, {'MEMRA_GPU_LEASE_FILE': 'CPU fixture'}), patch('builtins.print'):
                try: runner.run(args)
                except BaseException as error: failure = error
            result = json.loads((out / 'result.json').read_text())
            self.assertEqual(len(json.loads((out / 'cases.json').read_text())), 12)
            self.assertTrue((out / 'different-numerical-program.log').exists())
            if fault:
                self.assertIsNotNone(failure, 'finalization failure was ignored')
                self.assertIn(result['status'], ('failed', 'incomplete'), 'baseline finalization left stale passed result')
                if fault in ('timeout', 'terminate'): telemetry.kill.assert_called_once()
            else:
                self.assertIsNone(failure)
                self.assertEqual(result['status'], 'passed')
                self.assertEqual(result['evidence_manifest_sha256'], runner.digest(out / 'files-sha256.json'))
                self.assertNotIn('result.json', json.loads((out / 'files-sha256.json').read_text()))

    def test_telemetry_timeout_refuses_final_pass(self): self.exercise('timeout')
    def test_telemetry_error_refuses_final_pass(self): self.exercise('terminate')
    def test_variant_cleanup_failure_refuses_final_pass(self): self.exercise('variant')
    def test_manifest_failure_refuses_final_pass(self): self.exercise('manifest')
    def test_final_identity_failure_refuses_final_pass(self): self.exercise('verify')
    def test_atomic_publication_failure_preserves_incomplete(self): self.exercise('publish')
    def test_success_after_all_cleanup_and_evidence(self): self.exercise(None)


if __name__ == '__main__': unittest.main()
