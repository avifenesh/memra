#!/usr/bin/env python3
"""CPU controls for native caller-runner refusal and owned-child cleanup."""
import importlib.util
import hashlib
import json
import os
import re
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('qualify_callers', Path(os.environ.get('REWRITE_RUNNER_UNDER_TEST', Path(__file__).with_name('qualify-callers.py'))))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def transfer_fixture(mode):
    """Independent transcript fixture for the reviewed native gate; never GPU evidence."""
    if mode == 'conformance':
        return '''PASS v1 transfer_cancel native CUDA
PASS v1.1 transfer_complete_cancel native CUDA
PASS v1.1 transfer_lifetime native events + injected observation loss + graph retention
PASS v1.1 transfer_zero_accept Unsupported NVMe preserves owned input
PASS v1.1 acceptance exhaustive native mixed batch; rejected sibling blocks publication
PASS v1.2 transfer_completion_bytes native CUDA; stale epochs, ready publication, take once, authentic consumer fence
PASS additive source retirement Busy while source consumer bound; host destination survives source release
PASS v1.3 transfer_source_retirement native CUDA
PASS v1.3 device_hand_back native CUDA
PASS additive dropped destination retains backing and charge until graph retirement and acknowledgement
PASS rule cancelled-restore-recovers-source native CUDA
PASS rule cancel-refused-after-source-consumed native CUDA
PASS native governor zero after controlled drain
'''
    lines = []
    for size in (4096, 65536, 1048576, 16777216, 67108864, 268435456):
        digest = hashlib.sha256(str(size).encode()).hexdigest()
        lines.append(f'PASS native D2H-H2D roundtrip bytes={size} N=1 expected_sha256={digest} '
                     f'actual_sha256={digest} byte_exact=true source_freed_host_live=true '
                     'handback_no_copy=true governor_zero=true')
    return '\n'.join(lines) + '\n'


class TransferRunnerTests(unittest.TestCase):
    def exercise(self, mode=None, text=None, code=0):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root, out, owned = base / 'repo', base / 'evidence', base / 'owned'
            root.mkdir()
            commands = []
            def fake(command, environment, target, timeout):
                commands.append(command)
                self.assertEqual(command[0], str(owned / 'target/release/tier-transfer-gate'))
                self.assertEqual(environment['MEMRA_GPU_LEASE_FILE'], 'CPU fixture')
                self.assertNotIn('MEMRA_REWRITE_BUNDLE', environment)
                target.mkdir(parents=True)
                output = text if command[1] == mode and text is not None else transfer_fixture(command[1])
                (target / 'stdout.log').write_text(output)
                (target / 'stderr.log').write_text('CPU scheduling fixture\n')
                return code if command[1] == mode else 0
            # No model or roster exists, and no roster argument is provided.
            args = SimpleNamespace(phase='transfer', model=None, out=out,
                                   build_record=owned / 'build.json', timeout_seconds=5)
            error = None
            with patch.object(runner, 'ROOT', root), \
                    patch.object(runner.build, 'verify_build_record', return_value=({}, 'a' * 64)), \
                    patch.object(runner.admission, 'verify_source') as source, \
                    patch.object(runner.admission, 'verify_lease', return_value={'requested_uuids': ['owned']}), \
                    patch.object(runner, 'plain', side_effect=fake), \
                    patch.object(runner.controller, 'run_controlled') as controlled, \
                    patch.object(runner.subprocess, 'check_output', return_value='gpu_uuid,pid\n'), \
                    patch.object(runner.subprocess, 'Popen') as telemetry, \
                    patch.dict(os.environ, {'MEMRA_GPU_LEASE_FILE': 'CPU fixture'}), patch('builtins.print'):
                try:
                    runner.run(args)
                except RuntimeError as caught:
                    error = caught
                source.assert_not_called()
                controlled.assert_not_called()
                self.assertIn('--loop-ms=250', telemetry.call_args.args[0])
                telemetry.return_value.terminate.assert_called_once()
                telemetry.return_value.wait.assert_called_once()
            result = json.loads((out / 'result.json').read_text())
            cases = json.loads((out / 'cases.json').read_text())
            self.assertFalse(result['model_support_promotion'])
            self.assertFalse(result['serving_binary_qualification'])
            manifest = json.loads((out / 'files-sha256.json').read_text())
            for case in cases:
                for stream in ('stdout', 'stderr'):
                    path = f'cases/{case["case"]}/{stream}.log'
                    self.assertEqual(manifest[path], runner.build.digest(out / path))
                    self.assertEqual(case[f'{stream}_sha256'], manifest[path])
            if mode is None:
                self.assertIsNone(error)
                self.assertEqual(result['status'], 'passed')
                self.assertEqual([command[1] for command in commands], ['conformance', 'roundtrip'])
                self.assertEqual(len(cases[0]['validated_output']['markers']), 13)
                self.assertEqual(len(cases[1]['validated_output']['roundtrips']), 6)
            else:
                self.assertIsNotNone(error, 'invalid transcript/exit accepted')
                self.assertEqual(result['status'], 'failed')
                self.assertFalse(cases[-1]['passed'])
                self.assertEqual(commands[-1][1], mode)
                if code == 0:
                    self.assertIn('validation_error', cases[-1])

    def test_complete_schedule_needs_no_model_or_roster(self):
        self.exercise()

    def test_conformance_census_matches_current_native_producer(self):
        source = (Path(__file__).resolve().parents[2]
                  / 'crates/memra-engine/src/bin/tier_transfer_gate.rs').read_text()
        markers = re.findall(r'println!\(\s*"(PASS [^"\n]+)"', source)
        conformance = [line for line in markers
                       if not line.startswith('PASS native D2H-H2D roundtrip ')]
        self.assertEqual(len(conformance), 13)
        self.assertCountEqual(conformance, runner.TRANSFER_CONFORMANCE)
        self.assertCountEqual(conformance, transfer_fixture('conformance').splitlines())

    def test_every_conformance_assertion_is_mandatory_and_unique(self):
        lines = transfer_fixture('conformance').splitlines()
        for index in range(len(lines)):
            for bad in (lines[:index] + lines[index + 1:], lines + [lines[index]]):
                with self.subTest(index=index, count=len(bad)):
                    self.exercise('conformance', '\n'.join(bad))
        for bad in ('', '\n'.join(lines).replace('Busy', 'NotBusy'),
                    '\n'.join(lines + ['PASS incomplete assertion'])):
            with self.subTest(text=bad): self.exercise('conformance', bad)

    def test_every_roundtrip_size_and_matching_hash_is_mandatory(self):
        lines = transfer_fixture('roundtrip').splitlines()
        for index in range(len(lines)):
            for bad in (lines[:index] + lines[index + 1:], lines + [lines[index]],
                        lines[:index] + [lines[index].replace('actual_sha256=', 'actual_sha256=f')] + lines[index + 1:]):
                with self.subTest(index=index, text=bad): self.exercise('roundtrip', '\n'.join(bad))
        digest = hashlib.sha256(b'4096').hexdigest()
        text = '\n'.join(lines)
        for bad in ('', text.replace(f'actual_sha256={digest}', 'actual_sha256=' + '0' * 64),
                    text.replace('bytes=4096 ', 'bytes=4097 '), text.replace('N=1 ', 'N=2 '),
                    text + '\nPASS unrecognized result', text.replace('PASS ', ' PASS ', 1)):
            with self.subTest(text=bad): self.exercise('roundtrip', bad)

    def test_roundtrip_lifecycle_flags_cannot_be_missing_or_false(self):
        text = transfer_fixture('roundtrip')
        for flag in ('byte_exact', 'source_freed_host_live', 'handback_no_copy', 'governor_zero'):
            for bad in (text.replace(f'{flag}=true', f'{flag}=false', 1),
                        text.replace(f'{flag}=true', '', 1)):
                with self.subTest(flag=flag, text=bad): self.exercise('roundtrip', bad)

    def test_complete_output_cannot_override_failed_or_cancelled_exit(self):
        for mode in ('conformance', 'roundtrip'):
            for code in (1, 2, -signal.SIGTERM, -signal.SIGKILL):
                with self.subTest(mode=mode, code=code): self.exercise(mode, code=code)


class CallerRunnerTests(unittest.TestCase):
    def test_complete_schedule_preserves_scope_and_requires_each_native_case(self):
        self.check_schedule('callers')

    def test_selected_schedule_requires_supported_callers_refusals_and_real_fallback(self):
        self.check_schedule('supported-callers')

    def test_failed_selected_fallback_quarantines_index_and_fails_phase(self):
        self.check_schedule('supported-callers', fail_fallback=True)

    def test_selected_preflight_never_quarantines_an_existing_unowned_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            out = base / 'existing'
            index = out / 'retained-eg-bundle/rewrite-receipts.tsv'
            index.parent.mkdir(parents=True)
            index.write_text('unowned preserved index')
            args = SimpleNamespace(phase='supported-callers', model=base / 'model', out=out,
                                   build_record=base / 'owned/build.json', timeout_seconds=5)
            with patch.object(runner, 'ROOT', base / 'repo'), \
                    patch.object(runner.admission, 'verify_lease', return_value={}), \
                    patch.object(runner.admission, 'verify_source', return_value={}), \
                    patch.object(runner.build, 'verify_build_record', return_value=({}, 'a' * 64)), \
                    patch.object(runner, 'selected_failure') as failure:
                with self.assertRaises(FileExistsError):
                    runner.run(args)
                failure.assert_not_called()
            self.assertEqual(index.read_text(), 'unowned preserved index')
            self.assertFalse((out / 'result.json').exists())

    def check_schedule(self, phase, fail_fallback=False):
        selected = phase == 'supported-callers'
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, out, owned = base / 'repo', base / 'evidence', base / 'owned'
            root.mkdir()
            owned.mkdir()
            record = {'tests': {key: {'artifact': {'path': f'test-target/release/deps/{key}'}}
                                for key in ('worker', 'repack', 'gemma-prime')}, 'binaries': {}}
            launched, controlled = [], []

            def fake(command, environment, target, _timeout):
                target.mkdir(parents=True)
                launched.append((command, environment))
                if selected:
                    self.assertEqual(environment['MEMRA_FAST'], '0')
                    self.assertEqual(environment['CUDA_FORCE_PRELOAD_LIBRARIES'], '1')
                if 'inspect' in command:
                    bundle = Path(command[-1])
                    bundle.mkdir()
                    (bundle / 'artifact.lock').write_text('synthetic scheduling fixture')
                    text = 'CPU inspect fixture\n'
                elif command[1] == 'retained-capture':
                    text = 'RETAINED_CAPTURE_PASS\n'
                elif command[1] == 'retained-eg-capture':
                    (Path(command[-1]) / 'rewrite-receipts.tsv').write_text('CPU provisional index')
                    text = 'RETAINED_EAGER_GRAPH_CAPTURE_PASS\n'
                elif command[1] == 'retained-eg-prime-api-refusal':
                    text = 'UNQUALIFIED_PRIME_API_REFUSAL_PASS\n'
                elif command[1] == 'retained-eg-prime-production-refusal':
                    text = 'UNQUALIFIED_PRIME_PRODUCTION_REFUSAL_PASS\n'
                elif command[1] == 'retained-eg-prime-fallback':
                    text = 'NATIVE_MATH_FAILED fallback fixture\n' if fail_fallback else 'EAGER_PRIME_FALLBACK_PASS\n'
                elif Path(command[0]).name == 'worker':
                    text = 'NATIVE_WORKER_BOUNDARY_PASS\ntest result: ok. 1 passed; 0 failed; 0 ignored;\n'
                elif Path(command[0]).name == 'repack':
                    text = 'test result: ok. 1 passed; 0 failed; 0 ignored;\n'
                else:
                    text = 'RETAINED_EAGER_GRAPH_CALLER_PASS\n' if selected else 'RETAINED_CALLER_PASS\n'
                (target / 'stdout.log').write_text(text)
                (target / 'stderr.log').write_text('')
                return 1 if fail_fallback and command[1] == 'retained-eg-prime-fallback' else 0

            def control(command, target, timeout, *, environment, check_invariants):
                check_invariants()
                controlled.append(command)
                return {'controller_returncode': fake(command, environment, target, timeout)}

            args = SimpleNamespace(phase=phase, model=base / 'model', out=out,
                                   build_record=owned / 'build.json', timeout_seconds=5)
            with patch.object(runner, 'ROOT', root), \
                    patch.object(runner.build, 'verify_build_record', return_value=(record, 'a' * 64)), \
                    patch.object(runner.admission, 'verify_source', return_value={'fixture': True}), \
                    patch.object(runner.admission, 'verify_lease', return_value={'requested_uuids': ['owned']}), \
                    patch.object(runner, 'plain', side_effect=fake), \
                    patch.object(runner.controller, 'run_controlled', side_effect=control), \
                    patch.object(runner.subprocess, 'check_output', return_value='gpu_uuid,pid\n'), \
                    patch.object(runner.subprocess, 'Popen'), \
                    patch.dict(os.environ, {'MEMRA_GPU_LEASE_FILE': 'fixture', 'CUDA_FORCE_PRELOAD_LIBRARIES': '0'}), patch('builtins.print'):
                if fail_fallback:
                    with self.assertRaisesRegex(RuntimeError, 'prime-fallback failed'):
                        runner.run(args)
                else:
                    runner.run(args)
            self.assertEqual(len(launched), 27 if selected else 26)
            self.assertEqual(len(controlled), 10 if selected else 11)
            cases = json.loads((out / 'cases.json').read_text())
            self.assertTrue(all(case['passed'] for case in cases[:-1]))
            self.assertEqual(cases[-1]['passed'], not fail_fallback)
            self.assertEqual(sum(case['kind'] == 'cpu-inspect' for case in cases), 1)
            workers = [(command, env) for command, env in launched if Path(command[0]).name == 'worker']
            self.assertEqual({(env['REWRITE_PROBE_CASE'], env['REWRITE_PROBE_DRIFT']) for _, env in workers},
                             {(case, drift) for case in runner.WORKER_CASES for drift in ('library', 'env')})
            self.assertTrue(all('MEMRA_REWRITE_BUNDLE' not in env for _, env in workers))
            result = json.loads((out / 'result.json').read_text())
            self.assertEqual(result['serving_binary_qualification'], False)
            if selected:
                self.assertEqual(result['selected_surfaces'], ['decode-eager', 'decode-graph'])
                self.assertEqual(result['withheld_surfaces'], ['carried-prime'])
                self.assertFalse(result['positive_prime_coverage'])
                modes = [command[1] for command, _ in launched if Path(command[0]).name == 'rewrite_identity_gate']
                expected = ['retained-eg-capture'] + [f'retained-eg-{api}-{drift}'
                    for api in ('step', 'prof-apply', 'prof-launch', 'prof-read') for drift in ('library', 'environment')]
                expected += ['retained-eg-prime-api-refusal', 'retained-eg-prime-production-refusal', 'retained-eg-prime-fallback']
                self.assertEqual(modes, expected)
            if fail_fallback:
                self.assertEqual(result['status'], 'failed')
                bundle = out / 'retained-eg-bundle'
                self.assertFalse((bundle / 'rewrite-receipts.tsv').exists())
                self.assertEqual((bundle / 'failed-caller-index.tsv').read_text(), 'CPU provisional index')
                manifest = json.loads((out / 'files-sha256.json').read_text())
                self.assertIn('retained-eg-bundle/failed-caller-index.tsv', manifest)

    def test_native_unit_result_cannot_be_vacuous_ignored_or_failed(self):
        self.assertTrue(runner.test_success('test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 400 filtered out;\n'))
        for text in ('', 'test result: ok. 0 passed; 0 failed; 0 ignored;',
                     'test result: ok. 0 passed; 0 failed; 1 ignored;',
                     'test result: FAILED. 0 passed; 1 failed; 0 ignored;'):
            self.assertFalse(runner.test_success(text))

    def test_invalid_timeout_refuses_before_lease_or_gpu_work(self):
        with patch.object(runner.admission, 'verify_lease') as lease:
            for timeout in (0, -1, float('nan'), float('inf'), 86401):
                with self.assertRaises(ValueError):
                    runner.run(SimpleNamespace(timeout_seconds=timeout))
            lease.assert_not_called()

    def test_plain_child_exit_and_raw_output_preserved(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner.admission, 'verify_lease'):
            out = Path(directory) / 'case'
            code = runner.plain([sys.executable, '-c', 'import sys; print("raw witness"); sys.exit(7)'],
                                os.environ.copy(), out, 5)
            self.assertEqual(code, 7)
            self.assertEqual((out / 'stdout.log').read_text(), 'raw witness\n')

    def test_lost_lease_kills_and_reaps_the_owned_child(self):
        calls = 0
        def lease():
            nonlocal calls
            calls += 1
            if calls > 3:
                raise RuntimeError('deliberate lost lease')
        with tempfile.TemporaryDirectory() as directory, patch.object(runner.admission, 'verify_lease', side_effect=lease):
            out = Path(directory) / 'case'
            with self.assertRaisesRegex(RuntimeError, 'lost lease'):
                runner.plain([sys.executable, '-c', 'import os,time; print(os.getpid(),flush=True); time.sleep(30)'],
                             os.environ.copy(), out, 5)
            pid = int((out / 'stdout.log').read_text().strip())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_controller_invariant_failure_reaps_without_memory_access(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner.controller, 'require_linux'), \
                patch.object(runner.controller, 'validate_stopped_group') as group:
            result = runner.controller.run_controlled(
                [sys.executable, '-c', 'import time; time.sleep(30)'], Path(directory) / 'case', 5,
                environment=os.environ.copy(),
                check_invariants=lambda: (_ for _ in ()).throw(RuntimeError('lost lease')))
            self.assertEqual(result['status'], 'failed')
            self.assertIn('lost lease', result['error'])
            self.assertIsNotNone(result['child_returncode'])
            group.assert_not_called()


class FinalizationTests(unittest.TestCase):
    def exercise(self, phase, fault):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, out, owned = base / 'repo', base / 'evidence', base / 'owned'
            root.mkdir()
            binary = owned / 'target/release'
            binary.mkdir(parents=True)
            roster = base / 'roster'
            roster.write_text('CPU fixture')
            record = {'binaries': {}, 'tests': {
                key: {'artifact': {'path': f'test-target/release/deps/{key}'}}
                for key in ('worker', 'repack', 'gemma-prime')}}
            for name in ('kernel-check', 'run-spec', 'argmax-margin-probe', 'tier-transfer-gate'):
                path = binary / name
                path.write_bytes(b'CPU executable fixture')
                record['binaries'][name] = {'sha256': runner.build.digest(path)}
            finalizing = False
            verified_after_cleanup = False
            signals_sent = 0
            telemetry = Mock()
            waits = 0
            def wait(**kwargs):
                nonlocal finalizing, waits
                finalizing = True
                waits += 1
                if fault == 'timeout' and waits == 1:
                    raise subprocess.TimeoutExpired(['nvidia-smi'], kwargs['timeout'])
                return 0
            telemetry.wait.side_effect = wait
            if fault == 'terminate':
                telemetry.terminate.side_effect = OSError('deliberate terminate failure')
            def verify(*_args):
                nonlocal verified_after_cleanup, signals_sent
                if finalizing:
                    verified_after_cleanup = True
                    if fault == 'sig_verify':
                        signals_sent += 1
                        os.kill(os.getpid(), signal.SIGTERM)
                    if fault == 'verify':
                        raise RuntimeError('deliberate final source/binary/lease failure')
                return record, 'a' * 64
            original_write = runner.write
            def write(path, value):
                nonlocal signals_sent
                if path.name == 'files-sha256.json' and fault == 'manifest':
                    raise OSError('deliberate manifest failure')
                if path.name == 'result.json.pending' and value.get('status') == 'passed':
                    self.assertTrue(finalizing and verified_after_cleanup)
                    self.assertTrue((out / 'files-sha256.json').exists())
                    if fault == 'publish':
                        raise OSError('deliberate atomic publication failure')
                original_write(path, value)
                if fault == 'sig_pending' and path.name == 'result.json.pending' and value.get('status') == 'passed':
                    signals_sent += 1
                    os.kill(os.getpid(), signal.SIGTERM)
            original_replace = Path.replace
            def replace(path, target):
                nonlocal signals_sent
                if fault == 'sig_replace' and path.name == 'result.json.pending' and json.loads(path.read_text())['status'] == 'passed':
                    signals_sent += 1
                    os.kill(os.getpid(), signal.SIGTERM)
                if fault == 'replace_after' and path.name == 'result.json.pending' and json.loads(path.read_text())['status'] == 'passed':
                    original_replace(path, target)
                    raise OSError('deliberate failure after atomic PASS replacement')
                return original_replace(path, target)
            def fake(command, environment, target, timeout):
                target.mkdir(parents=True)
                if 'inspect' in command:
                    bundle = Path(command[-1]); bundle.mkdir()
                    (bundle / 'artifact.lock').write_text('CPU fixture')
                if phase == 'supported-callers' and command[1] == 'retained-eg-capture':
                    bundle = Path(command[-1])
                    (bundle / 'rewrite-receipts.tsv').write_text('CPU provisional index')
                    capture = bundle / 'retained-eg-capture'; capture.mkdir()
                    (capture / 'PASS').write_text('CPU capture seal')
                text = ('RETAINED_CAPTURE_PASS\nRETAINED_CALLER_PASS\nNATIVE_WORKER_BOUNDARY_PASS\n'
                        'RETAINED_EAGER_GRAPH_CAPTURE_PASS\nRETAINED_EAGER_GRAPH_CALLER_PASS\n'
                        'UNQUALIFIED_PRIME_API_REFUSAL_PASS\nUNQUALIFIED_PRIME_PRODUCTION_REFUSAL_PASS\nEAGER_PRIME_FALLBACK_PASS\n'
                        'test result: ok. 1 passed; 0 failed; 0 ignored;\n=== RELEASE BATTERY PASS ===\n')
                if phase == 'transfer':
                    text = transfer_fixture(command[1])
                (target / 'stdout.log').write_text(text)
                (target / 'stderr.log').write_text('raw stderr witness\n')
                return 0
            def controlled(command, target, timeout, *, environment, check_invariants):
                check_invariants()
                return {'controller_returncode': fake(command, environment, target, timeout)}
            args = SimpleNamespace(phase=phase, model=base / 'model', out=out, roster=roster,
                                   build_record=owned / 'build.json', timeout_seconds=5)
            failure = None
            with patch.object(runner, 'ROOT', root), \
                    patch.object(runner.build, 'verify_build_record', side_effect=verify), \
                    patch.object(runner.admission, 'verify_source', return_value={'fixture': True}), \
                    patch.object(runner.admission, 'verify_lease', return_value={'requested_uuids': []}), \
                    patch.object(runner, 'plain', side_effect=fake), \
                    patch.object(runner.controller, 'run_controlled', side_effect=controlled), \
                    patch.object(runner.subprocess, 'check_output', return_value='gpu_uuid,pid\n'), \
                    patch.object(runner.subprocess, 'Popen', return_value=telemetry), \
                    patch.object(runner, 'write', side_effect=write), \
                    patch.object(Path, 'replace', replace), \
                    patch.dict(os.environ, {'MEMRA_GPU_LEASE_FILE': 'CPU fixture'}), patch('builtins.print'):
                try:
                    runner.run(args)
                except BaseException as error:
                    failure = error
            result = json.loads((out / 'result.json').read_text())
            if fault and fault.startswith('sig_'):
                print(json.dumps({'signal_control': phase, 'fault': fault, 'signals_sent': signals_sent, 'status': result['status'], 'had_error': failure is not None}))
            self.assertTrue(list((out / 'cases').rglob('stdout.log')))
            if fault:
                self.assertIsNotNone(failure, 'finalization failure was ignored')
                self.assertIn(result['status'], ('failed', 'incomplete'), 'finalization failure left a stale passed result')
                if phase == 'supported-callers':
                    bundle = out / 'retained-eg-bundle'
                    self.assertFalse((bundle / 'rewrite-receipts.tsv').exists(), 'failed selected phase left an installable index')
                    self.assertEqual((bundle / 'failed-caller-index.tsv').read_text(), 'CPU provisional index')
                    self.assertTrue(result['selected_bundle_quarantined'])
                    if fault != 'manifest':
                        self.assertEqual(result['evidence_manifest_sha256'], runner.build.digest(out / 'files-sha256.json'))
                        manifest = json.loads((out / 'files-sha256.json').read_text())
                        self.assertIn('retained-eg-bundle/failed-caller-index.tsv', manifest)
                        self.assertNotIn('retained-eg-bundle/rewrite-receipts.tsv', manifest)
                    else:
                        self.assertIsNone(result['evidence_manifest_sha256'])
                if fault in ('timeout', 'terminate'):
                    telemetry.kill.assert_called_once()
                    self.assertGreaterEqual(telemetry.wait.call_count, 1)
            else:
                self.assertIsNone(failure)
                self.assertEqual(result['status'], 'passed')
                self.assertTrue(verified_after_cleanup)
                self.assertEqual(result['evidence_manifest_sha256'], runner.build.digest(out / 'files-sha256.json'))
                self.assertNotIn('result.json', json.loads((out / 'files-sha256.json').read_text()))

    def test_cleanup_timeout_never_leaves_passed_result(self):
        for phase in ('callers', 'supported-callers', 'transfer', 'battery'):
            with self.subTest(phase=phase): self.exercise(phase, 'timeout')

    def test_cleanup_error_never_leaves_passed_result(self):
        for phase in ('callers', 'supported-callers', 'transfer', 'battery'):
            with self.subTest(phase=phase): self.exercise(phase, 'terminate')

    def test_manifest_failure_never_leaves_passed_result(self):
        for phase in ('callers', 'supported-callers', 'transfer', 'battery'):
            with self.subTest(phase=phase): self.exercise(phase, 'manifest')

    def test_final_invariant_failure_never_leaves_passed_result(self):
        for phase in ('callers', 'supported-callers', 'transfer', 'battery'):
            with self.subTest(phase=phase): self.exercise(phase, 'verify')

    def test_atomic_publication_failure_keeps_incomplete_result(self):
        for phase in ('callers', 'supported-callers', 'transfer', 'battery'):
            with self.subTest(phase=phase): self.exercise(phase, 'publish')

    def test_success_is_sealed_after_cleanup_and_final_verification(self):
        for phase in ('callers', 'supported-callers', 'transfer', 'battery'):
            with self.subTest(phase=phase): self.exercise(phase, None)

    def test_selected_failure_after_atomic_replace_revokes_index_and_pass_result(self):
        self.exercise('supported-callers', 'replace_after')


if __name__ == '__main__':
    unittest.main()
