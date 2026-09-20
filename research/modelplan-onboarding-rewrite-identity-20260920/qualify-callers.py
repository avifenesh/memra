#!/usr/bin/env python3
"""Execute bounded native caller or generic-battery cases from an owned build.

Requires the central GPU wrapper, a clean final checkout, and native build.json.
All output lives in a new directory outside the checkout. This runner does not rent,
merge, manufacture qualification, or turn a test-executable receipt into a server receipt.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


admission = module('rewrite_admission_runner', 'qualify-native.py')
controller = module('rewrite_env_controller', 'native_env_controller.py')
build = admission.build_record
GRAPH_CASES = ('step', 'prof-apply', 'prof-launch', 'prof-read', 'prime-run')
WORKER_CASES = ('advance_sample_emit', 'advance_token_emit', 'step_session',
                'step_session_prefill', 'prefill_tick', 'step_session_async_chain')
TEST_NAME = 'worker::rewrite_native_tests::native_worker_boundary'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def publish_result(out, result):
    # An interrupted/failed final write leaves the earlier incomplete receipt intact.
    pending = out / 'result.json.pending'
    write(pending, result)
    pending.replace(out / 'result.json')


def finalize_evidence(out, telemetry, telemetry_log, verify):
    errors = []
    if telemetry is not None:
        try:
            telemetry.terminate()
            telemetry.wait(timeout=15)
        except BaseException as error:
            errors.append(f'telemetry cleanup: {type(error).__name__}: {error}')
            try:
                telemetry.kill()
                telemetry.wait(timeout=15)
            except BaseException as reap_error:
                errors.append(f'telemetry kill/reap: {type(reap_error).__name__}: {reap_error}')
    try:
        telemetry_log.close()
    except BaseException as error:
        errors.append(f'telemetry log close: {type(error).__name__}: {error}')
    manifest_sha = None
    try:
        excluded = {out / 'files-sha256.json', out / 'result.json', out / 'result.json.pending'}
        write(out / 'files-sha256.json', {str(path.relative_to(out)): build.digest(path)
              for path in sorted(out.rglob('*')) if path.is_file() and path not in excluded})
        manifest_sha = build.digest(out / 'files-sha256.json')
    except BaseException as error:
        errors.append(f'evidence manifest: {type(error).__name__}: {error}')
    try:
        controller.check_deadline(float('inf'))  # Deferred interruption must not become success.
        verify()
    except BaseException as error:
        errors.append(f'final invariants: {type(error).__name__}: {error}')
    return errors, manifest_sha


def plain(command, environment, out, timeout_seconds):
    out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + timeout_seconds
    process = None
    with (out / 'stdout.log').open('wb') as stdout, (out / 'stderr.log').open('wb') as stderr:
        try:
            admission.verify_lease()
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout,
                                       stderr=stderr, stdin=subprocess.DEVNULL, start_new_session=True)
            while process.poll() is None:
                controller.check_deadline(deadline)
                admission.verify_lease()
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
            admission.verify_lease()
            return process.returncode
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def test_success(text):
    return re.search(r'^test result: ok\. 1 passed; 0 failed; 0 ignored;', text, re.M) is not None


def run(args):
    if not math.isfinite(args.timeout_seconds) or not 0 < args.timeout_seconds <= 86400:
        raise ValueError("timeout-seconds must be finite and in (0, 86400]")
    lease = admission.verify_lease()  # Before any GPU query or child.
    out, record_path = args.out.resolve(), args.build_record.resolve()
    model = args.model.resolve() if args.model else None
    if args.phase == 'callers' and model is None:
        raise ValueError('--model is required for caller probes')
    if out == ROOT or ROOT in out.parents:
        raise RuntimeError('native evidence must be outside the clean checkout')
    binaries = record_path.parent / 'target/release'
    record, record_sha = build.verify_build_record(ROOT, record_path, binaries)
    source = (admission.verify_source(model) if args.phase == 'callers' else
              {'roster': str(args.roster.resolve()), 'roster_sha256': build.digest(args.roster)})
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'lease.json', lease)
    write(out / 'source.json', source)
    write(out / 'build-record.json', record)
    write(out / 'build-binding.json', {'path': str(record_path), 'sha256': record_sha})
    base = {key: value for key, value in os.environ.items()
            if not key.startswith('MEMRA_') and not key.startswith('REWRITE_PROBE_')}
    base['MEMRA_GPU_LEASE_FILE'] = os.environ['MEMRA_GPU_LEASE_FILE']
    if args.phase == 'callers':
        base['MEMRA_FAST'] = '0'
    cases = []

    def verify():
        admission.verify_lease()
        build.verify_build_record(ROOT, record_path, binaries, record_sha)

    def case(name, command, changes=None, marker=None, environment_drift=False, unit=False):
        verify()
        directory = out / 'cases' / name
        environment = {**base, **(changes or {})}
        command = [str(value) for value in command]
        if environment_drift:
            result = controller.run_controlled(command, directory, args.timeout_seconds,
                                               environment=environment,
                                               check_invariants=admission.verify_lease)
            code = result['controller_returncode']
        else:
            code = plain(command, environment, directory, args.timeout_seconds)
        verify()
        stdout = (directory / 'stdout.log').read_text(errors='replace')
        stderr = (directory / 'stderr.log').read_text(errors='replace')
        text = stdout + stderr
        passed = code == 0 and (marker is None or marker in text) and (not unit or test_success(text))
        entry = {'case': name, 'kind': 'cpu-inspect' if name.startswith('inspect-') else 'native',
                 'command': command, 'returncode': code, 'passed': passed,
                 'required_marker': marker, 'one_native_test_required': unit,
                 'stdout_sha256': build.digest(directory / 'stdout.log'),
                 'stderr_sha256': build.digest(directory / 'stderr.log')}
        cases.append(entry)
        write(out / 'cases.json', cases)
        print(json.dumps(entry), flush=True)
        if not passed:
            raise RuntimeError(f'native case {name} failed; see {directory}')

    def inspect(name, bundle):
        case(name, [binaries / 'memra', 'model', 'inspect', model, '--against', 'qwen3', '--out', bundle])

    verify()
    # The only native queries before cases: selected-card exclusivity and telemetry.
    inventory = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv'], text=True)
    (out / 'compute-entry.csv').write_text(inventory)
    if any(line.split(',')[0].strip() in lease['requested_uuids'] for line in inventory.splitlines()[1:]):
        raise RuntimeError('a compute process occupies the selected GPU')
    telemetry_log = (out / 'all-card-telemetry-250ms.csv').open('wb')
    summary = {'status': 'incomplete', 'phase': args.phase, 'cases': 0,
               'build_record_sha256': record_sha, 'model_support_promotion': False,
               'serving_binary_qualification': False}
    publish_result(out, summary)
    telemetry = None
    failure = None
    with controller.cleanup_signals():
        try:
            telemetry = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,uuid,name,index,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu',
            '--format=csv', '--loop-ms=250'], stdout=telemetry_log, stderr=subprocess.STDOUT)
            if args.phase == 'callers':
                bundle = out / 'retained-bundle'
                inspect('inspect-retained', bundle)
                capture = {'MEMRA_ARTIFACT_LOCK': str(bundle / 'artifact.lock')}
                case('retained-capture', [binaries / 'rewrite_identity_gate', 'retained-capture', model, bundle],
                     capture, 'RETAINED_CAPTURE_PASS')
                strict = {**capture, 'MEMRA_REWRITE_BUNDLE': str(bundle)}
                for api in GRAPH_CASES:
                    for drift in ('library', 'environment'):
                        name = f'retained-{api}-{drift}'
                        case(name, [binaries / 'rewrite_identity_gate', name, model, bundle], strict,
                             'RETAINED_CALLER_PASS', environment_drift=drift == 'environment')
                worker = record_path.parent / record['tests']['worker']['artifact']['path']
                for api in WORKER_CASES:
                    for drift in ('library', 'env'):
                        name = f'worker-{api}-{drift}'
                        worker_bundle = out / 'worker-bundles' / name
                        worker_bundle.mkdir(parents=True, exist_ok=False)
                        # The artifact is shared; qualification is not. Each worker process
                        # measures and binds its own fresh eager receipt in this test binary.
                        shutil.copyfile(bundle / 'artifact.lock', worker_bundle / 'artifact.lock')
                        settings = {'REWRITE_PROBE_CASE': api, 'REWRITE_PROBE_DRIFT': drift,
                                    'REWRITE_PROBE_SOURCE': str(model), 'REWRITE_PROBE_BUNDLE': str(worker_bundle),
                                    'MEMRA_ARTIFACT_LOCK': str(worker_bundle / 'artifact.lock')}
                        if api == 'step_session_async_chain':
                            settings['MEMRA_ASYNC_CHAIN'] = '2'
                        case(name, [worker, TEST_NAME, '--exact', '--ignored', '--nocapture', '--test-threads=1'],
                             settings, 'NATIVE_WORKER_BOUNDARY_PASS', environment_drift=drift == 'env', unit=True)
                repack = record_path.parent / record['tests']['repack']['artifact']['path']
                for layout in ('stacked', 'per_expert'):
                    name = f'native_nvfp4_{layout}_cache_lifecycle'
                    case(name, [repack, name, '--exact', '--ignored', '--nocapture', '--test-threads=1'],
                         {'MEMRA_ARTIFACT_LOCK': '', 'MEMRA_ST_REPACK_DISK': '1', 'MEMRA_ST_PINNED': '0',
                          'REWRITE_REPACK_OUT': str(out / f'repack-{layout}')}, unit=True)
            else:
                # The authoritative battery resolves ROOT/target/release. Only create missing
                # links; never replace an existing executable. Verify each consumed path.
                links = []
                root_bin = ROOT / 'target/release'
                root_bin.mkdir(parents=True, exist_ok=True)
                try:
                    for name in ('kernel-check', 'run-spec', 'argmax-margin-probe'):
                        link, executable = root_bin / name, binaries / name
                        if link.exists() or link.is_symlink():
                            if link.resolve() != executable.resolve():
                                raise RuntimeError(f'unowned existing battery executable: {link}; use a fresh checkout')
                        else:
                            link.symlink_to(executable)
                            links.append(link)
                        if build.digest(link) != record['binaries'][name]['sha256']:
                            raise RuntimeError('battery executable hash mismatch')
                    case('required-generic-battery', [ROOT / 'tools/release-battery.sh', '--roster', args.roster.resolve()],
                         marker='=== RELEASE BATTERY PASS')
                    for name in ('kernel-check', 'run-spec', 'argmax-margin-probe'):
                        if build.digest(root_bin / name) != record['binaries'][name]['sha256']:
                            raise RuntimeError('battery executable changed during execution')
                finally:
                    for link in links:
                        if link.is_symlink() and link.resolve() == (binaries / link.name).resolve():
                            link.unlink()
            verify()
        except BaseException as error:
            failure = error
        errors, manifest_sha = finalize_evidence(out, telemetry, telemetry_log, verify)
    summary.update({'cases': len(cases), 'evidence_manifest_sha256': manifest_sha})
    if failure is not None or errors:
        summary.update({'status': 'failed', 'error': str(failure) if failure else None,
                        'finalization_errors': errors})
        publish_result(out, summary)
        if failure is not None:
            raise failure
        raise RuntimeError('; '.join(errors))
    summary['status'] = 'passed'
    publish_result(out, summary)  # Last operation: teardown, evidence and checks already succeeded.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--build-record', type=Path, required=True)
    parser.add_argument('--phase', choices=('callers', 'battery'), default='callers')
    parser.add_argument('--roster', type=Path, default=ROOT / 'tools/release-roster.tsv')
    parser.add_argument('--timeout-seconds', type=float, default=600)
    run(parser.parse_args())
