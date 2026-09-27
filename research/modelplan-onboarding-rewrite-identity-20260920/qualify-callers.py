#!/usr/bin/env python3
"""Execute bounded native caller, transfer or generic-battery cases from an owned build.

Requires the central GPU wrapper, a clean final checkout, and native build.json.
All output lives in a new directory outside the checkout. This runner does not rent,
merge, manufacture qualification, or turn a test-executable receipt into a server receipt.
"""
import argparse
from collections import Counter
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
EAGER_GRAPH_CASES = ('step', 'prof-apply', 'prof-launch', 'prof-read')
WORKER_CASES = ('advance_sample_emit', 'advance_token_emit', 'step_session',
                'step_session_prefill', 'prefill_tick', 'step_session_async_chain')
TEST_NAME = 'worker::rewrite_native_tests::native_worker_boundary'
TRANSFER_CONFORMANCE = (
    'PASS v1 transfer_cancel native CUDA',
    'PASS v1.1 transfer_complete_cancel native CUDA',
    'PASS v1.1 transfer_lifetime native events + injected observation loss + graph retention',
    'PASS v1.1 transfer_zero_accept Unsupported NVMe preserves owned input',
    'PASS v1.1 acceptance exhaustive native mixed batch; rejected sibling blocks publication',
    'PASS v1.2 transfer_completion_bytes native CUDA; stale epochs, ready publication, take once, authentic consumer fence',
    'PASS additive source retirement Busy while source consumer bound; host destination survives source release',
    'PASS v1.3 transfer_source_retirement native CUDA',
    'PASS v1.3 device_hand_back native CUDA',
    'PASS additive dropped destination retains backing and charge until graph retirement and acknowledgement',
    'PASS rule cancelled-restore-recovers-source native CUDA',
    'PASS rule cancel-refused-after-source-consumed native CUDA',
    'PASS native governor zero after controlled drain',
)
TRANSFER_SIZES = (4096, 65536, 1048576, 16777216, 67108864, 268435456)


def transfer_conformance(text):
    lines = [line for line in text.splitlines() if line.strip().startswith('PASS')]
    if Counter(lines) != Counter(TRANSFER_CONFORMANCE):
        raise ValueError('transfer conformance requires every exact PASS marker once, with no extras')
    return {'markers': lines}


def transfer_roundtrip(text):
    rows = []
    for line in text.splitlines():
        if not line.strip().startswith('PASS'):
            continue
        match = re.fullmatch(
            r'PASS native D2H-H2D roundtrip bytes=([1-9][0-9]*) N=1 '
            r'expected_sha256=([0-9a-f]{64}) actual_sha256=([0-9a-f]{64}) '
            r'byte_exact=true source_freed_host_live=true handback_no_copy=true governor_zero=true', line)
        if match is None or match[2] != match[3]:
            raise ValueError('transfer roundtrip requires exact rows, equal SHA256 and all lifecycle assertions')
        rows.append({'bytes': int(match[1]), 'expected_sha256': match[2], 'actual_sha256': match[3]})
    if Counter(row['bytes'] for row in rows) != Counter(TRANSFER_SIZES):
        raise ValueError('transfer roundtrip requires each of the six byte sizes exactly once')
    return {'roundtrips': rows, 'N': 1, 'byte_exact': True, 'source_freed_host_live': True,
            'handback_no_copy': True, 'governor_zero': True}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


finalization = module('native_finalization', 'native_finalization.py')


def publish_result(out, result):
    finalization.publish_result(out, result, writer=write,
                                check_cancelled=lambda: controller.check_deadline(float('inf')))


def finalize_evidence(out, telemetry, telemetry_log, verify):
    def checked():
        controller.check_deadline(float('inf'))
        verify()
        controller.check_deadline(float('inf'))
    return finalization.finalize_evidence(out, telemetry, telemetry_log, checked,
                                         writer=write, digest=build.digest)


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


def selected_failure(out, error):
    """Quarantine through the final publication boundary, then seal the failed tree."""
    cleanup_errors = []
    index = out / 'retained-eg-bundle/rewrite-receipts.tsv'
    try:
        if index.exists():
            index.rename(index.with_name('failed-caller-index.tsv'))
    except BaseException as failure:
        cleanup_errors.append(f'selected bundle quarantine: {failure}')
    try:
        result = json.loads((out / 'result.json').read_text())
    except (OSError, ValueError):
        result = {'phase': 'supported-callers', 'model_support_promotion': False,
                  'serving_binary_qualification': False}
    result.update(status='failed', selected_failure=str(error),
                  selected_bundle_quarantined=not index.exists(),
                  evidence_manifest_sha256=None)
    try:
        result['evidence_manifest_sha256'] = finalization.write_evidence_manifest(
            out, writer=write, digest=build.digest)
    except BaseException as failure:
        cleanup_errors.append(f'selected failure manifest: {failure}')
    result['selected_failure_cleanup_errors'] = cleanup_errors
    publish_result(out, result)  # Failure publication never turns cancellation into PASS.


def run(args):
    owned = {}
    try:
        return run_owned(args, owned)
    except BaseException as error:
        if getattr(args, 'phase', None) == 'supported-callers' and 'out' in owned:
            selected_failure(owned['out'], error)
        raise


def run_owned(args, owned):
    if not math.isfinite(args.timeout_seconds) or not 0 < args.timeout_seconds <= 86400:
        raise ValueError("timeout-seconds must be finite and in (0, 86400]")
    lease = admission.verify_lease()  # Before any GPU query or child.
    out, record_path = args.out.resolve(), args.build_record.resolve()
    model = args.model.resolve() if args.model else None
    caller_phase = args.phase in ('callers', 'supported-callers')
    selected = args.phase == 'supported-callers'
    if caller_phase and model is None:
        raise ValueError('--model is required for caller probes')
    if out == ROOT or ROOT in out.parents:
        raise RuntimeError('native evidence must be outside the clean checkout')
    binaries = record_path.parent / 'target/release'
    record, record_sha = build.verify_build_record(ROOT, record_path, binaries)
    if caller_phase:
        source = admission.verify_source(model)
    elif args.phase == 'transfer':
        source = {'kind': 'native-transfer-fixture', 'model_required': False}
    else:
        source = {'roster': str(args.roster.resolve()), 'roster_sha256': build.digest(args.roster)}
    out.mkdir(parents=True, exist_ok=False)
    owned['out'] = out  # Never quarantine a pre-existing/unowned evidence directory.
    write(out / 'lease.json', lease)
    write(out / 'source.json', source)
    write(out / 'build-record.json', record)
    write(out / 'build-binding.json', {'path': str(record_path), 'sha256': record_sha})
    base = {key: value for key, value in os.environ.items()
            if not key.startswith('MEMRA_') and not key.startswith('REWRITE_PROBE_')}
    base['MEMRA_GPU_LEASE_FILE'] = os.environ['MEMRA_GPU_LEASE_FILE']
    if caller_phase:
        base['MEMRA_FAST'] = '0'
    if selected:
        # CUDA/NVVM dependencies must exist before the native model freezes its
        # library identity. This is an exec-time setting, never in-process set_var.
        base['CUDA_FORCE_PRELOAD_LIBRARIES'] = '1'
    cases = []

    def verify():
        admission.verify_lease()
        build.verify_build_record(ROOT, record_path, binaries, record_sha)

    def case(name, command, changes=None, marker=None, environment_drift=False, unit=False, validate=None):
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
        text = stdout + '\n' + stderr
        passed = code == 0 and (marker is None or marker in text) and (not unit or test_success(text))
        entry = {'case': name, 'kind': 'cpu-inspect' if name.startswith('inspect-') else 'native',
                 'command': command, 'returncode': code, 'passed': passed,
                 'required_marker': marker, 'one_native_test_required': unit,
                 'stdout_sha256': build.digest(directory / 'stdout.log'),
                 'stderr_sha256': build.digest(directory / 'stderr.log')}
        if validate is not None:
            try:
                entry['validated_output'] = validate(text)
            except ValueError as error:
                passed = False
                entry.update(passed=False, validation_error=str(error))
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
    if selected:
        summary.update(selected_surfaces=['decode-eager', 'decode-graph'],
                       withheld_surfaces=['carried-prime'],
                       positive_prime_coverage=False,
                       original_three_surface_campaign='failed; 24 later cases unrun',
                       startup_environment={'CUDA_FORCE_PRELOAD_LIBRARIES': '1', 'MEMRA_FAST': '0'})
    publish_result(out, summary)
    telemetry = None
    failure = None
    with controller.cleanup_signals():
        try:
            telemetry = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,uuid,name,index,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu',
            '--format=csv', '--loop-ms=250'], stdout=telemetry_log, stderr=subprocess.STDOUT)
            if caller_phase:
                prefix = 'retained-eg' if selected else 'retained'
                bundle = out / ('retained-eg-bundle' if selected else 'retained-bundle')
                inspect('inspect-retained', bundle)
                capture = {'MEMRA_ARTIFACT_LOCK': str(bundle / 'artifact.lock')}
                case(f'{prefix}-capture', [binaries / 'rewrite_identity_gate', f'{prefix}-capture', model, bundle],
                     capture, 'RETAINED_EAGER_GRAPH_CAPTURE_PASS' if selected else 'RETAINED_CAPTURE_PASS')
                strict = {**capture, 'MEMRA_REWRITE_BUNDLE': str(bundle)}
                for api in EAGER_GRAPH_CASES if selected else GRAPH_CASES:
                    for drift in ('library', 'environment'):
                        name = f'{prefix}-{api}-{drift}'
                        case(name, [binaries / 'rewrite_identity_gate', name, model, bundle], strict,
                             'RETAINED_EAGER_GRAPH_CALLER_PASS' if selected else 'RETAINED_CALLER_PASS',
                             environment_drift=drift == 'environment')
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
                if selected:
                    for mode, marker in (
                        ('prime-api-refusal', 'UNQUALIFIED_PRIME_API_REFUSAL_PASS'),
                        ('prime-production-refusal', 'UNQUALIFIED_PRIME_PRODUCTION_REFUSAL_PASS'),
                        ('prime-fallback', 'EAGER_PRIME_FALLBACK_PASS'),
                    ):
                        case(f'{prefix}-{mode}', [binaries / 'rewrite_identity_gate', f'{prefix}-{mode}', model, bundle],
                             strict, marker)
            elif args.phase == 'transfer':
                for mode, validator in (('conformance', transfer_conformance), ('roundtrip', transfer_roundtrip)):
                    case(f'transfer-{mode}', [binaries / 'tier-transfer-gate', mode], validate=validator)
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
    parser.add_argument('--phase', choices=('callers', 'supported-callers', 'transfer', 'battery'), default='callers')
    parser.add_argument('--roster', type=Path, default=ROOT / 'tools/release-roster.tsv')
    parser.add_argument('--timeout-seconds', type=float, default=600)
    run(parser.parse_args())
