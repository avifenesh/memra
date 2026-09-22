#!/usr/bin/env python3
"""Fresh admitted non-PLE Gemma HPOST export/pooling proof; not model qualification."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time

spec = importlib.util.spec_from_file_location('identity_native', Path(__file__).with_name('qualify-native.py'))
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
ROOT = native.ROOT


def case_names():
    return ([f'direct-p{p}-t{n}' for p in (0, 11) for n in (15, 16, 17)]
            + [f'batch-{n}-row{i}' for n in (15, 17) for i in range(3)])


def validate_export_census(raw):
    names = re.findall(r'^GEMMA_CASE_PASS case=(\S+)', raw, re.M)
    if len(names) != 12 or set(names) != set(case_names()):
        raise RuntimeError('incomplete or duplicate Gemma stream census')
    f32_names = re.findall(r'^GEMMA_F32_CASE_PASS case=(\S+)', raw, re.M)
    expected = {f'f32-p{p}-t{n}' for p in (0, 11) for n in (15, 16, 17)}
    if len(f32_names) != 6 or set(f32_names) != expected:
        raise RuntimeError('incomplete or duplicate F32 prerequisite census')
    generators = re.findall(r'^GEMMA_F32_GENERATE_CASE_PASS case=(\S+)', raw, re.M)
    expected = {f'{api}-t{n}' for api in ('generate', 'generate-with') for n in (15, 16, 17)}
    if len(generators) != 6 or set(generators) != expected:
        raise RuntimeError('incomplete or duplicate F32 generator census')
    if raw.count('GEMMA_F32_PARITY_PASS streams=6 atol=0.005\n') != 1:
        raise RuntimeError('missing or duplicate F32 prerequisite verdict')
    if raw.index('GEMMA_F32_PARITY_PASS') > raw.index('GEMMA_CASE_PASS'):
        raise RuntimeError('HPOST comparison preceded F32 prerequisite')


def cross_hpost(off, on):
    """Same raw T1 program and actual pooling result in both fresh process modes."""
    results = {}
    for case in case_names():
        suffixes = ['t1-raw', 'expected-normalized', 'actual-pooling']
        suffixes += [f'{kind}-logits-{i}' for kind in ('t1', 'actual') for i in range(5)]
        for suffix in suffixes:
            name = f'{case}-{suffix}.f32'
            left, right = off / name, on / name
            if not left.is_file() or not right.is_file() or not left.stat().st_size:
                raise RuntimeError(f'missing/empty cross-HPOST output: {name}')
            a, b = native.digest(left), native.digest(right)
            results[name] = {'off': a, 'on': b, 'bitwise_equal': a == b}
    return results


def fail_attempt(out, summary, error, errors):
    # Revoke PASS before any fallible archive/index operation.
    summary.update(status='failed', error=str(error), finalization_errors=list(errors))
    try:
        native.finalization.publish_result(out, summary, writer=native.write_json)
    except BaseException as publication_error:
        summary['finalization_errors'].append(f'failed status publication: {publication_error}')
        # Preserve any earlier result without leaving a live PASS if writing the
        # replacement failed. Quarantine indexes even if this rename also fails.
        try:
            result = out / 'result.json'
            prior = out / 'result.publication-error.json'
            if result.exists():
                if prior.exists():
                    raise RuntimeError('refusing to overwrite the prior publication error')
                result.rename(prior)
        except BaseException as archive_error:
            summary['finalization_errors'].append(f'result quarantine: {archive_error}')
    for hpost in (0, 1):
        index = out / f'hpost-{hpost}/rewrite-receipts.tsv'
        try:
            if index.exists():
                # A previous failure archive must never prevent revoking the live
                # admission index. link() reserves a same-filesystem destination
                # without overwriting any existing evidence, including a race.
                number = 0
                while True:
                    suffix = '' if number == 0 else f'-{number}'
                    destination = index.with_name(f'rewrite-receipts.failed{suffix}.tsv')
                    try:
                        os.link(index, destination)
                        break
                    except FileExistsError:
                        number += 1
                index.unlink()
        except BaseException as archive_error:
            summary['finalization_errors'].append(f'index quarantine: {archive_error}')
    try:
        summary['evidence_manifest_sha256'] = native.finalization.write_evidence_manifest(
            out, writer=native.write_json, digest=native.digest)
    except BaseException as archive_error:
        summary['finalization_errors'].append(f'evidence manifest: {archive_error}')
    native.finalization.publish_result(out, summary, writer=native.write_json)


def run_owned(command, log, env, deadline):
    """All owned subprocesses, including GPU inventory, share deadline/cleanup rules."""
    native.controller.check_deadline(deadline)
    native.verify_lease()
    process = subprocess.Popen([str(x) for x in command], cwd=ROOT, env=env,
                               stdout=log, stderr=subprocess.STDOUT)
    try:
        while process.poll() is None:
            native.controller.check_deadline(deadline)
            native.verify_lease()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
        native.controller.check_deadline(deadline)
        native.verify_lease()
        return process.returncode
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=10)


def run(args):
    if os.environ.get('DOCS_RS') is not None:
        raise RuntimeError('DOCS_RS is not a native build')
    out, model, binaries = args.out.resolve(), args.model.resolve(), args.bin_dir.resolve()
    if out.is_relative_to(ROOT.resolve()) or out.exists():
        raise RuntimeError('use a fresh output namespace outside the source checkout')
    if not model.is_file() or model.suffix != '.gguf' or not re.fullmatch('[a-f0-9]{64}', args.model_sha256):
        raise RuntimeError('select a real GGUF and its independently approved complete SHA256')
    lease = native.verify_lease()
    build, build_sha = native.build_record.verify_build_record(ROOT, args.build_record, binaries)

    def verify_model():
        if native.digest(model) != args.model_sha256:
            raise RuntimeError('selected Gemma source bytes changed')

    verify_model()
    out.mkdir(parents=True, exist_ok=False)
    native.write_json(out / 'source.json', {'model': str(model), 'bytes': model.stat().st_size,
                                           'sha256': args.model_sha256, 'family': 'gemma4_dense'})
    native.write_json(out / 'lease.json', lease)
    native.write_json(out / 'build-record.json', build)
    native.write_json(out / 'build-record-binding.json', {'sha256': build_sha})
    base = {k: v for k, v in os.environ.items() if not k.startswith('MEMRA_')}
    base.update(MEMRA_GPU_LEASE_FILE=os.environ['MEMRA_GPU_LEASE_FILE'], MEMRA_FAST='0', NVIDIA_TF32_OVERRIDE='0')
    deadline = time.monotonic() + args.timeout
    cases = []
    telemetry = None
    telemetry_log = (out / 'telemetry.csv').open('wb')
    summary = {'status': 'incomplete', 'scope': 'admitted non-PLE Gemma HPOST prime exports and actual hidden_postnorm_row consumer',
               'support_promotion': False, 'build_record_sha256': build_sha, 'tolerance': 0.005,
               'hpost_modes': [0, 1], 'streams_per_mode': 12}
    native.finalization.publish_result(out, summary, writer=native.write_json)

    def invariant():
        native.controller.check_deadline(deadline)
        native.verify_lease()
        native.build_record.verify_build_record(ROOT, args.build_record, binaries, build_sha)
        if telemetry is not None and telemetry.poll() is not None:
            raise RuntimeError('telemetry stopped before the native campaign completed')

    def case(name, command, env, marker=None):
        invariant()
        started = time.monotonic()
        code = None
        try:
            with (out / f'{name}.log').open('wb') as log:
                code = run_owned(command, log, env, deadline)
            invariant()
            raw = (out / f'{name}.log').read_text(errors='replace')
            if code != 0 or (marker and marker not in raw):
                raise RuntimeError(f'{name} failed; preserve the attempt, do not retry')
            if name.startswith('export-pooling-'):
                validate_export_census(raw)
        finally:
            cases.append({'case': name, 'command': [str(x) for x in command], 'exit_code': code,
                          'seconds': time.monotonic()-started,
                          'log_sha256': native.digest(out / f'{name}.log') if (out / f'{name}.log').exists() else None})
            native.write_json(out / 'cases.json', cases)

    failure = None
    with native.controller.cleanup_signals():
        try:
            with (out / 'compute-entry.csv').open('wb') as log:
                code = run_owned(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], log, base, deadline)
            if code != 0:
                raise RuntimeError('compute inventory failed; raw output preserved')
            activity = (out / 'compute-entry.csv').read_text()
            if any(row.split(',')[0].strip() in lease['requested_uuids'] for row in activity.splitlines()):
                raise RuntimeError('leased GPU has an existing compute process')
            telemetry = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu', '--format=csv', '--loop-ms=250'], stdout=telemetry_log, stderr=subprocess.STDOUT)
            for hpost in (0, 1):
                bundle = out / f'hpost-{hpost}'
                env = {**base, 'MEMRA_SPEC_HPOST': str(hpost)}
                case(f'inspect-{hpost}', [binaries/'memra', 'model', 'inspect', model, '--against', 'gemma4_dense', '--out', bundle], env)
                env['MEMRA_ARTIFACT_LOCK'] = str(bundle/'artifact.lock')
                # Existing independent verify-prefill vs T1 gate creates real Eager admission.
                # Its existing pack policy is unchanged. A failed prerequisite stops the phase.
                case(f'admit-eager-{hpost}', [binaries/'rewrite_identity_gate', 'capture', model, bundle], env,
                     'REWRITE_IDENTITY_GATE_PASS mode=capture')
                env['MEMRA_REWRITE_BUNDLE'] = str(bundle)
                case(f'export-pooling-{hpost}', [binaries/'rewrite_identity_gate', 'gemma-hpost', model, bundle], env,
                     f'GEMMA_HPOST_PASS hpost={str(bool(hpost)).lower()} streams=12')
            comparisons = cross_hpost(out/'hpost-0/gemma-hpost', out/'hpost-1/gemma-hpost')
            native.write_json(out/'cross-hpost.json', comparisons)
            if not all(x['bitwise_equal'] for x in comparisons.values()):
                raise RuntimeError('HPOST changed raw T1, logits or actual pooling bytes')
            invariant()
            verify_model()
        except BaseException as error:
            failure = error

        def final_verify():
            native.controller.check_deadline(deadline)
            native.verify_lease()
            native.build_record.verify_build_record(ROOT, args.build_record, binaries, build_sha)
            verify_model()

        errors, manifest_sha = native.finalization.finalize_evidence(out, telemetry, telemetry_log, final_verify,
            writer=native.write_json, digest=native.digest)
        summary.update(evidence_manifest_sha256=manifest_sha, cases=len(cases))
        try:
            if failure is not None or errors:
                raise RuntimeError(f'Gemma phase failed: {failure}; finalization={errors}')
            summary['status'] = 'passed'
            native.finalization.publish_result(out, summary, writer=native.write_json,
                check_cancelled=lambda: native.controller.check_deadline(deadline))
        except BaseException as error:
            # Every failure, including cancellation during PASS publication, revokes these new
            # indexes while preserving their original bytes. Historical bundles are untouched.
            fail_attempt(out, summary, error, errors)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-sha256', required=True)
    parser.add_argument('--bin-dir', type=Path, required=True)
    parser.add_argument('--build-record', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 3600:
        parser.error('timeout must be between 1 and 3600 seconds')
    run(args)
