#!/usr/bin/env python3
"""Native #542 admission checks. Run GPU phase only inside the central per-card lock wrapper."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
REPO = 'Qwen/Qwen3-0.6B'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
WEIGHT_SHA = 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
WEIGHT_BYTES = 1503300328
UUID = re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')
_build_spec = importlib.util.spec_from_file_location('native_build_record', Path(__file__).with_name('native_build_record.py'))
build_record = importlib.util.module_from_spec(_build_spec)
_build_spec.loader.exec_module(build_record)


def digest(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def prepare(path):
    """CPU/network only: stage the fixed source revision without running a GPU command."""
    path.mkdir(parents=True, exist_ok=True)
    api = f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true'
    with urllib.request.urlopen(api, timeout=60) as response:
        metadata = json.load(response)
    if metadata['sha'] != REVISION:
        raise RuntimeError('source revision mismatch')
    known = {'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
             'vocab.json', 'merges.txt', 'chat_template.jinja', 'model.safetensors', 'LICENSE'}
    names = sorted(x['rfilename'] for x in metadata['siblings'] if x['rfilename'] in known)
    if not {'config.json', 'tokenizer.json', 'model.safetensors'} <= set(names):
        raise RuntimeError('pinned source is missing required files')
    for name in names:
        target = path / name
        temporary = path / (name + '.download')
        # Existing weight bytes are reusable only when they match the independent pin.
        if name == 'model.safetensors' and target.exists() and digest(target) == WEIGHT_SHA:
            continue
        try:
            with urllib.request.urlopen(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}', timeout=120) as response, temporary.open('wb') as output:
                shutil.copyfileobj(response, output, 1024 * 1024)
            if name == 'model.safetensors' and (temporary.stat().st_size != WEIGHT_BYTES or digest(temporary) != WEIGHT_SHA):
                raise RuntimeError('pinned weight size/hash mismatch')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    write_json(path / 'qualification-source.json', {
        'source': f'{REPO}@{REVISION}',
        'files': {name: {'sha256': digest(path / name), 'bytes': (path / name).stat().st_size} for name in names},
    })
    print(f'PREPARED {REPO}@{REVISION}: {path}')


def ancestors(pid):
    result = set()
    while pid > 0 and pid not in result:
        result.add(pid)
        status = Path(f'/proc/{pid}/status').read_text()
        pid = int(next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:')))
    return result


def verify_lock_records(lease, visible, lineage, lock_rows, metadata):
    """Require ownership by the wrapper ancestor, not just a locked pathname or a marker."""
    requested = lease['requested_uuids']
    if not requested or len(set(requested)) != len(requested) or not all(UUID.fullmatch(x) for x in requested):
        raise RuntimeError('lease requires unique full physical GPU UUIDs')
    if visible.split(',') != requested:
        raise RuntimeError('CUDA_VISIBLE_DEVICES differs from requested physical GPU set/order')
    if lease['lock_order'] != sorted(requested) or set(lease['lock_files']) != set(requested):
        raise RuntimeError('lease does not contain exactly the complete sorted GPU lock set')
    wrapper = int(lease['wrapper_pid'])
    if wrapper not in lineage or int(lease['child_pid']) not in lineage or wrapper == os.getpid():
        raise RuntimeError('lease wrapper/child is not an ancestor of this session')
    for gpu in requested:
        expected_path = f'/tmp/memra-gpu-locks/{gpu}.lock'
        if lease['lock_files'][gpu] != expected_path:
            raise RuntimeError('noncanonical per-card lock path')
        info = metadata(expected_path)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError('GPU lock is not a regular file')
        device_inode = (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino)
        found = False
        for line in lock_rows.splitlines():
            fields = line.split()
            if len(fields) < 8 or fields[1:4] != ['FLOCK', 'ADVISORY', 'WRITE'] or fields[4] != str(wrapper):
                continue
            parts = fields[5].split(':')
            if len(parts) == 3 and (int(parts[0], 16), int(parts[1], 16), int(parts[2])) == device_inode:
                found = fields[6:] == ['0', 'EOF']
                if found:
                    break
        if not found:
            raise RuntimeError(f'wrapper ancestor does not hold exclusive FLOCK for {gpu}')


def verify_lease():
    if sys.platform != 'linux':
        raise RuntimeError('GPU phase requires Linux /proc lock verification')
    lease_path = os.environ.get('MEMRA_GPU_LEASE_FILE')
    if not lease_path:
        raise RuntimeError('MEMRA_GPU_LEASE_FILE missing: use the provided memra-gpu-run wrapper')
    lease = json.loads(Path(lease_path).read_text())
    verify_lock_records(lease, os.environ.get('CUDA_VISIBLE_DEVICES', ''), ancestors(os.getpid()),
                        Path('/proc/locks').read_text(), os.lstat)
    if len(lease['requested_uuids']) != 1:
        raise RuntimeError('this stage uses exactly one GPU; acquire only that one card')
    return lease


def verify_source(model):
    manifest = json.loads((model / 'qualification-source.json').read_text())
    if manifest['source'] != f'{REPO}@{REVISION}' or manifest['files']['model.safetensors']['sha256'] != WEIGHT_SHA:
        raise RuntimeError('identity stage requires the fixed official source pin; run --prepare first')
    for name, expected in manifest['files'].items():
        if Path(name).name != name or digest(model / name) != expected['sha256'] or (model / name).stat().st_size != expected['bytes']:
            raise RuntimeError(f'prepared source changed: {name}')
    return manifest


def mutate_weight(source, destination):
    destination.mkdir()
    for file in source.iterdir():
        if file.is_file():
            shutil.copy2(file, destination / file.name)
    weights = destination / 'model.safetensors'
    with weights.open('r+b') as stream:
        header_bytes = int.from_bytes(stream.read(8), 'little')
        header = json.loads(stream.read(header_bytes))
        name, tensor = next((n, t) for n, t in sorted(header.items()) if n != '__metadata__' and t['dtype'] in ('BF16', 'F16', 'F32'))
        offset = 8 + header_bytes + tensor['data_offsets'][0]
        stream.seek(offset)
        before = stream.read(1)
        stream.seek(offset)
        stream.write(bytes([before[0] ^ 1]))
    return {'tensor': name, 'byte_offset': offset, 'sha256': digest(weights),
            'original_sha256': WEIGHT_SHA, 'change': 'one payload low bit; shape/config unchanged'}


def case_passed(returncode, refusal, text):
    if refusal:
        return returncode == 1 and 'REWRITE_IDENTITY_GATE_FAIL:' in text and refusal in text
    return returncode == 0


def output_hashes(text, stage):
    pattern = re.compile(rf'^OUTPUT stage={re.escape(stage)} prompt=(\d+).* sha256=([0-9a-f]{{64}})$', re.M)
    pairs = pattern.findall(text)
    values = dict(pairs)
    if len(pairs) != len(values) or set(values) != {'0', '1', '2'}:
        raise RuntimeError(f'incomplete or duplicate native output hashes for {stage}')
    return values


def run(args):
    lease = verify_lease()  # No GPU command precedes the actual ancestor-FLOCK check.
    if os.environ.get('DOCS_RS') is not None:
        raise RuntimeError('DOCS_RS placeholder builds are not native qualification')
    model, out, binaries = args.model.resolve(), args.out.resolve(), args.binary_dir.resolve()
    if out == ROOT.resolve() or ROOT.resolve() in out.parents:
        raise RuntimeError('qualification output must be outside the clean source checkout')
    provenance, provenance_sha256 = build_record.verify_build_record(ROOT, args.build_record, binaries)

    def verify_build():
        build_record.verify_build_record(ROOT, args.build_record, binaries, provenance_sha256)

    source = verify_source(model)
    if args.mtp_model and (not args.mtp_sha256 or digest(args.mtp_model) != args.mtp_sha256):
        raise RuntimeError('MTP artifact requires its exact owner-supplied SHA-256')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'lease.json', lease)
    write_json(out / 'source.json', source)
    tools = {name: binaries / name for name in build_record.BINARIES}
    write_json(out / 'binaries.json', {name: row['sha256'] for name, row in provenance['binaries'].items()})
    write_json(out / 'build-record.json', provenance)
    write_json(out / 'build-record-binding.json', {'path': str(args.build_record.resolve()), 'sha256': provenance_sha256})
    (out / 'source-commit.txt').write_text(provenance['source']['commit'] + '\n')
    base = {k: v for k, v in os.environ.items() if not k.startswith('MEMRA_')}
    base['MEMRA_GPU_LEASE_FILE'] = os.environ['MEMRA_GPU_LEASE_FILE']
    results = []

    def case(name, command, changes=None, refusal=None):
        verify_lease()
        verify_build()
        start = time.monotonic()
        with (out / f'{name}.log').open('wb') as log:
            process = subprocess.Popen([str(x) for x in command], cwd=ROOT, env={**base, **(changes or {})}, stdout=log, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    verify_lease()
                    try:
                        process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        pass
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        verify_lease()  # A lost wrapper cannot turn an interrupted row into completed evidence.
        elapsed = time.monotonic() - start
        verify_build()  # Changed source/tool bytes cannot publish a passed row.
        text = (out / f'{name}.log').read_text(errors='replace')
        passed = case_passed(process.returncode, refusal, text)
        record = {'case': name, 'command': [str(x) for x in command], 'returncode': process.returncode,
                  'expected_refusal': refusal, 'passed': passed, 'wall_seconds': elapsed,
                  'log_sha256': digest(out / f'{name}.log')}
        results.append(record)
        write_json(out / 'cases.json', results)
        print(json.dumps(record), flush=True)
        if not passed:
            raise RuntimeError(f'{name} failed; see {out / (name + ".log")}')
        return text

    verify_build()  # Recheck after artifact preflight, before the FIRST GPU command.
    with (out / 'compute-entry.csv').open('wb') as log:
        subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv'], stdout=log, stderr=subprocess.STDOUT, check=True)
    for line in (out / 'compute-entry.csv').read_text().splitlines()[1:]:
        if line.split(',')[0].strip() in lease['requested_uuids']:
            raise RuntimeError('a compute process already occupies the locked GPU')
    verify_build()
    telemetry_file = (out / 'all-card-telemetry-250ms.csv').open('wb')
    telemetry = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,uuid,name,index,memory.total,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu', '--format=csv', '--loop-ms=250'], stdout=telemetry_file, stderr=subprocess.STDOUT)
    variant = out / 'wrong-weights'
    changed_binary = out / 'changed-executable'
    try:
        bundle = out / 'bundle'
        case('inspect', [tools['memra'], 'model', 'inspect', model, '--against', 'qwen3', '--out', bundle])
        capture = {'MEMRA_ARTIFACT_LOCK': str(bundle / 'artifact.lock')}
        strict = {**capture, 'MEMRA_REWRITE_BUNDLE': str(bundle)}
        captured = case('native-capture-and-reinstall', [tools['rewrite_identity_gate'], 'capture', model, bundle], capture)
        checked = case('native-positive-fresh-process', [tools['rewrite_identity_gate'], 'check', model, bundle], strict)
        expected = output_hashes(captured, 'installed-eager')
        actual = output_hashes(checked, 'check-eager')
        write_json(out / 'fresh-process-output-parity.json', {'expected': expected, 'actual': actual, 'passed': expected == actual})
        if expected != actual:
            raise RuntimeError('fresh-process eager output differs despite matching runtime identity')
        case('missing-bundle', [tools['rewrite_identity_gate'], 'check', model, bundle], {**strict, 'MEMRA_REWRITE_BUNDLE': str(out / 'missing')}, 'read artifact.lock')
        case('different-numerical-program', [tools['rewrite_identity_gate'], 'check', model, bundle], {**strict, 'MEMRA_FAST': '0'}, 'does not bind numeric_program_sha256=')
        write_json(out / 'weight-mutation.json', mutate_weight(model, variant))
        case('different-weights-same-geometry', [tools['rewrite_identity_gate'], 'check', variant, bundle], strict, 'does not bind artifact_sha256=')
        shutil.copy2(tools['rewrite_identity_gate'], changed_binary)
        with changed_binary.open('ab') as stream:
            stream.write(b'\nmemra-542-binary-identity-negative\n')
        write_json(out / 'binary-mutation.json', {'sha256': digest(changed_binary), 'original_sha256': digest(tools['rewrite_identity_gate']), 'change': 'appended inert bytes to ELF'})
        case('different-executable-same-source', [changed_binary, 'check', model, bundle], strict, 'does not bind implementation_sha256=')
        fresh = out / 'fresh-kv-control'
        fresh.mkdir()
        shutil.copy2(bundle / 'artifact.lock', fresh / 'artifact.lock')
        fresh_text = case('fresh-kv-output-control-unqualified', [tools['rewrite_identity_gate'], 'fresh-control', model, fresh], {'MEMRA_ARTIFACT_LOCK': str(fresh / 'artifact.lock')})
        if (fresh / 'rewrite-receipts.tsv').exists():
            raise RuntimeError('unqualified fresh-KV control emitted a qualification receipt')
        write_json(out / 'separate-program-outputs.json', {'cached': expected, 'fresh': output_hashes(fresh_text, 'fresh-kv-diagnostic'), 'same_program': False})
        replayed = case('cached-replay-after-isolated-fresh-control', [tools['rewrite_identity_gate'], 'check', model, bundle], strict)
        if output_hashes(replayed, 'check-eager') != expected:
            raise RuntimeError('isolated fresh-KV diagnostic changed cached eager output')
        case('legacy-argmax-regression', [tools['run-gen'], model, '1', '2', '3', '4'], {'MEMRA_NGEN': '32'})
        case('legacy-batch-regression', [tools['decode-batch-gate'], model, '--mode', 'config', '--batch', '2', '--steps', '16'])
        if args.mtp_model:
            if not args.mtp_sha256 or digest(args.mtp_model) != args.mtp_sha256:
                raise RuntimeError('MTP artifact requires its exact owner-supplied SHA-256')
            case('mtp-k1-through-k8', [tools['run-spec'], args.mtp_model, '1', '2', '3', '4'], {'MEMRA_NGEN': '32'})
        verify_build()
        write_json(out / 'result.json', {'status': 'passed', 'scope': 'single-card native identity admission and named regression cases only',
            'build_record_sha256': provenance_sha256,
            'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'mtp': 'ran' if args.mtp_model else 'pending exact artifact', 'multi_card': 'pending separate locked pair',
            'model_support_promotion': False, 'serving_binary_qualification': False})
    finally:
        telemetry.terminate()
        telemetry.wait(timeout=15)
        telemetry_file.close()
        changed_binary.unlink(missing_ok=True)
        if variant.exists():
            shutil.rmtree(variant)
        write_json(out / 'files-sha256.json', {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'files-sha256.json'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', type=Path, help='CPU/network-only download of the fixed source pin')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--binary-dir', type=Path, default=ROOT / 'target/release')
    parser.add_argument('--build-record', type=Path, help='required owned build.json from native_build_record.py')
    parser.add_argument('--mtp-model', type=Path)
    parser.add_argument('--mtp-sha256')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
    elif args.model and args.out:
        if not args.build_record:
            parser.error('--build-record is required; run native_build_record.py before acquiring GPUs')
        run(args)
    else:
        parser.error('use --prepare DIR or --model DIR --out NEW_RECEIPT_DIR')


if __name__ == '__main__':
    main()
