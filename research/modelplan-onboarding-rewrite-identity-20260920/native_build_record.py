#!/usr/bin/env python3
"""Build the native qualification tools and seal an owned build record (no GPU execution).

Run on the build host from a clean integrated checkout, before acquiring GPUs:
  python3 research/modelplan-onboarding-rewrite-identity-20260920/native_build_record.py \
    --out /scratch/admission-build-001 --cuda-arch 120a --nvcc /usr/local/cuda/bin/nvcc

The NEW output directory must be outside the checkout. It owns target/, cargo logs and
build.json; there is intentionally no command to bless existing target/release files.
Use build.json and target/release with qualify-native.py. Native worker/repack/Gemma
test executables are built in a separate test-target and sealed in the same record. The recipe uses default Cargo
features, --locked, release, a fresh target and a small explicit environment. Ambient
RUSTFLAGS, compiler wrappers and MEMRA tuning flags are not inherited. Cargo config,
compiler identities and effective environment are recorded and checked for changes.
This is an auditable local build record, not a signature or a hermetic toolchain claim.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PRODUCER = 'research/modelplan-onboarding-rewrite-identity-20260920/native_build_record.py'
RUNNER = 'research/modelplan-onboarding-rewrite-identity-20260920/qualify-native.py'
SCHEMA = 'memra-native-owned-build-v2'
BINARIES = ('memra', 'rewrite_identity_gate', 'run-gen', 'decode-batch-gate', 'run-spec', 'kernel-check', 'argmax-margin-probe', 'concat-prime-probe', 'tier-transfer-gate')
CALLER_RUNNER = 'research/modelplan-onboarding-rewrite-identity-20260920/qualify-callers.py'
ENV_CONTROLLER = 'research/modelplan-onboarding-rewrite-identity-20260920/native_env_controller.py'
GEMMA_RUNNER = 'research/modelplan-onboarding-rewrite-identity-20260920/qualify-gemma-hidden.py'
SOURCE_PATHS = ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', '.cargo', 'crates', PRODUCER, RUNNER,
                CALLER_RUNNER, ENV_CONTROLLER, GEMMA_RUNNER, 'research/modelplan-onboarding-rewrite-identity-20260920/native_finalization.py')
TEST_TARGETS = {
    'engine': ('memra-engine', '--lib', None, 'memra_engine'),
    'worker': ('memra-server', '--lib', None, 'memra_server'),
    'repack': ('memra-engine', '--test', 'native_repack_gpu', 'native_repack_gpu'),
    'gemma-prime': ('memra-engine', '--test', 'gemma4_chunked_prime_gpu', 'gemma4_chunked_prime_gpu'),
}
NATIVE_TOOLS = ('cc', 'c++', 'gcc', 'g++', 'ar', 'ld')
INHERITED_ENV = ('PATH', 'HOME', 'CARGO_HOME', 'RUSTUP_HOME', 'RUSTUP_TOOLCHAIN', 'TMPDIR')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_identity(path, executable=False):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or (executable and (not info.st_size or not os.access(path, os.X_OK))):
        raise RuntimeError(f'not a regular {"executable" if executable else "file"}: {path}')
    return {'sha256': digest(path), 'bytes': info.st_size}


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def source_identity(root):
    """Bind actual source bytes as well as HEAD; distrust index stat/skip-worktree hints."""
    head = git(root, 'rev-parse', 'HEAD').decode().strip()
    if git(root, 'status', '--porcelain=v1', '--untracked-files=all', '--ignore-submodules=none'):
        raise RuntimeError('native build requires a clean source checkout (including untracked files)')
    # Ignored source/config can still be picked up by rustc/nvcc/Cargo.
    if git(root, 'ls-files', '--others', '--ignored', '--exclude-standard', '--', 'crates', '.cargo'):
        raise RuntimeError('ignored files in build source/config directories are not bound to HEAD')
    algorithm = git(root, 'rev-parse', '--show-object-format').decode().strip()
    files = {}
    for row in git(root, 'ls-tree', '-rz', 'HEAD', '--', *SOURCE_PATHS).split(b'\0'):
        if not row:
            continue
        metadata, raw_name = row.split(b'\t', 1)
        mode, kind, oid = metadata.decode().split()
        name = os.fsdecode(raw_name)
        path = root / name
        if kind != 'blob' or mode not in ('100644', '100755'):
            raise RuntimeError(f'unsupported source entry: {name}')
        info = file_identity(path)
        blob = hashlib.new(algorithm, f'blob {info["bytes"]}\0'.encode())
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                blob.update(chunk)
        actual_mode = '100755' if path.stat().st_mode & 0o111 else '100644'
        if blob.hexdigest() != oid or actual_mode != mode:
            raise RuntimeError(f'source bytes/mode differ from HEAD: {name}')
        files[name] = {**info, 'mode': mode}
    for required in ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', PRODUCER, RUNNER, GEMMA_RUNNER):
        if required not in files:
            raise RuntimeError(f'missing tracked build input: {required}')
    if head != git(root, 'rev-parse', 'HEAD').decode().strip():
        raise RuntimeError('source changed during identity check')
    return {'commit': head, 'tree': git(root, 'rev-parse', 'HEAD^{tree}').decode().strip(),
            'files': files, 'content_sha256': content_hash(files)}


def config_identity(root, env):
    paths = {directory / '.cargo' / name for directory in (root, *root.parents)
             for name in ('config', 'config.toml')}
    cargo_home = Path(env.get('CARGO_HOME', str(Path(env['HOME']) / '.cargo')))
    paths.update(cargo_home / name for name in ('config', 'config.toml'))
    result = {}
    for path in sorted(paths):
        if path.exists():
            # Cargo's [env] can inject placeholder builds even with a clean shell.
            if b'DOCS_RS' in path.read_bytes():
                raise RuntimeError(f'DOCS_RS in Cargo config cannot qualify a native build: {path}')
            result[str(path)] = file_identity(path)
    return result


def build_environment(nvcc, arch):
    if 'DOCS_RS' in os.environ:
        raise RuntimeError('DOCS_RS placeholder builds are not native qualification')
    if arch not in ('120a', '100a', '90a', '89'):
        raise RuntimeError('explicit supported CUDA build architecture required')
    env = {key: os.environ[key] for key in INHERITED_ENV if key in os.environ}
    env.update({'MEMRA_NVCC': str(nvcc), 'MEMRA_CUDA_ARCH': arch, 'CUDA_VISIBLE_DEVICES': '',
                'CARGO_INCREMENTAL': '0'})
    return env


def tool_identity(path, version_args, root, env):
    path = path.resolve(strict=True)
    version = subprocess.check_output([str(path), *version_args], cwd=root, env=env, stderr=subprocess.STDOUT).decode().strip()
    if not version:
        raise RuntimeError(f'empty compiler version: {path}')
    return {'path': str(path), **file_identity(path, executable=True), 'version': version}


def build_inputs(root, env):
    tools = {}
    for name in ('cargo', 'rustc'):
        # Hash the selected compiler, not rustup's same-named proxy executable.
        path = subprocess.check_output(['rustup', 'which', name], cwd=root, env=env).decode().strip()
        tools[name] = tool_identity(Path(path), ['-Vv'], root, env)
    tools['nvcc'] = tool_identity(Path(env['MEMRA_NVCC']), ['--version'], root, env)
    for name in NATIVE_TOOLS:
        path = shutil.which(name, path=env['PATH'])
        if not path:
            raise RuntimeError(f'missing native build tool: {name}')
        tools[name] = tool_identity(Path(path), ['--version'], root, env)
    return {'environment': env, 'tools': tools, 'cargo_configs': config_identity(root, env)}


def verify_build_inputs(root, inputs):
    """Verify recorded compiler bytes without executing anything supplied by the record."""
    env, tools = inputs['environment'], inputs['tools']
    allowed_env = set(INHERITED_ENV) | {'MEMRA_NVCC', 'MEMRA_CUDA_ARCH', 'CUDA_VISIBLE_DEVICES', 'CARGO_INCREMENTAL', 'RUSTC'}
    if (not isinstance(env, dict) or not set(env) <= allowed_env
            or not all(isinstance(value, str) for value in env.values())
            or env.get('MEMRA_CUDA_ARCH') not in ('120a', '100a', '90a', '89')
            or env.get('CUDA_VISIBLE_DEVICES') != '' or env.get('CARGO_INCREMENTAL') != '0'):
        raise RuntimeError('missing/invalid native build environment')
    if set(tools) != {'cargo', 'rustc', 'nvcc', *NATIVE_TOOLS}:
        raise RuntimeError('incomplete build compiler identities')
    for tool in tools.values():
        path = Path(tool['path'])
        if (not path.is_absolute() or not isinstance(tool['version'], str) or not tool['version']
                or file_identity(path, executable=True) != {key: tool[key] for key in ('sha256', 'bytes')}):
            raise RuntimeError('build compiler bytes changed since owned build')
    if (env['RUSTC'] != tools['rustc']['path'] or str(Path(env['MEMRA_NVCC']).resolve()) != tools['nvcc']['path']
            or inputs['cargo_configs'] != config_identity(root, env)):
        raise RuntimeError('build tools/config inputs changed since owned build')


def build_command(inputs, out):
    return [inputs['tools']['cargo']['path'], 'build', '--locked', '--release',
            '--message-format=json', '--target-dir', str(out / 'target'),
            '-p', 'memra-cli', '--bin', 'memra', '-p', 'memra-engine',
            *[arg for name in BINARIES[1:] for arg in ('--bin', name)]]


def cargo_artifacts(path, binary_dir):
    artifacts, finished = {}, []
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if not isinstance(event, dict):
            raise RuntimeError('invalid Cargo build event')
        if event.get('reason') == 'build-finished':
            finished.append(event.get('success') is True)
        if event.get('reason') != 'compiler-artifact' or not event.get('executable'):
            continue
        name = event['target']['name']
        if name not in BINARIES:
            continue
        if name in artifacts or event.get('fresh') is not False or Path(event['executable']) != binary_dir / name:
            raise RuntimeError(f'not a fresh owned Cargo executable: {name}')
        artifacts[name] = file_identity(binary_dir / name, executable=True)
    if finished != [True] or set(artifacts) != set(BINARIES):
        raise RuntimeError('incomplete/unsuccessful Cargo build: missing completion or executable outputs')
    return artifacts


def test_command(inputs, out, key):
    package, selector, name, _ = TEST_TARGETS[key]
    command = [inputs['tools']['cargo']['path'], 'test', '--locked', '--release', '--no-run',
               '--message-format=json', '--target-dir', str(out / 'test-target'), '-p', package, selector]
    return command + ([name] if name else [])


def test_artifact(path, out, key):
    artifacts, finished = [], []
    expected_name = TEST_TARGETS[key][3]
    root = (out / 'test-target/release/deps').resolve()
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event.get('reason') == 'build-finished':
            finished.append(event.get('success') is True)
        if (event.get('reason') != 'compiler-artifact' or not event.get('executable')
                or event.get('target', {}).get('name') != expected_name
                or event.get('profile', {}).get('test') is not True):
            continue
        executable = Path(event['executable'])
        if event.get('fresh') is not False or executable.parent.resolve() != root or executable.is_symlink():
            raise RuntimeError(f'not a fresh owned native test executable: {key}')
        artifacts.append({'path': str(executable.relative_to(out)), **file_identity(executable, executable=True)})
    if finished != [True] or len(artifacts) != 1:
        raise RuntimeError(f'incomplete/unsuccessful native test build: {key}')
    return artifacts[0]


def build_tests(root, out, inputs):
    records = {}
    for key in TEST_TARGETS:
        command = test_command(inputs, out, key)
        events, stderr = out / f'{key}-events.jsonl', out / f'{key}-stderr.log'
        with events.open('xb') as stdout, stderr.open('xb') as errors:
            result = subprocess.run(command, cwd=root, env=inputs['environment'], stdout=stdout,
                                    stderr=errors, check=False)
        if result.returncode != 0:
            raise RuntimeError(f'native test build {key} failed ({result.returncode}); see {stderr}')
        records[key] = {'command': command, 'artifact': test_artifact(events, out, key),
                        'logs': {file.name: file_identity(file) for file in (events, stderr)}}
    return records


def verify_test_builds(out, inputs, records):
    if set(records) != set(TEST_TARGETS):
        raise RuntimeError('incomplete native test build records')
    for key, record in records.items():
        events, stderr = out / f'{key}-events.jsonl', out / f'{key}-stderr.log'
        if (record['command'] != test_command(inputs, out, key)
                or record['artifact'] != test_artifact(events, out, key)
                or record['logs'] != {file.name: file_identity(file) for file in (events, stderr)}):
            raise RuntimeError(f'native test executable/log/command changed since build: {key}')


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def build(root, out, nvcc, arch):
    root, out, nvcc = root.resolve(), out.resolve(), nvcc.resolve(strict=True)
    if out == root or root in out.parents:
        raise RuntimeError('owned build output must be outside the source checkout')
    source = source_identity(root)
    env = build_environment(nvcc, arch)
    inputs = build_inputs(root, env)
    env['RUSTC'] = inputs['tools']['rustc']['path']
    out.mkdir(parents=True, exist_ok=False)  # Never stamp or reuse a previous build.
    command = build_command(inputs, out)
    started = utc_now()
    with (out / 'cargo-events.jsonl').open('xb') as events, (out / 'cargo-stderr.log').open('xb') as log:
        process = subprocess.run(command, cwd=root, env=env, stdout=events, stderr=log, check=False)
    if process.returncode != 0:
        raise RuntimeError(f'owned native build failed ({process.returncode}); see {out / "cargo-stderr.log"}')
    binaries = cargo_artifacts(out / 'cargo-events.jsonl', out / 'target/release')
    tests = build_tests(root, out, inputs)
    # Test builds use a separate target and may not replace qualified production tools.
    if binaries != cargo_artifacts(out / 'cargo-events.jsonl', out / 'target/release'):
        raise RuntimeError('test compilation changed production executables')
    if source_identity(root) != source or build_inputs(root, env) != inputs:
        raise RuntimeError('source/build inputs changed during owned build; no success record emitted')
    record = {'schema': SCHEMA, 'producer': file_identity(root / PRODUCER), 'status': 'success',
              'started_utc': started, 'completed_utc': utc_now(), 'returncode': process.returncode,
              'source': source, 'inputs': inputs, 'inputs_sha256': content_hash(inputs),
              'command': command, 'binaries': binaries, 'tests': tests,
              'logs': {name: file_identity(out / name) for name in ('cargo-events.jsonl', 'cargo-stderr.log')}}
    # A killed/failed/incomplete build leaves logs, never a success-shaped build.json.
    temporary = out / 'build.json.tmp'
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    temporary.replace(out / 'build.json')
    return record


def unique_fields(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f'duplicate build record field: {key}')
        value[key] = item
    return value


def verify_build_record(root, path, binary_dir, expected_sha256=None):
    """CPU-only admission. Recheck against the initial digest before each native case."""
    if path is None:
        raise RuntimeError('missing --build-record; run native_build_record.py BEFORE locking GPUs')
    path, binary_dir, root = path.resolve(), binary_dir.resolve(), root.resolve()
    try:
        payload = path.read_bytes()
        record_sha256 = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None and record_sha256 != expected_sha256:
            raise RuntimeError('build record changed after preflight')
        record = json.loads(payload, object_pairs_hook=unique_fields)
        if (record['schema'] != SCHEMA or record['status'] != 'success'
                or type(record['returncode']) is not int or record['returncode'] != 0
                or not record['started_utc'] or not record['completed_utc']):
            raise RuntimeError('not a completed owned native build record')
        if record['producer'] != file_identity(root / PRODUCER):
            raise RuntimeError('build record producer differs from current source')
        if record['source'] != source_identity(root):
            raise RuntimeError('stale-source build record: clean checkout does not match built source')
        inputs = record['inputs']
        if record['inputs_sha256'] != content_hash(inputs):
            raise RuntimeError('build input content hash mismatch')
        verify_build_inputs(root, inputs)
        if binary_dir != path.parent / 'target/release' or record['command'] != build_command(inputs, path.parent):
            raise RuntimeError('binary directory/command is not the recorded owned build')
        expected_logs = {name: file_identity(path.parent / name) for name in ('cargo-events.jsonl', 'cargo-stderr.log')}
        if record['logs'] != expected_logs:
            raise RuntimeError('owned build logs changed or incomplete')
        if set(record['binaries']) != set(BINARIES) or record['binaries'] != cargo_artifacts(path.parent / 'cargo-events.jsonl', binary_dir):
            raise RuntimeError('stale-binary build record: executable content mismatch')
        verify_test_builds(path.parent, inputs, record['tests'])
        return record, record_sha256
    except (OSError, KeyError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f'missing/incomplete native build record or inputs: {error}') from error


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True, help='new owned build directory, outside checkout')
    parser.add_argument('--nvcc', type=Path, required=True, help='exact CUDA compiler (recorded and forced)')
    parser.add_argument('--cuda-arch', required=True, choices=('120a', '100a', '90a', '89'))
    args = parser.parse_args()
    build(ROOT, args.out, args.nvcc, args.cuda_arch)
    print(f'BUILT: {args.out.resolve() / "build.json"}')


if __name__ == '__main__':
    main()
