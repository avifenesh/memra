#!/usr/bin/env python3
"""Build native gate, then collect both transfer cells under one canonical lock."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(out, name, command, **kwargs):
    with (out / (name + '.log')).open('xb') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log,
                                stderr=subprocess.STDOUT, **kwargs)
    (out / (name + '.exit')).write_text(str(result.returncode) + '\n')
    if result.returncode:
        raise RuntimeError(f'{name} failed: {result.returncode}')


def worker(args):
    capture(args.out, 'lock-proof', [sys.executable, str(ROOT/'tools/tier-lock-proof.py'),
        '--fd', str(args.worker), '--lock', '/tmp/memra-gpu.lock'],
        pass_fds=(args.worker,), timeout=10)
    binary = ROOT/'target/release/tier-transfer-gate'
    before = sha(binary)
    for case in ('conformance', 'roundtrip'):
        capture(args.out, case, [str(binary), case], timeout=300)
    assert sha(binary) == before, 'binary changed during cells'
    (args.out/'binary.json').write_text(json.dumps({'sha256': before,
        'qualification': False}, indent=2)+'\n')


def collect(args):
    args.out.mkdir(parents=True, exist_ok=False)
    subprocess.run(['git', 'diff', '--exit-code', '--quiet'], cwd=ROOT, check=True)
    subprocess.run(['git', 'diff', '--cached', '--exit-code', '--quiet'], cwd=ROOT, check=True)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    env = os.environ.copy()
    env['PATH'] = '/root/.cargo/bin:/usr/local/cuda/bin:' + env['PATH']
    env['MEMRA_CUDA_ARCH'] = '120a'
    env['MEMRA_NVCC'] = '/usr/local/cuda/bin/nvcc'
    env.pop('DOCS_RS', None)
    for key in ('RUNPOD_POD_ID', 'CONTAINER_ID'):
        env.pop(key, None)
    capture(args.out, 'build', ['cargo', 'build', '--release', '-p', 'memra-engine',
        '--bin', 'tier-transfer-gate', '--offline', '-j', '8'], env=env, timeout=3600)
    identity = {'source_commit': source, 'qualification': False,
        'binary_sha256': sha(ROOT/'target/release/tier-transfer-gate'),
        'runner_sha256': sha(Path(__file__)),
        'collector_sha256': sha(ROOT/'tools/tier-battery.py'),
        'contract_sha256': sha(ROOT/'crates/memra-tier/src/conformance/revision_v13.rs')}
    (args.out/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    deadline = time.monotonic()+3600
    attempt = 0
    while True:
        out = args.out/f'attempt-{attempt:02d}'
        command = [sys.executable, str(ROOT/'tools/tier-battery.py'), '--rig', 'pro-single',
            '--timeout', '700', '--storage-root', str(ROOT), '--allow-unproven-storage',
            '--external-lock', '--out', str(out), '--execute', sys.executable,
            str(Path(__file__).resolve()), '--out', str(out), '--worker', '@COLLECTOR_LOCK_FD@']
        raw = args.out/f'attempt-{attempt:02d}-driver.log'
        with raw.open('xb') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log,
                stderr=subprocess.STDOUT, env=env, timeout=760)
        if result.returncode == 0:
            return
        refused = (not (out/'lock.json').exists() and
                   '[Errno 11] Resource temporarily unavailable' in raw.read_text())
        if not refused or time.monotonic()+60 > deadline:
            raise RuntimeError('collector failed or bounded lock wait exhausted')
        print(f'canonical lock busy: attempt {attempt}; retry in 60s', flush=True)
        time.sleep(60)
        attempt += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    args.out = args.out.resolve()
    worker(args) if args.worker is not None else collect(args)
