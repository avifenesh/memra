#!/usr/bin/env python3
"""One collector lock for eight direct storage cells and N=1 pread plumbing.

The driver retries only canonical-lock contention: every 60s, at most 60 minutes.
The worker proves the collector's inherited lock before touching its own scratch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
LABEL = 'block-device ext4 (virtio; NVMe ancestry provider-claimed, not proven)'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(args):
    proof = subprocess.check_output([sys.executable, str(ROOT/'tools/tier-lock-proof.py'),
        '--fd', str(args.worker), '--lock', '/tmp/memra-gpu.lock'],
        pass_fds=(args.worker,), text=True, timeout=10)
    (args.out/'worker-lock.json').write_text(proof)
    compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                      '--format=csv,noheader'], text=True, timeout=10)
    if compute.strip():
        raise RuntimeError('competing GPU process under collector lock')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    identity = dict(source_commit=source, storage_class=LABEL, qualification=False,
        storage_binary_sha256=sha(ROOT/'target/release/storage-bench'),
        pread_probe_sha256=sha(Path(__file__).with_name('pread_baseline.py')),
        runner_sha256=sha(Path(__file__)), collector_sha256=sha(ROOT/'tools/tier-battery.py'))
    (args.out/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    scratch = Path(tempfile.mkdtemp(prefix='day8-storage-batch-', dir=ROOT))
    try:
        for size in (264, 4097, 1048576, 4194568):
            for operation in ('roundtrip', 'restore'):
                name = f'direct-{operation}-{size}'
                command = [str(ROOT/'target/release/storage-bench'), operation,
                           str(scratch/str(size)), str(size), 'direct']
                capture(args.out, name, command)
        capture(args.out, 'pread-baseline', [sys.executable,
            str(Path(__file__).with_name('pread_baseline.py')), str(scratch/'pread-fixture')])
    finally:
        shutil.rmtree(scratch)


def capture(out, name, command):
    raw = out/(name+'.log')
    with raw.open('xb') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180)
    row = dict(cell=name, command=command, exit_code=result.returncode,
               raw_log=raw.name, sha256=sha(raw), bytes=raw.stat().st_size, qualification=False)
    with (out/'subcells.jsonl').open('a') as log:
        log.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)
    if result.returncode:
        raise RuntimeError(f'{name} failed; see retained raw log')


def collect(args):
    args.out.mkdir(parents=True, exist_ok=False)
    # Fail closed unless this is the explicitly specified block-device filesystem.
    fs = subprocess.check_output(['findmnt', '-n', '-o', 'SOURCE,FSTYPE', '-T', str(ROOT)], text=True).split()
    if fs != ['/dev/vda1', 'ext4']:
        raise RuntimeError('expected /dev/vda1 ext4; refuse storage mislabel')
    with (args.out/'filesystem.log').open('w') as log:
        log.write('storage_class='+LABEL+'\n'); log.flush()
        for command in (['stat','-f',str(ROOT)], ['findmnt','-T',str(ROOT),'-o','TARGET,SOURCE,FSTYPE'],
                        ['ls','-ld','/sys/block/vda/device'], ['readlink','-f','/sys/block/vda/device']):
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=10)
    deadline = time.monotonic()+3600
    attempt = 0
    while True:
        out = args.out/f'attempt-{attempt:02d}'
        command = [sys.executable, str(ROOT/'tools/tier-battery.py'), '--rig', 'pro-single',
            '--timeout', '600', '--storage-root', str(ROOT), '--allow-unproven-storage',
            '--external-lock', '--out', str(out), '--execute', sys.executable,
            str(Path(__file__).resolve()), '--out', str(out), '--worker', '@COLLECTOR_LOCK_FD@']
        raw = args.out/f'attempt-{attempt:02d}-driver.log'
        with raw.open('xb') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, timeout=660)
        if result.returncode == 0:
            return
        lock_refusal = (not (out/'lock.json').exists() and
                        '[Errno 11] Resource temporarily unavailable' in raw.read_text())
        if not lock_refusal or time.monotonic()+60 > deadline:
            raise RuntimeError('collector failed or bounded lock wait exhausted; receipts retained')
        print(f'canonical collector lock busy; attempt {attempt}; retry in 60s', flush=True)
        time.sleep(60)
        attempt += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    args.out = args.out.resolve()
    worker(args) if args.worker is not None else collect(args)
