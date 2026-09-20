#!/usr/bin/env python3
"""Pre-registered single-PRO G2 campaign; no engine/default or serving qualification.

--launch owns the collector for the entire campaign, waiting at most 45 minutes.
Worker descendants remain in the collector process group and inherit its lock FD.
"""
import argparse
import contextlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SIZES = [4096, 65536, 1048576, 16777216, 268435456]
MIN_NS = 250_000_000
TARGET_NS = 500_000_000


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = load('battery', ROOT/'tools/tier-battery.py')
C = load('checker', ROOT/'research/spill-f-20260919/check-h2d-output.py')
L = load('lockproof', ROOT/'tools/tier-lock-proof.py')


def next_copies(copies, samples):
    fastest = min(s['wall_ns'] for s in samples)
    B.require(fastest > 0, 'nonpositive calibration timing')
    n = max(copies + 1, math.ceil(copies * TARGET_NS / fastest))
    B.require(n <= 100_000, '250ms calibration cannot fit probe copies limit')
    return n


def check_samples(text, size, copies, order, scored=False):
    samples = C.check(text, size, copies, order)
    for row in samples:
        B.require(row['evidence_class'] == 'n1-plumbing-not-qualified', 'not native samples')
        B.require(all(row['power_before'][key] == '600.00 W' for key in ('power.limit', 'power.max_limit')),
                  'requires 600/600W throughout every visit')
        if scored:
            B.require(row['wall_ns'] >= MIN_NS, 'scored visit shorter than 250ms; campaign not scored')
    return samples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launch', action='store_true')
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lock-fd', type=int)
    a = p.parse_args()
    if a.launch:
        original = B.campaign_lock

        @contextlib.contextmanager
        def bounded_lock(rig, inherit=False):
            deadline = time.monotonic() + 2700
            while True:
                ctx = original(rig, inherit)
                try:
                    owned = ctx.__enter__()
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise ValueError('canonical GPU lock occupied for 45 minutes; no campaign run')
                    print('WAIT: canonical GPU lock occupied; retry in <=30 seconds', flush=True)
                    time.sleep(min(30, max(0, deadline-time.monotonic())))
                else:
                    break
            try:
                yield owned
            finally:
                ctx.__exit__(None, None, None)

        B.campaign_lock = bounded_lock
        sys.argv = ['tier-battery.py', '--rig', 'pro-single', '--timeout', '1800',
                    '--out', str(a.out/'collector'), '--external-lock', '--execute',
                    sys.executable, str(Path(__file__).resolve()), '--probe', str(a.probe.resolve()),
                    '--out', str(a.out/'visits'), '--lock-fd', '@COLLECTOR_LOCK_FD@']
        B.main()
        return
    B.require(a.lock_fd is not None, 'worker requires inherited collector lock')
    proof = L.verify(a.lock_fd, B.LOCKS['pro-single'])
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'worker-lock.json').write_text(json.dumps(proof, indent=2)+'\n')
    identity = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'binary_sha256': B.digest(a.probe), 'worker_sha256': B.digest(Path(__file__)),
                'probe_source_sha256': B.digest(ROOT/'crates/memra-engine/src/bin/h2d_probe.rs'),
                'collector_sha256': B.digest(ROOT/'tools/tier-battery.py'),
                'protocol_sha256': B.digest(Path(__file__).with_name('G2-PROTOCOL.md')),
                'qualification': False}
    (a.out/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')

    def visit(size, copies, order, name, scored=False):
        inventory = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                              '--format=csv,noheader'], text=True)
        (a.out/(name+'.compute.log')).write_text(inventory)
        B.require(not inventory.strip(), 'competing GPU process; campaign aborted')
        raw = a.out/(name+'.log')
        command = [str(a.probe.resolve()), '--bytes', str(size), '--copies', str(copies),
                   '--order', order, '--direction', 'both', '--repeats', '1']
        # Do not start a new process group: outer collector timeout owns every child.
        with raw.open('xb') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    pass_fds=(a.lock_fd,), check=False)
        B.require(result.returncode == 0, 'probe failed; raw log '+raw.name)
        samples = check_samples(raw.read_text(), size, copies, order, scored)
        with (a.out/'samples.jsonl').open('a') as stream:
            for sample in samples:
                stream.write(json.dumps({'phase': 'scored' if scored else 'calibration',
                                         'visit': name, 'raw_log': raw.name,
                                         'raw_sha256': B.digest(raw), **sample})+'\n')
            stream.flush(); os.fsync(stream.fileno())
        print(json.dumps({'visit': name, 'copies': copies, 'minimum_wall_ns': min(s['wall_ns'] for s in samples),
                          'scored': scored}), flush=True)
        return samples

    # Calibration and scoring share one uninterrupted lock. Freeze ALL copy counts first.
    calibrated = {}
    for size in SIZES:
        copies = 1000 if size <= 1048576 else 10
        for attempt in range(5):
            samples = visit(size, copies, 'ab', f'cal-{size}-{attempt}')
            if min(s['wall_ns'] for s in samples) >= TARGET_NS:
                calibrated[size] = copies
                break
            copies = next_copies(copies, samples)
        B.require(size in calibrated, 'calibration failed within five attempts')
    (a.out/'calibration.json').write_text(json.dumps(calibrated, indent=2)+'\n')
    for size in SIZES:
        for pair in range(5):
            for order in ('ab', 'ba'):
                visit(size, calibrated[size], order, f'score-{size}-{pair}-{order}', scored=True)
    B.require(B.digest(a.probe) == identity['binary_sha256'], 'probe changed during campaign')
    print('RESULT '+json.dumps({'campaign': 'G2', 'status': 'all-visits-complete', 'samples': 200,
                               'n_per_size_direction_arm': 10, 'ab_pairs': 5, 'ba_pairs': 5,
                               'minimum_visit_ms': 250, 'qualification': False}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, AssertionError) as error:
        print('REFUSED: '+str(error), file=sys.stderr)
        sys.exit(2)
