#!/usr/bin/env python3
"""One collector-owned N=1 CLI plumbing matrix; no calibration/scoring/medians."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tee_in_collector_group(command, raw):
    with raw.open('xb') as log:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(child.stdout.readline, b''):
            log.write(line)
            log.flush()
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
        child.stdout.close()
        code = child.wait()
    return code


B = load('battery', ROOT/'tools/tier-battery.py')
C = load('checker', Path(__file__).with_name('check-h2d-output.py'))
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lock-fd', type=int, required=True)
    a = p.parse_args()
    proof = subprocess.run([sys.executable, str(ROOT/'tools/tier-lock-proof.py'),
                            '--fd', str(a.lock_fd), '--lock', B.LOCKS['rtx5090']],
                           pass_fds=(a.lock_fd,), capture_output=True, text=True, check=True)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'worker-lock.json').write_text(proof.stdout)
    identity = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                        text=True).strip(),
                'probe_source_sha256': B.digest(ROOT/'crates/memra-engine/src/bin/h2d_probe.rs'),
                'binary_sha256': B.digest(a.probe),
                'collector_sha256': B.digest(ROOT/'tools/tier-battery.py'),
                'worker_sha256': B.digest(Path(__file__)), 'qualified': False}
    (a.out/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    visits = 0
    for size in [4096, 16777216]:
        for copies in [1, 1000]:
            for order in ['ab', 'ba']:
                apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                                '--format=csv,noheader'], text=True)
                B.require(not apps.strip(), 'competing GPU process; plumbing aborted')
                raw = a.out/f'{size}-{copies}-{order}.log'
                command = [str(a.probe), '--bytes', str(size), '--copies', str(copies),
                           '--order', order, '--direction', 'both', '--repeats', '1']
                # Remain in the collector worker's process group: the outer 300 s
                # timeout must kill the probe too, before releasing the canonical lock.
                # B.tee_run starts a fresh session and is unsuitable for this nesting.
                code = tee_in_collector_group(command, raw)
                B.require(code == 0, 'probe failed: '+raw.name)
                visits += len(C.check(raw.read_text(), size, copies, order))
    print('RESULT '+json.dumps({'record': 'copies-plumbing-matrix', 'visits': visits,
                               'n_per_size_copies_direction_arm_order': 1,
                               'qualified': False, 'medians_published': False}), flush=True)


if __name__ == '__main__':
    main()
