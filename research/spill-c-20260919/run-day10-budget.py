#!/usr/bin/env python3
"""Two bounded N=1 GPU bank budget cells through the canonical collector; no GPU bypass.

Cell a (`budget-refuse-7`): a budget worth seven slots must be REFUSED before load
finishes (collector status `refused`, exit 2). Cell b (`budget-exact-8`): a budget
worth exactly eight slots must run gen to MATCH with explicit evictions. The day-ten
build lives in its own target dir; the frozen day-nine binaries are never touched.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

RECORD = 860160  # largest expert record of the approved artifact (day nine: max_expert_bytes)
SLOT = RECORD + 8  # one native GPU slot: record plus the eight-byte tail pad
LOCK_REFUSED = 'REFUSED: [Errno 11] Resource temporarily unavailable'
# (flag, budget bytes, expected collector outcome). The host cell runs through the
# pressure-refusal.py red arm, which must see the native token.
CELLS = [('--expert-bank-gpu-bytes', 7 * SLOT, 'refused'), ('--expert-bank-gpu-bytes', 8 * SLOT, 'complete'),
         ('--expert-bank-host-bytes', 1, 'refused')]
NAMES = {(7 * SLOT, '--expert-bank-gpu-bytes'): 'budget-refuse-7', (8 * SLOT, '--expert-bank-gpu-bytes'): 'budget-exact-8',
         (1, '--expert-bank-host-bytes'): 'budget-host-refuse-1'}
FROZEN_SOURCE = '148e7f0e9994a1c35dd3e0891dae559c377561e0'


def classify(returncode, console_text, capture_path):
    """Collector outcome: lock-refused (retry), refused (cell a), complete (cell b), failed."""
    if returncode == 2 and console_text.strip() == LOCK_REFUSED and not capture_path.exists():
        return 'lock-refused'
    if not capture_path.exists():
        return 'failed'
    status = json.loads(capture_path.read_text())['status']
    if returncode == 0 and status == 'executed-not-qualified':
        return 'complete'
    if returncode == 2 and status == 'refused':
        return 'refused'
    return 'failed'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(root, target_dir):
    """Binary identities at run time. `worktree_head` is the checkout; the source the
    binaries were built from is in build.json (the verifier joins the two)."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    return {'checked_utc': now, 'worktree_head': head,
            'binary_sha256': {name: sha256(root / target_dir / 'release' / name) for name in ['run-gen', 'run-spec']},
            'frozen': {'runtime_source': FROZEN_SOURCE,
                       'binary_sha256': {name: sha256(root / 'target/release' / name) for name in ['run-gen', 'run-spec']}}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--target-dir', default='target-day10')
    parser.add_argument('--artifact', default='/root/artifacts/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf')
    parser.add_argument('--timeout', default='1800')
    parser.add_argument('--max-minutes', type=float, default=60)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    status_path = args.out / 'budget-status.json'
    (args.out / 'binary-manifest.json').write_text(json.dumps(manifest(root, args.target_dir), indent=2) + '\n')
    started = time.monotonic()
    completed = []
    for flag, budget, expected in CELLS:
        case = NAMES[(budget, flag)]
        attempt = 0
        while True:
            attempt += 1
            target = args.out / f'{case}-attempt{attempt}'
            argv = ['python3', str(root / 'tools/tier-battery.py'), '--rig', 'pro-single',
                    '--timeout', args.timeout, '--out', str(target), '--execute']
            if flag == '--expert-bank-host-bytes':
                argv += ['python3', str(root / 'research/spill-c-20260919/pressure-refusal.py')]
            argv += ['env', 'MEMRA_MOE_RESIDENT=0', 'MEMRA_NGEN=32',
                     str(root / args.target_dir / 'release/run-gen'), args.artifact, '55', '88', '13',
                     '--experts-via-tier', f'{flag}={budget}']
            status_path.write_text(json.dumps({'state': 'running', 'case': case, 'attempt': attempt,
                                               'completed': completed}) + '\n')
            console = args.out / f'{case}-attempt{attempt}-console.log'
            with console.open('wb') as log:
                run = subprocess.run(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            outcome = classify(run.returncode, console.read_text(), target / 'command.capture.json')
            if outcome == expected:
                completed.append({'case': case, 'directory': target.name, 'outcome': outcome,
                                  'flag': flag, 'budget_bytes': budget, 'attempts': attempt})
                break
            if outcome != 'lock-refused' or time.monotonic() - started >= args.max_minutes * 60:
                status_path.write_text(json.dumps({'state': 'failed' if outcome != 'lock-refused' else 'lock-refused',
                                                   'case': case, 'outcome': outcome, 'exit': run.returncode,
                                                   'attempts': attempt, 'completed': completed}) + '\n')
                raise SystemExit(run.returncode or 1)
            time.sleep(60)
    (args.out / 'binary-postcheck.json').write_text(json.dumps(manifest(root, args.target_dir), indent=2) + '\n')
    status_path.write_text(json.dumps({'state': 'complete', 'completed': completed}) + '\n')


if __name__ == '__main__':
    main()
