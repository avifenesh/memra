#!/usr/bin/env python3
"""One bounded N=1 repeatability cell through the canonical collector; no GPU bypass."""
import argparse
import json
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--gate', choices=['gen', 'spec'], default='spec')
parser.add_argument('--slots', default='9986')
parser.add_argument('--max-minutes', type=float, default=60)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[2]
case = f'pressure-{args.gate}-on-repeat'
status = args.out / 'repeat-status.json'
started = time.monotonic()
attempt = 0
while True:
    attempt += 1
    target = args.out / f'{case}-attempt{attempt}'
    argv = ['python3', str(root / 'tools/tier-battery.py'), '--rig', 'pro-single',
            '--timeout', '900', '--out', str(target), '--execute', 'env',
            'MEMRA_MOE_RESIDENT=0', 'MEMRA_NGEN=32', f'MEMRA_MOE_SLOTS={args.slots}',
            str(root / f'target/release/run-{args.gate}'),
            '/root/artifacts/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf', '55', '88', '13',
            '--experts-via-tier']
    status.write_text(json.dumps({'state': 'running', 'case': case, 'attempt': attempt}) + '\n')
    console = args.out / f'{case}-attempt{attempt}-console.log'
    with console.open('wb') as log:
        run = subprocess.run(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    if run.returncode == 0:
        status.write_text(json.dumps({'state': 'complete', 'case': case,
                                      'directory': target.name, 'attempts': attempt}) + '\n')
        break
    text = console.read_text()
    lock_refused = (run.returncode == 2 and text.strip() ==
                    'REFUSED: [Errno 11] Resource temporarily unavailable')
    if not lock_refused or time.monotonic() - started >= args.max_minutes * 60:
        status.write_text(json.dumps({'state': 'refused' if lock_refused else 'failed',
                                      'case': case, 'exit': run.returncode,
                                      'attempts': attempt}) + '\n')
        raise SystemExit(run.returncode)
    time.sleep(60)
