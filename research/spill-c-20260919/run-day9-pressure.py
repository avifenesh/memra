#!/usr/bin/env python3
"""Bounded canonical-collector waits; no GPU command can bypass the collector."""
import argparse
import json
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[2]
started = time.monotonic()
completed = []
for gate, arm in [('gen', 'on'), ('spec', 'on'), ('gen', 'off'), ('spec', 'off')]:
    case = f'pressure-{gate}-{arm}'
    attempt = 0
    while True:
        attempt += 1
        target = args.out / f'{case}-attempt{attempt}'
        argv = ['python3', str(root / 'tools/tier-battery.py'), '--rig', 'pro-single',
                '--timeout', '900', '--out', str(target), '--execute', 'env',
                'MEMRA_MOE_RESIDENT=0', 'MEMRA_NGEN=32', 'MEMRA_MOE_SLOTS=9986',
                str(root / f'target/release/run-{gate}'),
                '/root/artifacts/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf', '55', '88', '13']
        if arm == 'on':
            argv.append('--experts-via-tier')
        (args.out / 'pressure-status.json').write_text(json.dumps(
            {'state': 'running', 'case': case, 'attempt': attempt, 'completed': completed}) + '\n')
        console = args.out / f'{case}-attempt{attempt}-console.log'
        with console.open('wb') as log:
            run = subprocess.run(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        if run.returncode == 0:
            completed.append({'case': case, 'directory': target.name})
            break
        text = console.read_text()
        lock_refused = (run.returncode == 2 and text.strip() ==
                        'REFUSED: [Errno 11] Resource temporarily unavailable')
        if not lock_refused or time.monotonic() - started >= 3600:
            (args.out / 'pressure-status.json').write_text(json.dumps(
                {'state': 'refused' if lock_refused else 'failed', 'case': case,
                 'exit': run.returncode, 'completed': completed}) + '\n')
            raise SystemExit(run.returncode)
        time.sleep(60)
(args.out / 'pressure-status.json').write_text(json.dumps(
    {'state': 'complete', 'completed': completed}) + '\n')
