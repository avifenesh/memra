#!/usr/bin/env python3
"""Day-eleven receipts on the target card: build, CPU-only inspect, three N=1 collector cells.

`--build` compiles run-gen/run-spec from the checked-out day-eleven tree into its own
target dir and writes build.json + build.log; the frozen day-nine and day-ten binaries are
never rebuilt. `--inspect` builds memra-cli (no CUDA) and runs `memra model inspect` on the
approved artifact so the plan-derived catalog can be replayed from memra's own census.
`--cells` runs, through the canonical collector only: (a) run-gen with the plan-derived
installer and default budgets (MATCH expected), (c) run-spec with a seven-slot GPU budget
(REFUSED expected, collector `refused`), (b) run-spec K=1..8 with the exact eight-slot
budget (SELF-CONSISTENCY PASS expected). No GPU command runs outside the collector.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

RECORD = 860160  # largest expert record of the approved artifact (day nine: max_expert_bytes)
SLOT = RECORD + 8  # one native GPU slot: record plus SLOT_TAIL_PAD_BYTES
LOCK_REFUSED = 'REFUSED: [Errno 11] Resource temporarily unavailable'
FROZEN = {'day9': ('148e7f0e9994a1c35dd3e0891dae559c377561e0', 'target'),
          'day10': ('79353d53d7f69713177caf2e9830b48f9e600fa3', 'target-day10')}
# (case, gate, extra flags, expected collector outcome), in run order.
CELLS = [('gen-default', 'run-gen', [], 'complete'),
         ('spec-refuse-7', 'run-spec', [f'--expert-bank-gpu-bytes={7 * SLOT}'], 'refused'),
         ('spec-exact-8', 'run-spec', [f'--expert-bank-gpu-bytes={8 * SLOT}'], 'complete')]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def head(root):
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()


def binaries(root, target_dir):
    return {name: sha256(root / target_dir / 'release' / name) for name in ['run-gen', 'run-spec']}


def manifest(root, target_dir):
    """Binary identities at run time: the day-eleven pair plus both frozen pairs."""
    return {'checked_utc': now(), 'worktree_head': head(root),
            'binary_sha256': binaries(root, target_dir),
            'frozen': {day: {'runtime_source': source, 'binary_sha256': binaries(root, directory)}
                       for day, (source, directory) in FROZEN.items()}}


def tool_version(argv):
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=60).stdout.strip().splitlines()[-1]
    except (OSError, IndexError, subprocess.TimeoutExpired):
        return None


def build(root, out, target_dir):
    command = ['cargo', 'build', '--release', '-j', '16', '-p', 'memra-engine', '--bin', 'run-gen', '--bin', 'run-spec']
    env = os.environ.copy()
    env['CARGO_TARGET_DIR'] = str(root / target_dir)
    log = out / 'build.log'
    with log.open('wb') as sink:
        run = subprocess.run(command, cwd=root, env=env, stdout=sink, stderr=subprocess.STDOUT)
    receipt = {'finished_utc': now(), 'source': head(root), 'cargo_target_dir': str(root / target_dir),
               'command': ' '.join(command), 'exit': run.returncode,
               'binary_sha256': binaries(root, target_dir) if run.returncode == 0 else None,
               'frozen_binary_sha256': {day: binaries(root, directory) for day, (_, directory) in FROZEN.items()},
               'build_log_sha256': sha256(log), 'nvcc': tool_version(['nvcc', '--version']),
               'rustc': tool_version(['rustc', '--version'])}
    (out / 'build.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if run.returncode != 0:
        raise SystemExit(run.returncode)


def inspect(root, out, artifact, against):
    """CPU-only: memra-cli has no CUDA dependency; the receipt is memra's own census and plan."""
    env = os.environ.copy()
    env['CARGO_TARGET_DIR'] = str(root / 'target-cli')
    log = out / 'inspect-build.log'
    with log.open('wb') as sink:
        run = subprocess.run(['cargo', 'build', '--release', '-j', '16', '-p', 'memra-cli', '--bin', 'memra'],
                             cwd=root, env=env, stdout=sink, stderr=subprocess.STDOUT)
    if run.returncode != 0:
        raise SystemExit(run.returncode)
    target = out / 'inspect'
    command = [str(root / 'target-cli/release/memra'), 'model', 'inspect', artifact, '--against', against, '--out', str(target)]
    with (out / 'inspect.log').open('wb') as sink:
        run = subprocess.run(command, cwd=root, stdout=sink, stderr=subprocess.STDOUT)
    files = sorted(p.name for p in target.iterdir()) if target.is_dir() else []
    receipt = {'finished_utc': now(), 'source': head(root), 'command': command, 'exit': run.returncode,
               'memra_sha256': sha256(root / 'target-cli/release/memra'),
               'files': {name: {'bytes': (target / name).stat().st_size, 'sha256': sha256(target / name)} for name in files}}
    (out / 'inspect.json').write_text(json.dumps(receipt, indent=2) + '\n')


def classify(returncode, console_text, capture_path):
    """Collector outcome: lock-refused (retry), refused, complete, failed."""
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


def cells(root, out, target_dir, artifact, timeout, max_minutes):
    status_path = out / 'cells-status.json'
    (out / 'binary-manifest.json').write_text(json.dumps(manifest(root, target_dir), indent=2) + '\n')
    started = time.monotonic()
    completed = []
    for case, gate, flags, expected in CELLS:
        attempt = 0
        while True:
            attempt += 1
            target = out / f'{case}-attempt{attempt}'
            argv = ['python3', str(root / 'tools/tier-battery.py'), '--rig', 'pro-single',
                    '--timeout', str(timeout), '--out', str(target), '--execute',
                    'env', 'MEMRA_MOE_RESIDENT=0', 'MEMRA_NGEN=32',
                    str(root / target_dir / 'release' / gate), artifact, '55', '88', '13',
                    '--experts-via-tier'] + flags
            status_path.write_text(json.dumps({'state': 'running', 'case': case, 'attempt': attempt,
                                               'completed': completed}) + '\n')
            console = out / f'{case}-attempt{attempt}-console.log'
            with console.open('wb') as log:
                run = subprocess.run(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            outcome = classify(run.returncode, console.read_text(), target / 'command.capture.json')
            if outcome == expected:
                completed.append({'case': case, 'directory': target.name, 'outcome': outcome,
                                  'gate': gate, 'flags': flags, 'attempts': attempt})
                break
            if outcome != 'lock-refused' or time.monotonic() - started >= max_minutes * 60:
                status_path.write_text(json.dumps({'state': 'failed' if outcome != 'lock-refused' else 'lock-refused',
                                                   'case': case, 'outcome': outcome, 'exit': run.returncode,
                                                   'attempts': attempt, 'completed': completed}) + '\n')
                raise SystemExit(run.returncode or 1)
            time.sleep(60)
    (out / 'binary-postcheck.json').write_text(json.dumps(manifest(root, target_dir), indent=2) + '\n')
    status_path.write_text(json.dumps({'state': 'complete', 'completed': completed}) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--target-dir', default='target-day11')
    parser.add_argument('--artifact', default='/root/artifacts/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf')
    parser.add_argument('--against', default='qwen35moe')
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--max-minutes', type=float, default=120)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--cells', action='store_true')
    args = parser.parse_args()
    steps = [name for name in ['build', 'inspect', 'cells'] if getattr(args, name)] or ['build', 'inspect', 'cells']
    args.out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    if 'build' in steps:
        build(root, args.out, args.target_dir)
    if 'inspect' in steps:
        inspect(root, args.out, args.artifact, args.against)
    if 'cells' in steps:
        cells(root, args.out, args.target_dir, args.artifact, args.timeout, args.max_minutes)


if __name__ == '__main__':
    main()
