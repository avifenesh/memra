#!/usr/bin/env python3
"""Serialize a native-session gate or paired study and retain its evidence.

Run on the dedicated Linux GPU instance. No renting, deployment, or process eviction.
"""
import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def gpu_processes():
    return subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
        '--format=csv,noheader',
    ], text=True).strip()


def metrics_from_log(text):
    start = text.rfind('PERF_METRICS_START\n')
    if start < 0:
        raise ValueError('run has no completed metrics block')
    body, marker, _ = text[start + len('PERF_METRICS_START\n'):].partition('\nPERF_METRICS_END')
    if not marker:
        raise ValueError('incomplete metrics block')
    return json.loads(body)['scenarios']


def audit_control_ids(root, seed, arms, turns=8):
    for turn in range(1, turns + 1):
        for kind in ['prompt', 'output']:
            reference = root / f'{seed}-native' / f'turn-{turn}.{kind}.ids'
            want = reference.read_bytes()
            if not want.strip():
                raise ValueError(f'empty native {kind} at turn {turn}')
            for arm in arms:
                got = root / f'{seed}-{arm}' / f'turn-{turn}.{kind}.ids'
                if got.read_bytes() != want:
                    raise ValueError(f'{kind} identity failure: seed={seed}, turn={turn}, arm={arm}')


def summarize(records):
    arms = sorted({r['arm'] for r in records})
    pooled = {}
    for arm in arms:
        rows = [r for r in records if r['arm'] == arm]
        pooled[arm] = sum(r['tokens'] for r in rows) / sum(r['elapsed_s'] for r in rows)
    by_seed = {seed: {r['arm']: r for r in records if r['seed'] == seed}
               for seed in sorted({r['seed'] for r in records})}
    if 'learned' not in arms or 'native' not in arms:
        return {'pooled_e2e_tok_s': pooled, 'pairs': len(by_seed), 'calibration_only': True}
    gains = [100 * (r['learned']['e2e_tok_s'] / r['native']['e2e_tok_s'] - 1)
             for r in by_seed.values()]
    fixed_gains = [100 * (r['learned']['e2e_tok_s'] / r['fixed']['e2e_tok_s'] - 1) for r in by_seed.values()] if 'fixed' in arms else []
    return {'paired_learned_vs_fixed_percent': fixed_gains, 'pooled_e2e_tok_s': pooled, 'paired_learned_vs_native_percent': gains,
            'median_paired_gain_percent': statistics.median(gains),
            'wins': sum(g > 0 for g in gains), 'pairs': len(gains),
            'worst_pair_percent': min(gains)}


def select_cycles(gate=False, start=0, stop=8):
    if not 0 <= start < stop <= 8 or (gate and (start, stop) != (0, 8)):
        raise ValueError('invalid cycle range; gates use the full default range')
    if gate:
        return [(0, ('native', 'measured', 'fixed', 'learned'))]
    balanced = [('fixed','native','learned','measured'), ('native','measured','fixed','learned'),
                ('measured','learned','native','fixed'), ('learned','fixed','measured','native')] * 2
    return [(cycle, balanced[cycle]) for cycle in range(start, stop)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', choices=['qwen', 'gemma'], default='qwen')
    p.add_argument('--wait-lock', action='store_true')
    for name in ['binary', 'target', 'workload', 'out']:
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--draft', type=Path, help='Gemma assistant file; Qwen uses embedded MTP')
    p.add_argument('--lock', choices=['/tmp/memra-5090.lock', '/tmp/memra-gpu.lock'], required=True)
    p.add_argument('--gate', action='store_true', help='greedy identity only; no scored performance verdict')
    p.add_argument('--max-new', type=int, default=1024)
    p.add_argument('--ctx', type=int, default=32768)
    p.add_argument('--seed', type=int, default=20260920)
    p.add_argument('--cycle-start', type=int, default=0)
    p.add_argument('--cycle-stop', type=int, default=8)
    p.add_argument('--artifact-manifest', type=Path, required=True)
    p.add_argument('--source-commit', required=True, help='immutable source commit used for this binary')
    p.add_argument('--schedule', type=Path, help='Frozen explicit cycles and arm orders')
    p.add_argument('--fixed-depths', type=Path, help='Frozen arm-label to fixed-K map')
    a = p.parse_args()
    fixed_depths = json.loads(a.fixed_depths.read_text()) if a.fixed_depths else {}
    reserved = {'native', 'measured', 'fixed', 'learned'}
    maximum = 7 if a.family == 'qwen' else 5
    if any(not re.fullmatch(r'[a-z][a-z0-9_]*', key) or key in reserved
           or type(value) is not int or not 1 <= value <= maximum
           for key, value in fixed_depths.items()):
        raise ValueError('Invalid or reserved fixed-depth label')
    if a.schedule:
        entries = json.loads(a.schedule.read_text())
        cycles = [(entry['cycle'], tuple(entry['order'])) for entry in entries]
        if not cycles or len({cycle for cycle, _ in cycles}) != len(cycles):
            raise ValueError('Empty or duplicate cycle schedule')
        for cycle, order in cycles:
            if type(cycle) is not int or cycle < 0 or not order or len(order) != len(set(order)):
                raise ValueError('Invalid cycle or arm order')
            if not set(order) <= reserved | set(fixed_depths):
                raise ValueError('Unknown arm in frozen schedule')
            if a.gate and 'native' not in order:
                raise ValueError('Correctness gates require a native reference')
    else:
        cycles = select_cycles(a.gate, a.cycle_start, a.cycle_stop)
    if (a.family == 'gemma') != (a.draft is not None):
        raise ValueError('only Gemma requires a separate MTP assistant file')
    if not re.fullmatch(r'[0-9a-f]{40}', a.source_commit):
        raise ValueError('source-commit must be a full 40-character commit id')
    # Public artifact hashes, not credentials or a dump of the caller's environment.
    manifest = json.loads(a.artifact_manifest.read_text())
    verified = set()
    for row in manifest:
        path = a.artifact_manifest.parent / row['local_file']
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError(f'artifact hash mismatch: {row["local_file"]}')
        verified.add(path.resolve())
    required = [a.target, a.draft] if a.family == 'gemma' else [a.target]
    if any(path.resolve() not in verified for path in required):
        raise ValueError('target, draft and ranks must be the files covered by the artifact manifest')
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    lock = open(a.lock, 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | (0 if a.wait_lock else fcntl.LOCK_NB))
    except BlockingIOError:
        raise SystemExit('GPU lock is busy; no run started')
    busy = gpu_processes()
    if busy:
        raise SystemExit(f'GPU has an existing workload; no run started:\n{busy}')
    env = os.environ.copy()
    # Refuse unpinned inherited Memra settings instead of mixing another lane into this one.
    inherited = sorted(k for k in env if k.startswith('MEMRA_'))
    if inherited:
        raise SystemExit('unset inherited Memra experiment variables first: ' + ', '.join(inherited))
    env.update(MEMRA_SPEC_ADAPT='1', MEMRA_SPEC_ADAPT_FLOOR='1',
               MEMRA_SPEC_CAPMAX='7' if a.family == 'qwen' else '5',
               MEMRA_SPEC_PMIN='0', MEMRA_SPEC_PMIN_INROUND='0')
    identity = {
        'binary_sha256': digest(a.binary), 'workload_sha256': digest(a.workload),
        'source_commit': a.source_commit,
        'runner_sha256': digest(Path(__file__)),
        'family': a.family, 'draft_route': 'embedded-mtp' if a.family == 'qwen' else 'gemma-mtp-assistant', 'vocabulary_head': 'full-in-every-arm',
        'artifacts': manifest, 'mode': 'correctness-only' if a.gate else 'measurement',
        'native_session_not_http': True,
        'settings': {k: v for k, v in env.items() if k.startswith('MEMRA_')},
        'fixed_depths': fixed_depths,
        'schedule_sha256': digest(a.schedule) if a.schedule else None,
        'fixed_depths_sha256': digest(a.fixed_depths) if a.fixed_depths else None,
        'max_new': a.max_new, 'ctx': a.ctx, 'cycles': [cycle for cycle, _ in cycles],
        'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,power.limit,power.max_limit', '--format=csv'], text=True),
    }
    (a.out / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    records = []
    for cycle, order in cycles:
        seed = a.seed + cycle
        for arm in order:
            if busy := gpu_processes():
                raise SystemExit(f'GPU became busy before {arm}; stopping without eviction:\n{busy}')
            run = a.out / f'{seed}-{arm}'
            log_path = a.out / f'{seed}-{arm}.log'
            runtime_arm = f'fixed:{fixed_depths[arm]}' if arm in fixed_depths else arm
            cmd = [str(a.binary.resolve()), str(a.target.resolve()), str(a.draft.resolve()) if a.draft else 'embedded',
                   str(a.workload.resolve()), str(run.resolve()), runtime_arm, str(seed),
                   str(a.max_new), str(a.ctx), '0' if a.gate else '0.7']
            if a.gate:
                cmd.append('gate')
            (a.out / f'{seed}-{arm}.command.json').write_text(json.dumps(cmd) + '\n')
            print(f'start seed={seed} arm={arm}', flush=True)
            with log_path.open('w') as log, (a.out / f'{seed}-{arm}.gpu.csv').open('w') as telemetry:
                monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm', '--format=csv', '--loop-ms=250'], stdout=telemetry, stderr=subprocess.STDOUT)
                try:
                    child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                    contamination = None
                    while child.poll() is None:
                        time.sleep(1)
                        processes = gpu_processes().splitlines()
                        if len(processes) > 1:
                            contamination = '\n'.join(processes)
                            child.terminate()
                            try:
                                child.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                child.kill()
                                child.wait()
                            break
                    result = child.wait()
                except BaseException:
                    if 'child' in locals() and child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                    raise
                finally:
                    monitor.terminate()
                    monitor.wait(timeout=10)
            if contamination:
                (a.out / f'{seed}-{arm}.contamination.txt').write_text(contamination)
                raise SystemExit('concurrent GPU processes detected; stopped this run, no score accepted')
            if result:
                raise SystemExit(f'{arm} exited {result}; inspect {log_path}')
            log_text = log_path.read_text()
            vocabulary = 248320 if a.family == 'qwen' else 262144
            if f'full_vocab={vocabulary}' not in log_text:
                raise ValueError('full vocabulary engagement missing')
            if a.family == 'qwen' and f'draft_vocab={vocabulary} mtp=embedded' not in log_text:
                raise ValueError('embedded MTP full draft head engagement missing')
            metrics = metrics_from_log(log_text)[runtime_arm]
            with (run / 'turns.tsv').open() as f:
                turns = list(csv.DictReader(f, delimiter='\t'))
            expected_turns = 16 if a.family == 'gemma' else 8
            if len(turns) != expected_turns or metrics['turns'] != expected_turns or sum(int(t['output_tokens']) for t in turns) != metrics['tokens']:
                raise ValueError('turn metrics and summary disagree')
            records.append({'seed': seed, 'arm': arm, **metrics})
            (a.out / 'runs.json').write_text(json.dumps(records, indent=2) + '\n')
            print(f'completed seed={seed} arm={arm} tokens={metrics["tokens"]}', flush=True)
        # Synchronization must not change native-ladder output even with sampling.
        if 'native' in order:
            audit_control_ids(a.out, seed, [arm for arm in order if arm != 'native'] if a.gate else [arm for arm in order if arm == 'measured'], turns=16 if a.family == 'gemma' else 8)
    report = {'status': 'greedy-identity-pass', 'arms': list(cycles[0][1]), 'turns_each': 16 if a.family == 'gemma' else 8} if a.gate else summarize(records)
    (a.out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
