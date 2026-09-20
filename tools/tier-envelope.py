#!/usr/bin/env python3
"""G2 opaque-byte envelope: one bounded cell under the tier collector, no medians.

Default N=5 AB and N=5 BA. --correctness-only permits N=1, never scoring.
The worker refuses without an owning inherited canonical FD. No GPU runs in --plan.
"""
import argparse
import csv
import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('envelope_battery', ROOT/'tools/tier-battery.py')
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)
SIZES = [4096 * 4**i for i in range(10)]


def orders(rounds):
    return [(r, order, arm) for r in range(rounds) for order in ('AB', 'BA')
            for arm in (('pageable', 'pinned') if order == 'AB' else ('pinned', 'pageable'))]


def validate_visit(row, size, direction, arm, copies):
    B.require(row.get('record') == 'sample' and row.get('schema_version') == 1, 'unknown probe record')
    for key, value in {'bytes': size, 'direction': direction, 'arm': arm, 'copies': copies,
                       'completed_bytes': size*copies, 'verified_bytes': size, 'identity': True}.items():
        B.require(type(row.get(key)) is type(value) and row[key] == value, 'probe mismatch: '+key)
    B.require(isinstance(row.get('expected_sha256'), str) and
              B.re.fullmatch('[0-9a-f]{64}', row['expected_sha256']) and
              row['expected_sha256'] == row.get('actual_sha256'), 'payload hash mismatch')
    for key in ('wall_ns', 'event_ms', 'setup_ns', 'verify_ns'):
        B.require(type(row.get(key)) in (int, float) and math.isfinite(row[key]) and row[key] >= 0,
                  'invalid timing: '+key)
    B.require(row['wall_ns'] > 0 and row['unix_end_ns'] >= row['unix_start_ns'], 'vacuous interval')
    B.require(row['power_before'] == row['power_after'], 'native per-arm power changed')


def power():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,power.limit,power.max_limit',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True,
                            timeout=10, check=True)
    rows = list(csv.reader(result.stdout.splitlines()))
    B.require(len(rows) == 1 and len(rows[0]) == 3, 'G2 requires one GPU')
    index, cap, maximum = (float(v.strip()) for v in rows[0])
    B.require(index == 0 and math.isfinite(cap) and math.isfinite(maximum) and 0 < cap <= maximum,
              'missing/invalid enforced power cap')
    return {'power_limit_w': cap, 'power_max_limit_w': maximum, 'raw': result.stdout}


def preflight(root):
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    B.require(available >= 4*1024**3, 'less than 4 GiB available host RAM')
    stat = os.statvfs(root)
    B.require(stat.f_bavail*stat.f_frsize >= 20*1024**3, 'less than 20 GiB available disk')
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                            capture_output=True, text=True, timeout=10, check=True)
    B.require(not result.stdout.strip(), 'GPU has a competing compute application')
    return {'mem_available_bytes': available, 'disk_available_bytes': stat.f_bavail*stat.f_frsize,
            'compute_apps': result.stdout}


def telemetry_check(path, started_utc, ended_utc, expected_cap):
    """Conservative coverage: complete probe-process windows, not just copy time."""
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    B.require(len(rows) >= 2, 'insufficient telemetry')
    stamps = []
    for raw in rows:
        row = {k.strip(): v.strip() for k, v in raw.items()}
        B.require(row['index'] == '0', 'unexpected telemetry device')
        stamps.append(datetime.datetime.strptime(row['timestamp'], '%Y/%m/%d %H:%M:%S.%f').timestamp())
        for field, value in [('power.limit [W]', expected_cap['power_limit_w']),
                             ('power.max_limit [W]', expected_cap['power_max_limit_w'])]:
            B.require(float(row[field].split()[0]) == value, 'power cap changed/unparseable')
    gaps = [b-a for a, b in zip(stamps, stamps[1:])]
    B.require(all(0 < gap <= .5 for gap in gaps), 'telemetry gap >500 ms or nonmonotonic')
    B.require(stamps[0] <= started_utc and stamps[-1] >= ended_utc,
              'telemetry does not cover full visit windows')
    return {'status': 'pass', 'samples': len(rows), 'max_gap_seconds': max(gaps)}


def visit(args, phase, round_id, order, copies):
    name = f'{phase}-{round_id}-{order}'
    before = power()
    raw = args.out/(name+'.log')
    command = [str(args.probe), '--bytes', str(args.bytes), '--direction', args.direction,
               '--order', order.lower(), '--repeats', '1', '--copies', str(copies)]
    # Only called by a proof-checked worker under the collector's whole-window lock.
    code, expired = B.tee_run(command, raw, timeout=240, echo=False,
                             pass_fds=(args.worker,), shared_group=True)
    after = power()
    pair = {'schema_version': 1, 'kind': 'envelope-pair', 'phase': phase,
            'round': round_id, 'order': order, 'exit_code': code, 'timed_out': expired,
            'power_before': before, 'power_after': after, 'qualification': False,
            'raw_log': B.descriptor(args.out, raw)}
    B.append_cell(args.out/'pairs.jsonl', pair)  # Retain failed/refused invocations too.
    B.require(code == 0 and not expired, 'probe failed; see raw log (N1-only probe cannot score G2)')
    values = [json.loads(line) for line in raw.read_text().splitlines() if line.startswith('{')]
    results = [r for r in values if r.get('record') == 'RESULT']
    B.require(len(results) == 1 and results[0].get('comparator_red_rejected') is True,
              'missing RESULT/comparator red refusal')
    samples = [r for r in values if r.get('record') == 'sample']
    B.require(len(samples) == 2, 'expected one arm pair')
    controls = [r for r in values if r.get('record') == 'control']
    B.require(len(controls) == 4 and all(r.get('identity') is True for r in controls),
              'missing two-pattern arm controls')
    B.require(before == after, 'power changed within pair')
    visits = []
    arms = ('pageable', 'pinned-cacheable') if order == 'AB' else ('pinned-cacheable', 'pageable')
    for sample, arm in zip(samples, arms):
        validate_visit(sample, args.bytes, args.direction, arm, copies)
        for field, key in [('power.limit', 'power_limit_w'), ('power.max_limit', 'power_max_limit_w')]:
            B.require(float(sample['power_before'][field].split()[0]) == before[key], 'per-arm cap mismatch')
        row = {**pair, 'kind': 'envelope-visit', 'arm': arm, 'sample': sample,
               'started_monotonic_ns': sample['mono_start_ns'], 'ended_monotonic_ns': sample['mono_end_ns'],
               'started_unix_seconds': sample['unix_start_ns']/1e9,
               'ended_unix_seconds': sample['unix_end_ns']/1e9}
        B.append_cell(args.out/'envelope.jsonl', row)
        visits.append(row)
    return visits


def calibrate(args):
    """Discarded bounded calibration; one fixed count, covering the FASTER arm."""
    target_ns = 350_000_000 if args.correctness_only else 3_000_000_000
    copies = 1
    attempts = []
    for attempt in range(5):
        rows = visit(args, 'calibration', attempt, 'AB', copies)
        fastest_ns = min(r['sample']['wall_ns'] for r in rows)
        attempts.append({'copies': copies, 'minimum_wall_ns': fastest_ns})
        record = {'copies': copies, 'target_ns': target_ns, 'minimum_visit_ns': 250_000_000,
                  'discarded': True, 'attempts': attempts, 'qualification': False}
        (args.out/'calibration.json').write_text(json.dumps(record, indent=2)+'\n')
        if fastest_ns >= target_ns:
            return copies
        B.require(copies < 100000, 'calibration capped before required visit duration')
        # Count scaling is approximate: launch/warmup overhead shrinks per copy.
        # Headroom avoids asymptotic under-target retries as clocks settle.
        copies = min(100000, max(copies+1, math.ceil(1.25*copies*target_ns/fastest_ns)))
    raise ValueError('calibration did not converge; no rehearsal/scored samples run')


def worker(args):
    proof = subprocess.run([sys.executable, str(ROOT/'tools/tier-lock-proof.py'), '--fd', str(args.worker),
                            '--lock', B.LOCKS[args.rig]], pass_fds=(args.worker,), capture_output=True,
                           text=True, timeout=5, check=True)
    (args.out/'worker-lock.json').write_text(proof.stdout)
    pre = preflight(args.out)
    identity = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'binary_sha256': B.digest(args.probe), 'collector_sha256': B.digest(ROOT/'tools/tier-battery.py'),
                'runner_sha256': B.digest(Path(__file__)), 'preflight': pre, 'qualification': False,
                'thermal_class': 'short-cell-allocation-warm', 'pinned_allocator': 'CUDA-cacheable-flags-zero',
                'correctness_only': args.correctness_only}
    (args.out/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    # Inner copies never increase N. Even the N=1 rehearsal covers a sampler interval.
    copies = calibrate(args)
    visits = []
    for round_id in range(args.rounds):
        for order in ('AB', 'BA'):
            preflight(args.out)  # A competing application invalidates/aborts the cell.
            pair = visit(args, 'correctness' if args.correctness_only else 'sample', round_id, order, copies)
            B.require(all(r['sample']['wall_ns'] >= 250_000_000 for r in pair),
                      'visit below 250 ms; retained but not an accepted rehearsal')
            visits.extend(pair)
    # Leave room for a final sampler point covering the last visit; no GPU work here.
    time.sleep(.3)
    print('RESULT '+json.dumps({'kind': 'envelope-cell', 'visits': len(visits), 'rounds_per_order': args.rounds,
                               'copies': copies, 'correctness_only': args.correctness_only,
                               'qualification': False}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', action='store_true')
    p.add_argument('--bytes', type=int, default=4096, choices=SIZES)
    p.add_argument('--direction', choices=['h2d', 'd2h'], default='h2d')
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--correctness-only', action='store_true')
    p.add_argument('--rig', choices=['rtx5090'], default='rtx5090')
    p.add_argument('--probe', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--worker', type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    B.require((args.correctness_only and args.rounds == 1) or
              (not args.correctness_only and args.rounds >= 5), 'N<5 is correctness-only; use N=1 explicitly')
    if args.plan:
        print(json.dumps({'cells': [{'bytes': n, 'direction': d} for n in SIZES for d in ('h2d', 'd2h')],
                          'orders': orders(args.rounds), 'timeout_seconds': 300, 'telemetry_interval_ms': 250,
                          'qualification': False, 'medians_published': False}, indent=2))
        return 0
    B.require(args.out is not None and args.probe is not None, '--out and --probe required')
    args.out = args.out.resolve()
    args.probe = args.probe.resolve(strict=True)
    B.require(args.probe.is_file(), 'probe binary missing')
    if args.worker is not None:
        worker(args)
        return 0
    args.out.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(ROOT/'tools/tier-battery.py'), '--rig', args.rig, '--timeout', '300',
               '--out', str(args.out/'collector'), '--external-lock', '--execute', sys.executable,
               str(Path(__file__).resolve()), '--worker', '@COLLECTOR_LOCK_FD@', '--rig', args.rig,
               '--probe', str(args.probe), '--out', str(args.out), '--bytes', str(args.bytes),
               '--direction', args.direction, '--rounds', str(args.rounds)]
    if args.correctness_only:
        command.append('--correctness-only')
    # The collector owns raw stdout/stderr and process-group timeout teardown.
    code = subprocess.run(command, check=False).returncode
    capture = B.validate_cell(args.out/'collector/CELL.jsonl')
    summary = {'kind': 'envelope-summary', 'qualification': False, 'medians_published': False,
               'correctness_only': args.correctness_only, 'collector_exit_code': code,
               'scoring_eligible': False, 'telemetry': {'status': 'not-checked'}}
    if code == 0:
        try:
            rows = [json.loads(line) for line in (args.out/'envelope.jsonl').read_text().splitlines()]
            samples = [r for r in rows if r['phase'] != 'calibration']
            B.require(len(samples) == 4*args.rounds, 'incomplete paired orders')
            B.require(all(r['sample']['wall_ns'] >= 250_000_000 for r in samples), 'visit below 250 ms')
            summary['minimum_visit_ns'] = min(r['sample']['wall_ns'] for r in samples)
            cap = rows[0]['power_before']
            B.require(all(r['power_before'] == r['power_after'] == cap for r in rows), 'power cap changed across visits')
            summary['telemetry'] = telemetry_check(args.out/'collector/command.gpu.csv',
                                                  rows[0]['started_unix_seconds'], rows[-1]['ended_unix_seconds'], cap)
            seconds = sum(r['sample']['wall_ns'] for r in samples)/1e9
            summary['aggregate_timed_seconds'] = seconds
            summary['scoring_eligible'] = not args.correctness_only and args.rounds >= 5 and seconds >= 60
            summary['power'] = cap
        except (ValueError, KeyError, OSError) as error:
            summary['telemetry'] = {'status': 'unqualified', 'reason': str(error)}
    summary['capture_status'] = capture['status']
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    return code


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print('tier-envelope: REFUSED: '+str(error), file=sys.stderr)
        sys.exit(2)
