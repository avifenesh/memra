"""Gate, calibrate and freeze fixed K before opening held-out evaluations."""
import argparse
import collections
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import run_study as study

LANE = Path(__file__).resolve().parent
ARMS = ('fixed', 'native', 'learned', 'measured', 'calibrated')


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def schedule():
    base = (0, 1, 4, 2, 3)
    rows = [tuple(ARMS[(i + offset) % 5] for i in base) for offset in range(5)]
    return rows + [tuple(reversed(row)) for row in rows]


def select_depth(groups, maximum):
    totals = {k: [0, 0.0] for k in range(1, maximum + 1)}
    for group in groups:
        assert {r['arm'] for r in group} == {f'k{k}' for k in totals}
        for row in group:
            k = int(row['arm'][1:])
            assert row['elapsed_s'] >= 60 and row['tokens'] > 0 and not row['correctness_only']
            totals[k][0] += row['tokens']
            totals[k][1] += row['elapsed_s']
    rates = {k: tokens / seconds for k, (tokens, seconds) in totals.items()}
    winner = max(rates, key=lambda k: (rates[k], -k))
    return winner, rates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--binaries', type=Path, required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    save(root / 'schedule.json', schedule())
    for name in ['workloads.lock.json', 'PLAN.md', 'controller.py', 'run_study.py']:
        (root / name).write_bytes((LANE / name).read_bytes())

    def status(**fields):
        save(root / 'status.json', {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields})
        print(json.dumps(fields), flush=True)

    def attempt(family, phase, cycle, order, workload, mapping, gate=False):
        name = f'{family}-mtp-followup-{phase}-{cycle}'
        spec = root / (name + '-schedule.json')
        fixed = root / (name + '-depths.json')
        save(spec, [{'cycle': cycle, 'order': order}])
        save(fixed, mapping)
        target = args.models / family
        (target / 'artifacts.lock.json').write_bytes((LANE / f'{family}-artifacts.lock.json').read_bytes())
        seed = (20262000 if family == 'qwen' else 20263000) + (1000 if phase == 'eval' else 0)
        cmd = [sys.executable, str(LANE / 'run_study.py'), '--family', family,
               '--binary', str(args.binaries / ('mtp-depth-study' if family == 'qwen' else 'gemma-depth-study')),
               '--target', str(target / 'target.gguf'), '--workload', str(LANE / workload),
               '--out', str(root / name), '--lock', '/tmp/memra-5090.lock', '--wait-lock',
               '--max-new', '256' if phase == 'short' else '1024', '--ctx', '49152',
               '--seed', str(seed), '--artifact-manifest', str(target / 'artifacts.lock.json'),
               '--source-commit', args.source, '--schedule', str(spec), '--fixed-depths', str(fixed)]
        if family == 'gemma':
            cmd += ['--draft', str(target / 'assistant.gguf')]
        if gate:
            cmd += ['--gate']
        status(state='running', family=family, phase=phase, cycle=cycle, receipt_dir=name)
        with (root / (name + '-driver.log')).open('w') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name} failed with exit {result.returncode}; keep all receipts and stop')
        rows = json.loads((root / name / 'runs.json').read_text())
        assert [r['arm'] for r in rows] == list(order)
        return {'cycle': cycle, 'seed': seed + cycle, 'receipt_dir': name, 'records': rows}

    try:
        for family, maximum in [('qwen', 7), ('gemma', 5)]:
            depths = {f'k{k}': k for k in range(1, maximum + 1)}
            gate_order = ['native', 'measured', 'fixed', 'learned', *depths]
            attempt(family, 'short', 0, gate_order, 'workload-short.txt', depths, gate=True)
            attempt(family, 'long', 0, gate_order, 'calibration-a.txt', depths, gate=True)
            calibration = []
            for cycle, workload in enumerate(['calibration-a.txt', 'calibration-b.txt']):
                order = list(depths) if cycle == 0 else list(reversed(depths))
                calibration.append(attempt(family, 'calibration', cycle, order, workload, depths))
            winner, rates = select_depth([g['records'] for g in calibration], maximum)
            selection = {
                'family': family, 'selected_k': winner, 'pooled_calibration_tok_s': rates,
                'selected_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'rule': 'Highest pooled output tokens / request E2E seconds; exact ties select smaller K',
                'calibration': calibration,
                'calibration_runs_sha256': {g['receipt_dir']: study.digest(root / g['receipt_dir'] / 'runs.json') for g in calibration},
                'workloads_lock_sha256': study.digest(LANE / 'workloads.lock.json'),
            }
            save(root / f'{family}-selection.json', selection)
            status(state='selected', family=family, fixed_k=winner, calibration_rates=rates)
            groups = []
            for cycle, order in enumerate(schedule()):
                groups.append(attempt(family, 'eval', cycle, order, f'heldout-{ "a" if cycle % 2 == 0 else "b" }.txt', {'calibrated': winner}))
                save(root / f'{family}-selected-sets.json', groups)
            import analyze
            report = analyze.analyze(root / f'{family}-selected-sets.json', root, family)
            save(root / f'{family}-audit.json', report)
            status(state='family-complete', family=family, rates=report['pooled_e2e_tok_s'])
        save(root / 'DONE.json', {'families': ['qwen', 'gemma'], 'sets_each': 10})
        status(state='complete')
    except BaseException as exc:
        status(state='failed', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    main()
