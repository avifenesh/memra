"""Audit warm-session tapes, learner continuity and recorded depth/round costs."""
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from audit_reuse import audit_reuse

IDENTITY_FIELDS = (
    'binary_sha256', 'source_commit', 'runner_sha256', 'audit_reuse_sha256',
    'artifacts', 'gpu', 'settings', 'max_new', 'ctx', 'mode', 'cache_policy',
)


def table(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def round_summary(rows):
    if not rows:
        return None
    steps = [int(r['draft_depth']) for r in rows]
    ns = sum(int(r['elapsed_ns']) for r in rows)
    emitted = sum(int(r.get('emitted', r.get('committed'))) for r in rows)
    return {'rounds': len(rows), 'depth_histogram': dict(sorted(Counter(steps).items())),
            'mean_depth': sum(steps) / len(rows), 'round_seconds': ns / 1e9,
            'mean_round_ms': ns / len(rows) / 1e6, 'emitted_in_rounds': emitted,
            'emitted_per_round': emitted / len(rows),
            'round_tokens_per_second': emitted / (ns / 1e9)}


def audit_policy_stats(run, family, rounds):
    """Final state must account for eligible observations from all eight requests."""
    path = run / ('depth-stats.tsv' if family == 'qwen' else 'depth-stats-1.tsv')
    stats = table(path)
    totals = {}
    for row in rounds:
        if row['eligible_for_learning'] != 'true':
            continue
        depth = int(row['draft_depth'])
        count, tokens, elapsed = totals.get(depth, (0, 0, 0))
        totals[depth] = (
            count + 1,
            tokens + int(row.get('emitted', row.get('committed'))),
            elapsed + int(row['elapsed_ns']),
        )
    observed = set()
    for row in stats:
        depth = int(row['verify_rows']) - 1 if family == 'qwen' else int(row['draft_depth'])
        assert depth not in observed, 'duplicate policy statistic'
        observed.add(depth)
        actual = (int(row['rounds']), int(row['tokens']), int(row['elapsed_ns']))
        assert actual == totals.get(depth, (0, 0, 0)), (
            'learner state lost or duplicated observations across requests', depth, actual, totals.get(depth))
    assert set(totals) <= observed, 'policy state omitted an observed depth'


def phases(rows, family):
    """Replay only the frozen scheduling rule, not counterfactual rewards."""
    maximum, initial = (7, 3) if family == 'qwen' else (5, 5)
    tagged = {'initial_exploration': [], 'periodic_probe': [], 'exploitation': []}
    conversation = -1
    for row in rows:
        index = (int(row['turn']) - 1) // 8
        if index != conversation:
            conversation = index
            blocks = {k: 0 for k in range(1, maximum + 1)}
            count, since_probe, next_probe = 0, 0, 1
            expected, phase = initial, 'initial_exploration'
        tagged[phase].append(row)
        if row['eligible_for_learning'] != 'true':
            continue
        depth = int(row['draft_depth'])
        if expected is None:
            expected = depth
        assert depth == expected, ('policy replay mismatch', family, row)
        count += 1
        if count < 16:
            continue
        count = 0
        blocks[depth] += 1
        unseen = [k for k, n in blocks.items() if n == 0]
        if unseen:
            expected, phase = unseen[0], 'initial_exploration'
        elif since_probe >= 8:
            expected, phase = next_probe, 'periodic_probe'
            next_probe = next_probe % maximum + 1
            since_probe = 0
        else:
            phase, expected = 'exploitation', None
            since_probe += 1
    return tagged


def audit_selection(receipts, family):
    selection = json.loads((receipts / f'{family}-selection.json').read_text())
    lock_path = receipts / 'workloads.lock.json'
    assert hashlib.sha256(lock_path.read_bytes()).hexdigest() == selection['workloads_lock_sha256']
    lock = json.loads(lock_path.read_text())
    maximum = 7 if family == 'qwen' else 5
    totals = {k: [0, 0.0] for k in range(1, maximum + 1)}
    groups = selection['calibration']
    assert [g['cycle'] for g in groups] == [0, 1]
    sealed_identity = None
    for group in groups:
        folder = receipts / f"{family}-calibration-{group['cycle']}"
        assert group['receipt_dir'] == folder.name
        records = json.loads((folder / 'runs.json').read_text())
        assert records == group['records'], 'selection changed its calibration records'
        expected = list(range(1, maximum + 1))
        if group['cycle'] == 1:
            expected.reverse()
        assert [r['arm'] for r in records] == [f'k{k}' for k in expected]
        identity = json.loads((folder / 'identity.json').read_text())
        current_identity = {key: identity[key] for key in IDENTITY_FIELDS}
        if sealed_identity is None:
            sealed_identity = current_identity
        assert current_identity == sealed_identity, 'calibration runtime identity changed'
        assert identity['mode'] == 'measurement'
        workload = f"{family}-calibration-{'a' if group['cycle'] == 0 else 'b'}.txt"
        assert identity['workload_sha256'] == lock[workload]['sha256']
        assert identity['cache_policy'] == 'stable-prompt-checkpoint-reuse-required'
        assert identity['vocabulary_head'] == 'full-in-every-arm'
        for record in records:
            k = int(record['arm'][1:])
            assert record['seed'] == group['seed']
            assert record['turns'] == 8 and not record['correctness_only']
            run = folder / f"{record['seed']}-{record['arm']}"
            turns = table(run / 'turns.tsv')
            assert len(turns) == 8
            assert sum(int(t['output_tokens']) for t in turns) == record['tokens']
            elapsed = sum(float(t['elapsed_s']) for t in turns)
            assert math.isfinite(elapsed) and elapsed > 0
            assert math.isclose(elapsed, record['elapsed_s'], abs_tol=1e-6)
            assert audit_reuse(run) == record['reuse']
            rounds = table(run / 'rounds.tsv')
            assert all(int(r['draft_depth']) == k for r in rounds if r['eligible_for_learning'] == 'true')
            audit_policy_stats(run, family, rounds)
            totals[k][0] += record['tokens']
            totals[k][1] += elapsed
    rates = {k: tokens / seconds for k, (tokens, seconds) in totals.items()}
    for k, rate in rates.items():
        assert math.isclose(rate, selection['pooled_calibration_tok_s'][f'k{k}'], rel_tol=1e-8)
    winner = max(rates, key=lambda k: (rates[k], -k))
    assert winner == selection['selected_k'], 'selected K was not the calibration winner'
    calibration_hashes = {lock[f'{family}-calibration-{letter}.txt']['sha256'] for letter in ['a', 'b']}
    heldout_hashes = {lock[f'{family}-heldout-{letter}.txt']['sha256'] for letter in ['a', 'b']}
    assert calibration_hashes.isdisjoint(heldout_hashes)
    return winner, heldout_hashes, sealed_identity


def analyze(ledger, receipts, family, expected_sets=10):
    groups = json.loads(ledger.read_text())
    calibrated_k, heldout_hashes, calibrated_identity = audit_selection(receipts, family)
    assert [g['cycle'] for g in groups] == list(range(expected_sets))
    schedule = json.loads((receipts / 'schedule.json').read_text())
    workloads_lock = json.loads((receipts / 'workloads.lock.json').read_text())
    seed_base = 20267000 if family == 'qwen' else 20269000
    arms = ['fixed', 'native', 'measured', 'calibrated', 'learned']
    all_records, traces = [], {a: [] for a in arms}
    tagged = {p: [] for p in ['initial_exploration', 'periodic_probe', 'exploitation']}
    initial_prompts, hardware, binary = {}, None, None
    workloads = set()
    turns_expected, vocab, maximum, fixed = (8, 248320, 7, 3) if family == 'qwen' else (8, 262144, 5, 5)
    expected_rows = 0
    for group in groups:
        assert group['seed'] == seed_base + group['cycle'], 'duplicate or wrong evaluation seed'
        name = group['receipt_dir']
        assert re.fullmatch(r'(qwen|gemma)-eval-[0-9]+', name), 'invalid receipt directory'
        folder = receipts / name
        ident = json.loads((folder / 'identity.json').read_text())
        assert {key: ident[key] for key in IDENTITY_FIELDS} == calibrated_identity, (
            'evaluation runtime differs from calibration')
        assert ident['vocabulary_head'] == 'full-in-every-arm'
        assert ident['cache_policy'] == 'stable-prompt-checkpoint-reuse-required'
        assert ident['fixed_depths']['calibrated'] == calibrated_k
        assert ident['settings']['MEMRA_SPEC_PMIN'] == '0'
        assert ident['settings']['MEMRA_SPEC_PMIN_INROUND'] == '0'
        assert ident['settings']['MEMRA_SPEC_ADAPT_FLOOR'] == '1'
        assert 'MEMRA_FRSPEC_TRIM' not in ident['settings']
        assert hashlib.sha256((folder / 'runner.py').read_bytes()).hexdigest() == ident['runner_sha256']
        if hardware is None:
            hardware, binary = ident['gpu'], ident['binary_sha256']
        assert (hardware, binary) == (ident['gpu'], ident['binary_sha256'])
        workload = ident['workload_sha256']
        assert workload in heldout_hashes, 'evaluation used a calibration workload'
        workloads.add(workload)
        records = group['records']
        assert records == json.loads((folder / 'runs.json').read_text()), 'ledger differs from raw runs'
        assert [r['arm'] for r in records] == schedule[group['cycle']], 'arm order changed'
        workload_name = f"{family}-heldout-{'a' if group['cycle'] % 2 == 0 else 'b'}.txt"
        assert workload == workloads_lock[workload_name]['sha256'], 'wrong held-out workload'
        assert len(records) == 5 and {r['arm'] for r in records} == set(arms)
        assert all(r['seed'] == group['seed'] for r in records)
        for record in records:
            arm = record['arm']
            run = folder / f"{record['seed']}-{arm}"
            assert not (folder / f"{record['seed']}-{arm}.contamination.txt").exists()
            log = (folder / f"{record['seed']}-{arm}.log").read_text()
            assert f'full_vocab={vocab}' in log
            if family == 'qwen':
                assert f'draft_vocab={vocab} mtp=embedded' in log
            turns = table(run / 'turns.tsv')
            assert len(turns) == turns_expected == record['turns']
            assert not record['correctness_only']
            assert math.isfinite(record['elapsed_s']) and record['elapsed_s'] > 0
            assert math.isclose(record['tokens'] / record['elapsed_s'], record['e2e_tok_s'], rel_tol=1e-7), (
                'per-run throughput differs from token/time totals')
            assert abs(sum(float(t['elapsed_s']) for t in turns) - record['elapsed_s']) < 1e-6
            assert sum(int(t['output_tokens']) for t in turns) == record['tokens']
            command = json.loads((folder / f"{record['seed']}-{arm}.command.json").read_text())
            runtime_arm = f'fixed:{calibrated_k}' if arm == 'calibrated' else arm
            assert len(command) == 10 and command[5] == runtime_arm
            assert int(command[6]) == record['seed'] and Path(command[3]).name == workload_name
            assert int(command[7]) == 2048 and int(command[8]) == 49152 and float(command[9]) == 0.7
            reuse = audit_reuse(run)
            assert reuse == record['reuse']
            for turn in turns:
                n = int(turn['turn'])
                assert int(turn['seed']) == record['seed'], 'turn seed differs from invocation'
                prompt = (run / f'turn-{n}.prompt.ids').read_bytes()
                output = (run / f'turn-{n}.output.ids').read_bytes()
                assert len(prompt.splitlines()) == int(turn['prompt_tokens'])
                assert len(output.splitlines()) == int(turn['output_tokens']) > 0
                assert all(0 <= int(x) < vocab for x in output.splitlines())
                if n == 1:
                    assert 16384 <= int(turn['prompt_tokens']) <= 16896
                    initial_prompts.setdefault(workload, prompt)
                    assert prompt == initial_prompts[workload]
                if arm == 'measured':
                    reference = folder / f"{record['seed']}-native"
                    assert prompt == (reference / f'turn-{n}.prompt.ids').read_bytes()
                    assert output == (reference / f'turn-{n}.output.ids').read_bytes()
            rounds = table(run / 'rounds.tsv')
            if arm == 'native':
                assert not rounds
            else:
                assert rounds
                assert all(1 <= int(r['draft_depth']) <= maximum for r in rounds)
                if arm in ('fixed', 'calibrated'):
                    depth = fixed if arm == 'fixed' else ident['fixed_depths']['calibrated']
                    assert all(int(r['draft_depth']) == depth for r in rounds if r['eligible_for_learning'] == 'true')
                assert sum(int(r['elapsed_ns']) for r in rounds) / 1e9 <= record['elapsed_s']
            traces[arm].extend(rounds)
            if arm in ('learned', 'fixed', 'calibrated'):
                audit_policy_stats(run, family, rounds)
            if arm == 'learned':
                for phase, items in phases(rounds, family).items():
                    tagged[phase].extend(items)
            all_records.append(record)
            expected_rows += len(turns)
    pooled = {a: sum(r['tokens'] for r in all_records if r['arm'] == a) /
              sum(r['elapsed_s'] for r in all_records if r['arm'] == a) for a in arms}
    comparisons = {}
    for control in ['fixed', 'native', 'measured', 'calibrated']:
        pairs = []
        for group in groups:
            rows = {r['arm']: r for r in group['records']}
            pairs.append({'seed': group['seed'], 'control_tok_s': rows[control]['e2e_tok_s'],
                          'learned_tok_s': rows['learned']['e2e_tok_s'],
                          'change_percent': 100 * (rows['learned']['e2e_tok_s'] / rows[control]['e2e_tok_s'] - 1)})
        gains = [p['change_percent'] for p in pairs]
        comparisons[control] = {'pooled_change_percent': 100 * (pooled['learned'] / pooled[control] - 1),
                                'median_pair_percent': statistics.median(gains),
                                'wins': sum(g > 0 for g in gains), 'pairs': pairs}
    return {'family': family, 'planned_sets': expected_sets, 'observed_sets': len(groups), 'complete_planned_matrix': len(groups) == expected_sets, 'audited_runs': len(all_records), 'audited_turns': expected_rows,
            'initial_prompt_tokens_by_workload': {key: len(value.splitlines()) for key, value in initial_prompts.items()}, 'hardware': hardware,
            'binary_sha256': binary, 'workload_sha256': sorted(workloads), 'pooled_e2e_tok_s': pooled,
            'comparisons': comparisons, 'rounds': {a: round_summary(v) for a, v in traces.items()},
            'learned_phases': {p: round_summary(v) for p, v in tagged.items()},
            'phase_note': 'Scheduling phases replayed and checked from eligible rounds. Phase rates are descriptive, not counterfactual no-probe speedups.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['qwen', 'gemma'], required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-sets', type=int, choices=range(1,11), default=10)
    args = parser.parse_args()
    result = analyze(args.ledger, args.receipts, args.family, args.expected_sets)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['family', 'audited_runs', 'audited_turns', 'pooled_e2e_tok_s']}, indent=2))
