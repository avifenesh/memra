"""Audit selected paired MTP sets and describe recorded depth/round costs."""
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path


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


def analyze(ledger, receipts, family, expected_sets=10):
    groups = json.loads(ledger.read_text())
    assert [g['cycle'] for g in groups] == list(range(expected_sets))
    arms = ['fixed', 'native', 'measured', 'calibrated', 'learned']
    all_records, traces = [], {a: [] for a in arms}
    tagged = {p: [] for p in ['initial_exploration', 'periodic_probe', 'exploitation']}
    initial_prompts, hardware, binary = {}, None, None
    workloads = set()
    turns_expected, vocab, maximum, fixed = (8, 248320, 7, 3) if family == 'qwen' else (16, 262144, 5, 5)
    expected_rows = 0
    for group in groups:
        name = group['receipt_dir']
        assert re.fullmatch(r'(qwen|gemma)-mtp-[a-z0-9-]+', name), 'invalid receipt directory'
        folder = receipts / name
        ident = json.loads((folder / 'identity.json').read_text())
        assert ident['vocabulary_head'] == 'full-in-every-arm'
        assert ident['settings']['MEMRA_SPEC_PMIN'] == '0'
        assert ident['settings']['MEMRA_SPEC_PMIN_INROUND'] == '0'
        assert ident['settings']['MEMRA_SPEC_ADAPT_FLOOR'] == '1'
        assert 'MEMRA_FRSPEC_TRIM' not in ident['settings']
        assert hashlib.sha256((folder / 'runner.py').read_bytes()).hexdigest() == ident['runner_sha256']
        if hardware is None:
            hardware, binary = ident['gpu'], ident['binary_sha256']
        assert (hardware, binary) == (ident['gpu'], ident['binary_sha256'])
        workload = ident['workload_sha256']
        workloads.add(workload)
        records = group['records']
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
            assert math.isfinite(record['elapsed_s']) and record['elapsed_s'] >= 60
            assert abs(sum(float(t['elapsed_s']) for t in turns) - record['elapsed_s']) < 1e-6
            assert sum(int(t['output_tokens']) for t in turns) == record['tokens']
            for turn in turns:
                n = int(turn['turn'])
                prompt = (run / f'turn-{n}.prompt.ids').read_bytes()
                output = (run / f'turn-{n}.output.ids').read_bytes()
                assert len(prompt.splitlines()) == int(turn['prompt_tokens'])
                assert len(output.splitlines()) == int(turn['output_tokens']) > 0
                assert all(0 <= int(x) < vocab for x in output.splitlines())
                assert turn['resumed'] == 'false'
                if n == 1 or (family == 'gemma' and n == 9):
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
