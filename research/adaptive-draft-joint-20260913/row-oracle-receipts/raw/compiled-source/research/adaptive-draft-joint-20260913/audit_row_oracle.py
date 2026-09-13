"""Audit reached-state evidence and fit a pre-registered offline utility table."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def swap_label(state, candidate):
    # Initial tail is unique and remains frozen: remove row index core_rows.
    survivors = [x for x in state['active_top'] if x[2] != state['core_rows']]
    winner, score, _ = survivors[0]
    token, value = candidate
    if abs(value-score) <= state['tolerance']:
        return None
    if value > score:
        winner = token
    elif len(survivors) > 1 and abs(score-survivors[1][1]) <= state['tolerance']:
        return None
    return int(winner == state['target']) - int(state['correct'])


def check_counterexamples():
    state = {'active_top': [[1, 4.0, 0], [2, 3.0, 4096], [3, 2.0, 1]],
             'core_rows': 4096, 'target': 99, 'correct': False, 'tolerance': 1e-5}
    assert swap_label(state, [99, 3.5]) == 0  # Missing but insertion cannot win.
    assert swap_label(state, [99, 5.0]) == 1  # Missing and insertion repairs.
    assert swap_label({**state, 'target': 1, 'correct': True}, [99, 5.0]) == -1
    mutable = {**state, 'active_top': [[1, 4.0, 4096], [99, 3.0, 0], [3, 2.0, 1]]}
    assert swap_label(mutable, [50, 1.0]) == 1  # Removal repairs with low filler.
    assert swap_label({**mutable, 'target': 1, 'correct': True}, [50, 1.0]) == -1
    assert swap_label(state, [99, 4.0]) is None  # No unsupported tie claim.


def analyze(bank):
    root = bank/'raw/row-oracle'
    runs = [json.loads(x) for x in (root/'runs.jsonl').read_text().splitlines()]
    manifest = json.loads((root/'manifest.json').read_text())
    for name, expected in manifest['prompts'].items():
        assert sha(root/'prompts'/name) == expected
    states = []
    outputs = defaultdict(set)
    for r in runs:
        assert sha(root/(r['id']+'.log')) == r['raw_sha256']
        assert r['status'] in ['pass', 'expected_refusal']
        if r['status'] == 'expected_refusal':
            assert r['phase'] == 'admission' and r['exit_code'] != 0
            continue
        assert r['emitted'] == 128 and r['exit_code'] == 0
        outputs[r['prompt']].add(r['tokens_sha256'])
        assert r['config']['MEMRA_GEMMA_TRIM_FREEZE'] == '1'
        assert all(r['config'][k] == '0' for k in ['MEMRA_SPEC_ADAPT', 'MEMRA_SPEC_PMIN', 'MEMRA_SPEC_PMIN_INROUND'])
        if r['probe']:
            trace = root/(r['id']+'.jsonl')
            assert sha(trace) == r['trace_sha256']
            observations = [json.loads(x) for x in trace.read_text().splitlines()]
            if r['phase'] == 'collection':
                for s in observations:
                    s.update(split=r['split'], domain=r['domain'], prompt=r['prompt'])
                states.extend(observations)
    assert all(len(v) == 1 for v in outputs.values())
    headroom = {}
    for split in ['train', 'calibration', 'heldout']:
        for domain in ['code', 'prose']:
            rr = [s for s in states if s['split'] == split and s['domain'] == domain]
            valid = [s for s in rr if s['overlap_argmax_match']]
            wrong = [s for s in valid if not s['correct']]
            headroom[split+'-'+domain] = {
                'prompt_groups': len({s['prompt'] for s in rr}), 'states': len(rr),
                'numerically_eligible': len(valid), 'max_overlap_abs_error': max(s['overlap_max_abs_error'] for s in rr),
                'wrong_proposals': len(wrong), 'missing_targets': sum(not s['target_in_head'] for s in wrong),
                'addition_rescues': sum(s['addition_rescue'] for s in wrong),
                'removal_rescues': sum(s['removal_rescue'] for s in wrong),
                'swap_rescues': sum(s['swap_rescue'] for s in wrong),
                'any_one_swap_rescue': sum(s['addition_rescue'] or s['removal_rescue'] or s['swap_rescue'] for s in wrong),
                'correct_states_exposed_to_top16_harm': sum(s['correct'] and s['harmful_outside_top16'] > 0 for s in valid)}
    counts = defaultdict(lambda: [0, 0])
    frequency = defaultdict(int)
    for s in states:
        if s['split'] != 'train' or not s['overlap_argmax_match']:
            continue
        frequency[s['target']] += 1
        for c in s['outside_top16']:
            label = swap_label(s, c)
            if label is not None:
                counts[c[0]][0] += label
                counts[c[0]][1] += 1
    utilities = {token: total/(n+4) for token, (total, n) in counts.items()}

    def evaluate(split, threshold, policy):
        groups = defaultdict(list)
        actions = 0
        ambiguous = 0
        for s in states:
            if s['split'] != split or not s['overlap_argmax_match']:
                continue
            # All policies share the same numerical eligibility, including no-op.
            if any(swap_label(s, c) is None for c in s['outside_top16']):
                ambiguous += 1
                continue
            value = 0
            if policy == 'highest_outside':
                choice = s['outside_top16'][0]
            elif policy == 'frequency':
                choice = max(s['outside_top16'], key=lambda c: (frequency.get(c[0], 0), -c[0]))
                if frequency.get(choice[0], 0) == 0:
                    choice = None
            elif policy == 'learned':
                choice = max(s['outside_top16'], key=lambda c: (utilities.get(c[0], 0), -c[0]))
                if utilities.get(choice[0], 0) <= threshold:
                    choice = None
            else:
                choice = None
            if choice is not None:
                value = swap_label(s, choice)
                if value is None:
                    ambiguous += 1
                    continue
                actions += 1
            groups[s['prompt']].append(value)
        return {'prompt_groups': len(groups), 'states': sum(len(x) for x in groups.values()),
                'actions': actions, 'ambiguous_action_states_excluded': ambiguous,
                'net_one_step_correctness_changes': sum(sum(x) for x in groups.values()),
                'prompt_mean_delta': statistics.mean(statistics.mean(x) for x in groups.values()),
                'per_prompt_delta': {k: statistics.mean(v) for k, v in groups.items()}}

    grid = {str(t): evaluate('calibration', t, 'learned') for t in [0, 0.01, 0.025, 0.05, 0.1, 1.0]}
    threshold = max(map(float, grid), key=lambda t: (grid[str(int(t) if t == 0 else t)]['prompt_mean_delta'], t))
    # Scores/calibration are fixed above. Held-out labels never enter fitting/selection.
    estimator = {'prior_exposures': 4, 'candidate_rows': len(utilities),
        'table': {str(k): {'sum_delta': counts[k][0], 'n': counts[k][1], 'utility': v} for k, v in utilities.items()},
        'calibration_grid': grid, 'selected_threshold': threshold, 'training_target_frequency': dict(frequency),
        'metrics': {split: {p: evaluate(split, threshold, p) for p in ['no_change', 'highest_outside', 'frequency', 'learned']} for split in ['train', 'calibration', 'heldout']},
        'scope': 'Offline one-step candidate admission to a fixed victim using expensive probe scores; not online policy, speedup, or novelty evidence.'}
    pairs = {}
    for r in runs:
        if r['phase'] == 'cost':
            pairs.setdefault((r['prompt'], r['repeat']), {})[r['probe']] = r
    log_ratios = defaultdict(list)
    for (prompt, repeat), p in pairs.items():
        log_ratios[prompt].append(math.log(p[True]['request_seconds']/p[False]['request_seconds']))
    values = [statistics.mean(x) for x in log_ratios.values()]
    rng = random.Random(20260913)
    boot = sorted(math.exp(statistics.mean(rng.choices(values, k=len(values)))) for _ in range(10000))
    cost = {'paired_runs': len(pairs), 'prompt_groups': len(values), 'probe_over_control_request_ratio': math.exp(statistics.mean(values)),
            'bootstrap_95_percentile': [boot[250], boot[9749]], 'group_log_ratios': log_ratios}
    assert len(runs) == 193 and len(pairs) == 48 and len(outputs) == 48
    return {'raw_runs_verified': len(runs), 'headroom': headroom, 'cost': cost, 'estimator': estimator,
            'scope': 'Sparse reached-state oracle diagnostic; no extrapolation to whole-block acceptance or serving throughput.'}


if __name__ == '__main__':
    check_counterexamples()
    p = argparse.ArgumentParser()
    p.add_argument('bank', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = analyze(args.bank)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'estimator'}, indent=2))
    print(json.dumps({'threshold': result['estimator']['selected_threshold'], 'heldout': result['estimator']['metrics']['heldout']}, indent=2))
