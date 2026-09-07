#!/usr/bin/env python3
"""Validate saved component or sampled ABBA receipts, then print their summary."""
import hashlib
import json
from pathlib import Path
import struct
import sys


def fields(line):
    return dict(part.split('=', 1) for part in line.split()[1:] if '=' in part)


def component(root):
    lines = (root / 'gate.log').read_text().splitlines()
    rows = [fields(line) for line in lines if line.startswith('SPLITK_COMPONENT ')]
    assert len(rows) == 32, f'expected eight tokens, both projections and ranks: {len(rows)}'
    assert {(int(r['token']), r['device'], r['gu']) for r in rows} == {
        (t, d, g) for t in range(8) for d in ('0', '1') for g in ('0', '1')}
    timing = [fields(l) for l in lines if l.startswith('SPLITK_TIMING ')]
    for token in range(8):
        for gu in ('0', '1'):
            assert sum(int(r['slots']) for r in rows if r['gu'] == gu and int(r['token']) == token) == 6
    scored = [r for r in rows if int(r['slots']) > 0]
    for r in scored:
        for flag in ('numeric_ok', 'finite', 'deterministic', 'canaries'):
            assert r[flag] == '1', (flag, r)
        assert r['numeric_class'] == 'moe_m1_adaptive_splitk_f32_fixed_order'
        for arm in ('0', '1'):
            samples = [float(t['us']) for t in timing if
                       (t['token'], t['device'], t['gu'], t['arm']) ==
                       (r['token'], r['device'], r['gu'], arm)]
            assert len(samples) == 20 and all(t > 0 for t in samples)
        r['performance_ok'] = float(r['candidate_us']) <= float(r['oracle_us']) * 1.02
    assert any(l.startswith('PASS real routed M1 split-K component tokens=8') for l in lines)
    table = []
    for gu in ('1', '0'):
        for slots in range(1, 7):
            group = [r for r in scored if r['gu'] == gu and int(r['slots']) == slots]
            row = {'projection': 'GU' if gu == '1' else 'down', 'slots': slots, 'cells': len(group)}
            if group:
                oracle = sum(float(r['oracle_us']) for r in group) / len(group)
                candidate = sum(float(r['candidate_us']) for r in group) / len(group)
                row.update(oracle_us=oracle, candidate_us=candidate, speedup=oracle/candidate,
                           worst_candidate_over_oracle=max(float(r['candidate_us']) / float(r['oracle_us']) for r in group),
                           accepted=all(r['performance_ok'] for r in group) and (gu != '1' or slots > 2 or oracle/candidate >= 2))
            table.append(row)
    return {'validated': True, 'regime': 'warm matched planes', 'tokens': 8,
            'acceptance_rule': 'each cell candidate <= 1.02x oracle; pooled GU speedup >= 2x for slots 1 and 2; no down 2x requirement',
            'accepted_observed_slots': all(r.get('accepted', True) for r in table) and all(any(r['projection'] == 'GU' and r['slots'] == n and r['cells'] > 0 for r in table) for n in (1, 2)),
            'unobserved_slots': [r for r in table if not r['cells']],
            'table': table, 'rows': rows}


def full(root):
    arms = {'oracle': [], 'splitk': []}
    combined = (root / 'perf-abba.log').read_text() if (root / 'perf-abba.log').exists() else None
    if combined:
        sections = combined.split('ABBA_ARM index=')
        assert len(sections) == 5
        header = sections[0]
        assert header.count('[load] topology:') == 1
    correctness_pair = (root / 'correctness-pair.log').read_text() if (root / 'correctness-pair.log').exists() else None
    order = ('oracle', 'splitk', 'splitk', 'oracle') if combined else ('splitk', 'oracle', 'oracle', 'splitk')
    for i, arm in enumerate(order, 1):
        if combined:
            section = sections[i]
            assert section.startswith(f'{i-1} splitk={str(arm == "splitk").lower()} ')
            log = header + section
        else:
            log = (root / f'perf-{i}-{arm}.log').read_text()
        def rows(prefix):
            return [json.loads(l[len(prefix):]) for l in log.splitlines() if l.startswith(prefix)]
        protocol, = rows('PROTOCOL ')
        assert protocol['attention_tp'] and protocol['repeats'] == 5
        assert protocol['timing_scope'] == 'sample_plus_forward_envelope'
        assert protocol['sampling_in_timing'] and protocol['sampler_order'] == 'radix'
        assert log.count('[load] topology:') == 1
        measures = rows('MEASURE ')
        tokens = rows('TOKENS ')
        assert len(measures) == len(tokens) == 5
        for m, t in zip(measures, tokens):
            assert m['eligible'] and not m['looped'] and not m['eos']
            assert m['generated_tokens'] == m['forward_calls'] == 256
            assert m['state_pos'] == 512 and m['attention_tp']
            assert m['attention_rank_calls'] == [11008, 11008]
            assert m['attention_ar_calls'] == 11008 and m['ar_refusals'] == [0, 0]
            assert m['wo_a_calls'] == 0
            for key in ('gu_m1_calls', 'gu_half2_calls', 'down_half2_calls'):
                assert m[key] == (22016 if arm == 'oracle' else 0), (i, key, m[key])
            assert hashlib.sha256(b''.join(struct.pack('<I', v) for v in t['ids'])).hexdigest() == m['generated_sha256']
            for key in ('generated_sha256', 'final_logits_sha256', 'final_cache_digest', 'final_hidden_digest'):
                assert m[key] == measures[0][key], (i, key)
        summary, = rows('SUMMARY ')
        assert summary['repeats'] == summary['eligible_repeats'] == 5
        assert 'PASS sampled attention TP2' in log
        expected = f'MOE_PROGRAM splitk={str(arm == "splitk").lower()} component=false'
        assert expected in log
        engagement = [fields(l) for l in log.splitlines() if l.startswith('MOE_ENGAGEMENT ')]
        assert len(engagement) == 10
        for r in engagement:
            assert r['splitk'] == str(arm == 'splitk').lower()
            assert int(r['gu']) == int(r['down'])
            assert (int(r['gu']) > 0) == (arm == 'splitk')
            assert int(r['gu']) == (22016 if arm == 'splitk' else 0)
        arms[arm].append({'run': i, 'tok_s': summary['sampled_envelope_tok_s'],
                          'wall_ns': summary['decode_wall_ns'], 'tokens': summary['generated_tokens'],
                          'rows_tok_s': [m['headline_decode_tok_s'] for m in measures],
                          'generated_sha256': measures[0]['generated_sha256'],
                          'final_logits_sha256': measures[0]['final_logits_sha256']})
    for arm, runs in arms.items():
        assert len({r['generated_sha256'] for r in runs}) == 1, arm
        assert len({r['final_logits_sha256'] for r in runs}) == 1, arm
        if correctness_pair:
            sections_c = correctness_pair.split('CORRECTNESS_ARM splitk=')
            assert len(sections_c) == 3
            correctness = sections_c[1 if arm == 'oracle' else 2]
            assert correctness.startswith(str(arm == 'splitk').lower())
            assert correctness.count('REFUSAL_GATE ') == 6
        elif arm == 'splitk':
            correctness = (root / 'correctness-splitk.log').read_text()
        else:
            continue
        assert 'PASS plain-only all-layer TP/EP' in correctness and 'refusal_boundary=true' in correctness
    def pooled(arm):
        return 1e9 * sum(r['tokens'] for r in arms[arm]) / sum(r['wall_ns'] for r in arms[arm])
    tokens_identical = arms['oracle'][0]['generated_sha256'] == arms['splitk'][0]['generated_sha256']
    # The two arms intentionally use different numeric classes. Cross-arm
    # identity is a reported observation, never an admission condition.
    return {'validated': True, 'cross_arm_tokens_identical': tokens_identical,
            'sampler_order': 'radix', 'order': list(order),
            'model_loads': 1 if combined else 4, 'fresh_request_state_each_row': True,
            'timing_scope': 'sample_plus_forward_envelope',
            'oracle_tok_s': pooled('oracle'), 'splitk_tok_s': pooled('splitk'),
            'delta_pct': 100 * (pooled('splitk') / pooled('oracle') - 1), 'runs': arms}


root = Path(sys.argv[2])
result = {'component': component, 'full': full}[sys.argv[1]](root)
print(json.dumps(result, indent=2))

if sys.argv[1] == 'component' and not result['accepted_observed_slots']:
    sys.exit(1)
