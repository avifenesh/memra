#!/usr/bin/env python3
"""Select a small set of source-bound tests covering explicit behavior edges.

Selection is planning, not a pass receipt. Uncovered or stale edges refuse a reduced
battery. Controls and mandatory regressions cannot be optimized away.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def contract_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def select(required, tests, context, root):
    if not isinstance(required, list):
        raise ValueError('required edges must be a list')
    required = set(required)
    if any(not isinstance(edge, str) or not edge for edge in required):
        raise ValueError('edges must have nonempty names')
    if not required:
        return {'decision': 'no-change', 'selected': [], 'uncovered': [],
                'edge_decisions': [], 'qualification': False, 'reason': 'no validation requested or run'}
    catalog = {}
    ineligible = {}
    for test in tests:
        name = test['id']
        if not isinstance(name, str) or not name or name in catalog:
            raise ValueError('duplicate or empty test ID')
        cost = test['cost']
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost <= 0:
            raise ValueError('test cost must be positive and finite')
        if not isinstance(test['covers'], list) or any(not isinstance(x, str) or not x for x in test['covers']):
            raise ValueError('test coverage must name its independently asserted edges')
        catalog[name] = test
        reasons = []
        scope = test.get('scope', {})
        if not isinstance(scope, dict) or any(context.get(k) != v for k, v in scope.items()):
            reasons.append('model/hardware/numeric/request scope mismatch')
        if test.get('kind', 'cpu') not in ('cpu', 'gpu'):
            raise ValueError('unknown test execution kind')
        if test.get('kind') == 'gpu' and not {'model', 'artifact', 'hardware', 'numeric_program'} <= scope.keys():
            reasons.append('native coverage requires model/artifact/hardware/numeric identity')
        inputs = test.get('inputs', {})
        if not inputs:
            reasons.append('no source-bound coverage contract')
        for path, expected in inputs.items():
            resolved = (root / path).resolve()
            if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
                reasons.append('missing or escaping coverage input: ' + path)
            elif hashlib.sha256(resolved.read_bytes()).hexdigest() != expected:
                reasons.append('coverage input changed: ' + path)
        if reasons:
            ineligible[name] = reasons

    def bundle(name, active=()):
        if name not in catalog:
            raise ValueError('unknown control: ' + name)
        if name in active:
            raise ValueError('control dependency cycle')
        result = {name}
        for control in catalog[name].get('controls', []):
            result.update(bundle(control, (*active, name)))
        return result

    bundles = {name: bundle(name) for name in catalog}
    for name, members in bundles.items():
        bad = sorted(members & ineligible.keys())
        if bad:
            ineligible.setdefault(name, []).append('ineligible required test/control: ' + ', '.join(bad))

    selected = set()
    for name, test in catalog.items():
        if test.get('mandatory'):
            selected.update(bundles[name])
    invalid_mandatory = sorted(selected & ineligible.keys())
    if invalid_mandatory:
        return {'decision': 'expand', 'selected': [], 'uncovered': sorted(required),
                'ineligible': ineligible, 'reason': 'mandatory regression/control is unavailable',
                'qualification': False}

    def covered(names):
        return set().union(*(set(catalog[n]['covers']) for n in names)) if names else set()

    uncovered = required - covered(selected)
    choices = []
    while uncovered:
        candidates = []
        for name, members in bundles.items():
            if name in ineligible or members <= selected:
                continue
            added = members - selected
            gain = covered(added) & uncovered
            if gain:
                cost = sum(catalog[n]['cost'] for n in added)
                candidates.append((-len(gain) / cost, cost, name, added, gain))
        if not candidates:
            break
        _, cost, name, added, gain = min(candidates, key=lambda x: x[:3])
        selected.update(added)
        choices.append({'test': name, 'adds': sorted(added), 'new_edges': sorted(gain), 'cost': cost})
        uncovered -= gain

    contract = {'schema': 'memra-edge-contract-v1', 'required_edges': sorted(required),
                'context': context, 'tests': {n: catalog[n] for n in sorted(selected)}}
    return {'decision': 'scoped' if not uncovered else 'expand', 'selected': sorted(selected),
            'context': context,
            'contract': contract, 'contract_id': contract_digest(contract),
            'uncovered': sorted(uncovered), 'ineligible': ineligible, 'choices': choices,
            'edge_decisions': [{'edge': edge, 'covered': edge not in uncovered,
                                'tests': sorted(n for n in selected if edge in catalog[n]['covers'])}
                               for edge in sorted(required)],
            'cost': sum(catalog[n]['cost'] for n in selected),
            'optimization': 'deterministic greedy set cover, not a claimed global optimum',
            'isolation': 'test boot/reset contracts are unchanged; selection never merges process state',
            'qualification': False}


def validate_results(plan, results, context, root):
    """A selected composite test must pass every independently named edge/control."""
    if plan['decision'] != 'scoped':
        raise ValueError('a plan with uncovered edges cannot produce a scoped pass')
    if plan.get('context') != context:
        raise ValueError('execution context changed after selection')
    contract = plan.get('contract')
    if not contract or contract_digest(contract) != plan.get('contract_id'):
        raise ValueError('missing or changed source/coverage contract')
    if contract['context'] != context or set(contract['tests']) != set(plan['selected']):
        raise ValueError('selected tests or context disagree with the bound contract')
    if set(results) != set(plan['selected']):
        raise ValueError('missing or unexpected selected test result')
    for name in plan['selected']:
        result = results[name]
        test = contract['tests'][name]
        if result.get('contract_id') != plan['contract_id']:
            raise ValueError('result belongs to a different source/coverage contract: ' + name)
        for path, expected in test['inputs'].items():
            actual = (root / path).resolve()
            if not actual.is_relative_to(root.resolve()) or not actual.is_file() or hashlib.sha256(actual.read_bytes()).hexdigest() != expected:
                raise ValueError('coverage source changed before result admission: ' + path)
        if result.get('context') != context or result.get('status') != 'passed':
            raise ValueError('failed or mismatched test result: ' + name)
        executed, skipped = result.get('executed'), result.get('skipped')
        if type(executed) is not int or executed <= 0 or type(skipped) is not int or skipped != 0:
            raise ValueError('empty or skipped test result: ' + name)
        for edge in test['covers']:
            if result.get('edges', {}).get(edge) != 'passed':
                raise ValueError('selected test/control lacks an explicit passing assertion: ' + name + '/' + edge)
    covered = set().union(*(set(test['covers']) for test in contract['tests'].values()))
    if set(contract['required_edges']) - covered:
        raise ValueError('bound contract leaves required edges uncovered')
    return {'status': 'passed', 'edges': len(contract['required_edges']), 'contract_id': plan['contract_id'], 'qualification': False,
            'scope': 'declared behavior assertions only; native receipt admission remains separate'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--results', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    plan = select(manifest['required_edges'], manifest['tests'], manifest['context'], args.root)
    if args.results:
        print(json.dumps(validate_results(plan, json.loads(args.results.read_text()), manifest['context'], args.root), indent=2))
    else:
        print(json.dumps(plan, indent=2))
    return 2 if plan['decision'] == 'expand' else 0


if __name__ == '__main__':
    raise SystemExit(main())
