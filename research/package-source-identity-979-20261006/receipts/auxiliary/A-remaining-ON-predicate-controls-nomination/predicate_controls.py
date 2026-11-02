#!/usr/bin/python3 -I
"""Unexecuted CPU predicate-control proposal; Root owns review and execution.

Parse runner source as data. Never import either runner or its helper. Only the
selected predicate AST is compiled, with a restricted in-memory namespace.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re


CANDIDATE = Path('/home/evidence-user/.cache/evidence-manager-20261006/A-remaining-ON-reviewed/run_cell.py')
CANDIDATE_SHA256 = '1657d15eec6a587a14922445743945a594ef466a5d18240a0e388029e96fee43'
OLD_RUNNER = Path('/home/evidence-user/.cache/evidence-manager-20261006/A-serving-v3-reviewed/run_cell.py')
OLD_RUNNER_SHA256 = '05ebe7f41d17f92b3e634d137bf3dabef3ff27a7282f17f8b75e9b36c6e63bc6'
RETAINED = Path('/home/evidence-user/.local/state/evidence-campaign-20261002/receipts/manager/package-identity-979/current-main-335e-context-standard-1800/actual-rig-serving600/cache-off')
NO_HIT = 'no prefix-store hit with same-request spec restore and acceptance'


def source_ast(path, expected_sha256):
    raw = path.read_bytes()
    # Hash only these two small Python source files, never model/runtime artifacts.
    assert hashlib.sha256(raw).hexdigest() == expected_sha256, str(path)+' source drift'
    source = raw.decode('utf-8')
    return source, ast.parse(source, filename=str(path))


def select_candidate():
    source, tree = source_ast(CANDIDATE, CANDIDATE_SHA256)
    main, = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main']
    matches = []
    for parent in ast.walk(main):
        body = getattr(parent, 'body', None)
        if not isinstance(body, list):
            continue
        for index, node in enumerate(body):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == 'prefix_store_spec_hits'):
                matches.append(body[index:index+3])
    nodes, = matches
    assert len(nodes) == 3
    assignment, loop, check = nodes
    assert isinstance(assignment.value, ast.List) and not assignment.value.elts
    assert isinstance(loop, ast.For) and isinstance(loop.target, ast.Name) and loop.target.id == 'row'
    assert ast.dump(loop.iter) == ast.dump(ast.parse('turns[1:]', mode='eval').body)
    assert isinstance(check, ast.Assert) and isinstance(check.test, ast.Name)
    assert check.test.id == 'prefix_store_spec_hits'
    assert isinstance(check.msg, ast.Constant) and check.msg.value == NO_HIT
    selected = ast.Module(body=nodes, type_ignores=[])
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(selected))
    compiled = compile(selected, str(CANDIDATE)+':predicate-only', 'exec', dont_inherit=True, optimize=0)
    return compiled, {
        'path': str(CANDIDATE), 'sha256': CANDIDATE_SHA256,
        'line_start': assignment.lineno, 'line_end': check.end_lineno,
        'source': '\n'.join(ast.get_source_segment(source, node) for node in nodes),
        'compiled_nodes': ['Assign(prefix_store_spec_hits)', 'For(row in turns[1:])', 'Assert(prefix_store_spec_hits)'],
    }


def select_old():
    source, tree = source_ast(OLD_RUNNER, OLD_RUNNER_SHA256)
    main, = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main']
    expected_branch = ast.dump(ast.parse("arm == 'cache-off'", mode='eval').body)
    branch, = [node for node in ast.walk(main)
               if isinstance(node, ast.If) and ast.dump(node.test) == expected_branch]
    check, = branch.orelse
    assert isinstance(check, ast.Assert)
    expected_expression = ast.parse("any(row['cached_tokens'] > 0 for row in turns[1:])", mode='eval').body
    assert ast.dump(check.test) == ast.dump(expected_expression)
    compiled = compile(ast.Expression(body=check.test), str(OLD_RUNNER)+':old-ON-expression-only',
                       'eval', dont_inherit=True, optimize=0)
    return compiled, {
        'path': str(OLD_RUNNER), 'sha256': OLD_RUNNER_SHA256,
        'line': check.lineno, 'source': ast.get_source_segment(source, check.test),
    }


def retained_case():
    turns = []
    files = []
    for index in range(1, 9):
        path = RETAINED / ('turn'+str(index)+'.response')
        value = json.loads(path.read_bytes())
        usage = value['usage']
        turns.append({'turn': index, 'id': value['id'], 'prompt_tokens': usage['prompt_tokens'],
                      'cached_tokens': usage['prompt_tokens_details']['cached_tokens'],
                      'spec': usage['spec']})
        files.append(str(path))
    log_path = RETAINED / 'server.log'
    log = log_path.read_text()
    assert len({row['id'] for row in turns}) == 8
    assert all(row['spec']['rounds'] > 0 and row['spec']['drafted'] > 0 for row in turns)
    assert all(len(re.findall(r'^\[meter\] admit id='+re.escape(row['id'])+r' ', log, re.M)) == 1
               for row in turns)
    assert '[worker] spec-affinity: rewound to ' in log
    assert '[prefix-cache] hit:' not in log and '[prefix-cache] spec restore:' not in log
    files.append(str(log_path))
    return turns, log, files


def synthetic_cases():
    turns = [{'turn': index, 'id': 'synthetic-turn-'+str(index), 'prompt_tokens': 128,
              'cached_tokens': 96 if index == 2 else 0,
              'spec': {'rounds': 1, 'drafted': 3, 'accepted': 1}}
             for index in range(1, 9)]
    hit = '[prefix-cache] hit: 96 of 128 prompt tokens from cache (model package-probe)'
    restore = '[prefix-cache] spec restore: 96 of 128 prompt tokens + draft plane from cache [suffix queued] (model package-probe)'

    def log_for(lines, marker_turn=2):
        blocks = []
        for row in turns:
            blocks.append('[meter] admit id='+row['id']+' tenant=default lane=interactive model="package-probe"')
            if row['turn'] == marker_turn:
                blocks.extend(lines)
        return '\n'.join(blocks)+'\n'

    yield 'synthetic_matching_store_spec_accepted_positive', turns, log_for([hit, restore]), True
    yield 'synthetic_wrong_request', turns, log_for([hit, restore], marker_turn=3), False
    yield 'synthetic_wrong_cached_count', turns, log_for([hit.replace('96 of', '95 of'), restore.replace('96 of', '95 of')]), False
    yield 'synthetic_wrong_prompt_count', turns, log_for([hit.replace('of 128', 'of 129'), restore.replace('of 128', 'of 129')]), False
    yield 'synthetic_restore_count_disagrees', turns, log_for([hit, restore.replace('96 of', '95 of')]), False
    yield 'synthetic_missing_restore', turns, log_for([hit]), False
    yield 'synthetic_missing_hit', turns, log_for([restore]), False
    yield 'synthetic_reversed_order', turns, log_for([restore, hit]), False
    accepted_zero = copy.deepcopy(turns)
    accepted_zero[1]['spec']['accepted'] = 0
    yield 'synthetic_accepted_zero', accepted_zero, log_for([hit, restore]), False


def evaluate(name, turns, log, expected_accept, candidate, old):
    # No filesystem, process, network or import builtins enter the selected AST.
    old_namespace = {'__builtins__': {'any': any}, 'turns': copy.deepcopy(turns)}
    old_accept = bool(eval(old, old_namespace, old_namespace))
    namespace = {'__builtins__': {'str': str, 'AssertionError': AssertionError},
                 're': re, 'turns': copy.deepcopy(turns), 'current_log': log}
    assertion = None
    try:
        exec(candidate, namespace, namespace)
    except AssertionError as error:
        assertion = str(error)
    accepted = assertion is None
    assert old_accept, name+': expected the old any-cached predicate to accept'
    assert accepted == expected_accept, name+': unexpected candidate outcome'
    hits = namespace.get('prefix_store_spec_hits')
    if expected_accept:
        assert hits == [{'turn': 2, 'id': 'synthetic-turn-2',
                         'hit': '[prefix-cache] hit: 96 of 128 prompt tokens from cache (model package-probe)',
                         'spec_restore': '[prefix-cache] spec restore: 96 of 128 prompt tokens + draft plane from cache [suffix queued] (model package-probe)'}]
    else:
        # Missing input, extraction errors and other exceptions cannot count as a red pass.
        assert assertion == NO_HIT, name+': wrong refusal reason: '+repr(assertion)
        assert hits == [], name+': refusal retained unexpected hits'
    return {'case': name, 'data_kind': 'retained_OFF_bytes' if name == 'retained_OFF_affinity' else 'synthetic',
            'old_ON_any_cached_accepts': old_accept, 'candidate_accepts': accepted,
            'expected_candidate_accepts': expected_accept, 'candidate_assertion': assertion,
            'prefix_store_spec_hits': hits, 'status': 'PASS'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New CPU-control result path; never overwrite receipts')
    args = parser.parse_args()
    assert args.output.is_absolute() and args.output.parent.is_dir() and not args.output.exists()
    candidate, candidate_source = select_candidate()
    old, old_source = select_old()
    turns, log, retained_files = retained_case()
    results = [evaluate('retained_OFF_affinity', turns, log, False, candidate, old)]
    for name, rows, fixture_log, expected in synthetic_cases():
        results.append(evaluate(name, rows, fixture_log, expected, candidate, old))
    report = {
        'status': 'PASS', 'scope': 'Exact extracted ON-predicate CPU controls only',
        'source': '335e2cbcbdb808ffa62688d40a299a3dd58ab51e',
        'candidate_extraction': candidate_source, 'old_ON_extraction': old_source,
        'retained_files': retained_files, 'retained_turns': turns, 'cases': results,
        'original_GPU_job': 'd6f7265ceab8', 'original_GPU_job_status': 'FAILED and unchanged',
        'runner_imported_or_executed': False, 'runtime_helper_imported': False,
        'artifact_hashes_or_server_or_GPU_executed': False,
        'model_or_runtime_or_serving_qualification': False,
        'limits': 'Synthetic positives prove predicate discrimination only. Actual cache-ON server engagement remains untested here. The enclosing runner checks, Budget, HTTP, ownership, FD9 and teardown are outside this extracted predicate control.',
    }
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': 'PASS', 'cases': len(results), 'output': str(args.output), 'qualification': False}))


if __name__ == '__main__':
    main()
