#!/usr/bin/env python3
"""Source-bound public CPU CI routing. Native qualification stays separate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import shlex
import sys

import validation_plan
from cpu_workflow_inputs import _regular, _policy, POLICY_PATH, COMMANDS

REPOSITORY = 'avifenesh/memra'
OWNER = 'avifenesh'
OWNER_ID = 55848801
SHA = re.compile(r'[0-9a-f]{40}')
INVENTORY = 'tools/ci_merge_validation.json'
FULL_JOBS = ('changes', 'gates', 'boundary', 'build', 'clippy', 'server-tests',
             'portable-suites', 'engine-tests', 'arch-coverage', 'publish-dryrun')
MERGE_GUARDS = {
    'action-pins': ['bash', 'tools/check-action-pins.sh'],
    'workflow-keys': ['python3', 'tools/check-workflow-keys.py'],
    'conflict-markers': ['bash', 'tools/check-conflict-markers.sh'],
    'docs-registry': ['bash', 'tools/docs-registry-census.sh'],
    'publish-members': ['bash', 'tools/workspace-publish-census.sh'],
    'stub-abi': ['python3', 'tools/stub-abi-census.py'],
    'arch-matrix': ['bash', 'tools/arch-matrix-census.sh'],
}


class Refused(ValueError):
    pass


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Refused('duplicate JSON key')
        result[key] = value
    return result


def read_json(path):
    """Read a bounded regular event or receipt, without following parent links."""
    path = Path(path)
    if '..' in path.parts:
        raise Refused('noncanonical JSON input path')
    if not path.is_absolute():
        path = Path.cwd() / path
    directory = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=directory)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 2 * 1024 * 1024:
                raise Refused('input is not a bounded regular JSON file')
            with os.fdopen(descriptor, 'rb', closefd=False) as stream:
                raw = stream.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raise Refused('JSON input grew beyond its limit')
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)
    return json.loads(raw, object_pairs_hook=unique_object), hashlib.sha256(raw).hexdigest()


def object_field(value, key):
    field = value.get(key) if type(value) is dict else None
    if type(field) is not dict:
        raise Refused('missing object field: ' + key)
    return field


def commit(value):
    if type(value) is not str or not SHA.fullmatch(value) or value == '0' * 40:
        raise Refused('missing immutable commit')
    return value


def owner(value):
    return (type(value) is dict and value.get('login') == OWNER
            and type(value.get('id')) is int and value['id'] == OWNER_ID)


def repository(value):
    return (type(value) is dict and value.get('full_name') == REPOSITORY
            and owner(value.get('owner')))


def route(event_name, event, *, force_full=False):
    """Actor, labels and commit authors never confer owner-PR mode."""
    result = {'schema': 'memra-public-ci-route-v1', 'mode': 'full',
              'reason': 'unknown event or repository lineage',
              'event': event_name, 'qualification': False}
    try:
        if not repository(object_field(event, 'repository')):
            return result
        if force_full:
            result['reason'] = 'explicit complete CPU inventory'
        elif event_name == 'pull_request':
            pull = object_field(event, 'pull_request')
            base, head = object_field(pull, 'base'), object_field(pull, 'head')
            result.update(base=commit(base.get('sha')), head=commit(head.get('sha')))
            if not repository(object_field(base, 'repo')) or base.get('ref') != 'main':
                return result
            head_repository = object_field(head, 'repo')
            if (not owner(pull.get('user')) or not owner(head_repository.get('owner'))
                    or type(head_repository.get('full_name')) is not str
                    or not head_repository['full_name'].startswith(OWNER + '/')):
                result['reason'] = 'external PR author or unknown head lineage'
                return result
            result.update(mode='thin', reason='owner-authored PR with pinned repository lineage')
        elif event_name == 'push' and event.get('ref') == 'refs/heads/main':
            result.update(base=commit(event.get('before')), head=commit(event.get('after')),
                          mode='thin', reason='main push under explicit thin CPU policy')
        elif event_name in ('schedule', 'workflow_call', 'workflow_dispatch'):
            result['reason'] = 'complete CPU inventory event'
        elif event_name == 'push' and str(event.get('ref', '')).startswith('refs/tags/'):
            result['reason'] = 'tag CPU validation; native publication gates remain'
    except (Refused, TypeError):
        result.update(mode='full', reason='malformed event; complete CPU validation required')
    return result


def full_cpu_plan(root, head, reason):
    """Force the existing inventory, including present normal/native contracts."""
    tree = validation_plan.Tree(root, commit(head))
    plan = validation_plan.full(reason)
    validation_plan.preserve_contract_obligations(plan, tree, tree)
    plan.update(head=head, cpu_inventory='complete', qualification=False)
    return plan


def run_steps(text):
    """Inventory literal run fields, retaining exact bodies and parent job names."""
    job = label = None
    result = []
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        match = re.fullmatch(r'  ([a-zA-Z][a-zA-Z0-9_-]*):\n', line)
        if match:
            job, label = match[1], None
        match = re.fullmatch(r'      - (?:name|id): (.*)\n', line)
        if match:
            label = match[1]
        if line.startswith('        run:'):
            if job not in FULL_JOBS:
                continue
            if label is None:
                raise Refused('unidentified full workflow command')
            body = [line]
            if line.strip() == 'run: |':
                for following in lines[i + 1:]:
                    if following.strip() and not following.startswith('          '):
                        break
                    body.append(following)
                while body and not body[-1].strip():
                    body.pop()
            command = ''.join(body)
            result.append({'job': job, 'step': label,
                           'sha256': hashlib.sha256(command.encode()).hexdigest()})
    if len({(row['job'], row['step']) for row in result}) != len(result):
        raise Refused('duplicate full workflow command identity')
    return result


def inventory(root):
    data, _ = read_json(Path(root) / INVENTORY)
    if (type(data) is not dict or set(data) != {'schema', 'full_steps', 'guards', 'contracts',
                                             'full_workflow_sha256', 'public_workflow_sha256'}
            or data['schema'] != 'memra-ci-merge-inventory-v1'
            or data['full_workflow_sha256'] != hashlib.sha256(validation_plan.read_local_input(
                root, '.github/workflows/ci.yml', binary=True)).hexdigest()
            or data['public_workflow_sha256'] != hashlib.sha256(validation_plan.read_local_input(
                root, '.github/workflows/ci-public.yml', binary=True)).hexdigest()
            or data['contracts'] != sorted(validation_plan.TOOL_CONTRACTS)
            or data['full_steps'] != run_steps(validation_plan.read_local_input(root, '.github/workflows/ci.yml'))
            or type(data['guards']) is not list
            or [row.get('id') for row in data['guards']] != list(MERGE_GUARDS)):
        raise Refused('incomplete or stale CI execution inventory')
    for row in data['guards']:
        if set(row) != {'id', 'inputs', 'cpu'} or row['cpu'] != MERGE_GUARDS[row['id']]:
            raise Refused('unknown merge guard command')
        if type(row['inputs']) is not list or not row['inputs']:
            raise Refused('missing merge guard input closure')
        for path in row['inputs']:
            _regular(validation_plan.LocalTree(root), path)
    # Thin native compile/ABI/SASS/admission commands are the full workflow's
    # existing identities. Duplication in YAML must not permit coverage drift.
    public = validation_plan.read_local_input(root, '.github/workflows/ci-public.yml')
    native = run_steps(public.replace('  arch:\n', '  arch-coverage:\n', 1))
    for row in data['full_steps']:
        if row['job'] == 'arch-coverage' and row not in native:
            raise Refused('thin native compile coverage differs from full inventory')
    full = validation_plan.read_local_input(root, '.github/workflows/ci.yml')
    begin = full.index('      - name: DSV4 sampled, drift and composition CPU contract tests\n')
    end = full.index('\n      # Reuse one bounded artifact', begin)
    dense = full[begin:end].replace(
        'DSV4 sampled, drift and composition CPU contract tests',
        'DSV4 native CPU control admission').replace('needs.changes', 'needs.route')
    dense = ''.join(line for line in dense.splitlines(keepends=True)
                    if not line.startswith('          cargo test --release -p memra-engine --bin dsv4_tp_ep_sampled_perf_gate'))
    if public.count(dense) != 1:
        raise Refused('thin native dense admission differs from full inventory')
    return data


def source_plan(root, receipt, head):
    """A routing or policy bootstrap cannot authorize its own omissions."""
    head = commit(head)
    if receipt['mode'] != 'thin':
        return full_cpu_plan(root, head, receipt['reason'])
    base = commit(receipt.get('base'))
    before, after = validation_plan.Tree(root, base), validation_plan.Tree(root, head)
    try:
        inventory(root)
        if receipt['event'] == 'pull_request':
            validation_plan.git(root, 'merge-base', '--is-ancestor', commit(receipt.get('head')), head)
        elif receipt.get('head') != head:
            raise Refused('event head does not match candidate source')
        for name in ('tools/public_ci.py', 'tools/ci_merge_validation.json',
                     'tools/validation_plan.py', 'tools/cpu_workflow_inputs.py', POLICY_PATH,
                     '.github/workflows/ci-public.yml'):
            if _regular(before, name) != _regular(after, name):
                raise Refused('routing or merge policy bootstrap/change: ' + name)
        plan = validation_plan.event_plan(root, receipt['event'], base, base, head)
        # Unknown dependency ownership retains the complete CPU inventory.
        if plan['mode'] == 'full':
            return full_cpu_plan(root, head, plan['reason'])
        # Boundary policy changes need its complete checkout and allowlist proof.
        if any(c['id'] == 'public-boundary' for c in plan['cpu_contracts']):
            return full_cpu_plan(root, head, 'public boundary policy/input changed')
        for row in inventory(root)['guards']:
            for path in row['inputs']:
                if _regular(before, path) != _regular(after, path):
                    raise Refused('merge guard implementation changed: ' + path)
        plan.update(ci_mode='thin', qualification=False)
        return plan
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        return full_cpu_plan(root, head, 'unproven thin source/input ownership: ' + str(error))


def execute_contracts(root, plan):
    """Execute shipped command identities, never arbitrary commands from a receipt."""
    if (type(plan) is not dict or plan.get('schema') != 'memra-validation-plan-v1'
            or plan.get('ci_mode') != 'thin' or plan.get('qualification') is not False):
        raise Refused('contract execution requires a thin source plan')
    head = commit(plan.get('head'))
    if validation_plan.git(root, 'rev-parse', 'HEAD').decode().strip() != head:
        raise Refused('plan source does not match checkout')
    for path in ('tools/public_ci.py', 'tools/validation_plan.py',
                 'tools/cpu_workflow_inputs.py', POLICY_PATH, INVENTORY, '.github/workflows/ci-public.yml'):
        if _regular(validation_plan.Tree(root, head), path) != _regular(
                validation_plan.LocalTree(root), path):
            raise Refused('execution helper differs from pinned source')
    policy_rows = _policy(_regular(validation_plan.Tree(root, head), POLICY_PATH)[0])
    rows = plan.get('cpu_contracts')
    if type(rows) is not list or any(type(row) is not dict for row in rows):
        raise Refused('missing selected contract identities')
    names = [row.get('id') for row in rows]
    if (any(type(name) is not str or name not in validation_plan.TOOL_CONTRACTS for name in names)
            or len(names) != len(set(names))):
        raise Refused('unknown or duplicate contract identity')
    expected = [{'id': name, **validation_plan.TOOL_CONTRACTS[name]} for name in names]
    if rows != expected:
        raise Refused('contract receipt differs from shipped registry')
    # Includes input preflight for workflow-only labels, while normal execution
    # remains delegated to the same existing contract runner.
    available = validation_plan.cpu_contract_names(root, ','.join(names) if names else 'none')
    normal = [name for name in names
              if not validation_plan.TOOL_CONTRACTS[name].get('workflow_only')]
    if set(available) != set(normal):
        raise Refused('execution input reader expanded beyond the selected plan')
    executed = []
    guards = inventory(root)['guards']
    for row in guards:
        for path in row['inputs']:
            if _regular(validation_plan.Tree(root, head), path) != _regular(validation_plan.LocalTree(root), path):
                raise Refused('merge guard differs from pinned source: ' + path)
        subprocess.run(row['cpu'], cwd=root, check=True)
    for name in names:
        contract = validation_plan.TOOL_CONTRACTS[name]
        inputs = (next(row['inputs'] for row in policy_rows
            if row['id'] == name) if contract.get('workflow_only') else contract['inputs'])
        for path in inputs:
            if _regular(validation_plan.Tree(root, head), path) != _regular(validation_plan.LocalTree(root), path):
                raise Refused('selected contract differs from pinned source: ' + path)
        if contract.get('workflow_only'):
            for line in COMMANDS[name]:
                command = shlex.split(line)
                if command[0] != 'python3':
                    raise Refused('unexpected workflow caller executable')
                subprocess.run([sys.executable, *command[1:]], cwd=root, check=True)
        else:
            validation_plan.run_cpu_contract(contract, root)
        executed.append(name)
    return {'schema': 'memra-thin-contract-execution-v1', 'head': head,
            'selected': names, 'executed': executed, 'successful': executed,
            'merge_guards': [row['id'] for row in guards],
            'qualification': False}


def merge_result(plan, needs, execution=None):
    """Missing, failed or cancelled selected work can never become merge success."""
    if type(plan) is not dict or plan.get('schema') != 'memra-validation-plan-v1':
        raise Refused('missing source plan')
    commit(plan.get('head'))
    if type(needs) is not dict:
        raise Refused('missing job results')
    mode = 'full' if plan.get('mode') == 'full' else plan.get('ci_mode')
    if mode not in ('thin', 'full'):
        raise Refused('unknown validation mode')
    required = ['route']
    if mode == 'full':
        required.append('full')
        if (type(needs.get('full')) is not dict or
                needs['full'].get('outputs', {}).get('validated_head') != plan['head']):
            raise Refused('full CPU result is not bound to the candidate source')
    else:
        required.extend(['merge-validation', 'boundary'])
        jobs = plan.get('jobs')
        if (type(jobs) is not dict or set(jobs) != set(validation_plan.JOBS)
                or any(type(value) is not bool for value in jobs.values())):
            raise Refused('incomplete component decisions')
        required.extend(name for name in ('build', 'clippy', 'arch') if jobs[name])
        selected = [row['id'] for row in plan['cpu_contracts']]
        if (type(execution) is not dict
                or execution.get('schema') != 'memra-thin-contract-execution-v1'
                or execution.get('head') != plan['head']
                or execution.get('qualification') is not False
                or execution.get('merge_guards') != list(MERGE_GUARDS)
                or any(execution.get(key) != selected
                       for key in ('selected', 'executed', 'successful'))):
            raise Refused('selected contract execution is missing or incomplete')
    for name in required:
        if (type(needs.get(name)) is not dict
                or needs[name].get('result') != 'success'):
            raise Refused('required job did not succeed: ' + name)
    return {'schema': 'memra-ci-merge-result-v1', 'head': plan['head'], 'mode': mode,
            'required_jobs': required, 'success': True, 'qualification': False}


def boundary(root, plan, out):
    if plan.get('ci_mode') != 'thin' or plan.get('mode') != 'scoped':
        raise Refused('range boundary validation requires a thin plan')
    head, base = commit(plan.get('head')), commit(plan.get('base'))
    if validation_plan.git(root, 'rev-parse', 'HEAD').decode().strip() != head:
        raise Refused('boundary checkout differs from candidate')
    commits = validation_plan.git(root, 'rev-list', '--reverse', head, '--not', base).decode()
    if not commits.strip():
        raise Refused('empty public boundary range')
    import tempfile
    with tempfile.TemporaryDirectory(prefix='memra-ci-boundary-') as directory:
        path = Path(directory) / 'commits'
        path.write_text(commits)
        subprocess.run([sys.executable, 'tools/check-public-boundary.py', 'check',
                        '--commits-file', str(path), '--summary-only'], cwd=root, check=True)
    out.write_text(json.dumps({'schema': 'memra-ci-boundary-range-v1', 'head': head,
                              'base': base, 'commits': commits.splitlines(),
                              'whole_tree_allowlist_drift': False}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    entry = sub.add_parser('route')
    entry.add_argument('--event-name', required=True)
    entry.add_argument('--event', type=Path, required=True)
    entry.add_argument('--force-full', action='store_true')
    entry.add_argument('--out', type=Path, required=True)
    entry.add_argument('--github-output', type=Path)
    planning = sub.add_parser('plan')
    planning.add_argument('--route', type=Path, required=True)
    planning.add_argument('--head', required=True)
    planning.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    planning.add_argument('--out', type=Path, required=True)
    planning.add_argument('--github-output', type=Path)
    executing = sub.add_parser('contracts')
    executing.add_argument('--plan', type=Path, required=True)
    executing.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    executing.add_argument('--out', type=Path, required=True)
    finishing = sub.add_parser('result')
    finishing.add_argument('--plan', type=Path, required=True)
    finishing.add_argument('--needs', type=Path, required=True)
    finishing.add_argument('--execution', type=Path)
    finishing.add_argument('--out', type=Path, required=True)
    full = sub.add_parser('full-plan')
    full.add_argument('--head', required=True)
    full.add_argument('--out', type=Path, required=True)
    full.add_argument('--github-output', type=Path)
    full.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    full_result = sub.add_parser('full-result')
    full_result.add_argument('--head', required=True)
    full_result.add_argument('--needs', type=Path, required=True)
    full_result.add_argument('--github-output', type=Path)
    boundary_parser = sub.add_parser('boundary')
    boundary_parser.add_argument('--plan', type=Path, required=True)
    boundary_parser.add_argument('--out', type=Path, required=True)
    boundary_parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    sub.add_parser('check-inventory')
    args = parser.parse_args()
    if args.command == 'check-inventory':
        print(json.dumps(inventory(Path(__file__).resolve().parents[1]), sort_keys=True))
        return
    if args.command == 'full-result':
        inventory(Path(__file__).resolve().parents[1])
        needs, _ = read_json(args.needs)
        if validation_plan.git(Path(__file__).resolve().parents[1], 'rev-parse', 'HEAD').decode().strip() != commit(args.head):
            raise Refused('complete CPU result source differs from checkout')
        if any(type(needs.get(name)) is not dict or needs[name].get('result') != 'success'
               for name in FULL_JOBS):
            raise Refused('complete CPU inventory did not succeed')
        if args.github_output:
            with args.github_output.open('a') as stream:
                stream.write('validated_head=' + commit(args.head) + '\n')
        print('Complete CPU inventory passed: ' + commit(args.head))
        return
    if args.command == 'boundary':
        boundary(args.repo, read_json(args.plan)[0], args.out)
        return
    if args.command == 'full-plan':
        inventory(args.repo)
        plan = full_cpu_plan(args.repo, args.head, 'explicit complete CPU inventory')
        args.out.write_text(json.dumps(plan, indent=2) + '\n')
        if args.github_output:
            import contextlib
            with args.github_output.open('a') as stream, contextlib.redirect_stdout(stream):
                validation_plan.emit(plan)
        print(json.dumps(plan, sort_keys=True))
        return
    if args.command in ('contracts', 'result'):
        plan, _ = read_json(args.plan)
        if args.command == 'contracts':
            receipt = execute_contracts(args.repo, plan)
        else:
            if validation_plan.git(Path(__file__).resolve().parents[1], 'rev-parse', 'HEAD').decode().strip() != commit(plan.get('head')):
                raise Refused('merge result source differs from checkout')
            needs, _ = read_json(args.needs)
            execution = read_json(args.execution)[0] if args.execution else None
            receipt = merge_result(plan, needs, execution)
        args.out.write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt, sort_keys=True))
        return
    if args.command == 'plan':
        receipt, route_digest = read_json(args.route)
        if (type(receipt) is not dict or receipt.get('schema') != 'memra-public-ci-route-v1'
                or receipt.get('mode') not in ('thin', 'full')
                or receipt.get('qualification') is not False):
            raise Refused('missing or malformed route receipt')
        plan = source_plan(args.repo, receipt, args.head)
        plan['route_sha256'] = route_digest
        args.out.write_text(json.dumps(plan, indent=2) + '\n')
        if args.github_output:
            with args.github_output.open('a') as stream:
                stream.write('ci_mode=' + ('full' if plan['mode'] == 'full' else 'thin') + '\n')
                stream.write('packages=' + ','.join(plan['packages']) + '\n')
                stream.write('requires_cuda=' + str(plan['requires_cuda']).lower() + '\n')
                for name in ('build', 'clippy', 'arch', 'engine', 'server'):
                    stream.write(name + '=' + str(plan['jobs'][name]).lower() + '\n')
        print(json.dumps(plan, sort_keys=True))
        return
    event, digest = read_json(args.event)
    receipt = route(args.event_name, event, force_full=args.force_full)
    receipt['event_sha256'] = digest
    receipt['router_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.write_text(json.dumps(receipt, indent=2) + '\n')
    if args.github_output:
        with args.github_output.open('a') as stream:
            stream.write('mode=' + receipt['mode'] + '\n')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
