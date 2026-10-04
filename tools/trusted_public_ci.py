#!/usr/bin/env python3
"""Isolated entry from an immutable base checkout, with a separate candidate root."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

CODE_INPUTS = ('trusted_public_ci.py', 'public_ci.py', 'validation_plan.py',
               'cpu_workflow_inputs.py', 'sparse_input_preflight.py',
               'support_record_inputs.py', 'skip-census.py', 'validation_inputs.json',
               'ci_merge_validation.json', 'cpu_workflow_contracts.json')


def trusted_modules(root, head):
    if not sys.flags.isolated or not re.fullmatch('[0-9a-f]{40}', head):
        raise ValueError('trusted entry requires isolated Python and an immutable base')
    def git(*args):
        return subprocess.check_output(['git', '--no-replace-objects', '-C', str(root), *args])
    if git('rev-parse', 'HEAD').decode().strip() != head:
        raise ValueError('trusted checkout does not match event base')
    directory = os.open(root / 'tools', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name in CODE_INPUTS:
            metadata = git('ls-tree', head, '--', 'tools/' + name).split(b'\t')[0].split()
            if len(metadata) != 3 or metadata[0] not in (b'100644', b'100755'):
                raise ValueError('missing or nonregular trusted input: ' + name)
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            with os.fdopen(fd, 'rb') as stream:
                info = os.fstat(stream.fileno())
                mode = b'100755' if info.st_mode & stat.S_IXUSR else b'100644'
                if not stat.S_ISREG(info.st_mode) or mode != metadata[0]:
                    raise ValueError('trusted input mode differs: ' + name)
                if stream.read() != git('cat-file', 'blob', metadata[2].decode()):
                    raise ValueError('trusted input bytes differ: ' + name)
    finally:
        os.close(directory)
    # -I excludes both the candidate cwd and PYTHONPATH/sitecustomize. Add only
    # this checked base directory, never a candidate script/module directory.
    sys.path.insert(0, str(root / 'tools'))
    import public_ci
    return public_ci


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('route-plan', 'contracts', 'result', 'boundary'))
    parser.add_argument('--trusted-head', required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--event-name')
    parser.add_argument('--event', type=Path)
    parser.add_argument('--head')
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--github-output', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--needs', type=Path)
    parser.add_argument('--execution', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    root = Path(__file__).absolute().parents[1]
    candidate = args.repo.absolute()
    if ('..' in args.repo.parts or root.parent != candidate.parent
            or root == candidate or root in candidate.parents or candidate in root.parents):
        raise ValueError('trusted and candidate checkouts must be separate siblings')
    ci = trusted_modules(root, args.trusted_head)
    if args.command == 'route-plan':
        if not args.event or not args.event_name or not args.head or not args.out_dir:
            parser.error('route-plan requires event, head and output directory')
        if ci.validation_plan.git(candidate, 'rev-parse', 'HEAD').decode().strip() != ci.commit(args.head):
            raise ValueError('candidate checkout does not match requested execution head')
        event, digest = ci.read_json(args.event)
        route = ci.route(args.event_name, event)
        route.update(event_sha256=digest, trusted_base=args.trusted_head,
                     router_sha256=hashlib.sha256((root / 'tools/public_ci.py').read_bytes()).hexdigest())
        route_path = args.out_dir / 'route.json'
        route_path.write_text(json.dumps(route, indent=2) + '\n')
        plan = ci.source_plan(candidate, route, args.head)
        plan.update(trusted_base=args.trusted_head,
                    route_sha256=hashlib.sha256(route_path.read_bytes()).hexdigest())
        (args.out_dir / 'validation-plan.json').write_text(json.dumps(plan, indent=2) + '\n')
        if args.github_output:
            with args.github_output.open('a') as stream:
                stream.write('ci_mode=' + ('full' if plan['mode'] == 'full' else 'thin') + '\n')
                stream.write('packages=' + ','.join(plan['packages']) + '\n')
                stream.write('requires_cuda=' + str(plan['requires_cuda']).lower() + '\n')
                for name in ('build', 'clippy', 'arch', 'engine', 'server'):
                    stream.write(name + '=' + str(plan['jobs'][name]).lower() + '\n')
        print(json.dumps(plan, sort_keys=True))
        return
    plan, _ = ci.read_json(args.plan)
    if plan.get('trusted_base') != args.trusted_head:
        raise ValueError('plan does not bind the immutable base entry')
    if ci.validation_plan.git(candidate, 'rev-parse', 'HEAD').decode().strip() != plan.get('head'):
        raise ValueError('candidate checkout does not match the source plan')
    if args.command == 'contracts':
        result = ci.execute_contracts(candidate, plan)
    elif args.command == 'boundary':
        ci.boundary(candidate, plan, args.out)
        return
    else:
        needs, _ = ci.read_json(args.needs)
        execution = ci.read_json(args.execution)[0] if args.execution else None
        result = ci.merge_result(plan, needs, execution)
    args.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
