#!/usr/bin/env python3
"""Compile independent CPU fixture readers and compare their declared input reach."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def command(args, **kwargs):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=30, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--bundles', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    baseline = module(args.baseline, 'baseline_compiled_inputs')
    candidate = module(args.candidate, 'candidate_compiled_inputs')
    records = []
    cases = (
        ('conditional-module', 'research/inner.md', True),
        ('raw-direct-module', 'research/inner.md', True),
        ('direct-module', 'research/inner.md', False),
        ('split-concat-include', 'research/actual/expected.md', True),
        ('literal-include', 'research/actual/expected.md', False),
    )
    for label, target, omitted in cases:
        bundle = args.bundles / (label + '-control.bundle')
        with tempfile.TemporaryDirectory(prefix='memra-compiled-input-') as temp:
            repo = Path(temp) / 'fixture'
            command(['git', 'clone', '--quiet', str(bundle.resolve()), str(repo)])
            after = command(['git', '-C', str(repo), 'rev-parse', 'HEAD']).stdout.strip()
            before = command(['git', '-C', str(repo), 'rev-parse', 'HEAD^']).stdout.strip()
            changed = command(['git', '-C', str(repo), 'diff', '--name-only', before, after]).stdout.splitlines()
            assert changed == [target], changed
            executions = []
            for phase, revision, expected in (('before', before, 'BEFORE'), ('after', after, 'AFTER')):
                command(['git', '-C', str(repo), 'checkout', '--quiet', revision])
                source = repo / 'crates/memra-server/src/lib.rs'
                binary = args.out / (label + '-' + phase)
                compiled = command(['rustc', '--edition=2024', str(source), '-o', str(binary.resolve())])
                output = command([str(binary.resolve())], cwd=repo).stdout.strip()
                assert output == expected, (label, phase, output)
                executions.append({'phase': phase, 'revision': revision, 'output': output,
                    'binary_sha256': sha(binary), 'source_sha256': sha(source),
                    'compiler_stdout': compiled.stdout, 'compiler_stderr': compiled.stderr})
            old = baseline.event_plan(repo, 'push', '', before, after)
            new = candidate.event_plan(repo, 'push', '', before, after)
            assert old['changed'] == new['changed'] == [target]
            assert old['jobs']['server'] is (not omitted), (label, old)
            assert new['jobs']['server'] and new['mode'] == 'full', (label, new)
            assert not old['native']['qualification'] and not new['native']['qualification']
            assert executions[0]['binary_sha256'] != executions[1]['binary_sha256']
            records.append({'label': label, 'changed': changed, 'bundle_sha256': sha(bundle),
                'executions': executions, 'baseline_plan': old, 'candidate_plan': new})
    proof = {'cpu_fixture_only': True, 'native_qualification': False,
        'baseline_helper_sha256': sha(args.baseline), 'candidate_helper_sha256': sha(args.candidate),
        'collector_sha256': sha(Path(__file__)), 'rustc': command(['rustc', '-vV']).stdout,
        'cases': records}
    (args.out / 'PROOF.json').write_text(json.dumps(proof, indent=2) + '\n')
    print('compiled-input replay: three executed omissions repaired; two independent positive controls retained; native qualification=false')


if __name__ == '__main__':
    main()
