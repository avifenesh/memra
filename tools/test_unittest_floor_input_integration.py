#!/usr/bin/env python3
"""One real Git witness for the shared floor executable input closure."""
import ast
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import tomllib

import validation_plan as current

ROOT = Path(__file__).absolute().parent.parent
SHARED = ('tools/unittest-floor.sh', 'tools/unittest_floor.py')
LEGACY = '71a91c6fce00fda25a2562387c1bf521e259c23a'
LEGACY_PINS = {
    'validation_plan.py': '326558061ee7e8bf0d6057c15a0dd0a2563e0b10fae82d0a47130afd45e53385',
    'support_record_inputs.py': 'd3548a8fa3f765046e1864efd78c028d596151971a45908dad38230f021276d5',
    'cpu_workflow_inputs.py': 'd8041dac1acbdd232b1662f60673766fe55f639c488e8229c27f00b689aee937',
    'skip-census.py': 'd89ce391d748499214df2279a48d719b9e67cad6a2838c77ebcef46640883a7d',
    'validation_inputs.json': '71e17d5f91f44bae768047ad4c9f2c40486a291e35e123e6a7e54f16fd34d377',
    'cpu_workflow_contracts.json': '12543703a339d6457162499bb2f80795fe2222349eed88103b7543b0dab90bd0',
    'unittest-floor.sh': '19b9d1ee25d8bc31de732a0284ac30be161cda7119fe15acfac1863eafff605d',
    'unittest_floor.py': 'be5cff7c72f651c13829b3f07d35e13cb3ebd4ed7ad439bee5d9705a17551ac7',
}


@contextmanager
def isolated_git(scratch):
    """Cover planner subprocesses too, without changing the caller's HOME."""
    config = scratch / 'gitconfig'; config.write_text('')
    hooks = scratch / 'hooks'; hooks.mkdir()
    templates = scratch / 'templates'; templates.mkdir()
    saved = {k: v for k, v in os.environ.items() if k.startswith('GIT_')}
    for key in saved:
        del os.environ[key]
    settings = {'core.hooksPath': str(hooks), 'init.templateDir': str(templates),
                'commit.gpgsign': 'false', 'tag.gpgsign': 'false',
                'gc.auto': '0', 'maintenance.auto': 'false',
                'core.fsmonitor': 'false', 'core.untrackedCache': 'false',
                'core.autocrlf': 'false', 'core.fileMode': 'true'}
    os.environ.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=str(config),
                      GIT_CONFIG_COUNT=str(len(settings)))
    for index, (key, value) in enumerate(settings.items()):
        os.environ['GIT_CONFIG_KEY_' + str(index)] = key
        os.environ['GIT_CONFIG_VALUE_' + str(index)] = value
    try:
        yield
    finally:
        for key in list(os.environ):
            if key.startswith('GIT_'):
                del os.environ[key]
        os.environ.update(saved)


def legacy_blob(name):
    raw = current.git(ROOT, 'show', LEGACY + ':tools/' + name)
    assert hashlib.sha256(raw).hexdigest() == LEGACY_PINS[name], name
    row = current.git(ROOT, 'ls-tree', LEGACY, '--', 'tools/' + name).decode().split()
    assert row[0] == ('100755' if name in ('unittest-floor.sh', 'skip-census.py') else '100644'), row
    return raw


def run(root, *args, expected=0):
    result = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=60)
    if expected == 0:
        assert result.returncode == 0, (args, result.stdout, result.stderr)
    else:
        assert result.returncode != 0, (args, result.stdout, result.stderr)
    return result


def git(root, *args):
    return run(root, 'git', *args).stdout.strip()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def preflight_refuses(module, root, name):
    try:
        module.cpu_contract_names(root, name)
    except module.Refused as error:
        assert 'selected contract input is missing: ' + name in str(error), error
        return
    raise AssertionError('missing shared executable was admitted: ' + name)


def registry(source):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'TOOL_CONTRACTS' for t in n.targets))
    assert not any(isinstance(n, (ast.Call, ast.Attribute, ast.Lambda)) for n in ast.walk(node.value))
    return eval(compile(ast.Expression(node.value), '<literal registry>', 'eval'),
                {'__builtins__': {}}, {})


def preserved(candidate, before, names):
    restored = copy.deepcopy(candidate)
    for name in names:
        assert restored[name]['inputs'][:2] == list(SHARED)
        restored[name]['inputs'] = restored[name]['inputs'][2:]
    assert restored == before, 'command/floor/presence/requirements/native obligation changed'


def main():
    if not __debug__:
        raise SystemExit('integration requires assertions')
    before_source = legacy_blob('validation_plan.py').decode()
    after_source = (ROOT / 'tools/validation_plan.py').read_text()
    before = registry(before_source)
    names = [n for n, c in current.TOOL_CONTRACTS.items() if c['cpu'][0] == SHARED[0]]
    assert len(names) == 10
    preserved(current.TOOL_CONTRACTS, before, names)
    drift_refusals = []
    for name in names:
        for field in ('cpu', 'presence', 'required', 'python_requirements', 'native'):
            mutant = copy.deepcopy(current.TOOL_CONTRACTS)
            value = mutant[name].get(field)
            mutant[name][field] = (not value if isinstance(value, bool) else
                                   value + ['unapproved-drift'] if isinstance(value, list) else
                                   value + '.drift' if isinstance(value, str) else True)
            try:
                preserved(mutant, before, names)
            except AssertionError:
                drift_refusals.append(name + ':' + field)
            else:
                raise AssertionError('contract field drift survived: ' + name + ':' + field)
        mutant = copy.deepcopy(current.TOOL_CONTRACTS)
        mutant[name]['cpu'][-1] = str(int(mutant[name]['cpu'][-1]) - 1)
        try:
            preserved(mutant, before, names)
        except AssertionError:
            drift_refusals.append(name + ':floor')
        else:
            raise AssertionError('floor drift survived: ' + name)
    events = []
    with tempfile.TemporaryDirectory(prefix='memra-floor-input-') as folder, isolated_git(Path(folder)):
        scratch = Path(folder)
        old_dir = scratch / 'old-tools'; old_dir.mkdir()
        for name in LEGACY_PINS:
            body = legacy_blob(name)
            path = old_dir / name
            path.write_bytes(body)
            path.chmod(0o755 if name in ('unittest-floor.sh', 'skip-census.py') else 0o644)
            if name != 'validation_plan.py':
                live = ROOT / 'tools' / name
                assert live.read_bytes() == body, 'candidate import/metadata drift: ' + name
                assert bool(live.stat().st_mode & 0o100) == bool(path.stat().st_mode & 0o100), name
        old = load(old_dir / 'validation_plan.py', 'original_floor_inputs')
        root = scratch / 'repo'; root.mkdir()
        git(root, 'init', '-q')
        git(root, 'config', 'user.name', 'Floor Input Fixture')
        git(root, 'config', 'user.email', 'floor@example.invalid')
        inputs = set(SHARED)
        for name in names:
            inputs.update(before[name]['inputs'])
        metadata = tomllib.loads(current.git(ROOT, 'show', LEGACY + ':docs/support-records.toml').decode())
        for record in metadata['record']:
            for paths in record.get('evidence', {}).values():
                for path in paths:
                    if path.startswith('ci:'):
                        continue
                    inputs.add(path)
                    inputs.update(str(PurePosixPath(path).parent / leaf)
                                  for leaf in ('artifact.lock', 'tiny-gate.tsv'))
        inputs.update(('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml',
                       'tools/validation_inputs.json', 'tools/cpu_workflow_inputs.py',
                       'tools/cpu_workflow_contracts.json',
                       'research/modelplan-onboarding-hy3-20260830/tiny/gates.txt'))
        inputs.update(current.git(ROOT, 'ls-tree', '-r', '--name-only', LEGACY, '--', 'crates').decode().splitlines())
        tracked = set(current.git(ROOT, 'ls-tree', '-r', '--name-only', LEGACY).decode().splitlines())
        inputs &= tracked
        rows = current.git(ROOT, '--literal-pathspecs', 'ls-tree', '-r', '-z', LEGACY, '--', *sorted(inputs)).split(b'\0')
        objects = []
        for row in rows:
            if row:
                meta, name = row.split(b'\t', 1)
                mode, kind, oid = meta.decode().split()
                assert mode in ('100644', '100755'), (name, mode)
                objects.append((name.decode(), mode, oid))
        raw = subprocess.run(['git', '-C', str(ROOT), 'cat-file', '--batch'],
                             input=''.join(oid + '\n' for _, _, oid in objects).encode(),
                             capture_output=True, check=True, timeout=60).stdout
        cursor = 0
        fixture_manifest = []
        for name, mode, oid in objects:
            end = raw.index(b'\n', cursor)
            header = raw[cursor:end].decode().split(); assert header[:2] == [oid, 'blob']
            size = int(header[2]); body = raw[end + 1:end + 1 + size]; cursor = end + 2 + size
            path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body); path.chmod(0o755 if mode == '100755' else 0o644)
            fixture_manifest.append({'path': name, 'mode': mode, 'git_blob': oid,
                                     'sha256': hashlib.sha256(body).hexdigest()})
        assert cursor == len(raw)
        # This owned suite exercises the unchanged real admission executable,
        # without executing unrelated real model/provider collectors.
        suite = root / 'owned-suite'; suite.mkdir()
        (suite / 'test_owned.py').write_text('import unittest\nclass Cases(unittest.TestCase):\n'
                                           ' def test_one(self): self.assertEqual(1,1)\n'
                                           ' def test_two(self): self.assertTrue(True)\n')
        git(root, 'add', '.')
        git(root, 'commit', '-q', '-m', 'Original pinned contract input fixture')
        pinned = git(root, 'rev-parse', 'HEAD')
        assert len(current.workspace(current.Tree(root, pinned))[0]) == 14
        for name in names:
            assert current.cpu_contract_names(root, name) == [name]
        events.append('all10 exact original declarations and real input preflight pass')
        # Prove usable source analysis before guard controls. A fixture outage
        # returning full cannot be mistaken for the retained coverage predicate.
        ordinary_path = root / 'tools/cache-meter-gate.py'
        ordinary_path.write_bytes(ordinary_path.read_bytes() + b'\n# analysis control\n')
        git(root, 'add', 'tools/cache-meter-gate.py'); git(root, 'commit', '-q', '-m', 'Ordinary analysis control')
        analysis = current.event_plan(root, 'push', '', pinned, git(root, 'rev-parse', 'HEAD'))
        assert analysis['mode'] == 'scoped' and analysis['reason'] == 'dependency closure plus declared input contracts', analysis['reason']
        assert [c['id'] for c in analysis['cpu_contracts']] == ['cache-meter']
        old_analysis = old.event_plan(root, 'push', '', pinned, git(root, 'rev-parse', 'HEAD'))
        assert old_analysis['mode'] == 'scoped', old_analysis['reason']
        git(root, 'reset', '-q', '--hard', pinned)

        sibling = root / SHARED[1]; sibling.unlink()
        assert old.cpu_contract_names(root, 'cache-meter') == ['cache-meter']
        failed = run(root, str(root / SHARED[0]), str(suite), 'test_*.py', '2', expected=1)
        assert 'unittest_floor.py' in failed.stderr and 'No such file' in failed.stderr, failed
        for name in names:
            preflight_refuses(current, root, name)
        # Deliberate omission must fail the same refusal assertion, not setup.
        for name in names:
            original = current.TOOL_CONTRACTS[name]['inputs']
            current.TOOL_CONTRACTS[name]['inputs'] = [p for p in original if p != SHARED[1]]
            try:
                try: preflight_refuses(current, root, name)
                except AssertionError: pass
                else: raise AssertionError('omitted Python input mutant survived')
            finally: current.TOOL_CONTRACTS[name]['inputs'] = original
        events.append('original preflight accepts, original shell fails, all10 corrected refusals and omission mutants')
        git(root, 'restore', '--source', pinned, '--', SHARED[1])
        successful = run(root, str(root / SHARED[0]), str(suite), 'test_*.py', '2')
        assert 'ran 2 tests (floor 2)' in successful.stdout, successful
        for name in names: assert current.cpu_contract_names(root, name) == [name]
        events.append('exact Git restore then original wrapper executes both owned predicates')

        shell = root / SHARED[0]; shell.unlink()
        for name in names:
            preflight_refuses(current, root, name)
            original = current.TOOL_CONTRACTS[name]['inputs']
            current.TOOL_CONTRACTS[name]['inputs'] = [p for p in original if p != SHARED[0]]
            try:
                try: preflight_refuses(current, root, name)
                except AssertionError: pass
                else: raise AssertionError('omitted shell input mutant survived')
            finally: current.TOOL_CONTRACTS[name]['inputs'] = original
        git(root, 'restore', '--source', pinned, '--', SHARED[0])
        assert shell.stat().st_mode & 0o100
        events.append('all10 absent shell refusals and omission mutants preserve original executable mode')
        for shared in SHARED:
            path = root / shared; path.write_bytes(path.read_bytes() + b'\n# source change\n')
            git(root, 'add', shared); git(root, 'commit', '-q', '-m', 'Shared admission source change')
            candidate = git(root, 'rev-parse', 'HEAD')
            old_plan = old.event_plan(root, 'push', '', pinned, candidate)
            plan = current.event_plan(root, 'push', '', pinned, candidate)
            assert plan['mode'] == old_plan['mode'] == 'full'
            assert old_plan['reason'] == 'unmodelled input: ' + shared, old_plan['reason']
            assert plan['reason'] == 'shared unittest admission implementation changed', plan['reason']
            assert plan['jobs'] == old_plan['jobs'] and plan['native']['scope'] == old_plan['native']['scope'] == 'full'
            assert not plan['native']['qualification']
            # Remove the complete-selection guard from a real compiled planner.
            guard = "    if any(path in ('tools/unittest-floor.sh', 'tools/unittest_floor.py') for path in paths):\n        return full('shared unittest admission implementation changed', paths)\n"
            assert after_source.count(guard) == 1
            mutant_path = old_dir / 'mutant_guard.py'; mutant_path.write_text(after_source.replace(guard, ''))
            mutant = load(mutant_path, 'floor_guard_mutant')
            wrong = mutant.event_plan(root, 'push', '', pinned, candidate)
            assert wrong['mode'] == 'scoped' and wrong['reason'] == 'dependency closure plus declared input contracts', wrong['reason']
            assert {c['id'] for c in wrong['cpu_contracts']} == set(names)
            try: assert wrong['mode'] == 'full'
            except AssertionError: pass
            else: raise AssertionError('removed-full-guard mutant survived: ' + wrong['reason'])
            git(root, 'reset', '-q', '--hard', pinned)
        events.append('both shared source changes remain full; removed-full-guard mutants refuse')
        path = root / 'tools/cache-meter-gate.py'; path.write_bytes(path.read_bytes() + b'\n# ordinary contract change\n')
        git(root, 'add', str(path.relative_to(root))); git(root, 'commit', '-q', '-m', 'One original collector source change')
        one = current.event_plan(root, 'push', '', pinned, git(root, 'rev-parse', 'HEAD'))
        assert one['mode'] == 'scoped' and [c['id'] for c in one['cpu_contracts']] == ['cache-meter']
        assert one['native']['requirements'] == sorted(before['cache-meter']['native'])
        assert not any(one['jobs'].values()) and not one['native']['qualification']
        events.append('ordinary one-contract scope and original native obligation preserved')
        git(root, 'reset', '-q', '--hard', pinned)
        (root / 'tools/unknown-input.py').write_text('# unmodelled executable input\n')
        git(root, 'add', 'tools/unknown-input.py'); git(root, 'commit', '-q', '-m', 'Unknown input')
        unknown = current.event_plan(root, 'push', '', pinned, git(root, 'rev-parse', 'HEAD'))
        assert unknown['mode'] == 'full' and unknown['reason'] == 'unmodelled input: tools/unknown-input.py', unknown['reason']
        assert all(unknown['jobs'].values()) and unknown['native']['scope'] == 'full'
        assert not unknown['native']['qualification']
        events.append('unknown input retains full selection without analysis failure')
        git(root, 'reset', '-q', '--hard', pinned)
        assert git(root, 'diff', '--exit-code') == ''
    assert current.TOOL_CONTRACTS == registry(after_source)
    print(json.dumps({'ok': True, 'contracts': names, 'events': events,
                      'original_missing_sibling_stderr': failed.stderr,
                      'original_restored_stdout': successful.stdout,
                      'planner_sha256': hashlib.sha256(after_source.encode()).hexdigest(),
                      'legacy_commit': LEGACY, 'legacy_source_pins': LEGACY_PINS,
                      'fixture_manifest': fixture_manifest,
                      'field_drift_refusals': drift_refusals,
                      'scratch_retired': True, 'qualification': False}, indent=2))


if __name__ == '__main__':
    main()
