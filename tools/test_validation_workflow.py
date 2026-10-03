"""Real workflow bytes and Git fixtures for the finite CPU insertion boundary."""

import copy
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

import cpu_workflow_inputs as policy


ROOT = Path(__file__).resolve().parent.parent


class MemoryTree:
    def __init__(self, files, modes=None):
        self.files = dict(files)
        self.modes = {name: '100644' for name in files}
        self.modes.update(modes or {})

    def read_bytes(self, name):
        return self.files[name]

    def input_modes(self, *names, recursive=True):
        return {name: self.modes[name] for name in names if name in self.files}


class GitTree:
    def __init__(self, repo, ref, env):
        self.repo, self.ref, self.env = repo, ref, env

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args],
                                       env=self.env, stderr=subprocess.PIPE)

    def read_bytes(self, name):
        return self.git('show', self.ref + ':' + name)

    def input_modes(self, *names, recursive=True):
        args = ['--literal-pathspecs', 'ls-tree']
        if recursive:
            args.append('-r')
        result = {}
        for row in self.git(*args, '-z', self.ref, '--', *names).split(b'\0'):
            if row:
                metadata, name = row.split(b'\t', 1)
                result[name.decode()] = metadata.split()[0].decode()
        return result


class CpuWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.raw_policy = (ROOT / policy.POLICY_PATH).read_bytes()
        self.rows = json.loads(self.raw_policy)['contracts']
        self.workflow = (ROOT / policy.WORKFLOW_PATH).read_text()
        self.files = {policy.POLICY_PATH: self.raw_policy}
        for row in self.rows:
            for name in row['inputs']:
                self.files[name] = (ROOT / name).read_bytes()
        self.bare = self.workflow
        for row in self.rows:
            self.assertEqual(self.bare.count(row['block']), 1)
            self.bare = self.bare.replace(row['block'], '', 1)

    def tree(self, workflow, changes=None, modes=None):
        files = dict(self.files)
        files[policy.WORKFLOW_PATH] = workflow.encode()
        files.update(changes or {})
        return MemoryTree(files, modes)

    def pair(self):
        return self.tree(self.bare), self.tree(self.workflow)

    def refuse_head(self, text):
        with self.assertRaises(ValueError):
            policy.eligible_additions(self.tree(self.bare), self.tree(text))

    def test_each_known_addition_is_admitted(self):
        for row in self.rows:
            with self.subTest(contract=row['id']):
                head = self.bare.replace(row['anchor'], row['block'] + row['anchor'], 1)
                self.assertEqual(policy.eligible_additions(self.tree(self.bare), self.tree(head)),
                                 [row['id']])

    def test_all_three_real_literal_blocks_are_admitted(self):
        self.assertEqual(policy.eligible_additions(*self.pair()), list(policy.CONTRACT_IDS))

    def test_existing_step_bytes_remain_when_middle_step_is_added(self):
        row = self.rows[1]
        before = self.workflow.replace(row['block'], '', 1)
        self.assertEqual(policy.eligible_additions(self.tree(before), self.tree(self.workflow)),
                         [row['id']])

    def test_no_additions_refuses_instead_of_empty_success(self):
        for text in (self.workflow, self.bare):
            with self.subTest(text=text[:12]), self.assertRaises(ValueError):
                policy.eligible_additions(self.tree(text), self.tree(text))

    def test_native_job_and_global_changes_are_not_normalized(self):
        replacements = [('key: ci-build-120a', 'key: changed-native-cache'),
                        ('cargo test --release -p memra-server', 'cargo test -p memra-server'),
                        ('toolchain: "1.97.1"', 'toolchain: "nightly"'),
                        ('group: ci-${{ github.event.pull_request.number || github.ref }}',
                         'group: changed-workflow-concurrency'),
                        ('timeout-minutes: 30', 'timeout-minutes: 31')]
        for old, new in replacements:
            with self.subTest(old=old):
                self.assertIn(old, self.workflow)
                self.refuse_head(self.workflow.replace(old, new, 1))

    def test_unknown_extra_step_refuses(self):
        text = self.workflow.replace(self.rows[-1]['anchor'],
                    '      - name: CPU-only unknown\n        run: python3 tools/unknown.py\n\n'
                    + self.rows[-1]['anchor'], 1)
        self.refuse_head(text)

    def test_removed_step_refuses_even_with_another_valid_addition(self):
        base = self.workflow.replace(self.rows[1]['block'], '', 1)
        head = self.workflow.replace(self.rows[0]['block'], '', 1)
        with self.assertRaises(ValueError):
            policy.eligible_additions(self.tree(base), self.tree(head))

    def test_reordered_approved_steps_refuse(self):
        first, second = self.rows[:2]
        self.refuse_head(self.workflow.replace(first['block'] + second['block'],
                                              second['block'] + first['block'], 1))

    def test_duplicate_step_or_name_refuses(self):
        row = self.rows[0]
        self.refuse_head(self.workflow.replace(row['block'], row['block'] * 2, 1))
        self.refuse_head(self.workflow.replace(row['anchor'],
            row['block'].replace('python3 tools/build_expert_tier_plan.py --self-test',
                                'echo masked') + row['anchor'], 1))

    def test_registered_block_in_other_job_refuses(self):
        row = self.rows[0]
        block_in_boundary = self.bare.replace('  boundary:\n',
                    '  other:\n    steps:\n' + row['block'] + '  boundary:\n', 1)
        self.refuse_head(block_in_boundary)

    def test_outside_anchor_slot_refuses(self):
        row = self.rows[1]
        self.refuse_head(self.bare.replace('      - name: Immutable GitHub Action pin census\n',
                    row['block'] + '      - name: Immutable GitHub Action pin census\n', 1))

    def test_step_properties_and_swallowed_errors_refuse(self):
        row = self.rows[1]
        for extra in ('        if: false\n', '        continue-on-error: true\n',
                      '        env:\n          PYTHONOPTIMIZE: 1\n',
                      '        shell: bash\n'):
            with self.subTest(extra=extra):
                self.refuse_head(self.workflow.replace(row['block'],
                    row['block'].replace('        run:', extra + '        run:'), 1))
        self.refuse_head(self.workflow.replace('run: python3 tools/run_sft_gen_contract.py',
                                              'run: python3 tools/run_sft_gen_contract.py || true', 1))

    def test_optimized_or_changed_command_refuses(self):
        for command in ('python3 -O tools/run_sft_gen_contract.py',
                        'python3 tools/unknown_contract.py'):
            with self.subTest(command=command):
                self.refuse_head(self.workflow.replace('python3 tools/run_sft_gen_contract.py',
                                                       command, 1))

    def test_each_frozen_dependency_change_or_missing_input_refuses(self):
        for row in self.rows:
            text = self.bare.replace(row['anchor'], row['block'] + row['anchor'], 1)
            for name in row['inputs']:
                with self.subTest(contract=row['id'], path=name):
                    head = self.tree(text, {name: self.files[name] + b'\n# changed input\n'})
                    with self.assertRaises(ValueError):
                        policy.eligible_additions(self.tree(self.bare), head)
                    del head.files[name]
                    with self.assertRaises(ValueError):
                        policy.eligible_additions(self.tree(self.bare), head)

    def test_regular_modes_are_preserved_and_special_types_refuse(self):
        for name in (policy.WORKFLOW_PATH, policy.POLICY_PATH, policy.HELPER_PATH):
            for mode in ('120000', '160000', '040000', 'unsupported', '100755'):
                with self.subTest(path=name, mode=mode), self.assertRaises(ValueError):
                    policy.eligible_additions(self.tree(self.bare), self.tree(self.workflow,
                                             modes={name: mode}))
        base, head = self.pair()
        base.modes[policy.HELPER_PATH] = head.modes[policy.HELPER_PATH] = '100755'
        self.assertEqual(policy.eligible_additions(base, head), list(policy.CONTRACT_IDS))

    def test_head_only_policy_and_changed_policy_refuse(self):
        base, head = self.pair()
        del base.files[policy.POLICY_PATH]
        with self.assertRaises(ValueError):
            policy.eligible_additions(base, head)
        self.refuse_policy(self.raw_policy + b'\n', head_only=True)

    def refuse_policy(self, raw, head_only=False):
        base, head = self.pair()
        head.files[policy.POLICY_PATH] = raw
        if not head_only:
            base.files[policy.POLICY_PATH] = raw
        with self.assertRaises(ValueError):
            policy.eligible_additions(base, head)

    def test_duplicate_policy_keys_and_unsupported_schema_refuse(self):
        self.refuse_policy(self.raw_policy.replace(b'"schema": 1', b'"schema": 1, "schema": 1', 1))
        for schema in (True, 2):
            data = json.loads(self.raw_policy); data['schema'] = schema
            self.refuse_policy(json.dumps(data).encode())

    def test_policy_unknown_fields_ids_paths_and_missing_caller_refuse(self):
        original = json.loads(self.raw_policy)
        for mutation in ('unknown-field', 'unknown-id', 'path', 'omitted-caller', 'duplicate-input'):
            data = copy.deepcopy(original); row = data['contracts'][0]
            if mutation == 'unknown-field': row['if'] = 'false'
            elif mutation == 'unknown-id': row['id'] = 'unknown-caller'
            elif mutation == 'path': row['inputs'][0] = '../outside.py'
            elif mutation == 'omitted-caller': row['inputs'].remove('tools/run_expert_tier_contract.py')
            else: row['inputs'].append(row['inputs'][0])
            with self.subTest(mutation=mutation): self.refuse_policy(json.dumps(data).encode())

    def test_base_known_id_cannot_select_a_different_program_or_producer_mode(self):
        for changed in ('python3 tools/different_program.py',
                        'python3 tools/build_expert_tier_plan.py'):
            data = json.loads(self.raw_policy)
            row = data['contracts'][0]
            row['block'] = row['block'].replace(
                'python3 tools/build_expert_tier_plan.py --self-test', changed)
            row['inputs'].append('tools/different_program.py')
            base, head = self.pair()
            for tree in (base, head):
                tree.files[policy.POLICY_PATH] = json.dumps(data).encode()
                tree.files['tools/different_program.py'] = b'# coherent wrong-program fixture\n'
                tree.modes['tools/different_program.py'] = '100644'
            head.files[policy.WORKFLOW_PATH] = self.workflow.replace(
                self.rows[0]['block'], row['block'], 1).encode()
            with self.subTest(command=changed), self.assertRaises(ValueError):
                policy.eligible_additions(base, head)

    def test_tabs_crlf_and_scalar_masking_refuse(self):
        for text in (self.workflow.replace('\n', '\r\n'),
                     self.workflow.replace('        run:', '\t       run:', 1),
                     self.workflow.replace('        run: |', '        run: >', 1),
                     self.workflow.replace('        run: |', '        run: "unterminated', 1)):
            with self.subTest(prefix=text[:30]): self.refuse_head(text)

    def test_ambiguous_job_and_anchor_refuse(self):
        for text in (self.workflow.replace('  gates:\n', '  gates: &shared\n', 1),
                     self.workflow.replace('    if: ${{ !cancelled() }}', '    if: false', 1),
                     self.workflow.replace(self.rows[0]['anchor'], self.rows[0]['anchor'] * 2, 1)):
            with self.subTest(prefix=text[:30]): self.refuse_head(text)

    def test_unmodelled_base_gates_shape_cannot_authorize_insertions(self):
        for old, new in [('  gates:\n', '  gates: &shared\n'),
                         ('        run: |', '        run: >'),
                         ('    if: ${{ !cancelled() }}', '    if: false'),
                         ('    needs: changes', '    needs: [changes]')]:
            with self.subTest(new=new), self.assertRaises(ValueError):
                def changed(text):
                    start = text.index('  gates:\n')
                    return text[:start] + text[start:].replace(old, new, 1)
                policy.eligible_additions(self.tree(changed(self.bare)),
                                          self.tree(changed(self.workflow)))

    def test_native_bytes_and_existing_floor_changes_refuse(self):
        floor = re.search(r"'test_validation_\*\.py' ([0-9]+)", self.workflow)
        self.assertIsNotNone(floor)
        old_floor = floor.group()
        reduced = "'test_validation_*.py' " + str(int(floor[1]) - 1)
        for old, new in [(old_floor, reduced),
                         ('MEMRA_CUDA_ARCH: "120a"', 'MEMRA_CUDA_ARCH: "100a"')]:
            # This is a coherent actual workflow edit, not a made-up replacement
            # that leaves the accepted fixture untouched.
            with self.subTest(old=old):
                self.assertIn(old, self.workflow)
                self.refuse_head(self.workflow.replace(old, new, 1))

    def test_unrelated_compiled_input_is_not_this_helpers_scope(self):
        base, head = self.pair()
        head.files['crates/example/src/lib.rs'] = b'include_bytes!("../../../tools/fixture");'
        self.assertEqual(policy.eligible_additions(base, head), list(policy.CONTRACT_IDS))
        # The planner must still analyze this path and compiled consumers.

    def test_real_git_policy_and_dependency_custody(self):
        with tempfile.TemporaryDirectory(prefix='memra-cpu-workflow-') as folder:
            root = Path(folder); repo = root / 'repo'; repo.mkdir()
            for directory in ('home', 'hooks', 'templates'):
                (root / directory).mkdir()
            (root / 'gitconfig').write_text('')
            env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
            env.update(HOME=str(root / 'home'), GIT_CONFIG_NOSYSTEM='1',
                       GIT_CONFIG_GLOBAL=str(root / 'gitconfig'))
            def git(*args):
                return subprocess.check_output(['git', '-C', str(repo), *args],
                                               env=env, stderr=subprocess.PIPE)
            git('-c', 'init.templateDir=' + str(root / 'templates'), 'init', '-q')
            for key, value in [('user.name', 'Fixture'), ('user.email', 'fixture@example.invalid'),
                               ('commit.gpgsign', 'false'), ('core.hooksPath', str(root / 'hooks')),
                               ('core.autocrlf', 'false')]:
                git('config', key, value)
            files = dict(self.files); files[policy.WORKFLOW_PATH] = self.bare.encode()
            for name, value in files.items():
                target = repo / name; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(value)
            git('add', '.'); git('commit', '-qm', 'known policy and frozen dependencies')
            before = git('rev-parse', 'HEAD').decode().strip()
            (repo / policy.WORKFLOW_PATH).write_text(self.workflow)
            git('add', '.'); git('commit', '-qm', 'literal CPU additions')
            after = git('rev-parse', 'HEAD').decode().strip()
            self.assertEqual(policy.eligible_additions(GitTree(repo, before, env),
                             GitTree(repo, after, env)), list(policy.CONTRACT_IDS))
            (repo / policy.HELPER_PATH).write_text('# changed classifier\n')
            git('add', '.'); git('commit', '-qm', 'changed admission dependency')
            with self.assertRaises(ValueError):
                policy.eligible_additions(GitTree(repo, before, env), GitTree(repo, 'HEAD', env))


if __name__ == '__main__':
    unittest.main()
