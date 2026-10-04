"""One real Git/event witness for public CI modes, source edges and conclusions."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import re
import importlib.util

import public_ci as ci
import validation_plan as vp

ROOT = Path(__file__).resolve().parents[1]


def publication_dependency(text, consumer):
    """Parse literal jobs scope, after the repository's duplicate-key refusal."""
    spec = importlib.util.spec_from_file_location('workflow_keys', ROOT / 'tools/check-workflow-keys.py')
    walker = importlib.util.module_from_spec(spec); spec.loader.exec_module(walker)
    try:
        jobs = walker.walk(text)
    except (walker.Refused, walker.Unsupported) as error:
        raise AssertionError('workflow mapping refused') from error
    if consumer not in jobs or 'cpu-suite' not in jobs or text.count('\njobs:\n') != 1:
        raise AssertionError('missing publication jobs')
    jobs_text = text.split('\njobs:\n', 1)[1]
    fields = list(re.finditer(r'^  ([A-Za-z0-9_-]+):\n', jobs_text, re.M))
    bodies = {m[1]: jobs_text[m.end():fields[i + 1].start() if i + 1 < len(fields) else len(jobs_text)]
              for i, m in enumerate(fields)}
    if re.findall(r'^    needs: (.*)$', bodies[consumer], re.M) != ['cpu-suite']:
        raise AssertionError('publication job does not depend on full CPU validation')
    if re.findall(r'^    uses: (.*)$', bodies['cpu-suite'], re.M) != ['./.github/workflows/ci.yml']:
        raise AssertionError('publication preflight is not the existing full workflow')


class PublicCiIntegration(unittest.TestCase):
    def test_modes_sources_commands_and_results(self):
        with tempfile.TemporaryDirectory(prefix='memra-public-ci-') as folder:
            scratch = Path(folder)
            hooks, templates = scratch / 'hooks', scratch / 'templates'
            hooks.mkdir(); templates.mkdir()
            config = scratch / 'gitconfig'; config.write_text('')
            env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
            env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=str(config),
                       PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
                       OMP_NUM_THREADS='1')
            repo = scratch / 'repo'

            def command(args, *, ok=True, cwd=repo):
                result = subprocess.run(args, cwd=cwd, env=env, text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                if ok:
                    self.assertEqual(result.returncode, 0, result.stdout)
                else:
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                return result.stdout

            command(['/usr/bin/git', '-c', 'init.templateDir=' + str(templates),
                     'clone', '--shared', '--no-checkout', str(ROOT), str(repo)], cwd=scratch)
            for key, value in [('core.hooksPath', str(hooks)), ('user.name', 'Fixture'),
                               ('user.email', 'fixture@example.invalid'), ('commit.gpgsign', 'false')]:
                command(['/usr/bin/git', 'config', key, value])
            command(['/usr/bin/git', 'sparse-checkout', 'set', 'tools', 'crates', 'docs', '.github'])
            command(['/usr/bin/git', 'checkout', '-q', '--detach', vp.git(ROOT, 'rev-parse', 'HEAD').decode().strip()])

            def commit_file(name, text):
                path = repo / name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
                command(['/usr/bin/git', 'add', '--sparse', name])
                command(['/usr/bin/git', 'commit', '-qm', 'fixture input'])
                return command(['/usr/bin/git', 'rev-parse', 'HEAD']).strip()

            source = command(['/usr/bin/git', 'rev-parse', 'HEAD']).strip()
            self.assertEqual(len(vp.workspace(vp.Tree(repo, source))[0]), 14)
            descriptor = {'full_name': ci.REPOSITORY,
                          'owner': {'login': ci.OWNER, 'id': ci.OWNER_ID}}
            external = {'login': 'contributor', 'id': 42}

            def event(base, head, *, author=None, head_repo=None):
                return {'repository': descriptor, 'actor': ci.OWNER,
                        'pull_request': {'user': author or descriptor['owner'],
                            'base': {'sha': base, 'ref': 'main', 'repo': descriptor},
                            'head': {'sha': head, 'ref': 'topic', 'repo': head_repo or descriptor}}}

            def routed(name, data):
                file = scratch / 'event.json'; file.write_text(json.dumps(data))
                out = scratch / 'route.json'
                command([sys.executable, str(repo / 'tools/public_ci.py'), 'route',
                         '--event-name', name, '--event', str(file), '--out', str(out)])
                return json.loads(out.read_text())

            def planned(route, head):
                route_file = scratch / 'route.json'; route_file.write_text(json.dumps(route))
                out = scratch / 'plan.json'
                command([sys.executable, str(repo / 'tools/public_ci.py'), 'plan',
                         '--route', str(route_file), '--head', head, '--repo', str(repo), '--out', str(out)])
                return json.loads(out.read_text())

            # Actual graph, immutable source, ordinary owner docs change.
            head = commit_file('research/public-ci-integration.md', 'fixture docs\n')
            owned = routed('pull_request', event(source, head))
            self.assertEqual(owned['mode'], 'thin')
            thin = planned(owned, head)
            self.assertEqual(thin.get('ci_mode'), 'thin', thin['reason'])
            self.assertFalse(any(thin['jobs'].values()))
            self.assertEqual(thin['native']['scope'], 'none')
            # Author, not actor, confers mode; stable id and fork lineage matter.
            foreign = routed('pull_request', event(source, head, author=external))
            self.assertEqual(foreign['mode'], 'full')
            self.assertTrue(all(planned(foreign, head)['jobs'].values()))
            fork = dict(descriptor, full_name=ci.OWNER + '/memra-fork')
            self.assertEqual(routed('pull_request', event(source, head, head_repo=fork))['mode'], 'thin')
            self.assertEqual(routed('pull_request', event(source, head,
                author={'login': ci.OWNER, 'id': 42}))['mode'], 'full')
            self.assertEqual(routed('push', {'repository': descriptor, 'ref': 'refs/heads/main',
                                            'before': source, 'after': head})['mode'], 'thin')
            for mode in ('schedule', 'workflow_call', 'workflow_dispatch', 'unknown'):
                full = planned(routed(mode, {'repository': descriptor}), head)
                self.assertTrue(all(full['jobs'].values()))
                self.assertEqual(full['cpu_inventory'], 'complete')
                self.assertFalse(full['qualification'])
                self.assertTrue(any(c['native'] for c in full['cpu_contracts']))
            self.assertEqual(routed('pull_request', {'repository': descriptor})['mode'], 'full')
            # Event source cannot be replayed against an unrelated candidate.
            self.assertEqual(planned(routed('push', {'repository': descriptor,
                'ref': 'refs/heads/main', 'before': source, 'after': source}), head)['mode'], 'full')

            # Owned real workflow restoration: exact literal SFT step, no fake command.
            policy = json.loads((repo / 'tools/cpu_workflow_contracts.json').read_text())
            block = next(row['block'] for row in policy['contracts'] if row['id'] == 'sft-generator-caller')
            workflow = (repo / '.github/workflows/ci.yml').read_text()
            bare = commit_file('.github/workflows/ci.yml', workflow.replace(block, '', 1))
            restored = commit_file('.github/workflows/ci.yml', workflow)
            selected = planned(routed('pull_request', event(bare, restored)), restored)
            self.assertEqual(selected.get('ci_mode'), 'thin', selected['reason'])
            self.assertEqual([row['id'] for row in selected['cpu_contracts']], ['sft-generator-caller'])
            execution_file = scratch / 'execution.json'
            plan_file = scratch / 'selected-plan.json'; plan_file.write_text(json.dumps(selected))
            output = command([sys.executable, str(repo / 'tools/public_ci.py'), 'contracts',
                              '--plan', str(plan_file), '--repo', str(repo), '--out', str(execution_file)])
            self.assertEqual(output.count('SFT CPU contract: PASS: original=9 executed=18'), 1, output)
            execution = json.loads(execution_file.read_text())
            self.assertEqual(execution['executed'], ['sft-generator-caller'])
            # A schema-valid shortened mutable policy cannot omit producer checks
            # from the pinned execution tuple, even with the command unchanged.
            policy_path = repo / 'tools/cpu_workflow_contracts.json'
            policy_original = policy_path.read_bytes()
            shortened = json.loads(policy_original)
            sft_row = next(row for row in shortened['contracts'] if row['id'] == 'sft-generator-caller')
            sft_row['inputs'] = ['tools/cpu_workflow_inputs.py', 'tools/run_sft_gen_contract.py']
            policy_path.write_text(json.dumps(shortened))
            try:
                refused = command([sys.executable, str(repo / 'tools/public_ci.py'), 'contracts',
                                   '--plan', str(plan_file), '--repo', str(repo),
                                   '--out', str(execution_file)], ok=False)
                self.assertIn('execution helper differs from pinned source', refused)
                self.assertNotIn('SFT CPU contract: PASS', refused)
            finally:
                policy_path.write_bytes(policy_original)
            # Coherent masked/swallowed full caller changes leave its run body
            # intact. Whole execution-shape binding must refuse the actual CLI.
            full_path = repo / '.github/workflows/ci.yml'
            for property_line in ('        if: ${{ false }}\n',
                                  '        continue-on-error: true\n'):
                full_path.write_text(workflow.replace(
                    '        run: python3 tools/run_sft_gen_contract.py',
                    property_line + '        run: python3 tools/run_sft_gen_contract.py', 1))
                try:
                    command([sys.executable, str(repo / 'tools/public_ci.py'), 'check-inventory'], ok=False)
                finally:
                    full_path.write_text(workflow)
            # Removing native admission while updating the public shape hash must
            # still refuse. A coherent inventory cannot authorize that omission.
            public_path = repo / '.github/workflows/ci-public.yml'
            public_bytes = public_path.read_text()
            inventory_path = repo / ci.INVENTORY
            inventory_bytes = inventory_path.read_text()
            begin = public_bytes.index('      - name: DSV4 native CPU control admission\n')
            end = public_bytes.index('      - uses: actions/cache/save@', begin)
            omitted = public_bytes[:begin] + public_bytes[end:]
            public_path.write_text(omitted)
            changed_inventory = json.loads(inventory_bytes)
            changed_inventory['public_workflow_sha256'] = __import__('hashlib').sha256(omitted.encode()).hexdigest()
            inventory_path.write_text(json.dumps(changed_inventory))
            try:
                command([sys.executable, str(repo / 'tools/public_ci.py'), 'check-inventory'], ok=False)
            finally:
                public_path.write_text(public_bytes)
                inventory_path.write_text(inventory_bytes)
            # Conclusive actual CLI result and a coherent missing-execution mutant.
            needs = {name: {'result': 'success'} for name in
                     ('route', 'merge-validation', 'boundary', 'build', 'clippy', 'arch')}
            needs_file = scratch / 'needs.json'; needs_file.write_text(json.dumps(needs))
            result_file = scratch / 'result.json'
            result_args = [sys.executable, str(repo / 'tools/public_ci.py'), 'result',
                           '--plan', str(plan_file), '--needs', str(needs_file),
                           '--execution', str(execution_file), '--out', str(result_file)]
            command(result_args)
            self.assertTrue(json.loads(result_file.read_text())['success'])
            execution['successful'] = []; execution_file.write_text(json.dumps(execution))
            command(result_args, ok=False)
            # A planted command identity cannot become arbitrary receipt execution.
            bad = json.loads(plan_file.read_text()); bad['cpu_contracts'][0]['cpu'] = ['echo', 'false pass']
            plan_file.write_text(json.dumps(bad))
            command([sys.executable, str(repo / 'tools/public_ci.py'), 'contracts',
                     '--plan', str(plan_file), '--repo', str(repo), '--out', str(execution_file)], ok=False)
            plan_file.write_text(json.dumps(selected))
            required_input = repo / 'tools/test_sft_gen_contract.py'
            input_bytes = required_input.read_bytes(); required_input.unlink()
            command([sys.executable, str(repo / 'tools/public_ci.py'), 'contracts',
                     '--plan', str(plan_file), '--repo', str(repo), '--out', str(execution_file)], ok=False)
            required_input.write_bytes(input_bytes)
            # Source freeze catches uncommitted execution-helper tampering before commands.
            helper = repo / 'tools/validation_plan.py'; original = helper.read_text()
            helper.write_text(original + '\n# planted stale source\n')
            command([sys.executable, str(repo / 'tools/public_ci.py'), 'contracts',
                     '--plan', str(plan_file), '--repo', str(repo), '--out', str(execution_file)], ok=False)
            helper.write_text(original)

            # Compiled consumer, native graph and unknown producer retain conservative coverage.
            native = commit_file('crates/memra-server/src/public_ci_fixture.rs', '// native fixture\n')
            native_plan = planned(routed('pull_request', event(restored, native)), native)
            self.assertTrue(native_plan['jobs']['arch'])
            self.assertTrue(native_plan['jobs']['server'])
            self.assertEqual(native_plan['native']['scope'], 'serving')
            include = commit_file('crates/memra-server/src/public_ci_fixture.rs',
                                  'const INPUT: &str = include_str!("../../../.github/workflows/ci.yml");\n')
            no_step = commit_file('.github/workflows/ci.yml', workflow.replace(block, '', 1))
            with_step = commit_file('.github/workflows/ci.yml', workflow)
            mixed = planned(routed('pull_request', event(no_step, with_step)), with_step)
            self.assertEqual(mixed['mode'], 'full')
            self.assertTrue(all(mixed['jobs'].values()))
            unknown = commit_file('tools/public_ci_unknown.py', '# unknown producer\n')
            self.assertEqual(planned(routed('pull_request', event(with_step, unknown)), unknown)['mode'], 'full')
            # Router bootstrap / source-policy edits do not authorize their own omissions.
            changed = commit_file('tools/public_ci.py', (repo / 'tools/public_ci.py').read_text() + '\n# change\n')
            self.assertEqual(planned(routed('pull_request', event(unknown, changed)), changed)['mode'], 'full')

            full = ci.full_cpu_plan(repo, changed, 'integration full inventory')
            for name in ('build', 'clippy', 'server', 'engine', 'portable', 'core', 'lanes', 'arch', 'publish'):
                self.assertTrue(full['jobs'][name])
            full_needs = {'route': {'result': 'success'}, 'full': {'result': 'success',
                           'outputs': {'validated_head': changed}}}
            self.assertTrue(ci.merge_result(full, full_needs)['success'])
            full_needs['full']['outputs']['validated_head'] = source
            with self.assertRaises(ci.Refused): ci.merge_result(full, full_needs)
            full_needs['full']['outputs']['validated_head'] = changed
            full_needs['full']['result'] = 'failure'
            with self.assertRaises(ci.Refused): ci.merge_result(full, full_needs)
            # Publication dependencies precede all effects; native/tag guards are still present.
            for name, consumer in [('release.yml', 'guard'), ('publish.yml', 'publish')]:
                value = (ROOT / '.github/workflows' / name).read_text()
                publication_dependency(value, consumer)
                self.assertIn('tools/release_qualification.py verify', value)
                self.assertIn('tools/release-guard.sh', value)
                self.assertNotIn('secrets: inherit', value)
                misplaced = value.replace('  ' + consumer + ':\n    needs: cpu-suite\n',
                                          '  ' + consumer + ':\n', 1)
                misplaced = misplaced.replace('on:\n', 'on:\n    needs: cpu-suite\n', 1)
                with self.assertRaises(AssertionError):
                    publication_dependency(misplaced, consumer)
            # Compiling caller-author guard mutant drives the real CLI. The same
            # expected external-mode assertion must fail, then source is restored.
            router = repo / 'tools/public_ci.py'; router_source = router.read_text()
            guard = "not owner(pull.get('user')) or "
            self.assertEqual(router_source.count(guard), 1)
            router.write_text(router_source.replace(guard, '', 1))
            try:
                wrong = routed('pull_request', event(source, head, author=external))
                with self.assertRaises(AssertionError): self.assertEqual(wrong['mode'], 'full')
            finally:
                router.write_text(router_source)
            print('public-ci integration: PASS: real14graph modes/source/commands/statuses; '
                  'SFT executed once; wrong routing/missing execution/stale source refused; qualification=false')


if __name__ == '__main__':
    unittest.main()
