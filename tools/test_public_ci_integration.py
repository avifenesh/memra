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

            def command(args, *, ok=True, cwd=repo, extra_env=None):
                result = subprocess.run(args, cwd=cwd, env=dict(env, **(extra_env or {})), text=True,
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
            # Materialize the exact existing support evidence for the live guard.
            # This is source transport, never model/runtime qualification.
            support_tree = vp.Tree(repo, source)
            support = vp.support_record_data_inputs(support_tree)
            present = set(support_tree.paths('research', 'docs'))
            for path in set(support['required']) | (set(support['optional']) & present):
                target = repo / path; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(support_tree.read_bytes(path))
            # The conflict checker reads tracked eligible source/doc/data paths,
            # excluding raw/receipt/log trees. Materialize exactly that census.
            conflict_paths = [path for path in support_tree.paths() if ci.conflict_input(path)]
            conflict_modes = ci.data_input_modes(support_tree, conflict_paths)
            missing_conflict = [path for path in conflict_paths if not (repo / path).exists()]
            for path, data in ci.pinned_data(repo, source, missing_conflict):
                target = repo / path
                target.parent.mkdir(parents=True, exist_ok=True)
                if conflict_modes[path] == '120000':
                    target.symlink_to(data.decode())
                else:
                    target.write_bytes(data)
                    target.chmod(0o755 if conflict_modes[path] == '100755' else 0o644)
            # Support evidence may have materialized an eligible executable
            # before this transport pass. Restore its exact Git executable mode.
            for path in conflict_paths:
                self.assertIn(conflict_modes[path], ('100644', '100755', '120000'))
                if conflict_modes[path] == '120000' and not (repo / path).is_symlink():
                    (repo / path).unlink()
                    (repo / path).symlink_to(support_tree.read_bytes(path).decode())
                elif conflict_modes[path] != '120000':
                    (repo / path).chmod(0o755 if conflict_modes[path] == '100755' else 0o644)
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

            # The published vulnerable caller actually executes a hostile PR
            # router/full-plan. The corrected caller uses isolated base code and
            # workflow-literal full selection, even from the candidate cwd.
            trusted = scratch / 'trusted'
            command(['/usr/bin/git', '-c', 'init.templateDir=' + str(templates),
                     'clone', '--shared', '--no-checkout', str(repo), str(trusted)], cwd=scratch)
            command(['/usr/bin/git', 'sparse-checkout', 'set', 'tools'], cwd=trusted)
            command(['/usr/bin/git', 'checkout', '-q', '--detach', source], cwd=trusted)
            router_path = repo / 'tools/public_ci.py'
            router_original = router_path.read_text()
            hostile = '''import pathlib, sys
a = sys.argv
out = pathlib.Path(a[a.index('--out') + 1])
if a[1] == 'route':
    out.write_text('{"mode":"thin","qualification":false}')
else:
    out.write_text('{"mode":"full","jobs":{"build":false,"clippy":false,"arch":false}}')
    output = pathlib.Path(a[a.index('--github-output') + 1])
    output.write_text('ci_mode=thin\\nbuild=false\\nclippy=false\\narch=false\\nserver=false\\nengine=false\\nportable=false\\ncore=false\\nlanes=false\\npublish=false\\nrequires_cuda=false\\ncontracts=none\\n')
'''
            malicious_head = commit_file('tools/public_ci.py', hostile)
            event_file = scratch / 'hostile-event.json'
            event_file.write_text(json.dumps(event(source, malicious_head, author=external)))
            old_route = scratch / 'old-route.json'
            command([sys.executable, 'tools/public_ci.py', 'route', '--event-name', 'pull_request',
                     '--event', str(event_file), '--out', str(old_route)])
            with self.assertRaises(AssertionError):
                self.assertEqual(json.loads(old_route.read_text())['mode'], 'full')
            old_workflow = command(['/usr/bin/git', 'show',
                '2ab8eec8ddd5b27ecaaf22f55ddf728151be560c:.github/workflows/ci.yml'])
            self.assertIn('tools/ci-change-class.sh full "$GITHUB_SHA"', old_workflow)
            old_outputs = scratch / 'old-outputs'
            command(['bash', 'tools/ci-change-class.sh', 'full', malicious_head,
                     str(scratch / 'old-plan.json'), str(old_outputs)])
            with self.assertRaises(AssertionError):
                self.assertNotIn('build=false', old_outputs.read_text())
            # Also tamper the producer, rather than relying on the router only.
            producer = repo / 'tools/ci-change-class.sh'; producer_original = producer.read_text()
            producer_head = commit_file('tools/ci-change-class.sh', '#!/bin/sh\necho hostile-producer-ran > "' + str(scratch / 'producer-marker') + '"\n')
            marker = scratch / 'python-marker'
            customization = 'from pathlib import Path\nPath(' + repr(str(marker)) + ').write_text("candidate Python import ran")\n'
            commit_file('sitecustomize.py', customization)
            hostile_head = commit_file('tools/json.py', customization + 'raise RuntimeError("candidate json module imported")\n')
            attack_env = {'PYTHONPATH': str(repo) + os.pathsep + str(repo / 'tools')}
            command([sys.executable, 'tools/public_ci.py', 'route', '--out', str(old_route)],
                    extra_env=attack_env)
            self.assertTrue(marker.exists())
            marker.unlink()
            refused = command([sys.executable, '-c', 'import json'], extra_env=attack_env, ok=False)
            self.assertIn('candidate json module imported', refused)
            self.assertTrue(marker.exists())
            marker.unlink()
            event_file.write_text(json.dumps(event(source, hostile_head, author=external)))
            out_dir = scratch / 'trusted-output'; out_dir.mkdir()
            trusted_args = [sys.executable, '-I', str(trusted / 'tools/trusted_public_ci.py'),
                            'route-plan', '--trusted-head', source, '--repo', str(repo),
                            '--event-name', 'pull_request', '--event', str(event_file),
                            '--head', hostile_head, '--out-dir', str(out_dir)]
            command(trusted_args, extra_env=attack_env)
            isolated_plan = json.loads((out_dir / 'validation-plan.json').read_text())
            self.assertEqual(isolated_plan['mode'], 'full')
            self.assertTrue(all(isolated_plan['jobs'].values()))
            self.assertEqual(isolated_plan['head'], hostile_head)
            self.assertFalse(marker.exists())
            self.assertFalse((scratch / 'producer-marker').exists())
            public_text = (ROOT / '.github/workflows/ci-public.yml').read_text()
            route_job = public_text.split('\n  route:\n', 1)[1].split('\n  full:\n', 1)[0]
            self.assertIn("github.event.pull_request.user.id == 55848801", route_job)
            self.assertIn("github.event.pull_request.head.repo.owner.id == 55848801", route_job)
            self.assertIn("steps.plan.outputs.ci_mode == 'thin' && 'thin' || 'full'", route_job)
            self.assertNotIn('python3 tools/public_ci.py', route_job)
            route_body = route_job.split('        run: |\n', 1)[1].split('      - uses: actions/upload-artifact', 1)[0]
            route_body = ''.join(line[10:] if line.startswith('          ') else line for line in route_body.splitlines(keepends=True))
            trusted_entry = trusted / 'tools/trusted_public_ci.py'
            entry_bytes = trusted_entry.read_bytes(); trusted_entry.unlink()
            bootstrap_outputs = scratch / 'bootstrap-outputs'
            try:
                command(['bash', '-e', '-c', route_body], cwd=scratch,
                        extra_env=dict(attack_env, OWNER_MODE='true', TRUSTED_CHECKOUT='success',
                                       TRUSTED_HEAD=source, EVENT_NAME='pull_request',
                                       GITHUB_SHA=hostile_head, GITHUB_WORKSPACE=str(scratch),
                                       GITHUB_EVENT_PATH=str(event_file), GITHUB_OUTPUT=str(bootstrap_outputs)))
                self.assertIn('ci_mode=full', bootstrap_outputs.read_text())
                self.assertEqual(json.loads((scratch / 'validation-plan.json').read_text())['mode'], 'full')
                self.assertFalse(marker.exists())
            finally:
                trusted_entry.write_bytes(entry_bytes)
            # Run the actual new full selection body with BOTH hostile helpers
            # and candidate import traps present. Its outputs are workflow literals.
            full_text = (ROOT / '.github/workflows/ci.yml').read_text()
            changes = full_text.split('\n  changes:\n', 1)[1].split('\n  gates:\n', 1)[0]
            literal_outputs = changes.split('    outputs:\n', 1)[1].split('    steps:\n', 1)[0]
            for name in ('build', 'clippy', 'server', 'engine', 'portable', 'core', 'lanes', 'arch', 'publish', 'requires_cuda'):
                self.assertIn('      ' + name + ': "true"\n', literal_outputs)
            body = changes.split('        run: |\n', 1)[1].split('      - name: Retain', 1)[0]
            body = ''.join(line[10:] if line.startswith('          ') else line for line in body.splitlines(keepends=True))
            command(['bash', '-e', '-c', body], extra_env=dict(attack_env, GITHUB_SHA=hostile_head))
            literal_plan = json.loads((repo / 'validation-plan.json').read_text())
            self.assertTrue(all(literal_plan['jobs'].values()))
            self.assertEqual(literal_plan['head'], hostile_head)
            self.assertFalse(marker.exists())
            self.assertFalse((scratch / 'producer-marker').exists())
            (repo / 'validation-plan.json').unlink()
            complete_body = full_text.split('\n  complete:\n', 1)[1].split('        run: |\n', 1)[1]
            complete_body = ''.join(line[10:] if line.startswith('          ') else line for line in complete_body.splitlines(keepends=True))
            full_names = ('changes', 'gates', 'boundary', 'build', 'clippy', 'server-tests', 'portable-suites', 'engine-tests', 'arch-coverage', 'publish-dryrun')
            full_results = {name: {'result': 'success'} for name in full_names}
            complete_output = scratch / 'complete-output'
            complete_env = dict(attack_env, GITHUB_SHA=hostile_head, GITHUB_OUTPUT=str(complete_output),
                                JOB_RESULTS=json.dumps(full_results))
            command(['bash', '-e', '-c', complete_body], extra_env=complete_env)
            self.assertIn('validated_head=' + hostile_head, complete_output.read_text())
            full_results['build']['result'] = 'skipped'
            complete_env['JOB_RESULTS'] = json.dumps(full_results)
            self.assertIn('Complete CPU inventory did not succeed',
                          command(['bash', '-e', '-c', complete_body], extra_env=complete_env, ok=False))
            self.assertFalse(marker.exists())
            # A physically changed immutable loader dependency cannot import.
            trusted_router = trusted / 'tools/public_ci.py'; trusted_original = trusted_router.read_text()
            trusted_router.write_text(trusted_original + '\n# changed trusted bytes\n')
            refused = command(trusted_args, extra_env=attack_env, ok=False)
            self.assertIn('trusted input bytes differ: public_ci.py', refused)
            trusted_router.write_text(trusted_original)
            wrong_root = list(trusted_args); wrong_root[wrong_root.index('--repo') + 1] = str(trusted)
            self.assertIn('separate siblings', command(wrong_root, ok=False))
            # Restore the actual candidate tree before the remaining legacy
            # graph/caller controls. No override of HOME or runtime identity.
            commit_file('tools/public_ci.py', router_original)
            commit_file('tools/ci-change-class.sh', producer_original)
            command(['/usr/bin/git', 'rm', '--sparse', 'sitecustomize.py', 'tools/json.py'])
            command(['/usr/bin/git', 'commit', '-qm', 'restore hostile fixture'])
            restored_head = command(['/usr/bin/git', 'rev-parse', 'HEAD']).strip()
            event_file.write_text(json.dumps(event(source, restored_head)))
            command(trusted_args[:-4] + ['--head', restored_head, '--out-dir', str(out_dir)], extra_env=attack_env)
            self.assertEqual(json.loads((out_dir / 'validation-plan.json').read_text())['ci_mode'], 'thin')
            self.assertFalse(marker.exists())

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
            # Git executable mode follows the owner bit, not group/other bits.
            executable = repo / 'tools/check-action-pins.sh'
            original_mode = executable.stat().st_mode & 0o777
            for mode in (0o700, 0o744):
                executable.chmod(mode)
                self.assertEqual(vp.LocalTree(repo).input_modes('tools/check-action-pins.sh', recursive=False)
                                 ['tools/check-action-pins.sh'], '100755')
                ci.pin_data_inputs(repo, restored, ['tools/check-action-pins.sh'])
            executable.chmod(0o654)
            with self.assertRaises(ci.Refused):
                ci.pin_data_inputs(repo, restored, ['tools/check-action-pins.sh'])
            executable.chmod(original_mode)
            regular = repo / 'docs/ROUTER.md'; regular_mode = regular.stat().st_mode & 0o777
            regular.chmod(0o641)
            self.assertEqual(vp.LocalTree(repo).input_modes('docs/ROUTER.md', recursive=False)
                             ['docs/ROUTER.md'], '100644')
            ci.pin_data_inputs(repo, restored, ['docs/ROUTER.md'])
            # Coherent all-bits mutation misclassifies this actual Git-clean file.
            with self.assertRaises(AssertionError):
                self.assertEqual('100755' if regular.stat().st_mode & 0o111 else '100644', '100644')
            regular.chmod(regular_mode)
            # Actual tracked contained aliases are part of the conflict census.
            # A changed target string must refuse without following it.
            alias = repo / 'research/qwen4exp-bringup-20260829/round2-box-receipts/expand-goldens.py'
            self.assertTrue(alias.is_symlink())
            alias_target = os.readlink(alias)
            self.assertEqual(alias_target, 'bin/expand-goldens.py')
            alias.unlink(); alias.symlink_to('bin/make-ladder-ids.py')
            try:
                refused = command([sys.executable, str(repo / 'tools/public_ci.py'), 'contracts',
                                   '--plan', str(plan_file), '--repo', str(repo), '--out', str(execution_file)], ok=False)
                self.assertIn('pinned contained alias differs', refused)
            finally:
                alias.unlink(); alias.symlink_to(alias_target)
            # A harmless physical doc change cannot stamp HEAD merely because
            # its static census still passes. Exact data binding must refuse.
            registry_doc = repo / 'docs/ROUTER.md'; registry_original = registry_doc.read_bytes()
            registry_doc.write_bytes(registry_original + b'\n')
            try:
                refused = command([sys.executable, str(repo / 'tools/public_ci.py'), 'contracts',
                                   '--plan', str(plan_file), '--repo', str(repo), '--out', str(execution_file)], ok=False)
                self.assertRegex(refused, r'merge guard data (?:length )?differs from pinned source: docs/ROUTER\.md')
            finally:
                registry_doc.write_bytes(registry_original)
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
            self.assertIn('coalescer', ci.guard_ids(native_plan))
            # Real production CUDA path retains thin routing and the PDL guard.
            # No nvcc/native execution: the existing CPU checker sees an actual
            # first-statement omission in the source it ordinarily censuses.
            cuda_path = 'crates/memra-engine/cu/dsv4_sampler.cu'
            cuda_original = (repo / cuda_path).read_text()
            cuda_green = commit_file(cuda_path, cuda_original + '\n// owned guard fixture\n')
            cuda_plan = planned(routed('pull_request', event(native, cuda_green)), cuda_green)
            self.assertEqual(cuda_plan.get('ci_mode'), 'thin', cuda_plan['reason'])
            self.assertEqual(cuda_plan['native']['scope'], 'full')
            self.assertIn('pdl-chain', ci.guard_ids(cuda_plan))
            plan_file.write_text(json.dumps(cuda_plan))
            output = command([sys.executable, str(repo / 'tools/public_ci.py'), 'contracts',
                              '--plan', str(plan_file), '--repo', str(repo), '--out', str(execution_file)])
            self.assertIn('pdl-chain:', output)
            cuda_execution = json.loads(execution_file.read_text())
            self.assertIn('pdl-chain', cuda_execution['merge_guards'])
            self.assertIn('coalescer', cuda_execution['merge_guards'])
            # The source census cannot pass against a missing pinned CUDA input.
            required_cuda = repo / 'crates/memra-engine/cu/memra_pdl_chain.cuh'
            required_bytes = required_cuda.read_bytes(); required_cuda.unlink()
            try:
                refused = command([sys.executable, str(repo / 'tools/public_ci.py'), 'contracts',
                                   '--plan', str(plan_file), '--repo', str(repo), '--out', str(execution_file)], ok=False)
                self.assertTrue('PDL input census is missing' in refused or
                                "No such file or directory: 'memra_pdl_chain.cuh'" in refused,
                                refused)
            finally:
                required_cuda.write_bytes(required_bytes)
            needs_file.write_text(json.dumps(needs))
            command(result_args)
            # An otherwise complete receipt with the PDL execution omitted must
            # refuse, reproducing the old seven-guard omission at the result seam.
            cuda_execution['merge_guards'].remove('pdl-chain')
            execution_file.write_text(json.dumps(cuda_execution))
            refused = command(result_args, ok=False)
            self.assertIn('selected contract execution is missing or incomplete', refused)
            cuda_bad = commit_file(cuda_path, cuda_original.replace('    MEMRA_PDL_CHAIN_ENTRY();\n', '', 1))
            bad_plan = planned(routed('pull_request', event(cuda_green, cuda_bad)), cuda_bad)
            self.assertEqual(bad_plan.get('ci_mode'), 'thin', bad_plan['reason'])
            self.assertEqual(bad_plan['native']['scope'], 'full')
            plan_file.write_text(json.dumps(bad_plan))
            refused = command([sys.executable, str(repo / 'tools/public_ci.py'), 'contracts',
                               '--plan', str(plan_file), '--repo', str(repo), '--out', str(execution_file)], ok=False)
            self.assertIn('dsv4_sample_prepare does not open with MEMRA_PDL_CHAIN_ENTRY()', refused)
            cuda_restored = commit_file(cuda_path, cuda_original)
            header_path = 'crates/memra-engine/cu/dsv4_replay_control.cuh'
            header = commit_file(header_path, (repo / header_path).read_text() + '\n// header guard fixture\n')
            header_plan = planned(routed('pull_request', event(cuda_restored, header)), header)
            self.assertEqual(header_plan.get('ci_mode'), 'thin', header_plan['reason'])
            self.assertEqual(header_plan['native']['scope'], 'full')
            self.assertIn('pdl-chain', ci.guard_ids(header_plan))
            # --list is weaker than coverage: an actual undocumented read in an
            # owned Rust source must fail the selected live coverage guard.
            runtime_path = 'crates/memra-server/src/public_ci_fixture.rs'
            runtime = commit_file(runtime_path, 'fn fixture() { let _ = std::env::var("MEMRA_PUBLIC_CI_UNDOCUMENTED_FIXTURE"); }\n')
            runtime_plan = planned(routed('pull_request', event(header, runtime)), runtime)
            self.assertEqual(runtime_plan.get('ci_mode'), 'thin', runtime_plan['reason'])
            self.assertIn('flags-coverage', ci.guard_ids(runtime_plan))
            plan_file.write_text(json.dumps(runtime_plan))
            refused = command([sys.executable, str(repo / 'tools/public_ci.py'), 'contracts',
                               '--plan', str(plan_file), '--repo', str(repo), '--out', str(execution_file)], ok=False)
            self.assertIn('MEMRA_PUBLIC_CI_UNDOCUMENTED_FIXTURE', refused)
            self.assertIn('UNCOVERED runtime names', refused)
            runtime_restored = commit_file(runtime_path, '// native fixture\n')
            # Gate OFF-arm and live support-state checks are merge guards, not
            # substituted by their refusal fixtures or a list-only census.
            off_path = 'crates/memra-server/src/bin/public_ci_off_fixture.rs'
            off = commit_file(off_path, 'fn main() { std::env::remove_var("MEMRA_PUBLIC_CI_OFF_FIXTURE"); }\n')
            off_plan = planned(routed('pull_request', event(runtime_restored, off)), off)
            self.assertEqual(off_plan.get('ci_mode'), 'thin', off_plan['reason'])
            self.assertIn('gate-off-arms', ci.guard_ids(off_plan))
            plan_file.write_text(json.dumps(off_plan))
            refused = command([sys.executable, str(repo / 'tools/public_ci.py'), 'contracts',
                               '--plan', str(plan_file), '--repo', str(repo), '--out', str(execution_file)], ok=False)
            self.assertIn('VACUOUS GATE HAZARD', refused)
            off_restored = commit_file(off_path, 'fn main() {}\n')
            readme = (repo / 'README.md').read_text()
            support_bad = commit_file('README.md', readme + '\nNativeQualified\n')
            support_plan = planned(routed('pull_request', event(off_restored, support_bad)), support_bad)
            self.assertEqual(support_plan.get('ci_mode'), 'thin', support_plan['reason'])
            self.assertIn('support-state', ci.guard_ids(support_plan))
            plan_file.write_text(json.dumps(support_plan))
            refused = command([sys.executable, str(repo / 'tools/public_ci.py'), 'contracts',
                               '--plan', str(plan_file), '--repo', str(repo), '--out', str(execution_file)], ok=False)
            self.assertIn('check-support-states: FAIL', refused)
            support_restored = commit_file('README.md', readme)
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
