#!/usr/bin/env python3
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import validation_plan as vp


class ValidationPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='memra-validation-plan-')
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.g('init', '-q')
        self.g('config', 'user.email', 'test@example.invalid')
        self.g('config', 'user.name', 'Fixture')
        self.g('config', 'commit.gpgsign', 'false')
        self.g('config', 'core.hooksPath', '/dev/null')
        self.graph = {
            'memra-gguf': [], 'memra-tokenizer': ['memra-gguf'],
            'memra-reference': ['memra-gguf', 'memra-tokenizer'],
            'memra-cli': ['memra-reference', 'memra-gguf', 'memra-tokenizer'],
            'memra-tier': [], 'memra-kv': ['memra-tier', 'memra-gguf'],
            'memra-sampling': [], 'memra-validate': [], 'memra-runtime': ['memra-gguf'],
            'memra-lanes': [], 'memra-probe': [],
            'memra-engine': ['memra-gguf', 'memra-reference', 'memra-kv', 'memra-sampling', 'memra-runtime', 'memra-validate'],
            'memra-server': ['memra-engine', 'memra-tokenizer', 'memra-lanes'],
        }
        self.put('Cargo.toml', '[workspace]\nmembers = ' + json.dumps(['crates/' + n for n in self.graph]) + '\n')
        for name, dependencies in self.graph.items():
            data = '[package]\nname = ' + json.dumps(name) + '\n[dependencies]\n'
            data += ''.join(f'{d} = {{path = "../{d}"}}\n' for d in dependencies)
            self.put(f'crates/{name}/Cargo.toml', data)
            self.put(f'crates/{name}/src/lib.rs', '// fixture\n')
        self.put('README.md', 'docs\n')
        self.base = self.commit()

    def g(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args], stderr=subprocess.PIPE).decode().strip()

    def put(self, name, content):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def commit(self):
        self.g('add', '-A'); self.g('commit', '-qm', 'fixture')
        return self.g('rev-parse', 'HEAD')

    def plan(self, paths):
        tree = vp.Tree(self.repo, 'HEAD')
        return vp.make_plan(paths, tree, tree)

    def test_server_change_does_not_run_dependency_unit_suites(self):
        p = self.plan(['crates/memra-server/src/responses_api.rs'])
        self.assertEqual(p['packages'], ['memra-server'])
        self.assertTrue(p['jobs']['server'])
        self.assertFalse(p['jobs']['engine'])
        self.assertFalse(p['jobs']['portable'])
        self.assertFalse(p['jobs']['core'])
        self.assertTrue(p['jobs']['arch'])  # server has architecture-conditioned callers
        self.assertEqual(p['native']['scope'], 'serving')
        self.assertFalse(p['native']['qualification'])

    def test_cli_change_has_no_cuda_build(self):
        p = self.plan(['crates/memra-cli/src/lib.rs'])
        self.assertEqual(p['packages'], ['memra-cli'])
        self.assertTrue(p['jobs']['portable'])
        self.assertFalse(p['requires_cuda'])
        self.assertFalse(p['jobs']['server'])

    def test_engine_change_reaches_server_not_unmodified_core_tests(self):
        p = self.plan(['crates/memra-engine/src/decode.rs'])
        self.assertEqual(p['packages'], ['memra-engine', 'memra-server'])
        self.assertTrue(p['jobs']['engine'])
        self.assertTrue(p['jobs']['server'])
        self.assertFalse(p['jobs']['core'])

    def test_upstream_change_reaches_transitive_dependents(self):
        p = self.plan(['crates/memra-gguf/src/config.rs'])
        self.assertTrue({'memra-cli', 'memra-reference', 'memra-engine', 'memra-server', 'memra-tokenizer', 'memra-kv'} <= set(p['packages']))
        self.assertTrue(p['jobs']['core'])
        self.assertTrue(p['jobs']['portable'])

    def test_docs_only_has_no_compilation_or_native_claim(self):
        p = self.plan(['README.md', 'research/lane/RESULT.md'])
        self.assertEqual(p['mode'], 'scoped')
        self.assertFalse(any(p['jobs'].values()))
        self.assertEqual(p['native']['scope'], 'none')

    def test_flags_is_a_real_build_input_not_just_documentation(self):
        p = self.plan(['docs/FLAGS.md'])
        self.assertEqual(p['packages'], ['memra-engine', 'memra-server'])
        self.assertTrue(p['jobs']['build'])

    def test_declared_python_contract_avoids_native_compilation(self):
        p = self.plan(['tools/q35-cold-mixed-gate.py'])
        self.assertFalse(any(p['jobs'].values()))
        self.assertEqual(p['cpu_contracts'][0]['id'], 'q35-cache')
        self.assertEqual(p['native']['scope'], 'harness')
        self.assertTrue(p['native']['requirements'])

    def test_metrics_collectors_select_separate_floors_and_keep_native_obligations(self):
        for name in ('cache-meter', 'metrics-live'):
            for path in vp.TOOL_CONTRACTS[name]['inputs']:
                self.put(path, '# fixture\n')
        self.commit()
        for name, path, floor in [('cache-meter', 'tools/cache-meter-gate.py', '3'),
                                  ('metrics-live', 'tools/metrics-live-gate.py', '7')]:
            with self.subTest(name=name):
                plan = self.plan([path])
                self.assertEqual([c['id'] for c in plan['cpu_contracts']], [name])
                self.assertEqual(plan['cpu_contracts'][0]['cpu'][-1], floor)
                self.assertFalse(any(plan['jobs'].values()))
                self.assertTrue(plan['native']['requirements'])
                self.assertFalse(plan['native']['qualification'])

    def test_background_collector_selects_its_cpu_floor_and_native_obligations(self):
        contract = vp.TOOL_CONTRACTS['background-chat-text']
        for path in contract['inputs']:
            self.put(path, '# fixture\n')
        self.commit()
        for path in contract['inputs']:
            with self.subTest(path=path):
                plan = self.plan([path])
                self.assertEqual([c['id'] for c in plan['cpu_contracts']], ['background-chat-text'])
                self.assertEqual(plan['cpu_contracts'][0]['cpu'][-1], '9')
                self.assertTrue(plan['native']['requirements'])
                self.assertFalse(plan['native']['qualification'])

    def test_background_contract_refuses_partial_or_deleted_inputs(self):
        contract = vp.TOOL_CONTRACTS['background-chat-text']
        for path in contract['inputs']:
            self.put(path, '# fixture\n')
        for path in contract['inputs']:
            with self.subTest(missing=path):
                (self.repo / path).unlink()
                with self.assertRaisesRegex(vp.Refused, 'background-chat-text'):
                    vp.cpu_contract_names(self.repo, 'background-chat-text')
                self.put(path, '# fixture\n')

    def test_shared_metrics_parser_selects_both_collectors(self):
        for name in ('cache-meter', 'metrics-live'):
            for path in vp.TOOL_CONTRACTS[name]['inputs']:
                self.put(path, '# fixture\n')
        self.commit()
        plan = self.plan(['tools/prometheus_metrics.py'])
        self.assertEqual({c['id'] for c in plan['cpu_contracts']}, {'cache-meter', 'metrics-live'})
        self.assertFalse(any(plan['jobs'].values()))
        self.assertEqual(len(plan['native']['requirements']), 2)

    def test_shared_metrics_parser_does_not_invent_an_absent_collector(self):
        for path in vp.TOOL_CONTRACTS['cache-meter']['inputs']:
            self.put(path, '# fixture\n')
        self.commit()
        plan = self.plan(['tools/prometheus_metrics.py'])
        self.assertEqual([c['id'] for c in plan['cpu_contracts']], ['cache-meter'])
        self.assertFalse(any(plan['jobs'].values()))

    def test_partial_metrics_contract_cannot_silently_skip_its_test_or_parser(self):
        contract = vp.TOOL_CONTRACTS['metrics-live']
        for path in contract['inputs']:
            self.put(path, '# fixture\n')
        self.assertEqual(vp.cpu_contract_names(self.repo, 'metrics-live'), ['metrics-live'])
        for path in ('tools/test_prometheus_metrics.py', 'tools/prometheus_metrics.py'):
            with self.subTest(missing=path):
                (self.repo / path).unlink()
                with self.assertRaisesRegex(vp.Refused, 'metrics-live'):
                    vp.cpu_contract_names(self.repo, 'metrics-live')
                self.put(path, '# fixture\n')

    def test_shared_collector_selects_sampled_tests_only_when_present(self):
        paths = ['tools/collect-serving-qualification.py']
        self.assertEqual([c['id'] for c in self.plan(paths)['cpu_contracts']], ['serving-qualification'])
        self.put('tools/collect-sampled-mtp.py', '# imports serving collector\n')
        self.commit()
        p = self.plan(paths)
        self.assertEqual({c['id'] for c in p['cpu_contracts']}, {'serving-qualification', 'sampled-mtp'})
        self.assertFalse(any(p['jobs'].values()))

    def test_sampled_dependency_pin_selects_its_contract(self):
        self.put('tools/sampled-mtp-requirements.txt', 'numpy==2.3.5\n')
        self.commit()
        p = self.plan(['tools/sampled-mtp-requirements.txt'])
        self.assertEqual([c['id'] for c in p['cpu_contracts']], ['sampled-mtp'])
        self.assertFalse(any(p['jobs'].values()))

    def test_network_guard_selects_core_and_its_server_consumer(self):
        manifest = (self.repo / 'Cargo.toml').read_text().replace('members = [', 'members = ["crates/memra-net-guard", ')
        self.put('Cargo.toml', manifest)
        self.put('crates/memra-net-guard/Cargo.toml', '[package]\nname="memra-net-guard"\n')
        server = 'crates/memra-server/Cargo.toml'
        self.put(server, (self.repo / server).read_text() + 'memra-net-guard = {path="../memra-net-guard"}\n')
        self.commit()
        p = self.plan(['crates/memra-net-guard/src/lib.rs'])
        self.assertEqual(p['packages'], ['memra-net-guard', 'memra-server'])
        self.assertTrue(p['jobs']['core'])
        self.assertTrue(p['jobs']['server'])
        self.assertFalse(p['jobs']['engine'])

    def test_python_contract_environment_is_private_and_cleaned(self):
        contract = vp.TOOL_CONTRACTS['sampled-mtp']
        with mock.patch.object(vp.subprocess, 'run') as run:
            vp.run_cpu_contract(contract, self.repo)
        setup, install, check = run.call_args_list
        directory = Path(setup.args[0][-1])
        self.assertEqual(setup.args[0][1:3], ['-m', 'venv'])
        self.assertEqual(install.args[0][-2:], ['-r', str(self.repo / contract['python_requirements'])])
        self.assertEqual(check.args[0], contract['cpu'])
        self.assertEqual(check.kwargs['env']['PATH'].split(os.pathsep)[0], str(directory / 'bin'))
        self.assertEqual(check.kwargs['env']['OPENBLAS_NUM_THREADS'], '1')
        self.assertEqual(check.kwargs['env']['OMP_NUM_THREADS'], '1')
        self.assertFalse(directory.exists())

    def test_dependency_failure_cannot_pass_or_leave_environment(self):
        with mock.patch.object(vp.subprocess, 'run', side_effect=[None, subprocess.CalledProcessError(1, 'pip')]) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                vp.run_cpu_contract(vp.TOOL_CONTRACTS['sampled-mtp'], self.repo)
        self.assertEqual(run.call_count, 2)
        self.assertFalse(Path(run.call_args_list[0].args[0][-1]).exists())

    def test_unknown_tool_or_source_expands(self):
        for path in ('tools/new-builder.py', 'unknown.rs'):
            with self.subTest(path=path):
                self.assertEqual(self.plan([path])['mode'], 'full')

    def test_unreferenced_research_receipt_is_not_an_engine_build(self):
        p = self.plan(['research/new-campaign/receipts/output.json'])
        self.assertFalse(any(p['jobs'].values()))

    def test_native_probe_registry_input_keeps_native_obligation(self):
        self.put('tools/fast-gate/models.tsv', 'g12\targmax\t/model\tresearch/prompt-ids.txt\t20\t-\n')
        self.put('tools/fast-gate/accept-cells.tsv', 'accept\tresearch/e2e/prompts/*.txt\n')
        self.commit()
        for path in ('research/prompt-ids.txt', 'research/e2e/prompts/short.txt'):
            with self.subTest(path=path):
                p = self.plan([path])
                self.assertFalse(any(p['jobs'].values()))
                self.assertEqual(p['native']['scope'], 'harness')
                self.assertIn(path, ' '.join(p['native']['requirements']))

    def test_runtime_research_fixture_selects_its_reader(self):
        self.put('crates/memra-server/src/lib.rs', 'let x = std::fs::read_to_string("research/fixture.json");')
        self.commit()
        self.assertEqual(self.plan(['research/fixture.json'])['packages'], ['memra-server'])

    def test_directory_fixture_join_selects_descendant_reader(self):
        self.put('crates/memra-server/src/lib.rs', 'let p = root.join("../../research/sample").join("input.json");')
        self.commit()
        self.assertEqual(self.plan(['research/sample/input.json'])['packages'], ['memra-server'])

    def test_literal_glob_characters_cannot_hide_input_or_directory(self):
        for source, path in (
                ('include_str!("../../../research/input[1].txt");', 'research/input[1].txt'),
                ('let p = root.join("research/fixtures[1]");', 'research/fixtures[1]/input.json'),
                ('let p = format!("research/fixtures[1]/{case}/input.json");', 'research/fixtures[1]/a/input.json')):
            with self.subTest(path=path):
                self.put('crates/memra-server/src/lib.rs', source)
                self.commit()
                self.assertEqual(self.plan([path])['packages'], ['memra-server'])

    def test_literal_directory_glob_characters_do_not_hide_symlink(self):
        self.put('crates/memra-server/src/lib.rs', 'let p = root.join("research/fixtures[1]");')
        self.put('docs/real.md', 'fixture')
        (self.repo / 'research/fixtures[1]').mkdir(parents=True)
        (self.repo / 'research/fixtures[1]/alias.md').symlink_to('../../docs/real.md')
        self.commit()
        self.assertEqual(self.plan(['docs/real.md'])['mode'], 'full')

    def test_runtime_pattern_keeps_its_glob_interpretation(self):
        self.put('crates/memra-server/src/lib.rs', 'let pattern = "research/fixtures/*.json"; let dir = Path::new(pattern).parent().unwrap(); std::fs::read_dir(dir);')
        self.commit()
        self.assertEqual(self.plan(['research/fixtures/input.json'])['packages'], ['memra-server'])

    def test_dynamic_research_directory_keeps_all_receipts_affected(self):
        self.put('crates/memra-server/src/lib.rs', 'let p = root.join("research");')
        self.commit()
        self.assertEqual(self.plan(['research/new-campaign/result.json'])['packages'], ['memra-server'])

    def test_standalone_research_probe_keeps_its_own_native_obligation(self):
        p = self.plan(['research/new-campaign/probe.cu'])
        self.assertFalse(any(p['jobs'].values()))
        self.assertTrue(p['native']['requirements'])

    def test_build_inputs_expand(self):
        for path in ('Cargo.lock', '.cargo/config.toml', 'crates/memra-server/Cargo.toml',
                     'crates/memra-engine/build.rs', '.github/workflows/ci.yml'):
            with self.subTest(path=path):
                self.assertTrue(all(self.plan([path])['jobs'].values()))

    def test_multiline_and_raw_literal_fixtures_reach_consumer(self):
        self.put('crates/memra-server/src/lib.rs', 'include_str!(\n "../../../research/fixture.md"\n);\ninclude_bytes!(r#"../../../research/raw.bin"#);')
        self.commit()
        for path in ('research/fixture.md', 'research/raw.bin'):
            p = self.plan([path])
            self.assertEqual(p['packages'], ['memra-server'])

    def test_manifest_concat_fixture_reaches_consumer(self):
        self.put('crates/memra-server/src/lib.rs', 'include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../research/fixture.md"));')
        self.commit()
        self.assertEqual(self.plan(['research/fixture.md'])['packages'], ['memra-server'])

    def test_multiliteral_concat_is_a_real_fixture_dependency(self):
        self.put('crates/memra-server/src/lib.rs', 'include_str!(concat!("../../../research/", "fixture.md"));')
        self.commit()
        self.assertEqual(self.plan(['research/fixture.md'])['packages'], ['memra-server'])

    def test_dynamic_absolute_and_escaped_include_paths_expand(self):
        for expression in ['env!("UNREGISTERED_PATH")', '"/absolute/data.md"', r'"../../../research/fixt\x75re.md"']:
            self.put('crates/memra-server/src/lib.rs', f'include_str!({expression});')
            self.commit()
            self.assertEqual(self.plan(['research/fixture.md'])['mode'], 'full')

    def test_includes_in_comments_and_source_strings_are_not_dependencies(self):
        self.put('crates/memra-server/src/lib.rs', 'let fixture = r#"include_str!(env!("NOT_CODE"))"#;\n// include_str!(env!("ALSO_NOT_CODE"))\n')
        self.commit()
        self.assertFalse(any(self.plan(['README.md'])['jobs'].values()))

    def test_cross_crate_source_include_reaches_the_reader(self):
        self.put('crates/memra-cli/src/lib.rs', 'include_str!("../../memra-server/src/lib.rs");')
        self.commit()
        self.assertEqual(self.plan(['crates/memra-server/src/lib.rs'])['packages'], ['memra-cli', 'memra-server'])

    def test_unregistered_build_script_cannot_hide_external_inputs(self):
        self.put('crates/memra-server/build.rs', 'fn main() { std::fs::read_to_string("../../README.md").unwrap(); }')
        self.commit()
        self.assertEqual(self.plan(['README.md'])['mode'], 'full')

    def test_publish_includes_selected_packages_forward_dependencies(self):
        args = vp.publish_packages('memra-cli', self.repo)
        names = set(args[1::2])
        self.assertEqual(names, {'memra-cli', 'memra-reference', 'memra-tokenizer', 'memra-gguf'})

    def test_local_untracked_source_is_not_an_empty_diff(self):
        self.put('crates/memra-server/src/new.rs', 'pub fn new() {}')
        p = vp.local_plan(self.repo, self.base)
        self.assertEqual(p['packages'], ['memra-server'])

    def test_local_index_masked_source_expands(self):
        self.g('update-index', '--assume-unchanged', 'crates/memra-server/src/lib.rs')
        self.put('README.md', 'changed')
        self.assertEqual(vp.local_plan(self.repo, self.base)['mode'], 'full')

    def test_binary_dependency_edge_has_a_reason_for_both_arms(self):
        p = self.plan(['crates/memra-server/src/lib.rs'])
        edge = next(e for e in p['edge_decisions'] if e['from'] == 'memra-engine' and e['to'] == 'memra-server')
        self.assertFalse(edge['affected'])
        self.assertIn('unchanged declared inputs', edge['reason'])
        p = self.plan(['crates/memra-engine/src/lib.rs'])
        edge = next(e for e in p['edge_decisions'] if e['from'] == 'memra-engine' and e['to'] == 'memra-server')
        self.assertTrue(edge['affected'])

    def test_old_tree_include_cannot_disappear_from_plan(self):
        self.put('crates/memra-server/src/lib.rs', 'include_str!("../../../research/old.md");')
        old = self.commit()
        self.put('crates/memra-server/src/lib.rs', '// removed include\n')
        new = self.commit()
        p = vp.make_plan(['research/old.md'], vp.Tree(self.repo, old), vp.Tree(self.repo, new))
        self.assertEqual(p['packages'], ['memra-server'])

    def test_unreadable_manifest_expands(self):
        self.put('crates/memra-cli/Cargo.toml', 'not toml')
        self.commit()
        self.assertEqual(self.plan(['README.md'])['mode'], 'full')

    def test_new_dependency_expands(self):
        self.put('crates/memra-server/Cargo.toml', '[package]\nname="memra-server"\n[dependencies]\nmemra-cli={path="../memra-cli"}\n')
        new = self.commit()
        p = vp.make_plan(['crates/memra-server/Cargo.toml'], vp.Tree(self.repo, self.base), vp.Tree(self.repo, new))
        self.assertEqual(p['mode'], 'full')

    def test_renamed_workspace_dependency_resolves_package_identity(self):
        self.put('crates/memra-server/Cargo.toml', '[package]\nname="memra-server"\n[dependencies]\nengine_alias={package="memra-engine",path="../memra-engine"}\n')
        self.commit()
        self.assertIn('memra-server', self.plan(['crates/memra-engine/src/lib.rs'])['packages'])

    def test_dev_dependency_is_not_ignored(self):
        self.put('crates/memra-server/Cargo.toml', '[package]\nname="memra-server"\n[dev-dependencies]\nmemra-cli={path="../memra-cli"}\n')
        self.commit()
        self.assertIn('memra-server', self.plan(['crates/memra-cli/src/lib.rs'])['packages'])

    def test_target_dependency_is_not_ignored(self):
        self.put('crates/memra-server/Cargo.toml', '[package]\nname="memra-server"\n[target.\'cfg(unix)\'.dependencies]\nmemra-cli={path="../memra-cli"}\n')
        self.commit()
        self.assertIn('memra-server', self.plan(['crates/memra-cli/src/lib.rs'])['packages'])

    def test_invalid_event_refs_and_empty_diff_expand(self):
        for event, base, before, head in [('workflow_dispatch', '', '', 'HEAD'),
                                          ('push', '', '0' * 40, 'HEAD'),
                                          ('pull_request', 'bad-ref', '', 'HEAD'),
                                          ('push', '', 'HEAD', 'HEAD')]:
            self.assertEqual(vp.event_plan(self.repo, event, base, before, head)['mode'], 'full')

    def test_actual_pr_diff_uses_merge_base(self):
        self.g('checkout', '-qb', 'topic')
        self.put('crates/memra-server/src/lib.rs', '// topic\n')
        topic = self.commit()
        self.g('checkout', '-q', self.base)
        self.put('crates/memra-engine/src/lib.rs', '// base advanced\n')
        base = self.commit()
        p = vp.event_plan(self.repo, 'pull_request', base, '', topic)
        self.assertEqual(p['packages'], ['memra-server'])

    def test_ci_merge_tree_uses_dependencies_introduced_on_base(self):
        self.g('checkout', '-qb', 'topic')
        self.put('crates/memra-server/src/lib.rs', '// topic\n')
        topic = self.commit()
        self.g('checkout', '-q', self.base)
        self.put('crates/memra-cli/Cargo.toml', '[package]\nname="memra-cli"\n[dependencies]\nmemra-server={path="../memra-server"}\n')
        base = self.commit()
        self.g('checkout', '-q', 'topic')
        self.g('merge', '--no-edit', base)
        candidate = self.g('rev-parse', 'HEAD')
        p = vp.event_plan(self.repo, 'pull_request', base, '', candidate)
        self.assertEqual(p['packages'], ['memra-cli', 'memra-server'])
        self.assertEqual(p['head'], candidate)
        self.assertTrue(p['jobs']['portable'])

    def test_adding_changed_inputs_never_removes_required_jobs(self):
        samples = ['crates/memra-server/src/lib.rs', 'crates/memra-cli/src/lib.rs',
                   'crates/memra-sampling/src/lib.rs', 'docs/FLAGS.md']
        for a in samples:
            for b in samples:
                before = self.plan([a])['jobs']
                after = self.plan([a, b])['jobs']
                self.assertTrue(all(not needed or after[job] for job, needed in before.items()))

    def test_malformed_changed_paths_expand(self):
        for path in ('../secret', '/absolute', 'file\nname'):
            self.assertEqual(self.plan([path])['mode'], 'full')

    def test_cargo_argument_validation_falls_back_to_workspace(self):
        for value in ('', 'missing-package', 'memra-server;touch /tmp/x', 'memra-server,memra-server'):
            self.assertEqual(vp.cargo_packages(value, self.repo), ['--workspace'])
        self.assertEqual(vp.cargo_packages('memra-server', self.repo), ['-p', 'memra-server'])

    def test_full_fallback_emits_every_job_true(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            vp.emit(vp.full('failure'))
        values = dict(line.split('=', 1) for line in stream.getvalue().splitlines())
        for job in vp.JOBS:
            self.assertEqual(values[job], 'true')
        self.assertEqual(values['packages'], '')

    def test_full_plan_keeps_a_deleted_optional_contract(self):
        for name in ('collect-serving-qualification.py', 'test_collect_serving_qualification.py'):
            self.put('tools/' + name, '# known optional contract\n')
        before = self.commit()
        for name in ('collect-serving-qualification.py', 'test_collect_serving_qualification.py'):
            (self.repo / 'tools' / name).unlink()
        self.put('Cargo.lock', 'changed lock')
        after = self.commit()
        p = vp.event_plan(self.repo, 'push', '', before, after)
        self.assertEqual(p['mode'], 'full')
        self.assertIn('serving-qualification', {x['id'] for x in p['cpu_contracts']})

    def test_required_contracts_survive_unavailable_git(self):
        p = vp.event_plan(self.repo, 'push', '', 'unknown', 'unknown')
        self.assertTrue({'q35-cache', 'physical-gpu', 'support-records'} <= {x['id'] for x in p['cpu_contracts']})

    def test_explicit_unaffected_contracts_do_not_install_dependencies(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            vp.emit(self.plan(['README.md']))
        values = dict(line.split('=', 1) for line in stream.getvalue().splitlines())
        self.assertEqual(values['contracts'], 'none')
        self.assertEqual(vp.cpu_contract_names(self.repo, values['contracts']), [])

    def test_missing_contract_selection_still_runs_all_available(self):
        expected = []
        for name, contract in vp.TOOL_CONTRACTS.items():
            if contract.get('required'):
                expected.append(name)
                for path in contract['inputs']:
                    self.put(path, '# fixture\n')
        self.assertEqual(vp.cpu_contract_names(self.repo, ''), expected)

    def test_default_contract_runner_refuses_partial_or_deleted_required_inputs(self):
        for c in vp.TOOL_CONTRACTS.values():
            if c.get('required'):
                for path in c['inputs']:self.put(path, '# fixture\n')
        (self.repo / 'tools/test_resolve_physical_gpu.py').unlink()
        with self.assertRaisesRegex(vp.Refused, 'physical-gpu'):
            vp.cpu_contract_names(self.repo, '')
        (self.repo / 'tools/resolve-physical-gpu.py').unlink()
        with self.assertRaisesRegex(vp.Refused, 'physical-gpu'):
            vp.cpu_contract_names(self.repo, '')

    def test_source_symlink_target_expands_for_transitive_module_resolution(self):
        self.put('research/source.rs', 'pub fn f() {}\n')
        link = self.repo / 'crates/memra-server/src/linked.rs'
        link.symlink_to('../../../research/source.rs')
        self.commit()
        self.assertEqual(self.plan(['research/source.rs'])['mode'], 'full')

    def test_nested_external_rust_sources_expand(self):
        self.put('crates/memra-server/src/lib.rs', 'include!("../../../research/outer.rs");')
        self.put('research/outer.rs', 'include!("inner.rs");')
        self.put('research/inner.rs', 'pub const ANSWER: u8 = 42;')
        before = self.commit()
        self.put('research/inner.rs', 'pub const ANSWER: u8 = ;')
        after = self.commit()
        plan = vp.event_plan(self.repo, 'push', '', before, after)
        self.assertEqual(plan['mode'], 'full')
        self.assertTrue(plan['jobs']['server'])

    def test_cross_crate_compiled_include_propagates_transitive_inputs(self):
        self.put('crates/memra-server/src/lib.rs', 'include!("../../memra-probe/src/helper.rs");')
        self.put('crates/memra-probe/src/helper.rs', 'include_str!("../../../research/input.txt");')
        self.commit()
        self.assertEqual(self.plan(['research/input.txt'])['packages'], ['memra-probe', 'memra-server'])
        self.assertEqual(self.plan(['crates/memra-probe/src/nested.rs'])['packages'], ['memra-probe', 'memra-server'])

    def test_cross_crate_source_consumer_chain_reaches_the_final_reader(self):
        self.put('crates/memra-server/src/lib.rs', 'include!("../../memra-probe/src/helper.rs");')
        self.put('crates/memra-probe/src/helper.rs', '#[path="../../memra-lanes/src/lib.rs"] mod helpers;')
        self.put('crates/memra-lanes/src/lib.rs', 'include_str!("../../../research/input.txt");')
        self.commit()
        self.assertEqual(self.plan(['research/input.txt'])['packages'], ['memra-lanes', 'memra-probe', 'memra-server'])

    def test_external_data_symlink_and_runtime_alias_expand(self):
        self.put('research/real.md', 'fixture')
        (self.repo / 'research/alias.md').symlink_to('real.md')
        for reader in ('include_str!("../../../research/alias.md");',
                       'let s = std::fs::read_to_string("research/alias.md");'):
            with self.subTest(reader=reader):
                self.put('crates/memra-server/src/lib.rs', reader)
                self.commit()
                self.assertEqual(self.plan(['research/real.md'])['mode'], 'full')

    def test_runtime_parent_traversal_selects_the_canonical_fixture(self):
        for spelling, target in (
                ('research/fixtures/../expected.md', 'research/expected.md'),
                ('docs/fixtures/../expected.md', 'docs/expected.md'),
                ('research/../docs/expected.md', 'docs/expected.md'),
                ('research/../README.md', 'README.md')):
            with self.subTest(spelling=spelling):
                self.put('research/fixtures/.keep', '')
                self.put('docs/fixtures/.keep', '')
                self.put('crates/memra-server/src/lib.rs',
                         f'let fixture = std::fs::read_to_string("{spelling}");')
                self.put(target, 'before')
                before = self.commit()
                self.put(target, 'after')
                after = self.commit()
                plan = vp.event_plan(self.repo, 'push', '', before, after)
                self.assertEqual(plan['mode'], 'scoped')
                self.assertEqual(plan['packages'], ['memra-server'])
                self.assertTrue(plan['jobs']['server'])

    def test_runtime_parent_traversal_keeps_deleted_and_renamed_inputs(self):
        for rename in (False, True):
            with self.subTest(rename=rename):
                self.put('research/fixtures/.keep', '')
                self.put('research/expected.md', 'fixture')
                self.put('crates/memra-server/src/lib.rs',
                         'let fixture = std::fs::read_to_string("research/fixtures/../expected.md");')
                before = self.commit()
                if rename:
                    (self.repo / 'research/expected.md').rename(self.repo / 'research/renamed.md')
                else:
                    (self.repo / 'research/expected.md').unlink()
                after = self.commit()
                plan = vp.event_plan(self.repo, 'push', '', before, after)
                self.assertEqual(plan['mode'], 'scoped')
                self.assertEqual(plan['packages'], ['memra-server'])
                self.assertIn('research/expected.md', plan['changed'])

    def test_runtime_parent_traversal_to_root_retains_the_whole_subtree(self):
        self.put('research/fixtures/.keep', '')
        self.put('crates/memra-server/src/lib.rs',
                 'let fixture = std::fs::read_dir("research/fixtures/../..");')
        self.commit()
        self.assertEqual(self.plan(['README.md'])['packages'], ['memra-server'])
        self.assertEqual(self.plan(['crates/memra-probe/src/lib.rs'])['packages'],
                         ['memra-probe', 'memra-server'])

    def test_runtime_parent_traversal_through_a_symlink_expands(self):
        self.put('research/actual/nested/.keep', '')
        self.put('research/actual/expected.md', 'before')
        (self.repo / 'research/alias').symlink_to('actual/nested')
        self.put('crates/memra-server/src/lib.rs',
                 'let fixture = std::fs::read_to_string("research/./alias/../expected.md");')
        before = self.commit()
        self.put('research/actual/expected.md', 'after')
        after = self.commit()
        plan = vp.event_plan(self.repo, 'push', '', before, after)
        self.assertEqual(plan['mode'], 'full')
        self.assertTrue(plan['jobs']['server'])
        self.assertIn('symlink', plan['reason'])

    def test_runtime_parent_traversal_old_symlink_side_expands(self):
        self.put('research/actual/nested/.keep', '')
        alias = self.repo / 'research/alias'
        alias.symlink_to('actual/nested')
        self.put('crates/memra-server/src/lib.rs',
                 'let fixture = std::fs::read_to_string("research/./alias/../expected.md");')
        before = self.commit()
        alias.unlink()
        self.put('research/alias/.keep', '')
        self.put('research/expected.md', 'fixture')
        after = self.commit()
        self.assertEqual(vp.event_plan(self.repo, 'push', '', before, after)['mode'], 'full')

    def test_runtime_unresolved_or_escaping_parent_traversal_expands(self):
        for spelling, reason in (
                ('research/missing/../expected.md', 'unresolved'),
                ('research/../../outside.md', 'escapes'),
                ('research/{directory}/../expected.md', 'pattern'),
                ('research/*/../expected.md', 'pattern'),
                ('research/[ab]/../expected.md', 'pattern')):
            with self.subTest(spelling=spelling):
                self.put('research/expected.md', 'fixture')
                self.put('crates/memra-server/src/lib.rs',
                         f'let fixture = std::fs::read_to_string("{spelling}");')
                self.commit()
                plan = self.plan(['research/expected.md'])
                self.assertEqual(plan['mode'], 'full')
                self.assertIn(reason, plan['reason'])

    def test_runtime_normalized_paths_preserve_literal_and_format_reach(self):
        for spelling in ('research/expected.md', 'research/./expected.md',
                         'research/{family}/expected.md',
                         'research/fixtures/../{family}/expected.md'):
            with self.subTest(spelling=spelling):
                self.put('research/fixtures/.keep', '')
                self.put('crates/memra-server/src/lib.rs',
                         f'let fixture = std::fs::read_to_string("{spelling}");')
                self.commit()
                target = 'research/one/expected.md' if '{' in spelling else 'research/expected.md'
                plan = self.plan([target])
                self.assertEqual(plan['mode'], 'scoped')
                self.assertEqual(plan['packages'], ['memra-server'])

    def test_include_tokens_can_be_separated_by_whitespace_and_comments(self):
        for separator in (' ', '\n', ' /* note */ '):
            for path in ('README.md', 'research/input.txt'):
                with self.subTest(separator=separator, path=path):
                    self.put('crates/memra-server/src/lib.rs', f'include_str{separator}! ("../../../{path}");')
                    self.commit()
                    self.assertEqual(self.plan([path])['packages'], ['memra-server'])

    def test_module_path_whitespace_comments_and_escapes_expand(self):
        for attribute in ('# [ path = "../../../research/outer.rs" ]',
                          '#[ /* note */ path = r#"../../../research/outer.rs"# ]',
                          '#[path = "../../../research/outer\\x2ers"]'):
            with self.subTest(attribute=attribute):
                self.put('crates/memra-server/src/lib.rs', attribute + ' mod outer;')
                self.commit()
                self.assertEqual(self.plan(['research/outer.rs'])['mode'], 'full')

    def test_conditional_module_path_expands_transitive_inputs_without_cfg_guessing(self):
        for attribute in (
            '#[cfg_attr(all(), path="../../../research/outer.rs")]',
            '#[cfg_attr(any(), path="../../../research/outer.rs")]',
            '#[cfg_attr(feature="variant", path="../../../research/outer.rs")]',
            '#[cfg_attr(all(), cfg_attr(all(), path="../../../research/outer.rs"))]',
            '#[cfg_attr(all(), path=concat!("../../../", "research/outer.rs"))]',
            '#[cfg_attr(all(), r#path="../../../research/outer.rs")]',
        ):
            with self.subTest(attribute=attribute):
                self.put('crates/memra-server/src/lib.rs', attribute + ' mod outer;')
                self.put('research/outer.rs', 'pub const INPUT: &str = include_str!("inner.md");')
                self.put('research/inner.md', 'changed')
                self.commit()
                plan = self.plan(['research/inner.md'])
                self.assertEqual(plan['mode'], 'full')
                self.assertTrue(plan['jobs']['server'])
                self.assertFalse(plan['native']['qualification'])

    def test_conditional_attribute_comments_strings_and_nonpath_do_not_invent_modules(self):
        for source in (
            '// #[cfg_attr(all(), path="../../../other.rs")] mod outer;',
            'const TEXT: &str = r#"#[cfg_attr(all(), path="../../../other.rs")] mod outer;"#;',
            '#[cfg_attr(all(), allow(dead_code))] fn harmless() {}',
        ):
            with self.subTest(source=source):
                self.put('crates/memra-server/src/lib.rs', source)
                self.commit()
                self.assertEqual(self.plan(['README.md'])['mode'], 'scoped')

    def test_unterminated_conditional_attribute_expands(self):
        self.put('crates/memra-server/src/lib.rs', '#[cfg_attr(all(), path="outer.rs") mod outer;')
        self.commit()
        self.assertEqual(self.plan(['README.md'])['mode'], 'full')

    def test_split_concat_compiled_include_keeps_physical_symlink_traversal(self):
        self.put('research/actual/nested/directory.md', 'directory')
        self.put('research/actual/expected.md', 'changed')
        self.put('research/expected.md', 'different lexical target')
        (self.repo / 'research/alias').symlink_to('actual/nested')
        for source in (
            'const INPUT: &str = include_str!("../../../research/alias/../expected.md");',
            'const INPUT: &str = include_str!(concat!("../../../", "re", "search/alias/../expected.md"));',
            'const INPUT: &[u8] = include_bytes!(concat!("../../../", "re", "search/alias/../expected.md"));',
        ):
            with self.subTest(source=source):
                self.put('crates/memra-server/src/lib.rs', source)
                self.commit()
                plan = self.plan(['research/actual/expected.md'])
                self.assertEqual(plan['mode'], 'full')
                self.assertTrue(plan['jobs']['server'])
                self.assertFalse(plan['native']['qualification'])

    def test_external_source_symlink_expands(self):
        link = self.repo / 'crates/memra-server/src/linked.rs'
        link.symlink_to('/outside/source.rs')
        self.commit()
        self.assertEqual(self.plan(['README.md'])['mode'], 'full')


class FeatureProgramTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='memra-feature-program-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'Cargo.toml').write_text('[workspace]\nresolver="2"\nmembers=["a","b","shared"]\n')
        for name in ('a', 'b', 'shared'):
            (self.root / name / 'src').mkdir(parents=True)
        (self.root / 'shared/Cargo.toml').write_text('[package]\nname="memra-shared"\nversion="0.1.0"\nedition="2024"\n[features]\na=[]\nb=[]\n')
        (self.root / 'shared/src/lib.rs').write_text('#[cfg(not(all(feature="a",feature="b")))]\ncompile_error!("workspace feature union missing");\npub fn x(){}\n')
        for name in ('a', 'b'):
            (self.root / name / 'Cargo.toml').write_text(f'[package]\nname="memra-{name}"\nversion="0.1.0"\nedition="2024"\n[dependencies]\nmemra-shared={{path="../shared",features=["{name}"]}}\n')
            (self.root / name / 'src/main.rs').write_text('fn main(){memra_shared::x();}\n')
        # The helper validates actual workspace directories under crates/ in production;
        # this tiny compiler fixture exercises the command semantics directly.
        subprocess.run(['cargo', 'generate-lockfile', '--offline'], cwd=self.root,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    def test_selected_package_feature_drift_is_detected(self):
        same, reason = vp.feature_compatible(['-p', 'memra-a'], self.root, development=True)
        self.assertFalse(same)
        self.assertIn('memra-shared', reason)

    def test_matching_feature_program_can_use_scoped_clippy(self):
        p = self.root / 'a/Cargo.toml'
        p.write_text(p.read_text().replace('features=["a"]', 'features=["a","b"]'))
        same, reason = vp.feature_compatible(['-p', 'memra-a'], self.root, development=True)
        self.assertTrue(same, reason)

    def test_binary_filter_keeps_feature_union_without_building_other_binary(self):
        env = dict(os.environ, CARGO_BUILD_JOBS='1', RUSTC_WRAPPER='', SCCACHE_DISABLE='1')
        good = subprocess.run(['cargo', 'build', '--offline', '--workspace', '--bin', 'memra-a'],
                              cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(good.returncode, 0, good.stderr)
        self.assertTrue((self.root / 'target/debug/memra-a').exists())
        self.assertFalse((self.root / 'target/debug/memra-b').exists())
        bad = subprocess.run(['cargo', 'build', '--offline', '-p', 'memra-a', '--bin', 'memra-a'],
                             cwd=self.root, env=env, capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn('workspace feature union missing', bad.stderr)

    def test_unavailable_feature_graph_keeps_workspace(self):
        with mock.patch.object(vp.subprocess, 'check_output', side_effect=OSError('unavailable')):
            same, reason = vp.feature_compatible(['-p', 'memra-a'], self.root, development=False)
        self.assertFalse(same)
        self.assertIn('unavailable', reason)


if __name__ == '__main__':
    unittest.main()
