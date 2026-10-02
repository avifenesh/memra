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

    def test_unknown_tool_or_source_expands(self):
        for path in ('tools/new-builder.py', 'unknown.rs'):
            with self.subTest(path=path):
                self.assertEqual(self.plan([path])['mode'], 'full')

    def test_unreferenced_research_receipt_is_not_an_engine_build(self):
        p = self.plan(['research/new-campaign/receipts/output.json'])
        self.assertFalse(any(p['jobs'].values()))

    def test_runtime_research_fixture_selects_its_reader(self):
        self.put('crates/memra-server/src/lib.rs', 'let x = std::fs::read_to_string("research/fixture.json");')
        self.commit()
        self.assertEqual(self.plan(['research/fixture.json'])['packages'], ['memra-server'])

    def test_directory_fixture_join_selects_descendant_reader(self):
        self.put('crates/memra-server/src/lib.rs', 'let p = root.join("../../research/sample").join("input.json");')
        self.commit()
        self.assertEqual(self.plan(['research/sample/input.json'])['packages'], ['memra-server'])

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

    def test_source_symlink_target_is_a_real_input(self):
        self.put('research/source.rs', 'pub fn f() {}\n')
        link = self.repo / 'crates/memra-server/src/linked.rs'
        link.symlink_to('../../../research/source.rs')
        self.commit()
        self.assertEqual(self.plan(['research/source.rs'])['packages'], ['memra-server'])

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
