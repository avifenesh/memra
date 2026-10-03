"""Boundary selection collisions, refusal paths, and actual strict runner controls."""

import io
import json
import re
import os
from pathlib import Path
import shutil
import subprocess
import unittest

import run_public_boundary_contract as runner
import validation_plan as vp

ROOT = Path(__file__).resolve().parent.parent


class BoundaryContractTests(unittest.TestCase):
    def setUp(self):
        import test_validation_plan as fixtures
        self.fixture = fixtures.ValidationPlanTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.repo = self.fixture.repo
        self.inputs = ['tools/check-public-boundary.py', 'tools/test_public_boundary.py',
                       'tools/public-boundary-policy.toml', 'tools/public-boundary-allowlist.jsonl',
                       'tools/run_public_boundary_contract.py']
        for name in self.inputs:
            self.fixture.put(name, (ROOT / name).read_text())
        self.base = self.fixture.commit()

    def plan(self, path):
        return vp.make_plan([path], vp.Tree(self.repo, 'HEAD'), vp.Tree(self.repo, 'HEAD'))

    def test_each_real_boundary_input_selects_only_cpu_contract(self):
        self.assertEqual(vp.TOOL_CONTRACTS['public-boundary']['inputs'], self.inputs)
        for name in self.inputs:
            with self.subTest(name=name):
                p = self.plan(name)
                self.assertEqual(p['mode'], 'scoped')
                self.assertEqual([c['id'] for c in p['cpu_contracts']], ['public-boundary'])
                self.assertFalse(any(p['jobs'].values()))
                self.assertEqual(p['packages'], [])
                self.assertEqual(p['native'], {'scope': 'none', 'requirements': [], 'qualification': False})
                self.assertEqual(set(p['omitted']), set(vp.JOBS))

    def test_actual_committed_changes_and_old_include_keep_all_obligations(self):
        name = 'tools/public-boundary-policy.toml'
        self.fixture.put('crates/memra-server/src/lib.rs', 'pub const POLICY: &str = include_str!("../../../' + name + '");')
        before = self.fixture.commit()
        self.fixture.put(name, (ROOT / name).read_text() + '\n# changed policy\n')
        self.fixture.put('crates/memra-server/src/lib.rs', '// include removed\n')
        after = self.fixture.commit()
        p = vp.event_plan(self.repo, 'push', '', before, after)
        self.assertEqual(p['mode'], 'scoped')
        self.assertEqual(p['packages'], ['memra-server'])
        self.assertTrue(p['jobs']['build'] and p['jobs']['server'] and p['requires_cuda'])
        self.assertEqual([c['id'] for c in p['cpu_contracts']], ['public-boundary'])
        self.assertEqual(p['native']['scope'], 'serving')
        self.assertFalse(p['native']['qualification'])

    def test_cargo_owner_collision_retains_package_and_contract(self):
        self.fixture.put('tools/Cargo.toml',
                         '[package]\nname="memra-lanes"\nversion="0.0.0"\nedition="2024"\n')
        self.fixture.put('tools/src/lib.rs', '// CPU Cargo owner fixture\n')
        members = ['crates/' + n for n in self.fixture.graph if n != 'memra-lanes'] + ['tools']
        self.fixture.put('Cargo.toml', '[workspace]\nresolver="3"\nmembers=' + json.dumps(members) + '\n')
        shutil.rmtree(self.repo / 'crates/memra-lanes')
        for name in self.fixture.graph:
            if name == 'memra-lanes':
                continue
            manifest = self.repo / f'crates/{name}/Cargo.toml'
            manifest.write_text(manifest.read_text().replace(
                '[package]\n', '[package]\nversion="0.0.0"\nedition="2024"\n').replace(
                'path = "../memra-lanes"', 'path = "../../tools"'))
        actual = subprocess.run(['cargo', 'metadata', '--offline', '--no-deps', '--format-version', '1'],
                                cwd=self.repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(actual.returncode, 0, actual.stderr)
        metadata = json.loads(actual.stdout)
        lanes = next(p for p in metadata['packages'] if p['name'] == 'memra-lanes')
        server = next(p for p in metadata['packages'] if p['name'] == 'memra-server')
        dependency = next(d for d in server['dependencies'] if d['name'] == 'memra-lanes')
        self.assertEqual(Path(lanes['manifest_path']), self.repo / 'tools/Cargo.toml')
        self.assertEqual(Path(dependency['path']), self.repo / 'tools')
        self.fixture.commit()
        p = self.plan(self.inputs[0])
        self.assertEqual(p['mode'], 'scoped')
        self.assertEqual(p['packages'], ['memra-lanes', 'memra-server'])
        self.assertTrue(p['jobs']['build'] and p['jobs']['lanes'])
        self.assertEqual([c['id'] for c in p['cpu_contracts']], ['public-boundary'])
        self.assertFalse(p['native']['qualification'])

    def test_missing_each_input_expands_plan_and_refuses_execution(self):
        for name in self.inputs:
            with self.subTest(name=name):
                path = self.repo / name
                original = path.read_bytes()
                path.unlink()
                self.fixture.commit()
                self.assertEqual(self.plan(name)['mode'], 'full')
                with self.assertRaises(vp.Refused):
                    vp.cpu_contract_names(self.repo, 'public-boundary')
                path.write_bytes(original)
                self.fixture.commit()

    def test_symlink_type_expands_both_old_and_new_tree_and_refuses_execution(self):
        name = self.inputs[0]
        path = self.repo / name
        original = path.read_bytes()
        path.unlink()
        path.symlink_to(Path(self.inputs[1]).name)
        before = self.fixture.commit()
        self.assertEqual(self.plan(name)['mode'], 'full')
        with self.assertRaisesRegex(vp.Refused, 'unsafe type'):
            vp.cpu_contract_names(self.repo, 'public-boundary')
        path.unlink(); path.write_bytes(original)
        after = self.fixture.commit()
        self.assertEqual(vp.event_plan(self.repo, 'push', '', before, after)['mode'], 'full')

    def test_populated_gitlink_ancestor_expands_local_and_git_tree_plans(self):
        self.fixture.g('rm', '-r', '--cached', 'tools')
        self.fixture.g('update-index', '--add', '--cacheinfo', '160000,' + self.base + ',tools')
        self.assertTrue((self.repo / self.inputs[0]).is_file())
        indexed = self.fixture.g('write-tree')
        for head in (vp.LocalTree(self.repo), vp.Tree(self.repo, indexed)):
            with self.subTest(tree=head.ref):
                p = vp.make_plan([self.inputs[0]], vp.Tree(self.repo, self.base), head)
                self.assertEqual(p['mode'], 'full')
                self.assertFalse(p['native']['qualification'])

    def test_directory_fifo_and_parent_symlink_refuse_before_contract_start(self):
        name = self.inputs[0]
        path = self.repo / name
        original = path.read_bytes()
        for kind in ('directory', 'fifo'):
            with self.subTest(kind=kind):
                path.unlink()
                path.mkdir() if kind == 'directory' else os.mkfifo(path)
                with self.assertRaises(vp.Refused):
                    vp.cpu_contract_names(self.repo, 'public-boundary')
                path.rmdir() if kind == 'directory' else path.unlink()
                path.write_bytes(original)
        (self.repo / 'tools').rename(self.repo / 'real-tools')
        (self.repo / 'tools').symlink_to('real-tools', target_is_directory=True)
        with self.assertRaisesRegex(vp.Refused, 'unsafe type'):
            vp.boundary_contract_inputs(vp.LocalTree(self.repo), index_metadata=False)
        # C's census guard may refuse this parent before boundary admission.
        # Both actual entry points must refuse; neither may start the contract.
        with self.assertRaises(vp.Refused):
            vp.cpu_contract_names(self.repo, 'public-boundary')

    def test_unmodelled_tool_reader_and_build_inputs_expand(self):
        for path in ('tools/new-boundary-reader.py', 'Cargo.toml', 'Cargo.lock', '.github/workflows/ci.yml'):
            with self.subTest(path=path):
                p = self.plan(path)
                self.assertEqual(p['mode'], 'full')
                self.assertTrue(all(p['jobs'].values()))
                self.assertIn('public-boundary', [c['id'] for c in p['cpu_contracts']])
        self.fixture.put('crates/memra-server/src/lib.rs', 'include_str!(env!("UNKNOWN_BOUNDARY_INPUT"));')
        self.fixture.commit()
        self.assertEqual(self.plan(self.inputs[0])['mode'], 'full')

    def test_required_default_selection_and_genuine_unaffected_none(self):
        for c in vp.TOOL_CONTRACTS.values():
            if c.get('required'):
                for name in c['inputs']:
                    if name not in self.inputs:
                        self.fixture.put(name, '# fixture\n')
        self.fixture.put_support_data_reader_fixture()
        self.assertIn('public-boundary', vp.cpu_contract_names(self.repo, ''))
        self.assertEqual(vp.cpu_contract_names(self.repo, 'none'), [])
        self.assertIn('public-boundary', [c['id'] for c in vp.full('unknown')['cpu_contracts']])

    def test_actual_declared_command_executes_real_sixty_controls(self):
        result = subprocess.run(vp.TOOL_CONTRACTS['public-boundary']['cpu'], cwd=ROOT,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(result.returncode, 0, result.stdout)
        count = re.search(r'public-boundary contract: PASS: executed=(\d+) floor=60 skipped=0', result.stdout)
        self.assertIsNotNone(count, result.stdout)
        self.assertGreaterEqual(int(count.group(1)), runner.MINIMUM)

    def test_unconditional_workflow_security_coverage_and_prepush_unchanged(self):
        workflow = (ROOT / '.github/workflows/ci.yml').read_text()
        workflow = '\n'.join(line for line in workflow.splitlines() if not line.lstrip().startswith('#'))
        job = workflow.split('\n  boundary:', 1)[1].split('\n  build:', 1)[0]
        self.assertNotRegex(job, r'(?m)^\s+(?:if|needs):')
        for command in ('python3 tools/test_public_boundary.py',
                        'python3 tools/check-public-boundary.py check',
                        'python3 tools/check-public-boundary.py verify-allowlist'):
            self.assertIn(command, job)
        self.assertIn('--commits-file', (ROOT / 'tools/hooks/pre-push').read_text())
        self.assertIn('--refs', (ROOT / '.github/workflows/boundary-refs.yml').read_text())


class BoundaryRunnerTests(unittest.TestCase):
    def suite(self, count, *, skip=False, fail=False, expected_failure=False):
        class Case(unittest.TestCase):
            def runTest(self):
                if skip: self.skipTest('planted skip')
                if fail: self.fail('planted failure')
        if expected_failure:
            Case.runTest = unittest.expectedFailure(Case.runTest)
        return unittest.TestSuite(Case() for _ in range(count))

    def test_complete_suite_passes_and_empty_short_skipped_failed_suites_refuse(self):
        for count, skip, fail, expected in ((60, False, False, 0), (0, False, False, 1),
                                          (59, False, False, 1), (60, True, False, 1),
                                          (60, False, True, 1)):
            with self.subTest(count=count, skip=skip, fail=fail):
                self.assertEqual(runner.run(self.suite(count, skip=skip, fail=fail), stream=io.StringIO()), expected)
        for fail in (True, False):
            with self.subTest(expected_failure=True, assertion_fails=fail):
                self.assertEqual(runner.run(self.suite(60, fail=fail, expected_failure=True),
                                            stream=io.StringIO()), 1)


if __name__ == '__main__':
    unittest.main()
