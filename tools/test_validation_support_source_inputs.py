"""Content and fixture-copy type closure of the pinned CPU census readers."""

import importlib.util
import os
from pathlib import Path
import shutil
import unittest
from unittest import mock

import support_record_inputs as data
import validation_plan as vp

ROOT = Path(__file__).resolve().parent.parent


class SupportSourceInputs(unittest.TestCase):
    def setUp(self):
        import test_validation_support_record_inputs as fixtures
        self.fixture = fixtures.SupportRecordDataInputs()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.repo = self.fixture.repo
        self.base = self.fixture.base

    def plan(self, path):
        tree = vp.Tree(self.repo, 'HEAD')
        return vp.make_plan([path], tree, tree)

    def assert_census(self, plan):
        self.assertEqual(plan['mode'], 'scoped')
        self.assertEqual([c['id'] for c in plan['cpu_contracts']], ['support-records'])
        self.assertFalse(plan['native']['qualification'])

    def test_live_inventory_matches_independent_actual_checker_paths(self):
        spec = importlib.util.spec_from_file_location('actual_support_checker', ROOT / 'tools/check-support-states.py')
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        expected = {str(p.relative_to(ROOT)) for p in checker.doc_paths(ROOT)}
        expected.update(str(p.relative_to(ROOT)) for p in (ROOT / checker.PACK_DIR).glob('*/mod.rs'))
        expected.update((checker.PACK_DIR + '/mod.rs', checker.CLI_SRC))
        resolved = data.source_docs(vp.LocalTree(ROOT))
        self.assertTrue(resolved['active'])
        self.assertEqual(set(resolved['inputs']), expected)
        prefixes = ('crates', data.PACK_ROOT, data.CLI_SOURCE, 'docs')
        self.assertLessEqual(set(vp.Tree(ROOT, 'HEAD').input_modes(*prefixes, recursive=False)), set(prefixes))

    def test_fixed_current_and_potential_documents_select_only_census(self):
        for path in ('README.md', 'STATUS.md', 'AGENTS.md', 'docs/new.md',
                     'docs/nested/new.md', 'docs/.md', 'docs/archive-sibling/new.md'):
            with self.subTest(path=path):
                plan = self.plan(path)
                self.assert_census(plan)
                self.assertFalse(any(plan['jobs'].values()))

    def test_source_inputs_keep_native_package_and_probe_obligations(self):
        pack_paths = (data.PACK_ROOT + '/hy3/mod.rs', data.PACK_ROOT + '/new_pack/mod.rs',
                      data.PACK_ROOT + '/mod.rs')
        for path in (*pack_paths, data.CLI_SOURCE):
            with self.subTest(path=path):
                plan = self.plan(path)
                self.assert_census(plan)
                graph, owners = vp.workspace(vp.Tree(self.repo, 'HEAD'))
                expected, _ = vp.closure({vp.owner(path, owners)}, graph)
                self.assertEqual(set(plan['packages']), expected)
                self.assertTrue(plan['jobs']['build'])
                self.assertTrue(plan['jobs']['clippy'])
                if path in pack_paths:
                    self.assertTrue(plan['jobs']['server'])
                    self.assertTrue(plan['jobs']['engine'])
                    self.assertTrue(plan['requires_cuda'])
                    self.assertEqual(plan['native']['scope'], 'native-impact')

    def test_content_exclusions_do_not_invent_a_census_or_native_waiver(self):
        for path in ('docs/archive/old.md', 'docs/new.txt', 'docs/new.MD', 'research/new.md',
                     data.PACK_ROOT + '/hy3/not-mod.rs', data.PACK_ROOT + '/hy3/nested/mod.rs'):
            with self.subTest(path=path):
                plan = self.plan(path)
                self.assertFalse(data.reads_source_doc(path))
                if path in ('docs/new.txt', 'docs/new.MD'):
                    self.assertEqual(plan['mode'], 'full')
                else:
                    self.assertEqual(plan['mode'], 'scoped')
                    self.assertNotIn('support-records', [c['id'] for c in plan['cpu_contracts']])
                if path.startswith(data.PACK_ROOT):
                    self.assertTrue(plan['jobs']['engine'])
                    self.assertTrue(plan['requires_cuda'])

    def test_creation_deletion_and_before_after_content_inventory(self):
        before = self.base
        for path in ('docs/new.md', data.PACK_ROOT + '/new_pack/mod.rs', 'STATUS.md'):
            for present in (True, False):
                with self.subTest(path=path, present=present):
                    if present:
                        self.fixture.fixture.put(path, '// new input\n')
                    else:
                        (self.repo / path).unlink()
                    after = self.fixture.fixture.commit()
                    plan = vp.event_plan(self.repo, 'push', '', before, after)
                    before = after
                    self.assert_census(plan)
                    self.assertEqual(plan['changed'], [path])

    def test_copy_root_symlink_types_expand_even_for_excluded_content(self):
        for path in ('docs/archive/old.md', 'docs/nested/link.md', 'docs/directory-link',
                     data.PACK_ROOT + '/hy3/not-mod.rs', data.PACK_ROOT + '/directory-link'):
            with self.subTest(path=path):
                link = self.repo / path
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to('absent-owned-target')
                self.fixture.fixture.commit()
                self.assertEqual(self.plan(path)['mode'], 'full')
                self.assertEqual(vp.local_plan(self.repo, self.base)['mode'], 'full')
                link.unlink()
                self.fixture.fixture.commit()
        self.fixture.fixture.put('.gitignore', 'docs/archive/ignored.md\ndocs/ignored.md\n')
        self.fixture.fixture.commit()
        ignored = self.repo / 'docs/archive/ignored.md'
        ignored.parent.mkdir(parents=True, exist_ok=True)
        ignored.symlink_to('absent-owned-target')
        tree = vp.LocalTree(self.repo)
        self.assertEqual(tree.input_modes('docs')[str(ignored.relative_to(self.repo))], '120000')
        with self.assertRaisesRegex(data.InputContractError, 'ambiguous support fixture copy input type'):
            data.source_docs(tree)
        ignored.unlink()
        ignored.write_text('excluded readable content\n')
        self.assertTrue(data.source_docs(tree)['active'])
        (self.repo / 'docs/ignored.md').write_text('actual ignored reader content\n')
        with self.assertRaisesRegex(data.InputContractError, 'ignored support content input'):
            data.source_docs(tree)
        (self.repo / 'docs/ignored.md').unlink()
        ignored.unlink()
        with mock.patch.object(vp.os, 'scandir', side_effect=PermissionError('owned copy enumeration observer')):
            with self.assertRaisesRegex(PermissionError, 'owned copy enumeration observer'):
                data.source_docs(tree)
            plan = vp.make_plan(['README.md'], vp.Tree(self.repo, 'HEAD'), tree)
            self.assertEqual(plan['mode'], 'full')
            self.assertIn('owned copy enumeration observer', plan['reason'])
        shutil.rmtree(self.repo / 'docs')
        (self.repo / 'docs').symlink_to('absent-owned-docs', target_is_directory=True)
        self.fixture.fixture.commit()
        self.assertEqual(self.plan('docs/new.md')['mode'], 'full')

    def test_gitlinks_and_untracked_directory_types_expand(self):
        path = 'docs/archive/opaque'
        self.fixture.fixture.g('update-index', '--add', '--cacheinfo', '160000,' + self.base + ',' + path)
        self.fixture.fixture.g('commit', '-qm', 'fixture gitlink')
        self.assertEqual(self.plan(path)['mode'], 'full')
        mode = vp.LocalTree(self.repo).input_modes(path, recursive=False)
        self.assertEqual(mode, {})
        self.fixture.fixture.g('rm', '--cached', path)
        self.fixture.fixture.commit()
        self.fixture.fixture.put('STATUS.md/nested.txt', 'wrong fixed-file type\n')
        self.fixture.fixture.commit()
        self.assertEqual(self.plan('STATUS.md')['mode'], 'full')
        self.assertIn('ambiguous support fixture copy input type',
                      vp.local_plan(self.repo, self.base)['reason'])
        shutil.rmtree(self.repo / 'STATUS.md')
        self.fixture.fixture.commit()
        self.fixture.fixture.put(data.PACK_ROOT, 'wrong copy-root type\n')
        self.fixture.fixture.commit()
        self.assertEqual(self.plan(data.PACK_ROOT)['mode'], 'full')
        (self.repo / data.PACK_ROOT).unlink()
        self.fixture.fixture.commit()
        record = self.repo / data.RECORDS
        canonical = record.read_bytes()
        record.unlink()
        os.mkfifo(record)
        tree = vp.LocalTree(self.repo)
        with mock.patch.object(tree, 'read', side_effect=AssertionError('type preflight must precede content I/O')):
            with self.assertRaisesRegex(data.InputContractError, 'ambiguous support fixture copy input type'):
                data.source_docs(tree)
        record.unlink()
        record.write_bytes(canonical)

    def test_unknown_reader_and_noncanonical_source_paths_expand(self):
        for path in ('docs/a/../new.md', data.PACK_ROOT + '/../mod.rs', 'docs/with\nnewline.md'):
            with self.subTest(path=path):
                self.assertEqual(self.plan(path)['mode'], 'full')
        self.fixture.fixture.put('tools/check-support-states.py', '# changed reader\n')
        self.fixture.fixture.commit()
        self.assertEqual(self.plan('docs/new.md')['mode'], 'full')


if __name__ == '__main__':
    unittest.main()
