"""Transport-only closure for the pinned support fixture's copied receipt parents."""
import os
from pathlib import Path
import tomllib
import unittest
from unittest import mock

import support_record_inputs as data
import validation_plan as vp

ROOT = Path(__file__).resolve().parent.parent
LEGACY = 'research/modelplan-onboarding-hy3-20260830/tiny'

class ReceiptCopies(unittest.TestCase):
    def setUp(self):
        import test_validation_support_source_inputs as fixtures
        self.fixture = fixtures.SupportSourceInputs()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.repo = self.fixture.repo
        self.g = self.fixture.fixture.fixture.g
        self.commit = self.fixture.fixture.fixture.commit
        self.put = self.fixture.fixture.fixture.put
        self.base = self.fixture.base
        self.parent = 'research/support-fixture'

    def plan(self, path):
        tree = vp.Tree(self.repo, 'HEAD')
        return vp.make_plan([path], tree, tree)

    def test_live_roots_match_independent_fixture_copy_inventory(self):
        records = tomllib.loads((ROOT / data.RECORDS).read_text())['record']
        cited = {p for r in records for ps in r.get('evidence', {}).values() for p in ps
                 if not p.startswith('ci:')}
        expected = {str(Path(p).parent) for p in cited} | {LEGACY}
        self.assertEqual(set(data.receipt_copies(vp.LocalTree(ROOT))['roots']), expected)

    def test_regular_excluded_content_adds_no_blanket_contract(self):
        name = self.parent + '/unread-content.md'
        self.put(name, 'NativeQualified excluded receipt content\n')
        self.commit()
        self.assertEqual(self.plan(name)['cpu_contracts'], [])
        self.assertEqual(self.plan(name)['packages'], [])
        self.assertFalse(self.plan(name)['native']['qualification'])

    def test_broken_and_readable_links_expand_in_both_tree_modes(self):
        for parent in (self.parent, LEGACY):
            for target in ('missing-owned-target', 'gates.txt'):
                with self.subTest(parent=parent, target=target):
                    before = self.g('rev-parse', 'HEAD')
                    path = self.repo / parent / 'copy-link.txt'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.symlink_to(target)
                    unsafe = self.commit()
                    self.assertEqual(vp.event_plan(self.repo, 'push', '', before, unsafe)['mode'], 'full')
                    self.assertEqual(self.plan(str(path.relative_to(self.repo)))['mode'], 'full')
                    with self.assertRaisesRegex(data.InputContractError, 'receipt copy input type'):
                        data.receipt_copies(vp.LocalTree(self.repo))
                    path.unlink()
                    after = self.commit()
                    self.assertEqual(vp.event_plan(self.repo, 'push', '', unsafe, after)['mode'], 'full')

    def test_ignored_fifo_and_unreadable_regular_inputs_expand(self):
        self.put('.gitignore', self.parent + '/ignored-copy.txt\n')
        self.commit()
        path = self.repo / self.parent / 'ignored-copy.txt'
        os.mkfifo(path)
        try:
            with self.assertRaisesRegex(data.InputContractError, 'receipt copy input type'):
                data.receipt_copies(vp.LocalTree(self.repo))
        finally:
            path.unlink()
        path.write_text('excluded content\n')
        path.chmod(0)
        try:
            if not os.access(path, os.R_OK):
                with self.assertRaisesRegex(data.InputContractError, 'receipt copy input type'):
                    data.receipt_copies(vp.LocalTree(self.repo))
        finally:
            path.chmod(0o600)
        self.assertTrue(data.receipt_copies(vp.LocalTree(self.repo))['active'])

    def test_gitlink_copy_ancestor_is_rejected_when_populated(self):
        self.g('rm', '--cached', '-r', self.parent)
        self.g('update-index', '--add', '--cacheinfo', '160000,' + self.base + ',' + self.parent)
        self.g('commit', '-qm', 'fixture populated receipt ancestor gitlink')
        self.assertTrue((self.repo / self.parent).is_dir())
        self.assertEqual(self.plan('README.md')['mode'], 'full')
        with self.assertRaisesRegex(data.InputContractError, 'receipt copy input type'):
            data.receipt_copies(vp.LocalTree(self.repo))

    def test_enumeration_error_refuses_a_narrow_plan(self):
        with mock.patch.object(vp.os, 'scandir', side_effect=PermissionError('owned receipt enumeration')):
            with self.assertRaisesRegex(PermissionError, 'owned receipt enumeration'):
                data.receipt_copies(vp.LocalTree(self.repo))

    def test_required_and_optional_content_semantics_are_preserved(self):
        import test_validation_support_record_inputs as fixture_data
        for path in (fixture_data.GATE, fixture_data.LOCK, fixture_data.TINY):
            plan = self.plan(path)
            self.assertEqual([c['id'] for c in plan['cpu_contracts']], ['support-records'])
            self.assertEqual(plan['packages'], [])
            self.assertFalse(plan['native']['qualification'])
        for path in ('docs/new.md', data.CLI_SOURCE):
            plan = self.plan(path)
            self.assertIn('support-records', [c['id'] for c in plan['cpu_contracts']])

    def test_new_parent_and_repository_root_ambiguity(self):
        import test_validation_support_record_inputs as fixture_data
        import shutil
        shutil.rmtree(self.repo / LEGACY)
        self.commit()
        self.assertEqual(self.plan('README.md')['mode'], 'full')
        self.assertIn('copy directory is missing', self.plan('README.md')['reason'])
        self.put(LEGACY + '/gates.txt', 'Config=passed\n')
        self.commit()
        record = self.repo / data.RECORDS
        record.write_text(fixture_data.metadata('research/new-parent/gates.txt'))
        self.put('research/new-parent/gates.txt', 'Config=passed\n')
        self.commit()
        roles = data.receipt_copies(vp.Tree(self.repo, 'HEAD'))
        self.assertEqual(set(roles['roots']), {'research/new-parent', LEGACY})
        shutil.rmtree(self.repo / 'research/new-parent')
        self.commit()
        self.assertEqual(self.plan('README.md')['mode'], 'full')
        self.assertIn('copy directory is missing', self.plan('README.md')['reason'])
        record.write_text(fixture_data.metadata('root-gates.txt'))
        self.put('root-gates.txt', 'Config=passed\n')
        self.commit()
        self.assertEqual(self.plan('README.md')['mode'], 'full')
        self.assertIn('copies repository root', self.plan('README.md')['reason'])

if __name__ == '__main__':
    unittest.main()
