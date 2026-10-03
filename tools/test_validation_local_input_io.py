"""Real local planning input types and caller outcomes; no native execution."""
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import validation_plan as vp
import test_validation_plan as fixtures


class LocalInputIO(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ValidationPlanTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.repo
        self.tree = vp.LocalTree(self.root)

    def test_regular_text_and_raw_bytes_preserve_original_semantics(self):
        path = self.root / 'data.txt'
        raw = 'עברית\r\nnext\n'.encode()
        path.write_bytes(raw)
        self.assertEqual(self.tree.read_bytes('data.txt'), raw)
        self.assertEqual(self.tree.read('data.txt'), 'עברית\nnext\n')
        self.assertEqual(vp.workspace(self.tree)[0], vp.workspace(vp.Tree(self.root, self.fixture.base))[0])
        self.assertEqual(vp.cargo_packages('memra-lanes', self.root), ['-p', 'memra-lanes'])

    def test_fifo_and_directory_readers_refuse_before_content(self):
        for kind in ('fifo', 'directory'):
            path = self.root / kind
            os.mkfifo(path) if kind == 'fifo' else path.mkdir()
            for reader in (self.tree.read, self.tree.read_bytes):
                with self.subTest(kind=kind, reader=reader.__name__):
                    with mock.patch.object(vp.os, 'fdopen', side_effect=AssertionError('content engaged')):
                        with self.assertRaisesRegex(vp.Refused, 'regular'):
                            reader(kind)

    def test_leaf_and_parent_links_are_refused_before_reading_target(self):
        self.fixture.put('real/content.txt', 'target\n')
        (self.root / 'leaf.txt').symlink_to(self.root / 'real/content.txt')
        (self.root / 'alias').symlink_to(self.root / 'real', target_is_directory=True)
        (self.root / 'broken').symlink_to(self.root / 'missing', target_is_directory=True)
        for name in ('leaf.txt', 'alias/content.txt', 'broken/content.txt'):
            with self.subTest(name=name), mock.patch.object(vp.os, 'fdopen', side_effect=AssertionError('content engaged')):
                with self.assertRaisesRegex(vp.Refused, 'symlink'):
                    self.tree.read(name)

    def test_stat_to_fifo_replacement_is_nonblocking_and_refused(self):
        self.fixture.put('race.txt', 'regular\n')
        path = self.root / 'race.txt'
        real_open = os.open
        swapped = []
        def replace(name, flags, *args, **kwargs):
            if name == 'race.txt' and not swapped:
                self.assertTrue(flags & os.O_NONBLOCK)
                path.unlink(); os.mkfifo(path); swapped.append(True)
            return real_open(name, flags, *args, **kwargs)
        with mock.patch.object(vp.os, 'open', side_effect=replace):
            with self.assertRaisesRegex(vp.Refused, 'regular'):
                self.tree.read_bytes('race.txt')
        self.assertEqual(swapped, [True])

    def test_stat_to_symlink_replacement_refuses_actual_target_content(self):
        self.fixture.put('race.txt', 'regular\n'); self.fixture.put('target.txt', 'must not read')
        path = self.root / 'race.txt'; real_open = os.open; swapped = []
        def replace(name, flags, *args, **kwargs):
            if name == 'race.txt' and not swapped:
                path.unlink(); path.symlink_to(self.root / 'target.txt'); swapped.append(True)
            return real_open(name, flags, *args, **kwargs)
        with mock.patch.object(vp.os, 'open', side_effect=replace):
            with self.assertRaisesRegex(vp.Refused, 'symlink'):
                self.tree.read_bytes('race.txt')
        self.assertEqual(swapped, [True])

    def test_descriptor_cleanup_on_success_and_failure(self):
        self.fixture.put('regular.txt', 'data'); os.mkfifo(self.root / 'fifo')
        before = set(os.listdir('/proc/self/fd'))
        self.tree.read('regular.txt')
        with self.assertRaises(vp.Refused): self.tree.read('fifo')
        with self.assertRaises(vp.Refused): self.tree.read('missing/file')
        self.assertEqual(before, set(os.listdir('/proc/self/fd')))

    def test_root_cargo_fifo_expands_actual_planning_before_read(self):
        (self.root / 'Cargo.toml').unlink(); os.mkfifo(self.root / 'Cargo.toml')
        with mock.patch.object(vp.os, 'fdopen', side_effect=AssertionError('content engaged')):
            plan = vp.make_plan(['Cargo.toml'], vp.Tree(self.root, self.fixture.base), self.tree)
        self.assertEqual(plan['mode'], 'full')
        self.assertTrue(all(plan['jobs'].values()))
        self.assertFalse(plan['native']['qualification'])

    def test_unrelated_cargo_manifest_fifo_refuses_command_planning(self):
        path = self.root / 'crates/memra-server/Cargo.toml'; path.unlink(); os.mkfifo(path)
        with mock.patch.object(vp.os, 'fdopen', side_effect=AssertionError('content engaged')):
            with self.assertRaisesRegex(vp.Refused, 'regular'):
                vp.cargo_packages('memra-lanes', self.root)

    def test_own_registry_fifo_refuses_real_include_reader(self):
        original = Path(vp.__file__).resolve()
        with tempfile.TemporaryDirectory(prefix='memra-own-registry-') as owned:
            import importlib.util
            tools = Path(owned); copied = tools / 'validation_plan.py'; shutil.copyfile(original, copied)
            os.mkfifo(tools / 'validation_inputs.json')
            spec = importlib.util.spec_from_file_location('own_registry_reader', copied)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            _, owners = module.workspace(module.LocalTree(self.root))
            with self.assertRaisesRegex(module.Refused, 'regular'):
                module.included_inputs(module.LocalTree(self.root), owners)

    def test_tracked_census_reader_fifo_refuses_raw_byte_adapter(self):
        self.fixture.put_support_data_reader_fixture(); self.fixture.commit()
        path = self.root / 'tools/check-support-states.py'; path.unlink(); os.mkfifo(path)
        with mock.patch.object(vp.os, 'fdopen', side_effect=AssertionError('content engaged')):
            with self.assertRaisesRegex(vp.Refused, 'regular'):
                vp.support_record_data_inputs(self.tree)

    def test_snapshot_reads_remain_git_bytes_not_local_filesystem(self):
        original = (self.root / 'Cargo.toml').read_bytes()
        path = self.root / 'Cargo.toml'; path.unlink(); os.mkfifo(path)
        snapshot = vp.Tree(self.root, self.fixture.base)
        self.assertEqual(snapshot.read_bytes('Cargo.toml'), original)
        self.assertEqual(snapshot.read('Cargo.toml').encode(), original)

    def test_missing_and_noncanonical_paths_remain_refusals(self):
        for name in ('missing', '../outside', '/outside', 'a//b', './a', 'a\\b'):
            with self.subTest(name=name), self.assertRaises(vp.Refused):
                self.tree.read(name)


if __name__ == '__main__':
    unittest.main()
