"""Real local planning input types and caller outcomes; no native execution."""
import os
from pathlib import Path
import shutil
import socket
import stat
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
        for kind in ('fifo', 'directory', 'socket'):
            path = self.root / kind
            if kind == 'fifo':
                os.mkfifo(path)
            elif kind == 'directory':
                path.mkdir()
            else:
                endpoint = socket.socket(socket.AF_UNIX)
                self.addCleanup(endpoint.close)
                endpoint.bind(str(path))
            real_open = os.open
            def refuse_special_open(name, *args, **kwargs):
                if name == kind:
                    raise AssertionError('known special input opened')
                return real_open(name, *args, **kwargs)
            for reader in (self.tree.read, self.tree.read_bytes):
                with self.subTest(kind=kind, reader=reader.__name__):
                    with mock.patch.object(vp.os, 'fdopen', side_effect=AssertionError('content engaged')), \
                            mock.patch.object(vp.os, 'open', side_effect=refuse_special_open):
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
        manifests = list((self.root / 'crates').glob('*/Cargo.toml'))
        safe = [p for p in manifests if p != path]
        real_fdopen = os.fdopen
        for unsafe_first in (True, False):
            with self.subTest(unsafe_first=unsafe_first):
                content_reads = []
                def observe(descriptor, *args, **kwargs):
                    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                        raise AssertionError('unsafe manifest content engaged')
                    content_reads.append(descriptor)
                    return real_fdopen(descriptor, *args, **kwargs)
                ordered = [path, *safe] if unsafe_first else [*safe, path]
                with mock.patch.object(Path, 'glob', return_value=iter(ordered)), \
                        mock.patch.object(vp.os, 'fdopen', side_effect=observe):
                    with self.assertRaisesRegex(vp.Refused, 'regular'):
                        vp.cargo_packages('memra-lanes', self.root)
                self.assertEqual(len(content_reads), 0 if unsafe_first else len(safe))

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

    def test_real_consumers_refuse_leaf_aliases_and_replacement_races(self):
        import importlib.util
        self.fixture.put_support_data_reader_fixture(); self.fixture.commit()
        with tempfile.TemporaryDirectory(prefix='memra-consumer-registry-') as owned:
            tools = Path(owned)
            shutil.copyfile(vp.__file__, tools / 'validation_plan.py')
            shutil.copyfile(Path(vp.__file__).with_name('validation_inputs.json'), tools / 'validation_inputs.json')
            spec = importlib.util.spec_from_file_location('consumer_registry', tools / 'validation_plan.py')
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            cases = [
                ('root', self.root / 'Cargo.toml', vp,
                 lambda: vp.make_plan(['Cargo.toml'], vp.Tree(self.root, self.fixture.base), self.tree)),
                ('cargo', self.root / 'crates/memra-server/Cargo.toml', vp,
                 lambda: vp.cargo_packages('memra-lanes', self.root)),
                ('reader', self.root / 'tools/check-support-states.py', vp,
                 lambda: vp.support_record_data_inputs(self.tree)),
                ('registry', tools / 'validation_inputs.json', module,
                 lambda: module.included_inputs(self.tree, module.workspace(self.tree)[1])),
            ]
            for name, target, consumer, execute in cases:
                original = target.read_bytes()
                saved = target.with_name(target.name + '.saved'); saved.write_bytes(original)
                for kind in ('alias', 'race-alias', 'race-fifo'):
                    with self.subTest(consumer=name, kind=kind):
                        real_open = os.open; swapped = []
                        if kind == 'alias':
                            target.unlink(); target.symlink_to(saved)
                        def replace(leaf, flags, *args, **kwargs):
                            if leaf == target.name and not swapped:
                                self.assertTrue(flags & os.O_NONBLOCK)
                                if kind != 'alias':
                                    target.unlink()
                                    if kind == 'race-fifo': os.mkfifo(target)
                                    else: target.symlink_to(saved)
                                swapped.append(True)
                            return real_open(leaf, flags, *args, **kwargs)
                        try:
                            with mock.patch.object(consumer.os, 'open', side_effect=replace):
                                if name == 'root':
                                    plan = execute()
                                    self.assertEqual(plan['mode'], 'full')
                                    self.assertTrue(all(plan['jobs'].values()))
                                    self.assertFalse(plan['native']['qualification'])
                                else:
                                    with self.assertRaises(consumer.Refused): execute()
                            if kind != 'alias': self.assertEqual(swapped, [True])
                        finally:
                            target.unlink(); target.write_bytes(original)
                saved.unlink()

    def test_real_descendant_consumers_refuse_ancestor_aliases_and_anchor_races(self):
        self.fixture.put_support_data_reader_fixture(); self.fixture.commit()
        for relative, execute in (
                ('crates/memra-server', lambda: vp.cargo_packages('memra-lanes', self.root)),
                ('tools', lambda: vp.support_record_data_inputs(self.tree))):
            target = self.root / relative
            saved = target.with_name(target.name + '-saved')
            real_open = os.open
            for race in (False, True):
                with self.subTest(relative=relative, race=race):
                    swapped = []
                    if not race: target.rename(saved); target.symlink_to(saved, target_is_directory=True)
                    def replace(part, flags, *args, **kwargs):
                        if race and part == target.name and not swapped:
                            target.rename(saved); target.symlink_to(saved, target_is_directory=True); swapped.append(True)
                        return real_open(part, flags, *args, **kwargs)
                    try:
                        with mock.patch.object(vp.os, 'open', side_effect=replace):
                            with self.assertRaises(vp.Refused): execute()
                        if race: self.assertEqual(swapped, [True])
                    finally:
                        target.unlink(); saved.rename(target)

    def test_opened_parent_stays_anchored_when_its_path_is_replaced(self):
        self.fixture.put('parent/input.txt', 'original\r\n')
        self.fixture.put('replacement/input.txt', 'different\n')
        parent = self.root / 'parent'; saved = self.root / 'saved-parent'
        real_open = os.open; swapped = []
        def replace(part, flags, *args, **kwargs):
            descriptor = real_open(part, flags, *args, **kwargs)
            if part == 'parent' and not swapped:
                parent.rename(saved)
                parent.symlink_to(self.root / 'replacement', target_is_directory=True)
                swapped.append(True)
            return descriptor
        before = set(os.listdir('/proc/self/fd'))
        with mock.patch.object(vp.os, 'open', side_effect=replace):
            self.assertEqual(self.tree.read_bytes('parent/input.txt'), b'original\r\n')
        self.assertEqual(swapped, [True])
        self.assertEqual(before, set(os.listdir('/proc/self/fd')))


if __name__ == '__main__':
    unittest.main()
