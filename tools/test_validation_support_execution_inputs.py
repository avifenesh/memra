"""Real filesystem refusal before support-census content or CPU execution."""

import hashlib
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import support_record_inputs as data
import validation_plan as vp


ROOT = Path(__file__).resolve().parent.parent


class SupportExecutionInputs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='memra-support-execution-')
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        for name in (*data.READERS, data.RECORDS):
            target = self.repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        (self.repo / data.RECORDS).write_text(
            '[[record]]\n[record.gates]\nConfig="passed"\n'
            '[record.evidence]\nConfig=["ci:verify-tiny"]\n')
        self.tree = data.DirectoryTree(self.repo)

    def assert_no_content_read(self):
        with mock.patch.object(self.tree, 'read', side_effect=AssertionError('content read')), \
                mock.patch.object(self.tree, 'read_bytes', side_effect=AssertionError('byte read')):
            with self.assertRaisesRegex(data.InputContractError, 'regular|symlink|incomplete'):
                data.resolve(self.tree)

    def test_real_fifo_directory_and_socket_refuse_before_any_content_read(self):
        for name in (*data.READERS, data.RECORDS):
            path = self.repo / name
            original = path.read_bytes()
            for kind in ('fifo', 'directory', 'socket'):
                with self.subTest(name=name, kind=kind):
                    path.unlink()
                    if kind == 'fifo':
                        os.mkfifo(path)
                    elif kind == 'directory':
                        path.mkdir()
                    else:
                        endpoint = socket.socket(socket.AF_UNIX)
                        endpoint.bind(str(path))
                    try:
                        self.assert_no_content_read()
                    finally:
                        if kind == 'socket':
                            endpoint.close()
                        if kind == 'directory':
                            path.rmdir()
                        else:
                            path.unlink()
                        path.write_bytes(original)

    def test_leaf_symlink_to_regular_fifo_or_missing_refuses_before_reads(self):
        path = self.repo / data.RECORDS
        original = path.read_bytes()
        for kind in ('regular', 'fifo', 'missing'):
            with self.subTest(kind=kind):
                target = self.repo / ('target-' + kind)
                if kind == 'regular':
                    target.write_bytes(original)
                elif kind == 'fifo':
                    os.mkfifo(target)
                path.unlink()
                path.symlink_to(target)
                self.assert_no_content_read()
                path.unlink()
                path.write_bytes(original)

    def test_ancestor_symlink_including_broken_parent_refuses_before_reads(self):
        for name in ('tools', 'docs'):
            parent = self.repo / name
            moved = self.repo / (name + '-real')
            parent.rename(moved)
            for target in (moved, self.repo / 'missing-parent'):
                with self.subTest(name=name, target=target.name):
                    parent.symlink_to(target, target_is_directory=True)
                    try:
                        self.assert_no_content_read()
                    finally:
                        parent.unlink()
            moved.rename(parent)
        # Both bad parents must not masquerade as an absent optional census.
        for kind in ('broken-link', 'fifo', 'file', 'socket'):
            with self.subTest(both_parents=kind):
                endpoints = []
                parents = [self.repo / name for name in ('tools', 'docs')]
                moved = [self.repo / (name + '-saved') for name in ('tools', 'docs')]
                for parent, saved in zip(parents, moved):
                    parent.rename(saved)
                    if kind == 'broken-link':
                        parent.symlink_to(self.repo / 'missing-parent', target_is_directory=True)
                    elif kind == 'fifo':
                        os.mkfifo(parent)
                    elif kind == 'file':
                        parent.write_bytes(b'not a directory')
                    else:
                        endpoint = socket.socket(socket.AF_UNIX)
                        endpoint.bind(str(parent))
                        endpoints.append(endpoint)
                try:
                    self.assert_no_content_read()
                finally:
                    for endpoint in endpoints:
                        endpoint.close()
                    for parent, saved in zip(parents, moved):
                        parent.unlink()
                        saved.rename(parent)

    def test_read_bytes_remains_exact_and_regular_metadata_still_resolves(self):
        reader = next(iter(data.READERS))
        original = (self.repo / reader).read_bytes()
        crlf = original.replace(b'\n', b'\r\n')
        (self.repo / reader).write_bytes(crlf)
        self.assertEqual(self.tree.read_bytes(reader), crlf)
        self.assertNotEqual(hashlib.sha256(crlf).hexdigest(), data.READERS[reader])
        with self.assertRaises(data.UnmodelledReader):
            data.resolve(self.tree)
        (self.repo / reader).write_bytes(original)
        self.assertEqual(data.resolve(self.tree), {'required': [], 'optional': []})

    def test_unknown_reader_does_not_turn_fifo_into_typed_fallback(self):
        reader = self.repo / next(iter(data.READERS))
        reader.write_bytes(reader.read_bytes() + b'# changed reader\n')
        path = self.repo / data.RECORDS
        path.unlink()
        os.mkfifo(path)
        # Refusal reaches the public execution adapter with fallback enabled.
        with mock.patch.object(vp, '_SUPPORT_DATA', data):
            with self.assertRaisesRegex(vp.Refused, 'regular'):
                vp.support_record_data_inputs(self.repo, directory=True, allow_unknown_reader=True)

    def test_fifo_replacement_between_stat_and_open_refuses_without_blocking(self):
        path = self.repo / data.RECORDS
        real_open = os.open
        swapped = []

        def replace(name, flags, *args, **kwargs):
            if name == path.name and not swapped:
                self.assertTrue(flags & os.O_NONBLOCK)
                path.unlink()
                os.mkfifo(path)
                swapped.append(True)
            return real_open(name, flags, *args, **kwargs)

        with mock.patch.object(data.os, 'open', side_effect=replace):
            with self.assertRaisesRegex(data.InputContractError, 'regular'):
                self.tree.read_bytes(data.RECORDS)
        self.assertEqual(swapped, [True])

    def test_symlink_replacement_between_stat_and_open_refuses(self):
        path = self.repo / data.RECORDS
        target = self.repo / 'outside-content'
        target.write_bytes(b'should never be read')
        real_open = os.open
        swapped = []

        def replace(name, flags, *args, **kwargs):
            if name == path.name and not swapped:
                path.unlink()
                path.symlink_to(target)
                swapped.append(True)
            return real_open(name, flags, *args, **kwargs)

        with mock.patch.object(data.os, 'open', side_effect=replace):
            with self.assertRaisesRegex(data.InputContractError, 'symlink'):
                self.tree.read_bytes(data.RECORDS)
        self.assertEqual(swapped, [True])

    def test_parent_replacement_keeps_read_anchored_to_opened_directory(self):
        parent = self.repo / 'docs'
        moved = self.repo / 'docs-original'
        replacement = self.repo / 'docs-replacement'
        replacement.mkdir()
        (replacement / 'support-records.toml').write_bytes(b'outside content')
        original = (parent / 'support-records.toml').read_bytes()
        real_open = os.open
        swapped = []

        def replace(name, flags, *args, **kwargs):
            if name == 'support-records.toml' and not swapped:
                parent.rename(moved)
                parent.symlink_to(replacement, target_is_directory=True)
                swapped.append(True)
            return real_open(name, flags, *args, **kwargs)

        with mock.patch.object(data.os, 'open', side_effect=replace):
            self.assertEqual(self.tree.read_bytes(data.RECORDS), original)
        with self.assertRaises(data.InputContractError):
            self.tree.read_bytes(data.RECORDS)

    def test_descriptors_close_on_success_and_refusal(self):
        before = set(os.listdir('/proc/self/fd'))
        for _ in range(8):
            self.tree.read_bytes(data.RECORDS)
        path = self.repo / data.RECORDS
        path.unlink()
        os.mkfifo(path)
        for _ in range(8):
            with self.assertRaises(data.InputContractError):
                self.tree.read_bytes(data.RECORDS)
        self.assertEqual(set(os.listdir('/proc/self/fd')), before)

    def test_contracts_cli_refuses_fifo_before_any_cpu_command(self):
        path = self.repo / data.RECORDS
        path.unlink()
        os.mkfifo(path)
        with mock.patch.object(vp, 'ROOT', self.repo), \
                mock.patch.object(vp, '_SUPPORT_DATA', data), \
                mock.patch('sys.argv', ['validation_plan.py', 'contracts', '--selected', 'none']), \
                mock.patch.object(vp, 'run_cpu_contract') as run:
            with self.assertRaisesRegex(vp.Refused, 'regular'):
                vp.main()
            run.assert_not_called()
        for name in ('validation_plan.py', 'support_record_inputs.py'):
            shutil.copyfile(ROOT / 'tools' / name, self.repo / 'tools' / name)
        result = subprocess.run(
            [sys.executable, str(self.repo / 'tools/validation_plan.py'),
             'contracts', '--selected', 'none'], capture_output=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'not a regular file', result.stderr)
        self.assertNotIn(b'CPU contract:', result.stdout)

    def test_absent_fixture_and_partial_fixture_keep_original_contract(self):
        shutil.rmtree(self.repo / 'tools')
        shutil.rmtree(self.repo / 'docs')
        self.assertEqual(data.resolve(self.tree), {'required': [], 'optional': []})
        (self.repo / 'docs').mkdir()
        (self.repo / data.RECORDS).write_text('invalid metadata')
        with self.assertRaisesRegex(data.InputContractError, 'incomplete'):
            data.resolve(self.tree)


if __name__ == '__main__':
    unittest.main()
