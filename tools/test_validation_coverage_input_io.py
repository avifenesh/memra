"""Owned coverage input types/races; result rows are synthetic CPU metadata."""
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import validation_coverage as vc


class CoverageInputIO(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='memra-coverage-io-')
        self.addCleanup(self.tmp.cleanup)
        self.parent = Path(self.tmp.name)
        self.root = self.parent / 'root'; self.root.mkdir()
        self.outside = self.parent / 'outside'; self.outside.mkdir()
        self.raw = 'עברית\r\nsource\n'.encode()
        self.source = self.root / 'harness.py'; self.source.write_bytes(self.raw)
        self.external = self.outside / 'harness.py'; self.external.write_bytes(self.raw)
        self.context = {'hardware': 'cpu', 'case': 'owned IO, synthetic result metadata'}
        self.case = {'id': 'owned', 'cost': 1, 'covers': ['io'], 'mandatory': True,
                     'scope': self.context, 'inputs': {'harness.py': hashlib.sha256(self.raw).hexdigest()}}
        # This frozen healthy contract remains independent of a reader mutation.
        # An incorrect hash reader must reach a behavior assertion, not break setup.
        contract = {'schema': 'memra-edge-contract-v1', 'required_edges': ['io'],
                    'context': self.context, 'tests': {'owned': self.case}}
        self.plan = {'decision': 'scoped', 'selected': ['owned'], 'context': self.context,
                     'contract': contract, 'contract_id': vc.contract_digest(contract)}
        self.results = {'owned': {'contract_id': self.plan['contract_id'], 'status': 'passed',
                                 'context': self.context, 'executed': 1, 'skipped': 0, 'edges': {'io': 'passed'}}}

    def select(self): return vc.select(['io'], [self.case], self.context, self.root)
    def admit(self): return vc.validate_results(self.plan, self.results, self.context, self.root)

    def test_regular_bytes_text_and_context_preserve_semantics(self):
        self.assertEqual(vc.read_input(self.source, root=self.root), self.raw)
        self.assertEqual(vc.read_input(self.source, binary=False), self.raw.decode().replace('\r\n', '\n'))
        self.assertEqual(self.select()['decision'], 'scoped')
        self.assertEqual(self.admit()['status'], 'passed')
        self.assertFalse(self.admit()['qualification'])

    def test_contained_leaf_parent_and_absolute_source_aliases_remain_valid(self):
        (self.root / 'alias.py').symlink_to(self.source)
        (self.root / 'nested').mkdir(); (self.root / 'nested/file.py').write_bytes(self.raw)
        (self.root / 'directory-alias').symlink_to(self.root / 'nested', target_is_directory=True)
        for name in ('alias.py', 'directory-alias/file.py', str(self.source)):
            with self.subTest(name=name):
                self.case['inputs'] = {name: hashlib.sha256(self.raw).hexdigest()}
                plan = self.select(); self.assertEqual(plan['decision'], 'scoped')
                result = {'owned': {**self.results['owned'], 'contract_id': plan['contract_id']}}
                self.assertEqual(vc.validate_results(plan, result, self.context, self.root)['status'], 'passed')

    def test_missing_escaping_and_broken_sources_expand_or_refuse(self):
        for target in (self.external, self.outside / 'missing'):
            with self.subTest(target=target):
                self.source.unlink(); self.source.symlink_to(target)
                self.assertEqual(self.select()['decision'], 'expand')
                with self.assertRaisesRegex(ValueError, 'changed before result admission'): self.admit()

    def test_known_fifo_directory_socket_refuse_before_special_open(self):
        for kind in ('fifo', 'directory', 'socket'):
            with self.subTest(kind=kind):
                self.source.unlink()
                if kind == 'fifo': os.mkfifo(self.source)
                elif kind == 'directory': self.source.mkdir()
                else:
                    endpoint = socket.socket(socket.AF_UNIX); endpoint.bind(str(self.source)); self.addCleanup(endpoint.close)
                real_open = os.open
                def observe(name, *args, **kwargs):
                    if name == 'harness.py': raise AssertionError('known unsafe leaf opened')
                    return real_open(name, *args, **kwargs)
                try:
                    with mock.patch.object(vc.os, 'open', side_effect=observe):
                        self.assertEqual(self.select()['decision'], 'expand')
                        with self.assertRaisesRegex(ValueError, 'changed before result admission'): self.admit()
                finally:
                    self.source.rmdir() if kind == 'directory' else self.source.unlink()
                    self.source.write_bytes(self.raw)

    def test_leaf_replacement_alias_refuses_both_real_consumers(self):
        for execute in (self.select, self.admit):
            self.source.unlink(); self.source.write_bytes(self.raw)
            real_open = os.open; swapped = []
            def replace(name, flags, *args, **kwargs):
                if name == 'harness.py' and not swapped:
                    self.source.unlink(); self.source.symlink_to(self.external); swapped.append(True)
                return real_open(name, flags, *args, **kwargs)
            with self.subTest(consumer=execute.__name__), mock.patch.object(vc.os, 'open', side_effect=replace):
                if execute.__name__ == 'select': self.assertEqual(execute()['decision'], 'expand')
                else:
                    with self.assertRaises(ValueError): execute()
            self.assertEqual(swapped, [True])

    def test_leaf_replacement_fifo_is_nonblocking_and_refused(self):
        for execute in (self.select, self.admit):
            self.source.unlink(); self.source.write_bytes(self.raw)
            real_open = os.open; swapped = []
            def replace(name, flags, *args, **kwargs):
                if name == 'harness.py' and not swapped:
                    self.assertTrue(flags & os.O_NONBLOCK)
                    self.source.unlink(); os.mkfifo(self.source); swapped.append(True)
                return real_open(name, flags, *args, **kwargs)
            real_fdopen = os.fdopen
            def content(descriptor, *args, **kwargs):
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise AssertionError('nonregular opened object reached content')
                return real_fdopen(descriptor, *args, **kwargs)
            with self.subTest(consumer=execute.__name__), mock.patch.object(vc.os, 'open', side_effect=replace), \
                    mock.patch.object(vc.os, 'fdopen', side_effect=content):
                if execute.__name__ == 'select': self.assertEqual(execute()['decision'], 'expand')
                else:
                    with self.assertRaises(ValueError): execute()
            self.assertEqual(swapped, [True])

    def test_resolved_ancestor_replacement_refuses_opened_escape(self):
        nested = self.root / 'nested'; nested.mkdir(); (nested / 'harness.py').write_bytes(self.raw)
        saved = self.root / 'saved'; self.case['inputs'] = {'nested/harness.py': hashlib.sha256(self.raw).hexdigest()}
        real_open = os.open; swapped = []
        def replace(name, flags, *args, **kwargs):
            if name == 'nested' and not swapped:
                nested.rename(saved); nested.symlink_to(self.outside, target_is_directory=True); swapped.append(True)
            return real_open(name, flags, *args, **kwargs)
        with mock.patch.object(vc.os, 'open', side_effect=replace): self.assertEqual(self.select()['decision'], 'expand')
        self.assertEqual(swapped, [True])

    def test_opened_parent_keeps_original_bytes_after_path_replacement(self):
        nested = self.root / 'nested'; nested.mkdir(); (nested / 'harness.py').write_bytes(b'original')
        saved = self.root / 'saved'; real_open = os.open; swapped = []
        def replace(name, flags, *args, **kwargs):
            descriptor = real_open(name, flags, *args, **kwargs)
            if name == 'nested' and not swapped:
                nested.rename(saved); nested.symlink_to(self.outside, target_is_directory=True); swapped.append(True)
            return descriptor
        with mock.patch.object(vc.os, 'open', side_effect=replace):
            self.assertEqual(vc.read_input(nested / 'harness.py', root=self.root), b'original')
        self.assertEqual(swapped, [True])

    def test_descriptors_close_on_success_type_and_open_failure(self):
        before = set(os.listdir('/proc/self/fd'))
        vc.read_input(self.source, root=self.root)
        self.source.unlink(); os.mkfifo(self.source)
        with self.assertRaises(ValueError): vc.read_input(self.source, root=self.root)
        with self.assertRaises(ValueError): vc.read_input(self.root / 'missing/file', root=self.root)
        self.assertEqual(set(os.listdir('/proc/self/fd')), before)

    def documents(self):
        manifest = self.outside / 'manifest.json'; results = self.outside / 'results.json'
        manifest.write_text(json.dumps({'required_edges': ['io'], 'tests': [self.case], 'context': self.context}))
        results.write_text(json.dumps(self.results))
        return manifest, results

    def test_external_cli_documents_and_aliases_work_normal_and_optimized(self):
        manifest, results = self.documents()
        alias = self.parent / 'manifest-alias.json'; alias.symlink_to(manifest)
        for flags in ([], ['-O']):
            run = subprocess.run([sys.executable, *flags, vc.__file__, str(alias), '--root', str(self.root),
                                  '--results', str(results)], capture_output=True, timeout=5)
            self.assertEqual(run.returncode, 0, run.stderr.decode())
            self.assertEqual(json.loads(run.stdout)['status'], 'passed')
            self.assertFalse(json.loads(run.stdout)['qualification'])

    def test_cli_fifo_documents_refuse_without_blocking_normal_and_optimized(self):
        for which in ('manifest', 'results'):
            manifest, results = self.documents(); target = manifest if which == 'manifest' else results
            target.unlink(); os.mkfifo(target)
            for flags in ([], ['-O']):
                run = subprocess.run([sys.executable, *flags, vc.__file__, str(manifest), '--root', str(self.root),
                                      '--results', str(results)], capture_output=True, timeout=5)
                self.assertNotEqual(run.returncode, 0); self.assertEqual(run.stdout, b'')
                self.assertIn(b'not a regular file', run.stderr)
            target.unlink()

    def test_cli_open_replacement_fifo_refuses_both_document_readers(self):
        for which in ('manifest', 'results'):
            manifest, results = self.documents(); target = manifest if which == 'manifest' else results
            real_open = os.open; swapped = []
            def replace(name, flags, *args, **kwargs):
                if name == target.name and not swapped:
                    self.assertTrue(flags & os.O_NONBLOCK)
                    target.unlink(); os.mkfifo(target); swapped.append(True)
                return real_open(name, flags, *args, **kwargs)
            argv = [vc.__file__, str(manifest), '--root', str(self.root), '--results', str(results)]
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(vc.os, 'open', side_effect=replace):
                with self.assertRaisesRegex(ValueError, 'regular'): vc.main()
            self.assertEqual(swapped, [True]); target.unlink()


if __name__ == '__main__':
    unittest.main()
