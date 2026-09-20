#!/usr/bin/env python3
"""CPU-only verifier teeth; no GPU, fixture or collector execution."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('verify_day8', Path(__file__).with_name('verify_day8.py'))
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


class DescriptorTests(unittest.TestCase):
    def test_valid_nested_descriptor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'raw.log').write_bytes(b'raw')
            V.descriptors({'nested':[dict(path='raw.log', bytes=3,
                sha256=hashlib.sha256(b'raw').hexdigest())]}, root)

    def test_corrupt_bytes_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'raw.log').write_bytes(b'bad')
            with self.assertRaises(AssertionError):
                V.descriptors(dict(path='raw.log', bytes=3,
                    sha256=hashlib.sha256(b'raw').hexdigest()), root)

    def test_wrong_length_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'raw.log').write_bytes(b'raw')
            with self.assertRaises(AssertionError):
                V.descriptors(dict(path='raw.log', bytes=4,
                    sha256=hashlib.sha256(b'raw').hexdigest()), root)

    def test_path_escape_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(AssertionError):
                V.descriptors(dict(path='../raw.log', bytes=3, sha256='x'), Path(temp))

    def test_actual_native_capture(self):
        V.capture(V.ROOT/'conformance-retry1')
        V.capture(V.ROOT/'roundtrip')


if __name__ == '__main__':
    unittest.main()
