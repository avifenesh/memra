#!/usr/bin/env python3
"""CPU-only argument boundaries; stop before filesystem, lock, server or GPU use."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "request_fault_gate", Path(__file__).with_name("request-fault-gate.py")
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class ReachedFilesystem(Exception):
    pass


class PeerBoundaries(unittest.TestCase):
    def invoke(self, peers=None):
        argv = ["request-fault-gate", "--model", "unused", "--bin", "unused", "--out", "unused"]
        if peers is not None:
            argv += ["--peers", str(peers)]
        with patch.object(sys, "argv", argv), patch.object(gate, "Path", side_effect=ReachedFilesystem):
            gate.main()

    def test_minimum_and_last_valid_peer_count_reserve_a_fault_prompt(self):
        for peers in (1, len(gate.PROMPTS) - 1):
            with self.subTest(peers=peers), self.assertRaises(ReachedFilesystem):
                self.invoke(peers)

    def test_invalid_counts_refuse_before_any_io(self):
        for peers in (-1, 0, len(gate.PROMPTS), len(gate.PROMPTS) + 1):
            with self.subTest(peers=peers), contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(SystemExit) as raised:
                    self.invoke(peers)
                self.assertEqual(raised.exception.code, 2)
                self.assertEqual(output.getvalue().strip(), f"REFUSED: --peers must be 1..{len(gate.PROMPTS) - 1}")

    def test_default_three_peer_case_is_still_admitted(self):
        with self.assertRaises(ReachedFilesystem):
            self.invoke()


if __name__ == "__main__":
    unittest.main()
