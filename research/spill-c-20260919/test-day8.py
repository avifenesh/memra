#!/usr/bin/env python3
"""CPU red probes for the receipt replay and exact native-refusal classifier."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Day8(unittest.TestCase):
    def test_refusal_classifier_does_not_relabel_other_failures(self):
        refusal = load("refusal", "pressure-refusal.py")
        output = f'Error: "{refusal.REASON}"\n'
        self.assertEqual(refusal.verdict(1, output), (2, "REFUSED: " + refusal.REASON))
        for code, raw in [(0, output), (2, output), (1, "CUDA error"), (1, output + "[expert-host-slru] data"), (-9, output)]:
            with self.subTest(code=code, raw=raw):
                self.assertEqual(refusal.verdict(code, raw)[0], 1)

    def test_raw_and_command_identity_red_probes(self):
        verifier = load("verify8", "verify-day8.py")
        case = "8g-gen-on"
        source = verifier.RAW / case
        with tempfile.TemporaryDirectory(prefix="spill-c-receipt-test-") as temporary:
            verifier.RAW = Path(temporary)
            out = verifier.RAW / case
            shutil.copytree(source, out)
            self.assertEqual(verifier.replay(case)[0]["gpu_evictions"], 12091)
            capture = out / "command.capture.json"
            original = capture.read_text()
            parsed = json.loads(original)
            parsed["command"][-2] = "999"
            capture.write_text(json.dumps(parsed))
            with self.assertRaisesRegex(ValueError, "command identity changed"):
                verifier.replay(case)
            capture.write_text(original)
            log = out / "command.log"
            log.write_bytes(log.read_bytes().replace(b"prefill argmax=198", b"prefill argmax=199", 1))
            with self.assertRaisesRegex(ValueError, "receipt hash mismatch"):
                verifier.replay(case)


if __name__ == "__main__":
    unittest.main(verbosity=2)
