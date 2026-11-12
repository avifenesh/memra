#!/usr/bin/env python3
"""CPU tests for tools/hidden_trace_diff.py. Run: python3 -m unittest tools.test_hidden_trace_diff"""
import os
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "hidden_trace_diff.py")
BISECT = os.path.join(HERE, "..", "research", "glm53-flash-bringup-20260827", "forward-bisect-receipts")


def hexrow(values):
    return ",".join(struct.pack(">f", v).hex() for v in values)


def write_trace(path, stages, tokens="1,2,3"):
    with open(path, "w") as f:
        f.write("format\tmemra-hidden-trace-v1\n")
        if tokens is not None:
            f.write(f"tokens\t{tokens}\n")
        for stage, layer, values in stages:
            f.write(f"stage\t{stage}\t{layer}\t{len(values)}\t{hexrow(values)}\n")


def run(*args):
    proc = subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip().splitlines()[-1]


BASE = [1.0, 2.0, -0.5, 0.25]


def trunk(layer_values, extra=()):
    """expand, per-layer mixer/layer, collapse; layer_hidden appended after the run."""
    stages = [("expand", -1, BASE)]
    for i, v in enumerate(layer_values):
        stages.append(("mixer", i, v))
        stages.append(("layer", i, v))
    stages.append(("collapse", -1, layer_values[-1]))
    stages.extend(extra)
    for i, v in enumerate(layer_values):
        stages.append(("layer_hidden", i, v))
    return stages


class DiffTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.a = os.path.join(self.dir.name, "a.trace")
        self.b = os.path.join(self.dir.name, "b.trace")

    def tearDown(self):
        self.dir.cleanup()

    def test_identical_traces_are_clean(self):
        write_trace(self.a, trunk([BASE, BASE, BASE]))
        write_trace(self.b, trunk([BASE, BASE, BASE]))
        self.assertEqual(run(self.a, self.b)[0], 0)

    def test_exact_names_the_first_layer_that_moved_not_the_last(self):
        moved = [v * 1.001 for v in BASE]
        write_trace(self.a, trunk([BASE, BASE, BASE]))
        write_trace(self.b, trunk([BASE, moved, [v * 1.5 for v in BASE]]))
        code, text = run(self.a, self.b)
        self.assertEqual(code, 1)
        self.assertIn("FIRST DIVERGENCE at mixer@1", text)

    def test_collapse_sorts_after_the_layers(self):
        # Only the post-run layer_hidden lines carry layers: collapse must still sort last,
        # so a layer-1 change is named at layer 1, not at collapse.
        moved = [v * 1.001 for v in BASE]
        a = [("collapse", -1, BASE), ("logits", -1, BASE), ("layer_hidden", 0, BASE), ("layer_hidden", 1, BASE)]
        b = [("collapse", -1, moved), ("logits", -1, moved), ("layer_hidden", 0, BASE), ("layer_hidden", 1, moved)]
        write_trace(self.a, a)
        write_trace(self.b, b)
        self.assertIn("FIRST DIVERGENCE at layer_hidden@1", run(self.a, self.b)[1])

    def test_order_sensitive_stage_never_decides(self):
        write_trace(self.a, trunk([BASE, BASE], extra=[("route", 1, [1.0, 0.5, 2.0, 0.25])]))
        write_trace(self.b, trunk([BASE, BASE], extra=[("route", 1, [2.0, 0.25, 1.0, 0.5])]))
        self.assertEqual(run(self.a, self.b)[0], 0)

    def test_different_tokens_are_incomparable(self):
        write_trace(self.a, trunk([BASE]), tokens="1,2,3")
        write_trace(self.b, trunk([BASE]), tokens="1,2,4")
        self.assertEqual(run(self.a, self.b)[0], 2)

    def test_width_mismatch_is_incomparable(self):
        write_trace(self.a, trunk([BASE]))
        write_trace(self.b, trunk([BASE[:3]]))
        self.assertEqual(run(self.a, self.b)[0], 2)

    def test_quant_drift_is_not_a_step(self):
        # error grows ~4x per layer, the shape of quantized-vs-f32 compounding
        def off(e):
            return [BASE[0], BASE[1] + e, BASE[2], BASE[3]]
        # offsets x4 per layer: 1 - cosine grows ~16x per layer (under the 20x step rule) and
        # crosses the 1e-3 floor at the last layers
        offsets = [0.0008 * 4 ** i for i in range(6)]
        write_trace(self.a, trunk([BASE] * len(offsets)))
        write_trace(self.b, trunk([off(d) for d in offsets]))
        code, text = run("--mode", "quant", self.a, self.b)
        self.assertEqual(code, 0, text)
        self.assertIn("DRIFT", text)

    def test_quant_step_is_named(self):
        def off(e):
            return [BASE[0], BASE[1] + e, BASE[2], BASE[3]]
        write_trace(self.a, trunk([BASE] * 4))
        write_trace(self.b, trunk([off(1e-4), off(1e-4), off(2.0), off(2.0)]))
        code, text = run("--mode", "quant", self.a, self.b)
        self.assertEqual(code, 1)
        self.assertIn("STEP at layer@2", text)

    @unittest.skipUnless(os.path.exists(os.path.join(BISECT, "layer-diff-before.txt")), "banked bisect absent")
    def test_glm53_bisect_before_is_a_step_at_layer_3_in_the_moe_block(self):
        code, text = run("--table", "--mode", "quant", os.path.join(BISECT, "layer-diff-before.txt"))
        self.assertEqual(code, 1)
        self.assertTrue(text.startswith("STEP at layer@3"), text)
        for stage in ("route@3", "routed@3", "ffn@3"):
            self.assertIn(stage, text)

    @unittest.skipUnless(os.path.exists(os.path.join(BISECT, "layer-diff-after.txt")), "banked bisect absent")
    def test_glm53_bisect_after_is_drift_only(self):
        code, text = run("--table", "--mode", "quant", os.path.join(BISECT, "layer-diff-after.txt"))
        self.assertEqual(code, 0)
        self.assertIn("DRIFT, no step", text)


if __name__ == "__main__":
    unittest.main()
