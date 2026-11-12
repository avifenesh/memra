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

    def test_f32_names_the_first_stage_over_rel(self):
        small = [v * (1 + 1e-7) for v in BASE]
        big = [v * (1 + 1e-3) for v in BASE]
        write_trace(self.a, trunk([BASE, BASE, BASE]))
        write_trace(self.b, trunk([small, big, big]))
        code, text = run("--mode", "f32", self.a, self.b)
        self.assertEqual(code, 1)
        self.assertIn("FIRST DIVERGENCE at mixer@1", text)
        self.assertEqual(run("--mode", "f32", "--rel", "1e-2", self.a, self.b)[0], 0)

    def test_quant_nan_is_a_step_not_clean(self):
        nan = [float("nan")] * len(BASE)
        write_trace(self.a, trunk([BASE, BASE, BASE]))
        write_trace(self.b, trunk([BASE, nan, nan]))
        code, text = run("--mode", "quant", self.a, self.b)
        self.assertEqual(code, 1, text)
        self.assertIn("STEP at layer@1", text)

    def test_quant_zero_output_is_a_step_not_clean(self):
        zero = [0.0] * len(BASE)
        write_trace(self.a, trunk([BASE, BASE]))
        write_trace(self.b, trunk([BASE, zero]))
        self.assertEqual(run("--mode", "quant", self.a, self.b)[0], 1)

    def test_quant_checks_collapse_and_logits(self):
        flipped = [-v for v in BASE]
        a = trunk([BASE, BASE]) + [("logits", -1, BASE)]
        b = trunk([BASE, BASE]) + [("logits", -1, flipped)]
        write_trace(self.a, a)
        write_trace(self.b, b)
        code, text = run("--mode", "quant", self.a, self.b)
        self.assertEqual(code, 1, text)
        self.assertIn("STEP at logits@-1", text)

    def test_quant_names_an_embedding_fault_at_expand(self):
        moved = [BASE[0], BASE[1], -BASE[2], BASE[3]]
        a = trunk([BASE, BASE])
        b = [("expand", -1, moved)] + trunk([moved, moved])[1:]
        write_trace(self.a, a)
        write_trace(self.b, b)
        self.assertIn("STEP at expand@-1", run("--mode", "quant", self.a, self.b)[1])

    def test_a_truncated_candidate_is_incomparable(self):
        full = trunk([BASE, BASE, BASE, BASE])
        cut = [st for st in full if st[1] in (-1, 0) and st[0] != "collapse"]
        write_trace(self.a, full)
        write_trace(self.b, cut)
        self.assertEqual(run(self.a, self.b)[0], 2)
        self.assertEqual(run(self.b, self.a)[0], 2)

    def test_one_sided_expand_and_layer_hidden_are_allowed(self):
        # The engine side writes expand and no layer_hidden; reference `execute` the reverse.
        reference = [st for st in trunk([BASE, BASE]) if st[0] != "expand"]
        engine = [st for st in trunk([BASE, BASE]) if st[0] != "layer_hidden"]
        write_trace(self.a, reference)
        write_trace(self.b, engine)
        self.assertEqual(run(self.a, self.b)[0], 0)

    def test_one_sided_layer_hidden_is_allowed(self):
        engine = [st for st in trunk([BASE, BASE]) if st[0] != "layer_hidden"]
        write_trace(self.a, trunk([BASE, BASE]))
        write_trace(self.b, engine)
        self.assertEqual(run(self.a, self.b)[0], 0)
        self.assertEqual(run("--one-sided", "", self.a, self.b)[0], 2)

    def test_malformed_values_are_incomparable(self):
        write_trace(self.a, trunk([BASE]))
        with open(self.b, "w") as f:
            f.write("format\tmemra-hidden-trace-v1\ntokens\t1,2,3\n")
            f.write(f"stage\texpand\t-1\t4\t{hexrow(BASE)[:-4]}\n")
        self.assertEqual(run(self.a, self.b)[0], 2)
        with open(self.b, "w") as f:
            f.write("format\tmemra-hidden-trace-v1\ntokens\n")
        self.assertEqual(run(self.a, self.b)[0], 2)

    def test_a_table_without_rows_is_incomparable(self):
        with open(self.a, "w") as f:
            f.write("no table here\njust prose\n")
        self.assertEqual(run("--table", "--mode", "quant", self.a)[0], 2)

    def test_a_repeated_stage_keeps_its_first_place(self):
        # mixer@0 is emitted again after layer 1 (a later chunk): it stays layer 0's first
        # stage, so a change in layer 0's mixer is named before layer 1.
        moved = [v * 1.001 for v in BASE]
        a = [("mixer", 0, BASE), ("layer", 0, BASE), ("mixer", 1, BASE), ("layer", 1, BASE), ("mixer", 0, BASE)]
        # Both of layer 0's stages differ: mixer runs first, so it is the one named. Moving a
        # repeated key to its last position would put mixer after layer and name layer@0.
        b = [("mixer", 0, moved), ("layer", 0, moved), ("mixer", 1, moved), ("layer", 1, BASE), ("mixer", 0, moved)]
        write_trace(self.a, a)
        write_trace(self.b, b)
        self.assertIn("FIRST DIVERGENCE at mixer@0", run(self.a, self.b)[1])

    def test_mtp_layers_sort_after_the_trunk(self):
        # MTP taps carry layer index trunk + depth and run after the collapse: a collapse
        # fault is named at collapse, not at the MTP layer.
        moved = [v * 1.001 for v in BASE]
        a = [("expand", -1, BASE), ("layer", 0, BASE), ("layer", 1, BASE), ("collapse", -1, BASE), ("layer", 2, BASE)]
        b = [("expand", -1, BASE), ("layer", 0, BASE), ("layer", 1, BASE), ("collapse", -1, moved), ("layer", 2, moved)]
        write_trace(self.a, a)
        write_trace(self.b, b)
        self.assertIn("FIRST DIVERGENCE at collapse@-1", run(self.a, self.b)[1])

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
