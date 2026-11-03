#!/usr/bin/env python3
"""The push gate with the impact map: path rules first, then item-level narrowing.

A synthetic repository carries real rules (tools/change-impact.toml), a literal arch table and a
hand-labelled map; each case commits one diff and checks the cells change_impact.classify returns
with the map refiner on. Run: python3 tools/impact_map/test_gate.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import change_impact as ci  # noqa: E402
from impact_map import gate  # noqa: E402
import delta  # noqa: E402
from test_delta import CU_V1, ENGINE, KERNEL, RUST_V1, commit, run  # noqa: E402

ROOT = HERE.parents[1]
RULES = ci.load_rules((ROOT / ci.RULES_PATH).read_text(),
                      ci.ArchTable(accepted=("120a", "100a", "90a", "89"),
                                   caps={"12.0": "120a", "12.1": "120a", "10.0": "100a", "9.0": "90a", "8.9": "89"}))
ARCH = ci.ArchTable(accepted=("120a", "100a", "90a", "89"),
                    caps={"12.0": "120a", "12.1": "120a", "10.0": "100a", "9.0": "90a", "8.9": "89"})
EXTRA = "\npub fn hopper_only(x: u32) -> u32 {\n    x << 1\n}\n"


def build_map(repo, ref):
    sets = ["*",
            {"qwen35": {"plain": ["rtx5090"]}, "deepseek_v4": {"plain": ["rtxpro6000"]}},
            {"deepseek_v4": {"plain": ["rtxpro6000"]}},
            {"qwen35": {"plain": ["h100"]}}]
    reach_of = {"shared": 1, "forward": 1, "only_dsv4": 2, "hopper_only": 3, "k_add": 1}
    items = {}
    for path in (ENGINE, KERNEL):
        for i, (h, a, b, test, kind) in delta.scan(path, delta.blob(repo, ref, path)).items():
            name = i.split("::")[-1]
            items[i] = {"f": path, "a": a, "b": b, "h": h, "k": kind, "r": reach_of.get(name, 0),
                        "e": ["numeric"] if name != "k_add" else ["numeric", "speed"], "s": "test"}
    return {"schema": "memra-impact-map-v1", "source_commit": run(repo, "rev-parse", ref),
            "vocab": {"families": ["qwen35", "deepseek_v4"], "routes": ["plain"], "devices": ["rtx5090", "rtxpro6000", "h100"]},
            "reach_sets": sets, "items": items, "regions": [], "stats": {}}


class GateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        run(self.repo, "init", "-q")
        base = commit(self.repo, {ENGINE: RUST_V1 + EXTRA, KERNEL: CU_V1}, "code")
        self.base = commit(self.repo, {delta.MAP_PATH: json.dumps(build_map(self.repo, base)),
                                       delta.MODELS_PATH: json.dumps({"families": {}})}, "map")

    def tearDown(self):
        self.tmp.cleanup()

    def impact(self, files, use_map=True):
        head = commit(self.repo, files, "change")
        changes = ci.diff_changes(self.repo, self.base, head)
        refiner = gate.Refiner(self.repo, self.base, head, RULES, ARCH) if use_map else None
        return ci.classify(changes, RULES, ARCH, refiner=refiner)

    def cells(self, impact):
        return {(c["target"], c["scope"]) for c in impact["cells"]}

    def test_without_the_map_an_engine_edit_reaches_every_target(self):
        imp = self.impact({ENGINE: (RUST_V1 + EXTRA).replace("x << 1", "x << 2")}, use_map=False)
        self.assertEqual({t for t, _ in self.cells(imp)}, {"120a", "100a", "90a", "89"})

    def test_device_narrowing_from_item_reach(self):
        imp = self.impact({ENGINE: (RUST_V1 + EXTRA).replace("x << 1", "x << 2")})
        self.assertEqual(self.cells(imp), {("90a", "numeric")})

    def test_rustfmt_reflow_is_token_identical(self):
        imp = self.impact({ENGINE: (RUST_V1 + EXTRA).replace("x + 1", "x\n        + 1")})
        self.assertEqual(self.cells(imp), set())
        self.assertEqual(imp["files"][0]["class"], "comment")

    def test_change_on_unrostered_family_only_is_uncovered(self):
        imp = self.impact({ENGINE: (RUST_V1 + EXTRA).replace("x * 7", "x * 8")})
        self.assertEqual(self.cells(imp), set())
        self.assertIn("deepseek_v4", imp["uncovered"])

    def test_shared_item_keeps_battery_on_its_target_and_drops_perf_without_speed_effect(self):
        imp = self.impact({ENGINE: (RUST_V1 + EXTRA).replace("x + 1", "x + 2")})
        self.assertEqual(self.cells(imp), {("120a", "numeric")})

    def test_kernel_with_speed_effect_keeps_perf(self):
        imp = self.impact({KERNEL: CU_V1.replace("1.0f", "2.0f")})
        self.assertEqual(self.cells(imp), {("120a", "kernel"), ("120a", "perf")})

    def test_an_item_the_map_cannot_place_keeps_the_path_rule(self):
        imp = self.impact({ENGINE: RUST_V1 + EXTRA + "\npub fn brand_new() -> u32 { 1 }\n"})
        self.assertEqual({t for t, _ in self.cells(imp)}, {"120a", "100a", "90a", "89"})

    def test_map_at_the_base_is_the_one_in_force(self):
        # a change that also rewrites the map to narrow itself is judged by the base's map
        m = build_map(self.repo, self.base)
        for it in m["items"].values():
            it["r"] = 2
        imp = self.impact({ENGINE: (RUST_V1 + EXTRA).replace("x + 1", "x + 2"), delta.MAP_PATH: json.dumps(m)})
        self.assertEqual(self.cells(imp), {("120a", "numeric")})


if __name__ == "__main__":
    unittest.main()
