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
        for i, (h, a, b, test, kind, lh, _) in delta.scan(path, delta.blob(repo, ref, path)).items():
            name = i.split("::")[-1]
            if kind == "outside":
                continue  # the generator never places tokens outside items
            items[i] = {"f": path, "a": a, "b": b, "h": h, "l": lh, "k": kind, "r": reach_of.get(name, 0),
                        "e": ["numeric"] if name != "k_add" else ["numeric", "speed"], "s": "test"}
    return {"schema": "memra-impact-map-v1", "source_commit": run(repo, "rev-parse", ref),
            "vocab": {"families": ["qwen35", "deepseek_v4"], "routes": ["plain"], "devices": ["rtx5090", "rtxpro6000", "h100"]},
            "reach_sets": sets, "items": items, "regions": [], "stats": {}}


class Base(unittest.TestCase):
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


class GateTest(Base):

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
        # the engine edit is judged by the base's map (120a only); the map edit is a gate change
        self.assertEqual(self.cells(imp), {("120a", "numeric")} | {(t, "gate") for t in ("120a", "100a", "90a", "89")})


class GateSoundnessTest(Base):
    """Regressions from the independent review of #1032: each case once let a change skip a cell."""

    def drift(self, files, message="drift since the map"):
        # a commit between the map commit and the push base
        self.base = commit(self.repo, files, message)

    def test_shifted_lines_since_the_map_do_not_narrow_by_region(self):
        m = build_map(self.repo, self.base)
        m["regions"] = [{"file": ENGINE, "a": 6, "b": 8, "kind": "family", "families": ["qwen35"], "routes": ["*"], "devices": ["*"],
                         "effects": ["numeric"], "claim": "t:R0"}]
        self.base = commit(self.repo, {delta.MAP_PATH: json.dumps(m)}, "map with a region")
        shifted = "use std::fmt;\n\n\n" + RUST_V1 + EXTRA
        self.drift({ENGINE: shifted})
        imp = self.impact({ENGINE: shifted.replace("shared(x) * 2", "shared(x) * 5")})
        self.assertNotIn("deepseek_v4", imp["uncovered"])  # the region's stale lines did not narrow it
        self.assertEqual(self.cells(imp), {("120a", "numeric")})

    def test_a_caller_added_since_the_map_widens_the_callee(self):
        # shared (120a) starts calling hopper_only (90a in the map); a later edit of hopper_only only
        self.drift({ENGINE: (RUST_V1 + EXTRA).replace("    x + 1\n", "    hopper_only(x) + 1\n")})
        imp = self.impact({ENGINE: (RUST_V1 + EXTRA).replace("    x + 1\n", "    hopper_only(x) + 1\n").replace("x << 1", "x << 3")})
        self.assertIn(("120a", "numeric"), self.cells(imp))

    def test_arch_table_change_since_the_map_keeps_the_targets(self):
        self.drift({"crates/memra-engine/build.rs": "fn main() {}\n"})
        imp = self.impact({ENGINE: (RUST_V1 + EXTRA).replace("x << 1", "x << 2")})
        self.assertEqual({t for t, _ in self.cells(imp)}, {"120a", "100a", "90a", "89"})

    def test_production_item_moved_into_cfg_test_is_a_production_change(self):
        imp = self.impact({ENGINE: (RUST_V1 + EXTRA).replace("pub fn hopper_only", "#[cfg(test)]\npub fn hopper_only")})
        self.assertIn(("90a", "numeric"), self.cells(imp))

    def test_tokens_outside_items_keep_the_path_rule(self):
        imp = self.impact({ENGINE: "#![allow(dead_code)]\n" + RUST_V1 + EXTRA})
        self.assertEqual({t for t, _ in self.cells(imp)}, {"120a", "100a", "90a", "89"})
        imp = self.impact({KERNEL: "constexpr int kTile = 64;\n" + CU_V1})
        self.assertEqual({t for t, _ in self.cells(imp)}, {"120a", "100a", "90a", "89"})

    def test_an_item_placed_nowhere_keeps_the_path_rule(self):
        m = build_map(self.repo, self.base)
        m["reach_sets"].append({})  # a build-time door: no default cell
        for i, it in m["items"].items():
            if i.endswith("::only_dsv4"):
                it["r"] = len(m["reach_sets"]) - 1
        self.base = commit(self.repo, {delta.MAP_PATH: json.dumps(m)}, "map: only_dsv4 behind a build-time door")
        imp = self.impact({ENGINE: (RUST_V1 + EXTRA).replace("x * 7", "x * 9")})
        self.assertEqual({t for t, _ in self.cells(imp)}, {"120a", "100a", "90a", "89"})

    def test_a_file_the_scanner_does_not_read_keeps_the_path_rule(self):
        imp = self.impact({"crates/memra-engine/cu/tile.inl": "#define TILE 128\n"})
        self.assertEqual({t for t, _ in self.cells(imp)}, {"120a", "100a", "90a", "89"})

    def test_a_base_the_map_is_not_an_ancestor_of_does_not_narrow(self):
        m = build_map(self.repo, self.base)
        m["source_commit"] = "1" * 40  # a commit this repository does not have
        self.base = commit(self.repo, {delta.MAP_PATH: json.dumps(m)}, "map from elsewhere")
        imp = self.impact({ENGINE: (RUST_V1 + EXTRA).replace("x << 1", "x << 2")})
        self.assertEqual({t for t, _ in self.cells(imp)}, {"120a", "100a", "90a", "89"})

    def test_a_change_that_breaks_the_map_is_refused(self):
        head = commit(self.repo, {delta.MAP_PATH: "{not json"}, "broken map")
        with self.assertRaises(gate.MapError):
            gate.Refiner(self.repo, self.base, head, RULES, ARCH)

    def test_an_unreadable_base_map_falls_back_to_the_path_rules(self):
        self.base = commit(self.repo, {delta.MAP_PATH: json.dumps({"schema": "memra-impact-map-v1"})}, "truncated map")
        imp = self.impact({ENGINE: (RUST_V1 + EXTRA).replace("x << 1", "x << 2")})
        self.assertEqual({t for t, _ in self.cells(imp)}, {"120a", "100a", "90a", "89"})

    def test_the_environment_cannot_replace_the_map(self):
        import os
        m = build_map(self.repo, self.base)
        for it in m["items"].values():
            it["r"] = 3  # every item h100-only
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        p = Path(other.name) / "elsewhere.json"
        p.write_text(json.dumps(m))
        os.environ["MEMRA_IMPACT_MAP_FILE"] = str(p)
        try:
            imp = self.impact({ENGINE: (RUST_V1 + EXTRA).replace("x + 1", "x + 2")})
        finally:
            del os.environ["MEMRA_IMPACT_MAP_FILE"]
        self.assertEqual(self.cells(imp), {("120a", "numeric")})


if __name__ == "__main__":
    unittest.main()
