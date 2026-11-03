#!/usr/bin/env python3
"""Unit tests for the impact-map delta logic, on a synthetic repository and map.

Run: python3 tools/impact_map/test_delta.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import delta  # noqa: E402
import ledger  # noqa: E402
import rust_items  # noqa: E402

ENGINE = "crates/memra-engine/src/model.rs"
KERNEL = "crates/memra-engine/cu/k.cu"

RUST_V1 = """\
pub fn shared(x: u32) -> u32 {
    x + 1
}

pub fn forward(family: u32, x: u32) -> u32 {
    if family == 1 {
        // qwen35 arm
        shared(x) * 2
    } else {
        // deepseek arm
        shared(x) * 3
    }
}

pub fn only_dsv4(x: u32) -> u32 {
    x * 7
}
"""

CU_V1 = """\
extern "C" __global__ void k_add(float* y) {
    y[threadIdx.x] += 1.0f;
}
"""


def run(repo, *cmd):
    return subprocess.run(["git", "-C", str(repo), *cmd], check=True, capture_output=True, text=True).stdout.strip()


def commit(repo, files, msg):
    for path, text in files.items():
        p = Path(repo) / path
        p.parent.mkdir(parents=True, exist_ok=True)
        if text is None:
            p.unlink()
        else:
            p.write_text(text)
    run(repo, "add", "-A")
    run(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", msg)
    return run(repo, "rev-parse", "HEAD")


def build_map(repo, ref):
    """A hand-labelled map over the fixture: forward's two arms are family-narrowed regions."""
    items = {}
    sets = ["*",
            {"qwen35": {"plain": ["rtx5090"]}, "deepseek_v4": {"plain": ["rtxpro6000"]}},
            {"deepseek_v4": {"plain": ["rtxpro6000"]}},
            {"qwen35": {"plain": ["rtx5090"]}, "deepseek_v4": {"plain": ["rtxpro6000"]}}]
    reach_of = {"shared": 1, "forward": 1, "only_dsv4": 2, "k_add": 3}
    for path in (ENGINE, KERNEL):
        text = delta.blob(repo, ref, path)
        for i, (h, a, b, test, kind) in delta.scan(path, text).items():
            name = i.split("::")[-1]
            items[i] = {"f": path, "a": a, "b": b, "h": h, "k": kind, "r": reach_of.get(name, 0),
                        "e": ["numeric", "speed"], "s": "test"}
    regions = [
        {"file": ENGINE, "a": 6, "b": 8, "kind": "family", "families": ["qwen35"], "routes": ["*"], "devices": ["*"],
         "effects": ["numeric"], "claim": "t:R0"},
        {"file": ENGINE, "a": 9, "b": 12, "kind": "family", "families": ["deepseek_v4"], "routes": ["*"], "devices": ["*"],
         "effects": ["numeric"], "claim": "t:R1"},
    ]
    return {"schema": "memra-impact-map-v1", "source_commit": run(repo, "rev-parse", ref),
            "vocab": {"families": ["qwen35", "deepseek_v4"], "routes": ["plain", "mtp"], "devices": ["rtx5090", "rtxpro6000"]},
            "reach_sets": sets, "items": items, "regions": regions, "stats": {}}


class DeltaTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        run(self.repo, "init", "-q")
        self.base = commit(self.repo, {ENGINE: RUST_V1, KERNEL: CU_V1, "docs/a.md": "x\n"}, "base")
        m = build_map(self.repo, self.base)
        models = {"families": {"qwen35": [{"id": "qwen/qwen3.8-27b", "devices": ["rtx5090"], "routes": ["plain", "mtp"], "class": "vendor"}],
                               "deepseek_v4": [{"id": "deepseek-v4", "devices": ["rtxpro6000"], "routes": ["plain"], "class": "vendor"}]}}
        self.base = commit(self.repo, {delta.MAP_PATH: json.dumps(m), delta.MODELS_PATH: json.dumps(models)}, "map")

    def tearDown(self):
        self.tmp.cleanup()

    def cells(self, head):
        rep = delta.delta(self.repo, self.base, head)
        return rep, set(rep["cells"])

    def test_docs_only_reaches_nothing(self):
        head = commit(self.repo, {"docs/a.md": "y\n"}, "docs")
        rep, cells = self.cells(head)
        self.assertEqual(cells, set())

    def test_comment_only_reaches_nothing(self):
        head = commit(self.repo, {ENGINE: RUST_V1.replace("// qwen35 arm", "// the qwen35 arm, reworded")}, "comment")
        rep, cells = self.cells(head)
        self.assertEqual(cells, set())

    def test_rustfmt_only_reaches_nothing(self):
        head = commit(self.repo, {ENGINE: RUST_V1.replace("x + 1", "x\n        + 1")}, "fmt")
        _, cells = self.cells(head)
        self.assertEqual(cells, set())

    def test_change_inside_one_arm_narrows_to_that_family(self):
        head = commit(self.repo, {ENGINE: RUST_V1.replace("shared(x) * 3", "shared(x) * 4")}, "dsv4 arm")
        rep, cells = self.cells(head)
        self.assertEqual(cells, {("deepseek_v4", "plain", "rtxpro6000")})
        src = [it["source"] for f in rep["files"] for it in f.get("items", [])]
        self.assertEqual(src, ["map-region"])

    def test_change_outside_arms_takes_item_reach(self):
        head = commit(self.repo, {ENGINE: RUST_V1.replace("if family == 1", "if family == 2")}, "predicate")
        _, cells = self.cells(head)
        self.assertEqual(cells, {("qwen35", "plain", "rtx5090"), ("deepseek_v4", "plain", "rtxpro6000")})

    def test_family_local_item(self):
        head = commit(self.repo, {ENGINE: RUST_V1.replace("x * 7", "x * 8")}, "dsv4 only")
        _, cells = self.cells(head)
        self.assertEqual(cells, {("deepseek_v4", "plain", "rtxpro6000")})

    def test_new_item_takes_file_union(self):
        head = commit(self.repo, {ENGINE: RUST_V1 + "\npub fn brand_new() -> u32 { 1 }\n"}, "new fn")
        rep, cells = self.cells(head)
        self.assertEqual(cells, {("qwen35", "plain", "rtx5090"), ("deepseek_v4", "plain", "rtxpro6000")})
        self.assertIn("map-file", [it["source"] for f in rep["files"] for it in f.get("items", [])])

    def test_unknown_file_is_wide(self):
        head = commit(self.repo, {"crates/memra-engine/src/new_mod.rs": "pub fn f() {}\n"}, "new file")
        _, cells = self.cells(head)
        self.assertEqual(len(cells), 2 * 2 * 2)

    def test_kernel_change_needs_kernel_check(self):
        head = commit(self.repo, {KERNEL: CU_V1.replace("1.0f", "2.0f")}, "kernel")
        rep, cells = self.cells(head)
        self.assertTrue(rep["kernel"])
        tests = delta.tests_for(rep)
        self.assertIn("kernel-check", tests[("qwen35", "plain", "rtx5090")])

    def test_stale_map_does_not_narrow_by_region(self):
        # a change moves forward's lines first; a later edit inside the dsv4 arm must not use the
        # stale region lines, so it takes the whole item's reach
        mid = commit(self.repo, {ENGINE: "\n\n" + RUST_V1}, "shift lines")
        head = commit(self.repo, {ENGINE: "\n\n" + RUST_V1.replace("shared(x) * 3", "shared(x) * 4")}, "dsv4 arm after shift")
        rep = delta.delta(self.repo, mid, head, map_ref=self.base)
        self.assertEqual(set(rep["cells"]), {("qwen35", "plain", "rtx5090"), ("deepseek_v4", "plain", "rtxpro6000")})

    def test_plan_picks_one_model_per_family_and_device(self):
        head = commit(self.repo, {ENGINE: RUST_V1.replace("x + 1", "x + 2")}, "shared")
        rep = delta.delta(self.repo, self.base, head)
        pl = delta.plan(self.repo, self.base, rep)
        self.assertEqual(sorted(pl["runs"]), ["rtx5090", "rtxpro6000"])
        self.assertEqual(sorted(pl["runs"]["rtx5090"]), ["qwen/qwen3.8-27b"])
        self.assertIn("run-gen-argmax", pl["runs"]["rtx5090"]["qwen/qwen3.8-27b"]["tests"])
        self.assertEqual(pl["runs"]["rtx5090"]["qwen/qwen3.8-27b"]["routes"], ["plain"])
        self.assertEqual(pl["uncovered"], [])

    def test_fingerprint_moves_only_on_path(self):
        m = delta.Map(build_map(self.repo, self.base))
        sc = ledger.TreeScanner(self.repo)
        dsv4 = ("deepseek_v4", "plain", "rtxpro6000")
        q35 = ("qwen35", "plain", "rtx5090")

        def fp(ref, cell):
            return delta.cell_fingerprints(m, sc.hashes(ref), {cell})[cell]
        fp0 = fp(self.base, dsv4)
        head = commit(self.repo, {KERNEL: CU_V1.replace("1.0f", "3.0f")}, "kernel only")
        self.assertNotEqual(fp0, fp(head, dsv4))
        head2 = commit(self.repo, {"docs/a.md": "z\n"}, "docs")
        self.assertEqual(fp(head, dsv4), fp(head2, dsv4))
        q0 = fp(head2, q35)
        head3 = commit(self.repo, {ENGINE: RUST_V1.replace("x * 7", "x * 9")}, "dsv4-only item")
        self.assertEqual(q0, fp(head3, q35))
        self.assertNotEqual(fp(head2, dsv4), fp(head3, dsv4))


    def test_plan_covers_routes_with_fewest_models_and_lists_uncovered(self):
        models = {"families": {"qwen35": [
            {"id": "a-vendor", "devices": ["rtx5090"], "routes": ["plain", "mtp"], "class": "vendor"},
            {"id": "b-own", "devices": ["rtx5090"], "routes": ["plain"], "class": "own"}]}}
        rep = {"cells": {("qwen35", "plain", "rtx5090"): {"why": {"x"}, "effects": {"numeric"}},
                         ("qwen35", "mtp", "rtx5090"): {"why": {"x"}, "effects": {"numeric"}},
                         ("qwen35", "vision", "rtx5090"): {"why": {"x"}, "effects": {"numeric"}},
                         ("deepseek_v4", "plain", "rtx5090"): {"why": {"x"}, "effects": {"speed"}}},
               "effects": {"numeric", "speed"}, "kernel": False}
        pl = delta.plan(self.repo, self.base, rep, models=models, ledger={"records": []})
        self.assertEqual(list(pl["runs"]["rtx5090"]), ["a-vendor"])
        self.assertEqual(pl["runs"]["rtx5090"]["a-vendor"]["tests"], ["run-gen-argmax", "run-spec"])
        self.assertEqual(sorted((u["family"], u["route"]) for u in pl["uncovered"]),
                         [("deepseek_v4", "plain"), ("qwen35", "vision")])

    def test_plan_skips_what_the_ledger_proves_for_the_same_fingerprint(self):
        models = {"families": {"qwen35": [{"id": "m", "devices": ["rtx5090"], "routes": ["plain"], "class": "own"}]}}
        rep = {"cells": {("qwen35", "plain", "rtx5090"): {"why": {"x"}, "effects": {"numeric"}}},
               "effects": {"numeric"}, "kernel": False, "fingerprints": {("qwen35", "plain", "rtx5090"): "fp1"}}
        led = {"records": [{"model": "m", "route": "plain", "device": "rtx5090", "test": "run-gen-argmax", "fingerprint": "fp1"}]}
        pl = delta.plan(self.repo, self.base, rep, models=models, ledger=led)
        self.assertEqual(pl["runs"], {})
        self.assertEqual(len(pl["proven"]), 1)
        led["records"][0]["fingerprint"] = "older"
        pl = delta.plan(self.repo, self.base, rep, models=models, ledger=led)
        self.assertEqual(pl["runs"]["rtx5090"]["m"]["tests"], ["run-gen-argmax"])


    def test_push_tier_is_one_device_per_family_and_push_gated_target(self):
        models = {"families": {"qwen35": [{"id": "m", "devices": ["rtx5090", "rtxpro6000", "h100"], "routes": ["plain"], "class": "own"}]}}
        cells = {("qwen35", "plain", d): {"why": {"x"}, "effects": {"numeric"}} for d in ("rtx5090", "rtxpro6000", "h100")}
        rep = {"cells": cells, "effects": {"numeric"}, "kernel": False}
        pl = delta.plan(self.repo, self.base, rep, models=models, ledger={"records": []},
                        device_target={"rtx5090": "120a", "rtxpro6000": "120a", "h100": "90a"}, push_gated={"120a"})
        self.assertEqual(list(pl["runs"]), ["rtx5090"])
        self.assertEqual(pl["daily"], {"h100": ["qwen35"], "rtxpro6000": ["qwen35"]})


class ScannerTest(unittest.TestCase):
    def test_raw_strings_and_lifetimes(self):
        src = 'fn a<\'x>(s: &\'x str) -> &\'x str { let _r = r#"}{"#; s }\nfn b() {}\n'
        names = [r["name"] for r in rust_items.scan("crates/c/src/lib.rs", src) if r["kind"] == "fn"]
        self.assertEqual(names, ["a", "b"])

    def test_unsafe_extern_block_decls(self):
        src = 'unsafe extern "C" {\n    pub fn memra_x(a: i32) -> i32;\n}\nunsafe extern "C" fn cb(p: *mut u8) {}\n'
        recs = rust_items.scan("crates/c/src/ffi.rs", src)
        self.assertEqual([(r["name"], r["container"]) for r in recs if r["kind"] == "fn"], [("memra_x", "extern"), ("cb", "mod")])

    def test_cfg_test_module_is_test(self):
        src = "fn live() {}\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn t() {}\n}\n"
        recs = {r["name"]: r["test"] for r in rust_items.scan("crates/c/src/lib.rs", src) if r["kind"] == "fn"}
        self.assertEqual(recs, {"live": False, "t": True})


if __name__ == "__main__":
    unittest.main()
