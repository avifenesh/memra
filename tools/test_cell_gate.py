#!/usr/bin/env python3
"""Push-gate decisions on per-cell evidence. CPU fixtures only: the records here are
fabricated in a temporary repository and are never GPU evidence."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cell_evidence as ce  # noqa: E402
import release_qualification as q  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BUILD_RS = '''fn detect_arch() -> String {
    let arch = match cap.as_deref() {
        Some("12.0") | Some("12.1") => "120a",
        Some("10.0") => "100a",
        Some("9.0") => "90a",
        Some("8.9") => "89",
        _ => "120a",
    };
    arch.to_string()
}
fn main() {
    assert!(
        matches!(cuda_arch.as_str(), "120a" | "100a" | "90a" | "89"),
        "MEMRA_CUDA_ARCH must be 120a (default), 100a (B200), 90a (Hopper), or 89 (portable eval)"
    );
}
'''
DEVICE = {"name": "NVIDIA GeForce RTX 5090 Laptop GPU", "uuid_sha256": "0" * 64,
          "compute_cap": "12.0", "driver": "CPU fixture"}


class Repo:
    def __init__(self, root):
        self.root = root
        self.repo = root / "repo"
        self.repo.mkdir()
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "CPU fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.git("config", "core.hooksPath", "/dev/null")
        files = {"crates/memra-engine/build.rs": BUILD_RS,
                 "crates/memra-engine/src/lib.rs": "pub fn f() {}\n",
                 "crates/memra-engine/cu/qmatvec.cu": "__global__ void k() {}\n",
                 "crates/memra-engine/cu/mmq_fp4_stub.cu": "__global__ void stub() {}\n",
                 "crates/memra-server/src/worker.rs": "pub fn step() {}\n",
                 "research/INDEX.md": "# Research\n",
                 "README.md": "# fixture\n",
                 "tools/release-roster.tsv": "own\tmodel-a\t/models/a.gguf\nvendor\tmodel-b\t/models/b.gguf\n"}
        for path, text in files.items():
            self.write(path, text)
        for name in ("change-impact.toml", *[Path(m).name for m in ce.MANIFESTS]):
            shutil.copy2(ROOT / "tools" / name, self.repo / "tools" / name)
        self.commit("seed")
        self.base = self.head()

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], check=True, capture_output=True).stdout

    def write(self, path, text):
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def commit(self, message):
        self.git("add", "-A")
        self.git("commit", "-qm", message)
        return self.head()

    def head(self):
        return self.git("rev-parse", "HEAD").decode().strip()

    def push(self, remote="refs/heads/main", oid=None, base=None):
        oid = oid or self.head()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            q.check_push(self.repo, [f"refs/heads/x {oid} {remote} {base or self.base}"], "qualified")
        return out.getvalue()

    def record(self, commit, kinds, *, device_class="rtx5090", device=DEVICE, target="120a", slowdown=0.0,
               name=None, accept=None, serve_tail="serve-smoke: 0 failed", perf_base=None, models=("model-a", "model-b")):
        out = self.root / f"evidence-{len(list(self.root.glob('evidence-*')))}"
        (out / "battery").mkdir(parents=True)
        shutil.copy2(self.repo / "tools/release-roster.tsv", out / "roster.tsv")
        meta = {"commit": commit, "inputs": ce.inputs_digest(self.repo, commit), "target": target,
                "device_class": device_class, "device": dict(device), "kinds": list(kinds),
                "src": str(self.repo), "binaries": {}, "started": "fixture", "accept_regression": accept}
        if "battery" in kinds:
            coverage = ce.coverage_module()
            required = coverage.required_cells([self.repo / m for m in ce.MANIFESTS])
            logs = [("kernel", "kernel", "".join(f"{c} CPU-fixture OK\n" for c in sorted(required))
                     + f"ALL GREEN ({len(required)} cells, 0 skipped)\n")]
            spec = "".join(f"[generate_spec K={k}] 32 tok\nself-consistency: PASS (identical to plain target)\n"
                           for k in range(1, 9)) + "=== SELF-CONSISTENCY PASS ===\n"
            for model in ("model-a", "model-b"):
                logs.append(("argmax", model, "SUMMARY flips=0 bad=0\n  PASS: CPU fixture calibrated verdict\n"))
                logs.append(("spec", model, spec))
            rows = []
            for seq, (kind, model, text) in enumerate(logs, 1):
                (out / "battery" / f"{seq}-{kind}.log").write_text(text)
                rows.append(f"{kind}\t{model}\t0\t{seq}-{kind}.log\n")
            (out / "battery/runs.tsv").write_text("".join(rows))
            meta["battery_exit"] = 0
        if "serving" in kinds:
            (out / "serving.log").write_text(f"PASS chat\n{serve_tail}\n")
            meta["serving_exit"] = 0
        if "perf" in kinds:
            (out / "perf").mkdir()
            runs = []
            for seq, arm in enumerate(ce.PERF_ORDER * len(models), 1):
                model = models[(seq - 1) // len(ce.PERF_ORDER)]
                rate = 100.0 + seq * 0.1 - (slowdown if arm == "B" else 0.0)
                log = f"perf/{seq:02d}-{arm}-{model}.log"
                (out / log).write_text(f"generated 128 tokens in 1.000s = {rate:.2f} tok/s (ST decode)\n")
                runs.append({"arm": arm, "model": model, "log": log})
            meta.update({"base": perf_base or self.base, "perf_baseline": True, "perf_runs": runs})
        record = ce.seal(out, meta)
        if record["status"] != "pass":
            return record
        dest = ce.bank(self.repo, out, name)
        self.commit(f"data: bank {dest.name}")
        return record


class CellGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.f = Repo(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_docs_research_and_cpu_need_no_device_evidence(self):
        self.f.write("research/lane/run.log", "raw\n")
        self.f.write("tools/fast-gate/plan.py", "print(1)\n")
        self.f.commit("research and tooling")
        self.assertIn("NO PUSH-GATED CELL", self.f.push())

    def test_kernel_change_without_evidence_names_the_cell(self):
        self.f.write("crates/memra-engine/cu/qmatvec.cu", "__global__ void k() { int x = 1; }\n")
        self.f.commit("kernel")
        with self.assertRaisesRegex(q.GateError, r"sm_120a \[kernel,perf\] needs battery\+perf evidence from one of: "
                                                 r"gb10, rtx5090, rtxpro6000"):
            self.f.push()

    def test_change_gated_to_other_targets_needs_nothing_at_push(self):
        self.f.write("crates/memra-engine/cu/mmq_fp4_stub.cu", "__global__ void stub() { __trap(); }\n")
        self.f.commit("stub")
        out = self.f.push()
        self.assertIn("NO PUSH-GATED CELL", out)
        self.assertIn("DEFERRED sm_90a", out)
        self.assertIn("DEFERRED sm_89", out)
        self.assertNotIn("sm_120a", out)

    def test_comment_only_kernel_edit_needs_nothing(self):
        self.f.write("crates/memra-engine/cu/qmatvec.cu", "// why\n__global__ void k() {}\n")
        self.f.commit("comment")
        self.assertIn("change impact: none", self.f.push())

    def test_matching_evidence_admits_and_partial_evidence_refuses(self):
        self.f.write("crates/memra-engine/cu/qmatvec.cu", "__global__ void k() { int x = 2; }\n")
        code = self.f.commit("kernel")
        self.f.record(code, ["battery"])
        with self.assertRaisesRegex(q.GateError, r"sm_120a \[kernel,perf\] needs perf evidence"):
            self.f.push()
        self.f.record(code, ["perf"], device_class="rtxpro6000",
                      device={**DEVICE, "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"})
        out = self.f.push()
        self.assertIn("CELL-QUALIFIED", out)
        self.assertIn("sm_120a battery,perf", out)

    def test_serving_change_needs_serve_smoke(self):
        self.f.write("crates/memra-server/src/worker.rs", "pub fn step() { let _ = 1; }\n")
        code = self.f.commit("scheduler")
        self.f.record(code, ["battery", "perf"])
        with self.assertRaisesRegex(q.GateError, r"needs serving evidence"):
            self.f.push()
        failed = self.f.record(code, ["serving"], serve_tail="serve-smoke: 2 failed")
        self.assertEqual(failed["status"], "fail")
        self.f.record(code, ["serving"])
        self.assertIn("CELL-QUALIFIED", self.f.push())

    def test_evidence_for_other_inputs_is_not_admitted(self):
        self.f.write("crates/memra-engine/cu/qmatvec.cu", "__global__ void k() { int x = 3; }\n")
        old = self.f.commit("kernel v1")
        self.f.record(old, ["battery", "perf"])
        self.f.write("crates/memra-engine/cu/qmatvec.cu", "__global__ void k() { int x = 4; }\n")
        self.f.commit("kernel v2")
        with self.assertRaisesRegex(q.GateError, "missing evidence"):
            self.f.push()

    def test_tampered_record_is_not_admitted(self):
        self.f.write("crates/memra-engine/cu/qmatvec.cu", "__global__ void k() { int x = 5; }\n")
        code = self.f.commit("kernel")
        self.f.record(code, ["battery", "perf"], name="tamper")
        log = next((self.f.repo / ce.CELLS / "tamper/battery").glob("2-argmax.log"))
        log.write_text("SUMMARY flips=1 bad=1\nFAIL\n")
        self.f.commit("edit raw evidence")
        out = io.StringIO()
        with self.assertRaisesRegex(q.GateError, "missing evidence"), contextlib.redirect_stdout(out):
            q.check_push(self.f.repo, [f"refs/heads/x {self.f.head()} refs/heads/main {self.f.base}"], "qualified")
        self.assertIn("evidence changed: battery/2-argmax.log", out.getvalue())

    def test_device_of_another_target_does_not_count(self):
        self.f.write("crates/memra-engine/cu/qmatvec.cu", "__global__ void k() { int x = 6; }\n")
        code = self.f.commit("kernel")
        self.f.record(code, ["battery", "perf"], device_class="h100",
                      device={**DEVICE, "name": "NVIDIA H100 80GB HBM3", "compute_cap": "9.0"})
        out = io.StringIO()
        with self.assertRaisesRegex(q.GateError, "missing evidence"), contextlib.redirect_stdout(out):
            q.check_push(self.f.repo, [f"refs/heads/x {self.f.head()} refs/heads/main {self.f.base}"], "qualified")
        self.assertIn("device, build target and record target disagree", out.getvalue())

    def test_perf_regression_refuses_unless_accepted(self):
        self.f.write("crates/memra-engine/cu/qmatvec.cu", "__global__ void k() { int x = 7; }\n")
        code = self.f.commit("kernel")
        self.f.record(code, ["battery"])
        slow = self.f.record(code, ["perf"], slowdown=20.0)
        self.assertEqual(slow["status"], "fail")
        self.assertIn("REGRESSED on model-a, model-b", slow["failures"][0])
        with self.assertRaisesRegex(q.GateError, "needs perf evidence"):
            self.f.push()
        self.f.record(code, ["perf"], slowdown=20.0, accept="reviewed: correctness fix costs 20%")
        self.assertIn("CELL-QUALIFIED", self.f.push())

    def test_separated_slowdown_inside_the_noise_floor_is_admitted(self):
        # B is about 1% slower on every run and every B run is below every A run: rank alone
        # called this a regression, yet the recorded run-to-run noise reaches it on identical code.
        self.f.write("crates/memra-engine/cu/qmatvec.cu", "__global__ void k() { int x = 13; }\n")
        code = self.f.commit("kernel")
        self.f.record(code, ["battery"])
        small = self.f.record(code, ["perf"], slowdown=1.0)
        self.assertEqual(small["status"], "pass", small["failures"])
        self.assertTrue(all(-ce.PERF_MIN_REGRESSION_PCT < v["delta_pct"] < 0 for v in small["kinds"]["perf"]["verdicts"].values()))
        self.assertIn("CELL-QUALIFIED", self.f.push())

    def test_perf_must_compare_against_the_push_base_and_cover_the_roster(self):
        self.f.write("crates/memra-engine/cu/qmatvec.cu", "__global__ void k() { int x = 10; }\n")
        inside = self.f.commit("kernel step 1")
        self.f.write("crates/memra-engine/cu/qmatvec.cu", "__global__ void k() { int x = 11; }\n")
        code = self.f.commit("kernel step 2")
        self.f.record(code, ["battery"])
        self.f.record(code, ["perf"], perf_base=inside, name="inside")
        self.f.record(code, ["perf"], models=("model-a",), name="short")
        out = io.StringIO()
        with self.assertRaisesRegex(q.GateError, "needs perf evidence"), contextlib.redirect_stdout(out):
            q.check_push(self.f.repo, [f"refs/heads/x {self.f.head()} refs/heads/main {self.f.base}"], "qualified")
        self.assertIn("is not this push's base", out.getvalue())
        self.assertIn("do not cover every roster model", out.getvalue())
        self.f.record(code, ["perf"])
        self.assertIn("CELL-QUALIFIED", self.f.push())

    def test_a_push_cannot_relax_its_own_rules(self):
        rules = self.f.repo / "tools/change-impact.toml"
        rules.write_text(rules.read_text().replace('name = "sm_120a"\npush_gate = true', 'name = "sm_120a"\npush_gate = false'))
        self.f.write("crates/memra-engine/cu/qmatvec.cu", "__global__ void k() { int x = 12; }\n")
        self.f.commit("turn the gate off and change a kernel")
        with self.assertRaisesRegex(q.GateError, r"sm_120a \[gate,kernel,perf\] needs battery\+perf\+serving"):
            self.f.push()

    def test_card_keyed_default_needs_that_card(self):
        self.f.write("crates/memra-server/src/worker.rs",
                     'pub fn step(n: &str) { if n.contains("RTX PRO 6000 Blackwell") { vmm(); } }\n')
        code = self.f.commit("PRO 6000 default")
        self.f.record(code, ["battery", "serving", "perf"])
        with self.assertRaisesRegex(q.GateError, r"defaults keyed on rtxpro6000\] needs battery\+perf evidence from rtxpro6000"):
            self.f.push()
        pro = {**DEVICE, "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition"}
        self.f.record(code, ["battery", "perf"], device_class="rtxpro6000", device=pro)
        self.assertIn("CELL-QUALIFIED", self.f.push())

    def test_tags_still_need_the_full_record_and_development_is_unchanged(self):
        self.f.write("research/lane/run.log", "raw\n")
        self.f.commit("research")
        with self.assertRaisesRegex(q.GateError, "no committed native qualification pointer"):
            self.f.push("refs/tags/v1.0.0")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            q.check_push(self.f.repo, [f"refs/heads/x {self.f.head()} refs/heads/topic {self.f.base}"], "development")
        self.assertIn("UNQUALIFIED DEVELOPMENT", out.getvalue())

    def test_impact_targets_trailer_narrows_and_evidence_commits_do_not_reset_it(self):
        self.f.write("crates/memra-engine/cu/qmatvec.cu", "__global__ void k() { int x = 9; }\n")
        self.f.git("add", "-A")
        self.f.git("commit", "-qm", "fix: Hopper arm\n\nImpact-Targets: 90a\nImpact-Reason: the edit sits in a 900-only branch")
        out = self.f.push()
        self.assertIn("narrowed by Impact-Targets: 90a", out)
        self.assertIn("NO PUSH-GATED CELL", out)
        self.f.write(ce.CELLS + "note/record.json", "{}\n")
        self.f.commit("data: metadata only")
        self.assertIn("NO PUSH-GATED CELL", self.f.push())
        self.f.write("research/lane/run.log", "raw\n")
        self.f.commit("an undeclared commit")
        with self.assertRaisesRegex(q.GateError, r"sm_120a \[kernel,perf\]"):
            self.f.push()

    def test_new_branch_compares_against_origin_main_or_everything(self):
        self.f.write("crates/memra-engine/cu/qmatvec.cu", "__global__ void k() { int x = 8; }\n")
        self.f.commit("kernel")
        # No origin/main ref: the empty tree is the base, so every path is new.
        with self.assertRaisesRegex(q.GateError, "missing evidence"):
            self.f.push("refs/heads/topic", base="0" * 40)
        self.f.git("update-ref", "refs/remotes/origin/main", self.f.head())
        self.f.write("research/lane/run.log", "raw\n")
        self.f.commit("research only past origin/main")
        self.assertIn("NO PUSH-GATED CELL", self.f.push("refs/heads/topic", base="0" * 40))

    def test_rebased_topic_compares_against_origin_main_not_its_old_tip(self):
        # The topic was pushed on the seed, then main took a kernel change, then the topic
        # was rebased onto main. Only the topic's own research change is new; main's kernel
        # change is main's, and judging it against the old tip demanded evidence for it.
        self.f.write("research/lane/run.log", "raw\n")
        old_tip = self.f.commit("topic on the seed")
        self.f.git("reset", "-q", "--hard", self.f.base)
        self.f.write("crates/memra-engine/cu/qmatvec.cu", "__global__ void k() { int x = 9; }\n")
        self.f.git("update-ref", "refs/remotes/origin/main", self.f.commit("main moves"))
        self.f.write("research/lane/run.log", "raw\n")
        self.f.commit("topic rebased onto main")
        self.assertIn("NO PUSH-GATED CELL", self.f.push("refs/heads/topic", base=old_tip))
        # Main itself still compares against its old value.
        with self.assertRaisesRegex(q.GateError, "missing evidence"):
            self.f.push("refs/heads/main", base=self.f.base)


    def rebase_onto(self, main_change, *lane):
        """Move main by `main_change` (path -> text) on the seed, then replay the `lane` commits
        on it, as a lane rebases before landing. Returns (new main tip, new lane head)."""
        f = self.f
        f.git("checkout", "-q", "-b", "moved", f.base)
        for path, text in main_change.items():
            f.write(path, text)
        main_tip = f.commit("main moves")
        for commit in lane:
            f.git("cherry-pick", "--allow-empty", commit)
        return main_tip, f.head()

    def kernel_lane_with_record(self):
        self.f.write("crates/memra-engine/cu/qmatvec.cu", "__global__ void k() { int x = 4; }\n")
        code = self.f.commit("kernel")
        self.f.record(code, ["battery", "perf"])
        banked = self.f.head()
        self.assertIn("CELL-QUALIFIED", self.f.push())
        return code, banked

    def test_a_research_only_main_move_keeps_the_record(self):
        code, banked = self.kernel_lane_with_record()
        main_tip, head = self.rebase_onto(
            {"research/other-lane/README.md": "# another lane\n", "research/INDEX.md": "# Research\nother-lane | ok\n"},
            code, banked)
        out = self.f.push(oid=head, base=main_tip)
        self.assertIn("CELL-QUALIFIED", out)
        self.assertIn("research path", out)

    def test_a_tools_or_engine_main_move_still_voids_the_record(self):
        for change in ({"tools/fast-gate/plan.py": "print(2)\n"},
                       {"crates/memra-engine/src/lib.rs": "pub fn f() { let _ = 1; }\n"},
                       {"research/other-lane/README.md": "# x\n", "README.md": "# fixture, edited\n"}):
            with self.subTest(change=sorted(change)):
                self.setUp()
                code, banked = self.kernel_lane_with_record()
                main_tip, head = self.rebase_onto(change, code, banked)
                with self.assertRaisesRegex(q.GateError, "missing evidence"):
                    self.f.push(oid=head, base=main_tip)
                self.tearDown()

    def test_a_research_file_a_crate_reads_still_voids_the_record(self):
        self.f.write("research/tables/rates.txt", "1\n")
        self.f.write("crates/memra-engine/src/lib.rs",
                     "pub fn f() -> &'static str { include_str!(\"../../../research/tables/rates.txt\") }\n")
        self.f.commit("a crate reads a research table")
        self.f.git("update-ref", "refs/heads/seed-with-table", self.f.head())
        self.f.base = self.f.head()
        code, banked = self.kernel_lane_with_record()
        main_tip, head = self.rebase_onto({"research/tables/rates.txt": "2\n"}, code, banked)
        with self.assertRaisesRegex(q.GateError, "missing evidence"):
            self.f.push(oid=head, base=main_tip)

    def test_research_drift_names_only_unread_research_paths(self):
        a = self.f.head()
        self.f.write("research/lane/run.log", "raw\n")
        b = self.f.commit("research")
        self.assertEqual(ce.research_drift(self.f.repo, a, b), ["research/lane/run.log"])
        self.f.write("tools/x.py", "print(1)\n")
        c = self.f.commit("tools")
        self.assertIsNone(ce.research_drift(self.f.repo, a, c))
        self.assertIsNone(ce.research_drift(self.f.repo, a, "0" * 40))


class PerfVerdict(unittest.TestCase):
    def verdict(self, a, b):
        runs = [{"arm": "A", "model": "m", "tps": x} for x in a] + [{"arm": "B", "model": "m", "tps": x} for x in b]
        return ce.judge_perf(runs)["m"]

    def test_complete_separation_inside_the_noise_floor_is_not_a_regression(self):
        verdict = self.verdict([100.0, 100.2, 100.4], [99.5, 99.6, 99.9])
        self.assertEqual(verdict["delta_pct"], -0.6)
        self.assertFalse(verdict["regressed"])

    def test_a_regression_needs_both_rank_and_effect(self):
        self.assertTrue(self.verdict([100.0, 100.2, 100.4], [96.9, 97.2, 97.5])["regressed"])
        self.assertTrue(self.verdict([100.0, 100.0, 100.0], [98.0, 98.0, 98.0])["regressed"], "the floor is inclusive")
        # A 3% median drop whose arms overlap is not separated: not a regression.
        self.assertFalse(self.verdict([100.0, 100.2, 100.4], [97.0, 97.2, 100.5])["regressed"])


class Toolchain(unittest.TestCase):
    def test_rustc_is_resolved_in_the_build_tree(self):
        # The record named the caller's default rustc (1.98.1) while cargo built the tree with
        # its pinned rust-toolchain.toml (1.97.1). The fake rustc answers like rustup's proxy:
        # a toolchain file in the working directory wins over the default.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / "bin"
            fake.mkdir()
            (fake / "rustc").write_text("#!/bin/sh\nif [ -f rust-toolchain.toml ]; then echo 'rustc 1.97.1 (pinned)'; "
                                        "else echo 'rustc 1.98.1 (default)'; fi\n")
            (fake / "nvcc").write_text("#!/bin/sh\necho 'Build cuda fixture'\n")
            for tool in fake.iterdir():
                tool.chmod(0o755)
            tree = root / "tree"
            tree.mkdir()
            (tree / "rust-toolchain.toml").write_text('[toolchain]\nchannel = "1.97.1"\n')
            self.assertNotEqual(Path.cwd().resolve(), tree.resolve())
            found = ce.toolchain(tree, {**os.environ, "PATH": f"{fake}{os.pathsep}{os.environ['PATH']}"})
            self.assertEqual(found, {"rustc": "rustc 1.97.1 (pinned)", "nvcc": "Build cuda fixture"})


class DeviceLock(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def gb10_lock(self, repo=ROOT):
        return tomllib.loads((repo / "tools/change-impact.toml").read_text())["device"]["gb10"]["lock"]

    def test_gb10_takes_the_spark_through_sparklock_with_a_per_box_unit(self):
        lock = self.gb10_lock()
        self.assertEqual(lock[:2], ["sparklock", "run"])
        self.assertEqual(lock[lock.index("--unit") + 1], "${MEMRA_SPARK_UNIT}")
        self.assertEqual(lock[-1], "--")
        self.assertFalse(any("/tmp" in arg for arg in lock), "a local /tmp flock is invisible to the Sparks' lease")
        filled = ce.lock_command("gb10", {"lock": lock}, {"MEMRA_SPARK_UNIT": "unit-a"})
        self.assertEqual(filled[filled.index("--unit") + 1], "unit-a")
        for environ in ({}, {"MEMRA_SPARK_UNIT": ""}):
            with self.assertRaisesRegex(ce.EvidenceError, "gb10 lock needs MEMRA_SPARK_UNIT"):
                ce.lock_command("gb10", {"lock": lock}, environ)

    def fixture(self):
        f = Repo(self.root)
        self.assertEqual(self.gb10_lock(f.repo)[0], "sparklock")
        return f

    def test_unset_unit_refuses_before_anything_builds(self):
        f = self.fixture()
        out = self.root / "out"
        with mock.patch.dict(os.environ), mock.patch.object(ce, "build", side_effect=AssertionError("must not build")):
            os.environ.pop("MEMRA_SPARK_UNIT", None)
            with self.assertRaisesRegex(ce.EvidenceError, "gb10 lock needs MEMRA_SPARK_UNIT"):
                ce.produce(f.repo, "HEAD", "gb10", ["battery"], out, work=self.root / "work", log=lambda _: None)
        self.assertFalse(out.exists())

    def test_refused_lease_runs_nothing(self):
        f = self.fixture()
        fake = self.root / "bin"
        fake.mkdir()
        argv = self.root / "sparklock.argv"
        # sparklock exits non-zero without running its command when the unit is held.
        (fake / "sparklock").write_text(f"#!/bin/sh\nprintf '%s\\n' \"$@\" > {argv}\necho 'sparklock: unit held' >&2\nexit 1\n")
        (fake / "sparklock").chmod(0o755)
        tree, bins = self.root / "tree", self.root / "bins"
        bins.mkdir()
        for name in ce.BINARIES:
            (bins / name).write_bytes(b"fixture")
        out = self.root / "out"
        with mock.patch.dict(os.environ, {"PATH": f"{fake}{os.pathsep}{os.environ['PATH']}", "MEMRA_SPARK_UNIT": "unit-a"}), \
                mock.patch.object(ce, "build", return_value=(tree, bins)), \
                mock.patch.object(ce, "toolchain", return_value={}):
            with self.assertRaisesRegex(ce.DeviceBusy, "sparklock exit 1"):
                ce.produce(f.repo, "HEAD", "gb10", ["battery"], out, work=self.root / "work", log=lambda _: None)
        args = argv.read_text().splitlines()
        self.assertEqual(args[:4], ["run", "--unit", "unit-a", "--ttl"])
        self.assertIn("gpu-phase", args)
        self.assertFalse((out / "gpu-phase.started").exists())
        self.assertFalse((out / "record.json").exists())

    def test_a_lock_command_error_is_not_busy(self):
        # sparklock exits 2 for a usage error (an unknown unit). That is a configuration error a
        # retry will not fix; reporting it as DeviceBusy would read "busy, retry" every day.
        f = self.fixture()
        fake = self.root / "bin"
        fake.mkdir()
        (fake / "sparklock").write_text("#!/bin/sh\necho 'sparklock: unknown unit: unit-z' >&2\nexit 2\n")
        (fake / "sparklock").chmod(0o755)
        tree, bins = self.root / "tree", self.root / "bins"
        bins.mkdir()
        for name in ce.BINARIES:
            (bins / name).write_bytes(b"fixture")
        out = self.root / "out"
        with mock.patch.dict(os.environ, {"PATH": f"{fake}{os.pathsep}{os.environ['PATH']}", "MEMRA_SPARK_UNIT": "unit-z"}), \
                mock.patch.object(ce, "build", return_value=(tree, bins)), \
                mock.patch.object(ce, "toolchain", return_value={}):
            with self.assertRaises(ce.EvidenceError) as caught:
                ce.produce(f.repo, "HEAD", "gb10", ["battery"], out, work=self.root / "work", log=lambda _: None)
        self.assertNotIsInstance(caught.exception, ce.DeviceBusy)
        self.assertIn("device lock command failed (sparklock exit 2)", str(caught.exception))
        self.assertFalse((out / "record.json").exists())



class ShownCommand(unittest.TestCase):
    def test_banked_serving_lines_name_no_tree_or_home(self):
        tree = Path.home() / ".cache/memra-cells/gb10/tree-121a"
        line = f"   Compiling memra-server v0.138.0 ({tree}/crates/memra-server)\n".encode()
        self.assertEqual(ce.shown_bytes(line, tree), b"   Compiling memra-server v0.138.0 (<tree>/crates/memra-server)\n")
        self.assertEqual(ce.shown_bytes(f"cache {Path.home()}/x\n".encode(), tree), b"cache ~/x\n")

    def test_build_log_names_no_home_directory(self):
        # build.log is banked with the record and published: the build lock lives under
        # $HOME, so the logged command must show `~`, not the user's home path.
        command = ["flock", str(Path.home() / ".cache/memra-rig-build.lock"), "nice", "-n", "15", "cargo"]
        shown = ce.shown_command(command)
        self.assertEqual(shown, "flock ~/.cache/memra-rig-build.lock nice -n 15 cargo")
        self.assertNotIn(str(Path.home()), shown)


if __name__ == "__main__":
    unittest.main()
