#!/usr/bin/env python3
"""Push-gate decisions on per-cell evidence. CPU fixtures only: the records here are
fabricated in a temporary repository and are never GPU evidence."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

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

    def record(self, commit, kinds, *, device_class="rtx5090", device=DEVICE, target="120a", slower=False,
               name=None, accept=None, serve_tail="serve-smoke: 0 failed"):
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
            for seq, arm in enumerate(ce.PERF_ORDER * 2, 1):
                model = "model-a" if seq <= len(ce.PERF_ORDER) else "model-b"
                rate = 100.0 + seq * 0.1 - (20.0 if slower and arm == "B" else 0.0)
                log = f"perf/{seq:02d}-{arm}-{model}.log"
                (out / log).write_text(f"generated 128 tokens in 1.000s = {rate:.2f} tok/s (ST decode)\n")
                runs.append({"arm": arm, "model": model, "log": log})
            meta.update({"base": self.base, "perf_baseline": True, "perf_runs": runs})
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
        slow = self.f.record(code, ["perf"], slower=True)
        self.assertEqual(slow["status"], "fail")
        self.assertIn("REGRESSED on model-a, model-b", slow["failures"][0])
        with self.assertRaisesRegex(q.GateError, "needs perf evidence"):
            self.f.push()
        self.f.record(code, ["perf"], slower=True, accept="reviewed: correctness fix costs 20%")
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


if __name__ == "__main__":
    unittest.main()
