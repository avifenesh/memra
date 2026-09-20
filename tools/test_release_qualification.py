#!/usr/bin/env python3
"""CPU contract simulations only. Fabricated temporary records are never GPU evidence."""
from __future__ import annotations
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import release_qualification as q

ROOT = Path(__file__).resolve().parents[1]
UUID = "GPU-00000000-0000-0000-0000-000000000001"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, sort_keys=True).encode() + b"\n")


class Fixture:
    def __init__(self, root):
        self.repo, self.out = root / "repo", root / "evidence"
        self.repo.mkdir(); self.out.mkdir()
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "CPU fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.git("config", "core.hooksPath", "/dev/null")
        for crate in ("engine", "kv", "tier", "tokenizer", "server"):
            f = self.repo / f"crates/memra-{crate}/src/lib.rs"
            f.parent.mkdir(parents=True); f.write_text("// CPU fixture source\n")
        (self.repo / "tools").mkdir()
        for name in ("kernel-check-27b.cells", "kernel-check-step35.cells", "release-coverage.py"):
            shutil.copy2(ROOT / "tools" / name, self.repo / "tools" / name)
        (self.repo / "Cargo.lock").write_text("# CPU fixture lock\n")
        self.models = {str(root / "model-a.gguf"): {"bytes": 32, "sha256": "a" * 64},
                       str(root / "model-b.gguf"): {"bytes": 64, "sha256": "b" * 64}}
        paths = list(self.models)
        self.oracle_dir = str(root / "oracles")
        self.models[self.oracle_dir + "/named-oracle.gguf"] = {"bytes": 32, "sha256": "d" * 64}
        (self.repo / "tools/release-roster.tsv").write_text(f"own\tmodel-a\t{paths[0]}\nvendor\tmodel-b\t{paths[1]}\n")
        (self.repo / "research").mkdir(); (self.repo / "research/INDEX.md").write_text("# Research\n")
        (self.repo / "research/runtime-input.json").write_text("{}\n")
        self.commit("seed")
        self.source = q.source_snapshot(self.repo)
        self.binary_dir = root / "fake-binaries"; self.binary_dir.mkdir()
        self.binaries = {}
        for name in q.BINARIES:
            p = self.binary_dir / name
            p.write_bytes(b"CPU FIXTURE, NOT EXECUTABLE: " + name.encode())
            self.binaries[name] = {**q.file_identity(p), "format": "ELF-x86_64"}
        self.put("source.json", self.source)
        (self.out / "build.log").write_text("CPU MOCK build observation; not a native build\n")
        self.build = {"schema": "memra-native-build-v2", "exit_code": 0,
                      "source": self.ref("source.json"), "source_before": self.source["inputs_sha256"],
                      "source_after": self.source["inputs_sha256"], "cuda_arch": "120a", "docs_rs": False,
                      "cuda_visible_devices": "", "rustc": "CPU mock", "nvcc": "CPU mock",
                      "platform": {"profile": "ubuntu-24.04", "machine": "x86_64", "glibc": "2.39"},
                      "command": ["cargo", "build", "--release", "--locked"], "log": self.ref("build.log"),
                      "recipe": {"policy": "controlled-cargo-v1", "cargo_home": "fresh-config-free",
                                 "checkout": "actual-git-blobs-modes-v1", "build_source": "owned-git-checkout",
                                 "cargo_config": "tracked-jobs-only",
                                 "compilers": {name: {"bytes": 32, "sha256": "c" * 64} for name in ("cargo", "rustc", "nvcc")}},
                      "binaries": self.binaries}
        self.put("build.json", self.build)
        (self.out / "topology.txt").write_text("CPU MOCK topology; GPU0 PIX CPU\n")
        shutil.copy2(self.repo / "tools/release-roster.tsv", self.out / "roster.tsv")
        manifests = {}
        for name in q.MANIFESTS:
            shutil.copy2(self.repo / name, self.out / Path(name).name)
            manifests[name] = self.ref(Path(name).name)
        self.hardware = {"devices": [{"index": "0", "uuid": UUID, "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                                      "compute_cap": "12.0", "driver_version": "CPU MOCK"}],
                         "headroom_query": {"nvml_index": 0, "uuid": UUID},
                         "topology_sha256": self.ref("topology.txt")["sha256"]}
        self.lease = {"wrapper_pid": 10, "child_pid": 11, "requested_uuids": [UUID], "lock_order": [UUID],
                      "lock_files": {UUID: f"/tmp/memra-gpu-locks/{UUID}.lock"},
                      "devices": [{"index": 0, "uuid": UUID, "name": self.hardware["devices"][0]["name"]}],
                      "state": "finished", "exit_code": 0, "child_exit_code": 0, "timed_out": False,
                      "interrupted_signal": None, "lingering_compute": [], "started_unix": 100, "finished_unix": 140}
        self.put("lease.json", self.lease)
        (self.out / "battery.log").write_text("CPU MOCK of battery results\n")
        (self.out / "telemetry.csv").write_text("timestamp,uuid,used,free,util,power,temp\n"
            + f"CPU MOCK,{UUID},1,2,3,4,5\n" * 2)
        env = {"CUDA_VISIBLE_DEVICES": q.digest(UUID.encode()),
               "MEMRA_KC_MODELS_DIR": q.digest(self.oracle_dir.encode())}
        self.run = {"schema": "memra-native-release-run-v1", "exit_code": 0,
                    "source_before": self.source["inputs_sha256"], "source_after": self.source["inputs_sha256"],
                    "binaries_before": copy.deepcopy(self.binaries), "binaries_after": copy.deepcopy(self.binaries),
                    "models_before": copy.deepcopy(self.models), "models_after": copy.deepcopy(self.models),
                    "oracle_directory": self.oracle_dir,
                    "numeric_environment": dict(env), "numeric_environment_after": dict(env),
                    "hardware": copy.deepcopy(self.hardware), "hardware_after": copy.deepcopy(self.hardware),
                    "lease_owner": {k: self.lease[k] for k in ("wrapper_pid", "child_pid", "requested_uuids")},
                    "started_unix": 110, "finished_unix": 130, "topology": self.ref("topology.txt"),
                    "command": ["bash", "tools/release-battery.sh", "--evidence-dir", "cells"],
                    "roster": self.ref("roster.tsv"), "manifests": manifests, "battery_log": self.ref("battery.log"),
                    "telemetry": self.ref("telemetry.csv"), "cells": []}
        required = q.coverage_module().required_cells([self.repo / p for p in q.MANIFESTS])
        self.cell("kernel", "kernel", "".join(f"{name} CPU-fixture OK\n" for name in sorted(required))
                  + f"ALL GREEN ({len(required)} cells, 0 skipped)\n")
        spec = "".join(f"[generate_spec K={k}] 32 tok\nself-consistency: PASS (identical to plain target)\n"
                       for k in range(1, 9)) + "=== SELF-CONSISTENCY PASS ===\n"
        for name in ("model-a", "model-b"):
            self.cell("argmax", name, "SUMMARY flips=0 bad=0\n  PASS: CPU MOCK calibrated verdict\n")
            self.cell("spec", name, spec)
        self.refresh()

    def git(self, *args):
        return q.git(self.repo, *args)

    def commit(self, message):
        self.git("add", "."); self.git("commit", "-qm", message)

    def put(self, name, value):
        write(self.out / name, value)

    def ref(self, name):
        return {"path": name, "sha256": q.digest((self.out / name).read_bytes())}

    def cell(self, kind, model, text):
        name = f"cells/{len(self.run['cells'])}-{kind}.log"
        (self.out / name).parent.mkdir(exist_ok=True)
        (self.out / name).write_text(text)
        self.run["cells"].append({"kind": kind, "model": model, "exit_code": 0, "log": self.ref(name)})

    def refresh(self, recompute_verdicts=True):
        self.put("build.json", self.build); self.put("run.json", self.run); self.put("lease.json", self.lease)
        if recompute_verdicts:
            self.verdicts = q.cell_verdicts(self.run, q.Evidence(self.out), self.repo, "HEAD")
        self.record = {"schema": "memra-release-qualification-v1", "status": "qualified",
                       **{key: self.ref(key + ".json") for key in ("source", "build", "run", "lease")},
                       "verdicts": self.verdicts}
        self.record["identity_sha256"] = q.object_digest({"source": self.source["inputs_sha256"],
            "build": self.record["build"]["sha256"], "models": self.run["models_before"],
            "numeric": self.run["numeric_environment"], "hardware": self.run["hardware"], "verdicts": self.verdicts})
        self.record["payloads"] = {str(p.relative_to(self.out)): q.sha256_file(p)
                                   for p in self.out.rglob("*") if p.is_file() and p.name != "record.json"}
        self.put("record.json", self.record)

    def verify(self, **kwargs):
        return q.validate_record(self.record, q.Evidence(self.out), self.repo, "HEAD", **kwargs)

    def publish(self):
        target = self.repo / q.PUBLICATION / "cpu-fixture-do-not-qualify"
        shutil.copytree(self.out, target)
        write(self.repo / q.POINTER, {"record": str(target.relative_to(self.repo) / "record.json"),
                                     "sha256": q.digest((target / "record.json").read_bytes())})
        self.commit("CPU mock record for checker tests only")


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="memra-547-cpu-")
        self.addCleanup(self.tmp.cleanup)
        self.f = Fixture(Path(self.tmp.name))

    def test_complete_content_binding_and_publication(self):
        result = self.f.verify(binaries=self.f.binary_dir, models=self.f.models, hardware=self.f.hardware)
        self.assertFalse(result["publication_equivalent"])
        sidecar = self.f.repo / "research/new-publication/RESULTS.md"
        sidecar.parent.mkdir(); sidecar.write_text("CPU fixture publication only\n")
        self.f.commit("publication addition")
        self.assertEqual(self.f.verify()["publication_additions"], ["research/new-publication/RESULTS.md"])
        self.f.publish()
        result = q.verify_published(self.f.repo, "HEAD")
        self.assertTrue(result["publication_equivalent"])
        self.assertEqual(result["tested_commit"], self.f.source["commit"])
        with self.assertRaisesRegex(q.GateError, "no native build for profile"):
            q.verify_published(self.f.repo, "HEAD", profile="ubuntu-22.04")

    def test_every_runtime_dependency_and_nonpublication_research_invalidates(self):
        paths = [f"crates/memra-{crate}/src/lib.rs" for crate in ("engine", "kv", "tier", "tokenizer", "server")]
        paths += ["Cargo.lock", "tools/release-coverage.py", "research/runtime-input.json",
                  "crates/memra-engine/Cargo.toml", "crates/memra-engine/build.rs",
                  "crates/memra-kv/Cargo.toml", "crates/memra-tier/src/transfers.rs",
                  "crates/memra-server/src/worker.rs", "crates/memra-gguf/src/source.rs",
                  ".cargo/config.toml", "rust-toolchain.toml"]
        for name in paths:
            with self.subTest(name=name):
                (self.f.repo / name).parent.mkdir(parents=True, exist_ok=True)
                (self.f.repo / name).write_text("different source\n")
                self.f.commit("alter dependency")
                with self.assertRaisesRegex(q.GateError, "source inputs changed"):
                    self.f.verify()
                self.f.git("reset", "--hard", self.f.source["commit"])

    def test_fresh_checkout_never_restamps_old_evidence(self):
        self.f.publish()
        checkout = Path(self.tmp.name) / "fresh"
        subprocess.run(["git", "clone", "-q", str(self.f.repo), str(checkout)], check=True)
        self.assertTrue(q.verify_published(checkout, "HEAD")["publication_equivalent"])
        (checkout / "crates/memra-kv/src/lib.rs").write_text("stale source\n")
        subprocess.run(["git", "-C", str(checkout), "-c", "user.name=CPU", "-c", "user.email=cpu@example.invalid",
                        "commit", "-qam", "new runtime"], check=True)
        # Fresh mtimes and an arbitrary perf-ci file cannot repair stale content.
        for p in checkout.rglob("*"):
            if p.is_file(): os.utime(p, None)
        with self.assertRaisesRegex(q.GateError, "source inputs changed"):
            q.verify_published(checkout, "HEAD")

    def test_unknown_tested_commit_is_not_an_accepted_source_label(self):
        self.f.source["commit"] = "f" * 40
        self.f.put("source.json", self.f.source)
        self.f.build["source"] = self.f.ref("source.json")
        self.f.refresh()
        with self.assertRaisesRegex(q.GateError, "tested source object missing"):
            self.f.verify()

    def test_stale_binary_and_model_inputs(self):
        (self.f.binary_dir / "run-spec").write_bytes(b"different binary")
        with self.assertRaisesRegex(q.GateError, "stale binary"):
            self.f.verify(binaries=self.f.binary_dir)
        with self.assertRaisesRegex(q.GateError, "stale model"):
            self.f.verify(models={"different": {"bytes": 32, "sha256": "c" * 64}})
        self.f.run["binaries_after"]["run-spec"]["sha256"] = "c" * 64
        self.f.refresh()
        with self.assertRaisesRegex(q.GateError, "runtime binary"):
            self.f.verify()

    def test_rig_numeric_and_physical_headroom_mismatch(self):
        with self.assertRaisesRegex(q.GateError, "stale rig"):
            self.f.verify(hardware={"devices": []})
        for mode in ("name", "visibility", "headroom"):
            with self.subTest(mode=mode):
                f = self.f
                original = copy.deepcopy(f.run)
                if mode == "name":
                    f.run["hardware"]["devices"][0]["name"] = "Other GPU"
                    f.run["hardware_after"] = copy.deepcopy(f.run["hardware"])
                elif mode == "visibility":
                    f.run["numeric_environment"]["CUDA_VISIBLE_DEVICES"] = q.digest(b"0")
                    f.run["numeric_environment_after"] = dict(f.run["numeric_environment"])
                else:
                    f.run["hardware"]["headroom_query"]["uuid"] = "GPU-other"
                    f.run["hardware_after"] = copy.deepcopy(f.run["hardware"])
                f.refresh()
                with self.assertRaises(q.GateError): f.verify()
                f.run = original; f.refresh()

    def test_changed_numeric_environment_and_capture_selection_refuse(self):
        self.f.run["numeric_environment_after"]["NVIDIA_TF32_OVERRIDE"] = q.digest(b"1")
        self.f.refresh()
        with self.assertRaisesRegex(q.GateError, "numerical environment changed"):
            self.f.verify()
        spec = importlib.util.spec_from_file_location("qualification_capture", ROOT / "tools/qualify-release.py")
        capture = importlib.util.module_from_spec(spec); spec.loader.exec_module(capture)
        other = "GPU-00000000-0000-0000-0000-000000000002"
        observations = [f"1, {other}, NVIDIA RTX PRO 6000 Blackwell Server Edition, 12.0, 595.58, pci, 97887\n", UUID + "\n"]
        with patch.object(capture.subprocess, "check_output", side_effect=observations):
            with self.assertRaisesRegex(q.GateError, "headroom queries NVML GPU0"):
                capture.observe_hardware([other])

    def test_missing_failed_duplicated_cells_and_skip_ceiling(self):
        for mutation in ("missing", "failed", "duplicate", "single-k", "skip-budget"):
            original = copy.deepcopy(self.f.run)
            original_files = {cell["log"]["path"]: (self.f.out / cell["log"]["path"]).read_bytes() for cell in original["cells"]}
            with self.subTest(mutation=mutation):
                if mutation == "missing": self.f.run["cells"].pop()
                elif mutation == "failed": self.f.run["cells"][0]["exit_code"] = 1
                elif mutation == "duplicate": self.f.run["cells"].append(copy.deepcopy(self.f.run["cells"][0]))
                else:
                    index = 2 if mutation == "single-k" else 0
                    cell = self.f.run["cells"][index]; p = self.f.out / cell["log"]["path"]
                    if mutation == "single-k":
                        p.write_text("[generate_spec K=4] 32 tok\nself-consistency: PASS (identical to plain target)\n=== SELF-CONSISTENCY PASS ===\n")
                    else:
                        text = p.read_text(); count = len(q.coverage_module().required_cells([self.f.repo / x for x in q.MANIFESTS]))
                        text = text.replace(f"ALL GREEN ({count} cells, 0 skipped)",
                            "".join(f"SKIP optional-{n} (unavailable fixture)\n" for n in range(12)) + f"ALL GREEN ({count+12} cells, 12 skipped)")
                        p.write_text(text)
                    cell["log"] = self.f.ref(cell["log"]["path"])
                self.f.refresh(recompute_verdicts=False)
                with self.assertRaises((q.GateError, ValueError)): self.f.verify()
            self.f.run = original
            for name, data in original_files.items(): (self.f.out / name).write_bytes(data)
            self.f.refresh()

    def test_lease_failure_cannot_be_sealed(self):
        for key, value in [("exit_code", 1), ("child_exit_code", 1), ("timed_out", True),
                           ("interrupted_signal", 15), ("lingering_compute", ["pid"]), ("state", "running")]:
            with self.subTest(key=key):
                old = self.f.lease[key]; self.f.lease[key] = value; self.f.refresh()
                with self.assertRaisesRegex(q.GateError, "lease did not finish"): self.f.verify()
                self.f.lease[key] = old; self.f.refresh()

    def test_modified_evidence_and_unsafe_paths_refuse(self):
        (self.f.out / "build.log").write_text("changed evidence")
        with self.assertRaisesRegex(q.GateError, "evidence changed"): self.f.verify()
        for path in ("../outside", "/absolute", "a/../b", "a\\b"):
            with self.assertRaises(q.GateError): q.safe_path(path)
        with self.assertRaises(q.GateError): q.json_bytes('{"status":1,"status":2}')

    def test_development_is_explicit_and_never_main_or_tag(self):
        head = self.f.source["commit"]
        q.check_push(self.f.repo, [f"refs/heads/topic {head} refs/heads/topic {'0'*40}"], "development")
        for remote in ("refs/heads/main", "refs/heads/master", "refs/tags/v1.0.0"):
            with self.assertRaisesRegex(q.GateError, "cannot push main or tags"):
                q.check_push(self.f.repo, [f"refs/heads/topic {head} {remote} {'0'*40}"], "development")
        with self.assertRaisesRegex(q.GateError, "no committed native qualification pointer"):
            q.verify_published(self.f.repo, "HEAD")

    def test_index_is_append_only_and_source_modes_are_bound(self):
        (self.f.repo / "research/INDEX.md").write_text("# Research\nnew publication\n")
        self.f.commit("append metadata"); self.f.verify()
        (self.f.repo / "research/INDEX.md").write_text("replaced old claims\n")
        self.f.commit("replace metadata")
        with self.assertRaisesRegex(q.GateError, "append-only"): self.f.verify()

    def test_real_hook_missing_models_never_grant_qualification(self):
        for name in ("release_qualification.py", "check_hardware_gate.py"):
            shutil.copy2(ROOT / "tools" / name, self.f.repo / "tools" / name)
        for name in ("update-perf-board.py", "check-public-boundary.py"):
            (self.f.repo / "tools" / name).write_text("import sys\nsys.exit(0)\n")
        census = self.f.repo / "tools/check-flags.sh"
        census.write_text("#!/bin/sh\nexit 0\n"); census.chmod(0o755)
        self.f.commit("stage real qualification hook with unrelated gates stubbed")
        head = self.f.git("rev-parse", "HEAD").decode().strip()
        def hook(remote, **extra):
            env = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
            env.update(extra)
            return subprocess.run(["sh", str(ROOT / "tools/hooks/pre-push"), "origin"],
                cwd=self.f.repo, env=env, input=f"refs/heads/topic {head} {remote} {'0'*40}\n",
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
        for model_dir in (str(Path(self.tmp.name) / "absent-models"), str(self.f.binary_dir)):
            result = hook("refs/heads/topic", MEMRA_MODELS_DIR=model_dir)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("UNQUALIFIED", result.stdout)
        result = hook("refs/heads/topic", MEMRA_RELEASE_QUALIFICATION_MODE="development")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("UNQUALIFIED DEVELOPMENT", result.stdout)
        self.assertIn("UNQUALIFIED", (self.f.repo / ".git/memra-gate-skips.log").read_text())
        for remote in ("refs/heads/main", "refs/tags/v1.0.0"):
            result = hook(remote, MEMRA_RELEASE_QUALIFICATION_MODE="development")
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("cannot push main or tags", result.stdout)
        result = hook("refs/heads/topic", MEMRA_SKIP_PERF_CI="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("is retired", result.stdout)

    def test_bank_copies_only_manifested_evidence(self):
        import argparse
        spec = importlib.util.spec_from_file_location("qualification_bank", ROOT / "tools/qualify-release.py")
        capture = importlib.util.module_from_spec(spec); spec.loader.exec_module(capture)
        outside = Path(self.tmp.name) / "not-evidence.txt"
        outside.write_text("must not be published")
        (self.f.out / "unmanifested-link").symlink_to(outside)
        capture.bank(argparse.Namespace(repo=self.f.repo, out=self.f.out, name="cpu-fixture-only", append=False))
        destination = self.f.repo / q.PUBLICATION / "cpu-fixture-only"
        self.assertFalse((destination / "unmanifested-link").exists())
        self.assertTrue((destination / "record.json").is_file())

    def test_release_and_push_wiring_and_retired_waiver(self):
        hook = (ROOT / "tools/hooks/pre-push").read_text()
        self.assertIn("release_qualification.py push", hook)
        self.assertNotIn("newest_row=", hook)
        self.assertNotIn("no model dir; perf-ci not enforced", hook)
        release = (ROOT / ".github/workflows/release.yml").read_text()
        publish = (ROOT / ".github/workflows/publish.yml").read_text()
        self.assertLess(release.index("release_qualification.py verify"), release.index("  build:"))
        self.assertLess(release.index("--binaries target/release"), release.index("name: Package binaries"))
        self.assertLess(publish.index("release_qualification.py verify"), publish.index("name: Publish to crates.io"))
        self.assertIn("inputs.publish == true", publish[:publish.index("release_qualification.py verify")])


if __name__ == "__main__":
    unittest.main()
