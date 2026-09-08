#!/usr/bin/env python3
"""CPU-only source/evidence refusal fixtures; never executes a model."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("sampled_gate", Path(__file__).with_name("check_sampled_instrument_gate.py"))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class SampledGateTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="sampled-instrument-test-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.git("init", "-q")
        self.git("config", "user.email", "fixture@example.invalid")
        self.git("config", "user.name", "Fixture")
        self.probe = self.root / gate.PROBE
        self.probe.parent.mkdir(parents=True)
        self.probe.write_text("old\n")
        self.commit()
        self.base = self.git("rev-parse", "HEAD").strip()
        self.probe.write_text("new\n")
        self.commit()
        tested = self.git("rev-parse", "HEAD").strip()
        self.start = int(self.git("show", "-s", "--format=%ct", "HEAD")) + 1
        self.bank = self.root / "evidence"
        self.bank.mkdir()
        self.rows = []
        history = []
        for i in range(8):
            history = history + [{"role": "user", "content": f"question {i}"}]
            body = {"model": "fixture", "messages": copy.deepcopy(history), "max_tokens": 512,
                    "stream": True, "stream_options": {"include_usage": True}, "cache_salt": "conversation"}
            path = self.bank / f"request-{i}.json"
            path.write_text(json.dumps(body))
            text = f"answer {i}"
            self.rows.append({"turn": i + 1, "strict_valid": True, "done": True, "error": None,
                "loop": False, "route": "dflash2", "spec_rounds": 2, "cached_tokens": 0 if i == 0 else 10,
                "fingerprint": "memra-1.0-fixture",
                "completion_tokens": 20, "wall_s": 1., "ttft_s": .1, "text": text,
                "text_sha256": gate.digest(text.encode()), "request_receipt": f"/original/visit/request-{i}.json",
                "request_sha256": gate.digest(path.read_bytes())})
            history = history + [{"role": "assistant", "content": text}]
        self.table = "Engine/gate source: " + tested + "\n" + "\n".join(
            f"{i} 1 1.0 1 1.0 0.1 yes" for i in range(12))
        self.identity = {"source_commit": "c" * 40, "binary_sha256": "b" * 64,
                         "visit": {"phase": "sampled", "speculative": True}}
        self.inputs = {"source_commit": "c" * 40, "binary": "server", "files_sha256": {"server": "b" * 64}}
        self.server_log = "[server] build: memra-1.0-fixture (id: source-tree, git: cccccccccccc)\n"
        self.window = {"start_unix": self.start, "end_unix": self.start + 80, "phase": "sampled"}
        self.receipt = {"schema": "memra.sampled-instrument-gate.v1", "tested_commit": tested,
                        "probe_sha256": gate.digest(self.probe.read_bytes()),
                        "request_root": str(self.bank), "original_request_root": "/original/visit"}

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.root), *args], stderr=subprocess.DEVNULL).decode()

    def commit(self):
        self.git("add", "crates")
        self.git("commit", "-qm", "fixture")

    def reference(self, name, raw):
        p = self.bank / name
        p.write_bytes(raw)
        return {"path": str(p), "sha256": gate.digest(raw)}

    def validate(self):
        inputs = self.reference("inputs.json", json.dumps(self.inputs).encode())
        self.identity["inputs_sha256"] = inputs["sha256"]
        self.receipt.update(identity=self.reference("identity.json", json.dumps(self.identity).encode()),
            inputs=inputs, server_log=self.reference("server.log", self.server_log.encode()),
            window=self.reference("window.json", json.dumps(self.window).encode()),
            rows=self.reference("rows.jsonl", ("\n".join(json.dumps(r) for r in self.rows)).encode()),
            correctness_table=self.reference("table.txt", self.table.encode()))
        raw = json.dumps(self.receipt).encode()
        p = self.bank / "receipt.json"
        p.write_bytes(raw)
        return gate.validate(self.root, self.base, p, gate.digest(raw), now=self.start + 100)

    def alter_request(self, i, change):
        p = self.bank / f"request-{i}.json"
        body = json.loads(p.read_text())
        change(body)
        p.write_text(json.dumps(body))
        self.rows[i]["request_sha256"] = gate.digest(p.read_bytes())

    def test_valid_eight_turns(self):
        self.assertTrue(self.validate()["passed"])

    def test_greedy_or_sampling_override_refused_even_when_rehashed(self):
        self.alter_request(0, lambda b: b.update(temperature=0))
        with self.assertRaises(ValueError): self.validate()

    def test_unrelated_prompts_are_not_continuation(self):
        self.alter_request(2, lambda b: b.update(messages=[b["messages"][-1]]))
        with self.assertRaises(ValueError): self.validate()

    def test_changed_assistant_history_refused(self):
        self.alter_request(1, lambda b: b["messages"][1].update(content="different"))
        with self.assertRaises(ValueError): self.validate()

    def test_bad_cache_route_loop_and_timing_refused(self):
        original = copy.deepcopy(self.rows)
        for change in ({"cached_tokens": 0}, {"route": "plain"}, {"spec_rounds": 0},
                       {"loop": True}, {"error": "error"}, {"wall_s": float("nan")}, {"done": False}):
            self.rows = copy.deepcopy(original)
            self.rows[1].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError): self.validate()

    def test_missing_turn_refused(self):
        self.rows.pop()
        with self.assertRaises(ValueError): self.validate()

    def test_tampered_request_refused(self):
        (self.bank / "request-0.json").write_text("{}")
        with self.assertRaises(ValueError): self.validate()

    def test_tampered_output_refused(self):
        self.rows[0]["text"] = "not measured"
        with self.assertRaises(ValueError): self.validate()

    def test_fingerprint_and_runtime_substitution_refused(self):
        self.rows[3]["fingerprint"] = "other-runtime"
        with self.assertRaisesRegex(ValueError, "fingerprint"): self.validate()
        self.rows[3]["fingerprint"] = "memra-1.0-fixture"
        self.identity["source_commit"] = "d" * 40
        with self.assertRaisesRegex(ValueError, "input manifest"): self.validate()

    def test_explicit_hook_mode_cannot_skip_on_non_engine_changes(self):
        hook = Path(__file__).parent / "hooks/pre-push"
        text = hook.read_text()
        block = text[text.index('engine_files=$(printf'):text.index('# ---- profile-blob')]
        for changed in ("crates/memra-gguf/src/lib.rs", "tools/unrelated.sh"):
            for skip in ("0", "1"):
                env = {**os.environ, "changed": changed, "range": self.base,
                       "MEMRA_HARDWARE_GATE": "sampled-instrument", "MEMRA_SKIP_PERF_CI": skip,
                       "MEMRA_SKIP_FLAGS_CENSUS": skip, "MEMRA_SAMPLED_RECEIPT": "",
                       "MEMRA_SAMPLED_RECEIPT_SHA256": ""}
                result = subprocess.run(["sh", "-eu", "-c", block], cwd=self.root,
                                        env=env, capture_output=True)
                with self.subTest(changed=changed, skip=skip): self.assertNotEqual(result.returncode, 0)

    def test_changed_probe_refused(self):
        self.probe.write_text("changed after measurement\n")
        with self.assertRaises(ValueError): self.validate()

    def test_runtime_change_refused(self):
        (self.probe.parent.parent / "lib.rs").write_text("runtime change\n")
        self.commit()
        with self.assertRaisesRegex(ValueError, "serving/library"): self.validate()

    def test_unrelated_tool_change_refused(self):
        p = self.root / "tools/serve.sh"
        p.parent.mkdir()
        p.write_text("changed launch behavior\n")
        self.git("add", "tools")
        self.git("commit", "-qm", "unrelated tool")
        with self.assertRaisesRegex(ValueError, "outside"): self.validate()

    def test_unexplained_numeric_flip_refused(self):
        self.table = self.table.replace("0 1 1.0 1 1.0 0.1 yes", "0 1 1.0 2 1.0 0.1 NO")
        with self.assertRaisesRegex(ValueError, "unexplained"): self.validate()

    def test_stale_or_future_measurement_refused(self):
        for start, end in ((self.start - 10, self.start + 80), (self.start, self.start + 1000)):
            self.window.update(start_unix=start, end_unix=end)
            with self.subTest(start=start), self.assertRaises(ValueError): self.validate()


if __name__ == "__main__":
    unittest.main()
