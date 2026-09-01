#!/usr/bin/env python3
"""CPU-only tests for the resumable SFT scale orchestrator."""

from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "sft-scale.py"
PIPELINE = Path("~/projects/darklanes/sft-pipeline").expanduser()
PROOF = Path("~/projects/sft-traces/corpus/deepseek-v4-flash-20260808.jsonl").expanduser()


def load_tool():
    spec = importlib.util.spec_from_file_location("sft_scale_tested", TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {TOOL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SftScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = load_tool()
        cls.templates = {
            template.template_id: template
            for template in cls.tool.GEN.build_templates()
        }

    def test_all_proof_records_convert_to_valid_k3_traces(self) -> None:
        traces = []
        with PROOF.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if row.get("record_type") != "trace":
                    continue
                trace = self.tool.k3_record(
                    row,
                    self.templates[row["template_id"]],
                    workspace=None,
                    proof=True,
                )
                traces.append(trace)
        self.assertEqual(len(traces), 24)

        with tempfile.TemporaryDirectory(prefix="sft-scale-test-") as directory:
            path = Path(directory) / "proof.jsonl"
            path.write_text(
                "".join(
                    json.dumps(trace, ensure_ascii=False) + "\n"
                    for trace in traces
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(PIPELINE / "validate_trace.py"),
                    str(path),
                ],
                text=True,
                capture_output=True,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_session_conversion_preserves_real_tools_and_final_answer(self) -> None:
        with PROOF.open(encoding="utf-8") as source:
            next(source)
            row = json.loads(next(source))
        trace = self.tool.k3_record(
            row,
            self.templates[row["template_id"]],
            workspace=None,
            proof=True,
        )
        roles = [message["role"] for message in trace["messages"]]
        self.assertEqual(roles[0], "user")
        self.assertEqual(roles[-1], "assistant")
        self.assertIn("tool", roles)
        self.assertTrue(trace["messages"][-1]["content"].strip())
        self.assertGreater(len(trace["tools"]), 0)
        calls = {
            call["id"]
            for message in trace["messages"]
            for call in message.get("tool_calls", [])
        }
        responses = {
            message["tool_call_id"]
            for message in trace["messages"]
            if message["role"] == "tool"
        }
        self.assertEqual(calls, responses)

    def test_variant_prompts_are_unique_for_pilot_depth(self) -> None:
        for base in self.templates.values():
            prompts = {
                self.tool.variant_prompt(base, ordinal)
                for ordinal in range(64)
            }
            self.assertEqual(len(prompts), 64)

    def test_empty_plan_balances_task_kinds(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sft-scale-plan-") as directory:
            root = Path(directory)
            paths = self.tool.Paths(
                corpus_repo=root,
                pipeline_dir=PIPELINE,
                progress=root / "PROGRESS.md",
                steering=root / "steering.md",
                work_root=root / ".work",
            )
            self.tool.ensure_directories(paths)
            planned = self.tool.plan_templates(paths, 100)
        counts = Counter(template.category for template in planned)
        self.assertEqual(counts, Counter({kind: 25 for kind in self.tool.TASK_KINDS}))
        scenario_counts = Counter(
            (template.category, template.scenario) for template in planned
        )
        for kind in self.tool.TASK_KINDS:
            values = [
                scenario_counts[(kind, scenario)]
                for scenario in {template.scenario for template in planned}
            ]
            self.assertLessEqual(max(values) - min(values), 1)

    def test_scale_task_ids_group_by_base_and_end_in_sequence(self) -> None:
        value = self.tool.task_id("bug_fix", "cache-economics", 37)
        self.assertEqual(value, "scale-bug-fix-cache-economics-000037")
        match = self.tool.SCALE_TASK_RE.match(value)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "bug-fix")
        self.assertEqual(match.group(2), "cache-economics")
        self.assertEqual(match.group(3), "000037")

    def test_proof_spend_includes_failed_exports(self) -> None:
        total, sessions = self.tool.proof_spend()
        self.assertEqual(len(sessions), 24)
        self.assertAlmostEqual(total, 0.192890322, places=9)

    def test_coordination_branch_fast_forwards_without_losing_batch_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sft-scale-sync-") as directory:
            repo = Path(directory)
            self._init_repo(repo)
            subprocess.run(
                ["git", "switch", "-c", self.tool.COORDINATION_BRANCHES[0]],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            (repo / "tagging.txt").write_text("tag\n", encoding="utf-8")
            self._commit_all(repo, "tag")
            tag_head = self._git(repo, "rev-parse", "HEAD")
            subprocess.run(
                ["git", "switch", "main"],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            batch = repo / "raw-batch.jsonl"
            batch.write_text("{}\n", encoding="utf-8")

            self.tool.sync_coordination_branches(repo)

            self.assertEqual(self._git(repo, "rev-parse", "HEAD"), tag_head)
            self.assertEqual(batch.read_text(encoding="utf-8"), "{}\n")

    def test_coordination_branch_divergence_stops_before_commit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sft-scale-sync-") as directory:
            repo = Path(directory)
            self._init_repo(repo)
            subprocess.run(
                ["git", "switch", "-c", self.tool.COORDINATION_BRANCHES[0]],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            (repo / "tagging.txt").write_text("tag\n", encoding="utf-8")
            self._commit_all(repo, "tag")
            subprocess.run(
                ["git", "switch", "main"],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            (repo / "scale.txt").write_text("scale\n", encoding="utf-8")
            self._commit_all(repo, "scale")

            with self.assertRaisesRegex(RuntimeError, "diverged"):
                self.tool.sync_coordination_branches(repo)

    def test_worker_auth_store_contains_only_openrouter(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sft-scale-auth-") as directory:
            parent = Path(directory)
            env = self.tool.isolated_opencode_environment(
                self.tool.scale_environment(),
                parent,
            )
            auth_path = Path(env["XDG_DATA_HOME"]) / "opencode" / "auth.json"
            auth = json.loads(auth_path.read_text(encoding="utf-8"))
            self.assertEqual(set(auth), {"openrouter"})
            self.assertEqual(auth_path.stat().st_mode & 0o777, 0o600)
            result = subprocess.run(
                ["opencode", "auth", "list"],
                env=env,
                text=True,
                capture_output=True,
            )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("OpenRouter", output)
        for barred in ("Nvidia", "Alibaba Token Plan", "Z.AI", "OpenAI"):
            self.assertNotIn(barred, output)

    def test_unordered_answer_patterns_recover_historical_false_rejects(self) -> None:
        rejects = Path("~/projects/sft-traces/rejects").expanduser()
        checked = 0
        for path in sorted(rejects.glob("scale-*.jsonl")):
            with path.open(encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    outcome = row.get("outcome") or {}
                    if outcome.get("method") != "answer_check":
                        continue
                    scenario = row["meta"]["scenario"]
                    final = row["messages"][-1]["content"]
                    self.assertRegex(final, re.compile(self.tool.ANSWER_PATTERNS[scenario]))
                    checked += 1
        self.assertGreaterEqual(checked, 20)

    def _init_repo(self, repo: Path) -> None:
        subprocess.run(
            ["git", "init", "-b", "main"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "SFT Scale Test"],
            cwd=repo,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "sft-scale@example.invalid"],
            cwd=repo,
            check=True,
        )
        (repo / "base.txt").write_text("base\n", encoding="utf-8")
        self._commit_all(repo, "base")

    def _commit_all(self, repo: Path, message: str) -> None:
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=repo,
            check=True,
            capture_output=True,
        )

    def _git(self, repo: Path, *args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()


if __name__ == "__main__":
    unittest.main()
