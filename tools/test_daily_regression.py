#!/usr/bin/env python3
"""Daily regression runner: results branch publication, the per-device issue lifecycle
and the skip/busy decisions. CPU fixtures only; GitHub calls are recorded, not made."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("daily_regression", HERE / "daily-regression.py")
dr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dr)


def sh(*args, cwd=None):
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


class Publish(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.origin = root / "origin.git"
        sh("git", "init", "-q", "--bare", str(self.origin))
        self.repo = root / "repo"
        sh("git", "init", "-q", "-b", "main", str(self.repo))
        for key, value in (("user.name", "CPU fixture"), ("user.email", "fixture@example.invalid"),
                           ("core.hooksPath", "/dev/null")):
            sh("git", "-C", str(self.repo), "config", key, value)
        (self.repo / "README.md").write_text("# fixture\n")
        sh("git", "-C", str(self.repo), "add", ".")
        sh("git", "-C", str(self.repo), "commit", "-qm", "seed")
        sh("git", "-C", str(self.repo), "remote", "add", "origin", str(self.origin))
        sh("git", "-C", str(self.repo), "push", "-q", "origin", "main")
        self.run_dir = root / "run"
        self.run_dir.mkdir()
        (self.run_dir / "record.json").write_text('{"status": "fail"}\n')
        self.tools = root / "tools"
        self.tools.mkdir()
        self.boundary(0)
        os.environ["MEMRA_DAILY_WORK"] = str(root / "work")

    def tearDown(self):
        os.environ.pop("MEMRA_DAILY_WORK", None)
        self.tmp.cleanup()

    def boundary(self, code):
        (self.tools / "check-public-boundary.py").write_text(f"import sys\nsys.exit({code})\n")

    def summary(self, commit, status="fail"):
        return {"date": "2026-11-02", "commit": commit, "device": "rtx5090", "target": "sm_120a",
                "status": status, "failing": ["battery: argmax cell model-b did not pass"] if status == "fail" else []}

    def test_results_branch_is_orphan_and_append_only(self):
        dr.publish(self.repo, "rtx5090", self.run_dir, self.summary("a" * 40), self.tools, self.repo)
        dr.publish(self.repo, "rtx5090", self.run_dir, self.summary("b" * 40, "pass"), self.tools, self.repo)
        log = sh("git", "--git-dir", str(self.origin), "log", "--format=%s", "regression/rtx5090").splitlines()
        self.assertEqual(log, ["data(regression): rtx5090 pass at bbbbbbbbbbbb",
                               "data(regression): rtx5090 fail at aaaaaaaaaaaa"])
        files = sh("git", "--git-dir", str(self.origin), "ls-tree", "-r", "--name-only", "regression/rtx5090").splitlines()
        self.assertNotIn("README.md", files, "results branch must not carry main's tree")
        self.assertIn("research/daily-regression/rtx5090/2026-11-02-aaaaaaaaaaaa/summary.json", files)
        history = sh("git", "--git-dir", str(self.origin), "show",
                     "regression/rtx5090:research/daily-regression/rtx5090/history.tsv").splitlines()
        self.assertEqual([row.split("\t")[2] for row in history], ["fail", "pass"])

    def test_public_boundary_refusal_does_not_push(self):
        self.boundary(1)
        with self.assertRaisesRegex(RuntimeError, "public-boundary"):
            dr.publish(self.repo, "rtx5090", self.run_dir, self.summary("a" * 40), self.tools, self.repo)
        self.assertEqual(sh("git", "--git-dir", str(self.origin), "branch", "--list", "regression/*"), "")


class Issue(unittest.TestCase):
    def gh(self, existing):
        calls = []

        def run(args, **kwargs):
            calls.append(args)
            out = ""
            if args[:3] == ["git", "-C", "repo"] or args[0] == "git":
                out = "git@github.com:avifenesh/memra.git"
            elif args[:3] == ["gh", "issue", "list"]:
                out = json.dumps(existing)
            elif args[:3] == ["gh", "issue", "create"]:
                out = "https://github.com/avifenesh/memra/issues/9"
            return subprocess.CompletedProcess(args, 0, out, "")
        return calls, run

    def summary(self, status):
        return {"date": "2026-11-02", "commit": "c" * 40, "status": status,
                "failing": ["perf: REGRESSED on qwen/qwen3.8-27b"] if status == "fail" else []}

    def verbs(self, calls):
        return [c[1:3] for c in calls if c[0] == "gh"]

    def test_failure_opens_one_issue_then_updates_it(self):
        calls, run = self.gh([])
        with mock.patch.object(dr.subprocess, "run", run):
            dr.update_issue("repo", "rtx5090", "sm_120a", self.summary("fail"), "link")
        self.assertIn(["issue", "create"], self.verbs(calls))
        create = next(c for c in calls if c[1:3] == ["issue", "create"])
        self.assertIn("Daily regression: rtx5090 (sm_120a)", create)
        self.assertIn("perf: REGRESSED on qwen/qwen3.8-27b", create[create.index("--body") + 1])
        title = "Daily regression: rtx5090 (sm_120a)"
        calls, run = self.gh([{"number": 9, "state": "CLOSED", "title": title}])
        with mock.patch.object(dr.subprocess, "run", run):
            dr.update_issue("repo", "rtx5090", "sm_120a", self.summary("fail"), "link")
        verbs = self.verbs(calls)
        self.assertNotIn(["issue", "create"], verbs)
        self.assertEqual([v for v in verbs if v[0] == "issue" and v[1] != "list"],
                         [["issue", "reopen"], ["issue", "edit"], ["issue", "comment"]])

    def test_pass_closes_an_open_issue_and_ignores_a_closed_one(self):
        title = "Daily regression: rtx5090 (sm_120a)"
        calls, run = self.gh([{"number": 9, "state": "OPEN", "title": title}])
        with mock.patch.object(dr.subprocess, "run", run):
            dr.update_issue("repo", "rtx5090", "sm_120a", self.summary("pass"), "link")
        self.assertIn(["issue", "close"], self.verbs(calls))
        calls, run = self.gh([{"number": 9, "state": "CLOSED", "title": title}])
        with mock.patch.object(dr.subprocess, "run", run):
            dr.update_issue("repo", "rtx5090", "sm_120a", self.summary("pass"), "link")
        self.assertEqual([v for v in self.verbs(calls) if v != ["issue", "list"]], [])


class RunDecisions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["MEMRA_DAILY_WORK"] = self.tmp.name

    def tearDown(self):
        os.environ.pop("MEMRA_DAILY_WORK", None)
        self.tmp.cleanup()

    def args(self, **extra):
        import argparse
        return argparse.Namespace(repo=Path("."), device="rtx5090", head="d" * 40, force=False,
                                  no_publish=True, accept_regression=None, **extra)

    def test_tested_head_is_skipped(self):
        Path(self.tmp.name, "state.json").write_text(json.dumps({"last_commit": "d" * 40, "last_date": "x"}))
        with mock.patch.object(dr, "git", side_effect=AssertionError("must not touch git")):
            self.assertEqual(dr.run(self.args()), 0)

    def test_busy_device_is_not_a_failure(self):
        import cell_evidence as ce
        import change_impact as ci
        arch = ci.ArchTable(accepted=("120a",), caps={"12.0": "120a"})
        rules = {"target": {"120a": {"name": "sm_120a"}}, "device": {"rtx5090": {"compute_cap": "12.0"}}}
        with mock.patch.object(ci, "arch_table_at", return_value=arch), \
                mock.patch.object(ci, "rules_at", return_value=rules), \
                mock.patch.object(ce, "produce", side_effect=ce.DeviceBusy("lock not obtained")), \
                mock.patch.object(dr, "publish", side_effect=AssertionError("must not publish")):
            self.assertEqual(dr.run(self.args()), 3)
        self.assertFalse(Path(self.tmp.name, "state.json").exists())


if __name__ == "__main__":
    unittest.main()
