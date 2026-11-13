#!/usr/bin/env python3
"""Offline tests for the Q35 cold-mixed gate and the sellgate grid law (memra#777).

The two banked integ68 receipts failed with every hit reporting cached_tokens 4832 against an
expected 4860 while all 20 requests returned exactly 60 tokens. 4832 is the entry the #602
prime-grid capture publishes for a 4860-token prompt on grid 32. These tests replay both receipts
through the fixed judging code (PASS), and the red twin puts the old expectation (the prompt
length) back and must FAIL, classed as a usage mismatch and not a token regression.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RECEIPTS = REPO / "research/spill-lead-20260919/integration-day12/integ68-q35ab"
BANKED = [RECEIPTS / "main-q35-cold-mixed.log", RECEIPTS / "integ68-q35-cold-mixed.log"]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GATE = _load("memra_q35_cold_mixed_gate", HERE / "q35-cold-mixed-gate.py")
HARNESS = GATE.load_sellgate(REPO / "research/sellgate-20260812/sellgate_replay.py")
sys.path.insert(0, str(HERE))
import cache_qualification  # noqa: E402

WORKLOAD = GATE.load_frozen_workload(HARNESS, REPO)


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("{")
    ]


def _write(rows: list[dict]) -> Path:
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".log", prefix="q35-gate-test-", delete=False, encoding="utf-8"
    )
    with handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return Path(handle.name)


class GridLaw(unittest.TestCase):
    CASES = [
        (4860, 32, 4832),
        (4848, 32, 4832),
        (4840, 32, 4800),  # remainder 8 < PRIME_MIN_T steps down one grid unit
        (4864, 32, 4864),  # on the grid: the prompt end
        (100, 32, 64),
        (70, 32, None),  # 32 < PREFIX_CACHE_MIN_TOKENS: the seed is refused
        (4860, 64, 4800),
        (4860, 128, 4736),
        (272, 32, 256),
    ]

    def test_capture_len_matches_the_server_law(self) -> None:
        for prompt, grid, entry in self.CASES:
            with self.subTest(prompt=prompt, grid=grid):
                self.assertEqual(HARNESS.capture_len(prompt, grid), entry)
                self.assertEqual(cache_qualification.capture_len(prompt, grid), entry)

    def test_expected_hit_cached_refuses_a_seedless_prompt(self) -> None:
        self.assertEqual(HARNESS.expected_hit_cached(4860, 32), 4832)
        with self.assertRaises(ValueError):
            HARNESS.expected_hit_cached(70, 32)

    def test_gdn_grid_mirrors_engine_env_read(self) -> None:
        cases = [
            ({}, 32),
            ({"MEMRA_GDN_CHUNK": "64"}, 64),
            ({"MEMRA_GDN_CHUNK": "48"}, 32),
            ({"MEMRA_GDN_CHUNK": "200"}, 128),
            ({"MEMRA_GDN_CHUNK": "16"}, 32),
            ({"MEMRA_GDN_CHUNK": "abc"}, 32),
            ({"MEMRA_GDN_CHUNK": "-5"}, 32),
            ({"MEMRA_GDN_CHUNK": "64\n"}, 32),
        ]
        for env, grid in cases:
            with self.subTest(env=env):
                self.assertEqual(HARNESS.gdn_grid(env), grid)
                self.assertEqual(cache_qualification.gdn_grid(env), grid)


class BankedReplay(unittest.TestCase):
    def test_both_banked_receipts_pass_under_the_grid_law(self) -> None:
        for log in BANKED:
            with self.subTest(log=log.name):
                summary = GATE.replay(HARNESS, WORKLOAD, 32, log)
                self.assertEqual(summary["banked_verdict"], "FAIL")
                self.assertEqual(summary["banked_expected_hit_cached_tokens"], [4860])
                self.assertEqual(summary["reported_hit_cached_tokens"], [4832])
                self.assertEqual(summary["expected_hit_cached_tokens"], 4832)
                self.assertEqual(summary["verdict"], "PASS", summary)
                self.assertEqual(summary["failure_classes"], [])
                self.assertEqual(summary["fail_lines"], [])
                self.assertEqual((summary["requests"], summary["hit_requests"]), (20, 18))

    def test_red_twin_old_expectation_fails_as_usage_mismatch(self) -> None:
        """The pre-fix law (a hit caches the whole prompt) must go red on a grid-aligned receipt."""
        for log in BANKED:
            with self.subTest(log=log.name), mock.patch.object(
                HARNESS, "expected_hit_cached", lambda prompt, grid: prompt
            ):
                summary = GATE.replay(HARNESS, WORKLOAD, 32, log)
                self.assertEqual(summary["verdict"], "FAIL")
                self.assertIn("usage_mismatch", summary["failure_classes"])
                self.assertNotIn("token_regression", summary["failure_classes"])
                self.assertEqual(summary["short_or_non_length"], [])
                self.assertIn(GATE.FAIL_LINES["usage_mismatch"], summary["fail_lines"])
                self.assertNotIn(GATE.FAIL_LINES["token_regression"], summary["fail_lines"])

    def test_wrong_grid_fails(self) -> None:
        summary = GATE.replay(HARNESS, WORKLOAD, 64, BANKED[0])
        self.assertEqual(summary["verdict"], "FAIL")
        self.assertIn("usage_mismatch", summary["failure_classes"])


class MutatedReceipts(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = _rows(BANKED[0])
        self.paths: list[Path] = []

    def tearDown(self) -> None:
        for path in self.paths:
            path.unlink(missing_ok=True)

    def _replay(self, rows: list[dict]) -> dict:
        path = _write(rows)
        self.paths.append(path)
        return GATE.replay(HARNESS, WORKLOAD, 32, path)

    def _first(self, rows: list[dict], kind: str, role: str | None = None) -> dict:
        return next(
            row
            for row in rows
            if row.get("kind") == kind and (role is None or row.get("cache_role") == role)
        )

    def test_hit_reporting_the_prompt_end_passes(self) -> None:
        """memra#1233: an exact re-send restores the whole prompt through the entry's tail."""
        rows = copy.deepcopy(self.rows)
        hit = self._first(rows, "request", "hit")
        hit["cached_tokens"] = 4860
        cell = self._first(rows, "cell")
        cell["counter_deltas"]["cached_tokens_in"] += 28
        cell["counter_deltas"]["prefix_cache_hit_tokens"] += 28
        cell["counter_deltas"]["prefix_cache_tail_hits"] = 1
        summary = self._replay(rows)
        self.assertEqual(summary["verdict"], "PASS", summary)

    def test_hit_reporting_the_prompt_end_without_a_tail_hit_fails(self) -> None:
        """A whole-prompt restore the server did not count as a tail hit is the off-grid
        prompt-end entry of memra#602, not a tail restore."""
        rows = copy.deepcopy(self.rows)
        hit = self._first(rows, "request", "hit")
        hit["cached_tokens"] = 4860
        cell = self._first(rows, "cell")
        cell["counter_deltas"]["cached_tokens_in"] += 28
        cell["counter_deltas"]["prefix_cache_hit_tokens"] += 28
        summary = self._replay(rows)
        self.assertEqual(summary["verdict"], "FAIL")

    def test_hit_reporting_neither_length_is_a_usage_mismatch(self) -> None:
        rows = copy.deepcopy(self.rows)
        hit = self._first(rows, "request", "hit")
        hit["cached_tokens"] = 4800
        cell = self._first(rows, "cell")
        cell["counter_deltas"]["cached_tokens_in"] -= 32
        cell["counter_deltas"]["prefix_cache_hit_tokens"] -= 32
        summary = self._replay(rows)
        self.assertEqual(summary["verdict"], "FAIL")
        self.assertEqual(summary["failure_classes"], ["usage_mismatch"])
        self.assertEqual(summary["fail_lines"], [GATE.FAIL_LINES["usage_mismatch"]])

    def test_short_request_is_a_token_regression_not_a_usage_mismatch(self) -> None:
        rows = copy.deepcopy(self.rows)
        miss = self._first(rows, "request", "miss")
        miss["completion_tokens"] = 26
        miss["finish_reason"] = "stop"
        cell = self._first(rows, "cell")
        cell["counter_deltas"]["tokens_out"] -= 34
        summary = self._replay(rows)
        self.assertEqual(summary["verdict"], "FAIL")
        self.assertIn("token_regression", summary["failure_classes"])
        self.assertIn("response_failure", summary["failure_classes"])
        self.assertNotIn("usage_mismatch", summary["failure_classes"])
        self.assertEqual(summary["fail_lines"][0], GATE.FAIL_LINES["token_regression"])

    def test_counter_drift_is_cell_accounting(self) -> None:
        rows = copy.deepcopy(self.rows)
        self._first(rows, "cell")["counter_deltas"]["cached_tokens_in"] += 1
        summary = self._replay(rows)
        self.assertEqual(summary["verdict"], "FAIL")
        self.assertEqual(summary["failure_classes"], ["cell_accounting"])

    def test_seed_restoring_neither_length_fails_as_seed_usage(self) -> None:
        rows = copy.deepcopy(self.rows)
        self._first(rows, "seed")["cached_tokens"] = 4800
        summary = self._replay(rows)
        self.assertEqual(summary["verdict"], "FAIL")
        self.assertIn("seed_failure", summary["failure_classes"])
        self.assertIn("usage_mismatch", summary["failure_classes"])


class SeedJudge(unittest.TestCase):
    ROW = {"ok": True, "text_sha256": "a" * 64}

    def test_seed_accepts_cold_grid_entry_and_prompt_end(self) -> None:
        for cached in (0, 4832, 4860):
            with self.subTest(cached=cached):
                row = dict(self.ROW, cached_tokens=cached)
                self.assertEqual(HARNESS.judge_seed("q35", 0, row, 4860, 32, {}), [])

    def test_seed_rejects_neither_length(self) -> None:
        row = dict(self.ROW, cached_tokens=4800)
        failures = HARNESS.judge_seed("q35", 0, row, 4860, 32, {})
        self.assertEqual(len(failures), 1)
        self.assertIn(HARNESS.FAIL_USAGE, failures[0])


class FakeServer:
    """A /metrics counter set and a completion route, for run_cell offline.

    `busy` names what still holds the server when the first scrape lands: "active" a session
    whose stream already ended (as the last hot-set seed of the GB10 record of 2026-11-13 did
    inside that cell's window), "queued" a queued request, "route" a dedicated route's running
    request (memra#501: its counters fold into the totals, not into the worker gauges). On the
    second scrape it finishes: completed +1, its last 4 tokens_out, the gauge back to 0. "never"
    keeps a live session forever."""

    def __init__(self, busy: str | None = None) -> None:
        self.lock = threading.Lock()
        self.scrapes = 0
        self.busy = busy
        self.m: dict = {key: 0 for key in HARNESS.COUNTERS}
        pending = busy in ("active", "queued", "route")
        self.m.update({"admitted": 8, "completed": 7 if pending else 8, "tokens_out": 476 if pending else 480})
        self.m["active_sessions"] = 1 if busy in ("active", "never") else 0
        self.m["queued_requests"] = 1 if busy == "queued" else 0
        self.m["routes"] = [{"name": "dsv4", "running": 1 if busy == "route" else 0, "waiting": 0}]

    def scrape(self, endpoint, timeout):
        with self.lock:
            self.scrapes += 1
            if self.busy in ("active", "queued", "route") and self.scrapes == 2:
                self.m["completed"] += 1
                self.m["tokens_out"] += 4
                self.m["active_sessions"] = self.m["queued_requests"] = 0
                self.m["routes"] = [{"name": "dsv4", "running": 0, "waiting": 0}]
                self.busy = None
            return json.loads(json.dumps(self.m))

    def request(self, endpoint, prompt, salt, workload, timeout, barrier=None, go=None, trace_id=None):
        if barrier is not None:
            barrier.wait(timeout=60)
        if go is not None:
            go.wait(timeout=60)
        hit = "-hot-" in salt
        cached = HARNESS.expected_hit_cached(len(prompt), 32) if hit else 0
        tokens = int(workload["completion_tokens"])
        with self.lock:
            for key, add in (("admitted", 1), ("completed", 1), ("tokens_out", tokens),
                             ("prompt_tokens_in", len(prompt)), ("cached_tokens_in", cached),
                             ("prefix_cache_hit_tokens", cached),
                             ("prefix_cache_hits" if hit else "prefix_cache_misses", 1)):
                self.m[key] += add
        now = time.monotonic()
        return {"ok": True, "prompt_tokens": len(prompt), "cached_tokens": cached,
                "completion_tokens": tokens, "ttft_ms": 1.0, "latency_ms": 2.0,
                "inter_token_ms": 1.0, "_started": now, "_ended": now}


class IdleWindow(unittest.TestCase):
    """The cell window opens on an idle server (wait_idle before run_cell's `before` scrape)."""

    ENDPOINT = HARNESS.Endpoint(label="q35", base="http://fake", model="q35-coldfix")

    def _cell(self, server: FakeServer, timeout: float = 600.0) -> dict:
        judged = {"response_ok": True, "usage_ok": True, "golden_ok": True, "golden_required": False}
        with mock.patch.object(HARNESS, "scrape", server.scrape), \
                mock.patch.object(HARNESS, "request", server.request), \
                mock.patch.object(HARNESS, "judge_request", return_value=judged):
            _, _, cells = HARNESS.run_cell(
                [self.ENDPOINT], WORKLOAD, "test", "mixed90", 1, 4, timeout, {}, 32
            )
        return cells[0]

    def _clean(self, cell: dict) -> None:
        self.assertEqual(cell["counter_deltas"]["completed"], cell["requests_n"])
        self.assertEqual(cell["counter_deltas"]["tokens_out"], cell["completion_tokens"])
        self.assertEqual(cell["integrity_failures"], [])

    def test_a_session_live_at_the_window_start_stays_out_of_the_cell(self) -> None:
        self._clean(self._cell(FakeServer("active")))

    def test_a_queued_request_at_the_window_start_stays_out_of_the_cell(self) -> None:
        self._clean(self._cell(FakeServer("queued")))

    def test_a_busy_route_at_the_window_start_stays_out_of_the_cell(self) -> None:
        self._clean(self._cell(FakeServer("route")))

    def test_a_server_that_never_idles_fails_the_cell(self) -> None:
        with self.assertRaisesRegex(HARNESS.NotIdle, r"not idle before the cell window after 0\.3 s"):
            self._cell(FakeServer("never"), timeout=0.3)

    def test_metrics_without_the_idle_counters_fail_the_cell(self) -> None:
        server = FakeServer()
        del server.m["queued_requests"]
        with self.assertRaisesRegex(HARNESS.NotIdle, "no integer queued_requests"):
            self._cell(server)

    def test_the_seed_retention_check_waits_for_idle(self) -> None:
        server = FakeServer("never")
        server.m["prefix_cache_entries"] = int(WORKLOAD["hot_cache_entries"])
        with mock.patch.object(HARNESS, "scrape", server.scrape), \
                mock.patch.object(HARNESS, "request", server.request), \
                mock.patch.object(HARNESS, "judge_seed", return_value=[]), \
                self.assertRaisesRegex(HARNESS.NotIdle, "not idle before the hot-set retention check"):
            HARNESS.seed_hot_set([self.ENDPOINT], WORKLOAD, "test", 0.3, {}, 32)

    def test_a_not_idle_run_fails_the_gate_with_its_own_line(self) -> None:
        summary = GATE.gate_summary(HARNESS, WORKLOAD, 32, [], [], [], ["q35: server not idle"])
        self.assertEqual(summary["verdict"], "FAIL")
        self.assertEqual(summary["failure_classes"], ["not_idle"])
        self.assertEqual(summary["fail_lines"], [GATE.FAIL_LINES["not_idle"]])
        self.assertEqual(summary["not_idle"], ["q35: server not idle"])


class FailLines(unittest.TestCase):
    def test_every_class_has_a_distinct_line(self) -> None:
        self.assertEqual(set(GATE.FAIL_LINES), set(GATE.CLASS_ORDER))
        self.assertEqual(len(set(GATE.FAIL_LINES.values())), len(GATE.FAIL_LINES))
        self.assertNotIn("regression", GATE.FAIL_LINES["usage_mismatch"].split("(")[0])


if __name__ == "__main__":
    unittest.main()
