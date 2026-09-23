"""Scenario contracts from wire/health/process fixtures; no process, HTTP or GPU I/O."""

import copy
import json
import unittest

from serving_lifecycle import account_generations
from serving_policy import evaluate_scenario
from serving_release import ServingGateError, account_attempts


NS = 1_000_000_000
BOOT = "edc35d2a-aedc-4528-8ab4-a91bf32dcb9b"
OWNER = {"pid": 100, "start_identity": BOOT + ":1234"}
PROCESS = {"pid": 100, "ppid": 90, "pgid": 100, "start_time": "1234",
           "identity_source": "linux_proc_start_ticks", "state": "S"}


def encoded(value):
    return json.dumps(value).encode()


def chat(prompt=20, completion=2, text="finished", reason="stop"):
    return {"model": "gate", "choices": [{"index": 0, "finish_reason": reason,
            "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                      "total_tokens": prompt + completion}}


def observation(key, start, end, status, body, **extra):
    return dict(id=key, started_ns=int(start * NS), finished_ns=int(end * NS),
                status=status, body=encoded(body), server_identity=dict(OWNER), **extra)


def health(key, start, status="ok", generation=7):
    return observation(key, start, start + .01, 200,
                       {"status": status, "worker": {"phase": "busy", "generation": generation}},
                       path="/health")


def request(key, **extra):
    return dict(id=key, model="gate", wire="chat_json", prompt_tokens={"min": 1, "max": 32},
                completion_tokens={"min": 1, "max": 64}, **extra)


def derive(bundle):
    bundle["accounting"] = account_attempts(bundle["config"]["requests"], bundle["attempts"])
    bundle["generations"] = (account_generations(bundle["accounting"], bundle["attempts"],
                                                 bundle["health_samples"])
                             if bundle["config"]["scenario"] == "completed_group" else None)
    return bundle


def completed_group():
    return derive({"config": {"scenario": "completed_group", "server_identity": dict(OWNER),
                              "requests": [request("first"), request("second")]},
                   "attempts": [observation("first", 1.2, 3.1, 200, chat()),
                                observation("second", 1.3, 3.2, 200, chat(text="other"))],
                   "health_samples": [health("before", 1), health("after", 3.4)]})


def drain():
    bundle = {"config": {"scenario": "drain", "server_identity": dict(OWNER),
                          "stop_reason": "drain_cell", "drain_timeout_ns": 3 * NS,
                          "retry_after_s": 3, "requests": [request("running", role="inflight"),
                          dict(id="new", model="gate", wire="chat_json", role="new_admission")]},
              "attempts": [observation("running", 1.2, 3.1, 200, chat()),
                           observation("new", 2.3, 2.4, 503,
                                       {"error": {"code": "draining", "type": "server_error",
                                                  "message": "server draining"}},
                                       headers=[["Retry-After", "3"]])],
              "health_samples": [health("before", 1), health("draining", 2.1, "draining"),
                                 health("after", 3.4, "draining")],
              "status_samples": [observation("ready", 2.2, 2.25, 503, {"status": "not_ready"},
                                               path="/readyz", headers=[["retry-after", "3"]])],
              "lifecycle": {
                  "schema": "memra-owned-server-v1", "state": "finished", "boot_id": BOOT,
                  "linux_subreaper": True, "server": dict(PROCESS), "errors": [],
                  "timeouts": {"startup": 1, "overall": 10, "drain": 3, "kill": 1},
                  "stop": {"reason": "drain_cell", "monotonic": 2.0,
                           "server_returncode_before_cleanup": None},
                  "server_exit": {"pid": 100, "wait_status": 0, "returncode": 0,
                                  "exit_code": 0, "signal": None, "before_cleanup": False,
                                  "identity": dict(PROCESS, state="Z"), "observed_monotonic": 4.0},
                  "observed_processes": [dict(PROCESS)],
                  "ownership_observations": [{"identity": dict(PROCESS),
                      "first_seen_monotonic": .5, "last_seen_monotonic": 1.1}],
                  "descendant_exits": [],
                  "descendant_retirements": [],
                  "signals": [{"pid": 100, "start_time": "1234", "signal": 15,
                               "result": "sent", "method": "pidfd", "monotonic": 2.05,
                               "sent_monotonic": 2.051, "finished_monotonic": 2.052}],
                  "cleanup": {"complete": True, "raw_final": True, "remaining": [],
                              "escalated": False, "elapsed_s": 2.1, "descendants_reaped": 0,
                              "finished_monotonic": 4.1, "descendants_retired": 0,
                              "reaping_scope": "linux_subreaper"},
                  "supervisor_exit": {"returncode": 0, "reaped": True}}}
    return derive(bundle)


class ServingPolicyTests(unittest.TestCase):
    def evaluate(self, bundle):
        return evaluate_scenario(**bundle)

    def refuses(self, bundle, rederive=False):
        with self.assertRaises(ServingGateError):
            if rederive:
                derive(bundle)
            self.evaluate(bundle)

    def test_completed_group_and_drain_use_observations_and_do_not_mutate(self):
        for fixture in (completed_group, drain):
            bundle = fixture()
            before = copy.deepcopy(bundle)
            result = self.evaluate(bundle)
            self.assertEqual(bundle, before)
            self.assertEqual(result["planned"], 2)
            self.assertEqual(result["server_identity"], OWNER)
            self.assertTrue(result["completed"])
            self.assertNotIn("passed", result)
        result = self.evaluate(drain())
        self.assertEqual(result["refused_ids"], ["new"])
        self.assertEqual(result["wire_counts"], {"clean_success": 1, "refused": 1})
        self.assertEqual(result["sigterm_before_ns"], 2_050_000_000)
        self.assertEqual(result["deadline_ns"], 5 * NS)

    def test_summaries_cannot_hide_a_truncated_200_or_approve_a_skip(self):
        bundle = completed_group()
        bundle["config"]["requests"][0]["wire"] = "chat_sse"
        bundle["attempts"][0]["body"] = b'data: {"model":"gate","choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n'
        derive(bundle)
        bundle["accounting"].update(passed=True, counts={"clean_success": 2}, attempted=2)
        bundle["generations"].update(passed=True, generation_affected_requests=0)
        self.refuses(bundle)
        for kind in ("skip", "unknown", None):
            bundle = completed_group(); bundle["config"]["scenario"] = kind
            self.refuses(bundle)
        for field in ("passed", "skip", "optional"):
            bundle = completed_group(); bundle["config"][field] = True
            self.refuses(bundle)

    def test_empty_duplicate_missing_and_unplanned_ids_are_rejected(self):
        for location in ("attempts", "health_samples", "config", "accounting", "generations"):
            for mutation in ("empty_list", "blank_id", "duplicate", "missing", "unknown"):
                bundle = completed_group()
                key = "requests" if location in ("config", "accounting", "generations") else None
                rows = bundle[location][key] if key else bundle[location]
                if mutation == "empty_list": rows.clear()
                elif mutation == "blank_id": rows[0]["id"] = ""
                elif mutation == "duplicate": rows.append(copy.deepcopy(rows[0]))
                elif mutation == "missing": rows.pop()
                else: rows.append(dict(rows[0], id="unplanned"))
                # Extra health samples are meaningful if ordered; this duplicate-timed
                # addition deliberately cannot extend a valid observation interval.
                with self.subTest(location=location, mutation=mutation): self.refuses(bundle)

    def test_required_token_ranges_and_usage_reject_survivor_only_success(self):
        for counts in [(0, 2), (33, 2), (20, 0), (20, 65), (True, 2)]:
            bundle = completed_group()
            bundle["attempts"][0]["body"] = encoded(chat(*counts))
            self.refuses(bundle, rederive=True)
        for value in (None, {}, {"min": 32}, {"min": 33, "max": 32}, {"min": True, "max": 32}):
            bundle = completed_group(); bundle["config"]["requests"][0]["prompt_tokens"] = value
            self.refuses(bundle)
        for field in ("prompt_tokens", "completion_tokens"):
            bundle = completed_group(); del bundle["config"]["requests"][0][field]
            self.refuses(bundle)
        bundle = completed_group(); bundle["attempts"][0]["body"] = encoded(chat(text=""))
        self.refuses(bundle, rederive=True)

    def test_terminal_usage_chunk_and_native_json_are_supported(self):
        bundle = completed_group()
        bundle["config"]["requests"][0]["wire"] = "chat_sse"
        data = [{"model": "gate", "choices": [{"index": 0, "delta": {"reasoning": "thinking"}, "finish_reason": "length"}]},
                {"model": "gate", "choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22}}]
        bundle["attempts"][0]["body"] = b"".join(b"data: " + encoded(v) + b"\n\n" for v in data) + b"data: [DONE]\n\n"
        bundle["config"]["requests"][1]["wire"] = "native_json"
        bundle["attempts"][1]["body"] = encoded({"model": "gate", "text": "result", "tokens": [4, 5],
            "n_tokens": 2, "prompt_tokens": 20, "cached_tokens": 0, "elapsed_s": .1, "stop_reason": "MaxNew"})
        result = self.evaluate(derive(bundle))
        self.assertEqual([r["completion_tokens"] for r in result["completed"]], [2, 2])

    def test_generation_changes_and_stale_derived_rows_are_rejected(self):
        bundle = completed_group(); bundle["health_samples"][-1]["body"] = encoded({"worker": {"generation": 8}})
        derive(bundle)
        bundle["generations"].update(respawns_observed=0, generation_affected_requests=0, passed=True)
        self.refuses(bundle)
        for field, value in [("outcome", "clean_success"), ("wall_ns", 1)]:
            bundle = completed_group()
            if field == "outcome":
                bundle["attempts"][0]["body"] = b"invalid"
            else:
                bundle["accounting"]["requests"][0][field] = value
            self.refuses(bundle)
        bundle = completed_group(); bundle["generations"]["requests"][0]["generation_after"] = 8
        self.refuses(bundle)

    def test_all_capture_identities_and_intervals_are_checked(self):
        for location in ("attempts", "health_samples", "status_samples"):
            for change in ({"server_identity": {"pid": 101, "start_identity": OWNER["start_identity"]}},
                           {"server_identity": {"pid": 100, "start_identity": BOOT + ":9999"}},
                           {"started_ns": True}, {"finished_ns": 0}):
                bundle = drain(); bundle[location][0].update(change)
                with self.subTest(location=location, change=change): self.refuses(bundle)

    def test_drain_needs_real_primary_term_with_same_birth(self):
        for mutation in ("missing", "wrong_pid", "wrong_birth", "wrong_signal", "duplicate", "not_sent", "no_method"):
            bundle = drain(); signals = bundle["lifecycle"]["signals"]
            if mutation == "missing": signals.clear()
            elif mutation == "duplicate": signals.append(copy.deepcopy(signals[0]))
            else:
                key, value = {"wrong_pid": ("pid", 101), "wrong_birth": ("start_time", "9999"),
                              "wrong_signal": ("signal", 9), "not_sent": ("result", "already_exited"),
                              "no_method": ("method", None)}[mutation]
                signals[0][key] = value
            with self.subTest(mutation=mutation): self.refuses(bundle)
        for field in ("boot_id", "server", "server_exit"):
            bundle = drain()
            if field == "boot_id": bundle["lifecycle"][field] = "7ca6e379-cc57-435e-ad9d-84d63f73d9e6"
            elif field == "server": bundle["lifecycle"][field]["start_time"] = "9999"
            else: bundle["lifecycle"][field]["identity"]["start_time"] = "9999"
            self.refuses(bundle)

    def test_drain_requires_crossing_inflight_and_later_admission(self):
        for which, start, end in [(0, 2.06, 3.1), (0, 1.2, 2.05), (0, 1.2, 5.1),
                                 (1, 2.04, 2.4), (1, 2.3, 5.1), (1, 4.0, 4.1)]:
            bundle = drain(); bundle["attempts"][which].update(started_ns=int(start * NS), finished_ns=int(end * NS))
            with self.subTest(which=which, start=start, end=end): self.refuses(bundle, rederive=True)
        for role in ("inflight", "new_admission"):
            bundle = drain()
            keep = [r for r in bundle["config"]["requests"] if r["role"] == role]
            bundle["config"]["requests"] = keep
            bundle["attempts"] = [a for a in bundle["attempts"] if a["id"] in {r["id"] for r in keep}]
            self.refuses(bundle, rederive=True)

    def test_health_200_is_not_draining_and_ready_503_needs_retry(self):
        bundle = drain()
        for row in bundle["health_samples"]:
            row["body"] = encoded({"status": "ok", "worker": {"generation": 7}})
        self.refuses(bundle, rederive=True)
        for changes in ({"status": 200}, {"body": b'{"status":"ready"}'},
                        {"body": b'{"status":"not_ready","status":"ready"}'},
                        {"path": "/health"}, {"headers": []},
                        {"transport_error": {"kind": "read", "message": "partial"}},
                        {"started_ns": NS}, {"finished_ns": 5 * NS + 1}):
            bundle = drain(); bundle["status_samples"][0].update(changes)
            with self.subTest(changes=changes): self.refuses(bundle)
        bundle = drain(); bundle["status_samples"] *= 2; self.refuses(bundle)
        bundle = drain(); bundle["status_samples"] = []; self.refuses(bundle)

    def test_new_admission_must_be_draining_not_other_503_or_200(self):
        for status, payload in [(200, chat()), (429, {"error": {"code": "draining", "type": "server_error", "message": "busy"}}),
                                (503, {"error": {"code": "overloaded", "type": "server_error", "message": "busy"}}),
                                (503, {"error": {"code": "draining", "type": "other", "message": "busy"}})]:
            bundle = drain(); bundle["attempts"][1].update(status=status, body=encoded(payload))
            with self.subTest(status=status, payload=payload): self.refuses(bundle, rederive=True)
        for value in ([], [["Retry-After", "4"]], [["Retry-After", "0"]],
                      [["Retry-After", "Tue, 21 Sep 2026 00:00:00 GMT"]],
                      [["Retry-After", "3"], ["retry-after", "3"]], {"Retry-After": "3"}):
            bundle = drain(); bundle["attempts"][1]["headers"] = value
            with self.subTest(value=value): self.refuses(bundle)

    def test_cleanup_summary_cannot_hide_raw_exit_signal_or_errors(self):
        for section, change in [("server_exit", {"returncode": -9, "wait_status": 9, "exit_code": None, "signal": 9}),
                                ("server_exit", {"before_cleanup": True}),
                                ("stop", {"server_returncode_before_cleanup": 0}),
                                ("cleanup", {"remaining": [dict(PROCESS)]}),
                                ("cleanup", {"escalated": True}),
                                ("cleanup", {"raw_final": False}),
                                ("supervisor_exit", {"returncode": 1})]:
            bundle = drain(); bundle["lifecycle"][section].update(change)
            bundle["lifecycle"]["passed"] = True
            with self.subTest(section=section, change=change): self.refuses(bundle)
        for change in ({"errors": ["sync failed"]}, {"controller_error": "lost"},
                       {"linux_subreaper": False}, {"emergency_signals": []}):
            bundle = drain(); bundle["lifecycle"].update(change); self.refuses(bundle)
        for section in ("signals", "server_exit", "cleanup", "observed_processes", "supervisor_exit"):
            bundle = drain(); del bundle["lifecycle"][section]; self.refuses(bundle)

    def test_deadlines_are_frozen_finite_and_no_other_stop_cause_is_a_drain(self):
        for field in ("elapsed_s",):
            for value in (3.000000001, -1, True, float("nan"), float("inf"), 10**5000):
                bundle = drain(); bundle["lifecycle"]["cleanup"][field] = value
                with self.subTest(value_type=type(value).__name__): self.refuses(bundle)
        for change in ({"observed_monotonic": 5.1}, {"observed_monotonic": 1.9}):
            bundle = drain(); bundle["lifecycle"]["server_exit"].update(change); self.refuses(bundle)
        bundle = drain(); bundle["lifecycle"]["timeouts"]["drain"] = 10; self.refuses(bundle)
        for reason in ("cancelled", "overall_timeout", "controller_eof", "supervisor_signal_15"):
            bundle = drain(); bundle["config"]["stop_reason"] = reason
            bundle["lifecycle"]["stop"]["reason"] = reason
            self.refuses(bundle)

    def test_late_healthy_samples_cannot_hide_a_worker_restart(self):
        bundle = drain()
        bundle["health_samples"][-1]["body"] = encoded({"status": "draining", "worker": {"generation": 8}})
        self.refuses(bundle, rederive=True)

    def test_drain_last_health_before_last_response_uses_lifecycle_boundary(self):
        # Pure ordering fixture: the final probe completes while the stream is
        # active; its terminal response is followed immediately by listener exit.
        # There is no invented post-response HTTP call or final worker counter.
        bundle = drain()
        bundle["config"]["requests"][0]["wire"] = "chat_sse"
        bundle["attempts"][0]["body"] = (
            b'data: {"model":"gate","choices":[{"index":0,"delta":{"content":"finished"},"finish_reason":null}]}\n\n'
            b'data: {"model":"gate","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            b'data: {"model":"gate","choices":[],"usage":{"prompt_tokens":20,"completion_tokens":2,"total_tokens":22}}\n\n'
            b'data: [DONE]\n\n')
        bundle["health_samples"][-1] = health("last_alive", 2.5, "draining")
        bundle["lifecycle"]["server_exit"]["observed_monotonic"] = 3.15
        bundle["lifecycle"]["cleanup"]["elapsed_s"] = 1.2
        bundle["lifecycle"]["cleanup"]["finished_monotonic"] = 3.2
        derive(bundle)
        # The old full-bracket census cannot be used on this legitimate ordering.
        with self.assertRaisesRegex(ServingGateError, "do not bracket"):
            account_generations(bundle["accounting"], bundle["attempts"], bundle["health_samples"])
        result = self.evaluate(bundle)
        self.assertEqual(result["terminal_boundary"], "owned_process_exit_and_cleanup")
        census = result["generation_observations"]
        self.assertEqual(census["observed_generation_changes"], 0)
        self.assertEqual(census["observed_generation_increments"], 0)
        self.assertEqual(census["generation_unqualified_ids"], ["running"])
        self.assertEqual(census["clean_latency_ns"], [])
        self.assertIsNone(census["terminal_generation_evidence"])
        self.assertIs(census["lifecycle_exit_proves_final_generation"], False)
        completion = result["completed"][0]
        self.assertEqual(completion["generation_scope"], "unobserved_tail")
        self.assertIsNone(completion["generation_affected"])
        self.assertIsNone(completion["clean_latency_ns"])
        self.assertEqual(completion["completion_tokens"], 2)
        self.assertEqual(result["refused_ids"], ["new"])
        self.assertGreater(completion["finished_ns"], census["last_health_finished_ns"])

    def test_drain_censored_tail_cannot_skip_terminal_refusal_or_cleanup(self):
        for corruption in ("truncated", "not_refused", "dirty_cleanup", "wrong_owner", "no_draining_health"):
            bundle = drain()
            bundle["health_samples"][-1] = health("last_alive", 2.5, "draining")
            if corruption == "truncated":
                bundle["config"]["requests"][0]["wire"] = "chat_sse"
                bundle["attempts"][0]["body"] = b'data: {"model":"gate","choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n'
            elif corruption == "not_refused":
                bundle["attempts"][1].update(status=200, body=encoded(chat()))
            elif corruption == "dirty_cleanup":
                bundle["lifecycle"]["cleanup"]["remaining"] = [dict(PROCESS)]
            elif corruption == "wrong_owner":
                bundle["lifecycle"]["server"]["start_time"] = "other"
            else:
                for sample in bundle["health_samples"]:
                    sample["body"] = encoded({"status": "ok", "worker": {"generation": 7}})
            with self.subTest(corruption=corruption):
                self.refuses(bundle, rederive=True)

    def test_interval_only_generation_observations_do_not_manufacture_coverage(self):
        bundle = drain()
        # Even the new refusal can finish beyond the last health interval.
        bundle["health_samples"].pop()
        result = self.evaluate(bundle)
        self.assertEqual(result["generation_observations"]["generation_unqualified_ids"], ["running", "new"])
        self.assertTrue(all(row["generation_affected"] is None for row in
                            result["generation_observations"]["requests"]))
        for change in ("observed_restart", "decreasing", "malformed", "after_exit", "overlap", "no_before"):
            broken = copy.deepcopy(bundle)
            if change in ("observed_restart", "decreasing", "malformed"):
                value = {"observed_restart": 9, "decreasing": 6, "malformed": True}[change]
                broken["health_samples"][-1]["body"] = encoded({"status": "draining", "worker": {"generation": value}})
            elif change == "after_exit":
                broken["health_samples"].append(health("post_exit", 4.1, "ok"))
            elif change == "overlap":
                broken["health_samples"][0]["finished_ns"] = int(2.11 * NS)
            else:
                broken["health_samples"][0].update(started_ns=int(1.3 * NS), finished_ns=int(1.4 * NS))
            with self.subTest(change=change): self.refuses(broken)

    def test_strict_completed_group_bracketing_is_unchanged(self):
        bundle = completed_group()
        # Supply the previously derived strict result; a now-shorter raw health
        # interval cannot be laundered by those stale clean-generation rows.
        bundle["health_samples"][-1] = health("last_alive", 2.5)
        self.refuses(bundle)
        bundle = completed_group()
        bundle["generations"] = None
        self.refuses(bundle)
        self.assertEqual(len(self.evaluate(completed_group())["completed"]), 2)

    def test_drain_rejects_precomputed_whole_request_generation_claims(self):
        bundle = drain()
        bundle["generations"] = account_generations(bundle["accounting"], bundle["attempts"],
                                                     bundle["health_samples"])
        self.refuses(bundle)
        # It still computes brackets if real during-drain health happens to cover
        # a completed request, without requiring them or fabricating final health.
        bundle["generations"] = None
        census = self.evaluate(bundle)["generation_observations"]
        self.assertEqual(census["generation_unqualified_ids"], [])
        self.assertEqual(census["clean_latency_ns"], [int(1.9 * NS)])

    def test_server_exit_before_buffered_terminal_and_probe_reads_finish(self):
        # Capture-order fixture, not a live socket run: all requests reached the
        # collector's verified owner before exit. The server wrote replies and
        # exited; the clients then drained their complete buffered wire bodies.
        # The pure predicate has no source for a server last-write timestamp.
        bundle = drain()
        bundle["config"]["requests"][0]["wire"] = "chat_sse"
        bundle["attempts"][0]["body"] = (
            b'data: {"model":"gate","choices":[{"index":0,"delta":{"content":"finished"},"finish_reason":"stop"}]}\n\n'
            b'data: {"model":"gate","choices":[],"usage":{"prompt_tokens":20,"completion_tokens":2,"total_tokens":22}}\n\n'
            b'data: [DONE]\n\n')
        bundle["attempts"][0]["finished_ns"] = 4_600_000_000
        bundle["attempts"][1]["finished_ns"] = 4_500_000_000
        bundle["status_samples"][0]["finished_ns"] = 4_300_000_000
        bundle["health_samples"][-1]["finished_ns"] = 4_200_000_000
        result = self.evaluate(derive(bundle))
        self.assertEqual(result["exit_after_ns"], 4 * NS)
        self.assertEqual(result["deadline_ns"], result["capture_deadline_ns"])
        self.assertEqual(result["capture_deadline_ns"], 5 * NS)
        self.assertEqual(set(result["client_reads_finished_after_exit_observation"]),
                         {"running", "new", "ready", "after"})
        self.assertGreater(result["completed"][0]["finished_ns"], result["exit_after_ns"])
        self.assertIsNone(result["completed"][0]["generation_affected"])
        self.assertIsNone(result["completed"][0]["clean_latency_ns"])
        census = result["generation_observations"]
        self.assertEqual(census["last_health_finished_ns"], 4_200_000_000)
        self.assertEqual(census["last_generation_observation_latest_ns"], 4 * NS)
        self.assertEqual(census["generation_unqualified_ids"], ["running", "new"])
        self.assertIsNone(census["terminal_generation_evidence"])

        # Same complete buffered data cannot authorize a new post-exit capture,
        # a foreign responder, or a client that exceeds the frozen allowance.
        for location, index in [("attempts", 0), ("attempts", 1),
                                ("status_samples", 0), ("health_samples", -1)]:
            for corruption in ("start_at_exit", "start_after_exit", "late_read", "foreign"):
                broken = copy.deepcopy(bundle)
                row = broken[location][index]
                if corruption == "start_at_exit": row["started_ns"] = 4 * NS
                elif corruption == "start_after_exit": row["started_ns"] = 4 * NS + 1
                elif corruption == "late_read": row["finished_ns"] = 5 * NS + 1
                else: row["server_identity"] = {"pid": 101, "start_identity": BOOT + ":1235"}
                with self.subTest(location=location, corruption=corruption):
                    self.refuses(broken, rederive=True)
        # Client read allowance never extends the actual shutdown/cleanup budget.
        broken = copy.deepcopy(bundle)
        broken["lifecycle"]["server_exit"]["observed_monotonic"] = 5.01
        self.refuses(broken)
        broken = copy.deepcopy(bundle)
        broken["lifecycle"]["cleanup"]["elapsed_s"] = 3.01
        self.refuses(broken)

    def test_buffered_body_requires_terminal_and_cannot_extend_read_deadline(self):
        for location, index in [("attempts", 0), ("attempts", 1),
                                ("status_samples", 0), ("health_samples", -1)]:
            bundle = drain()
            bundle[location][index]["finished_ns"] = 5 * NS
            self.evaluate(derive(bundle))  # inclusive frozen allowance
            bundle[location][index]["finished_ns"] += 1
            self.refuses(bundle, rederive=True)
        bundle = drain()
        bundle["config"]["requests"][0]["wire"] = "chat_sse"
        bundle["attempts"][0].update(finished_ns=4_600_000_000,
            body=b'data: {"model":"gate","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}\n\n')
        self.refuses(bundle, rederive=True)

    def test_orphan_term_after_primary_exit_is_allowed_but_premature_term_is_not(self):
        bundle = drain()
        child = dict(PROCESS, pid=101, ppid=100, start_time="1235")
        receipt = bundle["lifecycle"]
        receipt["observed_processes"].append(child)
        receipt["ownership_observations"].append(dict(identity=child, first_seen_monotonic=1., last_seen_monotonic=3.99))
        receipt["signals"].append(dict(pid=101, start_time="1235", signal=15,
                                       result="sent", method="pidfd", monotonic=4.01,
                                       sent_monotonic=4.011, finished_monotonic=4.012))
        receipt["descendant_exits"].append(dict(pid=101, wait_status=0, returncode=0,
            exit_code=0, signal=None, identity=dict(child, state="Z"), observed_monotonic=4.04))
        receipt["cleanup"]["descendants_reaped"] = 1
        self.evaluate(bundle)
        receipt["descendant_exits"][0].update(wait_status=15, returncode=-15, exit_code=None, signal=15)
        self.evaluate(bundle)
        receipt["signals"][-1]["monotonic"] = 2.06
        self.refuses(bundle)

    def test_unexplained_child_signal_exit_does_not_pass_as_clean(self):
        bundle = drain()
        child = dict(PROCESS, pid=101, ppid=100, start_time="1235")
        bundle["lifecycle"]["observed_processes"].append(child)
        bundle["lifecycle"]["ownership_observations"].append(dict(identity=child, first_seen_monotonic=1., last_seen_monotonic=3.99))
        bundle["lifecycle"]["descendant_exits"].append(dict(pid=101, wait_status=15, returncode=-15,
            exit_code=None, signal=15, identity=child, observed_monotonic=4.04))
        bundle["lifecycle"]["cleanup"]["descendants_reaped"] = 1
        self.refuses(bundle)

    def test_inclusive_token_limits_and_nanosecond_deadline(self):
        bundle = completed_group()
        bundle["attempts"][0]["body"] = encoded(chat(prompt=1, completion=1, text=" "))
        bundle["attempts"][1]["body"] = encoded(chat(prompt=32, completion=64))
        self.evaluate(derive(bundle))
        bundle = drain()
        bundle["lifecycle"]["server_exit"]["observed_monotonic"] = 5.0
        bundle["lifecycle"]["cleanup"]["elapsed_s"] = 3.0
        bundle["lifecycle"]["cleanup"]["finished_monotonic"] = 5.0
        self.evaluate(bundle)
        bundle["lifecycle"]["server_exit"]["observed_monotonic"] = 5.000000001
        self.refuses(bundle)

    def test_invalid_wire_and_transport_cannot_be_green_via_summary(self):
        for wire in (None, [], "skip"):
            bundle = completed_group(); bundle["config"]["requests"][0]["wire"] = wire
            self.refuses(bundle)
        bundle = completed_group()
        bundle["attempts"][0]["transport_error"] = {"kind": "cancelled", "message": "closed"}
        derive(bundle)
        bundle["accounting"]["counts"] = {"clean_success": 2}
        self.refuses(bundle)

    def test_pre_send_time_cannot_prove_request_spanned_delayed_sigterm(self):
        bundle = drain()
        event = bundle["lifecycle"]["signals"][0]
        # Client terminal at 3.1 followed a lookup START at 2.05, but it preceded
        # the actual signal syscall. Move independent probes beyond delivery so
        # rejection is specifically the inflight completion relation.
        event.update(sent_monotonic=3.2, finished_monotonic=3.21)
        bundle["attempts"][1].update(started_ns=3_300_000_000, finished_ns=3_350_000_000)
        bundle["status_samples"][0].update(started_ns=3_360_000_000, finished_ns=3_370_000_000)
        bundle["health_samples"][1] = health("draining", 3.22, "draining")
        self.refuses(bundle, rederive=True)
        bundle["attempts"][0]["finished_ns"] = 3_250_000_000
        result = self.evaluate(derive(bundle))
        self.assertEqual(result["sigterm_after_ns"], 3_200_000_000)
        for field in ("sent_monotonic", "finished_monotonic"):
            broken = drain(); del broken["lifecycle"]["signals"][0][field]
            self.refuses(broken)
        for values in [dict(sent_monotonic=2.04), dict(finished_monotonic=2.04),
                       dict(sent_monotonic=2.06, finished_monotonic=2.052)]:
            broken = drain(); broken["lifecycle"]["signals"][0].update(values)
            self.refuses(broken)

    def test_known_child_needs_a_reap_or_actual_identity_retirement(self):
        bundle = drain(); record = bundle["lifecycle"]
        child = dict(PROCESS, pid=101, ppid=100, start_time="1235")
        record["observed_processes"].append(child)
        record["ownership_observations"].append(dict(identity=child,
            first_seen_monotonic=1., last_seen_monotonic=2.1))
        with self.assertRaisesRegex(ServingGateError, "lacks exit or identity retirement"):
            self.evaluate(bundle)
        # Primary reaped its own worker. The supervisor cannot assert a wait
        # status; it directly observes that the recorded birth no longer exists.
        retirement = dict(pid=101, identity=child, kind="identity_disappeared", source="linux_proc_stat",
            observation="absent", replacement_start_time=None, exit_status=None,
            last_seen_monotonic=2.1, started_monotonic=2.2, finished_monotonic=2.21)
        record["descendant_retirements"] = [retirement]
        record["cleanup"]["descendants_retired"] = 1
        result = self.evaluate(bundle)
        self.assertEqual(result["retired_without_wait_status"], [dict(pid=101, start_time="1235", exit_status=None)])
        for kind in ("omit", "duplicate", "same_birth", "read_error", "invent_status", "stale_last_seen", "late"):
            broken = copy.deepcopy(bundle); rows = broken["lifecycle"]["descendant_retirements"]
            if kind == "omit":
                rows.clear(); broken["lifecycle"]["cleanup"]["descendants_retired"] = 0
            elif kind == "duplicate":
                rows.append(copy.deepcopy(rows[0])); broken["lifecycle"]["cleanup"]["descendants_retired"] = 2
            elif kind == "same_birth": rows[0].update(observation="different_birth", replacement_start_time="1235")
            elif kind == "read_error": rows[0]["observation"] = "permission_error"
            elif kind == "invent_status": rows[0]["exit_status"] = 0
            elif kind == "stale_last_seen": rows[0]["last_seen_monotonic"] = 1.9
            else: rows[0].update(started_monotonic=4.4, finished_monotonic=4.5)
            with self.subTest(kind=kind): self.refuses(broken)
        retirement.update(observation="different_birth", replacement_start_time="9999")
        self.evaluate(bundle)  # Old birth retired; no signal is authorized to new birth.

    def test_reap_and_action_must_precede_actual_cleanup_finish(self):
        bundle = drain(); record = bundle["lifecycle"]
        child = dict(PROCESS, pid=101, ppid=90, start_time="1235")
        record["observed_processes"].append(child)
        record["ownership_observations"].append(dict(identity=child,
            first_seen_monotonic=1., last_seen_monotonic=3.99))
        record["descendant_exits"] = [dict(pid=101, wait_status=0, returncode=0, exit_code=0,
            signal=None, identity=child, observed_monotonic=4.04)]
        record["cleanup"]["descendants_reaped"] = 1
        self.evaluate(bundle)
        record["descendant_exits"][0]["observed_monotonic"] = 4.5
        with self.assertRaisesRegex(ServingGateError, "after actual cleanup finish"):
            self.evaluate(bundle)  # Before deadline=5, but after actual cleanup=4.1.
        record["cleanup"].update(finished_monotonic=4.6, elapsed_s=2.6)
        self.evaluate(bundle)
        record["cleanup"]["elapsed_s"] = 2.1
        self.refuses(bundle)  # Direct finish and elapsed cannot contradict each other.
        for mutation in ("late_action", "missing_finish", "premature_finish"):
            broken = drain()
            if mutation == "late_action": broken["lifecycle"]["signals"][0]["finished_monotonic"] = 4.2
            elif mutation == "missing_finish": del broken["lifecycle"]["cleanup"]["finished_monotonic"]
            else: broken["lifecycle"]["cleanup"].update(finished_monotonic=3.9, elapsed_s=1.9)
            with self.subTest(mutation=mutation): self.refuses(broken)


if __name__ == "__main__":
    unittest.main()
