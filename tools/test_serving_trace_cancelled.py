"""Typed partial-return protocol controls. CPU fixtures never establish native C4."""

import copy
import json
import os
from pathlib import Path
import unittest

from serving_release import ServingGateError
from serving_trace import decode_lifecycle_log, validate_cancel_trace
from test_serving_trace import KEY, Script, fixture, options, peer, wire


def version2(rows):
    rows = copy.deepcopy(rows)
    for row in rows:
        row["schema"] = "memra-request-lifecycle-v2"
        if row["quantum"] is not None:
            row["quantum"]["cancelled_at_boundary"] = None
    return rows


def partial():
    rows = version2(fixture(remaining=None))
    end = next(i for i, r in enumerate(rows) if r["event"] == "prime_quantum_end")
    rows[end]["event"] = "prime_quantum_cancelled"
    for row in rows[end:]:
        row["phase"] = "prime_cancelled"
        row["quantum"].update(completed=False, remaining_chunks=None,
            cancelled_at_boundary={"chunk": 1, "rows_done": 8, "rows_total": 16})
    return rows


def closed_rows(rows):
    return [r for r in rows if r["quantum"] is not None and r["quantum"]["end_ns"] is not None]


class TypedReturnTests(unittest.TestCase):
    def refuse(self, rows, message="."):
        with self.assertRaisesRegex(ServingGateError, message):
            validate_cancel_trace(wire(rows), **options())

    def test_partial_return_has_distinct_facts_without_completing_requested_rows(self):
        rows = partial()
        before = copy.deepcopy(rows)
        facts = validate_cancel_trace(wire(rows + peer()), **options())
        self.assertEqual(rows, before)
        q = facts["drop_spanning_quantum"]
        self.assertIs(q["completed"], False)
        self.assertIsNone(q["remaining_chunks"])
        self.assertEqual(q["cancelled_at_boundary"], {"chunk": 1, "rows_done": 8, "rows_total": 16})
        self.assertEqual(facts["retirement"]["phase"], "prime_cancelled")
        self.assertEqual(facts["retirement"]["retirement_site"], "ActiveSession")
        self.assertEqual(len(facts["decoded_log"]["census"]), 2)
        # Unqualified normal peer history stays in the census, without being promoted.
        self.assertEqual(facts["decoded_log"]["census"][1]["reported_errors"],
                         ["trace_ended_without_explicit_retirement"])

    def test_historical_v1_and_v2_success_unknown_remaining_still_mean_the_same(self):
        for scenario in ("cancel_queued", "cancel_prime", "cancel_decode"):
            for remaining in (None, 0, 2):
                with self.subTest(scenario=scenario, remaining=remaining):
                    old = fixture(scenario, remaining=remaining)
                    a = validate_cancel_trace(wire(old), **options(scenario))
                    b = validate_cancel_trace(wire(version2(old)), **options(scenario))
                    self.assertEqual(a["target"], b["target"])
                    self.assertEqual(a["retirement"]["phase"], b["retirement"]["phase"])
                    if scenario == "cancel_prime":
                        self.assertIs(b["drop_spanning_quantum"]["completed"], True)
                        self.assertEqual(b["drop_spanning_quantum"]["remaining_chunks"], remaining)

    def test_schema_version_is_exact_and_immutable_in_every_trace(self):
        for version in ("memra-request-lifecycle-v1", "memra-request-lifecycle-v3"):
            rows = partial()
            for row in rows: row["schema"] = version
            self.refuse(rows)
        rows = partial(); rows[0]["schema"] = "memra-request-lifecycle-v1"
        self.refuse(rows, "changed lifecycle schema")
        rows = partial(); rows[-1]["schema"] = "memra-request-lifecycle-v1"
        rows[-1]["quantum"].pop("cancelled_at_boundary")
        self.refuse(rows)
        # Even an unrelated trace cannot quietly change its producer schema.
        extra = peer(); extra[0]["schema"] = "memra-request-lifecycle-v2"
        self.refuse(partial() + extra, "changed lifecycle schema")

    def test_v2_bare_displaced_escaped_malformed_and_nonfinite_lines_refuse(self):
        good = wire(partial())
        for tail in (b'{"schema":"memra-request-lifecycle-v2"}\n',
                     b'{"\\u0073chema":"memra-request-lifecycle-v2",bad\n',
                     b'{"schema":"other","schema":"memra-request-lifecycle-v2"}\n',
                     b'noise [request-lifecycle] {"schema":"memra-request-lifecycle-v2"}\n'):
            with self.subTest(tail=tail), self.assertRaises(ServingGateError):
                decode_lifecycle_log(good + tail)
        for before, after in ((b'"rows_done":8', b'"rows_done":8,"rows_done":8'),
                              (b'"rows_done":8', b'"rows_done":NaN')):
            with self.assertRaises(ServingGateError): decode_lifecycle_log(good.replace(before, after))
        with self.assertRaises(ServingGateError): decode_lifecycle_log(good[:-1])

    def test_invalid_boundary_counts_types_and_call_identity_refuse(self):
        for name, value in (("chunk", -1), ("chunk", True), ("chunk", 8),
                            ("rows_done", 0), ("rows_done", 16), ("rows_done", 17),
                            ("rows_done", "8"), ("rows_total", 15), ("rows_total", 17),
                            ("rows_total", 2**64)):
            with self.subTest(name=name, value=value):
                rows = partial()
                for row in closed_rows(rows): row["quantum"]["cancelled_at_boundary"][name] = value
                self.refuse(rows)
        rows = partial()
        for row in closed_rows(rows):
            row["quantum"].update(rows=17)
            row["quantum"]["cancelled_at_boundary"]["rows_total"] = 17
        self.refuse(rows, "identity/rows")
        for key in ("chunk", "rows_done", "rows_total"):
            rows = partial()
            for row in closed_rows(rows): row["quantum"]["cancelled_at_boundary"].pop(key)
            self.refuse(rows)

    def test_cancel_metadata_cannot_be_added_to_success_or_ordinary_failure(self):
        for kind in ("success", "remaining_zero", "ordinary_event", "missing_boundary", "new_work"):
            rows = partial()
            for row in closed_rows(rows):
                q = row["quantum"]
                if kind == "success": q["completed"] = True
                if kind == "remaining_zero": q["remaining_chunks"] = 0
                if kind == "missing_boundary": q["cancelled_at_boundary"] = None
            if kind == "ordinary_event":
                next(r for r in rows if r["event"] == "prime_quantum_cancelled")["event"] = "prime_quantum_end"
            if kind == "new_work":
                next(r for r in rows if r["event"] == "receiver_closed")["event"] = "decode_start"
            with self.subTest(kind=kind): self.refuse(rows)
        # A genuine failed return remains refused even if it later retires cleanly,
        # including when producer validity flags are dishonestly flipped to true.
        self.refuse(version2(fixture(failed=True)), "failed or missing quantum")

    def test_drop_receiver_return_and_retirement_order_cannot_be_invented(self):
        for change in ("missing_drop", "drop_after_return", "receiver_before_return", "early_retirement",
                       "eof", "wrong_site", "unknown_site", "missing_final", "overflow", "requeue"):
            rows = partial()
            ix = {r["event"]: i for i,r in enumerate(rows)}
            if change == "missing_drop": rows.pop(ix["http_body_drop"])
            elif change == "drop_after_return": rows[ix["http_body_drop"]], rows[ix["prime_quantum_cancelled"]] = rows[ix["prime_quantum_cancelled"]], rows[ix["http_body_drop"]]
            elif change == "receiver_before_return": rows[ix["receiver_closed"]], rows[ix["prime_quantum_cancelled"]] = rows[ix["prime_quantum_cancelled"]], rows[ix["receiver_closed"]]
            elif change == "early_retirement": rows[ix["retired"]], rows[ix["prime_quantum_cancelled"]] = rows[ix["prime_quantum_cancelled"]], rows[ix["retired"]]
            elif change == "eof": rows[ix["http_body_drop"]].update(event="http_body_eof", http_body="normal_eof")
            elif change in ("wrong_site", "unknown_site"):
                for row in rows[ix["retired"]:]: row["retirement_site"] = "WorkerQueue" if change == "wrong_site" else None
            elif change == "missing_final": rows.pop()
            elif change == "overflow":
                for row in rows[ix["receiver_closed"]:]: row["receiver_close_cause"] = "event_queue_overflow"
                rows[ix["receiver_closed"]]["observed_close_cause"] = "event_queue_overflow"
            else: rows[ix["receiver_closed"]]["event"] = "requeued"
            with self.subTest(change=change): self.refuse(rows)

    def test_self_consistent_late_drop_and_receiver_first_still_refuse(self):
        # Rebuild contiguous snapshots instead of relying on trivial sequence gaps.
        for receiver_first in (False, True):
            s = Script(); s.state["schema"] = "memra-request-lifecycle-v2"
            s.rows[0]["schema"] = s.state["schema"]; s.bind(); s.quantum_start()
            s.state["quantum"]["cancelled_at_boundary"] = None
            s.rows[-1]["quantum"]["cancelled_at_boundary"] = None
            if receiver_first:
                s.add("http_body_drop", http_body="dropped_before_eof")
                s.add("receiver_closed", receiver_close_cause="receiver_dropped", observed_close_cause="receiver_dropped")
            s.add("prime_quantum_cancelled", phase="prime_cancelled", quantum_active=False,
                  quantum={**s.state["quantum"], "end_ns":s.next_ns(), "completed":False,
                           "remaining_chunks":None, "cancelled_at_boundary":{"chunk":0,"rows_done":8,"rows_total":16}})
            if not receiver_first:
                s.add("http_body_drop", http_body="dropped_before_eof")
                s.add("receiver_closed", receiver_close_cause="receiver_dropped", observed_close_cause="receiver_dropped")
            s.add("retired", retirement="aborted", retirement_site="ActiveSession"); s.add("trace_end")
            self.refuse(s.rows, "follow HTTP drop and precede receiver closure")

    def test_misbound_suppressed_and_disappearing_partial_history_refuse(self):
        for field, value in (("worker_generation", 2), ("worker_route", "other"), ("model", "other"),
                             ("pid", 18), ("suppressed_events", 1)):
            rows = partial()
            for row in rows:
                if row[field] is not None: row[field] = value
            with self.subTest(field=field): self.refuse(rows)
        rows = partial(); rows[-1]["quantum"]["cancelled_at_boundary"]["chunk"] = 2
        self.refuse(rows, "snapshot disagrees")
        rows = partial(); rows.insert(-1,copy.deepcopy(rows[-2])); self.refuse(rows)
        for scenario in ("cancel_queued", "cancel_decode"):
            with self.assertRaises(ServingGateError): validate_cancel_trace(wire(partial()), **options(scenario))


@unittest.skipUnless(os.environ.get("MEMRA_TYPED_TRACE_FIXTURES"), "requires exported actual-source Rust Trace cases")
class ActualSourceReplayTests(unittest.TestCase):
    def test_existing_prospective_collector_accepts_v2_start_and_preserves_identity(self):
        from serving_cancel_phase import _decode_prefix, _prospective_phase, _target_prefix

        raw = (Path(os.environ["MEMRA_TYPED_TRACE_FIXTURES"]) / "partial_pending.jsonl").read_bytes()
        decoded = decode_lifecycle_log(raw)
        start = next(entry for entry in decoded["records"] if entry["record"]["event"] == "prime_quantum_start")
        pid = start["record"]["pid"]
        required = {"scenario": "cancel_prime", "scope": {"model": "fixture"}}
        program = {"client_trace_key": KEY, "server_identity": {"pid": pid},
                   "requests": [{"role": "target", "path": "/v1/completions"}],
                   "trace": {"worker_generation": 1, "worker_route": "shared_gpu_worker",
                             "quantum_routes": ["plain-prime-call"]}}
        prefix = b"\n".join(raw.split(b"\n")[:start["line"]]) + b"\n"
        candidate = _prospective_phase(_target_prefix(_decode_prefix(prefix), required, program), "cancel_prime")
        self.assertEqual(candidate, start)
        complete = _target_prefix(decoded, required, program)
        self.assertEqual(complete[-1]["record"]["event"], "trace_end")
        self.assertEqual(complete[-1]["record"]["quantum"]["id"], candidate["record"]["quantum"]["id"])
        self.assertIs(complete[-1]["record"]["quantum"]["completed"], False)

    def test_real_rust_partial_success_and_adverse_traces(self):
        root = Path(os.environ["MEMRA_TYPED_TRACE_FIXTURES"])
        cases = {"partial_pending":True, "partial_body":True, "overflow":False,
                 "ordinary_error":False, "late_drop":False, "completed_return":True,
                 "unknown_remaining":True, "missing_retirement":False, "early_retirement":False}
        self.assertEqual({p.stem for p in root.glob("*.jsonl")}, set(cases))
        for case, accepted in cases.items():
            raw = (root/(case+".jsonl")).read_bytes()
            decoded = decode_lifecycle_log(raw)
            pid = decoded["records"][0]["record"]["pid"]
            args = {**options(), "expected_server_identity":{"pid":pid,"start_identity":"CPU-fixture-not-native-birth"},
                    "expected_model":"fixture", "expected_http_route":"/v1/completions",
                    "expected_worker_generation":1, "expected_quantum_routes":["plain-prime-call"]}
            with self.subTest(case=case):
                if not accepted:
                    with self.assertRaises(ServingGateError): validate_cancel_trace(raw, **args)
                    continue
                facts = validate_cancel_trace(raw, **args)
                q = facts["drop_spanning_quantum"]
                self.assertEqual(facts["target"]["request_id"], case)
                if case.startswith("partial"):
                    self.assertEqual(q["cancelled_at_boundary"], {"chunk":0,"rows_done":32,"rows_total":64})
                    self.assertIs(q["completed"], False)
                    self.assertIsNone(q["remaining_chunks"])
                    self.assertEqual(facts["retirement"]["phase"], "prime_cancelled")
                else:
                    self.assertIs(q["completed"], True)
                    self.assertIsNone(q["cancelled_at_boundary"])
                    self.assertEqual(q["remaining_chunks"], None if case=="unknown_remaining" else 0)


if __name__ == "__main__":
    unittest.main()
