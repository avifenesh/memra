"""Pure protocol fixtures/corruption controls. No socket, model, process or GPU proof."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from serving_release import ServingGateError
from serving_trace import decode_lifecycle_log, validate_cancel_trace


KEY = "0123456789abcdef0123456789abcdef"
OTHER_KEY = "fedcba9876543210fedcba9876543210"
OWNER = {"pid": 17, "start_identity": "outer-verified-boot:71"}


def wire(rows, *, prefix=b"[request-lifecycle] "):
    return b"".join(prefix + json.dumps(r, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"
                    for r in rows)


class Script:
    """Fixture snapshots specified by the script, not by the validator under test."""
    def __init__(self, trace_id=1, pid=17):
        self.rows = []
        self.state = dict(schema="memra-request-lifecycle-v1", pid=pid, trace_id=trace_id, seq=0,
            ordinary_event_limit=8192, clock="trace_elapsed_ns", at_ns=0, event="trace_start",
            client_trace_key=None, request_id=None, model=None, http_route="/v1/chat/completions",
            worker_generation=None, worker_route=None, phase="unbound", quantum_active=False,
            quantum=None, http_body="open", receiver_close_cause=None, observed_close_cause=None,
            retirement=None, retirement_site=None, bindings_complete=False, sequence_valid=True,
            first_error=None, suppressed_events=0, evidence_only=True)
        self.add("trace_start")

    def add(self, event, **snapshot):
        self.state.update(seq=self.state["seq"] + 1, at_ns=self.next_ns(),
                          event=event, observed_close_cause=None)
        self.state.update(copy.deepcopy(snapshot))
        self.rows.append(copy.deepcopy(self.state))

    def next_ns(self):
        return self.state["at_ns"] + 10

    def bind(self, key=KEY, request_id="minted-target"):
        self.add("client_trace_key_bound", client_trace_key=key)
        self.add("request_bound", request_id=request_id, model="gate")
        self.add("worker_bound", worker_generation=7, worker_route="shared_gpu_worker",
                 phase="bound", bindings_complete=True)
        self.add("queued", phase="queued")

    def quantum_start(self, number=1, route="target_prime", rows=16):
        self.add("prime_quantum_start", phase="prime", quantum_active=True,
                 quantum=dict(id=number, route=route, rows=rows, start_ns=self.next_ns(),
                              end_ns=None, completed=None, remaining_chunks=None))

    def quantum_end(self, remaining=0, completed=True):
        self.add("prime_quantum_end", quantum_active=False,
                 phase="prime_finished" if completed and remaining == 0 else "prime",
                 quantum={**self.state["quantum"], "end_ns": self.next_ns(),
                          "completed": completed, "remaining_chunks": remaining})


def fixture(scenario="cancel_prime", *, remaining=0, site=None, normal_eof=False,
            omit_quantum_end=False, before_decode_prime=False, failed=False):
    script = Script(); script.bind()
    if scenario == "cancel_prime":
        script.quantum_start()
    if scenario == "cancel_decode":
        if before_decode_prime:
            script.quantum_start(); script.quantum_end(remaining)
        script.add("decode_start", phase="decode")
    if normal_eof:
        script.add("http_body_eof", http_body="normal_eof")
        script.add("http_body_drop", http_body="dropped_after_eof")
    elif scenario == "cancel_queued":
        script.add("http_pending_drop", http_body="pending_dropped")
    else:
        script.add("http_body_drop", http_body="dropped_before_eof")
    if scenario == "cancel_prime" and not omit_quantum_end:
        script.quantum_end(remaining, completed=not failed)
    script.add("receiver_closed", receiver_close_cause="receiver_dropped", observed_close_cause="receiver_dropped")
    script.add("retired", retirement="aborted",
               retirement_site=site or ("WorkerQueue" if scenario == "cancel_queued" else "ActiveSession"))
    script.add("trace_end")
    return script.rows


def peer():
    script = Script(2); script.bind(OTHER_KEY, "minted-normal-peer")
    script.add("decode_start", phase="decode")
    script.add("http_body_eof", http_body="normal_eof")
    script.add("http_body_drop", http_body="dropped_after_eof")
    # Current production has no normal-completion retirement hook. This is the
    # actual unqualified end shape, not a synthetic successful-retirement claim.
    script.add("trace_end", sequence_valid=False, first_error="trace_ended_without_explicit_retirement")
    return script.rows


def options(scenario="cancel_prime"):
    return dict(scenario=scenario, client_trace_key=KEY, expected_server_identity=dict(OWNER),
                expected_model="gate", expected_http_route="/v1/chat/completions",
                expected_worker_route="shared_gpu_worker", expected_worker_generation=7,
                expected_quantum_routes=["target_prime", "draft_fill"])


def find(rows, name):
    return next(r for r in rows if r["event"] == name)


class TraceSyntaxTests(unittest.TestCase):
    def reject(self, data, message=None):
        with self.assertRaisesRegex(ServingGateError, message or ".") as error:
            decode_lifecycle_log(data)
        self.assertEqual(error.exception.code, "invalid_lifecycle")

    def test_all_records_and_normal_peer_errors_stay_in_syntax_census(self):
        rows = fixture() + peer(); data = b"server startup\n\n" + wire(rows)
        decoded = decode_lifecycle_log(data)
        self.assertEqual([r["record"] for r in decoded["records"]], rows)
        self.assertEqual(decoded["non_lifecycle_lines"], 2)
        self.assertEqual(sum(c["records"] for c in decoded["census"]), len(rows))
        normal = next(c for c in decoded["census"] if c["trace_id"] == 2)
        self.assertEqual(normal["reported_retirement_records"], 0)
        self.assertEqual(normal["reported_errors"], ["trace_ended_without_explicit_retirement"])
        self.assertEqual(normal["scope"], "syntax_and_census_only")

    def test_truncation_utf8_and_displaced_lifecycle_frames_refuse(self):
        data = wire(fixture())
        for bad in (data[:-1], data[:-20], data + b"partial", data + b"\xff\n", b"", b"only log text\n",
                    data.replace(b"[request-lifecycle] ", b" [request-lifecycle] ", 1),
                    data.replace(b"[request-lifecycle] ", b"", 1),
                    b"[request-lifecycle] \n" + data):
            self.reject(bad)

    def test_duplicate_json_members_and_nonfinite_numbers_are_not_coerced(self):
        data = wire(fixture())
        self.reject(data.replace(b'"seq":1,', b'"seq":1,"seq":1,', 1), "duplicate JSON member")
        self.reject(data.replace(b'"rows":16,', b'"rows":16,"rows":16,', 1), "duplicate JSON member")
        for number in (b"NaN", b"Infinity", b"-Infinity", b"1e9999", b"1.0", b"true"):
            self.reject(data.replace(b'"at_ns":10,', b'"at_ns":' + number + b",", 1))
        self.reject(data.replace(b'"at_ns":10,', b'"at_ns":' + b"9" * 5000 + b",", 1))

    def test_reserved_marker_anywhere_refuses_even_with_malformed_or_foreign_schema(self):
        for tail in (b'host-prefix [request-lifecycle] {"schema":\n',
                     b'host-prefix [request-lifecycle] {"schema":"broken-schema"}\n',
                     b'host-prefix [request-lifecycle malformed\n'):
            with self.subTest(tail=tail):
                self.reject(wire(fixture()) + tail, "framing")

    def test_escaped_schema_cannot_hide_bare_or_displaced_post_end_record(self):
        rows = fixture()
        extra = copy.deepcopy(rows[-1]); extra["seq"] += 1; extra["at_ns"] += 10
        for key in (b'"schema"', b'"\\u0073chema"'):
            for value in (b'"memra-request-lifecycle-v1"', b'"\\u006demra-request-lifecycle-v1"'):
                encoded = wire([extra]).replace(b'"schema"', key).replace(
                    b'"memra-request-lifecycle-v1"', value)
                for tail in (b"host-prefix " + encoded, encoded.removeprefix(b"[request-lifecycle] ")):
                    with self.subTest(key=key, value=value, tail=tail[:40]):
                        self.reject(wire(rows) + tail, "framing")
                # Correct framing exposes the duplicate to ordinary reconstruction.
                with self.assertRaisesRegex(ServingGateError, "record after trace_end"):
                    validate_cancel_trace(wire(rows) + encoded, **options())

    def test_reserved_bare_schema_survives_malformed_and_duplicate_members(self):
        for tail in (b'{"\\u0073chema":"\\u006demra-request-lifecycle-v1","later":\n',
                     b'{"schema":"other","\\u0073chema":"memra-request-lifecycle-v1"}\n'):
            self.reject(wire(fixture()) + tail, "framing")

    def test_ordinary_text_and_unrelated_json_remain_counted(self):
        ordinary = (b'ordinary host text with "quoted strings"\n'
                    b'{"schema":"unrelated","value":"memra-request-lifecycle-v1"}\n'
                    b'{"value":"schema","another":"memra-request-lifecycle-v1"}\n'
                    b'{ordinary non-JSON diagnostic}\n')
        rows = fixture()
        result = validate_cancel_trace(ordinary + wire(rows), **options())
        self.assertEqual(len(result["decoded_log"]["records"]), len(rows))
        self.assertEqual(result["decoded_log"]["non_lifecycle_lines"], 4)

    def test_unterminated_escaped_string_is_consumed_once(self):
        # Each quote after the first is escaped. Searching again from every such
        # quote made the first attempted framing fix take quadratic time.
        code = '''from serving_trace import decode_lifecycle_log
from test_serving_trace import fixture, wire
import json
counts = []
for suffix in (b"", bytes([92])):
    text = bytes([34]) + bytes([92, 34]) * 131072 + suffix
    counts.append(decode_lifecycle_log(text + bytes([10]) + wire(fixture()))["non_lifecycle_lines"])
print(json.dumps(counts))
'''
        run = subprocess.run([sys.executable] + (["-O"] if not __debug__ else []) + ["-c", code],
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent),
                 "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True, timeout=5)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), [1, 1])

    def test_schema_keys_enums_and_retirement_contract_are_exact(self):
        for name in fixture()[0]:
            rows = fixture(); del rows[0][name]; self.reject(wire(rows), "missing")
        changes = {"schema": "old", "ordinary_event_limit": 256, "event": "prime_start",
                   "clock": "monotonic_ns", "phase": "complete", "http_body": "cancelled",
                   "retirement": "completed", "retirement_site": "worker_queue",
                   "receiver_close_cause": "client_cancel", "first_error": "invented", "skip": True}
        for key, value in changes.items():
            rows = fixture(); rows[0][key] = value
            self.reject(wire(rows))

    def test_integer_boolean_nullable_and_utf8_byte_bounds_are_strict(self):
        for name in ("pid", "trace_id", "seq", "at_ns", "ordinary_event_limit", "suppressed_events"):
            for value in (True, -1, 2**64, 1.5, "1", None, []):
                rows = fixture(); rows[0][name] = value; self.reject(wire(rows))
        for name in ("quantum_active", "bindings_complete", "sequence_valid", "evidence_only"):
            rows = fixture(); rows[0][name] = 1; self.reject(wire(rows))
        for name in ("model", "request_id", "worker_route"):
            for value in ("", "é" * 129, "\ud800", [], 17):
                rows = fixture(); find(rows, "queued")[name] = value
                # ASCII escaping lets the decoder, rather than Python.encode, see lone surrogates.
                data = b"".join(b"[request-lifecycle] " + json.dumps(r).encode() + b"\n" for r in rows)
                self.reject(data)

    def test_quantum_snapshot_shape_and_numbers_are_bounded(self):
        for name in fixture()[5]["quantum"]:
            rows = fixture(); del find(rows, "prime_quantum_start")["quantum"][name]; self.reject(wire(rows))
        for name, value in (("rows", 0), ("id", False), ("start_ns", -1), ("rows", 2**64),
                            ("route", []), ("completed", True), ("remaining_chunks", 0)):
            rows = fixture(); find(rows, "prime_quantum_start")["quantum"][name] = value; self.reject(wire(rows))
        for name, value in (("completed", None), ("end_ns", False), ("remaining_chunks", True), ("end_ns", 1)):
            rows = fixture(); find(rows, "prime_quantum_end")["quantum"][name] = value; self.reject(wire(rows))

    def test_unrelated_malformed_record_is_not_discarded_after_target_selection(self):
        rows = fixture() + peer(); rows[-1]["worker_generation"] = True
        with self.assertRaises(ServingGateError): validate_cancel_trace(wire(rows), **options())
        rows = fixture() + peer(); rows[-1]["ignored_extra"] = "bad"
        with self.assertRaises(ServingGateError): validate_cancel_trace(wire(rows), **options())

    def test_line_trace_and_emitted_record_bounds_refuse(self):
        self.reject(b"[request-lifecycle] " + b" " * 16384 + b"{}\n", "record byte bound")
        rows = [Script(i + 1).rows[0] for i in range(1025)]
        self.reject(wire(rows), "trace census")
        self.reject(wire([fixture()[0]] * 8195), "emitted-record bound")


class TraceTargetTests(unittest.TestCase):
    def evaluate(self, rows, scenario="cancel_prime", **overrides):
        args = {**options(scenario), **overrides}
        return validate_cancel_trace(wire(rows), **args)

    def reject(self, rows, message=None, scenario="cancel_prime", **overrides):
        with self.assertRaisesRegex(ServingGateError, message or ".") as error:
            self.evaluate(rows, scenario, **overrides)
        self.assertEqual(error.exception.code, "invalid_lifecycle")

    def test_three_scoped_positives_return_facts_without_qualification(self):
        for scenario in ("cancel_queued", "cancel_prime", "cancel_decode"):
            rows = fixture(scenario); original = copy.deepcopy(rows)
            result = self.evaluate(rows, scenario)
            self.assertEqual(rows, original)
            self.assertEqual(result["target"]["request_id"], "minted-target")
            self.assertEqual(result["server_identity"], OWNER)
            self.assertEqual(result["scope"], "target_lifecycle_facts_only")
            self.assertNotIn("qualified", result); self.assertNotIn("qualification", result)
            self.assertLess(result["http_drop"]["seq"], result["retirement"]["seq"])

    def test_valid_target_survives_interleaved_unqualified_peer_without_erasing_it(self):
        a, b = fixture(), peer(); rows = []
        for i in range(max(len(a), len(b))):
            if i < len(a): rows.append(a[i])
            if i < len(b): rows.append(b[i])
        result = self.evaluate(rows)
        self.assertEqual(len(result["decoded_log"]["records"]), len(a) + len(b))
        normal = result["decoded_log"]["census"][1]
        self.assertEqual(normal["reported_errors"], ["trace_ended_without_explicit_retirement"])
        self.assertEqual(normal["reported_retirement_records"], 0)

    def test_missing_truncated_reordered_duplicate_and_post_end_target_records_refuse(self):
        rows = fixture(); self.reject(rows[1:], "sequence|trace_start")
        rows = fixture(); self.reject(rows[:-1], "final trace_end")
        rows = fixture(); rows[5], rows[6] = rows[6], rows[5]; self.reject(rows, "sequence")
        rows = fixture(); rows.insert(5, copy.deepcopy(rows[5])); self.reject(rows, "sequence")
        rows = fixture(); rows[-1]["seq"] += 1; self.reject(rows, "sequence")
        rows = fixture(); tail = copy.deepcopy(rows[-1]); tail.update(seq=tail["seq"] + 1, at_ns=tail["at_ns"] + 1)
        self.reject(rows + [tail], "after trace_end")

    def test_monotonic_order_accepts_equal_clock_ticks_but_not_clock_reversal(self):
        rows = fixture("cancel_queued"); rows[2]["at_ns"] = rows[1]["at_ns"]
        self.evaluate(rows, "cancel_queued")
        rows[2]["at_ns"] -= 1; self.reject(rows, "non-monotonic", "cancel_queued")
        rows = fixture("cancel_queued")
        for r in rows: r["at_ns"] = 2**64 - 1
        self.reject(rows, "clock overflow", "cancel_queued")

    def test_client_key_is_unique_across_all_traces_even_foreign_or_unqualified(self):
        self.reject(fixture(), "absent", client_trace_key=OTHER_KEY)
        for foreign_pid in (17, 88):
            other = peer()
            for r in other:
                r["pid"] = foreign_pid
                if r["client_trace_key"] is not None: r["client_trace_key"] = KEY
            self.reject(fixture() + other, "ambiguous")
        for key in (KEY.upper(), KEY[:-1], "g" * 32, " " + KEY, None):
            self.reject(fixture(), "invalid client", client_trace_key=key)

    def test_pid_minted_identity_model_http_route_and_generation_cannot_drift(self):
        for name, value in (("pid", 18), ("request_id", "rebound"), ("model", "other"),
                            ("http_route", "/v1/completions"), ("worker_route", "different"),
                            ("worker_generation", 8), ("client_trace_key", OTHER_KEY)):
            rows = fixture(); find(rows, "queued")[name] = value
            self.reject(rows)
        rows = fixture()
        for r in rows: r["pid"] = 18
        self.reject(rows, "PID differs")
        for name, value in (("expected_model", "other"), ("expected_http_route", "/v1/completions"),
                            ("expected_worker_route", "foreign"), ("expected_worker_generation", 8)):
            self.reject(fixture(), **{name: value})

    def test_minted_id_alias_in_unrelated_owned_trace_refuses(self):
        other = peer()
        for r in other:
            if r["request_id"] is not None: r["request_id"] = "minted-target"
        self.reject(fixture() + other, "minted request ID is ambiguous")

    def test_fake_valid_flag_and_phase_snapshot_cannot_create_evidence(self):
        rows = fixture(); find(rows, "http_body_drop")["phase"] = "decode"
        self.reject(rows, "phase snapshot")
        rows = fixture(); find(rows, "queued")["bindings_complete"] = False
        self.reject(rows, "bindings_complete snapshot")
        rows = fixture(); find(rows, "retired")["retirement_site"] = None
        rows[-1]["retirement_site"] = None
        self.reject(rows, "explicit retirement site")

    def test_stale_queued_phase_does_not_supply_workerqueue_or_active_phase_proof(self):
        rows = fixture("cancel_queued", site="ActiveSession")
        self.reject(rows, "retirement site", "cancel_queued")
        self.reject(rows, "spanning actual quantum", "cancel_prime")
        self.reject(rows, "actual decode", "cancel_decode")
        rows = fixture("cancel_prime", site="WorkerQueue")
        self.reject(rows, "retirement site")

    def test_missing_or_relabelled_site_never_becomes_queue_retirement(self):
        rows = fixture("cancel_queued")
        find(rows, "retired")["retirement_site"] = None; rows[-1]["retirement_site"] = None
        self.reject(rows, "retirement site", "cancel_queued")
        rows[-1]["retirement_site"] = "WorkerQueue"
        self.reject(rows, "retirement_site snapshot", "cancel_queued")

    def test_normal_eof_and_mixed_pending_body_drop_are_not_cancellation(self):
        self.reject(fixture(normal_eof=True), "without normal EOF")
        rows = fixture("cancel_queued")
        r = copy.deepcopy(find(rows, "http_pending_drop")); r.update(event="http_body_drop", http_body="dropped_before_eof")
        index = rows.index(find(rows, "http_pending_drop")) + 1
        rows.insert(index, r)
        for i, r in enumerate(rows, 1): r.update(seq=i, at_ns=i * 10)
        self.reject(rows, "mixed pending/body", "cancel_queued")

    def test_receiver_overflow_and_close_before_http_drop_refuse(self):
        rows = fixture("cancel_queued")
        for r in rows:
            if r["receiver_close_cause"]: r["receiver_close_cause"] = "event_queue_overflow"
            if r["observed_close_cause"]: r["observed_close_cause"] = "event_queue_overflow"
        self.reject(rows, "overflow", "cancel_queued")
        script = Script(); script.bind()
        script.add("receiver_closed", receiver_close_cause="receiver_dropped", observed_close_cause="receiver_dropped")
        script.add("http_pending_drop", http_body="pending_dropped")
        script.add("retired", retirement="aborted", retirement_site="WorkerQueue"); script.add("trace_end")
        self.reject(script.rows, "out of order", "cancel_queued")

    def test_false_quantum_return_plus_abort_cannot_hide_behind_forged_validity(self):
        self.reject(fixture(failed=True), "failed or missing quantum return")
        rows = fixture(failed=True)
        for r in rows[7:]: r.update(sequence_valid=False, first_error="failed_prime_quantum")
        self.reject(rows, "invalid/suppressed")

    def test_unknown_remaining_is_retained_without_becoming_completion(self):
        result = self.evaluate(fixture(remaining=None))
        self.assertIsNone(result["drop_spanning_quantum"]["remaining_chunks"])
        self.assertEqual(result["retirement"]["phase"], "prime")
        rows = fixture(remaining=None)
        for r in rows:
            if r["event"] in ("prime_quantum_end", "receiver_closed", "retired", "trace_end"): r["phase"] = "prime_finished"
        self.reject(rows, "phase snapshot")
        self.reject(fixture("cancel_decode", before_decode_prime=True, remaining=None),
                    "unknown is not zero", "cancel_decode")

    def test_decode_after_real_completed_prime_preserves_quantum_history(self):
        result = self.evaluate(fixture("cancel_decode", before_decode_prime=True), "cancel_decode")
        self.assertEqual(len(result["quanta"]), 1)
        self.assertEqual(result["quanta"][0]["remaining_chunks"], 0)
        self.assertLess(result["decode_start"]["seq"], result["http_drop"]["seq"])

    def test_target_and_draft_quantums_keep_individual_identity_and_return_snapshots(self):
        script = Script(); script.bind()
        script.quantum_start(1, "target_prime", 1024); script.quantum_end(remaining=2)
        script.quantum_start(2, "draft_fill", 512); script.quantum_end(remaining=1)
        script.quantum_start(3, "target_prime", 16)
        script.add("http_body_drop", http_body="dropped_before_eof")
        script.quantum_end(remaining=None)
        script.add("receiver_closed", receiver_close_cause="receiver_dropped", observed_close_cause="receiver_dropped")
        script.add("retired", retirement="aborted", retirement_site="ActiveSession"); script.add("trace_end")
        result = self.evaluate(script.rows)
        self.assertEqual([q["id"] for q in result["quanta"]], [1, 2, 3])
        self.assertEqual([q["route"] for q in result["quanta"]], ["target_prime", "draft_fill", "target_prime"])
        self.assertEqual(result["drop_spanning_quantum"]["id"], 3)
        self.assertIsNone(result["drop_spanning_quantum"]["remaining_chunks"])
        bad = copy.deepcopy(script.rows)
        find(bad, "http_body_drop")["quantum"] = copy.deepcopy(bad[6]["quantum"])
        find(bad, "http_body_drop")["quantum_active"] = False
        self.reject(bad, "quantum_active snapshot")

    def test_second_retirement_cannot_upgrade_an_unknown_site(self):
        rows = fixture("cancel_queued")
        find(rows, "retired")["retirement_site"] = None
        second = copy.deepcopy(find(rows, "retired")); second["retirement_site"] = "WorkerQueue"
        rows.insert(-1, second)
        for i, r in enumerate(rows, 1): r.update(seq=i, at_ns=i * 10)
        self.reject(rows, "after retirement", "cancel_queued")

    def test_missing_return_or_retirement_before_actual_return_refuses(self):
        self.reject(fixture(omit_quantum_end=True), "before actual return")
        rows = fixture(); rows.pop(-2)
        rows[-1]["seq"] -= 1
        self.reject(rows, "incomplete trace end")

    def test_quantum_ids_actual_rows_routes_and_snapshots_cannot_be_substituted(self):
        for event_name, field, value in (("prime_quantum_start", "id", 2),
                                       ("prime_quantum_end", "rows", 17),
                                       ("prime_quantum_end", "route", "draft_fill"),
                                       ("prime_quantum_end", "start_ns", 1),
                                       ("http_body_drop", "id", 8)):
            rows = fixture(); find(rows, event_name)["quantum"][field] = value; self.reject(rows)
        self.reject(fixture(), "route differs", expected_quantum_routes=["foreign"])
        self.reject(fixture(), "trusted actual quantum route", expected_quantum_routes=[])

    def test_cancellation_after_last_return_is_not_a_spanning_prime_quantum(self):
        script = Script(); script.bind(); script.quantum_start(); script.quantum_end(remaining=1)
        script.add("http_body_drop", http_body="dropped_before_eof")
        script.add("receiver_closed", receiver_close_cause="receiver_dropped", observed_close_cause="receiver_dropped")
        script.add("retired", retirement="aborted", retirement_site="ActiveSession"); script.add("trace_end")
        self.reject(script.rows, "spanning actual quantum")

    def test_overflow_requeue_and_suppression_are_never_waived_by_summaries(self):
        rows = fixture(); rows[-1].update(sequence_valid=False, first_error="lifecycle_event_limit", suppressed_events=1)
        self.reject(rows, "invalid/suppressed")
        for event_name in ("overflow", "requeued"):
            rows = fixture(); rows[5]["event"] = event_name
            self.reject(rows, "overflow/requeue")
        rows = fixture(); rows[-1]["passed"] = True
        self.reject(rows, "unknown or missing")

    def test_duplicate_binding_transition_is_not_authorized_by_immutable_values(self):
        rows = fixture("cancel_queued")
        r = copy.deepcopy(rows[2]); rows.insert(3, r)
        for i, r in enumerate(rows, 1): r.update(seq=i, at_ns=i * 10)
        self.reject(rows, "duplicate lifecycle transition", "cancel_queued")

    def test_trusted_expected_argument_types_and_unknown_scenarios_refuse(self):
        for key, value in (("scenario", "skip"), ("expected_worker_generation", True),
                           ("expected_server_identity", {"pid": True, "start_identity": "boot"}),
                           ("expected_server_identity", {"pid": 17, "start_identity": ""}),
                           ("expected_quantum_routes", "target_prime"),
                           ("expected_quantum_routes", ["target_prime", "target_prime"]),
                           ("expected_worker_route", []), ("expected_model", "")):
            args = options(); args[key] = value
            with self.assertRaises(ServingGateError): validate_cancel_trace(wire(fixture()), **args)


if __name__ == "__main__":
    unittest.main()
