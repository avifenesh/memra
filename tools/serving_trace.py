"""Pure C4 lifecycle-log decoding and target cancellation facts, not qualification.

decode_lifecycle_log(raw) reads exact bytes of the owned server's mixed stderr.
Lifecycle records MUST be complete ``[request-lifecycle] {JSON}\n`` lines. Other
lines are counted, not interpreted; displaced prefixes/bare lifecycle JSON refuse.
Every lifecycle record, including unrelated/unqualified peers, gets strict schema,
type and bounds checks. The returned records retain log order and line numbers.
The census reports producer claims; it never promotes a normal peer's missing
retirement into success. Target transition validation is separate below.

validate_cancel_trace(raw, *, scenario, client_trace_key,
    expected_server_identity, expected_model, expected_http_route,
    expected_worker_route, expected_worker_generation, expected_quantum_routes)

All expected values come from the independently trusted outer capture/program:
server identity is {"pid": int, "start_identity": "<boot-id>:<start-ticks>"}; routes
are explicit strings, and quantum_routes is a unique list (empty only when no
quantum route is expected). No records may select/relax their own expected scope.
The supplied 32-lowercase-hex key must join exactly one (pid, trace_id) across the
ENTIRE log, even if another occurrence is in an unqualified/foreign trace. The
minted request ID is then learned from that trace's single immutable request_bound
event, not equated with the client key. The outer layer can also bind it to wire IDs.

The target is reconstructed from transitions, not phase/sequence_valid claims.
Queue facts require explicit WorkerQueue retirement and a real queue observation;
stale queued phase and missing quantum/header records cannot supply the site.
Prime facts narrowly require a nonzero observed quantum spanning HTTP drop, its
successful return or explicit v2 typed partial-cancellation return before retirement,
and ActiveSession. Between-quantum cancellation does not satisfy this contract.
Unknown remaining work stays unknown, never prime completion. Decode requires real
decode_start before drop and ActiveSession. Ordinary failed quanta always refuse.

The producer emits v2. Historical v1 keeps its exact schema and success-only return
contract; a trace cannot change version partway through. v2 adds a nullable quantum
cancelled_at_boundary={chunk,rows_done,rows_total} and prime_quantum_cancelled event.
The exact typed return keeps completed:false and remaining_chunks:null, transitions
to prime_cancelled, and must span the HTTP drop before receiver closure/retirement.
It permits no further work or normal EOF, never whole-prime success. Its boundary
rows/counts describe the selected call, not total request work or GPU quiescence.
Drop must precede receiver_dropped and aborted retirement; EOF, overflow, requeue,
missing/inconsistent snapshots, suppression and incomplete target history refuse.

Timestamps are trace_elapsed_ns, not collector CLOCK_MONOTONIC; sequence order
breaks equal-clock ties. Returned host-return/site facts do NOT prove GPU quiescence,
no earlier work or cache hits. No native/whole-scenario pass is returned. Full peer,
wire, recovery, exact source/binary, controller/process birth, lease and manifest
checks remain outer-owned. A matching PID here does not independently prove boot
identity or authentic bytes. The computed log digest is a binding aid, not authority.

Fixed parser resource ceilings: 64 MiB log, 16 KiB per lifecycle line, 131072 total
records, 1024 traces, 8194 emitted records per trace. These bounds cannot be waived
by inputs. The producer ordinary_event_limit must be exactly 8192. No I/O occurs.
"""

import hashlib
import json
import re

from serving_completion import _same
from serving_release import ServingGateError, json_object, require


_PREFIX = b"[request-lifecycle] "
_SCHEMA_V1 = "memra-request-lifecycle-v1"
_SCHEMA = "memra-request-lifecycle-v2"
_SCHEMAS = frozenset((_SCHEMA_V1, _SCHEMA))
_KEY = re.compile(r"[0-9a-f]{32}\Z")
# Consume unterminated strings too, so escaped quotes are never retried as
# opening quotes and a malformed ordinary line cannot cause quadratic scans.
_STRING_TOKEN = re.compile(rb'"(?:[^"\\]|\\.)*(?:"|\\?\Z)')
_U64 = 2**64 - 1
_LIMIT = 8192
_MAX_LOG = 64 * 1024 * 1024
_MAX_LINE = 16 * 1024
_MAX_RECORDS = 131072
_MAX_TRACES = 1024
_FIELDS = frozenset("schema pid trace_id seq ordinary_event_limit clock at_ns event client_trace_key "
                    "request_id model http_route worker_generation worker_route phase quantum_active quantum "
                    "http_body receiver_close_cause observed_close_cause retirement retirement_site "
                    "bindings_complete sequence_valid first_error suppressed_events evidence_only".split())
_QFIELDS = frozenset("id route rows start_ns end_ns completed remaining_chunks".split())
_EVENTS_V1 = frozenset("trace_start request_bound client_trace_key_bound worker_bound queued decode_start "
                    "prime_quantum_start prime_quantum_end http_body_eof http_body_drop http_pending_drop "
                    "receiver_closed requeued retired overflow trace_end".split())
_EVENTS = _EVENTS_V1 | {"prime_quantum_cancelled"}
_WORKER_EVENTS = frozenset("queued decode_start prime_quantum_start prime_quantum_end prime_quantum_cancelled receiver_closed "
                           "requeued retired".split())
_PHASES_V1 = frozenset("unbound bound queued prime prime_finished decode".split())
_PHASES = _PHASES_V1 | {"prime_cancelled"}
_HTTP = frozenset("open normal_eof dropped_before_eof dropped_after_eof pending_dropped".split())
_CLOSE = frozenset(("receiver_dropped", "event_queue_overflow"))
_SITES = frozenset(("WorkerQueue", "ActiveSession"))
_ROUTES = frozenset(("/v1/completions", "/v1/chat/completions"))
_ERRORS = frozenset("abort_retirement_without_close_observation body_drop_after_pending_drop "
    "decode_before_prime_or_quantum_end duplicate_client_trace_key_binding duplicate_http_body_drop "
    "duplicate_or_late_http_eof duplicate_or_out_of_order_http_pending_drop duplicate_or_out_of_order_queue "
    "duplicate_receiver_close duplicate_request_binding duplicate_retirement duplicate_worker_binding "
    "failed_prime_quantum http_drop_without_request_binding http_eof_without_request_binding "
    "http_pending_drop_without_request_binding invalid_client_trace_key invalid_or_oversized_request_binding "
    "invalid_quantum_route_or_rows invalid_worker_route late_client_trace_key_binding lifecycle_clock_overflow "
    "lifecycle_event_limit missing_request_or_worker_binding overlapping_prime_quantums poisoned_lifecycle "
    "quantum_after_prime_finished_decode_or_receiver_close quantum_end_without_start request_requeued "
    "retirement_before_quantum_return trace_ended_with_open_quantum trace_ended_without_binding "
    "trace_ended_without_explicit_retirement trace_ended_without_http_lifetime_end worker_bound_before_request "
    "worker_event_after_retirement".split())


def _need(condition, message):
    require(condition, message, "invalid_lifecycle")


def _keys(value, fields, label):
    _need(type(value) is dict and value.keys() == fields, f"unknown or missing {label} fields")


def _integer(value, label, minimum=0, maximum=_U64):
    _need(type(value) is int and minimum <= value <= maximum, f"invalid {label} integer/bound")


def _text(value, label, maximum=256, nullable=False):
    if nullable and value is None:
        return
    _need(type(value) is str and bool(value), f"invalid {label} string")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError as error:
        raise ServingGateError(f"invalid {label} Unicode", "invalid_lifecycle") from error
    _need(size <= maximum, f"{label} exceeds byte bound")


def _enum(value, choices, label, nullable=False):
    _need((nullable and value is None) or (type(value) is str and value in choices),
          f"unknown {label}")


def _client_key(value):
    _need(type(value) is str and _KEY.fullmatch(value) is not None, "invalid client trace key")


def _schema(row):
    _keys(row, _FIELDS, "lifecycle record")
    _enum(row["schema"], _SCHEMAS, "lifecycle schema")
    _need(row["clock"] == "trace_elapsed_ns", "unknown lifecycle schema/clock")
    legacy = row["schema"] == _SCHEMA_V1
    _integer(row["ordinary_event_limit"], "ordinary event limit")
    _need(row["ordinary_event_limit"] == _LIMIT, "ordinary event limit is not 8192")
    _integer(row["pid"], "PID", 2, 2**32 - 1)
    _integer(row["trace_id"], "trace ID", 1)
    _integer(row["seq"], "sequence", 1)
    _integer(row["at_ns"], "elapsed timestamp")
    _integer(row["suppressed_events"], "suppression count")
    for name in ("quantum_active", "bindings_complete", "sequence_valid", "evidence_only"):
        _need(type(row[name]) is bool, f"{name} must be a bool")
    _need(row["evidence_only"] is True, "lifecycle is not evidence-only")
    _enum(row["event"], _EVENTS_V1 if legacy else _EVENTS, "event")
    _enum(row["phase"], _PHASES_V1 if legacy else _PHASES, "phase")
    _enum(row["http_body"], _HTTP, "HTTP lifetime")
    _enum(row["http_route"], _ROUTES, "HTTP route")
    _enum(row["retirement"], {"aborted"}, "retirement", nullable=True)
    _enum(row["retirement_site"], _SITES, "retirement site", nullable=True)
    for name in ("receiver_close_cause", "observed_close_cause"):
        _enum(row[name], _CLOSE, "receiver close cause", nullable=True)
    _enum(row["first_error"], _ERRORS if legacy else _ERRORS | {"invalid_prime_cancellation_boundary"},
          "producer error", nullable=True)
    _need(row["sequence_valid"] == (row["first_error"] is None), "producer error/validity disagree")
    for name in ("request_id", "model", "worker_route"):
        _text(row[name], name, nullable=True)
    _need((row["request_id"] is None) == (row["model"] is None), "partial request binding")
    if row["worker_generation"] is not None:
        _integer(row["worker_generation"], "worker generation")
    _need((row["worker_generation"] is None) == (row["worker_route"] is None), "partial worker binding")
    if row["client_trace_key"] is not None:
        _client_key(row["client_trace_key"])
    _need(row["retirement"] is not None or row["retirement_site"] is None, "site without retirement")
    q = row["quantum"]
    if q is None:
        _need(not row["quantum_active"], "active quantum snapshot missing")
        return
    _keys(q, _QFIELDS if legacy else _QFIELDS | {"cancelled_at_boundary"}, "quantum")
    for name in ("id", "rows"):
        _integer(q[name], "quantum " + name, 1)
    _text(q["route"], "quantum route")
    _integer(q["start_ns"], "quantum start")
    _need(q["start_ns"] <= row["at_ns"], "quantum starts after observation")
    if q["remaining_chunks"] is not None:
        _integer(q["remaining_chunks"], "remaining chunks")
    if q["end_ns"] is None:
        _need(row["quantum_active"] and q["completed"] is None and q["remaining_chunks"] is None,
              "open quantum has terminal fields or is inactive")
    else:
        _integer(q["end_ns"], "quantum end")
        _need(not row["quantum_active"] and type(q["completed"]) is bool
              and q["start_ns"] <= q["end_ns"] <= row["at_ns"], "invalid closed quantum snapshot")

    if not legacy:
        cancelled = q["cancelled_at_boundary"]
        if cancelled is not None:
            _keys(cancelled, {"chunk", "rows_done", "rows_total"}, "partial-cancellation boundary")
            _integer(cancelled["chunk"], "cancelled chunk")
            _integer(cancelled["rows_done"], "cancelled rows done", 1)
            _integer(cancelled["rows_total"], "cancelled rows total", 1)
            _need(cancelled["chunk"] < cancelled["rows_done"] < cancelled["rows_total"] == q["rows"]
                  and q["end_ns"] is not None and q["completed"] is False
                  and q["remaining_chunks"] is None,
                  "invalid typed partial-cancellation identity or completion")



def _bare_lifecycle_schema(line):
    # This only discovers reserved content to REFUSE, including escaped keys and
    # values in otherwise malformed JSON. It never decodes an accepted record.
    # Full-object parsing here could discard the schema when a later member is
    # malformed or duplicated. Ordinary text and unrelated JSON stay permissible.
    previous_schema, previous_end = False, 0
    for token in _STRING_TOKEN.finditer(line):
        try:
            value = json.loads(token[0])
        except (ValueError, UnicodeError):
            value = None
        if (previous_schema and line[previous_end:token.start()].strip() == b":"
                and isinstance(value, str) and re.fullmatch(r"memra-request-lifecycle-v[0-9]+", value)):
            return True
        previous_schema, previous_end = value == "schema", token.end()
    return False


def decode_lifecycle_log(raw):
    """Decode ALL lifecycle rows and a syntax-only census; no target is approved."""
    _need(type(raw) is bytes and 0 < len(raw) <= _MAX_LOG, "lifecycle log byte bound/type")
    _need(raw.endswith(b"\n"), "truncated log: missing final newline")
    try:
        raw.decode("utf-8")
    except UnicodeError as error:
        raise ServingGateError("invalid log UTF-8", "invalid_lifecycle") from error
    records, traces, other = [], {}, 0
    # Split only on LF: Unicode separators inside JSON strings are not log lines.
    for number, line in enumerate(raw.split(b"\n")[:-1], 1):
        if not line.startswith(_PREFIX):
            _need(b"[request-lifecycle" not in line and not _bare_lifecycle_schema(line),
                  f"lifecycle line {number}: displaced/malformed lifecycle framing")
            other += 1
            continue
        _need(len(line) <= _MAX_LINE, f"lifecycle line {number}: record byte bound")
        try:
            row = json_object(line[len(_PREFIX):])
            _schema(row)
        except ServingGateError as error:
            raise ServingGateError(f"lifecycle line {number}: {error}", "invalid_lifecycle") from error
        key = row["pid"], row["trace_id"]
        history = traces.setdefault(key, [])
        _need(not history or history[0]["schema"] == row["schema"], "trace changed lifecycle schema")
        history.append(row)
        _need(len(traces) <= _MAX_TRACES, "trace census exceeds bound")
        _need(len(traces[key]) <= _LIMIT + 2, "trace exceeds emitted-record bound")
        records.append({"line": number, "record": row})
        _need(len(records) <= _MAX_RECORDS, "lifecycle record census exceeds bound")
    _need(records, "no lifecycle records")
    census = []
    for (pid, trace_id), rows in traces.items():
        census.append({"pid": pid, "trace_id": trace_id, "records": len(rows),
            "trace_start_records": sum(r["event"] == "trace_start" for r in rows),
            "trace_end_records": sum(r["event"] == "trace_end" for r in rows),
            "client_trace_keys": sorted({r["client_trace_key"] for r in rows if r["client_trace_key"] is not None}),
            "reported_errors": sorted({r["first_error"] for r in rows if r["first_error"] is not None}),
            "reported_suppressed_events": max(r["suppressed_events"] for r in rows),
            "reported_retirement_records": sum(r["event"] == "retired" for r in rows),
            "scope": "syntax_and_census_only"})
    return {"log_sha256": hashlib.sha256(raw).hexdigest(), "records": records,
            "census": census, "non_lifecycle_lines": other}


def _target_history(entries, expected, client_key):
    state = dict(client_trace_key=None, request_id=None, model=None, worker_generation=None,
                 worker_route=None, phase="unbound", quantum_active=False, quantum=None,
                 http_body="open", receiver_close_cause=None, retirement=None, retirement_site=None,
                 bindings_complete=False)
    observations, quanta = {}, []
    previous_ns = -1
    for index, entry in enumerate(entries, 1):
        row, line = entry["record"], entry["line"]
        context = f"target line {line} ({row['event']})"

        def need(ok, message):
            _need(ok, context + ": " + message)

        event = row["event"]
        need(row["seq"] == index, "missing, duplicate or reordered sequence")
        need(row["at_ns"] >= previous_ns, "non-monotonic trace clock")
        need(row["at_ns"] < _U64, "producer clock overflow cannot form valid evidence")
        previous_ns = row["at_ns"]
        need(row["sequence_valid"] and row["first_error"] is None and row["suppressed_events"] == 0,
             "invalid/suppressed producer history: " + str(row["first_error"]))
        need(event not in ("overflow", "requeued"), "overflow/requeue cannot supply cancellation evidence")
        need(index <= _LIMIT or (index == _LIMIT + 1 and event == "trace_end"), "ordinary event limit exceeded")
        need(row["http_route"] == expected["http_route"], "foreign HTTP route")
        need("trace_end" not in observations, "record after trace_end")
        if event != "trace_start":
            need("trace_start" in observations, "missing trace_start")
        if state["phase"] == "prime_cancelled":
            need(event in ("receiver_closed", "retired", "trace_end"),
                 "work or HTTP transition after typed partial cancellation")
        if event in _WORKER_EVENTS:
            need(state["bindings_complete"], "worker observation before complete binding")
            need(state["retirement"] is None, "worker observation after retirement")
        if event not in ("prime_quantum_start", "prime_quantum_end"):
            need(event not in observations, "duplicate lifecycle transition")

        if event == "trace_start":
            need(index == 1, "late trace_start")
        elif event == "client_trace_key_bound":
            need(state["client_trace_key"] is None and state["http_body"] == "open"
                 and state["retirement"] is None, "late/duplicate client correlation binding")
            need(row["client_trace_key"] == client_key, "foreign client correlation binding")
            state["client_trace_key"] = client_key
        elif event == "request_bound":
            need(state["request_id"] is None and row["request_id"] is not None, "missing/duplicate minted ID")
            need(row["model"] == expected["model"], "foreign model binding")
            state.update(request_id=row["request_id"], model=row["model"])
        elif event == "worker_bound":
            need(state["request_id"] is not None and state["worker_generation"] is None,
                 "worker binding is missing, repeated or precedes request binding")
            need(row["worker_generation"] == expected["worker_generation"]
                 and row["worker_route"] == expected["worker_route"], "foreign worker generation/dispatch route")
            state.update(worker_generation=row["worker_generation"], worker_route=row["worker_route"], phase="bound")
        elif event == "queued":
            need(state["phase"] == "bound", "out-of-order queue observation")
            state["phase"] = "queued"
        elif event == "prime_quantum_start":
            need(not state["quantum_active"] and state["phase"] in ("bound", "queued", "prime")
                 and state["receiver_close_cause"] is None, "overlap or quantum after completion/decode/close")
            q = row["quantum"]
            need(q is not None and row["quantum_active"] and q["end_ns"] is None
                 and q.get("cancelled_at_boundary") is None,
                 "start lacks an actual open quantum")
            need(q["id"] == len(quanta) + 1 and q["start_ns"] == row["at_ns"]
                 and q["route"] in expected["quantum_routes"], "quantum ID, start or route differs")
            state.update(phase="prime", quantum_active=True, quantum=dict(q))
            quanta.append({"start_seq": index, "end_seq": None, **q})
        elif event == "prime_quantum_end":
            need(state["quantum_active"], "quantum return without matching start")
            q = row["quantum"]
            need(q is not None and q["completed"] is True, "failed or missing quantum return")
            closed = {**state["quantum"], "end_ns": row["at_ns"], "completed": True,
                      "remaining_chunks": q["remaining_chunks"]}
            need(_same(q, closed), "quantum identity/rows/return snapshot changed")
            state.update(quantum_active=False, quantum=closed)
            if q["remaining_chunks"] == 0:
                state["phase"] = "prime_finished"
            quanta[-1].update(end_seq=index, **closed)
        elif event == "prime_quantum_cancelled":
            need(row["schema"] == _SCHEMA and state["quantum_active"],
                 "typed cancellation without an actual v2 open quantum")
            q = row["quantum"]
            need(q is not None and q["cancelled_at_boundary"] is not None,
                 "typed cancellation lacks boundary identity")
            closed = {**state["quantum"], "end_ns": row["at_ns"], "completed": False,
                      "remaining_chunks": None, "cancelled_at_boundary": q["cancelled_at_boundary"]}
            need(_same(q, closed), "cancelled quantum identity/rows/return snapshot changed")
            need(state["http_body"] in ("pending_dropped", "dropped_before_eof")
                 and state["receiver_close_cause"] is None,
                 "typed partial cancellation must follow HTTP drop and precede receiver closure")
            state.update(phase="prime_cancelled", quantum_active=False, quantum=closed)
            quanta[-1].update(end_seq=index, **closed)
        elif event == "decode_start":
            need(not state["quantum_active"] and state["phase"] in ("bound", "queued", "prime_finished"),
                 "decode before observed prime completion (unknown is not zero)")
            state["phase"] = "decode"
        elif event in ("http_body_eof", "http_body_drop", "http_pending_drop"):
            need(state["request_id"] is not None, "HTTP lifetime without minted request binding")
            current = state["http_body"]
            if event == "http_body_drop":
                need(current in ("open", "normal_eof"), "duplicate or mixed pending/body drop")
                state["http_body"] = "dropped_after_eof" if current == "normal_eof" else "dropped_before_eof"
            else:
                need(current == "open", "duplicate or out-of-order EOF/pending drop")
                state["http_body"] = "normal_eof" if event == "http_body_eof" else "pending_dropped"
        elif event == "receiver_closed":
            need(state["receiver_close_cause"] is None, "duplicate receiver closure")
            need(row["observed_close_cause"] == "receiver_dropped", "queue overflow is not receiver drop")
            state["receiver_close_cause"] = "receiver_dropped"
        elif event == "retired":
            need(not state["quantum_active"] and state["receiver_close_cause"] == "receiver_dropped",
                 "retirement before actual return/receiver closure")
            need(row["retirement"] == "aborted", "not an observed aborted retirement")
            state.update(retirement="aborted", retirement_site=row["retirement_site"])
        elif event == "trace_end":
            need(state["bindings_complete"] and state["retirement"] == "aborted"
                 and not state["quantum_active"]
                 and state["http_body"] in ("pending_dropped", "dropped_before_eof", "dropped_after_eof"),
                 "incomplete trace end")
        else:
            need(False, "unsupported target event")
        state["bindings_complete"] = state["request_id"] is not None and state["worker_generation"] is not None
        for name, value in state.items():
            need(_same(row[name], value), f"{name} snapshot disagrees with reconstructed events")
        need(row["observed_close_cause"] == ("receiver_dropped" if event == "receiver_closed" else None),
             "close cause outside its observation")
        if event not in ("prime_quantum_start", "prime_quantum_end"):
            observations[event] = row
    _need("trace_end" in observations, "target missing final trace_end")
    _need(state["client_trace_key"] == client_key, "target lacks explicit client key binding")
    return state, observations, quanta


def validate_cancel_trace(raw, *, scenario, client_trace_key, expected_server_identity,
                          expected_model, expected_http_route, expected_worker_route,
                          expected_worker_generation, expected_quantum_routes):
    """Return target-bound facts and the complete decoded census, or ServingGateError."""
    _enum(scenario, {"cancel_queued", "cancel_prime", "cancel_decode"}, "cancellation scenario")
    _client_key(client_trace_key)
    _keys(expected_server_identity, {"pid", "start_identity"}, "expected server identity")
    _integer(expected_server_identity["pid"], "expected PID", 2, 2**32 - 1)
    _text(expected_server_identity["start_identity"], "expected birth identity", 512)
    _text(expected_model, "expected model")
    _enum(expected_http_route, _ROUTES, "expected HTTP route")
    _text(expected_worker_route, "expected worker dispatch route")
    _integer(expected_worker_generation, "expected worker generation")
    _need(type(expected_quantum_routes) is list and len(expected_quantum_routes) <= 32,
          "expected quantum routes must be a bounded list")
    for route in expected_quantum_routes:
        _text(route, "expected quantum route")
    _need(len(set(expected_quantum_routes)) == len(expected_quantum_routes), "duplicate expected quantum route")
    _need(scenario != "cancel_prime" or expected_quantum_routes, "prime requires a trusted actual quantum route")
    decoded = decode_lifecycle_log(raw)
    candidates = {(e["record"]["pid"], e["record"]["trace_id"]) for e in decoded["records"]
                  if e["record"]["client_trace_key"] == client_trace_key}
    _need(len(candidates) == 1, "client trace key is absent or ambiguous across traces/PIDs")
    pid, trace_id = next(iter(candidates))
    _need(pid == expected_server_identity["pid"], "target PID differs from trusted owned server")
    entries = [e for e in decoded["records"] if (e["record"]["pid"], e["record"]["trace_id"]) == (pid, trace_id)]
    expected = dict(model=expected_model, http_route=expected_http_route, worker_route=expected_worker_route,
                    worker_generation=expected_worker_generation, quantum_routes=expected_quantum_routes)
    state, events, quanta = _target_history(entries, expected, client_trace_key)
    _need(not any(e["record"]["pid"] == pid and e["record"]["trace_id"] != trace_id
                  and e["record"]["request_id"] == state["request_id"] for e in decoded["records"]),
          "minted request ID is ambiguous across owned-server traces")
    drops = [events[k] for k in ("http_pending_drop", "http_body_drop") if k in events]
    _need(len(drops) == 1 and "http_body_eof" not in events,
          "target needs exactly one HTTP drop without normal EOF")
    drop, closed, retired = drops[0], events["receiver_closed"], events["retired"]
    _need(drop["seq"] < closed["seq"] < retired["seq"] < events["trace_end"]["seq"],
          "drop, receiver closure, after-release retirement and trace end are out of order")
    site = "WorkerQueue" if scenario == "cancel_queued" else "ActiveSession"
    _need(retired["retirement_site"] == site, "missing or wrong explicit retirement site for cancellation")
    quantum = None
    if scenario == "cancel_queued":
        _need("queued" in events and events["queued"]["seq"] < drop["seq"]
              and drop["phase"] == "queued", "no queued observation before cancellation")
        _need(not quanta and "decode_start" not in events, "worker work contradicts selected queue cancellation")
    elif scenario == "cancel_prime":
        _need(drop["phase"] == "prime" and drop["quantum_active"] and drop["quantum"] is not None,
              "prime cancellation lacks a spanning actual quantum")
        quantum = quanta[drop["quantum"]["id"] - 1]
        _need(quantum["start_seq"] < drop["seq"] < quantum["end_seq"] < retired["seq"],
              "prime start/drop/return/retirement chronology differs")
        _need("decode_start" not in events, "decode observation contradicts selected prime cancellation")
    else:
        _need("prime_quantum_cancelled" not in events, "typed prime cancellation cannot establish decode")
        _need("decode_start" in events and events["decode_start"]["seq"] < drop["seq"]
              and drop["phase"] == "decode", "no actual decode observation before cancellation")
    return {"scope": "target_lifecycle_facts_only", "scenario": scenario,
            "server_identity": dict(expected_server_identity), "client_trace_key": client_trace_key,
            "target": {"pid": pid, "trace_id": trace_id, "request_id": state["request_id"],
                       "model": state["model"], "http_route": expected_http_route,
                       "worker_route": state["worker_route"], "worker_generation": state["worker_generation"]},
            "http_drop": drop, "receiver_closed": closed, "retirement": retired,
            "trace_end": events["trace_end"], "decode_start": events.get("decode_start"),
            "drop_spanning_quantum": quantum, "quanta": quanta, "decoded_log": decoded,
            "limits": "Host observations only; no GPU quiescence, cache/no-work, wire/recovery or native qualification proof."}
