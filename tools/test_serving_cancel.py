"""Pure refusal controls and real CPU HTTP/process diagnostics, never native proof."""

import copy
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import serving_cancel
import serving_http
from serving_cancel import collect_cancel_cell, evaluate_cancel_cell, validate_cancel_program
from serving_manifest import required_cells
from serving_process import OwnedServer, ReadinessEvidence
from serving_release import ServingGateError, account_attempts


SCOPE = {"id": "tiny", "model": "gate", "route": "normal", "profile": "text-generation-v1"}
OWNER = {"pid": 17, "start_identity": "boot-id:71"}


def raw(value):
    return json.dumps(value).encode()


def success(wire):
    usage = {"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 9}
    if wire == "chat_json":
        return raw({"model": "gate", "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "42"}}], "usage": usage})
    if wire == "chat_sse":
        return b"data: " + raw({"model": "gate", "choices": [{"index": 0, "finish_reason": "stop",
                              "delta": {"content": "42"}}], "usage": usage}) + b"\n\ndata: [DONE]\n\n"
    terminal = {"stop_reason": "MaxNew", "n_tokens": 1, "prompt_tokens": 8, "cached_tokens": 0, "elapsed_s": .1}
    if wire == "native_json":
        return raw({**terminal, "model": "gate", "text": "42", "tokens": [4]})
    return (b"event: message\ndata: " + raw({"model": "gate", "id": 4, "text": "42"})
            + b"\n\nevent: done\ndata: " + raw(terminal) + b"\n\n")


def typed_error(wire):
    error = raw({"error": {"code": "deadline_exceeded", "message": "request exceeded timeout",
                           "type": "timeout_error", "param": None}})
    if wire == "chat_json":
        return error
    if wire == "chat_sse":
        return b"data: " + error + b"\n\ndata: [DONE]\n\n"
    return b"event: error\ndata: " + error + b"\n\n"


def config(scenario="cancel_decode", wire="chat_sse"):
    required = next(r for r in required_cells(
        Path(__file__).with_name("serving-release.cells.json").read_bytes(), [SCOPE])
                    if r["scenario"] == scenario)
    requests = []
    for role in ("peer", "target", "recovery"):
        request = {"id": role, "role": role, "model": "gate", "wire": wire,
                   "path": "/v1/chat/completions" if wire.startswith("chat_") else "/v1/completions",
                   "payload": {"model": "gate", "stream": wire.endswith("_sse"),
                               "messages": [{"role": "user", "content": role}],
                               "fixture_role": role, "fixture_wire": wire}}
        if role != "target":
            request.update(prompt_tokens={"min": 1, "max": 32}, completion_tokens={"min": 1, "max": 16})
        requests.append(request)
    program = {"cell_id": required["id"], "scope": dict(SCOPE), "server_identity": dict(OWNER),
               "mode": "timed_disconnect_diagnostic", "requests": requests,
               "timing": {"cancel_after_s": .08, "observe_after_s": .04, "poll_s": .02, "overall_s": 8}}
    return required, program


def bundle(scenario="cancel_decode", wire="chat_sse"):
    required, program = config(scenario, wire)
    attempts = []
    for role, start, end in (("peer", 100_000_000, 300_000_000),
                             ("target", 110_000_000, 220_000_000),
                             ("recovery", 400_000_000, 500_000_000)):
        attempts.append({"id": role, "server_identity": dict(OWNER), "method": "POST",
                         "path": program["requests"][0]["path"], "started_ns": start, "finished_ns": end,
                         "status": 200, "headers": [["x-request-id", "minted-" + role]],
                         "body": b"data: {" if role == "target" else success(wire),
                         "transport_error": {"kind": "cancelled", "message": "capture cancelled"}
                         if role == "target" else None})
    health = [{"id": "health-" + str(n), "server_identity": dict(OWNER), "method": "GET", "path": "/health",
               "started_ns": at, "finished_ns": at + 1, "status": 200, "transport_error": None,
               "body": raw({"status": "ok", "worker": {"generation": 7, "phase": "busy"}})}
              for n, at in enumerate((1, 350_000_000, 600_000_000))]
    return {"required": required, "program": program, "attempts": attempts, "health_samples": health,
            "cancellation": {"id": "target", "server_identity": dict(OWNER), "invoke_started_ns": 100_000_000,
                             "set_before_ns": 200_000_000, "set_after_ns": 200_000_100}}


class CancelPolicyTests(unittest.TestCase):
    def rejects(self, value, message=None):
        with self.assertRaisesRegex(ServingGateError, message or "."):
            evaluate_cancel_cell(**value)

    def test_all_scenarios_and_wires_have_valid_diagnostics_but_no_native_pass(self):
        for scenario in ("cancel_queued", "cancel_prime", "cancel_decode"):
            for wire in ("chat_json", "chat_sse", "native_json", "native_sse"):
                value = bundle(scenario, wire); original = copy.deepcopy(value)
                result = evaluate_cancel_cell(**value)
                self.assertEqual(value, original)
                self.assertEqual(result["state"], "unqualified")
                self.assertFalse(result["qualification"])
                self.assertIsNone(result["observed_target_phase"])
                self.assertIsNone(result["server_cleanup"])
                self.assertEqual(result["accounting"]["counts"], {"clean_success": 2, "client_cancelled": 1})
                self.assertEqual(len(result["generations"]["requests"]), 3)
                self.assertEqual(len(result["missing_evidence"]), 3 if scenario == "cancel_prime" else 2)

    def test_global_idle_phase_and_success_summaries_cannot_qualify(self):
        value = bundle()
        for sample in value["health_samples"]:
            sample["body"] = raw({"status": "ok", "worker": {"generation": 7, "phase": "idle"},
                                  "passed": True, "cleanup": True, "request_id": "target"})
        value["attempts"][1].update(passed=True, cleanup=True, observed_phase="decode")
        self.assertFalse(evaluate_cancel_cell(**value)["qualification"])

    def test_full_denominator_and_role_definitions_cannot_shrink(self):
        for i in range(3):
            value = bundle(); value["attempts"].pop(i); self.rejects(value, "missing or unknown")
            value = bundle(); value["attempts"].append(value["attempts"][i]); self.rejects(value, "duplicate")
            value = bundle(); value["program"]["requests"][i]["id"] = ""; self.rejects(value)
        for key in ("peer", "target", "recovery"):
            value = bundle()
            value["program"]["requests"] = [r for r in value["program"]["requests"] if r["role"] != key]
            value["attempts"] = [r for r in value["attempts"] if r["id"] != key]
            self.rejects(value)

    def test_prime_allows_no_peer_but_still_needs_prime_bound_proof(self):
        value = bundle("cancel_prime")
        value["program"]["requests"].pop(0); value["attempts"].pop(0)
        result = evaluate_cancel_cell(**value)
        self.assertEqual(result["accounting"]["attempted"], 2)
        self.assertIn("request_bound_prime_chunk_cancellation_bound_unavailable", result["missing_evidence"])

    def test_unknown_or_weakened_manifest_and_program_refuse(self):
        for key in bundle()["required"]["requirements"]:
            value = bundle(); value["required"]["requirements"][key] = 1; self.rejects(value)
        for scenario in ("skip", "drain", [], None):
            value = bundle(); value["required"]["scenario"] = scenario; self.rejects(value)
        for key, val in (("mode", "cancel_when_global_busy"), ("cell_id", "foreign"), ("skip", True)):
            value = bundle(); value["program"][key] = val; self.rejects(value)

    def test_invalid_limits_and_probe_collisions_refuse_before_io(self):
        for key in config()[1]["timing"]:
            for val in (True, 0, -1, float("inf"), float("nan"), "1"):
                required, program = config(); program["timing"][key] = val
                with self.assertRaises(ServingGateError): validate_cancel_program(required, program)
        required, program = config(); program["requests"][0]["id"] = "cancel-probe-1"
        with self.assertRaises(ServingGateError): validate_cancel_program(required, program)

    def test_foreign_identity_wrong_route_and_minted_id_reuse_refuse(self):
        for i in range(3):
            for key, val in (("server_identity", {**OWNER, "start_identity": "other"}),
                             ("method", "GET"), ("path", "/other"), ("body", "not bytes")):
                value = bundle(); value["attempts"][i][key] = val; self.rejects(value)
        value = bundle(); value["attempts"][1]["headers"] *= 2; self.rejects(value)
        value = bundle(); value["attempts"][1]["headers"] = value["attempts"][0]["headers"]; self.rejects(value)

    def test_missing_server_id_is_unknown_not_inferred_from_client_header(self):
        value = bundle(); value["attempts"][1].update(status=None, headers=[])
        result = evaluate_cancel_cell(**value)
        self.assertIsNone(result["server_request_ids"]["target"])
        self.assertIn("target_server_request_id_not_observed", result["missing_evidence"])

    def test_late_cancel_cannot_hide_a_complete_terminal_for_any_wire(self):
        for wire in ("chat_json", "chat_sse", "native_json", "native_sse"):
            value = bundle(wire=wire); value["attempts"][1]["body"] = success(wire)
            self.rejects(value, "complete response")
        value = bundle(); value["attempts"][1]["transport_error"] = None
        self.rejects(value, "client-cancelled")

    def test_target_transport_failure_and_http_errors_are_not_cancellation(self):
        for kind in ("timeout", "connect", "read", "disconnect"):
            value = bundle(); value["attempts"][1]["transport_error"]["kind"] = kind; self.rejects(value)
        for status in (400, 429, 503):
            value = bundle(); value["attempts"][1]["status"] = status; self.rejects(value)

    def test_decoded_target_server_error_is_failure_even_with_client_cancel_flag(self):
        for wire in ("chat_json", "chat_sse", "native_sse"):
            with self.subTest(wire=wire):
                value = bundle(wire=wire); value["attempts"][1]["body"] = typed_error(wire)
                before = copy.deepcopy(value)
                with self.assertRaises(ServingGateError) as caught:
                    evaluate_cancel_cell(**value)
                self.assertEqual(caught.exception.code, "typed_error")
                self.assertIn("deadline_exceeded", str(caught.exception))
                self.assertEqual(value, before, "failure must not rewrite raw cancellation/body")
                wire_counts = account_attempts(value["program"]["requests"], value["attempts"])["counts"]
                self.assertEqual(wire_counts, {"clean_success": 2, "client_cancelled": 1})

    def test_peer_recovery_truncation_transport_and_bad_usage_never_disappear(self):
        for i in (0, 2):
            for key, val in (("body", b"data: {}\n\n"), ("body", b""),
                             ("transport_error", {"kind": "cancelled", "message": "cancelled"})):
                value = bundle(); value["attempts"][i][key] = val; self.rejects(value)
            value = bundle(); value["program"]["requests"][i]["prompt_tokens"]["min"] = 9
            self.rejects(value)

    def test_cancellation_interval_must_be_observed_and_spanned(self):
        for key, val in (("set_before_ns", 100_000_000), ("set_before_ns", 221_000_000),
                         ("set_after_ns", 190_000_000), ("set_before_ns", True),
                         ("id", "peer"), ("server_identity", {**OWNER, "pid": 18})):
            value = bundle(); value["cancellation"][key] = val; self.rejects(value)
        value = bundle(); value["attempts"][0]["finished_ns"] = 190_000_000; self.rejects(value, "peer")
        value = bundle(); value["attempts"][2]["started_ns"] = 250_000_000; self.rejects(value, "recovery")

    def test_client_can_observe_set_before_cancelling_thread_returns(self):
        value = bundle()
        # Event.set wakes the HTTP thread; its actual cancellation result and
        # finish can precede the cancelling thread's next scheduled instruction.
        value["cancellation"]["set_after_ns"] = 330_000_000
        before = copy.deepcopy(value)
        result = evaluate_cancel_cell(**value)
        self.assertEqual(result["accounting"]["counts"], {"clean_success": 2, "client_cancelled": 1})
        self.assertEqual(value, before)
        # The peer must span the *observed* upper bound (target finish), even
        # though it need not span the later caller scheduling observation.
        value["attempts"][0]["finished_ns"] = 219_000_000
        self.rejects(value, "peer")
        value = before
        value["attempts"][1]["body"] = success("chat_sse")
        self.rejects(value, "complete response")

    def test_health_cannot_hide_generation_changes_or_unbracketed_targets(self):
        for key, val in (("path", "/metrics"), ("method", "POST"), ("status", 503),
                         ("body", b"{"), ("transport_error", {"kind": "read", "message": "truncated"}),
                         ("body", raw({"status": "ok", "worker": {"generation": 8}})),
                         ("id", "target")):
            value = bundle(); value["health_samples"][-1][key] = val; self.rejects(value)
        value = bundle(); value["health_samples"].pop(0); self.rejects(value, "bracket")


# This is an HTTP/protocol fixture, not Memra telemetry. Fixture events are used
# ONLY by tests to establish what the fixture did; serving_cancel never parses them.
SERVER = r'''
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json,select,signal,socket,sys,time,threading
from pathlib import Path
mode=sys.argv[2]
lock=threading.Lock()
active=set()
def log(event, **kw): print(json.dumps(dict(event=event,**kw)),flush=True)
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def reply(self,data,extra=0):
        self.send_response(200);self.send_header('Content-Length',str(len(data)+extra))
        self.send_header('x-request-id','minted-'+self.headers.get('X-Request-Id',''))
        self.end_headers();self.wfile.write(data);self.wfile.flush()
    def do_GET(self):
        with lock: count=len(active)
        if self.path=='/metrics':
            self.reply(json.dumps(dict(active_sessions=count)).encode(),5 if mode=='bad_metrics' else 0)
        else:
            self.reply(json.dumps(dict(status='ok',worker=dict(generation=0,phase='busy' if count else 'idle'))).encode())
    def do_POST(self):
        p=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        role,wire=p['fixture_role'],p['fixture_wire']
        label=self.headers['X-Request-Id']
        with lock: active.add(label)
        log('request',role=role,id=label)
        try:
            if mode in ('truncated_target','cancel_before_eof','typed_error','finish_first'):
                # Fixture-only rendezvous: the peer is genuinely in flight before
                # the target response. The collector never consumes these files.
                if role=='peer':
                    Path('peer-started').touch()
                    end=time.monotonic()+4
                    while not Path('release-eof').exists() and time.monotonic()<end: time.sleep(.005)
                    if not Path('release-eof').exists(): raise RuntimeError('fixture peer barrier expired')
                elif role=='target':
                    end=time.monotonic()+4
                    while not Path('peer-started').exists() and time.monotonic()<end: time.sleep(.005)
                    if not Path('peer-started').exists(): raise RuntimeError('fixture start barrier expired')
                    if mode in ('cancel_before_eof','typed_error'):
                        data=bytes.fromhex(p['fixture_error']) if mode=='typed_error' else b'data: {'
                        self.reply(data,5)
                        end=time.monotonic()+4
                        while not Path('release-eof').exists() and time.monotonic()<end: time.sleep(.005)
                        if not Path('release-eof').exists(): raise RuntimeError('fixture EOF barrier expired')
                        log('eof_released',id=label,observed_ns=time.monotonic_ns())
                        return
            if role=='target' and mode not in ('finish_first','truncated_target'):
                self.send_response(200);self.send_header('x-request-id','minted-'+label);self.end_headers()
                self.wfile.write(b'data: {');self.wfile.flush()
                end=time.monotonic()+2
                while time.monotonic()<end:
                    if select.select([self.connection],[],[],.01)[0] and self.connection.recv(1,socket.MSG_PEEK)==b'':
                        log('client_eof',id=label)
                        if mode=='leaked_target':
                            log('cleanup_withheld',id=label)
                            time.sleep(.8)
                        break
                return
            if role=='peer': time.sleep(.25)
            if role=='target' and mode=='truncated_target': self.reply(b'data: {',5);return
            data=bytes.fromhex(p['fixture_response'])
            if role=='recovery' and mode=='truncated_recovery': data=b'data: {}\n\n'
            self.reply(data)
        finally:
            with lock: active.discard(label)
            log('retired',id=label)
signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))
with ThreadingHTTPServer(('127.0.0.1',int(sys.argv[1])),Handler) as server:
    print('FIXTURE_READY',flush=True)
    server.serve_forever(poll_interval=.01)
'''


def read(output, ref):
    data = (output / ref["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise AssertionError("evidence hash mismatch")
    return json.loads(data)


class CancellationOrder:
    """Test-only scheduling barriers around the real timer and HTTP reader.

    EOF-first delays the timer until invoke's finally has marked its own done
    event and joins it; the HTTP capture has already returned its real result.
    Cancel-first releases the timer on a later target read, after prior bytes
    were recorded by capture_request, while the fixture withholds EOF. A typed
    error fixture requires its entire known body before releasing the timer;
    TCP fragmentation must not turn this into a fixed read-count assumption.
    Neither branch fabricates an HTTP outcome, timestamp or cancellation receipt.
    """

    def __init__(self, order, root, *, before_cancel_body=None):
        self.order, self.root = order, root
        self.release = threading.Event()
        self.proof = {"order": order, "errors": []}
        self.target_reads = 0
        self.delivered = bytearray()
        self.before_cancel_body = before_cancel_body

    def release_timer(self):
        self.proof["timer_released_ns"] = time.monotonic_ns()
        self.release.set()

    def install(self, stack):
        start, join = threading.Thread.start, threading.Thread.join
        capture, read1 = serving_cancel.capture_request, serving_http._StrictHTTPResponse.read1

        def ordered_start(thread):
            if thread.name == "serving-cancel-timer":
                original = thread.run
                def gated():
                    self.proof["timer_waiting_ns"] = time.monotonic_ns()
                    if not self.release.wait(4):
                        self.proof["errors"].append("timer barrier expired")
                        return
                    original()
                thread.run = gated
            return start(thread)

        def ordered_join(thread, *args, **kwargs):
            if thread.name == "serving-cancel-timer" and self.order == "eof_first":
                # This join is in invoke's finally, AFTER its done.set().
                self.release_timer()
                self.root.joinpath("release-eof").touch()
            return join(thread, *args, **kwargs)

        def observe_capture(**kwargs):
            if kwargs["request_id"] != "target":
                return capture(**kwargs)
            event = kwargs["cancel_event"]
            original_set = event.set
            def observed_set():
                self.proof["event_set_before_ns"] = time.monotonic_ns()
                original_set()
                self.proof["event_set_after_ns"] = time.monotonic_ns()
                self.root.joinpath("release-eof").touch()
            event.set = observed_set
            try:
                result = capture(**kwargs)
                self.proof["capture_finished_ns"] = result["finished_ns"]
                self.proof["capture_returned_ns"] = time.monotonic_ns()
                return result
            finally:
                event.set = original_set

        def observe_read(response, *args, **kwargs):
            target = response.getheader("x-request-id") == "minted-target"
            if target:
                self.target_reads += 1
                ready = bool(self.delivered) and (self.before_cancel_body is None
                        or self.delivered == self.before_cancel_body)
                if ready and not self.release.is_set() and self.order == "cancel_first":
                    # A new read begins only after capture_request has recorded
                    # all bytes returned by its previous reads.
                    self.proof["release_read_entered_ns"] = time.monotonic_ns()
                    self.proof["release_read_index"] = self.target_reads
                    self.proof["delivered_before_release_bytes"] = len(self.delivered)
                    self.proof["delivered_before_release_sha256"] = hashlib.sha256(self.delivered).hexdigest()
                    self.release_timer()
            chunk = read1(response, *args, **kwargs)
            if target:
                self.delivered.extend(chunk)
            return chunk

        stack.enter_context(patch.object(threading.Thread, "start", ordered_start))
        stack.enter_context(patch.object(threading.Thread, "join", ordered_join))
        stack.enter_context(patch.object(serving_cancel, "capture_request", observe_capture))
        stack.enter_context(patch.object(serving_http._StrictHTTPResponse, "read1", observe_read))


class CancellationOrderTests(unittest.TestCase):
    def test_typed_error_timer_waits_for_all_fragments_to_reach_capture_loop(self):
        expected = typed_error("chat_sse")
        layouts = ([expected], [expected[:6], expected[6:]],
                   [bytes([value]) for value in expected])
        for fragments in layouts:
            with self.subTest(fragments=len(fragments)), tempfile.TemporaryDirectory() as tmp:
                observer = CancellationOrder("cancel_first", Path(tmp), before_cancel_body=expected)
                captured = bytearray();pending = iter([*fragments, b""])
                seen = []
                class Response:
                    def getheader(self, name):return "minted-target"
                def read_fragment(_response, *args, **kwargs):
                    seen.append((len(captured), observer.release.is_set()))
                    return next(pending)
                response = Response()
                with ExitStack() as stack:
                    stack.enter_context(patch.object(serving_http._StrictHTTPResponse, "read1", read_fragment))
                    observer.install(stack)
                    for fragment in fragments:
                        chunk = serving_http._StrictHTTPResponse.read1(response, 16384)
                        self.assertEqual(chunk, fragment)
                        self.assertFalse(observer.release.is_set())
                        captured.extend(chunk)
                    self.assertEqual(captured, expected)
                    self.assertEqual(serving_http._StrictHTTPResponse.read1(response, 16384), b"")
                self.assertTrue(all(not released for _, released in seen[:-1]))
                self.assertEqual(seen[-1], (len(expected), True))
                self.assertEqual(observer.proof["release_read_index"], len(fragments) + 1)
                self.assertEqual(observer.proof["delivered_before_release_bytes"], len(expected))

    def test_incomplete_or_different_error_body_does_not_release_timer(self):
        expected = typed_error("chat_sse")
        for body in (expected[:-1], b"X" + expected[1:]):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as tmp:
                observer = CancellationOrder("cancel_first", Path(tmp), before_cancel_body=expected)
                pending = iter([body, b""])
                class Response:
                    def getheader(self, name):return "minted-target"
                with ExitStack() as stack:
                    stack.enter_context(patch.object(serving_http._StrictHTTPResponse, "read1",
                                                    lambda *a, **k: next(pending)))
                    observer.install(stack)
                    response = Response()
                    serving_http._StrictHTTPResponse.read1(response, 16384)
                    serving_http._StrictHTTPResponse.read1(response, 16384)
                self.assertFalse(observer.release.is_set())
                self.assertNotIn("timer_released_ns", observer.proof)


class CancelProcessTests(unittest.TestCase):
    def run_capture(self, mode="normal", *, native_listener=False, timing=None, interrupt_timer=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); script = root / "fixture.py"; script.write_text(SERVER)
            with socket.socket() as port_socket:
                port_socket.bind(("127.0.0.1", 0)); port = port_socket.getsockname()[1]
            class FixtureServer(OwnedServer):
                def receipt(self):
                    value = super().receipt()
                    if sys.platform != "linux":
                        # Darwin has no boot_id in OwnedServer. Only this CPU
                        # fixture supplies a labelled stand-in; native requires Linux.
                        value = {**value, "boot_id": "fixture-only-no-linux-boot-id"}
                    return value
            server = FixtureServer(argv=[sys.executable, str(script), str(port), mode], env={}, cwd=str(root),
                                 evidence_dir=str(root / "process"), startup_timeout=5, overall_timeout=15,
                                 drain_timeout=2, kill_timeout=1)
            result = None
            try:
                server.start()
                deadline = time.monotonic() + 4
                while time.monotonic() < deadline:
                    if server.output_path.exists() and "FIXTURE_READY" in server.output_path.read_text(): break
                    time.sleep(.01)
                self.assertIn("FIXTURE_READY", server.output_path.read_text())
                server.mark_ready(ReadinessEvidence(server.identity, "owned_output", {"fixture": "FIXTURE_READY"}))
                required, program = config()
                if timing is not None:
                    program["timing"] = timing
                program["server_identity"] = {"pid": server.identity.pid,
                    "start_identity": server.receipt()["boot_id"] + ":" + server.identity.start_time}
                for request in program["requests"]:
                    request["payload"]["fixture_response"] = success(request["wire"]).hex()
                    request["payload"]["fixture_error"] = typed_error(request["wire"]).hex()
                output = root / "capture"
                options = dict(server=server,host="127.0.0.1",port=port,output=output,
                    http=dict(connect_timeout=.5,read_timeout=1,wall_timeout=3,max_body_bytes=16384))
                # Portable runs use a real process+HTTP but do NOT prove Linux procfs
                # listener ownership. That is the separate Linux-only control below.
                ordering = ("eof_first" if mode in ("truncated_target", "finish_first") else
                            "cancel_first" if mode in ("cancel_before_eof", "typed_error") else None)
                self.last_order_proof, self.last_cancellations = None, []
                with ExitStack() as stack:
                    if not native_listener:
                        proof = ReadinessEvidence(server.identity, "listener_identity", {"fixture_only": True})
                        stack.enter_context(patch.object(serving_cancel, "prove_listener", return_value=proof))
                    if ordering:
                        observer = CancellationOrder(ordering, root,
                            before_cancel_body=typed_error("chat_sse") if mode == "typed_error" else None)
                        observer.install(stack)
                    elif interrupt_timer:
                        original_start = threading.Thread.start
                        def start(thread):
                            original_start(thread)
                            if thread.name == "serving-cancel-timer":
                                raise KeyboardInterrupt("fixture interrupted after native timer started")
                        stack.enter_context(patch.object(threading.Thread, "start", start))
                    result = collect_cancel_cell(required, program, **options)
                if ordering:
                    self.last_order_proof = observer.proof
                    (root / "ordering.json").write_text(json.dumps(observer.proof, indent=2) + "\n")
                    self.assertEqual(observer.proof["errors"], [])
                self.last_cancellations = [read(output, ref) for ref in result["cancellation_events"]]
                self.assertIsNone(server.receipt().get("stop"), "borrowed server was stopped by collector")
                index = json.loads((output / "capture.json").read_text())
                for path, digest in index["payloads"].items():
                    self.assertEqual(hashlib.sha256((output / path).read_bytes()).hexdigest(), digest)
                rows = [read(output, ref) for ref in result["observations"]]
                for row in rows:
                    row["body"] = (output / row["body"]["path"]).read_bytes()
                evaluation = read(output, result["evaluation"]) if "evaluation" in result else None
                wire = read(output, result["wire_accounting"]) if "wire_accounting" in result else None
                logs = server.output_path.read_text()
                return result, rows, evaluation, wire, logs
            finally:
                receipt = server.close(reason="fixture_complete")
                artifacts = os.environ.get("MEMRA_CANCEL_TEST_EVIDENCE_DIR")
                if artifacts:
                    destination = Path(tempfile.mkdtemp(prefix=mode + "-", dir=artifacts))
                    shutil.copytree(root, destination / "fixture")
                self.assertEqual(receipt["server_exit"]["returncode"], 0, receipt)
                self.assertTrue(receipt["cleanup"]["complete"], receipt)
                self.assertFalse(receipt["cleanup"]["escalated"], receipt)
                self.assertFalse(any(t.name.startswith(("serving-client-", "serving-http-", "serving-cancel-"))
                                     for t in threading.enumerate()))

    def test_real_target_disconnect_clean_peers_and_recovery_still_unqualified(self):
        result, rows, evaluation, wire, logs = self.run_capture()
        self.assertEqual(result["state"], "unqualified", result)
        self.assertEqual(result["unattempted_ids"], [])
        self.assertEqual(wire["counts"], {"clean_success": 2, "client_cancelled": 1})
        self.assertIsNone(evaluation["server_cleanup"])
        self.assertIn('"event": "client_eof"', logs)
        self.assertIn('"event": "retired", "id": "target"', logs)
        self.assertEqual({r["id"] for r in rows}, {"peer", "target", "recovery"})
        self.assertEqual(logs.count('"event": "request"'), 3, "inference retried")

    def test_real_successful_recovery_cannot_hide_withheld_target_cleanup(self):
        result, rows, evaluation, wire, logs = self.run_capture("leaked_target")
        self.assertEqual(result["state"], "unqualified", result)
        self.assertIn('"event": "cleanup_withheld"', logs)
        self.assertNotIn('"event": "retired", "id": "target"', logs)
        self.assertEqual(wire["counts"], {"clean_success": 2, "client_cancelled": 1})
        self.assertIsNone(evaluation["server_cleanup"])

    def test_real_target_finished_before_cancel_is_failed_not_phase_success(self):
        result, rows, evaluation, wire, logs = self.run_capture("finish_first")
        self.assertEqual(result["state"], "failed", result)
        self.assertEqual(wire["counts"], {"clean_success": 3})
        self.assertEqual(result["cancellation_events"], [])
        self.assertEqual(len(rows), 3)
        self.assertIsNone(evaluation)

    def test_real_truncated_target_does_not_become_client_cancellation(self):
        result, rows, evaluation, wire, logs = self.run_capture("truncated_target")
        target = next(r for r in rows if r["id"] == "target")
        order = self.last_order_proof
        self.assertLessEqual(target["finished_ns"], order["capture_returned_ns"])
        self.assertLess(order["capture_returned_ns"], order["timer_released_ns"])
        self.assertNotIn("event_set_before_ns", order)
        self.assertEqual(result["cancellation_events"], [])
        self.assertEqual(result["state"], "failed", result)
        self.assertEqual(wire["counts"], {"clean_success": 2, "transport_error": 1})
        self.assertEqual(target["transport_error"]["kind"], "read")
        self.assertIn("IncompleteRead", target["transport_error"]["message"])
        self.assertEqual(len(rows), 3)

    def assert_cancel_before_eof(self, rows, logs):
        target = next(r for r in rows if r["id"] == "target")
        order = self.last_order_proof
        self.assertEqual(len(self.last_cancellations), 1)
        cancellation = self.last_cancellations[0]
        eof = next(json.loads(line) for line in logs.splitlines() if '"event": "eof_released"' in line)
        self.assertLess(target["chunks"][0]["observed_ns"], order["release_read_entered_ns"])
        self.assertLessEqual(order["timer_released_ns"], cancellation["set_before_ns"])
        self.assertLessEqual(cancellation["set_before_ns"], order["event_set_before_ns"])
        self.assertLessEqual(order["event_set_after_ns"], cancellation["set_after_ns"])
        self.assertLess(order["event_set_after_ns"], eof["observed_ns"])
        self.assertEqual(target["transport_error"]["kind"], "cancelled")
        return target

    def test_real_cancellation_before_fixture_eof_remains_client_cancelled(self):
        result, rows, evaluation, wire, logs = self.run_capture("cancel_before_eof")
        self.assert_cancel_before_eof(rows, logs)
        self.assertEqual(wire["counts"], {"clean_success": 2, "client_cancelled": 1})
        self.assertEqual(result["state"], "unqualified", result)
        self.assertFalse(result["qualification"])

    def test_real_typed_error_before_cancel_fails_with_all_raw_rows_preserved(self):
        result, rows, evaluation, wire, logs = self.run_capture("typed_error")
        target = self.assert_cancel_before_eof(rows, logs)
        self.assertEqual(target["body"], typed_error("chat_sse"))
        self.assertEqual(self.last_order_proof["delivered_before_release_bytes"], len(target["body"]))
        self.assertEqual(self.last_order_proof["delivered_before_release_sha256"],
                         hashlib.sha256(target["body"]).hexdigest())
        self.assertEqual(target["chunks"][-1]["end_offset"], len(target["body"]))
        self.assertLess(target["chunks"][-1]["observed_ns"], self.last_cancellations[0]["set_before_ns"])
        self.assertEqual(result["state"], "failed", result)
        self.assertTrue(any("deadline_exceeded" in error for error in result["errors"]), result)
        self.assertEqual(wire["counts"], {"clean_success": 2, "client_cancelled": 1})
        self.assertEqual(len(rows), 3)
        self.assertEqual(result["unattempted_ids"], [])
        self.assertIsNone(evaluation)

    def test_real_http200_truncated_recovery_remains_in_failed_denominator(self):
        result, rows, evaluation, wire, logs = self.run_capture("truncated_recovery")
        self.assertEqual(result["state"], "failed", result)
        self.assertEqual(len(rows), 3)
        self.assertEqual(wire["attempted"], 3)
        self.assertNotEqual(wire["counts"].get("clean_success"), 2)

    def test_real_probe_framing_failure_keeps_all_inference_attempts(self):
        result, rows, evaluation, wire, logs = self.run_capture("bad_metrics")
        self.assertEqual(result["state"], "failed", result)
        self.assertEqual(len(rows), 3)
        self.assertEqual(wire["counts"], {"clean_success": 2, "client_cancelled": 1})

    def test_real_overall_expiry_marks_recovery_unattempted_without_stopping_owner(self):
        timing = {"cancel_after_s": .025, "observe_after_s": .03, "poll_s": .01, "overall_s": .14}
        result, rows, evaluation, wire, logs = self.run_capture(timing=timing)
        self.assertEqual(result["state"], "failed", result)
        self.assertIn("recovery", result["unattempted_ids"])
        self.assertEqual(result["planned"], 3)
        self.assertEqual(result["captured_attempts"] + len(result["unattempted_ids"]), 3)
        self.assertNotIn('"role": "recovery"', logs)

    def test_native_timer_launch_interruption_joins_timer_and_retains_other_rows(self):
        result, rows, evaluation, wire, logs = self.run_capture(interrupt_timer=True)
        self.assertEqual(result["state"], "failed", result)
        self.assertEqual(result["unattempted_ids"], ["target", "recovery"])
        self.assertIn("target", result["client_errors"])
        self.assertEqual([r["id"] for r in rows], ["peer"])
        self.assertEqual(result["cancellation_events"], [])

    @unittest.skipUnless(sys.platform == "linux", "actual listener ownership requires Linux procfs")
    def test_linux_owned_listener_and_process_with_actual_cancellation(self):
        result, rows, evaluation, wire, logs = self.run_capture(native_listener=True)
        self.assertEqual(result["state"], "unqualified", result)
        self.assertEqual(wire["counts"], {"clean_success": 2, "client_cancelled": 1})
        self.assertFalse(evaluation["qualification"])


if __name__ == "__main__":
    unittest.main()
