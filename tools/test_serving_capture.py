"""Plan/evidence controls plus a real Linux owned HTTP-process capture."""

import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import serving_capture
from serving_capture import EvidenceStore, collect, collect_clients, encoded, file_identity, validate_plan
from serving_evidence import read_group_capture
from serving_process import ProcessIdentity, ReadinessEvidence
from serving_release import ServingGateError


SERVER = r'''
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json,signal,sys,time,threading
METRICS_MODE = "normal"
METRICS_BAD_CALL = 1
class Handler(BaseHTTPRequestHandler):
    completed = 0
    metrics_calls = 0
    lock = threading.Lock()
    def log_message(self,*args): pass
    def do_GET(self):
        print("HTTP_EVENT "+json.dumps(["GET",self.path,self.headers.get("X-Request-Id")]),flush=True)
        if self.path == "/metrics":
            with Handler.lock:
                Handler.metrics_calls += 1
                number,completed = Handler.metrics_calls,Handler.completed
            mode = METRICS_MODE if number == METRICS_BAD_CALL else "normal"
            # Prefix counters intentionally differ from total cached-token counts.
            value={"prompt_tokens_in":completed*4,"cached_tokens_in":completed*2,
                   "computed_tokens_in":completed*2,"prefix_cache_hit_tokens":0}
            data=json.dumps(value).encode()
            if mode == "malformed": data=b'{"prompt_tokens_in":'
            if mode == "non200": data=b'{"error":{"code":"metrics_unavailable"}}'
            self.send_response(503 if mode == "non200" else 200)
            self.send_header("Content-Length",str(len(data)+(5 if mode == "truncated" else 0)))
            self.end_headers();self.wfile.write(data)
            return
        value={"status":"ok","models":["gate"],"worker":{"generation":0}}
        self.reply(json.dumps(value).encode())
    def reply(self,data):
        self.send_response(200);self.send_header("Content-Length",str(len(data)))
        self.end_headers();self.wfile.write(data)
    def do_POST(self):
        print("HTTP_EVENT "+json.dumps(["POST",self.path,self.headers.get("X-Request-Id")]),flush=True)
        request=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        time.sleep(.03)
        with Handler.lock: Handler.completed += 1
        if request["stream"]:
            self.reply(b'data: {"model":"gate","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}\n\n')
        else:
            value={"model":"gate","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5,"prompt_tokens_details":{"cached_tokens":0}}}
            self.reply(json.dumps(value).encode())
signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))
with ThreadingHTTPServer(("127.0.0.1",int(sys.argv[1])),Handler) as server:
    server.serve_forever(poll_interval=.02)
'''


def plan(root, port=12345):
    request = {"id": "one", "wire": "chat_json", "model": "gate", "path": "/v1/chat/completions",
               "payload": {"model": "gate", "stream": False, "max_tokens": 8,
                           "messages": [{"role": "user", "content": "A short question"}], "temperature": 0}}
    return {"schema": "memra-serving-capture-plan-v1",
            "server": {"argv": [sys.executable, str(root / "server.py"), str(port)],
                       "env": {}, "cwd": str(root), "host": "127.0.0.1", "port": port,
                       "startup_timeout": 5, "overall_timeout": 20, "drain_timeout": 1,
                       "kill_timeout": 1},
            "identities": {"server_binary": sys.executable, "fixture_source": str(root / "server.py")},
            "http": {"connect_timeout": .3, "read_timeout": 1, "wall_timeout": 2,
                     "max_body_bytes": 16384},
            "groups": [{"id": "serial", "mode": "serial", "requests": [request]}]}


def read_blob(output, ref):
    data = (output / ref["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise AssertionError("fixture evidence blob differs from recorded hash")
    return data


def observation(output, ref):
    row = json.loads(read_blob(output, ref))
    row["body"] = read_blob(output, row["body"])
    return row


class CaptureInputTests(unittest.TestCase):
    def test_blob_identity_and_original_bytes_survive_deduplication(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = EvidenceStore(Path(temporary) / "capture")
            raw = b"data: incomplete\xff\r\n"
            ref = store.put(raw)
            self.assertEqual(ref, store.put(raw))
            self.assertEqual((store.root / ref["path"]).read_bytes(), raw)
            self.assertEqual(len(store.payloads), 1)
            store.index({"qualification": False, "response": ref})
            result = json.loads((store.root / "capture.json").read_text())
            self.assertFalse(result["qualification"])
            self.assertEqual(result["payloads"][ref["path"]], hashlib.sha256(raw).hexdigest())
            with self.assertRaises(FileExistsError):
                EvidenceStore(store.root)

    def test_plan_rejects_ambiguous_requests_and_unbounded_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            original = plan(Path(temporary).resolve())
            self.assertEqual(validate_plan(original), {"one"})
            variants = []
            bad = copy.deepcopy(original);bad["groups"][0]["requests"] *= 2;variants.append(bad)
            bad = copy.deepcopy(original);bad["groups"][0]["requests"][0]["payload"]["model"] = "other";variants.append(bad)
            bad = copy.deepcopy(original);bad["groups"][0]["requests"][0]["payload"]["stream"] = True;variants.append(bad)
            bad = copy.deepcopy(original);bad["http"]["wall_timeout"] = float("inf");variants.append(bad)
            bad = copy.deepcopy(original);bad["server"]["host"] = "example.com";variants.append(bad)
            bad = copy.deepcopy(original);bad["server"]["env"]["MEMRA_API_KEY"] = "secret";variants.append(bad)
            bad = copy.deepcopy(original);bad["identities"]["server_binary"] = "/other";variants.append(bad)
            for bad in variants:
                with self.subTest(plan=bad), self.assertRaises(ServingGateError):
                    validate_plan(bad)

    def test_file_identity_reads_actual_replaced_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary).resolve() / "artifact"; p.write_bytes(b"first")
            first = file_identity(p)
            p.unlink();p.write_bytes(b"second")
            self.assertNotEqual(file_identity(p), first)
            self.assertEqual(file_identity(p)["bytes"], 6)

    def test_metrics_flag_is_optional_but_never_coerced(self):
        with tempfile.TemporaryDirectory() as temporary:
            original = plan(Path(temporary).resolve())
            self.assertEqual(validate_plan(original), {"one"})
            for value in (False, True):
                config = copy.deepcopy(original);config["groups"][0]["metrics"] = value
                before = copy.deepcopy(config)
                self.assertEqual(validate_plan(config), {"one"})
                self.assertEqual(config, before)
            for value in (None, 0, 1, "true", "false", [], {}):
                config = copy.deepcopy(original);config["groups"][0]["metrics"] = value
                with self.subTest(value=value), self.assertRaisesRegex(ServingGateError, "metrics flag"):
                    validate_plan(config)

    def test_generated_probe_ids_cannot_alias_requests_or_each_other(self):
        with tempfile.TemporaryDirectory() as temporary:
            original = plan(Path(temporary).resolve())
            original["groups"][0]["metrics"] = True
            second = copy.deepcopy(original["groups"][0]);second["id"] = "later"
            second["requests"][0]["id"] = "two"
            original["groups"].append(second)
            for index in (0, 1):
                for request_id in ("serial-before", "serial-after", "serial-metrics-before",
                                   "serial-metrics-after", "later-before", "later-metrics-after",
                                   "startup-health", "startup-health-2", "startup-health-999"):
                    config = copy.deepcopy(original)
                    config["groups"][index]["requests"][0]["id"] = request_id
                    with self.subTest(index=index, request_id=request_id), self.assertRaises(ServingGateError):
                        validate_plan(config)
            # Distinct group names can still generate the SAME health/metrics ID.
            for reverse in (False, True):
                config = copy.deepcopy(original);config["groups"][1]["id"] = "serial-metrics"
                if reverse: config["groups"].reverse()
                with self.assertRaisesRegex(ServingGateError, "probe IDs collide"):
                    validate_plan(config)
            config = copy.deepcopy(original);config["groups"][0]["id"] = "startup-health"
            with self.assertRaisesRegex(ServingGateError, "startup namespace"):
                validate_plan(config)
            # Unused optional metrics IDs are not reserved when capture is disabled.
            config = plan(Path(temporary).resolve())
            config["groups"][0]["requests"][0]["id"] = "serial-metrics-before"
            self.assertEqual(validate_plan(config), {"serial-metrics-before"})

    def test_metrics_collection_control_flow_with_explicit_process_and_http_doubles(self):
        # Pure collector control: no ownership/HTTP qualification is inferred from
        # these doubles. LinuxCaptureTests below exercise the real process/socket.
        for flag, fault, phase in [(None, None, None), (False, None, None), (True, None, None),
                                    (True, "malformed", "before"), (True, "non200", "after"),
                                    (True, "truncated", "before"), (True, "truncated", "after")]:
            with self.subTest(flag=flag, fault=fault, phase=phase), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve();(root / "server.py").write_text(SERVER)
                config = plan(root)
                if flag is not None: config["groups"][0]["metrics"] = flag
                identity = ProcessIdentity(101, 100, 101, "7", "test_double", "S")
                owner = {"pid": 101, "start_identity": "fixture-boot:7"}
                receipt = {"boot_id": "fixture-boot", "state": "running", "errors": [],
                           "cleanup": {"complete": True, "escalated": False}, "server_exit": {"returncode": 0}}
                server = Mock(identity=identity, _supervisor=object())
                server.receipt.return_value = receipt
                server.close.side_effect = lambda *, reason: {**receipt, "stop": {"reason": reason}}
                events, tick, startups = [], [0], [0]
                def proof(*args, **kwargs):
                    events.append("listener")
                    return ReadinessEvidence(identity, "listener_identity", {"test_double": True})
                def capture(**kwargs):
                    label, path = kwargs["request_id"], kwargs["path"]
                    events.append(label);tick[0] += 10
                    status, transport = 200, None
                    if path == "/metrics":
                        raw = b'{"prompt_tokens_in":8,"cached_tokens_in":4,"prefix_cache_hit_tokens":0}'
                        if label.endswith(phase or "not-a-phase"):
                            if fault == "malformed": raw = b'{"prompt_tokens_in":'
                            if fault == "non200": status, raw = 503, b"unavailable"
                            if fault == "truncated": transport = {"kind": "read", "message": "IncompleteRead"}
                    elif path == "/health":
                        raw = encoded({"status": "ok", "models": ["gate"], "worker": {"generation": 0}})
                        if label.startswith("startup-health"):
                            startups[0] += 1
                            if startups[0] == 1: status = 503
                    else:
                        raw = encoded({"model": "gate", "choices": [{"index": 0, "finish_reason": "stop",
                                      "message": {"role": "assistant", "content": "ok"}}],
                                      "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}})
                    return {"id": label, "status": status, "body": raw, "headers": [],
                            "started_ns": tick[0], "finished_ns": tick[0] + 1, "transport_error": transport}
                with patch.object(serving_capture, "OwnedServer", return_value=server), \
                     patch.object(serving_capture, "prove_listener", side_effect=proof), \
                     patch.object(serving_capture, "capture_request", side_effect=capture):
                    output = root / "capture";result = collect(config, output)
                self.assertFalse(result["qualification"])
                self.assertEqual(result["state"], "failed" if fault == "truncated" else "captured", result)
                group = result["groups"][0]
                self.assertEqual(group["state"], result["state"])
                expected = ["listener", "startup-health", "listener", "startup-health-2",
                            "listener", "listener", "serial-before"]
                if flag: expected.append("serial-metrics-before")
                expected.append("one")
                if flag: expected.append("serial-metrics-after")
                expected += ["serial-after", "listener"]
                self.assertEqual(events, expected)
                self.assertEqual(len(set(e for e in events if e != "listener")), len(events) - events.count("listener"))
                self.assertEqual(json.loads(read_blob(output, group["wire_accounting"]))["attempted"], 1)
                self.assertIn("generation_accounting", group)
                for key in ("metrics_before", "metrics_after"):
                    if not flag:
                        self.assertNotIn(key, group)
                        continue
                    row = observation(output, group[key])
                    self.assertEqual(row["server_identity"], owner)
                    self.assertEqual((row["method"], row["path"]), ("GET", "/metrics"))
                    if fault and key.endswith(phase):
                        if fault == "malformed": self.assertEqual(row["body"], b'{"prompt_tokens_in":')
                        if fault == "non200": self.assertEqual((row["status"], row["body"]), (503, b"unavailable"))
                        if fault == "truncated": self.assertEqual(row["transport_error"]["kind"], "read")

    def test_serial_interruption_preserves_completed_client_immediately(self):
        saved, errors = [], {}
        def invoke(request, cancel):
            if request["id"] == "two":
                self.assertEqual(saved, [{"id": "one", "body": b"original bytes"}])
                raise KeyboardInterrupt("second client interrupted")
            return {"id": "one", "body": b"original bytes"}
        with self.assertRaises(ServingGateError):
            collect_clients([{"id": "one"}, {"id": "two"}], "serial", invoke, saved.append, .5, errors)
        self.assertEqual(saved[0]["body"], b"original bytes")
        self.assertIn("KeyboardInterrupt", errors["two"])
        self.assertFalse(any(t.name in ("serving-client-one", "serving-client-two")
                             for t in threading.enumerate()))

    def test_concurrent_interrupt_cancels_and_joins_before_return(self):
        entered, saved, errors = threading.Event(), [], {}
        actual_join, interrupted = threading.Thread.join, [False]
        def invoke(request, cancel):
            entered.set()
            self.assertTrue(cancel.wait(timeout=2), "caller never cancelled the client")
            return {"id": request["id"], "body": b"partial", "transport_error": {"kind": "cancelled"}}
        def interrupt_join(thread, *args, **kwargs):
            if thread.name == "serving-client-held" and not interrupted[0]:
                self.assertTrue(entered.wait(timeout=1))
                interrupted[0] = True
                raise KeyboardInterrupt("caller interrupted while client blocked")
            return actual_join(thread, *args, **kwargs)
        with patch.object(threading.Thread, "join", interrupt_join), self.assertRaises(KeyboardInterrupt):
            collect_clients([{"id": "held"}], "concurrent", invoke, saved.append, .5, errors)
        self.assertEqual(saved, [{"id": "held", "body": b"partial", "transport_error": {"kind": "cancelled"}}])
        self.assertFalse(any(t.name == "serving-client-held" for t in threading.enumerate()))

    def test_interruption_inside_thread_start_still_joins_native_worker(self):
        saved, errors = [], {}
        original_start = threading.Thread.start
        def invoke(request, cancel):
            self.assertTrue(cancel.wait(timeout=2))
            time.sleep(.03)  # Cleanup must await this worker, not just set cancel.
            return {"id": request["id"], "transport_error": {"kind": "cancelled"}, "body": b""}
        def interrupted_start(thread, *args, **kwargs):
            original_start(thread, *args, **kwargs)
            if thread.name == "serving-client-started":
                raise KeyboardInterrupt("after native thread start")
        with patch.object(threading.Thread, "start", interrupted_start), self.assertRaises(KeyboardInterrupt):
            collect_clients([{"id": "started"}], "concurrent", invoke, saved.append, .5, errors)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["transport_error"]["kind"], "cancelled")
        self.assertFalse(any(t.name == "serving-client-started" for t in threading.enumerate()))

    def test_genuine_launch_failure_keeps_previous_result_and_never_joins_unstarted(self):
        saved, errors = [], {}
        original_start, original_join = threading.Thread.start, threading.Thread.join
        def start(thread, *args, **kwargs):
            if thread.name == "serving-client-unstarted":
                raise RuntimeError("native thread allocation failure")
            return original_start(thread, *args, **kwargs)
        def join(thread, *args, **kwargs):
            self.assertNotEqual(thread.name, "serving-client-unstarted")
            return original_join(thread, *args, **kwargs)
        with patch.object(threading.Thread, "start", start), patch.object(threading.Thread, "join", join):
            with self.assertRaisesRegex(RuntimeError, "native thread allocation failure"):
                collect_clients([{"id": "one"}, {"id": "unstarted"}], "serial",
                                lambda r, _: {"id": r["id"], "body": b"first"}, saved.append, .5, errors)
        self.assertEqual(saved, [{"id": "one", "body": b"first"}])
        self.assertIn("unstarted", errors)


class MetricsWireFixtureTests(unittest.TestCase):
    def test_real_metrics_http_fixture_and_raw_store_on_any_host(self):
        # Execute the fixture handler, not its process launch/signal handler. This
        # proves actual HTTP framing and bytes locally, NOT Linux listener ownership.
        for mode in ("normal", "malformed", "non200", "truncated"):
            for bad_call in (1, 2):
                with self.subTest(mode=mode, call=bad_call), tempfile.TemporaryDirectory() as temporary:
                    events = []
                    namespace = {"print": lambda value, **_: events.append(json.loads(value.removeprefix("HTTP_EVENT ")))}
                    exec(SERVER.split("signal.signal(", 1)[0], namespace)
                    namespace.update(METRICS_MODE=mode, METRICS_BAD_CALL=bad_call)
                    with namespace["ThreadingHTTPServer"](("127.0.0.1", 0), namespace["Handler"]) as http:
                        thread = threading.Thread(target=http.serve_forever, kwargs={"poll_interval": .01},
                                                  name="metrics-wire-fixture")
                        thread.start()
                        try:
                            rows = []
                            for label, path, method, body in (
                                    ("before", "/metrics", "GET", b""),
                                    ("one", "/v1/chat/completions", "POST", b'{"model":"gate","stream":false}'),
                                    ("after", "/metrics", "GET", b"")):
                                rows.append(serving_capture.capture_request(request_id=label, path=path, method=method,
                                    port=http.server_port, body=body, headers={"X-Request-Id": label},
                                    connect_timeout=.5, read_timeout=1, wall_timeout=2, max_body_bytes=16384))
                        finally:
                            http.shutdown();thread.join(timeout=2)
                            self.assertFalse(thread.is_alive(), "metrics fixture did not stop")
                    self.assertEqual(events, [["GET", "/metrics", "before"],
                                             ["POST", "/v1/chat/completions", "one"], ["GET", "/metrics", "after"]])
                    self.assertIsNone(rows[1]["transport_error"])
                    selected = rows[0 if bad_call == 1 else 2]
                    if mode == "truncated":
                        self.assertEqual(selected["transport_error"]["kind"], "read")
                        self.assertIn("IncompleteRead", selected["transport_error"]["message"])
                    else:
                        self.assertIsNone(selected["transport_error"])
                    if mode == "non200":
                        self.assertEqual((selected["status"], selected["body"]),
                                         (503, b'{"error":{"code":"metrics_unavailable"}}'))
                    elif mode == "malformed":
                        self.assertEqual(selected["body"], b'{"prompt_tokens_in":')
                    else:
                        self.assertEqual(json.loads(rows[0]["body"])["prompt_tokens_in"], 0)
                        after = json.loads(rows[2]["body"])
                        self.assertEqual((after["prompt_tokens_in"], after["cached_tokens_in"],
                                          after["prefix_cache_hit_tokens"]), (4, 2, 0))
                    store = EvidenceStore(Path(temporary) / "raw")
                    for row in rows:
                        # HTTP header tuples become JSON arrays in the store;
                        # their names/values and the separate body bytes stay exact.
                        expected = {**row, "headers": [list(pair) for pair in row["headers"]]}
                        self.assertEqual(observation(store.root, store.observation(row)), expected)


@unittest.skipUnless(sys.platform == "linux", "real owned listener capture requires Linux procfs")
class LinuxCaptureTests(unittest.TestCase):
    def setup_plan(self, root):
        (root / "server.py").write_text(SERVER)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0));port = listener.getsockname()[1]
        return plan(root, port)

    def read_ref(self, output, ref):
        data = (output / ref["path"]).read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), ref["sha256"])
        return json.loads(data)

    def assert_metrics_brackets(self, output, group):
        before, after = [observation(output, group[key]) for key in ("health_before", "health_after")]
        metrics = [observation(output, group[key]) for key in ("metrics_before", "metrics_after")]
        attempts = [observation(output, ref) for ref in group["observations"]]
        left, right = [self.read_ref(output, group[key])["details"]
                       for key in ("listener_before", "listener_after")]
        self.assertLessEqual(left["finished_ns"], before["started_ns"])
        self.assertLessEqual(before["finished_ns"], metrics[0]["started_ns"])
        self.assertLessEqual(metrics[0]["finished_ns"], min(r["started_ns"] for r in attempts))
        self.assertLessEqual(max(r["finished_ns"] for r in attempts), metrics[1]["started_ns"])
        self.assertLessEqual(metrics[1]["finished_ns"], after["started_ns"])
        self.assertLessEqual(after["finished_ns"], right["started_ns"])
        ids = [r["id"] for r in [before, *metrics, *attempts, after]]
        self.assertEqual(len(ids), len(set(ids)))
        for row in metrics:
            self.assertEqual((row["method"], row["path"]), ("GET", "/metrics"))
            self.assertEqual(row["server_identity"], before["server_identity"])
            self.assertLessEqual(row["started_ns"], row["finished_ns"])
        return metrics

    def test_real_metrics_counters_are_raw_and_all_request_outcomes_remain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve();config = self.setup_plan(root)
            group = config["groups"][0];group.update(metrics=True, mode="concurrent")
            group["requests"] = [dict(group["requests"][0], id=f"parallel-{i}") for i in range(3)]
            group["requests"][1].update(wire="chat_sse", payload=dict(group["requests"][1]["payload"], stream=True))
            expected_identities = {name: file_identity(path) for name, path in config["identities"].items()}
            output = root / "capture";result = collect(config, output)
            self.assertEqual(result["state"], "captured", result["errors"])
            self.assertFalse(result["qualification"])
            captured = result["groups"][0]
            first, last = self.assert_metrics_brackets(output, captured)
            self.assertIsNone(first["transport_error"]);self.assertIsNone(last["transport_error"])
            self.assertEqual(json.loads(first["body"]), {"prompt_tokens_in": 0, "cached_tokens_in": 0,
                                                        "computed_tokens_in": 0, "prefix_cache_hit_tokens": 0})
            self.assertEqual(json.loads(last["body"]), {"prompt_tokens_in": 12, "cached_tokens_in": 6,
                                                       "computed_tokens_in": 6, "prefix_cache_hit_tokens": 0})
            accounting = self.read_ref(output, captured["wire_accounting"])
            self.assertEqual(accounting["attempted"], 3)
            self.assertEqual(accounting["counts"], {"clean_success": 2, "truncated_200": 1})
            self.assertEqual(len(self.read_ref(output, captured["generation_accounting"])["requests"]), 3)
            raw = (output / "capture.json").read_bytes()
            decoded = read_group_capture(raw, lambda path: (output / path).read_bytes(),
                expected_capture_sha256=hashlib.sha256(raw).hexdigest(), expected_plan=config,
                expected_identities=expected_identities)
            self.assertEqual(decoded["groups"][0]["metrics_samples"], [first, last])
            self.assertEqual(decoded["groups"][0]["accounting"]["counts"], accounting["counts"])
            self.assertFalse(decoded["qualification"])
            index = json.loads((output / "capture.json").read_text())
            for name, digest in index["payloads"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
            events = [json.loads(line.removeprefix("HTTP_EVENT ")) for line in
                      (output / "process/output.log").read_text().splitlines() if line.startswith("HTTP_EVENT ")]
            events = [row for row in events if not row[2].startswith("startup-health")]
            self.assertEqual(events[:2], [["GET", "/health", "serial-before"],
                                           ["GET", "/metrics", "serial-metrics-before"]])
            self.assertEqual({r[2] for r in events[2:5]}, {"parallel-0", "parallel-1", "parallel-2"})
            self.assertEqual(events[5:], [["GET", "/metrics", "serial-metrics-after"],
                                          ["GET", "/health", "serial-after"]])

    def test_real_absent_or_false_metrics_flag_adds_no_gets(self):
        for flag in (None, False):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve();config = self.setup_plan(root)
                if flag is not None: config["groups"][0]["metrics"] = flag
                output = root / "capture";result = collect(config, output)
                self.assertEqual(result["state"], "captured", result["errors"])
                self.assertNotIn("metrics_before", result["groups"][0])
                self.assertNotIn("metrics_after", result["groups"][0])
                events = [json.loads(line.removeprefix("HTTP_EVENT ")) for line in
                          (output / "process/output.log").read_text().splitlines() if line.startswith("HTTP_EVENT ")]
                self.assertEqual([r for r in events if r[0] == "GET"],
                                 [["GET", "/health", "startup-health"], ["GET", "/health", "serial-before"],
                                  ["GET", "/health", "serial-after"]])
                self.assertEqual([r for r in events if r[0] == "POST"], [["POST", "/v1/chat/completions", "one"]])

    def test_real_non200_and_invalid_metrics_json_remain_observable_not_qualified(self):
        for mode in ("non200", "malformed"):
            for call in (1, 2):
                with self.subTest(mode=mode, call=call), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary).resolve();config = self.setup_plan(root)
                    config["groups"][0]["metrics"] = True
                    (root / "server.py").write_text(SERVER.replace('METRICS_MODE = "normal"', f'METRICS_MODE = "{mode}"')
                                                   .replace('METRICS_BAD_CALL = 1', f'METRICS_BAD_CALL = {call}'))
                    expected_identities = {name: file_identity(path) for name, path in config["identities"].items()}
                    output = root / "capture";result = collect(config, output)
                    self.assertEqual(result["state"], "captured", result["errors"])
                    self.assertFalse(result["qualification"])
                    group = result["groups"][0]
                    row = self.assert_metrics_brackets(output, group)[call - 1]
                    self.assertIsNone(row["transport_error"])
                    self.assertEqual(row["status"], 503 if mode == "non200" else 200)
                    self.assertEqual(row["body"], b'{"error":{"code":"metrics_unavailable"}}'
                                     if mode == "non200" else b'{"prompt_tokens_in":')
                    self.assertEqual(self.read_ref(output, group["wire_accounting"])["attempted"], 1)
                    self.assertIn("generation_accounting", group)
                    raw = (output / "capture.json").read_bytes()
                    decoded = read_group_capture(raw, lambda path: (output / path).read_bytes(),
                        expected_capture_sha256=hashlib.sha256(raw).hexdigest(), expected_plan=config,
                        expected_identities=expected_identities)
                    self.assertEqual(decoded["groups"][0]["metrics_samples"][call - 1], row)
                    self.assertFalse(decoded["qualification"])

    def test_real_metrics_framing_failure_retains_both_probes_and_planned_requests(self):
        for call in (1, 2):
            with self.subTest(call=call), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve();config = self.setup_plan(root)
                config["groups"][0]["metrics"] = True
                (root / "server.py").write_text(SERVER.replace('METRICS_MODE = "normal"', 'METRICS_MODE = "truncated"')
                                               .replace('METRICS_BAD_CALL = 1', f'METRICS_BAD_CALL = {call}'))
                output = root / "capture";result = collect(config, output)
                self.assertEqual(result["state"], "failed", result)
                self.assertFalse(result["qualification"])
                group = result["groups"][0]
                self.assertEqual(group["state"], "failed")
                row = self.assert_metrics_brackets(output, group)[call - 1]
                self.assertEqual(row["transport_error"]["kind"], "read")
                self.assertIn("IncompleteRead", row["transport_error"]["message"])
                # The bytes parse as JSON, but the HTTP body was NOT complete.
                self.assertIsInstance(json.loads(row["body"]), dict)
                headers = {k.lower(): v for k, v in row["headers"]}
                self.assertEqual(int(headers["content-length"]), len(row["body"]) + 5)
                self.assertIn(row["id"], group["errors"])
                self.assertEqual(len(group["observations"]), 1)
                self.assertEqual(self.read_ref(output, group["wire_accounting"])["counts"], {"clean_success": 1})
                self.assertEqual(len(self.read_ref(output, group["generation_accounting"])["requests"]), 1)
                lifecycle = self.read_ref(output, result["lifecycle"])
                self.assertEqual(lifecycle["stop"]["reason"], "capture_error")
                self.assertEqual(lifecycle["server_exit"]["returncode"], 0)

    def test_real_serial_concurrent_and_truncated_client_all_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve();config = self.setup_plan(root)
            second = copy.deepcopy(config["groups"][0]);second.update(id="concurrent", mode="concurrent")
            second["requests"] = [dict(second["requests"][0], id=f"parallel-{i}") for i in range(3)]
            bad = second["requests"][1];bad["wire"] = "chat_sse";bad["payload"] = dict(bad["payload"], stream=True)
            config["groups"].append(second)
            expected_identities = {name: file_identity(path) for name, path in config["identities"].items()}
            output = root / "capture";result = collect(config, output)
            self.assertEqual(result["state"], "captured", result["errors"])
            self.assertFalse(result["qualification"])
            self.assertEqual(len(result["groups"]), 2)
            self.assertEqual(result["identities_before"], result["identities_after"])
            second_result = self.read_ref(output, result["groups"][1]["wire_accounting"])
            self.assertEqual(second_result["counts"], {"clean_success": 2, "truncated_200": 1})
            self.assertEqual(second_result["attempted"], 3)
            before = self.read_ref(output, result["groups"][1]["health_before"])
            self.assertEqual((before["method"], before["path"]), ("GET", "/health"))
            for ref in result["groups"][1]["observations"]:
                attempt = self.read_ref(output, ref)
                self.assertEqual(attempt["server_identity"], before["server_identity"])
                self.assertEqual((attempt["method"], attempt["path"]), ("POST", "/v1/chat/completions"))
            lifecycle = self.read_ref(output, result["lifecycle"])
            self.assertEqual(lifecycle["server_exit"]["returncode"], 0)
            self.assertTrue(lifecycle["cleanup"]["complete"])
            index = json.loads((output / "capture.json").read_text())
            for name, digest in index["payloads"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
            raw = (output / "capture.json").read_bytes()
            decoded = read_group_capture(raw, lambda path: (output / path).read_bytes(),
                expected_capture_sha256=hashlib.sha256(raw).hexdigest(),
                expected_plan=config, expected_identities=expected_identities)
            self.assertFalse(decoded["qualification"])
            self.assertEqual(decoded["groups"][1]["accounting"]["counts"], second_result["counts"])

    def test_foreign_port_never_becomes_our_readiness(self):
        with tempfile.TemporaryDirectory() as temporary, socket.socket() as foreign:
            root = Path(temporary).resolve();config = self.setup_plan(root)
            foreign.bind(("127.0.0.1", config["server"]["port"]));foreign.listen();foreign.settimeout(.1)
            output = root / "capture";result = collect(config, output)
            self.assertEqual(result["state"], "failed")
            self.assertFalse(result["qualification"])
            self.assertEqual(result["groups"], [])
            self.assertNotIn("startup_listener", result)
            with self.assertRaises(TimeoutError):
                foreign.accept()

    def test_truncated_health_preserves_wire_rows_without_generation_timings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve();config = self.setup_plan(root)
            code = SERVER.replace('class Handler(BaseHTTPRequestHandler):',
                                  'class Handler(BaseHTTPRequestHandler):\n    health_calls = 0')
            code = code.replace('self.reply(json.dumps(value).encode())\n    def reply',
                'Handler.health_calls += 1\n        data=json.dumps(value).encode()\n'
                '        self.send_response(200)\n'
                '        truncated = self.headers.get("X-Request-Id", "") != "startup-health"\n'
                '        self.send_header("Content-Length",str(len(data)+(1 if truncated else 0)))\n'
                '        self.end_headers();self.wfile.write(data)\n    def reply', 1)
            (root / "server.py").write_text(code)
            output = root / "capture";result = collect(config, output)
            self.assertEqual(result["state"], "failed", result)
            self.assertTrue(result["groups"], result["errors"])
            group = result["groups"][0]
            self.assertEqual(len(group["observations"]), 1)
            self.assertNotIn("generation_accounting", group)
            self.assertEqual(self.read_ref(output, group["wire_accounting"])["counts"], {"clean_success": 1})
            for key in ("health_before", "health_after"):
                self.assertIsNotNone(self.read_ref(output, group[key])["transport_error"])

    def test_real_deadline_cannot_hide_behind_clean_sigterm_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve();config = self.setup_plan(root)
            config["server"]["overall_timeout"] = 6
            output, instances = root / "capture", []
            original_server, original_proof = serving_capture.OwnedServer, serving_capture.prove_listener
            class ObservedServer(original_server):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs);instances.append(self)
            count = [0]
            def delayed_last_proof(*args, **kwargs):
                proof = original_proof(*args, **kwargs);count[0] += 1
                if count[0] == 4:
                    deadline = time.monotonic() + 8
                    while time.monotonic() < deadline and instances[0].receipt().get("state") != "finished":
                        time.sleep(.02)
                    self.assertEqual(instances[0].receipt()["state"], "finished")
                return proof
            with patch.object(serving_capture, "OwnedServer", ObservedServer), \
                 patch.object(serving_capture, "prove_listener", delayed_last_proof):
                result = collect(config, output)
            self.assertEqual(result["state"], "failed")
            self.assertTrue(any("overall_timeout" in error for error in result["errors"]), result["errors"])
            lifecycle = self.read_ref(output, result["lifecycle"])
            self.assertEqual(lifecycle["server_exit"]["returncode"], 0)
            self.assertEqual(lifecycle["stop"]["reason"], "overall_timeout")

    def test_interrupted_group_cannot_send_after_server_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve();config = self.setup_plan(root)
            config["groups"][0]["mode"] = "concurrent"
            (root / "server.py").write_text(SERVER.replace('def do_POST(self):',
                'def do_POST(self):\n        print("POST_RECEIVED",flush=True)'))
            entered, interrupted = threading.Event(), [False]
            original_capture, original_join = serving_capture.capture_request, threading.Thread.join
            def delayed_capture(**kwargs):
                if kwargs["request_id"] == "one":
                    entered.set()
                    self.assertTrue(kwargs["cancel_event"].wait(timeout=2))
                return original_capture(**kwargs)
            def interrupt_join(thread, *args, **kwargs):
                if thread.name == "serving-client-one" and not interrupted[0]:
                    self.assertTrue(entered.wait(timeout=1));interrupted[0] = True
                    raise KeyboardInterrupt("interrupted before client connect")
                return original_join(thread, *args, **kwargs)
            output = root / "capture"
            with patch.object(serving_capture, "capture_request", delayed_capture), \
                 patch.object(threading.Thread, "join", interrupt_join):
                result = collect(config, output)
            self.assertEqual(result["state"], "failed")
            rows = result["groups"][0]["observations"]
            self.assertEqual(len(rows), 1)
            self.assertEqual(self.read_ref(output, rows[0])["transport_error"]["kind"], "cancelled")
            self.assertNotIn("POST_RECEIVED", (output / "process/output.log").read_text())
            self.assertFalse(any(t.name in ("serving-client-one", "serving-http-one")
                                 for t in threading.enumerate()))


if __name__ == "__main__":
    unittest.main()
