"""Actual local socket/process controls for the two drain observation seams."""

import copy
import base64
import shutil
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from serving_http import capture_request
from serving_process import ProcessIdentity, _signal_process, process_identity
from test_serving_http import peer


class HttpObserverTests(unittest.TestCase):
    def capture(self, port, **kwargs):
        events = []
        original = kwargs.get("observer")
        if original is not None:
            def observe(value):
                event = dict(value)
                if "data" in event:
                    event["data_base64"] = base64.b64encode(event.pop("data")).decode()
                events.append(event)
                original(value)
            kwargs["observer"] = observe
        result = capture_request(request_id="observed", port=port, path="/v1/chat/completions",
                                 body=b"{}", read_timeout=.5, wall_timeout=.7, **kwargs)
        evidence = os.environ.get("MEMRA_DRAIN_TEST_EVIDENCE_DIR")
        if evidence:
            target = Path(tempfile.mkdtemp(prefix="http-observer-",dir=evidence))
            (target/"result.json").write_text(json.dumps({"scope":"actual CPU loopback observer control",
                "events":events,"result":{**result,"body_base64":base64.b64encode(result["body"]).decode(),"body":None}},indent=2))
        return result

    def test_immutable_actual_endpoints_headers_and_first_prefix(self):
        events = []
        def observe(value):
            with self.assertRaises(TypeError): value["id"] = "changed"
            if value["event"] == "headers":
                self.assertIsInstance(value["headers"], tuple)
                with self.assertRaises(TypeError): value["headers"][0][0] = "changed"
            events.append(value)
        with peer(lambda sock: sock.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nX-Test: yes\r\n\r\nhello")) as port:
            result = self.capture(port, observer=observe)
            self.assertEqual(events[0]["peer"], ("127.0.0.1", port))
        self.assertIsNone(result["transport_error"])
        self.assertEqual([e["event"] for e in events], ["connected", "headers", "first_body"])
        self.assertEqual(events[-1]["data"], result["body"])
        self.assertEqual(events[-1]["observed_ns"], result["first_body_byte_ns"])
        self.assertEqual(events[-1]["end_offset"], len(result["body"]))
        self.assertEqual(tuple(result["headers"]), events[1]["headers"])
        self.assertEqual(set(result), {"id","started_ns","finished_ns","status","headers","body",
                                     "chunks","first_body_byte_ns","transport_error"})
        count = len(events); frozen = copy.deepcopy(result)
        time.sleep(.06)
        self.assertEqual((len(events),result), (count,frozen))
        self.assertFalse(any(t.name == "serving-http-observed" for t in threading.enumerate()))

    def test_exception_at_each_stage_closes_socket_and_retains_available_bytes(self):
        for stage in ("connected", "headers", "first_body"):
            events = []
            def observe(value):
                events.append(value["event"])
                if value["event"] == stage: raise RuntimeError("observer fault")
            with self.subTest(stage=stage), peer(lambda sock: sock.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello")) as port:
                result = self.capture(port, observer=observe)
            self.assertEqual(result["transport_error"]["kind"], "observer")
            self.assertIn("observer fault", result["transport_error"]["message"])
            self.assertEqual(events[-1], stage)
            self.assertEqual(result["body"], b"hello" if stage == "first_body" else b"")
            self.assertFalse(any(t.name == "serving-http-observed" for t in threading.enumerate()))

    def test_cancel_from_observer_does_not_wait_for_watchdog_or_send_late_request(self):
        cancelled = threading.Event()
        events = []
        def observe(value):
            events.append(value["event"]); cancelled.set()
        with peer(lambda sock: self.fail("request sent after connected observer cancelled")) as port:
            result = self.capture(port, observer=observe, cancel_event=cancelled)
        self.assertEqual(events, ["connected"])
        self.assertEqual(result["transport_error"]["kind"], "cancelled")
        self.assertIsNone(result["status"])

    def test_slow_callback_cannot_turn_expired_capture_into_success(self):
        def observe(value):
            if value["event"] == "connected": time.sleep(.12)
        with peer(lambda sock: self.fail("request sent after observer consumed wall deadline")) as port:
            result = capture_request(request_id="observer-deadline", port=port, path="/", body=b"",
                                     wall_timeout=.05, observer=observe)
        self.assertEqual(result["transport_error"]["kind"], "timeout")
        self.assertIsNone(result["status"])

    def test_observer_failure_remains_distinct_when_cancellation_also_happens(self):
        cancelled = threading.Event()
        def observe(value):
            if value["event"] == "first_body":
                cancelled.set(); raise ValueError("observer refused retained prefix")
        with peer(lambda sock: sock.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello")) as port:
            result = self.capture(port, observer=observe, cancel_event=cancelled)
        self.assertEqual(result["body"], b"hello")
        self.assertEqual(result["transport_error"]["kind"], "observer")
        self.assertIn("cancelled", result["transport_error"]["message"])

    def test_first_prefix_is_bounded_even_when_wire_exceeds_cap(self):
        events = []
        with peer(lambda sock: sock.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 12\r\n\r\nabcdefghijkl")) as port:
            result = self.capture(port, observer=events.append, max_body_bytes=8)
        self.assertEqual(events[-1]["data"], b"abcdefgh")
        self.assertEqual(result["body"], b"abcdefgh")
        self.assertEqual(result["transport_error"]["kind"], "read")


SUPERVISOR = r'''
import json,sys,time
sys.path.insert(0,sys.argv[1])
import serving_process as p
mode=sys.argv[2]
original_signal=p._signal_process
def signal_attempt(owner,number):
    if mode=='signal_error' and number==15:
        now=time.monotonic()
        return dict(pid=owner.pid,start_time=owner.start_time,signal=15,result='error',
                    error='injected signal refusal',monotonic=now,finished_monotonic=time.monotonic())
    return original_signal(owner,number)
p._signal_process=signal_attempt
original_publish=p._atomic_json
failed=False
def publish(path,value):
    global failed
    if mode=='publication_error' and value.get('signals') and not failed:
        failed=True
        raise OSError('injected signal receipt publication failure')
    original_publish(path,value)
p._atomic_json=publish
p._supervise(json.loads(sys.stdin.buffer.readline()))
'''

SERVER = r'''
import signal,sys,time
mode=sys.argv[1]
def term(*_):
    print('TERM_RECEIVED',flush=True)
    if mode!='immediate':time.sleep(.45)
    raise SystemExit(0)
signal.signal(signal.SIGTERM,term)
print('READY',flush=True)
if mode=='already_reaped':
    time.sleep(.15)
    raise SystemExit(7)
while True:time.sleep(.01)
'''


class SignalPublicationTests(unittest.TestCase):
    def run_case(self, mode):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); evidence = root/"process"; evidence.mkdir()
            (root/"supervise.py").write_text(SUPERVISOR); (root/"server.py").write_text(SERVER)
            config = {"argv":[sys.executable,str(root/"server.py"),mode],"env":{},"cwd":str(root),
                      "evidence_dir":str(evidence),"timeouts":{"startup":2,"overall":5,"drain":1,"kill":1}}
            with (root/"supervisor.log").open("wb") as output:
                sup = subprocess.Popen([sys.executable,str(root/"supervise.py"),str(Path(__file__).parent),mode],
                    stdin=subprocess.PIPE,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                def send(value):
                    sup.stdin.write((json.dumps(value)+"\n").encode());sup.stdin.flush()
                def receipt():
                    try:return json.loads((evidence/"receipt.json").read_text())
                    except FileNotFoundError:return {}
                observed = None
                try:
                    send(config); limit=time.monotonic()+3
                    while time.monotonic()<limit:
                        current=receipt()
                        if current.get('server') and (evidence/'output.log').exists() and 'READY' in (evidence/'output.log').read_text():break
                        time.sleep(.01)
                    self.assertTrue(current.get('server'))
                    if mode != 'already_reaped':
                        send({'op':'ready','proof':{'owner':current['server'],'method':'owned_output','details':{'fixture':True}}})
                        send({'op':'stop','reason':'drain_cell'})
                        while time.monotonic()<limit:
                            current=receipt()
                            if current.get('signals'):
                                observed=current;break
                            time.sleep(.005)
                    else:
                        while time.monotonic()<limit and receipt().get('server_exit') is None:
                            time.sleep(.01)
                    sup.stdin.close();sup.wait(timeout=6)
                    final=receipt()
                    self.assertEqual(final['state'],'finished')
                    return observed,final
                finally:
                    if sup.poll() is None:
                        current=receipt(); owner=current.get('server')
                        if owner:
                            actual=process_identity(owner['pid'])
                            if actual and actual.start_time==owner['start_time']:_signal_process(actual,signal.SIGKILL)
                        sup.terminate();sup.wait(timeout=3)
                    evidence_root=os.environ.get("MEMRA_DRAIN_TEST_EVIDENCE_DIR")
                    if evidence_root:
                        target=Path(tempfile.mkdtemp(prefix="signal-"+mode+"-",dir=evidence_root))
                        (root/"observed-live.json").write_text(json.dumps(observed,indent=2))
                        shutil.copytree(root,target/"fixture")

    def test_post_syscall_record_is_published_while_primary_still_drains(self):
        observed,final=self.run_case('held')
        self.assertIsNotNone(observed)
        self.assertIsNone(observed['server_exit'])
        self.assertEqual(observed['signals'][0]['result'],'sent')
        self.assertLessEqual(observed['signals'][0]['sent_monotonic'],observed['signals'][0]['finished_monotonic'])
        self.assertEqual(final['server_exit']['returncode'],0)
        self.assertTrue(final['cleanup']['complete'])

    def test_immediate_exit_preserves_the_actual_signal_and_original_exit(self):
        observed,final=self.run_case('immediate')
        self.assertIsNotNone(observed)
        self.assertEqual(final['server_exit']['returncode'],0)
        self.assertTrue(any(s['signal']==15 and s['result']=='sent' for s in final['signals']))

    def test_signal_error_is_observed_and_never_becomes_clean_drain(self):
        observed,final=self.run_case('signal_error')
        self.assertEqual(observed['signals'][0]['result'],'error')
        self.assertNotIn('sent_monotonic',observed['signals'][0])
        self.assertTrue(final['errors'])
        self.assertFalse(final['cleanup']['complete'])

    def test_signal_publication_failure_uses_existing_error_cleanup(self):
        _,final=self.run_case('publication_error')
        self.assertTrue(any('publication failure' in error for error in final['errors']))
        self.assertFalse(final['cleanup']['complete'])
        self.assertEqual(final['signals'][0]['result'],'sent')

    def test_already_reaped_primary_has_no_invented_sent_signal(self):
        _,final=self.run_case('already_reaped')
        self.assertEqual(final['server_exit']['returncode'],7)
        self.assertFalse(any(s.get('signal')==15 and s.get('result')=='sent' for s in final['signals']))


if __name__ == '__main__':unittest.main()
