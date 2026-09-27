"""CPU protocol tests. Synthetic phase records are not native phase/GPU proof."""

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

import serving_cancel_phase as phase
import serving_http
from serving_cancel_phase import collect_phase_cancel_cell, validate_phase_program, _PinnedLog, _decode_prefix, _target_prefix, _prospective_phase
from serving_process import OwnedServer, ReadinessEvidence
from serving_release import ServingGateError
from serving_trace import validate_cancel_trace
from test_serving_cancel import config, success, typed_error
from test_serving_trace import Script, fixture as trace_fixture, wire


KEY = "0123456789abcdef0123456789abcdef"


def plan(scenario="cancel_decode"):
    required, old = config(scenario)
    return required, {"schema": "memra-phase-cancel-program-v1", "mode": "phase_cancel",
        "cell_id": required["id"], "scope": required["scope"], "server_identity": old["server_identity"],
        "client_trace_key": KEY, "trace": {"worker_generation": 7, "worker_route": "shared_gpu_worker",
                                           "quantum_routes": ["target_prime", "draft_fill"]},
        "endpoint": {"host": "127.0.0.1", "port": 12345},
        "http": {"connect_timeout": .5, "read_timeout": 2, "wall_timeout": 4, "max_body_bytes": 16384},
        "timing": {"trigger_timeout_s": 1.5, "retirement_timeout_s": 2, "poll_s": .02, "overall_s": 12},
        "requests": old["requests"]}


class PhaseInputTests(unittest.TestCase):
    def test_frozen_schema_has_no_timer_or_callback_waiver(self):
        required, program = plan(); before = copy.deepcopy(program)
        self.assertEqual(set(validate_phase_program(required, program)), {"target", "peer", "recovery"})
        self.assertEqual(program, before)
        for name, value in (("cancel_after_s", .1), ("phase_callback", "trust_me"), ("skip", True)):
            bad = copy.deepcopy(program); bad[name] = value
            with self.assertRaises(ServingGateError): validate_phase_program(required, bad)
        for field in program["timing"]:
            for value in (0, -1, True, float("nan"), float("inf")):
                bad = copy.deepcopy(program); bad["timing"][field] = value
                with self.assertRaises(ServingGateError): validate_phase_program(required, bad)
        bad = copy.deepcopy(program); bad["requests"][0]["id"] = "cancel-probe-1"
        with self.assertRaises(ServingGateError): validate_phase_program(required, bad)

    def test_pinned_log_retains_partial_lines_and_detects_prefix_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "output.log"; path.write_bytes(b"startup\npartial")
            reader = _PinnedLog(path)
            try:
                meta, offset, delta = reader.read()
                self.assertEqual((offset, delta, meta["complete_bytes"]), (0, b"startup\npartial", 8))
                self.assertIsNone(_decode_prefix(reader.raw))
                with path.open("ab") as f: f.write(b"-line\n")
                meta, offset, delta = reader.read()
                self.assertEqual((offset, delta), (15, b"-line\n"))
                path.write_bytes(b"STARTUP\npartial-line\n")
                with self.assertRaisesRegex(ServingGateError, "prefix mutated") as caught: reader.read()
                self.assertEqual(caught.exception.observed, path.read_bytes())
                self.assertEqual(reader.raw, b"startup\npartial-line\n")
            finally: reader.close()

    def test_missing_replaced_truncated_and_symlink_logs_refuse(self):
        for change in ("missing", "replacement", "truncation", "symlink"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "output.log"; path.write_bytes(b"original\n")
                reader = _PinnedLog(path)
                try:
                    reader.read()
                    if change == "missing": path.unlink()
                    elif change == "replacement":
                        other = path.with_name("new"); other.write_bytes(b"original\n"); os.replace(other, path)
                    elif change == "symlink":
                        other = path.with_name("new"); path.rename(other); path.symlink_to(other)
                    else: path.write_bytes(b"orig")
                    with self.assertRaises(ServingGateError): reader.read()
                finally: reader.close()

    def test_actual_append_after_pread_then_observed_size_shrink_refuses(self):
        # No fstat mock: pread returns the real old bytes, then another writer
        # appends real bytes before the reader's post-read stat observation.
        for truncate in (False, True):
            with self.subTest(truncate=truncate), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "output.log"; path.write_bytes(b"original\n")
                reader = _PinnedLog(path)
                original_pread = os.pread
                def append_after_pread(*args):
                    data = original_pread(*args)
                    with path.open("ab") as output: output.write(b"observed append bytes\n!")
                    return data
                try:
                    with patch.object(os, "pread", append_after_pread):
                        first, _, _ = reader.read()
                    self.assertEqual(first["size_before"], 9)
                    self.assertEqual(first["size_after"], path.stat().st_size)
                    self.assertGreater(first["size_after"], len(reader.raw))
                    if truncate:
                        path.write_bytes(b"original\n")
                        with self.assertRaisesRegex(ServingGateError, "below observed size") as caught:
                            reader.read()
                        self.assertEqual(caught.exception.metadata["size_before"], 9)
                    else:
                        second, offset, delta = reader.read()
                        self.assertEqual(second["size_before"], first["size_after"])
                        self.assertEqual(offset, 9)
                        self.assertEqual(delta, b"observed append bytes\n!")
                finally: reader.close()

    def test_shrink_since_open_and_within_actual_read_refuses(self):
        for when in ("since_open", "during_read"):
            with self.subTest(when=when), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "output.log"; path.write_bytes(b"original\n")
                reader = _PinnedLog(path); original_pread = os.pread
                def truncate_after_pread(*args):
                    data = original_pread(*args)
                    path.write_bytes(b"orig")
                    return data
                try:
                    if when == "since_open":
                        path.write_bytes(b"orig")
                        with self.assertRaisesRegex(ServingGateError, "below observed size"): reader.read()
                    else:
                        with patch.object(os, "pread", truncate_after_pread):
                            with self.assertRaisesRegex(ServingGateError, "below observed size"): reader.read()
                finally: reader.close()

    def test_no_record_yet_is_distinct_from_reviewed_framing_errors(self):
        self.assertIsNone(_decode_prefix(b"ordinary\npartial"))
        self.assertIsNone(_decode_prefix(b"[request-lifecycle] {"))
        for data in (b"prefix [request-lifecycle] {\n", b"[request-lifecycle] {\n",
                     b'{"\\u0073chema":"memra-request-lifecycle-v1",bad\n'):
            with self.assertRaises(ServingGateError): _decode_prefix(data)

    def test_prospective_phase_needs_latest_matching_observation_not_global_health(self):
        for scenario, marker in (("cancel_queued", "queued"), ("cancel_prime", "prime_quantum_start"),
                                  ("cancel_decode", "decode_start")):
            required, program = plan(scenario)
            rows = trace_fixture(scenario); index = next(i for i,r in enumerate(rows) if r["event"] == marker)
            selected = _target_prefix(_decode_prefix(wire(rows[:index+1])), required, program)
            self.assertEqual(_prospective_phase(selected, scenario)["record"]["event"], marker)
            self.assertIsNone(_prospective_phase(selected[:-1], scenario))
        required, program = plan("cancel_prime"); rows = trace_fixture()
        index = next(i for i,r in enumerate(rows) if r["event"] == "prime_quantum_end")
        # Remove the HTTP drop to represent a between-quantum snapshot; syntax-only trigger must decline it.
        rows = [r for r in rows[:index+1] if r["event"] != "http_body_drop"]
        self.assertIsNone(_prospective_phase(_target_prefix(_decode_prefix(wire(rows)), required, program), "cancel_prime"))

    def test_false_syntactic_candidate_still_fails_complete_parser(self):
        required, program = plan("cancel_prime"); rows = trace_fixture()
        rows.pop(3)  # Worker binding event omitted; later snapshots falsely claim bound.
        prefix = rows[:next(i for i,r in enumerate(rows) if r["event"] == "prime_quantum_start")+1]
        self.assertIsNotNone(_prospective_phase(_target_prefix(_decode_prefix(wire(prefix)), required, program), "cancel_prime"))
        with self.assertRaises(ServingGateError):
            validate_cancel_trace(wire(rows), scenario="cancel_prime", client_trace_key=KEY,
                expected_server_identity=program["server_identity"], expected_model="gate",
                expected_http_route="/v1/chat/completions", expected_worker_route="shared_gpu_worker",
                expected_worker_generation=7, expected_quantum_routes=["target_prime"])

    def test_foreign_binding_or_ambiguous_key_is_not_a_wait_condition(self):
        required, program = plan(); rows = trace_fixture("cancel_decode")[:6]
        for field, value in (("pid", 18), ("model", "other"), ("worker_generation", 8), ("worker_route", "other")):
            bad = copy.deepcopy(rows)
            for row in bad:
                if row[field] is not None: row[field] = value
            with self.assertRaises(ServingGateError): _target_prefix(_decode_prefix(wire(bad)), required, program)
        other = copy.deepcopy(rows)
        for row in other: row["trace_id"] = 2
        with self.assertRaisesRegex(ServingGateError, "ambiguous"):
            _target_prefix(_decode_prefix(wire(rows+other)), required, program)


# The subprocess emits SYNTHETIC protocol state. No Memra phase/model work runs.
SERVER = r'''
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json,os,select,signal,socket,sys,threading,time
from pathlib import Path
sys.path.insert(0,sys.argv[3])
from test_serving_trace import Script
mode,scenario=sys.argv[2],sys.argv[4]
output_lock=threading.Lock()
def log(value):
    with output_lock: print(json.dumps(value),flush=True)
def emit(row):
    with output_lock: os.write(1,b'[request-lifecycle] '+json.dumps(row,separators=(',',':')).encode()+b'\n')
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def do_GET(self):
        data=b'{"status":"ok","models":["gate"],"worker":{"generation":7}}'
        self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def do_POST(self):
        p=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        role=p['fixture_role']; label=self.headers['X-Request-Id']; key=self.headers.get('x-memra-trace-id')
        log(dict(event='HTTP',role=role,key=key,id=label))
        if role=='peer':
            Path('peer-started').touch()
            end=time.monotonic()+6
            while not Path('target-finished').exists() and time.monotonic()<end: time.sleep(.005)
        if role!='target':
            data=bytes.fromhex(p['fixture_response'])
            self.send_response(200);self.send_header('x-request-id','minted-'+label)
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);return
        end=time.monotonic()+4
        while not Path('peer-started').exists() and time.monotonic()<end: time.sleep(.005)
        s=Script(pid=os.getpid());s.bind(key,'minted-target')
        if mode=='foreign_model':
            for row in s.rows:
                if row['model'] is not None:row['model']='other'
        for row in s.rows:
            if mode=='false_prefix' and row['event']=='worker_bound':continue
            if mode=='no_phase' and row['event']=='queued':continue
            emit(row)
        def add(event,**kwargs):s.add(event,**kwargs);emit(s.rows[-1])
        if scenario!='cancel_queued':
            self.send_response(200);self.send_header('x-request-id','minted-target');self.end_headers()
        if mode=='fragmented_error':
            data=bytes.fromhex(p['fixture_error'])
            for piece in (data[:11],data[11:37],data[37:]):self.wfile.write(piece);self.wfile.flush();time.sleep(.01)
            end=time.monotonic()+4
            while not Path('error-observed').exists() and time.monotonic()<end:time.sleep(.005)
            if not Path('error-observed').exists():raise RuntimeError('error observer barrier expired')
        if mode!='no_phase':
            if scenario=='cancel_prime':s.quantum_start();emit(s.rows[-1])
            elif scenario=='cancel_decode':add('decode_start',phase='decode')
        if mode=='fast_eof':
            data=bytes.fromhex(p['fixture_response']);self.wfile.write(data);self.wfile.flush()
            add('http_body_eof',http_body='normal_eof');add('http_body_drop',http_body='dropped_after_eof')
            add('trace_end',sequence_valid=False,first_error='trace_ended_without_explicit_retirement')
            Path('target-finished').touch();return
        if scenario!='cancel_queued' and mode!='fragmented_error':self.wfile.write(b'data: {');self.wfile.flush()
        end=time.monotonic()+6
        while time.monotonic()<end:
            if select.select([self.connection],[],[],.01)[0] and self.connection.recv(1,socket.MSG_PEEK)==b'':break
        if mode=='stale_at_drop' and scenario=='cancel_prime':s.quantum_end();emit(s.rows[-1])
        add('http_pending_drop' if scenario=='cancel_queued' else 'http_body_drop',
            http_body='pending_dropped' if scenario=='cancel_queued' else 'dropped_before_eof')
        if scenario=='cancel_prime' and mode not in ('stale_at_drop','no_phase'):
            s.quantum_end(remaining=None,completed=mode!='failed_quantum');emit(s.rows[-1])
        add('receiver_closed',receiver_close_cause='receiver_dropped',observed_close_cause='receiver_dropped')
        if mode!='missing_retirement':
            add('retired',retirement='aborted',retirement_site='WorkerQueue' if scenario=='cancel_queued' else 'ActiveSession')
            add('trace_end')
        Path('target-finished').touch()
signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))
if mode=='preexisting_key':
    previous=Script(pid=os.getpid());previous.bind('0123456789abcdef0123456789abcdef','minted-previous')
    for row in previous.rows:emit(row)
with ThreadingHTTPServer(('127.0.0.1',int(sys.argv[1])),Handler) as server:
    print('SYNTHETIC_PHASE_PRODUCER_READY',flush=True)
    if mode=='partial_before_launch':os.write(1,b'[request-lifecycle] {"schema":')
    server.serve_forever(poll_interval=.01)
'''


def obj(root, ref):
    data=(root/ref['path']).read_bytes()
    if hashlib.sha256(data).hexdigest()!=ref['sha256']:raise AssertionError('blob hash mismatch')
    return json.loads(data)


class PhaseProcessTests(unittest.TestCase):
    def run_case(self, scenario="cancel_decode", mode="normal", *, real_listener=False, interrupt=False, long_poll=False, inside_set=False, abort_poll=False):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();script=root/'fixture.py'
            source = SERVER
            if inside_set:
                source = source.replace("not Path('target-finished').exists()", "not Path('release-peer').exists()")
            script.write_text(source)
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            class FixtureServer(OwnedServer):
                def receipt(self):
                    value=super().receipt()
                    return value if sys.platform=='linux' else {**value,'boot_id':'fixture-only-no-linux-boot-id'}
            server=FixtureServer(argv=[sys.executable,str(script),str(port),mode,str(Path(__file__).parent),scenario],
                env={},cwd=str(root),evidence_dir=str(root/'process'),startup_timeout=5,overall_timeout=20,drain_timeout=2,kill_timeout=1)
            try:
                server.start();until=time.monotonic()+4
                while time.monotonic()<until and (not server.output_path.exists() or 'SYNTHETIC_PHASE_PRODUCER_READY' not in server.output_path.read_text()):time.sleep(.01)
                self.assertIn('SYNTHETIC_PHASE_PRODUCER_READY',server.output_path.read_text())
                server.mark_ready(ReadinessEvidence(server.identity,'owned_output',{'synthetic_fixture':True}))
                required,program=plan(scenario);program['endpoint']['port']=port
                program['server_identity']={'pid':server.identity.pid,'start_identity':server.receipt()['boot_id']+':'+server.identity.start_time}
                for request in program['requests']:
                    request['payload']['fixture_response']=success(request['wire']).hex()
                    request['payload']['fixture_error']=typed_error(request['wire']).hex()
                if long_poll:
                    program['http']['wall_timeout'] = .5
                    program['timing'] = {'trigger_timeout_s':9, 'retirement_timeout_s':9, 'poll_s':8, 'overall_s':25}
                output=root/'capture'
                scheduled, abort_timers, aborted = [], [], threading.Event()
                source = Path(phase.__file__).read_text().splitlines()
                read_line = next(i for i,l in enumerate(source,1) if l.strip() == 'metadata, decoded = read_log()' and i > 350)
                after_line = next(i for i,l in enumerate(source,1) if l.strip() == 'after = time.monotonic_ns()')
                poll_line = next(i for i,l in enumerate(source,1) if l.strip() == 'wait_poll(group_cancel, limit, done)')
                def schedule(frame, event, arg):
                    if event == 'line' and frame.f_code.co_name == 'invoke' and Path(frame.f_code.co_filename).name == 'serving_cancel_phase.py':
                        if long_poll and frame.f_lineno == read_line and not scheduled:
                            until = time.monotonic() + .3
                            while time.monotonic() < until and b'"event":"decode_start"' not in server.output_path.read_bytes(): time.sleep(.001)
                            if b'"event":"decode_start"' not in server.output_path.read_bytes(): raise RuntimeError('phase barrier not observed')
                            scheduled.append(time.monotonic_ns())
                        if inside_set and frame.f_lineno == after_line and not scheduled:
                            if not frame.f_locals['done'].wait(3): raise RuntimeError('actual HTTP cancellation did not finish')
                            scheduled.append(time.monotonic_ns())
                            (root/'release-peer').touch()
                        if abort_poll and frame.f_lineno == poll_line and not abort_timers:
                            cancel = frame.f_locals['group_cancel']
                            def interrupt_group():
                                cancel.set(); aborted.set()
                            timer = threading.Timer(.1, interrupt_group)
                            abort_timers.append(timer); timer.start()
                    return schedule
                with ExitStack() as stack:
                    if not real_listener:
                        stack.enter_context(patch.object(phase,'prove_listener',return_value=ReadinessEvidence(server.identity,'listener_identity',{'synthetic_fixture_only':True})))
                    if interrupt:
                        start=threading.Thread.start
                        def launch(thread):
                            start(thread)
                            if thread.name=='phase-http-target':raise KeyboardInterrupt('controlled native target-thread launch interruption')
                        stack.enter_context(patch.object(threading.Thread,'start',launch))
                    if mode=='fragmented_error':
                        original=serving_http._StrictHTTPResponse.read1;received=[0]
                        def read(response,*args,**kwargs):
                            if response.getheader('x-request-id')=='minted-target' and received[0]==len(typed_error('chat_sse')):
                                (root/'error-observed').touch()
                            data=original(response,*args,**kwargs)
                            if response.getheader('x-request-id')=='minted-target':received[0]+=len(data)
                            return data
                        stack.enter_context(patch.object(serving_http._StrictHTTPResponse,'read1',read))
                    if long_poll or inside_set: threading.settrace(schedule)
                    began = time.monotonic()
                    try: result=collect_phase_cancel_cell(required,program,server=server,output=output)
                    finally:
                        threading.settrace(None)
                        for timer in abort_timers: timer.join(timeout=1)
                    if abort_poll:
                        self.assertTrue(aborted.is_set())
                        self.assertTrue(all(not timer.is_alive() for timer in abort_timers))
                        # Operational cancellation bound, not a performance claim:
                        # an 8-second poll must not hold the watcher after abort.
                        self.assertLess(time.monotonic()-began, 3)
                    if long_poll or inside_set: self.assertEqual(len(scheduled),1)
                sealed = (output/'capture.json').read_bytes()
                self.assertFalse(any(t.name.startswith(('serving-client-','serving-http-','phase-http-')) for t in threading.enumerate()))
                time.sleep(.06)
                self.assertEqual((output/'capture.json').read_bytes(),sealed)
                self.assertFalse(result['qualification']);self.assertIsNone(server.receipt().get('stop'))
                index=json.loads((output/'capture.json').read_text())
                for name,digest in index['payloads'].items():self.assertEqual(hashlib.sha256((output/name).read_bytes()).hexdigest(),digest)
                rows=[]
                for ref in result['observations']:
                    row=obj(output,ref);row['body']=(output/row['body']['path']).read_bytes();rows.append(row)
                self.assertEqual(len(result['request_denominator']),3)
                self.assertEqual(result['captured_attempts']+len(result['unattempted_ids'])+len(result['unobserved_invoked_ids']),3)
                final_log=(output/result['log_observed']['path']).read_bytes() if 'log_observed' in result else b''
                combined=b''.join((output/item['raw']['path']).read_bytes() for item in result['log_chunks'])
                self.assertEqual(combined,final_log)
                trigger=obj(output,result['trigger']) if 'trigger' in result else None
                facts=obj(output,result['trace_facts']) if 'trace_facts' in result else None
                counts=obj(output,result['wire_accounting'])['counts'] if 'wire_accounting' in result else {}
                logs=server.output_path.read_text()
                self.assertFalse(any(t.name.startswith(('serving-client-','serving-http-','phase-http-')) for t in threading.enumerate()))
                return result,rows,trigger,facts,counts,logs
            finally:
                receipt=server.close(reason='fixture_complete')
                artifact=os.environ.get('MEMRA_PHASE_TEST_EVIDENCE_DIR')
                if artifact:
                    dest=Path(tempfile.mkdtemp(prefix=scenario+'-'+mode+'-',dir=artifact));shutil.copytree(root,dest/'fixture')
                self.assertEqual(receipt['server_exit']['returncode'],0,receipt)
                self.assertTrue(receipt['cleanup']['complete'],receipt)
                self.assertFalse(receipt['cleanup']['escalated'],receipt)

    def test_actual_loopback_phase_driven_cancellation_and_correlation_isolation(self):
        for scenario in ('cancel_queued','cancel_prime','cancel_decode'):
            with self.subTest(scenario=scenario):
                result,rows,trigger,facts,counts,logs=self.run_case(scenario)
                self.assertEqual(result['state'],'captured',result['errors'])
                self.assertEqual(counts,{'clean_success':2,'client_cancelled':1})
                self.assertEqual(facts['target']['request_id'],'minted-target')
                self.assertLessEqual(trigger['log_read']['finished_ns'],trigger['cancellation']['set_before_ns'])
                self.assertEqual(trigger['complete_byte_watermark'],trigger['log_read']['complete_bytes'])
                http=[json.loads(l) for l in logs.splitlines() if l.startswith('{"event": "HTTP"')]
                self.assertEqual(len(http),3)
                self.assertEqual([r['key'] for r in http if r['role']=='target'],[KEY])
                self.assertTrue(all(r['key'] is None for r in http if r['role']!='target'))
                if scenario=='cancel_prime':self.assertIsNotNone(facts['drop_spanning_quantum'])

    def test_real_http_finishes_inside_actual_event_set_interval(self):
        result,rows,trigger,*_=self.run_case(inside_set=True)
        target=next(row for row in rows if row['id']=='target')
        self.assertLess(trigger['cancellation']['set_before_ns'],target['finished_ns'])
        self.assertLess(target['finished_ns'],trigger['cancellation']['set_after_ns'])
        self.assertEqual(result['state'],'captured',result['errors'])

    def test_valid_long_poll_missing_retirement_joins_before_return(self):
        result,rows,trigger,*_=self.run_case(mode='missing_retirement',long_poll=True)
        self.assertIsNotNone(trigger)
        self.assertEqual(result['state'],'failed')
        self.assertTrue(any('retirement observation deadline' in error for error in result['errors']),result)
        self.assertEqual({row['id'] for row in rows},{'peer','target'})
        self.assertEqual(result['unattempted_ids'],['recovery'])
        self.assertEqual(result['unobserved_invoked_ids'],[])

    def test_actual_group_abort_interrupts_long_poll_and_preserves_target(self):
        result,rows,trigger,*_=self.run_case(mode='missing_retirement',long_poll=True,abort_poll=True)
        self.assertIsNotNone(trigger)
        self.assertEqual(result['state'],'failed')
        self.assertTrue(any('phase capture interrupted' in error for error in result['errors']),result)
        self.assertIn('target',{row['id'] for row in rows})
        self.assertEqual(result['unattempted_ids'],['recovery'])
        self.assertEqual(result['unobserved_invoked_ids'],[])

    def test_false_or_stale_trigger_and_failed_quantum_never_capture_facts(self):
        for mode in ('false_prefix','stale_at_drop','failed_quantum'):
            with self.subTest(mode=mode):
                result,rows,trigger,facts,counts,logs=self.run_case('cancel_prime',mode)
                self.assertEqual(result['state'],'failed')
                self.assertIsNotNone(trigger)
                self.assertIsNone(facts)
                self.assertEqual(len(rows),2)
                self.assertEqual(result['unattempted_ids'],['recovery'])

    def test_no_phase_or_missing_retirement_times_out_with_full_denominator(self):
        for mode in ('no_phase','missing_retirement'):
            with self.subTest(mode=mode):
                result,rows,trigger,facts,counts,logs=self.run_case('cancel_decode',mode)
                self.assertEqual(result['state'],'failed')
                self.assertTrue(any('deadline' in s for s in result['errors']),result)
                self.assertEqual(len(rows),2)
                self.assertEqual(result['unattempted_ids'],['recovery'])

    def test_fast_eof_and_fragmented_typed_error_are_failed_controls(self):
        for mode in ('fast_eof','fragmented_error'):
            with self.subTest(mode=mode):
                result,rows,trigger,facts,counts,logs=self.run_case('cancel_decode',mode)
                self.assertEqual(result['state'],'failed')
                if mode=='fragmented_error':
                    target=next(r for r in rows if r['id']=='target')
                    self.assertEqual(target['body'],typed_error('chat_sse'))
                    self.assertLess(target['chunks'][-1]['observed_ns'],trigger['cancellation']['set_before_ns'])
                    self.assertTrue(any('deadline_exceeded' in s for s in result['errors']),result)
                    self.assertEqual(len(rows),3)

    def test_binding_failure_and_interrupted_launch_leave_no_late_HTTP_or_threads(self):
        for mode,interrupt in (('foreign_model',False),('normal',True)):
            with self.subTest(mode=mode):
                result,rows,trigger,facts,counts,logs=self.run_case(mode=mode,interrupt=interrupt)
                self.assertEqual(result['state'],'failed')
                self.assertIsNone(facts)
                self.assertNotIn('"role": "recovery"',logs)

    def test_preexisting_key_refuses_before_any_request_is_invoked(self):
        result,rows,trigger,facts,counts,logs=self.run_case(mode='preexisting_key')
        self.assertEqual(result['state'],'failed')
        self.assertTrue(any('already exists' in s for s in result['errors']),result)
        self.assertEqual(rows,[])
        self.assertEqual(result['request_invocations'],[])
        self.assertEqual(result['unattempted_ids'],['peer','target','recovery'])
        self.assertNotIn('"event": "HTTP"',logs)

    def test_incomplete_prelaunch_line_cannot_hide_key_reuse(self):
        result,rows,trigger,facts,counts,logs=self.run_case(mode='partial_before_launch')
        self.assertEqual(result['state'],'failed')
        self.assertTrue(any('prelaunch log prefix deadline' in s for s in result['errors']),result)
        self.assertEqual(rows,[])
        self.assertEqual(result['request_invocations'],[])
        self.assertEqual(result['unattempted_ids'],['peer','target','recovery'])
        self.assertNotIn('"event": "HTTP"',logs)
        self.assertTrue(logs.endswith('[request-lifecycle] {"schema":'))

    @unittest.skipUnless(sys.platform=='linux','actual listener/birth proof needs Linux procfs')
    def test_linux_actual_listener_phase_fixture(self):
        result,*_=self.run_case('cancel_decode',real_listener=True)
        self.assertEqual(result['state'],'captured',result['errors'])


if __name__=='__main__':unittest.main()
