"""Owned local HTTP/process fixtures; simulated epochs are not native worker proof."""
import hashlib,json,os,shutil,socket,sys,tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
from serving_process import OwnedServer,ReadinessEvidence
from serving_worker_failure_capture import collect_worker_failure_cell
from serving_worker_failure_evidence import read_worker_failure_capture
from test_serving_worker_failure import config

class WorkerFailureCaptureTests(unittest.TestCase):
    def run_case(self,mode='normal',multiple=False,interrupt=False,delay_finish=False):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();script=Path(__file__).parent/'fixtures/worker_failure_http_server.py'
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            server=OwnedServer(argv=[sys.executable,str(script),str(port),mode],env={'MEMRA_PANIC_AFTER':'1','MEMRA_WORKER_RESPAWN':'1'},
                cwd=str(root),evidence_dir=str(root/'process'),startup_timeout=4,overall_timeout=15,drain_timeout=2,kill_timeout=1)
            try:
                server.start();limit=time.monotonic()+3
                while time.monotonic()<limit and (not server.output_path.exists() or 'READY' not in server.output_path.read_text()):time.sleep(.01)
                self.assertIn('READY',server.output_path.read_text())
                server.mark_ready(ReadinessEvidence(server.identity,'owned_output',{'CPU_fixture':True}))
                required,program=config();program['endpoint']['port']=port
                if delay_finish:
                    program['http']['wall_timeout']=.5
                    program['timing']={'prefix_timeout_s':.7,'fault_timeout_s':1,'recovery_timeout_s':2,'poll_s':.01,'overall_s':6}
                owner=server.identity;receipt=server.receipt()
                program['server_identity']={'pid':owner.pid,'start_identity':receipt.get('boot_id',owner.identity_source)+':'+owner.start_time}
                program['identities']={'server_binary':sys.executable,'source_fixture':str(script.resolve())}
                if multiple:
                    import copy
                    extra=copy.deepcopy(program['requests'][0]);extra['id']='victim2';program['requests'].insert(1,extra)
                from contextlib import ExitStack
                with ExitStack() as stack:
                    if sys.platform!='linux':
                        stack.enter_context(patch('serving_worker_failure_capture.prove_listener',return_value=ReadinessEvidence(owner,'listener_identity',{'CPU_fixture_only':True})))
                    if interrupt:
                        start=threading.Thread.start
                        def launch(thread):
                            if interrupt=='before' and thread.name=='wf-pressure':
                                import signal
                                os.kill(os.getpid(),signal.SIGINT)
                                return
                            start(thread)
                            if thread.name=='wf-pressure':raise KeyboardInterrupt('controlled native pressure-thread launch interruption')
                        stack.enter_context(patch.object(threading.Thread,'start',launch))
                    if delay_finish:
                        import serving_http
                        line=next(i for i,s in enumerate(Path(serving_http.__file__).read_text().splitlines(),1)
                                  if s.strip()=='result["finished_ns"] = time.monotonic_ns()')
                        paused=[]
                        def pause_actual_client(frame,event,arg):
                            if (event=='line' and frame.f_code.co_name=='capture_request' and frame.f_lineno==line
                                    and Path(frame.f_code.co_filename).name=='serving_http.py'
                                    and frame.f_locals.get('request_id')=='victim' and not paused):
                                paused.append(time.monotonic_ns());time.sleep(9)
                            return pause_actual_client
                        previous_trace=threading.gettrace();threading.settrace(pause_actual_client)
                        stack.callback(threading.settrace,previous_trace)
                    value=collect_worker_failure_cell(required,program,server=server,output=root/'capture')
                    if delay_finish:self.assertEqual(len(paused),1)
                cap=(root/'capture/capture.json').read_bytes();index=json.loads(cap)
                for name,digest in index['payloads'].items():self.assertEqual(hashlib.sha256((root/'capture'/name).read_bytes()).hexdigest(),digest)
                def obj(ref):return json.loads((root/'capture'/ref['path']).read_text())
                facts=obj(value['facts']) if value['facts'] else None
                lifecycle=obj(value['lifecycle'])
                self.assertFalse(value['qualification'])
                self.assertEqual(len(value['request_denominator']),len(program['requests']))
                self.assertFalse(any(t.name.startswith(('wf-pressure','serving-client-','serving-http-')) for t in threading.enumerate()))
                time.sleep(.05);self.assertEqual((root/'capture/capture.json').read_bytes(),cap)
                if value['state']=='captured':
                    descriptor=obj(value['log_descriptor'])
                    result=read_worker_failure_capture(cap,lambda name:(root/'capture'/name).read_bytes(),
                        expected_capture_sha256=hashlib.sha256(cap).hexdigest(),expected_required=required,expected_program=program,
                        expected_server={'argv':server._config['argv'],'cwd':str(root),'env':server._config['env'],'timeouts':server._config['timeouts'],'output_path':str(server.output_path)},
                        expected_identities=value['identities_before'],expected_log_identity={k:descriptor[k] for k in ('path','device','inode','controller_pid')})
                    self.assertFalse(result['qualification'])
                return value,facts,lifecycle
            finally:
                if server.receipt().get('state')!='finished':server.close(reason='fixture_cleanup')
                evidence=os.environ.get('MEMRA_WORKER_FAILURE_EVIDENCE_DIR')
                if evidence:
                    dest=Path(tempfile.mkdtemp(prefix=mode+'-',dir=evidence));shutil.copytree(root,dest/'fixture')
    def test_actual_worker_epoch_episode_keeps_typed_failure_and_clean_recovery(self):
        value,facts,life=self.run_case()
        self.assertIsNotNone(facts,value['errors'])
        self.assertEqual(facts['wire_counts'],{'clean_success':2,'typed_error':1})
        self.assertEqual(facts['generation_observations']['respawns_observed'],1)
        self.assertEqual(life['server_exit']['returncode'],0)
        self.assertTrue(life['cleanup']['complete'])
        if sys.platform=='linux':self.assertEqual(value['state'],'captured',value['errors'])
        else:
            self.assertEqual(value['state'],'failed')
            self.assertTrue(any('Linux birth' in e for e in value['errors']))
    def test_multiple_admitted_victims_are_all_retained(self):
        value,facts,_=self.run_case(multiple=True)
        self.assertIsNotNone(facts,value['errors'])
        self.assertEqual(len(facts['typed_failures']),2)
        self.assertEqual(value['unattempted_ids'],[])
        self.assertEqual(value['unobserved_invoked_ids'],[])
        if sys.platform=='linux':self.assertEqual(value['state'],'captured',value['errors'])
    def test_actual_fragmented_nonterminal_prefix_remains_usable(self):
        value,facts,_=self.run_case('fragmented_nonterminal')
        self.assertIsNotNone(facts,value['errors'])
        self.assertEqual(facts['typed_failures'][0]['pre_error_frames'],1)
    def test_delayed_real_http_client_is_settled_before_final_index(self):
        value,_,life=self.run_case(delay_finish=True)
        self.assertEqual(value['state'],'failed')
        self.assertEqual(value['unobserved_invoked_ids'],[])
        self.assertEqual([r['id'] for r in value['request_denominator'] if r['status']=='captured'],['victim','trigger'])
        self.assertTrue(any('settlement' in v for v in value['client_errors'].values()))
        self.assertEqual(life['server_exit']['returncode'],0)
    def test_truncated_missing_DONE_and_request_fault_substitutes_refuse(self):
        for mode in ('truncated','no_done','request_fault'):
            with self.subTest(mode=mode):
                value,facts,_=self.run_case(mode)
                self.assertEqual(value['state'],'failed')
                self.assertIsNone(facts)
                self.assertEqual(value['unattempted_ids'],[])
                self.assertEqual(len(value['observations']),3)
    def test_missing_generation_bad_health_and_repeat_fault_refuse(self):
        for mode in ('no_generation','wrong_health','repeat_fault'):
            with self.subTest(mode=mode):
                value,facts,_=self.run_case(mode)
                self.assertEqual(value['state'],'failed')
                self.assertIsNone(facts)
                self.assertEqual(len(value['request_denominator']),3)
    def test_launch_interruption_joins_owned_clients_and_stops_owner(self):
        for point in ('before',True):
            with self.subTest(point=point):
                value,_,life=self.run_case(interrupt=point)
                self.assertEqual(value['state'],'failed')
                self.assertTrue(life['cleanup']['complete'])
    @unittest.skipUnless(sys.platform=='linux','full owned-process/listener replay requires Linux procfs')
    def test_linux_complete_immutable_replay(self):
        value,_,_=self.run_case()
        self.assertEqual(value['state'],'captured',value['errors'])

def _launch_signal_probe(scenario):
    """Actual SIGINT in a tiny owned subprocess; no HTTP/server/GPU work."""
    import faulthandler,inspect,signal
    from serving_worker_failure_capture import _owned_clients
    rows=[];errors={};observed={};fired=[]
    start=threading.Thread.start;bootstrap=threading.Thread._bootstrap_inner
    handler=signal.getsignal(signal.SIGINT)
    entered=threading.Event();release=threading.Event();auxiliary=None

    def delayed_bootstrap(thread):
        if thread.name.startswith('serving-client-'):
            observed['real_native_thread_entered']=True
            observed['ident_before_bootstrap']=thread.ident
            entered.set();release.wait()
        return bootstrap(thread)

    def interrupt_pending_native():
        if entered.wait(2):
            os.kill(os.getpid(),signal.SIGINT)
        release.set()

    def interrupted_start(thread):
        if not thread.name.startswith('serving-client-'):return start(thread)
        observed['ident_before_start']=thread.ident
        if scenario=='runtime_before':raise RuntimeError('controlled pre-native allocation failure')
        if scenario=='before' or (scenario=='before_second' and thread.name.endswith('-second')):
            os.kill(os.getpid(),signal.SIGINT)
            return
        start(thread)
        if scenario=='after' or (scenario=='after_second' and thread.name.endswith('-second')):
            os.kill(os.getpid(),signal.SIGINT)

    native_line=None
    if scenario=='registered_before_native':
        lines,first=inspect.getsourcelines(start)
        native_line=next(first+i for i,line in enumerate(lines)
            if '_start_joinable_thread(' in line or '_start_new_thread(' in line)

    def at_native_boundary(frame,event,arg):
        if event=='line' and frame.f_code is start.__code__ and frame.f_lineno==native_line and not fired:
            thread=frame.f_locals['self']
            observed['registered_before_sigint']=thread in threading._limbo
            observed['ident_before_native_call']=thread.ident
            fired.append(True);os.kill(os.getpid(),signal.SIGINT)
        return at_native_boundary

    faulthandler.dump_traceback_later(2,repeat=False)
    try:
        if scenario=='native_pending_ident':
            auxiliary=threading.Thread(target=interrupt_pending_native,name='probe-signal-sender');auxiliary.start()
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch.object(threading.Thread,'start',interrupted_start))
            if scenario=='native_pending_ident':stack.enter_context(patch.object(threading.Thread,'_bootstrap_inner',delayed_bootstrap))
            if native_line is not None:
                previous=sys.gettrace();sys.settrace(at_native_boundary);stack.callback(sys.settrace,previous)
            requests=[{'id':'first'},{'id':'second'}] if scenario.endswith('_second') else [{'id':'recovery'}]
            _owned_clients(requests,'serial',lambda request,cancel: {'id':request['id']},rows.append,.1,errors)
        observed['exception']=None
    except BaseException as error:
        observed['exception']=type(error).__name__
    finally:
        release.set()
        if auxiliary is not None:auxiliary.join(2)
        faulthandler.cancel_dump_traceback_later()
    observed.update(rows=rows,errors=errors,handler_restored=signal.getsignal(signal.SIGINT) is handler,
        live_threads=[t.name for t in threading.enumerate() if t.name.startswith(('serving-client-','probe-signal-'))])
    print(json.dumps(observed),flush=True)


class ClientWriterSettlementTests(unittest.TestCase):
    def signal_control(self,scenario):
        import subprocess
        argv=[sys.executable,'-B',*(['-O'] if sys.flags.optimize else []),str(Path(__file__).resolve()),'--launch-probe',scenario]
        run=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=4,
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
        self.assertEqual(run.returncode,0,run.stdout.decode())
        value=json.loads(run.stdout.decode().splitlines()[-1])
        self.assertTrue(value['handler_restored']);self.assertEqual(value['live_threads'],[])
        return value

    def test_actual_sigint_before_and_after_native_start(self):
        for scenario in ('before','after','before_second','after_second'):
            with self.subTest(scenario=scenario):
                value=self.signal_control(scenario)
                self.assertEqual(value['exception'],'KeyboardInterrupt')
                self.assertEqual(value['rows'],[{'id':'first'}] if scenario.endswith('_second') else [])

    def test_sigint_registration_and_missing_ident_are_not_no_launch_proof(self):
        registered=self.signal_control('registered_before_native')
        self.assertTrue(registered['registered_before_sigint'])
        self.assertIsNone(registered['ident_before_native_call'])
        self.assertEqual(registered['exception'],'KeyboardInterrupt');self.assertEqual(registered['rows'],[])
        pending=self.signal_control('native_pending_ident')
        self.assertTrue(pending['real_native_thread_entered'])
        self.assertIsNone(pending['ident_before_bootstrap'])
        self.assertEqual(pending['exception'],'KeyboardInterrupt');self.assertEqual(pending['rows'],[])

    def test_normal_start_and_proven_pre_native_runtime_failure(self):
        normal=self.signal_control('normal')
        self.assertIsNone(normal['exception']);self.assertEqual(normal['rows'],[{'id':'recovery'}])
        failure=self.signal_control('runtime_before')
        self.assertEqual(failure['exception'],'RuntimeError');self.assertEqual(failure['rows'],[])

    def test_deadline_cancels_but_waits_for_actual_persistence(self):
        from serving_worker_failure_capture import _owned_clients
        entered=threading.Event();release=threading.Event();returned=threading.Event()
        observed={};rows=[];errors={}
        def invoke(request,cancel):
            observed['cancel']=cancel
            return {'id':request['id']}
        def persist(row):
            entered.set();release.wait();rows.append(row)
        def controller():
            try:_owned_clients([{'id':'writer'}],'concurrent',invoke,persist,.05,errors)
            except BaseException as error:observed['error']=error
            finally:returned.set()
        thread=threading.Thread(target=controller,name='test-writer-controller');thread.start()
        try:
            self.assertTrue(entered.wait(1))
            self.assertTrue(observed['cancel'].wait(1))
            self.assertFalse(returned.is_set())
            self.assertTrue(any(t.name=='serving-client-writer' and t.is_alive() for t in threading.enumerate()))
        finally:
            release.set();thread.join(3)
        self.assertFalse(thread.is_alive());self.assertTrue(returned.is_set())
        self.assertEqual(rows,[{'id':'writer'}]);self.assertIsInstance(observed['error'],TimeoutError)
        self.assertTrue(errors)

if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--launch-probe':_launch_signal_probe(sys.argv[2])
    else:unittest.main()
