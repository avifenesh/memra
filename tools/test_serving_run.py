"""Outer-policy/refusal and real adapter replay controls; no native qualification.

Temporary Git policy and records are synthetic checker inputs. Adapter positives
replay existing complete raw fixtures, not passing summary booleans.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import release_qualification as q
import serving_run as run
from serving_capture import encoded
from serving_release import ServingGateError
from test_serving_group_gate import bound_fixture
from test_serving_overload_gate import fixture as overload_fixture
from test_serving_cancel_evidence import Packet as PhasePacket
from test_serving_drain_evidence import Packet as DrainPacket
from test_serving_worker_failure_evidence import Packet as WorkerPacket
from test_release_qualification import Fixture, UUID
from test_serving_policy import BOOT

ROOT=Path(__file__).resolve().parents[1]


class ControllerBindingTests(unittest.TestCase):
    """Fresh Python processes run production imports/binding; no native stand-ins."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.repo=self.root/'repo';self.repo.mkdir()
        shutil.copytree(ROOT/'tools',self.repo/'tools',ignore=shutil.ignore_patterns('__pycache__'))
        (self.repo/'.gitignore').write_text('__pycache__/\n')
        for args in [('init','-q'),('config','user.name','CPU fixture'),('config','user.email','fixture@example.invalid'),
                     ('config','core.hooksPath','/dev/null'),('add','.'),('commit','-qm','controller source')]:
            subprocess.run(['git','-C',str(self.repo),*args],check=True,capture_output=True)
        self.head=subprocess.check_output(['git','-C',str(self.repo),'rev-parse','HEAD'],text=True).strip()
        self.entry=self.repo/'tools/serving-run.py'

    def invoke(self, *, entry=None, before='', after='', cli=False):
        env=dict(os.environ)
        for key in ('PYTHONPYCACHEPREFIX','PYTHONPATH','PYTHONOPTIMIZE'):env.pop(key,None)
        command=[sys.executable,'-B',*(['-O'] if sys.flags.optimize else [])]
        if cli:command += [str(entry or self.entry),'--expected-head',self.head,'--repo',str(self.repo),'--help']
        else:
            code="""import json,runpy,sys,types
from pathlib import Path
repo,entry,head=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
sys.path.insert(0,str(repo/'tools'))
"""+before+"""
n=runpy.run_path(str(entry),run_name='controller_test')
source=n['producer'].clean_source(repo)
"""+after+"""
result=n['observed_controller_identity'](repo,head,source)
print(json.dumps({'identity':result,'cached_marker':getattr(n['q'],'TEST_CACHED_MARKER',False)}))
"""
            command += ['-c',code,str(self.repo),str(entry or self.entry),self.head]
        return subprocess.run(command,env=env,capture_output=True,text=True,timeout=30)

    def refused(self,result,reason):
        self.assertNotEqual(result.returncode,0,result.stdout)
        self.assertIn(reason,result.stderr)

    def test_fresh_controller_records_actual_source_closure(self):
        r=self.invoke();self.assertEqual(r.returncode,0,r.stderr)
        identity=json.loads(r.stdout)['identity']
        for path,value in identity.items():
            p=self.repo/path;self.assertEqual(value['sha256'],q.digest(p.read_bytes()))
            self.assertEqual(value['mode'],'100755' if p.stat().st_mode&0o111 else '100644')
        r=self.invoke(cli=True);self.assertEqual(r.returncode,0,r.stderr)

    def test_ignored_helper_bytecode_is_not_executed(self):
        p=self.repo/'tools/release_qualification.py';raw=p.read_bytes()
        p.write_bytes(raw+b'\nTEST_CACHED_MARKER = True\n')
        cache=p.parent/'__pycache__'/('release_qualification.'+sys.implementation.cache_tag+
            ('.opt-1' if sys.flags.optimize else '')+'.pyc')
        py_compile.compile(str(p),cfile=str(cache),doraise=True,optimize=int(bool(sys.flags.optimize)),
            invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
        p.write_bytes(raw)
        r=self.invoke();self.assertEqual(r.returncode,0,r.stderr)
        self.assertFalse(json.loads(r.stdout)['cached_marker'])

    def test_foreign_complete_entry_closure_refuses_before_imports(self):
        alternate=self.root/'alternate';shutil.copytree(self.repo/'tools',alternate/'tools')
        self.refused(self.invoke(entry=alternate/'tools/serving-run.py',cli=True),'origins differ from requested repository')

    def test_helper_byte_drift_refuses_before_imports(self):
        p=self.repo/'tools/serving_http.py';p.write_bytes(p.read_bytes()+b'\nraise RuntimeError("UNBOUND_HELPER_EXECUTED")\n')
        r=self.invoke(cli=True);self.refused(r,'bytes/modes differ')
        self.assertNotIn('UNBOUND_HELPER_EXECUTED',r.stderr)

    def test_helper_mode_drift_refuses_before_imports(self):
        p=self.repo/'tools/serving_http.py';p.chmod(p.stat().st_mode^0o100)
        self.refused(self.invoke(cli=True),'bytes/modes differ')

    def test_helper_symlink_refuses_before_imports(self):
        p=self.repo/'tools/serving_http.py';outside=self.root/'helper.py';p.rename(outside);p.symlink_to(outside)
        self.refused(self.invoke(cli=True),'helper origin is not a regular source file')

    def test_preloaded_helper_cannot_capture(self):
        self.refused(self.invoke(before="m=types.ModuleType('serving_http');m.capture_request=None;sys.modules['serving_http']=m\n"),
                     'capture requires a fresh process')

    def test_postimport_bytes_and_modes_are_rechecked(self):
        for change in ["p.write_bytes(p.read_bytes()+b'\\n# drift\\n')", "p.chmod(p.stat().st_mode^0o100)"]:
            with self.subTest(change=change):
                p=self.repo/'tools/serving_http.py';raw=p.read_bytes();mode=p.stat().st_mode
                r=self.invoke(after="p=repo/'tools/serving_http.py';"+change+'\n')
                self.refused(r,'bytes/modes differ');p.write_bytes(raw);p.chmod(mode)


def policy_fixture(repo):
    roster=q.read_roster((repo/'tools/release-roster.tsv').read_bytes())
    manifest=(ROOT/run.MANIFEST).read_bytes();(repo/run.MANIFEST).write_bytes(manifest)
    scopes=[]
    for i,model in enumerate(roster):
        scope={'id':'model-'+str(i),'model':'gate'+str(i),'route':'reviewed-fixture-route','profile':'text-generation-v1'}
        scopes.append({'roster_id':model['id'],'scope':scope,'artifacts':{'model':{'path':model['path'],'bytes':8,'sha256':'a'*64}},
            'hardware':{'devices':[{'name':'CPU fixture device','compute_cap':'12.0'}]},'cells':[
                {'scenario':c['scenario'],'program':{'cell_id':c['id'],'scope':scope,'mode':'serial',
                    'requests':[{'id':'not-a-real-native-program'}]},'server':{'env':{'MEMRA_MODELS':scope['model']+'='+model['path']},
                    'timeouts':{'startup':1,'overall':10,'drain':3,'kill':1},
                    'http':{'connect_timeout':1,'read_timeout':1,'wall_timeout':2,'max_body_bytes':16384}}}
                for c in run.required_cells(manifest,[scope])]})
    policy={'schema':'memra-serving-policy-v1','scopes':scopes};(repo/run.POLICY).write_bytes(encoded(policy))
    for name in run.CONTROLLER_FILES:
        path=repo/name;path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():path.write_text('# synthetic source fixture only\n')
    return policy


class ServingPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.f=Fixture(Path(self.tmp.name));self.policy=policy_fixture(self.f.repo);self.f.commit('fixture policy')
        self.source=q.source_snapshot(self.f.repo)
    def publish_policy(self,value):
        (self.f.repo/run.POLICY).write_bytes(encoded(value));self.f.commit('fixture policy mutation')
    def test_policy_requires_all_roster_models_and_eleven_cells(self):
        policy,manifest=run.load_policy(self.f.repo,'HEAD',self.source)
        self.assertEqual(len(run.required_cells(manifest,[s['scope'] for s in policy['scopes']])),22)
        for mutate in (lambda p:p['scopes'].pop(), lambda p:p['scopes'][0]['cells'].pop(),
                       lambda p:p['scopes'][0]['scope'].update(id='model-1')):
            p=copy.deepcopy(self.policy);mutate(p);self.publish_policy(p)
            with self.assertRaises(ServingGateError):run.load_policy(self.f.repo,'HEAD')
    def test_policy_is_immutable_git_input_not_mutable_or_run_selected_scope(self):
        (self.f.repo/run.POLICY).write_text('{"schema":"attacker"}')
        self.assertEqual(run.load_policy(self.f.repo,'HEAD')[0],self.policy)
        source=copy.deepcopy(self.source);source['files'].pop(run.POLICY)
        with self.assertRaisesRegex(ServingGateError,'input of the tested build'):run.load_policy(self.f.repo,'HEAD',source)
    def test_self_selected_artifact_alias_or_dynamic_policy_owner_refuses(self):
        for mutate in (lambda p:p['scopes'][0]['artifacts']['model'].update(path='/other/model'),
                       lambda p:p['scopes'][0]['cells'][0]['server']['env'].update(MEMRA_MODELS='foreign=/other/model'),
                       lambda p:p['scopes'][0]['cells'][0]['program'].update(server_identity={'pid':1}),
                       lambda p:p['scopes'][0]['hardware'].update(devices=[])):
            p=copy.deepcopy(self.policy);mutate(p);self.publish_policy(p)
            with self.assertRaises((ServingGateError,q.GateError)):run.load_policy(self.f.repo,'HEAD')
    def test_missing_policy_is_explicit_refusal(self):
        (self.f.repo/run.POLICY).unlink();self.f.commit('remove policy')
        with self.assertRaisesRegex(ServingGateError,'required tracked policy/input missing'):run.load_policy(self.f.repo,'HEAD')
    def test_bound_hardware_has_no_generic_gpu0_requirement(self):
        # Existing physical-lease controls cover the full multidevice validator;
        # this test ensures the serving policy keeps the route's declared class.
        p=copy.deepcopy(self.policy);p['scopes'][0]['hardware']['devices']*=3;self.publish_policy(p)
        self.assertEqual(len(run.load_policy(self.f.repo,'HEAD')[0]['scopes'][0]['hardware']['devices']),3)
    def test_materialization_changes_only_operating_identity(self):
        scope=self.policy['scopes'][0];cell=scope['cells'][0]
        cell['program']['requests']=[{'id':'r','model':'gate0','wire':'chat_json','path':'/v1/chat/completions',
            'payload':{'model':'gate0','stream':False,'messages':[{'role':'user','content':'frozen oracle prompt'}]}}]
        runtime={'server_identity':{'pid':100,'start_identity':BOOT+':1234'},'port':18000,'cwd':'/source',
                 'binary':'/built/memra-server','output_path':'/owned/capture/process/output.log','client_trace_key':None}
        before=copy.deepcopy(cell)
        p,launch,plan,ids=run.materialize(scope,cell,runtime,[UUID],self.f.binaries)
        self.assertEqual(cell,before);self.assertEqual(p['requests'],cell['program']['requests'])
        self.assertEqual(launch['env']['CUDA_VISIBLE_DEVICES'],UUID)
        self.assertEqual(plan['groups'][0]['requests'][0]['payload'],p['requests'][0]['payload'])
        self.assertEqual(ids['server_binary']['sha256'],self.f.binaries['memra-server']['sha256'])
    def test_missing_or_duplicate_scope_run_refuses_before_adapter_dispatch(self):
        policy,manifest=run.load_policy(self.f.repo,'HEAD',self.source)
        self.f.put('policy.json',policy);(self.f.out/'manifest.json').write_bytes(manifest)
        stage={'schema':'memra-serving-release-v1','source':self.f.ref('source.json'),'build':self.f.ref('build.json'),
               'policy':self.f.ref('policy.json'),'manifest':self.f.ref('manifest.json'),'runs':[]}
        bindings={'repo':self.f.repo,'head':'HEAD','source':self.source,'source_reference':stage['source'],
                  'build':self.f.build,'build_reference':stage['build']}
        with self.assertRaisesRegex(ServingGateError,'denominator'):run.validate_serving_run(stage,q.Evidence(self.f.out),manifest,bindings)
        with patch('serving_run.replay_cell',side_effect=AssertionError('must not dispatch')):
            with self.assertRaises(ServingGateError):run.validate_serving_run(stage,q.Evidence(self.f.out),manifest,bindings)


class LiveLeaseTests(unittest.TestCase):
    def sample(self):
        controller={'pid':12,'ppid':11,'pgid':12,'start_time':'123','identity_source':'linux_proc_start_ticks','state':'R'}
        child={**controller,'pid':11,'ppid':10,'start_time':'122'};wrapper={**controller,'pid':10,'ppid':1,'start_time':'121'}
        lease={'requested_uuids':[UUID],'child_pid':11,'wrapper_pid':10,'lock_files':{UUID:f'/tmp/memra-gpu-locks/{UUID}.lock'}}
        value={'started_ns':10,'finished_ns':20,'unix_ns':1_000_000_000,'boot_id':BOOT,'controller':controller,
            'ancestors':[controller,child,wrapper], 'locks':[{'uuid':UUID,'path':lease['lock_files'][UUID],
                'device_major':8,'device_minor':1,'inode':77,'raw':'1: FLOCK ADVISORY WRITE 10 08:01:77 0 EOF'}]}
        return value,lease,controller,{'devices':[{'uuid':UUID}]}
    def test_raw_flock_ancestry_and_birth_positive(self):
        self.assertEqual(run.lease_observation(*self.sample()),(10,20))
    def test_summary_boolean_cannot_replace_actual_lock_or_owned_ancestry(self):
        for change in (lambda x:x[0]['locks'][0].update(raw='1: POSIX ADVISORY READ 10 08:01:77 0 EOF'),
                       lambda x:x[0]['locks'][0].update(inode=78),lambda x:x[0]['ancestors'].pop(1),
                       lambda x:x[0].update(locks=[]),lambda x:x[0]['controller'].update(start_time='not-birth')):
            value=list(copy.deepcopy(self.sample()));change(value)
            with self.assertRaises((ServingGateError,ValueError)):run.lease_observation(*value)


class AdapterReplayTests(unittest.TestCase):
    def test_group_and_overload_dispatch_replays_actual_raw_fixtures(self):
        for scenario in sorted(run.GROUP):
            b=overload_fixture() if scenario=='overload_recovery' else bound_fixture(scenario)
            result=run.replay_cell(b['required'],b['program'],b['capture_bytes'],b['evidence_reader'],
                capture_sha256=b['expected_capture_sha256'],launch=None,plan=b['expected_plan'],
                identities=b['expected_identities'],log_identity=None)
            self.assertFalse(result['qualification']);self.assertGreater(result['verified_payloads'],0)
            with self.assertRaises(ServingGateError):run.replay_cell(b['required'],b['program'],b['capture_bytes']+b' ',b['evidence_reader'],
                capture_sha256=b['expected_capture_sha256'],launch=None,plan=b['expected_plan'],identities=b['expected_identities'],log_identity=None)
    def test_worker_adapter_replays_frozen_full_packet(self):
        p=WorkerPacket();raw=encoded(p.capture)
        result=run.replay_cell(p.required,p.program,raw,p.blobs.__getitem__,capture_sha256=q.digest(raw),
            launch=p.launch,plan=None,identities=p.identities,log_identity=p.log_identity)
        self.assertFalse(result['qualification']);self.assertEqual(result['facts']['planned'],3)
    def test_phase_and_drain_dispatch_replay_raw_history_and_buffered_reads(self):
        p=PhasePacket();raw=encoded(p.capture);e=p.expected
        result=run.replay_cell(e['required'],e['program'],raw,p.blobs.__getitem__,capture_sha256=q.digest(raw),
            launch=e['server'],plan=None,identities={},log_identity=e['log_identity'])
        self.assertFalse(result['qualification']);self.assertGreater(result['verified_payloads'],0)
        p=DrainPacket();raw=encoded(p.capture)
        result=run.replay_cell(p.required,p.program,raw,p.blobs.__getitem__,capture_sha256=q.digest(raw),
            launch=p.launch,plan=None,identities=p.identities,log_identity=None)
        self.assertFalse(result['qualification']);self.assertGreater(result['verified_payloads'],0)

    def test_all_required_dispatches_are_explicit(self):
        targets={'cancel_queued':'serving_cancel_evidence.read_phase_capture','cancel_prime':'serving_cancel_evidence.read_phase_capture',
                 'cancel_decode':'serving_cancel_evidence.read_phase_capture','drain':'serving_drain_evidence.read_drain_capture'}
        for scenario,target in targets.items():
            with patch(target,return_value={'scope':'dispatch-control-only'}) as function:
                run.replay_cell({'scenario':scenario},{},b'raw',lambda p:b'',capture_sha256='a'*64,
                    launch={},plan=None,identities={},log_identity={})
                self.assertEqual(function.call_count,1)
        with self.assertRaises(ServingGateError):run.replay_cell({'scenario':'unknown'},{},b'',lambda p:b'',
            capture_sha256='a'*64,launch={},plan=None,identities={},log_identity={})


class ReleaseStageTests(unittest.TestCase):
    def startup_control(self, failure=False, module_path=None):
        from unittest.mock import create_autospec
        from serving_http import capture_request
        from serving_capture import EvidenceStore
        from serving_process import ProcessIdentity, ReadinessEvidence
        from test_serving_policy import PROCESS
        spec=importlib.util.spec_from_file_location('serving_runner_startup',module_path or ROOT/'tools/serving-run.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        owner=ProcessIdentity(**PROCESS);ready=[];requests=[]
        class Server:
            identity=owner
            _config={'timeouts':{'startup':1}}
            def receipt(self):return {'boot_id':BOOT}
            def mark_ready(self,proof):ready.append(proof)
        def observed(**kwargs):
            self.assertEqual(kwargs['body'],b'');requests.append(kwargs)
            return {'id':kwargs['request_id'],'status':200,'body':b'partial' if failure else encoded(
                {'status':'ok' if kwargs['path']=='/health' else 'ready','models':['gate'],'worker':{'generation':0,'phase':'idle'}}),
                'transport_error':{'kind':'read','message':'truncated'} if failure else None}
        http=create_autospec(capture_request,side_effect=observed)
        with tempfile.TemporaryDirectory() as temp:
            store=EvidenceStore(Path(temp)/'startup')
            with patch.object(runner,'capture_request',http),patch.object(runner,'prove_listener',return_value=ReadinessEvidence(owner,'listener_identity',{'CPU_signature_fixture':True})):
                kwargs=dict(endpoint={'host':'127.0.0.1','port':18000},http={'connect_timeout':1,'read_timeout':1,'wall_timeout':1,'max_body_bytes':16384},store=store,scope={'model':'gate'})
                if failure:
                    with self.assertRaises(ServingGateError):runner.readiness(Server(),**kwargs)
                else:runner.readiness(Server(),**kwargs)
            saved=json.loads((store.root/'capture.json').read_text())
            self.assertEqual(saved['state'],'failed' if failure else 'ready')
            self.assertEqual(len(saved['observations']),1 if failure else 2)
            self.assertEqual(len(ready),0 if failure else 1)
            if failure:
                row=json.loads((store.root/saved['observations'][0]['path']).read_text())
                self.assertEqual((store.root/row['body']['path']).read_bytes(),b'partial')
            else:self.assertEqual([r['path'] for r in requests],['/health','/readyz'])
    def test_startup_binds_real_http_signature_and_empty_get_body(self):
        self.startup_control()
    def test_startup_transport_failure_keeps_raw_denominator(self):
        self.startup_control(failure=True)
    def test_stage_cannot_overwrite_generic_record_namespace(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'stage';root.mkdir();out=Path(temp)/'out';out.mkdir()
            (root/'generic-record.json').write_bytes(b'forged')
            (root/'stage.json').write_bytes(encoded({'payloads':{'generic-record.json':q.digest(b'forged')},'serving':{}}))
            with self.assertRaisesRegex(ServingGateError,'another release namespace'):run.import_stage(root,out)
            self.assertFalse((out/'generic-record.json').exists())
    def test_failed_v2_seal_evidence_is_not_overwritten(self):
        import argparse
        spec=importlib.util.spec_from_file_location('qualify_seal_test',ROOT/'tools/qualify-release.py')
        producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);(path/'generic-record.json').write_bytes(b'original failed seal')
            with self.assertRaisesRegex(q.GateError,'already attempted'):producer.seal(argparse.Namespace(out=path))
            self.assertEqual((path/'generic-record.json').read_bytes(),b'original failed seal')
    def test_battery_refuses_missing_serving_before_any_gpu_work(self):
        result=subprocess.run(['bash',str(ROOT/'tools/release-battery.sh')],capture_output=True,text=True,timeout=10)
        self.assertNotEqual(result.returncode,0);self.assertIn('required --serving-record is missing',result.stderr)
    def test_serving_hardware_observation_does_not_query_nvml_gpu0(self):
        spec=importlib.util.spec_from_file_location('qualify_test',ROOT/'tools/qualify-release.py')
        producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)
        calls=[]
        def observation(argv,**kwargs):
            calls.append(argv)
            if argv[1:3]==['topo','-m']:return b'CPU MOCK topology'
            return '3, '+UUID+', NVIDIA RTX PRO 6000 Blackwell Server Edition, 12.0, CPU DRIVER, 0000:03:00.0, 97280\n'
        with patch.object(producer.subprocess,'check_output',side_effect=observation),patch.object(producer,'platform_identity',return_value={'profile':'CPU'}):
            hardware,_=producer.observe_hardware([UUID],generic=False)
        self.assertIsNone(hardware['headroom_query'])
        self.assertEqual(hardware['devices'][0]['index'],'3')
        self.assertFalse(any('-i' in call and call[call.index('-i')+1]=='0' for call in calls))

    def test_seal_cli_requires_serving_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            result=subprocess.run([sys.executable,'-B',str(ROOT/'tools/qualify-release.py'),'seal','--out',temp,
                '--lease',temp+'/lease.json','--oracles',temp],capture_output=True,text=True,timeout=10)
            self.assertNotEqual(result.returncode,0);self.assertIn('serving',result.stderr)

if __name__=='__main__':unittest.main()
