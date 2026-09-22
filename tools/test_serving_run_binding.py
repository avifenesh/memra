"""Outer binding composition tests with explicitly simulated adapter outcomes.

Only replay_cell is stubbed in these tests: source/build/policy/lease/process/
startup/cell/record binding runs for real. Actual adapter raw fixture replay is
separately exercised in test_serving_run. Nothing here is native evidence.
"""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import release_input_view as view
import release_qualification as q
import serving_run as run
from serving_capture import encoded
from serving_release import ServingGateError
from test_release_qualification import Fixture, UUID
from test_serving_run import policy_fixture
import test_serving_run as run_tests
from test_serving_worker_failure_evidence import Packet


def blob(root, value):
    data=value if isinstance(value,bytes) else encoded(value)
    name='blobs/'+q.digest(data);path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    return {'path':name,'sha256':q.digest(data)}


def shift(value,base,pid):
    if isinstance(value,list):return [shift(x,base,pid) for x in value]
    if not isinstance(value,dict):return value
    out={}
    for k,v in value.items():
        if k in ('pid','pgid') and v==100:out[k]=pid
        elif 'monotonic' in k and type(v) in (int,float):out[k]=v+base
        else:out[k]=shift(v,base,pid)
    return out


class BindingFixture:
    def __init__(self,root):
        self.f=Fixture(root);f=self.f
        policy=policy_fixture(f.repo)
        for scope in policy['scopes']:
            scope['artifacts']['model'].update(f.models[scope['artifacts']['model']['path']])
            scope['hardware']['devices']=[{k:f.hardware['devices'][0][k] for k in ('name','compute_cap')}]
            for cell in scope['cells']:
                p=cell['program'];p['requests']=[{'id':'r','model':scope['scope']['model'],'path':'/v1/chat/completions','wire':'chat_json',
                    'payload':{'model':scope['scope']['model'],'stream':False}}]
                if cell['scenario']=='overload_recovery':p['requests'][0]['role']='peer'
                if cell['scenario'] not in run.GROUP:p['http']=cell['server']['http']
        (f.repo/run.POLICY).write_bytes(encoded(policy));f.commit('complete binding fixture policy')
        f.source=q.source_snapshot(f.repo);f.put('source.json',f.source)
        f.build.update(source=f.ref('source.json'),source_before=f.source['inputs_sha256'],source_after=f.source['inputs_sha256'],
            input_view_before=view.identity(f.source),input_view_after=view.identity(f.source))
        f.run.update(source_before=f.source['inputs_sha256'],source_after=f.source['inputs_sha256']);f.refresh()
        self.policy=policy;self.controllers=run.controller_identity(f.repo,'HEAD',f.source)
        manifest=(f.repo/run.MANIFEST).read_bytes();f.put('serving/policy.json',policy)
        (f.out/'serving/manifest.json').write_bytes(manifest)
        self.stage={'schema':'memra-serving-release-v1','source':f.ref('source.json'),'build':f.ref('build.json'),
            'policy':f.ref('serving/policy.json'),'manifest':f.ref('serving/manifest.json'),'runs':[]}
        self.scope_runs=[]
        for si,scope in enumerate(policy['scopes']):
            directory=f.out/'serving/runs'/scope['scope']['id'];directory.mkdir(parents=True)
            observe,lease,controller,_=run_tests.LiveLeaseTests().sample()
            native_lease=copy.deepcopy(f.lease);native_lease.update(started_unix=900,finished_unix=2200)
            (directory/'lease.json').write_bytes(encoded(native_lease))
            before=copy.deepcopy(observe);before.update(started_ns=1_000_000_000,finished_ns=2_000_000_000,unix_ns=1001_000_000_000)
            after=copy.deepcopy(observe);after.update(started_ns=130_000_000_000,finished_ns=131_000_000_000,unix_ns=1130_000_000_000)
            # The live ancestry's first PID is the actual scoped controller ID.
            captured={'schema':'memra-native-serving-scope-v1','scope_id':scope['scope']['id'],'state':'captured','errors':[],
                'source_before':f.source['inputs_sha256'],'source_after':f.source['inputs_sha256'],'build_sha256':self.stage['build']['sha256'],
                'binaries_before':copy.deepcopy(f.binaries),'binaries_after':copy.deepcopy(f.binaries),
                'models_before':{v['path']:{k:v[k] for k in ('bytes','sha256')} for v in scope['artifacts'].values()},
                'controllers_before':self.controllers,'controllers_after':self.controllers,'lease_owner':f.run['lease_owner'],
                'hardware':f.hardware,'hardware_after':f.hardware,'numeric_environment':{'CUDA_VISIBLE_DEVICES':q.digest(UUID.encode())},
                'started_unix':1000,'finished_unix':1132,'started_ns':500_000_000,'finished_ns':132_000_000_000,
                'topology':blob(directory,(f.out/'topology.txt').read_bytes()),'lease':{'path':'lease.json','sha256':q.digest(encoded(native_lease))},
                'lease_before':blob(directory,before),'lease_after':blob(directory,after),'controller':controller,'cells':[]}
            captured['models_after']=copy.deepcopy(captured['models_before'])
            for ci,cell in enumerate(scope['cells']):
                base=10*(ci+1);pid=100+si*20+ci;caproot=f'cells/{ci:03d}/capture';root=directory/caproot
                root.mkdir(parents=True)
                runtime={'server_identity':{'pid':pid,'start_identity':observe['boot_id']+':1234'},'port':18000,'cwd':'/fixture',
                    'binary':'/fixture/server','output_path':str(root/'process/output.log'),
                    'client_trace_key':'a'*32 if cell['scenario'] in run.PHASE else None}
                program,launch,plan,identities=run.materialize(scope,cell,runtime,[UUID],f.binaries)
                packet=Packet();life=shift(packet.value(packet.capture['lifecycle']),base,pid)
                life.update(argv=launch['argv'],cwd=launch['cwd'],env_keys=sorted(launch['env']),env_sha256=q.digest(q.canonical(launch['env'])),
                    timeouts=launch['timeouts'],output_path=launch['output_path']);life['supervisor']['ppid']=controller['pid']
                life['stop']['reason']='serving_scope_complete'
                index={'lifecycle':blob(root,life),'started_ns':int((base+.01)*1e9),'finished_ns':int((base+7.2)*1e9)}
                (root/'capture.json').write_bytes(encoded(index))
                entry={'id':program['cell_id'],'scope':scope['scope'],'scenario':cell['scenario'],'state':'captured','errors':[],
                    'runtime':runtime,'capture':{'path':caproot+'/capture.json','sha256':q.digest(encoded(index))},'capture_root':caproot,
                    'log_identity':None,'started_ns':base*1_000_000_000,'finished_ns':int((base+7.5)*1e9),'startup':None}
                if cell['scenario'] not in run.GROUP:
                    startroot=directory/f'cells/{ci:03d}/startup';startroot.mkdir()
                    startup={'schema':'memra-serving-startup-v1','state':'ready','observations':[],'listeners':[],'errors':[],
                             'lifecycle':blob(startroot,life),'payloads':{}}
                    def proof(t):
                        owner=copy.deepcopy(life['server'])
                        return {'method':'listener_identity','owner':owner,'details':{'schema':'memra-linux-listener-v1',
                            'endpoint':program['endpoint'],'boot_id':observe['boot_id'],'owner_before':owner,'owner_after':owner,
                            'clock':'monotonic_ns','started_ns':int((base+t)*1e9),'finished_ns':int((base+t+.001)*1e9),
                            'samples':[{'inodes':[77],'primary_fds':[5]},{'inodes':[77],'primary_fds':[5]}]}}
                    startup['listeners']=[blob(startroot,proof(.002)),blob(startroot,proof(.008))]
                    for n,(path,status,t) in enumerate([('/health','ok',.004),('/readyz','ready',.006)],1):
                        raw=encoded({'status':status,'models':[scope['scope']['model']],'worker':{'generation':0,'phase':'idle'}})
                        row={'id':'startup-'+str(n),'method':'GET','path':path,'status':200,'body':blob(startroot,raw),'server_identity':runtime['server_identity'],
                            'transport_error':None,'started_ns':int((base+t)*1e9),'finished_ns':int((base+t+.001)*1e9)}
                        startup['observations'].append(blob(startroot,row))
                    startup['payloads']={str(p.relative_to(startroot)):q.digest(p.read_bytes()) for p in startroot.rglob('*') if p.is_file()}
                    (startroot/'capture.json').write_bytes(encoded(startup));entry['startup']={'path':str((startroot/'capture.json').relative_to(directory)),
                        'sha256':q.digest(encoded(startup))}
                captured['cells'].append(entry)
            (directory/'run.json').write_bytes(encoded(captured));ref={'path':str((directory/'run.json').relative_to(f.out)),'sha256':q.digest(encoded(captured))}
            self.stage['runs'].append(ref);self.scope_runs.append((directory,captured))
        self.bindings={'repo':f.repo,'head':'HEAD','source':f.source,'source_reference':self.stage['source'],'build':f.build,'build_reference':self.stage['build']}
        self.manifest=manifest
    def refresh(self):
        for ref,(directory,captured) in zip(self.stage['runs'],self.scope_runs):
            (directory/'run.json').write_bytes(encoded(captured));ref['sha256']=q.digest(encoded(captured))
    def verify(self):
        self.refresh()
        return run.validate_serving_run(self.stage,q.Evidence(self.f.out),self.manifest,self.bindings)


class OuterBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.b=BindingFixture(Path(self.tmp.name))
        self.adapter=patch('serving_run.replay_cell',return_value={'verified_payloads':1});self.mock=self.adapter.start();self.addCleanup(self.adapter.stop)
    def test_complete_outer_binding_accounts_every_source_scope_cell(self):
        value=self.b.verify();self.assertEqual(len(value['cells']),22);self.assertEqual(self.mock.call_count,22)
    def test_skip_missing_duplicate_or_extra_cell_refuses(self):
        saved=copy.deepcopy(self.b.scope_runs[0][1]['cells'])
        for change in (lambda c:c.pop(),lambda c:c.append(c[0]),lambda c:c[0].update(state='skipped'),
                       lambda c:c[0].update(skip=True),lambda c:c[0].update(scenario='unknown')):
            self.b.scope_runs[0][1]['cells']=copy.deepcopy(saved);change(self.b.scope_runs[0][1]['cells'])
            with self.assertRaises((ServingGateError,q.GateError)):self.b.verify()
    def test_source_elf_model_controller_hardware_and_owner_substitution_refuse(self):
        record=self.b.scope_runs[0][1];saved=copy.deepcopy(record)
        for key in ('source_after','binaries_after','models_after','controllers_after','hardware_after'):
            record.clear();record.update(copy.deepcopy(saved));record[key]={} if type(record[key]) is dict else '0'*64
            with self.subTest(key=key),self.assertRaises((ServingGateError,q.GateError)):self.b.verify()
        record.clear();record.update(saved);record['cells'][1]['runtime']['server_identity']=copy.deepcopy(record['cells'][0]['runtime']['server_identity'])
        with self.assertRaisesRegex(ServingGateError,'reused'):self.b.verify()
    def test_cell_outside_live_lease_and_live_lock_substitution_refuse(self):
        record=self.b.scope_runs[0][1];record['cells'][0]['started_ns']=0
        with self.assertRaises(ServingGateError):self.b.verify()
        record['cells'][0]['started_ns']=10_000_000_000
        folder=self.b.scope_runs[0][0];obs=q.Evidence(folder).obj(record['lease_before']);obs['locks'][0]['raw']='1: FLOCK ADVISORY WRITE 999 08:01:77 0 EOF'
        record['lease_before']=blob(folder,obs)
        with self.assertRaises(ServingGateError):self.b.verify()
    def test_raw_startup_or_lifecycle_substitution_refuses(self):
        record=self.b.scope_runs[0][1];cell=next(c for c in record['cells'] if c['scenario'] in run.PHASE)
        folder=self.b.scope_runs[0][0];path=folder/cell['startup']['path'];startup=q.json_bytes(path.read_bytes())
        startup['observations'].pop();path.write_bytes(encoded(startup));cell['startup']['sha256']=q.digest(encoded(startup))
        with self.assertRaisesRegex(ServingGateError,'probe denominator'):self.b.verify()
    def test_captured_server_cannot_come_from_another_host_boot(self):
        record=self.b.scope_runs[0][1];cell=record['cells'][0];folder=self.b.scope_runs[0][0]
        root=folder/cell['capture_root'];index=q.json_bytes((root/'capture.json').read_bytes())
        life=q.Evidence(root).obj(index['lifecycle']);life['boot_id']='edc35d2a-aedc-4528-8ab4-a91bf32dcb9c'
        index['lifecycle']=blob(root,life);(root/'capture.json').write_bytes(encoded(index))
        cell['capture']['sha256']=q.digest(encoded(index))
        with self.assertRaisesRegex(ServingGateError,'another host boot'):self.b.verify()
    def test_full_v2_requires_unchanged_generic_and_serving_identity(self):
        f=self.b.f;serving=self.b.verify();f.put('generic-record.json',f.record);f.put('serving/stage.json',self.b.stage)
        verdicts={'generic':f.verdicts,'serving':serving}
        record={'schema':'memra-release-qualification-v2','status':'qualified','source':f.ref('source.json'),'build':f.ref('build.json'),
            'generic':f.ref('generic-record.json'),'serving':f.ref('serving/stage.json'),'verdicts':verdicts}
        record['payloads']={str(p.relative_to(f.out)):q.digest(p.read_bytes()) for p in f.out.rglob('*') if p.is_file() and p.name!='record.json'}
        record['identity_sha256']=q.object_digest({'source':f.source['inputs_sha256'],'build':record['build']['sha256'],
            'generic':record['generic']['sha256'],'serving':record['serving']['sha256'],'verdicts':verdicts})
        result=q.validate_record(record,q.Evidence(f.out),f.repo,'HEAD')
        self.assertIn('required-release',result['qualification'])
        for field in ('serving','generic'):
            changed=copy.deepcopy(record);changed[field]['sha256']='0'*64
            with self.assertRaises(q.GateError):q.validate_record(changed,q.Evidence(f.out),f.repo,'HEAD')

if __name__=='__main__':unittest.main()
