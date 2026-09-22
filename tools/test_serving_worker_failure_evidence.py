"""Synthetic Linux-shaped replay controls, separate from actual POSIX capture tests."""
import copy,hashlib,json,unittest
from serving_capture import encoded
from serving_worker_failure import evaluate_worker_failure_cell
from serving_worker_failure_evidence import read_worker_failure_capture
from serving_release import ServingGateError,account_attempts
from test_serving_worker_failure import bundle,ns
from test_serving_policy import BOOT,PROCESS,drain

class Packet:
    def __init__(self):
        b=bundle();self.required=b['required'];self.program=b['program'];self.blobs={};self.capture={'payloads':{}}
        self.identity={'pid':100,'start_identity':BOOT+':1234'};self.program['server_identity']=dict(self.identity)
        for row in b['attempts']+b['probes']:row['server_identity']=dict(self.identity)
        self.launch={'argv':['/fixture/server'],'cwd':'/fixture','env':b['launch']['env'],
            'timeouts':{'startup':1,'overall':12,'drain':3,'kill':1},'output_path':'/fixture/output.log'}
        self.identities={'server_binary':{'bytes':7,'sha256':hashlib.sha256(b'fixture').hexdigest()}}
        self.log_identity={'path':'/fixture/output.log','device':1,'inode':2,'controller_pid':80}
        core={'schema':'memra-owned-server-v1','state':'ready','argv':self.launch['argv'],'cwd':'/fixture',
            'output_path':self.launch['output_path'],'env_keys':sorted(self.launch['env']),
            'env_sha256':hashlib.sha256(json.dumps(self.launch['env'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'timeouts':self.launch['timeouts'],'server':dict(PROCESS),'boot_id':BOOT,'started_monotonic':.001,
            'supervisor':{'pid':90,'ppid':80,'pgid':90,'start_time':'1200','identity_source':'linux_proc_start_ticks','state':'S'},
            'ready':{'monotonic':.005,'owner':dict(PROCESS),'method':'owned_output','details':{'synthetic':True}},
            'stop':None,'server_exit':None,'errors':[]}
        life=drain()['lifecycle'];life.update(copy.deepcopy(core));life['state']='finished'
        life['stop']={'reason':'worker_failure_capture_complete','monotonic':6.,'server_returncode_before_cleanup':None}
        life['server_exit']={'pid':100,'identity':dict(PROCESS),'observed_monotonic':7.,'before_cleanup':False,
            'returncode':0,'wait_status':0,'exit_code':0,'signal':None}
        life['signals']=[{'pid':100,'start_time':'1234','signal':15,'result':'sent','method':'pidfd',
                         'monotonic':6.05,'sent_monotonic':6.051,'finished_monotonic':6.052}]
        life['cleanup'].update(finished_monotonic=7.1,elapsed_s=1.1)
        life['ownership_observations'][0].update(first_seen_monotonic=.002,last_seen_monotonic=6.9)
        self.capture.update(schema='memra-worker-failure-capture-v1',state='captured',qualification=False,clock='monotonic_ns',
            started_ns=ns(.01),finished_ns=ns(7.5),program=self.obj(self.program),required=self.obj(self.required),
            identities_before=copy.deepcopy(self.identities),identities_after=copy.deepcopy(self.identities),process_observations=[],listeners=[],
            observations=[],probes=[],request_invocations=[],prefixes={},prefix_confirmed_ns=ns(1.2),log_chunks=[],log_observations=[],
            lifecycle=self.obj(life),stop_call=self.obj({'started_ns':ns(5.9),'finished_ns':ns(7.3),'reason':'worker_failure_capture_complete','error':None}),
            errors=[],client_errors={},unattempted_ids=[],unobserved_invoked_ids=[],
            request_denominator=[{'id':r['id'],'role':r['role'],'status':'captured'} for r in self.program['requests']])
        for label,start,end in [('before',.02,.03),('after',4.6,4.61)]:
            self.capture['process_observations'].append(self.obj({'label':label,'started_ns':ns(start),'finished_ns':ns(end),'receipt':core,'observed_identity':dict(PROCESS)}))
        for label,start,end in [('before',.04,.05),('after',4.5,4.51)]:
            proof={'method':'listener_identity','owner':dict(PROCESS),'details':{'schema':'memra-linux-listener-v1','clock':'monotonic_ns',
                'endpoint':self.program['endpoint'],'started_ns':ns(start),'finished_ns':ns(end),'boot_id':BOOT,'owner_before':dict(PROCESS),'owner_after':dict(PROCESS),
                'samples':[{'inodes':[123],'primary_fds':[9]},{'inodes':[123],'primary_fds':[9]}]}}
            self.capture['listeners'].append(self.obj({'label':label,'proof':proof}))
        self.capture['log_descriptor']=self.obj({**self.log_identity,'fd':9,'regular_file':True,'opened_before_ns':ns(.06),'opened_after_ns':ns(.061)})
        self.capture['log_baseline']=self.put(b['baseline_log']);self.capture['log_final']=self.capture['log_observed']=self.put(b['log'])
        old=0
        for raw,start,end in [(b['baseline_log'],.07,.08),(b['log'],7.4,7.41)]:
            self.capture['log_chunks'].append({'offset':old,'bytes':len(raw)-old,'raw':self.put(raw[old:])});old=len(raw)
            self.capture['log_observations'].append(self.obj({'started_ns':ns(start),'finished_ns':ns(end),'device':1,'inode':2,'size_before':len(raw),
                'size_after':len(raw),'named_device':1,'named_inode':2,'named_after_device':1,'named_after_inode':2,'bytes':len(raw),'complete_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}))
        reqs={r['id']:r for r in self.program['requests']}
        for row in b['attempts']+b['probes']:
            record={**row,'body':self.put(row['body'])}
            self.capture['observations' if row['id'] in reqs else 'probes'].append(self.obj(record))
            body=encoded(reqs[row['id']]['payload']) if row['id'] in reqs else b''
            self.capture['request_invocations'].append(self.obj({'id':row['id'],'method':row['method'],'path':row['path'],'body':self.put(body),'started_ns':row['started_ns']-1}))
        for name,prefix in b['prefixes'].items():self.capture['prefixes'][name]=self.obj({**prefix,'body':self.put(prefix['body'])})
        self.capture['wire_accounting']=self.obj(account_attempts(self.program['requests'],b['attempts']))
        self.capture['facts']=self.obj(evaluate_worker_failure_cell(self.required,self.program,launch=self.launch,attempts=b['attempts'],probes=b['probes'],prefixes=b['prefixes'],log=b['log'],baseline_log=b['baseline_log']))
    def put(self,data):
        digest=hashlib.sha256(data).hexdigest();name='blobs/'+digest;self.blobs[name]=data;self.capture['payloads'][name]=digest
        return {'path':name,'sha256':digest}
    def obj(self,value):return self.put(encoded(value))
    def value(self,ref):return json.loads(self.blobs[ref['path']])
    def change(self,ref,mutate):
        value=self.value(ref);mutate(value);return self.obj(value)
    def evaluate(self,**kwargs):
        raw=encoded(self.capture)
        options={'expected_capture_sha256':hashlib.sha256(raw).hexdigest(),'expected_required':self.required,'expected_program':self.program,
                 'expected_server':self.launch,'expected_identities':self.identities,'expected_log_identity':self.log_identity}
        return read_worker_failure_capture(raw,self.blobs.__getitem__,**{**options,**kwargs})

class WorkerFailureEvidenceTests(unittest.TestCase):
    def test_complete_bound_replay_and_completion_order(self):
        p=Packet();p.capture['observations'].reverse();before=copy.deepcopy(p.capture)
        result=p.evaluate();self.assertFalse(result['qualification']);self.assertEqual(result['facts']['planned'],3)
        self.assertEqual([r['id'] for r in result['attempts']],['victim','trigger','recovery']);self.assertEqual(p.capture,before)
    def test_full_raw_census_digest_and_external_identity_are_required(self):
        for field in ['observations','probes','process_observations','listeners','request_invocations','log_observations']:
            p=Packet();p.capture[field].pop()
            with self.subTest(field=field),self.assertRaises(ServingGateError):p.evaluate()
        p=Packet();ref=p.put(b'diagnostic');p.blobs[ref['path']]=b'altered'
        with self.assertRaises(ServingGateError):p.evaluate()
        with self.assertRaises(ServingGateError):Packet().evaluate(expected_capture_sha256='0'*64)
        p=Packet();p.capture['identities_before']['server_binary']['sha256']='0'*64;p.capture['identities_after']['server_binary']['sha256']='0'*64
        with self.assertRaises(ServingGateError):p.evaluate()
    def test_launch_program_scope_log_and_ownership_substitution_refuse(self):
        for change in [lambda p:p.launch['env'].update(MEMRA_PANIC_AFTER='2'),lambda p:p.program['requests'][1]['payload'].update(max_tokens=100),
                       lambda p:p.log_identity.update(inode=22)]:
            p=Packet();change(p)
            with self.assertRaises(ServingGateError):p.evaluate()
        p=Packet();p.capture['process_observations'][0]=p.change(p.capture['process_observations'][0],lambda r:r['observed_identity'].update(start_time='999'))
        with self.assertRaises(ServingGateError):p.evaluate()
    def test_missing_prefix_or_terminal_and_false_clean_summaries_refuse(self):
        p=Packet();p.capture['prefixes'].clear()
        with self.assertRaises(ServingGateError):p.evaluate()
        for field in ['observations','probes']:
            p=Packet();p.capture[field][0]=p.change(p.capture[field][0],lambda r:r.update(body=p.put(b'{}')))
            with self.assertRaises(ServingGateError):p.evaluate()
        p=Packet();p.capture['facts']=p.obj({'passed':True})
        with self.assertRaises(ServingGateError):p.evaluate()
    def test_failed_or_incomplete_cleanup_and_post_exit_http_refuse(self):
        p=Packet();p.capture['state']='failed'
        with self.assertRaises(ServingGateError):p.evaluate()
        p=Packet();p.capture['lifecycle']=p.change(p.capture['lifecycle'],lambda r:r['cleanup'].update(escalated=True))
        with self.assertRaises(ServingGateError):p.evaluate()
        p=Packet();p.capture['request_invocations'][0]=p.change(p.capture['request_invocations'][0],lambda r:r.update(started_ns=ns(8)))
        with self.assertRaises(ServingGateError):p.evaluate()

    def test_self_consistent_early_error_prefix_refuses_at_terminal_boundary(self):
        p=Packet();row=p.value(p.capture['observations'][0])
        body=p.blobs[row['body']['path']].split(b'\n\n',1)[1]
        row['body']=p.put(body);row['chunks'][0]['end_offset']=11;row['chunks'][-1]['end_offset']=len(body)
        p.capture['observations'][0]=p.obj(row)
        prefix=p.value(p.capture['prefixes']['victim']);prefix.update(body=p.put(body[:11]),end_offset=11)
        p.capture['prefixes']['victim']=p.obj(prefix)
        attempts=[]
        for ref in p.capture['observations']:
            value=p.value(ref);value['body']=p.blobs[value['body']['path']];attempts.append(value)
        p.capture['wire_accounting']=p.obj(account_attempts(p.program['requests'],attempts))
        with self.assertRaisesRegex(ServingGateError,'prefix already touches terminal'):p.evaluate()

if __name__=='__main__':unittest.main()
