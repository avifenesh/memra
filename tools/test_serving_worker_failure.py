"""Focused source-contract tests with explicit synthetic observations, no native proof."""
import copy,json,unittest
from pathlib import Path
from serving_worker_failure import (PANIC_LINE,RESPAWN_LINE,CLOSED_MESSAGE,DETAIL,validate_program,
                                   evaluate_worker_failure_cell)
from serving_release import ServingGateError

OWNER={'pid':17,'start_identity':'synthetic-boot:71'}
def ns(v):return int(v*1e9)
def encode(v):return json.dumps(v,separators=(',',':')).encode()
def config():
    scope={'id':'fixture','model':'gate','route':'plain','profile':'text-generation-v1'}
    manifest=json.loads((Path(__file__).parent/'serving-release.cells.json').read_text())
    requirements=next(c['requirements'] for c in manifest['profiles']['text-generation-v1'] if c['id']=='worker_failure_recovery')
    required={'id':'fixture/worker_failure_recovery','scope':scope,'scenario':'worker_failure_recovery','requirements':requirements}
    requests=[]
    for name,role in [('victim','victim'),('trigger','trigger'),('recovery','recovery')]:
        row={'id':name,'role':role,'model':'gate','path':'/v1/chat/completions','wire':'chat_sse' if role=='victim' else 'chat_json',
             'payload':{'model':'gate','messages':[{'role':'user','content':'fixture'}],'stream':role=='victim','fixture_role':role}}
        if role!='victim':row.update(prompt_tokens={'min':1,'max':32},completion_tokens={'min':1,'max':16})
        requests.append(row)
    return required,{'schema':'memra-worker-failure-program-v1','mode':'worker_failure_recovery','cell_id':required['id'],'scope':scope,
        'server_identity':dict(OWNER),'endpoint':{'host':'127.0.0.1','port':12345},'identities':{'server_binary':'/fixture/server'},
        'http':{'connect_timeout':.5,'read_timeout':3,'wall_timeout':5,'max_body_bytes':16384},
        'timing':{'prefix_timeout_s':2,'fault_timeout_s':2,'recovery_timeout_s':3,'poll_s':.01,'overall_s':12},'requests':requests}

def observation(name,start,end,status,body,path='/v1/chat/completions',method='POST',retry=False):
    return {'id':name,'started_ns':ns(start),'finished_ns':ns(end),'status':status,'body':body,'method':method,'path':path,
            'server_identity':dict(OWNER),'transport_error':None,'headers':[['Retry-After','2'],['retry-after-ms','2000']] if retry else [],
            'first_body_byte_ns':ns(start+.01),'chunks':[{'observed_ns':ns(start+.01),'end_offset':len(body)}]}

def bundle():
    required,program=config()
    prefix=b'data: {"model":"gate","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}\n\n'
    error=b'data: '+encode({'error':{'message':CLOSED_MESSAGE,'type':'server_error','param':None,'code':'overloaded'}})+b'\n\ndata: [DONE]\n\n'
    clean=encode({'model':'gate','choices':[{'index':0,'finish_reason':'stop','message':{'content':'completed','role':'assistant'}}],
                  'usage':{'prompt_tokens':12,'completion_tokens':2,'total_tokens':14}})
    victim=observation('victim',1,2.2,200,prefix+error)
    victim['chunks']=[{'observed_ns':ns(1.1),'end_offset':len(prefix)},{'observed_ns':ns(2.19),'end_offset':len(prefix+error)}]
    victim['first_body_byte_ns']=ns(1.1)
    attempts=[victim,observation('trigger',1.5,1.9,200,clean),observation('recovery',4,4.2,200,clean)]
    probes=[]
    for path,t,status,g,phase in [('/health',.5,200,0,'idle'),('/readyz',.6,200,0,'idle'),
            ('/health',1.95,503,0,'dead'),('/readyz',2.05,503,0,'dead'),
            ('/health',2.4,503,1,'loading'),('/readyz',2.5,503,1,'loading'),
            ('/health',3.1,200,1,'idle'),('/readyz',3.2,200,1,'idle'),
            ('/health',4.3,200,1,'idle'),('/readyz',4.4,200,1,'idle')]:
        value={'status':('ok' if path=='/health' else 'ready') if status==200 else ('unhealthy' if path=='/health' else 'not_ready'),
               'models':['gate'],'worker':{'generation':g,'phase':phase}}
        if status==503:value['detail']=DETAIL
        probes.append(observation('wf-probe-'+str(len(probes)+1),t,t+.01,status,encode(value),path,'GET',status==503))
    return {'required':required,'program':program,'launch':{'env':{'MEMRA_PANIC_AFTER':'1','MEMRA_WORKER_RESPAWN':'1'}},
        'attempts':attempts,'probes':probes,'prefixes':{'victim':{'observed_ns':ns(1.1),'end_offset':len(prefix),'body':prefix}},
        'baseline_log':b'READY\n','log':('READY\n'+PANIC_LINE+'\n'+RESPAWN_LINE+'\n').encode()}

class WorkerFailureTests(unittest.TestCase):
    def reject(self,b):
        with self.assertRaises((ServingGateError,ValueError)):evaluate_worker_failure_cell(**b)
    def test_complete_worker_recovery_counts_driver_errors_and_recovery(self):
        b=bundle();before=copy.deepcopy(b);r=evaluate_worker_failure_cell(**b)
        self.assertEqual(r['wire_counts'],{'clean_success':2,'typed_error':1})
        self.assertEqual(r['generation_observations']['respawns_observed'],1)
        self.assertEqual(r['typed_failures'][0]['error_code'],'overloaded')
        self.assertFalse(r['qualification']);self.assertEqual(b,before)
    def test_storage_order_does_not_change_facts(self):
        a=bundle();b=copy.deepcopy(a);b['attempts'].reverse()
        self.assertEqual(evaluate_worker_failure_cell(**a),evaluate_worker_failure_cell(**b))
    def test_missing_duplicate_skip_or_weakened_program_refuses(self):
        for change in [lambda b:b['attempts'].pop(),lambda b:b['attempts'].append(b['attempts'][0]),
                       lambda b:b['program'].update(skip=True),lambda b:b['required']['requirements'].update(typed_failure=False),
                       lambda b:b['program']['requests'][0].update(id='wf-probe-1')]:
            b=bundle();change(b);self.reject(b)
    def test_injection_and_request_fault_are_not_interchangeable(self):
        for env in [{},{'MEMRA_PANIC_AFTER':'2','MEMRA_WORKER_RESPAWN':'1'},
                    {'MEMRA_PANIC_AFTER':'1','MEMRA_WORKER_RESPAWN':'0'},
                    {'MEMRA_PANIC_AFTER':'1','MEMRA_WORKER_RESPAWN':'1','MEMRA_FAULT_INJECT_CACHE_SALT':'x'}]:
            b=bundle();b['launch']['env']=env;self.reject(b)
        b=bundle();b['attempts'][0]['body']=b['attempts'][0]['body'].replace(b'overloaded',b'worker_fault');self.reject(b)
    def test_exact_terminal_error_does_not_accept_truncation_success_or_late_frames(self):
        for mutate in [lambda raw:raw.replace(b'data: [DONE]\n\n',b''),lambda raw:raw+b'data: {}\n\n',
                       lambda raw:raw.replace(b'"finish_reason":null',b'"finish_reason":"stop"'),
                       lambda raw:raw.replace(CLOSED_MESSAGE.encode(),b'worker unavailable'),
                       lambda raw:raw.replace(b'"param":null',b'"param":"x"')]:
            b=bundle();b['attempts'][0]['body']=mutate(b['attempts'][0]['body']);self.reject(b)
        b=bundle();b['attempts'][0]['transport_error']={'kind':'cancelled','message':'client left'};self.reject(b)
    def test_unrelated_health_failure_wrong_generation_and_second_failure_refuse(self):
        for idx,key,value in [(2,'detail','GPU fault'),(2,'status','ok')]:
            b=bundle();body=json.loads(b['probes'][idx]['body']);body[key]=value;b['probes'][idx]['body']=encode(body);self.reject(b)
        for generation in (0,2,True):
            b=bundle()
            for row in b['probes'][4:]:
                value=json.loads(row['body']);value['worker']['generation']=generation;row['body']=encode(value)
            self.reject(b)
        b=bundle();b['probes'][-1]=copy.deepcopy(b['probes'][2]);self.reject(b)
    def test_health_and_request_owner_routes_and_transport_cannot_be_substituted(self):
        for key,value in [('server_identity',{'pid':18,'start_identity':'other'}),('path','/metrics'),
                          ('transport_error',{'kind':'read','message':'framing'})]:
            b=bundle();b['probes'][2][key]=value;self.reject(b)
        b=bundle();b['attempts'][-1]['server_identity']['pid']=18;self.reject(b)
    def test_prefix_and_recovery_chronology_are_nonvacuous(self):
        for change in [lambda b:b['prefixes']['victim'].update(observed_ns=ns(2)),
                       lambda b:b['attempts'][0].update(finished_ns=ns(1.4)),
                       lambda b:b['attempts'][-1].update(started_ns=ns(2)),
                       lambda b:b['probes'].pop(),lambda b:b['attempts'][1].update(body=b'{}')]:
            b=bundle();change(b);self.reject(b)
    def test_panic_and_respawn_log_identity_order_and_one_shot_are_required(self):
        for log in [b'READY\n',('READY\n'+RESPAWN_LINE+'\n'+PANIC_LINE+'\n').encode(),
                    ('READY\n'+PANIC_LINE+'\n'+RESPAWN_LINE+'\n'+PANIC_LINE+'\n').encode(),
                    ('READY\n'+PANIC_LINE.replace('=1','=2')+'\n'+RESPAWN_LINE+'\n').encode()]:
            b=bundle();b['log']=log;self.reject(b)
    def test_error_fallback_is_retained_but_does_not_replace_quoted_fault_proof(self):
        b=bundle();value=json.loads(b['probes'][4]['body']);value['detail']='worker fault';b['probes'][4]['body']=encode(value)
        self.assertEqual(evaluate_worker_failure_cell(**b)['planned'],3)
        for row in b['probes'][2:4]:
            value=json.loads(row['body']);value['detail']='worker fault';row['body']=encode(value)
        self.reject(b)

    def test_first_complete_probe_pair_defines_observation_deadlines(self):
        b=bundle();b['program']['timing']['fault_timeout_s']=.7
        later=copy.deepcopy(b['probes'][2]);later.update(started_ns=ns(2.35),finished_ns=ns(2.36))
        b['probes'].insert(4,later)
        for i,row in enumerate(b['probes'],1):row['id']='wf-probe-'+str(i)
        self.assertEqual(evaluate_worker_failure_cell(**b)['planned'],3)
        # First health fits, but the matching readiness observation does not.
        b=bundle();b['program']['timing']['fault_timeout_s']=.5;self.reject(b)
        b=bundle();b['program']['timing']['recovery_timeout_s']=1.2;self.reject(b)

    def test_malformed_pre_error_delta_is_not_valid_partial_output(self):
        from serving_worker_failure import typed_victim
        for replacement in (b'17',b'{}',b'[]'):
            row=bundle()['attempts'][0]
            row['body']=row['body'].replace(b'"partial"',replacement)
            with self.subTest(replacement=replacement),self.assertRaises(ServingGateError):typed_victim(row,'gate')

    def test_fragment_of_terminal_error_before_driver_refuses(self):
        for with_nonterminal in (False,True):
            b=bundle();row=b['attempts'][0]
            boundary=row['body'].index(b'data: {"error"')
            if not with_nonterminal:row['body']=row['body'][boundary:];boundary=0
            end=boundary+11
            row['chunks'][0]['end_offset']=end;row['chunks'][-1]['end_offset']=len(row['body'])
            b['prefixes']['victim'].update(body=row['body'][:end],end_offset=end)
            with self.subTest(with_nonterminal=with_nonterminal):self.reject(b)
        # A first nonterminal chunk cannot hide an early error in a later read.
        b=bundle();b['attempts'][0]['chunks'][1]['observed_ns']=ns(1.2)
        with self.assertRaisesRegex(ServingGateError,'error bytes were observed before'):evaluate_worker_failure_cell(**b)

    def test_nonterminal_fragments_and_original_byte_boundaries_survive(self):
        for newline in (b'\n',b'\r\n',b'\r'):
            for bom in (b'',b'\xef\xbb\xbf'):
                b=bundle();row=b['attempts'][0]
                row['body']=bom+row['body'].replace(b'partial','\u00e9-partial'.encode()).replace(b'\n',newline)
                boundary=row['body'].index(b'data: {"error"')
                splits=(11,row['body'].index('\u00e9'.encode())+1,boundary)
                for end in splits:
                    row['chunks'][0]['end_offset']=end;row['chunks'][-1]['end_offset']=len(row['body'])
                    b['prefixes']['victim'].update(body=row['body'][:end],end_offset=end)
                    with self.subTest(newline=newline,bom=bom,end=end):
                        self.assertEqual(evaluate_worker_failure_cell(**b)['planned'],3)

if __name__=='__main__':unittest.main()
