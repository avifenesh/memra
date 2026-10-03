#!/usr/bin/env python3
"""Bounded native chat/text job acceptance, using stored/receipt events instead of wait polls."""
import argparse, copy, hashlib, json, math, os, re, secrets, socket, subprocess, threading, time, tomllib
import urllib.error, urllib.request
from pathlib import Path

DECODE_FIELDS={'temperature','top_p','top_k','min_p','seed','max_tokens','frequency_penalty','presence_penalty','repetition_penalty','reasoning','reasoning_effort','enable_thinking','chat_template_kwargs','include_reasoning'}
PHRASE='The quick brown fox jumps over the lazy dog while a quiet river flows past the old stone bridge.'

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()

def require(value,message):
    if not value:raise AssertionError(message)

def content(value,chat):
    if 'choices' in value:
        choice=value['choices'][0]
        return (choice.get('message',{}).get('content') or '') if chat else choice.get('text','')
    return value.get('text','')

def usage(value):
    if 'usage' in value:
        u=value['usage'];return {'prompt_tokens':u['prompt_tokens'],'cached_tokens':u.get('prompt_tokens_details',{}).get('cached_tokens',0),'completion_tokens':u['completion_tokens']}
    return {'prompt_tokens':value['prompt_tokens'],'cached_tokens':value['cached_tokens'],'completion_tokens':value['n_tokens']}

def verify_receipt(value,rows,chat,expected_outcome):
    require(len(rows)==1,'expected exactly one terminal callback')
    row=rows[0];require(row['outcome']==expected_outcome,'wrong terminal callback')
    u=usage(value)
    require(all(row[k]==v for k,v in u.items()),'terminal usage differs from delivered result')
    if chat:
        require(value.get('object')=='chat.completion','wrong chat envelope')
        require(value['choices'][0]['message']['role']=='assistant','missing assistant role')
    elif 'choices' in value:require(value.get('object')=='text_completion','wrong text envelope')
    else:
        require(isinstance(value.get('tokens'),list),'native token ids missing')
        require(len(value['tokens'])==u['completion_tokens'],'native token count mismatch')
    require(content(value,chat) or (chat and value['choices'][0]['message'].get('reasoning')),'empty completion')
    return u

def vendor_trace(log,profile,start_line=None):
    require('[server] listening on' in log,'missing listener boundary')
    lines=log.splitlines()
    listener=next(i for i,line in enumerate(lines) if '[server] listening on' in line)
    if start_line is not None:require(start_line>listener,'parameter witness begins before listener')
    live='\n'.join(lines[max(listener+1,start_line or 0):])
    bursts=[line for line in live.splitlines() if line.startswith('[skey] burst sampled=1 ')]
    require(bursts,'no native sampled parameter witness after listener')
    expected={'temp':profile['default_temperature'],'top_k':profile['default_top_k'],'top_p':profile['default_top_p'],'min_p':profile['default_min_p'],'pen_on':1}
    matched=[]
    for line in bursts:
        values={k:re.search(r'\b'+k+r'=([^ ]+)',line) for k in expected}
        require(all(values[k] and math.isclose(float(values[k][1]),v,abs_tol=1e-6) for k,v in expected.items()),'native sampled burst differs from fixture profile')
        matched.append(line)
    require(matched,'actual sampled parameters do not match thinking profile')
    return {'resolved_arm':'primary_thinking','observed':expected,'matching_bursts':len(matched),'first_http_burst':matched[0]}

def verify_deadline(started,rows):
    require(len(rows)==1 and rows[0]['observed_monotonic']-started>90,'native delivery did not exceed synchronous deadline')

def verify_progress(id,progress):
    require(progress and progress[0]['id']==id and progress[0]['completion_tokens']==32,'cancel lacked native progress')

class Server:
    def __init__(self,args,out,port,door=True,compat='native',context=32768):
        self.args,self.out,self.port=args,Path(out),port;self.out.mkdir(parents=True)
        self.base=f'http://127.0.0.1:{port}'
        self.cv=threading.Condition();self.receipts={};self.stored={};self.progress={};self.unsettled=[];self.lines=[]
        self.exited=False;self.listening=False;self.proc=None;self.monitor=None;self.reader=None;self.log=None;self.telemetry=None
        self.keys={name:secrets.token_hex(24) for name in ['owner','rotated','foreign']}
        env={k:v for k,v in os.environ.items() if not k.startswith('MEMRA_') or k in {'MEMRA_GPU_LOCK','MEMRA_CI_LOCK','MEMRA_CI_LOCK_HELD','MEMRA_RIG_LOCK_FD'}}
        ring=','.join(('acme' if k!='foreign' else 'blue')+':'+hashlib.sha256(v.encode()).hexdigest() for k,v in self.keys.items())
        env.update(MEMRA_MODELS='q9='+str(args.model.resolve()),MEMRA_CTX=str(context),MEMRA_ADDR=f'127.0.0.1:{port}',MEMRA_BACKGROUND_RESPONSES='1' if door else '0',MEMRA_PREFIX_CACHE_MB='0',MEMRA_REUSE_POOL='0',MEMRA_AFFINITY='0',MEMRA_API_KEYS=ring,MEMRA_COMPAT=compat,MEMRA_MODEL_METADATA=str(args.metadata.resolve()),MEMRA_SKEY_PROBE='1',MEMRA_DEBUG_SPEC='1')
        self.env=env
        public={k:('fixture credentials omitted' if k=='MEMRA_API_KEYS' else v) for k,v in env.items() if k.startswith('MEMRA_')}
        (self.out/'environment.json').write_text(json.dumps(public,indent=2)+'\n')
        self.http=(self.out/'http.jsonl').open('w')
    def __enter__(self):
        try:return self.start()
        except BaseException:
            self.__exit__(RuntimeError,None,None)
            raise
    def start(self):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);probe.bind(('127.0.0.1',self.port))
        subprocess.run(['bash','tools/port-guard.sh','check','background-chat-text',str(self.port)],check=True)
        self.log=(self.out/'server.log').open('w');self.telemetry=(self.out/'gpu-250ms.csv').open('w')
        self.monitor=subprocess.Popen(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=timestamp,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw','--format=csv','--loop-ms=250'],stdout=self.telemetry,stderr=subprocess.STDOUT,pass_fds=(9,))
        self.proc=subprocess.Popen([str(self.args.binary.resolve())],env=self.env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,pass_fds=(9,))
        def reader():
            with (self.out/'receipts.jsonl').open('w') as raw:
                for line in self.proc.stdout:
                    at=time.monotonic();self.log.write(line);self.log.flush()
                    with self.cv:
                        self.lines.append(line)
                        if '[server] listening on '+self.base in line:self.listening=True
                        for prefix,target in [('GATE_RECEIPT ',self.receipts),('GATE_STORED ',self.stored),('GATE_PROGRESS ',self.progress)]:
                            if line.startswith(prefix):
                                row=json.loads(line[len(prefix):]);row['observed_monotonic']=at
                                target.setdefault(row['id'],[]).append(row)
                                raw.write(json.dumps({'event':prefix.strip(),**row})+'\n');raw.flush()
                        if line.startswith('GATE_UNSETTLED '):self.unsettled.append(line.strip())
                        self.cv.notify_all()
                with self.cv:self.exited=True;self.cv.notify_all()
        self.reader=threading.Thread(target=reader);self.reader.start()
        with self.cv:ready=self.cv.wait_for(lambda:self.listening or self.exited,timeout=180) and self.listening and not self.exited
        if not ready:
            raise RuntimeError('server readiness failed')
        try:
            subprocess.run(['bash','-c','source tools/port-guard.sh; memra_port_owned background-chat-text "$1" "$2"','_',str(self.port),str(self.proc.pid)],check=True)
        except Exception:
            raise
        require(any('[server] OpenRouter metadata loaded: 1 model(s), sha256 '+digest(self.args.metadata) in line for line in self.lines), 'server did not load the sealed fixture metadata')
        return self
    def event(self,table,id,timeout=600):
        with self.cv:
            require(self.cv.wait_for(lambda:id in table or self.exited,timeout=timeout),'native event deadline')
            require(id in table,'server exited before native event')
            return copy.deepcopy(table[id])
    def request(self,method,path,body=None,key='owner',timeout=30,stream=False):
        headers={'Content-Type':'application/json'}
        if key is not None:headers['Authorization']='Bearer '+self.keys[key]
        request=urllib.request.Request(self.base+path,data=None if body is None else json.dumps(body).encode(),method=method,headers=headers)
        start=time.monotonic()
        try:
            response=urllib.request.urlopen(request,timeout=timeout)
        except urllib.error.HTTPError as error:response=error
        with response:
            raw=response.read().decode();status=response.status;response_headers=dict(response.headers)
        record={'start_monotonic':start,'duration':time.monotonic()-start,'method':method,'path':path,'key_role':key,'request':body,'status':status,'response_headers':response_headers,'raw':raw}
        self.http.write(json.dumps(record)+'\n');self.http.flush()
        if stream:
            events=[json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: ') and line[6:]!='[DONE]']
            require(('[DONE]' in raw or re.search(r'(?m)^event: done\r?$',raw)) and not any(e.get('error') is not None for e in events),'stream missing normal terminal marker')
            text=''.join((e.get('choices',[{}])[0].get('delta',{}).get('content') or '') if body.get('messages') else (e.get('choices',[{}])[0].get('text') or e.get('text','')) for e in events if e.get('choices',None)!=[])
            result={'text':text,'events':events};ident=response_headers.get('x-request-id')
        else:
            result=json.loads(raw);ident=result.get('id') or response_headers.get('x-request-id')
        return status,result,ident,start
    def result(self,id):
        self.event(self.stored,id)
        status,value,_,_=self.request('GET','/v1/jobs/'+id)
        require(status==200 and value['status'] in ['completed','incomplete','cancelled','failed'],'job not terminal after publication')
        return value
    def __exit__(self,*_):
        if self.proc:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:self.proc.wait(timeout=20)
                except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait()
            if self.reader:self.reader.join(timeout=5)
        if self.log:self.log.close()
        if self.monitor:
            if self.monitor.poll() is None:self.monitor.terminate()
            self.monitor.wait(timeout=5)
        if self.telemetry:self.telemetry.close()
        self.http.close()
        (self.out/'final.json').write_text(json.dumps({'receipt_count':sum(map(len,self.receipts.values())),'terminal_counts':{id:len(rows) for id,rows in self.receipts.items()},'unsettled':self.unsettled,'server_exit':self.proc.returncode if self.proc else None},indent=2))
        if not _ or _[0] is None:
            require(not self.unsettled,'unsettled native receipt')
            require(all(len(rows)==1 for rows in self.receipts.values()),'late duplicate terminal callback')

def short_body(chat,bg=False):
    body={'model':'q9','max_tokens':64,'temperature':0,'top_p':1,'top_k':0,'min_p':0,'presence_penalty':0,'frequency_penalty':0,'repetition_penalty':1,'seed':7}
    if chat:body.update(messages=[{'role':'user','content':'What is two plus two? Answer with one number.'}],enable_thinking=False)
    else:body['prompt']='What is two plus two? Answer with one number.'
    if bg:body['background']=True
    return body

def main(args):
    os.fstat(9);require(os.path.samefile('/proc/self/fd/9',os.environ['MEMRA_GPU_LOCK']),'wrong inherited GPU lease')
    require(digest(args.model)==args.model_sha,'cached artifact hash changed')
    profile=tomllib.loads(args.metadata.read_text())['models']['q9']
    require(profile['non_thinking_sampling']=={'temperature':0.7,'top_p':0.8,'top_k':20,'min_p':0.0,'presence_penalty':1.5,'repetition_penalty':1.0},'non-thinking profile absent/wrong')
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    manifest={'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'tracked_diff_sha256':hashlib.sha256(subprocess.check_output(['git','diff','HEAD','--binary'])).hexdigest(),'binary_sha256':digest(args.binary),'model_sha256':args.model_sha,'metadata_sha256':digest(args.metadata),'collector_sha256':digest(Path(__file__)),'context':32768,'phase':args.phase,'gpu':subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True)}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    summary=[]
    if args.phase=='short':
        with Server(args,out/'off',args.port,door=False,context=8192) as server:
            for chat in [False,True]:
                path='/v1/chat/completions' if chat else '/v1/completions'
                status,value,_,_=server.request('POST',path,short_body(chat,True));require(status==400 and value['error']['param']=='background','OFF did not explicitly refuse')
        for compat,offset in [('native',1),('openai',2)]:
            with Server(args,out/compat,args.port+offset,compat=compat,context=8192) as server:
                for chat in ([False,True] if compat=='native' else [False]):
                    path='/v1/chat/completions' if chat else '/v1/completions'
                    status,sync,id,_=server.request('POST',path,short_body(chat));require(status==200,'sync control failed');verify_receipt(sync,server.event(server.receipts,id),chat,'complete')
                    status,ack,id,_=server.request('POST',path,short_body(chat,True));require(status==200 and ack['status']=='queued','background acknowledgement failed')
                    result=server.result(id);verify_receipt(result,server.event(server.receipts,id),chat,'complete')
                    require(content(sync,chat)==content(result,chat) and usage(sync)==usage(result),'sync/background result identity failed')
                    for method,suffix in [('GET',''),('POST','/cancel')]:
                        require(server.request(method,'/v1/jobs/'+id+suffix,key='foreign')[0]==404,'foreign terminal ownership leak')
                    require(server.request('POST','/v1/jobs/'+id+'/cancel')[0]==409,'terminal cancel was not refused')
                    require(server.request('GET','/v1/jobs/'+id,key='rotated')[1]==result,'same-tenant key rotation failed')
                    summary.append({'cell':'short-identity','chat':chat,'compat':compat,'usage':usage(result)})
                    if compat=='native':
                        body={'model':'q9','background':True}
                        if chat:body['messages']=[{'role':'user','content':'What is two plus two? Answer briefly.'}]
                        else:body['prompt']='What is two plus two? Answer briefly.'
                        require(not DECODE_FIELDS.intersection(body),'bare probe contains decode overrides')
                        trace_start=len(server.lines)
                        status,ack,id,_=server.request('POST',path,body);require(status==200,'bare job refused')
                        result=server.result(id);verify_receipt(result,server.event(server.receipts,id),chat,'complete')
                        summary.append({'cell':'bare-default','chat':chat,'request':body,'usage':usage(result),'mode':'artifact/template default','parameter_proof':vendor_trace(''.join(server.lines),profile,trace_start),'trace_start_line':trace_start})
                if compat=='native':
                    summary.append({'cell':'default-parameters','proof':vendor_trace(''.join(server.lines),profile),'metadata_sha256':digest(args.metadata)})
    else:
        chat=args.phase=='long-chat';path='/v1/chat/completions' if chat else '/v1/completions'
        with Server(args,out/args.phase,args.port,context=32768) as server:
            body={'model':'q9','max_tokens':24000,'temperature':0,'top_p':1,'top_k':0,'min_p':0,'presence_penalty':0,'frequency_penalty':0,'repetition_penalty':1,'seed':7}
            if chat:
                body.update(messages=[{'role':'user','content':'Return the requested JSON array exactly. No commentary.'}],enable_thinking=False,response_format={'type':'json_schema','json_schema':{'name':'long_result','strict':True,'schema':{'type':'array','items':{'type':'string','const':PHRASE},'minItems':600,'maxItems':600}}})
            else:body['prompt']='Write a JSON array containing exactly 600 copies of this string, without abbreviating or skipping any copy: '+json.dumps(PHRASE)+'. Return only the array. Start now: ['
            # Streaming is the independent complete-result path, outside background delivery.
            status,golden,gid,_=server.request('POST',path,{**body,'stream':True},timeout=1000,stream=True);require(status==200,'streaming reference failed')
            server.event(server.receipts,gid)
            status,ack,id,started=server.request('POST',path,{**body,'background':True});require(status==200 and ack['status']=='queued','long acknowledgement failed')
            require(time.monotonic()-started<30,'queued acknowledgement took synchronous-length time')
            for method,suffix in [('GET',''),('POST','/cancel')]:require(server.request(method,'/v1/jobs/'+id+suffix,key='foreign')[0]==404,'foreign running ownership leak')
            result=server.result(id);rows=server.event(server.receipts,id)
            verify_receipt(result,rows,chat,'complete')
            verify_deadline(started,rows)
            require(content(result,chat)==golden['text'],'stored result differs from independent streaming reference')
            require(server.request('GET','/v1/jobs/'+id,key='rotated')[1]==result,'terminal result changed')
            if chat:require(json.loads(content(result,True))==[PHRASE]*600,'accepted long schema result was not preserved')
            summary.append({'cell':args.phase,'seconds':rows[0]['observed_monotonic']-started,'usage':usage(result),'status':result['status'],'full_text_sha256':hashlib.sha256(content(result,chat).encode()).hexdigest()})
            # Start the same native program again; cancel after its recorded 32nd token.
            status,ack,cid,_=server.request('POST',path,{**body,'background':True});require(status==200,'cancel submission failed')
            progress=server.event(server.progress,cid,timeout=120);verify_progress(cid,progress)
            status,cancelled,_,_=server.request('POST','/v1/jobs/'+cid+'/cancel');require(status==200 and cancelled['status']=='cancelled','live cancel did not settle')
            server.event(server.stored,cid);verify_receipt(cancelled,server.event(server.receipts,cid),chat,'cancel_partial')
            require(usage(cancelled)['completion_tokens']>=32,'cancel ran before observed native progress')
            require(content(result,chat).startswith(content(cancelled,chat)),'cancel lost original generated prefix')
            require(server.request('POST','/v1/jobs/'+cid+'/cancel')[0]==409,'repeat cancel should conflict')
            summary.append({'cell':'cancel','chat':chat,'usage':usage(cancelled),'prefix_preserved':True})
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['short','long-chat','long-text'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--binary',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--model-sha',required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--port',type=int,required=True)
    main(p.parse_args())
