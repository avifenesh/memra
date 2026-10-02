#!/usr/bin/env python3
"""Short native composition check: HTTP acknowledgement, job terminal callbacks and metrics."""
import argparse, hashlib, json, os, secrets, subprocess, sys, threading, urllib.error, urllib.request
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'tools'))
from prometheus_metrics import parse_samples,scalar
p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--port',type=int,default=18118);p.add_argument('--promtool',required=True);p.add_argument('--external-lock',type=int,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
assert os.path.samefile(f'/proc/self/fd/{a.external_lock}',os.environ['MEMRA_GPU_LOCK'])
subprocess.run(['bash','tools/port-guard.sh','check','metrics-background',str(a.port)],check=True)
def sha(path):
 with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
(a.out/'identity.json').write_text(json.dumps({'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_diff_sha256':hashlib.sha256(subprocess.check_output(['git','diff','HEAD','--binary'])).hexdigest(),'binary_sha256':sha(a.binary),'model_sha256':sha(a.model),'driver_sha256':sha(__file__),'model_bytes':a.model.stat().st_size,'context':8192,'sampling':'explicit temperature 0','cache':'disabled; reuse and affinity disabled','external_lock':a.external_lock},indent=2))
api,operator=secrets.token_hex(24),secrets.token_hex(24);model='metrics-bg';base=f'http://127.0.0.1:{a.port}'
env={k:v for k,v in os.environ.items() if not k.startswith('MEMRA_') or k in {'MEMRA_GPU_LOCK','MEMRA_CI_LOCK','MEMRA_CI_LOCK_HELD','MEMRA_RIG_LOCK_FD'}}
env.update(MEMRA_MODELS=model+'='+str(a.model.resolve()),MEMRA_ADDR=f'127.0.0.1:{a.port}',MEMRA_CTX='8192',MEMRA_MAX_SESSIONS='1',MEMRA_PREFIX_CACHE_MB='0',MEMRA_REUSE_POOL='0',MEMRA_AFFINITY='0',MEMRA_BACKGROUND_RESPONSES='1',MEMRA_API_KEY=api,MEMRA_METRICS_TOKEN=operator)
(a.out/'environment.json').write_text(json.dumps({k:('fixture credential omitted' if k in {'MEMRA_API_KEY','MEMRA_METRICS_TOKEN'} else v) for k,v in env.items() if k.startswith('MEMRA_')},indent=2))
condition=threading.Condition();records={};progress=set();stored={};unsettled=[];listening=False;ended=False
log=open(a.out/'server.log','w');telemetry=open(a.out/'gpu-250ms.csv','w')
monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,uuid,name,memory.used,memory.free,utilization.gpu,temperature.gpu','--format=csv','--loop-ms=250'],stdout=telemetry,stderr=subprocess.STDOUT,pass_fds=(a.external_lock,))
server=subprocess.Popen([str(a.binary.resolve())],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,pass_fds=(a.external_lock,))
def read_log():
 global listening,ended
 for line in server.stdout:
  log.write(line);log.flush()
  with condition:
   if '[server] listening on '+base in line:listening=True
   if line.startswith('GATE_RECEIPT '):
    row=json.loads(line.split(' ',1)[1]);records.setdefault(row['id'],[]).append(row)
   elif line.startswith('GATE_STORED '):
    row=json.loads(line.split(' ',1)[1]);stored[row['id']]=row
   elif line.startswith('GATE_PROGRESS '):progress.add(json.loads(line.split(' ',1)[1])['id'])
   elif line.startswith('GATE_UNSETTLED '):unsettled.append(line.strip())
   condition.notify_all()
 with condition:ended=True;condition.notify_all()
reader=threading.Thread(target=read_log);reader.start();http=[];checks={}
def wait_event(predicate,timeout):
 with condition:
  assert condition.wait_for(lambda:predicate() or ended,timeout) and predicate(),'native event deadline; inspect server.log'
def request(method,path,body=None):
 req=urllib.request.Request(base+path,method=method,data=None if body is None else json.dumps(body).encode(),headers={'Authorization':'Bearer '+api,'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(req,timeout=60) as r:status,value=r.status,json.loads(r.read())
 except urllib.error.HTTPError as r:status,value=r.code,json.loads(r.read())
 http.append({'method':method,'path':path,'status':status,'body':value});assert status==200,(status,value);return value
def snap(label):
 req=urllib.request.Request(base+'/metrics',headers={'Authorization':'Bearer '+operator,'Accept':'text/plain'})
 with urllib.request.urlopen(req,timeout=15) as r:text=r.read().decode()
 (a.out/(label+'.prom')).write_text(text)
 result=subprocess.run([a.promtool,'check','metrics'],input=text,text=True,capture_output=True,timeout=30)
 (a.out/(label+'.promtool.log')).write_text(result.stdout+result.stderr);assert result.returncode==0,result.stdout+result.stderr
 return parse_samples(text)
labels={'model':model,'route':'hybrid','lane':'interactive'}
def value(samples,name,**extra):
 try:return scalar(samples,name,**labels,**extra)
 except KeyError:return 0
try:
 wait_event(lambda:listening,180)
 before=snap('before')
 short={'model':model,'input':'What is two plus two? Answer with one number.','max_output_tokens':64,'temperature':0,'reasoning':{'effort':'none'}}
 ack=request('POST','/v1/responses',dict(short,background=True));assert ack['status']=='queued'
 ident=ack['id'];wait_event(lambda:ident in stored,60)
 complete=request('GET','/v1/responses/'+ident);assert complete['status']=='completed',complete
 after_complete=snap('after-complete')
 assert len(records.get(ident,[]))==1 and records[ident][0]['outcome']=='complete'
 for _ in range(3):assert request('GET','/v1/responses/'+ident)==complete
 after_polls=snap('after-polls')
 checks['queued_ack_counts_once']=value(after_complete,'memra_requests_total',code='200')-value(before,'memra_requests_total',code='200')==1
 checks['generation_success_counts_once']=value(after_complete,'memra_e2e_seconds_count')-value(before,'memra_e2e_seconds_count')==1
 checks['polls_do_not_count_http_or_generation']=value(after_polls,'memra_requests_total',code='200')==value(after_complete,'memra_requests_total',code='200') and value(after_polls,'memra_e2e_seconds_count')==value(after_complete,'memra_e2e_seconds_count')
 checks['completion_tokens_match_callback']=records[ident][0]['completion_tokens']==complete['usage']['output_tokens']==value(after_complete,'memra_completion_tokens_sum')-value(before,'memra_completion_tokens_sum')
 phrase='The quick brown fox jumps over the lazy dog while a quiet river flows past the old stone bridge.'
 long={'model':model,'input':'Return the requested JSON array exactly. No commentary.','background':True,'max_output_tokens':2048,'temperature':0,'reasoning':{'effort':'none'},'text':{'format':{'type':'json_schema','name':'metrics_partial','strict':True,'schema':{'type':'array','items':{'type':'string','const':phrase},'minItems':80,'maxItems':80}}}}
 ack2=request('POST','/v1/responses',long);assert ack2['status']=='queued';cancel_id=ack2['id']
 wait_event(lambda:cancel_id in progress,60)
 cancelled=request('POST','/v1/responses/'+cancel_id+'/cancel');assert cancelled['status']=='cancelled',cancelled
 wait_event(lambda:cancel_id in stored,30)
 assert len(records.get(cancel_id,[]))==1 and records[cancel_id][0]['outcome']=='cancel_partial'
 for _ in range(3):assert request('GET','/v1/responses/'+cancel_id)==cancelled
 # With one worker slot, completion of this request is a release/progress barrier.
 barrier=request('POST','/v1/responses',short);assert barrier['status']=='completed',barrier
 final=snap('final')
 checks['cancel_callback_once_and_partial']=len(records[cancel_id])==1 and 0<records[cancel_id][0]['completion_tokens']==cancelled['usage']['output_tokens']<2048
 checks['cancel_excluded_from_generation_success']=value(final,'memra_e2e_seconds_count')-value(after_complete,'memra_e2e_seconds_count')==1
 checks['only_three_generation_http_requests']=value(final,'memra_requests_total',code='200')-value(before,'memra_requests_total',code='200')==3
 checks['completion_histogram_excludes_cancelled_partial']=value(final,'memra_completion_tokens_sum')-value(before,'memra_completion_tokens_sum')==complete['usage']['output_tokens']+barrier['usage']['output_tokens']
 checks['partial_native_work_observed']=value(final,'memra_emitted_token_events_total')>=complete['usage']['output_tokens']+cancelled['usage']['output_tokens']+barrier['usage']['output_tokens']
 checks['acknowledgements_are_not_http_cancellations']=value(final,'memra_requests_cancelled_total')==0
 checks['no_unsettled_receipts']=not unsettled
 (a.out/'checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks),flush=True)
finally:
 (a.out/'http.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in http));(a.out/'callbacks.json').write_text(json.dumps({'receipts':records,'stored':stored,'unsettled':unsettled},indent=2))
 if server.poll() is None:
  server.terminate()
  try:server.wait(30)
  except subprocess.TimeoutExpired:server.kill();server.wait()
 reader.join();log.close();monitor.terminate();monitor.wait();telemetry.close()
 checks['no_unsettled_receipts']=not unsettled
 checks['one_callback_for_each_of_three_requests']=len(records)==3 and all(len(v)==1 for v in records.values())
 (a.out/'checks.json').write_text(json.dumps(checks,indent=2))
 (a.out/'callbacks.json').write_text(json.dumps({'receipts':records,'stored':stored,'unsettled':unsettled},indent=2))
 (a.out/'gpu-after-cleanup.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
raise SystemExit(0 if checks and all(checks.values()) else 1)
