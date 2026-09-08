# Archived mixed-RPS fixed-arrival design, adapted for same-binary prime-yield A/B.
# Run only on the authorized GPU after explicit handoff. Profiles/templates are read-only.
import sys,uuid,argparse,collections,fcntl,hashlib,json,math,os,pathlib,random,secrets,signal,socket,subprocess,threading,time,urllib.request,urllib.error
ap=argparse.ArgumentParser();ap.add_argument('--model',default='ornith');ap.add_argument('--label',default='ornith-pilot');ap.add_argument('--rates',default='0.2');ap.add_argument('--seconds',type=int,default=100);ap.add_argument('--mode',default='prefill');ap.add_argument('--drain',type=int,default=300);ap.add_argument('--limit',type=int,default=16);ap.add_argument('--arrival',default='regular');ap.add_argument('--sessions',type=int);ap.add_argument('--binary',required=True);ap.add_argument('--binary-sha256',required=True);ap.add_argument('--profile-source',required=True);ap.add_argument('--out-root',required=True);ap.add_argument('--arm',choices=['off','on'],required=True);ap.add_argument('--chunk',type=int,choices=[1024,4096],required=True);ap.add_argument('--long-request');ap.add_argument('--request-timeout',type=float,default=300);a=ap.parse_args()
base=pathlib.Path('/tmp/qwen-ornith-5090-capacity-20260908');root=pathlib.Path(a.out_root);root.mkdir(mode=0o700,parents=True,exist_ok=True);run=root/a.label;run.mkdir(mode=0o700)
lock=open('/tmp/memra-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip(),'GPU occupied'
orn=a.model=='ornith';source=pathlib.Path(a.profile_source);binary=pathlib.Path(a.binary);sha=a.binary_sha256;assert hashlib.file_digest(binary.open('rb'),'sha256').hexdigest()==sha
boot_nonce=uuid.uuid4().hex;workload_id='prefill-fairness-'+a.model
profile=json.loads((source/'profile.json').read_text());alias=profile['MEMRA_MODELS'].split('=')[0]
key=secrets.token_hex(24);(run/'key').write_text(key);(run/'key').chmod(0o600);(run/'keys.toml').write_text('[[keys]]\ntenant="mixed-rps-synthetic"\nsha256="'+hashlib.sha256(key.encode()).hexdigest()+'"\n');(run/'keys.toml').chmod(0o600)
(run/'models.toml').write_bytes((source/'models.toml').read_bytes())
with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
profile.update(MEMRA_ADDR=f'127.0.0.1:{port}',MEMRA_MODEL_METADATA=str(run/'models.toml'),MEMRA_API_KEYS=str(run/'keys.toml'),MEMRA_REQUEST_LEDGER=str(run/'ledger.jsonl'))
if a.sessions is not None:profile['MEMRA_MAX_SESSIONS']=str(a.sessions)
profile.update(MEMRA_PRIME_YIELD='1' if a.arm=='on' else '0',MEMRA_PRIME_CHUNK=str(a.chunk),MEMRA_TICK_TRACE='1',MEMRA_TTFT_TRACE='1')
(run/'profile.json').write_text(json.dumps(profile,indent=2));(run/'controller.py').write_bytes(pathlib.Path(__file__).read_bytes());(run/'identity.json').write_text(json.dumps({'binary':str(binary),'sha256':sha,'source_profile':str(source),'args':vars(a),'boot_nonce':boot_nonce,'workload_id':workload_id},indent=2))
log=(run/'server.log').open('w');proc=subprocess.Popen([str(binary)],cwd=run,env={'PATH':os.environ['PATH'],**profile,'MEMRA_METRICS_TOKEN':key},stdout=log,stderr=subprocess.STDOUT,start_new_session=True);(run/'server.pid').write_text(str(proc.pid));(run/'controller.pid').write_text(str(os.getpid()));print('LAUNCHED',json.dumps({'controller_pid':os.getpid(),'server_pid':proc.pid,'log':str(run/'server.log'),'binary_sha256':sha,'boot_nonce':boot_nonce}),flush=True)
identity=json.loads((run/'identity.json').read_text());identity.update(server_pid=proc.pid,server_start_ticks=pathlib.Path(f'/proc/{proc.pid}/stat').read_text().split()[21]);(run/'identity.json').write_text(json.dumps(identity,indent=2))
url=f'http://127.0.0.1:{port}';start=time.monotonic();abort=threading.Event();done=threading.Event();mu=threading.Lock();active={};progress={};rows=[];tele=[];phase='boot';faults=[]
def req(path,body=None,timeout=5):
 return urllib.request.urlopen(urllib.request.Request(url+path,data=json.dumps(body).encode() if body is not None else None,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}),timeout=timeout)
def metrics():
 with req('/metrics') as r:return json.load(r)
def observe():
 offset=0
 with (run/'telemetry.jsonl').open('w') as out:
  while not done.is_set():
   d={'t':time.monotonic()-start,'phase':phase}
   try:d['metrics']=metrics()
   except Exception as ex:d['metrics_error']=str(ex)
   try:d['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=4).strip()
   except Exception:pass
   with mu:
    d['client_inflight']=len(active);d['oldest_inflight_s']=max([time.monotonic()-v for v in active.values()],default=0);d['oldest_waiting_first_token_s']=max([time.monotonic()-v for i,v in active.items() if not progress.get(i)],default=0)
   with (run/'server.log').open() as f:f.seek(offset);delta=f.read();offset=f.tell()
   hits=[l for l in delta.splitlines() if any(x in l.lower() for x in ['out of memory','out_of_memory','step oom','panicked','cuda error','engine error'])]
   if hits:faults.extend(hits);abort.set()
   q=d.get('metrics',{}).get('queued_requests');
   if q is None:d['queue_unknown']=True;q=-1
   if d['oldest_waiting_first_token_s']>a.request_timeout or d.get('metrics',{}).get('step_oom_parks',0)>0:abort.set()
   tele.append(d);out.write(json.dumps(d)+'\n');out.flush();done.wait(1)
threading.Thread(target=observe,daemon=True).start()
def pct(xs,q=.95):
 xs=sorted(x for x in xs if x is not None);return round(xs[max(0,math.ceil(q*len(xs))-1)],3) if xs else None
sentence='The community garden has morning sunlight, two raised beds, a nearby water tap, compost, labels, gloves, and volunteers who can help each morning. '
def body_for(i,target,long=False):
 rng=random.Random(i+67191);words=['garden','orchard','nursery','greenhouse','allotment'];word=words[i%len(words)]
 lead=f'Synthetic case {workload_id} {i} {rng.randrange(100000000)}. Reference notes for planning a community {word}. Records describe independent observation days. Do not repeat the records.\n'
 n=max(1,int((target-110)/per_repeat));content=lead+sentence.replace('garden',word)*n
 task=('Write a detailed practical training handbook with 100 numbered lessons, each containing a worked example and at least 200 words. Continue until all lessons are complete.' if long else ['Give three practical recommendations in about 100 words.','Write a concise action plan in about 150 words.','Explain the two highest priority next steps in about 80 words.'][i%3])
 if not orn and not long:task='Give one practical first step in a short final sentence.'
 content+='\n'+task
 return {'model':alias,'messages':[{'role':'user','content':content}],'max_tokens':49152 if long else 2048,'stream':True,'stream_options':{'include_usage':True},'cache_salt':f'{a.label}-{i}'}
def one(i,cl,body,scheduled,stage):
 t=time.monotonic();r={'id':i,'class':cl,'stage':stage,'scheduled_s':scheduled-start,'dispatch_s':t-start,'schedule_lag_s':t-scheduled,'finish':None,'usage':{},'spec':{},'ttft_s':None,'content_chars':0,'reasoning_chars':0};last=None;gaps=[];events=[]
 try:
  with req('/v1/chat/completions',body,timeout=a.request_timeout) as response:
   r['http_status']=response.status;r['headers_s']=time.monotonic()-t
   for raw in response:
    now=time.monotonic()
    if cl=='longdecode' and now-t>120:
     r['deliberate_cut']=True;break
    if now-t>a.request_timeout:raise TimeoutError('hard request timeout')
    line=raw.decode(errors='replace').strip()
    if not line.startswith('data:'):continue
    data=line[5:].strip()
    if data=='[DONE]':r['done']=True;break
    ev=json.loads(data);events.append({'at_s':now-t,'data':ev})
    if ev.get('error'):r['stream_error']=ev['error']
    if ev.get('id'):r['request_id']=ev['id']
    if ev.get('usage'):r['usage']=ev['usage'];r['spec']=ev['usage'].get('spec',{})
    if ev.get('spec'):r['spec']=ev['spec']
    if ev.get('x_memra'):r['x_memra']=ev['x_memra']
    for c in ev.get('choices',[]):
     d=c.get('delta',{});text=d.get('content') or '';reason=d.get('reasoning') or d.get('reasoning_content') or ''
     if text or reason:
      if last is None:r['ttft_s']=now-t;progress[i]=True
      else:gaps.append(now-last)
      last=now;r['content_chars']+=len(text);r['reasoning_chars']+=len(reason)
     if c.get('finish_reason'):r['finish']=c['finish_reason']
   r['outcome']='deliberate_cancel' if r.get('deliberate_cut') else 'completed' if r.get('done') and r['finish']=='stop' and r['content_chars'] and not r.get('stream_error') else 'incomplete'
 except urllib.error.HTTPError as ex:r.update(http_status=ex.code,outcome='rejected' if ex.code in (429,503) else 'error',error=ex.read().decode(errors='replace'))
 except Exception as ex:r.update(outcome='cancelled' if abort.is_set() else 'error',error=str(ex))
 r['uncached_tokens']=r['usage'].get('prompt_tokens',0)-r['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)
 r.update(e2e_s=time.monotonic()-t,gap_p95_s=pct(gaps),gap_max_s=max(gaps,default=0),end_s=time.monotonic()-start)
 (run/f'{i}-request.json').write_text(json.dumps(body));(run/f'{i}-events.json').write_text(json.dumps(events))
 with mu:
  rows.append(r);active.pop(i,None)
  with (run/'requests.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
 return r
def summary(stage,rate,before,after,duration):
 rs=[r for r in rows if r['stage']==stage];ss={'stage':stage,'rate_offered_rps':len(rs)/duration,'small_rate_offered_rps':rate if a.mode=='prefill' else None,'duration_s':duration,'counts':dict(collections.Counter(r['outcome'] for r in rs)),'offered':len(rs),'server_delta':{k:after.get(k,0)-before.get(k,0) for k in ['admitted','completed','prompt_tokens_in','computed_tokens_in','cached_tokens_in','tokens_out','step_oom_parks','served_spec','served_dspark','served_plain']},'classes':{},'after_drain_metrics':{k:after.get(k) for k in ['queued_requests','active_sessions','admitted','completed','step_oom_parks']}}
 for cl in sorted(set(r['class'] for r in rs)):
  cr=[r for r in rs if r['class']==cl];ss['classes'][cl]={'n':len(cr),'completed':sum(r['outcome']=='completed' for r in cr),'p95_ttft_s':pct([r.get('ttft_s') for r in cr]),'p95_e2e_s':pct([r.get('e2e_s') for r in cr]),'p95_gap_s':pct([r.get('gap_p95_s') for r in cr]),'max_gap_s':max([r.get('gap_max_s',0) for r in cr],default=0),'ttft_bands':{str(b):sum(r.get('ttft_s') is not None and r['ttft_s']<=b for r in cr) for b in [2,5,10]},'input_tokens': [r.get('usage',{}).get('prompt_tokens') for r in cr],'output_tokens_including_reasoning':[r.get('usage',{}).get('completion_tokens') for r in cr]}
 ts=[x for x in tele if x['phase']==stage];ss['queue_peak']=max([x.get('metrics',{}).get('queued_requests',0) for x in ts],default=0);ss['queue_last']=ts[-1].get('metrics',{}).get('queued_requests') if ts else None;ss['oldest_inflight_peak_s']=max([x['oldest_inflight_s'] for x in ts],default=0);ss['faults']=faults;ss['aborted']=abort.is_set();(run/(stage+'-summary.json')).write_text(json.dumps(ss,indent=2));print('STAGE',json.dumps(ss),flush=True)
try:
 while time.monotonic()-start<180:
  if proc.poll() is not None:raise RuntimeError('server exited '+str(proc.returncode))
  try:
   with req('/readyz') as rr:
    if rr.status==200:break
  except Exception:pass
  time.sleep(1)
 else:raise TimeoutError('boot timeout')
 schema=metrics();(run/'metric-schema.json').write_text(json.dumps(sorted(schema)));assert all(k in schema for k in ['queued_requests','active_sessions','admitted','completed']), 'missing required metric schema';print('READY',proc.pid,flush=True);per_repeat=30 if orn else 31
 phase='calibration';b=body_for(-1,2048);r=one(-1,'2k',b,time.monotonic(),'calibration');actual=r['usage'].get('prompt_tokens',0)
 if r['outcome']!='completed':raise RuntimeError('calibration failed '+str(r))
 per_repeat=per_repeat*max(1,actual-110)/(2048-110);print('CALIBRATION',actual,r['usage'].get('completion_tokens'),r['ttft_s'],r['e2e_s'],flush=True)
 seeds=[]
 if a.mode in ('empirical','interference','prefill'):
  phase='seed-cache'
  for sid in range(8):
   sb=body_for(-100-sid,600);sb['messages'][0]['content']+=' Answer only with the word Ready.'
   sr=one(-100-sid,'seed',sb,time.monotonic(),'seed-cache')
   if sr['outcome']!='completed':raise RuntimeError('cache seed failed')
   ev=json.loads((run/f'{-100-sid}-events.json').read_text());answer=''.join(c.get('delta',{}).get('content','') or '' for x in ev for c in x['data'].get('choices',[]))
   seeds.append((sb['messages']+[{'role':'assistant','content':answer}],sb['cache_salt']))
 counter=0
 for stage_idx,rate in enumerate(map(float,a.rates.split(','))):
  if abort.is_set():break
  stage=f'stage{stage_idx}';phase=stage;before=metrics();t0=time.monotonic();threads=[];count=math.ceil(rate*a.seconds)
  if a.mode=='prefill':count=math.floor(rate*a.seconds)+1  # long at 0; 20 small arrivals in (0,100] at 0.20 RPS
  offsets=[j/rate for j in range(count)]
  if a.arrival=='burst':offsets=[(j//4)*4/rate for j in range(count)]
  if a.arrival=='jitterburst':
   rng=random.Random(782);offsets=sorted(max(0,j/rate+rng.uniform(-.3,.3)/rate) for j in range(count));offsets[8]=offsets[7];offsets[18]=offsets[17]
  if a.arrival=='poisson':
   rng=random.Random(513+stage_idx);offsets=[];arrival=0
   while arrival<a.seconds:offsets.append(arrival);arrival+=rng.expovariate(rate)
  count=len(offsets);dispatched=0;client_drops=0
  classes=[2048]*7+[8192]*2+[32000 if orn else 30000];random.Random(441).shuffle(classes)
  for j in range(count):
   scheduled=t0+offsets[j];time.sleep(max(0,scheduled-time.monotonic()))
   if abort.is_set():break
   i=counter;counter+=1;target=classes[j%10] if a.mode in ('mix','burst') else (2195 if j%10==9 else 1066);long=a.mode in ('interference','prefill') and j==0
   if long:target=128000 if a.mode=='prefill' else 60000
   cl=('longprefill' if a.mode=='prefill' else 'longdecode') if long else str(target);body=body_for(i,target,long and a.mode!='prefill')
   if long and a.long_request:
    body=json.loads(pathlib.Path(a.long_request).read_text());body.update(model=alias,max_tokens=2048,stream=True,stream_options={'include_usage':True},cache_salt=f'{a.label}-{i}')
    assert not any(k in body for k in ('temperature','top_k','top_p','min_p','seed')),'sampled performance template overrides vendor defaults'
   if a.mode in ('empirical','interference','prefill') and not long and j%10<8:
    sm,salt=seeds[j%8];tail=f'New observation {i}. '+sentence*max(1,int(360/per_repeat))+' Give one practical next step in a short final sentence.'
    body['messages']=sm+[{'role':'user','content':tail}];body['cache_salt']=salt;cl+='-partial-target'
   if a.mode=='empirical' and j%20==19:
    body['max_tokens']=8192;body['messages'][0]['content']+=' Write a detailed 1200-word explanation with examples.';cl+='-longer-output'
   elif a.mode in ('empirical','interference','prefill') and not long:body['messages'][0]['content']+=' Keep the final answer under 50 words.'
   with mu:
    full=len(active)>=a.limit
    if not full:active[i]=time.monotonic()
   if full:
    client_drops+=1
    row={'id':i,'stage':stage,'class':cl,'outcome':'client_limit_reject','scheduled_s':scheduled-start};rows.append(row)
    with (run/'requests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   else:
    dispatched+=1
    th=threading.Thread(target=one,args=(i,cl,body,scheduled,stage),daemon=True);th.start();threads.append(th)
  time.sleep(max(0,t0+a.seconds-time.monotonic()) if not abort.is_set() else 0);arrival_end=time.monotonic();phase=stage+'-drain';deadline=time.monotonic()+a.drain
  for th in threads:th.join(max(0,deadline-time.monotonic()))
  if any(th.is_alive() for th in threads):abort.set()
  after=metrics();timing={'start_s':t0-start,'planned_end_s':t0+a.seconds-start,'arrival_end_s':arrival_end-start,'drain_end_s':time.monotonic()-start,'planned_scheduled':count,'dispatched':dispatched,'client_limit_reject':client_drops,'abort_skipped':count-dispatched-client_drops,'arrival_kind':a.arrival};(run/(stage+'-timing.json')).write_text(json.dumps(timing,indent=2));summary(stage,rate,before,after,a.seconds)
  if abort.is_set():break
except Exception as ex:print('CONTROLLER_ERROR',type(ex).__name__,str(ex),flush=True);(run/'error.txt').write_text(str(ex));abort.set()
finally:
 if proc.poll() is None:
  os.killpg(proc.pid,signal.SIGTERM)
  try:proc.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
 time.sleep(2)
 with mu:
  for i in list(active):
   with (run/'requests.jsonl').open('a') as f:f.write(json.dumps({'id':i,'outcome':'cancelled','error':'controller shutdown with request outstanding'})+'\n')
 done.set();log.close();print('STOPPED',proc.pid,proc.returncode,'active',len(active),flush=True)

(run/'cleanup.json').write_text(json.dumps({'own_server_pid':proc.pid,'exit_code':proc.returncode,'gpu_compute_apps_while_lock_held':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip(),'boot_nonce':boot_nonce},indent=2))
lock.close()
if abort.is_set() or faults:sys.exit(1)
