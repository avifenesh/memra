import hashlib,json,os,pathlib,subprocess,threading,time
root=pathlib.Path.cwd(); out=root/'research/cache-metrics-20261002/issue777/live'; out.mkdir(parents=True,exist_ok=True)
model='/data/ai-ml/hf-models/qwen36-35b-a3b-mtp-gguf-5bc3e238/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf'
binary=root/'target/release/memra-server'
env={k:v for k,v in os.environ.items() if not k.startswith('MEMRA_') or k in ('MEMRA_GPU_LOCK','MEMRA_CI_LOCK','MEMRA_CI_LOCK_HELD','MEMRA_RIG_LOCK_FD')}
env.update(MEMRA_COMPAT='openai',MEMRA_MODELS='q35-coldfix='+model,MEMRA_ADDR='127.0.0.1:18110',MEMRA_SERVE_SPEC='0',MEMRA_CTX='8192',MEMRA_PREFIX_CACHE_MB='4096',MEMRA_REUSE_POOL='0',MEMRA_AFFINITY='0',MEMRA_MAX_SESSIONS='96',MEMRA_GDN_CHUNK='32')
assert env['MEMRA_RIG_LOCK_FD']=='9'
subprocess.run(['bash','tools/port-guard.sh','check','q35-c4','18110'],check=True)
identity={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'binary_sha256':hashlib.file_digest(open(binary,'rb'),'sha256').hexdigest(),'artifact_sha256':(out.parent/'model.sha256').read_text().split()[0],'artifact_revision':'5bc3e238d916f48a861bac2f8a1990a0e9b7e98d','artifact_bytes':os.stat(model).st_size,'env':{k:v for k,v in env.items() if k.startswith('MEMRA_')},'request_shape':{'concurrency':4,'requests':20,'hits':18,'misses':2,'prompt_tokens':4860,'completion_tokens':60,'temperature':0},'timeout_seconds':1200,'minimum_free_vram_mib':21000}
(out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
for label in ['before']:
 (out/f'gpu-{label}.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
ready=threading.Event(); started=False
log=open(out/'server.log','w'); telemetry=open(out/'gpu-250ms.csv','w')
monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw','--format=csv','--loop-ms=250'],stdout=telemetry,stderr=subprocess.STDOUT,pass_fds=(9,))
p=subprocess.Popen([str(binary)],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,pass_fds=(9,))
def copy_log():
 global started
 for line in p.stdout:
  log.write(line); log.flush()
  if '[server] listening on http://127.0.0.1:18110' in line:
   started=True; ready.set()
 ready.set()
t=threading.Thread(target=copy_log); t.start()
rc=1
try:
 if not ready.wait(300) or not started:
  raise RuntimeError('Server failed to start within 300 seconds; inspect server.log')
 subprocess.run(['bash','-c','source tools/port-guard.sh; memra_port_owned q35-c4 18110 "$1"','_',str(p.pid)],check=True)
 with open(out/'gate.jsonl','w') as f:
  result=subprocess.run(['python3','tools/q35-cold-mixed-gate.py','--base','http://127.0.0.1:18110','--model','q35-coldfix','--namespace','serve-smoke-q35-coldfix','--gdn-grid','32','--timeout','600'],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 rc=result.returncode
 import re
 carried=bool(re.search(r'^\[prime-batch\].*carried=[1-9]',(out/'server.log').read_text(),re.M))
 (out/'result.json').write_text(json.dumps({'gate_exit':rc,'carried_prime_batch_violation':carried},indent=2)+'\n')
 if carried: rc=1
finally:
 (out/'gpu-at-end.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
 if p.poll() is None:
  p.terminate()
  try:p.wait(20)
  except subprocess.TimeoutExpired:p.kill();p.wait()
 t.join(); log.close()
 monitor.terminate();monitor.wait();telemetry.close()
 (out/'gpu-after-cleanup.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
raise SystemExit(rc)
