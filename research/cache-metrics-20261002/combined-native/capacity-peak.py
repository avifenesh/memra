import argparse,json,socket,subprocess,sys,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'tools'))
from prometheus_metrics import parse_samples,scalar
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--api-key-file',type=Path,required=True);p.add_argument('--metrics-token-file',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--promtool',required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
api=a.api_key_file.read_text().strip();operator=a.metrics_token_file.read_text().strip();model='metrics-q9'
body={'model':model,'prompt':'Write at least ten thousand words about the design of a database. Include detailed examples and explanations of each component.','max_tokens':1024,'temperature':0,'stream':True,'cache_salt':'capacity-peak'}
request=urllib.request.Request(a.base+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+api})
frames=[];sample=None
with urllib.request.urlopen(request,timeout=180) as response:
 for line in response:
  if not line.startswith(b'data:'): continue
  raw=line[5:].strip()
  if raw==b'[DONE]': break
  event=json.loads(raw);frames.append(event)
  if len(frames)==256:
   req=urllib.request.Request(a.base+'/metrics',headers={'Authorization':'Bearer '+operator,'Accept':'text/plain'})
   with urllib.request.urlopen(req,timeout=15) as r: sample=r.read().decode()
   response.fp.raw._sock.shutdown(socket.SHUT_RDWR)
   break
(a.out/'request.json').write_text(json.dumps(body,indent=2));(a.out/'events.json').write_text(json.dumps(frames,indent=2))
assert sample is not None,'generation ended before the live-capacity barrier'
(a.out/'active.prom').write_text(sample)
samples=parse_samples(sample)
labels={'model':model,'route':'hybrid'}
used=scalar(samples,'memra_kv_used_bytes',**labels);capacity=scalar(samples,'memra_kv_capacity_bytes',**labels);active=scalar(samples,'memra_active_sessions',**labels)
checks={'kv_bytes_observed':0<used<=capacity,'active_session_observed':active==1,'device_free_positive':scalar(samples,'memra_device_free_bytes',device='0')>0,'device_sample_recent':0<=scalar(samples,'memra_device_memory_sample_age_seconds',device='0')<5}
r=subprocess.run([a.promtool,'check','metrics'],input=sample,text=True,capture_output=True,timeout=30)
(a.out/'promtool.log').write_text(r.stdout+r.stderr);checks['promtool']=r.returncode==0
(a.out/'checks.json').write_text(json.dumps({'checks':checks,'used':used,'capacity':capacity,'active':active},indent=2));print(json.dumps(checks))
raise SystemExit(0 if all(checks.values()) else 1)
