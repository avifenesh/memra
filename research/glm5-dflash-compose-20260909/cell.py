#!/usr/bin/env python3
"""One server boot; serial interleaved sampling/K arms and independent shadow captures."""
import hashlib,json,os,signal,subprocess,time,urllib.request,urllib.error
from pathlib import Path
R=Path(__file__).resolve().parent
out=R/'raw';out.mkdir(exist_ok=False)
ctl=Path('/root/dflash-compose-control')
ctl.mkdir(exist_ok=True)
for name,value in [('k','auto'),('mode','timing'),('flags','0')]:(ctl/name).write_text(value)
env={k:v for k,v in os.environ.items() if not k.startswith('MEMRA_')}
env.update(json.loads((R/'posture.json').read_text()))
(out/'env.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith('MEMRA_')},indent=2)+'\n')
(out/'gpu-before.txt').write_bytes(subprocess.check_output(['nvidia-smi','-q']))
log=(out/'server.log').open('wb')
server=subprocess.Popen(['/root/glm5-dflash-compose-target/release/memra-server'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'pid').write_text(str(server.pid)+'\n')
telemetry=subprocess.Popen(['python3',str(R/'telemetry.py')])
base='http://127.0.0.1:18997'
prompt=Path('/root/out-tpprime2/p32k/long.txt').read_text()
# Last arm isolates top-k by explicitly removing top-p.
def request(label,k,params,mode='timing',count=512,flags=0):
 for name,value in [('k',str(k)),('mode',mode),('flags',str(flags))]:
  pending=ctl/(name+'.next');pending.write_text(value);pending.replace(ctl/name)
 d=out/label;d.mkdir()
 body={'model':'zai/glm-5.3-flash','messages':[{'role':'user','content':prompt}],'max_tokens':count,'timeout_ms':900000,'stream':True,'stream_options':{'include_usage':True},**params}
 payload=json.dumps(body).encode();(d/'request.json').write_bytes(payload)
 offset=(out/'server.log').stat().st_size
 start=time.monotonic();first=None;chunks=[]
 try:
  req=urllib.request.Request(base+'/v1/chat/completions',data=payload,headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=900) as res:
   status=res.status
   for line in res:
    chunks.append(line)
    if line.startswith(b'data: {'):
     ev=json.loads(line[6:])
     if any(c.get('delta',{}).get('content') or c.get('delta',{}).get('reasoning_content') or c.get('delta',{}).get('reasoning') for c in ev.get('choices',[])):
      if first is None:first=time.monotonic()-start
  wall=time.monotonic()-start
  data=b''.join(chunks);(d/'response.sse').write_bytes(data)
  events=[json.loads(s[6:]) for s in data.decode().splitlines() if s.startswith('data: {')]
  usage=[e['usage'] for e in events if e.get('usage')]
  text=''.join(c.get('delta',{}).get('content','')+c.get('delta',{}).get('reasoning_content','')+c.get('delta',{}).get('reasoning','') for e in events for c in e.get('choices',[]))
  (d/'output.txt').write_text(text)
  result={'label':label,'flags':flags,'k':k,'mode':mode,'params':params,'http':status,'wall_s':wall,'ttft_s':first,'usage':usage[-1] if usage else None,'done':b'data: [DONE]' in data,'finish':[c.get('finish_reason') for e in events for c in e.get('choices',[]) if c.get('finish_reason')],'request_sha256':hashlib.sha256(payload).hexdigest(),'response_sha256':hashlib.sha256(data).hexdigest()}
  (d/'result.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result),flush=True)
  assert result['done'] and usage and usage[-1]['completion_tokens']>0,result
  if k != 0:
   assert usage[-1].get('spec',{}).get('drafted',0)>0, ('spec did not engage',result)
  else:
   assert not usage[-1].get('spec'), ('plain carried spec',result)
 except urllib.error.HTTPError as exc:
  (d/'http-error.txt').write_bytes(exc.read())
  raise
 finally:
  time.sleep(1)
  with (out/'server.log').open('rb') as f:f.seek(offset);(d/'server.log').write_bytes(f.read())
try:
 for _ in range(900):
  if server.poll() is not None:raise RuntimeError(f'server exited {server.returncode}')
  try:
   with urllib.request.urlopen(base+'/health',timeout=2) as res:
    if res.status==200:break
  except Exception:time.sleep(1)
 else:raise RuntimeError('health timeout')
 request('warm-spec','auto',{},count=64)
 request('warm-plain',0,{},count=64)
 # Each greedy arm captures emitted token IDs. MLA ON also runs same-input
 # current/candidate latent-row argmax and numerical-band checks before timing.
 for name,flags in [('off',0),('kda',1),('mla',2),('both',3),('all',7)]:
  request('greedy-'+name,6,{'temperature':0,'seed':39320260909},count=160,flags=flags|8)
 # Fixed seed paired PMIN comparison, three repeats of each arm. This is an
 # empirical divergence check; finite continuation histograms are not an
 # equality proof for conditional target distributions.
 for rep in range(3):
  for name,flags in ([('pmin_off',3),('pmin_on',7)] if rep%2==0 else [('pmin_on',7),('pmin_off',3)]):
   request(f'distribution-r{rep}-{name}',6,{'seed':39320260909},mode='agreement',flags=flags|16)
 arms=[('plain',0,0),('off',6,0),('kda',6,1),('mla',6,2),('both',6,3),('all',6,7),('auto','auto',7)]
 for rep in range(3):
  order=arms if rep==0 else list(reversed(arms)) if rep==1 else arms[3:]+arms[:3]
  for name,k,flags in order:
   request(f'timing-r{rep}-{name}',k,{},flags=flags)
 (R/'cell.exit').write_text('0\n')
finally:
 if server.poll() is None:os.killpg(server.pid,signal.SIGTERM)
 try:server.wait(timeout=45)
 except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
 log.close()
 telemetry.wait(timeout=10)
 (out/'gpu-after.txt').write_bytes(subprocess.check_output(['nvidia-smi','-q']))
