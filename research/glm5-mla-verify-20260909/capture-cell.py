#!/usr/bin/env python3
import json,os,signal,subprocess,time,urllib.request
from pathlib import Path
R=Path('/root/glm5-mla-verify-receipts')
posture=json.loads(Path('/root/wt-glm5-dflash-rootcause/research/glm5-dflash-rootcause-20260909/posture.json').read_text())
base='http://127.0.0.1:18997'
for context,prompt_path in [('p32k','/root/out-tpprime2/p32k/long.txt'),('p128k','/root/out-tpprime5/p128k/long.txt')]:
 out=R/context
 if out.exists() and all(len(list(out.glob(f't{t}-*.meta')))==11 for t in (2,4,7)):
  print(f'REUSE complete {context} captures',flush=True);continue
 out.mkdir(exist_ok=False)
 env={k:v for k,v in os.environ.items() if not k.startswith('MEMRA_')}
 env.update(posture)
 env.update(MEMRA_ADDR='127.0.0.1:18997',MEMRA_CTX='147456',MEMRA_SPEC_K='6',MEMRA_SPEC_TRACE='2',GLM5_MLA_CAPTURE_DIR=str(out),MEMRA_MODEL_METADATA=str(R/'models.toml'))
 if context=='p128k':env['MEMRA_GLM5_VISION']='0'
 (out/'posture.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith(('MEMRA_','GLM5_'))},indent=2)+'\n')
 (out/'gpu-before.txt').write_bytes(subprocess.check_output(['nvidia-smi','-q']))
 log=(out/'server.log').open('wb')
 server=subprocess.Popen(['/root/glm5-mla-verify-target/release/memra-server'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (out/'pid').write_text(str(server.pid)+'\n')
 try:
  for _ in range(900):
   if server.poll() is not None:raise RuntimeError(f'server exit {server.returncode}')
   try:
    with urllib.request.urlopen(base+'/health',timeout=2) as res:
     if res.status==200:break
   except Exception:time.sleep(1)
  else:raise RuntimeError('health timeout')
  for attempt in range(3):
   body={'model':'zai/glm-5.3-flash','messages':[{'role':'user','content':Path(prompt_path).read_text()}],'max_tokens':256,'stream':True,'stream_options':{'include_usage':True}}
   data=json.dumps(body).encode();(out/f'request-{attempt}.json').write_bytes(data)
   req=urllib.request.Request(base+'/v1/chat/completions',data=data,headers={'Content-Type':'application/json'})
   start=time.monotonic()
   try:
    with urllib.request.urlopen(req,timeout=1200) as res: response=res.read();status=res.status
   except urllib.error.HTTPError as exc:
    (out/f'http-error-{attempt}.txt').write_bytes(exc.read());raise
   (out/f'response-{attempt}.sse').write_bytes(response)
   usage=[json.loads(s[6:])['usage'] for s in response.decode().splitlines() if s.startswith('data: {') and json.loads(s[6:]).get('usage')]
   result={'context':context,'attempt':attempt,'http':status,'done':b'data: [DONE]' in response,'usage':usage,'elapsed':time.monotonic()-start,'captures':{t:len(list(out.glob(f't{t}-*.meta'))) for t in (2,4,7)}}
   (out/f'result-{attempt}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
   assert status==200 and result['done'] and usage
   if all(v==11 for v in result['captures'].values()):break
  else:raise RuntimeError('Missing requested width captures after three requests')
 finally:
  if server.poll() is None:
   os.killpg(server.pid,signal.SIGTERM)
   try:server.wait(timeout=45)
   except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
  log.close()
  (out/'gpu-after.txt').write_bytes(subprocess.check_output(['nvidia-smi','-q']))
(R/'capture.exit').write_text('0\n')
