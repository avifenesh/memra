#!/usr/bin/env python3
"""One sampled 64-token request, with profiler collection starting at spec round 0."""
import argparse, hashlib, json, os, signal, subprocess, time, urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('label')
p.add_argument('--k', type=int)
p.add_argument('--phase', action='store_true')
p.add_argument('--capture', type=Path)
a = p.parse_args()
root = Path(__file__).resolve().parent
out = root / 'raw' / a.label
out.mkdir(parents=True, exist_ok=False)
env = {k:v for k,v in os.environ.items() if not k.startswith('MEMRA_')}
env.update(json.loads((root/'posture.json').read_text()))
if a.k is not None:
    env['MEMRA_SPEC_K'] = str(a.k)
if a.phase:
    env.update(MEMRA_SPEC_PROF='1', MEMRA_SPEC_TRACE='2')
(out/'env.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith('MEMRA_')},indent=2)+'\n')
cmd=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=none','--force-overwrite=true','-o',str(out/'trace'),'/root/glm5-verify-tally-target/release/memra-server']
if a.capture:
    env['MEMRA_GLM5_VERIFY_TALLY']='capture:'+str(a.capture.resolve())
    cmd=[cmd[-1]]
    (out/'env.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith('MEMRA_')},indent=2)+'\n')
(out/'command.json').write_text(json.dumps(cmd)+'\n')
log=(out/'server.log').open('w')
proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'pid').write_text(str(proc.pid)+'\n')
try:
    for _ in range(600):
        if proc.poll() is not None:
            raise RuntimeError(f'server exited {proc.returncode}')
        try:
            with urllib.request.urlopen('http://127.0.0.1:18995/health',timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError('health timeout')
    prompt=Path('/root/out-tpprime2/p32k/long.txt').read_text()
    body={'model':'zai/glm-5.3-flash','messages':[{'role':'user','content':prompt}],'max_tokens':64,'stream':True,'stream_options':{'include_usage':True}}
    payload=json.dumps(body).encode()
    (out/'request.json').write_bytes(payload)
    start=time.monotonic()
    request=urllib.request.Request('http://127.0.0.1:18995/v1/chat/completions',data=payload,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=600) as r:
        response=r.read()
        status=r.status
    (out/'response.sse').write_bytes(response)
    events=[json.loads(line[6:]) for line in response.decode().splitlines() if line.startswith('data: ') and line[6:]!='[DONE]']
    usage=[x['usage'] for x in events if x.get('usage')]
    result={'http_status':status,'wall_s':time.monotonic()-start,'done':b'data: [DONE]' in response,'usage':usage,'request_sha256':hashlib.sha256(payload).hexdigest(),'response_sha256':hashlib.sha256(response).hexdigest()}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    assert result['done'] and usage and usage[-1]['completion_tokens']==64
finally:
    # TERM only the executable owned by this nsys session, letting nsys seal its report.
    tree=[row.split(None,2) for row in subprocess.check_output(['ps','-eo','pid,ppid,comm'],text=True).splitlines()[1:]]
    owned={proc.pid}
    for _ in range(len(tree)):
        old=len(owned)
        owned.update(int(pid) for pid,ppid,comm in tree if int(ppid) in owned)
        if len(owned)==old:
            break
    targets=[int(pid) for pid,ppid,comm in tree if int(pid) in owned and comm=='memra-server']
    for pid in targets:
        os.kill(pid,signal.SIGTERM)
    try:
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGTERM)
        proc.wait(timeout=30)
    log.close()
if (out/'trace.nsys-rep').exists():
    subprocess.run(['nsys','export','--type=sqlite','--force-overwrite=true','--output',str(out/'trace.sqlite'),str(out/'trace.nsys-rep')],check=True)
