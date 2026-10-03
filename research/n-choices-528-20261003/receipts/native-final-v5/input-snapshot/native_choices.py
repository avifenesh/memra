#!/usr/bin/env python3
"""Broker-only native n-choice collection. Retain successes and failures unchanged."""
import argparse
import ctypes
import hashlib
import http.client
import json
import os
import re
from pathlib import Path
import signal
import socket
import struct
import subprocess
import threading
import time

VENDOR={'temperature':1.0,'top_p':.95,'top_k':20,'min_p':0.0,'presence_penalty':1.5,'repetition_penalty':1.0}
NONTHINK={'temperature':.7,'top_p':.8,'top_k':20,'min_p':0.0,'presence_penalty':1.5,'repetition_penalty':1.0}

def save(path, value): path.write_text(json.dumps(value,indent=2)+'\n')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

class Pressure:
    """Brief lane-owned allocation; never changes the server or a foreign context."""
    def __init__(self,out):
        self.out=out;self.ptr=ctypes.c_void_p();self.api=ctypes.CDLL('/usr/local/cuda-13.1/lib64/libcudart.so')
        self.api.cudaSetDevice.argtypes=[ctypes.c_int]
        self.api.cudaMemGetInfo.argtypes=[ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
        self.api.cudaMalloc.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t]
        self.api.cudaFree.argtypes=[ctypes.c_void_p]
        self.state={}
    def __enter__(self):
        try:
            assert self.api.cudaSetDevice(0)==0
            free=ctypes.c_size_t();total=ctypes.c_size_t()
            assert self.api.cudaMemGetInfo(ctypes.byref(free),ctypes.byref(total))==0
            keep=1024*1024*1024
            assert free.value>keep*2,'insufficient free memory for this controlled pressure cell'
            amount=free.value-keep
            self.state={'before_free':free.value,'total':total.value,'allocated':amount,'keep_free':keep,'pid':os.getpid()}
            rc=self.api.cudaMalloc(ctypes.byref(self.ptr),amount);self.state['allocation_code']=rc
            assert rc==0,'owned CUDA allocation failed'
            assert self.api.cudaMemGetInfo(ctypes.byref(free),ctypes.byref(total))==0
            self.state['after_free']=free.value;save(self.out/'pressure.json',self.state)
            return self
        except BaseException:
            self.__exit__(None,None,None);raise
    def __exit__(self,*_):
        if self.ptr.value:
            self.state['free_code']=self.api.cudaFree(self.ptr);self.ptr=ctypes.c_void_p()
        self.state['own_context_reset_code']=self.api.cudaDeviceReset()
        save(self.out/'pressure.json',self.state)
        assert self.state.get('free_code',0)==0 and self.state['own_context_reset_code']==0,'owned pressure context failed cleanup'

class Native:
    def __init__(self,args):
        self.args=args;self.out=args.out; self.cv=threading.Condition();self.lines=[];self.events=[];self.ready=False
    def start(self):
        a=self.args;self.out.mkdir(parents=True)
        assert os.path.samefile('/proc/self/fd/9','/tmp/memra-5090.lock')
        assert os.environ['MEMRA_GPU_LOCK']=='/tmp/memra-5090.lock'
        assert sha(a.binary)==a.binary_sha and sha(a.model)==a.model_sha
        with socket.socket() as s:s.bind(('127.0.0.1',a.port))
        metadata='[models.fixture]\n'+''.join(f'default_{k}={v}\n' for k,v in VENDOR.items())
        metadata+='[models.fixture.non_thinking_sampling]\n'+''.join(f'{k}={v}\n' for k,v in NONTHINK.items())
        (self.out/'metadata.toml').write_text(metadata)
        env={k:v for k,v in os.environ.items() if not k.startswith('MEMRA_') or k in {'MEMRA_GPU_LOCK','MEMRA_CI_LOCK','MEMRA_CI_LOCK_HELD','MEMRA_RIG_LOCK_FD'}}
        env.update(MEMRA_MODELS=f'fixture={a.model}',MEMRA_CTX=str(a.context),MEMRA_ADDR=f'127.0.0.1:{a.port}',MEMRA_MAX_SESSIONS=str(a.sessions),MEMRA_PREFIX_CACHE_MB='0',MEMRA_MODEL_METADATA=str(self.out/'metadata.toml'),MEMRA_TTFT_TRACE='1',MEMRA_DEBUG_PRIMESEG='1',CHOICE_GATE_OUTPUT_BUDGET=str(a.output_budget))
        if a.reserve_mb is not None:env['MEMRA_ADMIT_RESERVE_MB']=str(a.reserve_mb)
        save(self.out/'env.json',{k:v for k,v in env.items() if k.startswith('MEMRA_') or k=='CHOICE_GATE_OUTPUT_BUDGET'})
        self.hardware=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
        self.process=subprocess.Popen([str(a.binary)],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,pass_fds=(9,))
        def read():
            try:
                with (self.out/'server.log').open('w') as f:
                    for line in self.process.stdout:
                        f.write(line);f.flush()
                        with self.cv:
                            self.lines.append(line.rstrip())
                            if '[server] listening on ' in line:self.ready=True
                            if line.startswith('CHOICE_'):
                                tag,payload=line.split(' ',1)
                                self.events.append({'tag':tag,'value':json.loads(payload),'line':len(self.lines)-1})
                            self.cv.notify_all()
            finally:
                with self.cv:self.cv.notify_all()
        self.reader=threading.Thread(target=read);self.reader.start()
        with self.cv:assert self.cv.wait_for(lambda:self.ready or self.process.poll() is not None,180) and self.ready, 'server failed startup'
    def request(self,name,path,payload,reset_after=None):
        folder=self.out/name;folder.mkdir();save(folder/'request.json',payload)
        conn=http.client.HTTPConnection('127.0.0.1',self.args.port,timeout=90)
        start=time.monotonic();line_start=len(self.lines)
        conn.request('POST',path,json.dumps(payload),{'Content-Type':'application/json'})
        response=conn.getresponse(); headers=dict(response.getheaders());save(folder/'headers.json',headers)
        record={'name':name,'status':response.status,'headers':headers,'started':start,'source_log_start':line_start,'disconnected':False}
        if payload.get('stream') and response.status==200:
            raw=[];packets=[];frames=0;data=[]
            while True:
                line=response.readline()
                if not line:break
                raw.append(line)
                if line.startswith(b'data:'):data.append(line[5:].strip())
                elif not line.strip() and data:
                    content=b'\n'.join(data);data=[]
                    packet='[DONE]' if content==b'[DONE]' else json.loads(content)
                    packets.append(packet)
                    if isinstance(packet,dict):
                        frames+=sum(bool(c.get('text') or c.get('delta',{}).get('content') or c.get('delta',{}).get('reasoning') or c.get('delta',{}).get('tool_calls')) for c in packet.get('choices',[]))
                    if reset_after is not None and frames>=reset_after:
                        assert conn.sock is not None
                        conn.sock.setsockopt(socket.SOL_SOCKET,socket.SO_LINGER,struct.pack('ii',1,0))
                        conn.close(); record['disconnected']=True;break
                    if packet=='[DONE]':break
            (folder/'wire.sse').write_bytes(b''.join(raw));save(folder/'packets.json',packets);record['frames']=frames
        else:
            raw=response.read();(folder/'wire.json').write_bytes(raw)
            try:record['body']=json.loads(raw)
            except ValueError:record['body_text']=raw.decode()
        conn.close();record['ended']=time.monotonic();record['source_log_end']=len(self.lines)
        identity=headers.get('x-request-id') or record.get('body',{}).get('id')
        record['id']=identity
        if identity and self.args.phase!='baseline':
            if '-refuse-' not in name:
                with self.cv:self.cv.wait_for(lambda:any(e['tag']=='CHOICE_DROP' and e['value']['id']==identity for e in self.events) or self.process.poll() is not None,10)
            record['callbacks']=[e for e in self.events if e['value'].get('id')==identity]
        record['worker_rows']=[e['value'] for e in self.events if e['tag']=='CHOICE_WORKER_ROW' and e['value'].get('group')==identity]
        record['prime_events']=[e['value'] for e in self.events if e['tag']=='CHOICE_PRIME' and e['value'].get('group')==identity]
        record['fork_lines']=[line for line in self.lines if line.startswith('[n-choice-prefill]') and ('group='+str(identity)+' ') in line]
        record['masked_steps']=sum(map(int,re.findall(r'\[constrained\] .*?: (\d+) masked steps','\n'.join(self.lines[line_start:]))))
        save(folder/'client.json',record);return record
    def stop(self):
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=10)
        self.reader.join(10);assert not self.reader.is_alive()
        save(self.out/'callback-events.json',self.events)

def main():
    p=argparse.ArgumentParser()
    for name in ['binary','model','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['binary-sha','model-sha','source']:p.add_argument('--'+name,required=True)
    p.add_argument('--port',type=int,default=18230);p.add_argument('--context',type=int,default=4096);p.add_argument('--sessions',type=int,default=8)
    p.add_argument('--output-budget',type=int,default=1000000);p.add_argument('--reserve-mb',type=int)
    p.add_argument('--phase',choices=['baseline','candidate','budget','kv','slots','deadline'],required=True)
    args=p.parse_args();assert not args.out.exists(),'new receipt directory required; never overwrite evidence';native=Native(args);results=[];status='failed'
    try:
        native.start()
        if args.phase=='baseline':
            for path,body in [('/v1/completions',{'model':'fixture','prompt':'Count upward from one, with a short item on each line.'}),('/v1/chat/completions',{'model':'fixture','messages':[{'role':'user','content':'Count upward from one, with a short item on each line.'}]})]:
                body.update(temperature=0,max_tokens=32,seed=73)
                results.append(native.request(path.strip('/').replace('/','-')+'-n1',path,body))
        elif args.phase=='candidate':
            for route in ['completions','chat/completions']:
                body={'model':'fixture','prompt':'Count upward from one, with a short item on each line.'} if route=='completions' else {'model':'fixture','messages':[{'role':'user','content':'Count upward from one, with a short item on each line.'}]}
                label=route.replace('/','-')
                results.append(native.request(label+'-n1','/v1/'+route,{**body,'temperature':0,'max_tokens':32,'seed':73}))
                for n in [2,4,8]:results.append(native.request(label+f'-greedy-n{n}','/v1/'+route,{**body,'n':n,'temperature':0,'max_tokens':32,'seed':73}))
                results.append(native.request(label+'-stream','/v1/'+route,{**body,'n':4,'stream':True,'stream_options':{'include_usage':True},'max_tokens':32,'seed':101}))
                results.append(native.request(label+'-stream-twin','/v1/'+route,{**body,'n':4,'max_tokens':32,'seed':101}))
                results.append(native.request(label+'-sampled','/v1/'+route,{**body,'n':3,'max_tokens':32,'seed':101}))
                results.append(native.request(label+'-sampled-repeat','/v1/'+route,{**body,'n':3,'max_tokens':32,'seed':101}))
                for i in range(3):results.append(native.request(label+f'-seed-{i}','/v1/'+route,{**body,'max_tokens':32,'seed':101+i}))
                for key,value in [('n',0),('n',9),('best_of',2)]:results.append(native.request(label+f'-refuse-{key}-{value}','/v1/'+route,{**body,key:value,'max_tokens':8}))
                results.append(native.request(label+'-n1-after','/v1/'+route,{**body,'temperature':0,'max_tokens':32,'seed':73}))
            bare={'model':'fixture','messages':[{'role':'user','content':'Reply with one short sentence confirming this fixture is ready.'}]}
            results.append(native.request('bare-default','/v1/chat/completions',bare))
            results.append(native.request('n-only-default','/v1/chat/completions',{**bare,'n':2}))
            results.append(native.request('n-only-default-fresh','/v1/chat/completions',{**bare,'n':2}))
            tools=[{'type':'function','function':{'name':'weather','parameters':{'type':'object','properties':{'city':{'type':'string','enum':['Paris']}},'required':['city'],'additionalProperties':False}}}]
            constrained={**bare,'messages':[{'role':'user','content':'Call weather for Paris.'}],'tools':tools,'tool_choice':'required','parallel_tool_calls':False,'n':2,'max_tokens':128,'reasoning_effort':'none','temperature':0,'seed':530}
            results.append(native.request('constrained-choices','/v1/chat/completions',constrained))
            results.append(native.request('constrained-choices-stream','/v1/chat/completions',{**constrained,'stream':True,'stream_options':{'include_usage':True}}))
            results.append(native.request('cancel','/v1/chat/completions',{'model':'fixture','messages':[{'role':'user','content':'List the integers from one upward. Continue until the output limit.'}],'n':4,'stream':True,'max_tokens':512,'seed':103},reset_after=8))
            results.append(native.request('cancel-recovery','/v1/chat/completions',{**bare,'temperature':0,'max_tokens':8,'seed':73}))
        elif args.phase=='deadline':
            payload={'model':'fixture','messages':[{'role':'user','content':'List integers upward with explanations. Keep going until the output limit.'}],'n':2,'timeout_ms':2000,'seed':73,'stop':['END']}
            results.append(native.request('deadline-partial','/v1/chat/completions',payload))
            results.append(native.request('deadline-recovery','/v1/chat/completions',{'model':'fixture','messages':[{'role':'user','content':'Reply with one short sentence.'}],'n':2,'max_tokens':8,'seed':73}))
        elif args.phase=='slots':
            payload={'model':'fixture','messages':[{'role':'user','content':'Count upward. Write one integer on each line and keep going until the output limit.'}],'n':4,'stream':True,'max_tokens':512,'seed':103}
            folder=args.out/'occupier';folder.mkdir();save(folder/'request.json',payload)
            conn=http.client.HTTPConnection('127.0.0.1',args.port,timeout=90)
            conn.request('POST','/v1/chat/completions',json.dumps(payload),{'Content-Type':'application/json'})
            response=conn.getresponse();assert response.status==200
            save(folder/'headers.json',dict(response.getheaders()))
            raw=[]; nonempty=False
            while not nonempty:
                line=response.readline();assert line,'occupier ended before overlap probe';raw.append(line)
                if line.startswith(b'data:'):
                    part=line[5:].strip()
                    if part and part!=b'[DONE]':
                        event=json.loads(part)
                        nonempty=any(c.get('delta',{}).get('content') or c.get('delta',{}).get('reasoning') for c in event.get('choices',[]))
            results.append(native.request('slots-contender','/v1/chat/completions',{'model':'fixture','messages':[{'role':'user','content':'Reply with one short sentence.'}],'max_tokens':8,'timeout_ms':1000,'seed':73}))
            assert conn.sock is not None
            conn.sock.setsockopt(socket.SOL_SOCKET,socket.SO_LINGER,struct.pack('ii',1,0));conn.close()
            (folder/'wire.sse').write_bytes(b''.join(raw))
            identity=dict(response.getheaders()).get('x-request-id')
            with native.cv:
                assert native.cv.wait_for(lambda:any(e['tag']=='CHOICE_DROP' and e['value']['id']==identity for e in native.events) or native.process.poll() is not None,10),'occupier drop missing'
            save(folder/'client.json',{'id':identity,'status':response.status,'disconnected':True,'callbacks':[e for e in native.events if e['value'].get('id')==identity]})
            results.append(native.request('slots-recovery','/v1/chat/completions',{'model':'fixture','messages':[{'role':'user','content':'Reply with one short sentence.'}],'n':4,'max_tokens':8,'seed':73}))
        elif args.phase=='kv':
            body={'model':'fixture','prompt':'Write a long numbered list.','max_tokens':8,'seed':73}
            results.append(native.request('kv-warm','/v1/completions',body))
            with Pressure(args.out):
                results.append(native.request('kv','/v1/completions',{**body,'n':8,'max_tokens':3072}))
            results.append(native.request('kv-recovery','/v1/completions',{**body,'n':8}))
        else:
            body={'model':'fixture','prompt':'Write a long numbered list.','n':4,'max_tokens':64,'seed':73}
            if args.phase=='budget':
                results.append(native.request('budget-singleton','/v1/completions',{**body,'n':1}))
            results.append(native.request(args.phase,'/v1/completions',body))
        status='complete'
    finally:
        if hasattr(native,'process'):native.stop()
        manifest={'status':status,'source':args.source,'binary_sha256':args.binary_sha,'model_sha256':args.model_sha,'phase':args.phase,'server_pid':getattr(getattr(native,'process',None),'pid',None),'hardware':getattr(native,'hardware',None),'server_exit':getattr(getattr(native,'process',None),'returncode',None),'results':results,'support_promotion':False,'helper_sha256':{name:sha(Path(__file__).with_name(name)) for name in ['native_choices.py','choice_verifier.py','replay_native_choices.py']},'files':{str(f.relative_to(args.out)):sha(f) for f in args.out.rglob('*') if f.is_file()}}
        save(args.out/'manifest.json',manifest)
    print(json.dumps({'status':status,'requests':len(results)}),flush=True)
if __name__=='__main__':main()
