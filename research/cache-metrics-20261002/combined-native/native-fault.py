"""One owned worker panic, with a second request already emitting. Expected fault is a red control."""
import concurrent.futures,json,threading,urllib.request
from pathlib import Path

def exercise(base,api,operator,out,panic_event,promtool):
    import subprocess,sys
    sys.path.insert(0,str(Path.cwd()/'tools'))
    from prometheus_metrics import parse_samples,scalar
    out.mkdir(parents=True,exist_ok=True);started=threading.Event()
    def stream_long():
        body={'model':'metrics-q9','prompt':'Write at least ten thousand words about databases, with detailed examples.','max_tokens':1024,'stream':True,'temperature':0,'cache_salt':'fault-long'}
        req=urllib.request.Request(base+'/v1/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+api,'Content-Type':'application/json'})
        events=[];done=False
        with urllib.request.urlopen(req,timeout=180) as r:
            for line in r:
                if not line.startswith(b'data:'): continue
                raw=line[5:].strip()
                if raw==b'[DONE]': done=True;break
                event=json.loads(raw);events.append(event)
                if len(events)>=32: started.set()
        row={'request':body,'events':events,'done':done}
        (out/'interrupted.json').write_text(json.dumps(row,indent=2));return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        long=pool.submit(stream_long)
        assert started.wait(30),'long request did not emit before fault trigger'
        body={'model':'metrics-q9','prompt':'Count from one to twenty in words, separated by commas.','max_tokens':1,'stream':False,'temperature':0,'cache_salt':'fault-trigger'}
        req=urllib.request.Request(base+'/v1/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+api,'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=60) as r: short=json.loads(r.read())
        (out/'trigger.json').write_text(json.dumps({'request':body,'response':short},indent=2))
        row=long.result(timeout=60)
    assert panic_event.wait(30),'owned server did not log the injected worker panic'
    req=urllib.request.Request(base+'/metrics',headers={'Authorization':'Bearer '+operator,'Accept':'text/plain'})
    with urllib.request.urlopen(req,timeout=15) as r: text=r.read().decode()
    (out/'after.prom').write_text(text);samples=parse_samples(text)
    labels={'model':'metrics-q9','route':'hybrid','lane':'interactive'}
    checks={'partial_tokens_before_fault':sum(bool(e.get('choices')) for e in row['events'])>=32,
            'typed_error_after_200':any(e.get('error',{}).get('code')=='overloaded' for e in row['events']),
            'no_successful_finish':not any(c.get('finish_reason') in ('length','stop') for e in row['events'] for c in e.get('choices',[])),
            'one_truncated_stream':scalar(samples,'memra_streams_truncated_total',**labels)==1,
            'one_incomplete_stream':scalar(samples,'memra_streams_incomplete_total',**labels)==1,
            'one_failed_response':scalar(samples,'memra_response_errors_total',**labels)==1,
            'worker_respawn_counted':scalar(samples,'memra_worker_respawns_total')==1,
            'kv_gauges_cleared_after_panic':scalar(samples,'memra_kv_used_bytes',model='metrics-q9',route='hybrid')==0,
            'partial_work_counted':scalar(samples,'memra_emitted_token_events_total',**labels)>=33,
            'no_client_cancellation':scalar(samples,'memra_requests_cancelled_total',**labels)==0}
    result=subprocess.run([promtool,'check','metrics'],input=text,text=True,capture_output=True,timeout=30)
    (out/'promtool.log').write_text(result.stdout+result.stderr);checks['promtool']=result.returncode==0
    (out/'checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks),flush=True)
    return 0 if all(checks.values()) else 1
