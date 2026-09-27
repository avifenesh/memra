"""CPU protocol fixture; simulated worker state, not model/CUDA/native proof."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json,signal,sys,threading,time
port=int(sys.argv[1]);mode=sys.argv[2]
lock=threading.Lock();fault=threading.Event();generation=0;phase='idle';failed=False
PANIC='MEMRA_PANIC_AFTER=1 fault injection: deliberate worker panic after 1 completed request(s)'
DETAIL='worker thread panicked: '+PANIC
MESSAGE='worker closed the stream without completing (worker restart in progress)'
def emit(value):print(value,flush=True)
def inject():
    global phase,failed,generation
    with lock:phase='dead';failed=True
    emit('[worker] PANIC in the GPU worker thread: '+PANIC)
    emit('[worker] respawn attempt 1/1 in 2s (reloading weights)')
    fault.set()
    def reload():
        global generation,phase,failed
        time.sleep(.3)  # compressed CPU fixture interval, never a native backoff claim
        with lock:generation=1 if mode!='no_generation' else 0;phase='loading'
        time.sleep(.15)
        with lock:phase='idle';failed=False
    threading.Thread(target=reload,daemon=True).start()
class Handler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def log_message(self,*_):pass
    def fixed(self,status,body,retry=False):
        data=json.dumps(body).encode();self.send_response(status)
        self.send_header('Content-Length',str(len(data)));self.send_header('Connection','close')
        if retry:self.send_header('Retry-After','2');self.send_header('retry-after-ms','2000')
        self.end_headers();self.wfile.write(data);self.wfile.flush()
    def do_GET(self):
        with lock:g,p,bad=generation,phase,failed
        if self.path not in ('/health','/readyz'):return self.fixed(404,{})
        body={'status':('unhealthy' if self.path=='/health' else 'not_ready') if bad else ('ok' if self.path=='/health' else 'ready'),
              'worker':{'generation':g,'phase':p},'models':['gate']}
        if bad:body['detail']=DETAIL if mode!='wrong_health' else 'unrelated failure'
        emit('GET '+self.headers.get('X-Request-Id','?'))
        self.fixed(503 if bad else 200,body,bad)
    def do_POST(self):
        global failed,phase
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        role=body['fixture_role'];emit('POST '+self.headers.get('X-Request-Id','?')+' '+role)
        if role=='victim':
            self.send_response(200);self.send_header('Transfer-Encoding','chunked');self.send_header('Connection','close');self.end_headers()
            def chunk(data):self.wfile.write(f'{len(data):x}\r\n'.encode()+data+b'\r\n');self.wfile.flush()
            try:
                prefix=b'data: {"model":"gate","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}\n\n'
                if mode=='fragmented_nonterminal':
                    chunk(prefix[:11]);chunk(prefix[11:])
                else:chunk(prefix)
                fault.wait(5)
                if mode=='truncated':return
                code='worker_fault' if mode=='request_fault' else 'overloaded'
                message='request fault' if mode=='request_fault' else MESSAGE
                data=('data: '+json.dumps({'error':{'code':code,'type':'server_error','param':None,'message':message}})+'\n\n').encode()
                if mode!='no_done':data+=b'data: [DONE]\n\n'
                chunk(data);self.wfile.write(b'0\r\n\r\n');self.wfile.flush()
            except (BrokenPipeError,ConnectionResetError):pass
            return
        response={'model':'gate','choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'completed'}}],
                  'usage':{'prompt_tokens':12,'completion_tokens':2,'total_tokens':14}}
        self.fixed(200,response)
        if role=='trigger':inject()
        elif role=='recovery' and mode=='repeat_fault':
            with lock:failed=True;phase='dead'
            emit('[worker] PANIC in the GPU worker thread: '+PANIC)
with ThreadingHTTPServer(('127.0.0.1',port),Handler) as server:
    def stop(*_):
        fault.set();threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,stop)
    emit('READY');server.serve_forever(poll_interval=.01)
