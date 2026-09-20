import json, os, subprocess, time, urllib.request, urllib.error
base = 'http://127.0.0.1:18090'
env = dict(os.environ, MEMRA_COMPAT='openai', MEMRA_ADDR='127.0.0.1:18090', MEMRA_CTX='4096', MEMRA_SERVE_SPEC='0', MEMRA_MODELS='g=/data/ai-ml/models/gemma4-12b-it-qat-29d0977/gemma-4-12b-it-qat-q4_0.gguf')
def call(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={'Content-Type':'application/json', 'x-request-id':'tokenize-live-531'})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r), r.headers.get('x-request-id')
    except urllib.error.HTTPError as e:
        return e.code, json.load(e), e.headers.get('x-request-id')
with open('/tmp/memra-lane-logs/tok-live-local/tok-server.log','w') as log:
    server = subprocess.Popen(['/home/avifenesh/projects/lane-bins/tokenize/memra-server'], env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for attempt in range(240):
            if server.poll() is not None:
                raise RuntimeError('server exited during load; inspect tok-server.log')
            try:
                if call('/readyz')[0] == 200:
                    break
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(2)
        else:
            raise TimeoutError('readyz did not become ready')
        model = call('/v1/models')[1]
        print('MODELS', json.dumps(model), flush=True)
        messages = [{'role':'user','content':'Say hi'}]
        status, tokenized, request_id = call('/v1/tokenize', {'model':'g','messages':messages})
        print('TOKENIZE_MESSAGES', status, json.dumps(tokenized), request_id, flush=True)
        assert status == 200 and request_id == 'tokenize-live-531'
        status, completion, _ = call('/v1/chat/completions', {'model':'g','messages':messages,'max_tokens':1})
        print('CHAT_COMPLETION', status, json.dumps(completion), flush=True)
        assert status == 200 and tokenized['count'] == completion['usage']['prompt_tokens']
        status, raw, _ = call('/v1/tokenize', {'model':'g','prompt':'Hello, world!'})
        assert status == 200
        status, decoded, _ = call('/v1/detokenize', {'model':'g','tokens':raw['tokens']})
        print('RAW_ROUNDTRIP', status, json.dumps(raw), json.dumps(decoded), flush=True)
        assert status == 200 and decoded['prompt'] == 'Hello, world!'
        status, oov, request_id = call('/v1/detokenize', {'model':'g','tokens':[4294967295]})
        print('OOV', status, json.dumps(oov), request_id, flush=True)
        assert status == 400 and oov['error']['param'] == 'prompt_ids' and request_id == 'tokenize-live-531'
        print('LIVE_CHECK_PASS', flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=40)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        print('SERVER_STOPPED', server.returncode, flush=True)
