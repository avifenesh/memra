#!/usr/bin/env python3
"""Native image URL acceptance with deterministic loopback fixtures and bounded failures."""
import argparse
import base64
import hashlib
import http.server
import json
import os
from pathlib import Path
import re
import signal
import socket
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary', type=Path, required=True)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--mmproj', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--server-port', type=int, required=True)
p.add_argument('--fixture-port', type=int, required=True)
p.add_argument('--external-lock', type=int, required=True)
p.add_argument('--door-off', action='store_true')
p.add_argument('--vision-kind', choices=('gemma', 'qwen'), default='gemma')
p.add_argument('--context', type=int, default=8192)
p.add_argument('--transport-only', action='store_true', help='No vision qualification; test HTTP controls with the text-only trunk')
p.add_argument('--test-binary', type=Path)
p.add_argument('--with-background', action='store_true', help='Exercise remote image admission with the background Responses switch enabled')
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
assert os.path.samefile(f'/proc/self/fd/{a.external_lock}', os.environ['MEMRA_GPU_LOCK'])
with socket.socket() as probe:
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    probe.bind(('127.0.0.1', a.server_port))

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def png(rgb):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    pixels = (b'\x00' + bytes(rgb) * 128) * 128
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 128, 128, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b'')

images = {'red': png((255, 0, 0)), 'blue': png((0, 0, 255))}
for colour, data in images.items():
    (a.out / f'{colour}.png').write_bytes(data)
hits = []
hits_lock = threading.Lock()
fixture_log = (a.out / 'fixture.jsonl').open('w')
class Fixture(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, *_):
        pass
    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        with hits_lock:
            hits.append(path)
            fixture_log.write(json.dumps({'at': time.monotonic(), 'path': path}) + '\n')
            fixture_log.flush()
        try:
            if path.startswith('/chain/'):
                left = int(path.rsplit('/', 1)[1])
                if left:
                    self.send_response(302)
                    self.send_header('Location', f'/chain/{left - 1}')
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                path = '/red.png'
            if path in ('/redirect-private', '/redirect-dns', '/redirect-scheme'):
                target = {'/redirect-private': 'http://169.254.169.254/latest/meta-data/', '/redirect-dns': f'http://localhost:{a.fixture_port}/red.png', '/redirect-scheme': 'file:///etc/passwd'}[path]
                self.send_response(302)
                self.send_header('Location', target)
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            if path == '/slow-header':
                time.sleep(11)
                path = '/red.png'
            if path.startswith('/slow-pass/'):
                time.sleep(7.5)
                path = '/red.png'
            if path in ('/large-header', '/budget-header'):
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Content-Length', str(12 * 1024 * 1024 + 1 if path == '/large-header' else 2 * 1024 * 1024))
                self.end_headers()
                self.close_connection = True
                return
            if path == '/grow':
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Transfer-Encoding', 'chunked')
                self.end_headers()
                data = b'x' * (256 * 1024)
                for _ in range(53):
                    self.wfile.write(f'{len(data):x}\r\n'.encode() + data + b'\r\n')
                self.wfile.write(b'0\r\n\r\n')
                return
            if path == '/status':
                self.send_response(503)
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            if path == '/bad-mime':
                data, content_type = b'<html>not an image</html>', 'text/html'
            else:
                data = images['blue' if path == '/blue.png' else 'red']
                content_type = 'image/png'
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data) + (100 if path == '/truncated' else 0)))
            self.end_headers()
            self.wfile.write(data)
            if path == '/truncated':
                self.close_connection = True
        except (BrokenPipeError, ConnectionResetError):
            pass

class FixtureServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

fixture = FixtureServer(('127.0.0.1', a.fixture_port), Fixture)
fixture_thread = threading.Thread(target=fixture.serve_forever, daemon=True)
fixture_thread.start()
url_root = f'http://127.0.0.1:{a.fixture_port}'
manifest = {'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'binary_sha256': sha(a.binary), 'model': str(a.model), 'model_sha256': sha(a.model), 'mmproj': str(a.mmproj), 'mmproj_sha256': sha(a.mmproj), 'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader'], text=True), 'context': a.context, 'temperature': 0, 'cache': 'prefix cache disabled', 'door': 'off' if a.door_off else 'on', 'vision_kind': a.vision_kind, 'spec': 'plain', 'scope': 'http_transport_only' if a.transport_only else 'native_vision', 'background_responses': a.with_background, 'fixture_sha256': {k: hashlib.sha256(v).hexdigest() for k, v in images.items()}}
(a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
env = {k: v for k, v in os.environ.items() if not k.startswith('MEMRA_') or k in {'MEMRA_GPU_LOCK', 'MEMRA_CI_LOCK', 'MEMRA_CI_LOCK_HELD', 'MEMRA_RIG_LOCK_FD'}}
env.update(MEMRA_MODELS='vision=' + str(a.model), MEMRA_GEMMA_VISION='1', MEMRA_GEMMA_MMPROJ=str(a.mmproj), MEMRA_CTX=str(a.context), MEMRA_ADDR=f'127.0.0.1:{a.server_port}', MEMRA_PREFIX_CACHE_MB='0', MEMRA_FETCH_URLS='0' if a.door_off else '1', MEMRA_FETCH_URLS_ALLOWED_HOSTS='127.0.0.1', MEMRA_API_KEY='image-gate-token')
env['MEMRA_SPEC'] = '0'
if a.with_background:
    env['MEMRA_BACKGROUND_RESPONSES'] = '1'
if a.vision_kind == 'qwen':
    env.pop('MEMRA_GEMMA_VISION', None)
    env.pop('MEMRA_GEMMA_MMPROJ', None)
    env['MEMRA_VISION_DIR'] = str(a.mmproj.parent)
if a.transport_only:
    env.pop('MEMRA_VISION_DIR', None)
    env['MEMRA_GEMMA_VISION'] = '0'
    env.pop('MEMRA_GEMMA_MMPROJ', None)
if not a.door_off and a.test_binary is not None:
    test_env = env.copy()
    test_env.update(IMAGE_FETCH_FIXTURE_URL=url_root, IMAGE_FETCH_FIXTURE_DIR=str(a.out))
    with (a.out / 'transport-unit.log').open('w') as log:
        result = subprocess.run([str(a.test_binary), '--ignored', '--exact', 'image_fetch::tests::controlled_http_fixture_rewrites_exact_image_bytes', '--nocapture'], env=test_env, stdout=log, stderr=subprocess.STDOUT, timeout=60, pass_fds=(a.external_lock,))
    assert result.returncode == 0, 'transport byte fixture failed; inspect transport-unit.log'
    report = (a.out / 'transport-unit.log').read_text()
    assert 'IMAGE_FETCH_FIXTURE_PASS' in report and '1 passed' in report, 'fixture must execute, not filter out'
server = subprocess.Popen([str(a.binary)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, pass_fds=(a.external_lock,))
ready, listening = threading.Event(), threading.Event()
def reader():
    with (a.out / 'server.log').open('w') as f:
        for line in server.stdout:
            f.write(line)
            f.flush()
            if '[server] listening on ' in line:
                listening.set()
                ready.set()
        ready.set()
reader_thread = threading.Thread(target=reader)
reader_thread.start()
http_log = (a.out / 'http.jsonl').open('w')
verdicts = {}
prompt = 'Is the dominant color of this square red or blue? Answer with one word, red or blue.'
def payload(urls, response=False):
    if response:
        content = [{'type': 'input_text', 'text': prompt}] + [{'type': 'input_image', 'image_url': u} for u in urls]
        return {'model': 'vision', 'input': [{'role': 'user', 'content': content}], 'max_output_tokens': 48, 'temperature': 0, 'reasoning': {'effort': 'none'}}
    content = [{'type': 'text', 'text': prompt}] + [{'type': 'image_url', 'image_url': {'url': u}} for u in urls]
    return {'model': 'vision', 'messages': [{'role': 'user', 'content': content}], 'max_tokens': 48, 'temperature': 0, 'reasoning_effort': 'none'}
def request(case, data, response=False):
    raw = json.dumps(data).encode()
    endpoint = '/v1/responses' if response else '/v1/chat/completions'
    req = urllib.request.Request(f'http://127.0.0.1:{a.server_port}' + endpoint, data=raw, headers={'Authorization': 'Bearer image-gate-token', 'Content-Type': 'application/json'})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=60) as result:
            status, body = result.status, result.read()
    except urllib.error.HTTPError as result:
        status, body = result.code, result.read()
    elapsed = time.monotonic() - started
    value = json.loads(body)
    http_log.write(json.dumps({'case': case, 'endpoint': endpoint, 'request_bytes': len(raw), 'request_sha256': hashlib.sha256(raw).hexdigest(), 'elapsed_s': elapsed, 'status': status, 'body': value}) + '\n')
    http_log.flush()
    return status, value, elapsed

def background_result(ident):
    deadline = time.monotonic() + 60
    sample = 0
    while time.monotonic() < deadline:
        req = urllib.request.Request(f'http://127.0.0.1:{a.server_port}/v1/responses/' + urllib.parse.quote(ident, safe=''), headers={'Authorization': 'Bearer image-gate-token'})
        with urllib.request.urlopen(req, timeout=min(10, max(0.1, deadline - time.monotonic()))) as result:
            status, value = result.status, json.loads(result.read())
        http_log.write(json.dumps({'case': 'background-poll-' + str(sample), 'status': status, 'body': value}) + '\n')
        http_log.flush()
        assert status == 200, value
        if value['status'] not in ('queued', 'in_progress'):
            return status, value, 0
        sample += 1
        threading.Event().wait(0.1)
    raise AssertionError('background remote image completion deadline')

def successful(result, response=False, colour='red'):
    status, value, _ = result
    assert status == 200, (status, value)
    if response:
        assert value['status'] == 'completed', value
        text = ''.join(c['text'] for item in value['output'] if item.get('type') == 'message' for c in item['content'] if c.get('type') == 'output_text')
    else:
        assert value['choices'][0]['finish_reason'] == 'stop', value
        text = value['choices'][0]['message']['content']
    words = re.findall('[a-z]+', text.lower())
    assert colour in words and ('blue' if colour == 'red' else 'red') not in words, text
    return text

def refuse(case, urls, code, response=False, **extra):
    data = payload(urls, response)
    data.update(extra)
    status, value, elapsed = request(case, data, response)
    assert status == 400, (case, status, value)
    assert value['error'].get('code') == code, (case, value)
    verdicts[case] = {'pass': True, 'elapsed_s': elapsed}
    return elapsed

try:
    assert ready.wait(180) and listening.is_set() and server.poll() is None, 'model startup failed; inspect server.log'
    red_uri = 'data:image/png;base64,' + base64.b64encode(images['red']).decode()
    if a.transport_only:
        status, control, _ = request('text-only-control', {'model':'vision', 'messages':[{'role':'user','content':'Reply OK.'}], 'temperature':0, 'reasoning_effort':'none', 'max_tokens':32})
        assert status == 200, (status, control)
        verdicts['text_only_endpoint_control'] = {'pass': True}
    else:
        inline = request('inline-red-control', payload([red_uri]))
        inline_text = successful(inline)
    if a.door_off:
        before = len(hits)
        for response in (False, True):
            status, value, _ = request('door-off-' + str(response), payload([url_root + '/red.png'], response), response)
            assert status == 400, value
            if not a.transport_only:
                assert 'disabled' in value['error']['message'], value
        assert len(hits) == before, hits
        verdicts['door_off_no_fetch'] = {'pass': True}
    else:
        before = len(hits)
        refuse('literal-public-requires-allowlist', ['http://8.8.8.8/x'], 'image_url_blocked')
        refuse('literal-private-blocked', ['http://10.0.0.1/x'], 'image_url_blocked')
        refuse('dns-loopback-blocked', [f'http://localhost:{a.fixture_port}/red.png'], 'image_url_blocked')
        assert len(hits) == before
        for case in ('redirect-private', 'redirect-dns', 'redirect-scheme'):
            before = len(hits)
            refuse(case, [url_root + '/' + case], 'image_url_blocked')
            assert len(hits) == before + 1, (case, hits[before:])
        before = len(hits)
        refuse('mixed-image-count', [red_uri] * 8 + [url_root + '/red.png'], None)
        assert len(hits) == before
        for route in ('large-header', 'grow'):
            refuse(route, [url_root + '/' + route], 'image_url_too_large')
        for route in ('bad-mime', 'status', 'truncated'):
            refuse(route, [url_root + '/' + route], 'image_url_unreachable')
        refuse('redirect-cap', [url_root + '/chain/6'], 'image_url_blocked')
        if a.transport_only:
            verdicts['production_fetch_exact_bytes_and_five_redirects'] = {'pass': True, 'evidence': 'transport-unit.log'}
        else:
            redirected = request('five-redirect-control', payload([url_root + '/chain/5']))
            assert successful(redirected) == inline_text
            verdicts['five_redirects_and_cap'] = {'pass': True}
            for response in (False, True):
                remote = request('remote-red-' + str(response), payload([url_root + '/red.png'], response), response)
                assert successful(remote, response) == inline_text
            blue_uri = 'data:image/png;base64,' + base64.b64encode(images['blue']).decode()
            blue_inline = successful(request('inline-blue-control', payload([blue_uri])), colour='blue')
            blue_remote = successful(request('remote-blue-control', payload([url_root + '/blue.png'])), colour='blue')
            assert blue_inline == blue_remote
            verdicts['native_vision_and_inline_url_identity'] = {'pass': True}
        if not a.transport_only:
            for response in (False, True):
                naked = payload([url_root + '/red.png'], response)
                for key in ('max_tokens', 'max_output_tokens', 'temperature', 'reasoning_effort', 'reasoning'):
                    naked.pop(key, None)
                assert set(naked) == ({'model', 'input'} if response else {'model', 'messages'})
                (a.out / ('vendor-default-responses-request.json' if response else 'vendor-default-chat-request.json')).write_text(json.dumps(naked, indent=2))
                result = request('vendor-default-' + str(response), naked, response)
                successful(result, response)
                verdicts['vendor_default_' + ('responses' if response else 'chat')] = {'pass': True, 'decode_fields': []}
        if a.with_background and not a.transport_only:
            refuse('background-literal-blocked', ['http://10.0.0.1/x'], 'image_url_blocked', response=True, background=True)
            elapsed = refuse('background-caller-deadline', [url_root + '/slow-header'], 'image_url_unreachable', response=True, background=True, timeout_ms=1000)
            assert 0.7 <= elapsed <= 4, elapsed
            submitted = payload([url_root + '/red.png'], True)
            submitted['background'] = True
            (a.out / 'background-remote-request.json').write_text(json.dumps(submitted, indent=2))
            status, queued, _ = request('background-remote-submit', submitted, True)
            assert status == 200 and queued['status'] == 'queued', queued
            assert successful(background_result(queued['id']), True) == inline_text
            verdicts['background_remote_image_and_preflight'] = {'pass': True, 'queued_then_completed': True, 'blocked_literal_refused_before_queue': True, 'caller_deadline_s': elapsed}
        for response in (False, True):
            refuse('whole-byte-budget-' + str(response), [url_root + '/budget-header'], 'image_url_too_large', response=response, _padding='x' * (192 * 1024 * 1024 - 1024 * 1024 - 4096))
        elapsed = refuse('per-image-timeout', [url_root + '/slow-header'], 'image_url_unreachable')
        assert 9 <= elapsed <= 14, elapsed
        elapsed = refuse('caller-deadline', [url_root + '/slow-header'], 'image_url_unreachable', timeout_ms=1000)
        assert 0.7 <= elapsed <= 4, elapsed
        elapsed = refuse('whole-pass-timeout', [url_root + f'/slow-pass/{i}' for i in range(3)], 'image_url_unreachable')
        assert 19 <= elapsed <= 24, elapsed
    verdicts['scope'] = 'http_transport_only' if a.transport_only else 'native_vision'
    verdicts['vision_acceptance'] = 'blocked: gemma4uv front end is unimplemented' if a.transport_only else 'passed'
    verdicts['pass'] = True
    (a.out / 'summary.json').write_text(json.dumps(verdicts, indent=2))
    print(json.dumps(verdicts), flush=True)
finally:
    http_log.close()
    if server.poll() is None:
        server.send_signal(signal.SIGINT)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)
    reader_thread.join(timeout=5)
    fixture.shutdown()
    fixture.server_close()
    fixture_thread.join(timeout=5)
    fixture_log.close()
