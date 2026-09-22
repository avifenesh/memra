"""Local CPU drain protocol fixture. No model, worker or GPU proof."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import signal
import sys
import threading
import time

port, mode = int(sys.argv[1]), sys.argv[2]
guard = threading.Lock()
draining = False
active = 0
seen = set()


def log(value):
    print(value, flush=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def fixed(self, status, body, extra=()):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Connection", "close")
        self.send_header("Content-Length", str(len(data)))
        for name, value in extra:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()

    def do_GET(self):
        log("HTTP " + self.headers.get("X-Request-Id", "unknown"))
        if self.path == "/health":
            if draining:
                with guard: seen.add("health")
            return self.fixed(200, {"status": "draining" if draining else "ok", "worker": {"generation": 7}})
        if self.path == "/readyz":
            with guard: seen.add("ready")
            if mode == "malformed_ready":
                return self.fixed(503, b"{", [("Retry-After", "3")])
            if mode == "framing_failure":
                self.send_response(503); self.send_header("Connection", "close")
                self.send_header("Transfer-Encoding", "chunked"); self.send_header("Retry-After", "3")
                self.end_headers(); self.wfile.write(b'22\r\n{"status":"not_ready","x":"hello"}\r\n0\r\n')
                self.wfile.flush(); return
            return self.fixed(503 if draining else 200, {"status": "not_ready" if draining else "ready"}, [("Retry-After", "3")])
        self.fixed(404, {})

    def do_POST(self):
        global active
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        name = self.headers.get("X-Request-Id", "unknown")
        log("HTTP " + name)
        if draining:
            with guard: seen.add("new")
            return self.fixed(503, {"error": {"code": "draining", "type": "server_error", "message": "draining"}}, [("Retry-After", "3")])
        with guard: active += 1
        try:
            self.send_response(200)
            self.send_header("Connection", "close")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("X-Fixture-Role", "inflight")
            self.end_headers()
            def chunk(data):
                self.wfile.write(f"{len(data):x}\r\n".encode()+data+b"\r\n"); self.wfile.flush()
            prefix = b'data: {"model":"gate","choices":[{"index":0,"delta":{"content":"working "},"finish_reason":null}]}\n\n'
            chunk(prefix)
            if mode != "fast_complete":
                until = time.monotonic()+8
                while time.monotonic()<until:
                    with guard: ready = draining and seen >= {"health", "ready", "new"}
                    if ready and mode != "stuck": break
                    time.sleep(.005)
                # Keep the real process/listener observable while the controller
                # processes the post-connect snapshots; not a latency assertion.
                time.sleep(.7)
            if mode == "truncated_stream": return
            if mode == "typed_error":
                chunk(b'data: {"error":{"code":"worker_failed","type":"server_error","message":"fixture fault"}}\n\ndata: [DONE]\n\n')
            else:
                chunk(b'data: {"model":"gate","choices":[{"index":0,"delta":{"content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":2,"total_tokens":14}}\n\ndata: [DONE]\n\n')
            self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
            log("TERMINAL " + name)
        except (BrokenPipeError, ConnectionResetError):
            log("DISCONNECTED " + name)
        finally:
            with guard: active -= 1


with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
    def term(*_args):
        global draining
        draining = True
        with guard: n = active
        log(f"[server] SIGTERM: draining ({n} in flight, deadline 3s)")
        def finish():
            begin = time.monotonic()
            while True:
                with guard: n = active
                if n == 0:
                    log(f"[server] drain complete in {time.monotonic()-begin:.1f}s; exiting")
                    server.shutdown(); return
                time.sleep(.005)
        threading.Thread(target=finish, daemon=True).start()
    signal.signal(signal.SIGTERM, term)
    log("READY")
    server.serve_forever(poll_interval=.01)
