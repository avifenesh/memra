#!/usr/bin/env python3
"""research/dflash-restorable-hit-20261103/cold-drafter-gate.sh against a stub memra-server, no GPU.

The stub answers /v1/models and /v1/completions on MEMRA_ADDR and writes the serving-shape log
lines the gate reads. Each case sets STUB_MODE:
  ok       - arm and reference agree, the arm logs one veto, one conversion, then tail restores;
  refuse   - every completion is refused with HTTP 400 (a 24 GB card at the GB10 sizes);
  diverge  - the arm's text differs from the reference's;
  noshape  - texts agree, but every arm step is vetoed and converted and none restores (the bug
             memra#1074 fixes);
  quiet    - like ok, but the stub never logs "[server] listening";
  squat    - the stub never serves: it starts a separate process that takes the port and answers.
Only ok may exit 0. Before 2026-11-06 the script exited 0 in all four of the first cases.
The port cases (a foreign server already on the port, squat, quiet) must refuse before any
request reaches a server the gate did not start: on 2026-11-06 a foreign server on 8093 answered
the ready wait and the reference run recorded it.
Needs bash, ss and waitfor on PATH; GATE_SCRIPT points the test at another copy of the script."""
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from shutil import which

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(os.environ.get(
    "GATE_SCRIPT", ROOT / "research/dflash-restorable-hit-20261103/cold-drafter-gate.sh"))
DRIVER = ROOT / "tools/dspark-cold-drafter-gate.py"

STUB = textwrap.dedent('''\
    #!/usr/bin/env python3
    import hashlib, json, os, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    MODE = os.environ["STUB_MODE"]
    ARM = os.environ.get("MEMRA_PREFIX_CACHE_MB", "0") != "0"
    host, port = os.environ["MEMRA_ADDR"].rsplit(":", 1)
    open(os.environ["STUB_MARK"], "a").write(f"started {os.getpid()}\\n")
    seen = {"n": 0}
    def log(line):
        print(line, flush=True)
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass
        def reply(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            self.reply(200, {"data": [{"id": "q38"}]})
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if MODE == "refuse":
                return self.reply(400, {"error": {"message": "prompt does not fit"}})
            prompt = body["prompt"]
            first = seen["n"] == 0
            seen["n"] += 1
            if ARM and not first:  # x0 is the plain prime; a and b steps follow
                if MODE == "noshape" or seen["n"] == 2:
                    log("[dspark] cache-preferred: stub")
                    log("[prefix-cache] DSPARK cold-drafter restore: stub")
                else:
                    log("[prefix-cache] DSPARK restore: stub")
                    # A declined conversion and the glm5 twin are not conversions: the shape
                    # check must not count them (it would read 2 and fail the passing case).
                    log("[prefix-cache] DSPARK cold-drafter restore declined (stub); plain hit")
                    log("[prefix-cache] glm5 cold-drafter restore: stub")
                if MODE != "noshape":
                    log("[prefix-cache] insert (dspark-boundary): stub")
            salt = "arm" if (ARM and MODE == "diverge") else ""
            text = hashlib.sha256((salt + prompt + str(body.get("temperature"))).encode()).hexdigest()[:24]
            self.reply(200, {"choices": [{"text": " " + text}],
                             "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": 8}})
    log("[server] build: stub")
    if MODE == "squat":
        import subprocess, time
        child = subprocess.Popen([sys.executable, os.environ["FOREIGN_SCRIPT"], port, os.environ["FOREIGN_HITS"]])
        open(os.environ["STUB_MARK"], "a").write(f"squatter {child.pid}\\n")
        time.sleep(600)
    srv = HTTPServer((host, int(port)), H)
    if MODE != "quiet":
        log(f"[server] listening on http://{host}:{port}")
    srv.serve_forever()
''')

# A server the gate did not start: answers /v1/models and /v1/completions like a plausible model
# and counts every completion it serves in the file named by its second argument.
FOREIGN = textwrap.dedent('''\
    import json, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    port, hits = int(sys.argv[1]), sys.argv[2]
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass
        def reply(self, obj):
            data = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            self.reply({"data": [{"id": "q38"}]})
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            open(hits, "a").write("completion\\n")
            self.reply({"choices": [{"text": " foreign"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    HTTPServer(("127.0.0.1", port), H).serve_forever()
''')


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@unittest.skipUnless(which("waitfor") and which("bash"), "needs bash and waitfor")
class ColdDrafterGateScript(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.stub = self.dir / "memra-server"
        self.stub.write_text(STUB)
        self.stub.chmod(0o755)
        self.foreign = self.dir / "foreign.py"
        self.foreign.write_text(FOREIGN)
        self.hits = self.dir / "foreign-hits"
        self.mark = self.dir / "stub-mark"
        self.procs = []

    def tearDown(self):
        for p in self.procs:
            p.kill()
            p.wait()
        if self.mark.exists():  # the squat stub's port-taking child outlives the stub
            for kind, pid in (l.split() for l in self.mark.read_text().splitlines()):
                if kind == "squatter":
                    try:
                        os.kill(int(pid), 9)
                    except ProcessLookupError:
                        pass
        self.tmp.cleanup()

    def gate(self, mode, vram="24463", port=None):
        out = self.dir / mode
        env = dict(os.environ, STUB_MODE=mode, GATE=str(DRIVER), GATE_VRAM_MIB=vram,
                   GATE_PORT=str(port or free_port()), STUB_MARK=str(self.mark),
                   FOREIGN_SCRIPT=str(self.foreign), FOREIGN_HITS=str(self.hits))
        for k in ("GATE_P_PARAS", "GATE_S_PARAS", "GATE_ADDR"):
            env.pop(k, None)
        p = subprocess.run(["bash", str(SCRIPT), str(self.stub), str(out)], env=env,
                           capture_output=True, text=True, timeout=300)
        report = (out / "compare.txt").read_text() if (out / "compare.txt").exists() else ""
        return p.returncode, p.stdout + p.stderr, report

    def foreign_server(self):
        """Start a server the gate did not start on a free port; returns the port once it listens."""
        port = free_port()
        p = subprocess.Popen([sys.executable, str(self.foreign), str(port), str(self.hits)])
        self.procs.append(p)
        for _ in range(100):
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    return port
            time.sleep(0.05)
        self.fail("foreign server did not listen")

    def assert_nothing_measured(self, mode, log):
        out = self.dir / mode
        self.assertFalse((out / "arm.jsonl").exists(), log)
        self.assertFalse((out / "ref.jsonl").exists(), log)
        self.assertFalse(self.hits.exists(), f"a foreign server served completions:\n{log}")

    def test_ok_passes_and_names_its_sizes(self):
        rc, log, report = self.gate("ok")
        self.assertEqual(rc, 0, log)
        self.assertIn("gate sizes GATE_P_PARAS=10 GATE_S_PARAS=14 memory.total=24463 MiB", report)
        self.assertIn("(GATE_VRAM_MIB override)", report)
        self.assertIn("SHAPE PASS", report)
        self.assertIn("COLD-DRAFTER GATE PASS", report)

    def test_refused_requests_fail(self):
        rc, log, report = self.gate("refuse")
        self.assertNotEqual(rc, 0, log)
        self.assertIn("HTTP 400", log)
        self.assertIn("arm run FAILED", report)
        self.assertIn("missing run file", report)

    def test_divergent_text_fails(self):
        rc, log, report = self.gate("diverge", vram="[N/A]")  # GB10 reports no dedicated memory
        self.assertNotEqual(rc, 0, log)
        self.assertIn("gate sizes GATE_P_PARAS=70 GATE_S_PARAS=80", report)
        self.assertIn("GATE FAIL", report)

    def test_reveto_shape_fails(self):
        rc, log, report = self.gate("noshape")
        self.assertNotEqual(rc, 0, log)
        self.assertIn("SHAPE FAIL", report)

    def test_occupied_port_is_refused_before_boot(self):
        port = self.foreign_server()
        # a previous run's passing arm log in the same out dir must not be read as this run's
        stale = self.dir / "ok"
        stale.mkdir()
        (stale / "serve-arm.log").write_text(
            "[dspark] cache-preferred: old\n[prefix-cache] DSPARK cold-drafter restore: old\n"
            "[prefix-cache] DSPARK restore: old\n[prefix-cache] insert (dspark-boundary): old\n")
        rc, log, report = self.gate("ok", port=port)
        self.assertNotEqual(rc, 0, log)
        self.assertIn(f"port {port} is already LISTENing", log)
        self.assertIn("GATE_PORT", log)
        self.assertFalse(self.mark.exists(), f"the gate started a server on an occupied port:\n{log}")
        self.assert_nothing_measured("ok", log)
        self.assertNotIn("SHAPE PASS", report)
        self.assertIn("no serve-arm.log", report)
        self.assertIn("COLD-DRAFTER GATE FAIL", report)

    def test_port_taken_after_the_preflight_is_refused(self):
        rc, log, report = self.gate("squat")
        self.assertNotEqual(rc, 0, log)
        self.assertIn("is NOT owned by our server", log)
        self.assert_nothing_measured("squat", log)
        self.assertIn("COLD-DRAFTER GATE FAIL", report)

    def test_responder_without_our_listening_line_is_refused(self):
        rc, log, report = self.gate("quiet")
        self.assertNotEqual(rc, 0, log)
        self.assertIn('has no "[server] listening on http://127.0.0.1:', log)
        out = self.dir / "quiet"
        self.assertFalse((out / "arm.jsonl").exists(), log)
        self.assertFalse((out / "ref.jsonl").exists(), log)
        self.assertIn("COLD-DRAFTER GATE FAIL", report)


if __name__ == "__main__":
    unittest.main()
