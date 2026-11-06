#!/usr/bin/env python3
"""research/dflash-restorable-hit-20261103/cold-drafter-gate.sh against a stub memra-server, no GPU.

The stub answers /v1/models and /v1/completions on MEMRA_ADDR and writes the serving-shape log
lines the gate reads. Each case sets STUB_MODE:
  ok       - arm and reference agree, the arm logs one veto, one conversion, then tail restores;
  refuse   - every completion is refused with HTTP 400 (a 24 GB card at the GB10 sizes);
  diverge  - the arm's text differs from the reference's;
  noshape  - texts agree, but every arm step is vetoed and converted and none restores (the bug
             memra#1074 fixes).
Only ok may exit 0. Before 2026-11-06 the script exited 0 in all four cases.
Needs bash and waitfor on PATH; GATE_SCRIPT points the test at another copy of the script."""
import os
import socket
import subprocess
import tempfile
import textwrap
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
                if MODE != "noshape":
                    log("[prefix-cache] insert (dspark-boundary) stub")
            salt = "arm" if (ARM and MODE == "diverge") else ""
            text = hashlib.sha256((salt + prompt + str(body.get("temperature"))).encode()).hexdigest()[:24]
            self.reply(200, {"choices": [{"text": " " + text}],
                             "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": 8}})
    log("[server] build: stub")
    HTTPServer((host, int(port)), H).serve_forever()
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

    def tearDown(self):
        self.tmp.cleanup()

    def gate(self, mode, vram="24463"):
        out = self.dir / mode
        env = dict(os.environ, STUB_MODE=mode, GATE=str(DRIVER), GATE_VRAM_MIB=vram,
                   GATE_ADDR=f"127.0.0.1:{free_port()}")
        for k in ("GATE_P_PARAS", "GATE_S_PARAS"):
            env.pop(k, None)
        p = subprocess.run(["bash", str(SCRIPT), str(self.stub), str(out)], env=env,
                           capture_output=True, text=True, timeout=300)
        report = (out / "compare.txt").read_text() if (out / "compare.txt").exists() else ""
        return p.returncode, p.stdout + p.stderr, report

    def test_ok_passes_and_names_its_sizes(self):
        rc, log, report = self.gate("ok")
        self.assertEqual(rc, 0, log)
        self.assertIn("gate sizes GATE_P_PARAS=10 GATE_S_PARAS=14 memory.total=24463 MiB", report)
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


if __name__ == "__main__":
    unittest.main()
