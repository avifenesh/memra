#!/usr/bin/env python3
"""One boot, the same concurrency several times: does the cost gate learn across waves?
usage: waves_once.py BIN OUTDIR ARM C REPEATS"""
import importlib.util, sys, os
spec = importlib.util.spec_from_file_location("g", os.path.join(os.path.dirname(__file__), "gate_ab.py"))
bin_, out, arm, c, reps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
sys.argv = ["gate_ab.py", bin_, out, "--reps", "0", "--cs", str(c)]
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
g.CS = [c] * reps
import subprocess, time, urllib.request, json
log = os.path.join(out, f"{arm}-waves-server.log")
env = dict(os.environ); env.update({"MEMRA_MODELS": g.a.models, "MEMRA_ADDR": f"127.0.0.1:{g.a.port}", "MEMRA_CTX": g.a.ctx})
if arm == "band": env["MEMRA_SPEC_GATE_MODEL"] = "0"
p = subprocess.Popen([bin_], env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(900):
        try: urllib.request.urlopen(g.BASE + "/v1/models", timeout=3).read(); break
        except Exception: time.sleep(1)
    g.wave(1, "warm")
    print(arm, [round(g.wave(c, r), 1) for r in range(reps)], flush=True)
finally:
    p.terminate(); p.wait(timeout=90)
