"""Real final-chunk diagnostic, target-only CLI, not a serving qualification."""
import fcntl
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

p = pathlib.Path("/root/qwen-prefill-attn-20260909")
root = p / "fa2-src"
fatbin, kernel = sys.argv[1:3]
profile = json.loads((p / "requal-profile.json").read_text())
request = json.loads((p / "request-131k.json").read_text())
prompt = root / "real-prompt.txt"
prompt.write_text(request["messages"][0]["content"])
binary = root / "fa2-real-chunk-probe-v1"
receipt = {"profile_sha256": hashlib.sha256((p / "requal-profile.json").read_bytes()).hexdigest(),
           "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
           "fatbin_sha256": hashlib.sha256((root / fatbin).read_bytes()).hexdigest(),
           "kernel": kernel, "scope": "baseline prefix then final 1024 rows, depth 32768; target-only diagnostic", "arms": []}
for arm in ["off", "on", "off-twin"]:
    lock = open("/tmp/memra-gpu.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        for _ in range(50):
            if not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True).strip():
                break
            time.sleep(.1)
        else:
            raise RuntimeError("compute list occupied under lock")
        env = os.environ.copy()
        env.update(profile)
        env["MEMRA_MODEL_METADATA"] = str(p / "requal-models.toml")
        for key in ["FA2_PROBE_ACTIVE", "FA2_PROBE_KERNEL", "FA2_PROBE_DUMP", "FA2_PROBE_FATBIN"]:
            env.pop(key, None)
        if arm == "on":
            env.update(FA2_PROBE_KERNEL=kernel, FA2_PROBE_FATBIN=str(root / fatbin))
        if arm != "off-twin":
            env["FA2_PROBE_DUMP"] = str(root / ("real-input-" + arm))
        command = [str(binary), profile["MEMRA_MODELS"].split("=", 1)[1], str(prompt), str(root / ("real-" + arm)), "32768"]
        start = time.time()
        with (root / ("real-" + arm + ".log")).open("w") as log:
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            print(json.dumps({"arm": arm, "pid": proc.pid}), flush=True)
            code = proc.wait()
        receipt["arms"].append(dict(arm=arm, pid=proc.pid, exit_code=code, elapsed_s=time.time()-start))
        (root / "real-chunk-run.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if code:
            raise RuntimeError(f"{arm} failed: {code}")
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    time.sleep(1)
