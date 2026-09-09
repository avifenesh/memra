"""Pinned on-box CLI gates, one canonical-lock GPU process per invocation."""
import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import subprocess
import time
import uuid

p = argparse.ArgumentParser()
p.add_argument("kind", choices=["margin", "spec", "eval", "layers"])
p.add_argument("--arm", choices=["0", "1"], default="1")
p.add_argument("--profile", action="store_true")
p.add_argument("--version", default="v2")
a = p.parse_args()
root = pathlib.Path("/root/qwen-prefill-attn-20260909")
out = root / "fa2-gates" / (a.kind + "-" + a.arm + "-" + uuid.uuid4().hex[:8])
out.mkdir(parents=True)
profile = json.loads((root / "requal-profile.json").read_text())
env = {k: v for k, v in os.environ.items() if not k.startswith(("MEMRA_", "FA2_PROBE_"))}
env.update(profile)
env.update(MEMRA_PRIME_ATTN_FA2=a.arm, MEMRA_MODEL_METADATA=str(root / "requal-models.toml"), MEMRA_SEED="5183")
model = profile["MEMRA_MODELS"].split("=", 1)[1]
if a.kind == "margin":
    cmd = [str(root / ("fa2-src/qwen-fa2-margin-gate-" + a.version)), model, str(out / "logits"), str(root / "margin-board-2048.txt"), str(root / "extra-margin-prompt.txt")]
elif a.kind == "spec":
    cmd = [str(root / ("fa2-src/run-spec-" + a.version)), model]
    env.update(MEMRA_PROMPT_FILE=str(root / "gate-prompt.txt"), MEMRA_CHAT="1", MEMRA_NGEN="64")
elif a.kind == "layers":
    cmd = [str(root / ("fa2-src/qwen-fa2-layer-probe-" + a.version)), model, str(root / "margin-board-2048.txt")]
else:
    cmd = [str(root / ("fa2-src/concat-prime-probe-" + a.version)), model, "nllwin", "--prompt-a", "@" + str(root / "fa2-serving-src/research/fp8st-20260804/mmq-v2/nll-window.txt"), "--window", "1024", "--chunk", "1024", "--jsonl", str(out / "per-token.jsonl")]
binary = pathlib.Path(cmd[0])
receipt = dict(kind=a.kind, arm=a.arm, nonce=uuid.uuid4().hex, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), profile_sha256=hashlib.sha256((root / "requal-profile.json").read_bytes()).hexdigest(), command=cmd, profile=profile)
if a.profile:
    cmd = ["/opt/nvidia/nsight-systems/2026.1.3/bin/nsys", "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--cuda-flush-interval=1000", "--sample=none", "--cpuctxsw=none", "--force-overwrite=true", "--output=" + str(out / "trace")] + cmd
print(out, flush=True)
with open("/tmp/memra-gpu.lock", "a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    for _ in range(50):
        busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True).strip()
        if not busy:
            break
        time.sleep(.1)
    else:
        raise RuntimeError("GPU occupied under lock: " + busy)
    start = time.time()
    with (out / "gate.log").open("w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipt.update(pid=proc.pid, started_utc=start)
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        code = proc.wait()
    receipt.update(exit_code=code, elapsed_s=time.time()-start)
    receipt["gpu_after"] = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True).strip()
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"output": str(out), "exit_code": code}), flush=True)
raise SystemExit(code)
