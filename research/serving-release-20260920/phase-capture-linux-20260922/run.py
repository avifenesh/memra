"""Bounded CPU validation driver; preserves failures without automatic retries."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
out = Path(sys.argv[1])
out.mkdir(exist_ok=False)
manifest = json.loads((root / "source-manifest.json").read_text())


def verify_source():
    hashes = {}
    for name, expected in manifest["files"].items():
        data = (root / name).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected["sha256"] or len(data) != expected["bytes"]:
            raise RuntimeError("source differs: " + name)
        hashes[name] = digest
    return hashes


before = verify_source()
record = {"source": manifest["source"], "driver_pid": os.getpid(),
          "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "python": sys.version, "GPU_work": False, "native_models": False, "runs": []}
(out / "source-before.json").write_text(json.dumps(before, indent=2) + "\n")
env = dict(os.environ, PYTHONPATH=str(root / "tools"), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
failure = False


def owned_fixture_processes():
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        decoded = [arg.decode(errors="replace") for arg in args if arg]
        if not decoded:
            continue
        supervisor = str(root / "tools/serving_process.py") in decoded and "--supervise" in decoded
        fixture = str(root / "tools") in decoded and any(arg.endswith("/fixture.py") for arg in decoded)
        if supervisor or fixture:
            found.append({"pid": int(proc.name), "argv": decoded})
    return found


for mode in ("normal", "optimized"):
    cache = out / (mode + "-bytecode"); cache.mkdir()
    mode_env = dict(env, PYTHONPYCACHEPREFIX=str(cache))
    fixtures = out / (mode + "-fixtures"); fixtures.mkdir()
    mode_env["MEMRA_PHASE_TEST_EVIDENCE_DIR"] = str(fixtures)
    mode_env["MEMRA_CANCEL_TEST_EVIDENCE_DIR"] = str(fixtures)
    options = ["-O"] if mode == "optimized" else []
    jobs = [("tests", [sys.executable, *options, "-m", "unittest", "test_serving_cancel_phase",
                      "test_serving_cancel", "test_serving_cancel_evidence", "test_serving_trace", "test_serving_release"]),
            ("three-phase-replay", [sys.executable, *options, str(root / "probe.py"), str(out / (mode + "-real-replay"))])]
    for kind, command in jobs:
        print(mode + " " + kind + " starting", flush=True)
        started = time.monotonic()
        try:
            run = subprocess.run(command, cwd=root, env=mode_env, capture_output=True, timeout=300)
            stdout, stderr, code = run.stdout, run.stderr, run.returncode
        except subprocess.TimeoutExpired as error:
            stdout, stderr, code = error.stdout or b"", error.stderr or b"", 124
        (out / (mode + "-" + kind + ".stdout")).write_bytes(stdout)
        (out / (mode + "-" + kind + ".stderr")).write_bytes(stderr)
        result = {"mode": mode, "kind": kind, "command": command, "exit_code": code,
                  "elapsed_s": time.monotonic() - started,
                  "stdout_sha256": hashlib.sha256(stdout).hexdigest(), "stderr_sha256": hashlib.sha256(stderr).hexdigest()}
        if kind == "tests":
            text = stderr.decode(errors="replace")
            counts = re.findall(r"Ran (\d+) tests", text)
            result["test_count"] = int(counts[-1]) if counts else None
            result["skipped"] = sum(int(n) for n in re.findall(r"skipped=(\d+)", text))
            if not counts or result["skipped"]:
                result["incomplete_Linux_test_scope"] = True
                failure = True
        record["runs"].append(result)
        (out / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        print(mode + " " + kind + " exit " + str(code), flush=True)
        if code:
            failure = True
            break
    if failure:
        break
remaining = owned_fixture_processes()
initial_remaining = remaining
cleanup_deadline = time.monotonic() + 25
while remaining and time.monotonic() < cleanup_deadline:
    time.sleep(.2)
    remaining = owned_fixture_processes()
record["namespace_cleanup"] = {"initial_remaining": initial_remaining, "final_remaining": remaining}
if remaining:
    failure = True
after = verify_source()
(out / "source-after.json").write_text(json.dumps(after, indent=2) + "\n")
record["source_before_after_equal"] = before == after
record["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
record["status"] = "failed" if failure else "passed"
(out / "run.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
raise SystemExit(1 if failure else 0)
