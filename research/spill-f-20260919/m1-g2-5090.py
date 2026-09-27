#!/usr/bin/env python3
"""G2 5090 half (OWED 23; M1-PREREG.md section D): D's G2 protocol on the local RTX 5090.

Run only as the collector's --execute child:
  tier-battery.py --rig rtx5090 --external-lock --out OUT/collector --execute \
    python3 m1-g2-5090.py --probe h2d-probe --out OUT/visits --lock-fd @COLLECTOR_LOCK_FD@

Reuses research/spill-d-20260919/run-g2.py's helpers (copy calibration, the output checker) and
its visit shape. Differences, all registered: the ten registered sizes; the canonical 5090 lock;
D's 600/600 W check (a PRO 6000 envelope) replaced by "every sample's power fields equal the
first sample's", because the laptop card has no settable 600 W cap. Scored visits must still
last at least 250 ms, and a competing compute application aborts the campaign.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("run_g2", ROOT / "research/spill-d-20260919/run-g2.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
gspec = importlib.util.spec_from_file_location("gpusampler", HERE / "m1-gpu-sampler.py")
GPU = importlib.util.module_from_spec(gspec)
gspec.loader.exec_module(GPU)  # telemetry amendment: recording only
SIZES = [4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456, 1073741824]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lock-fd", type=int, required=True)
    a = p.parse_args()
    proof = G.L.verify(a.lock_fd, G.B.LOCKS["rtx5090"])
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "worker-lock.json").write_text(json.dumps(proof, indent=2) + "\n")
    identity = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "binary_sha256": G.B.digest(a.probe), "worker_sha256": G.B.digest(Path(__file__)),
                "d_worker_sha256": G.B.digest(ROOT / "research/spill-d-20260919/run-g2.py"),
                "probe_source_sha256": G.B.digest(ROOT / "crates/memra-engine/src/bin/h2d_probe.rs"),
                "collector_sha256": G.B.digest(ROOT / "tools/tier-battery.py"), "sizes": SIZES, "qualification": False}
    (a.out / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    envelope = {}

    def check(text, size, copies, order, scored):
        samples = G.C.check(text, size, copies, order)
        for row in samples:
            G.B.require(row["evidence_class"] == "n1-plumbing-not-qualified", "not native samples")
            power = {k: row["power_before"][k] for k in ("power.limit", "power.max_limit")}
            if not envelope:
                envelope.update(power)
                (a.out / "power-envelope.json").write_text(json.dumps(envelope) + "\n")
            G.B.require(power == envelope, f"power fields changed: {power} != {envelope}")
            if scored:
                G.B.require(row["wall_ns"] >= G.MIN_NS, "scored visit shorter than 250ms; campaign not scored")
        return samples

    def visit(size, copies, order, name, scored=False):
        inventory = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                             "--format=csv,noheader"], text=True)
        (a.out / (name + ".compute.log")).write_text(inventory)
        G.B.require(not inventory.strip(), "competing GPU process; campaign aborted")
        raw = a.out / (name + ".log")
        gpu = GPU.GpuSampler(a.out / (name + ".gpu.csv")).start()
        command = [str(a.probe.resolve()), "--bytes", str(size), "--copies", str(copies),
                   "--order", order, "--direction", "both", "--repeats", "1"]
        with raw.open("xb") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, pass_fds=(a.lock_fd,), check=False)
        telemetry = gpu.stop()
        G.B.require(result.returncode == 0, "probe failed; raw log " + raw.name)
        samples = check(raw.read_text(), size, copies, order, scored)
        with (a.out / "samples.jsonl").open("a") as stream:
            for s in samples:
                stream.write(json.dumps({"phase": "scored" if scored else "calibration", "visit": name,
                                         "raw_log": raw.name, "raw_sha256": G.B.digest(raw),
                                         "gpu_telemetry": telemetry, **s}) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        print(json.dumps({"visit": name, "copies": copies, "minimum_wall_ns": min(s["wall_ns"] for s in samples),
                          "scored": scored}), flush=True)
        return samples

    calibrated = {}
    for size in SIZES:
        copies = 1000 if size <= 1048576 else 10
        for attempt in range(5):
            samples = visit(size, copies, "ab", f"cal-{size}-{attempt}")
            if min(s["wall_ns"] for s in samples) >= G.TARGET_NS:
                calibrated[size] = copies
                break
            copies = G.next_copies(copies, samples)
        G.B.require(size in calibrated, "calibration failed within five attempts")
    (a.out / "calibration.json").write_text(json.dumps(calibrated, indent=2) + "\n")
    for size in SIZES:
        for pair in range(5):
            for order in ("ab", "ba"):
                visit(size, calibrated[size], order, f"score-{size}-{pair}-{order}", scored=True)
    G.B.require(G.B.digest(a.probe) == identity["binary_sha256"], "probe changed during campaign")
    print("RESULT " + json.dumps({"campaign": "G2-5090", "status": "all-visits-complete", "samples": 400,
                                  "n_per_size_direction_arm": 10, "ab_pairs": 5, "ba_pairs": 5,
                                  "power_envelope": envelope, "qualification": False}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, AssertionError) as error:
        print("REFUSED: " + str(error), file=sys.stderr)
        sys.exit(2)
