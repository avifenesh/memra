#!/usr/bin/env python3
"""CPU stand-in for run-gen in m1-spill-runner tests: prints run-gen's real line shapes.

The arm is inferred from the env the runner sets (MEMRA_SPILL_IO, MEMRA_SPILL_PREAD_DEPTH,
MEMRA_MOE_MMAP_ADVICE). Decode tok/s per arm comes from the JSON map in M1_STUB_TPS; the stub
reads 1 MiB of the artifact so the visit has its own device I/O. M1_STUB_CORRUPT=<arm> changes
one token for that arm, M1_STUB_BAD_OVERREAD=<arm> misreports the direct over-read. Never a
measurement.
"""
import json
import os
import sys
import time


def arm_name(env):
    mode = env.get("MEMRA_SPILL_IO")
    depth = env.get("MEMRA_SPILL_PREAD_DEPTH", "2")
    if mode is None:
        return "mmap-normal" if env.get("MEMRA_MOE_MMAP_ADVICE") == "normal" else "mmap-random"
    return f"{mode}{depth}"


def main():
    env = os.environ
    arm = arm_name(env)
    ngen = int(env["MEMRA_NGEN"])
    tps = json.loads(env.get("M1_STUB_TPS", "{}")).get(arm, 10.0)
    with open(sys.argv[1], "rb") as f:
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        f.read(1 << 20)
    tokens = [1000 + i for i in range(ngen)]
    if env.get("M1_STUB_CORRUPT") == arm:
        tokens[7] += 1
    print("[spill] disk tier ON: free_vram=23116 MiB  free_pinnable_ram=0 MiB (MemAvailable*resolved_frac)")
    print("[spill] experts placed: 0 pinned (Tier 1), 30720 mmap'd from disk (Tier 2, 14514 MiB)")
    depth = None if env.get("MEMRA_SPILL_IO") is None else int(env.get("MEMRA_SPILL_PREAD_DEPTH", "2"))
    if depth is not None:
        print(f"[spill-pread] enabled: depth={depth} buffer_bytes=864256 payload_capacity=860160 "
              "total_pinned_bytes=0 (stub)")
    print("[ttft] prompt_tokens=71 prefill_wall_s=1.250 (verify-class batched prefill; stub)")
    print("verify-prefill argmax=1178  decode argmax=1178  logit maxdiff=5.258e-1  MATCH")
    time.sleep(0.8)
    dt = ngen / tps
    print(f"generated {ngen} tokens in {dt:.3f}s = {tps:.2f} tok/s (ST greedy decode)")
    print(f"tokens: {tokens}")
    print(f"MoE cache DECODE-WINDOW: 8 slots | hits=0 misses={ngen * 960} (hit-rate=0.0%) | "
          f"staged {ngen * 0.454:.2f} GB H2D (433.0 MB/token)")
    if depth is not None:
        reads = ngen * 960
        over = 4096 * reads if env.get("MEMRA_SPILL_IO") == "direct" else 0
        if env.get("M1_STUB_BAD_OVERREAD") == arm:
            over += 1
        print(f"spill worker DECODE-WINDOW: reads={reads} bytes={reads * 473088} waits=3 ring_full=0 fallbacks=0")
        over = 4096 * reads if env.get("MEMRA_SPILL_IO") == "direct" else 0
        if env.get("M1_STUB_BAD_OVERREAD") == arm:
            over += 1
        print(f"spill stages DECODE-WINDOW: worker_read_ms=1.000 demand_read_ms=0.000 wait_ms=0.500 "
              f"h2d_submits={reads} overread_bytes={over}")
        # OWED 26 G1 seam: M1_STUB_FALLBACKS=<arm> reports ring-busy mmap fallbacks for that arm.
        fb = 7 if env.get("M1_STUB_FALLBACKS") == arm else 0
        print(f"[spill-pread] reads={reads} bytes={reads * 473088} errors=0 short_reads=0 fallbacks={fb} "
              f"buffer_waits=3 ring_full=0 overread_bytes={over} worker_read_ns=1 demand_read_ns=0 "
              f"wait_ns=1 h2d_submits={reads}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
