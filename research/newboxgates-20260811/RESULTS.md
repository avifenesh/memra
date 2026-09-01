# New-box full gates and serving receipts — 2026-08-11

Status: **COMPLETE — PASS**

The exact v0.75 Step-3.7 PP-2 serving shape passed the requested one-hash matrix,
correctness battery, N=5 serving measurements, and 262k capacity comparison on the new
2x RTX PRO 6000 Workstation Edition box. Production serving and soak were restored after
every disruptive block and independently verified at handoff.

## Verdict

| Cell | N / shape | Verdict |
|---|---:|---|
| Pinned source, binary, golden, build, and rig identity | 1 | **PASS** |
| b1fix c=1 fresh boots | 10 boots, 10 requests | **PASS — 10/10 golden** |
| b1fix c=8 barrier | 5 boots, 40 requests | **PASS — 40/40 golden** |
| `kernel-check` and PP-2 decode B=1/2/4/8 | 1 battery | **PASS — all green / bit-identical** |
| `run-gen` step35 argmax | 1 | **PASS — both comparisons MATCH** |
| `run-spec` K=1..8 | 8 K values | **PASS — 8/8 self-consistent** |
| Step35 chunk/tick invariance and canary teeth | 4 gate runs | **PASS** |
| Serving latency and decode | N=5 per point | **PASS — 75/75 requests** |
| 262k capacity, ring OFF vs ON | N=1 per arm, c=24 offered | **PASS — 2 -> 12 sessions** |
| Final production server, stream, and soak | independent receipt | **PASS** |

## Pinned runtime and rig

- Remote source: clean detached `5911de40f48d1d2fe36a92fef7b9b41cebc792f2`, the v0.75
  source supplied for this lane.
- Server SHA-256:
  `1b2159b50c9bb5cf2703e9f159ac44b6f40d339db6fa078a8a0212ce1d54bf7b`.
- Golden response: 326 bytes, SHA-256
  `21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de`.
- Model entrypoint: `Step-3.7-flash-IQ4_XS-00001-of-00003.gguf`, 46,483,327,296 bytes;
  external MTP draft: `Step3.7-flash-mtp-Q8_0.gguf`, 3,707,276,416 bytes.
- Rig: 2x NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB each,
  driver 610.57.04, 600 W limits. The serving logs selected PP-2 devices `0,1` and native
  `cudaMemcpyPeerAsync`; the owner's prior byte probe established the CLEAN/no-bounce precondition.
- Toolchain: Rust/Cargo 1.97.1 and CUDA 13.1 (`nvcc` 13.1.115), auto `sm_120a`.
- Production/performance serving configuration: `MEMRA_CTX=262144`, `MEMRA_MOE_GROUPED=1`,
  `MEMRA_PREFILL_TICK=2048`, 2 GiB prefix cache, speculative serving off.

The gate build completed against the pinned source and left the deployed server hash unchanged.

## One-hash serving matrix

The existing `research/p0iso-20260810/run-box1.sh` harness supplied the golden protocol.
Each condition checked the source/binary/golden identities before traffic. Every cell used a
fresh server boot, retained request JSONL plus server/GPU logs, and stopped cleanly before the
next cell.

| Harness condition | Boots | Requests | Golden matches | Distinct completion hashes |
|---|---:|---:|---:|---:|
| `h2-c1` | 10 | 10 | 10 | 1 |
| `same` (c=8 barrier) | 5 | 40 | 40 | 1 |
| **Total** | **15** | **50** | **50** | **1** |

All 50 responses equal the pinned golden hash above. There were zero request errors and zero
golden divergences.

## Correctness battery

- `kernel-check`: `ALL GREEN`; GPU kernels matched CPU references.
- PP-2 decode: B=1,2,4,8, two split repetitions plus each unsplit reference, all logits
  bit-identical; `0 failing arm(s)`.
- `run-gen`: prefill/decode argmax `128799 == 128799` and batched-prime/tokenwise argmax
  `128799 == 128799`, both MATCH.
- `run-spec`: every K=1 through K=8 reported self-consistency PASS against the plain target.
- Step35 chunk gate: chunks 4096/513/512/256/64 were exact. Its legacy-arithmetic canary
  diverged and was caught.
- Step35 tick gate: budgets 0/1024/513/512/256/64 and split points 64/256/512 were exact.
  Its legacy-arithmetic canary diverged and was caught.

All eight gate commands exited zero; the two canary commands exit zero only after observing the
required divergence. The reducer independently marked all nine semantic checks true.

## N=5 serving medians

Method: five fresh server boots, with point order alternating forward/reverse by repetition under
one exclusive GPU lock. Each boot received one excluded warmup. Scored requests used streaming,
temperature 0, unique cache salts, no cached prompt tokens, and speculative serving disabled.
The 4k prompt was exactly 4,107 tokens. Decode cells generated exactly 512 tokens per stream and
required `finish_reason=length`.

| Shape | Primary N=5 median (full range) | Secondary median |
|---|---:|---:|
| Short TTFT | **0.13625 s** (0.13547-0.13693) | wall 0.46941 s |
| 4k cold TTFT | **5.77673 s** (5.71335-5.87501) | wall 5.85869 s |
| Decode c=1 | **98.30 total-window tok/s** (97.81-98.63) | 102.16 decode-window tok/s |
| Decode c=4 | **158.45 total-window tok/s** (158.36-158.84) | 168.35 decode-window tok/s |
| Decode c=8 | **173.62 total-window tok/s** (173.55-173.78) | 185.72 decode-window tok/s |

The primary decode rate includes prompt/TTFT through final drain; the secondary rate starts at the
first visible token. Across 25 scored point repetitions there were 75/75 successful requests,
zero cache hits, and zero wrong-length decode responses.

These medians supersede the provisional single-run values in
[`research/newbox-bench-20260811/RESULTS.md`](../newbox-bench-20260811/RESULTS.md): c=1
99.0 -> 98.30, c=4 161.3 -> 158.45, c=8 177.0 -> 173.62 tok/s, and 4k TTFT
5.227 -> 5.777 s. The new short-TTFT median remains inside the provisional N=3 range.

Thermal regime: continuous 500 ms NVML sampling. Across the five scored boots, maximum sampled
temperatures were 70/72 C, maximum powers 511.58/558.30 W, and maximum memory use
46,898/57,236 MiB. GPU memory returned to 2 MiB per device before production restoration.

## 262k capacity receipt

The unchanged `research/ringval-20260810/run-box1-capacity.sh` capbase workload ran one fresh
server per arm under one exclusive lock: c=24 simultaneous plain requests, requested and server
context 262,144, 64 generated tokens, temperature 0, continuous one-second NVML sampling, and all
requests allowed to drain. Both capacity arms used `MEMRA_PREFIX_CACHE_MB=0`,
`MEMRA_MAX_SESSIONS=64`, and `MEMRA_REUSE_POOL=2`. This is an N=1 capacity receipt for this exact
rig and shape.

| Receipt | Ring OFF | `MEMRA_SWA_RING=1` |
|---|---:|---:|
| Active sessions at first defer | **2** | **12** |
| Measured first-defer ratio | 1.0x | **6.0x** |
| Admission-modeled session cost | 21,894 MB | 6,123 MB |
| Effective free at first defer | 20,572 MB | 6,166 MB |
| Admission reserve | 1,611 MB | 1,611 MB |
| Peak sampled GPU0 / GPU1 used | 66,706 / 77,716 MiB | 81,266 / 91,444 MiB |
| Peak sampled combined used | 144,422 MiB | 172,710 MiB |
| Maximum sampled temperature GPU0/GPU1 | 56/56 C | 57/57 C |
| Requests completed | 24/24 | 24/24 |
| Captured failure lines / step-OOM parks | 0 / 0 | 0 / 0 |

The 2 -> 12 result reproduces the integer first-defer capacity row on this Workstation Edition
pair. It is not generalized beyond this N=1 box/configuration receipt.

## Serving restored at handoff

Every disruptive block has a `service-restored/restored.ok` receipt created only after
`/readyz`, `/v1/models`, a non-empty streamed completion ending in `[DONE]`, and a live soak PID.
An independent final receipt at `2026-08-11T00:09:31Z` then recorded:

- `/readyz`: `ready`, model present, `xid_warnings=0`.
- `/v1/models`: `stepfun/step-3.7-flash`, context length 262,144.
- Stream: `[DONE]`, non-empty 147-byte response, 32 completion tokens, fingerprint
  `memra-5911de40f48d`.
- Soak: 20 fresh captured rows, latest request 97 chunks with an empty error field, no recent errors.
- Processes: production `memra-server` PID 13667 and `/root/soak.py` PID 13803; both GPUs owned
  by that server at the final snapshot.

No request failure was captured. The final scan found no `CUDA_ERROR`, OOM, panic, worker-died,
server-fatal, or kernel Xid event; the logs' static list of watched fatal Xid codes is configuration,
not an incident.

## Raw evidence

- [Build, toolchain, binary identity, and live-service build receipts](raw/build-20260810T232828Z/)
- [Fifteen-boot one-hash matrix and reducer summary](raw/matrix-20260810T233141Z/)
- [Kernel, decode, run-gen, run-spec, and invariance logs](raw/correctness-20260810T234155Z/)
- [N=5 request JSONL, five server logs, NVML traces, and summary](raw/perf-20260810T235804Z/)
- [262k OFF/ON requests, server logs, NVML traces, and comparison](raw/capacity-20260811T000621Z/)
- [Independent final service, stream, process, GPU, and soak receipt](raw/final-service/)
- [SHA-256 manifest for all raw evidence files](raw/SHA256SUMS)

No runtime code, runtime default, model artifact, generated performance board, merge, tag, release,
or origin branch was changed. The only implementation added by this lane is the bounded research
harness; every disruptive mode restores production serving and soak from its exit trap. Nothing
was pushed.
