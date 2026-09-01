# SWA ring flag-ON validation — Box1

Status: **COMPLETE**

Recommendation: **ship-as-serving-config** for the validated Step-3.7 PP-2 / 262k shape;
keep `MEMRA_SWA_RING` default OFF.

The ring is exact across a physical wrap, preserves the fixed b1 serving golden under c=1 and
c=4, passes the applicable correctness battery, raises the measured first-defer point from 2 to
12 simultaneous 262k sessions on Box1, and safely declines a lapped affinity checkpoint into a
cold re-prime. This is an opt-in serving recommendation, not a runtime-default flip.

## Verdict

| Cell | N / shape | Verdict |
|---|---:|---|
| Exact-tip clean build and idle-GPU provenance | 1 build, 2x RTX PRO 6000 | **PASS** |
| Teacher-forced OFF vs ON after trunk and persistent-MTP wrap | 1 per arm, 9,216 tokens | **PASS — byte-identical sampled rows and full compared logit row** |
| Fixed-binary b1 one-hash probe, ring ON | c=1 | **PASS — 1/1 golden** |
| Ring-ON serving burst | c=4 | **PASS — 4/4 golden, zero errors/divergences** |
| `kernel-check`, PP decode, `run-gen`, `run-spec` K=1..8 | 1 run per gate | **PASS** |
| Step35 chunk/tick invariance plus canary teeth | 1 run per arm | **PASS** |
| 262k first-defer capacity | N=1 per OFF/ON arm, c=24 offered | **PASS — 2 -> 12 active sessions (6.0x)** |
| Lapped affinity decline and cold re-prime | N=1 | **PASS** |
| Unmodified stock `serve-smoke` | 1 run | **SCOPED RED — prefix-cache-only structural mismatch; all applicable cells pass** |

## Runtime provenance

- Source: clean `019428e217e297cb5981d201a4a520aee69222a6`, the merged b1fix base.
- Build host: Box1, two RTX PRO 6000 Blackwell Server Edition GPUs, PP stages `0,1`.
- Toolchain: `rustc 1.97.1`, `cargo 1.97.1`, CUDA 13.2 (`nvcc V13.2.51`), auto-detected
  `sm_120a`.
- Both GPUs reported 0 MiB used, P8, and no compute applications before and after the clean build.
- Clean `memra-server` SHA-256:
  `7f04f76715d637c46a379366a833d518aed9d465a5dcfd1ffee53be79d9b9cef`.
- The full clean-binary identity list is in
  [`raw/clean-build-20260810T074946Z/binary-identity.txt`](raw/clean-build-20260810T074946Z/binary-identity.txt).

The initial default-target build did rebuild the engine validation executables but left
`memra-server` stale at its pre-ring b1fix-parent hash. The first serving preflight detected the
missing `capped at 4639 rows` marker and sent **no requests**. That receipt is retained; all accepted
server, capacity, lap, and core-gate results use the later clean target. The wrap and full-logit
engine probes use the rebuilt exact-tip engine binaries whose identities are recorded in their
drivers. No result below uses the stale server.

## Wrap-crossing teacher-forced exactness

Both arms used the same first 9,216 token ids, the pinned IQ4_XS trunk, the external Q8_0 MTP
draft, `K=4`, stride 256, and 4,096-token chunks. The ring holds 4,639 physical rows. The first
chunk ends at 4,096; the second appends through 8,192 and therefore crosses the physical tail.
`replay_acceptance` allocates one trunk cache and one `MtpScratch` before the chunk loop, then
advances both at the same absolute chunk offset, so this cell crosses both the trunk SWA layers
and the persistent MTP scratch.

| Cross-arm comparison | OFF | ON | Verdict |
|---|---:|---:|---|
| Teacher-forced targets | 9,215 | 9,215 | identical |
| NLL/token | 0.78327 | 0.78327 | identical |
| Sampled replay rows | 35 | 35 | byte-identical SHA-256 `20e1a2a7...e7de` |
| Step-15 full logit row | 128,896 f32 / 515,584 B | 128,896 f32 / 515,584 B | byte-identical SHA-256 `0fdc84a9...c912` |
| Forced output tokens and summary | — | — | identical |

The full-logit harness's `disagreements=12/16` diagnostic is each arm versus the supplied force
tape; it is identical in both arms. The requested cross-arm comparison is byte-identical.

Two stopped diagnostics are preserved rather than hidden. The first replay attempt stopped before
measurement with `ERROR: model has no MTP/NextN head` because it supplied the wrong draft variable.
After the corrected replay succeeded, a separate diagnostic accidentally tokenized the entire
nominal prompt to 23,770 tokens and reported
`DriverError(CUDA_ERROR_INVALID_VALUE, "invalid argument")`; no cause is inferred and that run is
not used for a verdict.

## Serving exactness and correctness gates

The corrected clean server logged the ring shape on both fresh boots:

```text
[admission] "step": plain 83520 B/token (61248 capped at 4639 rows), spec 85376 B/token (63104 capped)
```

- b1fix one-hash, ring ON: c=1 returned the pinned 326-byte golden hash
  `21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de`, 1/1 match,
  zero errors, zero divergences.
- Barrier burst: c=4 returned that same hash 4/4, with zero errors and zero divergences.
- `kernel-check`: all GPU kernels matched CPU references.
- PP-2 decode gate: B=1,2,4,8, two split repetitions plus unsplit references, all
  bit-identical.
- `run-gen`: prefill/decode argmax `128799` MATCH; batched-prime/tokenwise argmax `128799`
  MATCH.
- `run-spec` with the external draft: self-consistency PASS for every K=1 through K=8.
- Step35 segmentation: naked chunks 4096/513/512/256/64 and tick budgets
  0/1024/513/512/256/64 with splits 64/256/512 were bit-identical. Both legacy-arithmetic
  canaries diverged and were caught, proving the gates have teeth.

### Stock serve smoke

The unmodified stock `serve-smoke` exits 1 and must not be called green. It has exactly one failed
top-level cell: flat prefix-cache accounting. Its 14 sub-failures all observe the ring contract's
deliberate zero-hit, zero-insert, zero-LCP behavior, accompanied by the explicit server line:

```text
[prefix-cache] refused for MEMRA_SWA_RING=1 Step35 session (flat-history snapshots/restores are excluded)
```

All applicable cells passed: plain APIs, SSE, completions, deterministic greedy output, three-way
concurrency, long generation, spec/plain greedy identity, every sampling-truncation arm, both
affinity modes, and no failed rewinds. The Gemma arm was skipped because its unrelated artifact is
absent. After `SIGTERM`, `0 in flight`, and `drain complete`, the server recorded the same
shutdown-only `CUDA_ERROR_DEINITIALIZED` pending-flush line present in the accepted flag-OFF b1fix
receipt; no request failed. Before a default flip, the stock smoke's flat-cache cell should be
replaced or explicitly skipped for ring sessions.

## Capacity at 262,144 context

Method: one fresh server per arm, N=1 per arm, identical model/PP-2 shape, plain K=0 requests with
`max_ctx=262144`, c=24 offered behind a start barrier, and continuous one-second `nvidia-smi`
sampling under one exclusive GPU lock. Capacity is the active-session count on the first captured
admission defer, matching the capbase methodology. All 24 requests in each arm were allowed to drain.

| Receipt | Ring OFF | Ring ON |
|---|---:|---:|
| Active sessions at first defer | **2** | **12** |
| Measured session-capacity ratio | 1.0x | **6.0x** |
| Admission-modeled session cost | 21,894 MB | 6,123 MB |
| Modeled cost ratio | 1.0x | **3.576x reduction** |
| Effective free at first defer | 20,535 MB | 6,129 MB |
| Admission reserve | 1,611 MB | 1,611 MB |
| Peak sampled GPU0 used | 66,705 MiB | 81,265 MiB |
| Peak sampled GPU1 used | 77,715 MiB | 91,443 MiB |
| Peak sampled combined used | 144,420 MiB | 172,708 MiB |
| Maximum sampled temperature GPU0/GPU1 | 38/40 C | 39/41 C |
| Requests completed | 24/24 | 24/24 |
| Captured failure lines | 0 | 0 |
| Step-OOM parks | 0 | 0 |

The shared residency posture was the same in both arms: stage 0 logged 45.72 GB experts plus
3.92 GB trunk; stage 1 logged 55.35 GB experts plus 3.92 GB trunk. Admission logged a 0 MB fixed
residual for both request-cost rows and the same 1,611 MB reserve. The runtime did not expose a
separate activation-high-water scalar in this run; activation and allocator high-water are present
in the per-GPU NVML peaks above. The 0 MB value is the admission fixed residual, not a claim that
activation memory is zero.

The 3.576x figure is the modeled per-session cost reduction. The observed first-defer count is the
integer Box1 result after fixed model residency and reserve: 2 -> 12, or 6.0x. It is not promoted
to a general 6x scaling claim; it is an N=1 capacity receipt for this exact box and serving shape.
Neither arm OOMed, and no failure cause is inferred from a successful defer-and-drain run.

## Lapped affinity decline

A ring-ON session was primed to 9,216 tokens and parked with a position-1,024 checkpoint, old enough
to be lapped by the 4,639-row ring. A 2,048-token affinity rewrite produced the required line:

```text
[worker] plain-affinity: declined (SWA ring lapped checkpoint 1024; 1 parked, 2048 prompt tokens; model step)
```

The request then computed all 2,048 tokens cold with zero cached tokens and zero affinity rewind.
Its 17-byte output was byte-identical to a fresh-server cold reference, SHA-256
`67f6e242e386ca51323360c58a1bfc4941d3d1781dc32051375765c1cfadfc02`. No crash occurred.

## Default eligibility

**Ship as a serving configuration:** operators may opt into `MEMRA_SWA_RING=1` for the validated
Step-3.7 IQ4_XS + Q8_0 MTP, PP-2, 262k-context Box1 shape. Keep the default OFF.

This lane does not authorize a runtime-default flip. Project policy still requires correctness,
memory, and throughput gates on the local RTX 5090 proof rig before changing the default. The stock
serve-smoke cache-metering cell also needs a ring-aware contract. No merge, tag, release, or origin
push was performed.

## Raw evidence

- [Clean build and binary identity](raw/clean-build-20260810T074946Z/)
- [Wrap replay OFF/ON](raw/wrap-20260810T064018Z/)
- [Full-logit OFF/ON comparison](raw/logits-20260810T065008Z/)
- [Serving golden and c=4 burst](raw/serve-exactness-20260810T075504Z/)
- [262k capacity OFF/ON](raw/capacity-20260810T075756Z/)
- [Lapped affinity decline](raw/lap-20260810T080122Z/)
- [Core correctness gates](raw/core-gates-20260810T080351Z/)
- [Chunk/tick invariance and canaries](raw/invariance-20260810T081034Z/)
- [Unmodified stock serve smoke](raw/serve-smoke-20260810T082015Z/)
- [Superseded initial build receipt](raw/build-20260810T063055Z/)
- [Stopped wrong-draft replay attempt](raw/wrap-20260810T063818Z/)
- [Caught stale-server serving preflight](raw/serve-exactness-20260810T074152Z/)
- [SHA-256 manifest for all 163 raw files](raw/SHA256SUMS)
