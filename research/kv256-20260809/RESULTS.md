# kv256 Option 2 — Step35 SWA bounded ring

Status: implemented behind `MEMRA_SWA_RING=1`, default **OFF**. Local unit/workspace and RTX
5090 flag-OFF gates are green. Flag-ON receipts are complete in
`research/ringval-20260810/RESULTS.md` and `research/newboxgates-20260811/RESULTS.md`; the
default remains OFF. Remaining doors are the `serve-smoke` prefix-cache cell (**SCOPED RED**
because ring sessions deliberately refuse flat-history prefix-cache operations) and the
default-flip policy. Capacity claims point to the measured **2 -> 12 / 6.0x** rows in those
receipts.

## Delivered

| commit | slice |
|---|---|
| `04eb306d` | flag parse, 512 + 4096 + 31 row geometry, Step35 SWA-only session allocation |
| `5ac690e2` | bounded append/rebase and physical read views, MTP scratch, lap declines, prefix-cache exclusion, exact admission accounting, tests |

The physical row cap is `min(max_ctx, 4639)`: the 512-row attention window, one legal maximum
4096-token prime chunk, and 31 rows of aligned-down-32 headroom. Only Step35 layers whose resolved
geometry has a window use it. Step35 global/full-history attention layers, Gemma4, Eagle scratch,
and every other architecture retain their prior full allocation.

`KvLayer.len` remains the absolute sequence length. A ring layer maps absolute rows relative to its
resident base; when an append would cross the physical tail, it copies only the audited aligned
live prefix to row zero and appends after it. This keeps each CUDA reader's range contiguous and
preserves the existing aligned prime-view contract. The changed readers are Step35 prime, eager
decode, batched decode, spec verify through eager replay, and the Step35 MTP scratch. No DC or graph
path was opened or changed.

Rollbacks preflight the aligned window needed at the target length. A lapped plain-affinity or spec
checkpoint declines through the existing cold/full-re-prime path; overwritten rows are never
treated as live. Prefix-cache snapshot, restore, and fanout refuse ring sessions and emit a
`[prefix-cache] refused ...` line because the v1 entry format is position-addressed flat history.

## Capacity arithmetic delivered by allocator and admission

At `max_ctx = 262,144`, the exact context-row component is:

`(total B/token - capped B/token) * 262,144 + capped B/token * 4,639`.

| session shape | full-slab B/token | ring-capped B/token | before | after | component reduction |
|---|---:|---:|---:|---:|---:|
| plain trunk | 83,520 | 61,248 | 20.391 GiB | 5.702 GiB | **3.576x** / 72.04% |
| spec trunk + MTP scratch | 85,376 | 63,104 | 20.844 GiB | 5.710 GiB | **3.650x** / 72.61% |

This realizes the earlier approximately 3.5x session-KV arithmetic in both the CUDA allocation and
the request admission estimate. Admission caps only the SWA byte class and continues to learn the
fixed residual. Per-plane 8-byte tail pads, length counters, recurrent state, model weights,
activation high-water, and allocator effects are outside the table; therefore **3.576x is a KV
component ratio, not a claim that box concurrency is already 3.576x**. The honest concurrent
full-262k session count is still a box1 measurement.

## Gates

All GPU checks below are one correctness run, not performance medians. The 5090 lock window began
idle at 31 MiB, 0% utilization, 52 C, and 9.57 W; it ended at 31 MiB, 0% utilization, 64 C, and
30.20 W. `MEMRA_SWA_RING` was explicitly unset throughout. Step-3.7 does not fit this 24 GB rig, so
the local battery proves dormant-path/shared-path identity on the installed Qwen3.6-35B artifact;
it does not substitute for flag-ON Step35 proof.

| gate | result | raw |
|---|---|---|
| focused ring geometry/wrap/identity/lap tests | 4/4 pass | included in workspace log |
| admission cap test | pass; server total is now 160/160 | included in workspace log |
| `cargo test --workspace` | PASS; engine 51 pass + 1 CUDA-only ignored, server 160 pass, all other suites/doc tests green | `raw/cargo-test-workspace-20260810.log` |
| release gate build | PASS | `raw/build-release-gates-20260810.log` |
| full `kernel-check`, RTX 5090, flag OFF | `ALL GREEN: kernels match CPU reference.` | `raw/kernel-check-flagoff-local5090-20260810.log` |
| `run-gen`, Qwen3.6-35B IQ4_XS, flag OFF | prefill/decode `MATCH`; batched-prime/tokenwise `MATCH` | `raw/run-gen-q35-flagoff-local5090-20260810.log` |
| `run-spec`, Qwen3.6-35B + own-trim draft, flag OFF | K=1..8 self-consistency PASS, 8/8 | `raw/run-spec-q35-k1-8-flagoff-local5090-20260810.log` |

The combined lock-window transcript is `raw/local5090-flagoff-battery-20260810.log`.

## Structural exclusions

- Step35 only; Gemma4 row-zero-addressed window/DC/graph kernels remain full-slab.
- Full-attention layers remain full-history and preserve their prior format and addressing.
- Step35 DC/graph decode remains refused by construction; this lane did not touch those paths.
- Prefix cache insert/lookup/fanout is disabled for ring sessions until a ring-aware entry format
  exists.
- A checkpoint whose required aligned window is older than the resident base declines and
  re-primes; no bytes are reconstructed.
- `MEMRA_PRIME_CHUNK` is capped at 4096 only while the ring door is on. Door-off explicit values,
  including `0 = monolithic`, retain the old schedule.

## Box1 follow-up — required before merge/tag/default discussion

1. Build this exact commit on box1 and record the commit, compiler/CUDA, artifact manifest, and
   clean PP-2 GPU state.
2. With the pinned Step-3.7 artifact, run flag OFF and flag ON on a prompt longer than 8192 tokens
   so both trunk SWA layers and the persistent MTP scratch cross a physical wrap. Retain raw logs
   and compare the same teacher-forced logits/tokens, not only each arm's internal argmax line.
3. Under flag ON, run full `kernel-check`, `run-gen` argmax/batched-prime agreement, and
   `run-spec` K=1..8, plus the existing Step35 PP-2 prime/batched/serve-shape gates.
4. Exercise and receipt the two deliberate declines: a lapped affinity rewind and prefix-cache
   insert/lookup/fanout refusal. Confirm the request takes the existing cold re-prime path.
5. At `MEMRA_CTX=262144`, capture admission's capped-byte log, per-stage CUDA allocations, and
   before/after maximum simultaneously resident sessions on the 96 GB pair. Report weights,
   activation residual, reserve, and any OOM line verbatim with concurrent GPU process state.
6. Leave the flag default OFF until all flag-ON exactness and capacity receipts above are committed.
