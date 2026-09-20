# WP-B day 10 — target-class active KV

Repository: **avifenesh/memra**, branch `lane/spill-b-20260919`.
All seven requested native cells completed and their raw receipts were pushed.
**Not merged, released, deployed, serving-qualified, or a default promotion.**

## Source and binary identity

| Surface | Native source commit | Binary SHA-256 |
| --- | --- | --- |
| Frozen baselines + original controls | `8ef562b12f16fd3378f866939be347df3a6d3d96` | `ad3f7aad046c46f5c4b996325557bd8cefdd7fd856984fabc5fedabc6908a0f1` |
| Isolated mapped-VA diagnostic | `61eb02acb75b3e8f2fc54c11e4ad32f43ce78457` | `9144cca9d2134811f5cdad7a1293c82664802c1a58cf75a3fc2c6244e60336fd` |
| Direct native construction | `c619b008ed9f359c51bca7e0d8d1e8a74edbc17a` | `e171aef5f71181c588b869c37bb0e85c50c2a237d974295436e9fa3ef24a7002` |

The initial merge includes integ5 `db16dc538f6920c9492c48c429c35e59b3d607e7`,
including the observed fixed-VA and `not-applicable-pooled` review fixes.
The diagnostic deliberately retains the original empty-plane allocation/swap
regime; direct allocation is enabled only in the subsequent measured binary.
This separates residual diagnosis from allocator-construction changes.

## Frozen target-card baselines

Hardware: **one RTX PRO 6000 Blackwell, 96 GB**, **600/600 W**.
Every GPU cell is an **N=1 development correctness probe**, not a timing campaign,
under `tools/tier-battery.py --rig pro-single`, `/tmp/memra-gpu.lock`, with
250 ms telemetry. Initial lock contention was retried boundedly; every refused
attempt is retained. No cross-card timing or serving/performance claim.

Checkpoint: Qwen3.8-27B native NVFP4/Q5K GGUF; native q8_0/q5_1 KV unchanged.
Program: tokenwise `decode_step_h`, trunk-only, no MTP or alternate prefill.
The two contexts use context minus 128 prompt tokens and 128 continuation tokens.

| Baseline completion file | SHA-256 |
| --- | --- |
| `pro-single-day10/baseline-8192/receipt/BASELINE.txt` | `224a9152f73b174698e24d234d97b47f253918c1657a8a1ff78554f67369d4d6` |
| `pro-single-day10/baseline-32768/receipt/BASELINE.txt` | `f11a8938316fe8063b2e12b9bf558581f0627086a70b52525b8e4319447af559` |

`BOX3-BASELINES.json` freezes all seven decoded continuation-surface hashes plus
artifact/binary/plan/prompt/source identity. Its SHA-256 is
`72db65b6a52784b14a54e3c76f8bfdd1aa58f02b8e7a159e2e27696cbc831990`.
These are **new target-card bundles, never interchangeable with RTX 5090 bundles**.

## G1 verdicts (verbatim)

```text
ACTIVE-8K G1 PASS
ACTIVE-32K physical reclaim/restore bit-identical, residual 2097152 B, class unclassified — not G1 PASS
ACTIVE-8K pooled control: not-applicable-pooled
```

Both the original and directly constructed VMM 8k cells receive the first verdict.
The original and isolated diagnostic 32k cells receive the second.
Every active cell matches all seven frozen continuation surfaces and the restored
prefix exactly: prompt, serialized plan, prefix/final state, 128 generated ids,
129 decision/final logit rows and final logits. All 32 active planes round-trip;
no source remains charged after demotion, and pinned charges equal logical bytes.

| Driver observation (bytes) | Original 8k VMM | Original 32k VMM | Direct-construction 8k VMM |
| --- | ---: | ---: | ---: |
| Queried granule | 2,097,152 | 2,097,152 | 2,097,152 |
| Rounded physical capacity | 301,989,888 | 1,040,187,392 | 301,989,888 |
| Released whole chunks | 201,326,592 | 905,969,664 | 201,326,592 |
| Retained edge/capacity chunks | 100,663,296 | 134,217,728 | 100,663,296 |
| Logical D2H / pinned bytes | 239,468,544 | 969,277,440 | 239,468,544 |
| Free before demote | 85,863,301,120 | 84,743,421,952 | 86,098,182,144 |
| Free after demote | 86,064,627,712 | 85,647,294,464 | 86,299,508,736 |
| Free after restore | 85,863,301,120 | 84,743,421,952 | 86,098,182,144 |
| Observed reclaim / reacquisition | 201,326,592 / 201,326,592 | 903,872,512 / 903,872,512 | 201,326,592 / 201,326,592 |
| Residual | 0 | **2,097,152** | 0 |

The pooled control has zero observed reclaim/reacquisition and zero trim release;
its G1 and fixed-VA fields are both **`not-applicable-pooled`**. Its unchanged
free-VRAM observation is 86,184,165,376 B. No process-footprint optimization or
performance attribution is inferred from different allocation-regime snapshots.

## Residual diagnostic: VA free did not return the granule

The probe acts on **actual demoted planes**: retained edge/capacity chunks are
unmapped **without releasing their physical handles**, then their VA reservation
is freed, re-reserved at the original address and the same retained handles remapped.
It reads `cuMemGetInfo` before unmap, after unmap, after VA free and after remap,
per plane. This isolates mapping/reservation metadata from physical-capacity release.

All 32 per-plane rows in `mapped-va-probe.tsv` show:

```text
mapped_va_release_delta_bytes=0
mapped_unmap_delta_bytes=0
mapped_va_roundtrip_equal=true
residual_bytes=2097152
residual_class=unclassified
```

Free VRAM remains **85,647,294,464 B** through spare-VA release, context sync,
actual mapped-range unmap/VA free/remap. Full restoration returns exactly to
**84,743,421,952 B**. The one-granule residual reproduces on this target card, just
as on the earlier RTX 5090; it is **not classified by this probe**.

A `va-reservation-page-table` class would require the residual to return only on
VA free (not unmap), with exact remap accounting. That did **not** happen. The
prior RTX 5090 probe freed only never-mapped spare VA, not the mapped-range probe.
Neither card supplies the required nonzero-residual classification; 32k stays
**not G1 PASS**. No relaxed equality, inferred page-table explanation, or further
unrequested diagnostic is used to promote it.

## Direct allocation refinement

`Cache::new_with_allocator(..., KvAllocator::Vmm)` selects native K/V allocation
at construction through `KvDev::alloc_kv_plane` / `Engine::alloc_vmm_u8`.
The final native 8k receipt observes **34 VMM / 0 pooled K/V planes at position 0**,
`construction=direct`, `empty_plane_swap=false`. Of those, 32 nonempty active planes
are demoted/restored. There is no pooled empty-plane bootstrap or swap in the final
source. The native build, clippy and 8k frozen-baseline/G1 rerun pass.

A backend without VMM support refuses explicitly; it never substitutes pooled
storage. All existing constructors, including planned and PP constructors, retain
pooled allocation. Recurrent/latent state allocation, formats, kernels and the
numerical program are unchanged. The gate-only `--kv-allocator vmm` door remains
default-OFF; **decide-by: 2026-10-04**. No new environment flag or serving registration.
New unsafe calls stay inside the existing `KvPlane` owner. A diagnostic failure
leaves the operand suspended and tracks reservation ownership for safe cleanup;
no token executes against it. No owning VMM `CudaSlice` escapes.

## Checks actually run

| Check | Result |
| --- | --- |
| Mac `cargo fmt --all -- --check` | PASS |
| Mac and Linux-target scoped all-target check, offline | PASS; cross-check is compilation only |
| Mac `cargo test -p memra-kv -p memra-tier --offline --no-fail-fast` | **262 passed** |
| Scoped all-target clippy, `-D warnings` | PASS |
| Four Python verdict tests, including 12 arithmetic/engagement mutation arms | PASS |
| `git diff --check`; `bash tools/check-flags.sh` | PASS |
| All three native release gate builds | PASS |
| Native engine/gate release clippy, `-D warnings`, diagnostic and final source | PASS |
| Native kv/tier release tests, diagnostic source (same final KV code) | **262 passed** |
| Seven completed collector cells + final strict offline replay | PASS archive/identity/exactness checks; **32k remains non-PASS for G1** |
| Additional Mac engine check | BLOCKED, exit 101: `spawn nvcc: ... No such file or directory` |
| Full GPU exactness/serving/PRO-pair battery; direct-allocation 32k | NOT RUN |

The Mac engine failure is retained in `day10-checks/engine-mac-check.log`;
native CUDA builds provide the target compilation evidence, not a fabricated Mac pass.
CPU commands/logs are under `day10-checks/`; native raw receipts under
`pro-single-day10/`. `day10-raw-manifest.json` seals membership and bytes.
Final logits are losslessly gzip archived; replay hashes decoded bytes.
`verify-day10.py --require-complete` checks journals, build/source/binary identity,
command/allocator engagement, power/telemetry, continuation, chunk arithmetic,
strict deltas and diagnostic classification. It **does not run CUDA**.

## Publication, boundaries and time

Baseline, control, diagnostic and injection receipts were committed and pushed after
each cell with hooks enabled. All raw failures here are pre-execution lock refusals;
no native GPU cell failed. No main/other-lane checkout or artifact was modified.
Only B's isolated checkout and receipts were used. The temporary remote runner is
removed after the final cell; the B lane stays open for lead integration and pending
qualification rather than being represented as a completed serving implementation.

This session used approximately **1.3 agent-hours**, including build/collector/lock
waits, below the approximately 3.5-hour session cap. The lane budget is **10 agent-days**;
prior sessions' cumulative usage is not reconstructed here. Remaining blockers are
32k residual classification and unrun serving/prefix/graph/spec/PP/PRO-pair surfaces,
not access to an owner-controlled rig.
