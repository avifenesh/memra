# Step + 27B + drafter co-residency VRAM pre-check

Status: **NO-GO for the scored A/B today — fit is not established.** This is not a measured
non-fit. The exact Qwen3.8-27B artifact and validated draft do not exist in the frozen campaign
manifest, so the load-bearing memory terms are unknown. The protocol requires execution to stop
until the target revision, quantization, optional draft, context cap, and hashes are frozen
([`PROTOCOL.md:20-24`](../beside-plan-20260811/PROTOCOL.md#L20-L24)). The official
[Qwen model catalog](https://huggingface.co/Qwen/models) was also checked while writing this note;
no exact target receipt was substituted from search results.

There are two narrower arithmetic verdicts:

- **All of Step plus Q27 on one physical card: NO-GO; Q27 session ceiling = 0.** The Step artifact
  alone is 105.0 GB and is explicitly “PP-2-or-nothing” on a 96 GB PRO 6000
  ([`docs/PERFORMANCE.md:524-527`](../../docs/PERFORMANCE.md#L524-L527)). This says nothing about
  which serving topology should be chosen; it only rules out that byte layout.
- **Protocol split, using the historical Qwen3.6 proxy only: the requested `c=1/2/4/8` ladder is
  arithmetically inside the accepted Step starting snapshot.** The deliberately conservative Q27
  terms below leave 22,096.52 MiB (21.58 GiB) after the `c=8` charge and transient floor, and cross
  zero between `c=24` and `c=25`. Those are planning bounds, not Qwen3.8 receipts and not
  safe-session claims. The A/B protocol itself asks for the `c=1/2/4/8` ladder
  ([`PROTOCOL.md:242-248`](../beside-plan-20260811/PROTOCOL.md#L242-L248)).

## Scope and units

The target is the rented two-card RTX PRO 6000 Blackwell pair, with 96 GB per card; the tracked
inventory reports 97,887 MiB per card
([`docs/PERFORMANCE.md:63-65`](../../docs/PERFORMANCE.md#L63-L65),
[`PROTOCOL.md:37-40`](../beside-plan-20260811/PROTOCOL.md#L37-L40)). `MiB` and `GiB` below are binary
units: `1 MiB = 2^20 bytes` and `1 GiB = 2^30 bytes`. Source `GB`/`MB` labels are preserved when a
log used decimal units; conversions are shown rather than silently relabeling them.

This is a memory-fit calculation only. It makes no business, pricing, interference, placement, or
serving-topology decision. No GPU workload ran and no model file was opened.

## Exact target ledger

The accepted Step process remains PP-2 across physical cards 0 and 1, and Arm B adds an independent
Q27 process on physical card 0 at served context 32,768
([`PROTOCOL.md:28-35`](../beside-plan-20260811/PROTOCOL.md#L28-L35),
[`PROTOCOL.md:44-55`](../beside-plan-20260811/PROTOCOL.md#L44-L55)). The Step baseline already
includes its pinned external draft and serving configuration, so its measured post-load free memory
is the correct starting point; adding Step's files again would double-count them.

The task's “27B NVFP4” bytes are recoverable only for the historical Qwen3.6 proxy. The tracked
Qwen3.8 preparation state selected an official FP8 safetensors source and explicitly excluded a
Qwen3.8 NVFP4 or local-requant bridge; it also recorded that the official target was not yet visible
([`cx-38prep PROGRESS.md:9-19`](../cx-38prep-20260808/PROGRESS.md#L9-L19)). Therefore this note does
not invent Qwen3.8 NVFP4 bytes.

| Budget term | Exact value available for the scored target? | Source and treatment |
|---|---:|---|
| Physical card capacity | 97,887 MiB each | Measured inventory in [`PROTOCOL.md:37-40`](../beside-plan-20260811/PROTOCOL.md#L37-L40); nominal hardware is 96 GB/card in [`docs/PERFORMANCE.md:63-65`](../../docs/PERFORMANCE.md#L63-L65). |
| Step load plan, card 0 | 45.72 GB experts + 3.92 GB trunk = 49.64 GB | Accepted load receipt summarized in [`PROTOCOL.md:37-40`](../beside-plan-20260811/PROTOCOL.md#L37-L40) and logged at [`server-w1...head.log:6-13`](../serve-ready-20260808/raw/server-w1-20260808T160419Z-head.log#L6-L13). |
| Step load plan, card 1 | 55.35 GB experts + 3.92 GB trunk = 59.27 GB | Accepted load receipt summarized in [`PROTOCOL.md:37-40`](../beside-plan-20260811/PROTOCOL.md#L37-L40) and logged at [`server-w1...head.log:8-13`](../serve-ready-20260808/raw/server-w1-20260808T160419Z-head.log#L8-L13). |
| Free after Step resident and serving, card 0 | 50,672 MiB | Target starting budget from [`PROTOCOL.md:37-42`](../beside-plan-20260811/PROTOCOL.md#L37-L42). |
| Free after Step resident and serving, card 1 | 40,332 MiB | Target control-card receipt from [`PROTOCOL.md:37-42`](../beside-plan-20260811/PROTOCOL.md#L37-L42). Q27 is not placed here in the protocol. |
| Qwen3.8 target resident weights | **UNKNOWN — needs-measurement** | Exact artifact is mandatory and the current 15–16 GiB Qwen3.6 value is explicitly proxy-only in [`PROTOCOL.md:20-24`](../beside-plan-20260811/PROTOCOL.md#L20-L24). The prior preparation lane also records that the intended exact target was not visible and no target directory existed at its check ([`preflight-20260808.log:55-56`](../cx-38prep-20260808/preflight-20260808.log#L55-L56)). |
| Qwen3.8 validated drafter resident weights | **UNKNOWN — needs-measurement** | The launch manifest still has an optional-draft placeholder, and plain decode is required if no exact pairing passes its battery ([`PROTOCOL.md:44-51`](../beside-plan-20260811/PROTOCOL.md#L44-L51), [`PROTOCOL.md:87-95`](../beside-plan-20260811/PROTOCOL.md#L87-L95)). |
| Second process / Engine / static model overhead | **UNKNOWN — needs-measurement** | Arm B starts a second process ([`PROTOCOL.md:44-49`](../beside-plan-20260811/PROTOCOL.md#L44-L49)). Current ownership is one Engine and CUDA context per worker ([`SCOPE.md:79-97`](../coresident-scope-20260811/SCOPE.md#L79-L97)); the target-specific GPU cost is not recorded. |
| Q27 prefix-cache residency | **UNKNOWN — needs-measurement** | The target launch manifest leaves `MEMRA_PREFIX_CACHE_MB` unpinned ([`PROTOCOL.md:87-94`](../beside-plan-20260811/PROTOCOL.md#L87-L94)). |
| Qwen3.8 KV bytes per session at context 32,768 | **UNKNOWN — needs-measurement** | Context is pinned, but bytes/token depends on the exact loaded model and plain/spec path; the runtime derives both shapes from that model ([`worker.rs:622-667`](../../crates/memra-server/src/worker.rs#L622-L667)). |
| Qwen3.8 fixed per-session residual | **UNKNOWN — needs-measurement** | The runtime learns a high-water residual from the observed effective-free delta ([`worker.rs:670-680`](../../crates/memra-server/src/worker.rs#L670-L680), [`worker.rs:3401-3424`](../../crates/memra-server/src/worker.rs#L3401-L3424)). No Qwen3.8 observation exists. |
| Spec-capable burst/transient floor | 1,536 MiB = 1.5 GiB | Runtime constant and rationale in [`worker.rs:847-862`](../../crates/memra-server/src/worker.rs#L847-L862). It is charged on top of request cost for a spec-capable path; a plain path charges the lesser of its request cost and this floor ([`worker.rs:1863-1876`](../../crates/memra-server/src/worker.rs#L1863-L1876)). |
| Qwen3.8 allocator-pool reserved/cached bytes after load and warm-up | **UNKNOWN — needs-measurement** | The protocol explicitly requires post-load, post-warm-up, peak, and final-floor receipts ([`PROTOCOL.md:65-68`](../beside-plan-20260811/PROTOCOL.md#L65-L68), [`PROTOCOL.md:292-300`](../beside-plan-20260811/PROTOCOL.md#L292-L300)). |
| Step memory movement during each overlapping load cell | **UNKNOWN — needs-measurement** | The accepted 50,672 MiB value is a starting snapshot, not a per-cell floor; the protocol requires measured values in every load cell ([`PROTOCOL.md:37-42`](../beside-plan-20260811/PROTOCOL.md#L37-L42), [`PROTOCOL.md:292-300`](../beside-plan-20260811/PROTOCOL.md#L292-L300)). |

For an exact receipt, define all terms in MiB:

```text
M38 = measured card-0 B-idle used-memory delta from Arm A
S38 = KV38(ctx=32768, selected path) + R38
H38(step_cell, c) = 50,672 - M38 - c*S38 - T38 - DeltaStep(step_cell)
c_max = floor((50,672 - M38 - T38 - DeltaStep(step_cell)) / S38)
```

`M38` is the measured whole incremental resident cost: target weights, validated drafter,
second-process/Engine overhead, prefix-cache residency, and allocator-pool state. Those diagnostic
components explain the delta but are not added to it again; allocator used/cached/reserved counters
overlap. `S38` is one full-cap live-or-retained session. `T38` is the applicable transient floor.
`DeltaStep` is the Step cell's increase from the accepted starting snapshot. The starting 50,672
MiB and context 32,768 come from
[`PROTOCOL.md:37-50`](../beside-plan-20260811/PROTOCOL.md#L37-L50); the session cost and reserve
semantics come from [`worker.rs:656-667`](../../crates/memra-server/src/worker.rs#L656-L667) and
[`worker.rs:1863-1876`](../../crates/memra-server/src/worker.rs#L1863-L1876).

The target-specific `M38`, `S38`, and `DeltaStep` terms are currently unknown for Qwen3.8; `T38`
also depends on whether the validated campaign path is speculative or plain. Therefore `H38` and
`c_max` are **UNKNOWN — needs-measurement**. A file-size substitution would not repair this: file
bytes do not recover allocator alignment, Engine/static allocations, warm-up arenas, cached pool
blocks, or the learned residual.

## Historical Qwen3.6 proxy — bounded arithmetic, not target evidence

The closest tracked payload used Qwen3.6-27B NVFP4 plus the selected own-trim drafter on a historical
Max-Q pair, not the current target rig. The rig ledger explicitly says those values are historical
only and never a current regression baseline
([`docs/PERFORMANCE.md:60-65`](../../docs/PERFORMANCE.md#L60-L65)). They may bound planning; they may
not be relabeled as Qwen3.8 or current-box measurements.

### Recoverable proxy components

| Proxy component | Value | Source / derivation |
|---|---:|---|
| Qwen3.6 target file | 15,705,920,064 bytes = 14,978.333 MiB = 14.6273 GiB | Exact file stat in [`artifact-manifest.log:9`](../27bab-20260810/raw/setup/artifact-manifest.log#L9); binary-unit division only. |
| Selected own-trim drafter file | 1,242,867,296 bytes = 1,185.291 MiB = 1.1575 GiB | Exact file stat in [`artifact-manifest.log:11`](../27bab-20260810/raw/setup/artifact-manifest.log#L11); the selected launch pair is pinned in [`q27-server.env:3-7`](../27bab-20260810/raw/C/q27-server.env#L3-L7). |
| Combined file payload | 16,948,787,360 bytes = 16,163.623 MiB = 15.7848 GiB | Sum of the preceding two sourced file sizes. This is not used as GPU residency. |
| Measured Q27 process allocation after initial content checks | 16,330 MiB | `nvidia-smi` per-process receipt in [`vram-both-resident-before.txt:3-8`](../27bab-20260810/raw/C/vram-both-resident-before.txt#L3-L8). It already includes the loaded target, selected drafter, one short speculative pool entry, and process overhead; the pool-entry count is in [`vram-both-resident-before.txt:17-18`](../27bab-20260810/raw/C/vram-both-resident-before.txt#L17-L18). |
| CUDA allocator pool at that receipt | 16,508,780,544 bytes reserved = 15,744 MiB; 16,315,563,700 bytes used + 193,216,844 bytes cached | Exact metrics in [`vram-both-resident-before.txt:17-18`](../27bab-20260810/raw/C/vram-both-resident-before.txt#L17-L18). Used plus cached equals reserved, so the cached 184.266 MiB is already inside the reserved pool and the 16,330 MiB process anchor; it must not be added again. |
| Speculative KV geometry | 33,408 bytes/token | Loaded-model admission line in [`q27-server.log:9-13`](../27bab-20260810/raw/C/q27-server.log#L9-L13). |
| Full served-context speculative KV | 1,094,713,344 bytes = 1,044 MiB = 1.01953125 GiB | `33,408 bytes/token * 32,768 tokens`; operands are sourced in [`q27-server.log:9-13`](../27bab-20260810/raw/C/q27-server.log#L9-L13) and [`q27-server.env:1-7`](../27bab-20260810/raw/C/q27-server.env#L1-L7). |
| Learned fixed residual | exact bytes **UNKNOWN — needs-measurement**; log reports 308 MB rounded | Receipt in [`q27-server.log:13-21`](../27bab-20260810/raw/C/q27-server.log#L13-L21); the emitter divides by decimal `1e6` and formats to zero decimal places in [`worker.rs:3412-3424`](../../crates/memra-server/src/worker.rs#L3412-L3424). The rounded value is below 309 MB; for the conservative calculation only, charge 309,000,000 bytes = 294.685 MiB. |
| Required unallocated transient floor | 1,536 MiB | [`worker.rs:847-862`](../../crates/memra-server/src/worker.rs#L847-L862). It remains free headroom; it is not another allocated pool. |
| Q27 prefix cache | 0 MB | Historical launch setting in [`q27-server.env:3-7`](../27bab-20260810/raw/C/q27-server.env#L3-L7). The scored target setting remains unknown. |

The reuse-pool limits are occupancy limits, not up-front allocations. The runtime default allows at
most 64 active interactive sessions, while the parked-entry global cap is 16
([`docs/FLAGS.md:98-105`](../../docs/FLAGS.md#L98-L105)). Actual live and parked entries consume
memory and must appear in the measured process/pool totals; no `16 * session-size` term is added
unless those 16 entries actually exist.

### Conservative proxy equation

Use the measured 16,330 MiB whole-process allocation rather than the smaller file payload, keep its
existing short pool entry in place, charge every campaign session as a new full-context speculative
session, round the learned residual upward to 309 MB, and preserve the full 1,536 MiB transient
floor:

```text
proxy_session = 1,044 MiB KV + (309,000,000 / 2^20) MiB residual
              = 1,338.685364 MiB

H36_proxy(c) = 50,672 - 16,330 - c*1,338.685364 - 1,536 MiB
```

The operands are the line-cited values in the proxy table. Keeping the already-present short pool
entry in the 16,330 MiB anchor and then charging all `c` sessions anew deliberately double-counts
that small entry instead of granting an optimistic credit.

| Additional full-cap Q27 sessions `c` | Conservative remaining card-0 headroom | Interpretation |
|---:|---:|---|
| 1 | 31,467.31 MiB = 30.73 GiB | Requested probe rung; arithmetic positive. |
| 2 | 30,128.63 MiB = 29.42 GiB | Required steady Q27 load in `B-active` is `c=2` ([`PROTOCOL.md:236-238`](../beside-plan-20260811/PROTOCOL.md#L236-L238)). |
| 4 | 27,451.26 MiB = 26.81 GiB | Requested probe rung ([`PROTOCOL.md:242-248`](../beside-plan-20260811/PROTOCOL.md#L242-L248)). |
| 8 | 22,096.52 MiB = 21.58 GiB | Highest requested probe rung; arithmetic positive. |
| 24 | 677.55 MiB = 0.66 GiB | Last positive integer under this proxy equation, but not a justified probe or safety claim. |
| 25 | -661.13 MiB = -0.65 GiB | First negative integer under this proxy equation. |

Thus the proxy-only capacity ceiling is `min(runtime 64, arithmetic 24) = 24` additional full-cap
Q27 sessions under the stated snapshot assumptions. The exact Qwen3.8 ceiling remains **UNKNOWN —
needs-measurement**. The protocol's requested highest rung is `c=8`, not `c=24`, and cross-rig,
cross-model arithmetic is not permission to extend it. The historical campaign established only
two persistent Q27 namespaces as a measured safe shape and explicitly did not establish a higher
safe bound ([`27bab RESULTS.md:85-97`](../27bab-20260810/RESULTS.md#L85-L97)).

The older coarse envelope reached the same qualitative proxy conclusion by charging 24–26 GiB for
the complete Qwen3.6 service and leaving 25.48–23.48 GiB from the 50,672 MiB starting budget
([`PROTOCOL.md:57-65`](../beside-plan-20260811/PROTOCOL.md#L57-L65)). The calculation above is more
conservative at `c=8` because it retains the measured whole-process anchor, charges a full-context
session plus residual per rung, and keeps the transient floor.

## Layout and session-fit verdicts

| Byte layout | Exact maximum Q27 sessions | Fit verdict for the scored A/B |
|---|---:|---|
| Step and Q27 both wholly on one 96 GB card | 0 | **NO-GO.** Step alone is PP-2-only at 105.0 GB ([`docs/PERFORMANCE.md:524-527`](../../docs/PERFORMANCE.md#L524-L527)). |
| Protocol layout: Step PP-2 on cards 0/1; Q27 process on card 0 | **UNKNOWN — needs-measurement** | **NO-GO today: unproven fit.** The historical Qwen3.6 proxy is positive through the planned `c=8` rung and has a proxy-only ceiling of `c=24`, but the exact Qwen3.8 terms are absent. |
| Same protocol layout during overlapping Step load | **UNKNOWN — needs-measurement** | Step's per-cell memory delta is also pending; every Arm-A/B memory row is still pending in [`PROTOCOL.md:292-300`](../beside-plan-20260811/PROTOCOL.md#L292-L300). |

“Forces a split” here means only that the tracked Step bytes cannot be placed on one 96 GB card.
It is not a recommendation to keep, change, or buy any topology.

## Receipt required to change the verdict

The fit verdict can become GO only after the target-box receipt resolves the unknowns without
changing the controlled object:

1. Freeze the exact Qwen3.8 target and optional validated draft revision, file sizes, SHA-256s,
   template, and context cap as required by
   [`PROTOCOL.md:20-24`](../beside-plan-20260811/PROTOCOL.md#L20-L24).
2. Record card-0/card-1 and per-process used/free values after Step warm-up, after Q27 load, after Q27
   warm-up, at each peak, and at the final free-memory floor as required by
   [`PROTOCOL.md:65-68`](../beside-plan-20260811/PROTOCOL.md#L65-L68) and
   [`PROTOCOL.md:292-300`](../beside-plan-20260811/PROTOCOL.md#L292-L300).
3. Capture the exact loaded-model plain/spec bytes per token and learned fixed residual, then solve
   `H38` instead of substituting Qwen3.6.
4. First prove the useful A/B floor: steady Q27 `c=2` with exactly two persistent namespaces while
   the Step cells overlap ([`PROTOCOL.md:236-238`](../beside-plan-20260811/PROTOCOL.md#L236-L238)).
   Then probe only the frozen `c=1/2/4/8` ladder
   ([`PROTOCOL.md:242-248`](../beside-plan-20260811/PROTOCOL.md#L242-L248)), retaining the transient
   floor and stopping on the protocol's captured failure rules
   ([`PROTOCOL.md:65-68`](../beside-plan-20260811/PROTOCOL.md#L65-L68)).

Until those receipts exist, **NO-GO means “do not spend the scored A/B window on an unpinned fit,”
not “Qwen3.8 has been shown not to fit.”**
