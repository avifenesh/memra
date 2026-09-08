# Full-token segmented replay: +2.64% plain envelope

On two RTX PRO 6000 Blackwell development GPUs, the bounded same-load 20-row
ABBA measured **42.799984 tok/s eager** and **43.931954 tok/s full-token graph**,
**+2.644789%**, with ten eligible rows per arm. The mean full-envelope saving is
0.602019 ms/token. This is a modest positive result, far below 120 tok/s; no
serving/default promotion or broader qualification follows automatically.

Measured source: `754438bb0ea46418b43b44a470ab2affa892cfcd`.
Release binary SHA-256:
`3cadc3b7aa17a24f068afb5a1478cc467898e8f25969c931f07960d0a7a80ce3`.
The raw request/source, model log, graph DOTs, controller exits, 250-ms telemetry,
process inventory and complete 20-row table are held in the companion private
ops namespace `full-token-replay-model-754438b-r1`. Model log SHA-256 and graph
hashes are recorded there with the independently validated summary.

## Scope and measured rows

Plain, one-token decode, 256 prime / 256 sampled outputs; all 43 layers use
attention TP2 and expert-ID EP. Device sampler and small-kernel diet ON, split-K
OFF, device caches, no DSpark, no host C4, no server or long-context ladder.
Both arms use the same f32x/RefFp8Round program and sampled configuration.

| Block | Arm | Rows | Pooled tok/s |
| --- | --- | ---: | ---: |
| A1 | eager | 5 | 42.700802 |
| B1 | graph | 5 | 43.881686 |
| B2 | graph | 5 | 43.982337 |
| A2 | eager | 5 | 42.899628 |

Graph rows range 43.536215..44.011611 tok/s; eager rows range
42.513599..42.976639. Every graph row exceeds every eager row in this window.
This is one experiment with ten rows/arm, not ten independent machine boots.
Telemetry over load and execution spans 30..62 C and 28..57 C on the two GPUs;
no power-limit or clock-limited attribution is made.

The timer includes live inputs, forward, both refusal reads, commit, head,
device sample and readback. The first scored B row includes capture/instantiate
time; later B rows retain the same four graphs. Primed state is copied into
existing allocations outside row timing. The common initial carry draw is
outside timing, the final next draw is inside timing, identically for both arms.
No hash/profile work is in the measured interval.

## Correctness and engagement

Before timing, all 256 changing-token steps matched eager sampled tokens, final
logit SHA-256 and both-rank live-cache/hidden digests through position 512.
The gate checks every step; it emits 65 progress records (initial position and
every C4 boundary). This covers C4/C128 emission and ring-wrap boundaries.

Each rank's retained forward graph contains 3,493 nodes, 3,240 kernels, 86 AR
nodes, one embedding and 86 HC-post nodes. Both complete layer paths are captured;
unsupported graph node types are absent. Successful commit graphs have 87 nodes
on rank 0 and 116 on rank 1, including the head and real sampler on rank 1.
Rank 1 has 43 copy nodes and 2 memset nodes, not 45 copies.
Device counters advance for each executed segment; capture counts remain `[1,1]`
for the correctness state and the separate scored state, with no recapture or
per-layer eager fallback. All per-block AR epochs are checked.

Six live refusals are armed after capture: each rank at layers 0/21/42 and
positions 259/383/511. Each leaves cache and position unchanged, advances only
the forward counters, and quarantines ordinary retry and prefix-reset retry.
Normal execution, missing-peer/partial-submit lifetime components, the actual
Rust fail-stop policy and targeted CUDA sanitizers have separate scoped receipts.

All 20 scored rows are eligible/non-looping and share these identities:

- Tokens SHA-256: `35e9e69e90047d273f266b7a4a71e0848d403be66eeebfcb1fe89a37d1caa433`.
- Final logits SHA-256: `37eb73d85d43b2237338b7eb069355838f166eafd95dbb9edd71464e2dcff9ca`.
- Host-reconstructed intended control-sequence SHA-256 (not a device readback): `ccb6de6553cf13a9bb02f5cf7eb44cfebbe08db432dba95e7ee2114edff09cbc`.
- Both cache digests: `5673480229060882075`; both hidden digests: `4231551965497114380`.

The controller exits zero; only the owned model process appears in the process
inventory. After completion the pair is empty and its shared GPU lock is free.
The shared model/pod remains available to its controller.

Verdict: keep the default-OFF diagnostic for root's qualification decision, with
decide-by 2026-09-22. No further experiment, merge or production rollout is started
by this result. Source review and hosted CI remain independent integration gates.
