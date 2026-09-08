# Full-token segmented replay: two bounded positive plain results

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

## Fresh-load reverse-order confirmation

The authorized follow-on at source `bd30a57bad9295c9668c95871dc650f3790f4c91`
measured **42.804086 tok/s eager** and **44.005344 tok/s graph**,
**+2.806409%**, saving **0.637743 ms/token**. Its exact schedule is
BBBBB AAAAA AAAAA BBBBB, ten eligible 256-output rows per arm. All 256 changing
steps, six live refusals, both-rank cache/hidden identity, epochs and graph census
checks passed. All row identities match the initial experiment. First scored B
capture is inside its timer; counters reach 2560 per segment/rank with one capture.

This is a separate source/binary from the initial result. Changes are harness
arm order, a separate profile-only selector and NVTX-only markers disabled during
scoring, with no CUDA/FFI or numerical dispatch change. Binary SHA-256:
`4880e3caabc7a8784789ceb4f9e8f83168ac77de12a56a9bc7fc41464a35ad9b`.
All hosted CI checks passed on this measured head. The clean controller exited
zero; process-map readback shows no profiler injection. Complete unfiltered rows,
source/control/graph hashes and validation are in private receipt namespace
`full-token-replay-model-baab-bd30a57-r1` and companion `BAAB-RESULT.md`.

## Separate profile, not scored timing

After BAAB, a separate process loaded the same binary/tape and profiled 32 steps
per arm from identical restored position-368 prefixes through position 400.
Both arms match sampled tokens, final state and epochs, including the C128
boundary. Controller EXIT=0; no MEASURE or scored-rate rows were emitted. Every
replay kernel event has a graph-node ID; each token has full embedding/43-layer
HC-post/86-AR-per-rank coverage.

Nsight node tracing adds overhead. Instrumented mean shared GPU forward spans
28.636 ms eager and 26.282 ms replay, while host steps are 30.537 and 30.305 ms.
Replay forward kernel-free time is 0.669 ms on rank 0 and 4.332 ms on rank 1,
with 3.633 ms of traced rank-start skew. The 22.078-ms host refusal read/drain
waits on queued forward execution; it is not additional removable time. Aggregate
graph-launch API duration is likewise not a native launch-cost estimate.
The replay sampler GPU window is 0.090 ms, then 0.016 ms to host step end.

Source/cadence accounting reconciles 575.25 extra replay kernels per rank/token
in this window: 396.125 inactive compressor emission/shift kernels and 179.125
live copy/control kernels. The always-launched emission sequence is at
`cu/dsv4_gpu.cu:6683`, called by `cmp_decode_batch_dev` in `dsv4_gpu.rs`.
A possible next mechanism is a small retained graph set selected by compressor
cadence, keeping live addresses and exact arithmetic. It is unimplemented; node
counts do not predict wall savings. Large GU/down/dense kernels still dominate
forward work. AR residence is not counted as removable wall. Split-K stays OFF.

Full phase tables, per-step interval unions, API/family counts, source-derived
node reconciliation and trace hashes are in companion `PROFILE-RESULT.md`,
namespace `full-token-replay-profile-bd30a57-r1`. Raw profiler blobs stay private
on the development host. Both authorized processes are complete; final inventory
is empty and the shared lock is free. No additional run, merge, default change
or serving admission is implied by these two modest positive results.
