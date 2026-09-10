# GLM PP1 verify graphs: NEGATIVE, 2026-09-09

**K6 graph ON is 79.191090 tok/s versus plain 79.957362: -0.958351%, below
the required +3% KEEP bar.** K6 graph OFF is 80.209701 tok/s. Auto graph ON
is 81.254014 tok/s, +1.621679% versus plain, also below +3%.
The new `MEMRA_GLM5_SPEC_VERIFY_GRAPH` door and its runtime additions are
removed in this lane. This does not refute every full-verifier graph design.
rev: 2026-09-23

## Source and mechanism

Base: `b737d98278a0132451ed7a152992d70f5270e494`, the compose branch behind
[Memra #399](https://github.com/avifenesh/memra/pull/399), including #388 KDA,
#394 MLA and default-OFF #393 causal PMIN. Archived candidate:
`6e073bd5a`; measured server SHA256:
`8bf811cec83f4cf9e725cb177c25cf2e761dcb4d73f6a1a517f413c2363ff52b`.
Research-only request controller and logit/token receipts are pinned in the
archive's instrumented source and source hashes. Both composition doors are
ON in every speculative decision arm; PMIN is the requested legacy .7 rule.
No fleet pin, serving default or production deployment changes.

[MECHANISM.md](MECHANISM.md) maps the source by file and line. The candidate
captures fixed-width KDA pieces, split at DFlash feature taps, with two pointer
phases per piece and graph sets for t2..7. MLA/DSA stays eager to preserve the
composed split-KV arithmetic. This is partial verifier capture, not a graph of
the entire round. Captures occur at every t2..7 across the final cell, while
1 GiB setup headroom leaves some pieces and widths eager within each request.
The first visit is eager, graph output self-checks are bitwise, host-copy nodes
refuse, and each graph launch has the shared headroom guard. Warm-stash lifetime,
rollback ownership, recurrent phase and tap destinations are explicit.
Unused warmup buffers are trimmed to normal workspace limits after each round;
session teardown releases unused graph-pool memory. These costs are measured.

The supplied t4 tally allows only 1.034122 ms of kernel-free gaps over 2933
launches (0.352582 us per launch), or 0.861639 ms idle excluding copies/memsets.
That is a screening bound, not expected saving. Plain PP1 already logs all
45 layers under its decode graph, with 11 MLA halves and an eager MLA middle.

## Oracle

Each pair uses KDA+MLA ON and differs only in graph policy. Complete worker
output IDs, each accepted round's token sequence, and SHA-256 of every f32
verify-logit byte stream are identical. Seed: 39320260909. Sampled requests
omit sampling parameters; model metadata supplies temperature=1/top_p=.95.
Greedy remains a correctness instrument and contributes no throughput rows.

| Pair | Output tokens per arm | Verify rounds per arm | Logits | Accepted sequence and token tape |
|---|---:|---:|---|---|
| greedy K3 OFF/ON | 160 | 62 | Identical | Identical |
| greedy K6 OFF/ON | 160 | 54 | Identical | Identical |
| sampled K3 OFF/ON | 160 | 68 | Identical | Identical |
| sampled K6 OFF/ON | 160 | 52 | Identical | Identical |

Greedy K3 and K6 token SHA256:
`c3d1c394e342f07841b0dece697e40d4db74d2a3dfe92d4640507e4a7d949b33`,
matching the compose MLA-ON tape. Zero graph self-check failures. ON requests
must contain actual captures and positive replay counters; fallback-only
requests failed earlier attempts rather than passing this oracle.

## Same-boot throughput

One development B200, sm100a, PP1; no production-pair access. One successful
boot, two excluded 64-token warmups, eight oracle requests and twelve timing
requests. Four arms, N=3, forward/reverse/rotated order. All 22 requests complete
HTTP 200 with terminal SSE. Every scored request uses all 29,781 cached prompt
tokens from the p32k fixture. All timing rows produce 512 output tokens. No loop
flags. Nominal 250 ms telemetry spans 33..48 C. No concurrent GPU tenant.
HTTP rate is completion_tokens/wall, including TTFT and capture. Tick ms/token
sums server decode ticks divided by actual output count. Synchronization at
verify boundaries is identical in both arms. Columns are independent medians.

| Arm | K | N | Accepted/round | HTTP ms/round | Tick ms/token | HTTP tok/s |
|---|---:|---:|---:|---:|---:|---:|
| plain | 0 | 3 | - | - | 12.318750 | 79.957362 |
| K6-off | 6 | 3 | 1.772973 | 33.269985 | 12.310029 | 80.209701 |
| K6-on | 6 | 3 | 1.647668 | 34.175333 | 12.497502 | 79.191090 |
| auto-on | auto | 3 | 1.480583 | 30.420813 | 12.186221 | 81.254014 |
K6 ON minus OFF HTTP ms/round is **+0.905348 ms**: signed saving
**-0.905348 ms/round**. Median verify ms/round saving is **-1.119820 ms**.
These random sampled requests have different continuations and accepted work,
so those differences are not matched-input estimates of the graph's kernel cost.

The fixed-seed K6 sampled oracle supplies a matched-input diagnostic: all 52
rounds are identical, with capture-inclusive mean verify saving **-3.487295 ms**.
Among 38 non-capture rounds at widths with an existing captured piece, mean
saving is **+0.015750 ms**, median **-0.089474 ms**. That subset is a partial-graph
and instrumented diagnostic, not a second serving benchmark. It does not show
useful steady verify saving in this cell. Full per-round pairs, including the
slow sampled K3 oracle, are in matched-rounds.json; none enter the timing table.

## Failed attempts and disposition

Attempts 1/2: capture INVALID_VALUE and zero replay; eager fallback was exact
but failed non-vacuity. Attempt 3: 69 captured pieces and 16 exact completed
rounds, then INVALID_VALUE during more setup near full VRAM. Attempt 4: complete
K6 greedy oracle, then admission 429. Attempt 5: all four oracles, then 429 after
the first timed plain request. The final cell includes setup headroom, graph
memory teardown and bounded ordinary workspace retention, and completes all
requests. Failed/partial attempts are retained and excluded from medians.

**NEGATIVE.** Per the door law, delete the new env read, graph dispatch additions
and dedicated warmup/teardown code; retain the existing pre-lane graph machinery
unchanged. The implementation commit and private source bundle preserve the
experiment. No recommendation to enable this graph or alter the known legacy
PMIN sampling contract follows from these measurements. rev: 2026-09-23

## Custody and validation

Private Darklanes PR #440:
`research/glm5-1m-b200-ship-20260906/receipts/dflash2-20260908/verify-graph-20260909/verify-graph-raw.tar.gz`.
Archive SHA256:
`e8c96cdffb5a0d0a9fa04f80e7c75f5afd1dc402038be00d1d87e007c2e786f0`.
All 272 manifest members verified after transfer. It contains all attempts,
requests/SSE, token and logit receipts, telemetry, build logs, measured
instrumented source and candidate Git bundle. The drafter weight, drafter
config and prompt hashes were rechecked against the compose receipt. The full
target mint was reused at the same path; no new full-shard rehash is claimed.

Remote release build and uninstrumented formatting passed. No Cargo, server,
bench or GPU gate ran on the rig. The commit formatting hook ran remotely under
the owner's no-rig-Cargo instruction. Push uses MEMRA_SKIP_PERF_CI=1; GitHub CI
remains the merge gate. This draft is a research bank, not a merge or deployment.

publicity: skipped - maintenance research record.

Claude-Session: https://claude.ai/code/session_01TFyR32RLUiSejCgrPm5nNj
