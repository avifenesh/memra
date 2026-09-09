# PP1 composed verify graph mechanism, 2026-09-09

Base: compose `b737d9827`, including KDA #388, MLA #394 and PMIN #393.
The measured candidate keeps KDA and MLA ON and legacy PMIN. References below
name the uninstrumented candidate archived by this lane's implementation commit.
The final branch removes this candidate after its NEGATIVE decision.

| Source | Mechanism and capture contract |
|---|---|
| `crates/memra-engine/src/glm_spec.rs:4275` | Spec round enters verify; draft RNG, acceptance, penalties and rollback remain outside capture. |
| `crates/memra-engine/src/glm_spec.rs:1506` | PP1 partitions the t-row trunk into KDA pieces and eager MLA layers. Pieces also end at DFlash feature-tap layers. Widths 2..7 have separate graph sets. |
| `crates/memra-engine/src/glm_spec.rs:1576` | Shared eager/captured layer body: mHC, norm, batched KDA projections and scan, device MoE routing, experts and residuals. No alternate model math. |
| `crates/memra-engine/src/glm_spec.rs:902` | First visit is eager. Subsequent visits warm the exact piece, prefill workspace, capture, census and compare output bits against eager before serving replay. |
| `crates/memra-engine/src/glm_spec.rs:935` | Ordered conv/SSM/alternate addresses and sizes distinguish both recurrent phases. Lookup also keys range and t. More than two pointer signatures refuses eager. |
| `crates/memra-engine/src/glm_spec.rs:1056` | A new capture needs 1 GiB setup headroom. Existing replay and self-check launches each check the shared 256 MiB launch floor. In-round tracing syncs, host rows, unsupported t and non-device routing decline. |
| `crates/memra-engine/src/glm_spec.rs:1108` | Warm rollback stashes stay alive outside capture. Their replacement must not free external allocations inside the graph. Rollback snapshots are refreshed outside capture, never baked request-owned addresses. |
| `crates/memra-engine/src/glm_spec.rs:1909` | DFlash feature taps write per-round sinks outside capture. Graph output at each tapped layer feeds the same contraction and copy. |
| `crates/memra-engine/src/lib.rs:11117` | Resident shared-expert ones avoid pageable HtoD constants. The capture census refuses any host-endpoint memcpy node. |
| `crates/memra-engine/src/lib.rs:13877` | Trim unused capture warmup workspace back to normal class/byte limits after each round; graph keepers and lent rollback stashes are separate owners. This cost is included in timing. |
| `crates/memra-engine/src/glm_spec.rs:796` | Session teardown destroys graph sets and trims unused graph memory on the owning PP1 device. Ordinary async-pool trimming alone did not recover admission. |

`hybrid_forward.rs:10441` is the older live-MLA program; it bypasses the
composed rows-exact split-KV dispatch. `:11052` preserves the requested MLA
program, whose DSA selection has position-dependent geometry. This candidate
therefore leaves MLA layers eager. It captures KDA pieces, not the entire
verify round or MLA halves. All t2..7 occur among successful captures across
the final cell, but the setup reserve leaves some pieces/widths eager within
each request. The negative result applies to this implementation and memory
posture, not to every possible full-verifier capture design.

PP1 plain uses `MEMRA_GLM5_DECODE_GRAPH`, not the TP symmetric door. Its live
receipt says `dev=0 stage=[0,45) runs=1 captured_layers=45 mla_halves=11
mla_mids=0`: two recurrent phases with an eager MLA middle. The admission and
launch guard is `glm5_decode_graph.rs:943`; `glm5_tp_sym_graph.rs` is the
rank-specific precedent, not this route.

## Price from the supplied tally

At t4: 2,933 launches, 1,034.122 us kernel-free gaps, 861.639 us idle after
excluding copies/memsets. Observed gap per launch is
`1034.122 / 2933 = 0.352581657 us`. Removing every such gap would save
`2933 * 0.352581657 = 1.034122 ms/round`; idle-only bound is 0.861639 ms.
This is an optimistic screening bound, not predicted saving. KDA pieces cover
only part of the round, graph-node issue gaps remain, and capture costs must
be repaid. The small-kernel means (2.1, 1.7, 4.2, 3.1 us) are execution times,
not CPU launch gaps. The corpus's H100 launch-cost difference is not a measured
B200 multiplier. Composition also changed kernels and width distribution.

## Capture and lifecycle evidence

Attempts 1 and 2 refused capture and completed through eager fallback; their
zero replay counters failed the oracle. Holding warm stashes outside capture
unblocked attempt 3: 69 pieces captured, first 16 rounds bitwise against OFF,
then INVALID_VALUE during more setup near full VRAM. A setup reserve let
attempt 4 pass the entire K6 greedy oracle, but its next request received 429.
Graph-pool trimming let attempt 5 pass all four oracles; ordinary warmup
workspace still caused a later 429. The final cell includes both cleanup paths
and completes all 22 requests in one boot, with no failed self-checks.

NVIDIA documents the separate graph allocation pool and its unused-memory
trim contract in [Physical Memory Footprint](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#physical-memory-footprint).
These failures and fixes are retained as receipts; fallback oracles and partial
requests do not enter the final throughput medians.
