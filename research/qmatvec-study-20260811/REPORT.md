# IQ4_XS qmatvec bandwidth feasibility — Step-3.7 B=1

Date: 2026-08-11

Lane: `lane/cx-qmatvec-study`

Scope: read-only source/evidence study at base `4fa5a266`; no GPU run and no kernel/runtime change.

## Decision

**GO** to a follow-up implementation experiment for a single-digit-percent-or-more bandwidth
recovery at frozen Step exactness. The first arm should copy the already-present wide-load,
`byte_perm` IQ4_XS expert dot into the dense `qmatvec_iq4_XS_dp4a` body while preserving its
group assignment, dp4a order, float accumulation, and 128-thread reduction tree.

This is a GO to measure, not a speedup receipt. Confidence is medium-high because:

- The current dense kernel still executes the scalar decode that the existing wide expert body was
  written to replace ([`qmatvec.cu:5387-5431`](../../crates/memra-engine/cu/qmatvec.cu#L5387-L5431)
  versus [`:5005-5045`](../../crates/memra-engine/cu/qmatvec.cu#L5005-L5045)).
- Current upstream llama.cpp uses the packed-load + `byte_perm` mechanism for exactly
  IQ4_XS × Q8_1.
- The same rewrite was bit-clean in memra's expert path and later measured **+2.5% end-to-end** on
  the 35B IQ4_XS model ([`rig5090.jsonl:215,217`](../tune-data/rig5090.jsonl#L215)). That is a
  structural transfer prior, not proof of the Step dense-trunk result.
- The profile shows the expected symptom: correct DRAM byte volume, but persistent L1TEX
  long-scoreboard stalls and some LG-instruction-queue pressure.

There is **not** a recoverable 49% over-fetch hole. The NCU **50.81%** value is useful cold replay
mechanism evidence, but unperturbed Nsys timing plus the exact tensor byte bill place the runtime
family at about **63.6% of card bandwidth**. Its absolute perfect-HBM ceiling is approximately
**1.80 ms/token**, versus 2.8363 ms measured: at most **1.03 ms/token**, **11.0% of the 9.401 ms
token wall**, or **+12.35% throughput** if every remaining qmatvec stall vanished. The plausible
first-arm win is much smaller.

## Evidence boundary and a correction to the headline rate

The committed Nsys window is the timing authority: N=32, 315 IQ4_XS launches/token,
2.8363 ms/token, and 30.17% of token wall across the serial PP pair
([`ncuspike RESULTS.md:39-55`](../ncuspike-20260811/RESULTS.md#L39-L55)). NCU sampled only device 0,
one launch per distinct `(symbol, grid, block)` configuration, with `--cache-control all` and 11
replay passes. The original report explicitly says NCU duration is not a throughput measurement
([`:81-104`](../ncuspike-20260811/RESULTS.md#L81-L104)); the driver records the filtering and cache
policy ([`box1-profile.sh:182-214`](../ncuspike-20260811/box1-profile.sh#L182-L214)).

That distinction matters here:

| surface | launches/token | logical bytes/token | time/token | effective logical rate | card BW |
|---|---:|---:|---:|---:|---:|
| Nsys device 0 | 154 | 1.505217 GB | 1.448200 ms | 1,039.37 GB/s | 65.08% |
| Nsys device 1 | 161 | 1.373822 GB | 1.388090 ms | 989.72 GB/s | 61.97% |
| **Nsys pair** | **315** | **2.879039 GB** | **2.836290 ms** | **1,015.07 GB/s** | **63.56%** |
| NCU device-0 replay, Nsys-time-weighted | 154 | configuration sample | replay duration | 811.4 GB/s | 50.81% |

The 2.879039 GB bill is 2.872947 GB of IQ4_XS weights, 1.702 MB of Q8 activation buffers, and
4.390 MB of output stores. The semantic launch map is reproduced in
[`analyze.py`](analyze.py): device 0 owns 22 layers (154 calls), device 1 owns 23 (161 calls), and
the counts assert against `summary.json`. Shapes come from the pinned GGUF receipt—for example,
`n_embd=4096`, dense FFN 11,264, shared FFN 1,280, and the 8,192/12,288 attention projections
([`gguf-header...txt:6-25,55-108`](../step37-bringup-20260802/raw/gguf-header-stepfun-iq4xs-shard1-20260802.txt#L6-L25)).

There is one profiling-granularity trap. Grid 4096 represents four input widths in the real walk:
attention-out K=8192/12288, dense-down K=11264, and shared-down K=1280. NCU's
`per-launch-config` key cannot distinguish kernel arguments. The first 4096-row launch follows the
SWA Q/K/gate sequence and has K=8192
([`cuda-gpu-trace.csv:11-34`](../ncuspike-20260811/raw/box1/nsys/cuda-gpu-trace.csv#L11-L34)), so that
is the shape countered for the 4096 configuration. The published 0.6038 ms/token is still the Nsys
sum of **all** 4096-row calls. Consequently, 50.81% must not be projected mechanically across that
mixed-K time or onto device 1.

The unperturbed rate above is an effective rate derived from known bytes, not a second DRAM counter.
It is justified as a ceiling sanity check because the countered launches independently show that
their actual DRAM bytes equal the logical matrix bill to within measurement/cache effects (next
section). The strictly compulsory weight-only HBM floor is 1.7990 ms/token; charging Q8 reads and
output stores to HBM gives 1.8028 ms/token. The honest ceiling is therefore 1.80 ms/token either
way.

## 1. Roofline placement: memory-intensity-bound, implementation latency-limited

### Byte and operation model

IQ4_XS stores 256 weights in 136 bytes: fp16 super-scale (2), high scale bits (2), four low-scale
bytes, and 128 packed nibbles. That is **0.53125 B/weight = 4.25 bpw**
([`qmatvec.cu:276-290`](../../crates/memra-engine/cu/qmatvec.cu#L276-L290); upstream's structure is
also [`block_iq4_xs`](https://github.com/ggml-org/llama.cpp/blob/ebb546b7e961bd46fd9ed0387ffd14ca86b6fe1b/ggml/src/ggml-common.h#L454-L460)).
At K=4096:

- one weight row is `4096 / 256 × 136 = 2,176 B`;
- memra's Stage-B Q8 buffer is 4,096 int8 values plus 128 f32 scales = **4,608 B**
  ([`qmatvec.cu:457-498`](../../crates/memra-engine/cu/qmatvec.cu#L457-L498));
- useful matvec work is conventionally `2NK` operations; and
- the full logical matrix bill is `N × 2,176 + 4,608 + N × 4` bytes.

If the activation were fetched from HBM separately for every row, intensity would be only
`8192 / (2176 + 4608 + 4) = 1.207 FLOP/B`. That is too pessimistic: the activation is shared by
all rows and is cache-resident. At matrix scope the measured shapes are **3.638–3.759 FLOP/B**,
approaching the weight-only asymptote `2 / 0.53125 = 3.7647 FLOP/B`:

| out_f | in_f | full logical bytes | arithmetic intensity | 1,597-GB/s roof |
|---:|---:|---:|---:|---:|
| 64 | 4096 | 0.1441 MB | 3.6377 FLOP/B | 5.809 TFLOP/s |
| 1280 | 4096 | 2.7950 MB | 3.7516 FLOP/B | 5.991 TFLOP/s |
| 8192 | 4096 | 17.8632 MB | 3.7568 FLOP/B | 6.000 TFLOP/s |
| 12288 | 4096 | 26.7924 MB | 3.7572 FLOP/B | 6.000 TFLOP/s |
| 4096 | 8192 | 17.8514 MB | 3.7593 FLOP/B | 6.004 TFLOP/s |

NVIDIA currently specifies the Server Edition card at **1,597 GB/s** and **120 FP32 TFLOP/s**
([official specifications](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)).
Even the conservative FP32 ridge is `120,000 / 1,597 = 75.14 FLOP/B`, twenty times the IQ4_XS
intensity. The useful-work bandwidth roof is only about 6 TFLOP/s. Thus the algorithm belongs on
the memory side of the roofline; it is nowhere near a compute-throughput roof.

### The missing bandwidth is not missing bytes

For each NCU configuration, `dram__bytes.sum.per_second × gpu__time_duration` reconstructs the
bytes below. “Logical” includes weight, one Q8 vector read, and output stores. Ratios just below
1.0 mean the small activation/output contribution did not all cross DRAM or reflect metric
rounding; weight bytes alone are fully accounted for.

| out_f | captured in_f | logical MB | measured DRAM/logical | card BW | occupancy | waves/SM | long scoreboard |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 4096 | 0.1441 | 1.0604× | 1.67% | 8.14% | 0.03 | 9.92 |
| 96 | 4096 | 0.2139 | 1.0389× | 2.43% | 8.14% | 0.05 | 10.17 |
| 1024 | 4096 | 2.2369 | 1.0021× | 17.69% | 43.57% | 0.54 | 18.00 |
| 1280 | 4096 | 2.7950 | 1.0015× | 21.07% | 54.98% | 0.68 | 21.14 |
| 4096 | 8192 | 17.8514 | 0.9996× | 54.99% | 74.20% | 2.18 | 16.82 |
| 8192 | 4096 | 17.8632 | 0.9987× | 56.48% | 74.01% | 4.36 | 12.30 |
| 11264 | 4096 | 24.5601 | 0.9985× | 63.56% | 74.07% | 5.99 | 10.95 |
| 12288 | 4096 | 26.7924 | 0.9985× | 66.18% | 73.89% | 6.54 | 10.13 |

The Nsys-time-weighted DRAM/logical ratio is **0.9991×**. There is no 2× transaction or cache-line
over-fetch story hiding behind the 50.81% rate. The 64/96-row kernels pay 4–6% fixed/sector
overhead, but together are tiny. The large matrices move essentially the minimum bytes once.

What remains is latency and issue throughput. Every sampled shape is led by long scoreboard
(10.13–21.14 warps per issue); the large shapes also show LG throttle of 2.77–3.69. NVIDIA defines
long scoreboard as waiting on an L1TEX dependency and recommends coalescing/cache locality, while
LG throttle points to an overfull local/global instruction queue and specifically motivates fewer,
wider loads and interleaved memory/math
([current Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)).
That matches the source: many narrow dependent loads feed each eight-dp4a group.

**Verdict for question 1:** the *operation* is memory-intensity-bound, but the current kernel is not
bandwidth-saturated. It is capped below the HBM roof by load/decode dependency latency and, for the
small grids, lack of enough CTAs. Bytes are already right; the instruction schedule is not.

## 2. Codebook serialization: historical constant-cache bug fixed; scalar lookup chain remains

The claim needs splitting into two eras.

1. **Current constant-cache serialization: refuted.** `kvalues_iq4nl_d` is aligned, plain
   `__device__` storage, explicitly **not** `__constant__`
   ([`qmatvec.cu:162-170`](../../crates/memra-engine/cu/qmatvec.cu#L162-L170)). Current llama.cpp
   likewise expands its CUDA table macro to `static const __device__`, not constant memory
   ([upstream `ggml-common.h:490-494`](https://github.com/ggml-org/llama.cpp/blob/ebb546b7e961bd46fd9ed0387ffd14ca86b6fe1b/ggml/src/ggml-common.h#L490-L494)).
   Memra already captured the historical bug: moving this table from constant to device storage
   improved 35B decode 129.5→149.7 tok/s, **+15.6%**, with gates green
   ([`rig5090.jsonl:155`](../tune-data/rig5090.jsonl#L155), commit `146179e5`). That win is already
   in the baseline and cannot be claimed again.

2. **Current indexed-lookup dependency: confirmed in source, strongly supported but not isolated by
   counters.** The hot dense kernel still constructs four low and four high packed int8 vectors per
   32-value group through 16 scalar `qs` byte positions and 32 dynamically indexed codebook byte
   expressions before issuing eight dp4a operations
   ([`qmatvec.cu:5399-5421`](../../crates/memra-engine/cu/qmatvec.cu#L5399-L5421)). The 16-byte
   codebook itself remains in L1, so those are not 32 DRAM fetches. They are still many narrow L1
   requests and a serial load→index→pack→dp4a dependency chain. Aggregate long-scoreboard counters
   cannot assign a percentage specifically to table versus weight versus activation loads; only a
   source-correlated/SASS implementation A/B can close that attribution.

Llama's current IQ4_XS dot avoids the scalar path. It reads four packed weight bytes at a time,
loads the 16-byte table as four 32-bit words, uses `__byte_perm` to materialize eight selected int8
values in two registers, then issues the same low/high dp4a pair
([`get_int_from_table_16`](https://github.com/ggml-org/llama.cpp/blob/ebb546b7e961bd46fd9ed0387ffd14ca86b6fe1b/ggml/src/ggml-cuda/vecdotq.cuh#L34-L80),
[`vec_dot_iq4_xs_q8_1`](https://github.com/ggml-org/llama.cpp/blob/ebb546b7e961bd46fd9ed0387ffd14ca86b6fe1b/ggml/src/ggml-cuda/vecdotq.cuh#L1337-L1362)).
Memra already contains the equivalent helper at `qmatvec.cu:174-190` and a value-identical IQ4_XS
wide group dot: one 64-bit header load, two 64-bit quant loads, four packed lookups, unchanged
low/high dp4a issue order, and unchanged final float expression (`:5005-5045`). The dense hot kernel
simply does not use it.

**Verdict for question 2:** “constant cache serializes the current kernel” is false. “The current
scalar codebook/weight decode creates avoidable load and dependency issue pressure” is a
high-confidence mechanism hypothesis with an exact in-tree replacement and a positive prior.

## 3. Launch shape: correct for K=4096, underfilled at small N, scalar loads poorly issued

The live launch is `grid=(out_f,m,1)`, `block=(128,1,1)`
([`lib.rs:4077-4095`](../../crates/memra-engine/src/lib.rs#L4077-L4095),
[`5999-6015`](../../crates/memra-engine/src/lib.rs#L5999-L6015)). The nearby “block 64” comment is
stale; all three launch sites use 128. Each CTA owns one output row and its four warps collaborate
over K, followed by a two-level CTA reduction (`qmatvec.cu:5391-5430`). This is **block-per-row,
not warp-per-row**.

At K=4096 there are 128 32-value groups, so each of the 128 threads handles exactly one group.
Current upstream llama.cpp also chooses four warps and one output row per CTA for B=1/K=4096
([`calc_nwarps`](https://github.com/ggml-org/llama.cpp/blob/ebb546b7e961bd46fd9ed0387ffd14ca86b6fe1b/ggml/src/ggml-cuda/mmvq.cu#L354-L369),
[`calc_rows_per_block`](https://github.com/ggml-org/llama.cpp/blob/ebb546b7e961bd46fd9ed0387ffd14ca86b6fe1b/ggml/src/ggml-cuda/mmvq.cu#L460-L475),
[`calc_launch_params`](https://github.com/ggml-org/llama.cpp/blob/ebb546b7e961bd46fd9ed0387ffd14ca86b6fe1b/ggml/src/ggml-cuda/mmvq.cu#L773-L782)).
For IQ4_XS, K=4096 has exactly 16 256-value blocks, equal to—not less than—the four-warp
small-K threshold, so upstream also stays at one row/CTA
([`mmvq.cu:863-902`](https://github.com/ggml-org/llama.cpp/blob/ebb546b7e961bd46fd9ed0387ffd14ca86b6fe1b/ggml/src/ggml-cuda/mmvq.cu#L863-L902)).
There is no obvious K=4096 launch-geometry mismatch to copy.

Occupancy is not the primary limiter for the large grids. N=4096 reaches 74.2% achieved versus
83.3% theoretical occupancy with 2.18 waves/SM; N=8192–12288 has 4.36–6.54 waves/SM and about 74%
occupancy. Those shapes have enough work resident but still wait on long scoreboard. By contrast,
N=64/96 provides only 64/96 CTAs across 188 SMs (0.03/0.05 waves per SM), and N=1024/1280 remains
below one wave. No per-CTA tuning can make the 64/96 projections fill the device; they need fusion
with adjacent work or acceptance that they are launch/latency floor cases.

The memory walk is globally complete but instruction-level awkward. Consecutive lanes own
consecutive 32-value groups. At any one scalar `qs[k*4+i]` instruction, lane addresses are 16 bytes
apart within each 136-byte superblock pattern, so each instruction uses only a small part of the
sectors it requests. Across the fully unrolled loop all sectors are eventually consumed—hence no
DRAM over-fetch—but via many narrow requests and dependent table loads. Packed 32/64-bit loads
improve request/issue efficiency without reducing the HBM byte bill.

The **315 launches/token** are model topology, not proof that each launch has the wrong block size.
They do expose a second lever: Q/K/head-gate and FFN gate/up projections consume the same Q8 input
and can be dispatched together without changing any row's arithmetic. The prior Q8_0 dense
gate/up fusion removed 64 launches/token and measured +0.94% end-to-end on the 188-SM rig
([`q27 RESULTS.md:225-243`](../q27-deepdive-20260805/RESULTS.md#L225-L243)); the same arm was flat,
not harmful, on 82 SM ([`local5090/VERDICT.md:66-86`](../q27-deepdive-20260805/local5090/VERDICT.md#L66-L86)).

**Verdict for question 3:** keep the 128-thread one-row geometry for the first K=4096 arm and fix
the dot body. Treat tiny-N underfill, K=1280 row packing, and projection fusion as separate second
arms. Changing the main reduction geometry first would mix an exactness-class change into the
mechanism test.

## 4. FP4/FP8 and the frozen-byte boundary

There is no lower-byte FP4/FP8 path that preserves this Step checkpoint's exact values:

- IQ4_XS is **136/256 = 0.53125 B/weight (4.25 bpw)**.
- Memra NVFP4 is **36/64 = 0.5625 B/weight (4.5 bpw)**
  ([`qmatvec.cu:324-334`](../../crates/memra-engine/cu/qmatvec.cu#L324-L334)): **5.88% more weight
  bytes**, plus a different E2M1/UE4M3 value set and macro-scale semantics.
- Native FP8 is approximately 1 B/weight before metadata, almost twice IQ4_XS's weight traffic.
- FP4 activation packing would halve the Q8 activation buffer, but all Q8 activation reads are only
  1.702 MB—**0.059%** of the 2.879 GB qmatvec logical bill per token. Even an impossible zero-byte
  activation would not move the bandwidth ceiling materially.

Blackwell's published FP4/FP8 tensor peaks therefore do not create a B=1 bandwidth win by
themselves. Hardware FP4 consumes E2M1 values; it cannot interpret the frozen nonlinear IQ4_NL
codebook directly. Re-quantizing weights to NVFP4/FP8 changes artifact bytes and values. Quantizing
activations to FP4 changes the numerical program and saves essentially none of the dominant weight
traffic. Expanding IQ4_XS into int8/FP4 on load increases resident bytes and still has to decode the
same frozen stream.

A lossless split-plane or walk-order mirror could retain values, but it cannot reduce the 4.25-bpw
weight bill. The NCU byte reconstruction says there is no DRAM over-fetch for such a mirror to
remove; any benefit would be instruction addressing only. The packed-load rewrite gets that
benefit without a second resident representation.

The AgentWorld IQ4_XS receipts establish that IQ4_XS has full fast-path coverage and passes the
standard exactness gates ([`agentworld README.md:29-41`](../agentworld-iq4xs-20260802/README.md#L29-L41));
they do not contain a dense-qmatvec bandwidth mechanism measurement and are not used as one here.

**Verdict for question 4:** NVFP4/FP8 is `[changes-bytes]` and ineligible for the frozen Step lane.
There is no exact lower-byte escape hatch; optimization must issue the existing 136-byte blocks
more effectively.

## Ranked follow-up rewrite arms

1. **Dense wide-load + `byte_perm` dot** — `[exactness-preserving]` `[match-llama]`.
   Inline or reuse the in-tree `expert_dot_iq4xs_g_v` recipe in `qmatvec_iq4_XS_dp4a`: aligned
   64-bit header/quant loads, packed table words, `get_int_from_table_16_d`, unchanged low/high
   dp4a order, `acc +=` order, and CTA reduction. Keep an alignment fallback. This directly removes
   the only clear source-level mismatch with llama, targets both long scoreboard and LG throttle,
   and has the +2.5% related-path prior. **First arm; do not change launch geometry with it.**

2. **Load-only software pipeline for K>4096** — `[exactness-preserving]` `[exceed]`.
   Once arm 1 is isolated, prefetch the next IQ4 header/quant words and Q8 scale while computing the
   current group; preserve arithmetic issue order. K=8192/11264/12288 gives each thread 2–3 groups,
   so this can add memory-level parallelism to the dominant 4096-row attention/dense-down calls.
   Risk: the present kernel already uses 43 registers/thread and has 83.3% theoretical occupancy;
   extra live state can lower occupancy enough to lose. Require SASS/register and NCU proof.

3. **Same-input projection fusion** — `[exactness-preserving]` `[exceed]`.
   Fuse Q/K/head-gate (three→one launch) and dense/shared gate+up (two→one) while executing the arm-1
   row body verbatim. Across 45 layers this can remove up to **135 of 315** IQ4 kernel launches and
   places tiny 64/96/1024-row work beside a large projection. It does not reduce weight bytes or fix
   large-shape bandwidth, so rank it behind the body rewrite despite the positive Q8 fusion prior.

4. **K=1280 multi-row CTA specialization** — `[exactness-preserving]` `[match-llama]`.
   Upstream's small-K policy packs multiple output rows when fewer than 16 IQ4_XS 256-value blocks
   are present; K=1280 has five. A memra version can carry several independent row accumulators while
   retaining each row's existing g mapping and reduction tree. It targets the 42 shared-down calls
   and wasted K lanes, not the K=4096 trunk. Measure separately: fewer CTAs and higher register use
   can outweigh activation reuse.

5. **Lossless split-plane/reordered mirror** — `[exactness-preserving]` `[exceed]`.
   Low priority/likely NO-GO. It preserves values but adds load-time work and resident VRAM while
   saving no HBM bytes; the counter data already show one complete weight stream. Reconsider only
   if arm 1's SASS still contains address-generation/narrow-load pressure that cannot be removed in
   the native layout.

6. **NVFP4/FP8/W4A4 conversion** — `[changes-bytes]` `[exceed]`.
   **Reject for this lane.** It violates frozen-model exactness, NVFP4 actually increases weight
   bytes by 5.88%, FP8 increases them much more, and FP4 activation savings are immaterial at B=1.

## Follow-up gate and explicit GO/NO-GO threshold

**GO** arm 1 with a target of at least **+10% relative qmatvec effective bandwidth** at identical
bytes (pair aggregate 1,015→at least 1,117 GB/s, or 2.836→at most 2.578 ms/token). That would save
about 0.258 ms/token and imply roughly **+2.8% end-to-end throughput** before interactions. It is a
single-digit percentage-point card-BW recovery (63.6→69.9%) and a double-digit relative kernel
recovery.

Promotion requires, in repository order:

1. byte-for-byte kernel output comparison to the current kernel at every observed `(out_f,in_f)`,
   including the alignment fallback;
2. `kernel-check` ALL GREEN, `run-gen` argmax MATCH, and `run-spec` K=1..8 self-consistency PASS;
3. same-lock, interleaved Nsys measurements with N≥32 per arm and raw logs retained; and
4. per-shape NCU showing lower long-scoreboard/LG pressure without higher DRAM bytes, followed by
   the designated 5090 and 2× PRO 6000 battery required by project doctrine.

If arm 1 is below a single-digit relative bandwidth gain after the packed loads are verified in
SASS, that arm is a **NO-GO for promotion**, not permission to change bytes. Proceed to the
load-only pipeline, fusion, and K=1280 arms independently. The true family ceiling remains the
1.80 ms/token HBM floor; no frozen-byte rewrite can exceed it.

## Reproduction and current external sources

Run `python3 research/qmatvec-study-20260811/analyze.py` from the repository root. It reads the
committed ncuspike `summary.json`, asserts the 154+161 launch map, and prints every roofline,
DRAM/logical, runtime-rate, and ceiling number used above.

Fast-moving external facts were refreshed on 2026-08-11:

- NVIDIA RTX PRO 6000 Blackwell Server Edition specifications:
  [1,597 GB/s, 120 FP32 TFLOP/s, and FP4/FP8 peaks](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/).
- Current llama.cpp source pinned for this comparison: commit
  [`ebb546b7e961bd46fd9ed0387ffd14ca86b6fe1b`](https://github.com/ggml-org/llama.cpp/commit/ebb546b7e961bd46fd9ed0387ffd14ca86b6fe1b).
- NVIDIA Nsight Compute 13.3 profiling guide:
  [memory requests, sectors, long scoreboard, and LG throttle](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
