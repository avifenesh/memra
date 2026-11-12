# Performance

Measured numbers with their conditions, and how they are taken. For readers who want a number
and the hardware it came from, and for maintainers checking a regression. Raw runs live in
[`research/`](../research/INDEX.md); every number here names its card, N and date.

Ratios against other engines appear only on this page and in `research/`, next to their
protocol. The two generated boards are frozen: the RTX 5090 Laptop board was last measured on
2026-08-02 and the H100 board on 2026-08-01. Neither covers the release roster (Ornith-1.5-35B-A3B,
Qwen3.8-27B), GB10 or the RTX PRO 6000 Max-Q; their numbers are under
[Current per-device evidence](#current-per-device-evidence).

## Method

- Every published median states its N and thermal regime. A single run is labeled as one and
  never headlines.
- A performance or default decision is a same-window interleaved A/B in both orders: three runs
  per arm by default, five on an anomaly (LAW:interleaved-ab). Cross-run and cross-day
  comparisons are invalid because clocks drift, competitor denominators included.
- Locked and free clocks never mix in one comparison. On 2026-08-06 unlocked clocks drifted 9%
  in-process, and 1860 MHz-locked numbers read below the 3090 MHz free-clock boost.
- Scored runs carry 250 ms telemetry. Serving claims also record TTFT, E2E, TPOT and ITL
  p50/p95/p99 plus request and token throughput.
- TTFT is labeled cold or warm. An unsalted repeat hits the prefix cache, so a cold number needs a
  fresh `cache_salt` per request with `cached_tokens=0` checked.
- One engine on the GPU at a time: a co-resident VRAM user forces spill and reads low. One scored
  campaign per multi-card box, under `flock /tmp/memra-gpu.lock`, because the cards of a pair share
  a PCIe path.
- **Rig law.** The RTX 5090 Laptop throttles: correctness gates on it are valid, its timing
  numbers are never a serving receipt. They are 5090 development and regression numbers. A
  correctness check is valid on any CUDA device; a kernel timing is serving-relevant only on the
  card class the kernel targets.
- The push gate's perf cells are `run-gen` decode A/B runs, A B B A A B per roster model,
  REGRESSED at complete separation plus a 2% median drop
  ([Release qualification](RELEASE-QUALIFICATION.md#change-impact-and-per-cell-evidence)).

The window rules and the false readings behind each one are in
[research/benchmarks.md](../research/benchmarks.md). Competitor engine builds and flags are in
[Competitor setup](COMPETITOR-SETUP.md).

## Rigs: what was measured on what

<a id="rigs--what-was-measured-on-what"></a>

Every number in this repo belongs to one rig. A cell moved to another rig is a different number:
the same kernel is 5-11% apart between two pod classes of one RTX PRO 6000 SKU, and about 2x apart
between a 188-SM and an 82-SM card. A rig label names the hardware and its measurement conditions,
nothing else. Labels that once named a rental provider are shortened here (`pro6000wk`,
`maxq-20260810`); the receipt files keep their original names.

| Rig label | Hardware | What it produced |
|---|---|---|
| RTX 5090 Laptop | GB203, 82 SM, 24 GB, 858 GB/s measured read | Development and single-card tuning; the 5090 board; `rtx5090` cell and daily-regression records |
| RTX PRO 6000 Max-Q | RTX PRO 6000 Blackwell Max-Q Workstation Edition, 96 GB, sm_120a; power limit recorded per record | Tag records since v0.139.0, one card per capture; `rtxpro6000` cell records |
| GB10 | DGX Spark, 48 SM, unified memory, sm_121a | `gb10` cell and daily-regression records; the GB10 decision records |
| `pro6000wk` | RTX PRO 6000 Blackwell Workstation, 96 GB, 188 SM, 600 W, clocks pinned at 2865 MHz | The 2026-08-04 27B board, including pp512 4118 tok/s on Qwen3.6-27B NVFP4 (N=5); a second pod class of the SKU ran 5-11% slower, so compare deltas within one pod only |
| `pro6000wk-pair` | 2x RTX PRO 6000 Blackwell Workstation, 96 GB each, 600 W, PCIe 5.0, native peer copies (probed clean) | Two-card receipts on the pair shape |
| `pro6000sv-pair` | 2x RTX PRO 6000 Blackwell Server Edition, 96 GB each, CUDA 13.2, Gen5 x16 P2P | The 2026-08 PP-2 lanes, Step-3.7-Flash bring-up, pre-release batteries ([coldfix](../research/coldfix-20260812/), [b1fix](../research/b1fix-20260810/)) |
| `maxq-20260810` | 2x RTX PRO 6000 Blackwell Max-Q Workstation, 96 GB each, 300 W, CUDA 13.1; PP-2 over host bounce | 2026-08 Max-Q serving numbers only, never a regression baseline ([27bab](../research/27bab-20260810/)) |
| `rig2x5090-serve` | 2x RTX 5090 32 GB, one card used | 2026-08 official-FP8 and multi-card receipts |
| H100 80 GB | 1x, 3x and 8x HBM3 | The H100 board, the replica fleet, QoS lanes, endurance |
| B200 | 1x and 2x NVIDIA B200, sm_100a | B200 kernel twins and GLM-5.3-Flash TP-2 receipts ([B200](rigs/b200.md)) |
| `rtx6000` | RTX PRO 6000 Server Edition | Hy3 spill and quantization research ([Hy3 spill](HY3-SPILL.md), `research/tune-data/cloud-rtx6000.jsonl`) |

## How the boards are generated

The `PERF-PLAIN`, `PERF-SPEC`, `PERF-DATE` and `PERF-H100` blocks below, and `PERF-MODELS` in
[Models](MODELS.md), are generated from
[`research/tune-data/current-board.json`](../research/tune-data/current-board.json) by
`tools/update-perf-board.py`. Edit the JSON and run `python3 tools/update-perf-board.py`; never
hand-edit between the markers. `python3 tools/update-perf-board.py --check` exits 1 on drift, and
the pre-push hook runs it. 5090 ratios are computed from the display-rounded values in the JSON;
H100 rows carry their receipted ratio. The JSON also holds rows the script does not render:
`plain_decode_depth` (the 6,257-token rows) and `extra_card_rows` (eight Gemma-4 rows). The
hand-written Gemma-4 table with its frozen llama.cpp columns is in the
[history](history/PERFORMANCE.md#gemma-4-qat-q4_0).
`research/tune-data/rig5090.jsonl` is the append-only research log.

## RTX 5090 Laptop board (2026-08-02)

Single-user decode on one RTX 5090 Laptop, memra and llama.cpp interleaved on the same rig with
the same prompts. The llama.cpp columns are frozen reference denominators recorded through
2026-08-03: regression anchors, not a live scoreboard, never refreshed in place. Builds and flags
for both columns: [Competitor setup](COMPETITOR-SETUP.md#frozen-baselines). None of these three
models is on today's release roster.

**Plain decode** (no speculation, tg128 at 512-token context):

<!-- PERF-PLAIN:START (generated by tools/update-perf-board.py; do not hand-edit, edit research/tune-data/current-board.json instead) -->
| Model | memra plain | llama.cpp plain | Ratio |
|---|---|---|---|
| Qwen3.5-9B NVFP4 (GGUF) | 137.3 | 121.2 | **1.13x** |
| Qwen3.6-27B NVFP4 (GGUF) | 47.6 | 43.7 | **1.09x** |
| Qwen3.6-35B-A3B MoE (IQ4_XS) | 187.0 | 164.9 | **1.13x** |
<!-- PERF-PLAIN:END -->

**Speculative decoding** (MTP head, both engines at their measured best; columns are the short
code, medium code and long agentic prompt classes):

<!-- PERF-SPEC:START (generated by tools/update-perf-board.py; do not hand-edit, edit research/tune-data/current-board.json instead) -->
| Model | memra spec | llama.cpp spec-best | Ratio |
|---|---|---|---|
| Qwen3.5-9B (K=3 + own-gen trimmed draft) | 281.0 / 211.7 / 187.1 | 122.2 / 121.5 / 117.7 | **2.30x** / **1.74x** / **1.59x** |
| Qwen3.6-27B (K=3 + own-gen trimmed draft) | 116.4 / 101.2 / 86.0 | 91.7 / 93.3 / 81.5 | **1.27x** / **1.08x** / **1.06x** |
| Qwen3.6-35B-A3B (K=2 + own-gen trimmed draft) | 302.4 / 253.0 / 270.7 | 234.7 / 207.2 / 235.8 | **1.29x** / **1.22x** / **1.15x** |
<!-- PERF-SPEC:END -->

<details><summary>Measurement provenance</summary>

<!-- PERF-DATE:START (generated by tools/update-perf-board.py; do not hand-edit, edit research/tune-data/current-board.json instead) -->
Measured 2026-08-02 on the tracked measuring rig (RTX 5090 Laptop) against llama.cpp built on the same machine, same exact prompts, both engines re-baselined the same day. Protocol: N=2+ medians, both engines interleaved in the same thermal window on the same rig, same exact prompts, no flags (the measured fast paths are the defaults); plain and depth rows re-measured 2026-08-02 after the deep fa_decode merge (d14d7d8d) as N=5 same-session interleaved medians with a fresh llama denominator (research/board-remeasure-20260802/); spec rows re-paired 2026-07-18, the 35B spec row re-paired 2026-08-02 under the mode-2 grouped-f16 naked default (sk visitor plus direct expert tiles) as N=3 medians (research/f16g-default-rearb-20260802/); Gemma card rows from the 2026-07-15 best-vs-best re-audit, the Gemma 12B plain row from the 2026-07-24 official N=5 cell stamp (research/gemma4-bringup/g12tg-cellstamp.log). Full per-run logs, every win and every loss: research/tune-data/ (Qwen) and research/gemma4-bringup/ (Gemma). The llama.cpp columns are frozen reference points recorded through 2026-08-03: regression anchors, not a live scoreboard. `research/tune-data/rig5090.jsonl` is the running record; a change that moves a published number updates `research/tune-data/current-board.json` and regenerates these blocks.
<!-- PERF-DATE:END -->

</details>

- **Depth.** At 6,257-token context every plain lead holds at 1.11-1.13x (`plain_decode_depth`,
  N=5 interleaved, [research/board-remeasure-20260802/](../research/board-remeasure-20260802/)).
- **Spec classes.** Short code and medium code are greedy; long agentic is sampled at temperature
  0.7 with distribution-exact rejection sampling, llama.cpp at the matched temperature. The 35B
  short-code llama.cpp bar is an EOS-suppressed continuation (the raw prompt ends at 1 token), not a
  clean win basis. Every spec row uses one trimmed draft built by the
  [draft regime](DRAFT-REGIME.md); prebuilt drafts are in the
  [bench repo](https://huggingface.co/tiyuvta/memra-bench).
- **Prefill** trails llama.cpp on this card; see [Open gaps](#open-gaps).

## H100 board (2026-08-01)

One H100 80 GB, memra against vLLM 0.26, frozen 2026-08-01. The Hopper lane was retired on
2026-09-02: `sm_90a` builds from source only, and the board has not been re-measured since. Its
promoted configurations and every refutation are in the Hopper lane ledger,
[ARCHITECTURE-H100.md](../ARCHITECTURE-H100.md); its status is on the [H100 card](rigs/h100.md).

<!-- PERF-H100:START (generated by tools/update-perf-board.py; do not hand-edit, edit research/tune-data/current-board.json instead) -->
Measured 2026-08-01 on 1× H100 80GB against vLLM 0.26: same box, every cell a same-session interleaved pair. One number per arm: end-to-end tok/s for 512 tokens generated on a ~2100-token real-text prompt, single request, total wall time; N=5 medians, argmax exactness gate green on every published row. Cross-artifact by design: vLLM runs the HF checkpoints H100 users deploy (w8a8, FP8-dynamic or bf16; it rejects these GGUFs), and memra runs its GGUF artifacts.

| Model | memra e2e | vLLM 0.26 e2e (artifact) | Ratio |
|---|---:|---:|---:|
| Gemma-4 12B | **146** | 81 (bf16) | **1.81x** |
| Qwen3.6-27B | **96** | 73 (FP8) | **1.31x** |
| Gemma-4 31B | **75** | 64 (FP8-dyn) | **1.18x** |
| Qwen3.5-9B | **204** | 176 (w8a8) | **1.16x** |
| Gemma-4 E4B | **193** | 168 (bf16) | **1.14x** |
| Qwen3.6-35B MoE | **226** | 215 (FP8) | **1.05x** |
| Gemma-4 26B MoE | **196** | 191 (FP8-dyn) | **1.02x** |
<!-- PERF-H100:END -->

The bf16 rows carry a quant advantage: those vLLM arms move about 4x the weight bytes. The board
has no end-to-end loss because decode wins 7 of 7; per-cell prefill still trails vLLM on the dense
models, where a Q8_0-exact int8 GEMM is mechanism-refuted on Hopper.

## Current per-device evidence

The release roster's numbers on today's code live with the evidence that gates it:

- **Push-gate perf records**: the `perf/` directory of
  `research/release-qualification/cells/<commit>-<device>-*perf*/`, a `run-gen` decode A/B per
  roster model (Ornith-1.5-35B-A3B, Qwen3.8-27B) on `rtx5090`, `gb10` or `rtxpro6000`. The
  `*-perf-conditions` records hold the thermal and power state of a run.
- **Daily regression**: the `regression/<device>` branches, under
  `research/daily-regression/<device>/`
  ([Release qualification](RELEASE-QUALIFICATION.md#daily-regression)).
- **Tag records and qualification notes**: `research/release-qualification/release-v*/` and
  `research/release-vX.Y.Z-*/README.md`, with the box conditions of each capture.
- **Default decisions with their A/B runs**: [docs/decisions/](decisions/README.md), for example
  the GB10 defaults (`GB10-*`), the cold speculative depth
  ([SPEC-K-COLD-LONG-K2](decisions/SPEC-K-COLD-LONG-K2.md),
  [SPEC-K-COLD-SHORT-5090](decisions/SPEC-K-COLD-SHORT-5090.md)) and the attention rungs
  ([FA-SPLIT-DEEP-RUNG](decisions/FA-SPLIT-DEEP-RUNG.md), [FA-VERIFY-ROWS](decisions/FA-VERIFY-ROWS.md)).
- **Model cards** cite their numbers with card, N and date: [Qwen3.8-27B](models/qwen38-27b.md),
  [Ornith-1.5-35B-A3B](models/ornith15-35b-a3b.md) and the others under [models](models/README.md).
- **Research lanes**: one verdict line per lane in [research/INDEX.md](../research/INDEX.md).

## Serving measurements

### Prefix-cache depth is the dominant serving lever

2026-08-13, one RTX PRO 6000 Blackwell Workstation, Qwen3.6-35B-A3B UD-IQ4_XS,
`MEMRA_PREFIX_CACHE_MB=49152`, experts resident, `MEMRA_SERVE_SPEC=0`. Shape: a 4,860-token
prompt plus 60 output tokens with a 90% shared prefix. N=3 rounds per width, medians. Raw JSON and
the harness: [research/canonflip-20260813/](../research/canonflip-20260813/). The only variable
between the arms is the depth of the resident prefix entry:

| arm | cache hit | c=8 req/s | c=16 req/s | c=16 p50 | c=16 out tok/s |
|---|--:|--:|--:|--:|--:|
| shallow entry covers the class | 39.6% | 2.57 | 2.72 | 5.84 s | 163.5 |
| entry covers the full prefix | 99.5% | 6.84 | **8.50** | **1.87 s** | 510.0 |

Cache depth alone moved throughput and latency 3.1x. The cause was a depth freeze, visible in the
LCP histogram (189 hits in the 1024-2048 bucket, none in 2048-4096): a shorter request had seeded
a 1,937-token entry for the class, and a hit never created a deeper one. It was not capacity
pressure: 5 entries and 423 MB of a 49 GiB budget were resident, with zero evictions. The seed
path has changed since: an entry deepens when the session's primed state is at least
`prefix_seed_deepen_min()` tokens deeper than the deepest covering entry (`maybe_prefix_seed` in
`crates/memra-server/src/worker.rs`; analysis in
[research/cacheinval-20260813/](../research/cacheinval-20260813/), H11). Measure hit depth, not
only hit rate: a high hit count against a shallow entry looks healthy while each request still pays
most of its prefill.

### Speculative decoding and the prefix cache

Same card, same fresh cache, same shape, spec the only variable:

| arm | cache hit | c=16 req/s | c=16 p50 |
|---|--:|--:|--:|
| `MEMRA_SERVE_SPEC=0` | 99.5% | 8.50 | 1.87 s |
| `MEMRA_SERVE_SPEC=1` | 18.4% | 2.14 | 7.40 s |

Spec is numerically exact here (the content and reasoning SHA-256 match across arms), so this is a
scheduling result. Offline, `run-spec` on the same artifact reports 1.51-1.70x over plain at
K=4..8 with 51.8-80.0% acceptance, because a CLI run has no cross-request cache to give up. On
shared-prefix serving shapes the prefix cache is worth more than the decode win, and
`MEMRA_SERVE_SPEC=0` is the measured setting for them; spec fits cache-poor shapes. The engine
default is spec on ([flags](FLAGS.md)).

### Concurrency envelope

On that card, throughput saturated near 9.9 req/s on a short-prompt shape and near 8.5 req/s on
the shared-prefix shape. Past c=8, latency grew almost linearly for little gain: c=8 is the
efficiency knee, and c=16 the widest width that held p50 under 1.8 s with a healthy cache. A
per-key `rate_limit` of 16 protected latency: at c=24 the endpoint returned 16 HTTP 200 and 8 clean
HTTP 429, with p50 unchanged.

### Batched decode, replicas and endurance

- Batched decode runs one z-batched tick across sequences, 16 wide on models with a bit-exact
  16-batch kernel class: +18.8% at c=16 on one RTX 5090 Laptop replica (same-mirror interleaved,
  N=4), and 654-659 tok/s per replica on the H100 serving tick (+25-36%).
- The 2026-08 serve gate replayed the same prompts at c=1 and c=16 and byte-compared every stream:
  greedy serving was isolated-identical under concurrent load at the defaults (serve against
  serve, not against a tokenwise oracle).
- A model that fits one card serves from a multi-GPU box as a replica fleet: 1,477 tok/s on 3
  H100s. A replica killed mid-load recovered in seconds, losing only its in-flight requests.
- Endurance: 140 minutes at c=96 on 8 H100s, 464,870 requests, 0 errors, 0 sheds, +0.045%
  throughput drift. That is Qwen3.5-9B-Q8_0, warm prefix, temperature 0.7 seeded, with a greedy
  probe hashing identical on all 8 replicas before and after.
- The boards above are bare-CLI numbers, not serve-path numbers. Mechanism and runbook:
  [Serving](SERVING.md).

### Pipeline parallel (PP-2)

PP-2 exists for capacity, not throughput: it lets a model that fits only across a pair serve at
all. On 2026-08-06, on 2x RTX PRO 6000 Blackwell Server Edition with Qwen3.5-9B (N=5 interleaved in
one lock hold), the batched split cost 0.995x, 0.989x and 0.986x at B=4, 8 and 16 against one
card, almost all of it the stage seam, not PCIe
([research/pp2-batch-20260806/](../research/pp2-batch-20260806/)). The dual-active default (two
independent batch waves, `MEMRA_DUAL_PP` unset) measured 158.065 aggregate tok/s against 133.553
for the serial rollback at c=8 (+18.354%, N=5 per arm, 2026-08-11,
[research/dualpp-flip-20260811/](../research/dualpp-flip-20260811/)). The placement-aware spec gate
keeps PP-2 on plain decode. Mechanism: [Serving](SERVING.md#pipeline-parallel-pp-2-serving).

## Open gaps

- **Prefill on the RTX 5090 Laptop** trails llama.cpp (0.59-0.78x on the board's models).
  llama.cpp runs NVFP4 prefill at W4A4 (FP4 activations), a numeric class memra's exactness gates
  reject, but that does not close the gap: on 2026-08-05, on an identical model file, 4k prefill
  measured 1.2k against 2.1k tok/s.
- **Interactive latency.** On 2026-08-05 (RTX 5090 Laptop, same model file, each engine at its
  everyday configuration, N=5 interleaved) llama.cpp led cold TTFT (0.19 s against 0.53 s), short
  agentic turns and raw prefill, and memra led long sampled decode by 17%
  ([research/memra-vs-llama-daily-20260805/](../research/memra-vs-llama-daily-20260805/)). memra's
  own first-text latency has moved since (round-cadence SSE, solo 0.41 to 0.12 s; admission yield,
  contended 1.60 to 0.15 s; [sse-cadence](../research/sse-cadence-20260805/),
  [admission](../research/admission-20260806/)), but was not re-paired against a competitor. memra
  makes no interactive-latency superiority claim.
- **Cold prime under concurrency.** On 2026-09-01 (RTX PRO 6000 Blackwell Server Edition,
  `unsloth/Qwen3.8-27B-NVFP4`, spec off), rebuilding an evicted context at N32 (32 sessions,
  0.93x of the vLLM device pool) measured memra cold TTFT p50 26.4-30.5 s against vLLM 0.28.0 at
  6.8 s (p95 11.2 s). The raw runs are not in this repository, and the split between queue wait and
  prime compute is not attributed yet.
- **NVFP4 speculative serve path**: 170.55 tok/s through the server against 186.72 bare (-8.66%,
  `pro6000wk`, 2026-08-04). The spec burst loop is a separate path from the plain fast path;
  not re-measured since.
- **External baseline.** The doctrine since 2026-09-12 is one external cell per served shape per
  month ([research/benchmarks.md](../research/benchmarks.md)); none has been banked yet.

## Other measured decisions

- **MTP learned depth** (2026-09-20, one RTX 5090 32 GB, 575 W): a learned depth controller lost
  2.45% against fixed K=3 on Qwen3.8-27B and gained 7.74% against fixed K=5 on Gemma 4 12B, but
  only 0.43% over native adaptation; not promoted
  ([decision](decisions/MTP-LEARNED-DEPTH-20260920.md)).
- **Release profile** (2026-11-02, RTX 5090 Laptop): fat LTO with one codegen unit cut
  `/v1/tokenize` on a 165,525-token chat by 6.3%, left TTFT, E2E and streaming flat, kept every
  output byte-identical and made `memra-server` 17% smaller ([decision](decisions/RELEASE-PROFILE.md)).
- **Ornith-1.5 pair-owner order** (2026-08-25): bit-exact across B=2..16, rejected on timing (4.5%
  to 6.8% slower on RTX PRO 6000, 5.3% on the RTX 5090 Laptop)
  ([decision](decisions/ORNITH-PAIR-OWNER.md)).

## History

<a id="bring-up-notes"></a>

The previous text of this page is in [history/PERFORMANCE.md](history/PERFORMANCE.md): the
2026-08-04 `pro6000wk` 27B board and the official-FP8 cell, the 2026-08-05 standing summary, the
serving exactness arc, the Gemma-4, Ornith-1.0 and AgentWorld end-to-end tables, the H100 flip
history and sm_90a feature list, the cold-prime gap analysis, the PP-2 and dual-PP cells, the IQ4_XS
qmatvec promotion, and the bring-up notes for KAT-Coder, Hy3, MiniMax-M3 and Step-3.7-Flash. Hopper
history is in [ARCHITECTURE-H100.md](../ARCHITECTURE-H100.md).
