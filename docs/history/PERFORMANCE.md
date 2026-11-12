# PERFORMANCE history

Records moved out of [docs/PERFORMANCE.md](../PERFORMANCE.md) and [docs/COMPETITOR-SETUP.md](../COMPETITOR-SETUP.md) on 2026-11-12, verbatim except for the removal
of private deployment detail: which box served what is dropped, and rig labels that named a rental
provider are shortened (`pro6000wk`, `maxq-20260810`; the receipt files keep their original names).
The current contract is in the main doc; this file is not.

## Opening notes of docs/PERFORMANCE.md (as of 2026-11-11)

This document carries the full tracked boards (generated from
[`research/tune-data/current-board.json`](../../research/tune-data/current-board.json) by
[`tools/update-perf-board.py`](../../tools/update-perf-board.py) — the README shows only a
few representative samples) and the depth behind them: how each cell moved, what was
refuted along the way, and where the raw runs live. The boards are a regression suite —
board-moving merges re-measure the tracked cells and regenerate every derived surface.
The measurement protocol is [`research/benchmarks.md`](../../research/benchmarks.md);
competitor engines run at their swept best
([`docs/COMPETITOR-SETUP.md`](../COMPETITOR-SETUP.md)); the H100 lane's append-only evidence
ledger — every promoted config, every mechanism refutation — is
[`ARCHITECTURE-H100.md`](../../ARCHITECTURE-H100.md).

Every published median states its N and thermal regime; every perf claim is a same-session
interleaved pair (cross-run and cross-day comparisons are clock-drift-invalid, including the
competitor denominator). Locked-clock corollary (measured 2026-08-06, lane/prefill-gemm:
unlocked clocks drift 9% in-process, and 1860 MHz-locked absolute numbers read BELOW the
3090 MHz free-clock boost): a clock-locked value is the only valid A/B denominator, and
locked and free-clock numbers must never be mixed in one comparison.

> **Competitor benching RESUMES as a monthly external baseline (owner call, 2026-09-12).**
> The 2026-08-03 stop is lifted. Every llama.cpp and vLLM column in this document is still a
> **frozen reference point** recorded on or before 2026-08-03 and stays labeled that way; frozen
> columns are never refreshed in place. From 2026-09-12 one external cell per served shape is
> measured once a month against the current pinned SGLang (primary) and vLLM (control) at their
> documented best configuration for the model and card, same-session interleaved with the current
> memra binary, and banked under `research/external-baseline-YYYYMM/` with commit, container
> digest and flags. A ratio in this file is context; the newest monthly cell is the verdict.
> Self-competition (memra vs its own previous cells) continues for kernel work and is not a
> substitute for the monthly cell. The doctrine banner also lives in
> [`research/benchmarks.md`](../../research/benchmarks.md).
>
> The 2026-08-05 head-to-head stays as recorded: same model file, each engine at its owner's daily
> config, N=5 interleaved, **llama.cpp leads on cold time-to-first-token (0.19 s vs 0.53 s), short
> agentic turns, and raw prefill**, while memra leads long-generation sampled decode by +17%
> ([`research/memra-vs-llama-daily-20260805/`](../../research/memra-vs-llama-daily-20260805/)).
> memra makes **no interactive-latency superiority claim** while that stands. The memra-side
> latency moves since then (round-cadence SSE solo first text 0.41 to 0.12 s, admission-yield
> contended first text 1.60 to 0.15 s;
> [`research/sse-cadence-20260805/`](../../research/sse-cadence-20260805/),
> [`research/admission-20260806/`](../../research/admission-20260806/)) are self-competition
> receipts that were never re-paired against a competitor; the first monthly cell is where that
> pairing happens.

> **Rig labels are load-bearing.** The generated tracked boards remain **RTX 5090 Laptop**
> and **H100 80 GB** receipts. The 5090 board is the development and single-card
> regression board; its existing cells stay labeled 5090. RTX PRO 6000 cells and pair
> receipts live outside those generated boards with their own rig labels, because mixing a
> 188-SM cell with an 82-SM cell is a 2x-class error. The registry of every rig that produced
> a number in this repo is [Rigs](#rigs--what-was-measured-on-what) directly below.

## Rigs — what was measured on what

Every number in this repo belongs to exactly one of these. A cell moved to another rig is a
different number, not the same number re-measured: the same kernel is 5-12% apart between the
two PRO 6000 pod classes and roughly 2x apart between a 188-SM and an 82-SM board.

| Rig label | Hardware | What it produces |
|---|---|---|
| **RTX 5090 Laptop** | GB203, 82 SM, 858 GB/s | Development and single-card tuning; the tracked 5090 regression board and development-iteration battery |
| `pro6000wk` | RTX PRO 6000 Blackwell Workstation 96 GB, 188 SM, 600 W, clocks pinned 2865 MHz, zero throttle | Historical 27B cells below; receipt label only |
| `pro6000wk-community` | Same SKU, a second pod class | Historical iteration receipts. Runs **5-11% slower** than the standard pod — compare only relative deltas within one pod |
| `rig2x5090-serve` | 2x RTX 5090 32 GB (single card used) | Historical official-FP8 and multi-card receipts; not the current target shape |
| `pro6000wk-pair` | 2x RTX PRO 6000 Blackwell Workstation Edition, 96 GB each, full 600 W, PCIe 5.0, driver 610.57.04; peer path byte-probed **CLEAN** — native `cudaMemcpyPeerAsync`, no host bounce | Two-card receipts on the pair shape |
| `maxq-20260810` (historical) | 2x RTX PRO 6000 Blackwell Max-Q Workstation Edition, 96 GB each, 300 W, CUDA 13.1; PP-2 used host bounce | Historical/Max-Q serving numbers only; never a current regression baseline (`research/27bab-20260810/`) |
| `pro6000sv-pair` | 2x RTX PRO 6000 Blackwell Server Edition 96 GB, sm_120a, CUDA 13.2, Gen5 x16 P2P | Retired. Pre-merge and pre-release battery receipts (`research/coldfix-20260812/`, `research/b1fix-20260810/`); source of the PP-2 lanes (batched decode, spec verdict, hardening) and Step-3.7-Flash bring-up. GPU windows held under `flock`, both cards verified at 0 MiB on entry and exit |
| H100 80 GB | 8x / 3x / 1x HBM3 pods | The H100 board, the replica fleet, QoS lanes, endurance |
| `rtx6000` (receipts `research/tune-data/cloud-rtx6000.jsonl`) | PRO 6000 Server Edition | Hy3 spill / quantization research (`docs/HY3-SPILL.md`) |

## Step-3.7 PP-2 receipts on 2x RTX PRO 6000 Workstation Edition, 2026-08-11 (untitled in the source)

*Moved 2026-11-12 from the rig notes of docs/PERFORMANCE.md, with the deployment wording of the
source (which box served what) left out.*

[`research/newbox-bench-20260811/RESULTS.md`](../../research/newbox-bench-20260811/RESULTS.md)
(provisional single runs) and [`research/newboxgates-20260811/RESULTS.md`](../../research/newboxgates-20260811/RESULTS.md)
(N=5 medians: short TTFT 0.136 s, 4k cold TTFT 5.777 s, decode 98.30/158.45/173.62 tok/s at
c=1/4/8; ring capacity 2 to 12). The earlier Max-Q/300 W pair receipts are in
`research/27bab-20260810/`; those numbers are Max-Q only and are not regression denominators for
the Workstation Edition pair.

## Historical 27B board — `pro6000wk` receipt (RTX PRO 6000 Blackwell 96 GB)

Rig `pro6000wk`, date **2026-08-04**, commit `2299ee0f`, temp max 43 C, zero throttle.
Two artifacts, both Qwen3.6-27B, interleaved in one session: **nv** =
`Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf` (15.7 GB), **q8** = `Qwen3.6-27B-Q8_0.gguf` (28.6 GB).
Receipts: [`research/pro6000-prod-20260804/`](../../research/pro6000-prod-20260804/), journal
`pro6000wk-runpod.jsonl`.

| Cell | Value | Protocol |
|---|---|---|
| Spec decode (MTP) K=3, nv, **bare CLI** | **186.7 tok/s** | N=5 process reps, median. 2.17x the same-run plain 86.20 |
| Spec decode K=3, nv, **through the serve surface** at c=1 | **170.5 tok/s** | N=5 median, server restarted per K, 0 err / 0 shed |
| Plain decode tg128, nv / q8 | 86.8 / 52.6 tok/s | N=5 medians |
| Aggregate at c=8, nv / q8 | **420.6** / 308.7 tok/s | N=3 passes, median; p50 2.42 s, 64 ok / 0 shed / 0 err per pass. Spec off, plain batched serve |
| TTFT cold, nv / q8 | **0.182** / 0.156 s | N=5 with per-rep `cache_salt`; median of reps 2-5, rep 1 excluded as one-time session warmup |
| TTFT warm (prefix hit), nv / q8 | 0.003 / 0.004 s | reps 2-5; 61x the cold number |
| Prefill pp512, nv / q8 | 4118 / 4591 tok/s | N=5, arms interleaved within every rep |
| Q8 96 GB residency lever | +57% agg at c=16/32 (486 vs 310 tok/s, p50 6.61 → 4.21 s), 63.7 GB resident | `q8rp/` |

Caveats that travel with these cells:

- **c=8 is the knee; saturation is not a win.** c8 420.6 → c16 421.9 → c32 423.0 is flat
  *while p50 doubles at every step* (2.43 → 4.84 → 9.67 s). The journal's own word for
  c16/c32 is "queueing, not throughput." c=32 is not a throughput ceiling.
- **The TTFT protocol trap.** An unsalted repeat request hits the prefix cache, so a TTFT
  number measured without a fresh `cache_salt` is a warm number wearing a cold label. Cold
  and warm are always stated separately.
- **These TTFT rows predate the felt-latency fixes** (round-cadence SSE 2026-08-05 +
  admission yield 2026-08-06). The rows above measure time to the first *token* on a plain
  request and stand as recorded; what changed is first *streamed text* on the spec tier —
  solo 0.41 → **0.12 s**, contended 1.60 → **0.15 s** at any `MEMRA_SPEC_BURST`, measured
  N=5 on the local 82-SM rig (27B NVFP4+MTP;
  [`research/sse-cadence-20260805/`](../../research/sse-cadence-20260805/),
  [`research/admission-20260806/`](../../research/admission-20260806/)). This board's rig has
  not re-measured them; do not mix the two rigs' latency numbers.
- **4118 and 4591 are two different artifacts**, not two configs of one.
- **Never present a single rep as the headline.** 170.6188 is the r4 rep; the N=5 median is
  170.55. 421.18 is the p1 pass; the N=3 median is 420.57.

Gate battery on that rig: `kernel-check` **ALL GREEN** naked (184 `OK`) and model-backed on
real 27B weights (263 `OK`); `run-gen` argmax **MATCH** both artifacts; `run-spec`
self-consistency **PASS** at K=1,2,3. **Do not say "K=1..8" about this rig** — this board
ran K=1..3 as a gate plus K=4/5 as perf cells. The full K=1..8 PASS battery lives on the
community board, on the NVFP4-MTP artifact
(`research/q27-deepdive-20260805/logs/gate-key48-runspec-K1to8.log`); Q8_0 cannot run it at
all (no MTP head, `RUNSPEC-Q8 rc=2`). The accurate form is "K=1..8 self-consistency is a
standing gate, run on the MTP-capable artifact."

### Official FP8 checkpoint — the 2.6x spec cell (a different rig)

Rig `rig2x5090-serve`, date **2026-08-04**, CUDA 13.0.1 / nvcc 13.0.88. Model: the official
`Qwen/Qwen3.6-27B-FP8` safetensors checkpoint (29 GB, e4m3, `weight_block_size [128,128]`,
407 block-128 scale grids). Receipts:
[`research/fp8ship-20260804/official/`](../../research/fp8ship-20260804/official/).

| Arm | tok/s | vs plain |
|---|---|---|
| ST plain | 48.99 | — |
| **ST spec, the checkpoint's own embedded MTP head** | **128.06** | **2.61x** |
| ST spec + own-trim drafter | 136.75 | 2.79x |
| ST e4m3 resident | 48.99 | flat *by construction* |

N=5 medians per arm, SSE `/v1/chat/completions`, greedy, max_tokens 128, pp512-class prompt,
fresh `cache_salt` per request with `cached_tokens=0` verified every rep, arms sequential in
one session, 32-44 C. Bit-identity on the official artifact: argmax 365==365, maxdiff 0.0 x3,
prefill logit vectors **bit-identical 993280/993280 bytes**. Load wall 843.9 → 291.6 s =
**2.89x** (N=3 interleaved).

The e4m3 arm is flat *because* every tensor on this checkpoint is block-128 and falls through
to the Q8_0 path — the win here is load time, not tok/s. Spec **triples TTFT** on this arm
(0.170 → 0.466 s). The GGUF Q8_0 reference row (53.63) is cross-protocol and cross-day, not
apples-to-apples. **There is no official-FP8 measurement on any PRO 6000** — do not merge the
two boards.

## Standing (2026-08-05)

Ten supported models on the 5090 Laptop, all fully gated; MTP-spec cells run 1.06–2.3x the
frozen llama.cpp references (one Gemma near-parity cell, 0.98x, still open), plain cells sit
at the DRAM wall or above.
Newest in: **Qwen-AgentWorld-35B-A3B** (#10, best-vs-best e2e 1.68/1.76/1.75x on the
UD-IQ4_XS re-pick). Just before it, **Ornith-1.0-35B** (#9) — AUTO-KQUANT (k-quant expert
banks join the grouped-f16 prefill lane by default: board-2048 prefill 3.14x) stacked on
resident-if-fits residency, and its own-gen drafter cleared the deployment bar on every
prompt class (best-vs-best e2e 1.31/1.14/1.12x,
[`research/q4k-expert-prefill-20260802/`](../../research/q4k-expert-prefill-20260802/)).
On the H100, a full per-model board against vLLM 0.26 (frozen 2026-08-01): **no end-to-end
losses** — seven wins of seven (1.02–1.81x); decode wins 7 of 7. Multi-user serving measured
on 3 H100s: 1,477 tok/s managed fleet, chaos-tested. Trimmed MTP drafter heads are
published ready-to-use at
[huggingface.co/tiyuvta/memra-bench](https://huggingface.co/tiyuvta/memra-bench).

The **plain-serve c=1 gap (task #70) has a measured fixed-solo opt-in**: explicit
`MEMRA_SERVE_B1FAST=1` routes a solo serve tick through the m=1 fused trunk (+8.33% q9 /
+5.19% q27 decode-only at c=1, N=5 order-paired, 5/5 wins each; serve c=1 was level with the
same-board `run-gen` denominator on the 82-SM rig). It is OFF by default since 2026-08-13:
the isolated-identical default keeps B=1 and B>=2 on one numeric program. Still open: the NVFP4
**spec** serve path (−8.66% pre-fix, its burst loop is a separate path) and the
interactive-latency gaps in the banner above. Both are tracked as gaps, not hidden.

## The serving exactness arc — a defect the discipline caught

The batched F32 prefill router GEMM was m-dependent, so under cross-request prefill batching
a MoE request's own expert routing could change with its co-arrivals — a real serving defect.
Scale of the defect, pre-fix: **121 of 760 (15.9%) `(layer, token)` pairs picked a different
expert *set*** depending on who arrived in the same batch (Ornith-1.0-35B Q4_K_M at
total_m=75, arm `exact0`), plus 217 more differing in order only.
The fix routes prefill through the decode path's m-invariant router/gate kernels
(`MEMRA_ROUTER_PREFILL_EXACT`, default ON), and a bit-identical batched twin then recovered
70% of the prefill it cost. The serving contract — greedy serving is isolated-identical under
concurrent load at defaults; a request's tokens never depend on its co-arrivals — is now
explicit and gated, not assumed: the serve gate replays the same prompts at c=1 and c=16 and
byte-compares every stream (16/16 on all four models post-fix; 7/16 and 6/16 with
`MEMRA_ROUTER_PREFILL_EXACT=0`, which is how the defect was found). Note the object: this is
**serve-vs-serve at c=1 vs c=16**, not identity against a single-token oracle — the
batched-plain path carries a documented, bounded near-tie flip class
([`research/plainbatch-20260804/`](../../research/plainbatch-20260804/)) and first-token
cross-config drift is a separate class again (docs/SERVING.md). Receipts:
[`research/concat-prime-exact-20260802/`](../../research/concat-prime-exact-20260802/),
[`research/fast-router-20260802/`](../../research/fast-router-20260802/).

The H100 35B cell tells the arc in one row: it slipped from 218/1.02x to dead-even when the
concat-prime fix landed (m-invariant prefill router + shexp gates, −13% Hopper expert
prefill) so a session's routing no longer depends on its co-arrivals — then a bit-identical
batched twin of the exact kernel recovered 82% of that cost (8136 pp2048,
kernel-check-pinned mism=0; `MEMRA_ROUTER_BATCH=0` is its perf-only rollback seam) and the
row reached 217/1.01x, ahead again with the contract held — then jumped to 226/1.05x when
direct-from-quant tile loaders reached ~100% of the expert bank (IQ4_XS/IQ3_S added to
Q4_K/Q6_K) and the single-kernel grouped GEMM flipped past cublas as the Hopper default
(round 55: mode-2 prime 13164 vs cublas 8627 tok/s, +52.6% interleaved x5, bit-identical
tiles by construction; expert prefill 8136 → 13258 on the row). Decode never moved; the
dense 27B row was never on the path (re-cell bit-stable). Receipts:
[`research/router-fix-recells-20260802/`](../../research/router-fix-recells-20260802/),
[`research/fast-router-20260802/`](../../research/fast-router-20260802/),
[`research/q35-recell-final-20260802/`](../../research/q35-recell-final-20260802/),
[`research/h100-flip-full-20260802/`](../../research/h100-flip-full-20260802/).

## RTX 5090 Laptop — tracked cells, vs the frozen llama.cpp reference

Single-user decode on the local **RTX 5090 Laptop** (82 SM); both engines interleaved on the
same rig, same prompts. The llama.cpp columns are **frozen reference denominators** recorded
through 2026-08-03 (benching stopped that day) — regression anchors, not a live scoreboard.

**Plain decode** (no speculation, tg128 at 512-token context):

(The generated `PERF-PLAIN` block is omitted here; the current block is in the main doc.)

**Speculative decoding** (MTP head, both at their measured best; columns = short code /
medium code / long agentic prompt classes):

(The generated `PERF-SPEC` block is omitted here; the current block is in the main doc.)

(The generated `PERF-DATE` block is omitted here; the current block is in the main doc.)

### Depth behavior

Depth is part of the contract: at 6.3k-token context every plain-decode lead holds
(1.11–1.13x, 2026-08-02 post-deep-fa re-measure — the depth cells are where the deep
fa_decode rewrite lands hardest, 35B +8.2% at d6257). The depth rows live in
`current-board.json` (`plain_decode_depth`), re-measured N=5 same-session interleaved
with a fresh llama denominator (`research/board-remeasure-20260802/`).

### Prefill, root-caused — and still open

5090 prefill trails llama.cpp (0.59–0.78x), root-caused: llama benches NVFP4 prefill at
W4A4 (FP4 activations), a numeric class memra's exactness gates reject. Output quality
outranks the prefill column — but the numerics explanation does not close the gap, and the
2026-08-05 dogfood run measured the same shape on an identical model file (4k prefill 1.2k
vs 2.1k tok/s). Prefill remains an open lane, not a settled trade.

### Speculative protocol notes

The three spec columns are three prompt classes: short code / medium code (greedy) / long
agentic (temp 0.7, distribution-exact rejection sampling). One asterisk: the 35B short-code
llama bar is an EOS-suppressed continuation (the raw short prompt EOSes at 1 token)
and is not a clean win basis. Every spec row uses one trimmed draft built by the standard
regime ([`docs/DRAFT-REGIME.md`](../DRAFT-REGIME.md)); prebuilt drafts live in the
[bench repo](https://huggingface.co/tiyuvta/memra-bench), or build your own:

```bash
./target/release/frspec-owngen model.gguf ranks.gguf 32768        # ranks from the model's OWN generations
tools/make-trimmed-draft.sh model.gguf ranks.gguf.txt draft.gguf  # extract + trim + quantize
```

### Gemma-4 (QAT Q4_0)

Same protocol, own campaign log
([`research/gemma4-bringup/rig5090-gemma4.jsonl`](../../research/gemma4-bringup/rig5090-gemma4.jsonl));
cells re-paired best-vs-best with llama's own draft depth swept per cell. Highlights
(full tables and the per-cell archaeology live in the campaign log). Hand-written table —
these cells are **not** in `current-board.json`, and their llama column is frozen at the
2026-07-15 best-vs-best re-audit:

| Cell | memra | llama.cpp (frozen ref) | Ratio |
|---|---|---|---|
| 12B MTP spec, 1.7k ctx (K=4 + own-gen trim) | 269.3 | 175.1 | **1.54x** |
| 12B MTP spec, chat (K=4 + own-gen trim) | 240.9 | 172.4 | **1.40x** |
| 31B MTP spec, chat (K=5 + own-gen trim) | 124.3 | 99.0 | **1.26x** |
| 31B MTP spec, 1.7k (K=6 + FR trim) | 97.3 | 83.9 | **1.16x** |
| 26B MTP spec, short ctx (K=4 + own-gen trim) | 328.1 | 298.0 | **1.10x** |
| E4B plain, short | 199.9 | 181.0 | **1.10x** |
| 26B plain, 4.9k ctx | 162.6 | 142.0 | 1.14x |
| plain decode elsewhere | — | — | 1.00–1.07x |

What buys the margins: an FP8 (e4m3) KV cache, occupancy-tuned attention tiles, wide-load
Q4_0 expert dots, FR-Spec drafter trims (150 MB → 18 MB at unchanged acceptance), a
per-model adaptive-K floor (+20% on 12B/31B spec at unchanged exactness), and an in-round
draft-confidence cut. One near-parity cell remains (26B 1.7k spec, 0.98x). Gemma plain
margins are thin where both engines sit at the DRAM wall (1.02–1.06x). Exact prompts and
llama.cpp's swept-best flags: [`docs/COMPETITOR-SETUP.md`](../COMPETITOR-SETUP.md).

### Ornith-1.0-9B (Q8_0)

Supported model #8, and the first community post-train to clear the deployment bar
(best-vs-best e2e ≥ 1.1x on every prompt class). Each engine at its measured best on the
same GGUF: memra runs the regime drafter at K=3 (donor-block own-gen trim — the published
model ships no MTP head, [`docs/DRAFT-REGIME.md`](../DRAFT-REGIME.md)); llama.cpp's best is
plain — its draftless speculative doors are structurally broken on this arch (M-RoPE
position faults, screened in the receipts). Interleaved N=3 medians, same session
([`research/ornith-bar-20260802/`](../../research/ornith-bar-20260802/)):

| Prompt class | memra e2e, 256 tok (spec K=3) | llama.cpp e2e (plain best) | Ratio |
|---|---:|---:|---:|
| code short | 1.395 s | 3.084 s | **2.21x** |
| code medium (1.8k ctx) | 2.050 s | 3.429 s | **1.67x** |
| agentic long (6.3k ctx) | 2.984 s | 4.376 s | **1.47x** |

Spec-vs-own-plain: 2.16/1.77/1.70x at 47-61% acceptance
([`research/ornith-drafters-20260801/`](../../research/ornith-drafters-20260801/)); build the
drafter with the standard two commands in [`docs/DRAFT-REGIME.md`](../DRAFT-REGIME.md)
(donor-block variant — donor pairs and receipts documented there).

### Ornith-1.0-35B (Q4_K_M MoE)

Supported model #9. A three-lane arc took it over the bar: resident-if-fits residency
(the 19.5 GB expert bank fits 24 GB — +50% decode over the old spill default), the
adopted own-gen donor-block drafter (K=2, 60-68% acceptance), and grouped-f16 expert
prefill (the Q4_K/Q6_K bank rides the grouped-f16 lane by default — AUTO-KQUANT then,
the mode-2 default now, same dispatch for this bank: board-2048 prefill 3.14x, pp512 +54%,
decode flat). Best-vs-best per class (memra = spec K=2; llama.cpp = plain, its measured
best on this arch); interleaved N=3, same session
([`research/q4k-expert-prefill-20260802/`](../../research/q4k-expert-prefill-20260802/)):

| Prompt class | memra e2e, 256 tok (spec K=2) | llama.cpp e2e (plain best) | Ratio |
|---|---:|---:|---:|
| code short (27 tok) | 1.033 s | 1.357 s | **1.31x** |
| code medium (1.8k ctx) | 1.618 s | 1.837 s | **1.14x** |
| agentic long (6.3k ctx) | 2.738 s | 3.053 s | **1.12x** |

The honest residual: short-prompt prefill (pp512 0.42x llama) — the per-pass k-quant → f16
dequant cost amortizes with prompt length (0.91x at 2048, a 1.06x memra win at 6.3k); the
no-dequant kill (Q4_K/Q6_K expert MMQ tile loaders) is priced, not built. Plain decode runs
1.086x.

### Qwen-AgentWorld-35B-A3B (UD-IQ4_XS MoE)

Supported model #10 — same stack as Qwen3.6-35B-A3B by construction (header bytes
verified), gated and benched on the UD-IQ4_XS re-pick. Best-vs-best e2e per class 1.68x /
1.76x / 1.75x (interleaved N=3 medians, spec K=2 own-gen drafter at 66-87% acceptance,
plain prefill 1.9-3.5x; [`research/agentworld-iq4xs-20260802/`](../../research/agentworld-iq4xs-20260802/)).
Quant guidance: use UD-IQ4_XS — the UD-Q4_K_M repack's Q5_K expert mix sits outside
fast-path coverage.

## H100 vs the frozen vLLM 0.26 reference



Rig: **1x H100 80 GB**. The vLLM column is a frozen
reference recorded 2026-08-01.

(The generated `PERF-H100` block is omitted here; the current block is in the main doc.)

*The bf16-row wins carry a quant advantage — those vLLM arms move ~4x the weight bytes;
the e2e board is loss-free because decode wins 7 of 7.*

### Flip history

The last e2e losses fell in two days. The two MoE expert-prefill cells closed in round 49:
the 35B when the grouped f16 expert lane reached full dequant coverage and became the
Hopper default (expert prefill +53%) — then +63% more at round 55 when direct-from-quant
tile loaders covered the whole bank and the in-house grouped kernel flipped past cublas —
and the 27B via K-quant f16 prefill mirrors (+54% pp2048) plus split-plane decode mirrors
(+16% decode, bit-identical). The 26B cell flipped from dead-even to a win in round 50,
when a cross-box interleaved arbitration caught that same f16g default *regressing* the
gemma expert class −6 to −8% and narrowed it per-model — the board is tuned per dispatch
class, not per flag. The last two decode losses fell to the shexp fused dot and a
router-default re-arbitration on real prompts.

### Refuted: a Q8_0-exact int8 GEMM on Hopper

Per-cell prefill still trails vLLM on the dense models — the int8-GEMM dtype edge: a
Q8_0-exact int8 GEMM is mechanism-refuted on Hopper (per-32-block rescale costs 5.4x naive
/ 17x pipelined; ptxas serializes cross-bank GMMA register reads), so crossing it means
w8a8-class numerics that change model outputs — an accuracy-bar decision with measured
receipts, not an engineering unknown. The e2e board stays loss-free because decode wins
7 of 7 and dominates the p2048/g512 shape. Full refutation ledger:
[`ARCHITECTURE-H100.md`](../../ARCHITECTURE-H100.md).

### Shipped on the sm_90a build

FA3-class prefill attention (TMA swizzled ring + wgmma, 4.8x the mma kernel), fused wgmma
GDN chunk kernels with varlen twins, f16 prefill mirrors on the cuBLASLt lane for the
Q8_0/Q4_0/Q4_K/Q5_K/Q6_K classes, K-quant split-plane decode mirrors (bit-identical layout
v2, 27B decode +15-16%), the grouped f16 expert lane (one single-kernel grouped GEMM over
the routed experts with direct-from-quant Q4_K/Q6_K/IQ4_XS/IQ3_S tile loaders and a 3-stage
deep tail — Hopper default for the qwen/silu MoE class, 35B expert prefill +53% at round 49
and +63% again at round 55 when it flipped past cublas; per-model off for the gemma class
after the round-50 regression arbitration), a batched serving decode tick (z-batched
attention + KV append, device-side sampling, lean logits: 654-659 tok/s/replica, +25-36%),
an MTP speculative serving fast lane (1.82x plain serving at c=1 on the 27B), cross-request
prefill batching, per-session CUDA-graph decode with kernel-class segment recapture, and
the Hopper wgmma toolkit
([`cu/wgmma_common.cuh`](../../crates/memra-engine/cu/wgmma_common.cuh)) — canonical
core-matrix pairings probed for bf16/tf32/s8: one byte-geometry, three MMA kinds.

- Evidence ledger (every verdict and refutation): [`ARCHITECTURE-H100.md`](../../ARCHITECTURE-H100.md)
- Flags + promoted defaults: [`docs/FLAGS.md` §7](../FLAGS.md)
- One-command battery (retired with the Hopper lane 2026-09-02; run the gates directly)

## Prefix-cache depth is the dominant serving lever (2026-08-13, RTX PRO 6000 WS, single card)

Measured on ONE RTX PRO 6000 Blackwell Workstation, Qwen3.6-35B-A3B UD-IQ4_XS,
Q35-only roster, `MEMRA_PREFIX_CACHE_MB=49152`, experts resident, `MEMRA_SERVE_SPEC=0`. Shape =
4,860-token prompt + 60 output tokens with a 90% shared prefix. N=3 rounds per width, medians.
Raw JSON and the harness: `research/canonflip-20260813/`.

The ONLY variable between these two arms is the depth of the resident prefix entry — same build,
same card, same shape, same request mix:

| arm | cache hit | c=8 req/s | c=16 req/s | c=16 p50 | c=16 out tok/s |
|---|--:|--:|--:|--:|--:|
| shallow entry covers the class | 39.6% | 2.57 | 2.72 | 5.84 s | 163.5 |
| entry covers the full prefix | 99.5% | 6.84 | **8.50** | **1.87 s** | 510.0 |

**3.1x in throughput and 3.1x in latency from cache depth alone.** The mechanism is a depth freeze,
confirmed live by the LCP histogram: 4,860-token requests landed 189 hits in the 1024-2048 bucket
with ZERO in 2048-4096, because a shorter request had already seeded a ~1,937-token entry for that
prefix class. Both deepening paths sit inside the `reused.is_none()` branch and the seed helper
returns early once `n_cached > 0`, so a hit permanently prevents the deeper entry from being created.
It is not capacity pressure: 5 entries and 423 MB of a 49 GiB budget were resident with zero
evictions. Analysis and the ranked fix are in `research/cacheinval-20260813/` (H11).

Consequence for anyone tuning this engine: measure prefix-cache HIT DEPTH, not just hit rate. A high
hit count against a shallow entry looks healthy in the counters while the request still pays most of
its prefill.

### Speculative decoding forfeits that cache — a 4x loss on cache-carried shapes

Same box, same fresh cache, same shape, spec the only variable:

| arm | cache hit | c=16 req/s | c=16 p50 |
|---|--:|--:|--:|
| `MEMRA_SERVE_SPEC=0` | 99.5% | 8.50 | 1.87 s |
| `MEMRA_SERVE_SPEC=1` | 18.4% | 2.14 | 7.40 s |

Spec is **numerically exact** here — the content+reasoning SHA-256 anchor is identical across both
arms — so this is a scheduling/economics result, not a correctness one. Offline it looks like the
opposite: `run-spec` on the same artifact reports 1.51-1.70x versus plain at K=4..8 with 51.8-80.0%
acceptance, because a CLI run has no cross-request cache to give up. That gap between the offline and
serving pictures is the finding worth carrying: on shared-prefix serving shapes the prefix cache is
worth more than the decode win, and `MEMRA_SERVE_SPEC=0` is the serving default for that measured
reason. Keep spec for cache-poor shapes.

### Concurrency envelope on that card

Throughput saturates near 9.9 req/s on a short-prompt shape and near 8.5 req/s on the shared-prefix shape.
Past c=8 latency grows close to linearly for negligible gain, so c=8 is the efficiency knee while
c=16 is the widest width still holding p50 under 1.8 s with a healthy cache. A per-key `rate_limit`
of 16 was verified to protect latency rather than degrade it: at c=24 the endpoint returned 16 x HTTP
200 plus 8 clean HTTP 429 with p50 unchanged.

## Serving performance

Batched decode runs one z-batched tick across sequences, chunked 16-wide on models with a
bit-exact 16-batch kernel class: +18.8% at c=16 on a 5090 Laptop single replica (same-mirror
interleaved N=4); 654-659 tok/s/replica on the H100 serving tick (+25-36%). Greedy
serving is isolated-identical under concurrent load at defaults (m-invariant prefill
router/gate kernels, serve-gate c=1-vs-c=16 byte identity — serve-vs-serve, not identity
against a tokenwise oracle). Multi-GPU boxes serve as a replica fleet — supervisor +
admission proxy + load harness, measured at 1,477 tok/s managed on 3 H100s,
chaos-tested: kill a replica mid-load and the breaker + supervisor recover in seconds with
only in-flight requests lost. A model too large for one card serves over **PP-2** instead
(pipeline-parallel, one engine, stage-split trunk) — see
[the PP-2 cells](#pipeline-parallel-pp-2--the-capacity-shape) below. Endurance: 140 minutes at c=96 on 8 H100s, 464,870
requests, 0 errors, 0 sheds, +0.045% throughput drift — a **9B-class** (Qwen3.5-9B-Q8_0)
warm-prefix result whose load ran at temp 0.7 seeded, with a separate greedy probe hashing
identical on all 8 replicas before and after. Runbook: [`docs/SERVING.md`](../SERVING.md).

**The plain-serve c=1 gap (task #70) — fixed-solo opt-in measured 2026-08-05, rolled back as a
default 2026-08-13.** Phase 1
measured serve c=1 at −11.74% vs the naked CLI on a Q8_0 27B cell (`memra-server` 46.09
tok/s N=3 median vs `run-gen` 52.22 single run, rig `pro6000wk-community`); the
measured cause was that the worker routed B=1 through the batched decode body, missing the
m=1 fusion chain. Explicit `MEMRA_SERVE_B1FAST=1` routes a `b_n==1` tick through
`decode_layers_eager` verbatim: **+8.33% q9 / +5.19% q27** decode-only at c=1 (N=5
order-paired, 5/5 wins each, c=8 flat), bit-identical to `decode_step_h` (strict gate1
PASSes with it ON and FAILed without), and put serve c=1 level with the same-board `run-gen`
denominator on the 82-SM rig. The optimization became an explicit fixed-solo door after dense
Q27 and Q35-MoE showed early EOS when live sessions crossed from the eager/graph program into
the batched program. The default pays the generic B=1 cost to preserve one numeric class across
co-residence changes. The 188-SM phase-1 cell has not been re-measured on the current core; its
−11.74% is history, not a current number. **Still open:** the NVFP4
**spec** serve path (−8.66% pre-fix: 170.55 serve vs 186.72 bare, rig `pro6000wk`)
— the spec tier's burst loop is a separate path the fast path does not touch. Receipts:
[`research/q27-deepdive-20260805/PHASE2-SPEC.md`](../../research/q27-deepdive-20260805/PHASE2-SPEC.md)
(phase 1), [`research/servepath-p2-20260805/`](../../research/servepath-p2-20260805/) (the
fix + the graph-door refutation). The published boards above are **bare-CLI** numbers; do
not read them as serve-path numbers.

### Cold-prime rate under concurrency: the receipted gap (competitive-bench debt 2, OPEN)

The 2026-09-01 competitive bench (one box, RTX PRO
6000 Blackwell Server Edition 96 GB, the same `unsloth/Qwen3.8-27B-NVFP4`
compressed-tensors bytes on every arm, spec OFF everywhere) measured rebuilding an evicted context by re-prefill at:

- N32 (0.93x of the vLLM device pool): memra cold TTFT p50 26.41-30.53 s vs vLLM 0.28.0
  cold p50 6.82 s / p95 11.21 s. The verdict names this "the cold-prime rate gap on this
  artifact class (our 27 s vs vLLM 7-11 s p50 at N32)".
- N12 (0.35x, fits): memra cold p50 34.12-36.26 s; vLLM had zero cold turns.

What this number is NOT: it is not a single-stream prefill claim. The bench's cold TTFT
column includes queue position behind concurrent primes and the decode interleave. It is
also not the 5090 W4A4 numeric-class gap documented above ("Prefill, root-caused"): that
one is single-stream and numeric; this one is concurrency-shaped, on the serving card
class.

Mechanism map from the worker (suspects, named for the diagnostic; none of this is a
measured attribution yet):

1. Interactive prime is time-sliced at the lane's `MEMRA_PREFILL_TICK` = 1024 tokens per session per
   scheduler tick. With K sessions priming, a 60k-token rebuild pays ~59 round-robin
   visits, each behind K-1 sibling chunks plus the batched decode phase (tick order:
   spec bursts, prefill chunks, batched decode).
2. The wide prime paths (the `SOLO_PREFILL_TICK_T` = 8192 widened solo prime, the
   batched GEMM prime gated on `cache.pos == 0`, the fanout dedup) disengage in exactly
   the N32/N56 regime: sessions are neither solo nor groupable. The 2026-08-28
   `solo_widen_fresh` fix recovered only the solo shape (an armed capture boundary had
   been silently costing 76% of cold TTFT on one measured shape).
3. Boundary-stopped primes (`ckpt_at`, `snapshot_at`, including the new
   `MEMRA_PREFIX_STABLE_BOUNDARY` arm) exclude their sessions from fanout and
   prime-batch by design; that trade is stated on the flag rows.
4. vLLM's arm on the same box runs continuous batching with chunked prefill folded into
   the running batch, so its cold p50 tracks compute, not queue position.

This is ENGINE work (scheduler and prime path), not ops, and it is GPU-diagnosis-first:
no code arc opens before the attribution cell splits queue-wait from prime-compute.

**Named GPU gate (PENDING, next hardware window), two cells:**

- **Cell A, single-stream prime rate:** one session, cold prompts at 30/45/60/80k
  tokens, bench artifact pins (`unsloth/Qwen3.8-27B-NVFP4` @ `57926bac...`), memra vs
  vLLM 0.28.0 on the same box class, `MEMRA_TTFT_TRACE=1` phase receipts per turn.
  Decides the kernel/numeric share of the gap.
- **Cell B, contention twin:** the bench N32 organic replay (corpus v2 + the banked
  driver), per-turn cold TTFT decomposed into queue-wait vs prime-compute from the ttft
  trace. Decides the scheduling share.
- Acceptance: the decomposition reproduces the observed ~27 s cold p50 within 15%, and
  the follow-up arc is opened on the dominant term with these receipts attached.

### Pipeline-parallel (PP-2) — the capacity shape

Rig: 2x RTX PRO 6000 Blackwell Server Edition 96 GB, sm_120a, CUDA 13.2, Gen5 x16
P2P (GPU windows under `flock`, both cards verified at 0 MiB on
entry and exit). Date 2026-08-06. Receipts:
[`research/pp2-batch-20260806/`](../../research/pp2-batch-20260806/),
[`research/pp2-spec-20260806/`](../../research/pp2-spec-20260806/),
[`research/pp2-hardening-20260806/`](../../research/pp2-hardening-20260806/).

**PP-2 exists for capacity, not throughput.** The replica fleet above is the throughput
answer; PP-2 is what lets a model that fits only across the pair serve at all. Read these
cells as "what the split costs", never as a scaling win.

q9, 64 steps, 512-token prompts, greedy, N=5 rep-major interleaved in one lock hold on one
binary (medians; `MEMRA_DECODE_BATCH_CAP=16` applied to all arms equally):

| Cell | Value | Protocol |
|---|---|---|
| Batched split cost, B=4 / B=8 / B=16 | **0.995x / 0.989x / 0.986x** | vs door-shut single device, same binary same lock hold |
| B=1 split cost (historical explicit fast-path arm) | **0.982x** (204.7 vs 208.4 tok/s) | N=5; `MEMRA_SERVE_B1FAST=1` now selects this fixed-solo arm; the default generic control measured 177.4 (0.851x) |
| Transport alone (C/B) | 0.986–0.997x of the seam | so almost all of the small loss is the seam, not PCIe |
| Placement symmetry (dev01 vs dev10) | agree within 0.3% | |
| Aggregate scaling under the split | B=8 = **3.65x** B=1 | intact |
| Spec OFF at c=8, dev10 | **875.1 tok/s**, 96/96, 0 err | the fastest arm in the lane |
| P2P vs a host bounce at boundary payload | **13.6–14.0x** (16 KB: 13.35 vs 0.954 GB/s) | link ~56 GB/s uni, ~107 bidir |

Caveats that travel with these cells:

- **The 1.786x (q9) / 1.905x (q27) pipelined figures are NOT serving throughput.** They were
  measured on a pre-recorded greedy token stream; the lane's own words: "Plain autoregressive
  serving cannot do that." Single-stream greedy gets the serial 0.996x row. Treat that old
  deferred-token experiment as quarantined historical evidence: it is not the B>=2,
  independent-wave serving path promoted below, which has its own PRO-pair re-gate and c=8
  serving receipt.
- **PP-2 defaults to plain decode through the placement-aware spec gate.** The old fatal
  spec placement was fixed and crash-gated, but spec loses every measured q9 and step35
  c=1/2/4 throughput cell. Default LOW=0/HIGH=1 admits no PP-2 spec sessions;
  `MEMRA_SPEC_GATE=0` retains the always-spec rollback and crash-gate arm. Mechanism and
  current matrix: [`docs/SERVING.md`](../SERVING.md#pipeline-parallel-pp-2-serving).
- **The serving primary follows the last `MEMRA_PP_DEVICES` entry**, the head stage. This
  v0.72 correction removes the old placement-order head peer-read asymmetry; both orders
  now measure in the same 111-112 tok/s spec class.

#### Dual-active PP-2 is now the naked default (2026-08-11)

`MEMRA_DUAL_PP` now resolves unset to Auto: an eligible PP-2 plain-decode tick with B>=2,
double boundary slots, and native peer transport schedules two independent batch waves so
one stage can work while the other wave advances. Unset `MEMRA_PP_OVERLAP` follows that
mode and is therefore ON in the naked eligible path. Auto degrades to the serial walker for
c=1, PP-3, host bounce, and other ineligible placements. `MEMRA_DUAL_PP=0` is the one-flag
rollback to the exact pre-flip serial path and also resolves unset overlap OFF; explicit
`MEMRA_DUAL_PP=1` retains the strict forced-request diagnostics described in
[`docs/FLAGS.md`](../FLAGS.md).

The flip commit `e94699eba` passed the 2x RTX PRO 6000 Server Edition battery:
`kernel-check` ALL GREEN; naked-default metrics reported 265 overlaps, 265 balanced slot
pairs, and zero slot collisions; and golden `21b8293f` matched at c=1,2,8,16 under both the
naked and rollback arms with zero divergences. At c=8, N=5/arm interleaved in one lock hold
with warmup discarded, the naked dual default measured **158.065 aggregate tok/s** versus
**133.553** for serial rollback, **+18.354%** (reported 40–44 C, 2,317–2,400 MHz). Full
reduction and raw receipts:
[`research/dualpp-flip-20260811/`](../../research/dualpp-flip-20260811/).

The follow-up deep-load sweep found the best width above the flip battery's c=8
anchor: **c=16 at 171.419 aggregate tok/s** (N=3 windows, +8.449% over the c8 anchor). The curve is non-monotonic — c=20 dips to 164.911 before c=24
recovers to 169.005 — so c=16 is the serving cap for this shape, with exactness 72/72 golden
matches across c=12–24 and TTFT p99 non-binding (1.523 s max in scored windows). Receipts:
[`research/loaddepth-20260812/`](../../research/loaddepth-20260812/).

#### Dense IQ4_XS qmatvec arm 1 — PRO promotion

The Step dense `qmatvec_iq4_XS_dp4a` arm now reuses the existing aligned wide-load plus
`byte_perm` group dot without changing weight bytes, launch geometry, integer dot order, or
the floating-point accumulation tree. On RTX PRO 6000 Server Edition silicon, the baseline,
candidate, and local RTX 5090 reference dumps all hashed to `e52f8369...accf`; byte compare
exited 0 and candidate `kernel-check` was ALL GREEN. The fixed 315-launch Step semantic mix
measured **2.863324 -> 2.673256 ms, -6.638% time / +7.110% logical throughput** (N=8/arm
ABBA interleaved, one GPU0 window, per-row 37–45 C, stock unlocked clocks).

NCU explains the mechanism without hiding the counter that moved the other way:
`lg_throttle` fell 2.900 -> 0.475 and L1TEX throughput fell 61.3875% -> 38.7725%, while
`long_scoreboard` **rose** 12.26 -> 27.39. Wider loads remove load-queue/unpack pressure;
with less issue work, the remaining stall mix is more visibly memory-dependency latency.
NCU replay is not the timing authority—the unprofiled interleaved wall time is—so this is a
**PROMOTION GO** for arm 1, not an end-to-end decode or tracked-board cell. Full PRO
reduction and raw receipts:
[`research/qmatvec-impl-20260811/RESULTS-pro.md`](../../research/qmatvec-impl-20260811/RESULTS-pro.md).

### Ornith-1.5 35B source-verbatim pair-owner study

The exact-width B=16 investigation tested whether ordering routed NVFP4 pairs
by owner expert could recover repeated-row locality without changing the B<=8
arithmetic. Every source-verbatim form is bit-exact across B=2..16, including
batch-composition identity, and the real B=12/B=16 decode-batch gates pass.

It does not replace rows. Clean RTX PRO 6000 timing rejects the serial-owner
forms by 4.5% to 6.8%; the final ordinary-warp owner-order form is 5.3% slower
on the clean local RTX 5090 Laptop. Its later PRO timing window had a concurrent
second-card tenant and is explicitly discarded under the multi-card measurement
law. The exact candidate and raw receipts remain banked for a future isolated
PRO recheck; cached CSR-NVFP4 remains excluded. Full record:
[`research/orndecode-pair-owner-20260825/RESULTS.md`](../../research/orndecode-pair-owner-20260825/RESULTS.md).

## Bring-up notes

- **KAT-Coder-V2.5** — onboarded with zero code change (native `qwen35moe` arch strings;
  argmax + chat gates green — [receipts](../../research/onboard-ornith-20260801/)). Decode
  anomaly RESOLVED: its IQ4_XS trunk was riding the f32 oracle path; default dp4a admission
  moved decode +81% to llama parity (1.016x) and flipped its drafter net-positive on
  code-short (1.25x e2e @K=2). The bar-binding gap is now prefill alone (0.169x — the
  IQ4_XS-trunk MMQ port is priced, [receipts](../../research/kat-anomaly-20260802/)).
- **Hy3 Layer103.5 overlay** — VRAM→RAM→dual-NVMe spill, correctness-gated end-to-end;
  serves at a 5.13 tok/s N=3 median, tuning toward 10
  ([`docs/HY3-SPILL.md`](../HY3-SPILL.md)).
- **MiniMax-M3 REAP50** — safetensors spill; loads + generates, router tuning open.
- **Step-3.7-Flash (`step35`)** — onboarded 2026-08-06 on a 2x RTX PRO 6000
  Blackwell Server Edition pair (sm_120a, CUDA 13.2); the official StepFun IQ4_XS
  artifact is 105.0 GB and fits only across the pair, so this SKU is PP-2-or-nothing. What
  landed: the `step35` arch (config parse, per-layer geometry and per-layer `n_head`), the
  `deepseek-v3` pre-tokenizer (Step was mis-tokenized without it — cross-checked against an
  independent engine), **split multi-shard GGUF support** (the artifact is 3 shards and the
  loader read 1 of 3), the attention mixer (dual-base partial RoPE, SWA 3:1, head-wise gate),
  two new math kernels with their kernel-check cells, and the StepFun ChatML dialect template
  (19 jinja-rendered goldens; reusing the qwen arm corrupted the prompt in 8 distinct ways).
  Gates on the pair: boots over PP-2 and generates coherently, `ppn-gate` **BIT-IDENTICAL**
  both arms (fence `[0,22,45]`), `run-gen` argmax **MATCH** at pp19/64tok, `kernel-check` **ALL
  GREEN** model-backed on the real weights, `chunkinv` PASS at the default config. MTP drafter
  wired and its gate passes both contracts (acceptance 0% → 77.8% at K=1 after the draft head
  was pointed at the NextN block's own `lm_head` instead of the trunk's).
  **This SKU is a spill path, not a resident one** — and that is the honest headline for its
  future numbers. `101.07 GB experts + 3.92 GB trunk vs 100.88 GB free` puts it on the SLRU
  cache even on 2x96 GB. That cache is measurably healthy (89.0% steady-state hit rate, 133.5
  MB per decode token on the 32-token boot, vs a 2678 MB/token Stage-1 baseline = 20.1x less
  PCIe), so the MoE cache is load-bearing here rather than incidental. One caveat travels with
  the decision itself: the residency check is **PP-blind in its numerator** — it sums every
  layer's expert bytes from the GGUF header, including layers living on the *other* card, and
  compares that against one stage's free VRAM. The verdict happens to be right on this SKU, but
  it would wrongly spill a bank that fits per-stage on a wider split. Not fixed: changing
  residency selection is perf-affecting and belongs behind an A/B.
  **v0.73 train update (2026-08-08):** the original chunk-dependence and caller-tick defects
  are closed and covered by `chunkinv35` / `tickinv35` plus canaries; the external MTP head,
  tokenizer byte check, `reasoning_effort`, PP-2 batched fresh-prime path, and request-owned
  speculative policy are live. The ungrouped PP-2 prime pipeline measured 330.0 / 401.8 /
  417.6 tok/s at pp512/2048/4096 (N=5, one interleaved lock hold); the later dynamic
  schedule measured 343.8 / 411.8 / 427.7 tok/s on a second box against fixed ranges, with only the
  pp512 gain material at this N. These scheduler lanes are separate from Lever C.
  Lever C's grouped arm measured 497.5 / 639.2 / 697.6 tok/s on the Step pair (N=5),
  but the required local 5090 transfer cell lost 75.3% on resident KAT, so
  `MEMRA_MOE_GROUPED=1` remains opt-in. At that explicit trial config, the serve surface
  measured 0.595 s short TTFT, 6.052 s 4k TTFT, 12.2 ms 4k cache-hit TTFT, and 146.1 tok/s
  aggregate at c=4 (N=3–8 by cell), with the complete serve-ready bar green. These are
  trial/serving receipts, not a tracked competitor-board row; the generated boards above do
  not move. Receipts: [`research/pipeprime-20260808/`](../../research/pipeprime-20260808/),
  [`research/microchunk-20260808/`](../../research/microchunk-20260808/),
  [`research/leverC-20260808/`](../../research/leverC-20260808/), and
  [`research/serve-ready-20260808/`](../../research/serve-ready-20260808/).
  The stated context stays **128K**: 256K KV fits the allocator on this box (20.4 GiB at
  the memra default, 89.95 GiB headroom after weights+MTP), but activation/graph-pool footprint
  and exactness at 256K are both unmeasured. The PP-blind residency numerator above also remains
  open.

## Full-head MTP learned-depth study (2026-09-20)

One RTX 5090 32GB, 575W, CUDA 13.1; full vocabulary heads, H learning and confidence
cuts off. Same model, sampler and cold priming in every arm. Eight balanced paired
sets per model; 64 selected scored runs / 768 turns. Initial prompt tokens are 3,703
for Qwen and 4,037 for Gemma, growing across eight turns; Gemma has two independent
conversations per run. Sampled 1K output cap, at least 10s warmup and 60s measured.

| MTP model, full head | Fixed E2E tok/s | Native adaptive E2E tok/s | Learned E2E tok/s | Learned vs fixed | Learned vs native |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B, fixed K=3, adaptive/learned 1–7 | 96.750 | 97.714 | 94.377 | −2.45% (1/8 wins) | −3.42% (0/8) |
| Gemma 4 12B, fixed K=5, adaptive/learned 1–5 | 175.325 | 188.080 | 188.889 | +7.74% (8/8) | +0.43% (6/8) |

These are native-session measurements on a frozen code-analysis workload, not HTTP
serving rates. Fixed K values are preselected native starting points, not calibrated
best-fixed oracles. The Gemma gain over fixed K=5 is repeatable; native adaptation
captures almost all of it. This cost learner is not promoted, and its runtime
prototype is retired with source and evidence preserved. Partial sets interrupted
by other GPU tests were excluded and retried in full. No DFlash measurements enter
this MTP comparison. [Protocol, audit and raw data](../../research/mtp-learned-depth-20260920/RESULTS.md)
and [decision](../decisions/MTP-LEARNED-DEPTH-20260920.md).

## Release profile: fat LTO on the host paths (2026-11-02, RTX 5090 Laptop)

The workspace release profile moved from Cargo's defaults (16 codegen units, thin-local LTO) to
opt-level 3, fat LTO and one codegen unit. The A/B used the same source, Qwen3.5-9B NVFP4 MTP GGUF
and the `balanced` platform profile, with serving rounds interleaved ABBAAB (three per arm).

- `/v1/tokenize` on a 165,525-token chat (template + BPE + JSON): 256.8 to 240.6 ms median
  (-6.3%; 60 calls per round), with server CPU per call 250 to 240 ms.
- Flat: short cached-prompt TTFT p50 58.1 vs 58.3 ms, short-request E2E p50 102.4 vs 102.9 ms, and
  greedy 512-token stream 3.08 vs 3.09 s.
- Byte-identical between arms: tokenizer ids, greedy and seeded sampled streams, fatbins and device
  ELFs.
- `memra-server` is 17% smaller (74.7 to 62.0 MB).

Build cost and raw rows are in [the decision record](../decisions/RELEASE-PROFILE.md) and
[research/release-profile-20261102/](../../research/release-profile-20261102/LANE.md).

## COMPETITOR-SETUP.md as of 2026-11-11

The whole of docs/COMPETITOR-SETUP.md before the 2026-11-12 rewrite: the July 2026 single-box
session on the RTX 5090 Laptop (llama.cpp b9743, vLLM 0.23.0, SGLang 0.5.9), headings demoted
one level. Local paths are replaced with `<llama.cpp>`, `<vllm-venv>`, `<sglang-venv>` and
`$MODELS`.

Beat-target reference for the memra benchmark. Every competitor is tuned to its **best** runnable single-stream config on this box (consumer Blackwell GB203, sm_120, 24 GB, 858 GB/s measured read wall, 82 SMs, thermal-bound). We beat them at their peak, not their defaults.

_Measured tok/s values are point-in-time records from the tuning session — both engines move. The current boards live in [docs/PERFORMANCE.md](../PERFORMANCE.md) and `research/tune-data/rig5090.jsonl`; this file documents configs (build flags, serve lines, model files), which change rarely._

Models: **Qwen3.5-9B** and **Qwen3.6-27B** (hybrid GDN: gated-deltanet linear-attn + periodic full-attn + MTP). The 35B-A3B MoE setups live in per-engine sections of the research dump; this is the 9B + 27B copy-paste reference.

Box prep (run once per session, before any engine):
```bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
gpu-full-power on
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader   # MUST be empty (serial!)
```
**SERIAL RULE:** never run two engines (or a bench + memra) at once. vLLM/SGLang GPU-memory
profiling RACES any other allocator during init. One engine on the GPU at a time, period.

---

### 0. TL;DR — which format fits 24 GB, per engine, per model

| Model | llama.cpp (GGUF) | vLLM (HF safetensors) | SGLang (HF safetensors) |
|-------|------------------|------------------------|--------------------------|
| 9B    | NVFP4 5.7 GB ✅   | NVFP4 modelopt ~5.3 GB ✅ (or bf16 ~18 GB, tight) | FP8 ~9–10 GB ✅ |
| 27B   | NVFP4+Q4_K_M 16 GB ✅ | NVFP4 modelopt ~14–15 GB ✅ | **NVFP4 modelopt_fp4 ~14–15 GB only** — FP8 (~27 GB) does NOT fit ❌ |

Hard non-starters on 24 GB (all engines): 27B bf16 (~54 GB), 27B FP8 (~27 GB), 35B Q6_K_XL (~31 GB).

---

### 1. llama.cpp — PEAK (this is the primary beat-target; numbers MEASURED on this box)

Build: b9743 (c57607016), `arch = sm_120a` SASS, CUDA 13.1, compile flags `GGML_CUDA_FA=ON GGML_CUDA_GRAPHS=ON GGML_CUDA_FA_ALL_QUANTS=ON`, `FORCE_MMQ=OFF`. Bins: `<llama.cpp>/build/bin/`. **Set `GGML_CUDA_GRAPH_OPT=1` before every run.**

#### 1a. llama-bench — RAW pp/tg (fair, kernel-speed apples-to-apples; CANNOT do MTP)
```bash
export GGML_CUDA_GRAPH_OPT=1
LB=<llama.cpp>/build/bin/llama-bench

# 9B NVFP4 (the winner format — beats Q8_0 decode by +50%)
$LB -m $MODELS/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf \
  -ngl 999 -fa 1 -ctk q8_0 -ctv q5_1 -p 2048 -n 128 -d 0,4096,8192 -r 5
#   MEASURED: pp512=6220 t/s, tg128=126.6 t/s (f16 KV); q8_0/q5_1 KV: 5961 pp / 123.9 tg; d8192: 5531 pp / 116.8 tg

# 27B NVFP4+Q4_K_M (14.7 GiB on disk; the file shows 16 GiB)
$LB -m $MODELS/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf \
  -ngl 999 -fa 1 -ctk q8_0 -ctv q5_1 -p 2048 -n 128 -d 0,4096,8192 -r 5
#   MEASURED: pp512=1980 t/s, tg128=42.1 t/s (raw, no MTP)
```

#### 1b. llama-server — PEAK single-stream WITH MTP spec-decode (the real beat-target for decode)
```bash
export GGML_CUDA_GRAPH_OPT=1
LS=<llama.cpp>/build/bin/llama-server

# 27B NVFP4 PEAK — MTP gives +58% decode (MEASURED 42.1 -> 66.6 t/s, accept 0.762, mean acc len 3.29)
# GOTCHA: the "-mtp.gguf" file has NO embedded NextN head — you MUST pass the external draft (-md).
$LS -m $MODELS/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf \
  -md $MODELS/qwen36-27b-nvfp4-mtp/mtp-Qwen3.6-27B-NVFP4.gguf \
  -ngl 999 -fa on -ctk q8_0 -ctv q5_1 -c 65536 --parallel 1 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.2 \
  --jinja --temp 0.6 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port 8099
#   MEASURED: TG 66.6 t/s (MTP on) vs 42.1 raw; peak VRAM 19.5/24 GiB @ 8K ctx (draft adds ~3.4 GiB)

# 9B NVFP4 PEAK (already 126 t/s raw; no separate draft file on disk -> run raw, MTP not wired for 9B here)
$LS -m $MODELS/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf \
  -ngl 999 -fa on -ctk q8_0 -ctv q5_1 -c 65536 --parallel 1 \
  --jinja --temp 0.6 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port 8099
```
- **Format/quant:** NVFP4 GGUF (native sm_120a FP4 tensor cores). Fastest + smallest. 9B = 5.7 GB, 27B = 16 GB.
- **Attention:** `-fa on` (llama.cpp's ggml FA kernel, sm_120a SASS — not FlashInfer/FA3/FA4).
- **KV quant:** `-ctk q8_0 -ctv q5_1` (~2% decode cost, lets 64k ctx fit). MTP draft ctx is f16.
- **Graphs:** ON (compiled-in), plus `GGML_CUDA_GRAPH_OPT=1`. No enforce-eager penalty.
- **Spec:** `--spec-type draft-mtp` (b9743 token; not old `mtp`). `--spec-draft-n-max 3` sweet spot.

#### Fits-24 GB note (llama.cpp)
9B NVFP4 (5.7 GB): trivial. 27B NVFP4 (16 GB) + MTP draft (5.9 GB) peaks 19.5 GiB @ 8K ctx, fits with room at 64k with q8_0/q5_1 KV. 27B bf16 (~54 GB) and 35B Q6_K_XL (~31 GB) do not fit.

---

### 2. vLLM 0.23.0 — PEAK

venv: `<vllm-venv>` (torch 2.11+cu130, CUDA 13.0, sees cap 12,0). Native
sm_120 SASS in `_C.abi3.so`; `cutlass_scaled_mm_supports_fp4(120)=True` (native NVFP4 W4A4 present).

> **Must download a pre-quantized HF checkpoint** — vLLM cannot run the on-disk GGUF efficiently
> (slow dequant path, no qwen3_5 hybrid GGUF loader). 27B/35B bf16 cannot even be loaded to quantize
> on the fly (needs full bf16 in VRAM first). Use modelopt-NVFP4 safetensors with **bf16-restored MTP**.

```bash
source <vllm-venv>/bin/activate

# 27B PEAK — HF modelopt-NVFP4 + MTP n=3 + FP8 KV + CUDA graphs ON
VLLM_ATTENTION_BACKEND=FLASHINFER \
vllm serve sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP \
  --quantization modelopt \
  --language-model-only --trust-remote-code \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.92 \
  --mamba-cache-mode align \
  --max-cudagraph-capture-size 32 \
  --enable-prefix-caching --enable-chunked-prefill \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'

# 9B PEAK — NVFP4 modelopt HF checkpoint (~5.3 GB, lots of headroom -> max-model-len 131072)
VLLM_ATTENTION_BACKEND=FLASHINFER \
vllm serve <qwen35-9b NVFP4 modelopt HF repo> \
  --quantization modelopt --language-model-only --trust-remote-code \
  --tensor-parallel-size 1 --max-model-len 131072 --max-num-seqs 1 \
  --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --gpu-memory-utilization 0.92 \
  --mamba-cache-mode align --max-cudagraph-capture-size 32 \
  --enable-prefix-caching --enable-chunked-prefill --reasoning-parser qwen3 \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
# Fallback 9B if no NVFP4 repo: local bf16 $MODELS/qwen35-9b-hf, --max-model-len ~24576
# (bf16 weights ~18 GB leave little KV room in 24 GB; tight but runs single-stream).
```
For the **honest no-spec decode number**, drop `--speculative-config`; re-add it for the tuned number.
- **Format:** HF modelopt-NVFP4 (W4A4, native sm_120 cutlass). Prefer `modelopt` over compressed-tensors
  (the latter is the SLOW Blackwell fallback AND usually dropped the MTP head → 0% accept).
- **Attention:** `VLLM_ATTENTION_BACKEND=FLASHINFER` (FlashInfer 0.6.12, supports 7.5≤cap≤12.1).
  NEVER FLASH_ATTN/FA3 on sm_120 (kernels are sm_80-only; + FA rejects fp8 KV). TRITON_ATTN is the fallback.
- **KV quant:** `--kv-cache-dtype fp8` (e4m3, +12% throughput; MUST pair with FlashInfer). NOT nvfp4 (sm_100 only).
- **Graphs:** ON (FULL_AND_PIECEWISE). NEVER `--enforce-eager` (~5–8× slower on sm_120).
  `--max-cudagraph-capture-size 32` avoids the Mamba-cache capture crash at low max-num-seqs.
- **Spec:** `{"method":"qwen3_5_mtp","num_speculative_tokens":3}` (auto-normalizes to mtp; n=3 sweet spot).

#### Fits-24 GB note (vLLM)
9B NVFP4 ~5.3 GB ✅ (huge headroom). 27B NVFP4 ~14–15 GB + fp8 KV ✅ (cap `--gpu-memory-utilization 0.92`; subtract 1–3 GB if display/compositor attached). 27B FP8 ~27 GB ❌, 27B bf16 ~54 GB ❌. Keep `--max-num-seqs 1` for fair single-stream bench (vLLM peaks at 2 concurrent on the 5090, but we hold single-stream across engines).

---

### 3. SGLang 0.5.9 — PEAK

venv: `<sglang-venv>` (torch 2.9.1+cu128, CUDA 12.8 → is_sm120_supported()
AND is_blackwell() both True). qwen35 is hybrid GDN → **full-attention backend MUST be `triton`** on
sm_120 in 0.5.9 (trtllm_mha is sm_100-only here; FlashInfer/FA3/FA4 are rejected by the hybrid-GDN
assertion). GGUF does NOT load (no GDN/mamba conv_state tensor-name mapping) → HF safetensors only.

```bash
source <sglang-venv>/bin/activate

# 9B — FP8 HF safetensors (~9–10 GB, fits with room)
SGLANG_USE_CUTEDSL_GDN_DECODE=1 \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-9B-FP8 \
  --tp-size 1 \
  --attention-backend triton \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.85 \
  --context-length 65536 \
  --cuda-graph-max-bs 8 \
  --chunked-prefill-size 4096 \
  --reasoning-parser qwen3 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --max-running-requests 1 --host 127.0.0.1 --port 30000

# 27B — dense FP8 (~27 GB) does NOT fit 24 GB. Use NVFP4 (modelopt_fp4) ~14–15 GB:
SGLANG_USE_CUTEDSL_GDN_DECODE=1 \
python -m sglang.launch_server \
  --model-path <Qwen3.6-27B NVFP4 modelopt_fp4 HF repo> \
  --quantization modelopt_fp4 \
  --tp-size 1 \
  --attention-backend triton \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.80 \
  --context-length 32768 \
  --cuda-graph-max-bs 4 \
  --chunked-prefill-size 2048 \
  --reasoning-parser qwen3 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --max-running-requests 1 --host 127.0.0.1 --port 30000
```
- **Format:** 9B = FP8 HF safetensors (reliable, official Qwen repos). 27B = **NVFP4 (modelopt_fp4) only**
  (FP8 27B exceeds 24 GB). GGUF/AWQ/GPTQ: skip (no hybrid loader / no published checkpoints / slower).
- **Attention:** `--attention-backend triton` MANDATORY (hybrid-GDN constraint on Blackwell).
- **KV quant:** `--kv-cache-dtype fp8_e4m3` (triton supports FP8 KV). Mamba SSM state dtype is separate.
- **Graphs:** ON. Small `--cuda-graph-max-bs` (4–8) for single-stream cuts capture memory / avoids OOM.
  Do NOT `--disable-cuda-graph`; do NOT `--enable-torch-compile`.
- **Spec:** `--speculative-algorithm NEXTN` (= MTP; auto-rewrites to EAGLE). `--speculative-eagle-topk 1`
  is REQUIRED on the triton/GDN path (topk>1 needs flashinfer/fa3 paging the constraint forbids).
  MTP head is embedded in the checkpoint — no separate draft path.

#### Fits-24 GB note (SGLang)
9B FP8 ~9–10 GB ✅. **27B FP8 ~27 GB ❌ — must use NVFP4 modelopt_fp4 (~14–15 GB) ✅.** Trim `--mem-fraction-static` to 0.80 and `--cuda-graph-max-bs` to 4 if OOM at load with fp8 KV + long ctx.

---

### 4. ENGINE THAT CANNOT RUN A MODEL (flagged)

- **SGLang 27B-dense FP8:** genuinely cannot run on 24 GB — FP8 weights are ~27 GB > 24 GB. Mitigation
  used above: NVFP4 (`modelopt_fp4`) checkpoint instead. If no NVFP4 27B repo is available, SGLang
  cannot serve the 27B at all on this box.
- **SGLang GGUF (any of these models):** cannot load — GGUFModelLoader has no tensor-name mapping for the
  GatedDeltaNet/mamba conv1d/conv_state/A_log weights. These GGUFs (Q8_0/NVFP4/IQ4_XS/Q6_K) are
  llama.cpp/memra-only. SGLang requires HF safetensors.
- **vLLM GGUF:** loads but is the slow dequant path with no qwen3_5 hybrid GGUF loader — not used; HF
  modelopt-NVFP4 required. 27B/35B bf16 cannot be loaded to quantize on-the-fly (won't fit to start).
- **vLLM/SGLang FlashAttention / FA3 / FA4:** prebuilt kernels are sm_80-only and do not run on sm_120;
  forcing them errors or PTX-JITs. Use FLASHINFER (vLLM) / triton (SGLang).

---

---

### 5. FAIR BENCHMARK PROTOCOL (identical across all 4 engines incl memra)

Fair benchmark requires: **same prompt, same generation length, single-stream, with warmup, N=5 medians, gpu-full-power on**, and prefill/decode tok/s extracted identically. Driver: `tools/bench.sh`.

#### 5.1 Invariants (held identical for all 4)
- **Hardware state:** `gpu-full-power on`; `--prio 3` where supported; GPU otherwise idle (serial).
- **Single stream:** parallelism = 1 everywhere (llama `--parallel 1`; vLLM `--max-num-seqs 1`;
  SGLang `--max-running-requests 1` + bench `--max-concurrency 1`; memra is inherently single-stream).
- **Prompt:** ONE fixed prompt of **P = 512 tokens** (prefill workload). Use the same tokenized prompt
  string for the server engines; for memra (token-id CLI) use a fixed 512-id list of the SAME content.
- **Generation length:** **N_GEN = 128** new tokens, **greedy / temp 0** (`ignore_eos`/`-n 128`), so
  acceptance and length are deterministic and the decode count is identical.
- **Context depth sweep:** `d = 0` for the headline; optionally `d = 4096, 8192` (llama-bench `-d`,
  servers via a primed conversation) for a long-context curve. Headline number is `d=0`.
- **Warmup:** 1 discarded run per engine (loads weights, JIT, captures CUDA graphs) before timed runs.
- **Repetitions:** **N = 5** timed runs; report the **median** (thermal-bound laptop drifts down under
  sustained load → median, not peak). `N` is overridable via `N=...` env.
- **KV quant:** matched where the engine supports it — llama `q8_0/q5_1`, vLLM/SGLang `fp8(_e4m3)`.
  memra currently uses its native KV (KV-quant kernels are task #2, not yet landed); note this in results.

#### 5.2 What we measure and HOW (identical definition)
- **Prefill tok/s = P / prefill_time**, where `prefill_time` is wall-clock from prompt submission to the
  first generated token (TTFT minus queueing), for a `P=512` prompt.
- **Decode tok/s = (N_GEN − 1) / decode_time**, where `decode_time` is wall-clock from the first to the
  last generated token (excludes prefill), for `N_GEN=128` greedy tokens.
- **Two decode numbers per engine** that supports spec: **(a) no-spec** (apples-to-apples kernel speed)
  and **(b) MTP/NEXTN-on** (peak tuned). Report both; memra must be compared against the no-spec number
  for kernel fairness and against the spec number for the headline "beat them at peak".

Extraction per engine:
| Engine | Tool | Prefill source | Decode source |
|--------|------|----------------|---------------|
| llama.cpp | `llama-bench -p 512 -n 128 -r 5` | `pp512` t/s | `tg128` t/s (no-spec). MTP via `llama-server` `prompt eval`/`eval` lines (`tg` with `--spec-type draft-mtp`). |
| vLLM | `vllm bench` / Python `LLM.generate` w/ `SamplingParams(max_tokens=128, ignore_eos=True)` | `prompt_throughput` / TTFT-derived | `decode_throughput` (run once without, once with `--speculative-config`). |
| SGLang | `sglang.bench_one_batch_server --batch-size 1 --input-len 512 --output-len 128` (or `bench_serving --max-concurrency 1`) | reported input throughput / TTFT | reported output throughput (run with/without `--speculative-algorithm NEXTN`). |
| memra | `run-gen <model> <512 prompt ids>` w/ `MEMRA_NGEN=128` | see 5.3 (currently NOT timed separately) | the printed `... tok/s` line (decode-only; prefill done via decode_step loop, excluded from the timer). |

#### 5.3 memra honesty notes (do not paper over)
- `run-gen` (`crates/memra-engine/src/bin/run_gen.rs`) currently times **decode only** (`tok/s` line at
  L46): it loops `decode_step` over the prompt to warm the cache, then times `N_GEN` greedy steps. It
  does **not** print a separate prefill tok/s, and prefill is done as a decode_step loop (not a batched
  prefill GEMM). For a fair PREFILL comparison, memra needs a `run-gen` addition that times
  `forward_last(prompt)` (the batched prefill path) and prints `prefill P/dt tok/s`. Until that lands,
  memra's prefill is reported as **"N/A (batched-prefill timing not yet in run-gen)"**, not faked.
- memra spec-decode (MTP) is roadmap task #9, not yet landed → memra's decode number is **no-spec**, so it
  must be compared against the competitors' **no-spec** decode for fairness, and we explicitly note that
  the competitors' headline uses MTP that memra does not yet have.
- memra KV-quant (task #2) not landed → memra KV is native; competitors use q8_0/q5_1 or fp8. Note it.

#### 5.4 Run order (serial)
1. `gpu-full-power on`; confirm GPU idle.
2. llama-bench (9B, 27B) → pp512 / tg128, N=5 median.
3. llama-server + MTP (27B) → decode-with-spec, N=5 median. Kill server.
4. memra run-gen (9B, 27B) → decode tok/s, N=5 median.
5. vLLM serve (9B, 27B) → no-spec then MTP decode, N=5 median. Kill server.
6. SGLang launch (9B, 27B) → no-spec then NEXTN decode, N=5 median. Kill server.
Each step fully releases the GPU before the next (`tools/bench.sh` enforces the serial guard).

---

### 6. EXPECTED RANKING — where memra must land to win

All targets are the MEASURED competitor numbers on this box (the bar to clear). Win = strictly faster
than the best competitor on that metric, single-stream, same protocol.

#### 9B (decode, tok/s)
| Engine | Format | Decode no-spec | Decode peak (spec) |
|--------|--------|----------------|---------------------|
| llama.cpp | NVFP4 | **126.6** (the bar) | 126.6 (no 9B draft on disk) |
| vLLM | NVFP4 | ~70–90 (FlashInfer, est.) | ~85–100 (MTP n=3) |
| SGLang | FP8 | ~70–90 (triton, est.) | higher w/ NEXTN |
| **memra (now)** | Q8_0 dp4a | **59.6** | n/a |
- **9B target: beat 126.6 tok/s decode (llama.cpp NVFP4).** memra is at 59.6 (Q8_0); the gap is ~2.1×.
  Closing it needs the decode host-round-trip removal (#1, done) + NVFP4 GEMM + CUDA-graph decode (#4).
  Realistic intermediate win: beat vLLM/SGLang no-spec (~70–90) first, then llama.cpp.

#### 27B (decode, tok/s)
| Engine | Format | Decode no-spec | Decode peak (spec) |
|--------|--------|----------------|---------------------|
| llama.cpp | NVFP4 | **42.1** (no-spec bar) | **66.6** (MTP, the peak bar) |
| vLLM | NVFP4 | est. lower than llama raw | ~85–100 (community 24 GB laptop, MTP) |
| SGLang | NVFP4 | est. | NEXTN |
| **memra (now)** | — | not yet benched at 27B decode | n/a |
- **27B no-spec target: beat 42.1 tok/s** (this is the apples-to-apples kernel bar — memra's first win).
- **27B peak target: beat 66.6 tok/s (llama.cpp MTP)** and the ~85–100 community vLLM-MTP number.
  memra reaches this only after MTP spec-decode (#9) lands. Until then, the honest claim is "beat the
  no-spec decode (42.1)".

#### Prefill (tok/s)
| Model | llama.cpp pp512 (bar) | memra |
|-------|------------------------|------|
| 9B | **6220** | N/A until batched-prefill timing in run-gen |
| 27B | **1980** | N/A until batched-prefill timing in run-gen |
- **Prefill targets: beat 6220 (9B) / 1980 (27B) pp512.** Requires the hand-written FA prefill (#3) +
  NVFP4 MMQ prefill path, and adding the prefill timer to run-gen (5.3). memra's structural prefill
  advantage candidate: it fits the 35B MoE in 24 GB where llama.cpp full-offload OOMs (already proven,
  argmax=1178 match) — the "wins where they OOM" angle is a real differentiator even before raw tok/s.

#### Where memra already wins (capability, not just speed)
- **35B-A3B MoE fits 24 GB (~4 GB peak via EDGE-1 selective expert staging)** where llama.cpp
  full-offload OOMs at 30.5 GB. That is a clean capability win to headline alongside the tok/s gaps.

#### Headline bar to beat (single number per metric, 24 GB sm_120, single-stream)
- 9B decode: **126.6 tok/s** (llama.cpp NVFP4).
- 27B decode no-spec: **42.1 tok/s**; 27B decode peak: **66.6 tok/s** (llama.cpp MTP).
- 9B prefill: **6220 pp512**; 27B prefill: **1980 pp512** (llama.cpp NVFP4).

The driver `tools/bench.sh` produces the memra-vs-llama.cpp side of this table automatically; vLLM/SGLang
are run separately (serial) via the section 2/3 commands and pasted into `research/benchmarks.md`.

### Gemma-4 pairing note (2026-07-15)

`llama-bench` `-fa auto` (the default) resolves flash-attention **off** for the gemma-4
GGUFs on this build and silently costs llama ~6-11% — always pass `-fa 1` when pairing
gemma cells (26B short reads 168 under auto vs 190 with `-fa 1`). KV-quant flags
(`-ctk q8_0 -ctv q8_0`) LOSE on gemma (26B short 174.7): llama's best gemma config is
plain `-fa 1` with f16 KV. Power state must be pinned per window (`gpu-full-power on|off`)
— both profiles pair fairly but are not comparable to each other.
