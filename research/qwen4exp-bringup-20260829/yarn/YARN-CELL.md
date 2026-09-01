# qwen4_exp YaRN long-context affordability cell

Owner question: "it will probably require yarn for full context the card offers, but need
to see if we can afford it on two cards." Box: cloud-eval frankfurt, 2x RTX PRO 6000
Blackwell 96 GB (97,887 MiB each); artifact `~/data/q48fn-nvfp4` (NVFP4 mint, rev
de4b8e4d); eager arm, single-request, f32 KV.

## VERDICT (the sentence to read alone)

**1M on two cards: NOT TODAY — and the blocker is residency plumbing, not YaRN.** YaRN
itself wires up and gates clean (factor-1.0 byte-identical on the real checkpoint, yarn
goldens 10/10 argmax). What 1M needs is KV *local to the compute card*: 1M of f32 KV is
**47.8 GiB** while card 0 has only **7.7 GiB** free after the 87.8 GiB trunk, and parking
KV on card 1 (measured, not assumed) collapses decode to **462 ms/token at a 4k fill vs
25.7 ms local — 18x** (UVA P2P gather). The route that fits is **TP2 residency** (trunk
halves ~44 GiB + *local* KV halves ~23.9 GiB = ~68 GiB/card, ~28 GiB headroom), and TP2 is
**decode-only today — TP2 prefill is not implemented**. That is the one named piece
between here and 1M.

**What two cards DO deliver today (measured, one card serving + one idle):** **100,000
tokens of context at 35.6 tok/s** decode, with decode **near-flat in depth** (22.3 ms at
4k -> 28.1 ms at 100k, +26% over a 24x depth increase) and the QSA bounded-attention claim
confirmed: `qsa.sdpa` is **7.6 ms/token at 4k AND at 100k**. Prefill to 100k is 8.4 min.

## 1. YaRN wiring (Q1) — DONE, gated

**How the engine ingests it: a config field.** `rope_parameters` (or `rope_scaling`) with
`rope_type: "yarn"`, `factor`, `original_max_position_embeddings` compiles to
`RopeFactors::Yarn` on every QSA rope plan — trunk layers *and* the MTP draft — and the
QSA indexer consumes the same main rotary (SEMANTICS.md §Rope), enforced by an
overlay-vs-attention rope-width check at load. No env var, no serving flag. The shipped
config (`rope_type: "default"`) compiles to `RopeFactors::None` and takes the historical
byte-exact path, so **yarn is default-OFF and the shipped config is untouched**.

The 1M artifact is a hardlink dir with only config.json rewritten (`mk-override.py`):
`factor 3.814697265625` (= 1,000,000/262,144), `original 262144`,
`max_position_embeddings 1000000`. (Symlinks are refused by the loader's snapshot
containment check — hardlinks are the working form.)

Math: transformers `_compute_yarn_parameters` twin — per-pair frequency **divisors** over
the 64 partial-rotary dims (correction range low=14 high=22 at the real shape) plus
`attention_factor = 0.1*ln(factor)+1 = 1.1338861` multiplied into cos/sin. Consumed by
`rope_neox_ffm_f32` (device) and the yarn-aware host indexer twin — one table, three
consumers. Pinned two ways: `transformers-yarn-params.tsv` (generated ON the box against
the installed transformers) and the memra-gguf unit test
`yarn_divisors_match_the_banked_transformers_receipt`. Unimplemented yarn keys
(`attention_factor`, `mscale`, `mscale_all_dim`, `truncate=false`) are **refused at parse**
rather than silently mis-scaled.

Gates (all green):

| gate | result |
|---|---|
| **factor-1.0 byte identity, REAL checkpoint** (the yarn ingest path vs the shipped config) | probe logits `max_abs 0.000e0`, KL 0.00000, top-20 overlap 20/20 on all 10 rows; greedy chains reproduce the banked mtp10 baseline exactly (`box-logits-compare-yarn-f1.tsv`) |
| **yarn goldens at native length** (fresh transformers goldens minted with the SAME yarn-1M config on the pinned BF16, then the NVFP4 yarn artifact gated against them) | logits argmax **10/10**, smooth per-layer envelope; greedy first-divergence -1/8/26/48 vs 1/0/-1/26 on the base gate — the same near-tie fork class, not a new error (`box-hidden-gate-yarn1m.tsv`, `box-greedy-gate-yarn1m.tsv`) |
| tiny `fixture-yarn` (factor 2 — the mscale path; tiny n_rot=2 cannot reach the divisor ramp, stated) | PASS vs memra-reference |
| tiny `yarn-identity` (factor 1.0) | **BIT-IDENTICAL** over all 25 prefill+decode rows |

**There is no golden oracle past native length** — transformers cannot run 500k here. Long
runs are therefore self-consistency gates: greedy chains stable, no NaN, `looped=false` at
every rung, and continuations banked decoded (`box-ceiling-continuations.txt`) — valid
Rust/CUDA source continuations of the real corpus at 4k, 32k, 65k and 100k.

## 2. Memory affordability (Q2) — measured

Measured KV+state cost: **48.9 KiB/token** (131,288-token state = 6,272 MiB), matching the
analytic 48 KiB (12 QSA layers x (K+V) x 512 dims x f32) plus state overhead. Trunk after
load: **89,971 MiB**, leaving **7,916 MiB** on card 0. Indexer raw+pooled keys are host
resident (~7.5 KiB/token, 168 GiB RSS headroom is ample); the device pooled mirror is
~0.5 KiB/token/QSA-layer.

| context | card0 MiB (measured) | card1 MiB | fits? |
|---|---|---|---|
| 4,096 | 96,125 | 3 | yes |
| 32,768 | 96,189 | 3 | yes |
| 65,536 | 96,253 | 3 | yes |
| 100,000 | **96,381** | 3 | **yes (1,506 MiB spare)** |
| 131,288 | 96,243 at state alloc | 3 | **state fits, first prefill chunk OOMs** |
| 200,254 | — | — | **OOM at state alloc** (needs 9.6 GiB of 7.9 GiB free) |
| 262,144 | 12.5 GiB KV | — | **no on card 0** |
| 1,000,000 | **47.8 GiB KV** | — | **no on card 0** |
| 1,000,000, KV on card 1 | trunk only | 47.8 GiB (fits) | **memory yes, speed no — 18x, see Q3** |
| 1,000,000, TP2 residency (**not implemented**) | ~68 GiB | ~68 GiB | fits analytically, ~28 GiB headroom |

**Single-card ceiling: ~100k-130k tokens** (f32 KV). Two independent reducers exist and
are *named, not claimed*: (a) a **bf16 KV cache** halves to 24.5 KiB/token (~200k+ on one
card; the eager arm is deliberately f32 — it is the exactness instrument); (b) the trunk
holds **both** f32 originals and bf16 twins (`trunk_bf16` keeps f32 as the A/B fallback) —
dropping the originals post-load would free ~6 GiB. Neither is needed for the TP2 route.

## 3. Speed at depth (Q3) — measured

x3 timing rounds per rung (fleet protocol 2026-08-30), round medians and spread named;
**no rung escalated** (all spreads <= 0.40%, well inside the 0.5% rule). Rounds are
consecutive decode segments on the same fill — a fresh 100k prefill per timing round is
prohibitive, stated. Greedy is the instrument; `looped=false` at every rung.

| fill | prefill segment | cumulative prefill | decode ms/token | tok/s | round medians (ms) | spread |
|---|---|---|---|---|---|---|
| 4,096 | 18.3 s | 18.3 s | 22.3 | **44.84** | 22.4 / 22.3 / 22.3 | 0.40% |
| 32,768 | 133.9 s | 152.2 s | 24.1 | **41.48** | 24.0 / 24.0 / 23.9 | 0.24% |
| 65,536 | 164.5 s | 316.7 s | 26.1 | **38.31** | 25.9 / 25.9 / 25.8 | 0.14% |
| 100,000 | 184.8 s | **501.5 s** | 28.1 | **35.56** | 28.1 / 28.1 / 28.1 | 0.11% |

Prefill is ~4.7-5.6 ms/token and near-linear -> **1M would prefill in ~85-95 min** (fine:
hardware time is the schedule).

**The QSA bounded-attention claim, receipted:** `qsa.sdpa` = **7.8 ms at 4k, 7.6 ms at
32k, 7.6 ms at 100k** — flat, because attention now reads only the <= 2052 selected KV
rows at any depth. This required the block-list kernel: the dense mask *reads every t_kv
row* (it only zeroes scores), so the historical masked path was O(context) bytes and
refused outright past t_kv 12288.

**The indexer's O(context) host selection, closed:** it was 4.1 ms at 4k -> **29.3 ms at
32k (52% of the token)** and quadratic across a long prefill. The device scorer
(`qsa_index_score_f32`, bit-identical scores) took the 32k token from **52.8 -> 24.1
ms/token** and leaves `qsa.idx_host` at **2.2 ms at 100k (6.9%)**.

Decode residual at 100k: `qsa.sdpa` 24.0%, `ple.host_ngram_gather` 15.4% (host n-gram
gather grows with history — the next named follow-up), `hyper.read` 10.3%, MoE 8.6%.

**KV-on-the-other-card arm, measured (this is why the verdict is TP2, not "put KV on card
1"):** at a 4k fill, prefill 519 s (vs 18.3 s local, 28x) and decode **462 ms/token (vs
25.7 ms, 18x)**, spread 9.44%. The attention kernel gathers K/V rows across PCIe with no
coalescing across head-blocks; effective P2P throughput ~220 MB/s.
(`box-ladder-kvdev1-partial.tsv`.)

**Spec at depth:** the machinery landed and is gated (ring-bounded wide stash + chunked
co-prefill, `mtp-spec-ring` byte-identity PASS) with a `--ladder-spec` driver, but the spec
rungs were NOT run on the box: the draft co-resident on card 0 costs ~9.5 GiB, which at
these depths is exactly the memory the KV needs, and the card-1 draft placement wants card
1 — the same card the KV would need. Spec at depth belongs with the TP2 residency work; no
spec-at-depth numbers are claimed here.

## 4. Engine work this cell required (each gated)

The eager arm could not run any long context before this: it refused past t_kv 12288, its
prefill was O(routed experts) per chunk, and its selection was O(context) on the host.

| change | gate | FLAGS row |
|---|---|---|
| `rope_neox_ffm_f32` + yarn plan/config/reference plumbing | tiny yarn arms + real f1 identity | (config field, no flag) |
| `sdpa_blocklist_f32` block-list attention | arm 0f oracle (bit-identical vs masked at t_kv 4096; host twin past the bound) + `fixture-longatt` (whole program forced, bit-identical) | `set_longatt` = AUTO |
| `qsa_index_score_f32` device scorer | arm 0g (per-score BIT equality + top-512 set equality) | `set_idx_dev` = ON |
| pooled indexer-key cache, parallel selection, `select_nth` top-k | identical sets by construction; battery green | — |
| chunked prefill (`prefill_extend`, head-skip + last-row head) | `prefill-extend` tolerance arm (1.9e-4) | — |
| chunked prefill rides the GROUPED MoE program | same | — |
| grouped-MoE slot sub-batching (grid.y caps at 65,535) | battery green | — |
| chunk-bounded state reserve + optional KV card placement | ladder receipts | — |
| spec ring + chunked co-prefill | `mtp-spec-ring` byte identity | (SpecOpts, default OFF) |

Two bugs the gates caught during the work, both real: nvcc's default FMA contraction made
one score in ~1e3 differ by 1 ULP (enough to flip a near-tie block out of the top-k), and
rewind truncated the host pooled cache without following it on the device mirror (stale
keys scored) — caught by the mtp11 lane's spec byte-identity arms.

## 5. Named follow-ups (in measured order)

1. **TP2 prefill** — the only thing between here and 1M on two cards (trunk halves + local
   KV halves; TP2 decode already exists with a 24/24 argmax gate).
2. **bf16 KV cache** — halves KV; also halves the attention bytes.
3. Drop the trunk's f32 originals when bf16 twins are resident (~6 GiB on card 0).
4. `ple.host_ngram_gather` (15.4% of the 100k token) — device gather / pinned prefetch.
5. Device top-k (host top-k over the dtoh'd scores is 1 MB/layer/token at 250k blocks).

## No product claims

qwen4_exp is not served. Nothing here is a roster row, a context claim, or a price; the
published-context claim would go through facts.json and its gates, and it is not proposed
by this cell.

## Receipts

Rig (tiny batteries, 24 summaries / failures=0): `tiny-gate-yarn.tsv`,
`tiny-gate-longctx.tsv`, `tiny-gate-specring.tsv`. Math pin:
`transformers-yarn-params.tsv`. Box: `box-ladder-ceiling.tsv` (the Q2/Q3 tables),
`box-ladder-smoke2-local.tsv` (host-scorer twin: 52.8 ms at 32k),
`box-ladder-kvdev1-partial.tsv` (the P2P arm), `box-ladder-smoke.tsv` (the per-expert
prefill and masked-decode before-state: 1007 s / 673 ms), `box-logits-compare-yarn-f1.tsv`,
`box-hidden-gate-yarn1m.tsv`, `box-greedy-gate-yarn1m.tsv`,
`box-ceiling-continuations.txt`, `mk-override.py`. Box dirs: `~/realgate/yarn/`.
Branch commits: 9924b3ab3, cb8ef020c, de98fbe9e, 1cad1ce6a, 668875b6d, 7c15c0586,
6413c3de3, 10fcf802f, 5a3a4f6ca, 495c1cb0a.

---

# ROUND 2 (2026-08-31): TP2 prefill + quantized KV, measured

Round-2 receipts and the full tables are in `../round2-box-receipts/LADDER.md`. Same card
class as round 1 (2x RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, 600 W), so the
round-1 rows above are the comparison.

## ROUND-2 VERDICT (replaces round 1's "NOT TODAY — blocker is residency plumbing")

**1M on two cards: NO.** A 1,000,000-token state does not allocate on either route.
Config that goes deepest: **ONE card**, `kvq` K=`q8_0` / V=`q5_1` + `idxq q8` (ship
defaults), yarn factor 3.8147, chunk 2048 — the second card is unused for depth. VRAM:
**11.08 KiB/token measured**, so 600,000 tokens allocate at **96,467 MiB of 97,887 (1,420
MiB free)** and 1M needs ~10.8 GiB against 7,916 MiB free, missing by ~2.9 GiB. Measured
single-card ceiling **~731k tokens**. tok/s at the deepest TIMED fill: **36.30 tok/s at
100,000 tokens** (27.6 ms/token) — and that row is the **f32** arm, so no ship-default
tok/s-at-depth row exists yet.

## The two things round 1 got wrong, now measured

1. **"The route that fits is TP2 residency."** It does not fit. TP2 prefill exists now, and
   it makes depth WORSE: it OOMs during the fill between 65,536 and 81,920 tokens, so its
   ceiling is *below* 100k while one card reaches ~731k. Mechanism, measured: TP2 post-load
   costs card 0 **2,784 MiB more** than single-card (92,755 vs 89,971), and **card 1 is flat
   at 43,603 MiB from fill=16,384 onward** while card 0 carries every additional byte. The
   split does not move the growing cache off the binding card. Card 1 sits with ~54 GiB free
   that long context cannot use. Round 1's "trunk halves ~44 GiB + local KV halves ~23.9 GiB
   = ~68 GiB/card" was a projection, and the shard is not that shape — card 0 keeps the full
   resident bank and card 1 gets a ~40 GiB half-bank copy.

2. **"1M of f32 KV is 47.8 GiB" was the right number for the wrong conclusion.** Quantizing
   the KV was necessary and it delivered: the single-card ceiling goes **~165k -> ~731k**, a
   **4.4x** depth increase (f32 measured at 49.0 KiB/token, reproducing round 1's 48.9). That
   is the real product win of round 2. It is still 1.4x short of 1M, and no amount of KV
   quantization closes that while the trunk holds 89,971 MiB of a 97,887 MiB card.

## What two cards DO deliver for depth today

Nothing that one card does not. **600,000 tokens allocates on one card** (96,467 MiB) and is
the deepest allocation proven; 262,144 allocates at 92,883 MiB. Neither has been FILLED and
timed yet — six spot reclaims on 2026-08-31, each replacement needing a ~174 GB mint download
plus a build before it can measure, exceeded the boxes' lifetimes. `--ladder-kv-dev1` (remote
KV) stays ruled out on round 1's 18x decode collapse; it was not re-measured.
