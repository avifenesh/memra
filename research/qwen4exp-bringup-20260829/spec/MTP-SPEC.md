# qwen4_exp MTP speculative decode — the mtp-spec lane (2026-08-30)

Owner target: 200+ tok/s single-request decode on the NVFP4 mint; the lever that carries
the plain shape past 90 (PROFILE-4 verdict: not reachable by shaves; spec is a
multiplier). Branch `qwen4exp-bringup-20260829`; box cloud-eval (2× RTX PRO 6000
Blackwell 96 GB); artifact `~/data/q48fn-nvfp4` (NVFP4 trunk + BF16 mtp graft shard).

## The draft program (SEMANTICS.md §MTP, resolved from SGLang PR #36497)

- Input fusion: `e = fc_embedding(zero-centered-RMSNorm_2560(embed(tok)))` broadcast
  over the 4 streams + per-stream `fc_hidden(FLAT GemmaRMSNorm_10240(trunk wide))`.
- ONE decoder layer at global index 48: QSA (own indexer, own weights) + the 512-expert
  MoE (BF16 in the graft; fused gate_up [512,1280,2560] + down [512,2560,640]).
- Exit through the draft's OWN `mtp.hyper_connection_mixer` into the SHARED trunk
  lm_head. The POST-LAYER WIDE state is the K>1 multi-step carrier.
- Alignment (SGLang): draft cache row i holds TARGET position i+1 — the draft pairs
  (token x_p, hidden h_{p-1}) at rope position p; position 0 never enters the draft.
  Engine: `pos_off = 1` on the draft's rope + indexer positions (reference-parity gates
  run `pos_off = 0`, matching the reference executor's row-indexed positions).

## Deliverable 1 — weights load + residency (PASS, 2026-08-30)

`LoadOptions::load_mtp` materializes the mtp.* namespace through the pack contract:
float rows into reference-layout weights (family (1+w) norm folds apply — GemmaRMSNorm
is the family class), the fused BF16 expert bank split at read and kept RAW, then
DEVICE-resident bf16 (`BankHalf::DeviceBf16`, ~5.0 GB — half the bytes of an f32
residency; the decode path runs per-selected-expert `qmatvec_bf16w_f32` row-offset
launches straight off the resident bytes, exact-widening products).

**Residency answer: trunk NVFP4 + draft co-resident on ONE card.**
post-load `95,283 / 97,887 MiB` (card 0), decode-timing 14.4 ms unchanged with the
draft resident. Receipt: `mtp1-load-run.log`. Headroom ~2.5 GiB — holds the K≤8 verify
stash (~1 GiB at K=8) + logits buffers at bench contexts.

**CORRECTION (mtp9, measured).** "Long-context serving would move the draft to card 1
(noted, not needed here)" was too weak. That headroom does not hold a spec run on an
agentic-length prompt AT ALL: the held-out spec-gate OOM'd on a 495-token prompt, and the
own-gen corpus had to skip every prompt over 400 tokens (724/691/675/602/512/502). Trunk
state + draft state + verify stash + prefill transients exceed what is left once the 5 GB
device-bf16 draft bank sits beside the NVFP4 trunk. Moving the draft bank to card 1 is a
PREREQUISITE for spec on real prompt lengths, not an optimization — see Deliverable 6.

## Deliverable 2 — draft forward (PASS)

Engine `mtp_draft_forward`: fusion + the one layer (trunk `build_layer_w` machinery on
the draft's own `MixerState` + indexer raw-key cache) + mixer exit + shared head; both
the BATCHED multi-row shape (draft prefill / accepted-token replay) and the t=1 chain
step. Returns `(logits [t,vocab], carrier [t,wide])`.

Gates:
- Tiny: `mtp-fixture` (reference `execute().mtp[0]` vs engine, batched + carrier +
  single-step chain, argmax every row, worst 7.9e-4) and `mtp-dir-bf16` (full loader
  path, DeviceBf16 bank, worst 5.2e-6). Receipt: gpu-eager/tiny-fixture-gate.tsv.
- Real checkpoint: `--draft-gate` — HOST reference MTP twin (`execute_mtp_standalone`,
  one checkpoint read serves both sides via `mtp_reference_weights`; only the mtp bank
  expands to f32) on the ENGINE's own captured trunk wide state. **20/20 argmax**
  (batched + step rows), worst abs 1.354e-4 rel 1.198e-4, KL 0.00000 every row,
  carrier worst rel 5.1e-5. Receipt: `draft-gate-nvfp4-mtp.tsv`.

## Deliverable 3 — multi-step drafting (K>1)

The carrier chains: step j feeds (d_{j-1}, carrier_{j-1}) at position p0+j-1. The K
ladder receipt (`spec-ladder-*.tsv`) banks per-K chains; byte-identity vs plain greedy
(deliverable 4's gate) is the coherence proof — every committed token equals the plain
chain's token.

## Deliverable 4 — verify + accept (the qwen38 t-parallel lesson, re-derived)

This trunk's verify pays (K+1)× through 36 GDN recurrences unless weight ops batch over
the K+1 column while state steps sequentially. Re-derivation for THIS geometry
(36 GDN 48V/16QK/128 + 12 QSA + PLE + hyper-gates + 512-expert MoE), with the
BYTE-IDENTITY contract (spec-on output == spec-off greedy output) engineered at the
ROW level — every verify row is bit-identical to the t==1 decode program:

| section | decode program (t==1) | verify chunk (1<t<=K+1) | identity class |
|---|---|---|---|
| trunk dense mats (GDN/QSA/shared/router/lm_head) | `qmatvec_bf16w` (multi4 stack = per-row VERBATIM) | `qmatvec_bf16w_mt` — weights read ONCE for all columns (`set_verify_mt`; grid.y=t twin = OFF arm) | bit-identical (oracle mt mode) |
| hyper read gates | hc-diet 3-launch | hc-diet MT stages (`set_verify_mt`): weight rows read ONCE, tokens inside, inline (x·inv)·nw — per-(row,token) chain VERBATIM; token-grid stages = OFF twin/fallback | bit-identical (oracle: mt AND t=3 grid vs per-token t=1, exact) |
| GDN conv | dwconv (hist rows) | same kernel, t rows (per-position tap order unchanged) | bit-identical |
| GDN scan | `gdn_scan_step` | per-COLUMN `gdn_scan_step_at` launches (same kernel/grid) + per-column state snapshot into the stash | bit-identical + rewind checkpoints |
| QSA attention | `sdpa_naive_mask` per (head,query) | same kernel; masked tail keys add exact zeros | bit-identical |
| QSA indexer proj | cuBLASLt m=1 | per-token m=1 launches (m>1 algorithms may differ) | bit-identical |
| indexer selection | host twin | host twin per row (deterministic) | identical |
| MoE routed | grouped sel matvec (gufuse) | ONE merged launch over every column's slots (gufuse tok_map, `set_verify_mt`) + per-token windowed combines; per-token loop = OFF twin | bit-identical (oracle tok_map mode) |
| PLE | cuBLASLt m=1 + host hashing | per-token m=1 launches + stash of pre-chunk history/normed rows | bit-identical |

Rewind (partial accept) is REPLAY-FREE (the VerifyCkpt precedent): GDN state restores
from the per-column snapshot; GDN conv and PLE history REBUILD from (pre-chunk history
++ chunk rows)[-pad:]; QSA KV/raw-keys/tokens truncate.

Spec loop (`spec_generate`): draft-prefill the prompt (one batched pass), then rounds of
{K-1 carrier-chained draft steps (d1 comes free from the previous round's replay-tip
row), ONE t=K+1 trunk verify with device per-row argmax (4t-byte dtoh), greedy accept
walk, replay-free rewind, batched accepted-token replay that carries the next tip row}.

Gates:
- Tiny (15 arms GREEN): `mtp-spec-tiny` — spec output BYTE-IDENTICAL to plain greedy
  (k=3, 24 tokens, all-reject worst case: 23 rounds of the a=0 rewind path);
  `mtp-rewind keep=1/2/3` — chunk+rewind+decode vs the full-sequence reference across
  an EOS PLE segment reset (worst 2.3e-4); hc-diet oracle t=3 BIT-IDENTITY.
- Real checkpoint: `--verify-bit-gate` (plain t==1 rows vs chunk rows, same fed tokens,
  every logit bit-identical), `--spec-gate` (plain vs spec chains per real prompt,
  byte identity, hard fail). RESULTS: see below.

## Deliverable 5 — measurement (mtp7 final battery; ledger in perf/PROFILE-5.md)

Single-card route, real prompt (prompts.tsv row 0 — the goldens probe's chain
degenerates and is banned from perf rows by the loop law; its receipts sit in mtp2 as
the lesson), greedy instrument, interleaved ×5 with 256 tokens per arm:

| arm | ms/token | tok/s |
|---|---|---|
| plain single-card (same run) | 14.86 | 67.28 |
| **spec K=5** | **8.37** | **119.50** (1.78×) |

K ladder (256 tokens): K=3 119.2, K=4 120.7, **K=5 121.6**, K=6 119.0, K=7 113.9,
K=8 104.5 — K=5 is the knee (accept 0.84, mean accept len 5.12); acceptance decay
beats the bigger window past K=6. Accept-length table = the `hist` column of
mtp7/spec-ladder (rounds by accepted count) + the spec-gate per-prompt rows
(accept 0.57-0.85, mean len 3.8-4.6 across the 4 real prompts).

TP2 route: a t-generic TP2 verify is NOT built (decode_step_tp2 is t==1-wired);
the TP2 PLAIN baseline is 12.9 ms/token (77.27, PROFILE-4) — single-card spec beats
the TP2 route by 1.55×, and the TP2 verify upper bound (plain ratio 1.12× on the
0.8 verify share) is ≤ ~9% — sequenced after the bigger levers (PROFILE-5 §Verdict).

Sampled probe (serving law): vendor defaults temp 1.0 / top_p 0.95 / top_k 20, fixed
seed — **ENGAGED 54/58 rounds** (hist 4,6,12,2,7,27 — 27 full-K rounds), accepted
199/290 drafted; receipt mtp7/spec-sampled-k5-nvfp4-final.tsv.

**Verdict vs the 200 target: NOT crossed — 119.5-126 measured (1.6× gap).** The 90
line PROFILE-4 could not cross is crossed with receipts. The honest residual +
projections (FR-Spec draft-head trim, draft/verify segment graphs, verify sel
restructure, TP2 verify): perf/PROFILE-5.md §Verdict.

## Deliverable 6 — mtp9: the residual's top two levers, both NEGATIVE (perf/PROFILE-6.md)

Interleaved ×5, 256 tokens/arm, real prompt, K=5, greedy instrument. Both seams shipped
default-OFF and both receipts CONFIRM those defaults:

| lever | OFF | ON | ratio |
|---|---|---|---|
| FR-Spec draft-head trim, N=5,538 | full head 8.26 ms = **121.03 tok/s** (accept 0.840, len 5.12) | 9.91 ms = 100.93 (accept 0.561, len 3.82) | **0.834** |
| verify scan-chain segment graphs | eager 9.96 ms = 100.38 | 9.97 = 100.31 | **0.999** |

- **Trim.** Own-gen ranks from the owner SXC pools + a composed real-shaped pack, chat
  template on, 97 prompts / 18 classes / 291 generations / **93,152 counted tokens →
  5,538 distinct ids** (2.2% of the 248,320 vocab; coverage 0.809/0.903/0.975 at
  1,024/2,048/4,096, 1.000 at 5,538; law-1 floor met 4.2×). The head got **44.8× cheaper**
  and acceptance paid more than that back. Every width from 1,024 to 5,538 loses; the width
  table is monotone in coverage. On an IN-CLASS held-out cell the loss shrinks 16.6% → 2.6%
  but never crosses zero. **Binding constraint is corpus SCALE, and law 1's ≥4×-topN floor
  does not catch it**: the floor bounds how well a given topN is ranked, not whether the
  corpus is big enough to discover a topN worth having.
- **Verify graphs.** 576 launches/round (the per-GDN-layer dwconv + t×(scan step + state
  snapshot) + conv roll, the only serially DEPENDENT all-device chain in the chunk) collapsed
  to 36 replays moves nothing. Launch issue is not this model's decode bottleneck at t=1 or
  t=K+1; PROFILE-5's "est. 1-3 ms/round" is retired with a receipt.
- **Exactness held throughout, which is the design claim being confirmed:** rep0 chains
  byte-identical full-vs-trim AND eager-vs-vgraph, all five trim widths reproduced the
  control chain exactly (five different d2t maps), verify-bit 24/24 bit-identical, spec-gate
  byte identity 4/4, tiny arms byte-identical with `vgraph` on and off.
- **At shipped defaults the round is perf-neutral:** spec K=5 8.34 ms = **119.97 tok/s**
  (mtp7: 119.50), ladder knee still **K=5** (121.55), sampled vendor-default probe ENGAGED
  54/58 rounds, tp2-gate 24/24 worst rel 3.018e-5 PASS.

### Two findings that outrank the verdicts

1. **The lane's headline prompt is the friendly one.** Every mtp2..mtp8 perf row used the
   same four goldens prompts; prompts.tsv row 0 accepts 0.840. On chat-template-rendered
   prompts the same full-vocab draft accepts **0.290-0.588** (len 2.46-3.76) → **55-96 tok/s,
   not 121**. Real and byte-gated, but the best case, not the serving case.
2. **Spec cannot run long prompts at this residency — measured, and it inverts §Deliverable 1's
   note.** The held-out spec-gate OOM'd on a 495-token prompt (after 0-3 passed byte-
   identically) and the corpus had to skip 6 prompts over 400 tokens: trunk state + draft
   state + verify stash + prefill transients do not fit the ~2.6 GiB left after the NVFP4
   trunk and the 5 GB device-bf16 draft bank go co-resident on card 0. Moving the draft bank
   to card 1 is therefore a **PREREQUISITE for spec on agentic-length prompts, not an
   optimization**. Card 1 is idle.

## Corpus-worthy findings (for promotion when the lane closes)

- The t-parallel verify law generalizes but the MECHANISM is per-kernel: grid.y=t
  re-reads weights per token (t-linear, measured 2.6× decode at t=5); the win only
  lands when each kernel's weight tile is read once per chunk with the per-(row,token)
  chain kept VERBATIM (bit-identity + 1.31× in one seam).
- Greedy-loop law strikes again: the goldens probe's argmax chain degenerates and
  faked a spec "slowdown" (18.5 vs 11-13 on real prompts) — spec acceptance is
  chain-content-sensitive, so spec perf rows are real-prompt-only, ALWAYS.
- Draft alignment for hyper-connection MTP: draft cache row i ↔ target position i+1
  (`pos_off=1`); the reference-parity gates run pos_off=0 to match the executor's
  row-indexed positions. Wrong alignment costs acceptance, never correctness.
- Sel warp packing (4 warps/block) measured NEGATIVE on decode and flat on verify —
  the sel slice is not SM-block-slot-limited (mtp6).
- **Own-gen corpora are sized by DISCOVERY, not just by the 4× floor** (mtp9). DRAFT-REGIME
  law 1's "corpus floor ≥4× topN tokens" bounds how well a given topN is RANKED; it says
  nothing about whether the corpus is large enough to discover a topN worth having. 93,152
  tokens over 97 prompts yielded only 5,538 distinct ids on a 248,320 vocab — floor met 4.2×,
  trim still −16.6%. Size the corpus by the distinct-id count the VOCAB needs first, then
  apply the 4× floor to that.
- **A trim's accept penalty is multiplicative in K, so a cheap draft does not unlock bigger
  K** (mtp9). Each chain step can propose an out-of-set token, so raising K multiplies the
  miss probability; PROFILE-5's "trim unlocks the K=8 arm" projection is the wrong sign.
- **Launch-issue levers are dead on this model at every t** (mtp9). The trunk's decode graphs
  bought +1.3% for a 2,400-launch cut; the verify's 576-launch serially DEPENDENT scan chain
  collapsed to 36 replays bought 0.999×. Do not propose graph work here again without a new
  mechanism.
- **Perf rows built on one prompt file measure that file** (mtp9). Full-vocab accept is 0.840
  on the goldens prompt and 0.290-0.588 on chat-template renders — a 2-3× spread in
  committed tokens per round, invisible for seven batteries because they shared prompts.tsv.
  Every spec perf claim needs at least one chat/agentic-shaped cell beside it.
- **A spec loss on a shape is an ADMISSION problem before it is an acceptance problem**
  (mtp10). The thinkon 0.87× was fixed without touching the draft: the p-min guard's
  zero-draft rounds (one plain step, still commits) + the accepted+1 window turn the
  break-even arithmetic per ROUND instead of per shape. Fixed break-even at K=5 is 2.87
  committed/round; the policy makes low-accept rounds cost ~1 plain step, so no shape
  can lose structurally. Byte identity is free: admission only shrinks the window.
- **64-token cells cannot see within-generation decay, and the decay is content, not
  state** (mtp10 traces): thinkon accept transitions 3.3 → ~2.0 plateau by position
  ~100 (target entropy 1.59 vs 0.91 nats at forks); carrier drift is IDENTICAL between
  accepting and rejecting rounds (rel L2 ≈ 1.1) — carrier fidelity is not what the
  draft sells; indexer selection is structurally inactive below position 2051.
- **Discovery, again, and now with a price** (mtp10): 405k own-gen tokens discover only
  11,854 distinct ids on a 248k vocab (4.3× tokens → 2.1× ids vs mtp9). Size trim
  corpora by the DISCOVERY curve; a 32k set here costs ~4M tokens (~28 GPU-hours).
- **The verify's non-GPU wall is per-layer host twins, not launch issue** (mtp10 nsys):
  36.5 ms verify = 24.5 GPU + ~12 host bubbles (48 router dtoh + 12 indexer masks per
  chunk). Graph work cannot remove them (routing is layer-sequential host math);
  device-side router/indexer twins are the lever, sized ≤ ~1/3 of verify wall.

## Deliverable 7 — mtp10: card-1 draft, the thinking-shape regression fixed (perf/PROFILE-7.md)

- **Card-1 placement (PROFILE-6 finding 2 closed).** `load_from_dir_dev1`: draft block +
  bank + a private same-bytes lm-head copy on card 1; wide seed rows cross P2P per round.
  Crossing **0.020-0.037 ms/round**; dev1-vs-single spec-ab ~1%; draft-gate 20/20 ON CARD
  1; the mtp9 OOM set (502-724-token prompts) passes spec-gate **6/6 byte identity**;
  card 0 frees 5.2 GB. The co-resident placement remains the gate-binary default (the
  regression twin); the dev1 placement is the serving prerequisite past ~400 prompt
  tokens.
- **The thinkon decay diagnosed (trace instrument, `--spec-trace`).** Content class, not
  length/carrier/indexer: target entropy at forks 1.59 nats (thinkoff 0.91), margins
  0.61 vs 1.31, 71% of misses are word starts; accept transitions 3.3 → ~2.0 plateau by
  position ~100; carrier drift is identical between accepting and rejecting rounds
  (rel L2 ≈ 1.1 — refuted); indexer selection is structurally full-causal below position
  2051 (refuted by construction). Draft rank of the missed target: median 2, max 321.
- **The fix — bounded admission, ported from the prior families:** p-min draft guard
  (MEMRA_SPEC_PMIN semantics incl. PMIN0 zero-draft rounds; sub-threshold token
  discarded uncounted; a guarded round verifies t==1 = one plain step that still commits
  a token) + the dflash accepted+1 adaptive window. Commits are always the target rows —
  byte identity by construction, spec-gate green under every arm. **Ship battery at
  `adapt k_lo=1 + pmin 0.3`** (interleaved ×5, 256 tok, sampled probe engaged per
  shape): thinkon **0.87× → 1.18×** (75.5 tok/s), efflow 0.93× → 1.22×, 724-token
  agentic 0.97× → 1.22×, thinkoff 1.50× → 1.56×, raw 1.78× → 1.73× (bench-only shape).
  NO shape regresses vs plain. Dyn-K decay (rolling-window floor) is built as the
  last-resort bound and UNUSED — no cell needed it.
- **Round-cost identity (owner item 1), thinkon K=5 fixed:** plain step 15.32; verify
  (t=6) 36.5 = GPU ≈ 24.5 (mt dense 8.6, MoE routed union 7.3, hc 4.6, sdpa 1.7, scan
  1.3) + ≈ 12 of per-layer host-twin bubbles (48 router dtoh + 12 indexer masks per
  chunk — a per-forward cost the admission policy directly reduces); draft chain 6.68
  (head-dominated); replay 0.76. Round ≈ 44 ms = 2.87 plain steps = the break-even
  accept length. Named follow-up levers: device-side router/indexer twins (≤ ~1/3 of
  verify wall), FR-Spec trim revival at the priced discovery scale (below).
- **FR-Spec retry at corpus scale (owner lever 2): DISCOVERY still binds; trim stays
  OFF.** 355 prompts (300 SXC, 75/pool; prompts to 940 tokens — dev1 removed the
  400-token skip), greedy 256 + sampled 512×2 → **404,851 counted tokens (floor for
  topN 32,768 met 3.1×) but only 11,854 distinct ids discovered** (4.8% of the 248k
  vocab). Trim A/B at N=11,854 (21× head cut, chains identical everywhere): raw fixed-
  K5 **0.882** (mtp9's −16.6% became −11.8% — scale helps, direction real), thinkoff
  at ship policy **0.905**, thinkon at ship policy 1.014. A 32k discovery set on this
  vocab prices at ~4M own-gen tokens (~28 GPU-hours) — the revival condition, stated.
- **Close-out (run H, merged tip 35a0b4c98):** tiny gate all arms PASS, verify-bit
  24/24, spec-gate 4/4 byte identity on raw AND thinkon at the final config, sampled
  probes ENGAGED; raw 114.8 tok/s, thinkon 75.4 (1.174×).

## Deliverable 8 — mtp11: the deferred round readback (owner-ordered port of spec.rs slice 2)

Audit + mature-pattern mapping: `mtp11/AUDIT.md`. The mtp10 round paid **2j+1 blocking
host syncs** (2 dtoh per drafted token: argmax + p-min confidence, each waiting out the
draft step's head-dominated tail; the verify argmax drain was already merged) plus a
pageable 10 KB embed h2d per chain step and a full [n, vocab] prefill dtoh (~934 MB at
940 prompt tokens, one row consumed).

**The port (commits 03686fb2a, 8f80bde1f; seams default-OFF per the flags law):**

- `SpecOpts::defer` — device-chained draft (argmax slots + a DRAFT-ENGINE-resident
  embed table, `arm_spec_devchain`: bf16 rows proven bit-clean value-by-value at arm
  time, f32 fallback, TRIM-RANK row order under a trim so the raw argmax index gathers
  its own next-step row); guard confidences in device slots; ONE chain drain per round.
  **This family's floor is a 2-drain round, not spec.rs's 1:** the layer-1 PLE block
  host-hashes the chunk's ACTUAL token ids into the host-resident n-gram table, so the
  verify cannot dispatch device-token-blind. t==1 steps (zero-draft rounds, dynk tail)
  commit via device argmax; the prefill dtoh is one row; prefill/bootstrap/replay
  embeds ride the same table (`HostDev`: 4t-byte htod + device gather; host embed
  stays the trim fallback). Structural syncs per round: 2j+1 -> 2.
- `SpecOpts::defer_guard_sync` — the sequential-guard sub-arm (per-step 4-byte prob
  readback, the chain stops exactly where the host arm stops). The default deferred
  guard drains the confidence window and truncates at the FIRST sub-threshold step
  (`spec_guard_trunc`, pure): same picks and counters bit-for-bit; the dispatched
  suffix past the stop is the cost delta the A/B measures (bounded by adapt k_lo=1).
- Found-and-fixed while auditing: decode graphs on an ARMED-verify state — the graphs
  tail carries neither the wide capture nor the argmax sink, so consecutive t==1
  zero-draft rounds silently skipped the replay's seed row (acceptance-only hazard,
  invisible to byte-identity gates). `graphs_mode` now requires `verify.is_none()`.
- `--defer-ab <reps>x<tokens>` (+ `--spec-ladder` for a whole K ladder in one model
  load): interleaved host/defer(/defer-gsync) arms, hard-fails on any chain OR
  admission-counter divergence.

Gates: rig tiny gate **19 arms PASS failures=0** (`mtp11/tiny-fixture-gate-mtp11.tsv`)
— defer byte identity vs plain AND counter identity vs host at pmin 0 / all-stop
pmin 0.5 / reversed-trim, on both table dtypes; `guard-trunc-pin` covers the mid-chain
stop the deterministic fixture cannot produce (stated in the receipt, plus the box
defer-ab at ship pmin 0.3 where mid-chain stops occur constantly).

**Box battery: DONE (perf/PROFILE-8.md, receipts mtp11/*.tsv).** Two findings:

1. **The 256-token spec-gate found a LATENT mtp10-era byte-identity defect** (gen 157,
   raw prompt 2): a prompt shorter than k_cap prefilled through the per-row DECODE
   programs on the armed state while the plain baseline prefilled FUSED — bit-different
   state from token 0, flipping the first 0.024-margin argmax. Reproduces at the
   mtp10-close commit; every prior green spec-gate ran 64 tokens. FIXED (94f1cecc2:
   exact chunks require base_pos > 0); gates grown (rewind-bit, rewind-bit-replay,
   armed-prefill-bit); all identity gates green at 256 tokens post-fix (raw 4/4,
   thinkon 4/4, long 6/6, gsync 4/4, verify-bit 24/24).
2. **The deferred round is perf-flat-to-negative on this family; both seams stay OFF.**
   K=5 per shape: defer 0.992-0.998x, gsync 1.000-1.016x, all inside 1.9-4.0% receipted
   spreads (x5 after both escalation rules fired). thinkon ladder: defer decays
   monotonically 1.009x (K=1) -> 0.992x (K=8) — the post-hoc guard dispatches past the
   stop (+0.20 ms/round dead drafts at K=5); gsync positive at all 7 rungs but never
   clears 2x pooled spread. The 0.67 ms K=1-class win does NOT reproduce: measured
   ~0.02 ms/round, because this family's draft step carries the router/indexer HOST
   TWINS inside it — the host already serializes there, so the per-step 4-byte dtoh
   only cost the inter-twin gap. The deferred chain's ceiling rises when the host-twin
   lane lands device-side routing/selection; PROFILE-8 is the re-measure baseline.
   Chain-embed table on the real artifact: bf16 BIT-CLEAN, 1,212.5 MiB on card 1,
   0.9 s arm. Sampled probe ENGAGED through the defer arm (71/150 accepting rounds);
   greedy thinkon 1.179x with rep0 byte identity (the mtp10 1.174x reproduces).

## Named, not built (honest residual levers)

- ~~FR-Spec head trim~~ — MEASURED NEGATIVE TWICE: mtp9 (−16.6% at N=5,538 from a 93k
  corpus) and mtp10 (−11.8% raw / −9.5% thinkoff-ship at N=11,854 from a 405k corpus
  WITH the card-1 placement and long prompts included — Deliverable 7). The binding
  constraint is DISCOVERY on this 248k vocab; revival is priced at a ~4M-token own-gen
  corpus (~28 GPU-hours), not re-derivation.
- ~~Draft-step CUDA graphs~~ — the verify twin was built and measured FLAT (mtp9, 0.999×) on
  the densest and only serially dependent chain in the round; the draft's chain is smaller
  and less dependent, so it is retired by the same receipt rather than tried separately.
- TP2-route verify: `decode_step_tp2` is t==1-wired (per-card halves + P2P joins +
  segment graphs); a t-generic TP2 verify is a parallel build of the whole half-kernel
  set. The TP2 plain win is 14.5→12.9 (1.12×) — it bounds the available spec gain.
- Batched sel matvec across verify columns (union-of-experts gather) — reads each
  routed expert's bytes once per chunk instead of once per token that routes to it.
- dspark-class tricks (draft-side vocab fold, verify graphs) — named per the qwen35
  precedent, not built.
