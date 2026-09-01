# qwen4_exp decode PROFILE-7 — the mtp10 round: card-1 draft, the thinkon regression diagnosed and fixed

Lane: mtp10, continuing spec/MTP-SPEC.md. Box: sbox-eval Frankfurt, 2× RTX PRO 6000
Blackwell 96 GB; artifact `~/data/q48fn-nvfp4`. Entry state from PROFILE-6: spec K=5 =
119.97 tok/s on the raw-continuation shape, but **0.87× (a REGRESSION) on the
vendor-default thinking render**, 45% zero-accept rounds, and spec OOMs past ~400 prompt
tokens (draft co-resident at 95.3/97.9 GB, card 1 idle). Owner direction for this round:
the primary target is ROUND OVERHEAD (a cheap round structurally cannot lose), admission
is the last-resort bound; port the prior families' confidence guard; retry the FR-Spec
mask at the prior lanes' corpus scale. Receipts: `spec/mtp10/`.

## 1. Card-1 draft placement (mechanical prerequisite) — DONE, crossing ≈ 22 µs/round

`load_from_dir_dev1`: the MTP block (weights + the ~5 GB DeviceBf16 expert bank) and a
private copy of the shared lm head (f32 + bf16 twin, same bytes) build on card 1; the
draft state, workspace, and (when armed) the FR-Spec trim live there too. What crosses
per round is the wide seed rows for the replay — (a+1) × 40 KB over PCIe P2P
(`memcpy_peer_async` on the draft stream) — plus the drafted token ids (4-byte dtoh
each, as before). Draft-engine identity is ENFORCED (`check_draft_engine`), because UVA
would make a wrong-engine call "work" at PCIe speed instead of failing.

- **Crossing cost, measured: 0.020-0.037 ms/round** (`# round-cost` lines, every mtp10
  receipt) — ~0.05% of a 44 ms round. The one-time prefill seed crossing is ~26 ms on a
  60-token prompt and ~95 ms at 724 tokens (n × 40 KB, once per generation).
- dev1 spec-ab twin: 8.41 vs single-card 8.33 ms/token on the raw shape (~1%, the P2P
  sync tax). Exactness: `--draft-gate` ON CARD 1 reproduces the mtp2 envelope exactly
  (20/20 argmax, worst rel 1.198e-4); verify-bit 24/24; spec-gate 4/4 byte identity;
  sampled probe hist bit-equal to mtp9's (same seed, same chain).
- Residency after the move: card 0 **89,971 MiB** (5.2 GB freed), card 1 9,555 MiB.
- **Long prompts unlocked (PROFILE-6 finding 2 closed):** the mtp9 OOM set (502-724
  token prompts, `spec/mtp10/long-prompts.tsv`) now runs — spec-gate **6/6 byte
  identity** at 128 tokens (`long/spec-gate-k5-mtp10-long.tsv`). New cell from it: on
  the 724-token agentic prompt, fixed K=5 spec is **0.97×** (20.25 vs plain 19.70
  ms/token, accept ~2.5/round) — the admission problem was never thinkon-only.

## 2. The round-cost identity (owner item 1), thinkon, K=5 fixed

Same-run interleaved numbers (`shapes/ab-spec-k5-rc-thinkon.tsv` round-cost lines):

| piece | ms | vs plain step |
|---|---|---|
| plain decode step | 15.32 | 1.00 |
| verify chunk (t=6) | 36.5 | **2.38×** |
| draft chain (5 steps, incl. 5 argmax dtoh syncs) | 6.68 | 0.44× |
| accepted-token replay | 0.76 | 0.05× |
| P2P crossing | 0.02 | ~0 |
| **round total** | **≈ 44.0** | **2.87×** |

So a K=5 round costs 2.87 plain steps → break-even accept length is 2.87 committed
tokens/round. Raw delivers 5.12 (1.78×), thinkoff 4.3 (1.50×), thinkon 2.46 (0.87×),
the 724-token agentic prompt ~2.5 (0.97×).

### Where the verify's 36.5 ms actually sit (phase-windowed nsys, `nsys/win-*.csv`)

`--profiler-window` now brackets the spec arm, so the kernel sums cover exactly one
K=5 thinkon spec run (the plain window brackets 38 decode-timing steps — note its
kernel rows are graph-hidden: t=1 trunk decode runs captured graphs, and nsys's default
graph tracing reports whole graphs, so the plain window undercounts; the SPEC window is
eager throughout and near-complete). Per round (~52 rounds), the verify's GPU kernels
sum to **≈ 24.5 ms** of the 36.5 ms wall:

| verify GPU slice | ms/round | note |
|---|---|---|
| `qmatvec_bf16w_mt` (weight-shared dense) | 8.6 | ~373 launches/chunk |
| MoE routed (`sel_gu_silu` + `sel_v3` merged tok_map) | 7.3 | union-of-experts bytes — the t-scaling physics |
| `hc_diet_*_mt` (read gates) | 4.6 | |
| `sdpa_naive_mask` | 1.7 | |
| `gdn_scan_step` (+snapshots) | 1.3 | the chain the FLAT vgraph covered |
| **GPU total** | **≈ 24.5** | |
| **non-GPU wall** | **≈ 12** | per-layer HOST-twin bubbles: 48 router dtoh round-trips + 12 indexer host masks per chunk — structural (routing is layer-sequential), and the t=1 step pays the same class per forward |

Consequences: (a) the round's non-GPU ~12 ms is a per-FORWARD cost, so the admission
policy (fewer, narrower chunks) attacks it directly — which the §4 numbers confirm;
(b) the draft chain's GPU is dominated by the full-vocab head (~0.75 ms × 6
reads/round ≈ 4.5 ms) — the FR-Spec trim's target (§6); (c) device-side router/indexer
twins (killing the 48+12 per-chunk host round-trips) are the largest remaining VERIFY
lever and a named follow-up lane, sized here at up to ~1/3 of verify wall; the
mt-kernel efficiency question from the sync-bounded ratios is bounded by (a)+(c) and
stays behind them.

## 3. The thinkon decay DIAGNOSED (trace receipts `trace/spec-trace-k5-trace-*.tsv`)

Per-round traces (accept position, fork margins, carrier drift), 256 tokens × 4 prompts
× {thinkon, thinkoff}, plain-twin byte identity asserted per prompt (hard fail), trace
mode only ADDS reads.

- **The decay is a transition with a plateau, not unbounded length decay**: thinkon
  mean accept 3.29 (positions 0-31) → ~1.9 by position ~100, then flat 2.0-2.2;
  zero-accept share 0.16 → 0.36-0.42 plateau. thinkoff shows NO positional decay
  (3.3-4.8 across the whole window). Hypothesis (a) — generation length per se — dead.
- **Mechanism is CONTENT CLASS (hypothesis b)**: at thinkon forks the TARGET itself is
  uncertain — target softmax entropy 1.59 nats vs thinkoff 0.91; target top-2 margin
  0.61 vs 1.31; 71% of missed tokens are word starts. The structured opening of a think
  block (task restatement, enumeration) accepts well; free reasoning prose is
  intrinsically branchy and argmax-vs-argmax coin flips lose. Severity is
  prompt-dependent (zero-share 0.45/0.31/0.17/0.11 across the four tasks) — so the fix
  must self-key per generation, not per shape.
- **Hypothesis (d) — carrier error — REFUTED**: the carrier seed is always a coarse
  approximation (rel L2 ≈ 1.1 on the FIRST chain seed) and is statistically identical
  between zero-accept rounds (1.10) and accepting rounds (1.12), in both shapes. Drift
  along the chain grows only mildly (1.12 → 1.41 by step 4). The draft's value is its
  token-level language modeling, not carrier fidelity — re-seeding cadence would fix
  nothing.
- **Hypothesis (c) — indexer selection divergence — dead by construction at these
  lengths**: every position < 2051 takes the indexer's structural full-causal fast path
  (budget 512 × block 4 — `indexer_mask_rows`, qwen4exp_gpu.rs), draft and target both.
  No selection ever differed.
- **The draft is never far wrong**: at forks, the draft's rank of the true target token
  is median 2, p90 10, max 321 — 100% inside the top 5,538 ids. (Two consequences: a
  wide FR-Spec trim cannot create fork misses on these shapes; and a confidence guard
  has real signal — fork-row draft margins are ~half the accepted-row margins.)

## 4. The fix: bounded spec admission (guard + adaptive window), ported from prior art

Ported per owner direction from the prior families (receipts in their lanes):
`MEMRA_SPEC_PMIN` semantics — stop the chain when the draft head's softmax confidence
in its own pick drops below p (sub-threshold token DISCARDED UNCOUNTED), including at
j=0 (`MEMRA_SPEC_PMIN0` zero-draft rounds: the verify is a plain t=1 step that still
commits one token — "unpredictable stretches never pay draft+verify overhead", the
llama 35B precedent); and `MEMRA_DFLASH_ADAPT`'s accepted+1 adaptive window (next round
drafts clamp(a+1, k_lo, K)). Confidence = the existing `prob_of_token` 2-pass kernels
(one extra ~1.3 GB/lm-head-read? no — one extra sum-exp pass over the logits row + a
4-byte dtoh per draft step). Both knobs only shrink the drafted window; commits are
always the target rows — byte identity by construction, and spec-gate ran GREEN under
every arm.

Sweep on thinkon (interleaved 5×256, plain ≈ 15.6 ms; `adm/ab-spec-k5-adm-*.tsv`):

| arm | ms/token | tok/s | vs plain | notes |
|---|---|---|---|---|
| fixed K=5 (mtp9 state) | 17.69 | 56.5 | **0.87×** | rounds 104, verify 36.5/round |
| pmin 0.3 | 14.85 | 67.3 | 1.05× | 75 guard stops, 27 zero-draft rounds |
| pmin 0.5 | 14.30 | 69.9 | 1.09× | 131 stops, 66 zero-draft |
| pmin 0.7 | 15.31 | 65.3 | 1.02× | over-guarded (126 zero-draft) |
| adapt k_lo=1 | 13.87 | 72.1 | 1.12× | verify shrinks to 24.6 ms/round mean |
| **adapt1 + pmin 0.3** | **13.30** | **75.2** | **1.17×** | chain 2.09, verify 22.7 ms/round |
| adapt1 + pmin 0.5 | 14.03 | 71.3 | 1.11× | |

And the guard does NOT tax the winning shape: thinkoff adapt1 9.80 (1.56×), adapt1 +
pmin0.5 9.70 (1.58×) vs fixed-K5 10.02 — the adaptive window IMPROVES thinkoff (its
fixed knee was K=3; the window self-keys there).

## 5. Ship battery at the chosen policy (adapt k_lo=1 + pmin 0.3), interleaved 5×256

Every shape, plain vs spec-with-admission, dev1 route, spec-gate byte identity GREEN
(5/5 runs) and the vendor-default sampled probe ENGAGED per shape (serving law) —
`ship/*.tsv`:

| shape | plain ms/tok | spec ms/tok | tok/s | vs plain | mtp9 fixed-K5 |
|---|---|---|---|---|---|
| **thinkon** (the model's DEFAULT render) | 15.59 | **13.24** | **75.5** | **1.18×** | **0.87× — the regression this lane was opened on** |
| efflow | 15.52 | 12.74 | 78.5 | **1.22×** | 0.93× |
| long agentic (724-token prompt) | 19.85 | 16.31 | 61.3 | **1.22×** | 0.97× (and OOM before dev1) |
| thinkoff | 15.32 | 9.81 | 101.9 | 1.56× | 1.50× |
| raw goldens (bench-only shape) | 15.07 | 8.69 | 115.1 | 1.73× | 1.78× |

The bar holds: NO shape regresses vs plain; the thinking shape gets +18%, the other
losing shapes +22%; thinkoff IMPROVES (the adaptive window self-keys to its K=3 knee);
raw gives back 4% of its win — and raw continuation is a measurement shape, not a
served one (every real request renders through the chat template).

Why this is also the OWNER'S round-overhead direction and not only admission: the guard
+ window cut the WASTE, which is the same milliseconds. thinkon round-cost at the
policy: chain 6.68 → 2.09 ms, verify 36.5 → 22.7 ms/round mean (smaller windows =
narrower chunks), zero-draft rounds pay ~1 plain step instead of 2.87. The residual
verify inefficiency (weight-shared dense sections at ~2× where ~1.1× is physics) is the
named kernel follow-up — phase-windowed nsys captures are banked for it (`nsys/`).

## 6. FR-Spec mask retry at the prior lanes' corpus scale (owner lever 2) — DISCOVERY still binds; trim stays OFF

The corpus was rebuilt at the ornith-publish scale and beyond (`corpus/owngen-owngen-mtp10.tsv`,
`ranks-owngen-big.txt.gz`, ledger `corpus-ids-big.tsv` on the box): 355 prompts (55
composed + **300 SXC prompts, 75 per owner pool**, 18 classes), greedy 256 + vendor-
default sampled 512 × 2 seeds, chat template on, **prompts up to 940 tokens now
INCLUDED** (the dev1 placement removed the 400-token skip that biased mtp9's ranks
against long contexts). Result: **404,851 counted tokens (the 131,072 floor for topN
32,768 met 3.1×) — but only 11,854 distinct ids discovered** (4.8% of the 248,320
vocab; coverage 1.000 at 16,384 because there is nothing left to rank). The mtp9 rule
("size an own-gen corpus by the distinct-id count it must produce for the VOCAB")
measured again: 4.3× the tokens bought only 2.1× the ids. The 32k-class set that worked
on ornith saturates there because ornith's vocab is 150k and its corpus reached full
coverage; on a 248k vocab, discovery of a 32k set projects to O(4M) own-gen tokens
(~28 GPU-hours at this box's rate) — priced, not re-derived.

Trim A/B at the full discovered set N=11,854 (21× head cut, `trim/ab-trim-*.tsv`,
chains identical to the full-head control in every run — exactness held):

| cell | full head | trim 11,854 | ratio |
|---|---|---|---|
| raw, fixed K=5 (the mtp9 −16.6% twin) | 8.63 ms, accept 0.840 | 9.78 ms, accept 0.610 | **0.882** |
| thinkoff at ship policy | — | — | **0.905** |
| thinkon at ship policy | 13.30 ms | 13.11 ms | 1.014 |

Corpus scale moved the raw loss 16.6% → 11.8% and brought the guarded thinkon cell to
breakeven-plus — the direction is real — but two of the three serving-relevant cells
still lose (unproposable tokens abort exactly the long accept runs that make spec pay),
so **the trim stays OFF**, with its revival condition now a priced number (the ~4M-token
discovery corpus) instead of a hope. Width sweep at 16,384/32,768/65,536 all clamp to
the 11,854 available ids (`trim/trim-sweep-k5-*.tsv`: 99.5-101.3 tok/s vs control
113.0, hist and chains identical across widths).

Guard interplay noted for any revival: under a trim the p-min confidence reads the
TRIMMED softmax (inflated), so the guard fires less (thinkon trim arm accept 0.550 vs
full 0.613 at the same pmin) — a revived trim re-tunes pmin.

## 7. Close-out gates at the merged branch tip (run H, HEAD 35a0b4c98)

origin/main merged into the lane (FLAGS conflict resolved keeping both sides; 252 lib
tests pass), box reset to the merged tip, rebuilt, and the rule-gate battery re-run at
the FINAL recommended config (dev1 + adapt k_lo=1 + pmin 0.3, trim OFF) —
`final/*.tsv`, `run-h.log`:

- Tiny fixture gate: every arm PASS (fixture, mtp-fixture, dir-bf16, dir-nvfp4 stacked
  + per-expert, mtp-dir-bf16, **mtp-spec-tiny byte identity**, mtp-rewind keep=1/2/3,
  kernel oracles).
- raw: verify-bit **24/24 bit-identical**, spec-gate **4/4 byte identity** (accept
  0.730-0.912 under the policy), plain 15.09 vs spec **8.71 ms/token (114.8 tok/s)**,
  sampled probe ENGAGED 54/90.
- thinkon: spec-gate **4/4 byte identity**, plain 15.58 vs spec **13.27 ms/token
  (75.4 tok/s, 1.174×)**, sampled probe ENGAGED 77/153 — the E3 numbers reproduce at
  the merged tip.

## Verdict, and the serving recommendation (owner decision input)

- **The thinking-shape regression is FIXED with receipts**: spec on the vendor-default
  thinkon render goes **0.87× → 1.18×** (75.5 tok/s), efflow 0.93× → 1.22×, the
  724-token agentic shape 0.97× → 1.22× (and runs at all — it OOM'd before dev1),
  thinkoff 1.50× → 1.56×, raw 1.78× → 1.73×. NO shape regresses vs plain; byte
  identity green under every configuration measured.
- **Serving recommendation (plainly): spec ON for every shape**, as ONE global config —
  the card-1 draft placement (`load_from_dir_dev1`) + K=5 ceiling + the bounded
  admission policy `adaptive k_lo=1 + pmin 0.3`. No per-shape switches: the policy
  self-keys per generation (the trace evidence shows the collapse is content-driven and
  prompt-dependent, not shape-static). The FR-Spec trim stays OFF (§6). Dyn-K decay
  stays OFF (built, unused — nothing needed the last-resort bound). The gate binary
  keeps everything explicit-flag (the unadorned instrument remains the regression
  twin); the serving integration, when qwen4_exp is wired for serving, adopts the
  recommended config as its default WITH these receipts.
- **Vs the 200 tok/s owner target: NOT crossed.** Best remains ~115-120 tok/s on the
  friendly raw shape; the honest serving numbers are now 75-102 tok/s on chat-template
  shapes — but they are now uniformly ABOVE plain (49-66 tok/s), which PROFILE-6 could
  not say. Residual levers, in measured order: (1) device-side router/indexer twins
  (≤ ~1/3 of verify wall — the per-forward host-twin bubbles, §2); (2) mt-kernel
  bandwidth efficiency on the weight-shared dense sections (bounded by §2's
  decomposition); (3) the priced 4M-token discovery corpus if the trim is ever revived;
  (4) TP2 t-generic verify (unchanged bound ≤ ~9%).

No claim, price, or roster row changes: qwen4_exp is not served; nothing here is a
customer-visible number.
