# qwen4_exp decode PROFILE-6 — the mtp9 round: FR-Spec draft trim + verify graphs

Lane: mtp9, continuing spec/MTP-SPEC.md. Box: cloud-eval frankfurt, 2× RTX PRO 6000
Blackwell 96 GB, single-card route; artifact `~/data/q48fn-nvfp4` (NVFP4 mint + BF16 mtp
graft). Entry state from PROFILE-5: spec K=5 **8.37 ms/token = 119.50 tok/s** interleaved
×5 over 256 tokens/arm (plain single-card same-run arm 14.86 ms = 67.28 tok/s), accept
0.84 / mean accept len 5.12, byte-identity green on every receipt. Owner target 200 tok/s.

PROFILE-5's residual, in its own measured order: (1) FR-Spec draft-head trim, (2) draft +
verify segment graphs, (3) verify sel restructure (NEGATIVE when tried as warp packing,
mtp6 — not retried), (4) TP2 t-generic verify (bounded ≤ ~9%). This round takes 1 and 2.

## Where the draft's milliseconds actually are (mtp5 spec-profile, sync-bounded)

Re-read before building anything, because the profile's `ms_per_round` column divides the
run's ONE prefill across every round and three of its top rows are prefill, not verify:

| section | ms/round | what it is |
|---|---|---|
| `mtp.lm_head` | **4.757** | the draft's FULL-VOCAB head — **54% of the draft's 8.81 ms/round** |
| `mtp.moe` | 1.813 | the draft layer's 512-expert MoE (DeviceBf16, per-selected-expert) |
| `mtp.qsa` | 1.110 | the draft layer's QSA |
| `mtp.hyper.read` + `.write` | 0.535 | the draft's read/write gates |
| `mtp.fuse` | 0.386 | input fusion |
| `mtp.exit` | 0.205 | the draft's exit mixer |

and the verify's own top rows, over `verify_ms`/round ≈ 39.7: `moe.sel_grouped` 9.555,
`hyper.read` 6.983, `gdn.proj` 5.210, `gdn.conv_scan` 2.982, `moe.shared` 2.596,
`gdn.norm_gate_out` 1.698, `moe.router` 1.540, `lm_head` 1.457, the four `qsa.*` 4.414.

`moe.dequant` 11.05, `moe.expert_gemms` 7.299 and `moe.idx_gather` 2.877 are **prefill**,
not per-round cost: at 177.2 "calls/round" they are 48 layers × ~48 per-expert resolves
from the single prompt prefill, divided by the run's 13 rounds. Reading them as verify cost
would have aimed this round at the wrong section entirely.

## Lever 1 — the FR-Spec draft-head trim

The draft scores only a top-N own-gen rank subset; the TARGET verify stays full-vocab, so
**exactness is untouched by construction** and only ACCEPTANCE can move (a token outside
the trim set is unproposable, i.e. a guaranteed one-round miss). Rows are gathered D2D from
the shared lm head, so a trimmed logit is bit-identical to its full-vocab twin at the same
row and only `out_f` changes: 248,320 → N.

### Corpus provenance (DRAFT-REGIME.md law 1)

Ranks are a distribution artifact of the EXACT artifact and requant, derived from its OWN
generations, chat template ON, covering every prompt CLASS served.

- **Prompt source.** The owner directive (2026-08-14, memory `sxc-corpora-for-rank-mints`;
  `LAW:real-prompts-for-spec` in agent-knowledge) is that rank-corpus prompts come from the
  SXC corpora and the owner agent-session pools. Those are on the RIG, not this box, so
  `spec/extract-sxc-prompts.py` runs there over
  `/home/avifenesh/projects/colbert-2/data/sessions` (4,080 jsonls: hermes 55, claude 1,543,
  codex 1,117, eigen 1,365) with the memory's own filter set (40-6,000 chars, ≥8 words,
  ≥0.6 alpha ratio, skip lines opening `<` `{` `[`, skip system-reminder/command wrappers,
  full-string dedup, whitespace-normalized) and **round-robin interleaving across pools, so
  any truncation of the output stays pool-balanced** rather than collapsing into one pool's
  distribution. 48 prompts, 12 per pool.
- **Plus a composed real-shaped pack** (`spec/make-corpus-prompts.py`): 55 prompts over 14
  classes — code, refactor/review, agentic tool-calling (the tools render with a real
  read_file/run_command/apply_patch tool set), agentic runbooks, reasoning, chat, JSON
  extraction, shell, log triage, translate, writing, the thinking-kwargs matrix
  (`enable_thinking=False` and `reasoning_effort=low` emit different distributions from the
  xhigh default), and long multi-turn agentic sessions with tool results in history. No
  synthetic filler: every prompt is a task a real caller would send.
- **103 prompts over 18 classes in the pack, 97 of them generatable** (6 over the measured
  400-token ceiling were skipped — see the residency finding below), all rendered through the
  artifact's own chat template. Prompts are INPUT ONLY; the counted distribution is the
  engine's own emitted tokens: greedy 256/prompt (the loop-law cap, which is what bounds
  greedy-loop damage in the counts) plus vendor-default sampled **384/prompt × 2 seeds**.
  384 rather than 512 is forced, not chosen: at 512 the longest surviving prompt's state pair
  did not fit the headroom (the OOM sequence below).
- **Each generation is counted only up to and INCLUDING its first end-of-turn id**
  (248046/248044). Post-EOS continuation is off-distribution and would pollute the corpus
  with whatever the model does after a turn it already finished.

### The held-out split, and why the accept numbers are lower bounds

Acceptance measured on a prompt whose own continuation was counted into the ranks is
optimistic by construction — the trim covers that exact chain. So:

- The four banked goldens prompts (`realgate/dump/prompts.tsv`) — **every perf row and
  every spec-gate row in this lane** — are absent from the counted corpus entirely. They are
  also raw continuations while the corpus is chat-shaped, so per law 1 ("ranks inherit their
  corpus MIX") they are the CONSERVATIVE cell: a trim measured on a class it was not derived
  from can only understate itself.
- Six further chat-shaped prompts (2 code, 2 agentic-tools, 2 reasoning) are held out into
  `heldout-prompts.tsv`, selected BY TEXT so reordering a pool cannot pull them back into
  the ranks, giving an in-class held-out cell as well.

## Lever 2 — verify scan-chain segment graphs (`set_verify_graphs`, default OFF)

Sized against this model's OWN graph receipt before building, not against a projection:
PROFILE-2 measured the trunk's decode graphs (84 graphs, host launches/token 2,932 → 531)
at **22.24 → 21.96 ms, +1.3%** — i.e. launch issue on this box is almost fully overlapped
with GPU work, so a graph is worth ~0.12 µs per launch removed. At ~800 launches/round for
draft + verify, a blanket graph build projects to well under 1%, which is why PROFILE-5's
"1-3 ms/round" estimate does not survive contact with the receipt.

What that argument does NOT cover is a **serially dependent** chain, where issue latency
cannot overlap because the next launch's input is the previous launch's output. There is
exactly one such chain in the verify, and it is also the densest: the per-GDN-layer
`dwconv` + t × (`gdn_scan_step_at` + per-column state snapshot) + conv-history roll — 14
launches at t=6, × 36 GDN layers = **576 launches/round**, every column reading and writing
the recurrent state. So the seam is built for that chain only, and its default stays OFF
until its own interleaved A/B says otherwise (FLAGS.md carries both arms and the reason).

Bit-identity is by construction: the chain is one shared closure, so the eager arm and the
captured arm issue the identical launch sequence with the identical parameters and baked
addresses — only the CPU issue path moves. Gates are `--verify-bit-gate` (bit-identical
rows) and `--spec-gate` (byte identity), never a tolerance.

## The corpus that came out (`spec/mtp9/owngen-coverage-final.tsv`)

97 generatable prompts (6 over the 400-token cap were skipped, below), 3 arms each (greedy
256 + sampled 384 × 2 vendor-default seeds), 291 generations, **93,152 counted tokens,
5,538 distinct ids out of a 248,320 vocab (2.2%)**.

| topN | global coverage | law-1 floor (4×topN) | met | per-class range (18 classes) |
|---|---|---|---|---|
| 1,024 | 0.80923 | 4,096 | yes | 0.731 - 0.908 |
| 2,048 | 0.90335 | 8,192 | yes | 0.843 - 0.959 |
| 4,096 | 0.97506 | 16,384 | yes | 0.957 - 1.000 |
| 5,538 (all) | 1.00000 | 22,152 | yes (4.2×) | 1.000 |

The qwen38 coverage law (top-N covers ≥99.5% of own-gen tokens) lands on the full 5,538-id
set: 248,320 → 5,538 is a **44.8× cut** on the draft head's bytes.

## Headline: both levers measured NEGATIVE

Interleaved ×5, 256 tokens/arm, real prompt, K=5, single-card, greedy instrument.

| A/B | OFF arm | ON arm | ratio |
|---|---|---|---|
| FR-Spec draft trim (`ab-trim-k5-n5538`) | full head **8.26 ms = 121.03 tok/s**, accept 0.840, len 5.12 | trim 9.91 ms = 100.93 tok/s, accept 0.561, len 3.82 | **0.8339 (−16.6%)** |
| verify scan-chain graphs (`ab-vgraph-k5`) | eager 9.96 ms = 100.38 tok/s | vgraph 9.97 ms = 100.31 tok/s | **0.9992 (flat)** |

### Trim width table (`trim-sweep-k5`), out-of-class cell — monotone, no rung wins

| draft head rows | tok/s | accept | mean accept len |
|---|---|---|---|
| 248,320 (full) | **116.42** | 0.840 | 5.12 |
| 5,538 | 100.21 | 0.561 | 3.82 |
| 4,096 | 97.41 | 0.550 | 3.76 |
| 3,072 | 83.71 | 0.446 | 3.24 |
| 2,048 | 75.12 | 0.380 | 2.91 |
| 1,024 | 71.28 | 0.360 | 2.81 |

The head did exactly what it was built to do; acceptance paid for all of it and more. 5.12
→ 3.82 committed tokens per round is −25% against a round that got only ~11% cheaper.

### In-class held-out cell — coverage matters, but does not cross zero

Chat-shaped prompts held out of the corpus by text, so the SAME mix as the ranks:

| draft head rows | tok/s | accept | mean accept len |
|---|---|---|---|
| 248,320 (full) | 57.30 (A/B) / 55.17 (sweep) | 0.290 | 2.46 |
| 5,538 | 55.83 (A/B) / 55.70 (sweep) | 0.232 | 2.17 |
| 4,096 | 53.66 | 0.215 | 2.08 |
| 2,048 | 48.08 | 0.181 | 1.91 |

Trim ratio 0.9743. So the loss shrinks **16.6% → 2.6%** when the cell's class is covered —
class coverage is a real component, exactly as law 1 says — but it does not become a win.

### Verdict on lever 1: corpus SCALE is the binding constraint, and law 1's floor does not catch it

93,152 tokens over 97 prompts can only ever DISCOVER 5,538 distinct ids, and 5,538 ids is
2.2% of this vocab. Law 1's "≥4× topN tokens" floor was met **4.2× over** and the trim still
lost, because that floor bounds how well a given topN is RANKED — not whether the corpus is
large enough to discover a topN worth having. qwen38's shipped trim is `frspec-sxc32768`,
6× wider, from a far larger pool. **New rule worth promoting: size an own-gen corpus by the
distinct-id count it must produce for the VOCAB, then apply the 4× floor to that.** For a
248k vocab targeting ~32k ids the floor is ~131k tokens, but the discovery requirement is
roughly an order of magnitude more.

### Verdict on lever 2: launch issue is not this model's decode bottleneck, at any t

The seam was built precisely because the trunk's +1.3%-for-2,400-launches receipt did not
cover a serially dependent chain. It now does: 576 launches/round collapsed to 36 replays
moves nothing (0.9992, inside a 0.11 ms per-arm spread). The graph craft is not the lever
here and PROFILE-5's "est. 1-3 ms/round" is retired with a receipt.

## Rule gates and the shipped config (`spec/mtp9/*-mtp9-defaults.tsv`, `*-mtp9-tp2.tsv`)

At the SHIPPED defaults (no trim armed, verify graphs OFF), the mtp9 code is perf-neutral:

- spec-ab interleaved ×5: plain 14.88 ms (67.22 tok/s) vs **spec K=5 8.34 ms (119.97 tok/s)**
  — mtp7 measured 119.50 on the same cell, so this round's code costs nothing at defaults.
- K ladder (256 tokens): K=3 119.10, K=4 120.69, **K=5 121.55**, K=6 118.87, K=7 113.81,
  K=8 104.55. **The knee is still K=5** (accept 0.840, len 5.12); the trim was supposed to
  move it and did not, because a cheaper draft never arrived in net terms. Accept decays
  0.890 → 0.662 across K=3..8 while mean length only reaches 6.24.
- Sampled vendor-default probe (serving law: temp 1.0 / top_p 0.95 / top_k 20, fixed seed):
  **ENGAGED 54/58 rounds**, accepted 199/290, hist 4,6,12,2,7,27 — identical to mtp7's
  receipt, as a fixed seed should be.
- Tiny arms: 15/15 PASS, and byte-identical between `vgraph` ON and OFF.
- Real checkpoint: verify-bit-gate 24/24 bit-identical; spec-gate byte identity 4/4 prompts
  (accept 0.565-0.846); greedy-gate 4/4.
- tp2-gate: **24/24 argmax, worst rel 3.018e-5, PASS** — the same class as mtp8, so this
  round's kernel touches remain bit-neutral on the TP2 route. Fresh TP2 plain decode-timing
  same run: 12.6 ms (79.67 tok/s).

## THE finding: spec decode is a NET LOSS on this model's vendor-default thinking shape

Residual item 2, run because acceptance (not kernels) is what was left. The SAME four
held-out tasks in three template shapes, full-vocab draft, shipped defaults, interleaved ×5
over 256 tokens/arm (`spec/mtp9/ab-spec-k5-shape-*.tsv`, `mtp9-shapes.log`):

| shape | plain | spec K=5 | spec vs plain | accept hist (rounds by accepted count 0..5) | tokens/round |
|---|---|---|---|---|---|
| **thinkon** (this model's DEFAULT render: "Reasoning effort … xhigh") | 15.28 ms = 65.44 tok/s | **17.65 ms = 56.66 tok/s** | **0.87× — SPEC LOSES** | 235,75,80,45,40,45 → **45% of rounds accept ZERO** | 2.46 |
| **thinkoff** (`enable_thinking=False`) | 15.09 ms = 66.29 | **9.94 ms = 100.63** | **1.52×** | 45,20,30,30,30,140 → 47% accept all 5, 15% accept zero | 4.34 |
| **efflow** (`reasoning_effort=low`) | 15.21 ms = 65.74 | 16.16 ms = 61.88 | 0.94× — spec loses | — | — |

Per-prompt spec-gate at 64 tokens looks healthy in every shape (accept 0.46-0.79, spec
8.7-12.8 ms), which is exactly the trap: **the penalty is a within-generation DECAY.** The
first tokens of a thinking turn are structured and accept well; the deeper the `<think>`
prose runs, the more rounds reject outright, so the longer the window the worse thinkon gets
— 10.1 ms at 64 tokens becomes 17.65 ms at 256. A 64-token cell cannot see this.

Not a greedy-loop artifact, and the direction proves it: a degenerating chain repeats cheap
high-accept tokens and would make spec look FASTER, not slower. The rep-0 chain is also
non-degenerate on inspection, and byte identity held (first_divergence −1 every shape).

Consequences, which are serving decisions and not perf trivia:

- The mtp7/mtp8 headline (119.5-121 tok/s) was measured on a RAW continuation with no chat
  template at all. On this model's own default chat render, spec at K=5 is **slower than
  plain decode**. Nothing in mtp2..mtp8 could have caught it: they shared one prompt file.
- `enable_thinking=False` is where the spec multiplier lives (1.52×). Whether that is a
  shape we may serve is an OWNER call — it changes model behaviour, not just speed — and it
  interacts with the reasoning-effort law (memory `reasoning-effort-unpinned-decode-cell`:
  an unpinned client measures think-prose, not the claim shape).
- Spec admission for qwen4_exp should therefore be SHAPE-AWARE rather than global. A single
  `MEMRA_SPEC_GATE`-style on/off would ship a regression to every thinking request.

### The K knee is SHAPE-DEPENDENT too (`spec-ladder-thinkoff-ladder.tsv`)

K ladder on the one shape where spec wins (thinkoff, 256 tokens, full-vocab draft):

| K | tok/s | accept | mean accept len |
|---|---|---|---|
| **3** | **107.90** | 0.785 | 3.37 |
| 4 | 106.54 | 0.735 | 3.94 |
| 5 | 102.04 | 0.671 | 4.34 |
| 6 | 95.86 | 0.615 | 4.65 |
| 7 | 96.90 | 0.609 | 5.22 |
| 8 | 90.91 | 0.561 | 5.45 |

The knee is **K=3**, not the K=5 this lane ships — so the shipped default is **5.7% off its
own knee on the shape that would actually be served** (107.90 vs 102.04). The mechanism is
simple once seen: the knee tracks ACCEPTANCE, and acceptance is per-shape (0.840 on the raw
goldens prompt, 0.671 here at K=5), so a lower-accept shape wants a SMALLER window because
each rejected draft wastes a bigger fraction of its chunk. **A global K default mis-serves
every shape it was not tuned on.** Sampled vendor-default probe on this shape: ENGAGED 54/77
rounds, accepted 183/385, hist 23,12,10,2,5,25.

## Two further findings that outrank the round's verdicts

### 1. The lane's headline prompt is the friendly one

Every mtp2..mtp8 perf row used the same four banked goldens prompts, and prompts.tsv row 0
accepts **0.840** at full vocab. On chat-template-rendered prompts the same full-vocab draft
accepts **0.290-0.588** (mean len 2.46-3.76) — **55-96 tok/s, not 121**. The 119.5-121
headline is real and byte-gated, but it is the best case, not the serving case, and no
earlier receipt in this lane could have revealed it because they all shared one prompt file.
Agentic and chat traffic is the shape actually sold.

### 2. Spec cannot run long prompts at this residency — now measured, and it inverts a note

The held-out spec-gate **OOM'd on its 495-token prompt** after passing prompts 0-3 byte-
identically, and the own-gen corpus had to skip 6 prompts over 400 tokens for the same
reason: trunk state + draft state + verify stash + prefill transients do not fit the
~2.6 GiB left after the NVFP4 trunk and the 5 GB device-bf16 draft bank go co-resident on
card 0 (`skipped_over_max_prompt=6, indices_lens=[(76,724),(60,691),(53,675),(54,602),
(34,512),(35,502)]`; the OOM's exact position is in `mtp9-heldout.log`).

MTP-SPEC.md carried this as "long-context serving would move the draft bank to card 1
(free) — noted, unmeasured". It is now measured, and it inverts: **the two-card placement is
a PREREQUISITE for spec on agentic-length prompts, not an optimization.** Card 1 is idle.

## Verdict vs the 200 tok/s owner target: NOT crossed, and this round moved it 0

Measured best remains **119.97-121.55 tok/s** at K=5 on the favourable prompt (55-96 on
chat/agentic shapes). PROFILE-5's residual items 1 and 2 are now closed NEGATIVE with
receipts, which removes ~30-45 tok/s of *projected* headroom from the stack rather than
adding any. Honest residual, re-ordered by what the measurements now say:

0. **OWNER CALL FIRST: does spec ship for qwen4_exp at all, and on which shapes?** The
   shape cells make this a decision, not a tuning task: spec is 0.87× on the default
   thinking render and 1.52× with thinking off. Serving it globally would regress every
   thinking request. Options are shape-aware spec admission, serving `enable_thinking=False`
   (a behaviour change, hence an owner call), or holding spec back for this model.
1. **Raise acceptance on the THINKING shape** — now the top engineering lever by a wide
   margin. 45% of thinkon rounds accept zero drafts, so the draft is being asked to predict
   reasoning prose it has no signal for. Vendor think-replay (the qwen38 precedent in memory
   `dspark-session-reuse-truths`) is the first candidate; a think-prose-weighted own-gen
   corpus is the second (and unlike the general trim, it targets the shape that is losing).
2. **Move the draft bank to card 1.** Not a perf lever first — a CAPABILITY one: it is what
   makes spec work at all on agentic-length prompts (finding 2), and it frees ~5 GB on card
   0 for a bigger verify stash (higher k_cap) at the same time. Card 1 is idle today.
3. **A much larger own-gen corpus, if the trim is revisited at all.** ~10× this corpus to
   discover a ~32k-id set; on this box that is ~7+ GPU-hours of generation and it needs the
   card-1 placement first so long prompts can participate (they are currently skipped, which
   is itself a coverage gap that biases any trim against long contexts).
4. **Verify sel restructure** — still the largest verify slice (`moe.sel_grouped` 9.555
   ms/round, ~25% of verify) but at ~86% of its bytes floor by the slot arithmetic, so the
   union-of-experts gather is worth ~5-15%, not a multiple. Warp packing was already
   NEGATIVE (mtp6).
5. **TP2 t-generic verify** — unchanged bound, ≤ ~9%, and it becomes more attractive only
   after (1) since that lane already puts weights on card 1.

No claim, price, or roster row changes: nothing here is a customer-visible number.

## Method notes banked from this round's own failures

- **Own-gen OOM was fragmentation, not capacity.** The first corpus pass died at prompt 42
  of 66 after 40 minutes: each generation sized its state to its own prompt, so 252
  differently-sized allocate/free cycles fragmented the ~2.6 GiB of post-load headroom and
  the first larger prompt class (the 317-512 token tools renders) had no block left to fit.
  Prompt 42 was not bigger than the run's peak — it was bigger than the holes. Fix: ONE
  capacity for every generation in the run. Second fix, because 40 GPU-minutes should never
  be a total loss: `--owngen-corpus-out` banks each finished generation's counted ids and
  the next invocation counts those rows from the file instead of regenerating them.
- **Corpus extension is free if indices are stable.** Owner prompts append AFTER the
  composed rows, so composed indices are byte-identical between the two packs and the resume
  ledger keeps every generation already banked. Verified by diff before relaunching.
- **A silent zero in a corpus extractor drops a whole class.** The SXC extractor's first run
  printed "48 owner prompts" and looked fine with claude and codex at ZERO, because those
  pools key their transcripts by project subdirectory and a flat listdir found no `.jsonl`.
  It now walks recursively AND exits non-zero on any pool that has files but yielded nothing.
