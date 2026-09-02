# qwen4_exp FR-Spec draft-head trim at corpus scale (memra#61, 2026-09-02)

The draft's full-vocab LM head (`mtp.lm_head`, 248,320 rows, run every chain step) is
**9.8% of the K=5 round** on the serving program (`../../ep2/spec-profile-k5-ep2A-rounds.tsv`,
thinkon shape, seams `idxsel,kvq,idxq`) plus `lm_head` 2.6% for the trunk verify. FR-Spec
trims the DRAFT head to a frequency-ranked subset. Exactness is untouched by construction:
the verify chunk stays full-vocab and the accept walk compares against it, so a trim can
only move ACCEPTANCE (`build_draft_trim` doc, and gated per arm below).

`../MTP-SPEC.md` measured this lever NEGATIVE twice and both receipts stand:

| lane | ranks | corpus | raw fixed-K5 | thinkoff ship | thinkon ship |
|---|---|---|---|---|---|
| mtp9 | N=5,538 | 93,152 own-gen tokens | 0.834 | | |
| mtp10 | N=11,854 | 404,851 own-gen tokens | 0.882 | 0.905 | 1.014 |

and named the binding constraint: **DISCOVERY on this 248,320-row vocab**, with a 32,768-id
set priced at ~4M own-gen tokens ~= **28 GPU-hours**. This lane spent **0 GPU-hours on the
mint** and reopened the lever at 32,768.

## 1. The mint (CPU only)

Method: the GLM-5.3 ranks mint (darklanes `research/glm53-ranks-mint-20260830/MINT.md`,
2026-08-30) ranks the ASSISTANT-SIDE emissions of the owner SXC session pools under the
model's own tokenizer. `extract_corpus.py` here is that extractor re-rendered for THIS
family's emission shape, read off the checkpoint's own `chat_template.jinja` assistant
branch (`{reasoning}\n</think>\n\n{content}`, then
`<tool_call>\n<function={name}>\n<parameter={k}>\n{v}\n</parameter>\n</function>\n</tool_call>`
per call, terminated `<|im_end|>`; the prompt supplies the opening `<think>\n`, so a corpus
line starts inside the reasoning block and never with `<think>`).

Corpus (`/home/avifenesh/projects/colbert-2/data/sessions/{claude,codex,eigen,hermes}`,
4,922 jsonls at mint time; `extract-stats.json`, `corpus/` is byte-reproducible and not
committed):

| class | tokens | distinct ids | top-32,768 covers | pools (tokens) |
|---|---|---|---|---|
| agentic | 36,993,491 | 56,769 | 99.69% of corpus | claude 15,438,295 / codex 15,772,595 / eigen 5,077,049 / hermes 705,552 |
| prose | 5,479,689 | 37,313 | 99.92% of corpus | claude 1,494,942 / codex 3,273,263 / eigen 556,379 / hermes 155,105 |

Turns extracted: claude 64,696 / codex 51,365 / eigen 16,704 / hermes 2,617 (claude and
codex capped at 48 MiB extracted each so the class stays pool-balanced).

**The corpus solves discovery: 56,769 distinct ids, 4.8x the 11,854 that 405k own-gen
tokens found.** It does NOT solve distribution, which is the finding in §3.

### Classes

| file | sha256 | size | what |
|---|---|---|---|
| `q4e-ranks-ogblend-32768.txt` | `75a47c461dd9247d948288f24f8897e4720826512993c66f4514f243ead837bc` | 188,191 | **the lane's artifact**: `0.5*normfreq(mtp10 own-gen) + 0.5*normfreq(mixed corpus)` — this model's OWN 404,851-token emission distribution for the head, the corpus for the tail it never discovered |
| `q4e-ranks-sxc32768.txt` | `fb84ad0efd8d8e4c64dd35bc4ecc6ac8fc2dabc7ad5a88b8c87c2d1edbebfbfc` | 188,202 | pure agentic corpus (the q38 `sxc32768` shape) — kept as the measured CONTROL, not a candidate |
| `q4e-ranks-prose-32768.txt` | `55064a2e73a88951f9e7a12d54ba2f997bb291e5cc8b11f23897f56e03a869fd` | 188,076 | pure prose corpus |
| `q4e-ranks-mixed-32768.txt` | `8c66b2bc5f0dc5b35cb0135d8dcaaef7e5ce6133990163e3966d5ec5acc09b03` | 188,178 | 50/50 agentic+prose blend |

Format: one token id per line, rank order, no header — what `read_ranks` and the loader's
txt arm parse. 32,768 unique ids each, every id < 248,320 (max id emitted 248,076, so the
`d2t < lm_head rows` assert clears with 244 rows of head padding to spare).

Rank law replicated from `memra_gguf::d2t::rank_top_n`: sort the WHOLE tokenizer id space
by (count desc, id asc), take the first N — unseen ids pad ascending into cover slots.

### Mint gates

1. **House tool** — `frspec-rank <artifact_dir> <out>.gguf 32768 corpus/<class>` from this
   lane's own release build. Reads the tokenizer memra actually loads (the artifact dir),
   so the counted id space is the SERVED one.
2. **Cross-implementation equality** — HF `tokenizers` 0.23.1 against the VENDOR
   `tokenizer.json` (`Qwen/Qwen3.8-Flash-Next@de4b8e4d43b9...`, sha256
   `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`), same read law (one
   encode per FILE, `add_special_tokens=False`), same rank law:
   - class totals and distinct-id counts **identical** (agentic 36,993,491 / 56,769;
     prose 5,479,689 / 37,313), per-file totals identical for all 8 files;
   - prose top-32,768 file **byte-identical**;
   - agentic top-32,768 **SET identical**, file differs at 639 of 32,768 positions, ALL
     inside equal-count tie groups, traced to exactly 4 ids whose counts differ by one:
     id 31374 `"י"` (140 house / 141 oracle) and the byte-fallback pieces 146, 112, 105.
     That is 1.1e-7 of the corpus, it needs whole-file context (both encoders agree
     exactly on the 611 yod-carrying lines in isolation, 25,969 tokens, and on a
     combining-mark sample, 806 tokens), and it cannot move the trim because the trim
     consumes the SET. The committed files carry the ORACLE ordering (vendor counts).
3. **Read-law calibration, because it bit once**: encoding line-by-line instead of
   whole-file inflates the corpus by 8.6% (BPE cannot merge across the newline joining two
   turns) and made the two implementations disagree for a reason belonging to neither. The
   four-way control on one 5 MB slice — memra+artifact, memra+vendor, HF+artifact,
   HF+vendor — returns exactly 705,552 tokens.
4. **Mechanical**: 32,768 unique ids per file, all in range, rank order.
5. **Can't-hallucinate eyeball**: the ogblend head decodes to `13 11 198 15 279 25 220 271`
   = the punctuation/whitespace/`the` class every real emission stream is made of.

### TRAP found in passing: the artifact's tokenizer.json is not the vendor's file

`~/data/q48fn-{nvfp4,yarn1m}/tokenizer.json` (sha256 `06b9509352d2...`) has the SAME vocab
(248,044 entries) and the SAME merges as the pinned vendor revision, but a **Qwen2-era
`pre_tokenizer` regex** (`[^\r\n\p{L}\p{N}]?\p{L}+` — no `\p{M}`) and different ByteLevel
`decoder` flags (`add_prefix_space/trim_offsets/use_regex` all true vs all false).

Serving is unaffected: memra takes the pre-tokenizer from `tokenizer_config.json`'s
`pretokenize_regex` and only checks that `tokenizer.json`'s is ByteLevel, so memra's ids
match the vendor's (measured: 806 = 806 on a combining-mark sample, where HF pointed at the
ARTIFACT file returns 816). The trap is for TOOLING: any HF-based oracle pointed at the
artifact directory silently tokenizes marks-carrying text differently from the served path.
Point oracles at the vendor revision, as this lane did.

## 2. The free estimator (why this is not another 28-GPU-hour own-gen run)

    tok/s_trim / tok/s_full = (1 - A*q) / (1 - H*(1 - N/V))

- `q` = out-of-set share of the target's own emitted tokens, measured on every real chain
  banked in this lane (`# rep0_*` / `# ids` lines of mtp10/mtp11 receipts: raw 1,280 tok,
  thinkon 1,792, thinkoff 1,792, efflow 1,280, long 512). A target token outside the
  trimmed head cannot be proposed, so it is a guaranteed miss at that chain step.
- `A` = per-shape amplification of `q` into mean-accept-length loss, **calibrated on the
  two banked negatives** (one out-of-set pick also derails the rest of that carrier chain).
- `H` = full-vocab draft-head share of the round, solved from each shape's own
  `draft_ms_share` full-vs-trim pair (head cost is linear in rows).
- `V` = 248,320.

| shape | q at N=11,854 | measured accept-len loss | A | H | refit ratio | mtp10 measured |
|---|---|---|---|---|---|---|
| raw (fixed K=5) | 0.1477 | 20.70% | 1.402 | 0.0945 | 0.8714 | 0.8824 |
| thinkon (ship) | 0.0954 | 4.69% | 0.491 | 0.0630 | 1.0140 | 1.0144 |
| thinkoff (ship) | 0.1563 | 27.25% | 1.744 | 0.0945 | 0.7995 | 0.9050 |

It refits raw and thinkon (within 1.1 and 0.04 points) and **under-predicts thinkoff by 10
points**, for a reason worth keeping: `predict()` holds round cost fixed apart from the
head, which is exact where the draft window is fixed (raw) or already at the adaptive floor
(thinkon, accept-len 1.92 at `k_lo=1`) but conservative where the window sits above the
floor — a lower-accept arm also drafts a SMALLER window, so part of the accept loss comes
back as a cheaper round. Predictions for thinkoff-like shapes are therefore floors.

Predicted ratios at the chosen width (full grid in `oracle-report.json`; the head saving
shrinks as N grows while coverage rises, so the knee is interior and lands at 32,768):

| shape | class | 8,192 | 16,384 | 24,576 | **32,768** | 49,152 | 65,536 |
|---|---|---|---|---|---|---|---|
| raw | sxc (corpus only) | 0.632 | 0.711 | 0.773 | **0.820** | 0.814 | 0.809 |
| raw | ogblend | 0.845 | 0.888 | 0.964 | **1.015** | 1.008 | 1.002 |
| thinkon | sxc | 0.991 | 1.034 | 1.050 | **1.054** | 1.049 | 1.045 |
| thinkon | ogblend | 1.024 | 1.041 | 1.054 | **1.052** | 1.048 | 1.043 |
| thinkoff | sxc | 0.526 | 0.637 | 0.764 | **0.785** | 0.780 | 0.775 |
| thinkoff | ogblend | 0.734 | 0.904 | 0.964 | **1.025** | 1.018 | 1.011 |

## 3. Coverage of the model's OWN emission mass — the finding the corpus does not fix

Share of mtp10's 404,851 own-generated tokens whose id is inside the top-N:

| top-N | sxc (corpus) | prose | mixed | ogblend |
|---|---|---|---|---|
| 4,096 | 0.7351 | 0.7670 | 0.7668 | 0.9193 |
| 8,192 | 0.8299 | 0.8667 | 0.8482 | 0.9704 |
| 16,384 | 0.9225 | 0.9261 | 0.9261 | 0.9978 |
| 32,768 | 0.9654 | 0.9657 | 0.9673 | 1.0000 |
| 65,536 | 0.9654 | 0.9657 | 0.9673 | 1.0000 |

The corpus classes PLATEAU at 0.965-0.967: **3.4% of what this model emits sits on ids that
36.9M tokens of other agents' sessions never emitted at all**, so no width of a
foreign-corpus ranking can reach them. That is why the pure-corpus class is predicted to
lose on code-shaped output (raw 0.820) while the own-gen-headed blend wins (1.015) at the
same width, and why cell C measures exactly that pair.

## 4. Battery

Card 0 of the lane box, serving caches (`MEMRA_Q4E_SEAMS=idxsel,kvq,idxq`; selgroup
default-ON since PR #56, nothing extra pinned), draft co-resident with the trunk
(`mtp_dev1=false` — one card, and every shape prompt here is 43-105 tokens, well inside the
~400-token co-resident ceiling `../MTP-SPEC.md` measured). Script: `frspec-cells.sh`.
Instrument changes in this lane's commit: the trim A/B now FLIPS the arm order on odd reps
and prints each arm's own spread, and the vendor-default sampled probe runs on BOTH trim
arms in one boot instead of on whichever arm the CLI left live.

### Pass 1 (`box-pre57/`, branch based on 24d775458 — before PR #57's StepPool shed)

x3, arm order flipped per rep, chains byte-identical in every arm
(`rep0_full_vs_trim_first_divergence = -1`), spec-gate byte identity 4/4 at 256 tokens with
the trim ARMED, sampled twin engaged on both arms:

| cell | shape | policy | ranks | full tok/s | trim tok/s | ratio | accept full->trim | len full->trim | draft share |
|---|---|---|---|---|---|---|---|---|---|
| A | thinkon | adapt k_lo=1, pmin 0.3 | ogblend | 81.79 | 86.09 | **1.0525** | 0.608 -> 0.603 | 1.94 -> 1.92 | 0.12 -> 0.07 |
| D | thinkoff | adapt k_lo=1, pmin 0.3 | ogblend | 98.01 | 103.87 | **1.0598** | 0.689 -> 0.689 | 2.88 -> 2.78 | 0.16 -> 0.07 |
| B | raw (bench) | fixed K=5 | ogblend | 128.64 | 128.97 | 1.0026 | 0.840 -> 0.745 | 5.12 -> 4.65 | 0.18 -> 0.09 |
| C | raw (bench) | fixed K=5 | **sxc (corpus only)** | 128.75 | 127.64 | 0.9913 | 0.840 -> 0.745 | 5.12 -> 4.65 | 0.18 -> 0.09 |

Every arm's within-arm spread cleared the 0.5% escalation threshold (thinkon 0.510/0.440,
thinkoff 1.832/3.439, raw 1.773/3.040, sxc-raw 2.088/2.226) and the raw verdict sat inside
the pooled spread, so rules (a) and (b) of LAW:interleave-x3-default both fired and the
pairs are re-cut at x5 below.

### Pass 2 — the claim rows, x5 on the shipped program (rebased onto v0.124.0 / c04c1da9b)

`qwen4exp_real_gate.frspec2`, binary sha256 `ba3d443bf309bf8d2f7001723f64d07f223770e8430d25f4a091dcfa30260485`.

| cell | shape | policy | full tok/s | trim tok/s | ratio | accept full->trim | len full->trim | spread full/trim |
|---|---|---|---|---|---|---|---|---|
| A2 | thinkon | adapt k_lo=1, pmin 0.3 | 84.62 | 90.23 | **1.0663** | 0.599 -> 0.611 | 1.94 -> 1.94 | 0.750% / 0.272% |

At x5 the full arm still spreads 0.750%, but the verdict margin (6.63%) is ~6x that and well
outside 2x the pooled spread, so rule (b) does not fire again. Per-rep sign is 5/5 for the
trim, in both arm orders. **Acceptance did not pay for the head this time**: accept rate
rose 0.599 -> 0.611 and mean accept length was unchanged at 1.94, because at this width the
only chain steps the trim can move are those whose full-vocab top-1 is out of set, and its
best in-set token matches the target about as often as the full head's did.

Supporting receipts, same boot: width sweep (below), hidden-state gate 10/10 argmax
agreement, `spec-gate` byte identity 4/4 at 256 tokens with the trim armed (accept
0.611/0.652/0.790/0.754 across the four thinkon prompts), and the vendor-default sampled
twin on BOTH arms — full 79.65 tok/s with 34/65 accepting rounds, trim 84.12 tok/s with
39/66, 128 tokens each (above the token floor of TRAP:short-sampled-row-fakes-tok-s; the
arms share one boot, and this row is an engagement receipt, not the perf claim).

### Width: chosen, not inherited

thinkon at ship policy, one run per width plus the full-vocab control, shipped program:

| draft head rows | tok/s | accept | mean accept len | draft share | chain == control |
|---|---|---|---|---|---|
| 248,320 (control) | 81.88 | 0.599 | 1.94 | 0.12 | yes |
| 8,192 | 86.26 | 0.584 | 1.91 | 0.06 | yes |
| 16,384 | 87.12 | 0.596 | 1.94 | 0.07 | yes |
| 32,768 | 88.20 | 0.611 | 1.94 | 0.07 | yes |

### Round-cost reconciliation

The profile puts `mtp.lm_head` at 9.8% of a FIXED-K=5 round. Under the ship policy the p-min
guard and the accepted+1 window shorten the chain, so the head is a smaller share of the
round: the measured `draft_ms_share` moves 0.12 -> 0.07 at 32,768, i.e. ~5 points of the
round, and the trim collects it. The two numbers describe different round shapes, not a
discrepancy.
