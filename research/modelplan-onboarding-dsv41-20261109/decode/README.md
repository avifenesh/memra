# DeepSeek-V4.1-Flash onboarding, step 6 (GPU): the decode loop, gated against the vendor's generations (2026-11-09)

Verdict: the V4.1 GPU route now generates. `dsv41_gpu::decode` runs a prefill and then one token per
call, each sequence's window rings, compressor groups, compressed KV, index keys and token history
carried as the vendor's `model.py` carries them, on one rank or N (the TP/EP partition). On one card,
teacher-forced with the vendor's own greedy tokens, its per-step distributions sit inside the vendor
program's own floor: 1.14x the floor's mean KL on 517 chat steps (16 to 18,343 tokens of context),
1.21x on 194 long-context steps (41k to 128k), and 1.21x the vendor's CPU-vs-GPU spread on the early
steps where the two GPU schedules coincide; at 257,185 tokens its two decode steps sit at or under
the floor. Free-running greedy generation reproduces the vendor's tokens exactly on 11 of 18 tapes
(every needle, 3.7k to 257k, included) and parts from them on the other 7 at a vendor near-tie each
time, 5 of them at a step where another valid run of the vendor program picks memra's token too.
Per layer, below and across the window, every decode step's attention index row, selection,
compressor emission and Engram hash ids equal the vendor decode program's bytes (122 index rows over
positions 16 to 131), and the same check rejects a loop that re-runs prefill semantics at every
position where the vendor's two programs differ. Replaying a generation layer by layer gives every
saved step back byte for byte, and N = 2 ranks give the one-rank bytes. The support state stays as it is: this is the decode
surface, not a serving path.

Hardware: RTX PRO 6000 Blackwell Max-Q (sm_120a, 96 GB), CUDA 13.4, rustc 1.97.1; the chat cases on
one card and the long cases on another, one card per run (the vendor runs on the same cards). Every
time below is a single run with the weights streamed from the page cache, not timed for publication.

## The loop

`crates/memra-engine/src/dsv41_gpu/decode.rs` (`Runner`, `Seq`, `Call`):

- A call at position 0 is the vendor's prefill program. Every later call is one token, the
  vendor's decode program: the window ring read whole, oldest slot first, empty slots -1
  (`get_window_topk_idxs` at `start_pos > 0`); the ratio-2 compressor's slot update, pooling when a
  pair completes; the indexer over `end_pos // ratio` keys with the candidate pool of this call;
  one Engram hash row from the token history. A generated token is always fed as a decode call.
  More than one token at a position past 0 is not a vendor program and is refused.
- What a sequence carries: per layer, the window ring (fp8-quantized KV, position p at slot
  p % 128) and the open compressor pair (fp32 kv and gate rows); per kv-source group (layers 2, 8,
  14, 20), the compressed KV and the index keys, appended as groups complete. The vendor keeps one
  cache per kv source and points its shared slots at the current group's; the layer code keeps one
  `Shared`, so the runner swaps each group's half in for its layers (`Shared::swap_group`) and keeps
  the per-forward half (selection, candidate pool) per call.
- `Runner::forward` runs a list of calls layer by layer: a layer's weights load once (or stay
  resident) and every call runs through it before the next layer, a sequence's calls in order. A
  call reads only state its own and earlier calls wrote, so a sequence's known tokens fed as many
  calls of one forward (teacher forcing, or replaying a saved generation) give the bytes of feeding
  them one forward at a time, with the weights streamed once instead of once per token. Every call
  runs alone through each block: batching sequences into one forward never changes a sequence's
  numbers.
- The ranks are a `tp::Group`. One rank is the one-card program (the grouped expert dispatch,
  byte-identical to the per-expert loop, `../tp/README.md`); N ranks the TP/EP partition.
- `dsv41-run` drives it: `--force FILE` (teacher-forced decode, all steps in one layer-by-layer
  pass), `--gen N` (greedy, stopping at `<｜end▁of▁sentence｜>`), `--cases FILE` (several sequences
  per run), `--resident N` (keep N layers on the card during the decode forwards), `--steps K`
  (stop after K decode forwards; a rerun with the same `--out` replays the saved tokens layer by
  layer, refuses to continue unless every replayed step comes back byte for byte, and goes on).
  Each step writes its top-32 ids and values, lse (f64), argmax, the token taken, and the full
  logits of the first 8 steps (the tapes' layout). The prefill-only paths (`--last-logits`,
  `MEMRA_ORACLE_OUT`, `--tf`) run on the same runner.

## Identity gates

| gate | result |
|---|---|
| prefill-only logits after the refactor onto the runner, against the banked one-card runs | identical sha256: capital (16 tokens) `a3de6c00...`, needle1200 (18,343) `50657b54...` (`receipts/runs/smoke.sha256`) |
| layer-by-layer replay against token-by-token generation (the resume check on every saved step: top-32 ids and values, lse, argmax, token, full logits) | byte-identical on all 704 greedy steps: 523 chat (13 sequences, the final verify run after 5 resumable leases), 178 long (4 sequences, after 4 leases) and 3 at 257k; every intermediate resume passed the same check (`receipts/runs/greedy-*-verify.log`) |
| memra's teacher-forced decode against memra's greedy run, on the steps fed the same tokens (separate processes, other batches of sequences) | byte-identical on all 203 shared steps of the 18 tapes (`forced_vs_greedy_identical` in `receipts/gates/{chat,long}.json`) |
| the committed binary against the receipts' runs (the chat greedy replay, and the chat teacher-forced run into a fresh directory) | byte-identical: the committed build (`dsv41-run` sha256 `b17478ec...`) replays the 523 chat greedy steps and rewrites all 65 chat step files that the earlier builds wrote (`receipts/runs/final-check.log`). The runs span three builds of this branch that differ by formatting, two loops rewritten as iterators and `Ctx::window_row_decode` factored out; the long and 257k verify runs also ran on the committed build |
| a decode run's step 0 against the step-4 runner's last-token logits (`--last-logits`) | identical at 18,343, 73,520, 128,745 and 257,185 tokens (`receipts/gates/step0-vs-banked-prefill.txt`) |
| the decode program's integer outcomes per layer against the vendor's (index rows, selections, compressor emissions, Engram ids) at positions 16 to 131, and a re-prefill red arm | 0 of 540 checks failed; the red arm rejected wherever the vendor's two programs differ (next sections) |
| N = 2 ranks on one card against N = 1, teacher-forced capital, gsm, hebrew (25 steps; every step file) | byte-identical (`receipts/runs/native-chat.log`, no `TP2 DIFFERS` line) |
| decode steps above the window against memra's own prefill of the same tokens (`decode_vs_prefill.py`) | 96 / 96 and 92 / 92 steps with bitwise-equal top-32 at 41k and 64k (`receipts/gates/decode-vs-prefill-*.json`) |

The last row says that past the window memra's decode and prefill programs are one program per
row (the ring order is position order there, and every memra kernel acts per row). The vendor's two
programs differ there too, by torch's shape-dependent GEMM kernels: its own prefill reproduces its
decode on 93/96 and 85/92 steps (`../gpu/README.md`).

## The floor and the band

The vendor reference is the tape's own oracle, teacher-forced with the tape's tokens
(`make_forced_oracle.py`: the oracle's generation loop takes the token of each step from a list and
also keeps each step's lse). It rewrites every tape exactly: the same top-32 ids and values on all
714 steps of the 18 tapes ("ref = tape" below), now with the lse a per-step KL needs.

The floor is the same vendor program in another valid realization, teacher-forced with the same
tokens: the global-max attention oracle (`../gpu/make_globalmax_oracle.py`, the schedule floor of
the prefill gates). Per step: KL(vendor || other) from both sides' top-32 plus one rest bucket (a
lower bound of the full KL, `../gpu/tf_kl.py`'s estimator, the same for memra and for the floor),
the exact KL on the 8 steps with full logits, and top-1 agreement with the vendor's greedy token.

Band, the rule of the prefill gates: a group passes when memra's mean KL is at most twice the
floor's and its top-1 agreement at most 1 point under the floor's (two binomial standard errors
when the group is small). Twice is the factor of every gate in this lane (the layer gates, the
checkpoint band, the prefill gates): memra differs from the reference in several places at once
(GEMM summation orders, the attention schedule, the fp8 roundings those move) where the floor
differs in one. What the band can and cannot see is measured below: it cannot tell the vendor's
decode program from its prefill program (the prefill arm).

The schedule floor is degenerate on short contexts: while an attention row holds at most 64 valid
keys, one key block, the tilelang schedule and the global max are the same arithmetic, so the floor
is exactly zero at a short prompt's prefill and small on the next steps. There the backend floor
applies, the one the layer gates and the checkpoint band use: the vendor program on the CPU
(`cpu_floor_kl.py`, the forced oracle with `--device cpu`, first 8 steps of four chat cases).

## Teacher-forced decode

Pooled per group, every step (the KL lower bound of `decode_gate.py`; `receipts/gates/chat.json`,
`long.json`):

| group | tapes | steps | KL memra | KL floor | ratio | top-1 memra | top-1 floor | band | verdict |
|---|---|---|---|---|---|---|---|---|---|
| chat, 16 to 18,343 tokens | 13 | 517 | 1.63e-02 | 1.43e-02 | 1.14 | 97.3% | 97.5% | KL <= 2.86e-02, top-1 >= 96.1% | PASS |
| chat, vendor entropy > 1 nat | | 22 | 9.84e-02 | 9.50e-02 | 1.04 | 77.3% | 81.8% | KL <= 1.90e-01, top-1 >= 65.4% | PASS |
| long, 41,098 to 128,745 tokens | 4 | 194 | 3.68e-02 | 3.05e-02 | 1.21 | 93.8% | 96.4% | KL <= 6.10e-02, top-1 >= 93.7% | PASS, by 0.1 point on top-1 |
| long, vendor entropy > 1 nat | | 9 | 3.96e-01 | 3.83e-01 | 1.03 | 33.3% | 44.4% | KL <= 7.66e-01, top-1 >= 11.3% | PASS |
| 257k needle (needle16800), 257,185 tokens | 1 | 3 | 1.62e-04 | 3.64e-05 | 4.44 | 100% | 100% | KL <= 7.29e-05, top-1 >= 99.0% | over the KL bar on its prefill row; see below |

Per tape (KL mean / max over the steps; first miss = the first step whose argmax is not the vendor's
token):

| tape | prompt | steps | ref = tape | KL memra | KL floor | top-1 memra | top-1 floor | first miss memra / floor |
|---|---|---|---|---|---|---|---|---|
| capital | 16 | 2 | yes | 1.36e-05 / 2.72e-05 | 3.34e-06 / 6.68e-06 | 2/2 | 2/2 | - / - |
| gsm | 42 | 20 | yes | 2.50e-02 / 3.77e-01 | 3.39e-03 / 3.63e-02 | 19/20 | 20/20 | 5 / - |
| code | 24 | 42 | yes | 1.95e-03 / 5.05e-02 | 3.04e-04 / 1.27e-02 | 42/42 | 42/42 | - / - |
| copy | 41 | 30 | yes | 4.42e-09 / 2.97e-08 | 5.15e-09 / 5.92e-08 | 30/30 | 30/30 | - / - |
| hebrew | 16 | 3 | yes | 2.37e-07 / 4.07e-07 | 4.20e-08 / 1.22e-07 | 3/3 | 3/3 | - / - |
| gsm_think | 68 | 50 | yes | 2.64e-03 / 2.47e-02 | 3.31e-03 / 3.99e-02 | 49/50 | 50/50 | 3 / - |
| story | 22 | 138 | yes | 3.19e-02 / 1.01e+00 | 3.42e-02 / 6.56e-01 | 132/138 | 132/138 | 19 / 19 |
| hebrew_para | 27 | 155 | yes | 1.89e-02 / 4.69e-01 | 1.50e-02 / 3.23e-01 | 150/155 | 148/155 | 23 / 13 |
| code_pal | 33 | 53 | yes | 7.03e-03 / 3.02e-01 | 2.19e-03 / 6.69e-02 | 52/53 | 53/53 | 19 / - |
| sys_json | 31 | 15 | yes | 7.61e-04 / 1.09e-02 | 4.23e-05 / 3.73e-04 | 15/15 | 15/15 | - / - |
| multiturn | 45 | 3 | yes | 1.40e-04 / 4.21e-04 | 2.88e-04 / 8.65e-04 | 3/3 | 3/3 | - / - |
| needle | 3,761 | 3 | yes | 2.87e-07 / 7.31e-07 | 3.68e-07 / 8.96e-07 | 3/3 | 3/3 | - / - |
| needle1200 | 18,343 | 3 | yes | 1.28e-06 / 3.38e-06 | 3.40e-06 / 9.87e-06 | 3/3 | 3/3 | - / - |
| decode32k_he_summary | 41,098 | 96 | yes | 5.52e-02 / 2.29e+00 | 4.13e-02 / 1.28e+00 | 90/96 | 92/96 | 13 / 13 |
| decode64k_en_continue | 64,443 | 92 | yes | 1.99e-02 / 4.78e-01 | 2.12e-02 / 3.10e-01 | 86/92 | 89/92 | 4 / 4 |
| needle4800 | 73,520 | 3 | yes | 2.11e-07 / 4.32e-07 | 4.09e-07 / 1.02e-06 | 3/3 | 3/3 | - / - |
| needle8400 | 128,745 | 3 | yes | 6.74e-05 / 1.98e-04 | 3.45e-05 / 1.02e-04 | 3/3 | 3/3 | - / - |
| needle16800 | 257,185 | 3 | yes | 1.62e-04 / 4.78e-04 | 3.64e-05 / 7.09e-05 | 3/3 | 3/3 | - / - |

The 257k needle's three steps are three near one-hot positions (the vendor puts 0.9988, 0.99998 and
0.9999999 on its token), and its row is over the KL bar as a number. Step by step, exact KL on the
full logits (memra / floor): step 0, the prompt's prefill row, 5.3e-04 / 1.4e-04, byte for byte the
step-4 runner's 257k logits, the single sample that step 4 judged on the 262,144-position
teacher-forced text instead, where memra sits at 0.93x-0.94x of this floor (`../gpu/README.md`); the
two decode steps at positions 257,185 and 257,186, 7.8e-06 / 3.9e-05 and 7.1e-08 / 6.6e-08. The decode
steps at 257k are at or under the floor.

The long group passes its top-1 rule by 0.1 point (93.8% against a 93.7% bar). Its misses sit on
the two long generations, where the floor misses too: memra 90/96 and 86/92, the global-max vendor
92/96 and 89/92, the vendor's own prefill of the same tokens 93/96 and 85/92 (`../gpu/README.md`).

On the steps with full logits the exact KL tells the same story at long context (memra 8.9e-03, the
floor 1.4e-02, 22 steps) and a different one on the chat tapes: memra 1.36e-02 against 2.39e-03 on
78 steps, 5.7x. Those are the first 8 steps of each chat tape, a short prompt's prefill row and the
decode rows right after it, and there the schedule floor is degenerate (previous section).

The backend floor for those early steps (`cpu_floor_kl.py`, `receipts/gates/cpu-floor.json`): exact
KL against the GPU tape on the first 8 steps of four chat tapes, for memra, the vendor on the CPU and
the vendor's global-max GPU realization, all fed the same tokens.

| tape | steps | KL memra | KL vendor CPU | KL vendor global-max | top-1 memra / CPU |
|---|---|---|---|---|---|
| gsm | 8 | 5.86e-02 | 5.12e-02 | 7.35e-03 | 7 / 7 |
| story | 8 | 1.45e-02 | 1.45e-02 | 1.38e-03 | 8 / 8 |
| code | 8 | 6.31e-03 | 3.08e-04 | 1.31e-07 | 8 / 8 |
| sys_json | 8 | 1.43e-03 | 8.94e-04 | 7.87e-05 | 8 / 8 |
| pooled | 32 | 2.02e-02 | 1.67e-02 | 2.20e-03 | 31 / 31 (of 32) |

Pooled, memra is at 1.21x the vendor's CPU-vs-GPU spread: PASS on the 2x band. The vendor's own CPU
run is 7.6x the global-max floor on the same steps, which is what makes that floor degenerate here.
Single positions swing both ways and far: on `code` memra is 20x the CPU run (its prefill row: memra
puts 0.84 on the opening fence where the GPU vendor puts 0.94; the CPU run's KL there is 2.5e-03,
memra's 5.1e-02), on `story` step 0 it is 36 times closer than the CPU run, and on `story` overall the
two match to three digits. One position is one draw from a heavy-tailed distribution; the band is
on the pool. Both the CPU vendor and memra miss the same `gsm` step (step 5, below).

## Greedy generation

`dsv41-run --gen` from each tape's prompt, greedy, stopping at `<｜end▁of▁sentence｜>` (200 tokens at
most for the chat tapes, 96 for the long ones, as the tapes were made). The exact-match prefix with
the vendor's tokens, where they part, and the vendor's own margins there (`decode_gate.py`).

| tape | vendor tokens | memra tokens | exact prefix | parts at step | vendor margin (logit) | vendor p1 / p2 | memra's token is the vendor's | another vendor run picks memra's token | near-tie |
|---|---|---|---|---|---|---|---|---|---|
| capital | 2 | 2 | all | - | | | | | |
| gsm | 20 | 20 | 5 | 5 | 0.665 | 0.564 / 0.290 | #2 | CPU run | yes |
| code | 42 | 42 | all | - | | | | | |
| copy | 30 | 30 | all | - | | | | | |
| hebrew | 3 | 3 | all | - | | | | | |
| gsm_think | 50 | 47 | 3 | 3 | 0.083 | 0.401 / 0.369 | #2 | - | yes |
| story | 138 | 126 | 19 | 19 | 1.138 | 0.634 / 0.203 | #2 | global-max | yes |
| hebrew_para | 155 | 175 | 23 | 23 | 0.006 | 0.495 / 0.492 | #2 | global-max | yes |
| code_pal | 53 | 54 | 19 | 19 | 0.502 | 0.622 / 0.377 | #2 | - | yes |
| sys_json | 15 | 15 | all | - | | | | | |
| multiturn | 3 | 3 | all | - | | | | | |
| needle | 3 | 3 | all | - | | | | | |
| needle1200 | 3 | 3 | all | - | | | | | |
| decode32k_he_summary | 96 | 96 | 13 | 13 | 0.485 | 0.462 / 0.285 | #2 | global-max | yes |
| decode64k_en_continue | 92 | 76 | 4 | 4 | 0.146 | 0.362 / 0.313 | #2 | global-max | yes |
| needle4800 | 3 | 3 | all | - | | | | | |
| needle8400 | 3 | 3 | all | - | | | | | |
| needle16800 | 3 | 3 | all | - | | | | | |

A divergence is a vendor near-tie when another valid run of the vendor program picks memra's token
at that step (the global-max realization, or the CPU run where it covers the step), or when the
vendor's top-2 margin there is no larger than the p99 of how far that margin moves between the
vendor's two GPU realizations over the group's steps (2.23 logits on the chat tapes, 3.57 on the
long ones; the median move is 0.39 and 0.63). Every divergence is the vendor's runner-up token, and
5 of the 7 are steps where another vendor run makes memra's choice. `gsm_think` parts at a 0.083
logit margin, `code_pal` at 0.50, under a quarter of the chat group's p99 move.

After the split the generations stay on task and give the same answers: `gsm` writes the same three
lines and `9` with two trailing spaces before each line break; `gsm_think` reaches the same `191`
after "We need answer." instead of "We need answer only final number."; `code_pal` puts a comma before
"ignoring" and writes the same function; `story` and `decode32k_he_summary` tell the same content in
other words; `decode64k_en_continue` opens "In the final chapter provided (Chapter 24, "The
Advocate")" and summarizes the same chapter; `hebrew_para`, split on a 0.006-logit tie between two
Hebrew words for "more than", goes on to another figure for the city's age (5,000 years against the
vendor's 3,000), both sides fluent. Full texts: `text` in the tapes, `native_text` in
`receipts/gates/{chat,long}.json`.

## The prefill arm

The same chat tapes scored by one teacher-forced prefill of the prompt plus the vendor's tokens
(`dsv41-run --tf`): the prefill program at every decode position, what a loop that re-prefilled
each token would run. Both arms against the vendor's decode program, split at the window
(`prefill_arm.py`, `receipts/gates/prefill-arm.json`):

| positions | steps | KL decode loop | KL prefill arm | top-1 decode loop | top-1 prefill arm | steps with bitwise-equal top-32 |
|---|---|---|---|---|---|---|
| below 128 | 427 | 1.616e-02 | 1.619e-02 | 415 | 417 | 0 |
| 128 and up | 90 | 1.705e-02 | 1.727e-02 | 88 | 85 | 6 |

Below the window the two programs never give the same bits (0 of 427 steps; the vendor's two
programs differ there too), yet at the logit level they sit equally far from the vendor's decode:
these logit gates cannot tell the vendor's decode program from its prefill program. The per-layer
byte gate below does, and it is what pins the loop to the decode program, with the ring-row unit
test (`decode_rows_read_the_ring_oldest_slot_first`) and the replay and forced-vs-greedy identities
above, which hold the loop to one program per sequence. The 6 equal steps above the window
are the two needles, whose prompts are past it; on `story` and `hebrew_para` the rows past 128
still read KV written below it.

## The decode program, per layer, byte for byte

`dsv41-gate decode` (`receipts/gates/bytegate-*.{json,log}`). The vendor program runs the
checkpoint with hooks on chosen layers (`../gpu/capture_layers.py`, the tapes' own oracle) through a
prefill and decode steps; memra runs each captured layer alone from the capture's inputs (stream,
pre-mix, the CSA2 state of the sources it reads), its own window ring and compressor pair seeded by
its own prefill of the prompt, then each decode step at its position: the per-layer program the
loop runs. Per step it compares the integer outcomes where a decode program and a prefill program
differ, byte for byte, with the vendor's: the attention index row (width and order: the window
ring oldest slot first with -1 for empty slots, then the compressed selection shifted by the
window), the indexer's selection on an index source, whether the ratio-2 compressor pools, and the
Engram hash ids of the new token. Three captures: `capital` (decode at 16 and 17, the ratio-2 pair
opening and closing), `gsm` (42 to 47) and `straddle126` (the story prompt plus the vendor's first
104 story tokens as a 126-token prompt, decode at 126 to 131: below the window, at it and past it;
`make_dec_tapes.py`). The capture of `gsm` reproduces the tape's logits bit for bit on all 7 steps.

| capture | positions | layers | attention index rows | selections | compressor emissions | Engram hash rows | failed |
|---|---|---|---|---|---|---|---|
| capital | 16-17 | 1, 2, 3, 14, 20, 21, 24 | 14 / 14 | 8 / 8 | 4 / 4 | 4 / 4 | 0 of 60 |
| gsm | 42-47 | 0, 1, 2, 3, 8, 14, 20, 21, 24 | 54 / 54 | 30 / 30 | 18 / 18 | 12 / 12 | 0 of 240 |
| straddle126 | 126-131 | 0, 1, 2, 3, 8, 14, 20, 21, 24 | 54 / 54 | 30 / 30 | 18 / 18 | 12 / 12 | 0 of 240 |

The red arm runs the same position under prefill semantics: one prefill of every position up to
it, the program a loop that re-prefills each token would run, on the six layers that hold all their
CSA2 state themselves (those of 0, 1, 2, 8, 14 and 20 each capture holds). The check must reject it wherever the vendor's own
prefill row is not its decode row, which by the vendor's `get_window_topk_idxs` is every position
but 127: below it the prefill row is narrower (positions 0..p, no empty slots), past it the ring row
holds slots, not positions, and the compressed entries are offset by the window instead of by the
prefill's length. At 127 both list 0..127 and the two programs read the same keys in the same order. Result:
rejected at all 74 layer-steps of positions 16-17, 42-47, 126 and 128-131, accepted at the 6 of
position 127, as required (part of each capture's check count above).

The streams are reported, not gated. Per layer the decode program sits closer to the vendor's
decode output than the re-prefill does below 127 (median rel L2 1.1e-03 against 9.0e-03 over 50
layer-steps, the re-prefill farther on 35 of them and equal on 14, where every valid key falls into
one 64-key block in both orders), and the two are bit-equal from 127 on (30 of 30). Through the 40
layers to the head that per-layer gap stays inside the realization noise (the prefill arm above);
the integer outcomes keep it visible.

## What remains for serve

- Session and batching: a `deepseek_v41` engine entry holding one `Seq` per request, with
  admission and memory accounting for its state (per layer a 128 KB window ring; per group the
  compressed KV, 1 KB per row, n/2 rows in the three ratio-2 groups and n rows in the ratio-1 group;
  index keys, 256 B per row; the candidate pool per call), free on finish or cancel. Today each
  sequence runs alone through each block; batching rows of several sequences into one GEMM needs a
  serving-shape gate that proves batched and solo bit-identical, or a rule that never mixes them
  inside a request (one numeric program per request).
- Weight residency: the one-card route streams about 7.5 GB per layer per forward (11.0 to 31.5 s
  per decode forward here with 11 layers resident, the slow end while another card streamed too).
  Serving is the 4-card stage at N = 4 with every share resident (76.5 GB per rank,
  `../tp/README.md`), or, on one card, a routed-expert cache (6 of 384 experts per token per layer)
  instead of whole-layer streaming.
- Sampler: greedy only. The vendor's `sample` (temperature with Gumbel-max), top-p / top-k,
  penalties, seeds, logprobs.
- Stop conditions: the end-of-sequence id and a token budget only. Stop strings need the
  detokenized suffix, and the dialect's own stop tokens.
- The `encoding_dsv41` renderer: render chat messages to ids (thinking modes, the reasoning budget)
  before the prefill, and parse DSML tool calls and reasoning out of the generated ids; set
  `ModelCaps{dsv4, dsv41}` from tokenizer detection and accept the numeric 1-100 reasoning budget
  over HTTP (`../encoding/`).
- Continuation: a later turn's tokens enter as decode calls one at a time (the vendor's program for
  them); chunked continuation is refused. Prefix caching would key on the whole carried state.
- The memra-server route (OpenAI surface, streaming) and the support record's Serve gate.
- Speed: per-call host round trips (MoE plan, window index rows, Engram gather), no CUDA graphs,
  plain GEMM tiles at M = 1. DSpark (`mtp.*`) drafts and the vision tower are not run.

## Commands

```bash
# inputs from the vendor's greedy tapes, and the forced oracle copies (reference and floor)
python3 prep_decode.py TAPES dec chat capital gsm code copy hebrew gsm_think story hebrew_para code_pal \
  sys_json multiturn needle needle1200
python3 prep_decode.py TAPES dec long decode32k_he_summary decode64k_en_continue needle4800 needle8400
python3 make_forced_oracle.py ORACLE_DIR oracle-forced
python3 make_forced_oracle.py ORACLE_GLOBALMAX_DIR oracle-globalmax-forced
python3 oracle-forced/run_oracle.py CKPT REF dec/chat-oracle.json ref --device cuda --threads 8 --full-logits 8 \
  --max-seq 140000 --index-qchunk 512 --expert-cache-gb 8          # long group: --expert-cache-gb 2
python3 oracle-globalmax-forced/run_oracle.py CKPT REF dec/chat-oracle.json floor ...   # same flags
python3 oracle-forced/run_oracle.py CKPT REF dec/cpu-oracle.json cpu-floor --device cpu --threads 12 \
  --full-logits 8 --max-seq 8192                                   # gsm, story, code, sys_json, 8 steps each
# memra: teacher-forced (one pass per run), greedy in resumable leases, a final replay-and-verify run
MEMRA_FULL_PREC=1 dsv41-run CKPT --token-map TOKEN_MAP --cases dec/chat-forced.txt --out native
MEMRA_FULL_PREC=1 dsv41-run CKPT --token-map TOKEN_MAP --cases dec/tf-needle8400.txt --out native  # one long tape per run
MEMRA_FULL_PREC=1 dsv41-run CKPT --token-map TOKEN_MAP --cases dec/chat-greedy.txt --gen 200 --out greedy \
  --resident 11 --steps 40     # repeated until no sequence is left, then once more (no --steps) to verify
MEMRA_FULL_PREC=1 dsv41-run CKPT --token-map TOKEN_MAP --cases dec/long-greedy.txt --gen 96 --out greedy \
  --resident 11 --steps 30
MEMRA_FULL_PREC=1 dsv41-run CKPT --token-map TOKEN_MAP --cases dec/tp2-forced.txt --out native-tp2 --devices 0,0
# the prefill arm: prompt + vendor tokens as one teacher-forced prefill per tape
python3 ../gpu/decode_vs_tf.py prep prefill-arm TAPES/capital.pt ...
MEMRA_FULL_PREC=1 dsv41-run CKPT --token-map TOKEN_MAP --ids prefill-arm/capital.ids.i32 --tf prefill-arm/capital
# gates
python3 decode_gate.py --tapes TAPES --ref ref --floor floor --native native --greedy greedy --cpu cpu-floor \
  --cases capital,gsm,... --name chat --tok CKPT --json chat.json
python3 cpu_floor_kl.py TAPES cpu-floor floor native gsm,story,code,sys_json --json cpu-floor.json
python3 prefill_arm.py --tapes TAPES --ref ref --native native --arm prefill-arm --cases capital,... --json prefill-arm.json
python3 decode_vs_prefill.py native/decode32k_he_summary GPU_LANE/dectf-decode32k_he_summary 41098
python3 tables_decode.py chat.json long.json
# the per-layer byte gate: vendor captures of decode steps, then memra per layer with the re-prefill red arm
python3 make_dec_tapes.py TAPES captape
python3 ../gpu/capture_layers.py ORACLE_DIR CKPT REF captape captures --ids gsm,straddle126 \
  --layers 0,1,2,3,8,14,20,21,24 --ops 0,1,2,3,8,14,20,21,24 --decode 6 --max-seq 270000 --index-qchunk 512
dsv41-gate decode CKPT captures/straddle126 --layers 0,1,2,3,8,14,20,21,24 --token-map TOKEN_MAP \
  --out bytegate-straddle126.json
dsv41-gate decode CKPT GPU_LANE_CAPTURES/capital --layers 1,2,3,14,20,21,24 --token-map TOKEN_MAP
```

Every GPU step ran as one lease through `memra-gpu-run` on one card with that card's box lock,
cores 24-31/56-63. Paths in the receipts are relative to the lane's run directory.

## Files

- `make_forced_oracle.py`: the oracle copy that takes the token of each step and keeps lse.
- `prep_decode.py`: prompt and token files, `--cases` files and oracle cases from the tapes.
- `decode_gate.py`: teacher-forced per-step KL and top-1 against the floor, greedy prefix and
  divergence analysis, the forced-vs-greedy byte check.
- `cpu_floor_kl.py`: the early steps against the vendor's CPU run.
- `decode_vs_prefill.py`: memra's decode steps against memra's prefill rows.
- `prefill_arm.py`: the decode loop against the re-prefill arm, split at the window.
- `tables_decode.py`: the tables above from the gate receipts.
- `make_dec_tapes.py`: the two capture prompts of the byte gate (`gsm`, `straddle126`).
- `receipts/gates/`: gate JSON and logs. `receipts/runs/`: every `dsv41-run` log, the vendor runs'
  summary lines (with the oracle files' sha256), the smoke hashes, the unit tests. `receipts/native/`,
  `receipts/greedy/`, `receipts/native-tp2/`: memra's per-step files (top-32, lse, argmax, tokens,
  meta; the full logits stay on the run box). `receipts/prefill-arm/`: the prefill arm's rows at the
  decode positions (`prefill_arm.py` reads them as it reads a full run directory).
