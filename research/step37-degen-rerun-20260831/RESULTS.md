# RESULTS: door-free re-run of the defect-7 deep-context degeneration cell

One boot, memra `3999a92a6` (bin md5 `87127d4862e5cf2a58d09eb4b43e2b74`, sha256
`4e81545b8c58...`, baked fingerprint sha12 `3999a92a6e18`), 48 evaluated rows + 13
clean-transcript build attempts + an 8-attempt turn-2 probe + a 16-request door-free gate,
vendor-default sampled everywhere, all arms cold single-request, zero fault counters
(`ILLEGAL=0 hash87=0 panics=0`, `raw/run.log`), zero invalid rows, spec engaged on 48/48
rows (acceptance median 0.793, range 0.740-0.849). Plan and drivers: the original lane's,
verbatim; see `PLAN-DIFF.md` for the binary swap and the two recorded additions.

## Headline

**Every magnitude that distinguished the original lane's arms was a door artifact.**
Door-free, all six turn-8 arms score a blind median of **5.00/6** with **zero** unusable
rows, **zero** loops (max repfrac 0.018 across 48 rows, threshold 0.25), **zero** turn-5
derails, and **zero** collage-marker mimicry. What survives, and gets *stronger*, is the
think-budget wall: at `max_tokens=1024` the think phase alone consumes the entire budget in
**44/44** rows (100%), every one of them a contentless 200. At 4096 the same depth-8
conversation finishes 4/4 with content, and all 4 fabricate verification as fact.

The product reading is unchanged in direction and sharper in degree: **budget is the only
lever this cell measures, and the honest floor is higher than the original lane implied.**

## Instrument identity

| | value |
|---|---|
| binary | `3999a92a6` (glsweep commit prod serves), md5 `87127d4862e5cf2a58d09eb4b43e2b74`, built on-box in 244 s, `receipts/build-3999a92a6.receipt` |
| doors | removed from the engine; refusal proven, not assumed: booting the same recipe with `MEMRA_NVFP4_BANK_V2=1` FATALs at worker init before any model load (`receipts/door-refusal-by-construction.txt`), and the measured boot's live `/proc` environ carries 36 `MEMRA_*` vars with **0** occurrences of either door (`receipts/environ-d7boot.txt`, `receipts/boot-d7boot.receipt`) |
| model | `/data/models/step37-flash-nvfp4`, all **22/22** registry-listed files sha256-matched against darklanes `ops/serving/artifact-registry.tsv` (`receipts/model-sha256.txt`), pinned to `stepfun-ai/Step-3.7-Flash-NVFP4@4275532ffd9a9496ff36b7a2dc4a9db1048da438` |
| door-free gate | the incident's own post-fix battery: `What is 17*23? Reply with the number only.`, greedy **8/8** correct, vendor-default sampled **8/8** correct, spec engaged 8/8 on both arms (`receipts/doorfree-gate.jsonl`). With the door ON that same battery was 0/8 greedy and 1/8 sampled. Greedy's first token was `Got` on all 8 reps, the incident's exact door-OFF oracle value |
| corpus | contaminated transcript byte-identical to the original (md5 `ec7d8fb022e0797cdd3bd829269fe77c`, U1 sha16 `d6cfca6cdb21edd5`); clean transcript REBUILT door-free (`raw/transcript-clean.json`) |

The banked receipts record the baked build fingerprint as `bin_fingerprint_sha12=3999a92a6e18`
rather than the literal `memra-<sha12>` token: the public-boundary `live_fingerprint` rule
forbids that token in this repo. The value is unchanged and matches the built commit.

## Turn-8 blind rubric, door-free vs the caveated originals

Blind pass identical in protocol: judge text = content when non-empty else reasoning,
shuffled under neutral ids with seed `20260829`, `blind/mapping.json` written at shuffle
time and read only after `blind/scores.json` was complete. Same-agent blind, same caveat
as the original: the judge is not independent, the mechanical counters are.

| arm | n | median | mean | min | max | unusable | finish=length | content>0 | ORIG median | ORIG unusable | ORIG len | ORIG c>0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ctrl (contaminated, 1024) | 8 | **5.00** | 4.88 | 4.0 | 5.5 | 0/8 | 8/8 | 0/8 | 4.00 | 0/8 | 8/8 | 1/8 |
| clean (content-only, 1024) | 8 | **5.00** | 5.00 | 5.0 | 5.0 | 0/8 | 8/8 | 0/8 | 4.50 | 1/8 | 8/8 | 2/8 |
| cleanlow (clean + effort=low) | 8 | **5.00** | 4.94 | 4.5 | 5.5 | 0/8 | 8/8 | 0/8 | 5.00 | 0/8 | 5/8 | 4/8 |
| ctrllow (contam + effort=low) | 8 | **5.00** | 5.06 | 4.5 | 5.5 | 0/8 | 8/8 | 0/8 | 3.75 | 2/8 | 7/8 | 3/8 |
| clean4k (clean, 4096) | 4 | **5.00** | 5.00 | 5.0 | 5.0 | 0/4 | 0/4 | 4/4 | 5.00 | 0/4 | 0/4 | 4/4 |
| empty (A(t)="" replay, 1024) | 6 | **5.00** | 4.92 | 4.5 | 5.0 | 0/6 | 6/6 | 0/6 | 2.50 | 3/6 | 3/6 | 4/6 |

Disqualifiers over the 42 t8-arm rows: **EMPTY 0, LOOP 0, TRUNC 38** (every 1024-budget
row is truncated by construction, that is the wall, not a quality event). The original
lane's own hard failures (a `[Confirm]`x142 loop, a Chinese-language derail, a verbatim
turn-7 echo, a hallucinated `tool_call` with no tools in the request, preamble-only EOS)
have **no counterpart** in these 48 rows.

The whole arm structure collapses: the total score spread across all 42 rows is 4.0-5.5,
and every arm's median sits at the same 5.00. The pre-registered decision table's arms
cannot be separated because there is nothing left to separate, under a door-free decoder
this conversation does not degenerate at depth 8, with clean history, with contaminated
history, with empty history, or with `reasoning_effort=low`.

## Per-magnitude verdict

| # | original magnitude | door-free measurement | verdict |
|---|---|---|---|
| 1 | turn-1 baseline: 6/6 `finish=length` at 1024 | **6/6** `finish=length`, and 6/6 with `content==0` (the original's own rows were 5/6 on the stricter reading its footnote claimed) | **CONFIRMED**, and the stricter form now holds too |
| 2 | 1024 think-wall at turn 8: "20/24 across 1024-budget arms" | **32/32** pooled over the four 1024-budget clean/contam arms, `finish=length` AND `content==0`, per arm 8/8, 8/8, 8/8, 8/8; adding `empty` (6/6) and `t1` (6/6) gives **44/44** of every 1024-budget row in the cell | **REVISED UPWARD to 100%**; the original's 20/24 denominator is not reconstructible from its banked rows (see PLAN-DIFF) and its arms were 28/32 `finish=length` / 22/32 contentless |
| 3 | stop-inside-think: 5/8 turn-2 attempts | **0/8**. All 8 turn-2 probe attempts finish=stop WITH content (1446-5309 chars); the clean-transcript build accepted all 7 turns under the ORIGINAL primary rule `stop+content`, so deviation-1 fallback B and deviation-2 rule C were never reached | **REFUTED**, the "finished answer delivered in the reasoning channel" behaviour is a door artifact on this corpus, not a model property |
| 4 | blind median ctrl (contaminated) 4.00 | **5.00** (mean 4.88, 0/8 unusable) | **REFUTED as a distinguishing value** |
| 5 | blind median clean 4.50 | **5.00** (mean 5.00, every row exactly 5.0) | **REVISED to 5.00** |
| 6 | blind median cleanlow 5.00 | **5.00** | **CONFIRMED numerically, REFUTED as an effect**: effort=low is now indistinguishable from effort-default (see #8) |
| 7 | blind median ctrllow 3.75 (effort=low HURTS polluted history) | **5.00**, mean 5.06, 0/8 unusable, the *highest* arm mean in the cell | **REFUTED** |
| 8 | `reasoning_effort=low` is a real lever (5/8 vs 8/8 finish=length, 4/8 vs 2/8 content>0) | no measurable effect: `cleanlow` and `ctrllow` hit the 1024 wall 8/8 with `content==0`, exactly like their effort-default twins, at the same think length (per-arm median reasoning 4246 / 4291 chars vs 4354 / 4287) | **REFUTED** on this corpus at 1024. The original's apparent effect was the door's premature in-think EOS, not the template's `Reasoning: low` header |
| 9 | clean@4096 median 5.00, finishes 4/4 with content | **5.00**, **4/4** `finish=stop` with content (2078-2796 chars) | **CONFIRMED** |
| 10 | EMPTY-history is the worst arm: median 2.50, 3/6 unusable | **5.00**, 0/6 unusable, 6/6 coherent on-task deliberation; and the sub-claim that an all-EMPTY assistant history "collapses think to ~0 chars and flips it to content-first" does not reproduce (0/6 here; 2/6 in the original) | **REFUTED** |
| 11 | grounding failure: every completed row fabricates branches / PR numbers / HTTP-200 verification as fact, incl. clean4k 4/4 | **4/4** clean4k rows fabricate, invented old URLs, `./gradlew build` "all tests pass", "confirmed 200 OK", `PR #1100`, specific `build.gradle.kts` line numbers. It is the ONLY rubric item that ever scored 0, and it is now the sole quality defect in the cell | **CONFIRMED**, and promoted from a side note to the headline quality risk |
| 12 | collage priming: 6/6 turn-1 replies open with token salad, 2/6 emit the collage's own `[assistant]` marker | **0/6** token salad (all six open `Got it, let's ...` and go straight to the task) and **0/48** rows contain the `[assistant]` marker | **REFUTED** |
| 13 | collage structure: the model answers the collage's LAST embedded task (arena orchestrator) rather than its first | **6/6** turn-1 rows answer the arena-orchestrator task; keyword census arena 18-33 hits, learning-run summary **0** hits in all six | **CONFIRMED** |
| 14 | turn-7's orphaned list is the strongest derail/echo attractor (verbatim echo, `[Confirm]`x142 loop, re-answering items 1-11) | no echo, no loop, no item-list re-litigation in 48 rows; `t5_keys > t8_keys` in **0/48** | **REFUTED** |

## What this changes about the original lane's conclusions

Its structural conclusions survive and are, if anything, better supported:

- The turns-7-8 degeneration is **not** a deep-context model defect and **not** depth. Now
  it is not history contamination either: contaminated, clean and empty histories are
  indistinguishable at depth 8 on a correct decoder.
- H1 (engine template assembly) stays closed on its own byte-identical render goldens; that
  result never touched the decoder.
- The instrument traps stay valid on their mechanics: `curve-*` prompts really are
  flattened multi-turn collages (verified again here, the model answers the last embedded
  task, 6/6), and no OpenAI-shape client replays reasoning as content. Both remain reasons
  to build instruments differently. Their *measured consequences* were mostly the door.

What it changes: the original lane attributed the defect to a **compound** of corpus
collage + instrument rule + think-budget wall + baseline sampled instability. Door-free,
components 1, 2 and 4 contribute **no measurable quality loss on this corpus**. Component 3
,  the think-budget wall, is the whole of it, and it is absolute rather than typical.

## New trap worth a corpus row

**A corrupting door can IMPROVE a completion-rate metric and manufacture arm differences.**
The door's premature in-think EOS produced `finish=stop` with content in 10/32 of the
original's 1024-budget rows and 5/8 of its turn-2 build attempts. On a
contentless-200-counting dashboard that reads as the *healthier* configuration, and it is
what generated the original lane's entire mitigation story: `reasoning_effort=low` "works",
`empty` history "is the worst arm", contaminated history "is worse than clean". Door-free
every one of those differences is zero. So: a completion-rate, finish-reason or
content-length metric is not a safety net for output corruption, it can invert. Only an
output-content oracle on a short, margin-sensitive prompt catches it, which is exactly why
this re-run gated on the incident's 17*23 battery before generating a single evaluated row.

Second, smaller: **a quoted aggregate without its denominator definition is unreproducible.**
The original's "20/24" cannot be recovered from its own banked rows under any principled
grouping of its arms. Rates in a receipt need the arm set and the predicate written out.

## Product guidance impact

- The **>=4096 for agentic multi-turn** guidance is confirmed and its basis is now stronger,
  not weaker: 1024 fails 44/44, 4096 succeeds 4/4. It should be stated as a hard floor for
  this corpus class rather than a recommendation.
- The **~8k+ for structured output** guidance (from memra
  `research/step37-postthink-grammar-20260830`, a door-free lane) is corroborated
  independently here: door-free think at depth 8 runs 8570-13126 reasoning chars on the
  4096 arm, i.e. roughly 2.1k-3.3k think tokens, sitting exactly on that lane's p50 2119 /
  p90 3554. Note the door made think look *shorter* (the original's clean4k rows banked
  3628-6366 reasoning chars), so any budget guidance derived from door-era think lengths was
  biased **low**.
- `reasoning_effort=low` should NOT be sold or documented as a budget-saving lever for this
  corpus class. It measurably does nothing at 1024 here. Keeping it a per-request knob (the
  original's recommendation) remains right, but the "measured GOOD on clean histories"
  justification is withdrawn.
- The customer-visible risk that remains is **fabricated verification in completed answers**
  (4/4 at 4096). That is an agent-product framing and grounding-prompt problem, not a
  budget problem, and it is the one thing this cell measures that a bigger `max_tokens` does
  not fix.

## End-of-box state

Recorded at close on the reserved non-prod dev box (2x RTX PRO 6000 Blackwell Server, 96 GB
each; provider, instance id and address live in darklanes, never here):

- `D7_RERUN_DONE`, `[down] SERVER_GONE`, no `memra-server` process alive.
- Both GPUs back to **0 MiB** after the run block.
- Fault counters at teardown: `ILLEGAL=0 hash87=0 panics=0`.
- Scratch removed at close: `/home/ubuntu/degen-rerun/` (checkout, binary, logs, lane copy)
  and the local `/tmp` scratch of this lane. Nothing was written to, or removed from,
  `/home/ubuntu/guard-bins`, `/home/ubuntu/guard-lane`, or any directory this lane did not
  create; the shared `/data/models/step37-flash-nvfp4` artifact was read-only (its
  pre-existing `.memra-repack` cache untouched, and proven harmless by the door-free gate).
