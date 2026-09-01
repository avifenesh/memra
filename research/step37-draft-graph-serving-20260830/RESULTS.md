# step37 draft graph on the serving shape: multi-head chain capture + in-graph filtered sampling

Lane: `lane/step37-draft-graph-serving-20260830`. Owner-ordered sequencing: this lane lands
BEFORE the TTFT/operating-point/vision-latency measurement lanes so the perf substrate is
final; it publishes NO serving perf claims - the A/B tables below are lane-internal receipts.

Predecessor: `research/step37-draft-graph-20260829/RESULTS.md` (the dcw door), which receipted
the two structural blockers this lane removes:

1. `graph_draft` carried an `mtp_extra.is_empty()` conjunct: at MEMRA_MTP_HEADS=3 (the
   QUALIFIED serving shape) capture was never even ATTEMPTED and the boot WARN never printed.
2. The sampled draft graph was pure-temp-only by exactness law, and step37 vendor defaults
   are temp 0.5 / top_p 0.9 - so even a 1-head boot never launched a captured chain for a
   vendor-default request.

Box: the rented dev box (2x RTX PRO 6000 Blackwell; provisioning receipts live in the private ops repo), artifact
`/data/models/step37-flash-nvfp4` verified shard-by-shard against HF
`stepfun-ai/Step-3.7-Flash-NVFP4` rev `4275532ffd9a9496ff36b7a2dc4a9db1048da438`:
all 14 LFS sha256 match (`raw/shards receipt inside raw/runspec.txt` header line).
Binaries built on-box from the lane branch (clone + explicit-refspec bundle fetch); the
FINAL receipts (run 3, commit `7cfcf73be`) bind to `dgs-run-spec` md5
`4643cc805b7535e83b5001483ee4572a` and `dgs-memra-server` md5
`4f1ed7c4dd8a3aaaf2571dffb5c25e51`, strings fingerprint chain_graph_flag=2
filtered_flag=1 chain_receipt=2 (rebuild-attribution law: the header of every OUT file
carries `git log -1`). Three battery runs total: run 1 aborted on the sub-floor incident
(below), run 2 exposed the accept-path stats duplication, run 3 = the banked receipts.

## 1. Design

### Blocker 1 removed: per-head single-row graph chain (`DraftChainGraphs`)

The step-modulo prefix-replay chain stays HOST-side - head selection, prefix length, and
the stored-seed feed replay `mtp_chain_forward_dev`'s exact launch order - and each
replayed ROW becomes one captured graph launch on that head's own scratch plane:

- `interior[i]`: head i with `with_head=false` - embed/norms/eh_proj/attention (the dcw
  windowed device-counter arm)/FFN/carrier, NO head matmul. Interior rows' logits are dead
  in the eager chain too (`mtp_chain_forward_dev` keeps only the last row), so the consumed
  bytes are identical while the captured chain skips the dead 128896x4096 head matmul the
  eager chain still pays per replay row.
- `last[i]`: head i with the mode tail - greedy argmax (+ p-min prob, + grammar-mask node
  when constrained) or the sampled in-graph categorical draw.

One `DraftChainGraphs` per mode (greedy / sampled), each owning its capture-retain keeper;
`chain_s` shares `s_key` and every drop rule with the single-head `graph_s`. Per round at
K=3/heads=3 the graph arm launches 6 graphs (1+2+3 rows) instead of ~6x the eager chain's
per-row kernel storm, with per-step `set_plane_len` + `g_pos`/`g_tok`/`g_seed` host writes.
Why not the T-padded batched twin: the per-(head,row) design reuses the receipted
single-head capture body verbatim (one `scratch_index` parameter), keeps graph-vs-eager
parity by construction (same launcher, same bucket, same order), and needs no new kernel.

Failure stays LOUD: a chain capture error trips the same `draft-graph capture failed` WARN
(the silent no-attempt hole is closed - door-off boots now WARN at heads=3), and capture
success prints a positive receipt: `[mtp-chain-graph] captured mode=<greedy|sampled>
heads=3 interior=3 last=3 ...` (the 3a lesson: WARN absence is never evidence).
Door: `MEMRA_MTP_CHAIN_GRAPH` default ON, `=0` prints a one-line disarm note and keeps the
eager chain.

### Blocker 2 removed: the truncation filter runs IN-GRAPH

`SampledGraphKey::graph_capturable()` (one predicate for capture guard, launch re-test, and
exactness guard): pure-temp always; filtered (top_k/top_p/min_p) when
`MEMRA_SPEC_GRAPH_FILTERED` (default ON) - the capture body then runs `filter_stats` into
persistent stat slots and the new `gumbel_perturb_filtered_ctr` kernel (device counter +
device (mx, th); arithmetic expression-for-expression the eager `gumbel_perturb_filtered_f32`)
before the in-graph argmax. Penalties never capture (per-round history cannot be baked).
Exactness: the draft draws from the SAME filtered distribution the verify's accept test
reconstructs - accept-side (th, z) recompute from the retained q (q_slots) with the same
deployment-keyed `filter_stats` program on the same bits, hence bit-identical to the
in-graph stats that shaped the draw. The pure-temp capture body is byte-identical to the
pre-lane graph (untouched branch).

`MEMRA_STEP35_DRAFT_DCW` (the kernel prerequisite: windowed device-counter draft attention)
flips default ON in the same lane; `=0` remains the rollback to the host-len eager arm plus
the named refusal.

### The sub-floor-cap hole the default flip exposed (found by the vision cell, fixed in-lane)

`step35_dcw_eligible` mirrored the launcher's WINDOW but not its BUCKET: `fa_decode_dcw`
gates on `bucket_max = min(window, scratch cap) >= 96` (the vec floor), and a SMALL spec
session (tiny prompt + tiny max_tokens; the vision battery's max_tokens=8 usage twin has
scratch cap ~62) is outside the domain even though window=512 clears it. With the door
newly ON by default, that session's capture WARNed and then the EAGER dcw chain - which
has no graceful fallback point - hard-failed the burst
(`[engine-error] fa_decode_dcw supports the default v3-vec class only`,
`raw/run1-aborted/srv-vision.log` line 616; the two vision usage-gate FAILs in run 1 were
this request dying, not a vision regression). Fix `2358f09f4`: eligibility now takes the
plane cap and mirrors the bucket gate, so sub-floor sessions take the host-len kvmod arm -
byte-for-byte the door-off serving - and the cap-site refusal names window and cap. All
batteries were re-run in full against fixed binaries (run 3 = the receipts below);
run-1 outputs are archived in `raw/run1-aborted/` for the incident trail.

## 2. Exactness gates (run-spec, curve-0400 real 613-token chat payload, NGEN=160; all banked in `raw/runspec.txt`)

16/16 cells green, `illegal=0 sentinel87=0 panic=0` and `skey q=0 EXACTNESS lines = 0` in
every cell. TP2 serving env (the vision-lane clean recipe, no removed doors).

| gate | cells | verdict |
|---|---|---|
| Greedy byte identity, heads=3, K=1..8 | g3-off / g3-eager / g3-graph / g3-disarm | SELF-CONSISTENCY PASS (byte-identical to plain generate) at every K in all four arms |
| Chain capture receipt, heads=3 greedy | g3-graph | `[mtp-chain-graph] captured mode=greedy heads=3 interior=3 last=3` x8 (one per fresh ctx), WARN=0 |
| WARN fires on THIS shape (door off) | g3-off | `capture_warn_count=8`, arm=eager (kvmod fallback), still PASS - the silent no-attempt hole is closed |
| Disarm note (chain door off) | g3-disarm | `disarm_note=1`, no capture receipts, PASS |
| Per-K acceptance identity, greedy | g3-eager vs g3-graph, g3p pair, h1g pair | ACCEPT-IDENTITY PASS (identical accepted/drafted rows at every K; g3-off identical too) |
| Sampled seeded twins, heads=3 vendor shape (temp .5 / top_p .9 / seed 4242), K=1..8 | s3-eager vs s3-graph | TOKEN-IDENTITY PASS (8 sampled streams byte-identical graph-vs-eager), seeded-rerun PASS per K, ACCEPT-IDENTITY PASS |
| Chain capture receipt, filtered sampled | s3-graph | `captured mode=sampled heads=3 interior=3 last=3 filtered=1` x16 (k is in s_key; each K recaptures), launch receipts `[skey] chain=graph_chain_s` |
| Serving-policy twins (K=3 PMIN=0.5 PMIN0=1) | g3p / s3p pairs | PASS + TOKEN-IDENTITY + ACCEPT-IDENTITY |
| Single-head regression, heads=1 | h1g / h1sf (filtered) / h1sp (pure-temp) pairs | PASS all; TOKEN-IDENTITY PASS; `[skey] chain=graph_s` launch receipts on both sampled shapes (filtered single-head capture is live; pure-temp body unchanged) |

Why the seeded-twin gate is decisive for sampled correctness: the engine has a seed knob
(`seed` per request / MEMRA_SEED in run-spec), spec sampling is counter-based Philox, and
run-spec additionally reruns each (seed, prompt, K) for reproducibility. Byte identity of
the full sampled stream graph-vs-eager at the same seed, on 8 K values plus the policy
twin, subsumes distribution-equivalence testing; the accept-side is additionally pinned by
per-K acceptance identity and by the zero `q=0` exactness-probe count (the
unconditional-accept signature of a draw/accept distribution mismatch).

## 3. Serving-surface gates (memra-server, `raw/server.txt`)

- WARN both directions on the QUALIFIED shape (W cells, heads=3, serving policy, SKEY probe
  on): door-off boot `capture_warn=5`, `arm=eager`, serving healthy (spec engaged, sampled
  accepted 247/298); door-on boot `capture_warn=0` with positive receipts
  `chain_captured_greedy=2 chain_captured_sampled=3`, first receipt
  `captured mode=sampled ... filtered=1` with a fresh per-request seed in the key.
- BONUS byte receipt: the greedy serving request's output sha is IDENTICAL across the two
  door arms (`2f6a5c073081fb58` both) - kvmod-eager vs captured-dcw-chain serve the same
  greedy bytes end-to-end (and the same sha holds on every P-cell boot of both arms, and
  held across all three binaries of the lane).
- Vendor-default sampled requests (NO sampling params) on the door-on boot: spec engaged
  (`usage.spec` rounds>0, accepted>0) AND the sampled chain captured - the exact shape that
  could never launch a captured chain before this lane.
- spec-on == spec-off byte gate (I cells, curve-0128 - the prime-identity prompt;
  curve-1000 is excluded by QUIRK:step37:prime-program-differs-by-spec): spec-on
  (operator-pin K=3, chain captured, rounds=335 accepted=495/634) sha `5cdfe8f292e33df7`
  == spec-off (MEMRA_SERVE_SPEC=0, `usage.spec ABSENT` - spec truly off) sha
  `5cdfe8f292e33df7`. An earlier run additionally banked pin-vs-automatic-policy identity
  at the same sha (the family-armed default engages spec even without MEMRA_SERVE_SPEC).

## 4. Perf receipts (lane-internal, NOT public claims; interleaved x5, one boot per cell, PID+nonce arm identity, `raw/server.txt`)

Vendor-default sampled (product shape, 3 stream reps/boot, curve-0400, max_tokens 400) and
greedy twin (temperature:0), K=3 PMIN=0.5 PMIN0=1, heads=3; binaries `4f1ed7c4` (server) /
`4643cc80` (run-spec) at commit 7cfcf73be:

| arm | sampled tok/s med (min..max) | greedy tok/s med (min..max) | sampled accept (agg) | greedy accept |
|---|---|---|---|---|
| graph (captured chain) | 103.76 (102.34..109.68) | 106.71 (106.53..106.94) | 1070/1367 = 78.3% | 207/286 = 72.4% (identical every boot) |
| eager (MEMRA_SPEC_NOGRAPH=1) | 104.46 (101.54..113.62) | 106.48 (106.17..106.66) | 1073/1368 = 78.4% | 207/286 = 72.4% (identical every boot) |

Pairwise per round (graph minus eager): sampled +0.19, -0.70, -3.94, -2.15, +0.80 (median
-0.70 inside an ~11 tok/s per-arm spread); greedy +0.04, +0.61, -0.13, +0.64, +0.23
(median +0.23). VERDICT: NEUTRAL on both shapes. TTFT medians: sampled 0.400 vs 0.398 s,
greedy 0.264 vs 0.262 s - the per-request capture cost is ~2-3 ms of TTFT (fresh
per-request seeds re-key chain_s, so the sampled chain recaptures once per request; a
device-resident seed would remove even that and stays a costed follow-up). The GREEDY
serving sha is `2f6a5c073081fb58` on EVERY boot of BOTH arms (and on both door arms of the
W cells): a standing byte-identity receipt on the real serving surface. Engagement:
`usage.spec` rounds>0 accepted>0 in all 20 probe rows; zero LOOP/EMPTY exclusions fired.

Mid-lane perf incident, receipted: run 2 (pre-`7cfcf73be`, `raw/server-run2-pre-statsfix.txt`)
measured the sampled graph arm at median pairwise -5.24 tok/s (~5%): the accept path was
paying a SECOND full-vocab filter_stats per used slot after the chain while the in-graph
nodes had already computed the stats. `7cfcf73be` reads the three scalars back per replay
(bit-exact - they are the values the in-graph perturb consumed); the table above is the
post-fix re-run, and the seeded-twin battery re-ran green on the same binary.

## 5. 8-turn larger-prompt cache-on twin (multi-turn law; `raw/multiturn.txt`)

agentic8 pool, ONE accumulating conversation per boot, vendor-default sampling, max_tokens
4096 (think-budget law), per-turn TTFT + tok/s + usage.spec (with acceptance_rate) +
cached prompt tokens, interleaved x2 per arm. Verdict:

- Per-turn decode tok/s: graph 103.2..109.3, eager 101.8..108.8 across all 32 turns - no
  per-turn regression shape attributable to the arm (turn-by-turn the arms sit inside each
  other's spread).
- Cache engagement PROVEN in both arms: `cached>0` turns in every conversation (e.g. graph
  rnd1 turns 2/6/7/8 with cached up to 4544 of prompt 5435; eager rnd2 turns 2/3/6 with
  cached up to 3200), matching `[spec-k] ... cached=` log fields. Which turns hit is
  stochastic (a think-heavy turn that terminates on length does not park its session -
  the known think-budget wall), and the pattern class is identical across arms.
- TTFT per turn tracks UNCACHED prompt length, not the arm (a cached turn-8 serves at
  ~0.46 s where an uncached one pays ~2.2 s, in BOTH arms).
- spec engaged on every turn (acceptance ~77-83%); zero WARN/ILLEGAL/#87/panic in all four
  boots.

## 6. Vision no-interaction cell (`raw/vision.txt`)

Vision-armed boot (MEMRA_STEP_VISION_DIR) on the final binary: every image request admits
`[spec-k] K=0 source=eligibility-fallback` (spec never crosses an image span), text
requests keep `K=3 source=operator-pin`, and the vision e2e gates pass - including the
+171/+670 prompt-token accounting that run 1 broke via the sub-floor-cap engine error
(now: `usage-171-per-image PASS delta=171`, `usage-670-tiled PASS delta=670`,
engine-error count 0; the sub-floor session logs the honest named capture refusal and
serves on the kvmod eager arm). The can't-hallucinate answer probes run under
vendor-default SAMPLING and carry single-draw judgment flakiness that is model behavior,
not tower or graph breakage: attempt 1 (`raw/vision-try1.txt`) drew one empty answer on
`img-single-640` (the receipted stop-inside-think quirk) with every other gate PASS;
attempt 2 (`raw/vision.txt`) passes `img-single-640` and instead names a white background
"light gray" on `img-multi` while perceiving both images' shapes and foregrounds exactly.
Every gate passed in at least one attempt, `[spec-k]` attribution is identical in both
(all image requests K=0, text K=3), and warn=1 in both is the sub-floor session's named
refusal, with zero engine errors.

## 7. Door decisions (FLAGS.md rows in this branch)

- `MEMRA_MTP_CHAIN_GRAPH` default ON - receipts above; `=0` = eager chain, disarm note.
- `MEMRA_SPEC_GRAPH_FILTERED` default ON - receipts above; `=0` = pure-temp-only capture.
- `MEMRA_STEP35_DRAFT_DCW` default flipped ON - it is now load-bearing for capture on the
  qualified shape; `=0` = pre-lane serving byte-for-byte (receipted by the g3-off cell and
  the W door-off boot).

Serving recipe delta for prod: NONE (defaults carry everything; a rollback needs only the
three `=0` seams above).
