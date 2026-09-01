# glm5_next T-parallel verify + rollback (lane/glm5-tparallel-verify, 2026-08-30)

The acceptance/rollback machinery that turns the native MTP draft head
(lane/glm5-mtp-remint @ ba8326602, `mtp_head_forward_mla_cached`) into speculative
decoding on the hc trunk. Before this lane, every spec entry point `refuse_hyper`'d;
the DFlash2 probe priced the arc GO at 3.06-4.66 tokens/cycle on real traffic, and
the corpus law says spec on this architecture pays ONLY with T-parallel verify
(sequential verify: throughput <= plain decode by construction).

## The walk-reuse decision

The verify of K drafted tokens is a t=K+1 walk over the SAME trunk. It rides the
**batched-decode walk's kernel classes** (`hyper_batch_range_decode`,
lane/glm53-batched-decode), NOT the prefill/prime walk:

- The batched walk's row-parallel ops are per-row BIT-EXACT vs the isolated t=1
  decode step at every width `1..=hyper_batch_cap()` (= `PRIME_MIN_T - 1` = 15, the
  shexp decode-exact knee measured in `../batched-decode-gate/31-KNEE-b16-forced.log`):
  block-per-token hc glue, per-row m=1 mix GEMM (`hyper::pre_exact`, the lt_ndep law),
  per-row MoE programs below t=16, `matmul_decode_exact` head. Per-row exactness is
  what makes spec-vs-plain BYTE IDENTITY achievable at all.
- The prime walk is a different numeric class (batched mix GEMM = cuBLASLt
  n-dependent; prefill MoE dispatch; prefill conv arm) and could never pass a
  byte-identity gate against plain decode.
- The knee bounds K+1 <= 15, i.e. K <= 14: the probe's K<=7 drafter and upstream's
  5/7-draft MTP configs fit with margin.

The one change from the batched walk: B independent sessions become K+1 SEQUENTIAL
positions of ONE session — KDA chains the t=1 recurrent step row->row (the scan
kernel is the same sequential program prefill and decode share, kda.rs module doc),
MLA appends+attends causally per row (`mla_attn_cached` t=1; row r attends
`0..pos0+r+1`). This is the "hyper rows-walk with causal verify appends" the
batched-decode lane's standing refusals named as missing.

## The rollback design (built and gated BEFORE the loop)

Adopted from the engine survey's upstream reading (ENGINE-SURVEY §1.8/§2.7: vLLM
keeps num_spec+1 per-step KDA state columns and commits the last accepted one;
SGLang's ReplaySSM keeps an input ring and replays the accepted prefix):

| plane | rollback | cost at K=7 |
|---|---|---|
| KDA recurrent + conv | **per-step state columns** (the vLLM design; the same `cols` arm `commit_verified_prefix` ships for qwen35 GDN): clone (conv, ssm) after each verify row; accept-j restores column j; full accept = resident state, no copy | 4 MiB state x 34 layers x 7 columns ~= **0.95 GiB in-round transient** (conv adds 288 KiB/layer/column, ~2%); the probe sized the session state class ~140 MB — this is K columns of it. Named diet follow-up: the GdnStash/ReplaySSM form (stash ~160 KB/row/layer scan inputs, rebuild by a prefix re-run of `memra_kda_scan_s128`, T-invariant t-loop) |
| MLA latent rows | truncate: `len = snap + keep`, `len_d` in lock-step (append-only, position-addressed — the kept rows ARE what plain decode would have written, per the decode-exact contract) | zero-copy |
| kpool index planes | the tail ring drains IN-CALL, so pool keys over drafted rows may FINALIZE mid-walk; rollback clamps `index_pools_ready` via `truncate_index_pool_keys` (the clamp the field's own doc demands); the `mla_kpool_indices` residency tripwire fails LOUDLY if any rewind forgets it | O(1) |
| MTP draft plane (il=n_trunk) | len reset by the loop (the draft lane's contract) + the same pool clamp; chain rows built from carrier hiddens always reset — accepted positions are re-fed with TRUE trunk h next round, and that re-warm doubles as the next round's first draft step | O(1) |

`CacheSnapshot`/`Cache::rollback` (memra-kv) deliberately NOT extended: their own doc
marks latent awareness as an asymmetric addition, and the pool clamp needs the
engine-side `MlaIndexerGeom::pool`. The glm5 ckpt lives in `glm_spec.rs`
(`Glm5VerifyCkpt`), covering exactly the planes a glm5_next trunk mutates.

## The loop (`generate_spec_glm5`, MEMRA_GLM5_SPEC default OFF)

prime -> warm the MTP plane over the prompt ((token[i+1], h_i) pairs — gap-free
horizon, the head refuses gaps) -> rounds of: feed newly-committed pairs (last call
= draft 1) -> chain K-1 drafts via the carrier -> verify in one t=K+1 walk ->
greedy longest-matching-prefix accept (the DFlash2 probe's rule; the verify bonus
token = argmax of row j) -> rollback -> re-seed from the accepted collapsed rows.
Wired into `generate_spec` behind `MEMRA_GLM5_SPEC` (FLAGS.md row in the same
commit); OFF = the standing `refuse_hyper`, byte-identical serving. `MEMRA_SPEC_HPOST`
refuses (the loop's h_seed contract is pre-output_norm; the HPOST twin needs its own
gate). The MTP_SPEC capability manifest is deliberately NOT extended: worker
`mtp_spec_capable` stays false, sealed bundles stay fail-closed until the
real-artifact qualification lane.

SAMPLED ACCEPTANCE (stated follow-up): the accept rule swaps to memra's existing
sampled spec contract — `spec::SpecSampling` + the rejection-sampling walk
(`u_j < p_j(x_j)/q_j(x_j)`, host Philox4x32-10 tag 0xFFFF_FFFE via `spec::host_u01`,
residual resampling on first rejection), the same contract the MEMRA_SPEC_TEMP route
and the dspark sampled-admission walk consume. One seam; walk and rollback unchanged.

## Gate (glm5_tparallel_verify_gpu; rig 5090, TF32 off, flock-serialized, 2026-08-30)

Fixture: the hyper-batch/ppn gate family (4 mHC streams, mean collapse, KDA +
DSA(MLA+kpool) alternating, dense L0 + sigmoid noaux_tc MoE, F32 + Q8_0 banks) plus
ONE NextN block; prompt 24 puts the trunk indexer in the sparse regime. Full log:
`rig-gate-run-20260830.log`.

| gate | result |
|---|---|
| 1 walk identity: verify rows vs plain `decode_step`, full logits | PASS — 8/8 rows bit-identical |
| 2 accept-j-then-continue vs the never-drafted chain, j = 0..=7, 12 continue steps each, full logits | PASS — bit-identical at every j (continuation stream deliberately differs from the drafted tokens) |
| 3 RED stale-KDA-state (reinstate post-row-K state after a j=2 rollback) | bites — 379 differing logits across the continuation |
| 4 RED pool-keys-finalized-past-j (reinstate un-clamped `index_pools_ready`) | bites — the named tripwire: "claims 8 finished pools but the cache holds only 6 complete pools before this call (25 rows / pool 4)" |
| 5 e2e spec-vs-plain greedy tape, K=1..7 natural drafter + forced full-accept (K=3: 15/15, K=7: 17/21 accepted) + forced-rejection sweep cycling j (14/42) | PASS — every tape byte-identical to plain decode |
| 6 RED rollback disabled + forced rejections | bites — loud failure ("latent cache overflow — 59 + 1 rows exceeds capacity 59": the un-truncated planes overflow), never a silent green |

Natural-drafter acceptance on the fixture is 0 (untrained deterministic weights) —
expected and irrelevant: these are byte-identity gates; acceptance quality is the
real artifact's number. Corrupted drafts yield IDENTICAL output to plain decode —
that is the property the loop sells.

Repro:
```
NVIDIA_TF32_OVERRIDE=0 flock /tmp/memra-5090.lock \
  cargo test -p memra-engine --test glm5_tparallel_verify_gpu -- --ignored --test-threads=1
```

## FR-Spec vocab masking (owner addition 2026-08-30 — the house recipe, the q38 way)

Both draft paths get FR-Spec masking; the MTP head lands here, DFlash2 (if ever
licensed) inherits the same MtpHead trim through the dspark selector, which already
consumes it by default when loaded.

Wired behind the EXISTING `MEMRA_FRSPEC_TRIM` contract — no new flag, because the
plumbing reaches `glm_spec.rs` cleanly: the loader's trim match consumes the embedded
head regardless of arch, `frspec_trim_own_head_name(45)` misses (glm5_next ships no
private MTP lm_head) and the gather falls back to the trunk head — EXACT BY CONTRACT
for this family (the NextN block projects through the trunk lm_head; the draft gate
pins `shared_head_head.is_none()` untrimmed), stronger than the tied-head argument
that arm was written for. `mtp_head_forward_mla_cached` then already projects over
the gathered rows; the loop's seam is the remap only: every draft argmax is a RANK
id mapped through d2t to true vocab BEFORE it is drafted, chained, or verified. The
VERIFY WALK stays full-vocab and untouched — a trimmed draft changes WHICH tokens get
drafted, never how they verify. Boot receipt: `[glm5-spec] draft head TRIMMED to N
rows (FR-Spec d2t engaged)`.

Gate 7 (`gpu_frspec_trim_equivalence_partial_and_skipped_remap_red`; rig 5090, TF32
off, 2026-08-30, `rig-gate-frspec-20260830.log`, full battery 7/7):

| arm | result |
|---|---|
| full-vocab FIXED-POINT-FREE permutation trim (rotation ranks, same 32 rows permuted) | PASS — all 133 remapped drafts IDENTICAL to the untrimmed arm's, tape identical, acceptance identical (an unwired remap cannot pass this: every draft would differ) |
| accept-j spot check (j=2) with the trim loaded | PASS — 12 continue steps bit-identical (the trim can never reach the verify/rollback planes) |
| RED: `skip_d2t_remap` (rank ids drafted as vocab ids — the q38 0/248 skipped-remap defect, silent to every exactness gate) | bites — 125/133 drafts diverge from the remapped arm while the OUTPUT TAPE STAYS byte-identical to plain decode; the silent class made loud |
| partial trim (top-16 of 32 rows) | PASS — restricted draft vocabulary, tape byte-identical |

Acceptance-at-k on the fixture is a plumbing receipt only (untrained weights, ~0 both
arms). The REAL acceptance trade is a number to be measured on the real artifact with
owner-minted ranks.

**INPUT DEPENDENCY — glm5 ranks mint (owner lane, CPU-only):** tokenize the SXC pools
with GLM's tokenizer (LAW:real-prompts-for-spec — never synthetic), frequency-rank
the 154880 vocab, emit the q38 plain-text format (one token id per line, rank order)
per traffic class (agentic/prose/mixed, the published q38 precedent). Until it lands,
the self-trim d2t arm (ranking consumed from a ranks list gathered over the trunk
head's own rows) is live and gated; the acceptance-at-k A/B on real prompts joins the
box A/B battery below.

## Box A/B plan (the MEMRA_GLM5_SPEC flip condition — NOT run in this lane)

Prerequisites, in order:
1. **ppN twin of the verify walk.** `glm5_verify_rows` refuses `pp_cuts` by name; the
   2-card serving shape (L2 owner ruling) needs the `[t, streams, n_embd]` stage-split
   twin mirroring `decode_step_batch_hyper_ppn`, plus the MTP head under the split
   (the head lives on the LAST stage with the lm_head). Gate: the same accept-j and
   e2e batteries under `stages=2`, both split arms.
2. **Real-artifact accept/rollback battery** on the deployed NVFP4 artifact and
   placement (MEMRA_GLM5_MTP=1 + MEMRA_GLM5_SPEC=1, research door, never a sealed
   bundle): spec-vs-plain greedy byte identity on real prompts (the banked gpf-ab
   pool), K=1..7, plus the quantized-class twin of the walk-identity gate (the
   fixture pins the F32/Q8_0 classes; the artifact's NVFP4/BF16 classes get their own
   banked first-diff).
3. **Acceptance measurement** (the projection input): tokens/cycle on real agent
   traffic with the native head — upstream measures 3.71-5.06 at 7 drafts / 1.36-2.05x
   at c=1; our number is ours to measure, never inferred (no-generic-support law).
3b. **Trim acceptance A/B** (rides the same battery): acceptance-at-k trim-on vs
   trim-off with the owner-minted per-class ranks on real prompts — the acceptance
   trade as a number (q38 precedent: acceptance byte-identical at a 3.9x narrower
   draft vocab; gemma: +2.8 to +5.8% at identical acceptance; ours is ours to
   measure).
4. **The A/B itself**: interleaved x5, flag ON vs OFF, on the 2-card serving shape,
   c=1 and the concurrency ladder (spec + batched-decode interaction is chunked by the
   worker; spec rounds are per-session), with the VENDOR-DEFAULT SAMPLED TWIN (no
   sampling params on the wire) and spec-engagement receipts from the server log
   (K>0 / accept counters — never a 200; the never-serve-greedy law), plus the 8-turn
   larger-prompt cache-on twin (multi-turn cache measurement law). Flip evidence =
   all arms green + the owner's read of the sampled rows.

Box time goes through the coordinator (L3 battery first, then the launch-diet
census). Full-clock Blackwell twin of the fixture battery: RAN 2026-08-30 in the
co-tenancy window on the 4-card box (RTX PRO 6000 Blackwell Server, CVD=2,3, TF32
off, commit bc7762c3): **6/6 PASS in 3.36s**, same red-arm bite signatures as the
rig — `box-gate-run-20260830.log`. Box left clean (queue logged, dir removed,
0 MiB on all cards).

## Deliberately NOT in this lane

- DFlash2 external drafter serving (CC BY-NC-ND; probe-only by law).
- MTP-plane prompt prefill beyond the loop's sequential warm (the t-parallel warm is
  a tuning lane).
- MTP_SPEC manifest extension, roster/perf/acceptance claims, any serving exposure
  beyond the default-OFF flag.
- The ReplaySSM-style column diet, the HPOST twin, Full/Linear mixer arms in the
  walk (each refused by name, each needs its own gate).
