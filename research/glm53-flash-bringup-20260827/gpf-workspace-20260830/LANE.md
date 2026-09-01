# gpf-workspace: bound the grouped-MoE prefill workspace, and make admission see it

Lane: `lane/glm5-gpf-workspace`, base `origin/lane/glm53-flash-bringup` @ 19d49a0b1
(2026-08-30). Owner brief: the 262k 2-card cell (branch `lane/glm5-262k-2card-receipt`
@ ca36fa839, `research/glm53-flash-bringup-20260827/262k-2card-20260830/LANE.md`)
measured a MONOLITHIC per-request prefill workspace of ~0.8 MiB/token/card that walls
the 2-card 9-GiB-free shape at ~7-8k prompt tokens, while admission charges glm5
`0 B/token x ctx + 155MB fixed` and admits everything. Two deliverables: (1) bound the
workspace per CHUNK, not per REQUEST; (2) a workspace-aware admission cost so an
over-deep request gets a clean 4xx before prime instead of a mid-stream engine OOM.

## 1. Attribution: what scales with REQUEST tokens, and why

THE ROOT CAUSE IS NOT INSIDE `moe_ffn_grouped_prefill_sigmoid`. Every buffer that
function allocates is proportional to ITS OWN `t`. The defect is which `t` it is
handed: on the pp door (`MEMRA_PP_STAGES=2`, the 2-card serving recipe) the base
tree's `prime_cache_hyper` short-circuits to `prime_cache_hyper_ppn` BEFORE the
`hyper_prime_ranges` chunk walk (hybrid_forward.rs, the "NOTE: the ppN twin primes
monolithically" comment), so ONE call carries the whole prompt and every per-call
transient scales with request tokens. Two levels of monolithicity compose:

* SERVER: glm5 is an `eager_only_model` (worker.rs), so `prefill_tick_take` takes the
  WHOLE prefill queue in one `prime_cache` call (`take = q`, not `q.min(budget)`).
* ENGINE: on the pp door that one call never reaches the chunk schedule. The
  single-engine walk chunks (lane/glm53-1m-context); the ppN twin did not — the named
  follow-up that lane/glm53-1m-demo built as 93927b1fac, unmerged on this base.

Sizes below use glm5_next pins (CENSUS.md): H = n_embd = 4096, S = hc streams = 4,
F = moe_intermediate_size = 2048, U = experts/token = 8, 11 MLA (LatentKvCache) layers
with latent width W = 512 (NoPE), indexer head dim Dh = 128 (index_width 2Dh = 256),
k-pool P = 4, topk 2048. f32 = 4 B, f16 = 2 B. `t` = tokens in ONE prime call
(= the whole request on the base tree), `X` = session cache length.

| # | allocation | site | size formula (bytes) | per token at glm5 dims |
|---|---|---|---|---|
| 1 | hc stream state, double-buffered | `hyper_range_prime` x + `hyper::post` out | 2·S·H·4·t | 128 KiB |
| 2 | pre/norm transients (y, h, z, ffn_out; mixes/gates small) | `hyper::pre`, `rms_norm` | ~4·H·4·t | 64 KiB |
| 3 | grouped-MoE staging: z16 (U·2H), g/u/act (3·U·4F), a16 (U·2F), d_csr + y_pair (2·U·4H), moe_out (4H) | `moe_ffn_grouped_prefill_sigmoid` | [U·(10H + 14F) + 4H]·t | 560 KiB |
| 4 | ppN boundary slots (tx + rx of `[t, S, H]`) | `prime_cache_hyper_ppn` / `PpNRt::tx/rx` | 2·S·H·4·t | 128 KiB |
| 5 | MLA q/attn transients (q planes heads·qk, attn heads·v) | `mla_attn_cached` core | ~2·64·256·4·t | 128 KiB |
| 6 | DSA k-pool SCORE plane | `mla_kpool_indices` | t·(X/P)·4 | 4·X/P per call token (ctx-coupled!) |
| 7 | DSA k-pool idx | `mla_kpool_indices` | t·(topk/P + tail)·4 | ~2 KiB |
| 8 | prime-tail hiddens + hn | `hyper_prime_tail` / `prime_chunk_hyper` exit | 2·H·4·t | 32 KiB |

Sum of the request-scaled classes 1-5,7,8 is ~1.0 MiB/token of ALLOCATION traffic per
card; the measured RETAINED high-water is ~0.8 MiB/token/card (vramwatch.csv: the
8,072-token prime raised dev1 by +6.3 GiB), consistent with the async pool retaining
the peak live set per size class. Class 3 alone is 0.55 MiB/token — the grouped arm is
the dominant term, but bounding IT alone would not have fixed the wall: classes
1/2/4/5/8 are another ~0.45 MiB/token of the same per-call scaling.

Request-scaled state that is NOT workspace (legitimate, admission's to charge):

| # | allocation | site | size formula (bytes/token) | at glm5 dims |
|---|---|---|---|---|
| 9 | MLA latent rows (f32, eager at session ctx) | `Cache::new_inner` LatentKvCache arm | Σ_latent W·4 = 11·512·4 | 22.0 KiB |
| 10 | resident k-pool keys (lazy) | `mla_kpool_indices` | Σ_latent Dh·4/P = 11·128 | 1.4 KiB |
| 11 | indexer tail ring | `Cache::new_inner` | FIXED (ring rows · 2Dh · 4 per layer) | ~0 (fixed ~60 MiB) |
| 12 | returned `hiddens` stack (spec prompt_h / capture) | chunked prime exit, LAST stage | H·4 per PROMPT token | 16 KiB |

ADMISSION SAW NONE OF IT: `cache_bytes_per_token_for_plan` (memra-kv) sums only
`LayerKind::FullAttention` KV planes. glm5's 34 KDA layers are `Recurrent` (0/token,
correct) and its 11 MLA layers are `StatePlan::LatentKvCache` — unmatched, so the
coefficient is literally 0 and the whole first-session cache allocation was learned
into the 155 MB "fixed residual" (`AdmissionCostModel::observe` measures the admit-time
free-VRAM delta; prime-time workspace growth is never observed at all). This is the
same accounting gap the prefix-latent lane named. The 262k cell's receipt line
(`request cost: ... = 0 B/token x ctx + 155MB fixed`) is exactly this arithmetic.

## 2. The bound (deliverable 1)

* PORT 93927b1fac (lane/glm53-1m-demo, cherry-pick -x): the ppN prime walks the SAME
  `hyper_prime_ranges` schedule as the single-engine walk, `queued_after + (t - end)`
  preserving the request-level `seq_end` invariant per chunk. Classes 1-5,7,8 then
  scale with `min(t, 4096)` (PRIME_CHUNK_MAX_TOKENS), not the request:
  W_chunk = 4096 x ~0.9 MiB ~= 3.5 GiB/card worst case, constant in prompt depth.
  What the port bounds: every per-call transient. What it does NOT bound: class 6
  (score plane, chunk x ctx/P — ctx-coupled by design of the k-pool selection), class
  12 (`hiddens`, prompt-scaled on the last stage — consumed by the MTP-spec prompt_h
  and the embed capture), and classes 9-11 (session state). Those move to admission.
* Persistent workspace across chunks (launch-diet WINDOW-20260830 §3.4, the step37
  DECODE_V2 class) is a REAL further increment (alloc churn -> stable buffers) but it
  is a host-time/pool-churn diet, not a capacity bound; the capacity math above is
  already closed by the chunk walk. Left named for the perf lane, not taken here.

## 3. Admission (deliverable 2)

Per-family, defaulting to current behavior elsewhere:

* memra-kv `cache_bytes_per_token_for_plan` now charges `StatePlan::LatentKvCache`
  layers: `W·4` latent row + `Dh·4/P` resident pool key (P from `cfg.glm5`; a latent
  plan without glm5 config charges P=1, conservative), + the flat indexer plane
  `2Dh·4` ONLY when the tail ring is disabled (`MEMRA_DSA_INDEX_RING=0`), mirroring
  `Cache::new_inner` exactly. Every non-latent family's coefficient is unchanged (the
  new arm matches a plan state no other family compiles).
* memra-server `AdmissionCostModel` gains a prefill-workspace shape from the engine
  (`HybridModel::hyper_prime_workspace_shape`, None for every non-hyper model),
  charged as `HyperPrimeWorkspaceShape::admission_bytes(prompt)`:
  - `chunk_token_bytes x hyper_prime_call_rows(prompt)` (classes 1-5,7; the call-rows
    helper re-derives the same env-sensitive schedule the prime walks, so the
    monolithic rollback `MEMRA_PRIME_CHUNK=0` is charged honestly too)
  - `call_rows x prompt/P x 4` (class 6, the k-pool score plane)
  - `H x 4 x prompt` (class 12, hiddens).
  KEYED ON PROMPT ROWS, not ctx_cap: `request_ctx_cap`'s ctx-bounded arm gives a
  `max_tokens`-omitted request the WHOLE server window, and charging the window would
  have refused every vendor-default short prompt on a deep-window box for workspace
  it never allocates (worker test's vendor-short arm pins this). The context planes
  keep charging ctx_cap in `context_bytes` — that IS the session's eager latent
  allocation (`Cache::new_planned` books `max_ctx x width` latent rows up front).
  The estimate is the FORMULA, not tonight's slope; the receipt slope is the
  validation (formula ~1.02 MiB/chunk-token >= measured ~0.8 MiB/token/card).
  Rollback seam: `MEMRA_ADMIT_PREFILL_WORKSPACE=0` (FLAGS.md row in this lane).
  NAMED CONSEQUENCE for the 2-card recipe, now visible instead of latent: a
  `max_tokens`-omitted request on a `MEMRA_CTX=262144` box charges (because it
  eagerly ALLOCATES) ~6.3 GB of latent planes — one such session fits, a second
  correctly defers/429s. That is the box's real capacity; if the product wants more
  vendor-default concurrency at 262k, the follow-up is lazy/growable latent
  allocation in the engine, not a smaller admission charge.
* Refusal path is the EXISTING admission shape: `[admit-oom] VRAM reject ... HTTP 429`
  (`EngineError::rate_limit`) when no attainable headroom exists, BEFORE
  `prepare`/prime touches the GPU; queue-defer (FIFO) while active sessions might
  drain, exactly as every other family behaves today.

## 4. Predicted arithmetic (prediction only; box receipts are the remaining rungs)

2-card recipe shape (9.0/8.7 GiB free at ready): request cost at depth D
~= 39.4 KiB/token x D (22.0 latent + 1.4 keys + 16 hiddens) + 4 KiB/token x D (score)
+ 3.5 GiB chunk workspace. Headroom 9.0 GiB => admission clears D up to ~130k tokens
per card-agnostic primary arithmetic; the physical last-stage card (hiddens + its ~5-6
latent layers + workspace) sustains a similar order. The 8k wall should move to a
>100k-class admission-governed depth, with over-budget requests 429ing cleanly. The
90 s TTFT ceiling still caps the PRODUCT at ~66k on this arm (737 tok/s prefill) —
unchanged by this lane.

3-card 1M arm (the last-stage 97.2 GiB DSA-kpool OOM, prefix-latent window done-line
2026-08-30T04:55Z): with the ppN chunk walk the per-call transients bound at 3.5 GiB,
but class 6 alone at X = 1M, C = 4096 is 4096 x (1,048,576/4) x 4 = 4.0 GiB per MLA
layer call (transient, pool-recycled across the stage's latent layers) and `hiddens`
at 1M is 16 GiB on the last stage — the 1M demo needs either the hiddens consumer
narrowed (plain sessions never read the full stack) or the last stage's latent-layer
count rebalanced. PREDICTION: chunked ppN + this admission model makes the 1M prime
admission-refused on the 3-card shape as configured (16 GiB hiddens + ~26 GiB latent
+ 4 GiB score + 3.5 GiB workspace > any single card's headroom there), i.e. the clean
4xx replaces the 97.2 GiB mid-prime OOM; the 1M SKU itself needs the hiddens diet
(named, not taken here).

## 5. Gates — GREEN and RED, run (receipts/, rig 5090, flock + TF32 off; tree SHA in receipts/BINARY-SHA)

| gate | arm | bar | result |
|---|---|---|---|
| `glm5_moe_grouped_prefill_gpu` | grouped-vs-sequential bit-gate re-run (incl. `wrong_programs_fail_the_gate` red arms, bitwise knee) | reference band + routing exactness | 8/8 PASS |
| `glm5_chunked_prime_gpu` | single-engine chunk walk vs `memra_reference` + monolithic sibling, depths 64/200/501 x chunks 16/32/37/128, decode continuation, teeth | calibrated band 2e-5 | 4/4 PASS |
| `hyper_connections_gpu` | the truth half of the ppN composition chain (unsplit walk vs reference) | reference band | 6/6 PASS |
| `glm5_prime_capacity` | transient sub-quadratic + residency assertions | capacity bound | 5/5 PASS |
| `glm5-hyper-ppn-gate` | CHUNKED ppN prime vs chunked unsplit prime over the SAME schedule, hiddens stack included: chunk=8 P=24 (N=2, N=3), chunk=32 P=200 (6 calls), monolithic control P=200, baseline P=6 | BIT identity, split hard-asserted | 5 invocations, all arms BIT-IDENTICAL |
| `glm5-hyper-ppn-gate` RED M5 | hiddens chunk copied to offset 0 | must FAIL on the NEW hiddens compare ONLY (invisible to logits/decode) | FAIL [prime-twin] 1/10 (`prime hiddens stack` 3072/3072), others green, exit=1 |
| `glm5-hyper-ppn-gate` RED M6 | first chunk never primes | must FAIL prime+decode | FAIL [prime-twin] 10/10, others green, exit=1 |
| `glm5_admission_cost` (CPU) | latent coefficient formula on the mini glm5 plan + per-stage split + chunk-bounded charge identities | exact-byte formulas | 3/3 PASS |
| worker admission tests (CPU) | the 262k cell's own rungs: 262k refused, 7,108 admitted, vendor-default 541-token prompt on a 262k window admitted; RED = the pre-lane 0 B/token model admits 262k | refusal arithmetic vs the banked free-at-ready | PASS (473/473 worker suite) |
| box rung (REMAINING, named) | 2-card recipe box, depth ladder 8k/16k/45k/130k through the serving surface + vendor-default sampled probe | 429 before prime on over-budget, no `[engine-error]` line; in-budget rungs serve | NOT RUN — needs the box |
| box rung (REMAINING, named) | 3-card 1M prime re-run under chunked ppN + admission | clean refusal replaces the 97.2 GiB mid-prime OOM (§4 prediction) | NOT RUN — needs the box |
