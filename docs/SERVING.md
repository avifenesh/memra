# Serving — the OpenAI surface and the replica fleet

This is the serve-surface doc: fleet topology and measured throughput, the isolation
contract, the OpenAI tools surface, the gateway listing surface (`/v1/models` schema,
rate-limit headers, graceful drain), safetensors/FP8 checkpoint serving, cross-request
prompt caching with per-tenant `cache_salt` isolation, and the honestly-stated numeric
edges of batched serving.

> Numbers here are engineering receipts, each labeled with its rig — see
> [Rigs](PERFORMANCE.md#rigs--what-was-measured-on-what) for what each label is. A number
> without its rig label is not a number: the same cell moves 5-12% between two pods of the
> same SKU and ~2x between a 188-SM and an 82-SM board. The open gaps stated below travel
> with the wins.

The qualified published shape is a 2x RTX PRO 6000 Blackwell pair. PP-3/PP-4 infrastructure is
experimental until its exact-topology and sampled-serving battery lands. Which provider hosts which role —
serving, verification, tuning — is a deployment fact and lives in darklanes, not here. The
Provider runbooks are not in this repo at all — they moved to darklanes with the rest of
deployment. What stays here is `deploy/systemd/`, the supervision contract.

Multi-GPU serving has two shapes, and which one applies is decided by whether the model fits
on one card:

- **Replica fleet** (the default for a model that fits): N independent `memra-server`
  processes, one engine per GPU (`Engine::new(0)`; `CUDA_VISIBLE_DEVICES` is the placement
  mechanism), fronted by an admission proxy. This is the throughput shape — see
  [Fleet tooling](#fleet-tooling).
- **Pipeline-parallel PP-2** (qualified for a model that fits only across the pair): ONE engine
  process, the layer trunk cut into stages, each stage's weights and KV resident on its own
  card. Opt-in via `MEMRA_PP_STAGES` / `MEMRA_PP_DEVICES`; see
  [Pipeline-parallel serving](#pipeline-parallel-pp-2-serving) below for what is gated and
  what is refused.
- **Pipeline-parallel PP-3/PP-4** (experimental): the same stage-local weight/KV and peer-copy
  runtime, with `MEMRA_PP_WAVE=1` filling stages from independent request or prompt-microchunk
  waves. It is default-off and has no published performance/support claim yet; see
  [the design decision](decisions/PRO6000-MULTICARD.md).

Tensor parallelism is neither. Step-3.7 has an exact model-specific TP2/TP4/EP implementation,
but that does not establish generic dense TP or transfer a topology/default to another model.

### Step topology contract (placement only)

Step-3.7's model-specific planner validates deterministic PP, TP, EP, and hybrid
rank placement through eight ranks. It checks the checkpoint's alternating 64/96
query-head geometry, eight KV heads, and native 128-row E4M3 scale-block
boundaries before CUDA allocation. Current product acceptance remains one through
four GPUs.

This contract refuses illegal layouts early. It is not evidence that serving
TP/EP, native P2P, or multi-card throughput is complete; those surfaces require
separate official-model and target-hardware gates.

## Fleet tooling

(Not to be confused with the OpenAI `tools` API surface — that is
"[OpenAI tools surface](#openai-tools-surface-serve-tools-lane-2026-08-02)" below.)

| tool | what it does |
|---|---|
| `tools/serve-fleet.sh start\|stop\|status\|restart` | declarative fleet supervisor: brings up `REPLICAS_PER_GPU` replicas per GPU in `GPUS`, fronts them with the proxy, health-loop restarts anything that dies. systemd-free; pidfiles under `$FLEET_RUN` |
| `tools/serve-proxy.py` | least-outstanding reverse proxy with per-backend admission cap (default 8 = the engine's exactness-tier batch width and the two-replicas-per-GPU anti-thrash bound). Bounded FIFO queue with deadline → 429 + Retry-After; `/health` + `/metrics` JSON. Request bodies default to 32 MiB with bounded buffering and can be raised explicitly to the server's 192 MiB ceiling via `--max-body-mb 192`; the proxy always rejects dual credentials and authenticates headers before accepting a body. |
| `tools/load-serve.py` | concurrent OpenAI-format load harness: aggregate output tok/s, p50/p95 latency, JSONL per load point |
| `tools/serve-smoke.sh` | OpenAI-surface smoke gate for a single server |
| `deploy/systemd/memra-server.service` | example unit for a **single supervised instance** (the other deployment shape — `serve-fleet.sh` is the systemd-free multi-replica path). `Type=notify` with `READY=1` after the models load and the socket binds, `WATCHDOG=1` pings only while inference is live, `STOPPING=1` + `EXTEND_TIMEOUT_USEC` so a drain is not SIGKILLed, and exit 70 (unrecoverable GPU) distinguished from exit 1 (bad config). Copy, do not symlink: every path is site-specific and the value is the supervision contract in the directive choices, each commented with the failure it prevents |

## Measured numbers (Qwen3.5-9B Q8_0; receipts in `research/`)

- **Single replica (H100, rented pod):** temp-0.7 c=8/16/32 medians **654/657/659 tok/s** after the
  batched decode tick (z-batched FA + KV append, device sampling, lean logits — +25-36%
  over the pre-batched tick; N=4, `research/batched-tick-inc2-20260801/`; chunk-8 era —
  see the exact-16 tier below).
- **Managed fleet, 3 rented H100s x 2 replicas (v0.60-validated):** **1,477 tok/s** through the
  admission proxy at c=96 (N=2 interleaved passes: 1477.0/1473.1), zero 429s/5xx —
  managed now matches the v0.59-era 1,480 direct number (the ~7% admission-overhead gap
  closed at the fleet level). Chaos-tested: SIGKILL a replica mid-load, breaker DOWN the
  same second, supervisor restart +2s, backend UP +9s, 8/768 requests lost (exactly the
  victim's in-flight cap), aggregate across the kill 1,487 tok/s; greedy hash identical
  on all 6 replicas in every condition, 18/18 (`research/fleet-v060-20260801/SUMMARY.md`).
  The proxy cap (8) was calibrated on the v0.59 core — the cap re-sweep is pending the
  next box window (stale-verdict risk flagged in the validation summary).
- **Single replica (RTX 5090 Laptop — the local rig, exact-16 tier):** with the Q8_0 split-plane mirror
  (`MEMRA_Q8RP=1` on 24GB; Hopper default), the worker auto-selects decode chunk 16 —
  c=16 median **494.5 tok/s vs 416.4** at chunk 8, same mirror, interleaved N=4
  (**+18.8%**; +33.8% vs the mirror-less baseline); c=32 at `MEMRA_CTX=2048` runs
  **502.1** with 128/128 ok (single run; `research/batched-tick-inc3-20260801/`).
- **Historical fixed-solo m=1 measurement** (serve-path phase 2, 2026-08-05): with
  `MEMRA_SERVE_B1FAST=1`, a lone session's tick runs `decode_layers_eager` verbatim instead of
  the batched body, inheriting the whole m=1 fusion chain. Order-paired N=5 on the 5090 (82 SM), decode-only `step_p50`:
  **+8.33%** on the 9B (123.7 → 134.1 tok/s, 5/5 wins) and **+5.19%** on the 27B (43.6 → 45.8,
  5/5); c=8 saturation flat (−0.00% / −0.18%). The arm closed the c=1 performance gap phase 1
  measured against naked decode and put serve c=1 level with the same-board `run-gen`
  denominator (134.8/134.5/134.0). It also **retired the solo graph door as a performance win**:
  `GraphSession` replay amortized the same launch overhead this removes outright, so with the
  fast path in place the door was a net loss at every length measured out to mt=1024. Both
  doors are OFF by default since 2026-08-13 because a request gaining a peer otherwise changes
  numeric program mid-stream. These remain fixed-solo opt-ins, not the serving default (FLAGS
  §serve; `research/servepath-p2-20260805/`, `research/eosclass-20260813/`).
- **Spec fast lane, now REQUEST-CONDITIONED inside one process** (lane/spec-gate, 2026-08-07 —
  this supersedes the "run spec and bulk as separate server processes" guidance): MTP
  speculative serving is a single-stream latency tier — 1.82x plain serving at c=1 on the 27B
  (131.8 vs 72.5 tok/s) — and plain batching overtakes between c=2 and c=4 because the spec
  path is a serial burst QUEUE, not a contended one (phase (a) steps each spec session's whole
  burst in a host loop; phase (c) excludes spec rows from batched decode). Pooling the verify
  is REFUTED at a 16-column exact-kernel width ceiling (`research/spec-scaling-20260806/`), so
  the answer is scheduling policy: **one server now admits spec only while `active+1 <= 2` and
  DEMOTES live spec sessions into the batched phase at `active >= 4`**, with `active==3` a
  hysteresis band and demotion one-way per session. The handoff is a real cache transfer
  (`(cache, next_pred)` into the session's cache + `device_next`, a carried pending flushed
  first) and is byte-exact for greedy: a session demoted mid-generation emits a stream
  byte-identical to one batched from the start. Measured q9 on the 5090, N=5 interleaved: the
  gated curve tracks spec at c=1-2 (251.2 tok/s, 1.81x over batched) and batched at c=4-8
  (504.7 tok/s, 2.03x over always-spec), with per-stream p50 at c=8 equal to batched's 1.963s
  rather than spec's 3.973s. Sampled and constrained spec sessions do not demote and stay on
  the serial path, bounded by the admit ceiling. The stated reason for excluding SAMPLED
  sessions — "its `next_pred` is a greedy argmax, so the handoff would inject a greedy token
  into a sampled stream" — **stopped being true on 2026-08-19** (`MEMRA_SPEC_SAMPLED_BOUNDARY`
  draws it from the request's target instead). Sampled demotion is therefore unblocked in
  principle and is a follow-up lane, NOT lifted with the boundary fix: it needs its own
  handoff-exactness cell (the batched row would continue a session whose Philox stream lives
  on the SpecSession) and mixing the two claims in one diff is how an untested handoff ships.
  Constrained sessions still never demote (unmasked argmax). One residual, disclosed: a first-wave TTFT p95 transient (0.423s vs never-spec's
  0.017s at c=4) confined to the at-most-`LOW` sessions admitted before a load ramp — p50
  matches never-spec; set `MEMRA_SPEC_GATE_LOW=0` to never admit spec if cold-ramp p95
  outweighs c=1 throughput. Flags: `MEMRA_SPEC_GATE` (rollback seam),
  `MEMRA_SPEC_GATE_LOW`/`_HIGH`. Receipts: `research/spec-serving-20260801/`,
  `research/spec-gate-20260806/`. **The defaults are placement-aware:** a current-train q9
  re-sweep keeps single-card at LOW=2/HIGH=4 (spec wins c=1/2, plain wins c=4), while
  sharded cross-device PP-2 uses LOW=0/HIGH=1 because plain wins every q9 and step35
  c=1/2/4 cell. Within admitted single-card requests, the 2026-08-08 K policy uses `K=3`
  for cold prompts and `K=2` when a prompt of at least 1024 tokens actually resumes at
  least 1024 tokens. This is intentionally shallower on cached-long: K=2 beat K=3 on
  both q9 and q27. `MEMRA_SPEC_K` remains the operator pin, including `0` for plain.
  Receipts: `research/specplace-20260808/`, `research/kpolicy-20260808/`.
- **Spec engages on prefix-cache hits** (v0.93.0, lane/spec-on-cache-hit — whole-entry
  restores only; **SAMPLED hits included since lane/sampled-hit-spec 2026-08-19**). A spec
  session's boundary capture publishes its prefix entry with the MTP
  draft plane and boundary hidden alongside the trunk KV (`PREFIX_ENTRY_LAYOUT_VERSION` 2;
  plane + hidden bytes counted in the entry budget). On an unconstrained, spec-eligible
  WHOLE-entry hit the worker re-arms a warm spec session instead of downgrading:
  qwen/MTP via `spec_session_from_restored` (restored trunk + draft plane + entry anchor;
  the engine feeds any prompt suffix itself, mirroring prefill_tick's program arms), gemma
  via `gemma_spec_session_from_restored` (drafter seed regenerated from a non-empty suffix
  feed; full-cover gemma hits stay plain by design). `spec_restore_refusal` refuses
  `entry_pos != fed_len`, so a partial restore can never route into a spec session — the
  mid-entry-restore hazard class the lcprestore rollback recorded stays closed. Every
  non-convertible shape (no draft plane, constrained, partial hit, conversion decline)
  serves plain exactly as before **and the downgrade log line names the reason**;
  `cached_tokens` counts the restored prefix only.
  - **Sampled hits.** v0.93.0 shipped this greedy-only and the DE deploy then measured the
    consequence: 3 cache hits, 3 plain downgrades, 0 restores, because the paying tenant's
    traffic is sampled — which is what the OpenAI surface defaults to (`temperature` 1.0).
    The greedy-only premise ("a sampled hit's first token must be host-sampled") did not
    hold against the code it was protecting: the cold sampled spec path did not host-sample
    its own first token either — it argmaxed it, as did every continuation burst. Sampled
    spec is distributionally exact, not byte-equal to plain sampling (see the `sampling`
    note under the spec-burst arm), so the restore's identity target is the COLD SPEC
    session, not plain: per seed, a full-cover sampled hit is byte-identical to the cold
    sampled leader, with the same accepted/drafted counts. **That argmax is itself fixed
    since lane/sampled-spec-quality (2026-08-19) — see "Sampled boundary tokens" below —
    and the restore draws its seed by the same rule from the same row, so the per-seed
    identity is unchanged.** Rollback: `MEMRA_SPEC_RESTORE_SAMPLED=0` restores the v0.93.0
    posture.
  - **Sampled restores are SOLO-ONLY** (`MEMRA_SPEC_RESTORE_LOAD_GUARD`, default on,
    2026-08-19). Measured on the sold shape: the lever is 1.623x solo (192 out) and 1.350x at
    c1 (60 out), then **0.669x at c2** and 0.813x at c16 — the crossover is between c1 and c2.
    Two concurrent sampled spec sessions already cost a third of aggregate throughput, because
    each burst is stepped in a serial host loop and spec sessions are excluded from batched
    decode, so two of them serialise what would otherwise be a 2-wide batch. The reason this
    cannot be cleaned up after the fact is that a sampled spec session is the ONE kind the
    tick's demotion sweep will not hand back to the batched path (its sampler history and
    Philox stream live on the session; that handoff is a separate, unmeasured lane). A greedy
    restore in the same position IS demoted the moment load arrives and measures 1.00x at c16,
    which is why the guard is sampled-only. So the admission is conservative up front: a
    sampled restore is admitted only when the request is SOLO — demand `max(worker-visible,
    HTTP in-flight)` <= 1, read as late as possible, after the restore's own KV copy, because
    the head of an arriving fan-out is invisible in `active + queue` at tick top. The refusal
    names itself: `[spec-restore-guard] sampled restore REFUSED: demand N > SOLO watermark 1
    (...)`. An operator `MEMRA_SPEC_K` pin or `MEMRA_SPEC_GATE=0` owns the policy and bypasses
    it.
  - **Sampled boundary tokens are SAMPLED** (`MEMRA_SPEC_SAMPLED_BOUNDARY`, default on).
    A burst's first emitted token does not come from the accept walk — it comes off a
    logits row that already exists (the prime's last row on a cold burst, the row after the
    last committed token on a continuation burst, the entry's boundary row on a converted
    hit). All three were `argmax` in both sampling regimes, so a sampled stream took a
    greedy token once per burst. MEASURED, not estimated, on 18 real agent prompts at the
    API's own default temperature (1.0), 256 tokens each: **3.52% of generated tokens were
    boundary tokens at c=1 (one per 28.4 — below the 1-in-32 the burst size suggests, since
    bursts also end on EOS, budget and admission yield), and 48.8% of those draws returned a
    different token than the argmax they replaced ⇒ 1.71% of a customer's tokens came from
    the wrong distribution.** At c=8 the rate falls to 0.50%, and the reason matters: the
    spec admit/demote gate keeps only 2 of 18 requests on the spec path under load, so the
    exposure is CONCENTRATED in the low-concurrency interactive tier, not diluted by scale.
    Every boundary token is now drawn from the request's
    own filtered/penalized target through the SESSION's Philox stream — never a second
    stream — using the same primitive the full-accept bonus uses, so `sample_check`'s
    distributional oracle covers it (arm 9: 20k draws, TV against the CPU-referenced target
    with the Monte-Carlo floor printed, teeth cells for the argmax rule and an unpenalized
    target). Greedy is byte-unaffected.
  - **Penalty history spans the SESSION** (`MEMRA_SPEC_PEN_SESSION`, default on). The burst
    seeded `pen_hist` from the `prompt` slice it was handed, so a continuation burst — the
    majority of a stream's tokens, and all of a converted hit's — restarted the penalty
    window from nothing and the burst's own boundary token was never recorded at all.
    `repetition_penalty` / `frequency_penalty` / `presence_penalty` therefore did not do
    what the API contract says over a long stream. The window is now the last
    `max(penalty_last_n, 64)` tokens of `committed ++ prompt`, matching the plain sampler's
    own history; byte-identical to the old posture for a cold turn-1 burst at the default
    window. This is also what LIFTED the penalized-sampled restore refusal: a restored
    session's `committed` is the whole prompt, so its window is the cold session's window.
    With the door shut the refusal returns and names the flag.
  - **A restored session republishes its own boundary**
    (`MEMRA_SPEC_RESTORE_REPUBLISH`, default on). Publication used to be armed for cold
    sessions only, so a namespace learned exactly ONE boundary: turn 3 of a growing
    conversation could never hit a longer prefix than turn 2, and multi-turn spec paid a
    constant rather than a growing share of prefill. A restore with a non-empty suffix now
    captures its own prompt-end boundary after the feed (whole-entry: `pos == fed_len`, so
    the mid-entry hazard class stays closed) and the existing publication sweep publishes
    it. Consequence to know before quoting the identity law on multi-turn traffic: the plain
    tier cannot publish an extended entry, so from turn 3 on a spec boot restores a LONGER
    boundary than a plain boot can and the two prime different suffix lengths — a near-tie
    can then flip, exactly as the banked r3 two-programs finding describes. `spec == plain`
    remains a per-shape law (same cache state, same program), not a per-conversation one.
    The state itself is gated by a round-trip cell: a full-cover hit on a republished entry
    reproduces its publisher's own continuation byte-for-byte.
  Gates: `tools/spec-on-cache-hit-gate.sh` (now IN the release battery) — hit-engage +
  byte-identity vs the spec-off replay on both models, the sampled cells (engagement,
  per-seed identity vs cold, acceptance parity, suffix reproducibility, a plane-less
  refusal that names itself, penalized sampled hits, the boundary-draw sites and their
  deviation, and the growth/round-trip cells), its `MEMRA_HITGATE_TEETH=1` arm — which shuts
  EVERY door in this arc and asserts the pre-lane behaviour, so the cells cannot pass
  vacuously — and the sold-shape floor (req/s 1.00x, hit rate 0.987) with spec engaged on
  hits. Receipts: `research/spec-cache-20260818/`, darklanes
  `research/spec-cache-20260818/SAMPLED-HIT.md` and `SAMPLED-QUALITY.md`.
- **The plain-serve c=1 gap (task #70) has a measured fixed-solo opt-in; the isolated-identical
  default deliberately pays that cost.** Phase 1 measured serve c=1 trailing the naked
  CLI **−11.74%** on a Q8_0 27B cell (`memra-server` 46.09 tok/s, N=3 median, vs `run-gen`
  naked 52.22, single run; rig `pro6000wk-runpod-community`, same commit and prompt); the
  measured cause — B=1 ran the batched body and missed the m=1 fusion chain — is exactly what
  explicit `MEMRA_SERVE_B1FAST=1` recovers. That optimization is unsafe for a session that can
  gain a peer and is no longer the default. The −11.74% number itself is from the original
  188-SM cell and has not been re-measured on the current core — do not quote it as current.
  Still open: the NVFP4
  **spec** serve path at **−8.66%** (serve 170.55 vs bare 186.72, rig `pro6000wk-runpod`,
  also a pre-H3 measurement) — the spec tier runs its own burst loop that the `b_n==1`
  fast path does not touch. Receipts: `research/q27-deepdive-20260805/RESULTS.md` §4
  (phase 1), `research/servepath-p2-20260805/RESULTS.md` (the fix + the H1 refutation).

## The isolation contract

Greedy serving is **isolated-identical under concurrent load at defaults**: a request's
output tokens are byte-identical whether it arrives alone or inside a full batch. This is
gated, not assumed — the serve gate replays the same prompts at c=1 and c=16 and
byte-compares every stream.

**Read the gate's exact scope before quoting the contract as unconditional.** The gate runs
16 prompts at **96 max_tokens** with all sessions arriving together, i.e. at *equal* depth.
Outside that shape a 768-token greedy request diverged from its own solo reference at byte
1347 (≈ token **331**) when it shared batched decode with sessions **staggered to different
depths**, and on a second run the divergence moved to byte 2379 (lane/spec-gate receipt,
`research/spec-gate-20260806/logs/exact/`, arm `REF_LOAD`).

**lane/iso-gap (task #91, 2026-08-07) reproduced that receipt on demand and attributed it —
the two mechanisms this paragraph used to name are both innocent** (receipts
`research/iso-gap-20260807/`):

- *Depth staggering moves nothing.* At the engine tick, with the program family held fixed,
  a co-resident session at ANY other depth — including across a `fa_split_keys` ladder-rung
  boundary, B=2..8, three rungs, 300-step horizons — changes **zero bits** of a session's
  logits (`iso-gap-probe`, 8 arms + canary). `decode_step_batch`'s rung guard is
  per-session-correct: every row either shares one rung (the seqs kernel then derives each
  session's split partition from its OWN `t_kv` — the ONE-PARTITION law) or all rows take the
  per-seq eager loop. The property is now pinned by the `isogap` fast-gate arm, which places
  the straddle per-rig.
- *The real carrier was the solo↔batched **program flip at the co-residence boundary**.* A solo
  session could run the m=1 fused trunk (`MEMRA_SERVE_B1FAST`) or GraphSession replay
  (`MEMRA_SERVE_GS`); the moment a second session arrived mid-stream, its ticks flipped to the
  batched body — a different FP composition. The iso-gap lane measured solo-vs-loaded divergence
  at byte 659 and proved that `B1FAST=0 GS=0` made solo and loaded streams byte-identical; the
  moving byte was arrival-tick jitter, not a depth-dependent batched-kernel defect.

That transition is no longer a legal default. Dense Q27 made the accepted drift correctness-
visible: with one serially seeded restored target and three serially seeded restored peers, changing
only peer-arrival delay reproduced the historical HTTP-200 11-token EOS hash at 50 and 225 ms.
The trace-free default arm produced two restored-hit hashes in N=5; the B1-off arm produced one.
The default therefore keeps B=1 and B>=2 on the generic batched body for every architecture, and
GraphSession is also off. `decode-batch-gate` config mode enforces B=1-vs-B=N bit identity under
that policy. `MEMRA_SERVE_B1FAST=1` and `MEMRA_SERVE_GS=1` remain explicit fixed-solo measurement
doors only; use them only when admission guarantees the request cannot gain a peer. The measured
cost of the one-class default is the inverse of the historical solo win: about 8.33% q9 and 5.19%
q27 decode-only at c=1 on the 82-SM rig; c>1 is unaffected except as a batch drains to one.

Qwen35-MoE retains a code-level eager-path exclusion as defense in depth. Its earlier sellgate
found the same numeric-class transition selecting EOS at completion tokens 15/17/25, and the
generic B=1 body made the mixed c=2 reducer one hash with 100/100 full-length requests.

The same model class is also excluded from **carried** cross-request prime batches. The
v0.81.0 coldhol default repeatedly concat-primed bounded continuation chunks for long cold
prompts; in the frozen Q35 mixed-c4 cell that path selected EOS at token 26 for both cold
misses. Serial priming and one complete fresh B=2 prime-batch control both reached 60/60, so
whole-fresh MoE batching remains available and dense Q27 retains continuation batching. The
MoE carried door stays closed until the prime-batch gate covers the real multi-chunk sequence
followed by the serving batched-decode trunk (`research/coldfix-20260812/`).

Mixed hit/cold scheduling separately protects cache-hit first-token latency. A full-prompt cache
hit already owns the logits needed for its first decode, so an admitted hit takes that decode
before unrelated synchronous cold prefill. When an interactive completion opens a client-
concurrency slot while cold prefill remains, the scheduler also uses the short
`MEMRA_PRIME_BATCH_HOLD_MS` window as a refill grace. That lets a not-yet-admitted replacement
cross the HTTP/channel boundary before the worker can enter the next cold prime call. The fence
ends after the hit's first token, and cold-only traffic retains dense continuation batching.
Frozen box1 N=5 validates that boundary: Q27 mixed-c4 hit TTFT is p50 18.497 / p95 19.820 ms
against the sold 18.573/21.565 ms envelope, and the clean-throughput knee remains c=16. The
strict cross-run reducer still flags c4 output 144.245 versus 144.462 tok/s (`-0.150%`); both
targets are SELLABLE, but that comparison remains explicit rather than being rounded away.

The contract is over **tokens**, and the default now removes the known FP-program transition as
well: both `b_n==1` and `b_n>=2` run the generic batched body, while GraphSession replay is off.
`decode-batch-gate` config mode requires bit-identical logits for B=1 versus B=N on that program.
The eager trunk remains covered separately in strict mode and can be selected explicitly with
`MEMRA_SERVE_B1FAST=1` only for fixed-solo measurement; its historical stream-level q9
NVFP4-MTP receipt remains in `research/servepath-p2-20260805/`. Qwen35-MoE remains excluded from
the eager door even when requested, as described above. This is also a fixed defect, not a
freebie: the batched
cuBLASLt prefill router and shared-expert-gate GEMMs were m-dependent, so under
cross-request prefill batching a MoE request's own expert selection changed with its
co-arrivals (the supported Qwen3.6-35B had the same defect as the onboards whose serve
gate exposed it — Ornith-35B 6/16, KAT 7/16, both 16/16 after the fix). Default
`MEMRA_ROUTER_PREFILL_EXACT` routes prefill through decode's m-invariant router/gate
kernels; a bit-identical batched twin recovers most of the prefill cost
(`MEMRA_ROUTER_BATCH`, FLAGS §3). Receipts: `research/concat-prime-exact-20260802/`,
`research/fast-router-20260802/`.

**Admission is VRAM-aware** (2026-08-02): once the first admitted session reveals the
model's per-session VRAM cost, further admissions require free ≥ 2x that cost — otherwise
the request *waits* in the same never-rejected FIFO as the session-count cap instead of
failing with a cache-alloc OOM (the c=16 8192-ctx failure mode under resident-if-fits,
caught by the serve gate as instant HTTP 400s — `research/fast-router-20260802/RESULTS.md`).
The first session always admits; an OOM with no active sessions is real capacity and
still errors loudly, with the CUDA error quoted.

### 64-client robustness (lane/admit-oom, 2026-08-06) — gated, not assumed

At `MEMRA_MAX_SESSIONS=64` with spec ON on a 24GB card, the 2026-08-02 cost model
under-charged the live burst and **every one of 64 streams died** with a quoted
`step error: DriverError(CUDA_ERROR_OUT_OF_MEMORY)` (0/64 well-formed, x3 runs; the worker
itself survived — it was never a hang or a panic). Two independent errors, both fixed:

- **The parked-session delta understated the live cost 1.49x**, and a roughly constant
  ~1.3 GiB draft-graph capture-arena transient is not proportional to session count at all,
  so no per-session headroom multiple could cover it. Admission now charges a flat
  `SPEC_SHRINK_RESERVE` (1.5 GiB) on **spec-capable models only** — the plain path is
  untolled and passed c=64 unaided.
- **Retires returned KV to the pinned async pool, invisible to driver `free`**, so the gate
  read a full card while gigabytes sat cached. The gate now reads `free + pool_cached`
  (deferrals 36 → 5, 59 sessions active sustained).
- **Step-OOM parks instead of killing**: a spec step that OOMs despite admission rebuilds
  its request and re-queues at the FRONT (`MEMRA_STEP_OOM_RETRIES`, default 3) — bounded,
  and only for a session that has emitted **nothing** and only on a quoted CUDA OOM, so a
  streamed prefix is never replayed to a client. Parking costs a re-prime: pure latency,
  never a correctness change.

Result: **64/64 well-formed, x3, peak 23.1 of 24.5 GB.** The c=8 no-regression control is
behaviorally identical (+0.49% agg tok/s, zero defer/park events). This is now a *gated*
property, not a claim: `tools/serve-stress-gate.sh` runs in `tools/local-ci.sh` and as the
`sstress` fast-gate arm, and it has teeth — `--teeth` forces the reserve to 16 MB and the
verdict inverts (11/64), so a gate observed only passing proves nothing.
Receipts: `research/admit-oom-20260806/`, `research/serving-density-20260806/VERDICT.md`.

### Config recommendation: send `max_tokens`

Admission sizes each session's KV ladder from the request's own bound. An explicit
`max_ctx` is authoritative. Without one, a finite request uses
`prompt_tokens + max_tokens + 8`; only a request that **omits `max_tokens`** falls back to
`MEMRA_CTX`. At `MEMRA_CTX=32768`, that fallback reserves ladder slack an unbounded client
may never use: measured **6.3% of a 96GB card at c=16 and 12.6% at c=32** stranded on the
9B — more than sealed-prefix duplication costs at the same shape. Right-sized requests
(explicit `max_tokens`) strand ~0%. Set an explicit `max_tokens` in serve configs and
client defaults, and keep the `MEMRA_CTX` fallback at the unbounded-request workload
rather than the maximum. Receipt: `research/serving-density-20260806/VERDICT.md` (Q1).

### Request body ceiling: 192 MiB, explicit, with a clean 413 (lane/hermes-fixes, 2026-08-19)

Every inference route carries an explicit `DefaultBodyLimit` of **192 MiB**, and a body over it
gets an OpenAI-shape `413` rather than a framework-shaped rejection.

The limit is explicit because axum's default is **2 MiB**, which silently capped the surface this
server advertises: a 262,144-token prompt sent as `prompt_ids` is ~2.8 MiB of JSON on its own, so
the advertised context could not be used through the advertised field. The ceiling is sized from
the advertised maxima rather than picked round — prompt 4 MiB + `VISION_MAX_IMAGES` (8) × 12 MiB
raw × 4/3 base64 = 128 MiB + 2 videos × 12 MiB raw GIF × 4/3 = 32 MiB + 4 MiB envelope/tools
headroom = a 168 MiB requirement, rounded to 192 MiB for headroom while staying finite.

Operators fronting the server with a proxy should raise the proxy's own body limit to match, or
the proxy becomes the real ceiling and returns its own error page instead of the API's 413.

Authenticated body parsing has a separate bounded admission gate for large uploads. Requests
with a declared body of at most 1 MiB use their own 32-slot pool, so a slow vision upload cannot
head-of-line block ordinary JSON calls and neither class can create unbounded parser tasks. The
body-read deadline starts at 90 seconds for an unknown-length body and adds a pessimistic 2 MiB/s
transfer budget for a declared length, capped at 180 seconds; this upload bound is separate from
the inference `timeout_ms` promise. If either pool is full, the request fails fast with 429
`body_admission_busy` rather than joining an unbounded waiter queue; it carries `Retry-After: 1`
and the `retry-after-ms` twin. A closed pool uses retryable 503 `body_admission_unavailable`.
All pre-body refusals retain the selected dialect's error body and request-id headers, including
the Anthropic `/v1/messages` contract.

## The exact-16 decode chunk tier

The batched tick decodes sessions in per-model chunks. Default width is **16 on models
where every matmul has a bit-exact 16-batch kernel class** (`decode_batch_exact16_ok`:
the b16 batched-mmvq family — Q8_0 qualifies only through its `_rp` mirror twin), **8
otherwise**; `MEMRA_DECODE_BATCH_CAP` stays the explicit measurement door. Qualifying
steps scope out every m>=16 GEMM/MMQ arm, so chunk-16 output is bit-identical to
isolated decode (gate2 bit-checked at steps 32 and 160). B=32 has no exact kernel class
— chunk policy stays <=16. On the H100 fleet model (9B Q8_0, mirror on by default) the
tier engages automatically on the next deploy; the H100 numbers above are chunk-8-era
and the chunk-16 fleet effect is pending on-box re-validation.

**Capacity envelope (24GB):** the mirror costs ~model-size VRAM, so c=32 sessions at the
default `MEMRA_CTX=8192` exceed VRAM (captured `CUDA_ERROR_OUT_OF_MEMORY` in the
pre-admission-wait receipts; ~27 sessions fit — since the VRAM-aware admission wait the
overflow queues instead of erroring). Set `MEMRA_CTX` to the workload — 2048 clears the
same cell (machine-specific config per the flags doctrine).

## Pipeline-parallel (PP-2) serving

For a model that fits only across two cards. Receipts:
[`research/pp2-batch-20260806/`](../research/pp2-batch-20260806/) (batched decode),
[`research/pp2-spec-20260806/`](../research/pp2-spec-20260806/) (the spec verdict),
[`research/pp2-hardening-20260806/`](../research/pp2-hardening-20260806/) (the fail-closed
guard). Rig for all three: 2x RTX PRO 6000 Blackwell Server Edition 96 GB, sm_120a, CUDA
13.2, SPOT box — **rented**, not owned. Flag reference: [FLAGS.md](FLAGS.md) `MEMRA_PP_*`.

The serving config, minimally:

```bash
MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 \
MEMRA_MODELS="big=/path/to/model.gguf" memra-server
```

The request-conditioned K policy selects `K=0` by default on this sharded PP-2 shape;
no explicit `MEMRA_SERVE_SPEC=0` is required. The server logs
`[pp] cross-device transport: stage0=dev0 stage1=dev1` when the split is live — a config that
silently did not split is the failure mode that banner exists to rule out.

**Exactness: the split adds zero deviation.** `decode-batch-gate --mode pp` records a
reference with the door OFF over the same loaded weights, replays the same token sequence
through the split, and compares every f32 logit of every row of every step bit by bit.
**0 differing bits** on all seven configs — `dev01`, `dev10` (reversed placement),
`singledev` (seam only, one card), `split5` (uneven cut), N=4 (`devices 0,0,1,1`), q27 (64
layers), and `wide` (B=12/16 under the `MEMRA_DECODE_BATCH_CAP=16` door). The B=1 fast path
is its own gate arm (arm 4) against the eager split, since it carries the accepted m=1 fusion
FP gap vs the batched body by design: **3,973,120 f32 logits bit-identical, 0 differing
bits**, across the same six configs.

**Cost: the boundary transfer does not bite at m>1.** q9, 64 steps, 512-token prompts,
greedy, N=5 rep-major interleaved in one lock hold on one binary (medians; cross-run
comparison would be clock-drift invalid):

| arm | B=1 | B=4 | B=8 |
|---|---|---|---|
| door shut, single device | 208.4 | 489.3 | 654.0 |
| split dev01 (**the serving config**) | **204.7** | 487.0 | 646.9 |
| ratio | 0.982x | **0.995x** | **0.989x** |

So batched PP-2 costs **0.5–1.5%** at B=4/8/16, and of that, transport is 0.986–0.997x of the
seam — almost all of the small loss is the seam, not PCIe. Both placement orders agree within
0.3%. Aggregate scaling survives the split: B=8 reaches 3.65x B=1's aggregate.

The B=1 row is a historical explicit-fast-path measurement worth keeping: opening the pp door originally dropped every solo
session off the m=1 fusion chain (the `b1_fast` guard included `pp_cuts().is_none()`), a
permanent **−14.9%** tax on exactly the request shape an interactive 2-card box serves. Fixed
by giving the split its own B=1 path — each stage runs its layer range through
`decode_layers_eager`. `MEMRA_SERVE_B1FAST=1` now opts into that fixed-solo path; the default
generic program is the historical control that measured 177 (0.851x). The PP eager gate remains
to validate the opt-in, not to define the serving default.

**PP-2 speculative policy.** Speculative serving over PP-2 is *correct* — the verify trunk
takes its own stage split and the bit-identity battery is 7/7 ALL GREEN. The old concurrent
crash was the ppN reverse-publication hole; #87 fixed it with stage-behind fences and closed
the formerly fatal placement at c=2/4/8. The later v0.72 head-affinity fix made both
placement orders the same 111-112 tok/s spec class.

Correctness does not make it the throughput winner. q9 spec ON/OFF measures
112.5/223.3, 112.3/340.3, and 112.1/593.4 tok/s at c=1/2/4. On the current batched step35
core the corresponding cells are 35.9/85.7, 36.2/101.6, and 36.7/121.7 (N=3). The
placement-aware default is therefore `MEMRA_SPEC_GATE_LOW=0`,
`MEMRA_SPEC_GATE_HIGH=1`: no PP-2 request enters the serial spec queue, and plain requests
do not pay the spec-only admission reserve. `MEMRA_SPEC_GATE=0` restores always-spec for
rollback and the #87 crash gate; explicit LOW/HIGH values remain measurement overrides.
Receipts: `research/pp2spec-crash-20260807/`, `research/v072-fix2-20260808/`,
`research/specplace-20260808/`.

**What refuses, deliberately.** The four decode paths that have no stage split
(`decode_step_batch`'s unsplit body, `decode_step_dc`, the graph capture wrapping dc, and
`decode_step_t*` spec verify) **fail closed** under an open pp door with a sharded
cross-device placement, behind one shared guard (`pp::refuse_unsplit_if_remote`). They were
not wrong, they were a silent perf cliff with a green battery: an unsplit trunk peer-reads
every remote stage's weights every step, measured **7.4 vs 208.9 tok/s at B=1 (28x)** and
**47.4 vs 657.0 at B=8 (13.9x)**. Exactness was never affected (peer reads return identical
bytes), which is exactly why a refusal rather than a warning was the right call.
`MEMRA_PP_ALLOW_UNSPLIT_BATCH=1` re-admits them as a measurement door only;
`MEMRA_PP_SHARD=0` is the non-measurement escape (weights all home — full speed, forfeits the
capacity PP-2 exists for).

### Experimental PP-3/PP-4 wavefront

The PP-N transport already places complete layer ranges, their weights, and their KV state on
each stage. `MEMRA_PP_WAVE=1` adds the missing 3–4 card serving schedule: the worker forms up to
one exact-width request wave per stage, and the engine advances cells by dependency order while
one host worker owns each stage. Prompt microchunks use the same shape. Every boundary retains
two explicit credits, so a slot cannot be reused until the downstream `rx` has recorded that
wave's read-complete event.

The door requires 3 or 4 distinct CUDA ordinals, native P2P, the resolved double-slot policy,
ModelPlan pipeline capability plus legal selected cuts, and a qualified batched decode rewrite.
Host bounce and repeated devices refuse. Per-device admission charges stage-local context and
learned fixed residency, then projects only missing capacity in the process-global grow-only
boundary slots (including the enabled Step concat-prime high-water). Generic dense/non-Step
concat prime has no cross-device PP split and is routed through individual wavefront primes. A
failed mutating wave taints every affected cache and the worker aborts those sessions instead of
retrying or parking partial state. `/metrics.pp_wave` exposes operator-only ticks, cells, and real
host-walker overlap. `MEMRA_PP_WAVE=0` returns to the serial PP-N walk.

This is implemented infrastructure, not a qualified model surface. Before enabling it outside a
research process, run the PP-N gate matrix in [Testing](TESTING.md#multi-gpu-pp-n-exactness-gates--run-on-the-multi-card-box)
on the exact 3/4-card RTX PRO 6000 host and retain sampled serving/performance receipts.

### v0.73 PP-2 prefill stack and Step trial receipt

Three prefill mechanisms compose on the current Step path, but their performance claims stay
separate:

| mechanism | naked behavior | rollback / scope | measured receipt |
|---|---|---|---|
| PP-2 prime pipeline | adjacent chunks overlap stage 0 of N+1 with stage 1 of N through concurrent stage-owned host walkers | `MEMRA_PRIME_PIPE=0` keeps the same boundaries and runs the serial split; `MEMRA_PRIME_PP=0` is the unsplit reference | 330.0 / 401.8 / 417.6 tok/s at pp512/2048/4096, N=5 interleaved on box2 |
| dynamic auto microchunks | naked PP-2 auto geometry keeps the fixed chunk count, shortens fill, and shrinks the drain tail | `MEMRA_PRIME_CHUNK_SCHED=fixed`; an explicit `MEMRA_PRIME_CHUNK` is authoritative and fixed | 343.8 / 411.8 / 427.7 tok/s, N=5 interleaved on box1; +1.4% / +0.3% / flat versus fixed |
| Step35 fresh-prime batch | simultaneous complete fresh prompts batch weight-streaming work at `m=sum(T)` while attention, KV state, and PP-stage ownership remain per request | `MEMRA_STEP35_PRIME_BATCH=0`; carried/dark-lane caches still use the single-prime fallback | +2.5% at B=2 and +2.3% at B=4 for T=520, N=5 paired on box2; server c=2/c=4 byte identity gated |

All three pass `kernel-check`, `run-gen`, `run-spec` K=1..8, the Step chunk/tick segmentation
family, and liveness canaries. Receipts:
[`research/pipeprime-20260808/`](../research/pipeprime-20260808/),
[`research/microchunk-20260808/`](../research/microchunk-20260808/), and
[`research/primebatch-20260808/`](../research/primebatch-20260808/).

Lever C is deliberately not a naked default. `MEMRA_MOE_GROUPED=1` gives the rented Step pair
its grouped expert path (497.5 / 639.2 / 697.6 tok/s at the three prompt classes, N=5), but the
local 5090 proof cell on resident KAT lost 75.3%; the default-flip gate therefore keeps it
opt-in. The Step trial config sets it explicitly. A sole fresh interactive request also widens
its outer prefill call to at most 8192 tokens unless `MEMRA_PREFILL_TICK` is explicitly set;
this removed the 1024-token segmentation tax without changing concurrent-session fairness.

At the explicit Step trial config, the serve surface measured 0.595 s short TTFT, 6.052 s 4k
TTFT, 12.2 ms 4k cache-hit TTFT, and 36.5 tok/s per stream at c=4. Its historical `~343 MB`
entry is a Step-model 4k example, not a sizing receipt for Q27 or Q35; that trial explicitly set
`MEMRA_PREFIX_CACHE_MB=2048`. With no override, memra now budgets two full-`MEMRA_CTX` entries of
the largest loaded model from trunk KV/recurrent geometry (excluding MTP/NextN head blocks) and
clamps that request to post-load driver-free VRAM minus the 1.5 GiB serving-transient reserve. An
explicit `MEMRA_PREFIX_CACHE_MB` remains authoritative and unclamped. Four simultaneous
different-prefix 4k primes later measured 580.5 tok/s aggregate versus
674 tok/s for the one-call solo class: the pair is compute-saturated, so no concurrent-prefill
scheduler landed. Scale this workload with another pair or a new compute mechanism, not a claim
that queueing can recover a 3K tok/s target. Receipts:
[`research/serve-ready-20260808/`](../research/serve-ready-20260808/) and
[`research/concprefill-20260808/`](../research/concprefill-20260808/).

Exact Q27/Q35 entry sizes come from `prefix_cache_bytes` deltas in
[`research/cachesize-20260813/`](../research/cachesize-20260813/):

| model | prefix tokens | device bytes | device MiB |
|---|---:|---:|---:|
| Q27 | 4,096 | 278,528,000 | 265.625 |
| Q27 | 4,860 | 301,215,744 | 287.2617 |
| Q27 | 8,192 | 400,162,816 | 381.625 |
| Q35 | 4,096 | 103,874,560 | 99.0625 |
| Q35 | 4,860 | 110,964,480 | 105.8240 |
| Q35 | 8,192 | 141,885,440 | 135.3125 |

Operator `/metrics` exposes permanent shape refusal as `prefix_cache_skips_budget` and temporary
live-lease pressure as `prefix_cache_skips_pinned`; the first refusal is also a loud server warning.

## OpenAI tools surface (serve-tools lane, 2026-08-02)

**STANDARD-SURFACE CONTRACT (2026-08-17).** Every model this engine serves to
customers speaks the same full surface, identically: the three wire formats
(`/v1/chat/completions`, `/v1/responses`, `/v1/messages`) and the tools surface
below. A chat template with no tools branch makes a model chat-only — that is a
support GAP, not a supported state: the tools branch is part of per-model support
(alongside quantization and drafter), built against the model's official template
and accepted only by a real agentic-CLI round-trip on the serving surface. The
gemma-4 launch shipped without one and answered 400 to every Codex/Claude Code
request ("chat template has no tools branch") — that class of gap now blocks a
model from being announced as served.

`/v1/chat/completions` accepts `tools`, `tool_choice` (`"auto"`|`"none"`; `"required"` and
named-function forms 400 — the grammar engine isn't wired to tool selection yet),
assistant-history `tool_calls`,
`role:"tool"` result turns, and `reasoning_effort`/`reasoning`. The path is **template +
parsing only — zero engine changes**:

- Tool schemas render into each model's own tools branch. Qwen3.5/3.6 uses its ChatML
  `<tool_call>`/`<function=…>` protocol; HY3 uses the pinned shipping template's suffixed
  `<tool_calls:opensource>` / `<arg_key:opensource>` / `<tool_responses:opensource>` protocol,
  including its no_think/low/high reasoning header. Both are reproduced byte-for-byte from the
  committed vendor template. Models whose template has no supported tools branch (plain gemma4
  and bare ChatML) reject `tools` with a 400 at admission.
- Emitted model-native tool-call blocks are parsed from the generated stream into OpenAI-shape
  `tool_calls` (streaming deltas + non-stream `message.tool_calls`, deterministic ids,
  `finish_reason:"tool_calls"`); argument values coerce per the declared JSON-schema types.
  **Malformed policy:** a block that does not parse is surfaced verbatim as content — never
  an error, never dropped bytes; unterminated blocks flush raw at end of generation.
- **`reasoning_effort` — one surface, per-arch native thinking control** (owner directive
  2026-08-07: every supported model is a thinking model). The reasoning-capable-model
  convention: `low|medium|high` = thinking ON, with the named level passed to templates
  that consume one; `none|minimal` = thinking OFF; **absent = the model's own default**
  (never overridden; no silent behavior change for existing deployments). The named levels
  are per-request token-saving dials, NOT a graded quality ladder: measured on the
  level-consuming step37 template (`research/step37-reasoning-effort-20260829/RESULTS.md`,
  cell12, n=8 vendor-default sampled per level), every level is honored and answers
  sanely, but reasoning depth is not monotone in the level (`high` landed below `medium`
  on both non-trivial prompts) and the absent-field default is the DEEPEST arm by a wide
  margin, so naming any level constrains the model relative to sending nothing. Name a
  level to spend fewer reasoning tokens; do not promise or expect more depth from a higher
  one. `reasoning: {enabled, effort}` (OpenRouter form) maps the same
  way; `{enabled: false}` is the explicit off, `{enabled: true}` thinking on at the
  template default. `xhigh|max|ultra` are accepted as clamp aliases for `high`
  (real default-config clients send them: codex `xhigh` on `/v1/responses`, Claude Code
  `xhigh` on `/v1/messages`). Any other value 400s. **The canonical set —
  `none|minimal|low|medium|high` plus the three clamp aliases — is ONE table
  (`canonical_effort`/`parse_think`) shared by all three request dialects** (chat
  `reasoning_effort`/`reasoning`, `/v1/responses` `reasoning.effort`, `/v1/messages`
  `output_config.effort`): same acceptance, same rejection, same downstream effect on
  every surface (issue #31). An explicit boolean switch (`reasoning.enabled`, Anthropic
  `thinking.type`) wins the on/off decision over the switch an effort level implies; the
  effort value is still validated and still supplies the level for level-consuming
  templates. Caller-visible consequence of the table below: on binary-switch templates
  (qwen class, gemma4) `low|medium|high` render IDENTICAL prompts — the level knob is
  connected only on level-consuming templates (step35, hy3), so tuning it elsewhere
  changes nothing. Per-model mapping (goldens rendered from
  each REAL shipped template: `research/step-sku-20260807/render-thinking-goldens.py`):

  | model class | native mechanism | absent (default) | none/minimal | low | medium | high |
  |---|---|---|---|---|---|---|
  | Qwen3.5/3.6, Ornith, AgentWorld, KAT (qwen ChatML class) | `enable_thinking` switch | thinking **ON** (open `<think>\n`, the template default) | closed `<think>\n\n</think>\n\n` | open `<think>` | open `<think>` | open `<think>` |
  | Gemma-4 family (12B/26B/31B/E4B) | `enable_thinking`, template default **false** | thinking **OFF** (closed `<\|channel>thought\n<channel\|>`) | closed channel | `<\|think\|>` system token + open turn | same | same |
  | Hy3 | template's own `reasoning_effort:` `no_think\|low\|high` | `no_think` (its jinja default) | `no_think` | `low`, open `<think:opensource>` | `low` (clamp — no medium level) | `high`, open think |
  | Step-3.7-Flash (`step35`) | `Reasoning: {level}` string in the system turn; `<think>` tail **unconditional** | no `Reasoning:` line (template default) | `Reasoning: low` (clamp — no off level) | `Reasoning: low` | `Reasoning: medium` | `Reasoning: high` |
  | GLM-5.3-Flash (`glm5_next`) | `<\|system\|>Reasoning Effort: {Low\|High\|Max}` line, ALWAYS rendered; `<think>` tail **unconditional**, no off switch anywhere in the template | `Reasoning Effort: Max` (the template's own `else` arm) | **400**: the template cannot close its think tail (`qwen_think && !think_switch`) | `Reasoning Effort: Low` | `Reasoning Effort: High` (maps UP: no medium rung, `high` is the middle rung; owner ruling 2026-09-02, issue #75) | `Reasoning Effort: High` |

  GLM-5.3-Flash is the second model (after deepseek-v4 0731) whose template distinguishes a
  rung **above** `high`: `xhigh`/`max`/`ultra` canonicalize to `max` there instead of clamping
  into `high`, so `Reasoning Effort: Max` — the model's own default — stays reachable by name.
  `medium` is the one rung it does not define; it maps to `high`, the middle ask onto the
  middle rung (the 2026-09-02 ruling superseded an earlier clamp-down to `low`). The law that
  ruling keeps: a sub-max ask never falls through the template's `else` arm to `Max`, because
  answering "reason less" with the model's deepest setting is the never-corrupt-clamp law read
  backwards.

  Level strings reach only templates that consume one (spawn-time `effort_levels` probe,
  keyed on the jinja's own `reasoning_effort is defined` input test — true for step35, Hy3
  and GLM-5.3-Flash, plus the `qwen_effort` and `glm5` dialect probes);
  binary-switch templates are driven by the on/off half alone, so prompts on models
  that never read a level cannot be perturbed by it. Serve-smoke receipts:
  `research/step-sku-20260807/raw/effort-smoke-*.log` (step35),
  `research/step-sku-20260807/raw/think-smoke-*.log` (qwen + gemma4 arms).
- **`enable_thinking` / `chat_template_kwargs` — the vLLM spelling of the same switch**
  (lane/reasoning-control-20260823). The vLLM-shaped ecosystem turns thinking off with
  top-level `enable_thinking: false` or `chat_template_kwargs: {"enable_thinking": false}`,
  and until this lane BOTH were deserialized away: `ChatCompletionReq` carries no
  `deny_unknown_fields`, so the request was accepted with 200 and served with reasoning ON.
  They are now first-class aliases of `reasoning.enabled`, with identical precedence and the
  identical rendered prompt (`reasoning_effort:"none"`, `reasoning:{enabled:false}`,
  `enable_thinking:false` and `chat_template_kwargs.enable_thinking:false` all render
  byte-identically). Three rules keep the surface honest:
  - **Every other `chat_template_kwargs` key is a 400 naming the key.** This server renders
    templates in Rust rather than executing jinja, so a kwarg it does not implement changes
    nothing about the prompt — accepting it would be the same defect one level down.
  - **Two explicit switches that disagree are a 400**, never a silent coin-flip
    (`enable_thinking` vs `reasoning.enabled`, or the two `enable_thinking` spellings).
  - **A client-explicit thinking-off request on a template whose `<think>` tail is
    UNCONDITIONAL is a 400** naming the model (`cannot disable reasoning`), instead of the
    200-plus-full-reasoning-block it used to get. `ThinkMode::NoThink` being "a documented
    no-op" on switchless templates was documented in the renderer and invisible at the API.
    Scoped to a client-explicit request: an operator `default_reasoning_effort` of
    `none` must never 400 a caller who sent nothing.
- **`max_tokens` is a SINGLE output budget covering reasoning AND content**, and reasoning
  is billed as completion tokens like any other generated token — reasoning is output, and
  there is no reasoning-versus-content distinction in this server's accounting. On a
  thinking-open request a small budget can therefore return `finish_reason:"length"` with an
  EMPTY `content`: measured on `qwen/qwen3.8-27b` at `max_tokens:3072` on a hard puzzle, 3/3
  reasoning-on reps returned empty `content` against 0/3 with `reasoning_effort:"none"`
  (darklanes `research/reasoning-schema-20260823/`). **This is expected behaviour, not a
  billing defect** — you can exhaust your budget thinking, exactly as you can exhaust it
  writing. The remedy is caller-side: send `reasoning_effort:"none"`, drop to a cheaper rung,
  or raise the cap. There is no separate reasoning budget and no `reasoning_tokens` field.
- **THINKING CONTENT IS RETURNED on every format** (owner ruling 2026-08-23: "also thinking
  content should be returned, not only the content itself"), never stripped server-side, and a
  reasoning-off request carries no thinking field at all rather than an empty one. The field per
  format, probe-receipted on both served models (darklanes
  `research/reasoning-schema-20260823/raw-thinking-delivery*.txt`, 16/16):

  | format | non-streaming | streaming |
  |---|---|---|
  | `/v1/chat/completions` | `message.reasoning` (string) + `message.reasoning_details[]` (`{type:"reasoning.text", text}`) | `delta.reasoning` |
  | `/v1/messages` | `content[]` block `{type:"thinking", thinking, signature:""}` | `content_block_start` type `thinking` + `thinking_delta` |
  | `/v1/responses` | output item `{type:"reasoning", summary:[{type:"summary_text", text}]}` | `response.reasoning_summary_text.delta` / `.done` |

- **REASONING IS ALWAYS DELIVERED, and there is no suppression mode.** Because reasoning is
  billed as output, withholding it would charge for output that was never sent. So
  `include_reasoning:false` and `reasoning.exclude:true` **stop the model reasoning** rather
  than hiding the text — they are first-class aliases of `reasoning.enabled:false`, render
  byte-identically to it, and inherit its named refusal on a template that cannot go off. The
  drop path is deleted from the streaming parser rather than left unreachable. If you want the
  reasoning text out of your view, discard the `reasoning` field client-side; if you want to
  stop paying for it, switch it off.
- **The reasoning-effort ladder is PER-MODEL and probed from the template, never assumed from
  a family name** (lane/reasoning-schema-20260823). Each model's lab is the authority on what
  its thinking control means, and the two are genuinely different:

  | model | binary off/on | graded levels | source |
  |---|---|---|---|
  | `qwen/qwen3.8-27b` | `enable_thinking` | **`xhigh` (default) / `medium` / `low`** — an instruction sentence at the head of the system turn; `medium` deliberately injects nothing; `high`/`max`/`ultra` fold onto `xhigh` per Qwen's own hosted-API mapping | Qwen/Qwen3.8-27B card + its `chat_template.jinja` |
  | `ornith-ai/ornith-1.5-35b-a3b` | `enable_thinking` | **none** — `reasoning_effort` appears zero times across every card in the org, both generations, all sizes | Ornith AI card + `chat_template.jinja` |

  Two consequences at the API. On a model **with** a ladder every rung renders a different
  prompt, pinned byte-for-byte against the vendor's own jinja (`ModelCaps::qwen_effort`,
  goldens under `research/reasoning-schema-20260823/`), **and the unset case renders the
  vendor's own default** — for qwen3.8 that is `xhigh` (adopted 2026-08-23; the house default
  is the lab's recommendation, and the old no-instruction bytes were the defect this lane
  fixed, not a behaviour to preserve). An operator who wants the historical bytes sets
  `default_reasoning_effort = "medium"` in the model registry — `medium` is the vendor's
  zero-steering rung — and that knob is deployment configuration, documented, not a hidden
  compatibility mode. On a model **without** a ladder, a graded level **TRANSLATES onto the
  binary axis as reasoning ON**: the caller asked for reasoning and gets reasoning, and the
  rendered prompt is byte-identical to an explicit `reasoning:{"enabled":true}` by construction
  (the template has no depth input for the rung to land on). Stock codex and Claude Code send
  `xhigh` on every request, so this mapping is what keeps default-config agent sessions working
  against binary-switch models like ornith; the named 400s remain for what is genuinely
  unhonourable — unknown keys, wrong types, contradictions, and an off-request the template
  cannot honour. `reasoning_effort:"none"` (off) and `reasoning:{"enabled":true}` (on) work on
  every model. Templates that default to thinking OFF (gemma4, hy3) never needed the
  translation: there a level really does flip reasoning on.
- **`minimal` = OFF, a deliberate divergence from Qwen's hosted API** (which maps
  `minimal -> low`, reasoning on briefly). Decided 2026-08-23 and stated here so nobody
  discovers it by surprise: "use the lab's model" governs each model's *behaviour*; `minimal`
  is a value of *our* API schema, whose promise is that the no-reasoning side is real. A caller
  sending `minimal` is reaching for the least reasoning we can give, and on this stack that is
  genuinely none — on a model that cannot go off, `minimal` is the same named 400 as `none`.
- **Every reasoning field this server cannot act on is a named 400, on every surface.** The
  `reasoning` object understands exactly `enabled`, `effort`, `exclude`; `reasoning.max_tokens`
  and any other key refuse by name, as do wrong-typed values (previously silent on chat while
  `/v1/messages` already refused them — one schema means one answer to the same malformed
  request everywhere). On `/v1/messages`, `thinking.budget_tokens` refuses: there is no lever
  that can cap a reasoning segment (`GenParams` is `max_new`/`max_ctx`/`eos`, and a `</think>`
  stop string would end the whole request), so accepting it would promise a spend cap we cannot
  keep. `thinking.type` accepts `enabled|disabled|adaptive` and names anything else;
  `output_config` accepts only `effort`. On `/v1/responses`, `reasoning.summary` accepts only
  `"auto"` because this server does not summarise — it returns the reasoning text verbatim.
  `chat_template_kwargs.preserve_thinking` refuses in **both** directions, because the qwen arm
  does not replay a prior assistant turn's `<think>` block at all — and note *which* value the
  vendor defaults to: qwen3.8's template replays when the kwarg is **absent**
  (`preserve_thinking is undefined or ... is true`), so `true` is what an upstream caller who
  sends nothing already gets. Accepting `false` as "what this server does" would have blessed the
  divergent value and refused the vendor's own default. Omitting the kwarg still serves.
  **Known gap, stated rather than implied:** memra's multi-turn qwen prompts drop a prior
  assistant turn's reasoning (`messages[].reasoning` / `reasoning_content`) where that template
  would have replayed it, so multi-turn q38 prompts are off the vendor's bytes for assistant turns
  that carry reasoning. Implementing the replay is a prompt-bytes change on every multi-turn
  request and is follow-up work, not part of this lane.
- **Isolation:** non-tools traffic bypasses the tools renderer AND the emission parser
  entirely (legacy render path, byte-identical streams); tools traffic is generation-
  identical for the identical rendered prompt (raw-completions bijection gate). ONE
  carve-out (v0.109.1): a template carrying the qwen3.8 reasoning-effort ladder routes
  even unset non-tools chats through the tools-capable renderer, because only that
  renderer injects the vendor's `xhigh` default on unset — v0.109.0 kept the bypass and
  served a split surface (the same unset request rendered the default with a `tools`
  array riding along and the historical bare bytes without one, live-probed on q38-nj
  2026-08-23). Ladder-less templates keep the bypass byte-identically. `usage`
  now carries worker-truth `prompt_tokens` (rendered tools block included) +
  `completion_tokens` + `total_tokens` on stream and non-stream shapes, with the
  prompt-caching split (`prompt_tokens_details.cached_tokens`) — see "Prompt caching"
  below. Tools requests cache like any other: the prefix cache keys on the rendered
  prompt's token ids, so a repeated tools block is a cacheable prefix (no special-casing).

Receipts: `research/serve-tools-20260802/` (round-trip transcripts N=3 greedy on
Qwen3.6-35B + AgentWorld, streaming schema checker, malformed-policy transcript,
tok-check usage crosscheck, cross-binary c1 refs + c1-vs-c16) and
`research/integrate-cache-20260802/` (tools x cache intersection gate).

## Embeddings and rerank — the capture surfaces (lane/embed-serve, 2026-08-26)

Two prefill-only routes read the final prompt position of a causal LM instead of
decoding from it. No decode step runs (`max_new: 0`); admission, lanes, budgets,
receipts, rate limits and `[meter]` lines are byte-identical to `/v1/completions` —
one admitted worker request per input, billed as prompt tokens, each under its own
ledger id `<x-request-id>.<index>` (the response keeps the parent id; a ledger that keys
debits by id as a replay guard would otherwise bill N inputs as one).

- `POST /v1/embeddings` (OpenAI schema): `{model, input: string|string[], dimensions?}`.
  The vector is the last-token post-final-norm hidden state, L2-normalized; `dimensions`
  applies MRL truncation (keep-first-N, re-normalize) — the Qwen3-Embedding pooling
  convention. `encoding_format` other than `"float"` is refused. Arrays are capped at 32
  inputs per request.
- `POST /v1/rerank` (Cohere-shaped): `{model, query, documents[], top_n?, instruction?,
  return_documents?}`. Each document is judged with the Qwen3-Reranker prompt (system +
  `<Instruct>/<Query>/<Document>` + forced empty think block) and scored as P("yes")
  over the {"yes","no"} logit pair at the final position. Models whose "yes"/"no" are
  not single vocabulary tokens are refused honestly. Documents capped at 64 per request.

Worker semantics (`worker::CaptureSpec`): capture requests bypass EVERY cross-request
KV reuse tier (prefix cache, continuation pool, spec resume — a cache hit would skip
the prime the capture reads from), never take the spec path, and prime alone (no
batched prime, no prefix fanout). Prompts at or above the prime floor pool from the
prime call's hidden stack; shorter prompts walk the tokenwise path and pool via
`decode_step_h` on the final token — same numeric program, hidden returned
(`prime_cache` hard-asserts T >= PRIME_MIN_T; do not lower that floor for capture).

Deployment posture: these are the SUBORDINATE surfaces by design — serve them on
batch-class keys so they ride the harvest lane and shed under interactive load (the
SLO admission protecting decode p99 is the isolation mechanism). A co-loaded
embedding/reranker model is the intended shape: `MEMRA_MODELS` accepts the extra
`alias=path` entry, and a model with no trained NextN block now loads PLAIN beside a
spec'd chat model even when `MEMRA_FRSPEC_TRIM` is set globally (the trim request is
ignored for headless models instead of fataling the whole worker).

No new environment flag: capture is request-driven (`/v1/embeddings`, `/v1/rerank`
are the only producers), so there is no default to decide or roll back — absent those
calls the serving process is byte-identical to before this lane.

## Model id resolution — a stripped vendor prefix is tolerated (2026-08-13)

Marketplaces normalize model ids before calling upstream. Onlist lists `qwen/qwen3.6-35b-a3b` and
then probes for the bare `qwen3.6-35b-a3b`, which produced a correct-but-fatal
`unknown model "qwen3.6-35b-a3b"; loaded: ["qwen/qwen3.6-35b-a3b"]` and cost two real requests
before it was found. The listing side offers no upstream-id override, so the tolerance lives here.

`canonical_model_id` runs at both completion handlers BEFORE anything downstream, because model
metadata limits, capability gates, the cache namespace, ledger pricing, and the worker's roster all
key off this id and must agree on one spelling. Rules:

- an **exact** alias always wins, so nothing already working can change meaning;
- otherwise, if exactly **one** loaded alias's segment after the last `/` equals the request, that
  alias is used, and the response's `model` field reports the canonical id;
- **ambiguity is deliberately not resolved** — with both `a/m` and `b/m` loaded, a request for `m`
  stays unknown rather than routing to the wrong weights and billing under the wrong schedule;
- a request that already contains `/` is never suffix-matched, and an unknown id still returns
  `model_not_found`.

`/v1/models` continues to advertise canonical ids only. This is inbound request tolerance, not a
second public name for a model.

## OpenAI compatibility contract (serve-compat lane, 2026-08-03)

The five gap-scan listing-blockers (`research/gap-scan-20260802/REPORT.md`), fixed and
gated by the official `openai` Python SDK against a live server
(`research/serve-compat-20260802/`):

- **Envelope:** every OpenAI-shape completion and stream chunk carries `id`
  (`chatcmpl-…`/`cmpl-…`), `created`, and `system_fingerprint`
  (`memra-<version>-<content id>`, baked at build, see **The build fingerprint** below);
  the id echoes as the `x-request-id` response header. The first stream delta
  carries `role:"assistant"`. Error bodies are the OpenAI object —
  `{"error": {"message","type","param","code"}}` — and mid-stream worker errors arrive as
  a final `data:` error chunk + `[DONE]`, never a named SSE event. SSE keep-alive comments
  flow every 5s (long-prompt prefill streams nothing before first token; OpenRouter cancels
  silent streams).

  **Precondition — which surface you are talking to.** Everything in this section describes the
  **OpenAI-shape** surface, and the stream terminator + the mid-stream error shape are gated on
  `chat || openai_compat()` (main.rs:1966, 2007). `openai_compat()` is true when
  `MEMRA_COMPAT=openai`, or when `MEMRA_COMPAT` is unset **and `MEMRA_API_KEY` is set** — the pi
  setup. On a **native-default** server (no `MEMRA_COMPAT`, no `MEMRA_API_KEY`) a streaming
  `/v1/completions` does the opposite of the sentence above: it emits a named `event: error` and a
  named `event: done`, with **no `data: [DONE]`**. That is deliberate, not a bug — native clients
  are memra's own tools, which do parse named events, and the validation harnesses rely on it.
  `/v1/chat/completions` is always OpenAI-shape (`chat` is true regardless). The shipped unit sets
  `MEMRA_COMPAT=openai` (`deploy/systemd/memra-server.service:92`), so a deployed server matches
  this section — but if you are testing a bare `memra-server` and your SDK reads a silent hang,
  this is why.
- **Reasoning separation:** on think-open prompts, `<think>` text routes to
  `message.reasoning` / `delta.reasoning` (+ `reasoning_details`, the OpenRouter
  dialect); `content` is post-think only. `include_reasoning:false` (or
  `reasoning: {exclude: true}`) drops the separated text. Non-think models keep
  byte-identical no-parser streams. **Gemma-4 dialect** (lane/gemma4-serve-gaps,
  2026-08-07): `<|channel>thought\n…\n<channel|>` blocks route to `reasoning` the same
  way — tags, the channel label and the bracketing newlines are syntax — and the splitter
  runs on *every* gemma4 chat request (channels can open mid-stream even under the
  closed-channel default). Turn-end control tokens (`<turn|>`, `<end_of_turn>`,
  `<|im_end|>`) stop generation (`eog_ids()` union) and never reach the client as text.
- **`max_tokens` omitted** ⇒ context-bounded budget (session ctx − prompt, capped at the
  model's trained context) — the OpenAI default-when-omitted semantics, not a silent
  128-token truncation. Explicit `max_tokens`/`max_completion_tokens` honored exactly.
- **`temperature` omitted ⇒ 1.0; `seed` omitted ⇒ fresh per request** (dogfood F4,
  2026-08-04). Both were `#[serde(default)]`, which is `0.0` for `f32` and `0` for `u64` —
  and both of those zeros are *meaningful values*, not "unset": temperature 0 is greedy
  argmax and seed 0 is a valid fixed stream. An omitting client (the OpenAI SDK's
  documented leave-it-out path, and this repo's own agentic driver) therefore got
  deterministic decoding pinned twice over: same context in, same token out, identical
  tool-call cycles forever. Now `temperature` omitted is OpenAI's documented 1.0 and an
  omitted `seed` draws fresh entropy per request. **Explicit values are honored exactly,
  including `temperature: 0` (greedy) and `seed: 0`** — every determinism gate in `tools/`
  and `research/` sends both explicitly, so all of them keep their behavior. Supply a
  `seed` whenever you want reproducibility; omit it to get variation.
  Corollary worth knowing: an omitted-`temperature` request is *pure* temperature-1.0
  sampling (`top_p` 1.0, `top_k`/`min_p` disabled, penalties off), which is exactly the
  regime that keeps the in-graph sampled draft chain — so the OpenAI default lands on the
  fast sampled-spec path, not a slow fallback.
- **Disconnect abort:** a hung-up client's session retires at the next tick (all serve
  paths: batched, graph, spec, legacy) and is billed to the abort point (the `[abort]`
  log line records prompt/cached/generated); queued requests from dead clients never
  reach the GPU. The abort now propagates PROMPTLY in one previously silent case: the
  event forwarder between worker and response used to notice a gone consumer only when
  the next event arrived, so a request still in prefill (producing nothing yet) kept its
  channel open indefinitely and no disconnect or deadline could cancel it.
- **`timeout_ms` request deadline** (all four surfaces — see the billing contract below).

## The build fingerprint (lane/real-system-fingerprint, 2026-09-01)

`system_fingerprint` is **`memra-<crate version>-<content id>`**, for example
`memra-0.123.0-6371ca8a0af4` (every concrete value in this section is this lane's head,
2026-09-01; the id moves with the source, which is the point). It is baked at compile time by
`crates/memra-server/build.rs` and the algorithm lives in `crates/memra-server/src/build_id.rs`,
which the build script `include!`s and the crate compiles as a module: one implementation,
two callers, so the tests re-derive the value instead of pinning a copy of the algorithm.

The `<content id>` is 12 lowercase hex: an FNV-1a-128 digest, folded to its top 48 bits, over
the workspace's compiled inputs: root `Cargo.toml` and `Cargo.lock` plus every `*.rs`,
`*.toml`, `*.cu`, `*.cuh`, `*.h` under `crates/`, keyed by workspace-relative path and
sorted, so it does not depend on `read_dir` order or on where the checkout lives. Uniqueness
class, not crypto: it is a build identity, not a tamper seal.

**Why content and not the git sha.** The field used to be `concat!("memra-", <git rev-parse>)`
with any git failure swallowed into the literal `"unknown"`. Two consequences, both real:

- **It degraded silently.** darklanes' release container (`serving/build-artifact.sh`)
  compiles as root over a uid-1000 read-only mount; before its `safe.directory` line landed
  on 2026-08-30 git aborted with "detected dubious ownership" and the build baked
  `unknown`. Prod answered `system_fingerprint: memra-unknown` to every request for a whole
  deploy generation. A build with a meaningless identity looked exactly like a good one.
- **It did not survive a history rewrite.** A rewrite changes every commit SHA while the
  bytes of the tree stay put, so a sha baked into a shipped binary becomes a dangling
  reference, and a fingerprint quoted in a published claim, a research receipt, or a
  customer's own response stops naming anything. A content id is unchanged by a rewrite
  (receipt in the lane PR: `git commit --amend` over this tree moved the commit sha
  `2bfd89fb74d2` to `07b909892f43` while the id stayed `6371ca8a0af4`).

**Deliberately NOT in the value: a build timestamp.** Two builds of the same source must
produce the same fingerprint, because darklanes' `tools/check-claim-builds.mjs --live`
compares it for **equality** against the pin published beside every performance figure; a
per-rebuild component would churn every pin and make the gate meaningless. Build time is an
artifact-registry fact (the artifact filename and the file's mtime), not an identity.

**The git sha is still baked, as an extra field, never as the identity:**
`MEMRA_BUILD_SHA` → `memra_server::BUILD_GIT_SHA`. It is allowed to read `unknown` there,
because there it is honest.

**No flag.** The real fingerprint is unconditional: there is no env var that turns it off,
downgrades it, or overrides it. A knob here would be a knob for publishing an unverifiable
identity (new-flags law: the default is a written decision, and this one is "no flag").

**Reading it without a GPU.** `memra-server --version` (also `-V`) prints the identity and
exits before any engine, GPU, or model work, so a deployed artifact can be identified on any
box and in the release container that produced it:

```
$ memra-server --version
memra-server 0.123.0
system_fingerprint memra-0.123.0-6371ca8a0af4
build_id_src source-tree
git_sha <this build's short sha, or `unknown` if it could not read a repo>
```

The same path is in `serve_with`, so darklanes' deployment binary answers `--version` too.

**The degraded path is loud on both sides.** If the workspace source tree is unreadable (a
vendored or packaged crate), the id falls back to a digest of the package's own identity,
still shaped, still never `unknown`, and:

- `build.rs` emits a `cargo:warning=memra build identity DEGRADED: <reason>` at build time;
- every boot prints `[server] WARNING: build identity is DEGRADED: <reason>` naming the
  consequence (published performance pins cannot be verified against it).

Every boot also prints the identity line unconditionally, as its first line:
`[server] build: memra-0.123.0-6371ca8a0af4 (id: source-tree, git: <short sha>)`.

**Gates** (`crates/memra-server/src/lib.rs`, `mod build_identity_tests`): the baked value is
non-empty, is not `memra-unknown`, contains no `unknown`, matches
`memra-<version>-<12 lowercase hex>` and names this crate version; the shape checker rejects
what shipped to prod **and** the old `memra-<sha>` form; the baked id is re-derived from the
working tree in a separate process and must match, which is what proves it is a function of
the source rather than of the build environment; two scans of one tree agree; and the id is
asserted to differ from `BUILD_GIT_SHA`, so history can never become the identity by
accident. The two OpenAI envelope tests now assert the full shape: they used to assert
`starts_with("memra-")`, which `memra-unknown` passes, which is how the defect sat inside a
tested surface all the way to a customer.

**Consumers.** darklanes `tools/check-claim-builds.mjs` reads `system_fingerprint` from a
live endpoint and compares it to the `build` / `*Build` pins carried by every published
performance figure (`PRODUCT-TRUTH.md` §2.7). While the field read `memra-unknown` that gate
had nothing to verify; both prod boxes need a binary from this lane before the `--live` half
means anything, and the pins move to the new shape as part of that deploy.

## Request deadlines and the billing promise (`timeout_ms`)

Owner ruling, 2026-08-23: *"if the time pass and we didnt responed in time we fail and we
dont bill. if the non response is our fault we should not bill."*

`timeout_ms` is an integer number of milliseconds, accepted in **`1000`..=`90000`**, on
**every** surface with identical semantics: `/v1/chat/completions`, `/v1/completions`,
`/v1/messages`, `/v1/responses`. Out of range, or any non-integer type, is a **named 400**
naming `timeout_ms`, stating the range, and pointing at streaming for longer work — never
a silent clamp, because quietly shortening a caller's deadline is the accepted-and-ignored
class the standard-surface law bans.

**Default when absent: `90000`.** Every request has a deadline whether or not the client
knows the parameter exists: *we answer inside 90 s or you don't pay.*

**AMENDED 2026-08-26** (owner: a 30k-token non-streaming request timed out —
lane/deadline-partial-20260826). That promise covered a request we fail to answer, and it
still does. What changed is that a non-streaming deadline no longer THROWS AWAY the tokens
it had already generated in order to produce that error:

- An infeasible non-streaming request is refused at admission with a named
  400 (`nonstream_deadline_infeasible`) naming the `max_tokens` that fits — instantly,
  before any GPU work, instead of after the full deadline.
- A deadline that lands mid-generation on the OpenAI-dialect surfaces DELIVERS what was
  produced and BILLS it, under the census outcome `deadline_partial` — the caller received
  those tokens. A deadline that lands with zero tokens still answers 408 `deadline_exceeded`
  and bills nothing.

Full behaviour, including which surfaces deliver partials and why the other two do not, is
in "Long non-streaming requests: what the deadline does now" below.

The 90 s maximum is a **platform fact, not a preference**: the fronting proxy fails a
non-streaming response whose time-to-headers reaches ~100 s (Cloudflare `524`), so a
larger promise would be broken above this server no matter what it did. Work that
legitimately needs longer belongs on a **stream**, where the deadline bounds only the time
to first token and the response may then run as long as it needs.

| | non-streaming | streaming (`stream: true`) |
|---|---|---|
| what the deadline bounds | the COMPLETE response | **time to first token only** |
| infeasible before it starts | refused at admission: `400` `nonstream_deadline_infeasible` | never gated |
| miss with SOME tokens produced | OpenAI-dialect surfaces: **`200` partial, BILLED** (`deadline_partial`, `finish_reason: "error"`); `/v1/messages` + `/v1/responses`: `408`, discarded | cannot happen — the parameter is spent at the first token |
| miss with ZERO tokens produced | `408`, generation cancelled, **not billed** | `408` **pre-header**, generation cancelled, **not billed** |
| after the first token | — | the parameter is **spent**; the stream runs to completion |

A missed deadline that delivered NOTHING is `408` with the standard error object
(`type: "timeout"`, `code: "deadline_exceeded"`), a message naming the effective deadline,
and **no** `Retry-After` — a miss says nothing about when a retry would fit, and inventing a
window
would be a promise this server cannot keep. `408` stays retryable (no
`x-should-retry: false`; SDKs retry it by default). The streaming 408 is deliberately
**pre-header**: the response is held until the first token, because once the first byte of
a `200` is written there is no status left for a router to act on and no honest way to say
"you don't pay". The admission wait counts against the deadline, so a request that can no
longer be answered in time is cancelled rather than served late.

**The billing promise, stated once:** *a request we fail on time grounds costs nothing; a
response you abandon is billed for what was generated.* A client that walks away
mid-stream after receiving tokens is the existing `abandoned` path — user fault, partial
billed. Everything that is OUR fault bills zero (the census below).

### Fault attribution: which outcomes may bill

Every request ends with exactly one ledger `outcome`, and only the outcomes in
`BILLABLE_OUTCOMES` may carry a debit — **three** of them, marked below. (Stated as the
constant rather than a number: the count has now gone stale twice, once in this file and
once in four places in `ledger.rs`, each time the billable set grew.)

| outcome | when | debit |
|---|---|---|
| `completed` | the response was delivered in full | exact |
| `abandoned` | the CLIENT walked away mid-generation | partial (what was generated) |
| `deadline_partial` | `timeout_ms` cut generation and we DELIVERED what was produced | partial (what was delivered) |
| `rejected` | refused or failed before/during generation (any 4xx/5xx) | **zero** |
| `deadline_exceeded` | `timeout_ms` elapsed before we responded | **zero** |
| `shed_deadline` | admission shed: estimated wait exceeded the deadline | **zero** |
| `shed_queue` | admission shed: absolute queue bound reached | **zero** |
| `shed_queue_wait` | admission shed: estimated wait exceeded the deployment's opt-in queue-wait ceiling (`MEMRA_QUEUE_WAIT_CEILING_S`) | **zero** |
| `drain_killed` | WE killed it at the drain deadline (SIGTERM) | **zero** |
| `crashed` | the handler panicked while holding the receipt | **zero** |

The invariant is enforced at the ledger's single terminal write point, not at each call
site: a row whose outcome may not bill has its `usage`, `unit_prices_usd`, `cost_usd` and
`budget` fields stripped and its reservation refunded in full, so neither the budget
journal nor any revenue report that sums `cost_usd` can count a request the customer was
never charged for. One test per outcome asserts the debit.

Three of those zero-debit outcomes are new because the paths existed and billed the wrong
way: a worker that died mid-stream, a drain-deadline kill, and a handler panic all reached
the receipt's `Drop` and were recorded as `abandoned` — i.e. **the customer paid for our
crash**, under an error code (`client_disconnected_or_handler_dropped`) that could not tell
the two apart. A mid-stream worker death is also now LOUD on every surface: it emits the
same error object the blocking path returns instead of a stream that simply stops.

### Backpressure: deadline-aware admission

The interactive lane queues beyond `MEMRA_MAX_SESSIONS` and never used to shed, so a
saturated box just got slower — nothing a client or a router could act on. At submission
time (never after: **an admitted request is never shed**) two gates always apply and a third is opt-in, all only
while the lane is at capacity:

1. **Deadline test** — if the estimated queue wait exceeds this request's remaining
   deadline, `429` + `Retry-After` = the estimate, outcome `shed_deadline`, not billed.
   The estimate reuses the `X-RateLimit-Reset` machinery (mean tokens/request × p50 step
   latency, scaled by the waves of queued work ahead). **Honestly coarse — a hint, not a
   promise**, and the shed message says so.
2. **Absolute bound** — `MEMRA_MAX_QUEUE_DEPTH` (default 4 × the session cap), outcome
   `shed_queue`, same headers. The load-shape-independent backstop.
3. **Queue-wait ceiling** (opt-in: `MEMRA_QUEUE_WAIT_CEILING_S` = N > 0, default 0 = off) —
   if that same wait estimate exceeds N seconds, `429` with the same headers, outcome
   `shed_queue_wait`, not billed. Independent of the caller's deadline: gate 1 never fires
   for a patient caller, which is how a burst past the session cap queued 133-137 s in
   silence on prod (2026-09-01). Judged after gates 1 and 2, so those answer first and
   unchanged. Flag row: docs/FLAGS.md.

All three carry the `X-RateLimit` trio, and all three are what the router's circuit breaker and
load spill consume. **Circuit breaking stays at the router**; the engine's contribution is
honest, prompt 429s. Dark lanes (`judge`/`harvest`) are untouched here — they already shed
at cap inside the worker.
- **Parameter breadth + honesty:** `frequency_penalty`/`presence_penalty`/
  `repetition_penalty` plumb to the sampler (last 8,192 tokens on every host, device, and
  speculative path; greedy+penalized
  keeps the host-sampled path). `response_format` `json_object`/`json_schema` is REAL
  constrained decoding (see the section below). Semantic params we can't honor 400 with
  the param named (`logit_bias`, `logprobs`/`top_logprobs`, `n != 1`, `best_of != 1`,
  unknown `response_format` types); cosmetic fields (`user`, `stream_options`) are
  accepted and ignored. Streams exclude stop-sequence text exactly like non-stream
  responses (holdback buffer).

## Gateway listing surface

OpenRouter's current Provider Monitor schema is version **2.4**, but it is not the old
flat/catalog shape: new integrations declare typed `input_modalities` and
`output_modalities`, with pricing and capacity nested on the modality they belong to.
The older flat provider document remains supported only for existing integrations.

memra keeps four views separate because provider schemas reject unknown or differently shaped
fields:

- **`GET /models`** keeps the historical OpenAI-style body byte-for-byte:
  `{"object":"list","data":[{"id":"<alias>","object":"model"}]}`. Existing pill/Hermes
  consumers stay on this default.
- **`GET /models?schema=openrouter`** is the OpenRouter Provider Monitor 2.4 document
  for a new provider integration. Use this full URL for the OpenRouter application.
- **`GET /models?schema=openmodels`** is the OpenModels provider feed. It returns the
  required `{data:[...]}` wrapper with simple modality arrays, per-token pricing,
  documented `supported_features`, and readiness/free/discount fields. Use this URL for an
  OpenModels provider application; do not substitute the OpenAI or OpenRouter view.
- **`GET /v1/models`** keeps the existing catalog-style enrichment
  (`context_length`, `architecture`, `pricing`, `top_provider`) for current clients.
  It is not the strict Provider Monitor document.

The Provider Monitor view derives what the process knows:

- `id` and `name` are the exact `MEMRA_MODELS` alias.
- Text `max_context_length` and `tokenizer` come from the loaded model — with the
  context claim CAPPED by the deployment's operational envelope
  (`max_prompt_length + max_output_length`) whenever the metadata pins both
  (2026-08-30, lane/glm5-docs-sweep): the checkpoint's trained `context_length` is a
  training fact, not a serving claim, and every catalog view (`/v1/models`,
  openrouter, openmodels) publishes `min(trained, envelope)`. The receipt that forced
  it: glm5_next declares 1,048,576 trained while the 3-card resident shape OOMs the 1M
  prime (`research/glm5-prefix-latent-20260830/box-window/WINDOW-STATUS.md`).
- Streaming and supported generation parameters come from the real HTTP surface;
  `tools` and `reasoning` appear only when the loaded template exposes them.
  A model declaring a non-chat `surface` (below) overrides all of that: it is not a
  completion surface, so it advertises no streaming and no generation parameters at
  all, whatever its template exposes.

Everything else is operator-declared in a TOML file named by
`MEMRA_MODEL_METADATA`. The same declarations supply both strict provider views: OpenModels maps
`cached_prompt` to `pricing.input_cache_read`. Its view fails with HTTP 400 unless every model has
a loaded context plus declared `created`, `max_output_length`, prompt/completion/cached-input
prices, `is_ready`, `is_free`, and `discount_to_user`; this prevents a superficially successful
but invalid provider feed. The file is optional for local serving. If configured, it is parsed
before the GPU worker starts; unknown fields, invalid price strings, invalid
quantization names, zero limits, or aliases absent from `MEMRA_MODELS` are fatal.

```toml
# /etc/memra/models.toml
[models."provider/model-id"]
description = "Qwen3.6 27B served by memra."
quantization = "nvfp4"
max_prompt_length = 262144
max_output_length = 262144
default_output_length = 8192 # runtime default only; not the advertised maximum
is_ready = true
# Set the real deployment location before submitting the application:
# datacenters = [{ country_code = "US", region = "actual-region" }]

[models."provider/model-id".pricing]
# Per-token USD strings, not per-million-token numbers.
prompt = "0.000000234"         # $0.234 / 1M input tokens
cached_prompt = "0.0000001638" # 70% of input; 30% cache discount
completion = "0.000001872"    # $1.872 / 1M output tokens

[models."provider/model-id".capacity]
# Optional honest declarations; omit values that are not measured.
prompt_tpm = 1000000
completion_tpm = 500000
request_rpm = 1000
concurrency = 16
```

```bash
MEMRA_MODELS="provider/model-id=/path/to/model.gguf" \
MEMRA_MODEL_METADATA=/etc/memra/models.toml \
memra-server

curl 'http://127.0.0.1:8080/models?schema=openrouter'
curl 'http://127.0.0.1:8080/models?schema=openmodels'
```

Supported metadata fields are `hugging_face_id`, `created`, `quantization`,
`description`, `surface`, `max_prompt_length`, `max_output_length`, `default_output_length`,
`default_reasoning_effort`, the `default_*` sampling keys below, `is_ready`, `is_free`,
`discount_to_user`, `openrouter_slug`, `datacenters`, `zdr`, and `hipaa`.
`surface` (`chat` | `embedding` | `rerank`, default `chat`, validated at boot) declares
which API surface the model serves, and is what every catalog view — `/v1/models`,
`?schema=openrouter`, `?schema=openmodels` — reads to decide the model `type`, its
`endpoints`, its output modality and whether the chat affordances (streaming, tools,
reasoning, structured output, `max_tokens` and any output ceiling) appear at all. It is
declared rather than inferred because embedding/rerank capability is only decided at
runtime, when the prime path either yields hidden state or does not, which is long after
the catalog row is built. Omit it and the row is byte-identical to a chat model's:

```toml
[models."qwen/qwen3-embedding-8b"]
surface = "embedding"   # serves /v1/embeddings, not /v1/chat/completions
```
`default_reasoning_effort` (`none|minimal|low|medium|high`, validated at boot) resolves a
request that expressed no reasoning choice on any surface as if the client had sent that
value — explicit client `reasoning_effort`/`reasoning` always wins, and models without
the key keep their template's own default.

### Per-model vendor sampling defaults

A request that omits a sampling field gets the MODEL VENDOR's own published recommendation for
that model, not greedy and not a house guess. Seven optional boot-validated keys:

| key | range | meaning |
|---|---|---|
| `default_temperature` | `(0, 2]` | **zero is refused** — see below |
| `default_top_p` | `(0, 1]` | 1.0 = disabled |
| `default_top_k` | `>= 0` | 0 = disabled (keep all) |
| `default_min_p` | `[0, 1)` | 0.0 = disabled |
| `default_presence_penalty` | `[-2, 2]` | 0.0 = off |
| `default_frequency_penalty` | `[-2, 2]` | 0.0 = off |
| `default_repetition_penalty` | `> 0` | 1.0 = off |

Rules:

- **Each key substitutes for exactly one OMITTED field.** An explicit client value always wins,
  on every surface. **An explicit `temperature: 0` still produces true greedy** — greedy is a
  caller decision and stays exactly reachable; it just stops being what an omitting client gets.
- **All four surfaces resolve identically** (`/v1/completions`, `/v1/chat/completions`,
  `/v1/messages`, `/v1/responses`) because they share one resolver, `resolve_sampler_config`.
  This is enforced by `vendor_sampling_defaults_are_identical_on_every_surface`, not by three
  copies of the logic agreeing.
- **Undeclared means untouched.** A parameter with no key keeps the API-standard default (temp
  1.0, top_p 1.0, top_k 0, min_p 0.0, penalties off). Where a vendor recommends nothing, declare
  nothing — do not invent a value.
- **`default_temperature = 0.0` is fatal at boot.** A zero *default* is greedy-by-default wearing
  a config hat: it would silently apply deployment-wide to every omitting client, which is the
  hazard these keys exist to remove (owner ruling 2026-08-19: *"we don't have to serve greedy, we
  measure greedy but we serve what the user chooses"*, *"we default to what are the
  recommendations"*, *"greedy can create issues"*).
- **Cite the vendor in the TOML comment.** The number belongs to the model's vendor, not to us;
  the citation is what stops a later cleanup pass from "simplifying" a deliberate value.
- **Models without any key are byte-identical to before these keys existed.** Architectures that
  publish their own API defaults still supply them from the engine (step35 = StepFun's 0.5/0.9);
  a metadata key outranks that arch default, and a partially-declared block falls through field
  by field rather than wholesale.

### Per-mode arms: `non_thinking_sampling`

Some vendors publish TWO sampling recommendations — one for thinking mode, one for non-thinking
(qwen3.8: thinking `1.0/0.95/20`, non-thinking `0.7/0.80/20` + `presence_penalty 1.5`). The flat
`default_*` keys above are the PRIMARY arm; an optional nested table declares the second:

```toml
[models."qwen/qwen3.8-27b".non_thinking_sampling]
# Same seven keys, unprefixed (the table name says which arm), same ranges, same
# boot validation, same zero-temperature refusal. An EMPTY table is refused.
temperature = 0.7
top_p = 0.8
top_k = 20
presence_penalty = 1.5
```

Rules, on top of everything above (which all still holds — an explicit client value is NEVER
overridden by either arm):

- **The request's RESOLVED thinking mode picks the arm** (`ModelSamplingDefaults::for_mode`).
  Any spelling that resolves to thinking-off — `reasoning_effort: "none"|"minimal"`,
  `enable_thinking: false`, `chat_template_kwargs.enable_thinking: false`,
  `reasoning: {"enabled": false}`, `include_reasoning: false`, Anthropic
  `thinking.type: "disabled"`, an operator `default_reasoning_effort = "none"` resolving an
  unset request, or `response_format` forcing the think switch off (switch-carrying
  templates only; post-think constrained requests on think-forced templates keep thinking
  ON and take the primary arm) takes the non-thinking arm for the sampling fields the
  client left unset. Every other mode (thinking on, or the template's own default) takes
  the primary arm.
- **A model without the table is byte-identical to before it existed**: one arm, every mode.
- **Arms never blend.** A field the vendor left out of the non-thinking arm falls to the
  API-standard default, not to the thinking arm's value — the two arms are separate vendor
  programs, and blending them would serve numbers no vendor published.
- **A deployment whose UNSET case should be non-thinking says so with
  `default_reasoning_effort = "none"`** — that resolves the unset request to thinking-off
  upstream, and the arm selection follows. `ThinkMode::Default` alone keeps the primary arm.
- `/v1/completions` is a raw-prompt surface with no thinking control: always the primary arm.

```toml
[models."google/gemma-4-31b-it"]
# VENDOR SOURCE: google/gemma-4-31B-it model card, "Best Practices" -> "1. Sampling
# Parameters" ("Use the following standardized sampling configuration across all use
# cases"), corroborated by the repo's own generation_config.json and by
# ai.google.dev/gemma/docs/core/model_card_4. Do not "clean up" these numbers.
default_temperature = 1.0
default_top_p = 0.95
default_top_k = 64
# Google recommends nothing for min_p or the penalties: left at the API standard, not invented.
```

**DEPLOY ORDER IS MANDATORY: binary first, then config.** `[models.*]` blocks are parsed with
`deny_unknown_fields`, so a binary that predates these keys **FAILS BOOT** on a config carrying
them — the same trap `default_reasoning_effort` created in v0.92.0. Roll the binary everywhere
first, confirm it is live, then render the config.

**PERFORMANCE NOTE.** Both currently served models' vendor recommendations include truncation
filters (`top_p` < 1 and a positive `top_k`). The in-graph sampled draft chain samples from the
raw softmax and can hold no per-row filter stats, so `spec.rs` engages the captured sampled graph
only in the pure-temperature regime and otherwise uses the eager draft chain
(`memra-sampling::Sampler::is_spec_sampling`, `spec.rs`'s `pure_temp`). Declaring `default_top_p`
/ `default_top_k` therefore moves the DEFAULT request shape onto the eager draft chain. Exactness
is unaffected — filters apply symmetrically to draft `q` and target `p` under the rejection
verify, so these requests stay spec-eligible and distribution-exact — but the throughput profile
of the default path changes, and a client can opt back into the fast regime with an explicit
`top_p: 1.0` + `top_k: 0`. `vendor_defaults_leave_the_pure_temp_sampled_spec_regime` pins this so
the flip stays measured rather than discovered.
`pricing` accepts `prompt`, `cached_prompt`, `cache_write`, `completion`,
`internal_reasoning`, and `request`; `capacity` accepts `prompt_tpm`,
`cached_prompt_tpm`, `completion_tpm`, `request_rpm`, and `concurrency`.

Future offers may use the same shape under `[planned_models."provider/model-id"]`. Planned entries
are parsed and validated at startup but never emitted or served. Promotion means moving the entry
under `[models]` and loading the identical alias through `MEMRA_MODELS`; no Rust change is needed.
Prices and capacities are omitted when undeclared. memra never turns an absent price
into `"0"`; use an explicit zero only for a genuinely free SKU.

The remaining gateway controls are battery-gated (`research/serve-tail-20260804/`):

- **Rate-limit headers:** `X-RateLimit-Limit` / `-Remaining` / `-Reset` (emitted
  lowercase on the wire, as HTTP/2 requires; capitalized here by convention — a client
  parsing headers into a case-sensitive dict must key on `x-ratelimit-*`) on both
  completion routes with concurrency-slot semantics — a per-lane atomic gauge whose
  RAII slot rides the SSE stream to completion, so `Remaining` is truthful for the
  whole life of a stream. Sheds carry `429 + Retry-After`; `MEMRA_RL_RESET_S` is the
  no-signal fallback for `Reset` (with traffic, Reset = mean tokens/request x p50 step
  latency). **`Remaining: 0` no longer means "you will wait" unconditionally** — since
  lane/deadline-billing the interactive lane sheds a saturated admission when the
  estimated wait cannot fit the request's `timeout_ms`, or when the absolute queue bound
  (`MEMRA_MAX_QUEUE_DEPTH`) is reached; see "Request deadlines" above.
- **Graceful drain:** SIGTERM flips `/health` to `status:"draining"` (still **200** — see
  Health below) and `/readyz` to **503**, new completion requests get `503 + Retry-After`,
  in-flight requests — streams included — run to `[DONE]` within the `MEMRA_DRAIN_S`
  deadline (default 30s), then the process exits 0. Live receipt: a 1024-token stream
  completed mid-drain.

## Health, readiness, and fault handling (serve-hardening lane, 2026-08-06)

Receipts: `research/serve-hardening-20260806/`. Example unit:
`deploy/systemd/memra-server.service`.

**The full route table**, since the sections above only introduce routes as they become
relevant (bind address `MEMRA_ADDR`, default `127.0.0.1:8080`):

| route | notes |
|---|---|
| `GET /health`, `GET /livez` | the same handler — inference liveness (below) |
| `GET /readyz` | routability (below) |
| `GET /v1/models` | the existing catalog-style enriched listing |
| `GET /models` | the byte-compatible OpenAI-style listing used by existing clients and smoke gates |
| `GET /models?schema=openrouter` | strict OpenRouter Provider Monitor schema 2.4; operator metadata comes from `MEMRA_MODEL_METADATA` |
| `GET /models?schema=openmodels` | OpenModels provider feed; pricing/readiness and other operator declarations come from `MEMRA_MODEL_METADATA` |
| `POST /v1/completions` | raw-prompt completions. **Streaming shape depends on `MEMRA_COMPAT`** — see the compatibility precondition above |
| `POST /v1/chat/completions` | always OpenAI-shape |
| `GET /metrics` | keyring completion credentials receive only their own tenant rows; operator scope adds process-wide counters, global prefix/LCP aggregates, current capacity/VRAM, background-job state, cumulative `spec`, and rolling `spec_tau` / `spec_accept_by_position` |
| `GET /yield/metrics` | the process-wide dark-lane yield view; a keyring deployment requires the dedicated operator metrics token |

The default loopback bind is intentionally convenient for no-key local development. A
non-loopback `MEMRA_ADDR` without `MEMRA_API_KEY` or `MEMRA_API_KEYS` is refused before
model initialization. `MEMRA_ALLOW_OPEN_BIND=1` is the explicit development override;
it leaves completion routes open but does **not** make either metrics route public. Once
any API-key source is configured—or the bind is non-loopback—metrics require a bearer.
Multi-tenant keyring deployments default to a tenant-only metrics view; exposing any
process-wide serving surface is an explicit operator choice:

- With `MEMRA_METRICS_TOKEN` configured, that token is **exclusive** for `/metrics` and
  `/yield/metrics`; completion API keys receive 403. The scrape principal sees all tenant
  rows plus operator-only gauges and aggregate blocks, so use this mode only for an operator
  trusted with fleet-wide usage.
- Without `MEMRA_METRICS_TOKEN`, a keyring completion credential may read `/metrics`, but the
  body contains only its own `tenants` row and `adsd_suspect_total` entry. Process-wide request,
  token, latency, admission, cache/pool, prefix, capacity/VRAM, background-job, and aggregate
  `spec` fields are absent. `/yield/metrics` is wholly process-wide and returns 403. Each caller
  still receives its own `tenants.*.cache_hit_token_ratio`.
- The legacy `MEMRA_API_KEY` remains one completion tenancy domain: without a keyring or metrics
  token it retains the existing cumulative counters, yield view, and all of that domain's raw
  `cache_salt` rows, but it is not an operator principal and still cannot see the separately
  operator-gated prefix/capacity/spec surfaces.

The operator-only fields can expose cross-tenant volume, latency, prefix shape, or the
memory/scheduler state used to time a [Fill-and-Squeeze attack](https://arxiv.org/abs/2602.07878).

No-key loopback development remains unauthenticated and unchanged. A configured metrics
token never authorizes completion routes.

**`/health` == `/livez` — inference liveness, not process liveness.** The GPU worker is
ONE `std::thread` owning the CUDA context. `/health` used to answer `{"status":"ok"}` off
the axum task, so a worker panic or a wedged card left a permanently green health check in
front of a box answering nothing. It now derives from a heartbeat the scheduler loop stamps
every iteration, plus a phase:

| worker phase | `/health` | why |
|---|---|---|
| `loading` | 503 | weights are not resident; the process answers nothing yet. On a FIRST load the port is not bound yet (bind follows the load), so a probe sees connection-refused — the same verdict for k8s and `serve-fleet.sh`. This state is reached over HTTP during a **respawn**, which is the case that matters |
| `idle` | 200 at any beat age | the worker blocks in `rx.recv()` — an idle server legitimately stamps nothing for hours, and a naive age check would call every quiet server dead |
| `busy` | 200 while FORWARD PROGRESS advances, 503 past `MEMRA_HEALTH_STALL_S` (120s) | work in flight must make progress. Two signals attest it and the verdict takes the fresher: the scheduler heartbeat (one loop pass) and the engine's prime odometer (one completed prime chunk, stamped where the chunk's logits are already host-side). A long monolithic prefill therefore reads BUSY-and-healthy while it is genuinely progressing; a wedged worker advances neither signal and still 503s within the bound (memra#50, 2026-09-03; `MEMRA_HEALTH_PROGRESS=0` restores beat-age-only semantics). What it does NOT catch: a hang inside ONE chunk (same detection time as before), a livelock that keeps completing chunks without finishing requests, and per-session starvation (the odometer is process-global) |
| `dead` / fault latched | 503 immediately | worker panic or fatal Xid — a latch, not a timeout, so the flip is instant |

The response body is `{status, models, worker:{phase, beat_age_ms, tick_max_ms,
stall_threshold_ms, forward_progress_age_ms, prime_progress:{rows, chunks, age_ms}|null,
generation, xid_warnings}}`, plus a top-level `detail` on a red (which is
where a quoted panic payload lands). `status` is `ok` / `draining` / `unhealthy` on
`/health`-`/livez` and `ready` / `not_ready` on `/readyz`. So a red is self-explaining and
`tick_max_ms` — the longest scheduler iteration this process actually observed — is the live
receipt for revisiting the threshold.

Every red probe 503 follows the same retry-header contract as request-path overloads.
Worker-related `/health` and `/readyz` failures use the worker supervisor's 2-second respawn
backoff (`Retry-After: 2` + `retry-after-ms: 2000`). A draining `/readyz` uses
`MEMRA_DRAIN_S`, clamped to the SDK-honored 1..=60 second window, with the matching millisecond
twin. Retryable probe responses never carry the contradictory `x-should-retry: false`.

**`/readyz` — should traffic be routed here?** Ready = model loaded AND worker alive AND
not draining. Unready is NOT a restart request: draining and loading are healthy states
that simply must not be routed to, which is exactly why liveness and readiness are
separate endpoints (k8s deprecated `/healthz` at v1.16 for this split). Queue pressure
deliberately does not flip readiness — the interactive lane queues FIFO and never sheds,
so a deep queue is work in progress; capacity backpressure belongs on the request path.
`tools/serve-proxy.py` probes `/readyz` for rotation; `tools/serve-fleet.sh` probes
`/health` for its restart decision. vLLM has no readiness endpoint (503 only on
`EngineDeadError`) and TGI a single `/health`.

**Worker panic → supervised.** The worker thread runs inside `catch_unwind`: a panic marks
health dead with the quoted panic payload, then ONE respawn is attempted after a **`2 x attempt`
second** backoff — 2 s at the default max of 1 (`MEMRA_WORKER_RESPAWN`; the sleep exists so a
panic from a transient device condition gives the driver time to settle instead of re-hitting it
immediately) — and failing that the process exits **70** so the supervisor restarts it whole.
**Two distinct paths reach exit 70**, and an operator reading `systemctl status` should be able to
tell them apart: the respawn budget running out (`STATUS=worker unrecoverable; exiting`), and a
respawn whose **weight reload itself failed** (`STATUS=respawn load failed; exiting`) — the second
is not a panic, and it exits rather than looping because a load failure will not fix itself.
Exit 70 is sysexits' `EX_SOFTWARE`, chosen so it reads distinctly from the startup FATAL paths,
which exit 1 ("the engine died" vs "bad config"). One attempt, deliberately — CUDA errors are sticky per process, so a
respawn loop against a poisoned context produces a box that looks alive and serves nothing.
Proved on a real CUDA worker, not only in tests (`MEMRA_PANIC_AFTER` fault injection,
`research/serve-hardening-20260806/logs/worker-death.txt`): panic → 503 on all three routes
with the quoted payload in `detail` within ~200 ms → weights reloaded → `generation` 0 → 1 →
the respawned worker served a real completion; with `MEMRA_WORKER_RESPAWN=0` the process
exited 70 and the port went refused. A request that arrived during the dead window was
**served by the respawn** — the supervisor owns the command channel across restarts, so
queued work survives a worker death.

**GPU faults (`MEMRA_GPU_WATCH`).** A watcher thread tails Xid lines (`/dev/kmsg`, falling
back to `journalctl -k -f`) and latches unhealthy on the fatal classes
(48/64/79/94/95/119/120), counting the rest as warnings. It also probes `nvidia-smi` for
uncorrectable ECC and row-remap failures every `MEMRA_GPU_WATCH_S` seconds (default 60 — the
audit's published detection commitment is "checks every 60 s", so treat it as a stated fact about
the instrumentation rather than a free knob). The design constraint: Blackwell's worst wedge
(Xid 119/120, GSP RPC timeout) emits nothing to the process **and hangs the query tools**,
so the probe runs as a killed-on-deadline child and its own timeout
(`MEMRA_GPU_PROBE_TIMEOUT_S`) is the alarm. Health reads only atomics, so a hung
`nvidia-smi` can never block a health answer. A GPU fault survives a worker respawn: a new
thread on a wedged card is not recovery.

**The supervision contract (`deploy/systemd/memra-server.service`) has three couplings you can
break silently.** The unit is an example to copy, but these are not stylistic choices — each is
sized against a server-side default, and changing one side alone produces a unit that looks
correct and misbehaves only during a failure:

| directive | value | the coupling |
|---|---|---|
| `WatchdogSec` | 180 | MUST exceed `MEMRA_HEALTH_STALL_S` (default 120). The heartbeat that feeds `/health` also feeds systemd, so a watchdog under the legitimate-stall bound restarts a *healthy* server mid-prefill. Raise both together if you raise `MEMRA_MAX_SESSIONS` or the context |
| `TimeoutStopSec` | 60 | MUST exceed `MEMRA_DRAIN_S` (default 30), or systemd SIGKILLs a drain that is finishing streams correctly. The server also sends `EXTEND_TIMEOUT_USEC`; the static floor covers a build that does not |
| `TimeoutStartSec` | 600 | MUST exceed the slowest cold load (~120 s measured for a 27B NVFP4 from page cache; cold NVMe on a large bank is slower). Startup silence is a load, not a hang |
| `StartLimitIntervalSec` / `StartLimitBurst` | 3600 / 4 | systemd's defaults (10 s / 5) are sized for millisecond daemons and **cannot trip at all** here — 5 starts do not fit in 10 s when each start takes ~120 s, so a crash loop restarts forever instead of failing the unit for a human. 4 starts per hour ≈ "if it cannot survive four full loads, page someone" |
| `RestartSec` / `RestartSteps` / `RestartMaxDelaySec` | 10 / 4 / 160 | a card that just threw an Xid needs the driver to settle; a tight loop makes recovery less likely. The ramp needs systemd ≥ 254 — on older systemd delete the last two lines and keep the flat 10 s |
| `OOMPolicy` | `kill` | the default `stop` reaps only the offending process, and the kernel OOM killer can take out ONE thread — classically the worker — leaving a process that accepts connections and can never serve them, which is the exact invisible death this lane removes. **Host memory only**: CUDA OOM is the 503 above, never a process kill |

Two more worth knowing before you deploy. `Type=notify` + `NotifyAccess=main` means `READY=1`
fires after the models load **and** the socket binds — `systemctl start` returning is a real
readiness signal, which is why `TimeoutStartSec` must be generous. And Xid visibility can be
silently absent: `kernel.dmesg_restrict=1` makes `/dev/kmsg` root-only, so an unprivileged unit
sees Xids only through `journalctl`; grant `AmbientCapabilities=CAP_SYSLOG` +
`CapabilityBoundingSet=CAP_SYSLOG` or accept the fallback to the probe-hang and ECC/remap
detectors, which need no kernel log. The watcher logs which source it got, so this is never a
silent downgrade. The unit is deliberately **not** `ProtectSystem=strict` — model paths,
`/dev/nvidia*`, and the CUDA cache need real filesystem access, and a wrong sandbox fails at
load time looking like a model bug.

**Error taxonomy.** Every engine failure used to be `400 invalid_request_error` — which no
OpenAI SDK retries, and which a router cannot distinguish from a malformed request. The
class now comes from the producer:

| condition | status | `type` | `code` | retry headers |
|---|---|---|---|---|
| malformed field, bad template, bad `response_format` | 400 | `invalid_request_error` | — | `x-should-retry: false` |
| prompt ≥ context cap | 400 | `invalid_request_error` | `context_length_exceeded` | `x-should-retry: false` |
| unknown model id | 400 | `invalid_request_error` | `model_not_found` | `x-should-retry: false` |
| dark-lane QoS shed (`x-lane` judge/harvest over budget) | 429 | `rate_limit_error` | `rate_limit_exceeded` | `Retry-After: 2` + `retry-after-ms: 2000` |
| `timeout_ms` out of range or wrong type | 400 | `invalid_request_error` | — | `x-should-retry: false` |
| `timeout_ms` deadline missed with ZERO tokens produced (or TTFT when streaming) | 408 | `timeout` | `deadline_exceeded` | none — a miss says nothing about when a retry fits; 408 is retryable by default |
| `timeout_ms` deadline missed mid-generation, OpenAI-dialect surfaces | **200** | — | `deadline_exceeded` in the response's `error` object, with `finish_reason: "error"` | n/a — the partial is delivered and billed (`deadline_partial`) |
| non-streaming request that cannot fit its deadline (refused at admission) | 400 | `invalid_request_error` | `nonstream_deadline_infeasible` | `x-should-retry: false` — retrying unchanged cannot fit either; the message names the `max_tokens` that would |
| interactive shed: estimated queue wait exceeds the request's deadline | 429 | `rate_limit_error` | `shed_deadline` | `Retry-After` = the wait estimate + matching `retry-after-ms` |
| interactive shed: absolute queue bound (`MEMRA_MAX_QUEUE_DEPTH`) reached | 429 | `rate_limit_error` | `shed_queue` | `Retry-After` = the wait estimate + matching `retry-after-ms` |
| interactive shed: estimated queue wait exceeds the deployment's opt-in ceiling (`MEMRA_QUEUE_WAIT_CEILING_S`, default off) | 429 | `rate_limit_error` | `shed_queue_wait` | `Retry-After` = the wait estimate + matching `retry-after-ms` |
| out of VRAM / step-OOM past its park budget / worker restarting | 503 | `server_error` | `overloaded` | `Retry-After: 5` + `retry-after-ms: 5000` |
| step, prefill, graph or constraint fault | 500 | `server_error` | `engine_error` | none (not time-bounded) |
| new request arriving during a drain | 503 | `server_error` | `draining` | `Retry-After: MEMRA_DRAIN_S` (≤60) + matching `retry-after-ms` |
| unknown `x-lane` value | 400 | `invalid_request_error` | `invalid_lane` | `x-should-retry: false` |
| batch-class api key requesting `x-lane: interactive` | 403 | `authentication_error` | — | `x-should-retry: false` |
| bad / disabled api key | 401 / 403 | `authentication_error` | — | `x-should-retry: false` |
| worker channel dropped (`cmd_tx.send` fails) | 503 | `server_error` | `overloaded` | `Retry-After: 2` + `retry-after-ms: 2000` |

Engine-fault bodies carry a stable sentence, never the producer text (memra#143, 2026-09-04):
`overloaded` reads "the model is temporarily at capacity; retry after the Retry-After delay",
`engine_error` reads "the engine could not complete this request; retry, and report the
request id if it persists". The producer text (a driver or executor Debug string such as
`step error: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")`) goes to the
`[engine-error] class=... <text>` server-log line on every construction and nowhere else; that
line is the operator's evidence and is why serving boxes must not send stdout/stderr to
`/dev/null` without a log file. Producer-authored refusals (shed reasons, request-shape
errors, constraint errors) keep their exact text.

Unknown model is a deliberate **400, not 404**: OpenRouter's uptime math counts 404s
against the provider and excludes 400s, and the `code` is what clients branch on either
way. `Retry-After` is always integer seconds ≤ 60 (RFC 9110 delay-seconds; litellm honors
only `0 < v ≤ 60`, openai-python abandons retry past 120s) with a matching
`retry-after-ms`, which openai-python reads first. A **mid-stream** failure — after the 200
is committed and no status code is left to change — emits the same error object as a
`data:` chunk and closes the connection.

The channel-drop path uses the same 2-second constant as the supervisor's first respawn delay, so
the HTTP hint cannot drift from the recovery ladder. The control-plane probe 503s described above
also pass through the shared contract builder; there are no bare 503 producers left in
`crates/memra-server/src`.

### ADSD acceptance-collapse class (operations)

[ADSD](https://arxiv.org/abs/2607.21804) shows that an adversarial suffix can collapse speculative
acceptance as cost amplification while the verifier still returns correct outputs. For keyed
serving, memra now records each retired speculative request into a bounded eight-request tenant
window and compares it with the same model's rolling 64-request history. An eligible baseline
**excluding that tenant's own rows** is preferred. If none exists, the detector falls back to the
tenant's older historical rows after excluding the preceding seven rows that join the current
sample in the short window, so the two populations do not overlap. Either baseline needs 16
samples and 512 drafted tokens; a tenant window needs at least 128 drafted tokens. A window is
collapse-shaped when its acceptance is at least 0.20 below baseline and its pooled two-proportion
z-score, accounting for sampling error in both windows, is at most -3.0. Three consecutive
collapse-shaped observations emit one `[adsd-suspect]` log and increment that tenant's
`adsd_suspect_total`. The incident latches until recovery within 0.10 of its comparator; a
historical comparator is frozen while latched so a persistent collapse cannot dilute its own
baseline and rearm.

This increment is **detection only**. It does not disable speculative decoding, evict a session,
alter lossless verification, or automatically throttle the tenant. Inspect the request-level
`usage.spec` evidence and log, then manually apply the existing tenant/lane rate limit if the
traffic is abusive. The counter follows the metrics isolation policy above: completion keys see
only their own entry, while the dedicated operator scrape token sees all tenants.

The strongest known adversary this acceptance-signal detector faces is
[Mistletoe](https://arxiv.org/abs/2605.14005): a null-space-projected, semantic-preserving attack
that degrades drafter-target agreement and collapses acceptance length and throughput while output
quality and perplexity remain normal.

## Safetensors checkpoint serving (serve-st + fp8-ship lanes, 2026-08-04)

`MEMRA_MODELS` accepts safetensors checkpoint directories (`config.json` +
`model.safetensors[.index.json]`) and repack dirs alongside GGUF paths — validated at
parse time (a bogus dir fails naming the missing file). Chat templates come from the
checkpoint's own tokenizer config (`from_hf_dir`); template-less dirs 400 with a pointer
to `/v1/completions`. Official Qwen FP8 block-128 checkpoints load bit-exact (GPU
dequant, load wall 843.9 → 291.6 s = **2.89x faster load**) and **spec decode runs out of
the box on the checkpoint's embedded MTP head** — **128.06 tok/s** from the checkpoint's
own `mtp.safetensors` (**2.61x** the same-run plain 48.99), 136.75 with an own-trim
drafter, on rig **`rig2x5090-serve`** (rented 2x RTX 5090; there is no official-FP8 cell
on any RTX PRO 6000 board — do not merge the two). The win is **load time, not decode
throughput**: the e4m3-resident arm is flat by construction (weights dequantize onto the
Q8_0 arm), and spec **triples TTFT** on this arm (0.170 → 0.466 s).
Receipts: `research/fp8ship-20260804/official/`.

The ST-spec exactness scare (#68) was root-caused to a serve-side bug that was never
ST-specific: the per-session persistent draft graph replayed with dangling pool
addresses (capture transients not retained + the fa-partials pool freeing grown-past
buffers the capture baked) — reproducible on GGUF session bursts at n>=600 too. Fixed
via capture-retain keepers on `DraftGraphCtx` + retire-on-grow for the fa partials pool;
the quarantine is lifted and dir checkpoints are spec-eligible by default
(`MEMRA_SERVE_SPEC=0` is the rollback door). Gate: `tools/serve-st-gate.sh` — item 3 pins
the CLI ST-dir branch and the server to identical greedy token streams on a 64-token
window, and item 4 pins the DEFAULT (spec-on) server against the **tokenwise serve
oracle** (`MEMRA_SERVE_SPEC=0 MEMRA_SERVE_BATCH=0` — same worker, plain decode) at a
400-token window, prefix-tolerant for burst overshoot. Note what item 4 is *not*: the
comparator is deliberately the tokenwise **serve** arm, not the run-gen CLI, because both
the batched-plain path and the CLI carry their own accepted near-tie FP classes at long
windows (see [first-token cross-config
drift](#first-token-cross-config-drift-batched-prime--stated-honestly) below). Do not
restate this gate as "token-identical to the CLI oracle".

## Constrained decoding (`response_format`) — lanes constrained + constrained-full, 2026-08-03

`/v1/chat/completions` honors `response_format` `{"type":"json_object"}` and
`{"type":"json_schema","json_schema":{...,"schema":{...}}}` as REAL constrained decoding:
the schema compiles to an [llguidance](https://github.com/guidance-ai/llguidance) grammar,
and each step's packed token bitset uploads to a stable per-session device buffer (~31KB
H2D) where `mask_logits_f32` bans disallowed tokens on device — BEFORE the same
device-sample / lean-logits / CUDA-graph / speculative paths unconstrained sessions ride.
No path is lost to being constrained.

### Compile isolation and limits

Schema compilation stays off the CUDA scheduler tick. Each loaded model owns a bounded eight-job
CPU compiler queue; first-use TokTrie/`ConstraintFactory` construction and per-request matcher
compilation run there while normal sessions continue stepping. Pre-admission rejects schemas over
512 KiB, 64 raw JSON levels, or 32 Ki JSON values as loud 400s
(`MAX_SCHEMA_BYTES`, `MAX_SCHEMA_DEPTH`, `MAX_SCHEMA_NODES`).

Every accepted compile has the five-second `CONSTRAINT_COMPILE_TIMEOUT` request deadline. An
overrun returns the retryable overloaded-class error `response_format compilation did not finish
within 5000 ms; retry with a smaller schema` and a fresh compiler drains later work. The supervisor
reaps workers that finish late, but four simultaneously abandoned workers
(`CONSTRAINT_ABANDONED_WORKER_CAP`) fail-close constrained compilation for that model until the
server worker restarts; later submissions are refused without spawning another worker. See the
[sec5 watchdog results](../research/sec5-20260811/RESULTS.md) and
[sec6 cap results](../research/sec6-20260811/RESULTS.md).

- **Cost:** plain constrained-greedy = **99.4% of unconstrained** (123.7 vs 124.4 tok/s,
  q9 N=3 same-session, local RTX 5090, Qwen3.5-9B NVFP4, 256-token greedy); per-step
  grammar compute 0.006–0.007 ms. **That 99.4% is the plain lane only — the speculative
  lane pays far more: 153.4 vs 194.4 = 79%.** Never quote 99.4% for the spec path. The
  remaining constrained-vs-unconstrained gap is draft acceptance under a tight grammar,
  not mask overhead.
- **Draft-side masking (lane/draft-mask, 2026-08-04):** the drafter is masked too. A
  constrained spec session clones the session's grammar matcher once per spec round
  (0.002 ms), advances the clone with each proposed token, and bans the illegal ids in the
  draft head's own logits — in-graph on the captured draft chain, permuted through `d2t`
  for trimmed draft heads. Proposals are legal by construction, so the verify-side
  truncation backstop (which stays, as the correctness backstop) stops firing:
  `gram_cuts` went 3/12, 3/15, 1/10, 28/30, 18/25 -> **0/N on every cell measured**.
  Bounded tight schema: acceptance 0.561 -> 0.651, 216.6 -> 227.5 tok/s (+5.0%, N=3 warm).
  Cells whose drafter already proposed legal tokens (json_object, loose prose) move inside
  noise; unconstrained traffic is inert. Rollback seam `MEMRA_DRAFT_MASK=0`. Receipts
  `research/draft-mask-20260804/`.
- **Exactness:** device-mask greedy is byte-identical to the host -inf oracle
  (`MEMRA_CONSTRAIN_HOST=1`), spec-constrained is byte-identical to plain-constrained,
  graphed is byte-identical to eager, draft-masking ON is byte-identical to OFF (greedy and
  seeded-sampled, 7 cells), and unconstrained requests are byte-identical to the pre-lane
  binary (the isolation contract). Kernel-check pins `mask_logits_col` bit-identity.
  One measured exception, documented because it is NOT a masking property: an unbounded
  schema that lets the model degenerate into arbitrary whitespace against a token cap has a
  draft-chain-SHAPE-dependent tail (verify batch shape T changes FP summation order, which
  flips argmax at the near-ties in that tail). The pre-lane binary shows the same
  divergence across `MEMRA_SPEC_K=3/2/1` on that cell with no draft-mask code present;
  with shape held fixed the arms are byte-identical. Bound the schema and it goes away.
- **Think interaction (updated lane/step37-postthink-grammar, 2026-08-30):** on a
  switch-carrying template, constrained requests force the template's no-think switch (a
  grammar masking from token 0 can never close an open `<think>` tail), byte-identical
  to before. On a think-FORCED template with a derivable think-close token contract
  (step37: the tokenizer's atomic `</think>` token, id 128799 on the NVFP4 artifact),
  constrained requests serve POST-THINK two-phase decoding instead: the think phase runs
  unconstrained exactly as the model was trained, with every end-of-generation id banned
  (so the response cannot end inside think: the receipted step37 EOS-inside-think
  quirk), and the grammar clamps every token from the close on. The close detector
  matches TOKEN IDS (rolling KMP over the emitted stream), never decoded text. The
  `reasoning` channel is present and streams as usual; `content` is the grammar-clamped
  JSON. Post-think sessions never ride spec (plain constrained decode; the `[spec-k]`
  admit line receipts K=0). `MEMRA_POSTTHINK_CEILING=<tokens>` (default off) force-closes
  a think that never ends. FAIL-CLOSED terminal: a budget (or stop sequence, or context
  bound) that ends generation INSIDE the think channel produced zero schema-constrained
  content, so the request returns a named 400 `invalid_request_error` naming the field
  (`max_tokens` / `stop`) and the reasoning-token count, never a 200 with empty content;
  mid-stream the same error object ends the stream. `finish_reason: length` remains
  possible only after the grammar engaged (non-empty, truncated content). A think-forced
  template with NO derivable close contract keeps the loud 400.
- Unknown `response_format` types remain loud 400s. `/v1/completions` (non-chat) carries
  no `response_format`.

Receipts: `research/constrained-20260803/` (v1) + `research/constrained-full-20260803/`
(full battery: every path, cross-path identity, three-way perf).

## Prompt caching (cross-request prefix cache) — 2026-08-02

Two caching tiers serve prompt tokens without recomputing them:

1. **Continuation pool** (pre-existing, `MEMRA_KV_REUSE`): a retired session parks its whole
   (prompt + generation) state; a new prompt that EXACTLY EXTENDS it resumes. Single-use,
   exact-extension only — a new session that merely shares a system prompt always missed.
2. **Cross-request prefix cache** (`MEMRA_PREFIX_CACHE_MB`, 0 = off): compact device snapshots
   of primed state at token boundaries, keyed by the exact token-id prefix within each model and
   cache namespace. All entries share one worker-global byte budget, with byte-budgeted segmented
   LRU (SLRU) eviction by default and plain global LRU under
   `MEMRA_PREFIX_CACHE_POLICY=lru`. With no override, the budget holds two full-`MEMRA_CTX`
   entries of the largest loaded model, clamped to post-load driver-free VRAM minus the
   serving-transient reserve; an explicit value remains authoritative. Entries are REUSABLE — a hit
   deep-copies the entry into the new session's cache, so one marketplace system prompt serves
   any number of sessions. Learning sequence for a shared-prefix pattern: request 1 seeds its
   full prompt, request 2 split-primes at the longest-common-prefix and inserts the boundary
   entry, request 3+ hit. Hybrid models are safe by construction: GDN conv/ssm state cannot be
   truncated to a shorter prefix, so the state is snapshotted AT the boundary while a fresh
   session primes — never rolled back.

**Exactness contract:** an entry stores the KV/recurrent bytes from WHATEVER prime config ran
(single, chunked, or concat batch-prime); decode from those bytes is deterministic, so a
cached hit is bit-identical to the run that computed the prefix — gated 16/16 partial-prefix
+ 16/16 full-prefix cached-vs-fresh greedy identity across depths
(`research/prompt-cache-20260802/gate-exact.jsonl`). Comparing a cached-hit stream against a
DIFFERENT prime config's fresh stream inherits the batched-prime near-tie first-token law
("First-token cross-config drift" below) — same documented class, reported not gated.

**Policy:** MTP spec sessions probe and publish the cache. Their boundary captures include the
trunk state, MTP draft plane, boundary logits, and hidden anchor needed to re-arm speculation on a
qualified whole-entry hit; an unqualified hit serves through the exact plain path. DFlash sessions
cannot restore their independent draft KV from a trunk entry, so load decides before lookup:
an admitted low-load DFlash request preserves but ignores any trunk-only hit and primes the full
prompt. An exact existing entry is promoted/refreshed without cached-token credit; otherwise the
first exact same-wave owner publishes one full-prompt trunk snapshot and peers skip duplicate
captures. A request shed to plain at higher load probes and consumes that unchanged entry through
the normal restore path. DFlash does not add an LCP or message-boundary capture arm, and disabling
the prefix-cache budget disables its capture too. Legacy round-robin mode
(`MEMRA_SERVE_BATCH=0`) bypasses the prefix cache.
The segmentation rules below describe the default `MEMRA_PREFIX_CACHE_POLICY=slru` path;
`MEMRA_PREFIX_CACHE_POLICY=lru` forces the protected share to 100% and restores plain global LRU.
New entries enter PROBATION and earn PROTECTED residency only on a successful reuse. The global
byte budget defaults to an 80% protected target and 20% probation target
(`MEMRA_PREFIX_CACHE_PROTECTED_PCT`); probation can borrow unused protected bytes, so a cold cache
uses the full budget and a large individually fitting entry is not refused merely because it is
larger than the nominal probation share. Protected overflow demotes protected LRU back to
probation, and capacity pressure evicts probation LRU before protected LRU. Thus one-hit scan
traffic cycles through probation instead of displacing entries that have demonstrated reuse.
If a pinned fanout snapshot cannot fit from probation plus the protected bytes its own promotion
would demote, that snapshot is not retained; participants continue from their private session
copies instead of evicting below the protected byte share.
Sessions always win over unpinned cache residency: a failed session-cache allocation evicts every
unpinned entry across both segments and retries before erroring. Entries leased by live hit/fanout
requests remain pinned until the last participant retires, then re-enter their current segment at
current recency. The `(model, cache_salt)` visibility boundary, global byte ceiling, and refusal
of an entry larger than the entire budget are unchanged.

**Same-window cold fanout (`MEMRA_PREFIX_DEDUP`, default on, 2026-08-08):** immediately before
fresh-prime batching, eligible cold requests are grouped only inside the exact
`(model, tenant-scoped cache namespace)` pool. One leader primes the budget-capped exact common
token prefix; each sibling deep-copies the snapshot, receives exact cached-token credit, and
holds one entry lease. Hashes may label receipts, but exact token equality decides membership,
and cross-tenant or cross-salt requests are never compared. The N=8 Step receipt computed one
1024-token prefix and served seven cached copies, reducing same-burst p50 TTFT from 22.263 s to
3.852 s; `MEMRA_PREFIX_DEDUP=0` restores eight independent cold primes. Security, accounting,
pinning, and API-key isolation receipts: [`research/prefixdedup-20260808/`](../research/prefixdedup-20260808/).

<a id="cache-salt-isolation"></a>

**Per-tenant isolation (`cache_salt`) — PC-ISO, 2026-08-02:** every cross-request reuse
tier (prefix cache, continuation pool, spec pool) keys on (model, cache namespace), not
model alone. The namespace comes from the optional `cache_salt` string field on
`/v1/completions` and `/v1/chat/completions` (the vLLM `cache_salt` design, OpenAI-
compatible extension): requests only share cached prefixes with requests carrying the
SAME salt, in either direction, so `usage.prompt_tokens_details.cached_tokens` can only
ever reflect the caller's own namespace's history — the CacheProbe/PROMPTPEEK cross-tenant
hit-oracle mitigation (`research/cache-tools-20260802/REPORT.md` §1.4/§4). No salt = the
default `""` namespace: single-tenant deployments behave exactly as before (no new env
knob — the namespace is a request field, not a flag). The prefix cache's byte budget stays
global across namespaces. Whole-session continuation/spec pools keep
`MEMRA_REUSE_POOL` entries per `(model, namespace)` for locality, so their pre-ceiling VRAM
exposure was roughly `MEMRA_REUSE_POOL × distinct live cache_salt namespaces × parked-entry
bytes` per populated tier. `MEMRA_REUSE_POOL_GLOBAL_CAP` now bounds the process-wide sum across
both tiers and all models/namespaces, evicting the globally oldest entry before a park would
cross it; see the [flag catalog](FLAGS.md) for sizing. Its default 16 preserves the measured
[27bab two-namespace Q27 shape](../research/27bab-20260810/RESULTS.md#vram-ledger) (16 spec
entries, zero OOM parks, 27.34 GB driver-free), while the salt-per-turn control reached 76
entries and a captured CUDA OOM. A gateway multiplexing many end-users through one API key —
the marketplace listing shape — MUST still set a per-end-user/session salt and bound active
fanout to avoid churn. Gates: `research/pc-iso-20260802/` (same-salt hit, cross-salt miss both
directions, default-namespace blindness; the integrate-cache intersection gate re-run
unmodified as the no-salt regression).

**Accounting:** every response shape carries OpenAI-schema usage with the worker-truth split —
`usage.prompt_tokens`, `completion_tokens`, `total_tokens`, and
`prompt_tokens_details.cached_tokens` (tokens resumed from ANY cache tier: continuation pool,
spec resume, or prefix cache — the field name providers report cached reads under: OpenAI,
OpenRouter, and Grok chat all use `prompt_tokens_details.cached_tokens`). Cached
prefill costs ~0 to serve and bills at 25% of input on the OpenRouter hy3 endpoints — the
margin lever (`research/or-provider-20260802/REPORT.md`).

**Host spill tier (`MEMRA_KV_HOST_MB`, default 0 = off; lane/kv-host-spill-20260830):** an
optional pinned-host RAM tier BEHIND the device prefix cache. Design law: the host tier
feeds the device cache and the restore path is untouched. A device capacity eviction
demotes the entry's bytes verbatim (already q8_0 K / q5_1 V at rest, no requantization)
into pinned cacheable host memory; a later probe that misses the device pool but exactly
matches a host entry (same exact-token / PC-ISO key rules) re-materializes a normal device
entry and serves through the existing restore, byte-lossless by construction. Two flags:
`MEMRA_KV_HOST_MB` (the per-stack budget in MiB, boot-clamped to MemAvailable x 0.6 with a
loud warning; pinned-alloc failure is loud and latches the tier off, never a silent
pageable fallback) and `MEMRA_KV_HOST_VERIFY` (diagnostic sha256 of the entry's logical
state at demote, re-checked at promote; too slow always-on for GB entries). Rollback seam:
`MEMRA_KV_HOST_MB=0` is byte-identical to today by construction, nothing ever reaches the
tier. Model exclusions ride the upstream `prefix_snapshot` refusals for free: step37's SWA
ring and glm5's latent planes cannot produce a demotable entry in the first place. Operator
receipts: `prefix_host_entries/bytes/demotions/promotions/demote_ms/promote_ms/
rejected_allocs` in `/metrics` (the `*_ms` fields are cumulative copy wall-time, the
tick-stall receipt), plus per-copy `[prefix-host]` log lines. See the
[flag catalog](FLAGS.md) rows for arms, receipts pointers, and the pending pod battery.

**Tenant lifecycle purge (lane/kv-tenancy-compaction-20260831, tiering spec §0.5):** key
revocation or tenant deletion must not leave that tenant's prompt bytes parked in pinned
host RAM. The engine exposes `RuntimeHandles.purge` (`PurgeHandle::purge_tenant(tenant)`)
beside the trim handle; the deployment binary wires it into its admin surface as
`/admin/tenants/{tenant}/purge` and calls it from BOTH the revocation and deletion paths.
Contract notes: the parameter is `{tenant}` (the keyring tenant id, the same string
`--gen-key <tenant>` took), never `{tenant_id}`; the purge removes every host-tier entry
across ALL of the tenant's end-user salts (the PC-ISO `t:<tenant>` row identity, the same
`scope_namespace -> meter_key` derivation that scoped the entries in), and ALSO sweeps the
tenant's unpinned DEVICE prefix entries so a later capacity eviction cannot demote the
purged bytes straight back into host RAM. Pinned device entries (leased by in-flight
sessions) are reported as `device_pinned_left`, not dropped: revocation never aborts an
admitted request; re-fire after those sessions retire when the report says pins remained.
Raw-salt (no-keyring) namespaces carry no tenant and never match. Receipts:
`prefix_host_purges/purged_entries/purged_bytes` in `/metrics` plus the per-purge
`[prefix-host] purge tenant=...` log line with both tiers' counts.

**Per-tenant host-pool share cap (`MEMRA_KV_HOST_TENANT_PCT`, default 50;
lane/kv-tenancy-compaction-20260831):** one tenant's maximum share of the
`MEMRA_KV_HOST_MB` budget, keyed on the same tenant row identity as the `tenants`
receipt. A demotion that would push the tenant past its share evaporates (checked before
the D2H copy, so it also skips the PCIe trip) instead of demoting, so one tenant can
never squeeze the others out of the pool; `100` disarms the check for single-tenant
deployments (the global byte-LRU then governs, exactly as before the flag). Receipt:
`prefix_host_tenant_rejects`, the boot line's `tenant share cap` clause, and the
per-evaporation `[prefix-host] demote evaporated at the tenant share cap` log line (the
exact production text; a gate greps this line, not a paraphrase). Full trade discussion
in the [flag catalog](FLAGS.md) row.

**Continuation-pool park compaction (`MEMRA_KV_PARK_COMPACT`, default 0 = off;
lane/kv-tenancy-compaction-20260831, tiering spec Arc C1):** a parked plain-pool session
keeps its full ladder-cap `Cache` allocation today (cap = `cache.max_ctx`, so a 6k-token
session can park ~1 GB on a 262k-ctx deployment). Armed, the park compacts the cache to
exactly its committed length through the same snapshot + checkpoint-restore machinery the
plain-affinity grow uses, and the resume grows it back to the request's own charged cap
before any suffix primes; failures in either direction fall back to today's behavior
(full-size park, or cold serve). SPEC/DSPARK pools are out of scope by design (they park
live engine sessions). Default off until the pod battery lands the resume byte-identity
gate and the step-OOM adjacency replay; see the [flag catalog](FLAGS.md) row and
`research/kv-tenancy-20260831/REPORT.md` for the pending cells.

## Cache-hit metering (lane/cache-metering, 2026-08-07)

The cache receipt surface is cumulative since process start, worker-truth, published every 32nd
tick AND on every request retire so a post-workload scrape is never stale. Global prefix/LCP rows
below are operator-only; completion credentials retain the base prompt totals and their permitted
`tenants` rows:

**Publish semantics (tick-boundary truth):** the worker owns every counter and copies a
snapshot into `/metrics` at scheduler-tick END, so within a tick the named log lines are the
authoritative per-event receipt, and a scrape racing a response's final tick can read totals
up to one tick stale (the terminal token event is sent mid-tick, before the retire sweep and
the publish). Since lane/kv-battery-fixups-20260831, any tick whose `prefix_host_*` event
counters moved ALSO forces the publish (`HostPrefixCache::telemetry_stamp`): before that, an
admit-tick promote and an idle-wake pause-sweep demote landed on ticks the 32-tick/retire
forces skipped, and the counters lagged the `[prefix-host]` log lines around quiesce; the
receipt of record is the 2026-08-31 battery's O6 cell, where a promote logged at 89.9 ms was
still reading `promotions=0 / promote_ms=0.0` in the scrape taken right after its response
completed (darklanes `research/kv-fastband-20260830/battery-20260831/RESULTS.md`, O6
finding). Gates assert on the log lines; metrics deltas are recorded fields.

| field | meaning |
|---|---|
| `prompt_tokens_in` / `cached_tokens_in` | every prompt token admitted / the subset served from any cache tier |
| `computed_tokens_in` | `prompt - cached` — the denominator of the revenue multiplier |
| `cache_hit_token_ratio` | global `cached / prompt`, token-weighted; operator scope only |
| `prefix_cache_hits/misses/inserts/evictions` | global prefix-cache probe outcomes + churn; operator scope only |
| `prefix_cache_hit_tokens` | global token-weighted hit mass (sum of served entry lengths); operator scope only |
| `prefix_cache_entries/bytes` | resident state; operator scope only |
| `prefix_host_entries/bytes/demotions/promotions/demote_ms/promote_ms/rejected_allocs` | pinned-host spill tier (`MEMRA_KV_HOST_MB`, lane/kv-host-spill-20260830): current gauges, tier round-trips, cumulative copy wall-time (the tick-stall receipt: ms per demotion = `demote_ms / demotions`), and alloc/copy failures; operator scope only |
| `prefix_host_purges/purged_entries/purged_bytes` | tenant lifecycle purges (`PurgeHandle`, lane/kv-tenancy-compaction-20260831): cumulative invocation count and host-tier entries/bytes removed, the receipt that a revocation/deletion actually cleared resident state; operator scope only |
| `prefix_host_tenant_rejects` | demotions evaporated at the per-tenant share cap (`MEMRA_KV_HOST_TENANT_PCT`, lane/kv-tenancy-compaction-20260831); a nonzero rate with low pool occupancy is the whale-tenant signature the cap bounds; operator scope only |
| `lcp_histogram` | global `{edges, counts}`: one sample per prefix-cache probe — served entry length on a hit, best LCP on a miss. Lower-edge buckets `[0,1,16,32,64,128,256,512,1024,2048,4096]`, last unbounded; `[64,512)` (buckets 4..=6) is the tick-seg segmentation window; operator scope only |
| `tenants` | per-tenant `{prompt_tokens_in, cached_tokens_in, cache_hit_token_ratio}` rows — absent until the first admit |
| `adsd_suspect_total` | per-tenant detection-only acceptance-collapse incident counters; absent until the first incident |

`tenants` composes with PC-ISO tenancy: rows key on the TENANT half of the namespace
(keyring deployments get one row per tenant across its end-user salts, `t:<tenant>`;
no-keyring deployments key on the raw `cache_salt`, `""` = the default namespace). Rows are
bounded (256): overflow traffic aggregates under `"(other)"`, so totals stay exact while a
salt-spraying client cannot grow the map. Spec-tier and non-batched requests never probe the
prefix cache and are absent from the histogram by construction (their cached tokens still
count in `cached_tokens_in` via the continuation/spec pools).

**The economics query** (`tools/cache_economics.py <metrics-url-or-json>`): turns a scrape
into the earning-model row — `revenue_multiplier = billed_prompt_tokens /
computed_prompt_tokens` at a chosen cached-token billing factor (`--cache-billing-factor`,
1.0 = cached bills full price, 0.25 = the OR cached-input tier), plus per-tenant multipliers
and the tick-seg window share. JSON row on stdout (ledger-appendable), summary on stderr.
For a live URL it sends `MEMRA_METRICS_TOKEN`, falling back to `MEMRA_API_KEY`, when set.

**Fleet receipt accumulator** (`tools/fleet-meter.sh`): the pre-listing hit-rate receipt for
controlled replay traffic. A one-shot scrape of `http://127.0.0.1:8002/metrics` appends only
the UTC timestamp, prompt/cached/computed counters, hit ratio, LCP histogram, tenants, and a
`restart` marker to `research/fleet-meter/rig5090-fleet.jsonl`. An unchanged scrape is
idempotently skipped. A failed scrape logs `skip` and exits successfully; it never starts,
stops, or otherwise mutates the owner-critical server. On a keyed deployment, set
`FLEET_METRICS_TOKEN` (or `MEMRA_METRICS_TOKEN`) in a mode-`0640` EnvironmentFile; the script
sends it through curl stdin so it is not exposed in the process arguments.

**Fleet replay driver** (`tools/fleet-replay.py`): run only in dev-idle windows against the
existing port-8002 deployment. Its low defaults are five minutes at 3 requests/minute,
89.5:1 prompt:completion, 12 carried synthetic sessions, four tenant-scoped `cache_salt`
values, and eight shared 1k-4k-token system-prompt/tool-schema templates; exponential
inter-arrival times and 2-4-turn session bursts exercise both prefix sharing and continuation.
Set `MEMRA_API_KEY` to the local deployment key and run
`tools/fleet-replay.py --duration 300`. Any meter interval driven by this tool is labeled
**replay-calibrated**: it is a controlled synthetic workload and must never be described as
organic traffic.

```bash
tools/fleet-meter.sh --once                         # cron/timer-safe snapshot
tools/fleet-meter.sh --loop --interval-minutes 30  # foreground accumulator
python3 tools/fleet-report.py                       # all UTC days
python3 tools/fleet-report.py --days 7              # rolling weekly view
```

The example `deploy/systemd/memra-fleet-meter.{service,timer}` runs the one-shot form every
30 minutes. Copy the units, override their `/opt/memra` path and service account for the
host, add a protected EnvironmentFile carrying the scrape token when server auth is enabled,
then enable the timer; do not point the meter at a public endpoint.

The report diffs cumulative counters and histograms. A counter regression (or an explicit
`restart=true`) starts a new segment whose current values count from zero, so restarts never
produce negative traffic. The first snapshot intentionally counts the server's existing
cumulative receipt. Later intervals are attributed to the UTC day of their ending snapshot,
which bounds day-edge uncertainty to the snapshot cadence. Each daily row shows fleet prompt
tokens, cached/computed splits, hit-token ratio and day-over-day change, the revenue
multiplier band at cached-token billing factors 0.25 and 1.0, tick-seg `[64,512)` probe
share, and detected restart count. Revenue and tick-seg math comes directly from
`tools/cache_economics.py`; the report does not carry a second formula.

**Exactness gate** (`tools/cache-meter-gate.py`, serve-smoke arm 7b): N requests sharing a
K-token `prompt_ids` prefix must meter exactly — seed/LCP-split requests `cached_tokens: 0`,
steady-state hits `cached_tokens == K`, a same-prefix request under a different `cache_salt`
cold (PC-ISO), `/metrics` totals closed-form, histogram bucket-exact, economics row
crosschecked. 26/26 on the 5090; disabling the cache inverts 16/26 (teeth). Overhead A/B
(pre-lane binary vs instrumented, both resident, interleaved x5, N=100/arm, prefix-hit
steady state): p50 −0.03%, p95 −0.19% — no measurable serve overhead (<0.5% p95 bar).
Receipts: `research/cache-meter-20260807/`.

## Spec-decode acceptance telemetry (lane/accept-telemetry, 2026-08-05)

Always-on per-draft-position acceptance counters, the llama.cpp #26389 / vLLM spec-decode
counter schema. WHY: the 2026-08-05 dogfood head-to-head found short-context sampled
acceptance at 0.55 vs 0.73 full-draft — a posthoc dig that this surface turns into a live
gauge (drafter health on a new checkpoint is readable in minutes, and the K-policy work
gets a per-position decay curve for free).

**`GET /metrics` — the operator-only `spec` block**, per model, cumulative since the model
loaded (models load once per server process, so counters reset on restart, never mid-run).
It is visible to `MEMRA_METRICS_TOKEN` and no-key loopback development, never to a completion
credential. Absent until the first spec burst — spec-off deployments see the exact pre-lane
payload:

```json
"spec": {
  "q9": {
    "rounds": 118, "drafted": 354, "accepted": 213,
    "acceptance_rate": 0.602, "tokens_per_round": 2.805,
    "pos_drafted":  [118, 118, 118],
    "pos_accepted": [96, 71, 46],
    "accept_rate_per_pos": [0.814, 0.602, 0.390]
  }
}
```

`accept_rate_per_pos[j]` = P(draft position j accepted | a round offered position j) — healthy
spec decode decays monotonically from position 0 (acceptance is a prefix walk: position j can
only be accepted if 0..j-1 were). Arrays are trimmed to the deepest position ever drafted
(up to 8 tracked positions; totals count deeper drafts too). Normalization matches
`MEMRA_SPEC_STATS`: a p-min-cut chain token is counted in neither drafted nor accepted. The
opt-in round-stream arm (`MEMRA_SPEC_STREAM=1`) keeps its accept counts on device, so under
it per-position arrays cover the standard-path rounds only; totals stay complete.

The first-class operator metrics use a rolling 30-second per-model window. `spec_tau` is the
mean accepted draft-prefix length (`accepted / rounds`, excluding the target bonus token), and
`spec_accept_by_position` carries accepted counts plus the offered denominator needed when K
varies by request:

```json
"spec_tau": { "q9": 1.805 },
"spec_accept_by_position": {
  "q9": {
    "window_seconds": 30.0, "rounds": 118,
    "offered":  [118, 118, 118],
    "accepted": [96, 71, 46],
    "accept_rate": [0.814, 0.602, 0.390]
  }
}
```

Both fields are absent until the rolling window has signal and follow the same operator-only scope
as the cumulative `spec` block. The cumulative block remains unchanged for compatibility.

**`usage.spec` — per-request summary.** Spec-decode requests carry their OWN
rounds/drafted/accepted + `acceptance_rate` in the response usage object (this request only —
pool-resumed sessions do not leak prior requests' counts). Additive and OpenAI-safe: official
SDKs ignore unknown usage fields, no existing field changes, and non-spec requests carry no
`spec` key at all.

**Cost:** fixed-size relaxed atomic adds at the round accounting the engine loop already does — zero
GPU syncs, zero per-token allocation, no hot-path lock (the worker merges per-burst deltas
into its own map; the metrics mutex is only taken on the existing 32nd-tick publish, plus a
force-publish when a spec session retires so one-shot requests are visible immediately).
Validation capture: `research/accept-telemetry-20260805/`.

## API keys — multi-key tenant auth (lane/api-keys, 2026-08-05)

Bearer auth that maps key → tenant, so cache isolation, QoS lane class, rate-limit
headers, and metering all key off a real tenant identity. Launch-shaped: a file-backed
keyring + a CLI, no web UI.

**Configuration.** `MEMRA_API_KEYS=/path/keys.toml` — TOML `[[keys]]` entries carrying
`sha256` (of the plaintext key — the plaintext is never stored), `tenant`
(`[A-Za-z0-9_-]+`), `lane` (`interactive` default | `batch`), `enabled`, and optional
`rate_limit`. An inline env form `tenant:sha256hex[:lane],...` exists for file-less
deploys. A malformed ring is a startup FATAL (never partially applied); the file
hot-reloads on mtime change (≤2s poll — chosen over SIGHUP: no signal thread, cannot be
missed), and a broken rewrite keeps the previous ring and logs loudly — auth never fails
open because of a typo.

A file-backed ring must be a private regular non-symlink file (0600/0640 class), owned
by the service uid, with exactly one hard link and size at most 8 MiB. Startup fails
before serving when this class is wrong; hot reload, `--gen-key`, and `--revoke-key`
revalidate it rather than weakening the rule after boot.

**Lifecycle CLI.**
```
memra-server --gen-key acme [--lane batch] [--rate-limit 4] [--keys /path/keys.toml]
memra-server --revoke-key <key-prefix> [--keys /path/keys.toml]
```
`--gen-key` prints the plaintext key (`mk-<tenant>-<48 hex>`) exactly ONCE on stdout and
appends the hash entry; newly created keyrings use mode `0640`. `--revoke-key` disables
by unambiguous prefix (or full key), writing and fsyncing a randomized hidden sibling
temporary file before an atomic rename over the live ring, so a concurrent hot reload
sees the complete old or new file.
A running server picks the revocation up on the next poll. `--keys` defaults to
`MEMRA_API_KEYS`.

**Request law.** `Authorization: Bearer <key>` on every `/v1` completion route:
- keyring match → that key's tenant context; **disabled key → 403** (actionable,
  distinct from unknown), **unknown key / missing header → 401**;
- `MEMRA_API_KEY` (the single static key — the daily driver and every serve script)
  keeps working unchanged as tenant `default`, with or without a keyring configured;
- neither configured → open (dev behavior), tenant `default`.

When `MEMRA_METRICS_TOKEN` is unset, a keyring API key authorizes `GET /metrics` but receives
only its own tenant rows; process-wide counters are absent and `GET /yield/metrics` returns 403.
The legacy single `MEMRA_API_KEY` domain retains its existing cumulative and yield views.
When `MEMRA_METRICS_TOKEN` is set, it is the exclusive bearer for both routes and completion
keys receive 403. The routes are unauthenticated with full visibility only when the bind resolves
entirely to loopback and none of `MEMRA_API_KEY`, `MEMRA_API_KEYS`, or `MEMRA_METRICS_TOKEN` is
configured.

**What the tenant identity drives:**
- **Cache isolation:** with a keyring configured, the PC-ISO namespace is
  `t:<tenant>␟<cache_salt>` — one tenant's keys share cached prefixes, different tenants
  never do, and the `␟` (US, `\x1f`) separator is excluded from tenant ids so a
  client-controlled `cache_salt` cannot forge another tenant's namespace. `cache_salt`
  still sub-scopes WITHIN a tenant (a gateway multiplexing end-users through one key
  keeps setting per-user salts). No keyring → the raw-salt namespace, byte-identical to
  PC-ISO behavior.
- **QoS lane class:** `interactive`-class keys behave exactly like pre-lane traffic
  (default lane interactive, any `x-lane` honored). `batch`-class keys default to the
  harvest lane and are refused `x-lane: interactive` with a 403 — a bulk key cannot
  claim the protected class, by omission or by header.
- **Rate limits:** per-key `rate_limit` is a concurrency-slot override; the effective
  cap is **min(override, global lane cap)** — the global cap stays authoritative, an
  override can only narrow. A request that arrives while its tenant already holds every
  configured slot is rejected before worker admission with `429 rate_limit_exceeded`;
  two simultaneous arrivals cannot both pass the cap. The `X-RateLimit-*` trio reports
  the binding cap, with `Remaining` counting the tighter of the tenant and lane gauges.
  Multiple keys under one tenant intentionally share that tenant's gauge; issue distinct
  tenants when recipients need independent caps.
- **Metering seam:** every admitted request logs one flat
  `[meter] admit id=<x-request-id> tenant=<t> lane=<l> model=<m>` line (for the items of a
  multi-input `/v1/embeddings` or multi-document `/v1/rerank` request the id is
  `<x-request-id>.<index>`, one line per item) — the public-repo
  half; the private fork's metering layer joins these against the worker-truth usage
  lines by request id for per-tenant billing.

Gate: `tools/apikeys-gate.sh` (unit laws + live two-tenant isolation proof via
cache-hit behavior; receipts `research/apikeys-20260805/`).

## Self-serve seams

THE BUSINESS TIER LEFT THIS REPO (engine-billing-extraction-20260829; owner razor
2026-08-29: "only engine is open, business is private"). The prepaid ledger, tenant
budgets, admission modes, per-tenant capture, and the `/admin` provisioning surface now
live in the deployment's own binary, compiled against this crate's public metering seam.

What the ENGINE owns, and exposes publicly:

- `memra_server::metering` — the seam vocabulary: `Metering` (enforces_limits /
  is_limited / reserve / open / captures / limits_health / drain_kill), `Receipt` (the
  full settle lifecycle: prompt usage, per-token counting, complete, deadline-partial,
  reject, named zero-debit outcomes, capture arming), `UsageCounts`, `AdmitError`, an
  opaque `Permit`, and `MeteringFactory`. Everything speaks tokens and verdicts, never
  money.
- `memra_server::ServerWiring` — `stock()` runs the open engine with NO accounting;
  `with_metering(factory)` compiles a deployment's implementation in;
  `claiming(var)` declares which deployment-surface env vars that binary consumes
  itself; `on_ready(RuntimeHandles { trim, metadata_reload, .. })` hands over
  the engine trim handle, the model-metadata reload handle, and the drain shutdown
  signal at worker-ready. A deployment surface MUST end
  and drop its `TrimHandle` (and its `PurgeHandle` and `HostHandoffHandle`, which
  wrap worker command senders like it) on the shutdown signal — the GPU worker
  only exits when every sender drops.
  (`MetadataReloadHandle` is memory-only and needs no drop.)
- The handler obligations behind that seam are tested here with a recording mock
  (deadline partial vs unbilled, worker-truth usage sync, admission-denial mapping,
  disconnect partials, byte-exact capture feeding); what the counts COST is the
  implementation's business and is tested with it.

`MEMRA_REQUEST_LEDGER`, `MEMRA_TENANT_BUDGETS`, `MEMRA_ADMIN_ADDR`,
`MEMRA_ADMIN_TOKEN_FILE`, and `MEMRA_CAPTURE_DIR` are startup FATALs on the stock
binary: they configure surfaces this build does not carry, and set-but-unread
configuration must not fail open.

Auth stays in the engine (`memra_server::auth`, `MEMRA_API_KEYS` / `MEMRA_API_KEY`):
a key file, tenant identity, per-key lane class and rate limits are engine security
surface; what a tenant may SPEND is not.

### Long non-streaming requests: what the deadline does now (2026-08-26)

`timeout_ms` (MIN 1000 / MAX 90000 / DEFAULT 90000) bounds a non-streaming response
end-to-end. The 90 s maximum is a platform fact, not a preference: the fronting proxy
fails a non-streaming response whose headers take ~100 s, so a longer promise would be
broken upstream of this server.

Two behaviours keep that ceiling from eating a customer's work:

1. **Infeasible requests are refused at admission, not after 90 s.** A non-streaming
   request whose pessimistic estimate (prompt at `MEMRA_PREFILL_FLOOR_TOK_S`, generation
   at `MEMRA_DECODE_FLOOR_TOK_S`) needs more than 1.5x its deadline gets an immediate
   400 `nonstream_deadline_infeasible` naming the `max_tokens` that would fit and the
   streaming alternative. No slot, no receipt, no GPU time.
2. **A deadline that lands mid-generation DELIVERS what was produced.** The response is
   200 with the tokens generated so far, `finish_reason: "error"`,
   `native_finish_reason: "deadline_exceeded"`, and an `error` object carrying
   `code: "deadline_exceeded"` and `metadata.error_type: "timeout"` — the OpenRouter
   dialect this server already uses for reasoning. Those tokens ARE billed: the caller
   received them. Only a deadline that lands before ANY token still answers
   408 `deadline_exceeded`, unbilled, because there is nothing to deliver.

`finish_reason: "length"` is deliberately NOT used for a time cut. No provider's
finish-reason enum has a time value (OpenAI, Anthropic, Google and the hosted resellers all mean
max_tokens by `length`/`MAX_TOKENS`), so reporting a deadline as `length` would tell a
caller to ask for more tokens when the truth is that it must stream.

**Surface coverage.** The feasibility gate runs on all four completion surfaces
(`/v1/completions`, `/v1/chat/completions`, `/v1/messages`, `/v1/responses`) from one
body. PARTIAL DELIVERY is OpenAI-dialect only, and deliberately: `/v1/messages` keeps the
408 because the Anthropic `stop_reason` enum has no time value (labelling a time cut
`max_tokens` would be the same lie), and `/v1/responses` keeps it because OpenAI defines
`incomplete_details.reason` as `max_output_tokens | content_filter` only. Both are stated
decisions, not omissions.

**Ledger.** A delivered partial writes the billable outcome `deadline_partial` with
`error_code: "deadline_exceeded"` — distinguishable from `completed` in every census. A
zero-token miss writes `deadline_exceeded`, unbilled.

**Streaming has no such ceiling**: a stream's deadline bounds only the time to first
token, and the stream then runs as long as it needs. For work that cannot fit a
synchronous window at all, streaming is the answer this server offers today.

## Session affinity — resuming a REWRITTEN conversation (lane/session-affinity, 2026-08-05)

Both reuse tiers above require the new prompt to EXTEND what is cached (token prefix, or
text prefix). Real agent clients do not extend — they REWRITE. The owner's client strips
`<think>` blocks out of prior assistant turns before re-sending them, so turn N's prompt is
not a prefix-extension of anything, both probes miss, the parked multi-GB session is
discarded, and every turn re-primes the whole growing conversation.

Affinity answers a different question: not "does this prompt extend that session's bytes?"
but "is this the SAME CONVERSATION?" — and then resumes it at a retained boundary.

**Two identity tiers (nomination only):**

- **Explicit** — the client names its conversation. Accepted from `session_id` or `user` in
  the request body of `/v1/completions` and `/v1/chat/completions`, or the `x-session-id`
  header. Body beats header (the body is the caller's own statement of identity; a header can
  be injected by an intermediary); `session_id` beats `user`. An explicit id on one side only
  never matches: a named conversation and an anonymous one are not the same conversation.
- **Implicit** — nothing named, so identity is STRUCTURAL: the conversation is split at its
  control tokens (the chat template's own role markers) and each segment contributes a hash of
  its first and last few tokens. A rewritten segment BODY does not perturb its hash, so the
  chain's leading run survives a think-strip; three shared segments are required before an
  implicit fingerprint may name a conversation (a bare system prompt is shared by every fresh
  conversation and must not cross-link them).

**Identity nominates, BYTES decide.** A nominated session is resumed only if the new prompt
reproduces its committed tokens EXACTLY up to the boundary its last turn checkpointed. A
fingerprint collision therefore costs one wasted comparison, never a wrong resume. If the
rewrite reached BELOW the boundary, affinity declines and the request re-primes in full —
correctness first. Declines are logged with their offsets (`history diverged at N of
checkpoint M`), because a silent decline is indistinguishable from a broken mechanism.

**The boundary.** Full-attention KV is truncatable by length, so a checkpoint copies only the
GDN conv/ssm recurrent state; the draft scratch needs no copy (the next turn's fill rewrites
it). WHERE it lands differs by tier, and the difference is load-bearing:

- **Plain tier — the LAST TURN-MARKER control token** (`plain_checkpoint_boundary`). Prompt-end
  is NOT a safe boundary, and this paragraph used to say it was. A rewriting client resends the
  prior turns but the template re-renders the LIVE assistant-generation header
  (`<|im_start|>assistant\n<think>\n`, or the closed-think scaffold
  `<think>\n\n</think>\n\n`), so turn N+1 diverges from turn N a few tokens BELOW prompt end —
  inside a header the client never sent. A prompt-end checkpoint therefore always sits past the
  divergence and declines 100% of the time (`research/cachespec-20260809/RESULTS.md` §P0).
- **Spec tier (`SpecCheckpoint`) — still PROMPT END, and therefore still inert on
  rewriting multi-turn traffic.** The plain-affinity fix was never ported: `ckpt_at` arming is
  gated `spec.is_none()`, and the spec capture asserts `pos == base + prompt.len()`. Live
  symptom, seen on the v0.93.0 DE box: `spec-affinity: declined (history diverged at 3115 of
  checkpoint 3119)` — a 4-token gap, exactly the closed-think scaffold; 2 tokens with think ON.
  Distinct from the banked off-by-one (that was `+1` from the init feed, now fenced by a
  debug_assert): this is boundary CHOICE, not arithmetic, which is why the observed gap tracks
  the template (1 / 2 / 4 / 21 in the banked logs). Porting it is its own lane — the cheap half
  (reuse the prime-split snapshot the prefix cache already pays for) is blocked by
  `spec prime split is cold-session-only`, so turn 3+ would re-arm at prompt end again.
  Multi-turn sampled/greedy traffic still gets the prefix-cache tier, which since
  lane/sampled-hit-spec re-arms spec on those hits.

**Scope.** Affinity is stored per (model, cache namespace), so it adds no cross-tenant reach
beyond what the reuse tiers already have: a `cache_salt` is an affinity boundary too.
Constrained (grammar) requests never resume. Resumed sessions respect the same evict-first +
right-size ladder as new ones, and are tested against the room the request actually needs, so
a right-sized session stays affinity-eligible.

`MEMRA_AFFINITY=0` turns the mechanism off (rollback seam / exactness A/B arm; the winner is
the default and needs no flag). Receipts, byte-identity gate, and TTFT curves:
`research/session-affinity-20260805/`.

## Multi-tenant QoS — the x-lane SLO gate (lane/qos-p95, 2026-08-02)

Requests may tag a service class via the `x-lane` header: `interactive` (protected;
also the default when the header is absent — naked traffic is byte-identical),
`judge` (prefill-shaped), or `harvest` (decode-shaped bulk). The gate is engine-side
admission control: interactive always admits (waits FIFO past `MEMRA_MAX_SESSIONS`,
never rejected); judge/harvest admit only while the measured interactive decode-step
p99 stays under their fraction of `MEMRA_SLO_P99_MS` (50ms default) and shed with an
immediate `429 + Retry-After` otherwise — dark work is never queued inside the engine.
Inside the tick, interactive decode rows batch first and dark-lane prefill runs after
decode within measured SLO headroom only. Per-lane counters + the engine-truth step
p50/p99 export at `GET /yield/metrics`.

Measured at fleet scale (8 replicas, Qwen3.5-9B-Q8_0 on rented H100s, c=96 harvest + c=4
interactive, 4 conditions interleaved, N=3 passes with full teardown/bring-up per cell,
`research/qos-p95-20260802/`): the lane-blind proxy FIFO alone inflates contended
interactive p95 to 7.15s (~4x alone); with lanes on and the proxy cap at 16 (so engine
admission owns the queue — the gate cannot fix a queue it never sees), p95 drops to
3.69s (~2x alone) at -11% bulk throughput vs the cap-16 ceiling. `MEMRA_SLO_P99_MS`
is the dial: 25ms makes contended interactive statistically equal to alone
(p50 1.637s / p95 2.158s) with bulk paying -67%. Lane knobs in [FLAGS.md §1](FLAGS.md).

**Attribution, required whenever the 7.15 → 3.69s figure is quoted:** raising the proxy
cap from 8 to 16 *by itself* moves p95 **7.15 → 4.335s** (that control cell is in the same
RESULTS.md); the lane gate accounts for **4.34 → 3.69s**. Roughly half the headline
improvement is the queue, not the engine gate — which is the point of the sentence above,
not a caveat to it. Quoting 7.15 → 3.69s as "what lanes do" is refutable from our own log.

## Streaming cadence + admission latency — the felt-TTFT arc (lanes sse-cadence 2026-08-05, admission-latency 2026-08-06)

Two fixes, one arc: solo first text went **0.41 s → 0.12 s** and contended first text
**1.60 s → 0.15 s** (27B NVFP4+MTP, K=3, local 5090, N=5 medians in one lock hold), and
neither number scales with `MEMRA_SPEC_BURST` anymore.

- **Round-cadence SSE** (lane/sse-cadence): spec-burst sessions used to emit ONE
  `Event::Token` per burst — at B128 that meant 2 chunks per response and 1.16 s to first
  text. The worker now flushes text at every spec-round commit through an `on_commit` seam
  in the engine's spec loop (same detokenize-tail + `utf8_delta` cursor, same
  EOS-text-never-streamed rule), so first text is ~0.12 s and inter-chunk gap p50 ~27 ms at
  ANY burst size for a solo stream (B32 fix-off was 0.41 s / 299 ms). Content is
  byte-identical either way — only chunk boundaries move. Throughput parity measured c=1
  and c=8. Rollback: `MEMRA_SSE_PER_BURST=1`. Receipts:
  `research/sse-cadence-20260805/VERDICT.md`.
- **Admission yield + cold-first ordering** (lane/admission-latency): a request arriving
  mid-burst used to wait the whole in-flight burst out (contended first text 0.54 s at B32 /
  1.60 s at B128, i.e. burst size set round-robin admission latency). Two pieces, one flag:
  a pending admit (`PENDING_ADMITS` gauge, polled by the round hook above) ends the
  in-flight burst at the next round boundary, and sessions that have emitted nothing yet
  burst before mid-generation peers. Contended first text is now **0.123 s (B32) / 0.152 s
  (B128)** — the solo class at any burst. Content byte-identical on/off, solo AND contended.
  The cost lives at c=8 saturation only: −3.4% agg tok/s for 3.8x better p50
  (newcomer-first vs lockstep-fair; p95 tail pays); c=1 parity. Rollback:
  `MEMRA_ADMIT_YIELD=0` (both pieces). Receipts: `research/admission-20260806/VERDICT.md`.

Burst default stays **B32** by the strict flip criterion, but the two old flip-blockers are
gone: B128 buys +8.4% (c=1) / +8.5% (c=8) and now trails B32's contended first text by one
29 ms round-cadence quantum instead of a 3x cliff — a live owner call, per the
`MEMRA_SPEC_BURST` row in FLAGS.md.

## Dead-darklane background jobs — valleys carry owner work (lane/darklane-training, 2026-08-07)

The standing lab thesis: idle serve capacity carries owner research/training jobs, yielding
instantly to paying traffic. This section is the ENGINE mechanics only — which jobs run,
what a valley is worth, and every scheduling-policy/economics question live in the product
repo; the seam between the two is exactly `MEMRA_BG_JOB` + the checkpoint protocol below.

**Valley detection** invents no sensor. The scheduler already flips its health phase to
IDLE precisely when `active` and `queue` are both empty, and the phase stamp refreshes the
heartbeat — so `phase == IDLE` + heartbeat age IS the idle duration, at zero new hot-path
cost. The `PENDING_ADMITS` gauge closes the HTTP→worker handoff gap (a submitted request
the worker hasn't popped is traffic, not idleness). Exposed two ways: **`/metrics
serve_idle_seconds`** (always published in the operator view; 0.0 the instant there is any work)
and the in-process `ValleySignal` hook (`darklane.rs`) the runner polls. Receipt — signal accrues
while idle, reads 0.0 sampled mid-generation, re-accrues from a fresh epoch after:
`research/darktrain-20260807/raw/valley-signal.log`.

**The lane class sits BELOW every serving lane.** Harvest is still a *request* class the
engine admits, schedules, and sheds; a background job is not a request at all — it runs
only while the engine has NOTHING (no interactive, no judge, no harvest, no queue, no
pending admits) and yields on the first sign of any of them. The hysteresis is asymmetric
on purpose: yield fires on the busy EDGE with no debounce (paying traffic never waits for
a threshold), resume waits a full `MEMRA_VALLEY_S` (default 2 s) of quiet, because a
between-requests gap in a live conversation is not a valley.

**Yield mechanism v1 — simplest honest first.** The job (`MEMRA_BG_JOB`, arbitrary
command) is a child process in its own process group; yield is SIGSTOP to the group,
resume SIGCONT. The bound is the poll interval (`MEMRA_BG_POLL_MS`, 25 ms) plus signal
delivery — **measured 19.4 ms median / 23.3 ms max** request-fired-to-job-stopped (N=5,
one per rep, i.e. one poll interval; target <500 ms). Serve-impact stress (N=5 interleaved
reps, fresh boot per rep per arm, c=8×16 streaming bursts vs an 8-spinner CPU job, 5090):
burst p95 delta **+0.77%**, TTFT p95 **+1.11%**, agg tok/s **−0.54%** — under the 2% bar
(`research/darktrain-20260807/raw/bgstress-n5.log`). Two operator truths: a SIGSTOPped
process KEEPS its memory (VRAM included — the budget is carved out for the life of the
job, not per valley), and the runner cleans up on drain (CONT→TERM→KILL past grace) while
PDEATHSIG covers the SIGKILL path, so no orphan ever stays frozen.

**GPU memory discipline** is fits-or-refused at launch: `MEMRA_BG_VRAM_MB` (default 0 =
CPU-only) is granted only while `min free across visible GPUs >= budget +
MEMRA_MOE_RESIDENT_HEADROOM_GB` — min, not sum, because on a PP-2 pair both cards carry
serve shards. Unreadable `nvidia-smi` = refusal (fail closed); a refused job retries next
valley (headroom moves as sessions retire). v1 enforces fit at launch; staying inside the
budget at runtime is the job's contract, and the VRAM-aware admission gate defends serving
against a job that lies the same way it defends against everything else.

**Checkpoint/resume — the training-class seam** (`MEMRA_BG_YIELD_MODE=checkpoint`), for
jobs whose stopped working set must not squat on VRAM. The "checkpoint callback" is
process-level: SIGUSR1 to the group means *checkpoint now and exit 75* (EX_TEMPFAIL); the
runner relaunches the same command next valley and the job resumes from its own file.
Exit 0 = complete, never relaunched; any other exit = failed, loud, never relaunched. A
job that outlives `MEMRA_BG_CKPT_GRACE_MS` (5 s) after SIGUSR1 is SIGKILLed — the yield
bound holds even against a wedged job, and semantics are at-least-once (a training step
may repeat, never be lost; checkpoint writes must be atomic — write-tmp-then-rename).
Write single-command jobs as `MEMRA_BG_JOB="exec python3 train.py ..."`: the command runs
under `sh -c`, and without `exec` the shell parent dies of the unhandled SIGUSR1 before
the job's exit 75 can propagate. The live-server receipt caught exactly this
(`raw/ckpt-serve.log`: "job exited None during preemption") — the runner's
during-preemption branch classifies ANY exit after SIGUSR1 as checkpointed-and-relaunch,
so the cycle still resumed from step 129 correctly, but `exec` is what makes the
protocol exit visible.
Toy proof: `tools/bg-ckpt-counter.py` (counter checkpoints on SIGUSR1, exits 75, resumes
from the file; the unit test `checkpoint_mode_preempts_and_resumes_counter` pins the whole
cycle GPU-free). An in-process trainer API can replace this seam later without touching
the valley/scheduler half.

Observability: `/metrics` gains a `bg` block only when `MEMRA_BG_JOB` is set (state,
launches/yields/resumes/preempts, ckpt_kills, last yield-signal micros, job pid, budget)
— unset deployments see the pre-lane payload byte-identical.

## Knobs

Serving flags (batch cap, device sampling, lean logits, prime batching, spec burst) are
cataloged in [FLAGS.md §7](FLAGS.md) under "Serving (memra-server)"; fleet topology knobs
(`GPUS`, `REPLICAS_PER_GPU`, `CAP`, ports, health cadence) are env-overridable at the top of
`tools/serve-fleet.sh`. The exactness contract holds under batching: `decode-batch-gate` runs
in `tools/local-ci.sh` in both modes — `--mode config --batch 8` (the default-env battery,
fused tier live in the reference) and `--mode strict --batch 4` under the equalized composition
(`MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1`) — and ran twice inside `tools/validate-h100.sh` before
that battery was retired with the Hopper lane (2026-09-02). Each invocation runs gate1 (B=1 vs
`decode_step_h`),
gate2 (per-seq isolation — batchmates must not change your stream), and gate3, whose three
sub-checks are (a) device-argmax == host-argmax of the same row, (b) sampled draws at B=N ==
the same metas at B=1, and (c) `gate3c`, lean-vs-full logits identity. **`gate3c` is a
sub-check of gate3, not a fourth gate** — gate3 prints one PASS/FAIL line covering all three,
so a green line is the only signal that (c) ran; the sub-check names surface in the output only
when one fails. The stage-split modes (`--mode pp`, `--mode ppspec`) SKIP gate1/2/3 by design —
they are single-device jurisdiction — and neither PP mode was ever wired into `validate-h100.sh`; PP
exactness has its own invocations (see [TESTING.md](TESTING.md)).

## First-token cross-config drift (batched prime) — stated honestly

Serving primes prompts BATCHED (`prime_cache`, prefill GEMMs) while the historical oracle
stream is tokenwise (`decode_step`, m=1). These are different numeric configs by design —
same law as forward-vs-decode and the decode-batch gate's config mode — so on near-tie
prompts the FIRST generated token of a request can differ from the tokenwise oracle
stream, and everything after it follows the new prefix. Measured on the six-model
2026-08-02 sweep (`research/prime-gate-coverage-20260802/`, 144 prompts): **10/144 first
tokens flip (~7%)**, every flip at a tokenwise top1-top2 margin <= 0.70, batched prime
bit-deterministic, no content leakage across chunk boundaries, and forward_last sides
with the batched prime in 8/10 flips — the tokenwise config is usually the outlier, so
this is config roulette on a near-tie, not a wrong path. On the gemma prefill lanes the
config can even move per PROCESS (cuBLASLt heuristic algo selection; one observed
instance in the 144-row double pass, bit-deterministic within a process). Dense Q8_0
models (9B judge, Ornith-9B — the fleet class) flipped 0/48. Consequences can be visible (the Qwen3.6-35B
pp512 probe greedy-emits `"\n"` + EOS at 2 tokens where the tokenwise stream writes 128):
within contract, but real. `MEMRA_PRIME_TOKENWISE=1` pins the oracle stream at prefill
cost; the run-gen `batched-prime` gate line + the `prime-gate` battery bound the class
(structured divergence fails hard, near-tie flips are reported).

### Chunked prefill is split-stable — since the grain-free fix (found 2026-08-05; mechanism corrected and FIXED same day)

The class below is **history**: since lane/chunkinv-flip, chunked prefill is bit-identical
across `MEMRA_PRIME_CHUNK` values by default (see "FIXED BY DEFAULT" further down). What
follows is the finding and root cause, kept because the mechanism correction is the
evidence for the fix. A sharper statement of the same class, found while building
serve-smoke check 10: **changing only the prefill chunk split changed greedy output.**
Arms were the same four recorded prompts with a per-turn `cache_salt` (so nothing
resumes — every request primes cold), `MEMRA_AFFINITY=0`, varying only `MEMRA_PRIME_CHUNK`:

| prompt tokens | 2048 vs 64 | 2048 vs 32 |
|---|---|---|
| 48 | identical | identical |
| 97 | identical | **differs @ char 45** |
| 149 | **differs @ char 172** | **differs @ char 52** |
| 195 | identical | identical |

No reuse required, and 149 tokens is far too short for a long-window explanation. Every
resume tier inherits this by construction: a resume primes `[rewind boundary .. end]` as
its own chunk sequence rather than one full prime.

**Mechanism — corrected 2026-08-05 by `lane/chunk-invariance`.** This section originally
said "a different split changes the reduction order in the prefill GEMMs." **That is
measurably wrong.** The prefill GEMM is m-INVARIANT: feeding the same activation rows at
m=32 and at m=33..80 leaves rows `[0,32)` BIT-IDENTICAL for both the quantized `wq` and the
`output` head, so growing a batch does not move an existing row's value. And the divergence
is not a distributed last-bit band — it is a **step at the first chunk boundary**: per-row
maxdiff is exactly `0.000e0` for every row before the boundary and O(1) (6.9) immediately
after it, with `first_div_pos` equal to the chunk size exactly in every arm.

The real cause is a numeric-**class** edge, in `full_attn_prime_fa_dispatch`
(`hybrid_forward.rs`), selected by `base_len == 0` — *"is this the first chunk?"*:

- chunk 0 → `fa_prefill` over this batch's **f32** K/V;
- every later chunk → `fa_prefill_view_ws` over the **q8_0/q5_1 quantized KV cache**.

So `MEMRA_PRIME_CHUNK` decides at which token position the prefill stops reading f32 K/V
and starts reading dequantized cache. Rows before that position are computed identically in
both configs (hence the bit-identity); rows after it carry q8_0/q5_1 quantization error, and
a near-tie argmax flips. Eliminated by measurement, not assumption: `MEMRA_PRIME_DEQW=0`
(the other quantized-cache FA kernel) diverges identically, and `MEMRA_GDN_CHUNKED=0`
(sequential GDN scan, no WY segmentation at all) still diverges — the GDN state carry is
**not** the cause.

**FIXED BY DEFAULT — grain-free (lane/chunkinv-flip, 2026-08-05).** The `base_len == 0`
f32 special case is gone: chunk 0 quantizes its K/V into the cache first and attends through
`fa_prefill_view_ws` exactly like every later chunk (quantize-then-attend). One numeric class
for every row means the chunk size cannot decide where a precision edge falls, so **chunked
prefill is byte-identical across `MEMRA_PRIME_CHUNK` values with no door and no grain knob**
(chunkinv gate, naked env, both pinned prompts EXACT at chunks 2048/64/32).
`MEMRA_PRIME_CHUNK` is again a pure memory/transient knob. Rollback seam:
`MEMRA_PRIME_F32CHUNK0=1` restores the legacy f32 first-chunk arithmetic (and is the gate
canary's injection). The interim `MEMRA_PRIME_INVARIANT`/`MEMRA_PRIME_GRAIN` pin-the-boundary
door was superseded by this fix and removed at v0.71 per the flags doctrine (the research
record keeps its history). History + root-cause receipts:
`research/chunk-invariance-20260805/VERDICT.md`; flip receipts:
`research/chunkinv-flip-20260805/`.

What this changes and what it does not:

1. Gates MAY now assert byte-equality between two prefills of the same prompt at different
   chunk boundaries — `tools/chunk-invariance-gate.sh` asserts exactly that as its default
   (`--expect-invariant`, no env). serve-smoke check 10's scoping note is retired with it.
2. The exactness CLASS of short (single-chunk) prompts changed at the flip: chunk 0 now
   reads quantized KV — the same arithmetic long prompts always had past the first boundary.
   Near-tie argmax flips vs the old f32-first-chunk output are the documented contract
   change (quantified teacher-forced in `research/chunkinv-flip-20260805/`), not a bug.

**Scope: this is a per-architecture property.** The fix above is a property of the
`full_attn_prime_fa_dispatch` path, and the gate runs on the shipped arches. A different
attention family can re-enter the class through its own door, and one did — twice, both
closed and both gated: the `step35` bring-up arch (Step-3.7-Flash) was **chunk-DEPENDENT
past its 512-token SWA window** via kernel *selection* (a chunk whose `t_kv` exceeded the
window took the f32 windowed floor while a chunk that fit took FA, so the FA rows formed a
prefix `P = c*floor(win/c)` and the verdict depended only on `P` — pinned by a
pre-registered 4/4 falsification battery incl. a one-token c=513-vs-512 verdict flip;
receipts [`research/step37-p2-20260806/`](../research/step37-p2-20260806/), commit `66a81371`).
**FIXED 2026-08-07 in two stages, both gated:**
(1) *within one `prime_cache` call* — the SWA arm keys on the request's `seq_end`, not the
chunk's `t_kv`, making `P` identically 0 at every chunk size; gate `chunkinv35` (+
`chunkinv35c` canary via `MEMRA_STEP35_SWA_TKV`), default measured +0.009%
([`research/step35-chunkfix-20260807/`](../research/step35-chunkfix-20260807/));
(2) *across calls* — serve splits a prompt over SEVERAL `prime_cache` calls (per-tick budgets,
dark lanes SLO-capped = load-dependent; plus the prefix-cache LCP split), so `prime_cache` now
carries `queued_after` and `seq_end = cache.pos + t + queued_after` is request-level whatever
the tick segmentation; gate `tickinv35` (+ `tickinv35c` canary via `MEMRA_PRIME_CALLLOCAL`,
whose `sp<L>` split arms also pin the off-grid-resume hole — vLLM #51113's second law)
([`research/tick-seg-20260807/`](../research/tick-seg-20260807/)).
The SECOND door opened when the SWA prefill moved from the f32 floor to the windowed hd128
FA stamp (lane/pp-prefill 2026-08-07, `MEMRA_STEP35_SWA_FA` seam): the FA kernel's
online-softmax tiles group keys relative to the **view start**, and the SWA view offset is
a chunk boundary — so an unaligned offset regrouped the same absolute keys into different
BK=32 tiles at different chunk sizes. Closed by aligning the view offset down to the tile
size (the ≤31 extra leading keys are fully masked for every query — a bitwise no-op in both
kernels, measured on the floor arm). `chunkinv35` caught the second door on its first
battery, and its canary (`MEMRA_STEP35_SWA_TKV=1`, restoring BOTH pre-fix halves) is
verified red-capable (`research/pp-prefill-20260807`, batteries 1-3).

The behavior is **gated in both directions**: fast-gate ids `chunkinv` / `chunkinvc`
(routed from the `hybrid_forward.rs` map row): the default arm asserts byte-identity naked;
the canary arm injects `MEMRA_PRIME_F32CHUNK0=1` and must break, proving the gate detects
the mechanism. Reproducers + raw rows:
`research/session-affinity-20260805/chunk-order-probe.py` and `chunk-order.jsonl` (12 rows =
3 chunk sizes x 4 prompts, each with its text; under two minutes on the 9B), plus the
engine-level root-cause arm `concat-prime-probe chunkinv` and
`research/chunk-invariance-20260805/`.

**The THIRD axis — cross-call splits on the HYBRID (the prime-grid law, 2026-08-21,
lane/spec-longctx).** `chunkinv` pins within-one-call chunking; `tickinv35` pins cross-call
splits on the SWA launch SKU; nothing pinned cross-call splits on the qwen hybrid, and the
class was live at serving context: identical 24k-token greedy requests produced THREE
distinct outputs keyed purely by which prime program ran (restored-24k vs cold-monolithic
vs restored-607 — GATES-SMOKE-20260821 B3, darklanes), and the plain arm's own
boundary-stop-vs-monolithic twin diverged on 13/16 turns at agent lengths with spec OFF
(the same smoke's B1, reattributed by FRSPEC-FIX §3.2 — it was never the dflash2 route's).
Measured law (`concat-prime-probe primepath`, NJ box, q38 27B trunk @ v0.99.0, 24k real
agentic prompt; receipts `research/multiturn-cache-20260821/LONGCTX-EXACTNESS-20260821.md`,
darklanes):

- a prompt primed as TWO `prime_cache` calls split at `L` is **bit-identical** to the
  monolithic prime **iff `L % gdn_chunk_size() == 0`** (aligned splits: rows_diff 0/23999,
  logits EXACT — even though the two programs' INTERNAL 2048-token chunk boundaries do not
  coincide, only the 32-grid matters);
- an off-grid split diverges from **exactly row `L`** onward (prefix rows bit-identical),
  and greedy flips land on near-tie margins (measured flip at the p0.0 margin of the
  reference stream's distribution, with the cross-arm delta at the contending ids above
  it — the argmax-margin-probe discriminator);
- under the sequential GDN scan (`MEMRA_GDN_CHUNKED=0`) every split is EXACT. Mechanism:
  the chunked WY GDN scan segments per prime call, so an off-grid call start shifts the
  fold grid and **materializes recurrent state at a point the monolithic program never
  computes**. Lawful FP behavior of the WY form — structurally unfixable at arbitrary
  split points, NOT a defect (the 2026-08-05 defect signature — an O(1) class edge the
  arithmetic crosses — is a different animal and stays gated).

What changed because of it: **serve rounds every boundary it CHOOSES down to the grid**
(`grid_align_boundary`, worker.rs — the plain/spec checkpoint stop, the prefix-cache
LCP-split capture, the message-boundary seed capture; `MEMRA_PRIME_GRID_ALIGN=0` is the
rollback seam), so the boundary-stopped prime, the checkpoint resume and the whole-entry
restore reproduce the cold monolithic bytes, at a cost of at most `gdn_chunk_size()-1`
re-primed suffix tokens per resume. Gated in both directions: `primegrid` / `primegridc`
(tools/prime-grid-gate.sh — aligned splits must be EXACT, off-grid divergence must be
CONFINED to the split row with near-tie-only flips; canary coarsens the grid with
`MEMRA_GDN_CHUNK=64` and must break). What this deliberately does NOT promise:
cross-prime-path byte-identity is **not a valid gate** where the paths are not
grid-equivalent — prompt-END seed entries (exact-repeat class, position unroundable by
design) still serve extension hits through an off-grid suffix prime, and
verbatim-extension continuation resumes keep decode-computed rows whose arithmetic a cold
prefill never reproduces (`primepath --hist` measures that arm: bounded logit
perturbation, flips only at near-ties). Those paths carry the documented
cached-hit-vs-fresh-prime near-tie contract above; the valid assertions everywhere are
per-program determinism, grid-aligned byte-identity, and confinement + near-tie-only
flips across programs.


## Serving-contract guarantees, and the receipt for each

Not a changelog — the changelog is generated from conventional commits into
[releases](https://github.com/avifenesh/memra/releases) by `tools/changelog.sh`, and "what
changed recently" belongs there and nowhere else. This is the standing list of what the serving
contract promises and which measurement proves it, written in the present tense because a
guarantee is not news. The receipts below are the only doc pointers those directories have.

- Serving fails closed on exposure: a non-loopback bind without a configured key source refuses
  to boot, `/metrics` and `/yield/metrics` require bearer auth whenever keys are configured or
  the bind is public, API-key comparisons are constant-time, and keyring rewrites are atomic
  ([receipt](../research/sec1-20260811/RESULTS.md)).
- The GGUF parser rejects truncated and malformed files with contextual errors instead of
  panicking the GPU worker — every byte-prefix of a valid file is covered by fixture tests
  ([receipt](../research/ggufhard-20260811/RESULTS.md)).
- The GPU Gumbel sampler keeps uniforms below 1.0, so no rare `+inf` winner injects an
  arbitrary token during a long sampled generation. The 262,144-token serving target remains in
  place ([receipt](../research/longdepth-20260809/RESULTS.md)).
- Explicit `max_tokens` is enforced exactly: token events, usage and scheduler accounting
  follow visible token IDs rather than speculative rounds
  ([receipt](../research/honesty-20260809/RESULTS.md)).
- Plain-decode session affinity checkpoints and resumes rewritten conversation histories,
  so later turns prime the new suffix instead of an ever-growing history
  ([receipt](../research/affinity-20260809/RESULTS.md)).
- New admissions can end an in-flight speculative burst at its next round boundary instead of
  waiting for the whole burst; request-sized KV allocation remains VRAM-aware
  ([receipt](../research/admission-20260806/VERDICT.md)).
- Cache boundaries were hardened: release builds retire prefix pins and client cache salts are
  bounded and validated ([receipt](../research/pinfix-20260809/RESULTS.md)).
- `/v1/models` advertises the parameters each loaded model actually supports
  ([gateway contract](#gateway-listing-surface)), and as of v0.86.3 renders
  pricing from the same `MEMRA_MODEL_METADATA` entry the request ledger bills from
  (cache prices under `input_cache_read`/`input_cache_write`) plus the declared input
  modalities in the modality string (`text+image+video->text`) — previously a priced,
  vision-serving endpoint reported a hardcoded `"0"` text-only stub.
- v0.87.0 adds the router-marketplace provider contract (v2) to the same catalog:
  `contract_version`, a `[provider]` metadata block (status URL, contacts, regions) with the
  server-truth error contract (429/503 + `Retry-After`, `insufficient_balance` quota code,
  `x-request-id` echo), per-model lifecycle and reliability-timeout blocks, capability
  booleans, and per-1M-token decimal-string prices computed by exact decimal shift from the
  billed per-token metadata.
- Step-3.7-Flash PP-2 decodes in one numeric class at every batch width: a greedy request
  returns the same bytes whether it decodes solo, joins a batch mid-generation, or starts
  batched. The transition matrix that proved and closed the load-history divergence is the
  receipt ([isolation](../research/p0iso-20260810/RESULTS.md),
  [fix](../research/b1fix-20260810/RESULTS.md)).
- Grouped expert prefill (`MEMRA_MOE_GROUPED=1`) looked promotable on two-card Step serving in August
  2026 — 4k streaming TTFT 10.96 s -> 7.26 s, N=5 per arm
  ([receipt](../research/grouped-serve-20260810/RESULTS.md)) — but a re-sweep on an RTX PRO 6000 pair
  **withdrew it**: the arm is slower there (2,687.5 vs 8,193.7 tok/s resident-KAT transfer, 0/5 paired
  wins) and it fails the Q35 mixed c=4 exact-token gate, truncating every request in the cell
  ([receipt](../research/groupedregate-20260813/RESULTS.md)). It stays off by default and should not be
  enabled; see `MEMRA_MOE_GROUPED` in [docs/FLAGS.md](FLAGS.md).
- An opt-in SWA ring (`MEMRA_SWA_RING=1`) right-sizes sliding-window KV for Step sessions:
  the 262k session KV component drops 3.6x — measured 2 -> 12 concurrent 262k sessions
  before the first defer; lapped checkpoints decline safely
  ([receipt](../research/kv256-20260809/RESULTS.md),
  [flag-on validation](../research/ringval-20260810/RESULTS.md)).
- A B=1 decode specialization for Step models recovers most of the one-class contract's
  cost: +4.4% sustained c=1 decode with byte-identical output (the batched walk was issuing
  90 arithmetic-free device copies per token at B=1)
  ([receipt](../research/eagerpar-20260810/RESULTS.md)).
- A host-staged PP boundary fallback (`MEMRA_PP_HOST_BOUNCE=1`, default off) serves
  byte-correct on hosts whose GPU peer-copy path reports success but does not preserve
  bytes — a failure mode a capability flag or bandwidth test does not reveal. Peer paths
  should be byte-probed at provisioning
  ([receipt](../research/hostbounce-20260810/RESULTS.md)).
- Graceful shutdown joins the GPU worker thread before exit, closing a restart race where
  a new server could boot while the old worker still held device state.
