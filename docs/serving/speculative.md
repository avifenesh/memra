# Speculative decoding in the server

How `memra-server` attaches drafters, decides which requests speculate and at what depth K, hands
speculating sessions back to batched decode under load, and paces a speculative stream. It is for
an operator choosing a drafter or reading the `[spec-gate]`, `[spec-k]` and `[spec-cost]` boot and
request lines. The engine side is in [ARCHITECTURE.md](../../ARCHITECTURE.md#8-speculative-decoding),
the drafter build recipe in [DRAFT-REGIME.md](../DRAFT-REGIME.md), flag cells in
[FLAGS.md](../FLAGS.md) and the counters in [metrics.md](metrics.md#speculative-decoding).

Greedy speculation is exact: the stream equals plain greedy decode token for token, whatever K,
pooling or demotion did.

## Drafter attach

| Drafter | Attach | Notes |
|---|---|---|
| MTP / NextN head in the GGUF | automatic | |
| A model's own MTP draft GGUF | `MEMRA_MODELS="name=/model.gguf+/draft.gguf"` | The draft must be a file. A missing or non-file draft is a boot FATAL. `hf:owner/repo[:file]` works for both parts |
| A process-wide MTP draft | `MEMRA_MTP_DRAFT=<draft.gguf>` | Replaces the MTP head of every model |
| FR-Spec trimmed draft head | `MEMRA_FRSPEC_TRIM=<ranks.txt \| frspec.gguf \| hf:...>` | Ranks from the model's own generations ([DRAFT-REGIME.md](../DRAFT-REGIME.md)); detected through the head's `d2t` map |
| DSpark (DFlash2) block drafter | `MEMRA_DSPARK_SPEC=1` + `MEMRA_DSPARK_DRAFT=<export_dir \| hf:...>` | GDN-hybrid trunks. Disables MTP spec for that model. Refuses to boot with `MEMRA_SPEC_DFLASH`, `MEMRA_GEMMA4_SPEC`, PP stages > 1, Step TP/EP or GLM-5 TP. A load failure panics |
| Gemma 4 assistant drafter | `MEMRA_DRAFT=<assistant.gguf>` | `MEMRA_GEMMA4_SPEC` defaults to K=5 when `MEMRA_DRAFT` is set; `0` turns it off. Dense gemma4 models other than E4B. A request speculates only when no other session is active; sampled requests without penalties are admitted (`MEMRA_GEMMA_SPEC_SAMPLED`, default on), constrained and penalized ones are not |
| GLM-5 MTP block | `MEMRA_GLM5_MTP=1` + `MEMRA_GLM5_SPEC=1` | Off by default. GLM-5 TP serving refuses every spec program and needs `MEMRA_SERVE_SPEC=0` |

An attach is a log line, not the absence of an error: `[mtp-draft] loading external MTP draft:
<file>` or `regime draft attached (...)`. `tools/assert-drafter-attached.sh <server.log>` checks
for it, because a trunk with an embedded head speculates without the file. A model whose pack
declares an external MTP head and gets none serves plain and logs `[worker] WARN: ... no MTP
drafter attached`. `MEMRA_DRAFT` is the gemma attach only: on another model it attaches nothing,
and it still changes engine defaults that key on it (its row in [FLAGS.md](../FLAGS.md)).

```bash
MEMRA_MODELS="main=/path/to/model.gguf+/path/to/mtp-draft.gguf" memra-server
```

## Admission gate

`MEMRA_SERVE_SPEC` (default on; `0` off) allows the spec path. A request speculates only when a
drafter is attached, the request is eligible, and the gate admits it. Speculation is off under a
PP host bounce. Two deciders share the gate.

**The concurrency band.** Single card, and every placement except sharded PP-2: admit while the
projected wave is at most `LOW` (2), and demote live speculating sessions at `active >= HIGH` (4).
The wave counts live sessions, this request, the queue and the requests still in flight between
the HTTP layer and the worker, so the head of a burst of arrivals does not speculate while the
rest of it queues. Between the two thresholds a live session keeps speculating and a new arrival
serves batched. `MEMRA_SPEC_GATE_LOW` and `_HIGH` override the defaults; `HIGH <= LOW` is
clamped to `LOW + 1` with a warning. `MEMRA_SPEC_GATE=0` is always-spec on every placement.

**The cost model** (`MEMRA_SPEC_GATE_MODEL`, default on). At boot every MTP-served model measures a
cost table on the device, logged as `[spec-cost] model=... verify_ms[1..16]=[...] plain_ms[1..]=[...]`:
verify forwards over 1 to 16 rows, plain batched steps up to the decode wave cap, and the
per-session round overhead. On the RTX PRO 6000 Blackwell Max-Q the overhead is fitted per draft
step, so each draft length pays its own drafting
([`research/spec-cost-draft-step-20261109/`](../../research/spec-cost-draft-step-20261109/)); every
other device charges one K=3 burst's cost to every draft length. Once the model has 64 rounds of
per-position acceptance, admission and demotion take the K in `0..=rung` (0 = plain) that
maximizes predicted aggregate tok/s at the projected concurrency:

- realized tick rates replace a prediction after 3 samples and scale the predictions near them;
- with no acceptance history yet, a calibrated model speculates at its rung when every session's
  verify rows fit one pooled forward (`n * (rung + 1) <= 16`), so the history fills;
- at 7 or more sessions with no measured speculative rate within one session of `n`, it serves
  plain rather than extrapolate;
- a tie within two standard errors goes to speculation when the speculative price is not backed by
  fresh samples, so a stale price gets re-measured
  ([`research/spec-cost-tie-break-20261110/`](../../research/spec-cost-tie-break-20261110/)).

Each verdict change logs `[spec-cost] model=... n=... rung=... -> K=... (predicted ... vs plain
... tok/s; ...)`. The band decides instead for an operator pin, the PP-2 default, explicit
`LOW`/`HIGH`, DSpark-served models, and models without a verdict. A model whose calibration does
not run logs `[spec-cost] calibration skipped: ... the band serves`; Qwen3.8-27B does not calibrate
on the 24 GB RTX 5090 Laptop, so the band and the K table decide there
([SPEC-K-COLD-SHORT-5090.md](../decisions/SPEC-K-COLD-SHORT-5090.md)). `MEMRA_SPEC_GATE_MODEL=0`
restores the band everywhere.

Plain requests pay the 1.5 GiB speculative transient reserve in admission only where some request
can speculate. Rationale and the measurements behind both deciders:
[SPEC-ADMISSION-GATE.md](../decisions/SPEC-ADMISSION-GATE.md),
[`research/batch-spec-20261101/COST-GATE.md`](../../research/batch-spec-20261101/COST-GATE.md).

## PP-2 default

On a sharded cross-device PP-2 placement (`MEMRA_PP_STAGES=2`, stages on two devices) the band
defaults to `LOW=0`, `HIGH=1`: no request speculates, plain requests skip the speculative reserve,
and the cost model does not decide. The boot line says so: `[spec-gate] policy
placement=pp2-cross-device LOW=0 HIGH=1 source=placement-default spec-admission=off`. Speculation
over PP-2 is correct (the verify takes its own stage split, `MEMRA_SPEC_PP`, default on) but slower
than batched decode at every measured concurrency: Qwen3.5-9B on two RTX PRO 6000 Blackwell
Server Edition cards measured 112.5 against 223.3 tok/s at c=1 and 112.1 against 593.4 at c=4
(2026-08-08, N=3; [`research/specplace-20260808/`](../../research/specplace-20260808/PROGRESS.md)).
`MEMRA_SPEC_GATE=0` forces it for tests. PP-3, PP-4 and same-device PP keep the single-card band;
they were not measured.

## K policy

Inside an admitted request K comes from a four-rung table, then the cost model may pick a lower K:

| Rung | Applies to | Default K | Override |
|---|---|---|---|
| cold-short | prompt under 1,024 tokens | 3 (6 on the GLM-5 route of a `100a` build) | `MEMRA_SPEC_K_COLD_SHORT` |
| cold-long | prompt of 1,024+ tokens, under 1,024 restored | 2 on the RTX 5090 Laptop GPU and the RTX PRO 6000 Blackwell Max-Q; 3 elsewhere, GB10 and the desktop RTX 5090 included | `MEMRA_SPEC_K_COLD_LONG` |
| cached-long | prompt and restored prefix both 1,024+ tokens, greedy requests only | 2 | `MEMRA_SPEC_K_CACHED_LONG` |
| cached-long, trimmed head | the same with an FR-Spec trimmed MTP head | 5 | `MEMRA_SPEC_K_CACHED_LONG_TRIM` |

A sampled request takes its cold rung whatever it restored
([`research/spec-k-resume-20261109/`](../../research/spec-k-resume-20261109/)). The cold-long rung
is keyed by device name: [SPEC-K-COLD-LONG-K2.md](../decisions/SPEC-K-COLD-LONG-K2.md). K=2 for
cold-short prompts on the RTX 5090 Laptop was measured and rejected:
[SPEC-K-COLD-SHORT-5090.md](../decisions/SPEC-K-COLD-SHORT-5090.md). The cached-long K=2 and the
trimmed-head K=5 come from [`research/kpolicy-20260808/`](../../research/kpolicy-20260808/) and
[`research/orndecode-20260822/`](../../research/orndecode-20260822/). A rung override must be a
positive integer; anything else warns and keeps the shipped rung.

`MEMRA_SPEC_K` pins K for every request, `0` meaning plain, and turns off the gate, the cost model
and demotion. An invalid pin warns and keeps the automatic policy. The boot prints the table
(`[spec-k] automatic table: ...`) and each request prints its choice:

```text
[spec-k] model="main" tenant="" K=3 source=cold-short prompt=812 cached=0 lcp=0 active=1 wave=1 placement=single-or-non-pp2
```

`source` is one of `operator-pin`, `oom-replay` (a replay after a step OOM keeps its K),
`pp2-placement`, `concurrency`, `cold-short`, `cold-long`, `cached-long`, `cost-model`,
`eligibility-fallback` (the request could not speculate) and `resume-floor(k=0->1)` (a resumed
spec session keeps at least K=1; the demotion sweep moves it to batched decode if load requires).

## How speculating sessions run

A speculating session runs a burst of rounds (`MEMRA_SPEC_BURST`, default 32 tokens) inside the
worker tick. Same-model MTP sessions on continuation bursts run as a pooled cohort: their bursts
advance in lockstep and each round's verifies share one target forward, chunked at 16 rows, with
the draft steps pooled too. Plain sessions of the same model can ride the cohort as one row per
round, byte-identical to their batched stream: device-sampled ones in any lane, host-sampled ones
(greedy with a penalty) in the interactive lane. A model with an interactive plain session that
cannot ride (constrained, vision, fork) takes no riders that tick; a judge or harvest one stays in
phase (c) and holds no rider back. DSpark sessions form their own
cohorts. Prime bursts, constrained sessions and the other routes step one session at a time.
Pooling needs a model whose verify can pool: an MTP head, the batched serving class, no PP cuts,
no sliding-gated MoE or gemma batch program, and the verify graph off. The verify graph is on by
default for GDN + routed-MoE models, so the 35B-A3B class speculates one session at a time.
Riders also need a dense FFN. Receipts:
[`research/batch-spec-20261101/`](../../research/batch-spec-20261101/).

## Demotion

At each tick the worker hands a live speculating session to batched decode when the band says
`active >= HIGH`, or when the cost model prices plain above every K at the live count. The
session's cache and next token move into a batched row; a carried pending token is committed first.
Demotion is one way: the session stays batched until it ends. The log line names the reason and the
sampling handoff:

```text
[spec-gate] demoted session to batched decode: 4 active >= HIGH=4 (model main, committed 913, generated 101, greedy)
```

Where the cost model owns the gate the reason reads `N active, cost model prefers plain (or
HIGH=4 without a model)`; a sampled session's clause reads `sampled, plain draws from philox event
N`.

- Greedy sessions demote on the MTP, DSpark and GLM-5 routes, byte-exact. Gemma assistant-drafter
  sessions do not demote; their solo admission bounds them.
- Sampled sessions demote on the MTP and DSpark routes. Plain draws continue at the spec session's
  next unused Philox event, so the text stays one sampled stream. On the MTP route that is the next
  position, and the demoted text is plain decode's.
- GLM-5 sessions demote greedy only.
- Constrained sessions never demote: their carried token is the unmasked verify result.

DSpark admission uses the band with its own rules: at most `LOW` DSpark rows live, sampled rows
included (`MEMRA_DSPARK_SAMPLED_WAVE`, default on). `MEMRA_SPEC_DEMOTE_AT` is a test door that
forces a demotion; it is never set in a deployment.

## Sampled speculation

On the MTP route a sampled round is verified by position (memra#1154): position `p` emits plain
decode's own draw from the target's row at `p`, one Philox event per position. A seeded request
writes the same text at every K, alone or beside other requests, cold or resumed, with or without a
demotion. Measured cost: about 2 points of acceptance per drafted token
([SAMPLED-VERIFY-POSITION-KEYED.md](../decisions/SAMPLED-VERIFY-POSITION-KEYED.md)). A sampled
burst carries its pending token into the next burst, as a greedy burst does
([SAMPLED-PENDING-CARRY.md](../decisions/SAMPLED-PENDING-CARRY.md)). The penalty window spans the
session, the last `max(penalty_last_n, 64)` tokens of `committed ++ prompt`, as plain decode's does
(`MEMRA_SPEC_PEN_SESSION`, default on).

Not covered by the position key:

- the DSpark, GLM-5 and gemma assistant-drafter routes, which keep the rejection walk: their text
  matches the target distribution but not plain decode's text;
- a penalized sampled request where the device penalty arm is off (`MEMRA_SERVE_DEVPENALTY`): one
  text across K, but not plain decode's;
- the rollback seams `MEMRA_SPEC_SAMPLED_BOUNDARY=0` (an argmax at each burst boundary),
  `MEMRA_SPEC_PEN_SESSION=0` (a burst-local penalty window) and `MEMRA_SPEC_REPLAY=1`.

Greedy requests with a penalty do not speculate on the MTP route. Speculation on prefix-cache
hits, the sampled-restore rule and the spec-pool sampler check are in
[prompt-cache.md](prompt-cache.md#speculation-on-cache-hits).

## Constrained speculation

Greedy requests with `response_format` speculate. The drafter is masked too: each round clones the
session's grammar matcher, advances the clone with each proposed token and bans illegal ids in the
draft head's logits (through `d2t` for a trimmed head), so proposals are legal by construction. The
verify-side truncation stays as the correctness backstop. Spec-constrained output is byte-identical
to plain-constrained.

These serve plain constrained decode instead:

- sampled constrained requests, and every request under the `MEMRA_CONSTRAIN_HOST=1` oracle;
- GDN + MoE trunks (the Qwen3.5/3.6 35B-A3B class), whose exact spec path replays bursts and cannot
  commit a grammar-masked bonus, and any model under `MEMRA_SPEC_REPLAY=1`;
- post-think sessions on think-forced templates (the `[spec-k]` line shows `K=0
  source=eligibility-fallback`).

Constrained sessions never join a pooled cohort, never demote and never resume a parked spec
session. Cost on Qwen3.5-9B NVFP4 + MTP, one RTX 5090, 256-token greedy, N=3 (2026-08-03 and
2026-08-04): constrained spec ran at 79% of unconstrained spec (153.4 against 194.4 tok/s), against
99.4% on the plain path; draft masking raised a tight bounded schema from 216.6 to 227.5 tok/s
([`research/constrained-full-20260803/`](../../research/constrained-full-20260803/),
[`research/draft-mask-20260804/`](../../research/draft-mask-20260804/)).

## Round cadence

- **Text per round.** The worker flushes text at every round commit, so a solo stream's first text
  and chunk gaps do not depend on the burst size: about 0.12 s to first text and a 27 ms p50 chunk
  gap on a 27B NVFP4 + MTP model, K=3, one RTX 5090 (2026-08-05, N=5;
  [`research/sse-cadence-20260805/`](../../research/sse-cadence-20260805/)). Content is
  byte-identical; only chunk boundaries move.
- **Admission yield** (`MEMRA_ADMIT_YIELD`, default on). A request waiting for admission ends the
  in-flight burst at the next round boundary, and sessions that have emitted nothing burst before
  mid-generation peers. Contended first text measured 0.123 s at B32 and 0.152 s at B128 against
  0.54 s and 1.60 s before; the cost is 3.4% of aggregate throughput at c=8 saturation
  ([`research/admission-20260806/`](../../research/admission-20260806/)). `0` selects lockstep
  fairness.
- **Entry-yield bound.** A session yields at a burst's entry with nothing emitted at most once in a
  row; at the next such entry it runs one round first. This keeps a long stream moving under a
  stream of short arrivals ([SPEC-ENTRY-YIELD-BOUND.md](../decisions/SPEC-ENTRY-YIELD-BOUND.md)).
- **Stall mark.** When a speculating session's last four burst entries each met a waiting request,
  the cost model marks that live count, and the waves its waiting requests are admitted into,
  plain at every draft length, so the session demotes and newcomers decode plain. A demoted
  session's plain steps that see a waiting request refresh the mark; it expires 64 s after the last
  refresh. Greedy text and seeded sampled text are unchanged
  ([SPEC-STALL-MARK.md](../decisions/SPEC-STALL-MARK.md)).
- **Newcomers prime before the pooled cohorts** (memra#1227, 2026-11-12). An MTP speculative
  session that has emitted nothing and has a same-model speculative peer on a model that pools (so
  its prime burst ends at one token) steps before the tick's pooled cohorts, and a session whose
  prime completed joins that tick's lockstep. Before, the cohorts stepped first, so a newcomer admitted
  at the tick top waited a whole lockstep burst (32 tokens per member) for its prime, and behind a
  single stream its second token waited that stream's burst. On GB10's Qwen3.8-27B the first token
  behind two or three long streams comes at 0.27 and 0.29 s where it came at 1.86 and 2.03 s, and
  closed-loop clients at c = 2..4 gain 9 to 11% aggregate. A long prime runs its first chunk there and
  then keeps the prime policy's place, with the cohort running one round between its chunks; a
  lone speculative session and the DFlash2, glm5 and gemma routes (a prime plus a whole first
  burst) keep their place after the cohorts; `MEMRA_ADMIT_YIELD=0` keeps the cohorts first. Session order and cohort membership are
  content-neutral, so texts are unchanged. A cohort under short arrivals now meets waiting
  requests at its burst entries, so the stall mark above applies to it as it does to one stream. A
  request admitted to plain decode is the next item's
  ([SPEC-COHORT-NEWCOMER.md](../decisions/SPEC-COHORT-NEWCOMER.md),
  [`research/spec-cohort-newcomer-20261112/`](../../research/spec-cohort-newcomer-20261112/)).
- **A plain session behind the bursts advances once per round** (memra#1244, 2026-11-12). A plain
  interactive request with a host sampler (greedy with a penalty) rides the pooled cohort like a
  device-sampled one, while the cost model keeps the streams speculative (on GB10's
  Qwen3.8-27B up to 4 riders beside 2 streams). While an interactive plain session gets its
  step only after the tick's
  bursts (constrained, vision, fork, a rider candidate no cohort carries, a prompt whose prime
  phase (b) runs this tick within one chunk), every MTP burst of the tick ends at its first
  round, in a tick with no DFlash2, glm5 or gemma burst. Judge and harvest sessions never engage
  it, never keep interactive riders in phase (c), and step once every N yielded ticks (N the
  rounds of a recent full burst; about half their burst cadence on GB10 while the yield holds).
  Before, such a session got one token per burst: on GB10 1.4 to 1.8 s between tokens beside one to three streams, 0.10 to 0.19 s now.
  The streams lose 1 to 5% beside a rider row in their forward and pay for the plain session's
  own forward while it is live otherwise (24 to 29% on GB10 beside a constrained request; 29 to
  41% on the RTX 5090, where the rider steps plain inside each round). Where the total falls,
  this is a trade of throughput for latency. Texts are unchanged; `MEMRA_ADMIT_YIELD=0` keeps
  full bursts. The DFlash2, glm5 and gemma bursts and a long prime's chunks are not covered
  (memra#1246, memra#1253) ([SPEC-ROUND-YIELD.md](../decisions/SPEC-ROUND-YIELD.md),
  [`research/spec-plain-round-yield-20261112/`](../../research/spec-plain-round-yield-20261112/)).
- **The burst ends with the stream.** The round that commits EOS or a stop id ends the burst, and
  so does a client that disconnected. Measured: up to 0.8 s less per ~100-token answer on GB10
  ([`research/gb10-decode-20261109/`](../../research/gb10-decode-20261109/)). Acceptance read after
  2026-11-09 is higher for the same text, so older acceptance figures do not compare.
- **A pooled member closes at its round.** A cohort member whose stream ended gets its usage and
  end of stream at that round, not when the whole cohort returns
  ([POOLED-MEMBER-CLOSE.md](../decisions/POOLED-MEMBER-CLOSE.md)).
- **A pooled verify fits the device** (memra#1266). Each member's verify holds its replay-free
  checkpoint (about 157 MB per draft column on Qwen3.8-27B) until its round finishes, so a pooled
  forward holds all of them at once. A member joins a forward only while the whole forward fits the
  allocatable memory (driver or unified-allocatable free plus the pool's cached blocks) less the
  256 MiB launch floor; the rest verify after it, bitwise the same tokens. Before the cohorts step,
  when that reading cannot fund the largest member's round, unleased prefix entries and then parked
  sessions are evicted. `/metrics` counts both (`spec_pool_capped_forwards_total`,
  `spec_cohort_reclaims_total`); the lines are `[spec-pool] pooled MTP verify capped by memory` and
  `[spec-pool] pre-cohort reclaim`
  ([POOLED-VERIFY-MEMORY-CAP.md](../decisions/POOLED-VERIFY-MEMORY-CAP.md)).

The burst size stays 32. `128` measured 8.4% more throughput at c=1 and trails 32 by one round
(about 29 ms) of contended first text; the trade is in the `MEMRA_SPEC_BURST` row of
[FLAGS.md](../FLAGS.md).

## Records and receipts

Decisions: [SPEC-ADMISSION-GATE](../decisions/SPEC-ADMISSION-GATE.md),
[SPEC-K-COLD-LONG-K2](../decisions/SPEC-K-COLD-LONG-K2.md),
[SPEC-K-COLD-SHORT-5090](../decisions/SPEC-K-COLD-SHORT-5090.md),
[SAMPLED-VERIFY-POSITION-KEYED](../decisions/SAMPLED-VERIFY-POSITION-KEYED.md),
[SAMPLED-PENDING-CARRY](../decisions/SAMPLED-PENDING-CARRY.md),
[SPEC-ENTRY-YIELD-BOUND](../decisions/SPEC-ENTRY-YIELD-BOUND.md),
[SPEC-COHORT-NEWCOMER](../decisions/SPEC-COHORT-NEWCOMER.md),
[POOLED-MEMBER-CLOSE](../decisions/POOLED-MEMBER-CLOSE.md),
[POOLED-VERIFY-MEMORY-CAP](../decisions/POOLED-VERIFY-MEMORY-CAP.md). Per-request `usage.spec` and the
acceptance counters: [metrics.md](metrics.md#speculative-decoding). The dated history is in the
[archived serving reference](../archive/SERVING-2026-11-12.md#streaming-cadence--admission-latency--the-felt-ttft-arc-lanes-sse-cadence-2026-08-05-admission-latency-2026-08-06).
