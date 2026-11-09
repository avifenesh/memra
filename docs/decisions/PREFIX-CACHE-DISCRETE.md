# Prefix cache on discrete cards: the budget leaves the session peak (2026-11-07)

**Status:** landed as the discrete-card default, measured on one RTX PRO 6000 Max-Q (2026-11-08: at
`MEMRA_MAX_SESSIONS=12` the budget is 54,277,046,272 B, the mixed shape's repeats hit 30/30 against 0/30 and the
agentic cells cache 99.5% of prompt tokens against 16.8-19.8%, with no OOM;
`research/prefix-budget-20261107/pro6000-maxq/`). Those cells never filled the cache: it peaked at 20.5 GB (mixed) and
11.2 GB (agentic) of the 54.3 GB budget, with no reclaim, so their free-memory floors say nothing about headroom. The
at-the-limit cell (`research/prefix-budget-20261107/pro6000-maxq/limit/`) fills the budget and the parked pool, then
loads the interactive cap: one arrival defers and the reclaim flushes the whole cache (57 entries, 54.2 GB), with no
OOM, pressure TTFT at most about 0.15 s behind main at every rank and the same repeat TTFT. At the limit the change keeps nothing over
main, and loses nothing against it (Owed, below). On the 24 GB RTX 5090 Laptop
the rule gives the old two-entry budget in every measured configuration, so it changes nothing there; the cards it
changes are the large ones, and the finding that opened the lane came from one of them (one RTX PRO 6000 Max-Q,
below). The eviction order stays plain LRU ([PREFIX-CACHE-POLICY.md](PREFIX-CACHE-POLICY.md)) and on the floor the
admission reclaim stays the flush: the lane measured a value-per-byte order, a partial reclaim and a live-free insert
clamp on the 5090 and kept none of them (Rejected, below). Above the floor the reclaim is partial since 2026-11-09
((c), below). Receipts: `research/prefix-budget-20261107/`; the full lane,
including the rejected levers' code, is banked on `bank/prefix-budget-gds-20261107`.

## Question

One RTX PRO 6000 Max-Q (96 GB), Qwen3.8-27B NVFP4 + Q5_K with its MTP head, served context 32,768 tokens, the
spec-cache-mixed gate (one greedy warm request with a 520-token shared prompt, then c8 unique-prompt load with six
greedy repeats of the shared prompt): `cached=[0, 0, 0, 0, 0, 0]`, 105 eviction lines.

- The discrete default was two full-context entries, 2,259,943,424 B, beside 86,614,081,536 B of boot free memory.
- Every entry carries its endpoint's GDN recurrent state, 156.9 MB on this model against 29.7 KB per token. A one-off
  request's 64-token seed is 158.8 MB; the shared 480-token spec-boundary entry is 172.1 MB.
- 14 seeds fill the budget. Under c8 load an entry lived about 4.8 s, and plain LRU evicted the shared entry before
  the first repeat arrived.

Two questions: (a) the discrete budget, and (b) what a full cache gives up first, in particular whether one-off
entries should be admitted on the same terms as shared ones.

## Decided

**(a) Budget: boot free less an estimate of the session peak.** The derived request on a discrete card is the larger of the
two-entry floor and boot free less the 1.5 GiB transient reserve, the admission driver headroom
(`MEMRA_ADMIT_DRIVER_HEADROOM_MB`, 2 GiB) and the session peak: every lane's resident-session cap (the interactive cap,
64 when batching, plus the judge and harvest caps) times one served-context session (the spec-path context planes at
the served context plus the recurrent state). No share is assumed: the cache gets the memory a full house of
served-context sessions would not use. That peak is an estimate, not a bound: it leaves out a request past
`MEMRA_CTX` on a naked boot (the bounded arm caps at the checkpoint context), the parked reuse-pool sessions (up to
`MEMRA_REUSE_POOL_GLOBAL_CAP`, 16), priming workspaces when memory-shaped admission is off (every Max-Q), a calibrated
admission floor above the 1.5 GiB reserve, the learned residual and a drafter's plane. Sessions stay first through
the machinery that already guarded the two-entry cache: the admission reclaim (the reclaim-on-defer flush), the
eviction and retry when a session allocation fails, the post-insert yield before decode and the step-OOM park. On the finding's box
with `MEMRA_MAX_SESSIONS=12` the request is 54,277,046,272 B (24 resident sessions x 1,190,789,120 B); at the default
caps, on the 24 GB RTX 5090, or when no model's context resolves, it is the floor (unit test
`discrete_prefix_budget_leaves_the_session_peak_and_keeps_the_two_entry_floor`). The budget is now read after the
vision towers load. With the finding's budget the policy simulator holds the mixed shape's shared entry through
every repeat under plain LRU (6/6) and every turn of the agentic replay (29/29).

**(b) Victim order and admission: unchanged** (plain LRU, every capture admitted, the reclaim flush). Each candidate
won one shape and lost or tied another on the 5090; the owner's rule keeps only levers that measure positive.

**(c) Admission reclaim above the floor: partial (2026-11-09, `research/partial-reclaim-20261109/`).** With the memory
door off, on a discrete card whose derived budget is above its two-entry floor and a single primary reading, a
deferred arrival frees its unbooked shortfall, the entry it will itself publish and what the priming sessions still
owe (prime transients and their uncapped pending seeds), oldest entry first, and the rest of the cache stays. Still
short after the settle, the rest of the unleased cache goes as before. A pass that freed bytes arms a booking: the
door-off gate books what the priming sessions owe (the transients, and the seeds capped at the cache's room) until
nothing is owed, so a later arrival of the same burst cannot pass on those bytes. The door's demoting flush, unified
memory, a per-device reading, an operator budget and the floor (every measured RTX 5090 configuration, the Max-Q at
the default caps) keep the flush. Measured on one RTX PRO 6000 Max-Q (2026-11-09, the at-the-limit cell, A B B A
against #1090's head): no flush, six partial passes per B arm freeing 18.2 GB of the 54.2 GB, the 16 repeats hit
16/16 (TTFT p50 116 ms against 10.46 s), no OOM line, every greedy row identical, pressure TTFT within 0.73 s of A
at every rank (seven of eight ranks faster); the mixed shape never reached the reclaim and its counts are identical.

## Measured

One RTX 5090 Laptop GPU (24 GB), Qwen3.8-27B NVFP4 + Q5_K with its MTP head, receipts in
`research/prefix-budget-20261107/rtx5090/`. On this card (a) is the floor, so the cells compare the (b) candidates.
Arms: base = main; gds = GreedyDual-Size over prompt tokens per byte (priority = clock + tokens per MB, the victim the
lowest priority, the clock takes its priority, every insert admitted) with a partial admission reclaim (evict in victim
order only until the arrival's shortfall, plus what priming sessions owe, is covered); lru = the partial reclaim alone;
flush = GreedyDual-Size with main's flush.

The mixed shape (`phase1a-final/`, three runs per arm, ABCD DCBA ABCD, `MEMRA_MAX_SESSIONS=12`):

| arm | repeats hit (each run) | repeat TTFT median ms | batch tok/s | batch TTFT p50 ms | reclaims |
|---|---|---|---|---|---|
| base | 3, 3, 3 | 656 | 96.0 | 1452 | 6 |
| lru (partial reclaim) | 2, 2, 3 | 689 | 98.0 | 1347 | 15 |
| gds (value per byte + partial reclaim) | 6, 6, 6 | 355 | 96.6 | 1456 | 11 |
| flush (value per byte + flush) | 4, 4, 5 | 431 | 95.3 | 1545 | 6 |
| off (one run) | 0 | 672 | 110.7 | 690 | 0 |

The agentic cell (`phase1b/`, `phase1c/`: 29 turns of 18 coding-agent sessions, prime pass then the measured pass,
T=1, 768 new tokens, two runs per arm AB BA):

| cell | arm | measured hits | cached fraction | TTFT p50 ms | TTFT p95 ms | agg tok/s |
|---|---|---|---|---|---|---|
| c4 | base | 14, 10 | 0.164 | 3964 | 19789 | 42.41 |
| c4 | gds | 13, 14 | 0.165 | 4013 | 18556 | 41.61 |
| c1 | base | 24, 24 | 0.194 | 3263 | 18569 | 26.95 |
| c1 | gds | 16, 16 | 0.182 | 3310 | 18464 | 26.96 |

Greedy texts matched the cache-off run on every compared row of every mixed-shape arm.

## Rejected

- **GreedyDual-Size value per byte.** It kept the mixed shape's shared entry through every repeat (6, 6, 6 against
  3, 3, 3; repeat TTFT 355 against 656 ms) and replayed the 2026-09-21 incident shape token for token like LRU
  (`phase1d/`, 10/10 ties at 29,600 computed tokens, 28/28 digests). It lost the agentic c1 cell: the coding-agent
  sessions share a 256-token spec-boundary prefix (165 MB, almost all recurrent state) that every new session hits,
  and its density (1.6 tokens per MB against about 25 for a 13k-token turn) put it first in line, so it was evicted
  between hits. At c1 both pairs read 16 against 24 hits, 33,568 against 35,616 cached tokens (4 against 12 hits on the
  256-token head), TTFT p50 +1.4%. The policy simulator reproduces the mechanism with a shared 256-token head (`sim/`: LRU
  58,752 cached tokens, value per byte 52,608; value per byte times hit count also 52,608, and it loses the incident
  shape). A follow-up that values an entry by the prompts that have shared it, not by its own length alone, starts
  from these receipts.
- **A partial admission reclaim.** With plain LRU it was flat to slightly worse on the mixed shape (2, 2, 3 hits
  against 3, 3, 3; batch TTFT 1347 against 1452 ms) and identical on agentic c4 (36-37 reclaims, 61-62 entries either
  way); it helped only beside value per byte. The flush it would replace grows with the budget: in the PRO 6000
  at-the-limit cell one deferred arrival (admission charged it 3.4 GB: KV, a 2.3 GB prefill workspace and fixed state;
  the shortfall was under 1 GB) flushed 54.2 GB, and the 16 repeats that followed
  all re-primed. That is where a partial reclaim earns its place: on the floor it stays rejected, above it it
  is (c).
- **GB10's live-free insert clamp on discrete cards.** Three runs per arm on the mixed shape (`phase1a-clamp/`): the
  shared entry hit 0, 0, 0 (LRU) and 1, 1, 1 (value per byte) of six repeats, against 6, 6, 6 without the clamp and
  2, 3, 3 on main. Under c8 load it held the cache to about one entry, and since an insert never evicts itself every
  new seed displaced the shared entry.
- **Second-hit admission** (insert an entry's state only when its prefix family is seen again). It delays every first
  reuse by one request, and a conversation's turn is reused exactly once, by the next turn. On the policy simulator
  with the real entry geometry it computed 45,376 tokens on the incident replay against LRU's 28,896 (+57%) and lost
  25,024 cached tokens of the agentic prime-then-measure cell; on the mixed shape it got 5 of 6 repeats.
- **A separate budget for recurrent state** (the SGLang shape: a Mamba state pool sized by
  `mamba_full_memory_ratio`, default 0.9, with its own LRU). On this engine a hybrid entry restores only with its
  endpoint's recurrent state, so a separate state budget either evicts whole entries (an entry-count cap) or strands
  KV bytes that serve nothing, and it needs a split ratio with no measurement behind it.
- **A free-memory share for the discrete budget** (GB10's half). No measurement says what share a discrete card needs;
  the session peak comes from the scheduler's own caps.

## Owed

- Measured 2026-11-08 (one RTX PRO 6000 Max-Q, `research/prefix-budget-20261107/pro6000-maxq/`): the
  spec-cache-mixed shape and the agentic c1 and c4 cells with (a), labelled as loaded-box timings (no quiet-box
  re-time is planned). Still owed: the default caps, where a Max-Q keeps the two-entry floor and still missed 0/18
  repeats.
- A partial admission reclaim at a large budget: (c), measured on the at-the-limit cell in
  `research/partial-reclaim-20261109/`. Still owed there: an RTX 5090 boot that derives a budget above the floor,
  and other burst shapes at the limit.
- GB10 keeps its own budget rule (the half share). One input moves: the budget is now read after the vision towers
  load, so a boot with a tower reads less free memory and gets a smaller half share. The sm_121a cell record of the
  landing commit covers a GB10 boot without a tower; no GB10 boot with a tower was measured.
