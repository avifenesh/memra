# Prefix cache on discrete cards: session-peak budget, value-per-byte victims, partial reclaim (2026-11-07)

**Status:** decided on the RTX 5090 Laptop (24 GB) and landed as the discrete-card default; the RTX PRO 6000
measurement is owed (the finding that opened the lane came from one RTX PRO 6000 Max-Q, below). Unified memory
(GB10) keeps its own budget rule, plain LRU and the reclaim flush until it is measured there. Receipts:
`research/prefix-budget-20261107/`. Supersedes, on discrete cards only, the eviction order of
[PREFIX-CACHE-POLICY.md](PREFIX-CACHE-POLICY.md) (plain LRU, 2026-09-21), which stays the unified-memory order and
the `MEMRA_PREFIX_EVICT=lru` rollback.

## Question

One RTX PRO 6000 Max-Q (96 GB), Qwen3.8-27B NVFP4 + Q5_K with its MTP head, served context 32,768 tokens, the
spec-cache-mixed gate (one greedy warm request with a 520-token shared prompt, then c8 unique-prompt load with six
greedy repeats of the shared prompt): `cached=[0, 0, 0, 0, 0, 0]`, 105 eviction lines.

- The discrete default was two full-context entries, 2,259,943,424 B, beside 86,614,081,536 B of boot free memory.
- Every entry carries its endpoint's GDN recurrent state, 156.9 MB on this model against 29.7 KB per token. A one-off
  request's 64-token seed is 158.8 MB; the shared 480-token spec-boundary entry is 172.1 MB.
- 14 seeds fill the budget. Under c8 load an entry lived about 4.8 s, and plain LRU evicted the shared entry before
  the first repeat arrived.

Two things to decide: (a) the discrete budget, and (b) what a full cache gives up first. Measuring (b) on the 5090
surfaced (c): the admission reclaim.

## Decided

**(a) Budget: what the scheduler can never need.** The derived request on a discrete card is the larger of the
two-entry floor and boot free less the 1.5 GiB transient reserve, the admission driver headroom
(`MEMRA_ADMIT_DRIVER_HEADROOM_MB`, 2 GiB) and the session peak: every lane's resident-session cap (the interactive cap,
64 when batching, plus the judge and harvest caps) times one served-context session (the spec-path context planes at
the served context plus the recurrent state). No share is assumed: the cache gets exactly the memory a full house of
served-context sessions cannot use. On the finding's box with `MEMRA_MAX_SESSIONS=12` that is 54,277,046,272 B
(24 resident sessions x 1,190,789,120 B); at the default caps, or on the 24 GB RTX 5090, the floor (unit test
`discrete_prefix_budget_leaves_the_session_peak_and_keeps_the_two_entry_floor`).

**(b) Victim order: GreedyDual-Size over prompt tokens per byte** (Cao and Irani, 1997). Each entry's priority is the
clock plus the tokens it saves per MB it holds, set on insert, on reuse and at the last lease release; the victim is the
lowest priority (ties oldest first) and the clock takes its priority. Every insert is admitted and never evicts itself.
Among entries of one density this is plain LRU.

**(c) Admission reclaim: free the shortfall, not the cache.** When admission is short, the reclaim evicts in victim
order until the arrival's unbooked shortfall plus what priming sessions still owe (their prefill workspaces and the
seeds they will publish) is covered, and empties the rest only if the re-read is still short.

## Measured

FILL

## Rejected

- **Second-hit admission** (insert an entry's state only when its prefix family is seen again). It delays every first
  reuse by one request, and a conversation's turn is reused exactly once, by the next turn. On the policy simulator
  with the real entry geometry (`research/prefix-budget-20261107/sim/policy-sim.txt`) it computed 45,376 tokens on
  the incident replay against LRU's 28,896 (+57%) and lost 25,024 cached tokens of the agentic prime-then-measure
  cell; on the mixed shape it got 5 of 6 repeats where value per byte got 6 of 6.
- **A separate budget for recurrent state** (the SGLang shape: a Mamba state pool sized by
  `mamba_full_memory_ratio`, default 0.9, with its own LRU). On this engine a hybrid entry restores only with its
  endpoint's recurrent state, so a separate state budget either evicts whole entries (an entry-count cap) or strands
  KV bytes that serve nothing; and it needs a split ratio with no measurement behind it.
- **GreedyDual-Size that refuses an insert it cannot rank above the residents.** Same as the shipped rule on the
  mixed and agentic shapes, but on the incident replay it refused six cohort returns and computed 30,528 tokens
  against LRU's 28,896: a refusal never advances the clock, so stale dense entries outlived fresh ones (the
  2026-09-21 losing shape). The shipped rule always admits and replays the incident token for token like LRU.
- **A free-memory share for the discrete budget** (GB10's half). No measurement says what share a discrete card needs;
  the session peak is a bound the scheduler enforces.
- **The cache taking every free byte above the admission floor**, sessions reclaiming as they arrive. It sizes the
  cache into memory sessions are entitled to; with the reclaim partial it would be safe, but the RTX PRO 6000 serving
  shapes it would change are not measured.

## Owed

- The RTX PRO 6000 run of the same cells (the finding's card class; no PRO 6000 was free during this lane).
- GB10: the value-per-byte order and the partial reclaim on unified memory.
