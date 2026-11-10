# A speculating session yields at a burst's entry at most once in a row (2026-11-10)

**Status:** adopted, memra#1201. Receipts:
[`research/spec-entry-yield-bound-20261110/`](../../research/spec-entry-yield-bound-20261110/).

## Question

A speculative burst ends early while a request waits for admission (the admission yield). The
verdict is read at every commit hook, including the first one, which runs before round 0. A greedy
burst that enters on its carried pending token has emitted nothing at that hook, so when a request
is waiting there the burst runs no round and emits nothing. On GB10 at the default gate this
stopped a long greedy stream outright: from two closed-loop clients of one-token requests with a
think time of 200 ms or less, the stream emitted 1 or 2 tokens in 30 s. It recovered only when the
arrivals stopped.

The trigger in the default configuration is the calibrated spec cost model (v0.139.0). It admits
the one-token newcomers to the speculative path, and the admission yield's cold-first ordering
runs their first bursts ahead of the stream. Their clients' next requests are therefore waiting
when the stream's next burst begins. On the RTX 5090 the 27B's calibration is refused, the band
admits the newcomers to plain decode, and the stream does not stall at two clients. The mechanism
itself (a yield at the burst's entry on a carried token) has shipped since v0.71.0, and with
`MEMRA_SPEC_GATE=0` it stalls the RTX 5090 too.

## Measurement

GB10, Qwen3.8-27B NVFP4 + Q5_K with its MTP head, default gate, greedy. One long streamed request
runs alone for 8 s, then closed-loop clients send one-token requests for 30 s, then it runs alone
for 12 s. Pass/fail style, single runs. The arms ran in one probe binary with per-cell knobs; at
four clients they ran in an ABBA order inside one boot. Every long text is prefix-equal to main's.

Long-stream tokens in the 30 s window, longest gap, and the short requests' mean latency:

| arm | 4 clients, 0 ms | 2 clients, 0 ms | 2 clients, 200 ms |
|---|---|---|---|
| main (no bound) | 1 / 30.7 s / 0.83 s | 1 / 30.3 s / 0.42 s | 1 / 30.1 s / 0.22 s |
| bound N=1 | 58, 53 / 1.8 s / 0.74 s | 74 / 0.94 s / 0.47 s | 142 / 0.73 s / 0.32 s |
| bound N=2 | 31, 31 / 2.6 s / 0.81 s | 50 / 1.36 s / 0.45 s | 93 / 0.93 s / 0.29 s |
| bound N=4 | 41, 38 / 4.3 s / 0.70 s | 26 / 2.2 s / 0.44 s | 57 / 1.35 s / 0.26 s |
| demote after two silent entries | 33, 33 / 2.6 s / 0.89 s | 62 / 1.33 s / 0.49 s | 106 / 0.91 s / 0.36 s |
| flush the carried token, no bound | 35, 34 / 0.92 s / 0.90 s | 64 / 0.50 s / 0.49 s | 106 / 0.49 s / 0.37 s |
| bound N=0 (one round at every waiting entry) | 83 / 0.97 s / 0.84 s | 143 / 0.52 s / 0.51 s | 228 / 0.51 s / 0.42 s |
| band owns the gate | 169 / 0.28 s / 0.36 s, demoted | 645 / 0.32 s / 1.67 s | 654 / 0.33 s / 1.68 s |
| zero-rate sample in the realized slot | 150 / 3.6 s / 0.39 s, demoted | 189 / 1.9 s / 0.31 s, demoted | 267 / 0.22 s / 0.25 s, demoted |

At four clients every arm that runs rounds ends in a demotion. Once rounds run, the 30 s
acceptance window ages below the 64 rounds a verdict needs, and the n=5 verdict falls back to the
band (5 >= HIGH). That spread blurs the comparison between N values at four clients. At two
clients nothing demotes.

The shipped bound against main v0.141.0, the full think-time sweep, one boot each on GB10
(`gb10/sweep-fix/`, `gb10/sweep-main141/`). Every long text is prefix-equal to main's.

| clients | think ms | long tokens in 30 s, main -> fix | longest gap s | short requests | short p50 s | short p95 s | fix demoted |
|---:|---:|---|---|---|---|---|---|
| 1 | 1000 | 631 -> **630** | 0.31 -> 0.31 | 24 -> 24 | 0.301 -> 0.302 | 0.304 -> 0.308 | no |
| 1 | 200 | 427 -> **424** | 0.32 -> 0.31 | 60 -> 62 | 0.307 -> 0.306 | 0.312 -> 0.31 | no |
| 1 | 50 | 229 -> **229** | 0.32 -> 0.32 | 97 -> 97 | 0.259 -> 0.26 | 0.262 -> 0.264 | no |
| 1 | 0 | 229 -> **230** | 0.32 -> 0.32 | 97 -> 98 | 0.31 -> 0.31 | 0.312 -> 0.312 | no |
| 2 | 1000 | 491 -> **488** | 0.53 -> 0.52 | 50 -> 50 | 0.22 -> 0.219 | 0.31 -> 0.311 | no |
| 2 | 200 | 1 -> **139** | 30.04 -> 0.73 | 143 -> 116 | 0.219 -> 0.321 | 0.225 -> 0.33 | no |
| 2 | 50 | 1 -> **94** | 30.21 -> 0.73 | 144 -> 125 | 0.369 -> 0.466 | 0.376 -> 0.474 | no |
| 2 | 0 | 1 -> **74** | 30.21 -> 0.94 | 145 -> 130 | 0.416 -> 0.439 | 0.421 -> 0.519 | no |
| 4 | 1000 | 228 -> **228** | 0.95 -> 0.95 | 98 -> 98 | 0.238 -> 0.24 | 0.254 -> 0.344 | no |
| 4 | 200 | 1 -> **50** | 30.49 -> 1.56 | 146 -> 136 | 0.633 -> 0.729 | 0.645 -> 0.743 | no |
| 4 | 50 | 1 -> **59** | 30.66 -> 1.57 | 147 -> 159 | 0.782 -> 0.781 | 0.796 -> 0.893 | yes |
| 4 | 0 | 1 -> **56** | 30.66 -> 1.77 | 147 -> 160 | 0.833 -> 0.835 | 0.844 -> 0.936 | yes |

Every cell advances, with a longest gap of 1.77 s against 30.0 to 30.7 s on main. One client is
unchanged. Where the bound acts, the short requests' p50 rises by at most 0.10 s and their p95 by
0.09 to 0.10 s.

## What was chosen and what was not

- **Chosen: the bound at N=1** (`SPEC_ENTRY_YIELD_MAX`). A session yields at a burst's entry
  with nothing emitted at most once in a row. At its next such entry it runs one round, and that
  round's hook yields. Where a burst ends is content-neutral, so texts are unchanged. The bound
  acts only after a silent entry yield, whoever owns the gate and on either sampling class: a
  sampled session that carries a token (memra#1186) takes the same path.
- **Why N=1.** Across N the trade is close to linear. In the two-client cells, where nothing
  demotes, the bound buys 10.0 to 12.5 long-stream tokens per second of added short-request
  latency at N = 0, 1, 2 and 4 alike. N only sets the split. N=1 keeps the longest gap under
  1.8 s in every measured cell (2.6 s at N=2, 4.3 s at N=4), and a waiting request pays at most
  one round (about 0.1 s on GB10) every second burst. N=0 (a round at every waiting entry)
  doubles that cost (short mean +23% and +87% at two clients, against +12% and +43%) and drops
  the newcomer-first ordering at every carried entry.
- **What it guarantees, and what it does not.** It guarantees no outright stall. It is a
  liveness guarantee, not a throughput policy. Under this pressure the stream still runs at about
  2 to 5 tok/s on GB10 (74 and 139 tokens in 30 s at two clients, 0 and 200 ms). When the band serializes the
  newcomers behind its bursts, the same stream keeps about 21 tok/s, and the newcomers wait about
  1.6 s. The share between the classes is memra#1211.
- **Rejected: demoting a session after two silent entries** (a price for the cost model: the
  session's realized rate over those ticks is zero). The demoted session takes a plain step every
  tick, which costs the short requests more than the bound does (mean 0.89 against 0.74 s at four
  clients, 0.36 against 0.32 s at two), and the stream gets fewer tokens (33 against 53 to 58, 106
  against 142). Its trade is 7.1 to 7.2 tokens per added short second. It also acts only where
  the cost model owns the gate.
- **Rejected: no bound, flush the carried token** (one target step at a yielding entry, so the
  entry emits the next token). It pays a target step every tick: a smaller gap than the bound's,
  but more short-request latency and fewer stream tokens in every cell (6.7 tokens per added
  short second).
- **Deferred to memra#1211: a zero-rate sample in the cost model's realized slot** at the tick's
  concurrency, on a session's second silent entry in a row. It did best for both classes in these
  cells (150 to 267 stream tokens, short p50 0.25 to 0.36 s): the verdict at that concurrency goes
  to K=0, the stream demotes and later newcomers are admitted to plain decode. It does not ship
  as built. A speculative slot is refreshed only by speculative ticks at its own concurrency, and
  the tie probe needs a spec price within two standard errors of plain, so a zero slot is never
  measured again and that concurrency is served plain until restart (the third slot cell shows
  it). Neighbors inherit the zero through `priced()`, whose choice between equal neighbors is
  memra#1212. A price that sees these ticks needs a decay or time-to-live on its zero samples and
  its own normal-concurrency proof.
- **Not changed: the cold-first ordering and the cost model's admission of short newcomers.**
  Changing the ordering serializes newcomers behind full bursts, which is what the band does and
  pays for with about 1.6 s of short-request latency. Those are policy questions with their own
  trade (memra#1211); the bound does not decide them.

## Workarounds for releases before the fix

- `MEMRA_SPEC_GATE_MODEL=0` (the band owns the gate): no stall. At two clients the stream keeps
  21.6 to 21.8 tok/s and a one-token request takes 1.6 s at p50. At two concurrent agentic
  sessions the band's aggregate is about 18% lower than the cost model's
  (`research/batch-spec-20261101/GB10-AGENTIC.md`).
- `MEMRA_ADMIT_YIELD=0` (lockstep fairness): no stall. A one-token request takes 2.3 to 3.0 s at
  p50, and short-request throughput falls 3 to 7 times.
