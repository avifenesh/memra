# A sampled speculative burst carries its pending token (2026-11-10)

**Status:** adopted for the MTP speculative route (GB10: sampled decode +8.4%, greedy flat). A
sampled burst ends on a pending token and stashes it on the session, as a greedy burst has since
2026-08-01; the next burst verifies it as round 0's first column and draws the next position there.
The solo trunk pass that committed the token at the burst's tail and the solo pass that fed the
boundary token at the next burst's start are gone. Every seeded text is the one the commit before
wrote. Two rollback seams keep the old tail. The DFlash2, GLM5, DSpark and gemma assistant-drafter
routes are not touched. Issue memra#1186. Receipts: [`research/spec-sampled-pending-carry-20261110/`
](../../research/spec-sampled-pending-carry-20261110/).

## Question

On GB10 a sampled speculative round of Qwen3.8-27B cost 10 to 12 ms of wall more than a greedy
one, and the split (`research/gb10-sampled-round-split-20261110/`) put all of it at the burst
boundary. A burst ends on a pending token: emitted to the client, not yet in the caches. The
sampled tail committed it with one T=1 trunk pass, drew the boundary token from that pass's
logits, and the next burst fed the boundary token with a second T=1 pass. A T=1 pass on GB10
costs 75 ms, 97% of a round's T=4 verify, and a burst is 32 tokens.

The greedy path has carried the pending token into the next burst's first verify since
2026-08-01. The sampled path could not: its accept walk needed the commit pass's logits for
round 0, and a boundary draw made anywhere else would have used other random numbers. Since
memra#1154 (`SAMPLED-VERIFY-POSITION-KEYED.md`) neither holds. The token at generated position
p is `argmax(filter(target row_p)/T + Gumbel(seed, p))`, the event keyed by the position, and a
round that has a pending token draws its first row from verify column 0.

Can a sampled burst carry, with no draw moved?

## What was chosen

The sampled tail stashes its pending token exactly as the greedy tail does
(`spec_burst_end`): the token stays out of `committed` and out of the caches, `next_pred` is
empty, `last_h` is the hidden of the last committed row, and the session counter stays at the
event of the position after the pending token. The next burst (`spec_burst_begin`) takes the
token as round 0's pending: no emission, no init feed, base 1. Round 0's verify commits it as
column 0 and its row 0 is that column, drawn at the session counter.

Why no draw moves, piece by piece against the tail it replaces:

| | the old tail | the carried round 0 |
|---|---|---|
| the row | the logits of a T=1 pass over the pending token | verify column 0 over the same token from the same cache state (the decode-exact contract the greedy accept walk and the K invariance of memra#1154 already rest on) |
| the event | `sctr`, then `sctr + 1` | `sctr0 = sctr`, row 0 |
| the penalty window | the burst's window: the session's committed tail, the prompt, every token the burst emitted | the next burst's window: the session's committed tail (all of the above but the pending token) plus the pending token, pushed at entry |
| the filter and the draw | `sample_boundary_token`: penalise, the row's own filter statistics, Gumbel perturb, argmax | the round's row draw: the same kernels, per row |

One predicate, `sampled_pending_carries`, gates the tail's stash and both entry flushes (the
serial entry and the pooled cohort's).

What stays a drawn boundary token: the cold prime's first token, a converted prefix-cache hit's
seed, and the token after a burst that ended with no pending token (a burst that ran no round
before its first carry: a one-token first burst, an admission yield right after the prime).

Where the token is still committed by a solo pass (`spec_flush_pending`, with the request's own
sampler, so the next token is drawn at the session counter as the old tail drew it):

- before a non-empty prime (a resume that primes; the worker parks only flushed sessions, so
  this is the engine's own guard);
- when the worker parks the session at the end of a request. The pass runs after the response
  has ended. A session the grid law frees is not flushed at all;
- when the worker demotes the session to plain decode. Plain decode then continues one position
  later with the counter the flush left.

A request's last token therefore stays pending when the request ends, on EOS, on a stop string,
on its budget or on a full context, as a greedy request's does. A client that disconnects drops
the session with it.

The context-full stop counts a sampled session's pending token. Before the carry the sampled
tail had committed it, so the worker's check (`committed + K + 3` against the cache size) saw
every emitted token and a sampled request stopped one token earlier than a greedy one. Without
the count the carry would let a sampled request that fills its context run one more round: the
right tokens, but a longer text than before. The count keeps the rule in emitted tokens; it
does not pin the stop to the same token. The last burst ends at the engine's context-edge
guard or on the last round's overshoot, both of which depend on round alignment, and the carry
shifts round boundaries by one position (main fed a boundary token outside any round at every
burst start). The text is plain decode's prefix either way. The burst-boundary gate's three
context-edge rows stop at the same token in both arms at the default burst; other burst
lengths were not run at the context edge.

## What was rejected

- **Carrying under `MEMRA_SPEC_SAMPLED_BOUNDARY=0`.** The seam emits an argmax at every burst
  boundary. A carried boundary has no boundary token: round 0 would draw the position, and the
  seam would no longer restore what it names. It keeps the tail pass.
- **Carrying a penalised request under `MEMRA_SPEC_PEN_SESSION=0`.** The seam restarts the
  penalty window at every burst. The boundary token is then drawn in the ending burst's window,
  and the next row in a window that starts at that token. A carried round 0 would draw the
  boundary position in a window holding only the pending token. It keeps the tail pass; an
  unpenalised request under that seam carries.
- **Leaving the pending token on a parked session** (no flush at retirement). A parked
  session's committed tokens are its resume key (the exact and the text prefix match), and a
  recurrent trunk rewinds to a checkpoint and clears it anyway. The flush is one pass per
  retired request, after the response, and only for a session that is parked. Not measured as
  a cost worth a second resume path.
- **A longer burst** (`MEMRA_SPEC_BURST=128` left 3.5 to 5.3 ms of the gap in the split's probe)
  moves the scheduler cadence for concurrent requests and removes nothing.

Not a rejection, a case with nothing to carry: under `MEMRA_SPEC_REPLAY=1` every round commits
its bonus by replay, so a burst never ends on a pending token, sampled or greedy.

## What landed before it

Three changes this one rests on landed first, each as its own fix:

- **A no-round burst leaves the carried token pending** (memra#1196). A continuation burst can
  end before its first round: the worker's admission yield says stop at the burst's first commit
  hook when a request waits. Such a burst re-stashed the carried token and still pushed it into
  `committed`, as if round 0 had verified it. Greedy text never showed it, because the verify
  reads the cache; a sampled penalty window is seeded from `committed`, so the carry needed it
  fixed.
- **The entry-yield bound** (memra#1201, `SPEC-ENTRY-YIELD-BOUND.md`). A burst that enters on a
  carried token emits nothing at its first commit hook. Under arrivals that keep a request waiting
  at every entry, such a session stopped emitting: greedy on main, and with this carry a sampled
  session too, where before it advanced one boundary token per burst. The bound lets a session
  yield silently at a burst's entry at most once in a row; at its next such entry it runs one
  round. It reads only whether the entry call carries a token, not the sampling class, so a
  sampled burst that enters on a carried token takes it exactly as a greedy one does. A sampled
  burst on a rollback seam that keeps the tail pass enters on its drawn boundary token, which the
  entry call emits, so the bound does not act there.
- **A stop string ends a speculative request on the token that completes it** (memra#1207). The
  burst ends at that round, so where a stop-ended request's last burst ends no longer depends on
  the burst length, and its `completion_tokens` equals plain decode's.

## Evidence

Qwen3.8-27B NVFP4 + Q5_K with its MTP head, GB10 and one RTX 5090 Laptop GPU; `main` = main
66ee31fcb6 (the bound and the stop-string fix in), candidate = that commit plus this change.
Pass/fail cells are single runs.

- **Seeded-invariance gate** (54 rows: K=0..3, alone, paired, resumed, the parked-prompt pair,
  a forced demotion): PASS on both binaries on both devices, and 54 of 54 rows carry main's
  text sha, completion, prompt and cached tokens. Acceptance per drafted token is within 0.7
  points of main at every K (GB10 K=3 0.425 and 0.426).
- **Burst-boundary gate** (new, 229 rows: EOS, a stop string, the token budget at 31 to 65,
  a small context window, a grammar-constrained answer, a resumed turn, a park, arrivals with
  the demotion gate off and on; sampled and greedy; four burst settings): every speculative
  row equals plain decode's text, finish reason and completion tokens on both binaries, and
  229 of 229 candidate rows carry main's on both devices. Tail draws 1,179 (GB10) and 1,157
  (RTX 5090) on main and 0 on the candidate; 1,252 and 1,222 sampled bursts entered on a
  carried token; every no-round burst of the yield boot entered on one (23 and 13).
- **Stop-string gate** (memra#1207's), temperature 1 and greedy, default burst and burst 2:
  13 of 13 stop rows equal to plain decode per sampler and boot on both devices.
- **The bound holds for the sampled class.** The bound's stall cell at temperature 1 (2
  clients at 0 and 200 ms think, 4 clients at 0 and 200 ms): the long stream advances in every
  cell. GB10 at the default gate: 62, 105, 35, 45 tokens in 30 s on main, 75, 130, 55, 52 on the
  candidate, longest gap 0.96, 0.73, 1.78 and 1.56 s, against 0.94, 0.73, 1.77 and 1.56 s in
  the bound's greedy rows. RTX 5090 with the gate off: 64, 115, 35, 44 against 73, 131, 40, 50,
  longest gap 0.95, 0.72, 1.84 and 1.61 s. Texts prefix-equal. Before the bound the carry
  stalled such a stream outright.
- **End of a request** (close and demote cells, 56 rows per device): text, finish reason and
  usage equal to main on every row. A sampled request closes 0.09 to 0.28 ms after its last
  chunk on GB10, against 74.6 to 78.6 ms on main (two rows apart: 180.7 -> 104.7 and 75.3 ->
  8.9), and 0.06 to 0.15 ms on the RTX 5090, against 33.9 to 36.7 ms: the closing tail pass.
  For a parked session that pass now runs after the response, at the park flush.
- **Rollback seams and the hit gate** (RTX 5090): under each seam the candidate writes main's
  text, with main's rounds where the seam keeps the tail; the hit gate and its teeth arm are
  green.
- **session-gate**: ALL TURNS MATCH on both devices, its sampled carry check included.
- **Timed cell on GB10** (six boots per arm, main cand cand main, twelve cold prompts per
  boot, 400-token answers, K=3): the nine sampled classes (temperature 1, top_p 0.95, penalised;
  34, 1.2k and 3.6k-token prompts) decode 1.069 to 1.096 times main's rate, geometric mean
  1.084, each class ahead in 36 of 36 boot pairs; the greedy control 0.998 to 1.001. A sampled
  round costs 99.97 to 103.72 ms against main's 106.86 to 111.65, within 2.1 ms of the greedy
  round at the same prompt. One candidate boot was voided (the previous lease's server outlived
  it into that boot's start) and replaced. The split that found the two passes had predicted 7
  to 9% (`research/gb10-sampled-round-split-20261110/`).

## Not covered

- The RTX 5090 timed cell: only pass/fail cells ran on that card. The mechanism is in shared code; the price of the two passes
  differs per device.
- The RTX PRO 6000, three or more sampled requests at once, a multi-head MTP chain.
- The routes that still use the rejection walk (DFlash2, GLM5, DSpark, the gemma assistant
  drafter): no pending carry for a sampled request there, unchanged.
