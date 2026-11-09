# The MTP spec pool under the grid law (memra#1093 item 1, 2026-11-09)

Verdict: the spec pool's off-grid resume is reachable at the RTX 5090 device default, and it breaks
warm == cold. The fix ships. On main 65fb297a59, 12 of 22 three-turn pair cells left the cold text:
one greedy (an affinity rewind to an off-grid checkpoint) and eleven sampled (T=1 and penalized,
on-grid rewinds included). On the fix, 22 of 22 are byte-equal. The cost is about +0.1 s TTFT on
the turns main resumed at the parked end (turn 2 of a short chat: 0.120 s to 0.216 s rewound, 0.222
s cold), because every exact path primes with a checkpoint stop. Turns that already rewound on the
grid, or that the prefix cache serves, are unchanged.

Model and rig for every cell: Qwen3.8-27B NVFP4 + Q5_K GGUF with its MTP head, no DFlash drafter
(the MTP spec route serves), RTX 5090 Laptop, context 8,192, two sessions, prefix cache at its
derived size (800 MB), every switch at the device default unless a row names one. Binaries: main
65fb297a59 (sha256 82645099dc93c91d, the cell-evidence build) and the fix (e992559fe295d782, built
from this lane's first commit before `cargo fmt`, which changed one line's wrapping).

## Is the path reachable at the device default?

Yes, from two tiers, by ordinary short chats. `probe_reach.py` (`probe-main/`, one boot on main,
twelve three-turn shapes, greedy):

- **Exact tier, parked end.** A 17-token chat turn with `max_tokens` omitted (the vendor default)
  parks at 49 committed rows; turn 2 (69 tokens) extends them exactly and resumes there:
  `spec-reuse: 49 committed tokens resumed`. Chat and a client-rendered `/v1/completions` replay
  both reach it. With `max_tokens` 256 the parked cap (281 rows) is too small for turn 2's cap, so
  the exact tier skips the session and the affinity tier takes the turn.
- **Affinity tier, off-grid checkpoint.** A prompt too short for a stable boundary
  (`plain_checkpoint_boundary` returns `None`) and not nominatable keeps the engine's prompt-end
  checkpoint. The affinity tier rewinds to it and primes the rest: `rewound to 17 of 69`,
  `49 of 122`, `30 of 306`, `69 of 122`. A suffix primed from an off-grid start places its next
  checkpoint relative to that start, so the chain stays off the grid: `286 of 583` (30 + 256).
- **Not reached:** prompts long enough for a prefix-cache entry (about 80 tokens and up). The
  prefix cache's on-grid spec restore claims those turns first (`spec restore: 928 of 1001`), as
  the #1098 receipts found. A prompt under the 64-token floor, or between the floor and the floor
  plus 16, publishes no entry (`seed REFUSED (grid)`), which is why short chats reach the pool.

## The cells

`spec_pool_pair_cells.py` through `pairs.sh`: one warm boot serves a three-turn conversation in
order (each turn exactly extends the previous prompt and answer), one cold boot per turn 2 and turn
3 primes that turn's prompt from scratch, and each warm answer (reasoning + content) must be
byte-equal to its cold twin. T=1 cases use seed 1234; penalized cases add frequency 0.3 and
presence 0.2. Shapes:

- `chat-short`, `raw-short`: three short questions.
- `chat-story`, `raw-story`: a story and two continuations.
- `chat-mid`: two-paragraph answers.
- `chat-think`: thinking on, with the reasoning passed back.
- `chat-grid`, `raw-grid`: the first question padded to a 32-token prompt through `/v1/tokenize`.
- `chat-exact`: a fixed system line that makes the first turn 32 tokens, and whose parked stream
  turn 2 extends exactly. `probe_exact.py` found it (`probe-exact-fix/`).
- `raw-long`: a long first turn, with the prefix cache off and a `user` id.

`open` means `max_tokens` omitted. `pairs-base/run1` to `run4` are main (run 1 yielded to a landing
record after eight cases; its chat-story rows carry the `offgrid` expectation they were registered
with before they became on-grid controls). `pairs-fix/run1` to `run3` are the fix. `TABLE.md` is
`summarize.py` over all of them:

| case | sampling | main bytes (t2/t3) | main warm path | fix bytes (t2/t3) | fix warm path |
|---|---|---|---|---|---|
| chat-short-open | greedy | PASS | parked end 49, affinity 69 (off grid) | PASS | affinity 32, 1 park skip |
| chat-short-open-t1 | T=1 | FAIL (69!=68/95!=100) | affinity 17, 49 (off grid) | PASS | 1 park skip, prefix restore 64 |
| chat-short-open-t1-pen | T=1 pen | FAIL (161!=147/217!=101) | affinity 17, 49 (off grid) | PASS | 1 park skip, prefix restore 64 |
| chat-short-256 | greedy | PASS | affinity 17, 49 (off grid) | PASS | affinity 32, 1 park skip |
| chat-short-256-t1 | T=1 | FAIL (69!=68/95!=100) | affinity 17, 49 (off grid) | PASS | 1 park skip, prefix restore 64 |
| chat-mid-open | greedy | **FAIL (1204!=1050/=)** | affinity 30, 286 (off grid) | PASS | 1 park skip, prefix restore 288 |
| raw-short-open | greedy | PASS | parked end 49, affinity 69 (off grid) | PASS | affinity 32, 1 park skip |
| chat-grid-open | greedy | PASS | affinity 32, 77 (off grid) | PASS | affinity 32, 1 park skip |
| chat-grid-open-t1 | T=1 | FAIL (9!=12/2!=122) | affinity 32, 60 (off grid) | PASS | affinity 32, 1 park skip |
| chat-grid-open-t1-pen | T=1 pen | FAIL (9!=12/79!=139) | affinity 32, 60 (off grid) | PASS | affinity 32, 1 park skip |
| raw-grid-open | greedy | PASS | affinity 32, 77 (off grid) | PASS | affinity 32, 1 park skip |
| chat-exact-open | greedy | PASS | parked end 64, affinity 84 (off grid) | PASS | rewind 64->32, affinity 64 |
| chat-exact-open-t1 | T=1 | FAIL (9!=12/=) | affinity 32, 60 (off grid) | PASS | affinity 32, 1 park skip |
| chat-exact-open-t1-pen | T=1 pen | FAIL (9!=22/75!=139) | affinity 32, 60 (off grid) | PASS | affinity 32, 1 park skip |
| raw-long-nopc | greedy | PASS | affinity 928, 960 | PASS | affinity 928, 960 |
| raw-long-nopc-t1 | T=1 | FAIL (13!=12/=) | affinity 928, 960 | PASS | affinity 928, 960 |
| chat-story-open | greedy | PASS | affinity 32, 256 | PASS | affinity 32, 256 |
| chat-story-open-t1 | T=1 | FAIL (1090!=1048/682!=563) | affinity 32, 256 | PASS | affinity 32, 256 |
| raw-story-open-t1 | T=1 | FAIL (1090!=1048/682!=563) | affinity 32, 256 | PASS | affinity 32, 256 |
| chat-think-1024 | greedy | PASS | affinity 32, 64 | PASS | affinity 32, 64 |
| chat-think-1024-t1 | T=1 | PASS | affinity 32, 64 | PASS | affinity 32, 64 |
| chat-think-1024-t1-pen | T=1 pen | FAIL (=/33!=0) | affinity 32, 64 | PASS | affinity 32, 64 |

(`N!=M` = warm and cold character counts of a turn that differs; `=` = byte-equal.)

Two classes fail on main:

1. **Off the grid.** `chat-mid-open` greedy: turn 2 rewound to the off-grid prompt-end checkpoint
   30 and left the cold text (1,204 against 1,050 characters). The other greedy cases on an
   off-grid resume stayed equal on their short answers: the state differs, and no near-tie flipped.
2. **The sampled stream.** Every sampled case with a resume fails, on-grid rewinds included
   (`chat-story-open-t1` rewinds to 32 and 256), except `chat-think-1024-t1`, whose answers are 2
   to 9 characters. The first warm token already differs ("Weeks" against "Years") with the same K
   on both arms (K=3, `cold-short`), so the boundary draw used other Philox counters. A pool resume
   kept the parked request's `sctr`/`uctr`; the field's own contract is "(0,0) at admit", and a cold
   session starts there.

## The change

For a recurrent (GDN) trunk:

- An exact or text hit rewinds to the session's turn checkpoint when that checkpoint is on the
  32-token grid, the prompt matches the committed tokens through it, and at least `PRIME_MIN_T`
  rows remain to prime (`spec_pool_grid_refusal`, `spec_pool_take_extension`). Otherwise the
  session is dropped and the request primes cold. A text hit that rewinds primes the prompt's own
  tokens, not the re-tokenized remainder, so the warm stream is the one a cold prime sees.
- The affinity tier takes only such a checkpoint.
- A retiring session with no on-grid checkpoint is freed, not parked (it could never resume
  exactly, and it would hold its trunk and draft planes). The check runs before the pending flush.
- Every pool resume starts the request's Philox counters at (0, 0).
- `/metrics`: `spec_pool_grid_rewinds`, `spec_pool_grid_declines`, `spec_pool_grid_park_skips`.

Why an on-grid checkpoint is cold state: the engine captures the turn checkpoint at a prime stop.
Every prime a resumable spec session runs now starts on the grid (0, an on-grid prefix-cache entry,
or a checkpoint that passed the check), and every stop the worker arms is on the grid, so the state
at an on-grid stop is the state a cold prime of those tokens leaves (the grid law). Attention-only
trunks resume where they stand, as before.

Rewind or decline: the rewind is used whenever it is exact, since it primes fewer rows from a live
session; the decline is the only exact option when no on-grid checkpoint exists.

## Cost

`cost_cell.py`, base against fix, boots in ABBA order per shape, three conversations per boot (a
distinct opener each), greedy, thinking off, streamed (TTFT = first content chunk). n = 6 per cell,
the rig otherwise idle (`cost/`):

| shape | turn | main TTFT median (s) | fix TTFT median (s) | main range | fix range |
|---|---|---:|---:|---|---|
| chat-short-open | 1 | 0.124 | 0.123 | 0.122-0.125 | 0.121-0.124 |
| chat-short-open | 2 | 0.119 | 0.222 | 0.119-0.124 | 0.222-0.223 |
| chat-short-open | 3 | 0.221 | 0.222 | 0.220-0.221 | 0.221-0.223 |
| chat-mid-open | 1 | 0.125 | 0.126 | 0.123-0.127 | 0.123-0.140 |
| chat-mid-open | 2 | 0.314 | 0.314 | 0.293-0.318 | 0.301-0.318 |
| chat-mid-open | 3 | 0.385 | 0.396 | 0.122-0.403 | 0.313-0.406 |
| chat-story-open | 1 | 0.127 | 0.126 | 0.124-0.132 | 0.124-0.128 |
| chat-story-open | 2 | 0.321 | 0.412 | 0.119-0.326 | 0.309-0.422 |
| chat-story-open | 3 | 0.322 | 0.334 | 0.315-0.324 | 0.318-0.421 |
| raw-long-nopc | 1 | 0.806 | 0.796 | 0.784-0.843 | 0.787-0.810 |
| raw-long-nopc | 2 | 0.228 | 0.226 | 0.224-0.236 | 0.225-0.228 |
| raw-long-nopc | 3 | 0.221 | 0.222 | 0.219-0.224 | 0.220-0.222 |

The +0.10 s on `chat-short-open` turn 2 is the inexact resume given back. With an opener the
story's first prompt leaves the grid, so its turn 2 goes cold too (+0.09 s). Turn 3 and the long
shapes are unchanged.

Rewind against decline on the same turn (`cost-exact/`, the `chat-exact` shape, one conversation
per boot, boots in the order base fix nopool nopool fix base, twice; `nopool` is the fix binary
with `MEMRA_REUSE_POOL=0`, so turn 2 primes cold as a decline does; n = 4 per cell):

| turn | main (parked end 64) | fix (rewind to 32) | nopool (cold) |
|---|---:|---:|---:|
| 1 | 0.123 (0.122-0.124) | 0.123 (0.122-0.123) | 0.122 (0.122-0.123) |
| 2 | 0.120 (0.119-0.120) | 0.216 (0.215-0.216) | 0.222 (0.222-0.222) |
| 3 | 0.221 (0.220-0.221) | 0.222 (0.222-0.223) | 0.223 (0.222-0.223) |

The rewind beats the decline by 6 ms on this 101-token turn. The gap to main is not the prime
length, and not session setup: the rewound session keeps its allocation and draft graph and still
takes 0.216 s. Every prime here that stops at a checkpoint takes about 0.22 s or more (turn 3 on
both arms, `raw-long-nopc` priming 69 rows from 928). A prime with no stop takes about 0.12 s (turn
1's 17 tokens, main's parked-end suffix). The fixed cost is in the checkpointed prime; it is not
traced here.

## Left open

- The 1..15-row plain-affinity suffix (item 3) is unchanged here. The spec-affinity tier applies
  the prime floor to a recurrent trunk as part of this law.
- The checkpointed prime's fixed cost on the RTX 5090 (about 0.1 s over a stop-free prime of the
  same turn, above). The grid law makes more turns pay it; finding where it goes (the snapshot,
  the boundary capture or the split prime's MTP walk) needs a trace.
- Long resumes where the cached prefix reaches 1,024 tokens pick `cached_long` K (K=2 against the
  cold K=3). No cell here crossed it; at T=1 such a resume could draw differently from cold.
