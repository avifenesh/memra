# WP-B day 41: O11, the grid-checkpoint rewind arm for the verbatim-extension resume (priced for the owner)

OWED.md O11. The lead's order (2026-09-25): pre-register and price the grid-checkpoint rewind arm, so the owner decides
the near-tie residual question on receipts; do not change the serving default.

The question. A continuation that exactly extends a parked session's committed tokens resumes today on the parked
state, which holds the previous turn's DECODED rows; a cold prime of the same prompt primes those rows. `docs/SERVING.md`
names this a near-tie residual ("verbatim-extension continuation resumes keep decode-computed rows whose arithmetic a
cold prefill never reproduces"); `CLAUDE.md`'s one-numeric-program rule reads such a crossing as a bug unless forbidden
or proven bit-identical. Measured on the 27B on the target card: 1 flip in 19 exact-extension resumes (DAY38 2.1), 3 in
30 on both arms and both orders (DAY38 2.3). A resume that rewinds to the entry's grid checkpoint (a prime-produced state
on the GDN prime grid) and re-primes from there is cold-identical by the grid law (`grid_align_boundary`: a prime split
on the grid is bit-identical to the monolithic prime). This day builds that arm behind a default-OFF door and measures
what it costs against keeping the decoded rows. Every cell is `executed-not-qualified`; no default moves; the contract
is the owner's.

## 1. Pre-registration

Committed and pushed before any day-41 code and before any day-41 cell. Nothing in section 1 changes after a number is
seen; a failed clause is recorded as it reads and a revision is a new, dated addendum pushed before its code.

### 1.1 The arm (`MEMRA_RESUME_GRID_REWIND`, default unset; decide-by 2026-10-09)

Unset: today's program, byte for byte. `1`:

- **Arming.** Every plain session arms its affinity checkpoint at `plain_checkpoint_boundary(prompt)` (grid-aligned,
  before the last turn marker, or `PLAIN_CKPT_RAW_GUARD` tokens before a markerless prompt's end), not only a
  nominatable prompt or a named conversation; every cold spec session arms its turn checkpoint by the same boundary
  (resumed spec sessions already arm one).
- **Plain pool.** An exact-extension hit (`continuation_reuse_index`) rewinds the entry to its checkpoint before the
  suffix primes, after the park-compact regrow when there is one, through the affinity path's own restore
  (`restore_cache_checkpoint`, `fed` truncated to the checkpoint, the checkpoint's logits). The primed suffix is then the
  parked prompt's tail past the checkpoint, the parked turn's generated tokens, and the new tokens. An entry with no
  checkpoint, or a lapped ring, is dropped and the request primes cold.
- **Spec pool.** An Exact or Text hit rewinds the session to its turn checkpoint (`spec_rewind_to_checkpoint`) and primes
  the token suffix from there (the text remainder is not used after a rewind). No checkpoint: cold.
- **Receipts.** One line per resume: `[kv-reuse] grid-rewind: <plain|spec> extension of <F> committed rows rewound to
  <P> (re-priming <R> rows; model <m>)`, or `[kv-reuse] grid-rewind: <plain|spec> declined (<why>); cold`.
- **Outside this arm, stated:** the DSpark pool's resume and the prefix cache's prompt-end seed extension hits (the
  other residual `docs/SERVING.md` names). Each is its own arm if the owner wants the same question answered there.

The arm's costs, by construction: (a) the re-primed rows per resume (the previous turn's generated tokens plus the
prompt tail past the checkpoint, at most `PLAIN_CKPT_RAW_GUARD` + 31 tokens on a raw prompt); (b) one checkpoint
snapshot per session on a recurrent model (156.9 MB on the 27B, DAY39 2.1), booked by the admission door's term when
armed; (c) an armed checkpoint takes a raw-prompt session out of the in-batch fanout and prime-batch paths
(`plain_ckpt_nominatable`'s documented cost).

### 1.2 Engine-level mechanism check (`concat-prime-probe primepath --hist K --rewind`)

A new `rewind` arm beside `hist`: turn 1 primed with a stop and a snapshot at the grid boundary `b` (the largest
multiple of the GDN chunk at or below the prompt end that leaves at least `PRIME_MIN_T` rows after it), primed on to its
end, the same K greedy tokens decoded, a rollback to the snapshot, then `seq[b..]` primed. Against the monolithic
reference the arm is expected `EXACT`; `hist` (the kept rows) prints its own verdict. The line also prints each arm's
suffix-prime wall time. Runs: the 27B on the target card, the 9B on the 5090, prompts of 6,144 and 30,720 tokens from
`docs/SERVING.md`, K in {32, 256}, suffix 64 tokens.

### 1.3 Serving cells (`day41-client.py`, one boot = one arm)

- **Shape RX (a raw agent loop).** Per conversation, `/v1/completions` with `prompt_ids`, greedy, streamed: turn 1 is
  L ids (on the grid) with `max_ctx = L + 2048` and `max_tokens = G`; turn 2 is turn 1's ids, turn 1's completion
  tokenized through `/v1/tokenize`, and 64 stream ids; turn 3 likewise from turn 2. Each turn's cold twin: the same ids
  and `max_ctx` in a fresh namespace. G in {32, 256} (both in every boot, as separate conversations). Lengths: 6,144 and
  30,720 on both cards, plus 122,880 on the target card. N = 5 conversations per (L, G).
- **Shape FX (the fanout cost).** 8 raw requests sharing a 4,096-token prefix, released together, `max_tokens = 16`:
  the in-batch fanout's reach per arm.
- **Routes.** The plain route (`MEMRA_SERVE_SPEC=0`) and the spec route (the default), each its own boots.
- **Arms and orders.** `keep` (door unset) and `rewind` (`=1`), one binary, both orders per route (O1 keep then rewind,
  O2 rewind then keep): 8 boots per card. Plus `off-prev`: the previous tip (`a803d3080`, no door) on the plain route
  with the RX shape at 6,144 only, for R3.
- Per row: tag, shape, route, arm, L, G, turn, cold, TTFT (first streamed token), E2E, `cached_tokens`, completion
  digest, prompt sha256, the pool-hit delta.

### 1.4 Clauses (the arm's correctness; the decision is not a rule)

- **R1 exactness.** On `rewind`, every resumed RX turn (`cached_tokens > 0`) equals its cold twin, on both routes, every
  length and G, both cards; every such resume prints a `grid-rewind:` line whose R equals the committed rows minus P.
- **R2 the resumes happened.** On both arms at least 80% of RX turns 2 and 3 resume (a tokenization miss misses on both
  arms and is counted).
- **R3 door OFF.** Every `keep` RX row at 6,144 on the plain route equals `off-prev`'s.
- **R4 health.** No `CUDA_ERROR_OUT_OF_MEMORY` line, no 503, no crash line on any boot.

Readings, no bound (the price the owner reads): per card, route, L and G: re-primed rows per resume on `rewind` against
the suffix rows `keep` primes; resumed-turn TTFT p50 and p95 per arm and their ratio; generated tokens over boot wall
time; resumed-against-cold flips per arm; idle driver free and pool-cached bytes; FX's prefix hits and TTFT per arm.
Every median states N and the 250 ms regime.

### 1.5 CPU before the cards

Census tests: the arm is read at the two pools' exact and text hits and at the two arming sites, and nowhere else;
unarmed, the probe and admission text are today's. The client and reader dry-run against a local stub.

### 1.6 Price

Code: about 0.5 agent-day (the two pools' rewind, arming, receipts, census, the probe arm, the client and reader).
Cells: the 5090 about 3 h once its reset is done (it is down since 01:25Z, Xid 119 then 154); the target card about
4.5 h (the probe 20 min, 8 serving boots, `off-prev`).
