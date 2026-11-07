# A recurrent continuation-pool resume rewinds to its grid checkpoint (memra#1091, 2026-11-07)

Verdict: at the RTX 5090 default a three-turn DFlash chat whose second turn had room for 256 tokens left the cold text on turn 3, because the plain continuation pool resumed the parked turn-2 session at its parked end (3,113 of 3,144 tokens: turn 2's prompt plus 4 decode-written rows, off the prime grid). A recurrent cache's exact-extension hit now rewinds to the session's grid checkpoint (3,072) and primes the rest; every fix arm is byte-equal to its cold twin, with and without `MEMRA_KV_PARK_COMPACT=1`.

## Cause

Reported by the 5090 lane on main's binary (adee274f0b): turn 3 warm 373 vs cold 372 characters, twice, with
`kv-reuse: 3113 of 3144 prompt tokens resumed`; with turn 2 at 96 tokens the parked cap does not cover turn 3,
the plain-affinity path rewinds to 3,072 and the turn passes. `plain_resume_cap_admits` (`entry_cap >= ctx_cap
|| park_compact`) is what lets `continuation_reuse_index` take the entry; it then resumed on the parked rows,
which the default path left unrewound (only `MEMRA_RESUME_GRID_REWIND` / `MEMRA_RESUME_EXACT`, default off,
rewound or settled). The rows past turn 2's prompt were written by decode, and the parked end is off the
32-token grid, so the GDN state there is not the cold prime's (docs/SERVING.md, the grid law).

## Change

`plain_pool_resume_at`: a recurrent cache resumes only at its grid checkpoint, with at least `PRIME_MIN_T`
rows to prime; no usable checkpoint primes cold; an attention-only cache resumes where it stands. The admit
site sends every recurrent exact-extension hit through `grid_rewind_plain`, which now decides by that rule.
Default-run test `the_continuation_pool_resume_obeys_the_restore_grid_law` (the receipt's 3,113 / 3,072 /
3,144 shape, the floor, no checkpoint, attention-only, and the wiring).

## Pair cells (RTX 5090 Laptop, `pair-cells/`)

`research/plain-restore-grid-20261107/grid_pair_cells.py` with `MAX_T2=256` (turn 2's max_tokens; the client
knob is new), base = 05a8a63503 (main), fix = 9c3a4edd98, switch at the device default (OFF):

| arm | warm env | turn 3 warm / cold chars | bytes | pool resumes (rows of prompt) | grid rewinds (from, to) | turn 1 sha |
|---|---|---|---|---|---|---|
| base-256 | `(device default)` | 372 / 371 | FAIL | [[3113, 3144]] | [] | 523ca36af0 |
| fix-256 | `(device default)` | 371 / 371 | pass | [[3072, 3144]] | [[3113, 3072]] | 523ca36af0 |
| base-256-compact | `MEMRA_KV_PARK_COMPACT=1` | 372 / 371 | FAIL | [[3113, 3144]] | [] | 523ca36af0 |
| fix-256-compact | `MEMRA_KV_PARK_COMPACT=1` | 371 / 371 | pass | [[3072, 3144]] | [[3113, 3072]] | 523ca36af0 |
