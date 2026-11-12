# Prime-grid law: every boundary the server chooses on a recurrent trunk lands on the GDN fold grid (2026-08-21)

**Status:** adopted 2026-08-21 for the boundaries the server chose then (the plain and spec
checkpoint stop, the prefix-cache LCP-split capture, the message-boundary seed capture). Extended
to every capture and resume site through 2026-11-09 (see [Extensions](#extensions)). No flag: the
`MEMRA_PRIME_GRID_ALIGN` seam was stripped on 2026-09-05. Code: `grid_align_boundary`,
`seed_capture_boundary`, `plain_pool_resume_at`, `spec_pool_grid_refusal`, `dspark_park_rewind`,
`dspark_cold_capture` and `prefix_restore_exact` in `crates/memra-server/src/worker.rs`;
`gdn_chunk_size()` (32) in `crates/memra-engine/src/lib.rs`; `PRIME_MIN_T` (16) in
`crates/memra-engine/src/hybrid_forward.rs`. Gates: fast-gate `primegrid` and `primegridc`
(`tools/prime-grid-gate.sh`), `tools/prefix-newest-turn-fits-gate.py`,
`tools/prefix-restore-identity-gate.py`, `tools/spec-on-cache-hit-gate.sh`.

The rationale below moved out of the serving doc on 2026-11-12. The old text is kept in
[docs/archive/SERVING-2026-11-12.md](../archive/SERVING-2026-11-12.md#chunked-prefill-is-split-stable--since-the-grain-free-fix-found-2026-08-05-mechanism-corrected-and-fixed-same-day)
("The third axis").

## Question

The server primes one prompt as several `prime_cache` calls whenever it stops at a boundary: a
turn checkpoint, a prefix-cache capture, a seed, a resume. Within one call, chunked prefill is
split-stable since 2026-08-05, and the `step35` sliding-window architecture is stable across calls
since 2026-08-07 (`tickinv35`). Nothing pinned cross-call splits on the Qwen hybrid trunk, whose
GDN layers run a chunked WY scan. Is a prompt primed in two calls bit-identical to one monolithic
prime?

## Measurement

The trigger: identical 24k-token greedy requests produced three distinct outputs keyed only by
which prime program ran (restored at 24k, cold monolithic, restored at 607), and the plain arm's
boundary-stop prime diverged from its monolithic twin on 13 of 16 agent-length turns with
speculation off.

The law, measured with `concat-prime-probe primepath` on the Qwen3.8-27B trunk at v0.99.0, one
RTX PRO 6000 Blackwell Workstation Edition, one real 24k-token agentic prompt (2026-08-21; the raw
runs are not in this repository):

- a prompt primed as two calls split at `L` is bit-identical to the monolithic prime if and only if
  `L % gdn_chunk_size() == 0` (aligned: rows_diff 0/23999, logits exact), although the two
  programs' internal 2,048-token chunk boundaries do not coincide;
- an off-grid split diverges from exactly row `L` on; earlier rows stay bit-identical, and greedy
  flips land on near-tie margins;
- under the sequential GDN scan (`MEMRA_GDN_CHUNKED=0`) every split is exact.

Mechanism: the chunked WY scan segments per prime call, so an off-grid call start shifts the fold
grid and materializes recurrent state at a point the monolithic program never computes. That is
lawful floating-point behaviour of the WY form, not a defect. It is a different class from the
2026-08-05 chunk-0 defect (an O(1) class edge the arithmetic crossed), which stays gated by
`chunkinv`.

## Decision

The server rounds every boundary it chooses down to the grid. A boundary-stopped prime, a
checkpoint resume and a whole-entry restore then reproduce the cold monolithic bytes. Rounding down
alone costs at most `gdn_chunk_size() - 1` (31) re-primed suffix tokens per resume. A boundary
placed against a known prompt end must also leave at least `PRIME_MIN_T` (16) tokens to prime in
one call (the W1 floor in `grid_align_boundary_within`), so a 1 to 15-token remainder steps one
more grid unit down: such a resume re-primes 16 to 47 tokens, or none on an on-grid prompt
([Operator consequences](#operator-consequences); the day-18 seed entry sits at most 47 tokens
short of the prompt end, [`DAY18.md`](../../research/spill-b-20260919/DAY18.md)). The gate runs both
directions: aligned splits must be exact, off-grid divergence must be confined to the split row
with near-tie-only flips, and the canary coarsens the grid (`MEMRA_GDN_CHUNK=64`) and must break.

## Rejected

- **Fixing the off-grid divergence in the scan.** The WY form cannot give the same state at an
  arbitrary split point; the divergence is lawful, so the boundaries move instead.
- **The sequential scan as the serving program.** It is exact at every split and stays the
  diagnostic arm that proves the mechanism; the serving trunk keeps the chunked scan.
- **A prompt-end seed left at the prompt end ("unroundable by design").** Measured below; replaced
  2026-09-21.
- **Resuming a recurrent cache at its parked end.** That end holds decode-written rows off the
  grid. Measured below; replaced by grid rewinds.
- **The DSpark restore switch as the fix on the RTX 5090.** Turning it on wins the same turn
  (0.277 s TTFT) but refuses one concurrency step sooner on that card
  ([`research/pool-rewind-dspark-20261108/`](../../research/pool-rewind-dspark-20261108/)).
- **Evicting the arriving prompt's own rewound DSpark session last in the admission reclaim.**
  Tried and dropped: on the agentic workload it never changed which session survived
  ([`research/pool-grow-dspark-20261108/`](../../research/pool-grow-dspark-20261108/)).

## Extensions

Measured on the RTX 5090 Laptop GPU at the device default unless the row says otherwise. "Base" is
the binary before the change.

| site | since | base | after | receipt |
|---|---|---|---|---|
| prompt-end seed (`seed_capture_boundary`) | 2026-09-21, memra#602 | a 12-turn chain of 11,000 + 150-token turns flipped a near-tie at turn 10 (`grid_off=8`); the sequential scan was identical 12/12 | the seed publishes at `capture_len(P)`; both restore gates red on the pre-fix binary, green on the fix on the RTX 5090 and on one RTX PRO 6000 | [`research/spill-b-20260919/DAY17.md`](../../research/spill-b-20260919/DAY17.md), [`DAY18.md`](../../research/spill-b-20260919/DAY18.md), [`DAY19.md`](../../research/spill-b-20260919/DAY19.md) |
| spec session publications | 2026-09-21 | a cold spec session published off the grid | its `capture_at` is the seed's grid boundary; spec on and off restore the same entry; `spec-on-cache-hit-gate.sh qwen` green on the fix, red on the pre-fix accounting | `DAY19.md` above |
| seed rows in the concat prime | 2026-11-07 | an off-grid cold session primed alone up to its seed: mixed c8 batch TTFT p50 1,304 ms with the cache on against 736 ms off | the seed row joins the batch with its take ending on the boundary: TTFT p50 1,304 to 935 ms with the cache on, greedy text identical on 95/95 rows | [C8-SERVING-24GB.md](C8-SERVING-24GB.md) |
| plain continuation pool | 2026-11-07, memra#1091 | a three-turn chat resumed at 3,113 of 3,144 tokens and left the cold text | rewinds to the grid checkpoint and primes the rest, or primes cold when there is none; cold costs about 0.1 s TTFT on a 3k-token turn, because the on-grid seed serves the bulk | [`research/pool-resume-grid-20261107/`](../../research/pool-resume-grid-20261107/), [`research/pool-resume-cost-20261107/`](../../research/pool-resume-cost-20261107/) |
| DFlash capture and the restore hit site | 2026-11-07 | turn 2 restored 2,992 tokens off the grid; turn 3 left the cold text (383 against 371 characters) | the cold capture cuts at the seed's grid boundary; a recurrent entry serves only a whole cover or an on-grid end with `PRIME_MIN_T` suffix rows; +0.15 s on a 2,992-token first turn (3.54 to 3.69 s, mean of 2) | [`research/plain-restore-grid-20261107/`](../../research/plain-restore-grid-20261107/), [`research/grid-law-cpu-test-20261107/`](../../research/grid-law-cpu-test-20261107/) |
| DSpark pool park | 2026-11-08, memra#1093 | turn 3 resumed at 3,114 of 3,144 tokens and left the cold text | a retiring session parks rewound to its last own grid capture: turn 3 resumes at 3,072, 9/9 pair cells byte-equal; about +0.11 s TTFT against the inexact parked-end resume it replaces (0.163 to 0.275 s; the cold re-prime main ran after PR #1098, 2.462 s) | [`research/pool-rewind-dspark-20261108/`](../../research/pool-rewind-dspark-20261108/) |
| DSpark rewound session growth | 2026-11-08, memra#1093 | 47 to 50 sessions per agentic boot parked rewound and none resumed | the session grows to the next turn's cap, then seats: 15/15 pair cells byte-equal, turn 2 TTFT 2.319 to 0.273 s; at agentic c1, 3 of 11 second turns now resume, 2.6-3.0 to 0.45-0.83 s TTFT (the other 8 lost their session to the admission reclaim or the cache-preferred veto) | [`research/pool-grow-dspark-20261108/`](../../research/pool-grow-dspark-20261108/) |
| MTP spec pool | 2026-11-09, memra#1093 item 1 | 12 of 22 three-turn cells left the cold text | rewinds only to an on-grid turn checkpoint with `PRIME_MIN_T` rows to prime, and a resume that primes a suffix starts its Philox stream at (0, 0): 22/22 byte-equal; about +0.1 s TTFT on turns the base resumed at the parked end (0.120 to 0.216 s; cold 0.222 s) | [`research/spec-pool-grid-20261109/`](../../research/spec-pool-grid-20261109/) |
| plain-affinity rewind floor | 2026-11-09, memra#1093 item 3 | an edited 175-token resend rewound to 160, primed 15 rows token by token and left the cold text, 3/3 (821 against 904 characters) | the tier declines below `PRIME_MIN_T` rows and the turn primes cold: byte-equal 3/3, TTFT 0.327 to 0.217 s (medians of 3) | [`research/affinity-floor-20261109/`](../../research/affinity-floor-20261109/) |

## Operator consequences

- A published entry's length is `capture_len(P)` for a prompt of `P` tokens: `P` when `P` is a
  multiple of 32, otherwise the largest multiple of 32 below `P` whose remainder is at least 16. A
  grid-entry hit reports `cached_tokens == capture_len(P_previous)`. An exact re-send of an off-grid
  prompt restores `capture_len(P)` and primes 16 to 47 tokens from a grid start. Since 2026-11-11
  (memra#1233) the entry also carries the publishing prompt's prompt-end tail. While the tail is
  resident, that re-send restores all `P` tokens and reports `cached_tokens == P`; every longer
  prompt still restores the grid entry ([PREFIX-PROMPT-END-TAIL](PREFIX-PROMPT-END-TAIL.md)).
- A prompt between the 64-token entry floor and the floor plus 16 publishes nothing:
  `[prefix-cache] seed REFUSED (grid)`, counted in `prefix_cache_seed_grid_refusals`. A refused
  recurrent hit prints `[prefix-cache] restore grid law: ...` and counts in
  `prefix_cache_restore_grid_refusals`.
- A recurrent session with no plain checkpoint primes a resumed turn cold: glm5_next hyper trunks, prompts with
  no turn markers and no `session_id` or `user`, DFlash sessions demoted under load, and boots with
  `MEMRA_AFFINITY=0` or PP host-bounce.

## What the law does not promise

- A verbatim-extension resume of an attention-only cache keeps decode-computed rows a cold prefill
  never reproduces: bounded logit perturbation, flips only at near-ties (`primepath --hist`).
- The `PRIME_MIN_T` floor covers recurrent entries only; an attention-only hit that leaves a 1 to
  15-token suffix still primes it token by token.
- `prefix_fanout_groups` takes the raw in-batch LCP as its capture length; that path is covered by
  inspection only.
- glm5_next KDA entries count as recurrent although KDA prefill is a sequential scan with no fold
  grid, so the glm5 spec route's off-grid prompt-end captures serve only whole covers. This is
  conservative and gains no exactness there.

The valid assertions everywhere are per-program determinism, grid-aligned byte identity, and
confinement with near-tie-only flips across programs.
