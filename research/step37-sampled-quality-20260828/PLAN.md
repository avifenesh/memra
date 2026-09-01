# Sampled-quality cell: does MEMRA_STEP_GEMM_PRIME_SUFFIX ship ON for Step-3.7-Flash?

Bytes are settled (gemm-suffix lane VERDICT.md: every prime path is m-dependent under a
decomposition fork). This cell answers the owner's remaining question: do the warm paths
produce WORSE ANSWERS, sampled, vendor-default?

## Box and pins

dev box (devbox2) <see darklanes research: devbox2 provisioning lane, 2026-08-28> (<devbox2-ip>, 2x RTX PRO 6000 Blackwell Server, SPOT).
Source /home/ubuntu/memra @ lane/step37-main-merge-20260828 tip 8695bdef4a.
Binary /home/ubuntu/memra/target/release/memra-server md5 f45c3623d958ca085eefd3207987812a
(verified before first boot; printed into every results header).
Model /root/models/step37-flash-nvfp4 (ephemeral NVMe). GPU lock /root/gemmprime.lock.

Serving env: ENVV from /root/agentic8.sh + MEMRA_LOAD_MTP=1 MEMRA_MTP_HEADS=3
MEMRA_SPEC_K=3 MEMRA_SPEC_PMIN=0.5 MEMRA_SPEC_PMIN0=1 MEMRA_CTX=262144 MEMRA_SERVE_SPEC=1.
Vendor-default sampled: NO temperature/top_p in any evaluated payload. max_tokens=1024.

## Conversation

8-turn growing conversation: turn 1 user = /root/curve-1000.json content, turns 2..8 user
= /root/agentic8.json strings 1..7 (idx 0..6). Canonical assistant replies A1..A7 are
generated ONCE (cold, fresh session per turn, vendor-default sampled, max_tokens 1024,
reply text = content if non-empty else reasoning, truncated to 4000 chars per the
agentic8.sh convention) and REUSED as the fixed transcript by every arm and sample.
Evaluated turns: 4 (mid context) and 8 (deep context), 8 samples per arm per turn.

## Arms

- COLD: fresh session_id, full conversation in one request, first requests on a fresh
  boot. Validity: no rewind, eng_fresh>0, eng_suffix=0, walk_suffix=0.
- WARM-GEMM (MEMRA_STEP_GEMM_PRIME_SUFFIX=1): conversation replayed sequentially on one
  session_id (prefix turns max_tokens=64, replies discarded; the transcript stays the
  canonical one). Validity: rewound=True on the evaluated turn and [gemm-prime] ENGAGED
  base>0 present, walk_suffix=0.
- WARM-WALK (MEMRA_STEP_GEMM_PRIME_SUFFIX=0, committed default): same replay. Validity:
  rewound=True and [gemm-prime] WALK base>0 present, eng_suffix=0.

Each warm sample replays the conversation fresh on its own session_id (no sample inherits
another sample's sampled replies; the fixed transcript keeps the answered conversation
identical across arms and samples).

Known instrument property, stated up front: with a fixed transcript the warm suffix is
(canonical reply + next user msg) rather than the pure new-user-msg suffix real serving
sees, because the server's own discarded prefix replies diverge from the canonical text at
the rewind point. The mechanism under test (suffix prime at base>0 through the batched vs
walk entry) is exercised on every warm evaluated turn; suffix lengths are banked per row.

## Interleaving and boots

Cycles of one sample per arm per turn; each cycle boots a door=0 server (COLD t4, COLD t8,
WALK t4, WALK t8) and a door=1 server (GEMM t4, GEMM t8), boot order alternating by cycle
parity. Cold rows are always the first requests of their boot. Cycle count x samples per
cycle = 8 total samples per arm per turn (granularity chosen after measuring boot time;
recorded in RESULTS).

## Hygiene

- /health 200 guard before every generation; empty output = disqualified row, kept.
- Engagement receipts parsed from the server log per request (gs-drive.py pattern).
- ILLEGAL / #87 counts scanned per boot; any nonzero is a launch-blocking finding.
- Raw generations, receipts, rubric, scores banked here and committed as they land
  (SPOT box; nothing lives only on the box).
- Blind judging: outputs shuffled and stripped to text-only before scoring; mapping
  file read only after all scores are written. RUBRIC.md committed before any
  generation existed.

## LAUNCH-BLOCKING FINDING (2026-08-28, during attempt 2): grow panic

An 8-turn growing conversation replayed sequentially on one session (the REAL agentic
customer shape) kills the GPU worker at replay turn 6 on this tip:

```
thread 'memra-gpu-worker' panicked at cudarc-0.19.8/src/driver/safe/core.rs:1795:36:
called `Option::unwrap()` on a `None` value
   4: <memra_engine::Engine>::copy_u8_into
   5: memra_engine::pp::restore_cache_checkpoint
   6: <memra_engine::hybrid::HybridModel>::spec_grow_and_rewind_to_checkpoint
   7: memra_server::worker::alloc_with_single_reclaim_retry
   8: memra_server::worker::admit
```

cudarc core.rs:1795 is `CudaSlice::slice_mut(bounds).unwrap()`: an out-of-bounds slice
restoring the cache checkpoint into the regrown session buffer.

Trigger geometry, 3/3 reproductions (raw in `raw/sq-attempt2-bank.tgz`):

- Sequential replay, door=1: grows 1552 -> 2780 -> 3777 -> 4890 -> 6126 rows succeed;
  the next grow (target ~7.2k rows, ctx 7090) panics.
- Sequential replay, door=0 (committed default): IDENTICAL failure at the same turn.
  The gemm-prime suffix door is NOT involved.
- One-grow probe (fresh 8007-token prime, then a single rewind+grow to ~10.2k target):
  panics on the FIRST grow. Trigger = grow TARGET SIZE (in 6126..~7.2k rows), not
  cumulative grows. Cold fresh primes are unaffected (9118 tokens, TTFT 3.5 s).

Blast radius: the worker respawns with a full weight reload (~2-3 min); every queued
request during the reload hits the 90 s first-token deadline while /health stays 200.
Any customer session whose conversation grows past ~6k rows (5-6 real agentic turns)
triggers it deterministically. Engine fix belongs to a memra lane
(pp::restore_cache_checkpoint bounds under HybridModel grow); it does NOT gate the
suffix door (both door arms fail identically, no suffix door needed, only a grow).

Instrument dodge used by this cell, no engine change: session capacity is fixed at
creation (prompt + max_tokens + 8) and rewinds never shrink it, so the FIRST prefix
request of each warm replay carries a large max_tokens (t4: 4600, t8: 9100) that
pre-sizes the session for the evaluated turn; every later turn is rewind +
suffix-prime with NO grow, preserving the true sequential warm shape. Two more
instrument traps paid for on the way: (a) the park pool holds ~2 sessions and
nominates by longest prefix match, so identical transcript bytes across rows let a
deeper-checkpoint leftover out-nominate the pre-sized session ("declined (history
diverged at X of checkpoint Y)") - fixed with a per-row nonce in the turn-1 text;
(b) the pre-size request must STOP generating within ~512 tokens or the SWA ring
laps its own checkpoint ("declined (SWA ring lapped checkpoint N)") - fixed with a
stop-string, since capacity is reserved from max_tokens, not from tokens generated.

## RESULTS (2026-08-29, judged blind against the pre-registered rubric)

Binary: isolated rebuild of tip 8695bdef4a, md5 09fe2d670d82931248d4b0733898e6f4
(the tasking's f45c3623 build was overwritten by a co-tenant lane; markers
verified, cuda-linked). 72 rows total, ALL VALID by engagement receipts (8 per arm
per turn at t4; 16 per arm at t8 after the extension), zero ILLEGAL / #87 / panics
across all 19 battery boots (the only panics on this box are the grow-bug
reproductions above). Suffix lengths matched across warm arms (gemm 1127-1155 vs
walk 1154-1158 tokens). Raw: raw/rows.jsonl, raw/gen/, blind/ (sealed scores +
mappings), raw/sq-final-bank.tgz + raw/sq-ext-bank.tgz (server logs).

Rubric scores (0-6, median [min..max], blind, position-randomized):

| arm  | t4 (n=8)          | t8 (n=16)          | t8 hard-DQ (loop/empty) | t8 task-derail |
|------|-------------------|--------------------|-------------------------|----------------|
| cold | 4.75 [3.5..6.0]   | 3.75 [0.0..6.0]    | 4                       | 2              |
| gemm | 4.25 [3.5..5.5]   | 5.00 [0.0..6.0]    | 4                       | 2              |
| walk | 5.00 [3.5..6.0]   | 5.00 [0.0..5.5]    | 1                       | 2              |

Pre-registered rule: WARM-GEMM is INDISTINGUISHABLE from COLD on both turns (median
delta -0.50 at t4, +1.25 at t8, both far inside COLD's own self-spread of 2.5 and
6.0; hard-DQ counts equal). WARM-WALK likewise. Direct warm-vs-warm Mann-Whitney at
t8: gemm vs walk p~0.91 (a round-1 walk advantage, 5.25 vs 2.00 medians at n=8,
collapsed under the n=16 extension - it was sampling noise, which is why the
extension was run). Nothing here distinguishes the door from the incumbent walk
suffix or from cold full primes on answer quality.

VERDICT (three-outcome shape): outcome (1). WARM-GEMM quality is indistinguishable
from COLD within COLD's own sampling spread, on both evaluated turns, with the
turn-8 claim carried by n=16 per arm. The door adds no quality harm; the flip is
clean on quality grounds. Neither warm arm degrades vs COLD, so nothing gates
affinity/session reuse either.

Perf bank (NOT part of the verdict): evaluated-turn TTFT median gemm 0.58 s vs walk
7.15 s (suffix ~1150 tokens; the 7.97x slope collapse from the gemm-suffix lane,
reproduced at the serving shape), cold 2.26 s (t4) / 3.59 s (t8). Spec acceptance
0.80-0.86 in every arm/turn; gemm t8 is the highest (0.855 median).

Separate model-quality finding, arm-independent: at turn 8 (deep agentic context,
vendor-default sampling, max_tokens 1024), 14/48 answers across ALL arms are
unusable: 8 degenerate loops (counting runs, yes-list echoes, verbatim re-quoting),
6 task derails (answering an EARLIER turn's task, usually the turn-5 forensic-PDF
prompt). Counts per arm are statistically indistinguishable (loops 3/4/1). The
canonical transcript's own turn-7 reply (generated cold, kept per protocol) is a
degenerate yes-list, and it visibly primes these failures identically for every
arm. This is a Step-3.7-Flash deep-context robustness issue, not a door issue.
