# WP-B day 38: O2, the `MEMRA_KV_PARK_COMPACT` deciding cell (decide-by 2026-10-06)

OWED.md O2. The door is default OFF. At a plain-pool park it copies the retiring session's committed rows into a
fed-length cache and frees the ladder-cap allocation; a resume re-allocates at the request's own charged cap and
restores the rows before the suffix primes (`compact_parked_plain_cache`, `plain_resume_cap_admits`). Its FLAGS row
names the cell that decides it (DAY28.md section 2, the day-28 addendum of `KV-RESIDENCY-DESIGN.md`), and the lane
report that landed it (`research/kv-tenancy-20260831/REPORT.md`, "PENDING GPU gates" 1, 2 and 5) names the same
three gates. This day runs that cell on both cards. Every cell is `executed-not-qualified`; no default moves; the
verdict is the owner's.

## 1. Pre-registration

Committed and pushed before any day-38 code and before any day-38 boot. Nothing in section 1 changes after a number
is seen; a failed clause is recorded as it reads, and a design change is a new, dated addendum pushed before its code.

### 1.1 The subject, isolated

The door acts on the plain continuation pool only (spec and DSpark pools park live engine sessions, out of scope by
design). Every boot of this cell therefore runs:

- `MEMRA_SERVE_SPEC=0` (the plain path, the only path the door reaches);
- `MEMRA_PREFIX_CACHE_MB=0` (the prefix cache off, so a continuation is served by the continuation pool or cold,
  and never by a prefix-cache restore that would bypass the pool);
- `MEMRA_REUSE_POOL` at its default (2 per model and namespace); every conversation runs in its own cache namespace
  (`cache_salt`), so each continuation meets only its own parked entry;
- greedy decoding, the day-26 model and context per card (the 9B at `MEMRA_CTX=65536` on the 5090; the 27B at the
  checkpoint's 262,144 on the target card).

### 1.2 The workload (`day38-client.py`, one boot = one arm)

Prompt ids are built once per length from `docs/SERVING.md` text through `/v1/tokenize` and cut to exact lengths.
Every resume boundary of the exact-extension shape is a multiple of 32 tokens (the GDN prime grid, the capture law
of memra#602), so a resumed suffix prime starts on the grid as a cold prime does.

- **Shape X, exact extension, with the compact, grow, re-park cycle.** Per conversation, `/v1/completions` with
  `prompt_ids`: turn 1 is the base (length L), `max_tokens=1`, so the parked committed sequence is exactly the prompt;
  turn 2 is turn 1 plus 64 ids, `max_tokens=1` (a resume, then a second park); turn 3 is turn 2 plus 64 ids,
  `max_tokens=32` (a resume of the re-parked entry). Each turn's cold twin is the same prompt in a fresh namespace.
- **Shape A, affinity rewind.** Per conversation, `/v1/chat/completions` with an `x-session-id` header: turn 1 is a
  system and a user message over the length-L text, `max_tokens=16`; turn 2 resends that history with the assistant
  message replaced by a fixed string and a new user message, `max_tokens=32`, so the render diverges after the
  checkpoint and the plain-affinity path rewinds. Each turn's cold twin is the same messages in a fresh namespace
  with no session id.
- Lengths L: 6,144 and 30,720 tokens on the 5090; 6,144, 30,720 and 122,880 on the target card. N = 5 conversations
  per shape per length, distinct salts. Per request the row carries the tag, shape, turn, length, rep, HTTP status,
  `usage`, `finish_reason`, the completion text's sha256, submit and done times, and `/metrics` before and after.

### 1.3 Arms, orders, boots

Arms: `off` (`MEMRA_KV_PARK_COMPACT` unset) and `on` (`=1`), one binary (the lane tip at the first boot), one boot per
arm, both orders: O1 (off then on) and O2 (on then off). Per card: 4 main boots, the fault boot (1.4 P2), and the
interaction pair (1.5). The local chain runs under `/tmp/memra-5090.lock`; the target card's under
`/tmp/memra-gpu.lock`; each boot after a bounded idle wait; never a signal to anything this lane did not start.

### 1.4 Acceptance clauses

- **P1 resume identity.** (a) Door identity: on every resumed turn of both shapes, both orders, the completion digest
  of `on` equals `off`'s, and every cold twin's digest equals too. (b) The resumes happened: on `on`, every shape-X
  turn 2 and turn 3 prints `[kv-reuse] park-compact grow:` and the boot's `continuation_pool_hits` rises by the number
  of resumed turns; on both arms every shape-A turn 2 prints `[worker] plain-affinity: rewound to`; on `on` every
  retire that parks prints `[kv-reuse] park-compact:` (none `failed`). (c) No turn of either arm answers non-200.
- **P2 step-OOM adjacency** (the fault boot, `on`, `MEMRA_STEP_OOM_FAULT=1`, shape X at 6,144 only): the forged OOM
  parks the first stepping session back to the queue (`[admit-oom] step OOM parked session back to queue`); that
  teardown writes no `park-compact` line (every request of this shape parks on a normal retire, so the boot's
  `park-compact:` count equals its number of HTTP 200 rows exactly, and the torn-down attempt adds none); the retried
  request completes, and its conversation's later turns resume with digests equal to the main `on` boot's.
- **P3 park-time copy cost** (a reading, no bound): every `[kv-reuse] park-compact: .. in <ms>` per fed length on both
  cards, N >= 5 per length, with p50, p95 and max, and the 250 ms telemetry regime.
- **P4 benefit** (a reading, no bound): the continuation pool's retained device bytes at idle after each main boot
  (`cuda_driver_free_bytes`, `cuda_pool_cached_bytes`, pool entries), `off` against `on`, and the resumed turns' E2E.

### 1.5 The interaction pair (labelled, outside the rule)

`MEMRA_KV_ALLOCATOR=vmm` (the O1 serving arm, DAY37) with the park door `off` and `on`, shape X at 6,144 only, one
boot each: the parked bytes and digests. Under vmm a plain-pool park already releases the rows past the position
without a copy; the pair shows what the door adds or costs on top.

### 1.6 The decision rule (stated, not chosen)

Per card class:

- **PROMOTE-ELIGIBLE** when P1 (a) to (c) and P2 pass. P3 and P4 go with it. The door reaches the plain path only;
  on the served spec path it writes nothing (DAY27.md 2.5).
- **FAIL (no reading)** when P1 or P2 fails: the cause is quoted, and the door is revised under a new pre-registration
  before the decide-by or recorded as failed at it.
- **A reading for the owner, not a rule:** if the interaction pair shows the door's parked-byte saving already
  present under vmm alone, the door is redundant wherever vmm serves; the owner decides the two doors together.

Neither card's reading is compared with the other's, and no timing crosses cards.

### 1.7 CPU work before the cards

The client and its reader (`day38-client.py`, `day38-read.py`) with a dry run against a stub; `bash -n` and
shellcheck on the runners. No engine or server change is planned; if the cell needs one, it is an addendum first.

### 1.8 Addendum A (2026-09-24, from reading the code, before any day-38 code or boot)

Writing the client against the code found that P2 as registered cannot run on the plain path, and a defect beside it:

- **P2's instrument does not reach the plain path.** `MEMRA_STEP_OOM_FAULT` forges its OOM at one point, the spec
  phase's step dispatch (`"spec step"`), which a plain session never enters. The plain path also has no step-OOM
  retry: the step-OOM park back to the queue exists in the spec phase only (and the admission door's prefill park).
  So on the plain path P2's first clause ("parks the first stepping session back to the queue") has no branch to hit.
- **The defect beside it (door OFF and ON alike).** The session-ending error arms that the plain path can reach send
  a typed error but leave the session parkable: the non-batching step arm (`MEMRA_SERVE_BATCH=0`), the prefill tick's
  arm, and the eager-only decode arm set neither `aborted` nor `oom_teardown`, so `retire_may_park` passes and the
  errored session's cache parks. A later exact-extension or affinity resume could then adopt state that a failed
  step left behind (a recurrent layer advanced by a token that is not in `fed`). The batched decode arm already sets
  `aborted` and does not park. The spec phase's honest-error arm has the same shape for a non-OOM error.
- **The fix, before its code.** Every arm that ends a session on an error marks it unparkable through one new flag
  (`errored`), which `retire_may_park` reads and which keeps the session out of the completion history exactly as an
  abort does; a census test pairs every typed-error send of the tick loop's session arms with that flag or with
  `aborted`. No successful request's program changes.
- **P2, re-specified for the plain path.** The fault door gains the plain decode dispatch as a second injection point
  (the batched chunk and the non-batching step; the same quoted synthetic OOM, before any device work; its FLAGS row
  updated). The fault boots run `on`, shape X at 6,144 with `MEMRA_STEP_OOM_FAULT=1`, in two modes: batched (the
  default) and `MEMRA_SERVE_BATCH=0`. Clauses: the forged OOM ends the first plain session it reaches with the typed
  step error; that session writes no `park-compact` line and parks nothing (the boot's `park-compact:` count equals
  its number of HTTP 200 rows, and that conversation's next turn runs cold with a digest equal to its cold twin's).
  The non-batching mode also runs once on the unfixed binary (the r3 binary of DAY37) as the red arm; the defect's
  expected reading there is one extra `park-compact:` line and a resumed next turn.
- **Unchanged:** P1, P3, P4, the interaction pair and the decision rule.

### 1.9 Addendum B (2026-09-24, before any day-38 boot)

Addendum A's red arm named "the r3 binary of DAY37". That binary has no plain-path injection point (the fault door
gained it in the fix commit), so the forged OOM would never fire there on the plain path. The red arm is instead the
fix tree with only the flag's effect on parking removed: `day38-red.patch` makes `retire_may_park` ignore `errored`
(the pre-fix predicate), and nothing else. It is checked to apply and to compile before the boot, and the build
records both trees' hashes. The fault boots also run a fourth shape-X turn per conversation (turn 3 plus 64 ids,
`max_tokens=32`, with its cold twin), so the conversation whose turn 3 takes the forged OOM has a next turn to read.
With one request in flight, the forged OOM fires on the first decode step of the boot, which is conversation r0's
turn 3 (turns 1 and 2 are `max_tokens=1` and never decode).

### 1.10 Addendum C (2026-09-24, before any day-38 boot): the binaries

DAY37 addendum E (1.14) changed the lane's on-demand release path before any day-38 boot, so the lane tip at the
first boot is `c6f9282c2` (r4). Green is that commit's `memra-server` (the same file as DAY37's r4 lane binary,
`580fe677...`); red is the same commit plus `day38-red.patch` (`f581e84d...`), built in a detached worktree
(`rtx5090-day38/binaries.sha256`, `red.source`). The earlier builds from `99fef8898` never ran and are deleted. The
target-card chain pins the same source and reuses the day-37 lane file as green when its source matches. Nothing
else changes.

## 2. Results

Written after the runs. Section 1 is unchanged.
