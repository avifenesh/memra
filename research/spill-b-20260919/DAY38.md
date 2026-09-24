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

## 2. Results

Written after the runs. Section 1 is unchanged.
