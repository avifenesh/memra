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

### 1.11 Addendum D (2026-09-25, after the target-card half read FAIL, before any code or boot of the revision)

Section 2.1 placed P1's failure and 2.2 corrects its fact 3: the verbatim-extension resume over decode-computed rows
is a named residual of the documented near-tie contract (`docs/SERVING.md`, "What the grid law still does NOT
promise"), so 1.4's cold-twin clause asserted, for shape X, what the engine does not promise; for shape A (the
affinity rewind to an on-grid checkpoint) the grid law does promise it. 1.2's workload also let `off` resume nowhere
at 6,144 and 30,720. P1 and P2 are re-registered; P3, P4, the interaction pair and the rule of 1.6 are unchanged.

- **Shape X' (replaces X).** Every turn sends `max_ctx = L + 512` (a request-supplied hard cap is the charged cap,
  `request_ctx_cap`), so a plain-parked entry fits the next turn on both arms and the door comparison is resume against
  resume. Turn 2 = turn 1's ids, then turn 1's completion text tokenized through `/v1/tokenize`, then stream ids to
  64 new tokens in all, `max_tokens=1`; turn 3 = turn 2's ids, turn 2's completion tokenized, stream ids to 64 new,
  `max_tokens=32`. Each turn's cold twin: the same ids and `max_ctx` in a fresh namespace.
- **P1'.** (a) Door identity: every X' turn 2 and 3 that resumed on both arms (`cached_tokens > 0` on both) has equal
  digests on `on` and `off`; every shape-A turn 2 has equal digests on `on`, `off` and its cold twin (the grid law).
  (b) The resumes happened: at least 80% of the X' turns 2 and 3 resume on both arms (a tokenization miss misses on
  both and is counted), every `on` resume prints `[kv-reuse] park-compact grow:`, and every shape-A turn 2 prints
  `[worker] plain-affinity: rewound to` on both arms. (c) No row answers non-200. Reading, no bound: X' resumed rows
  against their cold twins per arm (flips under the near-tie contract), with the resume row counts.
- **P2'.** `MEMRA_STEP_OOM_FAULT`'s non-batching injection point fires only on a session that is `prefill_done`, so the
  forged failure lands on a decode step after the prime (the batched point is already a decode chunk). Clauses as in
  addendum A: the errored session writes no `park-compact` line (the boot's count equals its HTTP 200 rows) and its
  conversation's next turn runs cold (equal to its cold twin, no pool hit). The red arm (non-batching, the fix tree
  with `day38-red.patch`) is expected to read one extra `park-compact` line and a resumed next turn; if it does not,
  the red arm has not exercised the defect and that is recorded.
- **Binaries:** the lane tip at the first boot of the revision (green) and green plus `day38-red.patch` (red). Both
  cards, the lengths of 1.2, both orders for the main boots.
- **Reader:** `day38d-read.py`, a new file (the registered `day38-read.py` reads the registered 5090 half unchanged).

## 2. Results

Written after the runs. Section 1 is unchanged.

### 2.1 The target-card half (one RTX PRO 6000 Blackwell Workstation Edition, 2026-09-24 19:06:59 to 21:57:34Z)

Green `3dc05d17...` (the day-37 lane file, source `c6f9282c2`), red `c31ff36d...` (green plus `day38-red.patch`), the 27B
at the checkpoint's context, lengths 6,144, 30,720 and 122,880. Receipts `pro-single-day38/box/`. The reader's lines,
verbatim:

```
DAY38 P1 card=pro6000 order=O1 rows=150 door_equal=149 door_differ=['X-30720-r3-t3'] cold_differ=['main-O1-on:X-30720-r3-t3'] non200=[] x_resumed_on=30 park_compact_grow=19 park_compact_on=150 failed=0 affinity_rewound={'main-O1-off': 20, 'main-O1-on': 16} a_resumed={'main-O1-off': 15, 'main-O1-on': 15} pool_hits={'main-O1-off': 20, 'main-O1-on': 35} faults=0 -> FAIL
DAY38 P1 card=pro6000 order=O2 rows=150 door_equal=149 door_differ=['X-30720-r3-t3'] cold_differ=['main-O2-on:X-30720-r3-t3'] non200=[] x_resumed_on=30 park_compact_grow=19 park_compact_on=150 failed=0 affinity_rewound={'main-O2-off': 20, 'main-O2-on': 16} a_resumed={'main-O2-off': 15, 'main-O2-on': 15} pool_hits={'main-O2-off': 20, 'main-O2-on': 35} faults=0 -> FAIL
DAY38 P2 card=pro6000 boot=fault-batch fired=1 errored_rows=['X-6144-r0-t1'] ok200=39 park_compact=39 next_turn=[{'tag': 'X-6144-r0-t2', 'equal_cold': True, 'pool_resumed': 0}] faults=0 -> PASS
DAY38 P2 card=pro6000 boot=fault-nobatch fired=1 errored_rows=['X-6144-r0-t1'] ok200=39 park_compact=39 next_turn=[{'tag': 'X-6144-r0-t2', 'equal_cold': True, 'pool_resumed': 0}] faults=0 -> PASS
DAY38 P2 card=pro6000 boot=fault-nobatch-red fired=1 errored_rows=['X-6144-r0-t1'] ok200=39 park_compact=39 next_turn=[{'tag': 'X-6144-r0-t2', 'equal_cold': True, 'pool_resumed': 0}] faults=0 -> PASS (red arm: the expected reading is FAIL with park_compact=ok200+1 and a resumed next turn)
DAY38 P3 card=pro6000 fed~6144 N=100 p50_ms=1.20 p95_ms=1.30 max_ms=1.30
DAY38 P3 card=pro6000 fed~30720 N=100 p50_ms=2.60 p95_ms=2.70 max_ms=2.70
DAY38 P3 card=pro6000 fed~122880 N=100 p50_ms=8.20 p95_ms=8.30 max_ms=8.30
DAY38 P4 card=pro6000 boot=main-O1-off idle driver_free=7564754944 pool_cached=6735256512 pool_reserved=93818191872 continuation_pool_entries=16 resumed_e2e_ms N=45 p50=1936.0 p95=51646.8
DAY38 P4 card=pro6000 boot=main-O1-on idle driver_free=4645519360 pool_cached=9678177920 pool_reserved=96737427456 continuation_pool_entries=16 resumed_e2e_ms N=45 p50=685.0 p95=9343.8
DAY38 P4 card=pro6000 boot=main-O2-off idle driver_free=7564754944 pool_cached=6735256512 pool_reserved=93818191872 continuation_pool_entries=16 resumed_e2e_ms N=45 p50=1968.0 p95=51678.8
DAY38 P4 card=pro6000 boot=main-O2-on idle driver_free=4645519360 pool_cached=9678177920 pool_reserved=96737427456 continuation_pool_entries=16 resumed_e2e_ms N=45 p50=700.0 p95=9352.0
DAY38 VMM-PAIR card=pro6000 rows=30 differ=[] vmm_off idle vmm_mapped=0 pool_cached=2567162688 trims=0; vmm_on idle vmm_mapped=0 pool_cached=2622667584 park_compact=30
```

Causes, placed from the receipts:

- **P1 fails in both orders, on the same row.** `X-30720-r3-t3` on the `on` arm differs from `off` and from its own cold
  twin; `off`'s row equals its cold twin. Three facts from the rows' `usage.prompt_tokens_details.cached_tokens` and
  the pool-hit deltas:
  1. **The workload's premise was wrong.** 1.2 said a `max_tokens=1` turn parks "exactly the prompt". It parks the
     prompt plus the one generated token: the resumed rows read `cached_tokens` = L + 1 at turn 2 (6,145, 30,721,
     122,881) and L + 65 at turn 3 (6,209, 30,785, 122,945). A turn resumes only when that generated token equals the
     next stream id (the exact-extension rule): 20 of the 30 shape-X turns 2 and 3 resumed on `on` (19 exact
     extensions and one on-grid rewind at 104,832; 19 `park-compact grow:` lines). P1 (b)'s `x_resumed_on=30` counted
     rows, not resumes.
  2. **The `off` arm never resumed at 6,144 or 30,720.** Every one of its shape-X turns reads `cached_tokens=0`: the
     plain-parked entry keeps turn 1's ladder cap, which does not fit the next turn's charged cap
     (`plain_resume_cap_admits`, the legacy contract). At 122,880 `off` rewound to an on-grid checkpoint (104,832 to
     108,800). The door's compacted entry regrows at the request's cap, so `on` resumes where `off` cannot. The door
     comparison of 1.4 therefore compared `on`'s resumes against `off`'s cold primes, not resume against resume.
  3. **The resume `on` performs is a second numeric program for the same request.** Every exact-extension resume
     starts at a state after a DECODED token and off the GDN prime grid (L + 1), while the cold prime of the same
     prompt primes that position inside a grid-aligned chunk. This is the measured law of `grid_align_boundary`
     (an off-grid split diverges from the split row on) and the "prime vs decode" pair of the one-numeric-program
     rule. 18 of the 19 exact-extension resumes kept the cold digest; `X-30720-r3-t3` (resumed at 30,785) did not,
     deterministically in both orders. The on-grid rewind kept it. The crossing belongs to the continuation pool's exact-extension resume, not to the compaction copy:
     the door makes it reachable on this workload because the compacted entry always fits.
- **P2 (green) PASS in both modes; the red arm did not exercise the defect.** The forged OOM fired on
  `X-6144-r0-t1`, not on turn 3 as addendum B predicted: a `max_tokens=1` turn decodes its one token (fact 1). In the
  non-batching mode the injection point is the session's first step, which is its prime: the session is not
  `prefill_done`, so it cannot park on either tree, and the red arm reads the same as green. The errored-session fix
  (`1c1e5cd49`) keeps its CPU census as its only evidence of a red; the red arm needs an injection that lands after
  the prime.
- **P3** (a reading): the park-time copy is 1.2, 2.6 and 8.2 ms at the three fed lengths (N=100 each).
- **P4** (a reading): resumed-turn E2E p50 1,936 and 1,968 ms on `off`, 685 and 700 ms on `on` (N=45 each; `off`'s
  rows are mostly cold primes, fact 2). At idle `on` holds more pool-cached bytes (9.68 GB against 6.74 GB).
- **VMM-PAIR** (outside the rule): no digest differs; `vmm_mapped=0` at idle on both. At 6,144 with `max_tokens` 1 to
  32 the planes are below `on_demand_pays`, so the allocator stayed pooled and the pair shows nothing about the two
  doors together.

The rule of 1.6 reads **FAIL (no reading)**: P1 fails. The cause is the continuation pool's resume program (fact 3),
which the door makes reachable, plus a workload that cannot compare resume against resume (facts 1 and 2). The door is
revised under a new pre-registration (addendum D, 1.11; 2.2 corrects fact 3).

### 2.2 Correction to 2.1's fact 3 (2026-09-25, from the records, before addendum D)

Fact 3 called the verbatim-extension resume "a second numeric program for the same request" as if it broke a promise.
The repo documents it as a named residual: `docs/SERVING.md` ("What the grid law still does NOT promise:
verbatim-extension continuation resumes keep decode-computed rows whose arithmetic a cold prefill never reproduces
... That path carries the documented cached-hit-vs-fresh-prime near-tie contract"), measured by `primepath --hist`
(darklanes `research/multiturn-cache-20260821/LONGCTX-EXACTNESS-20260821.md` P3: logits differ, maxdiff 0.45,
flips only at near-ties). `X-30720-r3-t3` is one such flip: one in 19 exact-extension resumes here. So 1.4's
cold-twin clause was a registration error of this lane for shape X, not a defect the door or the pool must fix. The
tension between that documented contract and `CLAUDE.md`'s one-numeric-program rule is recorded for the owner (OWED,
owner decisions) and not worked by this lane. Addendum D (1.11) re-registers P1 and P2.
