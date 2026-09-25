# WP-A day 47: OWED item 6, the by-reference demote routes off the tick (design V)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. OWED item 6 (Move 1 item 3, `OWNER-THREAD-OFFLOAD.md` day 17: "the
admission reclaim flush (`evict_all_demoting`), the pause sweep and the handoff keep the blocking program; ... The pause
sweep's park half holds live device state and needs the `Demoting` state to protect the park until publication"; ruling
47). Every cell `executed-not-qualified`. Behind `MEMRA_KV_HOST_CONTRACTS` (default OFF) and, for the pause sweep,
`MEMRA_KV_PAUSE_DEMOTE` (default OFF). No new flag.

## 1. Pre-registration (committed before any V code)

**The four by-reference call sites today** (`ContractD2h::OnTick`: the D2H, its receipt, the owner's wait and the bind
all inside the tick; every tenant waits for them):

1. The admission reclaim flush (`evict_all_demoting`, under `MEMRA_ADMIT_BY_MEMORY`): demotes resident entries up to
   the arrival's shortfall, then drops them, so the arrival admits in the same tick.
2. The handoff export (`host_handoff_export`): drain-demote, then write the file.
3. The pause sweep's shape 1 (the exact-fed plain continuation park): `prefix_snapshot` of the park into a fresh entry,
   the demote of that entry, and only after `Demoted` or `Evaporated` the park's release.
4. The pause sweep's shape 2 (the deepest resident device prefix entry prefixing the tape): the demote by reference,
   and only after `Demoted` or `Evaporated` its removal.

**What moves, and what does not, with the reason for each (stated before any code):**

- **Sites 3 and 4 move off the tick (design V).** A pause fires while tenants decode; today's blocking demote of a 27B
  entry holds every tenant for its copy and its bind (the stall cell below prices it).
- **Site 2 stays on the tick, by its contract.** `Cmd::ExportHostHandoff`: "serve-deploy calls it on the DRAINED blue
  slot after the edge flip, where a stalled tick has no one to stall; a slot with active/queued requests refuses unless
  `force`." The export must hold every entry before it writes; there is no tenant to protect.
- **Site 1 stays on the tick in this design, and goes to the lead as a question.** The flush exists to free the
  arrival's shortfall before this tick's admission decision. An off-tick D2H frees nothing before it lands (the planes
  return to the pool at the settle), so an off-tick flush means deferring the arrival until the landings, which is the
  memory-admission door's own decision (`admit_memory::decide`'s defer arm), a door lane B is measuring under the owner's
  2026-09-23 call (decide-by 2026-10-07). Whether item 6 should build a deferring flush inside that door is the lead's
  call; V does not touch it.

**Design V.**

1. **Shape 1: the snapshot demotes on the sink's route.** The boundary snapshot is a fresh entry the sweep owns (never
   resident), the sink's shape exactly: `host_demote_prefix_ref(.., ContractD2h::OffTick)`, the shell with the pending
   demote (`pending.dead`). The park stays where it is. The pending demote carries a `ParkRelease { pool_key, tape }`;
   its publication (`Demoted`) queues the release on `HostPrefixCache.park_releases`, and the run loop's tick top drains
   the queue right after its settle calls: the park in `reuse[pool_key]` whose `fed` equals `tape`, if still there, is
   released (`pause demote: plain park released off the tick ..`); a park the session consumed meanwhile is not there,
   and nothing is released. `Evaporated` (decided at the submit) releases at once, as today. Any failure queues nothing:
   the park stays (today's fail-closed rule).
2. **Shape 2: the entry leaves the device index into the sink's route, and comes back on a failure.** `pause_px_decision`
   unchanged (a pin or a post-arm touch loses). The chosen entry is removed from the device index (`px.remove_at`) and
   demoted as the sink's evicted entry (`host_demote_prefix_entry` with a `reinstate` mark on the pending demote). A
   request that hits it while it is `Demoting` parks (design P, days 29 and 38) and promotes after the publication,
   which is what today's blocking program gives that request too (it waits the whole demote behind the stalled tick,
   then finds the entry on the host). Every failure exit that leaves the shell whole (a typed refusal before or after
   the submit, a leaked ticket, a `Hashing` latch, the `flip-demote` fault) queues the whole shell on
   `HostPrefixCache.reinstate` instead of dropping it, and the tick top re-inserts it (`insert_demoting`, `pause demote
   failed: entry reinstated ..`): today's rule that a failed demote leaves the entry resident. `SourceQuarantined` (the
   planes stay with the transfer engine; the shell is not whole) drops it, as today's shape 2 does.
3. **The counters.** `pause_demotes` counts at the release (shape 1) and at the publication (shape 2); `pause_cancels`
   stays the sweep's (a candidate that submitted nothing).
4. **Censuses.** Both shapes call the off-tick route; the release is queued only on `Demoted`, drained at the tick top
   after the settle calls, and matches `fed == tape`; the reinstate queue is filled only on the whole-shell failure exits
   (their count pinned) and drained at the tick top; no pause path calls `ContractD2h::OnTick` any more; the flush and the
   export still do (their two sites pinned).

**Acceptance, stated before any code:**

- (a) CPU: the census above; unit cells for the release matcher (a consumed park releases nothing; an exact twin
  releases one) and for the reinstate queue's exits (a whole shell reinstated, a quarantined one dropped).
- (b) The gates ALL GREEN on each card: identity x4, failure OFF and ON, the fault gate default and plain, hit OFF and ON
  (the demote sink's routes are shared), and a new gate `tools/kv-host-pause-demote-gate.sh` (below) in the plain and
  the default boot.
- (c) The pause stall cell (`stall_cell.py --mode pause`, below), base against V, `--n 5`, 20 boots (o1 = base V x5, o2
  reversed), door ON and `MEMRA_KV_PAUSE_DEMOTE=1 MEMRA_KV_PAUSE_DEMOTE_MS=250` in both: per order, V's median
  pause-window stall at most a quarter of base's, and the tenant's text identical across every run of both arms.
- (d) In the same cell: every armed candidate demotes in both arms (the count of `pause demote: .. released` lines per
  boot equal between the arms' boots) and V's pause publication lands (every V boot's `demote digests landed` count
  covers its pause demotes).

**The gate** (`tools/kv-host-pause-demote-gate.sh`, one boot per cell, door ON, `MEMRA_KV_PAUSE_DEMOTE=1`,
`MEMRA_KV_PAUSE_DEMOTE_MS=300`; each cell byte-compared with a pause-OFF, door-OFF boot of the same turns): turn 1 is a
chat request with two tools declared and an agent system prompt that asks for a tool call (the darklanes pause battery's
shape; the gate's first check is that turn 1 ended in a tool call, `finish_reason: "tool_calls"`, else it fails typed:
the cell would measure nothing); turn 2 appends the tool call and a tool result.

- `clean`: after turn 1, the `pause demote` release lines for the boot's shapes (plain boot: shape 1 and shape 2;
  default boot: shape 2, spec parks being out of scope by design); turn 2 then reports `cached_tokens > 0` (a host hit
  that promotes); turns 1 and 2 byte-equal to the reference.
- `race`: `MEMRA_KV_HOST_FAULT=d2h-delay` (the first demote of the boot, the pause demote, held 3 s unlanded): turn 2
  is sent at once after the pause fires; it parks on the `Demoting` entry and promotes after the publication; byte-equal.
- `failure`: `MEMRA_KV_HOST_FAULT=contract-presubmit` (the first contract D2H refused before any op): the pause demote
  fails typed; the park is kept (`host copy did not publish; park kept`) and the shape-2 entry reinstated (`entry
  reinstated`); turn 2 is a device hit; byte-equal; the tier on.

**The stall cell** (`stall_cell.py --mode pause`, the five earlier modes and the two long modes byte-for-byte
unchanged): the tenant as ever; at its 24th token the intruder posts turn 1 of a fresh tool conversation (the gate's
tools and system prompt, a run-numbered ask so each run's entry is new); the pause fires 250 ms after it returns while
the tenant still streams. Per run: the pause window starts 200 ms after the intruder's response returned, and the
pause-window stall is the tenant's largest inter-token gap inside it minus the run's ITL p50. A run whose window holds no
`pause demote` line in the server log is not counted (and a boot with fewer than five counted runs is incomplete). Reader
`day47-reading.py`.

**Predictions.** Base's pause-window stall is the blocking demote's duration (tens of ms on the 9B's 64-token-class
entries, above 100 ms on the 27B); V's is its pre-submit (about 1 to 2 ms, the 27B's long entries up to about 20 ms). (c)
holds by a wide margin; (d) holds.

**What each card decides.** Each card its own (a) to (d). The target card runs first while the 5090 is down.

**Budget.** 1.5 agent-days: V's code and its censuses 0.4, the gate 0.4, the stall mode and reader 0.2, the sittings
0.5.

**Order.** V's code is written after design S3's target reading (item 4, DAY46), so a revert of either stays one clean
commit.

## 1a. Amendments before any V code (mechanics and the gate's cell readings; no clause or bound changes)

1. **No pause shape Block-settles.** Every demote route settles a pending demote first (`host_demote_settle_pending(..,
   Block, "a second demote")`, the one-batch rule), so a sweep that fired shape 1 and then shape 2 in one tick would hold
   the tick for shape 1's whole copy and hash, the stall V removes. So a shape whose demote cannot start because a demote
   is `Demoting` (shape 1's own, or the sink's) does not start: the candidate stays pending with that shape still owed
   and the sweep tries it again at a later tick (shape 1 before shape 2; a candidate whose owed shapes all resolved
   leaves the list; a later touch or pin cancels shape 2 as today). Census: no pause path reaches a `Block` settle.
2. **The gate's race cell reads per boot.** In the plain boot the pause demotes shape 1's snapshot first, so turn 2
   sent during the held copy resumes from the park itself (the park is released only at the publication); the
   publication then finds the park gone and releases nothing (`plain park already gone at the publication`), and shape
   2's demote runs at a later tick. In the default boot (spec parks out of scope) shape 2 is the first demote, and turn
   2 parks on the `Demoting` entry and promotes after the publication. Both byte-equal to the reference.
3. **The gate's failure cell reads per boot.** `contract-presubmit` refuses the boot's first contract D2H: in the plain
   boot, shape 1's (`host copy did not publish; park kept`; shape 2 then demotes cleanly); in the default boot, shape
   2's (`entry reinstated`). Both byte-equal, the tier on.
