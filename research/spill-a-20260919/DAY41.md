# WP-A day 41: OWED item 5, design K's promote-side fail-closed arms in serving-shape fault cells

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Rig: the local RTX 5090 Laptop GPU; the target card's half rides
a later sitting. Every cell `executed-not-qualified`. Behind `MEMRA_KV_HOST_CONTRACTS` (default OFF).

## 1. Pre-registration (committed before any day-41 code)

**The gap** (`OWED.md` item 5). Design K (`DAY34.md`) hands an off-tick promote's KV source views to the hash helper as
a `Sources` job, and the promote's settle latches the tier on three arms (`host_kv_planes_settle_promote`, step 5b):
`tier hash helper gone before the H2D checksums of ticket seq=S landed (..)` (the sources reply channel closed), `tier
H2D checksums never landed: ticket seq=S's N sources handed X ms ago, past the 10s deadline` and `tier H2D checksum
reply seq=R with D digests does not describe ticket seq=S of N sources`. The fault gate's `hash-helper-gone` exits the
helper on its FIRST job of any kind and `hash-never-lands` discards a `Hash` reply; in the gate's shapes the first job
is a demote's `Hash`, so none of K's three arms is reached by a serving-shape cell (only a GPU unit cell covers the
corrupt-lease refusal).

**The design.** Three one-shot values on the existing `MEMRA_KV_HOST_FAULT` row (no new name), read once at boot into the
helper like the day-28 two, each keyed on the helper's FIRST `Sources` job (a demote's `Hash` jobs before it run clean):
`sources-helper-gone` (the helper exits on it, the job dropped with it), `sources-never-land` (the helper hashes it and
discards the reply), `sources-foreign-reply` (the helper replies with the job's `seq + 1`). The helper prints one line
when it applies one (`[prefix-host] hash helper fault (MEMRA_KV_HOST_FAULT=<value>): ..`). No production path changes.

**The cells** (`tools/kv-host-contract-fault-gate.sh`, one per value, two boots each: door ON with the fault, then door
OFF as the byte reference; the promote cells' shape): r1 P_A seeds E_A; r2 P_B evicts E_A (a clean demote whose `Hash`
job lands and publishes); r3 P_A hits E_A on the host, its promote submits and hands its `Sources` job to the helper,
which takes the fault; r4 P_B. Checks: four completions served; exactly one helper-fault line; exactly one `TIER
DISABLED` line naming the arm's own words (above); no `[prefix-host] promote: ` publication in the boot; the parked
request re-admitted (r3 returned); `hash helper joined (the tier latched off)` once (the `never-land` and `foreign-reply`
helpers are idle when the tier latches; the `gone` helper has exited); no `leaked` or `Capacity` wording; r1 to r4
byte-equal to the door-OFF boot.

**Acceptance.** The three cells green in the fault gate's default and plain arms on the 5090 (the whole gate `ALL
GREEN`), and on the target card in a later sitting; a CPU cell for the three values' parsing and their keying on the
`Sources` job (a `Hash` job before it runs clean); the server lib suite and clippy clean.

**What each card decides.** Each card its own gate. Nothing is timed.

**Budget.** 0.25 agent-day.

## 2. The 5090 half, as it ran (`rtx5090-day41/`)

- The tip binary `a8e73bd2f377b66f..` (`7f3558229`, the day-41 code `4ca4bb36e` with later records only), one hold
  18:15:42Z to 18:22:07Z after two bounded busy attempts; `run.sh` ran the fault gate's default and plain arms through
  `--external-lock 9`.
- Verbatim: `gate fault-default rc=0 KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (229 ok, 0 FAIL) and `gate fault-plain
  rc=0 KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (229 ok, 0 FAIL); every check of the three new cells green in both arms.
- The arms' own lines (default arm): `hash helper fault (MEMRA_KV_HOST_FAULT=sources-helper-gone): the Sources job of
  ticket seq=4 (18 views) is dropped and the helper exits` then `TIER DISABLED: tier hash helper gone before the H2D
  checksums of ticket seq=4 landed (the sources reply channel closed)`; `.. sources-never-land): .. is hashed and its
  reply discarded` then `TIER DISABLED: tier H2D checksums never landed: ticket seq=4's 18 sources handed 10001.0ms ago,
  past the 10s deadline`; `.. sources-foreign-reply): .. is answered as seq + 1` then `TIER DISABLED: tier H2D checksum
  reply seq=5 with 18 digests does not describe ticket seq=4 of 18 sources`. No promote published in any of them, the
  helper joined at each latch, r1 to r4 byte-equal to door OFF.
- CPU: `day41_the_sources_faults_key_on_the_first_sources_job` (the census of the order and, behaviourally, a `Hash` job
  answered cleanly under each `Sources` fault), server lib 911 passed, clippy clean.
- **Verdict**: `DAY41 K-ARMS (5090) default ALL GREEN, plain ALL GREEN, the three arms each latch typed with no
  publication -> 5090 PASS; target card owed` (the next sitting's fault gate carries the cells).
