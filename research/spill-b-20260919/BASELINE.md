# WP-B day-1 baseline — 2026-09-19

> Historical day-1 baseline retained. Day-2 migration and current boundaries: `DAY2.md`, `HOSTPREFIX-EXTENSION.md`, `CELLS.md`.

Repository `avifenesh/memra`; base `c5a33b14ff7b6c75a3cd808dbc0ee4f10aa8b33f`.
Branch `lane/spill-b-20260919`, isolated worktree `../wt-spill-b`.
This is CPU control-plane work, not RAM/SSD active attention qualification.
No model execution, GPU access, serving-instance interaction, SSH retry, push, PR or tag.
The lead reported two unsuccessful bounded 5090 connection attempts; this session did not repeat them.

## Directly inspected baseline (line numbers at base, before module exports)

- `crates/memra-server/src/worker.rs:7370-7457`: HostPrefixEntry mirrors KV, recurrent
  conv/SSM, logits, hidden anchor, optional draft/DFlash/GLM state. HostPrefixCache is the
  existing byte-LRU pinned tier, with budget, tenant share/purge, model generation and
  allocation-failure latch. Extend this cache; do not build another payload owner.
- `worker.rs:8551-8607`: admission promotion checks model generation, reconstructs an
  ordinary device PrefixEntry, optionally checks its state digest, and retains the host twin.
  Transfer completion currently sits inside synchronous restore, not scheduler tickets.
- `worker.rs:6244-6279` (inventory D K4): existing `(model, namespace)` device prefix map,
  exact-token matching and pin/fanout semantics remain authoritative.
- `worker.rs:8631-8675` (inventory D S6): restart handoff is whole-image persistence, NOT
  an online NVMe active-cache implementation; preserve that distinction/version boundary.
- `crates/memra-server/src/admit_memory.rs:192-284`: live free device, unleased prefix
  bytes and free host bytes select Admit / bounded DemoteThenAdmit / Defer / Refuse.
  These are prefix-reclamation decisions, not permission to read attention operands off SSD.
- `crates/memra-kv/src/lib.rs:12-18`: the ordinary admitted KV is q8_0 K (34 bytes/32)
  and q5_1 V (24 bytes/32). Removed format switches are not a selectable f16/FP8 option.
- `memra-kv/src/lib.rs:415-477`: KvDev allocation/copy seam and contiguous per-layer
  byte planes with native token/head/dim order and stable len/base counter addresses.
- `memra-kv/src/lib.rs:2950-3010`: rollback truncates full KV, updates counters in place,
  restores conv/SSM and requires accepted-token replay by the same native caller. Lapped
  SWA checkpoint refuses. A snapshot is not a faultable active block manager.
- `docs/SERVING.md:1724-1908`: prefix/host policy and tenant isolation/handoff context read.
  Historical prose about generic host exclusions is not a new capability decision; the GLM
  extension already exists (inventory D K7). No unrelated docs cleanup in this lane.

## Extension boundary

Day 1 adds only `memra-kv::record` and `memra-kv::tiered`: compiled proposed contracts,
identity, exact layout validator, leased restore state machine, active epoch guard, policies,
and fake backend tests. No server/engine callsite changes yet; HostPrefixCache still owns all
runtime host images. Shared definitions will MOVE (not be copied) to lead-owned memra-tier
at freeze. No branch-local memra-tier scaffold was needed.

Next integration: HostPrefixCache implements the object-directory/lease adapter, retaining
its namespace, tenant caps, purge and handoff semantics. The worker's admission owns a
TierAdmissionPlan; only READY reaches its existing prefix restore/suffix prime boundary.
Active materialization is a separate adapter over existing attention operands (see proposal).
A shared governor must replace the isolated CPU test ledger before B/C integration.

## Evidence interpretation

Opaque f32/FP8/FP4 fixtures exercise bytes/alignment/required pools only. They neither add a
codec nor qualify a model encoding. Native q8_0/q5_1 byte fixtures use exact record sizes;
real Qwen3.8-27B state/token/logit gates remain queued in CELLS.md. The unrelated paused
model lane is not an oracle or a test cell.

Historical inputs read: plan 08 §§A–C, WP-B, Session B, F; inventory D K1–K7, S6, V1–V4;
reference E L11/L23/V5/V6/P1 and HiCache storage requirements; AGENTS.md, ROUTER.md,
research/INDEX.md, benchmarks.md, private GPU corpus index and gate-craft rule rows.
External systems are design references only; no external runtime dependency added.
