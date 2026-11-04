# On-demand KV planes are the serving default on the RTX PRO 6000 Workstation Edition (2026-11-04)

**Chosen:** on the full-power RTX PRO 6000 Blackwell Workstation Edition, the server builds the
full-attention K/V planes of a covered session (the MTP spec route and the plain route on one
device, no SWA ring, no latent planes, no ppN or TP stage ownership) as on-demand VMM planes. Each
plane reserves its whole capacity as a fixed virtual range and maps physical granules only as the
session's rows grow, inline on the owner thread. Releases run after an owner-stream event and are
reaped at the tick top. The admission gate subtracts the reserved-but-unbacked bytes of active
on-demand sessions (`vmm_owed`). No numeric program changes: the same kernels touch the same
bytes at addresses a session keeps for its life. `MEMRA_KV_ALLOCATOR` and `MEMRA_KV_VMM_GROW`
are deleted. `MEMRA_KV_VMM_FAULT` stays as the fault door of the A5 cells. Code:
`memra_engine::parallel::HardwareTarget::is_pro6000_workstation`, `crates/memra-server/src/kv_vmm.rs`.

**Rejected:**

- Pooled planes on that card, the previous default. A pooled session holds its whole capacity for
  its life. On the deciding cell the on-demand arm left about 24 GB more driver-free memory at
  idle on the mix (69.8 against 45.7 GB plain, 67.8 against 43.7 GB spec; DAY37 2.5 A4-IDLE).
- The helper (mapper-thread) placement on that card. Stage 0 on the first box read `inline`
  (`busy_p95_sum_us=55.3`) and the cell was run and read with inline grows.
- The on-demand planes on the RTX 5090 class. Its reading under the decision rule is FAIL (no
  reading): A1's admit-mem-burst cell went red on the pooled control arm (DAY37 2.7), and the
  repro placed that red on the missing DAY39 addendum-B booking terms, not on either allocator
  (DAY37 2.8). The rerun on the addendum-B tree (r5, `rtx5090-day37/build-r5.sh` and
  `chain-r5.sh`) has not read, so the 5090 keeps pooled planes.
- The Max-Q Workstation and Server editions and every other card class. They carry no stage-0
  probe and no deciding cell.
- Keeping `MEMRA_KV_ALLOCATOR` as a rollback seam. Owner, 2026-11-04: winners with positive
  receipts become defaults keyed to the card each receipt came from, and their switches go.

**Measurement that settled it:** `research/spill-b-20260919/DAY37.md`, the pre-registered rule of
1.7 on the r4 source across two sittings on two RTX PRO 6000 Blackwell Workstation Edition boxes
(2.5 and 2.6). A1 PASS (every gate cell on both allocators, A1-MIX 45 of 45 on both paths and
both orders, A1-STREAM 10 pairs with 0 differing rows), A2 GROW-G1 PASS (28 grows, drift 0,
tokens, logits and state equal), A5 PASS (ENSURE, BUILD1, BUILD64; MAPPER does not apply under
inline), A6 PASS, A7 PASS (no OOM line, no 503, no grow failure), A3 (i) PASS (inline grow
`owner_us` p99 139 us, N=4,420, and 141 us, N=1,714, under the 500 us bound) and A3 (ii) PASS
(TTFT p50 ratio 1.0229, ITL p99 0.9999, TPOT p50 0.9877, throughput 0.9998; N=10 per arm). The
rule reads PROMOTE-ELIGIBLE for that class. The two boxes' stage-0 probes disagree (55.3 against
271.2 us busy p95, on opposite sides of the rule's 100 us line); the serving grows read the same
on both under inline, which is the placement the cell qualified.

**What would change it:** the 5090 r5 cell reading PROMOTE-ELIGIBLE on its registered tree adds
the RTX 5090 class with its own stage-0 placement (`helper`, DAY37 2.1). A stage-0 probe and a
deciding cell on the Max-Q or Server Edition add that edition. `docs/decisions/KV-PHYSICAL-RECLAIM.md`
keeps the gate-only demote and restore record this serving default grew from.
