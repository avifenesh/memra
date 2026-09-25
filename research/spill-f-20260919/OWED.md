# WP-F owed ledger

Opened 2026-09-25 on the owner order of 2026-09-24: "every improvment and tunning should be
done, no shortcut or fast path". Every open item carries its source. Nothing here is dropped
for being slow; an item leaves only with a receipt, a verdict, or an owner decision.

Status words: **open** (work not started), **registered** (pre-registered, not run),
**blocked-box** (needs rented hardware the lane cannot acquire), **blocked-card** (needs the
local RTX 5090 after the owner's reset), **closed** (receipt or verdict named).

## M1: storage provenance and spill speed on proven local NVMe

| # | Item | Source | Status |
|---|---|---|---|
| 1 | M1 proof: the spill path's writable storage is physical local NVMe, proven in the running kernel from the mount to the PCI function, fail-closed | owner order 2026-09-24; lead prompt 2026-09-25; `NVME-DECISION.md` (parked NO-GO 2026-09-20); lead `HANDOVER-20260920.md` item 6 | registered in `M1-PREREG.md` part A; tool ready (item 2); run blocked-box |
| 2 | Proof tool `m1-nvme-proof.py` with fixture red controls for every fail-closed branch, plus live controls on this rig | this ledger; `M1-PREREG.md` A | **closed** 2026-09-25: 24/24 fixture and 7/7 live controls (`M1-PROOF-CONTROLS.md`) |
| 3 | B0 device envelope on the proven path (fio: positioned-read threads vs io_uring vs libaio at the worker's request sizes and depths; sequential write with fsync) | `M1-PREREG.md` B0; CLAUDE.md "Hy3 spilling" (measure the storage-to-compute pipeline; 70% headroom rule in `spill-a-20260919/CELLS.md`) | registered; blocked-box |
| 4 | B1 KV ObjectStore storage cells (`storage-bench` roundtrip and restore, buffered vs O_DIRECT read vs O_DIRECT read+write, six sizes, cold host cache) | `spill-a-20260919/CELLS.md` M1-row/bulk cells; `M1-PREREG.md` B1 | registered; item 9 landed; collector route needs item 14 applied; blocked-box |
| 5 | B2 KV host-tier handoff export/import on the proven path (fsync+rename export, cold import, restored completions byte-identical to cold) | `M1-PREREG.md` B2; `MEMRA_KV_HOST_HANDOFF` row in `docs/FLAGS.md` | registered; driver selection owed (item 13); blocked-box |
| 6 | B3 expert-bank spill cells: `MEMRA_SPILL_IO` mmap (random and normal advice), pread, worker depth 2, worker depth 16 (the measured baseline), direct, three cache regimes, round-robin balanced N=10 per arm | CLAUDE.md "Hy3 spilling" (compare O_DIRECT, io_uring and mapped access only against the measured worker baseline; stages measured together); `M1-PREREG.md` B3 | registered; needs items 7, 8, 10, 11, 12; blocked-box |
| 7 | Direct arm is 100% mmap fallback on the pinned artifact: 0 of 31,488 expert slices pass `direct_extent_aligned` (every tensor base offset is misaligned at `general.alignment=32`). Owed engine improvement: aligned over-read in the direct worker, byte identity against the mmap oracle | new finding, `m1-prereg/direct-alignment-census-qwen36-35b.json`; `per-expert-quant/README.md` ("aligned over-read or an aligned sidecar") | **landed CPU-verified** 2026-09-25 (`owed7/RESULTS.md`); GPU gates owed (box, then 5090) |
| 8 | Per-stage spill counters (worker read time, owner wait time, H2D submissions, over-read bytes) in the existing `MEMRA_SPILL_STATS` snapshot and the pool's drop line; no new env read | CLAUDE.md "Measure the stages together"; `PreadStats` carries counts only | **landed CPU-verified** 2026-09-25 (`owed8/RESULTS.md`); GPU values come with B3 |
| 9 | `storage-bench` separates read-call time from verification time (`io_ns`, write/commit time) | `StorageSample.io_ns` is `None` today; `M1-PREREG.md` B1 | **landed CPU-verified** 2026-09-25 (`owed9/RESULTS.md`) |
| 10 | Host/storage 250 ms sampler: `/proc/diskstats` for the proven leaf set, `/proc/meminfo`, `/proc/stat`, target-process `/proc/<pid>/io`; runs inside the collector-executed worker so the lock covers it | `spill-a-20260919/CELLS.md` "No telemetry means no score"; `M1-PREREG.md` B | open |
| 11 | Cache-regime helper: file-scoped `POSIX_FADV_DONTNEED` plus `mincore` verification (cold), full buffered read plus `mincore` (warm), and the bounded page-cache regime (lane-owned mlocked balloon with a MemAvailable floor) | `spill-a-20260919/IO-BASELINE.md` cold/warm method; `M1-PREREG.md` B3 regime (iii) | open |
| 12 | M1 cell runner (round-robin arm order, fresh process per visit, tee raw logs before parsing, identity check against the proof receipt before and after every visit) | `M1-PREREG.md` B | open |
| 13 | Pick the request driver that fills the host tier to a fixed byte size for B2 (lane B's restore-identity or twin gates) | `M1-PREREG.md` B2 | open |
| 14 | Collector storage label is a name match: `tools/tier-battery.py::capture_storage` marks `nvme-ancestry` when any `nvmeXnY` name appears in `lsblk -s`. That passes an emulated or fabric NVMe and can refuse a bind mount whose findmnt source carries a `[subdir]` suffix. Proposal to D and the lead: scored storage cells bind to the M1 proof receipt's identity triple instead | new finding; `tools/tier-battery.py` lines 503-527 | **patch and test delivered** 2026-09-25 (`owed14/README.md`); applying it is the lead's routing to D |
| 15 | B4 serving-shape cell (memra-server, c=1 and c=4, TTFT/E2E/TPOT/ITL p50/p95/p99, request and token throughput) for the worker baseline and any B3 candidate | CLAUDE.md "Active accelerator owners" serving-claim fields; `M1-PREREG.md` B4 | registered; blocked-box |
| 16 | io_uring decision input: B0's ring vs positioned-read threads at the worker operating point; implementation of memra's ring stays with the lead's assignment of `spill-a-20260919/IO-URING-PROPOSAL.md` | IO-URING-PROPOSAL.md (deferred until the bounded-pread baseline is measured); `M1-PREREG.md` B5 | registered; blocked-box |
| 17 | Mapped pinned-host (zero-copy) cold read-once arm: not implemented in the engine | CLAUDE.md "Hy3 spilling" pipeline list; `per-expert-quant/README.md` ladder rung 5 | open (candidate; needs B3 baseline first) |
| 18 | KV handoff O_DIRECT arm: not implemented (the handoff writes through a 4 MiB `BufWriter` and reads through `BufReader`) | `crates/memra-server/src/worker.rs` `host_handoff_export` / `open_host_handoff_import` | open (candidate; needs B2 baseline first) |
| 19 | 5090 halves: prove the local rig's spill path, then run the B3 subset there after the owner's reset | CLAUDE.md "Per-hardware arm selection" (both rigs before a default) | proof part **closed** 2026-09-25 (artifact store and root filesystem are `nvme-local-direct`, `M1-PROOF-CONTROLS.md`); cells blocked-card |
| 20 | An above-RAM expert-bank artifact for a storage-bound steady state without a balloon (the pinned 35B bank is 15.6 GB and fits every candidate box's page cache) | `M1-PREREG.md` B3 regime note | open (lead picks the artifact; lead-owned pins) |
| 21 | Historical "local NVMe `/scratch`" spill numbers from 2026-07-10 carry no in-repo ancestry proof. Not relabelled here; flagged to the owning lane | `per-expert-quant/evidence/spill-prefetch-cloudbox-20260710.md`, `spill-worker-ab-cloudbox-20260710.md` | open (flag only; not F-owned) |

## G2: host/device copies

| # | Item | Source | Status |
|---|---|---|---|
| 22 | The G2 matrix registered in `CELLS-ENVELOPE.md` has ten sizes; D's scored day-10 campaign on the target class covered five (4 KiB, 64 KiB, 1 MiB, 16 MiB, 256 MiB). Unrun: 16 KiB, 256 KiB, 4 MiB, 64 MiB, 1 GiB, through D's runner | `CELLS-ENVELOPE.md`; `spill-d-20260919/DAY10-VERIFICATION.md`, `G2-RESULTS.md` | registered; can ride the M1 box (B6); blocked-box |
| 23 | G2 5090 half (any size) | `CELLS-ENVELOPE.md` (rented-development 5090 envelope) | blocked-card |

## Closed or retired

| # | Item | Source | Verdict |
|---|---|---|---|
| 24 | Note to D: nested probe must stay in the collector worker's process group | `H2D-RESULTS.md` | closed: `tools/tier-envelope.py` launches the visit with `shared_group=True` (line 99) |
| 25 | BOX2 pre-destroy checklist | `BOX-HYGIENE.md` | retired: BOX2 destroyed 2026-09-20 (lead `HANDOVER-20260920.md`); its receipts are banked |
| 26 | `h2d_probe --copies` N=1 plumbing cell | `H2D-RESULTS.md` | closed: 32 identity-clean visits, `executed-not-qualified`; full G2 is D's runner |
| 27 | The parked VM-spend question ("does `vms_enabled` give a guest a physical NVMe?") | `NVME-DECISION.md` | closed 2026-09-25: the route's own documentation says local volumes attach to docker instances only, not VMs, and its VMs are KVM guests with virtual disks, so a VM cannot pass part A; the docker-plus-local-volume route replaces it (`M1-PREREG.md` C) |
