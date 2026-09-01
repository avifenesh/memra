# Box incident, the second bench box, 2026-08-28 — and why this lane stood down

(Box identity lives in the private ops repo, per the lane's own convention in `BRINGUP.md`.)

Written from the box's own logs, not from inference. Every timestamp is UTC and every claim below
has a command behind it.

## Timeline

| time | event | evidence |
|---|---|---|
| 13:30:19 | box up on kernel `6.8.0-1061` | `last -F` |
| 13:49:10 | unattended-upgrades runs and does **nothing** | `/var/log/unattended-upgrades/unattended-upgrades.log`: "No packages found that can be upgraded unattended" |
| 14:10:35 – 14:21:23 | **five `Xid 31` MMU faults**, all from `glm5-hyper-ppn-` (the PP lane's gate binary), on PCI `0000:32:00` | `journalctl -b -1` |
| ~14:14 | this lane's first GPU touch (the fixture gate suite) — **after** the first three Xids | build log finished 14:14; Xids at 14:10:35 / 14:11:15 / 14:11:58 |
| 14:16 – 14:23 | this lane's `IDOFF` server load + identity rows, 19.7 GB on card 0 | `cell-IDOFF.log` |
| 14:26:24 – 14:26:50 | an `apt` run installs kernel `6.8.0-1063` + ~30 package upgrades. No `Requested-By` line, and unattended-upgrades had already declined at 13:49, so it was neither this lane nor unattended-upgrades | `/var/log/apt/history.log` |
| 14:28:49 | **clean, orderly reboot** (`systemd-logind: System is rebooting`, full graceful shutdown, `Finished System Reboot`) — not a panic, not an OOM, no `oom-kill` anywhere in the boot | `journalctl -b -1` |
| 14:29:09 | box up on `6.8.0-1063`. `nvidia` DKMS module exists only for `1061`, so **both GPUs are dead**: `modprobe: FATAL: Module nvidia not found` | `dkms status`, `uname -r` |
| 14:30:26 | `apt-get install -y linux-headers-6.8.0-1063`, **Requested-By: ubuntu (1000)** — another operator is already remediating | `/var/log/apt/history.log` |
| ~14:32 | `nvidia/595.91.07` rebuilt for `1063`; `nvidia-smi` healthy again. **Not this lane's doing** | `dkms status` |
| 14:32 onward | PP lane running `glm5-hyper-ppn-gate 2 6 8` x12, its own command line reading `=== incidence, clean boot, exclusive box ===` | `ps -eo cmd` |

## CORRECTION (same day, after the coordinator produced counter-evidence)

An earlier version of this file claimed, under a heading reading "What this lane did NOT do",
that this lane "did not cause the Xid 31 faults". **That claim was too broad and it is
withdrawn.** What the timestamps actually support is narrower: the FIRST THREE faults (14:10:35,
14:11:15, 14:11:58) predate this lane having any CUDA context on the box, because the release
build did not finish until ~14:14. They say nothing about the two later ones (14:16:22, 14:21:23),
which land inside this lane's `IDOFF` server window, nor about the probe failures.

The coordinator's measurement settles it in the other direction:

> PP's peer probe failed **15/15 with 16384/16384 mismatched bytes** and Xid 31 MMU faults while
> an unrelated server held up to 20.9 GB on card 0. On the recovered idle box the same probe was
> **12/12 clean**.

That unrelated server was this lane's. So co-tenancy did not merely add noise to the PP lane's
cell — **it manufactured a result that read as catastrophic cross-device divergence**, and the
PP lane spent an incidence study characterising it. A "correctness arms tolerate a co-tenant"
assumption is what produced that, and it was wrong for a cross-device peer-integrity probe
specifically: a second process on the same card changes the allocation and mapping the probe is
testing. The lesson is not "check idle before timing arms", it is **check idle before anything
that shares a card with a memory-integrity measurement**.

Recorded here rather than quietly edited, because a receipt that revises a claim without saying
so is worse than the wrong claim.

## What this lane did NOT do

- Did not fire the first three Xid 31 faults (14:10:35 / 14:11:15 / 14:11:58): no CUDA context
  existed for this lane at those times. The later two are NOT disclaimed — see the correction
  above.
- Did not reboot the box, and did not run the 14:26 `apt` upgrade.
- Did not rebuild DKMS. The rebuild was already in flight from another operator by the time the
  driver state was diagnosed, and running a second `dkms autoinstall` against a live apt/dpkg
  transaction is how one broken thing becomes two.
- Did not restart the PP lane's runner. Their lane, their restart.

## Why the lane stood down instead of continuing correctness work

The grant said correctness arms may run concurrently and only timing arms need an idle box. That
was true when it was written. It stopped being true at 14:32: the PP lane's own command line
declares **`clean boot, exclusive box`**, and what it is running is an **incidence study** —
12 repetitions counting `PASS` / `PEER_PROBE_FAILED` / `OTHER` on a probabilistic failure.

A co-tenant is not neutral to that measurement. This lane's identity arm puts ~20 GB on card 0
(their gate is already holding 82 GB), changes allocation and eviction order, and would land
inside the exact statistic they are collecting. Between a standing grant and a live exclusivity
declaration from the other lane, the live declaration wins and the conflict gets surfaced rather
than resolved unilaterally. Timing was already blocked by `idle-check.sh`; correctness is now
voluntarily blocked too, pending the coordinator arbitrating a window.

## Two things the PP lane should be told (this lane is not acting on either)

1. **`Xid 31`, five times, before the reboot, from their gate binary** — reported here as
   diagnostic detail, NOT as a claim about origin (see the correction above; two of the five sit
   inside this lane's co-tenancy window). Verbatim:

   ```
   NVRM: Xid (PCI:0000:32:00): 31, pid=11572, name=glm5-hyper-ppn-, channel 0x00000002,
   intr 00000000. MMU Fault: ENGINE GRAPHICS GPC0 GPCCLIENT_T1_0 faulted @ 0x33_8404d000.
   Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_READ
   ```

   (also at 14:11:15 pid 20000, 14:11:58 pid 20928, 14:16:22 pid 30010, 14:21:23 pid 32427; the
   fault address alternates between `0x33_8404b000`/`0x33_8404d000` and `0x33_8804b000`.)

   A `FAULT_PDE ACCESS_TYPE_VIRT_READ` is a read through a device pointer whose page table entry
   is not valid — a stale or cross-device pointer, not a numerical bug. Their `peer byte-integrity
   probe FAILED` counter and these faults are very likely the same defect. It is the same hazard
   class this lane's `moe_fused_epi_token_q8` guards with its pass-2 re-verification: pointers
   collected before an operation that can invalidate them. Worth saying out loud because an
   incidence study will characterise the *rate* of a pointer bug rather than find it.

2. **`gdrdrv/2.5.2` was NOT rebuilt for `6.8.0-1063`.** `dkms status` shows `nvidia`,
   `efa` and `efa-nv-peermem` present for both kernels and `gdrdrv` only for `1061`. If their
   cross-device path touches GPUDirect, it is running degraded or absent right now, on the very
   boot they have labelled "clean". Theirs to fix; flagged, not touched.

## Standing hazard for the fleet, for the coordinator not for this lane

The reboot was clean and deliberate, and it still cost both lanes their in-flight work, because
a kernel upgrade and a reboot on a DKMS box means **the GPUs come back missing their driver**.
The same shape applies to any box carrying out-of-tree GPU modules, including the bench box and
the prod serving box. Two cheap guards, both box-owner calls:

- pin the kernel (`apt-mark hold the kernel image and metapackage`) on boxes that serve or measure, so
  a package upgrade cannot silently stage a driverless boot;
- make the reboot procedure include `dkms autoinstall -k $(uname -r)` **and** a `dkms status`
  check for every module, not just `nvidia` — the `gdrdrv` gap above is what a nvidia-only check
  misses.
