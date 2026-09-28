# OWED 20 sitting: storage-bound Step-3.7-Flash IQ4_XS (NOT RUN: closed by owner ruling, 2026-09-27)

**Owner ruling, 2026-09-27: option 3.** The storage-bound regime is the BOX27 balloon regime (iii),
already scored; this sitting is not run and nothing more is owed on it. The text, the lock, the
runner wrapper and the census stay as the record (`../M1-PREREG.md` section H).

**Market amendment (2026-09-27):** no 2x RTX PRO 6000 container offer has RAM below the bank
(184 GB and up), and a container cannot shrink its own kernel memory (`../M1-PREREG.md` section H,
market amendment; `container-capability-docker-default.log`). The registered route is a box whose
kernel is ours (a VM instance, marketplace VM type included, or bare metal) booted with `mem=64G`;
it goes to the owner because it changes what the pick assumed. The RAM check reads the effective
page-cache ceiling (the smaller of `MemTotal` and the cell's cgroup `memory.max`).

Registered in `../M1-PREREG.md` section H. Commands: `run-sitting.sh`. Artifact pin, census and
lock: `census.json`, `s20-arms.lock.json`.

## Box shape

| Need | Value | Why |
|---|---|---|
| Cards | 2x RTX PRO 6000 Blackwell (96 GB), the only GPU tenants, power at maximum | the qualified PP-2 topology of the artifact (`docs/models/step37-flash.md`); one card cannot hold the 104.99 GB trunk |
| Host RAM | page-cache ceiling below 101,072,240,640 bytes (the expert bank): `MemTotal` after a `mem=64G` boot on a VM or bare metal | the page cache must not be able to hold the bank, so every regime is storage-bound without a balloon; the sitting refuses a box at or above the bank |
| Storage | local PCIe NVMe, ext4 or xfs, passing the M1 proof, at least 300 GB free | the runner refuses without a PASS `nvme-local-direct` proof; the artifact (108.70 GB) and receipts live there |
| CPU | at least 16 threads | the 16-deep worker pools, one per PP stage, plus runner and samplers |
| Access | root or unlimited memlock; no swap in use | pinned buffers; a swapping host would make the regime something else |

## Order

1. RAM precondition, build (OWED 26 fix build), stage all four files byte-verified, M1 proof.
2. `reference`: the qualified PP-2 program without the disk tier; its token ids are the oracle.
3. Disk-tier smoke (six arms, one round) gated on tokens equal to the reference, zero fallbacks
   and zero demand-wait timeouts; `run-spec` K=1..8 PASS for worker16 with the disk tier. A
   mismatch stops the sitting: the disk tier on Step is then a different numeric program.
4. The storage-bound regime: ten rounds, six arms, B3's verdict rule.

Expected wall time: about 5 hours (stage 20 min at a typical link, build 5, proof 2, reference 10,
smoke 20, spec 20, the regime about 3.5 hours at 6 arms by 10 rounds of storage-bound visits).
