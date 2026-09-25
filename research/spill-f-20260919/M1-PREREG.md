# M1 pre-registration: prove local NVMe, then measure memra's spill speed on it

Registered 2026-09-25, before any rental request and before the proof tool or cell code.
Owner order 2026-09-24: "every improvment and tunning should be done, no shortcut or fast path".
Supersedes the 2026-09-20 NO-GO in `NVME-DECISION.md` as the plan of record; that file stays as
the history of why the VM route was parked. Nothing below has run. Open items: `OWED.md`.

The claim M1 exists to earn, and nothing wider: **the writable directory under memra's spill path
is on physical local NVMe, proven in the running kernel from the mount to the PCI function, and
memra's expert-bank and KV spill paths measured on that directory produce the recorded numbers
under the recorded conditions.** No model-support claim, no runtime default, no serving
qualification and no board update follows from M1 alone.

## A. The proof chain (tool: `m1-nvme-proof.py`, stdlib Python, read-only except A6)

One admitted class: **`nvme-local-direct`**. A path is admitted only when every step passes.
Every failure is recorded with its reason; the tool reports all failing reasons, not the first.

| Step | Check | Passes when | Fails closed on |
|---|---|---|---|
| A1 kernel | `/proc/cpuinfo` flags, `/sys/class/dmi/id/{sys_vendor,product_name,board_vendor}`, `/sys/hypervisor/type`, and a census of every `/sys/bus/pci/devices/*` vendor:device | no `hypervisor` CPU flag, DMI not a virtual platform, no hypervisor type, and no emulated platform device on the PCI bus (Q35/ICH9/i440FX/PIIX chipsets, QEMU root ports and bridges, virtio, VMware, VirtualBox, bochs); the census still catches a guest whose CPU flag is hidden | any guest kernel (a VM, or a container whose host is itself a VM). A guest can see an emulated NVMe with a PCIe path; in-guest identity cannot rule emulation out, so no guest class is admitted in this campaign |
| A2 path | `stat`, `os.access`, `realpath` | `P` is an existing, writable, lane-owned directory; no component resolves outside the mount under test | missing, read-only, or file (not directory) targets; the container root; another lane's directory |
| A3 mount | `statx(P)` (dev major:minor, `stx_mnt_id`, DIO alignment), `/proc/self/mountinfo`, `/proc/self/fdinfo/<fd>` of an fd on `P` | the three mount ids agree, the mountinfo entry covering `realpath(P)` carries the same major:minor, and the filesystem type is `ext4` or `xfs` | overlay, tmpfs, ramfs, squashfs, fuse and `fuse.*`, nfs, cifs/smb, ceph, 9p, virtiofs, lustre and other network or virtual filesystems; btrfs and zfs (anonymous `st_dev`, not traceable through `/sys/dev/block` here); any id disagreement |
| A4 block graph | `/sys/dev/block/<maj:min>` resolved, `partition` to parent, `slaves/` recursion (dm linear/crypt/thin, md RAID), NVMe native-multipath heads expanded to every path | every node is classified and every leaf is an NVMe namespace | loop (including a loop file on overlay or on a proven filesystem: a loop layer is a different I/O stack), virtio `vd*`, xen `xvd*`, SCSI/SATA `sd*`, `nbd*`, `rbd*`, `zram*`, `ram*`, `pmem*`, `mmcblk*`, a virtual node with no slaves, a RAID or dm set with any non-NVMe member, an unreadable node |
| A5 controller | `/sys/class/nvme/<ctrl>/{transport,model,firmware_rev,serial}`, controller `device` link, PCI `class`, `vendor`, `device`, `subsystem_vendor`, `current_link_speed/width`, `max_link_speed/width`, `numa_node` | `transport` is `pcie`, the controller resolves under `/sys/devices/pci*`, class `0x010802`, a vendor outside the emulated set, a non-empty model | NVMe-oF (`tcp`, `rdma`, `fc`, `loop` transports); emulated or virtual vendor ids 0x1b36, 0x1af4, 0x15ad, 0x1414, 0x1ae0, 0x1d0f, 0x80ee, 0x1ab8 and 0x8086:0x5845; emulator model strings; missing link fields |
| A6 I/O binding | a lane-owned file in `P`: 1 GiB nonuniform pattern written with O_DIRECT at the statx DIO alignment, `fdatasync`, read back with O_DIRECT; `/sys/block/<leaf>/stat` (and the top device's `stat`) sampled before and after each half | bytes read back equal bytes written (SHA-256); the leaf set's written and read sector deltas each cover at least the payload (`>= 2097152` sectors); the file is removed | O_DIRECT refusal (errno quoted), short I/O, byte mismatch, counter deltas below the payload, counters unreadable |
| A7 capacity | `statvfs(P)` | free space covers the staged artifact, fixtures and a 20 GiB reserve | below the reserve |
| A8 identity for cells | the proof emits `{device, filesystem_id, mount_id}` in exactly the shape `tools/tier-battery.py::filesystem_identity` produces | every scored cell re-reads the triple before and after; equal to the proof's | any change: the cell is retained and unscored |

Notes that bind the reading of a PASS:

- A container on a bare-metal host shares the host kernel, so its `/sys` block and PCI view is the
  kernel's own: a bind-mounted host directory (a provider local volume, a pod volume disk) traces
  exactly as it would on the host. That is the route this campaign uses (section C).
- A6's counter deltas are host-wide. The amount above the payload is foreign I/O and is recorded
  as the co-tenancy indicator; the proof passes on coverage, the cells gate on the indicator.
- Passing A proves where the bytes go. It does not qualify throughput, durability under power
  loss, or any consumer (KV restore, expert eviction, serving).
- Privacy: the raw capture (controller serial, PCI address, mount source, root and options, host
  paths, DMI strings) stays private, outside git. The public receipt carries classes, the
  drive model and firmware, link, NUMA, counter deltas, SHA-256 of the private capture, and hashes of
  serials. No provider, host, machine id, city or price enters a tracked file.

Command (inside the collector, canonical rig lock, before any cell):

```sh
python3 tools/tier-battery.py --rig pro-single --timeout 600 --out "$R/m1-proof" --execute \
  python3 research/spill-f-20260919/m1-nvme-proof.py --path /scratch/spill-f \
  --bind-bytes 1073741824 --private-out "$PRIV/m1-proof.json" --public-out "$R/m1-proof/PROOF.json"
```

Exit 0 is PASS, 3 is FAIL (reasons in both receipts), 2 is a refused invocation. **A box that
fails the proof yields no scored cell.** Its receipts are kept as a FAIL record.

## B. The spill cells (only after PASS, same box, same window)

### Common conditions

- **Binary:** one memra commit, built on the box before the window; every visit uses the same
  binaries (`run-gen`, `run-spec`, `memra-server`, `storage-bench`), SHA-256 recorded before and
  after. Arms differ only by environment.
- **Artifact:** `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`, 18,209,036,576 bytes, SHA-256
  `df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf` (the program's
  `APPROVED_SHA`), staged byte-identical onto `P` and re-hashed there; the staged hash is part
  of every visit receipt. Expert bank 15,600,713,728 bytes in 31,488 slices (census below).
- **Lock and exclusion:** the collector holds the canonical lock for the whole window
  (`/tmp/memra-gpu.lock` on the PRO card, `/tmp/memra-5090.lock` on the 5090). No builds, no
  second campaign, no other GPU process; compute apps captured before, after and at failure.
- **Telemetry (all at 250 ms):** the collector's GPU CSV (clocks, power, enforced cap, temperature,
  VRAM, utilization, PCIe link); the lane sampler: `/proc/diskstats` for the proven leaf set and
  the top device, NVMe hwmon temperature, `/proc/meminfo` (MemAvailable, Cached), `/proc/stat`,
  and the visit process's `/proc/<pid>/io`. Missing or gapped telemetry (gap over 500 ms) means
  the visit is unscored.
- **Cache regimes:** *cold*: file-scoped `POSIX_FADV_DONTNEED` on the artifact (or object files),
  then `mincore` shows 0 resident pages, then the process starts. *warm*: one full buffered read,
  `mincore` shows every page resident. *bounded page cache*: a lane-owned mlocked balloon leaves
  MemAvailable below half the expert bank, never below a 6 GiB floor; `mincore` sampled each
  visit; if `RLIMIT_MEMLOCK` or the floor forbids it the regime is refused with the reason and
  OWED 20 (an above-RAM artifact) carries it. No global cache drop, no cache-drop privilege.
- **Order:** round-robin. Ten rounds; odd rounds run the arm list forward, even rounds reversed.
  Every arm gets N=10 visits and every pair of arms meets five times in each relative order.
  Each visit is a fresh process; model load is timed separately and outside the scored window.
- **Raw first:** every child's stdout and stderr go through `tee` into the visit directory before
  any parser reads them; exit status and timeout status are kept. Failures are quoted.
- **Co-tenancy gate:** a visit whose foreign device bytes (leaf delta minus the process's
  `read_bytes`/`write_bytes`) exceed 2% of its device bytes is contaminated. More than two
  contaminated visits of any arm in a regime leaves that regime unscored for the window.
- **Thermal gate:** a drive hwmon reading at or above its `temp*_max` during a visit, or a changed
  GPU power cap, makes the visit unscored.

### B0 device envelope (CPU only; `fio` is an instrument here, never "memra spill speed")

Lane-owned 16 GiB file in `P`, written sequentially with direct I/O first (no holes), `fsync`.
Random reads, `--direct=1 --time_based --runtime=10 --ramp_time=2 --randrepeat=0
--output-format=json+`:

- block sizes 4096, 450560, 557056, 860160 and 1048576 bytes (the three middle sizes are the
  artifact's IQ3_S, IQ4_XS and Q6_K expert slices);
- depth 1, 2 and 16 (blocking pread, worker default, worker knee);
- engines: `psync` with `--numjobs=<depth>` (the worker's threads doing positioned reads),
  `io_uring --iodepth=<depth>`, `libaio --iodepth=<depth>`.

The full grid is N=1 per cell, descriptive. The io_uring screen is scored: 557056-byte reads at
depth 2 and 16, `psync` threads vs `io_uring`, 5 AB plus 5 BA visits. Sequential write
envelope: 4 MiB blocks, direct and buffered-plus-`fsync`, 30 s each, N=1, SLC regime noted.
Its measured sustained read rate is the denominator of the 70% route/SSD headroom rule.

### B1 KV ObjectStore storage (`storage-bench`, CPU)

Sizes 264, 4096, 4097, 1048576, 116654080 and 933232640 bytes; modes `buffered`, `uncached`
(O_DIRECT reads) and `direct` (O_DIRECT reads and writes); phases `roundtrip` (fresh empty
directory per visit) and `restore` (cold regime on the object files). Round-robin over the three
modes per size and phase, N=10 each. Scored: read-call time and write/commit time separately
(OWED 9), valid bytes, padded bytes, device bytes, `fallbacks=0`, byte-exact status. Buffered is
the baseline arm. Blocked on OWED 14 when run through the collector's storage guard.

### B2 KV host-tier handoff (memra-server, GPU)

The artifact in resident mode, `MEMRA_KV_HOST_MB=16384`, `MEMRA_KV_HOST_HANDOFF=$P/handoff.bin`.
A fixed prompt set fills the host tier to 1 GiB and to 8 GiB. Drain-export (fsync, rename),
restart, cold-regime import, then the same prompts: completions byte-identical to a cold-computed
reference; TTFT with and without the import. Recorded: the export line (entries, MB, ms), the
import DONE line (entries, MB, s), device bytes, fsync share. N=5 cycles per size, single arm
(the handoff has one I/O mode today; its O_DIRECT arm is OWED 18). Driver: OWED 13.

#### B2 amendment (2026-09-25, registered before the driver code; OWED 13)

Found while designing the driver: the in-tree `memra-server` exposes no route that triggers the
handoff export; only a deployment binary's `ServerWiring::on_ready` hook receives
`HostHandoffHandle`. The boot-time import needs no trigger (a file present at boot arms it, and
the idle loop drips it at a 1 ms wait). So B2 needs one instrument and one driver:

- **`kv-handoff-gate`** (new `memra-server` binary): `serve_with(ServerWiring::stock().on_ready(..))`;
  SIGUSR1 calls `HostHandoffHandle::export(force = false)` and prints
  `[handoff-gate] export ok <report json>` or `[handoff-gate] export refused: <reason>`; the task
  drops its handles on the drain shutdown signal (the worker exits only when every sender is
  dropped). No env read, no HTTP route, no change to the stock binary.
- **Engine log change:** the existing `[prefix-host] handoff export:` line gains `write_ms=` and
  `fsync_ms=` so the fsync share is measured, not inferred.
- **Driver `m1-handoff-driver.py`**, one collector cell per size, external lock verified:
  - model: the B3 artifact; server env `MEMRA_MODELS=gate=<artifact>`, `MEMRA_CTX=8192`,
    `MEMRA_MAX_SESSIONS=4`, `MEMRA_PREFIX_CACHE_MB=1024` (device budget small so entries demote),
    `MEMRA_KV_HOST_MB=16384`, `MEMRA_KV_HOST_HANDOFF=<P>/b2/handoff.bin`, greedy requests
    (`max_tokens=48`, `temperature=0`);
  - prompts: 128 distinct deterministic synthetic prompts of 6,500 plain words, generated on the
    box by `m1-b2-prompts.py` and checked against `m1-prereg/b2-prompts.manifest.json`; probes are
    prompts 1 to 4 plus a fixed suffix. (Amended the same day, before any run: the first
    registration said 96 prompts of about 7,000 tokens; the artifact's own tokenizer measured
    4,531 tokens for 4,500 words, too little to reach 8 GiB with margin. 6,500 words measure
    6,526 to 6,535 tokens with the probe suffix, inside the 8,144-token budget);
  - reference: one stock `memra-server` boot with `MEMRA_KV_HOST_MB=0` and no handoff gives the
    probes' cold texts;
  - cycle (N = 5 per size, sizes 1 GiB and 8 GiB of `prefix_host_bytes` plus the drain-demoted
    device entries): boot the gate; send prompts in order until `/metrics` `prefix_host_bytes`
    reaches the size (prompts exhausted is a cycle failure); SIGUSR1; wait for the export line
    (900 s); SIGTERM and wait; SHA-256 of the file, then the cold regime on it (`mincore` = 0);
    boot the gate again with the sampler on the proof's leaves; wait for
    `[prefix-host] handoff import DONE` (1800 s); send the four probes; SIGTERM;
  - a cycle passes when the export answered ok, the import finished with `skipped = 0`, and every
    probe has `cached_tokens > 0` and text equal to the reference;
  - recorded: export entries, MB, ms, write_ms, fsync_ms; file bytes and hash; import entries,
    MB, seconds, skipped; device write bytes over the export window and read bytes over the
    import window; probe `cached_tokens`. Descriptive medians with N stated; single arm; no
    default decision.
- CPU gates before the box: the gate binary builds and passes clippy; the driver runs end to end
  against a stub server that prints the exact log formats and serves `/v1/models`,
  `/v1/completions` and `/metrics`, including red controls (a refused export, an import with
  skips, a probe that misses the cache, a probe whose text differs).

### B3 expert-bank spill (the headline cells; `run-gen` and `run-spec`)

Common env (frozen in `m1-prereg/b3-arms.lock.json`): `MEMRA_SPILL_DISK=1
MEMRA_SPILL_PINNED_FRAC=0 MEMRA_MOE_CACHE=1 MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS=8
MEMRA_SPILL_STATS=1`, fixed text prompt (file hash in the lock), greedy, `MEMRA_NGEN=128`.
With 8 slots the hit rate is 0% and each decode token moves about 454 MB of expert bytes, so the
storage-to-GPU path is the whole decode loop.

| Arm | Env | Role |
|---|---|---|
| `worker16` | `MEMRA_SPILL_IO=worker MEMRA_SPILL_PREAD_DEPTH=16` | **measured baseline** (the Hy3 knee in `docs/FLAGS.md`) |
| `worker2` | `MEMRA_SPILL_IO=worker` (default depth 2) | default-depth row |
| `pread16` | `MEMRA_SPILL_IO=pread MEMRA_SPILL_PREAD_DEPTH=16` | blocking positioned-read oracle |
| `mmap-random` | unset `MEMRA_SPILL_IO` (mmap default) | mapped access, default advice |
| `mmap-normal` | `MEMRA_MOE_MMAP_ADVICE=normal` | mapped access with readahead |
| `direct16` | `MEMRA_SPILL_IO=direct MEMRA_SPILL_PREAD_DEPTH=16` | O_DIRECT worker; **refused until OWED 7's GPU gates pass** (the over-read landed CPU-verified 2026-09-25, `owed7/RESULTS.md`). Before it, 0 of 31,488 slices were O_DIRECT-admissible and the arm was 100% mmap fallback under a direct label |

Census receipt: `m1-prereg/direct-alignment-census-qwen36-35b.json` (header-only; header
SHA-256 `61329137...890ab`; `general.alignment=32`; slice lengths are 4096 multiples, every
tensor base offset is not). io_uring and mapped pinned-host arms do not exist in the engine;
they are B5 and OWED 17, not silent omissions.

1. **Correctness, before any timing, per arm:** `run-gen` prints `MATCH`; the 128 generated token
   ids are byte-identical to `mmap-random`'s (the byte oracle); spill stats show `errors=0
   short_reads=0 config_fallbacks=0` and, for `direct16`, `fallbacks=0` with the over-read byte
   count equal to the census prediction; `run-spec` K=1..8 prints `SELF-CONSISTENCY PASS` for
   `worker16` and `direct16`. A failing arm is refused, retained, and never timed.
2. **Timing, three regimes** (cold, warm, bounded page cache), round-robin N=10 per arm each.
   Scored per visit: gen-only decode tok/s, TTFT (`[ttft]` line), load time (separate), and the
   stage account: logical spill bytes (`[spill-pread]`), device read bytes (leaf diskstats),
   process read bytes, MoE misses, H2D bytes, the per-stage counters of OWED 8 (worker read
   time, owner wait time, buffer waits, ring-full events), CPU utilization, GPU utilization and
   PCIe receive rate.
3. **Verdict rule, per regime, arm X vs `worker16`:** within each round the X/baseline ratio of
   decode tok/s. **Winner**: median ratio at least 1.05, and at least 4 of 5 forward and 4 of 5
   reverse rounds agree in sign. **Loser**: median at most 0.95 with the same agreement.
   Otherwise **flat**. A verdict whose stage account contradicts the arm's mechanism (for
   example `direct16` device bytes far below its logical bytes, or a warm buffered arm reading
   the device) is **unexplained** and not scored until explained.
4. **What a verdict can decide:** nothing about defaults from this rig alone. A PRO-class winner
   becomes a candidate per-device default only after the 5090 half (OWED 19) and the owner's call;
   a flat or losing `MEMRA_SPILL_IO` arm goes to the door-hygiene decision in `docs/FLAGS.md`
   with these receipts.

### B4 serving shape (memra-server)

`worker16`, plus any B3 winner, in the regime where it won: a fixed 32-request set, 128 output
tokens, concurrency 1 and 4, 5 AB plus 5 BA. Recorded: TTFT, E2E, TPOT and ITL p50/p95/p99,
request and token throughput, plus the B3 stage account. Without a B3 winner, `worker16` alone
runs as a descriptive serving row.

### B5 io_uring decision input

From B0's scored screen: if `io_uring` beats `psync` threads by at least 5% at the worker's
operating point (with 4 of 5 rounds agreeing in each order), or matches it (median bandwidth ratio at least 0.97) at 10% less CPU per GiB (both clarifications registered 2026-09-25 with the runner, before any run), the lead assigns the memra ring from
`spill-a-20260919/IO-URING-PROPOSAL.md` and its own AB against `worker16` becomes a B3 arm;
otherwise io_uring stays deferred with this receipt as the reason.

### B6 remaining G2 sizes (host/device copies, through D's runner)

16 KiB, 256 KiB, 4 MiB, 64 MiB and 1 GiB, pinned vs pageable, both directions, 5 AB plus 5 BA,
per `CELLS-ENVELOPE.md`. Rides the same window; independent of A.

### Budget

Bootstrap and native build 40 min, staging and re-hash 10 min, proof 10 min, B0 25 min, B1 20
min, B2 40 min, B3 correctness 25 min, B3 timing about 4 h (three regimes, 60 visits each,
about 75 s per visit), B4 60 min, B6 25 min: **about 8 h of box time**, on one card.

## C. Rentable routes (read-only research 2026-09-25; identifiers and quotes in the private file)

| Route | What it gives | Can it pass A? |
|---|---|---|
| Marketplace docker instance plus a **local volume** at `/scratch` (the route's documentation: local volumes are "physically tied to the machine", attach to docker instances only) | a bind of a host directory into a container that shares the host kernel | **yes, if** the host is bare metal and the volume is a host-filesystem bind on NVMe; it fails closed if the volume is a loop image, the host is itself a VM, or the backing is not NVMe. Offers whose advertised disk is an emulator string mark a VM host and are excluded up front |
| Same marketplace, VM instance (`vms_enabled`) | a KVM guest with a virtual disk; local volumes do not attach to VMs | **no** (A1 and A4) |
| Pod provider with a host-local pod volume disk (not its network volume) | same container shape as the first route | yes under the same conditions; fallback route |
| The program's VM provider | PRO 6000 instance types list only dynamic (network block) storage | **no** |

Selection for the first route: one RTX PRO 6000 (target class), whole machine (`gpu_frac=1.0`,
so no co-tenant shares the drive), 600 W cap, a local volume of at least 500 GB, 94 GB or more
RAM, 16 or more cores, driver supporting CUDA 13.1 or newer, a real NVMe drive model in the
advertisement. Three offers met this on 2026-09-25; the private file ranks them.
