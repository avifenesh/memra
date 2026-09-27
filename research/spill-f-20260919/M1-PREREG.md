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

#### B1 amendment (2026-09-25, after the one-round B1 smoke, before any scored B1 visit)

Record of how it landed: written at 17:42Z, before the scored B1 run started at 17:43:57Z, but
its first commit was stopped by a failed shell chain (a zero-count `grep -c` exits 1), so the
box ran the pre-amendment runner. That runner records every input of the amended gate (device
and own read bytes per visit), so the scored B1 summary is recomputed offline with this gate by
`m1-b1-resummarize.py`; the run's in-process scoring is kept as recorded and labelled
superseded.

- Every visit is its own collector invocation with the exact `storage-bench` argv and
  `--storage-root`/`--storage-proof` (the collector refuses storage-bench behind a wrapper:
  `REFUSED: opaque storage command`); the runner orchestrates around them.
- Co-tenancy gate, restated for storage writes: per-process accounting cannot see the
  filesystem journal or the kernel's writeback threads (the smoke's roundtrip visits carried 15
  to 24 MB of such writes with no other tenant on the machine), and sub-MiB payloads are below
  the filesystem's own metadata traffic. B1 gates on reads: foreign read bytes at most
  max(2% of the visit's device read bytes, 1 MiB). Device write bytes are recorded per visit and
  not gated. B3's gate is unchanged (its visits are read-dominated; the smoke read 0.1 to 0.4%).

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

#### B2 amendment 2 (2026-09-26, after the first B2 attempt, before any passing cycle)

The first 1 GiB attempt (`box27/b2-1g-attempt1-spec-route`) never filled the host tier: all 128
fill prompts ran on the MTP spec route, where the prompt-end seed that feeds the prefix cache
does not publish (the grid-aligned seed is armed for plain sessions only; a spec session keeps
its own post-prime capture, which one-token fills never reach), so `prefix_host_bytes` stayed 0
and every cycle failed as "prompts exhausted". Every B2 boot (reference, export and import) now
runs with `MEMRA_SERVE_SPEC=0`, the documented production posture for shared-prefix serving
shapes (`docs/FLAGS.md`). Nothing else changes.

#### B2 amendment 3 (2026-09-26, after the first 8 GiB attempt)

The first 8 GiB attempt (`box27/b2-8g-attempt1-tenant-cap`) plateaued at 8,520,110,208 host
bytes (67 entries of 127 MB) with all 128 prompts sent: the default per-tenant share cap
(`MEMRA_KV_HOST_TENANT_PCT=50` of the 16 GiB budget) evicts a single tenant's oldest entries,
which are the probe prompts. The export (8,520 MB, 5.2 s) and import (6.1 s) still ran and are
kept as recorded. The cell is single-tenant by construction, so its rerun sets
`MEMRA_KV_HOST_TENANT_PCT=100` (`--tenant-pct 100`); the 1 GiB cell, which never reached the cap,
keeps the default. Nothing else changes.

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

#### B3 amendment 2 (2026-09-25, after the one-round smoke, before any scored B3 visit)

The smoke (`box27/b3-smoke`, one round, never scored) showed the registered common env cannot
force the disk tier on the current engine: `[spill] invalid MEMRA_SPILL_PINNED_FRAC="0"
(expected a finite fraction greater than 0 and at most 1); using 0.6`, then `[spill] experts
placed: 30720 pinned (Tier 1), 0 mmap'd from disk`. The `0` came from the 2026-08-11 spill smoke,
before the range check. Every smoke arm therefore measured pinned host RAM to the GPU (36.7 to
37.1 tok/s, N=1 each); kept as an unscored diagnostic of the pinned tier.

- Common env: `MEMRA_SPILL_PINNED_FRAC=0.000000001` (inside the accepted range; the pinnable
  budget becomes tens of bytes, below one expert, so nothing is pinned).
- New correctness gates per visit: `[spill] experts placed: 0 pinned ... 30720 mmap'd`, and no
  `[spill] invalid` or `[spill-pread] invalid` line.
- `run-gen`'s GGUF path prints no per-window spill lines (they exist on its safetensors path), so
  the direct gates and the stage account read the pool's whole-visit totals line
  (`[spill-pread] reads= ... overread_bytes= worker_read_ns= demand_read_ns= wait_ns=
  h2d_submits=`, prefill included): `errors=0 short_reads=0` for every positioned arm, and for
  `direct16` `fallbacks=0` and `overread_bytes == 4096 x reads`.
- Prefill time from the GGUF path's `prefill N tok in Xs` line (it has no `[ttft]` line).

#### B3 regime (iii) amendment (2026-09-25, registered on the box before any B3 visit)

The rented container refuses the registered mlocked balloon: `RLIMIT_MEMLOCK` is 8 MiB and
raising it fails (`ulimit: max locked memory: cannot modify limit: Operation not permitted`;
the container has no `CAP_SYS_RESOURCE`). Its cgroup v2 limits are `memory.max` =
127,295,029,248 bytes and `memory.swap.max` = 0. With swap forbidden for the cgroup, touched
anonymous memory cannot be reclaimed, so it bounds the page cache exactly as locked memory would.

- Mechanism: `m1-cache-regime.py balloon --touch`: anonymous mmap, one write per page, no mlock.
  Refused unless the process's cgroup reports `memory.swap.max` = 0.
- Size: `memory.max - anon - leave`, with `leave = 7,000,000,000` bytes for the visit process
  and its page cache (below half the 15,600,713,728-byte expert bank, as registered).
- Floor, restated for cgroup accounting: the balloon releases itself (exit 4) if
  `memory.max - anon` falls under 2 GiB, which prevents a cgroup OOM. The registered 6 GiB
  host-MemAvailable floor cannot apply here: the container's `/proc/meminfo` is host-wide.
- Proof the bound held, per visit: `mincore` residency of the artifact at visit end below 50% of
  the file, and device read bytes over the decode window of the same order as the logical spill
  bytes. A visit failing either is `regime_ok = false` and unscored.

### B4 serving shape (memra-server)

`worker16`, plus any B3 winner, in the regime where it won: a fixed 32-request set, 128 output
tokens, concurrency 1 and 4, 5 AB plus 5 BA. Recorded: TTFT, E2E, TPOT and ITL p50/p95/p99,
request and token throughput, plus the B3 stage account. Without a B3 winner, `worker16` alone
runs as a descriptive serving row.

#### B4 amendment (2026-09-25, registered on the box before any B4 visit)

- Request set: 32 prompts, each the B3 prompt text followed by ` Answer variant i of 32 in your
  own order.` (i = 1..32); SHA-256 of the joined set recorded in each run's `identity.json`.
- Transport: `POST /v1/completions`, `stream: true`, `stream_options.include_usage`, greedy
  (`temperature 0`), `max_tokens 128`; `c` client workers pull from the set in order.
- Server: `memra-server` with the B3 common env and the arm env (minus the `run-gen`-only
  prompt, chat and NGEN variables), `MEMRA_COMPAT=openai`, `MEMRA_CTX=8192`,
  `MEMRA_MAX_SESSIONS = max(4, c)`; a fresh boot per visit after the regime is applied.
- Per request: TTFT (first text frame), E2E, TPOT = (E2E - TTFT) / (tokens - 1), every
  inter-token gap. Per visit: p50/p95/p99 of each, request and token throughput over the
  request window. A visit with any request error, invalid telemetry or a failed regime is
  unscored. Tool: `m1-b4-serving.py`.

#### B4 arms (2026-09-26, registered after B3 and before any B4 visit)

Under the registered gate only the bounded regime scored, and there `worker16` beat every
challenger, so B4 has no registered winner: `worker16` runs as the registered descriptive
serving row. The warm regime's read-gate rescoring (applied after its data) found `mmap-random`
and `mmap-normal` 1.19x faster than `worker16`; they join as labelled post-hoc challengers, in the
warm regime where they won, in one round-robin schedule (`m1-b4-serving.py --arms
worker16,mmap-random,mmap-normal --regime warm`: every pair meets 5 times in each relative
order per concurrency). Their B4 verdicts are reported as post-hoc, never as registered.

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

## D. The 5090 half (OWED 19 and 23; registered 2026-09-26 before any 5090 cell)

Rig: the local laptop RTX 5090 (24 GB, enforced cap reported `[N/A]`, maximum 175 W), shared
with lanes B and C and occasionally another project's process. Storage: `/data` (LVM over a
PCIe 4.0 x4 NVMe), proven `nvme-local-direct` in `M1-PROOF-CONTROLS.md`; a fresh proof receipt of
`/data` is taken first and bound as in section B. Artifact: the same pinned GGUF under `/data`,
re-hashed before the first cell. Binaries: a local release build from the lane tip whose engine
source equals BOX27's build (`git diff --quiet ffff2d89a HEAD -- crates Cargo.toml Cargo.lock`),
`MEMRA_CUDA_ARCH=120a`, so both rigs measure one program. Lock `/tmp/memra-5090.lock`.

Rig rule, which changes the regimes: every process tree here runs under `systemd-run --user
--scope -p CPUQuota=1200% -p MemoryMax=20G`, and page cache is charged to that cgroup. So:

- **capped** (replaces cold here): cold start of each visit (DONTNEED plus `mincore` 0) inside a
  20 GiB cgroup with `MemorySwapMax=0`. The 18.2 GB artifact plus the process cannot all be
  cached, so this is neither the box's cold nor its bounded regime; it is the rig's ceiling.
- **bounded**: the same, with the cgroup's `MemoryMax` set to the measured peak visit RSS plus
  7,000,000,000 bytes (below half the bank), so the page cache cannot hold the bank. No balloon:
  the cgroup limit is the bound, verified per visit by `mincore` (below 50% at visit end).
- **warm** is not runnable under the rig rule (the file plus the process exceed 20 GiB); recorded
  as refused by the rig rule, liftable only by the owner.

Arms, order, correctness and verdict rule: exactly B3's (lock file, amendment 2 env). To share the
card, each round of six visits is its own collector cell holding the lock only for that round;
before each round the driver waits until the lock is free and no compute application is on the
card, and records every wait (start, end, blocking processes by name and memory). Rounds keep
their registered forward/reverse order, and verdicts pool the ten rounds of a regime. Inside a round, a visit
during which any other compute application appears on the card (polled every 5 s) is unclean
for timing, like a co-tenant on the drive. The
co-tenancy gate is unchanged; `/data` also carries this desktop's `~/.cache`, so contaminated
visits are expected and are excluded by the registered rule, not argued away.

G2 5090 half: all ten registered sizes (4 KiB to 1 GiB), pinned vs pageable, both directions, D's
protocol (calibrate each size to at least 500 ms per visit, then 5 AB plus 5 BA), the
`h2d-probe` built as above. The laptop has no settable 600 W envelope, so D's 600/600 W check is
replaced by: the power fields must be identical on every sample of the campaign (recorded
verbatim). Tool: `m1-g2-5090.py`.

No default follows from either half alone; with BOX27 these are the two rigs CLAUDE.md requires
for a per-device decision.

## E. OWED 18: the KV handoff O_DIRECT arm (registered 2026-09-26 before any code)

Baseline (B2 on BOX27): export is serialize plus sha256 plus a 4 MiB `BufWriter` into the page
cache, then `fsync`, about 1.5 to 1.7 GB/s end to end; import is a 4 MiB `BufReader`, validated
frame by frame, about 1.4 GB/s. Both are far below the drive, so the question is whether the page
cache copy and the fsync writeback are part of the engine-side cost.

Mechanism: `MEMRA_KV_HOST_HANDOFF_IO=direct` (default `buffered`, a default-OFF door with a
decide-by date 14 days after landing). Export writes the `.tmp` file with `O_DIRECT` through one
4096-aligned 4 MiB buffer: full buffers go straight to the device, the last block is padded with
zeros, the file is truncated to its logical length, then `fdatasync`, then the same rename.
Import reads with `O_DIRECT` through the same aligned buffer shape. The wire format does not
change: a direct-written file equals a buffered-written file byte for byte, and either reader
reads either file. A filesystem that refuses `O_DIRECT` fails the export or import loudly
(`O_DIRECT open refused`); there is no silent buffered fallback inside the direct arm.

Correctness first, before any timing:
1. Unit: the same header and frames through both writers give byte-identical files (sha256), for
   lengths that end on, before and after a 4096 boundary, including an empty entry list; each
   reader reads both files to identical entries; truncation and corrupt-frame behavior under the
   direct reader equal the buffered reader's (the existing `host_handoff_tests` cells run for
   both modes).
2. Cell, forced ON and OFF: the B2 cycle (`m1-handoff-driver.py`, same prompts manifest, spec
   off, 1 GiB) with each arm; every probe's text identical to cold, import entries equal export
   entries, zero skips.

Timing (5090 half now; the PRO 6000 half needs a target card): one collector cell per cycle
pair, idle-gated like section D, 1 GiB, five forward pairs (buffered then direct) and five
reverse pairs, alternating. Metrics per cycle: export ms, its write and fsync parts, import s.
Verdict per metric on the within-pair ratio direct / buffered: winner at median <= 0.95 with at
least 4 of 5 pairs below 1 in each order, loser at median >= 1.05 with the same agreement,
otherwise flat. The 8 GiB cell runs the same way if the process fits the rig's 20 GiB cgroup; if
it does not, that is recorded as refused by the rig rule. A 5090 result sets at most a 5090
default; the door stays default-OFF until both rigs have a row.

Section E, 5090 sizing (registered 2026-09-26 before any handoff cell): the gate boots use
`--host-mb 4096` for the 1 GiB cell and `--host-mb 12288` for the 8 GiB cell (BOX27 used 16384;
the rig's 20 GiB cgroup cannot hold a 16 GiB pinned tier plus the process). The 8 GiB cell also
uses `--tenant-pct 100` (B2 amendment 3). Scratch is `/data/cache/spill-f-b2` on the proven
filesystem; the prompt set is regenerated and checked against `b2-prompts.manifest.json`.
Driver: `m1-5090-rounds.py --regime handoff`, round k = one pair, `buffered,direct` for odd k,
`direct,buffered` for even k, ten rounds. No compile or other CPU campaign of this lane runs
while a scored cell runs.

Section D, bounded amendment (2026-09-26, after the capped smoke and before any bounded cell):
the registered bound "measured peak visit RSS plus 7,000,000,000" cannot use ru_maxrss. The
smoke's ru_maxrss is 17,802,168 kB on every arm, the full artifact, because run-gen maps the whole
file and file-backed mapped pages count as RSS; that bound (about 25.2 GB) would be looser than the
20 GiB capped regime. The bound is therefore the peak of RssAnon plus RssShmem (`/proc/<pid>/status`,
250 ms samples), taken once per arm in a capped-scope cold visit by `m1-anon-peak.py`, maximum over
the six arms, plus 7,000,000,000. That cell runs after the capped regime and before bounded;
its visits are sizing only, never scored. Everything else in bounded is unchanged, including the
per-visit residency check (below 50% of the artifact's pages at visit end).

## F. OWED 17: the cold read-once bypass, staged and mapped (registered 2026-09-26 before any code)

Fact that sets the question (5090 capped smoke, worker16): the 8-slot MoE cache hits 423 of
583,680 dispatches (0.1%); every expert block is read, copied host to device into a slot, used
once and evicted, 454 MB per decode token. The CLAUDE.md pipeline list and the per-expert-quant
ladder rung 5 name the candidate: serve such a block without the copy, by letting the kernel read
the pinned host buffer through its device address, and only for cold blocks.

Mechanism, one door, `MEMRA_MOE_COLD_BYPASS` (default `off`; default-OFF door with a decide-by
date 14 days after landing), active only on the positioned-read spill path (`MEMRA_SPILL_IO`
pread, worker or direct) and only at the call sites that enqueue their kernel inside the same
cache scope (the batch-1 decode expert GEMMs, f32 and q8). A doorkeeper decides "cold": a block's
first miss is not admitted to a slot and its id enters a bounded FIFO ghost list (4 x slots, at
least 64); a miss whose id is in the ghost list admits exactly as today. The two bypass forms:

- `staged`: the first-miss block is copied into one dedicated device scratch buffer (not a cache
  slot, never published) and the kernel reads it there. This isolates the admission change.
- `mapped`: no copy; the kernel reads the pinned buffer through `cuMemHostGetDevicePointer`. The
  buffer stays owned until an event recorded after that kernel completes, then returns to the
  pool. Refused loudly if the driver will not map the pinned buffers.

Every other call site keeps the exact current admission program. The bytes and the kernel are the
same in all three arms, so the token stream must be identical to the byte oracle.

Correctness first: unit cells for the doorkeeper (first miss bypasses, second admits, FIFO bound)
and for buffer ownership under `mapped` (the buffer is not reused before its event); then the B3
correctness gates per visit (placement, token ids equal the byte oracle, zero read errors, zero
mmap fallbacks), plus the bypass line: `[moe-bypass] mode=<m> bypassed=N ghost_admits=M` with
N > 0 on the bypass arms and absent on the baseline.

Timing (5090 half now; the PRO 6000 half needs a target card): arms `worker16` (baseline, the B3
lock row verbatim), `bypass-staged`, `bypass-mapped` (the same env plus the door), the capped
regime of section D (20 GiB swapless scope, cold start per visit, idle-gated round cells, GPU
co-tenant gate), ten rounds, forward and reverse order alternating. Verdict per arm vs worker16:
B3's rule verbatim (winner median >= 1.05 with at least 4 of 5 per order, loser <= 0.95, otherwise
flat; regime unscored if more than 2 contaminated visits in any arm), the B1 read gate reported
beside it as post hoc, as for B3 on the 5090. A 5090 winner sets at most a 5090 default; the door
stays default-OFF until both rigs have a row.

Section F, precedent (added 2026-09-26, still before any code or data): the `staged` arm is the
second-miss ghost filter with transient staging that was measured a net loss and removed on
2026-07-08 (`MEMRA_MOE_GHOST`, docs/FLAGS.md removed doors; 5090 spill 24.2 -> 25.0 tok/s with
it off, 2026-07-06), because every cold block paid two host-to-device copies. It stays in this
cell as the registered control that carries the admission change without zero-copy: `mapped`
changes the first miss from one copy (baseline) or two (staged) to none. If `staged` loses again,
its value is deleted from the door in the lane that measures it.

Section D, resync amendment (2026-09-26 about 07:53Z, after the requested 07:28Z rig reboot, before any
further 5090 cell):
1. The reboot remounted `/data` with mount id 250 (was 242; same device and filesystem). The
   collector correctly refused the stale proof on rounds 5 to 10, and the round driver wrongly
   continued past the refusal (fixed: a non-lost failure now stops the regime). `/data` is
   re-proven (`rtx5090/proof/`, PASS, mount id 250; the pre-reboot proof is kept in
   `rtx5090/proof-prereboot-mount242/`). Capped rounds 1 to 4 ran under the first proof, rounds 5
   to 10 run under the second; both are the same proven device and filesystem, every visit is a
   cold start, so the ten rounds still pool. The refused and interrupted cells are kept, never scored.
2. G2: the laptop reports `power.limit` as `[N/A]`, which the frozen `h2d-probe` refuses
   ("unknown power limit"). The probe now accepts exactly the literal `[N/A]` and records it
   verbatim; every other non-watt value still refuses, and the copy and timing code is unchanged.
   The G2 5090 probe is therefore built from the lane tip (hash in `rtx5090/build-g2/`), not from
   BOX27's engine source; G2 compares pinned with pageable inside one binary, so no cross-rig
   binary identity is claimed for it.
3. The mapped GPU ownership cell runs under `flock -n -E 75 /tmp/memra-5090.lock` after the same
   idle wait (the queue's first form failed on its own quoting before running anything).

Sections D and F, fallback amendment (2026-09-26, after the capped data and before any bounded
or f17 visit): the capped regime showed worker-path demand reads falling back to mmap when every
pinned buffer is busy (OWED 26), on arms the registered gate does not check. For the 5090 bounded
regime and for the OWED 17 cell, a visit whose `[spill-pread]` totals line shows any fallback is
unclean for timing (it still gates correctness, token ids against the oracle). It counts toward
the contamination limit like a co-tenant visit. Capped keeps its registered verdict unchanged.

Bounded sizing record (2026-09-26 08:55Z, before any bounded visit): two anon-peak runs, maximum
RssAnon plus RssShmem 864,223,232 and 864,210,944 bytes. The second run's sizing tool reported
`all_correct: false` because its direct16 visit fell back to mmap 9 times (OWED 26); that is not a
sizing fault, and the bound is the larger peak of the two runs plus 7,000,000,000:
MemoryMax = 7,864,223,232 bytes, passed to the queue explicitly.

Section E, 5090 sizing correction (2026-09-26 about 10:08Z, after the first 1 GiB pair failed,
before any rerun): with `--host-mb 4096` the default 50% tenant share cap is 2,147 MB, below the
1 GiB cell's 17 entries (17 x 127.2 MB = 2,162 MB), so the oldest entry (probe 1's prefix) was
evicted before export, in both arms alike (`handoff-1g` round 1: probes 2 to 4 restored 6,496
cached tokens with text identical to cold, probe 1 missed). BOX27's 16 GiB budget never bound the
cap. The 1 GiB cell therefore also runs with `--tenant-pct 100` (B2 amendment 3's rule, already
registered for the 8 GiB cell); host budget unchanged. The failed pair is kept as
`refused-handoff-1g-tenant-cap`, never scored.

## G. OWED 26: worker demand reads that fall back to mmap (registered 2026-09-26 about 11:18Z, before any code)

Routed to F by the coordinator (the spill positioned-read path). Mechanism, from the code: in
worker mode a demand read calls `submit_worker_with_admission(.., Demand)`; when no pinned buffer
is `Free` it returns `Ok(None)` and `dispatch_disk` reads the block through mmap with
`[spill-pread] falling back to mmap: worker read ring is busy`. The bytes are the same, but the
visit then runs two I/O programs. The blocking `pread` arm never does this (it waits in
`wait_for_one`), which is why `pread16` shows zero fallbacks everywhere.

G1, visibility first (applies to every cell that starts after this commit; recorded rows are
re-read, not rescored). The runner parses the fallback count from the `[spill-pread]` totals line
for every positioned-read arm and writes it into `visit.json` and the per-arm summary. A direct
arm with any fallback stays a correctness refusal (registered B3 amendment 2). Any other
positioned-read arm with a fallback is unclean for timing and counted toward the contamination
limit (the D and F fallback amendment, now in the runner itself, not only the pool tool). An mmap
arm that prints a positioned-read line is already refused.

G2, the fix (`fix(engine)`, no door: a fallback that is not an error is a defect). A demand submit
that finds no free buffer waits for one instead of switching program, in this order: reap
completed H2D events; if a buffer is in `H2d` with its event, wait on the oldest such event; else
if a buffer is `Reading` or `Canceled`, block on the next worker completion; else every buffer is
`Ready` or `Failed` and owned by a prefetch ticket, so the pool returns `HeldByPrefetch` and the
cache cancels one prefetch ticket (not the requested block) and retries; else (only buffers whose
H2D completion is unknown) synchronize the compute stream once. Each step waits on a concrete
in-flight completion, so the wait is finite while the device and the workers progress. Liveness
bound: a demand read that has waited 30 s in total errors loudly (`demand wait exceeded 30 s`), is
counted (`demand_wait_timeouts`), and only then takes the existing mmap error path; the bound is a
watchdog, never expected to fire. New counters on the totals line: `demand_waits`,
`demand_wait_ns`, `prefetch_cancels`, `demand_wait_timeouts`.

G2 red arm: a GPU unit cell fills every pinned buffer with completed reads marked `H2d` behind a
busy compute stream, then submits a demand read. On the pre-fix code the submit returns `None`
(red, run in a scratch worktree at the pre-fix commit with only the test added, its failing log
kept); on the fix it returns a ticket after waiting, with `demand_waits == 1`. A second cell holds
every buffer `Ready` under prefetch tickets and asserts `HeldByPrefetch`, then the cache-level
cancel-and-retry is exercised by the serving-shape check below.

G2 serving-shape check (5090, unscored, like the door-off diagnostic): `worker2` and `worker16`
visits on the fixed build, capped scope, cold start: zero fallbacks, token ids equal the byte
oracle, the new counters printed. The fix then becomes the build for the OWED 17 cell and for both
PRO 6000 sittings; the B3 capped and bounded rows keep the B3 build (engine source equal to BOX27's).

G3, re-reading recorded rows: a census of every committed receipt whose totals line shows
fallbacks (`owed26/CENSUS.md`): lane, cell, arm, visits affected, counts. Nothing recorded is
rewritten; the BOX27 worker2 correction note stays as written.

Telemetry amendment (coordinator, 2026-09-26, same time): the collector already writes a 250 ms
GPU CSV per cell (SM and memory clock, power draw and limits, temperature). From the next cell on,
every timed visit (runner visits, handoff export and import windows, G2 visits) also runs its own
250 ms `nvidia-smi` sampler with `clocks.sm`, `clocks.max.sm`, `power.draw`, `enforced.power.limit`,
`temperature.gpu` and `clocks_event_reasons.active`, and its record carries a summary: SM clock
median and minimum, power draw median and maximum, maximum temperature, and the share of samples
with each active throttle reason. Recording only, never a gate. Recorded cells get the same
per-visit summary post hoc from the collector CSV (no reason bits there).

Section F build amendment (2026-09-26, after the OWED 26 fix and before any timed OWED 17 visit):
the OWED 17 ten-round cell runs on the fix build (`owed26/build/`, the OWED 17 door plus the
OWED 26 demand wait), not on `owed17/build/`. Reason: on the earlier build every arm's visits carry
ring-busy fallbacks (smoke: 2,372, 3,646 and 8,287), which the fallback amendment makes unclean, so
that cell could only come out unscored while measuring a mixed program. It runs only after the
OWED 26 red/green cells and the serving-shape check pass. Arms, oracle, regime and verdict rule
are unchanged. Both PRO 6000 sittings use the same fix build.

Section F correctness addition (2026-09-26, before any spec cell with the door on): `run-spec`
K=1..8 self-consistency for `bypass-staged` and `bypass-mapped` on the fix build
(`m1-spec-cell.py` with the F lock, B3's verdict line `=== SELF-CONSISTENCY PASS ===`), on the 5090
before the timed cell and again in the PRO 6000 sitting.

Section G, first serving-shape check FAILED (2026-09-26 11:56Z) and its correction, registered
before the corrected code: the fix build's first visit (`worker2`) generated tokens equal to the
oracle but ran at 0.65 tok/s and stalled in run-gen's 32-step steady-state window; it was stopped
after 729.6 s (`owed26/5090/owed26-serve-attempt1`). Cause, from the code: `wait_for_any_buffer`
reaps H2D completions again; when an event completed between the submit's own reap and this one,
a buffer is free and no read is in flight, and the code still blocked in `recv_timeout` for the whole
30 s bound. Correction: if the second reap freed a buffer, or a buffer is free with nothing in
flight, return at once and retry the submit; every blocking wait on a worker completion is capped
at 50 ms per iteration and the loop re-evaluates, so a mis-placed wait costs at most 50 ms, and the
30 s liveness bound still ends the whole demand wait. Red arm: a GPU cell calls the wait with one
buffer free and nothing in flight, and a second with every H2D event already complete; both must
return within 1 s. On the e5d899500 build both take the full 30 s (red, run in the scratch worktree
at e5d899500 with only the cells added); on the corrected build they return at once. The serving-
shape check then reruns in full.

Section D, bounded OOM record (2026-09-26 14:53Z) and fix, before any bounded visit: the first
bounded cell died 3.6 s in, quoted from the kernel: `Memory cgroup out of memory: Killed process
643352 (python3) total-vm:17828376kB, anon-rss:7636208kB` (systemd: `The kernel OOM killer killed
some processes in this unit`). The runner's `sha()` read the whole 18.2 GB artifact into memory to
check its hash, which fits the 20 GiB capped scope but not the 7,864,223,232-byte bounded one.
Fix: `sha()` streams in 1 MiB blocks (the collector's own digest shape; same values). No visit
changes; the refused cell is kept as `refused-bounded-oom-runner-hash`.
