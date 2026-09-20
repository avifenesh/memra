# M1 storage decision — 2026-09-20

**Recommendation: NO-GO on spending for the currently advertised VM-capable
candidate without a storage-backing commitment.** The private candidate is
recorded in the lead's untracked `LANE-LOCAL-NVME.md`; no instance lifecycle
action was taken. This decision concerns obtaining M1 evidence, not whether
the candidate can run CUDA. GPU-copy G2 can proceed independently on BOX2.

## What a VM would and would not prove

`vms_enabled=true` is eligibility metadata, not proof that a particular
allocation is a VM, exposes local disks, or passes a physical NVMe controller
through to its guest. The advertised disk label and bandwidth do not bind a
writable filesystem to a device. Prior read-only discovery is retained in
`NVME-OFFER-SHORTLIST.md` and `NVME-PROOF-OPTIONS.md`.

A guest can prove its visible mount-to-block chain and its write/read/direct-I/O
behavior. A virtio disk proves a virtual block interface; an NVMe device name
can also be emulated. Neither alone proves physical local NVMe. M1 requires
either a physical controller passthrough chain with independently established
non-emulation, or an operator-supplied, exact guest-disk-to-host-backing mapping
and the complete host mount/block/NVMe ancestry. Network-backed or untraceable
virtual disks remain unproven even when fast and even when `O_DIRECT` succeeds.

## Spend gate

Change to conditional **GO** only after the operator confirms the exact
allocation type and commits to supplying either physical NVMe passthrough or
the matching host-backing evidence, plus a dedicated writable scratch mount.
The lead/owner must separately authorize lifecycle operations and a bounded
inspection window; this lane has no lifecycle authority. Refresh availability
and the private quote at that point. Do not spend merely to discover whether
`vms_enabled` means passthrough. Prefer a proven writable NVMe-backed bind on
existing capacity if it can be supplied.

Post-allocation admission is fail-closed: no physical ancestry, no writable
dedicated mount, insufficient free space, or missing required CUDA/tooling
means no scored M1 cells. A refused/unknown probe is evidence of that failure,
not permission to infer backing. No raw-device write, format, mount alteration,
cache drop, or privilege escalation is part of this sequence.

## Exact in-guest proof sequence

Run metadata probes first. Keep original captures private (controller serials,
namespace IDs, PCI addresses and mount options may identify a machine); publish
sanitized topology and hashes only. `P` is an operator-approved, lane-owned
scratch directory, never `/root/artifacts` or another lane's directory.

```sh
P=/scratch/spill-f
# Admission: supplied directory and actual filesystem, not an overlay alias.
test -d "$P" && test -w "$P"
findmnt -J -T "$P" -o TARGET,SOURCE,FSTYPE,MAJ:MIN
lsblk -J -o NAME,KNAME,PKNAME,MAJ:MIN,TYPE,SIZE,ROTA,TRAN,FSTYPE,MOUNTPOINTS
stat -f -c '%T %s %b %f %a' "$P"
# Inspect device identity/ancestry, including virtual buses, without creating nodes.
for D in /sys/block/*/device; do
  test -e "$D" || continue
  readlink -f "$D"
  for F in vendor model; do
    test ! -r "$D/$F" || cat "$D/$F"
  done
done
# Follow partition parents and every mapper/RAID slave from the mount's
# MAJ:MIN through /sys/dev/block/<major>:<minor>/slaves recursively.
# The executable traversal is in NVME-PROOF-OPTIONS.md.
lspci -nnk
# If nvme-cli and the already-exposed controller node are present:
if command -v nvme >/dev/null; then
  for D in /dev/nvme[0-9]*; do
    test -c "$D" || continue
    nvme id-ctrl "$D" -o json
  done
fi
```

A missing `nvme` command or controller node is recorded, never repaired with
`mknod`. `nvme id-ctrl` corroborates controller characteristics; it is not
independent proof against emulation. Compare guest and host mapping where
needed. Retain every RAID/mapper parent, not just one favorable member.

Next, and **only after ancestry acceptance**, run this disposable-file smoke
through `tools/tier-battery.py` with `--storage-root "$P" --timeout 300` and the
canonical rig lock. D's storage-root binding fix must be integrated so the
capture describes the filesystem actually written. Save the following as a
lane-owned script before invoking the collector (never run it bare):

```python
import hashlib, os, pathlib, sys, tempfile
root = pathlib.Path(sys.argv[1]).resolve(strict=True)
data = bytes(range(256)) * 16384  # 4 MiB, bounded and nonuniform
fd, name = tempfile.mkstemp(prefix="m1-proof-", dir=root)
try:
    with os.fdopen(fd, "wb", buffering=0) as out:
        view = memoryview(data)
        while view:
            written = out.write(view)
            if not written:
                raise RuntimeError("short write made no progress")
            view = view[written:]
        os.fdatasync(out.fileno())
    with open(name, "rb", buffering=0) as inp:
        returned = inp.read()
    assert returned == data, "M1 byte mismatch"
    print("M1_WRITE_FDATASYNC_IDENTITY", len(data), hashlib.sha256(data).hexdigest())
finally:
    os.unlink(name)
```

The collector command shape is `python3 tools/tier-battery.py --rig rtx5090
--timeout 300 --storage-root "$P" --out <unique-receipt-dir> --execute python3
<lane-owned-script> "$P"`. The snippet has not run; it is an admission recipe,
not a receipt. Retain exit/stderr, before/after mount identity and free-space
metadata, fdatasync outcome and byte identity. A successful buffered readback
may hit page cache; fdatasync success is not a power-loss test or a cold-read
benchmark. Subsequent direct-I/O cells establish their own alignment/API gates.

## M1 cells unlocked after the full proof

- Collector-bound local-storage baseline: bounded positioned read/write,
  byte identity, explicit durability step and exact filesystem/device chain.
- Matched mmap/page-fault versus worker positioned-I/O cells, with warm/cold
  conditions stated rather than inferred from timings.
- Aligned `O_DIRECT` admission and measured arm, retaining exact refusals;
  success is I/O semantics, not proof of uncached hardware access.
- io_uring admission and worker comparison where kernel/tool support exists;
  unavailable io_uring remains an unrun/refused arm, never silently replaced.
- Storage-to-pinned-host-to-GPU pipeline cells with identical payloads,
  bounded buffers and stage timing, after the separate GPU identity gates.

Each still needs its own correctness, exclusion, telemetry and balanced timing
protocol. Passing ancestry does not qualify active/prefix KV, expert eviction,
serving, multi-card transport or any particular model. BOX2 overlay results
remain development diagnostics; they are not retroactively relabeled NVMe.
