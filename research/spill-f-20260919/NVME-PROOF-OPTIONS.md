# M1: proving the writable storage path, not just discovering NVMe

Date: 2026-09-19. Status: **M1 unproven**. Read-only evidence:
[`raw/box2-inventory.json`](raw/box2-inventory.json) (19:01 UTC) and
[`raw/offers-storage-projection.json`](raw/offers-storage-projection.json)
(19:08 UTC). BOX2 is development hardware, not a serving qualification.
No mount, device-node creation, power change, or instance lifecycle action ran.

## What the current namespace proves

`lsblk` and sysfs expose three NVMe namespaces and md0 RAID0 (two NVMe
members). Their `/dev` nodes are absent. `/root` resolves to overlay, 200 GiB
quota; `/scratch` and `/workspace` are absent. md0-backed file binds appear
at container configuration files, but they are not a writable benchmark
directory. Device discovery and those file binds **do not connect overlay's
writable upper layer to md0**. Never benchmark or modify the configuration files.

A bind mount preserves an existing filesystem's `st_dev`/major:minor; it does
not manufacture physical ancestry. A provider-created bind of a dedicated,
empty directory from the md0 filesystem could expose that ancestry without
exposing the raw block device. It must be writable, independently mapped to
md0, and tied through `/sys/dev/block/9:0/slaves` to the NVMe members. Container
`/dev/md0` is not required for this read-only sysfs proof. A bind of `/root`
merely remains overlay. The tenant cannot recover an inaccessible host upperdir
by rebinding it; no privilege escalation or mounting an existing host filesystem
is proposed.

A loop file in overlay proves only `loop -> file -> overlay`, not physical
NVMe. Even `O_DIRECT` success proves an I/O API accepted the request, not the
backing device or bypass of every cache. A loop over an already-proven bind
adds a layer without helping M1. Creating `/dev` nodes would neither grant
cgroup device access nor establish a safe writable directory. Do not do it.

## Marketplace raw fields actually observed

The read-only invocation was:

```sh
~/.local/bin/vastai search offers --raw --limit 50 'num_gpus=1 gpu_name=RTX_5090'
```

Raw responses must stay in memory; run `search-storage-offers.py` to save only
the allowlisted projection. No account, instance, network identity or numeric
price is needed in public receipts.

- `vms_enabled`: observed boolean; two of 34 returned single-5090 offers were
  true. This is an eligibility hint, **not proof that a rented guest uses a VM,
  exposes a block device, or passes through physical NVMe**.
- `disk_name`, `disk_space`, `disk_bw`: reported model/capacity/bandwidth hints.
  An NVMe name or large advertised bandwidth cannot establish the mounted
  path's ancestry. One row even names a virtual disk.
- `avail_vol_ask_id`, `avail_vol_size`, `avail_vol_dph`: keys observed in the
  raw schema, but their values were deliberately not retained. They indicate
  volume-related offer metadata, not tested attachment support or local NVMe.
  Do not publish the identifier or price values.
- `hosting_type`, `resource_type`, `is_vm_deverified`: observed keys; values
  and semantics were not independently qualified. Do not infer VM/dedicated
  assignment from their names alone.
- `nw_disk_min_bw`, `nw_disk_avg_bw`, `nw_disk_max_bw`: observed keys, not a
  local-device provenance guarantee.
- `is_vm`, `bare_metal`, `volume_enabled`, `volume_support`: **not present**
  in this returned schema. Do not invent filters based on those fields.

Both narrower queries (`num_gpus=1 gpu_name=RTX_5090 vms_enabled=true` and
`vms_enabled=true`) timed out after 60 seconds. This is not zero availability.
The successful unfiltered sample already contained VM-eligible rows. Cost
classes in that projection are sample-relative terciles, not quotes: those two
rows fell in medium and dear classes. Availability and class must be rechecked
privately before any separately authorized action.

## Ranked options

| Rank | Option | What it can prove | Planning cost class | Missing condition |
|---|---|---|---|---|
| 1 | Ask for a dedicated writable NVMe/md0-backed bind on current development container | Writable filesystem major:minor maps through sysfs to NVMe, including any RAID layer | cheap (reuse; not a price quote) | Provider/owner must supply the path and approve access; none exists in current evidence |
| 2 | Separately authorized VM-eligible single-5090 allocation, contingent on storage topology evidence | Guest block device and direct-I/O behavior; physical NVMe only with passthrough or independently attested host backing | medium/dear in observed sample | `vms_enabled` alone is insufficient; virtio/QEMU remains virtual storage unless backing is proven |
| 3 | Separately authorized attached volume with documented local backing and mapped mount | Volume filesystem and block path, potentially local NVMe | medium planning assumption, not measured | Volume fields alone prove neither block exposure nor locality; network volume does not pass local-NVMe M1 |
| 4 | Overlay bind or loop workaround | Overlay/loop functionality only | cheap | Cannot pass M1; diagnostic only |

**One recommendation:** request a dedicated writable md0/NVMe-backed bind on
BOX2 and require the ancestry probe below before any storage score; keep M1
blocked if that cannot be supplied. GPU transfer G2 does not depend on M1.

## Exact read-only acceptance probes

Run only after the owner supplies a dedicated path (example `/scratch/spill-f`;
not assumed to exist). Commands inspect metadata and do not mount or write:

```sh
P=/scratch/spill-f
# Must exist as a writable directory, not a file bind or overlay alias.
test -d "$P" && test -w "$P"
findmnt -J -T "$P" -o TARGET,SOURCE,FSTYPE,MAJ:MIN
stat -f -c '%T %s %b %f %a' "$P"
lsblk -J -o NAME,MAJ:MIN,TYPE,SIZE,ROTA,TRAN,FSTYPE
# Avoid mount OPTIONS, serial numbers, UUIDs and host paths in public output.
python3 - "$P" <<'PY'
import json, os, pathlib, sys
p = pathlib.Path(sys.argv[1])
st = p.stat()
root = pathlib.Path('/sys/dev/block') / f'{os.major(st.st_dev)}:{os.minor(st.st_dev)}'
seen = set()
def visit(node):
    node = node.resolve()
    if node in seen:
        return
    seen.add(node)
    if not (node / 'dev').is_file():
        raise SystemExit('UNPROVEN: no sysfs block mapping')
    row = {'name': node.name, 'major_minor': (node / 'dev').read_text().strip()}
    slaves = sorted((node / 'slaves').glob('*'))
    row['slaves'] = [x.name for x in slaves]
    print(json.dumps(row))
    if (node / 'partition').exists():
        visit(node.parent)
    for slave in slaves:
        visit(slave)
visit(root)
PY
```

Accept only a complete mount-to-block-to-NVMe chain; record RAID members and
whether the namespace is virtual. A new path must additionally pass the lane's
bounded write/read identity and direct-I/O tests **through the collector**, on
lane-owned disposable files only. Those tests are separate from the read-only
ancestry probe and have not run here. No raw-device writes, global cache drops,
mount changes, or lifecycle commands are permitted in this lane.
