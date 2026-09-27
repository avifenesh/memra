# M1 proof tool: fixture and live controls (2026-09-25)

**`m1-nvme-proof.py` passes 24 of 24 fixture controls and 7 of 7 live controls on this rig.**
Every admitted topology returns no reason; every fail-closed branch of `M1-PREREG.md` section A
returns its reason. This validates the tool, not any rented box: no M1 cell has run, and the
local timings below are proof bookkeeping (a verified QD1 loop), not storage throughput.

## Fixture controls (`test-m1-nvme-proof.py`, raw `m1-proof-controls/fixture-tests.log`)

Fixture sysfs/proc trees use the kernel's relative symlinks, so the resolution code runs
unchanged. PASS cases: bare-metal NVMe partition; LVM over an NVMe partition; md RAID0 over two
NVMe namespaces; an NVMe native-multipath head whose path is PCIe. FAIL cases, each asserting its
own reason: `hypervisor` CPU flag; DMI naming a virtual platform; a guest with the CPU flag
hidden, caught by the PCI census (ICH9 LPC of a Q35 machine); an empty PCI census; a virtio
disk (leaf class and platform census both fire); an emulated NVMe controller (vendor 0x1b36 and
the emulator model string both fire); an NVMe-over-TCP namespace; an NVMe-over-TCP multipath
path; md RAID1 with a SATA member; a loop device; an unresolvable node; a node without a `dev`
attribute; a dm cycle (terminates, "no NVMe leaf"); an `Unknown` PCIe link speed. Plus mountinfo
overmount and escape parsing, component-aware prefix matching, stat parsing, nested
sanitization, the tmpfs CLI path, and refusal to overwrite a receipt.

## Live controls on this rig (bare metal, laptop RTX 5090 host, two PCIe 4.0 x4 NVMe drives)

Private captures stay outside git; public receipts in `m1-proof-controls/`. Every receipt
carries `tool_sha256` `d59851b624339e16d2290b21b94567a593645266eb0e5dd83d4490decbb78864`, the exact
tool committed beside it. The I/O binding is 1 GiB (2,097,152 sectors) in every PASS row.

| Control | Path shape | Verdict | Block graph | Leaf write / read sectors | Foreign write sectors (upper bound) |
|---|---|---|---|---|---|
| `root-partition-nvme` | directory on the root ext4 | **PASS** | `nvme0n1p2` partition, `nvme0n1` namespace | 2,100,616 / 2,097,152 | 3,464 |
| `lvm-nvme` | directory on the LVM ext4 (artifact store) | **PASS** | `dm-0` (LVM), `nvme1n1` | 2,097,584 / 2,097,152 | 432 |
| `bind-subdir-lvm-nvme` | a bind mount of a subdirectory of the LVM ext4 (mount root `/cache`) | **PASS** | `dm-0` (LVM), `nvme1n1` | 2,097,664 / 2,097,152 | 512 |
| `container-bind-root-nvme` | inside a docker container, a host directory bind-mounted at `/scratch` | **PASS** | `nvme0n1p2`, `nvme0n1` | 2,097,776 / 2,097,160 | 624 |
| `container-overlay-root` | inside the same container, its overlay root | **FAIL** | none | not run | not run |
| `tmpfs` | `/dev/shm` | **FAIL** | none | not run | not run |
| `squashfs-loop` | a read-only squashfs on a loop device | **FAIL** | `loop0` (loop) | not run | not run |

Run history: this is run 2. Run 1 (same seven paths, same verdicts; receipts kept in
`m1-proof-controls/run1/`) used a tool revision, SHA-256 `d92163c9...ed0ba7e`, whose detection
tables named a cloud provider in words the public-boundary gate refuses, so that revision's bytes
are not published. The names were removed (that provider's PCI vendor id stays in the census, so
its VMs and bare-metal hosts both still fail closed), the tool hash was added to every receipt,
and all fixtures and controls were rerun.

Verdict lines, verbatim from the logs:

```text
root-partition-nvme       M1-PROOF verdict=PASS class=nvme-local-direct reasons=0
lvm-nvme                  M1-PROOF verdict=PASS class=nvme-local-direct reasons=0
bind-subdir-lvm-nvme      M1-PROOF verdict=PASS class=nvme-local-direct reasons=0
container-bind-root-nvme  M1-PROOF verdict=PASS class=nvme-local-direct reasons=0
container-overlay-root    M1-PROOF verdict=FAIL class=None reasons=4   (A3 overlay, A3 anonymous device 0:149)
tmpfs                     M1-PROOF verdict=FAIL class=None reasons=4   (A3 tmpfs, A3 anonymous device 0:27)
squashfs-loop             M1-PROOF verdict=FAIL class=None reasons=6   (A2 not writable, A3 squashfs, A4 loop0)
```

What the controls establish:

- **The rented route's shape works as designed.** A container on a bare-metal host sees the host
  kernel's sysfs, and a bind-mounted host directory traces to its PCI NVMe controller exactly as
  on the host (`container-bind-root-nvme`), while the container's own overlay root fails closed.
- **Bind mounts are followed by mount id, not by path name.** In run 1, `bind-subdir-lvm-nvme`
  was first run under a mislabeled name ("root-nvme") because its path looked like a
  root-filesystem directory; the tool traced it to the LVM volume anyway. It was renamed to what
  it is and the real root-partition control was run separately.
- **The foreign-I/O indicator sees co-tenants.** Run 1's `lvm-nvme` overlapped other lanes'
  writes on this shared rig: 623,656 sectors (about 305 MiB) above the payload; run 2 saw 432.
  The proof passes on coverage; a scored cell with run 1's indicator would be contaminated
  (section B gate).
- **This rig's storage class is now proven** for the 5090 half (OWED 19): the artifact store
  (LVM over PCIe NVMe) and the root filesystem are `nvme-local-direct`. The 5090 cells still
  wait for the owner's card reset.

Timings in the receipts (about 0.5 s for the O_DIRECT write plus `fdatasync`, about 1.3 s for the
verified QD1 read) include pattern generation and byte comparison and are not throughput.
