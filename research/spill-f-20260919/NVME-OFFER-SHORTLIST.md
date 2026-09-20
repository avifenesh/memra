# NVMe-provable rental shortlist — 2026-09-20

**Recommendation: the single matching one-card RTX 5090 VM-capable class,
conditional on physical block ancestry evidence before rental.** No instance
was created. VM capability is not itself NVMe proof; the available search API
cannot certify PCIe storage passthrough. There is no unconditional proven offer.
The concrete offer identifier and current quote are in the lead's **untracked**
`LANE-LOCAL-NVME.md`; refresh availability before any owner-authorized rental.

## Read-only discovery

`search offers --help` documents `vms_enabled` as a boolean query field.
There is no separate VM instance-type option in search help. `--type` chooses
on-demand/reserved/bid pricing, not virtualization. Raw response keys include
`vms_enabled`, `is_vm_deverified`, `hosting_type`, `resource_type`; no
`is_vm`, `virt`, `machine_type` or `bare_metal` key was returned. Numeric
`hosting_type=0` has no documented mapping here and is not interpreted.

Queries (default verified/rentable filters also made explicit):

```text
verified=true rentable=true vms_enabled=true cuda_vers>=13.1 cpu_ram>=60 disk_bw>=3000 inet_down>=1000 inet_up>=1000
```

A broad VM query avoided GPU-name alias ambiguity and returned RTX 4090 and
RTX 5090 classes only. Filtering it to RTX 5090 or RTX PRO 6000 leaves **one**
concrete offer; no PRO 6000 match. The narrowed 5090 query returned the same
candidate. CLI RAM filter units are GB; raw RAM is reported in MB.

| Class | Cost class | Advertised properties | What can be proved |
|---|---|---|---|
| One RTX 5090, VM-capable, verified | cheap (single-card development candidate; actual quote private) | `vms_enabled=true`, `is_vm_deverified=false`, 126398 MB RAM, driver CUDA ceiling 13.2, disk 7812 MB/s, down/up 7314.6/5385.6 Mb/s, disk label `nvme` | A VM can expose its own mount/block topology. Physical local NVMe requires passthrough or independently captured host backing evidence; label and bandwidth are not proof. |

These are offer advertisements, not measured performance. CUDA ceiling does
not establish an installed 13.1 toolkit. Network tests are advertised values,
not guaranteed bandwidth. Cost class is not a bill or reservation.

## Proof method and acceptance probes

Select a VM image/configuration with root access and either a PCIe NVMe
controller passed through to the guest or an operator-provided host backing
capture for the exact guest disk. A generic virtio disk with no backing proof
still fails M1. A VM with an emulated NVMe controller also fails physical
ancestry without backing proof. Bare metal would allow a direct proof, but
no bare-metal offer flag or qualifying bare-metal candidate was discovered.

Read-only commands, with `PATH_UNDER_TEST` resolved to the proposed scratch
filesystem (never create, format or write a raw block device):

```sh
findmnt -T "$PATH_UNDER_TEST" -o TARGET,SOURCE,FSTYPE,OPTIONS,MAJ:MIN
lsblk -s -o NAME,TYPE,MAJ:MIN,PKNAME,TRAN,MODEL,MOUNTPOINTS
lspci -nnk
nvme list -o json
readlink -f /sys/dev/block/MAJOR:MINOR
# Follow every partition/mapper/slave to the physical controller.
ls -l /sys/dev/block/MAJOR:MINOR/slaves
nvidia-smi --query-gpu=name,driver_version,power.limit,power.max_limit --format=csv
nvcc --version
free -b
```

Record filesystem device → parent disk → NVMe namespace/controller → PCIe
ancestry, and establish that the controller is physical passthrough rather
than emulation. For virtio, require a matching host `findmnt`/`lsblk` capture
and mapping of guest disk to host backing file/LV and all physical parents.
Reject overlay, network backing, untraceable mapper/virtio ancestry, or
advertised `nvme` alone. Keep original topology captures private; publish a
sanitized proof with hashes and measurement conditions only.

Only after this proof and an installed CUDA toolkit check should an approved
collector perform O_DIRECT/storage cells. No current-container observation or
fast disk advertisement substitutes for that gate. If the seller cannot
supply the mapping, do not rent this candidate for M1; request explicit
passthrough/bare-metal capacity instead.
