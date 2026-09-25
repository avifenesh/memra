# OWED 14: bind the collector's NVMe label to the M1 proof (proposal for D, routed by the lead)

**Not applied.** `tools/tier-battery.py` is D-owned; this directory carries the change as
`tier-battery-storage-proof.patch` (SHA-256 `f264bae8...5990952`, applies cleanly with
`git apply` to `ae23ea91f`, whose collector is SHA-256 `4861050b...fcea732`) plus
`test_storage_proof.py`. Registration: `CPU-PREREG.md` OWED 14.

## Defect

`capture_storage` labels a root `nvme-ancestry` (`nvme_proven: true`) when any `nvmeXnY` name
appears in `lsblk -s` of the findmnt source. An emulated NVMe in a guest, an NVMe-oF namespace
and a RAID set with a network member all carry such names, and a bind mount's findmnt source
(`/dev/md0[/subdir]`) makes lsblk fail, so a real local volume reads as unproven. No archived
capture ever carried `nvme_proven: true` (searched 2026-09-25), so tightening breaks no receipt.

## Change

- `--storage-proof PROOF.json` (requires `--storage-root`): the M1 proof receipt from
  `research/spill-f-20260919/m1-nvme-proof.py`, public or private form.
- `m1_proof_binding`: schema `m1-nvme-proof-v1`, `verdict=PASS`, `class=nvme-local-direct`, no
  reasons, `tool_sha256` equal to the checkout's proof tool, and a live identity equal to the
  receipt's (device, mount id, filesystem id or its 16-hex SHA-256). The receipt is copied into
  the capture as `STORAGE-PROOF.json` and hash-bound.
- `capture_storage` labels `nvme-local-direct` only through that binding. Without a proof the
  name match is recorded as `nvme_name_hint` and the class is `nvme-name-only-unproven` (or
  `overlay-unproven`), which needs `--allow-unproven-storage` as today. A supplied proof that
  fails is recorded as `m1-proof-refused` with the reason and refuses even under
  `--allow-unproven-storage`: a bad proof never downgrades silently. The findmnt `[subdir]`
  suffix is stripped before lsblk.
- `validate_storage_record` (extracted from `validate_capture`, same checks plus one): an
  `nvme_proven` record must be class `nvme-local-direct` with the M1 label and a hash-valid
  archived proof whose verdict and tool hash match the binding.

## Evidence (raw logs in `cpu/`)

| Run | Result |
|---|---|
| `test_storage_proof.py` on the patched collector | 19 passed, 1 skipped (`owed14-patched-live.log` is the same suite with the live case on) |
| same, `M1_LIVE_PROOF=1` (real proof tool, 64 MiB binding, this bare-metal rig) | **20 passed** |
| D's `crates/memra-tier/tests/battery` suite on the patched collector | **87 passed** (`battery-suite-patched.log`) |
| `test_storage_proof.py` on the unpatched collector (red control) | every non-skipped case fails: 3 failures, 16 errors (`owed14-unpatched-red.log`) |

The green cases use a synthetic PASS receipt bound to the live identity, so the suite runs on
CI runners (VMs, where a real proof fails A1 by design). Suggested home once applied:
`crates/memra-tier/tests/battery/test_storage_proof.py` (its `ROOT` lookup already walks up to
the repo root).
