# WP-A day-five storage capture adapter (WP-D handoff)

`../storage_capture.py` stays lane-local. `battery-dispatch.patch` is the proposed
**explicit** `tools/tier-battery.py --validate <CELL.jsonl> --schema storage-cell
--out <new.jsonl>` dispatch fragment; the shared collector is not edited by WP-A.
Apply/rebase the fragment in WP-D's lane, then run `python3
research/spill-a-20260919/day5/test_capture.py`. Its isolated temporary checkout
applies the patch and exercises the real CLI without modifying the source tree.

Contract:

- Exactly one complete start/end attempt; same nonempty run ID, command and
  identity; exit zero and diagnostic classification on both journal and capture.
- Capture/raw stdout/diagnostic CSV/stderr/snapshot hashes verified. Empty CSV
  and stderr are allowed only as hash-checked diagnostics, never as positive
  250-ms telemetry evidence. No tier counters, logits or token pairs fabricated.
- Canonical collector GPU lock required. This validates its recorded assertion,
  not independently reconstructing past OS lock ownership.
- Exact storage-bench roundtrip/restore argv, directly or via the canonical
  `bash -c <shlex.join(argv)>` wrapper. Shell operators/noncanonical quoting are
  refused. The wrapper avoids nested Cargo `--` collector parsing ambiguity.
- One actual StorageSample from stdout, strict existing collector schema, exact
  requested length/backend and capture time window. Optional supplied envelope
  must match both the run ID and captured sample, including nulls and failures.
- Preserve timestamps, raw telemetry descriptors/status and unchanged sample
  counters. Output is `capture-matched-not-qualified`, always qualification=false.
  Physical NVMe bytes and transfer counters remain unknown when the source says
  null. This is not a forced ON/OFF pair, model gate or serving receipt.
- No automatic CELL promotion: the new schema is explicitly selected. No new
  dependencies, runtime flags or numerical programs.

`../rig-cells-a.sh --box2 <owned-overlay-scratch> --out <new-receipt-dir> --cell
<cell> --approved-non-serving` executes **one** cell. CPU tests/builds run bare;
CUDA-touching cells go through the canonical-lock collector. GPU inventory must
be empty first. Each cell records the power limit and maximum (400 W limit, not
full power in this campaign), and mount types only. Sync the receipt and
commit/push before the next cell. `--dry-run` creates nothing.

All storage numbers are **overlay, development, not spill speed**. Sysfs NVMe
names without exposed block device nodes do not establish the overlay backing.
