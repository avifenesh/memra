# The unified ragged step (v2)

Step 6 of the batch-spec lane. Same-model plain decoding sessions ride the pooled MTP cohort's
lockstep forwards as one-row segments, so a mixed tick runs one forward per round over spec
verify rows and plain decode rows together.

## 1. Epilogue split (pure refactor)

`decode_batch_epilogue` (decode_batch.rs) is now two stages that the batched step runs back to
back, in the old device order:

- `decode_batch_sample_stage`: pristine parks, device penalties, grammar masks, the device draws
  and the `[B]` token readback (skipped for the pending step). Returns a `SampledStack`.
- `decode_batch_sample_rows`: the lean logits parks and the host rows.

The `pos` bump and the pending-step handoff stay in `decode_batch_epilogue`. The split lets a
caller sample any `[n, n_vocab]` logits stack with the batched step's own epilogue, without the
`pos` bump (the pooled walk already advanced each cache).

### Receipts (RTX 5090 Laptop, q9 + owntrim draft)

Reference: the 3cd42279ee binaries (`~/.cache/memra-batchspec-bins/int`). Hashes in
`unified/refactor/binaries.sha256`.

- Serial battery (`battery/battery.sh`, `MEMRA_SPEC_GATE_MODEL=0`) against `int-band`:
  **BYTE-IDENTICAL** on every cell, including the 56 + 4 + 4 `[spec-acc]` lines
  (`unified/refactor/compare-battery-band.txt`).
- MTP pool cell (`battery/pool_cell.py`) against `pool-int-a`: **14/14 streams identical**
  (`unified/refactor/compare-pool-cell.txt`). Single run; its tok/s are not a speed claim.
