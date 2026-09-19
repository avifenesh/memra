# Day 6 — device publication and native expert binding

Repository: avifenesh/memra; lane/spill-c-20260919. Integration parent:
50c97665 (merged spill-integ3 and pushed with hooks).

## Step 1: frozen row-upload publication boundary

`bank/device_rows.rs` provides `RowUpload`: submit a single exact coalesced
host window through `TransferEngine::h2d`, validate segment byte counts/checksum,
current epochs, producer completion AND installed consumer wait, check the
owner-sealed ReadyView against the exact destination, then take the device lease.
Last-use registration, observed retirement and acknowledgement stay separate.
Rejected submission returns owned inputs; accepted errors leave a retryable
backend-owned ticket; Drop never pretends unknown DMA has retired.

`GatheredRows::submit_device` checks the supplied host bytes against every expanded
F32 bit (including duplicate order, negative zero and NaN payload), without
changing the numerical program. The PLE budget requests now retain the injected
governor's device-vector dimension so one governor can cover host and GPU work.
No MEMRA environment read or runtime default changed.

CPU fake tests cover producer-only refusal, stale epochs, short/corrupt data,
wrong-owner fence, single publication, cancellation, rejection ownership,
consumer-not-retired accounting, and native gather to H2D submission. Raw logs:
`raw/day6-device-cpu/`. Strict memra-tier clippy passes. A fixture initially
used mismatched governor dimensions and refused InvalidLayout; corrected to
exercise a single shared governor with a device vector.

**device-publish: NOT RUN — native CudaTransfers integration pending.**
This helper alone does not name a native device-publish seam. The existing
host-only GPU PLE gate verdict from day 5 is not device-transfer evidence.

Rented RTX 5090 cells must record configured power cap **400 W / 600 W maximum**,
verify current power metadata, use the canonical collector/lock and archive per
cell. No cell has run for this step.
