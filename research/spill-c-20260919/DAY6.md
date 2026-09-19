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

## Step 2: expert dispatch adapter and native integration refusal

Selected public source: `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` at HF revision
`5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`; file
`Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`, 18,209,036,576 bytes, SHA256
`df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf`.
Unauthenticated HF API returned 200; metadata receipt `raw/day6-model-access.json`.
This matches the historical artifact hash in `research/percard-20260812/RESULTS.md`.
The non-MTP repository has different bytes and is not substituted. No download
was started before establishing a usable native injection boundary.

`SlruExpertDispatch` binds an already-installed immutable bank/source map to
original `(layer, projection, expert)` IDs; refuses absent IDs, wrong projection,
wrong byte lengths, and multi-plane records (this payload-only slice does not
silently drop scales). A host demand stays open through H2D and only finishes on
explicit completion. The underlying `BankService::with_slru` retains resident
bytes across repeated demands; tests show 4 demands / 2 physical reads, exact
bytes, unknown-ID refusal, and explicit drained charges. Native intrusive SLRU
is not replaced by the O(n) CPU policy model.

**experts-via-tier: NOT RUN — native owner-thread injection boundary unresolved.**
`DAY6-EXPERT-DISPATCH-PROPOSAL.diff` is an UNAPPLIED, rejected integration probe,
not working code: storing the owner-thread bank and BankLease in `MoeSlotCache`
makes `Engine` non-Sync, failing scoped pipeline worker compilation. The frozen
Rc ownership is intentional; neither unsafe Send nor cross-worker substitution
is acceptable. The patch also still needs every direct warmup/admit bypass fenced.
Raw DOCS_RS compile diagnostics are retained in `raw/day6-device-cpu/engine-check.log`;
that run is a Mac compile probe, not native CUDA evidence. The unsafe/incomplete
wiring was removed from live code; only the independently CPU-tested adapter lands.
No model argmax/spec PASS is claimed; the artifact is available and below the cap,
so this is a runtime integration blocker, NOT an artifact/size excuse.
