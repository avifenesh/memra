# Attempt 7 diagnosis — Q27 restored-hit early EOS

This directory is the byte-for-byte copy of the interrupted remote
`/opt/dl-image/nvme/cx-cachesize/raw/scored/` segment recovered after the harness
process-group sweep. The remote source remains untouched.

## Reusable completed boots

- `r01-q27-b01024`: `sweep.exit=0`, final verdict `PASS`, seven clean cells.
- `r01-q27-b04096`: `sweep.exit=0`, final verdict `PASS`, seven clean cells.

Both boots ran after `CACHESIZE_LOCK_ACQUIRED` at `2026-08-13T00:05:06Z`, with the runner
holding both `/tmp/memra-gpu.lock` and `/tmp/memra-gpu-1.lock`. Their server failure scans are
empty. Per resume steering, these completed measurements must be reused rather than repeated.

## Excluded boot

`r01-q27-b08192` is not scoreable: `sweep.exit=1` and its final summary is `FAIL`. In the c=4
cell, paired working key `prefix_id=87` was a full 4,860-token cache hit but returned HTTP 200
with `finish_reason=stop` after 11 rather than 60 completion tokens. The request id was
`cmpl-a0261e2e33683fa5dee848700a988ff3`; its text hash was
`ddbbcf35ae93821874be64e15f63feecb2471bbe6ea0c23b63fa00b1e98a9b73`.

The exact same paired key was a miss in the 1,024 and 4,096 MiB boots and completed 60 tokens
with `finish_reason=length` both times. Its text matched the failed restored hit through the
11 visible tokens and then continued. Attempt 3 produced the same 11-token text hash on a Q27
restored hit at 16,384 MiB. This falsifies the earlier claim that batched working-set seeding was
the cause; attempt 7 used the corrected sequential seed path.

The failed cell's counters reconcile exactly: 20 admitted/completed requests, 14,580 cached
tokens, three hits, 17 misses, 1,151 output tokens, zero admission defers, and zero OOM parks.
The server-failure scan is empty. The captured evidence establishes a repeatable exact-length
failure on a restored hit. Because those hits and paired misses can follow different prime/decode
batch classes, it does not by itself establish cache corruption or distinguish snapshot/restore
state from a batch-shape-dependent numerical path. A serial cold-versus-hit oracle is required.
