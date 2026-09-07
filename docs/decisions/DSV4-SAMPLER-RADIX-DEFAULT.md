# DSV4 sampler defaults to radix, 2026-09-07

Choose radix candidate ordering for an unset `MEMRA_DSV4_SAMPLE_SORT`.
Keep `comparison` as the explicit legacy arm and byte-identity oracle for gates.
NaN rows still use the comparator; the per-thread gate override is unchanged.

The 2026-09-06 receipts establish identical candidate order over 49,643,520
elements and identical tokens in 1,536 sampled parameter cases. CPU sampler time
fell from 14.4-14.9 ms to 2.9-3.2 ms per row. Plain and DSpark decode gained
29-38% at 256 and 8192 context, with six observations per sorter/decoder arm.
Private ops receipt namespace: `sampler-order-20260906-*`.

The 2026-09-07 comparison-pinned attention-TP envelope observed teacher-forced
prime at 41.8 tok/s versus sampled decode at 24.9 tok/s on the same binary.
This supports the CPU sampler bottleneck; it is not a new sorter A/B.
Private ops receipt: `attention-tp-sampled-2383-r1`.

Ordering, probability arithmetic and position-keyed draws preserve token identity
by construction. Serving inherits the default through `dsv4_sample_row`.
Sampler-specific HTTP/release gates remain pending. The selector remains under
the gate-selector exception to door removal; selecting `comparison` restores the
legacy arm. See [FLAGS.md](../FLAGS.md) for the contract and decided-door ledger.
