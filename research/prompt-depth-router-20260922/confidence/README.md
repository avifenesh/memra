# Qwen code K=3 confidence study

**Keep K=3/C=0 as the research control. Positive-C throughput rows are
diagnostic, not equal-distribution speedups.** The sampled positive-C
path discards a low-confidence proposal before target verification
and can bias the target output distribution. The
[exactness proof](EXACTNESS.md) and [scoped verdict](VERDICT.md) govern
interpretation of the generated [development](RESULTS.md) and
[held-out](HELDOUT-RESULTS.md) reports.

The pinned Qwen3.8-27B NVFP4+Q5_K embedded-MTP study measured a fixed
K=3/C=0 control, positive C cutoffs and K=2/C=0 on one non-production
RTX 5090 with sampled 0.7/top-k 20/top-p 0.95 decoding. Development
selected C=0.15 at +6.81% pooled code tok/s; the separately qualified
held-out code set measured +1.03% pooled with a 4K loss of 4.98% and
12/24 paired wins. Those are measured rates of the executed path,
not proof of a faster exact sampler or improved code quality.

The [protocol](PROTOCOL.md) preserves the first 7/8 all-format
qualification failure and the [versioned code-only split](CODE-ONLY-V2-PROTOCOL.md).
The [learner design](LEARNER-DESIGN.md) starts with sampled exactness,
then a cost-inclusive oracle and conditional accepted-prefix
calibration. The [archive boundary](ARCHIVE-BOUNDARY.md) describes the
hash-pinned raw receipts and private publication review.
Memra #673 owns the MTP sampled-C correction and deployment-override
audit. No live C learner or served default was promoted.
