# HY3 native tuning receipt subset

These files bind the admitted four-card RTX PRO 6000 path to Memra source
`4e0b37aedc2f1920794f819a6593d5aa708959a4` and the artifact index
`0f22f6fc51ac7e39b7510a77c77098c4fd7c722e9e6cfdb9782247c37f1b6afd`.

The complete 99 MiB receipt tree, including raw full-vocabulary TSVs and the Nsight report, is
sealed in the operator's off-repository receipt archive. The checked-in manifests bind that
archive without publishing host paths.

The selected rank files and machine receipt were published under
`Tiyuvta/Hy3-NVFP4` at Hugging Face commit
`4e8bbadbdb97b5402cb5a3f997d941946b97c5b5`; pinned readback reproduced every checked-in
SHA-256.

This checked-in subset contains:

- the HY3-served completion-corpus seal and final 81,920-row rank artifact;
- the four V1/V2 grouped-prefill bank-oracle arms;
- the 91-cell final kernel gate and masked-MTP K=1..8 gate;
- binary and raw-receipt SHA-256 manifests;
- exact, all-Q8, gate/up-Q8, and down-Q8 logit verdicts;
- a public-safe seal for sampled c=1/c=4, cache, tool, and reasoning receipts, plus the
  symmetric teacher-forcing logs;
- the final Nsight kernel summary and report hash.

The admitted numeric class is gate/up-only Q8 activation inside the routed-expert program.
All-Q8 and down-only Q8 failed the frozen logit thresholds and remain rejected. The slot-major V2
layout is not a corruption source; the live defect was the grouped-prefill sktail call site
defaulting `in_f` to zero before upstream commit `1b18a61e8`.
