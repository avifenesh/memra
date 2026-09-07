# DSV4 device sampler

The first 10-cycle ABBA on the two-card RTX PRO 6000 development pair passed:
CPU radix **35.057077 tok/s**, device **40.277751 tok/s**, **+14.891926%**.
All 40 model rows (20 per arm) have identical generated tokens, final logits,
cache and hidden digests. Each row has 256 source-prime tokens and 256 sampled
forward calls. The timer includes sampling, forward and final drain:
`timing_scope=sample_plus_forward_envelope`.

These numbers are verbatim from the `ABBA` line in the private companion receipt
`gpu-sampler-42d9756-r1/gate.log`. Source:
`42d9756820f32276632f6107cf46a5e4eda1f539`. Controller exit: 0.
The component gate passed 512 rows, 256 per physical GPU, with identical tokens,
finite outputs and output canaries. The deterministic full-vocabulary tape has
129280 logits per row, ties, signed zeros, subnormals, dominant logits, penalties,
tiny top-p, and temperatures down to minimum positive normal f32.

Code delivery source `2a0255cb460324f3a9ac0ec975ca513de138437e` adds scratch guards,
stream/vocabulary validation, per-row engagement counts and accurate sampler
names in protocol/summary lines. Its CUDA sampler file is unchanged from the
first receipt (Git blob `0a4247df5b7e48fe4bb72dc41a517db3a6e910d8`).
The separate delivery gate receipt is `gpu-sampler-2a0255c-r1` in the companion
private ops tree. The draft PR records its current validation status; the
first receipt's PASS does not stand in for that separate run.

Default host; device opt-in, decide-by 2026-09-21. No promotion or merge.
CPU radix remains the oracle and rollback arm.

Numeric class: `device-f64-exp-tree-cdf-v1`. Candidate IDs and penalty arithmetic
retain CPU order. CUDA double exp and parallel unnormalized prefix sums can
change a nucleus cutoff or inverse-CDF decision at a rounding boundary relative
to the CPU libm/sequential normalized program. The frozen tape is token-identical;
this is not universal probability-bit identity. Nonfinite penalized inputs refuse.
Prefill/restore retain their existing host-row contract; subsequent plain forward
decode returns only the sampled u32 after the device chain.

Perf CI is skipped per the owner. All GPU gates run on the development pair.
The parent 120 tok/s plain objective remains open. No serving qualification.
