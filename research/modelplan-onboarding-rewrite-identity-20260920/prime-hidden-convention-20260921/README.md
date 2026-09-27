# Prime hidden-output convention follow-up

A separate Astra source assessment identified a hidden-output contract problem in
the first Eager-only fallback (`9733594e`, published as `45c0b1742`). Non-PLE
Gemma T1 decode returns pre-norm hidden rows even with `SPEC_HPOST` enabled. Its
ordinary text prime uses the generic prime epilogue, which returns post-norm
rows under that setting; the common pooling consumer then skips another norm.
Copying the T1 seed verbatim therefore gives that consumer the wrong representation.
This is distinct from the preserved Qwen3 T16 numerical failure and does not
establish a logit or cache error in that Gemma path.

The correction is confined to exported rows in `prime_cache_eager`. It uses the
canonical Gemma program and absence of the typed PLE operation to select the
existing output RMS normalization, with the same loaded norm weights and epsilon.
The result is a separate buffer used for the hidden stack and final seed. T1
logits, cache state, subsequent token inputs, ordinary/qualified prime, PLE and
generic-family conventions remain unchanged. No flag, new kernel, activation or
weight format is introduced.

Eight CPU controls pass. The new matrix covers generic/Gemma, PLE/non-PLE and
HPOST off/on with non-unit rows and nonuniform norm weights; it checks every row,
the final seed, normalization count, unchanged logits/cache/next-input dataflow,
and taint after a normalization error. Replaying frozen45c fails exactly the two
new controls. Frozen655 still reproduces the original strict fallback failures.
The CUDA arithmetic in the CPU harness is explicitly stubbed; these tests prove
dispatch, output convention and lifecycle, not numerical CUDA qualification.
Linux-target engine/gate/server clippy passes using DOCS_RS for compile-only checks.

The original T16/15/17/B3/carry/four-continuation gate remains intact. No native
run has occurred for this follow-up. Actual admitted Gemma/HPOST variants still
need exact-source native verification of exported normalized T1 rows and their
consumers before claiming that configuration qualified; the existing generic
RMS kernel battery and ordinary Gemma tests are supporting, not substitute evidence.
All655/0ea0 raw failures and all archived ELFs remain unchanged. Remote dispatch
is held by the coordinator while local review continues.
