# Current-main transfer census correction

At source `1cd7135d6f10021f8a892bedb9b41bce175e169d`, the native transfer
conformance child exited 0 and emitted thirteen PASS assertions. The runner expected
the older exact eleven-marker set, so it correctly refused the unmatched transcript
and stopped before roundtrip. That campaign remains FAILED; roundtrip remains UNRUN.

Current main added two real CUDA recovery assertions:

- `PASS rule cancelled-restore-recovers-source native CUDA`
- `PASS rule cancel-refused-after-source-consumed native CUDA`

The correction requires both, alongside all eleven earlier assertions. Exact Counter
equality still rejects missing, duplicate or extra markers. The six roundtrip sizes,
hash-equality checks and lifecycle predicates are unchanged. No native producer,
numerical tolerance, Qwen T16 oracle, selected 27-case scope or production code changes.

CPU controls cover deletion and duplication of every marker and reject extras. A
source-census regression pins the thirteen assertions emitted by the current producer;
the previous eleven-marker runner rejects the new complete transcript. All 22 caller
runner controls pass after the repair.

The earlier source-bound results remain useful: `1cd7135d` passed 27/27 selected
callers, the original T16 and its full matrix with zero error, baseline 13/13, and the
v3 generic battery. They are not qualification of this changed controller/source.
Freeze and review this revision before a fresh transfer attempt; never relabel the
failed attempt or infer that its unrun roundtrip passed.
