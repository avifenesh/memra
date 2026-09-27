# Unqualified fresh PrimeGraph program (#585)

This preserves the failed three-surface campaign at 3d1cd9ea on one RTX PRO 6000
Blackwell. True input `[1,2,3,4]`, bucket 16, FAST0 and exec-time CUDA dependency
preloading are fixed in the evidence. Seed max absolute error is 32.650116; graph
logits differ from the independent reference by 1.379926. The ordinary tokenwise
and teacher-forced quantized-cache references agree within 1.90735e-6. Matching
argmax does not satisfy the unchanged 0.005 absolute/relative policy.

No CarriedPrime receipt or full-capture seal was issued. The first prime input
failed; the 8/16 prime inputs, continuations and 24 subsequent caller cases were not
executed. The new selected Eager/Graph qualification plan does not relabel this
campaign or count positive prime coverage as passed.

`raw/` contains the original logs and float arrays. `offline-comparison.json` is
derived analysis, separate from the original failed result. `evidence.json` binds
each raw payload, result and offline comparison to the actual source, executable and artifact.

Source review finds direct in-tree PrimeGraph callers in diagnostic binaries.
Its public API is still exposed, and its pad-invariance claim does not establish
this configuration's equivalence. The shared CarriedPrime permission also gates
production batched prime, so neither that route's success nor its failure can be
inferred solely from this diagnostic graph. Permission remains absent.

The accepted #578 change concerns carried MTP sub-floor suffix scheduling, with
an already-restored cache. #427 concerns an aligned long q38 restore whose final
real suffix has 16 rows. This case is a fresh Qwen3-0.6B graph with four real tokens;
sixteen is its capture bucket. No shared root cause or fix is established by that
number alone. #585 tracks this separate unqualified program.
