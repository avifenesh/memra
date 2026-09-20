# Native single-card admission evidence

Pinned Qwen3-0.6B source staging passed, with every checkpoint/config/tokenizer hash matching
the prior independently staged host. Build003 passed at
`b2b0ff40714008884185e62cb6696a15b5aa0e34` using real CUDA13.1, jobs8, GPU visibility empty.

`server-admission-001` ran on one exclusively locked RTX PRO6000 Blackwell Server Edition.
Inspection passed; native capture failed on prompt[1,2,3,4]: fresh-F32-KV forward_last vs
quantized-cache tokenwise had matching argmax3 but max_abs3.1681318, beyond the unchanged
0.005+0.005|reference| policy. No receipt was installed and no qualification passed.
The wrapper finished normally with exit1, no timeout/signal, and no lingering compute.

`server-control-001` is the standing run-gen control on the same source/prompt. It passed
exit0: quantized-cache verify-prefill vs tokenwise decode maxdiff1.907e-6, argmax3 MATCH.
This is the numeric-class distinction documented in run_gen.rs; the failed cross-class
comparison is preserved, not waived or relabeled. No shared attention math was changed.

The correction gives fresh-KV monolithic forward its own manifest/receipt and refuses it
under eager-only qualification. The native gate now uses independent verify-prefill and
tokenwise paths of the cached-KV class, retains full output planes, and tests fresh-KV refusal.
A new build and fresh GPU receipt are required before claiming that correction qualified.
Storage is XFS on Ceph RBD; this is not an NVMe/spill performance qualification.
