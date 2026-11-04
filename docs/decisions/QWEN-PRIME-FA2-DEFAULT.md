# Qwen prime attention takes the FA2 class on its qualified profile (2026-11-04)

**Chosen:** the Qwen 24 Q / 4 KV / d256 causal prefill at t=16..1039 runs the FA2 attention
class (`fa_prefill_qw_fa2`, `fa_prefill_qw_fa2_prime_table`) by default on the 170-SM sm_120a
target (the RTX 5090) when the prime runs in 1024-row chunks (`MEMRA_PRIME_CHUNK=1024`). The
selection is compile-time arch, then the cached SM count, then the chunk
(`Engine::prime_attn_fa2_enabled`). `MEMRA_PRIME_ATTN_FA2` is deleted, together with the two
instruments that toggled it in one process (`qwen-fa2-margin-gate`, `qwen-fa2-layer-probe`).

**Rejected:** the existing quantized-KV prefill kernel as the default on that profile. It stays
the program for every other build, SM count, shape and chunk setting, the naked 4096-token chunk
cap included.

**Why the chunk stays a condition:** the kernel covers t=16..1039. At a wider chunk every full
chunk would stay on the existing class while only the folded tail took FA2, so one prime would
mix numerical classes and a restored suffix would stop matching its cold twin. That is the defect
the first integration hit at t<128. The receipt's every arm ran at chunk 1024, so the default is
held to that geometry rather than extended past its evidence.

**Measurement that settled it:** `research/qwen-prefill-attn-20260909/FA2.md` and
`fa2-receipt.json`, one 170-SM RTX 5090, one binary for both arms. Cold TTFT, three fresh boots
per arm: 8k 2.465 -> 2.450 s (-0.60%), 32k 11.000 -> 10.622 s (-3.43%), 131k 66.489 -> 60.376 s
(-9.19%). Same-binary ON run-spec K=1..8 matches plain; the frozen-prompt margin gate reads 0 flips
over 24 teacher-forced positions (max logit delta 1.057); four greedy cold/restore turns match
their cold twins; the boundary state hashes match at 8160; paired NLL delta -0.0011 with a
diagnostic 95% interval that includes zero; decode with a fixed OFF-prime state is bit-identical
over 192 steps. The receipt proposed this flip for the owner's batched release. Owner,
2026-11-04: positive-receipt winners become defaults keyed to the receipt's card, and their
switches go.

**Scope:** one card class (the 170-SM RTX 5090) and one geometry. The naked prime chunk is
unchanged, so a naked command reaches FA2 only on a deployment that sets chunk 1024. Whether
chunk 1024 with FA2 should replace the 4096 cap as the naked chunk on that profile is a separate
question: no receipt compares those two arms. On that profile `MEMRA_PRIME_CHUNK=1024` against
any other value is now a class change, which the `MEMRA_PRIME_CHUNK` row states.

**Not decided here:** `MEMRA_PRIME_KV_T3`, the three-plane staging door on the same geometry,
stays default OFF. Its receipt (`research/qwen-prefill-20260909/RESULTS.md`) is exact and faster,
but holds the default on an unpaired accumulated-state failure in one T3 boot that has no assigned
cause. That receipt measured the legacy class at chunk 1024 on this card, which is now FA2's
geometry, so T3 no longer engages there; its row carries the owner's call (delete it as
superseded, or measure it at the naked chunk on this card).
