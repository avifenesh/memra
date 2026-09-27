# MiMo native routing semantics

Owner: `codex-mimo-serving-max-20260923`, under `darklanes#1083`.
This branch depends on the fail-closed MiMo config intake in
`avifenesh/memra#682` at `6d77a2a5b0daae3d97041700e654bb3a4b3981a2`.
It does not register a MiMo pack or admit the checkpoint for native load.

Pinned source: `XiaomiMiMo/MiMo-V2.6-Flash-RL` revision
`3b38d063180c3e4aed9691fdc735f3d10b266ee4`, local source-config
SHA-256 `61bea4a0f7a0dd8969f8cae528761e26b697dd12ff63e98804c3f0945492e621`.
The vendor `modeling_mimo_v2.py` applies sigmoid to FP32 router logits,
adds `e_score_correction_bias` for top-k selection only, gathers the
unbiased scores as weights, normalizes the selected weights, and multiplies
by `routed_scaling_factor` (1.0 when its config value is null). The pinned
config declares one routing group and one selected group, so its group
selection equals ordinary top-k selection. Layer 0 is dense and layers
1–47 are routed MoE.

This slice retains both group fields, requires the single-group form
before compiling a typed plan, selects the existing sigmoid/selection-bias
router, and uses the declared dense/MoE layer frequency. It does not
claim an executable MiMo attention path: full/SWA geometry, learned
sink denominator, 0.707 V scaling before cache, fused QKV tensor binding,
checkpoint parity, GPU serving, and fleet gates remain separate work.
No customer route or published fact is changed.

No tests or gates may run on the local rig. The exact pinned-config and
router fixtures need hosted CI and an owned non-production-host test.
