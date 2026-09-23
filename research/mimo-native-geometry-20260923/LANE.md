# MiMo native full/SWA geometry

Owner: `codex-mimo-serving-max-20260923`, under `darklanes#1083`.
This branch depends on the fail-closed config and single-group router
drafts in `avifenesh/memra#682` and `#686`. No MiMo model pack is registered.

Pinned source: `XiaomiMiMo/MiMo-V2.6-Flash-RL` revision
`3b38d063180c3e4aed9691fdc735f3d10b266ee4`, pinned config SHA-256
`61bea4a0f7a0dd8969f8cae528761e26b697dd12ff63e98804c3f0945492e621`.
The 48 trunk layers have global attention at 0, 5, 11, 17, 23, 29, 35,
41 and 47; all others are window-128. Both use 64 query heads and
192-wide Q/K with 64 rotary dimensions and 128-wide V. Global layers
use 4 KV heads and RoPE theta 10,000,000; sliding layers use 8 KV heads
and theta 10,000.

This slice builds two declarative geometry classes from the pinned HF
config and tests their per-layer resolution. MiMo has no QK norm tensor;
its family-specific presence is marked absent. Typed model-plan
compilation explicitly refuses the still-unexpressed learned sink
denominator and 0.707 value scale before KV-cache write. This prevents
the new geometry table from making an incorrect generic attention
program appear executable. Fused QKV binding, native executor, parity,
GPU serving and fleet gates remain open. No customer route or product
fact changes.

No tests or gates may run on the local rig. Exact-head hosted CI and
off-rig pinned-config tests are required before this is reviewable.
