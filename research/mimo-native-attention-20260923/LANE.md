# MiMo typed attention arithmetic and CPU reference

Owner: `codex-mimo-serving-max-20260923`, under `darklanes#1083`.
This branch depends on `avifenesh/memra#682`, `#686`, and `#688` in
GitHub stack #687. It does not register a MiMo model pack.

Pinned source: `XiaomiMiMo/MiMo-V2.6-Flash-RL` revision
`3b38d063180c3e4aed9691fdc735f3d10b266ee4`, config SHA-256
`61bea4a0f7a0dd8969f8cae528761e26b697dd12ff63e98804c3f0945492e621`.
Its `modeling_mimo_v2.py` multiplies projected V by 0.707 before
updating the KV cache. Sliding layers concatenate each learned
`attention_sink_bias` logit to real QK logits before softmax, then
drop the sink probability before the V weighted sum. Thus the sink
affects the denominator and has no value vector.

This slice adds family-only attention math metadata to the typed plan
and makes the CPU reference apply that V scale before caching and
include the sink in a stable softmax denominator. A one-token fixture
has real logit 0 and sink logit ln(3), proving real probability 1/4;
it also checks the scaled cached V. The pinned config test checks
the nine global layers have no sink and a sliding layer requires one.
The new optional plan field is omitted from Debug for all other
families to preserve their existing plan identity representation.

This is not a runnable MiMo checkpoint. Its serialized `qkv_proj`
still needs an exact split/tensor contract, its 36,096 routed expert
projections need ModelOpt NVFP4 binding, and the target GPU executor,
tokenizer/template, checkpoint parity, quality, host-cache and fleet
serving gates remain open. No customer route or product fact changes.

No tests or gates may run on the local rig. Hosted exact-head CI and
off-rig pinned-config/reference tests are required.
