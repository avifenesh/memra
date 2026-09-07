# DSV4 intermediate expert TP candidate

This candidate is a source-connected, gate-only vertical slice for the DSV4 all-layer
intermediate-TP topology. It is not a serving default, a rate result, or a qualification receipt.

## Ownership and numeric class

Both ranks retain every trunk layer, the full attention state, the full router state, and the
global route domain. A selected expert id is therefore unchanged on both ranks. For every
ModelOpt NVFP4 expert, each rank owns:

- half of the gate output rows;
- half of the up output rows;
- half of the down input columns, packed to the local physical row stride.

The DSV4 geometry is the pinned `hidden=4096`, `expert_inter=2048`, `experts=256`,
`topk=6`, 43-trunk-layer shape. This is the DSV4 six-selected-expert path, not the GLM
eight-expert path. Global model config and the shared-expert FFN dimensions are unchanged.

Each rank produces a hidden-wide down partial. The eventual rank-0-then-rank-1 FP32 addition is
the named class `dsv4_modelopt_tp2_gu_rows_down_cols_f32_rank_reduce`; it is not claimed to be
bit-identical to the unsplit full-K dot.

## Source path

- `dsv4_topology.rs` adds a distinct `TpEpIntermediate` admission mode and a process-local
  gate setter. It cannot be selected by an environment variable or serving request.
- `dsv4_modelopt_split.rs` validates the full ModelOpt bank, packs GU row halves and strided
  down-K halves with device 2D copies, and builds the six-plane local pointer table.
- `dsv4_gpu.rs` loads the complete manifest first, then replaces each rank's temporary full
  bank with the packed half bank. It retains the full router ids/weights and scale-2 planes,
  creates local-only rank state, and reuses the existing one-shot rank-order reduction.
- `dsv4_grouped.rs` adds a half-intermediate workspace while keeping the existing grouped
  visitor, SwiGLU, FP8 activation quantization, down visitor, scatter, and contribution ABI.

The production whole-expert-ID TP/EP mode remains a separate topology. No PP shell or
expert-ID partition is reused as an intermediate-TP claim. No custom CUDA megakernel or
external dependency is introduced.

## Evidence status

Source-only review and rustfmt checks pass. No Cargo build, GPU run, model load, 1M prompt, or
performance claim was made for this candidate. The next gate must prove, on both ranks, full
route coverage, packed code/scale identity against the complete manifest, per-expert GU/down
geometry, non-owned-slot zeroing, rank-order reduction identity for the named class, and AR
engagement before any rate measurement.
