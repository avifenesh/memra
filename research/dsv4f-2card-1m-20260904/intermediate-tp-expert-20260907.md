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

## Gate binding (2026-09-08)

The stacked gate lane rebases source candidate `61bda2afe4f9375e6cd23d9ea6052d1230cc01fc`
on main `99c2c4e4d` (including #337). The candidate's stale finalization byte accounting referenced three
undefined variables; accounting now happens only in the per-layer pack. Attention TP2's
separate packed projections and join remain intact. The candidate raw-pointer scratch
helper dropped cudarc write guards before enqueue; the local scratch now enters the existing
enqueue body through typed reborrows. The hidden-width helper now reports the replicated
4096 axis. Admission also pins one shared expert and refuses disabling the matrix executor
after loading half banks.

Both existing gate binaries accept `MEMRA_DSV4_INTERMEDIATE_TP_GATE=1` before loading.
Only these executables read the selector; serving cannot select the candidate through an
environment variable or request. OFF selects the existing expert-ID EP program. Initial
qualification uses attention TP2, radix sampling and split-K OFF. PR #337 merged during implementation. The second rebase retains its split-K dispatch and
uses half-width GU scratch for intermediate banks. Initial intermediate gate selectors
refuse split-K CLI arms, and the standard arm explicitly sets split-K OFF.

Per rank/layer: 256 experts, 1024x4096 GU and 4096x1024 down, 1.5 GiB code bank,
192 MiB scale bank, and 1536 pointer entries. Shared experts remain full width and are
added once after the routed rank join on each replicated rank.

Load checks validate all expert projection shapes and pointer entries and compare packed
code/scale bytes for experts 0, 1, 127, 128, 254 and 255 against the complete manifest on
every layer/rank. Correctness captures every layer's real route prefix, original-slot
mapping, both GPU down partials and both GPU AR outputs for four teacher-forced positions,
repeated twice. CPU f32 rank0+rank1 must match every joined bit. A separate poisoned-buffer
restricted-domain control uses the real half banks to verify non-owned positive zero and
owned-slot identity. Production intermediate TP has no non-owned selected slots.

The existing six sticky-refusal cells cover 40043/40044 at positions 1, 3 and 127 with
attention TP2 engaged, unchanged cache/position and refused retries. Sampled timing excludes
all correctness capture and includes sampling plus forward. Five rows per fresh process;
ABBA order is intermediate/current/current/intermediate on the same binary, with per-arm
repeatability, pooled tokens divided by pooled wall time, and no cross-class token-identity
requirement.

Status: implementation pending remote build and full-model correctness. No GPU or Cargo
work ran on the local rig. Pushes use `MEMRA_SKIP_PERF_CI=1` with normal hooks; hosted CI
and the exclusively locked development pair provide validation. Raw receipts stay outside
this public repository under private namespaces `intermediate-tp-<sha7>-r<N>`.
