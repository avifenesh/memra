# Row launches past the CUDA grid-y cap split into row blocks (2026-11-11)

**Status:** adopted, memra#1200. Receipts:
[`research/grid-y-rows-20261111/`](../../research/grid-y-rows-20261111/).

## Question

CUDA caps grid dimensions y and z at 65,535 blocks. The batched embed gathers
(`embed_gather_device_t`, `_td`, `_tv` and qwen4exp's `embed_gather_rows_into`) launched with
the row count in grid y. So did the per-token dp4a matvecs (the GDN beta and alpha projections
land there), `qmatvec_f32`, the prefill GDN conv, the prefill KV row append and E4B's strided
row copy. A call over more rows failed at launch with a bare `CUDA_ERROR_INVALID_VALUE`.
`forward_last` on a 131,072-token prompt failed before layer 0's first norm (memra#1177), and
run-gen on a GGUF stops there past 65,535 tokens. The served prime is chunked under the cap, so
no default serving path reached it. E4B's monolithic prime and operator settings of the concat
prime could.

Three fixes were open: clamp every caller, change the kernels, or change the launchers.

## The rule chosen

A launcher whose kernel takes its row from grid y issues the rows in blocks of at most 65,535
(`hybrid_forward::grid_yz_row_blocks`). It launches once per block over sub-views at the block's
first row. The kernels are unchanged.

- **Row-only kernels** read and write only their own row, so a split cannot change a row: the
  gathers, the dp4a family (through one launcher, `dp4a_rows_launch`), `qmatvec_f32`, the KV row
  append (with the block's own `t0`), `copy_rows_strided`, and the MoE pair scatter (pair ids are
  absolute; the CSR offsets and the output are sliced).
- **The prefill GDN conv** reads the three rows before each row. Block 0 is the zero-history
  kernel as before. A later block runs the carried-ring twin `ssm_conv1d_gdn_state_f32` over a
  ring of the rows just before it, written by `ssm_conv_ring_update_f32`. The window reads the
  same values in the same 8-tap order, through the same SiLU and scatter.
- **At or below the cap** every launcher issues the one launch it issued before, with the same
  grid and the same arguments. Rows that work today are byte-identical by construction. Measured:
  forward_last logits at 3,540 tokens are byte-equal, main against the fix, on the RTX 5090.

These launches have no sub-view split, and they refuse by name instead of failing at launch:

- **The prime-graph KV append.** It reads its cache base and counter from the graph table, and
  it serves only chunked primes, which stay under the cap.
- **`qmatvec_mmvq`.** It serves large m only as a fallback after every GEMM tier declined.
- **`sdpa_naive_island_f32` and `sdpa_naive_w_f32`.** The first is the attention of a Gemma prime
  with a vision overlay; the second is the cacheless windowed prefill of a sliding-window trunk.
  Each keeps one f32 score per KV row in dynamic shared memory without the opt-in attribute, so its
  own limit (48 KiB less its static shared memory, about 12,288 rows) binds long before grid y
  does. Past that limit each now names itself and the row count.
- **`forward` and `forward_last` over a KDA layer past the cap.** The KDA prefill core
  (`kda_conv_silu`, `kda_gate`, `kda_gate_beta`, `kda_proj_fused6_raw`) keeps the row count in grid
  y. The model refuses before the first launch and names those kernels.

`forward_last` errors now name the row count and the last kernel the engine resolved.

## Rejected

- **Clamping callers.** This keeps every caller that can legitimately pass more rows broken:
  forward_last as the f32 reference for long prompts (memra#1178 needs one past 65,535), run-gen's
  argmax gate, and E4B's whole-prompt prime. It also leaves the next caller to rediscover the cap.
- **A grid-stride loop over y inside each kernel.** This is just as correct, but it changes every
  hot decode kernel it touches. That means new SASS, before/after perf numbers per kernel, and new
  KERNELS.md rows, all for row counts no decode step reaches. The launcher split leaves every
  kernel binary as it is.

## Not covered

These launches still carry a row count in grid y. None of them is reached past the cap on a
default path:

- **The carried-ring conv's prime form (`ssm_conv1d_gdn_state_pad`).** Every prime that reaches it
  is chunked under `PRIME_CHUNK_LAUNCH_CAP`.
- **The MoE router's plain forms.** They are reached only through the `MEMRA_ROUTER_BATCH=0` and
  `MEMRA_ROUTER_V2=0` rollback seams. The default batch form tiles rows by 8.
- **`qmatvec_view_inner`.** It fails only if one expert receives more than 65,535 rows in one call.
- **The naive attention kernels.** Portable (non-mma) builds fall back to them from `fa_prefill*`,
  and their own shared-memory limits bind first.
