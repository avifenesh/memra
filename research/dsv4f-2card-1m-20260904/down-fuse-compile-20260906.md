# DSV4 down-fusion compile receipt

Date: 2026-09-06

This is a compile-only receipt. No GPU or serving claim is made.

## Candidate

`MEMRA_F16G_DOWN_FUSE=1` is a default-OFF, matrix plain-only door. It keeps
the existing ModelOpt NVFP4 direct tail visitor and its f32 MMA accumulation,
but uses `dsv4_act_quant_fp8_half_kernel` to write the normalized half mirror
directly from the weighted SwiGLU row. The down visitor then folds `macro2`
into its final multiply and writes the original slot directly. The old path
remains the rollback: FP8 codes/scales, checked mirror gather, down visitor,
`scale_rows`, and original-slot scatter.

The candidate does not change expert weights, FP8 rounding, row scales, MMA
order, route order, or output accumulation class. It removes launch/data
movement boundaries only. It is not qualified until a same-process identity
gate compares the full routed output and intermediate half/scale planes, then
an interleaved target rate cell prices it against the rollback.

## Build

Command:

```text
cargo check -p memra-engine
```

Result:

```text
Finished `dev` profile [unoptimized + debuginfo] target(s) in 3m 37s
```

The build compiled both changed CUDA translation units with CUDA 13.1 / sm120a
and `-fmad=false` for `dsv4_gpu.cu`. The door is not enabled by this receipt.

## Next gate

On the owned pair, use one loaded process and alternate the gate-only
`Dsv4Gpu::set_grouped_down_fuse_for_gate` setter (or the env arm in separate
boots), with stream drains between arms. At each
of 256 and 8192 prompt rows compare: (1) weighted SwiGLU `H`, (2) FP8 codes /
scales versus the direct half/row-scale plane after decoding, (3) routed
contribution at original slots, and (4) final sampled/greedy output. Any
nonzero identity mismatch removes the candidate. Only a positive identity
cell proceeds to interleaved plain-rate measurement; the 1M proof is not
rerun.
