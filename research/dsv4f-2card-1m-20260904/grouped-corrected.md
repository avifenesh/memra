# Corrected grouped-prefill experiment

2026-09-05. Base `274b82fd5`; `MEMRA_DSV4_PREFILL_MOE=reference` remains default.

The reverted `f218cca95` experiment returned before computing the shared expert.
Its quality and whole-model timing conclusions remain invalid. This replacement
has two routed arms that both produce original-slot contributions, followed by
one mandatory combine and one complete shared-expert program. A value-level
composition gate checks actual checkpoint weights, nonzero shared contribution,
and total = routed + shared; explicit omitted/uncomputed-shared controls must fail.

It also restores the upstream activation contract. The old grouped path cast
raw f32 activations to half using arbitrary row-amax normalization. The new path
first performs the existing per-128 FP8 activation quantization, then transports
those exact values to half with a power-of-two row scale. Every element must
round-trip exactly; otherwise that row refuses before GEMM. The ModelOpt codes,
E4M3/16 micro-scales and F32 macro scale are read in place without GGUF/repacking.
MMA accumulation remains a separate numerical class, even when inputs are exact.

Current primary source confirms FP8 activation quantization before FP4 linear
and the shared-expert addition in MoE:
https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/model.py
(read 2026-09-05, linear and MoE.forward). SM120 implementation uses Memra's
existing f16 MMA visitor, not SM100-only tcgen05/TMEM machinery; PTX reference:
https://docs.nvidia.com/cuda/parallel-thread-execution/

`VerifyWs.is_prefill` is set by the prefill allocator and never by speculative
verification. Phase is not inferred from tmax. The serving width ceiling remains
64; the separate engine bring-up ceiling is now 512. This does not authorize a
512-row customer deployment.

Build the local transport gate (correctness only):

```sh
/usr/local/cuda-13.1/bin/nvcc -O3 -std=c++17 -arch=sm_120a \
  --expt-relaxed-constexpr -fmad=false -Xcompiler=-ffp-contract=off \
  tools/dsv4-fp8-half-mirror-gate.cu -lcublasLt -o target/dsv4-fp8-half-mirror-gate
flock -n /tmp/memra-5090.lock target/dsv4-fp8-half-mirror-gate
```

Initial transport gate passes all normal widths 128/256/2048/4096, duplicated
row mappings, tail guards and explicit underflow/NaN refusals. No local timing.
Clippy passes. The first locked attempt did not acquire the rig GPU and produced
an empty log; that is not a failed numeric test.

On the development pair, the next actual-model instrument is:

```sh
MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native \
MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_MOE_F16G=2 \
  ./dsv4_grouped_prefill_gate ./model ./dsv4_fixtures_ref.json
```

## Target-card results

Model gate SHA256 `58a900c88a5afa4346563812bdd5697d232c7b825a900c05ec7000dd39c8039d`.
Both RTX PRO 6000 cards pass the FP8/half transport gate. Actual-model composition
passes at layers 3/42 and widths 1/32/64: shared output is identical between arms,
and total equals the independently recomposed routed plus shared contribution.
The largest single-layer total delta is 0.000011444092.

This is not model identity: both width-32 and width-64 grouped primes change
1/16 fixed-seed sampled choices on the same forced continuation. Logit/TV drift
is characterized in `grouped-corrected-model-pro.log`, not accepted by a loose
quality band. Prefill/decode class consistency and sampled cache transparency
remain unresolved. No default or serving promotion.

## Wide transaction repair and exactness

The old commit path scattered all committed rows into a 128-token window. At
512 rows this submits four concurrent writers per destination slot. The new
`ring_commit_plan` selects only the newest window before D2D bounce/scatter.
CPU tests cover all starting offsets through two wraps and widths through 512,
requiring unique destinations and equality with sequential writes.

Wide gate SHA256 `e453f6c049da4c132f310ab7f9be40d4f2a43c895277c71ee477ff9b892105e4`.
A real 1025-token source prompt passes width-32/128/512 equality for logits,
every live trunk cache class, DSpark rings/confidence/rounds and sampled output.
It passes separately within the reference and grouped classes, not between them.
`MEMRA_F16G_SK=32` fixes the grouped kernel form across widths.
Raw receipt: `wide-prefill-pro.log`.

One diagnostic prefill row per width (seconds, not a serving decision):

| class | width 32 | width 128 | width 512 |
|---|---:|---:|---:|
| reference | 8.835038 | 10.037346 | 10.003692 |
| grouped FP8-QAT half mirror | 8.898866 | 7.529029 | 6.818843 |

393 engine library tests, two omission controls and five selected serving tests
pass. The transport synchronization checker reports zero errors. Full delivery,
long-context HTTP, sampled quality and target SLO gates remain open.

## Next mechanisms identified in source, not yet implemented

- Prefill calls `verify_batch_dev` for every teacher-forced chunk. Even when
  logits are not requested, it currently computes the full vocabulary head and
  per-row argmax before the caller discards them. An explicit prefill output
  policy can remove intermediate head work and ultimately compute only the last
  row, while preserving the public verification API's argmax/full-row contract.
- Widths above 32 currently dispatch dense and island-dot work in eight-row
  tiles. The newly gated 128/512 regime permits pricing 32-row tiles and one
  two-dimensional grid without changing each row's accumulation tree; existing
  serving widths must retain their qualified dispatch until tested.
- Indexer Q and compressed K each undergo per-128 FP4 quantization with one
  power-of-two scale per vector. A compact integer-dot path may preserve the
  exact dot while reducing arithmetic/storage. It needs a bounded-exponent
  proof, actual-QAT-data gates and cache-contract work; it is not implemented.
