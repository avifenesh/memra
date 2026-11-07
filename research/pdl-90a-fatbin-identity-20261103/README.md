# PDL entry wait on sm_90a: fatbin identity (2026-11-03)

Verdict: the PDL entry-wait change (`fix(cuda): keep the PDL entry wait on portable sm_90a`,
7e8b3673d4) leaves the sm_120a and sm_100a fatbins byte-identical and changes only sm_90a,
where it adds the wait (`ACQBULK`) and nothing else. Its GPU battery is Hopper-only.

## Question

The change rewrote the `MEMRA_PDL_ENTRY` guard in `cu/kernels.cu`, `cu/qmatvec.cu` and
`cu/flash_attn.cu` from `!MEMRA_PORTABLE_CUDA && __CUDA_ARCH__ >= 900` to
`__CUDA_ARCH__ >= 900`. build.rs defines `MEMRA_PORTABLE_CUDA` only for 89 and 90a, so the
two guards should agree on 120a and 100a. A `.cu` change otherwise owes the full battery, so
the claim needs a byte-level proof before the battery can be scoped.

## Method

`run.sh` extracts the three sources at 7e8b3673d4 and at its parent, then builds each with
the flags build.rs passes (`-gencode arch=compute_X,code=sm_X -O3 --fatbin`; 90a adds
`-DMEMRA_PORTABLE_CUDA=1 -DMEMRA_HOPPER_MMA=1`; the kf8vf8 variant adds
`-DMEMRA_KV_KFMT=1 -DMEMRA_KV_VFMT=2`). Every build runs from the same relative path, so the
only input that differs is the source. sm_90a is the positive control. Compiler: CUDA 13.4
(`nvcc.txt`). CPU only.

## Result

`cmp.txt` (full sha256 and sizes):

| arch | kernels | qmatvec | flash_attn | flash_attn_kf8vf8 |
|---|---|---|---|---|
| sm_120a | identical | identical | identical | identical |
| sm_100a | identical | identical | identical | identical |
| sm_90a | differs | differs | differs | differs |

`acqbulk.txt`: on sm_90a the fix adds 14 (`kernels`), 25 (`qmatvec`), 8 (`flash_attn`) and
8 (`flash_attn_kf8vf8`) `ACQBULK` instructions, the SASS of `cudaGridDependencySynchronize()`;
the parent has none. In `kernels` the only other SASS histogram change is two `NOP`s of
padding.

## Scope and what is still owed

This is a build identity, not a GPU run. It settles that 120a and 100a need no battery for
this change. On Hopper it still owes, unrun: kernel-check, decode-dc and graph-decode for
correctness, and a balanced interleaved H100 decode A/B (parent vs commit, both orders, three
runs each), because the wait changes H100 scheduling and the PERF-H100 board was measured with
it compiled out. If the A/B moves the numbers, `research/tune-data/current-board.json` and
PERF-H100 are refreshed in the same change.

Reproduce: `NVCC=<nvcc> research/pdl-90a-fatbin-identity-20261103/run.sh <repo> <work dir>`.
