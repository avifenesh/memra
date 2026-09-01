# qwen4_exp decode PROFILE-1 — after the launch-boundary attacks (2026-08-29)

Same box, same artifact, same prompts as PROFILE-0 (read that first for the method and the
before table). memra qwen4exp-bringup-20260829 @ a671196324; binary
`target/release/qwen4exp_real_gate` sha256 `a487e9129fec9af3…`. Greedy is the instrument.

## Headline

| | ms/token (warm) | tok/s | vs PROFILE-0 |
|---|---|---|---|
| PROFILE-0 (per-expert dispatch, unfused gates) | 78.5 | 12.74 | — |
| after (a) grouped NVFP4 expert matvec | 40.9 | 24.48 | 1.92× |
| **after (a)+(c) fused hyper gates** | **28.8** | **34.67** | **2.72×** |

Kernel launches per token fell **15,308 → 2,932** (5.22×) and pooled allocations
11,366 → 2,234 (5.09×). Owner target ~90 tok/s = 11.1 ms/token: **2.60× still to find.**

## Per-change interleaved A/B (×5, both arms in ONE run, fresh state + prefill per arm)

Every rep runs arm-off then arm-on back to back from a fresh state, the goldens prompt
prefill, and 4 discarded warmup steps; the other seam stays at its shipped default so each
table isolates one change. Ranges do not overlap in either table.

### (a) grouped NVFP4 selected-experts matvec — `ab-moe-nvfp4.tsv`

```
invocation: ./target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/perf1 \
    --label nvfp4 --goldens ~/realgate/dump --prompts ~/realgate/dump/prompts.tsv \
    --decode-timing 80 --profile 64 --ab-moe 5x40
```

| arm | mean of 5 means (ms) | min | max | tok/s |
|---|---|---|---|---|
| per_expert (before) | 78.10 | 78.00 | 78.16 | 12.80 |
| **sel_grouped (after)** | **40.91** | 40.79 | 41.05 | **24.44** |

**1.91×.** The rep-0 40-token greedy chains are IDENTICAL between arms
(`rep0_arm_chain_first_divergence -1`) despite the summation-order change.

### (c) fused gated-residual read gate — `ab-hc-nvfp4.tsv`

```
invocation: ... ~/realgate/perf2 --label nvfp4 --goldens ~/realgate/dump \
    --prompts ~/realgate/dump/prompts.tsv --compare-logits ~/realgate/out/probe-logits-bf16.bin \
    --decode-timing 80 --profile 64 --ab-moe 5x40 --ab-seam hc
```

| arm | mean of 5 means (ms) | min | max | tok/s |
|---|---|---|---|---|
| hc_unfused (before) | 40.41 | 40.38 | 40.44 | 24.75 |
| **hc_fused (after)** | **28.84** | 28.81 | 28.86 | **34.68** |

**1.40×**, and again identical rep-0 chains (`-1`) even though the inject scalars change
reduction tree.

### (b) indexer structural fast path

Not A/B'd on the box as a timing arm: PROFILE-0 measured the host twin at 0.47 ms/token
(0.4%), so it was never a perf rock. It is banked as a correctness-preserving simplification
(PROFILE-1 shows it at **0.020 ms/token**, a 23× cut on that section) and, more importantly,
as the thing that keeps the host twin flat instead of O(context) as T grows toward 2051.
Its no-op proof is structural (below budget the top-k selects every complete block) and the
tiny gate's budget-2 fixture still exercises the scoring arm at every position past 11.

## Per-section wall profile after (64 warm steps, T_kv 94→158)

Profiled wall 37.48 ms/token, attributed 35.11, unprofiled 28.84 — read shares, not
absolutes (`profile1-nvfp4.tsv`; the intermediate after-(a)-only table is
`profile0a-nvfp4-after-moe.tsv`).

| section | calls/token | ms/token | % attributed | PROFILE-0 ms/token |
|---|---|---|---|---|
| **hyper.read** | 96 | **9.38** | **26.7** | 30.01 |
| **moe.sel_grouped** | 48 | **5.25** | **15.0** | 56.94 (dequant+gemms+gather+reduce) |
| **gdn.proj** | 36 | **5.10** | **14.5** | 5.12 |
| gdn.norm_gate_out | 36 | 2.46 | 7.0 | 2.50 |
| moe.shared | 48 | 1.89 | 5.4 | 1.99 |
| gdn.conv_scan | 36 | 1.87 | 5.3 | 1.90 |
| lm_head | 1 | 1.67 | 4.8 | 1.67 |
| hyper.write | 96 | 1.61 | 4.6 | 1.65 |
| qsa.proj | 12 | 1.49 | 4.2 | 1.48 |
| moe.router | 48 | 1.44 | 4.1 | 1.46 |
| qsa.sdpa | 12 | 1.11 | 3.2 | 1.11 |
| qsa.gate_wo | 12 | 0.76 | 2.2 | 0.76 |
| qsa.idx_proj | 12 | 0.37 | 1.0 | 0.38 |
| ple.key_gate | 1 | 0.22 | 0.6 | 0.23 |
| ple.conv_write | 1 | 0.12 | 0.3 | 0.12 |
| qsa.mask_h2d | 12 | 0.10 | 0.3 | 0.11 |
| logits.dtoh | 1 | 0.10 | 0.3 | 0.09 |
| exit.mixer | 1 | 0.08 | 0.2 | 0.13 |
| entry.embed | 1 | 0.04 | 0.1 | 0.04 |
| ple.host_ngram_gather | 1 | 0.02 | 0.1 | 0.02 |
| **qsa.idx_host** | 12 | **0.02** | 0.1 | 0.47 |
| ple.h2d | 1 | 0.01 | 0.0 | 0.01 |

## Launch census after (nsys, same 8-step warm window)

| | PROFILE-0 /token | PROFILE-1 /token |
|---|---|---|
| kernel launches | 15,308 | **2,932** (cuLaunchKernel 1,714 + cudaLaunchKernel 1,218) |
| cuMemAllocAsync / FreeAsync | 11,366 | **2,234** |
| cuMemsetD8Async | 1,685 | 0 (the `zeros` temporaries are gone) |
| cuMemcpyHtoDAsync | 1,568 | 0 in the top rows (per-expert index uploads gone) |
| cuMemcpyDtoDAsync | 148 | 532 (the inject row extraction) |
| cuMemcpyDtoHAsync | 65 | 65 (unchanged: 48 router + 12 indexer + logits) |

New kernels behave as designed: `qmatvec_nvfp4_modelopt_sel_f32` 144 instances/token
(= 48 layers × 3 projections) at 29.9 µs each, `hc_inject_gates_f32` 96/token at 18.1 µs.
The 1,440 dequant + 1,440 upcast launches per token are simply absent.

Receipts: `profile1-nsys-decode8_*.csv` (before: `profile0-nsys-decode8_*.csv`).

## Correctness after both changes (real-checkpoint gate, re-run)

Everything matches the banked REAL-CHECKPOINT-GATE baseline:

| gate | banked baseline | PROFILE-1 re-run |
|---|---|---|
| logits argmax vs transformers | 10/10 | **10/10** |
| greedy 64-token chains (prompts 0-3) | none, 8, none, 48 | **none, 8, none, 48** |
| layer0 / layer47 max_abs | 7.258e-3 / 1.014e0 | 7.259e-3 / 1.018e0 |
| exit_mixer / logits max_abs | 1.702e1 / 6.437e0 | 1.703e1 / 6.421e0 |
| cross-arm KL row 9 (worst row) | 0.293 | 0.29485 |

The 4th-digit envelope moves are the documented inject reduction-tree class (1 ULP at the
logit scale); the greedy chains and every argmax are untouched. Tiny four-arm gate GREEN on
the rig for both changes plus the new arm-0 kernel oracle (worst rel 7.153e-7).
Receipts: `hidden-gate-nvfp4.tsv`, `greedy-gate-nvfp4.tsv`, `logits-compare-nvfp4.tsv`,
`run-profile1-nvfp4.log`, and `../gpu-eager/tiny-fixture-gate.tsv`.

## Residual: what the next lane should attack

28.84 ms/token, 2.60× from the 11.1 ms target. The remaining profile splits into two
different physics, and only one of them is still a launch problem:

1. **hyper.read, 9.38 ms (26.7%), still launch/occupancy bound.** 15 launches × 96 calls =
   1,440 launches/token = ~49% of all remaining launches; the weights it touches are ~26 MB
   per gate, i.e. ~267 GB/s effective — far from the card's bound. The 8 rank-320 GEMVs per
   gate are the residue: a batched/strided GEMV (or the cuBLASLt strided-batched form) takes
   15 → 8 launches, and **CUDA graphs** would take the whole per-token program to one
   replay. Graphs are the single highest-leverage item left and were out of this lane's scope.
2. **gdn.proj 5.10 + gdn.norm_gate_out 2.46 + gdn.conv_scan 1.87 = 9.43 ms (26.8%) is
   MEMORY bound on f32 trunk weights.** gdn.proj reads ~169 MB of f32 projection weights per
   token (36 layers × [16,480 × 2,560]) in 5.10 ms ≈ 1.3 TB/s — near the card's f32 read
   ceiling. No launch fix helps: the win is **not storing the trunk in f32**. A bf16 (or
   NVFP4/W4A4) trunk halves-or-quarters this slice and also cuts qsa.proj, lm_head, and the
   gate GEMVs. This is the biggest single remaining lever after graphs, and it needs the
   quantized-trunk/W4A4 lane.
3. **moe.sel_grouped 5.25 ms (15.0%)** is now real work: 144 launches reading
   48 × 10 × (2 × 640 × 2560 + 2560 × 640) NVFP4 bytes ≈ 1.18 GB/token in 5.25 ms
   ≈ 225 GB/s — well under the bound, so the kernel itself has headroom (one warp per output
   row, scalar nibble unpack, no dp4a, no vectorized loads). Porting the ornith dp4a/v2-bank
   craft (`qmatvec_nvfp4_sel_gu_into`, `sel_down8` folded combine, `MEMRA_SEL_GU_RPW`
   multirow) to this dialect is a straightforward 2-3× on this slice, worth ~3 ms.
4. **TP2** is untouched and the box has a second idle 96 GB card. The trunk is the memory-
   bound half (item 2), so splitting it across two cards attacks exactly the slice that
   launch fusion cannot.

Ordering suggestion by measured value: CUDA graphs (kills the launch residue wholesale) →
bf16/quantized trunk weights (kills the memory-bound residue) → sel-kernel dp4a port → TP2.
