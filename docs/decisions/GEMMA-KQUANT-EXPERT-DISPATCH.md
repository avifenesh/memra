# A gemma4_moe expert bank runs only kernels that decode its qtype (2026-11-11)

**Chosen:** every gemma4_moe device arm picks its expert kernel per projection qtype, the way the
qwen-class arms already do (memra#1234). The device arms admit the `expert_dp4a_supported` set:
Q4_0, IQ4_XS, IQ3_S, Q3_K, Q4_K and Q6_K. Three of their programs decode only the first three
(`q8_expert_dec_supported`) and read any other qtype with one of those layouts (the MMA GEMM and
the CSR twin as IQ3_S, the decode-once matvec as IQ4_XS). Each now takes a k-quant bank to the
kernel that carries its body:

1. **Prefill pairs arm, int8 MMA** (t >= 16, the default when `n_embd % 256 == 0`):
   `gemma_pairs_arm` selects it only when gate, up and down are all in the decode set. A model
   with any k-quant expert bank prefills on the dp4a arm (or on the grouped f16 lane where
   `MEMRA_MOE_F16G` opens it for gemma; that lane has its own k-quant dequant kernels and was
   not run here).
2. **Prefill pairs arm, dp4a** (`gemma_pairs_matvec`): per projection, the decode-once
   `moe_pairs_matvec_q8_dec` for the decode set (the program this arm always ran for them), else
   `moe_pairs_matvec_q8_em`, whose `expert_dot_g` has the k-quant bodies.
3. **Speculative verify, the CSR twin of the gate/up kernel** (t = 2..10,
   `moe_gate_up_gelu8_dev_q8_csr`): only when gate and up are in the decode set, else the rows
   twin (`moe_gate_up_gelu8_dev_q8_rows`), which is the t = 1 kernel per row and already ran at
   t = 11..15.

The Q4_0, IQ4_XS and IQ3_S programs are unchanged: the same kernels, the same bytes. The decode
kernels (t = 1), the rows twins and the down kernels already carried the k-quant bodies.

**Rejected:**

- *Refuse k-quant expert banks at load.* The kernels that decode them exist and the qwen-class
  arms run them; only the dispatch was missing.
- *Send a k-quant model to the host-staged loop.* That is another numeric program and far slower
  (memra#1225 measured 17.9 against 113.7 tok/s decode on gemma-4-26B-A4B).
- *Add k-quant tile loaders to the int8-MMA GEMM and the cached CSR decode.* New kernels with no
  artifact to qualify them on. The `_em` matvec is the qwen-class k-quant program; the gemma
  rows twin shares its k-quant dot bodies with the qwen kernels and already ran at t = 11..15.

**Measurement that settled it** (RTX 5090 Laptop GPU; receipts in
`research/gemma-kquant-dispatch-20261111/`). Fixture only: no k-quant gemma4_moe GGUF is on any
unit this repo tests on, so no real-checkpoint window or served run exists for this change.

The test (`tests/gemma4_moe_kquant_pairs_gpu.rs`) builds memra#1225's two-layer placement
fixture (128 experts, top 8) with three k-quant layouts, each next to a Q4_0 control: Q3_K
gate/up with Q4_0 down at ff 704, Q4_K gate/up with Q4_0 down at ff 704, and Q6_K gate/up/down
at ff 512. The first is what a stock llama.cpp Q3_K quantization gives gemma-4-26B-A4B: its
down k (704) is not a multiple of the 256-value superblock, and llama.cpp's fallback for Q3_K
there is Q4_0. (Its fallbacks for Q4_K and Q6_K are Q5_0 and Q8_0, which the device arms do
not admit, so those stock GGUFs of the 26B keep the host-staged loop.) The test compares the
engine's forward logits with `memra_reference::execute` over the dequantized weights; the score
of a width is the median over rows of the row's largest logit error over the largest reference
logit. A near-tie in the router's top-k that activation quantization flips moves a few rows in
any variant, the control included (largest row up to 0.12 at T = 130), so the median is the
score and the maximum is printed.

| variant | build, pairs arm | T=1 | T=8 | T=12 | T=16 | T=40 | T=130 | prime 16 vs 16 decode steps |
|---|---|---|---|---|---|---|---|---|
| Q4_0 control, ff 704 | base and fix, MMA | 0.0017 | 0.0021 | 0.0021 | 0.0016 | 0.0022 | 0.0021 | 0.0038 |
| Q3_K / Q4_0, ff 704 | base, MMA | 0.0024 | 0.155 | 0.0023 | 0.116 | 0.145 | 0.143 | 0.117 |
| Q3_K / Q4_0, ff 704 | base, dp4a | 0.0024 | 0.155 | 0.0023 | 0.133 | 0.132 | 0.147 | 0.132 |
| Q3_K / Q4_0, ff 704 | fix | 0.0024 | 0.0014 | 0.0023 | 0.0017 | 0.0022 | 0.0021 | 0.0027 |
| Q4_K / Q4_0, ff 704 | base, MMA | 0.0031 | 0.138 | 0.0023 | 0.101 | 0.122 | 0.132 | 0.101 |
| Q4_K / Q4_0, ff 704 | base, dp4a | 0.0031 | 0.138 | 0.0023 | 0.114 | 0.132 | 0.125 | 0.080 |
| Q4_K / Q4_0, ff 704 | fix | 0.0031 | 0.0020 | 0.0023 | 0.0018 | 0.0021 | 0.0021 | 0.0028 |
| Q4_0 control, ff 512 | base and fix, MMA | 0.035 | 0.0014 | 0.0018 | 0.0014 | 0.0017 | 0.0020 | 0.0039 |
| Q6_K / Q6_K, ff 512 | base, MMA and dp4a | 0.0015 | 0.089 | 0.0017 | inf | inf | inf | inf |
| Q6_K / Q6_K, ff 512 | fix | 0.0015 | 0.0017 | 0.0017 | 0.0015 | 0.0016 | 0.0019 | 0.0029 |

On the fix a k-quant model takes the dp4a arm whatever `MEMRA_GEMMA_MOE_MMA` says, so its two
runs are equal. Base is the tree with memra#1225 landed. The test bounds each k-quant score by three times
its control's (floor 0.002, never above 0.01) and the share of rows above 0.01 by a third.

Placement (memra#1225): each k-quant variant on a 360-slot SLRU cache (pinned admissions and
chunked expert runs) is bit-identical to its resident run on all ten rows, so the `_em` matvec
chunks by expert like `_dec`. The Q4_0 programs are unchanged: memra#1225's placement test
passes on the fix and gemma-4-26B-A4B QAT Q4_0's argmax window gives memra#1225's tables.

**Scope.** gemma4_moe with a k-quant routed-expert bank; fixture evidence on one device. The
shipped gemma-4-26B-A4B QAT Q4_0 artifact has only Q4_0 experts and runs the programs it ran
before.
