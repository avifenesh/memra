# gemma4_moe k-quant expert banks: per-qtype kernel dispatch (memra#1234)

RTX 5090 Laptop GPU (24 GB). Fixture only: no k-quant gemma4_moe GGUF is on any unit this repo
tests on, so there is no real-checkpoint window, served run or speed number here. Base is
f53bfef3cd (memra#1225 and memra#1232 landed; main's crates at d1ed663256 are the same); the fix
is this change on top of it. One card, one lease per
cell, single runs (exactness only, no timing). Binaries' sha256 in `binaries.sha256`. Decision
record: `docs/decisions/GEMMA-KQUANT-EXPERT-DISPATCH.md`.

## The defect

`gemma4_moe` admits the `expert_dp4a_supported` qtypes to its device arms: Q4_0, IQ4_XS, IQ3_S,
Q3_K, Q4_K and Q6_K. Three programs decode only Q4_0, IQ4_XS and IQ3_S and read any other qtype
through one of those branches:

| program | when | kernel | reads a k-quant bank as |
|---|---|---|---|
| pairs arm, int8 MMA | t >= 16, default when `n_embd % 256 == 0` | `mmq_iq_experts_kernel` | IQ3_S tiles |
| pairs arm, dp4a | t >= 16 under `MEMRA_GEMMA_MOE_MMA=0`, or `n_embd % 256 != 0` | `moe_pairs_matvec_q8_dec` | IQ4_XS (`expert_decode_g`) |
| verify CSR twin, gate/up | t = 2..10 | `moe_gate_up_gelu8_dev_q8_csr` | IQ3_S (`exp_decode_g_cached`) |

The issue named the first two (read from the code by memra#1225's independent review). The
third came out of this test: with only the pairs arm fixed, T = 8 stayed wrong in both k-quant
variants (`pairs-only/`). The decode kernels (t = 1), the verify rows twin (t = 11..15) and the
down kernels carry the k-quant bodies and were right.

## The fix

The qwen-class arms' per-qtype dispatch: the MMA arm only when gate, up and down are in the
decode set (`q8_expert_dec_supported`); on the dp4a arm `moe_pairs_matvec_q8_dec` for the decode
set and `moe_pairs_matvec_q8_em` otherwise, per projection; the CSR twin only when gate and up
are in the decode set, else the rows twin. No new kernel, no new flag. Q4_0, IQ4_XS and IQ3_S
models run the kernels they ran.

## The test

`crates/memra-engine/tests/gemma4_moe_kquant_pairs_gpu.rs`: memra#1225's two-layer gemma4_moe
placement fixture (128 experts, top 8, width 256, a 512-id vocabulary, a spread router) with the
expert banks in five variants:

- Q4_0 gate/up and down, ff 704: the control (the shipped QAT class);
- Q3_K gate/up, Q4_0 down, ff 704: the layout a stock llama.cpp Q3_K quantization gives
  gemma-4-26B-A4B. Its down k (704) is not a multiple of the 256-value k-quant superblock, so
  the quantizer falls back to a 32-value type there, and Q3_K's fallback is Q4_0
  (`tensor_type_fallback` in llama.cpp's `src/llama-quant.cpp`, read on 2026-11-11);
- Q4_K gate/up, Q4_0 down, ff 704. The same table sends Q4_K to Q5_0 and Q6_K to Q8_0, which
  the device arms do not admit, so a stock Q4_K or Q6_K GGUF of the 26B keeps the host-staged
  loop. This variant and the next hold the dispatch for those qtypes on other geometries;
- Q6_K gate/up and down, ff 512;
- Q4_0 gate/up and down, ff 512: that variant's control.

Each variant runs forward at T = 1, 8, 12, 16, 40 and 130 and is compared with
`memra_reference::execute` over the dequantized weights the engine reads, so the only error left
is the engine's activation and KV quantization. A width's score is the median over its rows of
the row's largest logit error over the largest reference logit; the maximum and the share of
rows above 0.01 are printed next to it. The median is the score because a near-tie in the
router's top-k that activation quantization flips moves a few rows a lot in any variant, the
Q4_0 control included (largest row 0.08 and 0.12 at T = 130), while a bank read with the wrong
layout moves every row. A k-quant score may be at most three times its control's (floor 0.002)
and never above 0.01, and at most a third of a width's rows may be above 0.01. Then a 16-token
prime's last row is compared with 16 decode steps. Last, each k-quant variant runs ten forwards
(the six widths, a 600-token and three 1,000-token prefills) on a 360-slot SLRU cache and must
be bit-identical to its resident run. The test refuses to run with `MEMRA_GEMMA_CSR`,
`MEMRA_GEMMA_MOE_FAST`, `MEMRA_GEMMA_MOE_PAIRS` or `MEMRA_MOE_F16G` set.

The k-quant encoders in the test are direct fits, not ggml's search. The reference reads the
same dequantized values as the engine, so encoder quality does not enter the comparison; a CPU
test checks they round-trip.

## Results

Median row error against the reference (`red-*.log` on base, `green-*.log` on the fix; the
int8-MMA default and `MEMRA_GEMMA_MOE_MMA=0`):

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

On the fix a model with a k-quant bank takes the dp4a arm whatever `MEMRA_GEMMA_MOE_MMA` says,
so its two runs are equal. The controls' dp4a-arm rows differ from their MMA rows in the fourth
digit at T >= 16 and are in the logs. The ff 512 control's T = 1 score is one row. On base every
row of the failing widths is above 0.01; on the fix the largest share is 3 of 12 rows (Q3_K at
T = 12; the controls reach 1 of 12 and 16 of 130).

Placement, 360-slot SLRU against resident, ten rows per variant:

| variant | build | SLRU calls: row / pinned / chunked | rows differing from resident |
|---|---|---|---|
| Q3_K / Q4_0, ff 704 | fix, both pairs settings | 0 / 16 / 4 | 0 of 10 |
| Q4_K / Q4_0, ff 704 | fix, both pairs settings | 0 / 16 / 4 | 0 of 10 |
| Q6_K / Q6_K, ff 512 | fix, both pairs settings | 0 / 16 / 4 | 0 of 10 |
| Q3_K / Q4_0, ff 704 | base, MMA | 0 / 16 / 4 | 0 of 10 |
| Q3_K / Q4_0, ff 704 | base, dp4a | 0 / 16 / 4 | 7 of 10 (every T >= 16) |
| Q4_K / Q4_0, ff 704 | base, both | 0 / 16 / 4 | 0 of 10 |
| Q6_K / Q6_K, ff 512 | base, both | 0 / 20 / 0 | 0 of 10 |

On base the two placements mostly agree on the same wrong numbers. The Q3_K dp4a row is the
exception: the decode-once kernel walks a 110 B Q3_K superblock as a 136 B IQ4_XS one, so it
reads past each row into whatever follows it, which differs between the slab and a cache slot.
The Q6_K variant's non-finite hidden state selects too few experts to chunk on base, so base
also fails that arm's non-vacuity check.

The Q4_0 programs are unchanged (`q4_0/`): memra#1225's placement test passes on the fix (0 of
20 rows differ in the roomy, mid and tight arms), and gemma-4-26B-A4B QAT Q4_0's 12-row
`board-2048` argmax window on the fix gives memra#1225's tables: resident eager
`3768249ed8dbaaf2`, resident batch `3f1daf1cbcf22f6a`, 360-slot SLRU eager `3768249ed8dbaaf2` (the table
hash `cells.txt` records; memra#1225's final cells record the same eager and batch hashes). The SLRU
cell's per-call trace lines are dropped from its log; `trace-summary.txt` counts them.

`pairs-only/`: an earlier revision of the test (largest-row score, no T = 12, no Q3_K variant,
no placement check) on a build with only the pairs-arm fix, before memra#1232. T >= 16 is repaired for the Q6_K
variant and T = 8 is unchanged from base in both k-quant variants (0.185 and 0.112 largest
row): the verify CSR twin.

## Scope

- Fixture only, one device. No real k-quant gemma4_moe checkpoint was run.
- Reach: a stock llama.cpp Q3_K quantization of gemma-4-26B-A4B (Q3_K gate/up, Q4_0 down) is
  on the device arms and was wrong on base at prefill and at speculative verify. Stock Q4_K and
  Q6_K quantizations of that model fall back to Q5_0 and Q8_0 down banks and stay on the
  host-staged loop, before and after. The shipped QAT Q4_0 artifact has only Q4_0 experts and is
  not affected by the defect or the fix.
- The grouped f16 lane (`MEMRA_MOE_F16G`, env-explicit for gemma) is selected before these
  arms and has its own k-quant dequant kernels; it was not run here.
- No speed claim: the `_em` matvec and the rows twin are the k-quant programs that exist.
- The test is `#[ignore]` (GPU) like memra#1225's placement test and is in no battery cell.

## Reproduce

Build `cargo test --release -p memra-engine --test gemma4_moe_kquant_pairs_gpu --no-run` on the
base and on the fix, copy the two test binaries to `bin/gemma4_moe_kquant_pairs_gpu-red` and
`-fix`, then `KQ_DIR=<dir> tools/run_kq.sh red red-mma`, `... red red-dp4a
MEMRA_GEMMA_MOE_MMA=0`, and the same two with `fix` and `green-`. `binaries.sha256`: `-red` is
the final test on the base, `-fix` the final test, probe and placement test on the
fix, `-pairs-only` the earlier test revision on the pairs-arm fix alone.
