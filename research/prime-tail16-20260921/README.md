# The 16-row prime segment (#427): the batched decode/verify tier ended one row too high

Verdict: a restored suffix whose final prime segment is exactly `PRIME_MIN_T` (16) rows is
bitwise with the one-call prime now. The cause was in the general matmul entries, not in the
restore protocol or the GDN scan: `matmul` and `matmul_pre` sent `m = 16` through the batched
small-`m` mmvq tier (`2..=16`, the decode/verify class) while every longer prime rode the generic
path. The tier now ends at `PRIME_MIN_T - 1` outside the verify-exact scope; inside that scope it
keeps its receipted `2..=16` width, and `matmul_decode_exact*` keep their own tiers (they are that
class). The one-call prime's logits did not change (same digest before and after), so no prefill
of 17+ rows moved; only 16-row prime calls did.

## Reproduction (local RTX 5090 Laptop, qwen35-9b NVFP4 MTP GGUF, 9,296 repo-text tokens)

`qwen-a4-continuation-gate <9B> prompt.txt 9296 16 48 80` (`raw/repro-before.log`):

| split | before |
| --- | --- |
| 9280 + 16 | DIFFERS |
| 9248 + 48 | ok |
| 9216 + 80 | ok |

The issue's table (served q38 mint and a calibrated A4 artifact, rented 5090) had the same
shape; the aligned tails of a 9,296-token prompt are exactly the ones congruent to 16 mod 32, so
16 was the only tail at or below 32 that the grid law lets through.

## Placement by door (`raw/doors-before.log`, tail 16 against the one-call prime)

| door | 16-row tail |
| --- | --- |
| default | DIFFERS |
| `MEMRA_GDN_CHUNKED=0` (sequential GDN scan) | DIFFERS (not the scan) |
| `MEMRA_FA_FLOOR=1`, `MEMRA_FA3=0`, `MEMRA_FA_BF16KV=0` | DIFFERS (not attention) |
| `MEMRA_GDN_MMA=0` | DIFFERS |
| `MEMRA_Q8RP=0` | DIFFERS |
| `MEMRA_MMVQ=0` | **ok** |
| `MEMRA_FAST=0` | **ok** |

`MEMRA_MMVQ=0` disables the mmvq class including its batched `_b2/_b4/_b8/_b16` twins; with it
off, m=16 falls to the generic path like m=48 and the digests agree. Reading the dispatch:
`matmul` takes the GEMM at `m >= 16` only when `out_f >= 128`; the small-`out_f` tensors (the ssm
gate projections and their kin) fall through to the tiers, where `(2..=16).contains(&m)` caught
m=16 and nothing else a prime ever produces.

## Fix and receipts (`raw/gate-after.log`)

`Engine::small_m_tier_max()` = 16 in the verify-exact scope, `PRIME_MIN_T - 1` otherwise, used by
`matmul` and `matmul_pre`. Gate on the fixed build, default doors: 9280+16, 9248+48, 9216+80,
9184+112, 9152+144, 9120+176, 9088+208 all `ok`, one-call digest `f15b4cd735a1296e` unchanged
from before the fix. `MEMRA_MMVQ=0` control still `ok`.

The continuation gate no longer exempts the 16-row tail and runs in `tools/local-ci.sh` after
prime-gate (tails 16/48/80 on the 9B, `MEMRA_CI_CONTGATE=0` skips).

The served Qwen3.8-27B NVFP4-Q5K mint (the artifact in the issue's table) on the same rig and
prompt, fixed build: 9280+16, 9248+48, 9216+80 all `ok` (`raw/gate-after-q38-27b.log`).

## Why the boundary sat there

`PRIME_MIN_T` is the floor of the batched prime program, and the small-`m` tier was written as
"m up to 16" from the decode/verify side, where 16 rows is the top verify width. The two numbers
coincided, so a 16-row prime was the one prefill call inside the decode tier. Any other floor would
have hidden it the same way; the fix expresses the tier's ceiling in terms of the prime floor so
they cannot drift back together.
