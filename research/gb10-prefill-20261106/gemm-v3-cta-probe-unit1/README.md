# F8F4 v3: CTA-local TMA destination, per launch (GB10 unit 1, 2026-11-08)

Ledger row 44. The landed v3 mainloop (`fcdae6aaf2`) issues its copies with
`cp.async.bulk.tensor.2d.shared::cluster`. For that form the compiler wraps every copy in a CTA-id check
and a call slow path with 16 local stores, all in the one issuing thread. The `.shared::cta` form
compiles on sm_121a without either. In CUDA 13.4 SASS of the follow-up TU, the v3 plain kernel goes
from 1,688 to 1,152 instructions. It keeps the same 128 FFMA, 192 FMUL and 32 QMMA, and loses its 16
CALL and 16 STL.

Builds (each TU's hash is in its directory's `SHA256SUMS`):

- **A:** the landed TU.
- **B:** the CTA form, plus v2 on the shared stage compute and epilogues (`plain-resid/tu-B.cu.txt`).
  The v2 kernels' SASS is identical to A, so the v2 times anchor both builds.
- **C:** the landed TU with only the destination changed (`gateup/tu-C.cu.txt`). It is used to
  time the fused gate/up silu launch on v3, which the landed entry keeps on v2.

Drivers call the engine entries with the image pre-pass included: `plain-resid/drv_cta.cu` (built
against A and B) and `gateup/drv_gateup.cu` (built against C). Each run uses T 2,048 or 4,096 and
11 launches per arm; the median is reported. Both in-process orders run, twice. A and B runs
interleave. Scripts: `cta-probe.sh`, `e1-probe.sh`, `probe_sum.py`.

| T | launch | v3 A | v3 B | B against A |
|---|---|---|---|---|
| 2,048 | down 5120 x 17408 | 3.893 ms | 3.665 ms | +6.2% |
| 2,048 | down + residual | 4.159 ms | 3.915 ms | +6.2% |
| 2,048 | ssm_out 5120 x 6144 | 1.425 ms | 1.365 ms | +4.4% |
| 2,048 | wqkv 10240 x 5120 | 2.356 ms | 2.311 ms | +2.0% |
| 4,096 | down 5120 x 17408 | 7.001 ms | 6.535 ms | +7.1% |
| 4,096 | down + residual | 7.368 ms | 6.973 ms | +5.7% |
| 4,096 | ssm_out 5120 x 6144 | 2.521 ms | 2.429 ms | +3.8% |
| 4,096 | wqkv 10240 x 5120 | 4.083 ms | 4.014 ms | +1.7% |

The fused gate/up silu launch (2 x 17408 x 5120) on v3 C against v2: 8.053 -> 7.152 ms (+12.6%) at T
2,048 and 14.968 -> 13.442 ms (+11.4%) at T 4,096, n = 4 each. The landed cluster form measured
-1.2 / -0.2% on this launch (ledger row 41).

Output bytes are identical between v2 and v3 in every run. Spills under the Spark's CUDA 13.0
(ptxas `-v`):

- The follow-up's v3 gate/up kernel spills 12 / 12 B. The landed cluster form spilled 28 / 32 B, and
  v2's gate/up kernel spills 12 / 16 B.
- The follow-up's v3 plain and residual kernels spill nothing. C's spill 8 / 8 B.
