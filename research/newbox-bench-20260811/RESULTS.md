# New two-card pair bench — 2026-08-11

Box: rented 2x RTX PRO 6000 WS **600W full power**
(old box was Max-Q 300W), PCIe 5.0 x16, driver 610.57.04, **peer path byte-verified
CLEAN** (0/16384 wrong both directions — serves on native cudaMemcpyPeerAsync, NO
host-bounce). $2.40/hr (old box $3.77). Binary: v0.75.0 tip, promoted serving config
(PP-2, MOE_GROUPED=1, PREFILL_TICK=2048, CTX=262144).

## Single-stream (N=3 short, N=1 others; streaming, temp=0)

| shape | new box | old box (Max-Q, bounce/peer mix) | delta |
|---|---:|---:|---|
| short TTFT | **0.133-0.148s** | 0.185-0.218s | **-32%** |
| short decode | **96.7 tok/s** | 69-74 | **+31-40%** |
| 4k cold TTFT | **5.227s** | 6.03-7.47 | **-13 to -30%** |
| depth decode (400 tok) | **101.0 tok/s** | 72.8 | **+39%** |

## Decode ladder (512-tok essays, distinct salts, total-window agg, single run)

| c | new box | old box | box1 (Server Ed.) |
|---|---:|---:|---:|
| 1 | 99.0 | 77.3 | 88.5 |
| 2 | 137.1 | 105.9 | 118.2 |
| 4 | 161.3 | 121.5 | 146.1 |
| 8 | **177.0** | 135.4 | 166.5 |

The new box BEATS box1 (previous best pair) at every rung — full 600W + PCIe 5.0 +
newer driver. The Max-Q tax thesis confirmed: it was ~15-20% and it is gone.

## Economy impact (receipts, same $/tok market prices)

- Cost: $89.5 -> **$57.6/day** (-36%) before any prepay; 1-mo prepay ~$46, 3-mo ~$40.
- Gross ceiling up ~30% with the throughput gain: Shape-B decode receipts now
  177 agg on-box (was 135-166 across boxes).
- Content verified coherent; one-hash + full gate battery = next block.
