#!/usr/bin/env python3
"""derive.py: prime attention work for Qwen3.8-27B at the served lengths (16 full-attention layers,
24/4 heads x 256, 4,096-row chunks, h2/h2p tiling: 64-row CTAs of two heads, 32-key tiles)."""
import math
H, HKV, D, L, C = 24, 4, 256, 16, 4096
for P in (32784, 131051, 259951):
    causal = 4 * H * D * P * (P + 1) / 2 * L
    ex = ws = qr = l2 = 0
    for c in range(math.ceil(P / C)):
        t0 = c * C; T = min(C, P - t0); Tkv = t0 + T
        ws += Tkv * 2 * HKV * D * 2          # bf16 K+V workspace written per layer-chunk
        qr += Tkv * 1856                      # q8_0 K + q5_1 V read by the dequant pass
        for qb in range(math.ceil(T / 64)):
            nt = (t0 + min(qb * 64 + 63, T - 1)) // 32 + 1
            ex += 12 * 2 * 64 * nt * 32 * D * 2 * 2
            l2 += 12 * nt * 32 * 2 * D * 2
    ex *= L; ws *= L; qr *= L; l2 *= L
    print(f"P={P}: causal {causal/1e15:.4f} PFLOP, executed {ex/1e15:.4f} PFLOP (+{100*(ex/causal-1):.2f}%), "
          f"at 102 TFLOPS {causal/102e12:.1f} s, workspace written {ws/1e9:.1f} GB, quantized KV read {qr/1e9:.1f} GB, "
          f"K/V streamed L2->SM {l2/1e12:.2f} TB, KV cache {P*31552/1e9:.2f} GB")
