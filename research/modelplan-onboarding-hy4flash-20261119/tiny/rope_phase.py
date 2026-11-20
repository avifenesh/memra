"""Largest RoPE phase difference between transformers' inverse frequencies and memra's, in f64.

transformers: inv_freq = 1 / base ** (arange(0, dim, 2) / dim), phase = position * inv_freq.
memra-reference: phase = position * base.powf(-(2i) / dim). Both rotate the trailing 64 dims."""
import math
import torch
for base in (1e7, 1e4):
    for dim in (64, 8):
        tf = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
        mm = torch.tensor([base ** (-(2 * i) / dim) for i in range(dim // 2)], dtype=torch.float64)
        for top in (544, 262_144, 1_048_576):
            pos = torch.arange(top, dtype=torch.float64)[:, None]
            d = (pos * tf - pos * mm).abs().max().item()
            print(f"base {base:g} dim {dim} positions <{top}: max phase difference {d:.2e} rad")
