# One nvfp4-rig decode token, per kernel (2026-11-22, af6259f73d)

nsys with graph-node tracing, ink2-gpu --gen 32 on p1 (short context) and on p3 at 4096 prompt tokens, the
last 14 decode calls; RTX 5090 Laptop, full clocks. Bytes are the weights a token reads (safetensors headers,
48 slots; experts 8 of 128 a slot) and the KV a token reads (e4m3 K and V, 8 KV heads x 128). Roofline
865 GB/s DRAM read. Raw: prof-bk (sqlite exports and logs), bk.py.

## c=1, short context: 14.49 ms wall (69 tok/s under the profiler), floor 3.58 ms

| Kernel | launches | us / token | bytes / token | GB/s | of 865 GB/s |
|---|---|---|---|---|---|
| ink2_expert_gemm_kernel (W4A4 experts) | 96 | 4322 | 1.959 GB | 453 | 52% |
| ink2g_gemv_nvfp4_jobs_kernel (dense NVFP4 + unembed) | 97 | 2754 | 0.986 GB | 358 | 41% |
| ink2g_gemv_bf16_kernel (router, rel, sink, BF16 rest) | 96 | 618 | 0.154 GB | 249 | 29% |
| ink2g_attn_t4_kernel + merge (f32t attention) | 96 | 1264 | ~4 MB KV | - | latency-bound |
| depth mix (dot 1100 + out 427) | 240 | 1527 | activations | - | latency-bound (f64) |
| norms | 153 | 744 | activations | - | latency-bound |
| controller (up 630 + down 578) | 192 | 1207 | in BF16 rest | - | latency-bound |
| conv | 192 | 588 | activations | - | latency-bound |
| act-quant | 96 | 433 | activations | - | latency-bound |
| heads, route, gate, sums, casts, groups, argmax, softcap, embed | ~585 | 668 | small | - | latency-bound |
| gaps between graph nodes | - | 382 | - | - | - |

Bandwidth kernels: 7.69 ms for 3.10 GB (403 GB/s, 47%). Latency-bound kernels: 6.4 ms, no bandwidth in
them. About 1690 kernel launches a token, in graphs.

## 4k context: 22.61 ms wall

The same kernels as c=1 within 3%, except ink2g_attn_t4_kernel: 6311 us a token (shipped 5773 us) for about
0.15 GB of KV (8 global slots x 4096 keys + 40 sliding slots x 1024 keys, 2 KB a key): 24 GB/s, 3% of the
roofline. Gaps between graph nodes grow to 3.50 ms (shipped 2.97 ms). Why nvfp4-rig is below shipped at 4k:
its attention reads the e4m3 cache one byte at a time and decodes each element in every query head of a
group (24 heads over 8 KV heads), so it costs more instructions than shipped's f32 cache. That is about
0.5 ms of the gap; the larger gaps account for the rest.

## The gaps, in the order they are attacked

1. f32t attention: 1.26 ms at c=1, 6.4 ms at 4k for well under 1 ms of KV. One block a (row, KV head, tile)
   serving its 3 query heads, the K and V rows loaded 16 bytes at a time and decoded once; the same
   per-key and per-output chains, so the same bits.
2. Latency-bound small kernels, 6.4 ms: fuse per layer (norm into the GEMV that reads it, mix dot and out,
   the controller pair).
3. Dense NVFP4 GEMV at 41% and the expert GEMM at 52% of the roofline.
4. Graph-node gaps at long context (3.5 ms at 4k).
