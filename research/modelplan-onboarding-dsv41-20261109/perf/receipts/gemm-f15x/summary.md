| shape | f14 ms (spread) | f15x ms (spread) | f15x / f14 | bytes |
|---|---|---|---|---|
| op=fp8 name=wq_a m=1 n=1280 k=5120 | 0.0126 (1.4%, n=3) | 0.0126 (1.1%, n=3) | 1.001 | same |
| op=fp8 name=wq_a m=64 n=1280 k=5120 | 0.0750 (0.1%, n=3) | 0.0750 (0.1%, n=3) | 1.000 | same |
| op=fp8 name=wq_a m=4096 n=1280 k=5120 | 0.2597 (0.1%, n=3) | 0.2767 (0.2%, n=3) | 1.065 | same |
| op=fp8 name=wq_a m=8192 n=1280 k=5120 | 0.5127 (0.2%, n=3) | 0.5328 (0.4%, n=3) | 1.039 | same |
| op=fp8 name=wkv m=1 n=512 k=5120 | 0.0120 (0.4%, n=3) | 0.0120 (0.3%, n=3) | 0.999 | same |
| op=fp8 name=wkv m=64 n=512 k=5120 | 0.0747 (0.1%, n=3) | 0.0749 (0.2%, n=3) | 1.003 | same |
| op=fp8 name=wkv m=4096 n=512 k=5120 | 0.1184 (0.3%, n=3) | 0.1270 (0.5%, n=3) | 1.072 | same |
| op=fp8 name=wkv m=8192 n=512 k=5120 | 0.2331 (0.2%, n=3) | 0.2497 (0.1%, n=3) | 1.071 | same |
| op=fp8 name=wq_b m=1 n=32768 k=1280 | 0.0388 (0.3%, n=3) | 0.0387 (0.4%, n=3) | 0.998 | same |
| op=fp8 name=wq_b m=64 n=32768 k=1280 | 0.0410 (0.3%, n=3) | 0.0411 (0.3%, n=3) | 1.002 | same |
| op=fp8 name=wq_b m=4096 n=32768 k=1280 | 1.7582 (0.2%, n=3) | 1.8752 (0.1%, n=3) | 1.067 | same |
| op=fp8 name=wq_b m=8192 n=32768 k=1280 | 3.4649 (0.1%, n=3) | 3.7215 (0.2%, n=3) | 1.074 | same |
| op=fp8 name=idx_wq_b m=1 n=4096 k=1280 | 0.0111 (0.6%, n=3) | 0.0111 (1.3%, n=3) | 1.000 | same |
| op=fp8 name=idx_wq_b m=64 n=4096 k=1280 | 0.0259 (0.4%, n=3) | 0.0258 (2.2%, n=3) | 0.997 | same |
| op=fp8 name=idx_wq_b m=4096 n=4096 k=1280 | 0.2406 (0.3%, n=3) | 0.2420 (0.1%, n=3) | 1.006 | same |
| op=fp8 name=idx_wq_b m=8192 n=4096 k=1280 | 0.4687 (0.3%, n=3) | 0.4789 (0.1%, n=3) | 1.022 | same |
| op=fp8 name=wo_b m=1 n=5120 k=8192 | 0.0297 (0.2%, n=3) | 0.0297 (0.3%, n=3) | 1.001 | same |
| op=fp8 name=wo_b m=64 n=5120 k=8192 | 0.1192 (0.3%, n=3) | 0.1190 (0.6%, n=3) | 0.999 | same |
| op=fp8 name=wo_b m=4096 n=5120 k=8192 | 1.5184 (0.1%, n=3) | 1.5618 (0.1%, n=3) | 1.029 | same |
| op=fp8 name=wo_b m=8192 n=5120 k=8192 | 3.1245 (0.0%, n=3) | 3.1902 (0.3%, n=3) | 1.021 | same |
| op=fp8 name=shared_w13 m=1 n=2304 k=5120 | 0.0144 (0.7%, n=3) | 0.0145 (0.3%, n=3) | 1.004 | same |
| op=fp8 name=shared_w13 m=64 n=2304 k=5120 | 0.0751 (0.2%, n=3) | 0.0750 (0.8%, n=3) | 1.000 | same |
| op=fp8 name=shared_w13 m=4096 n=2304 k=5120 | 0.4896 (0.2%, n=3) | 0.4936 (0.1%, n=3) | 1.008 | same |
| op=fp8 name=shared_w13 m=8192 n=2304 k=5120 | 0.8965 (0.1%, n=3) | 0.9204 (0.2%, n=3) | 1.027 | same |
| op=fp8 name=shared_w2 m=1 n=5120 k=2304 | 0.0156 (0.3%, n=3) | 0.0156 (0.3%, n=3) | 0.999 | same |
| op=fp8 name=shared_w2 m=64 n=5120 k=2304 | 0.0403 (0.8%, n=3) | 0.0400 (0.3%, n=3) | 0.994 | same |
| op=fp8 name=shared_w2 m=4096 n=5120 k=2304 | 0.4789 (0.1%, n=3) | 0.4945 (0.2%, n=3) | 1.033 | same |
| op=fp8 name=shared_w2 m=8192 n=5120 k=2304 | 0.9585 (0.1%, n=3) | 0.9847 (0.1%, n=3) | 1.027 | same |
| op=mxfp4_tiles name=exp_w13 experts=6 rows=1 bm=1 n=2304 k=5120 | 0.0516 (0.1%, n=3) | 0.0516 (0.3%, n=3) | 1.001 | same |
| op=mxfp4_tiles name=exp_w2 experts=6 rows=1 bm=1 n=5120 k=2304 | 0.0500 (0.7%, n=3) | 0.0498 (0.4%, n=3) | 0.995 | same |
| op=mxfp4_tiles name=exp_w13 experts=64 rows=64 bm=64 n=2304 k=5120 | 0.9722 (0.1%, n=3) | 0.9732 (0.1%, n=3) | 1.001 | same |
| op=mxfp4_tiles name=exp_w2 experts=64 rows=64 bm=64 n=5120 k=2304 | 0.8381 (0.0%, n=3) | 0.8386 (0.2%, n=3) | 1.001 | same |
| op=mxfp4_tiles name=exp_w13 experts=384 rows=64 bm=64 n=2304 k=5120 | 5.1762 (0.2%, n=3) | 5.1807 (0.1%, n=3) | 1.001 | same |
| op=mxfp4_tiles name=exp_w2 experts=384 rows=64 bm=64 n=5120 k=2304 | 4.8727 (0.0%, n=3) | 4.8704 (0.3%, n=3) | 1.000 | same |
| op=mxfp4_tiles name=exp_w13 experts=384 rows=512 bm=128 n=2304 k=5120 | 35.8876 (0.1%, n=3) | - | - | same |
| op=mxfp4_tiles name=exp_w2 experts=384 rows=512 bm=128 n=5120 k=2304 | 33.7181 (0.1%, n=3) | - | - | same |
| op=mxfp4_tiles name=exp_w13 experts=384 rows=512 bm=64 n=2304 k=5120 | - | 39.7683 (0.5%, n=3) | - | same |
| op=mxfp4_tiles name=exp_w2 experts=384 rows=512 bm=64 n=5120 k=2304 | - | 38.7518 (0.5%, n=3) | - | same |
