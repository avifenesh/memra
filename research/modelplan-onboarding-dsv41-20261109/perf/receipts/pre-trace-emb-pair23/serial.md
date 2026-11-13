
## cell 0: about 4096 rows (first 16-row mHC pass of 256 columns)
host TTFT (the cell's NVTX range) 474.2 ms
| span | ms |
|---|---|
| cell start -> hashing (the sequence's state, bookkeeping) | 0.00 |
| hashing (Engram hashes of the prompt) | 0.69 |
| hashing -> embed | 0.00 |
| embed (id check, ids' upload, the row gather launched on every rank) | 0.14 |
| cell start -> first kernel launch call (cudaLaunchKernel_v7000) | 0.76 |
| cell start -> first kernel on a device (v41_embed_rows_kernel, device 1) | 0.83 |
| cell start -> first 16-row mHC pass (layer 0) on device 0 | 0.94 |
CUDA calls before the first launch (all threads): cuMemcpyHtoDAsync_v2 2 / 0.04 ms; cuMemAllocAsync 6 / 0.02 ms; cuLibraryGetKernel 1 / 0.00 ms; cuCtxSetCurrent 1 / 0.00 ms; cuKernelGetName 2 / 0.00 ms
in the cell: cuMemHostAlloc 0 calls / 0.0 ms; cuMemAllocAsync of >= 1 ms 0 calls / 0.0 ms; host-to-device copies >= 1 MB: 25.2 MB pinned dev 1 0.44 ms, 25.2 MB pinned dev 0 0.44 ms, 25.2 MB pinned dev 0 0.44 ms, 25.2 MB pinned dev 1 0.44 ms

## cell 1: about 32768 rows (first 16-row mHC pass of 2048 columns)
host TTFT (the cell's NVTX range) 3326.0 ms
| span | ms |
|---|---|
| cell start -> hashing (the sequence's state, bookkeeping) | 0.01 |
| hashing (Engram hashes of the prompt) | 6.14 |
| hashing -> embed | 0.00 |
| embed (id check, ids' upload, the row gather launched on every rank) | 0.07 |
| cell start -> first kernel launch call (cudaLaunchKernel_v7000) | 6.20 |
| cell start -> first kernel on a device (v41_embed_rows_kernel, device 0) | 6.21 |
| cell start -> first 16-row mHC pass (layer 0) on device 0 | 7.21 |
CUDA calls before the first launch (all threads): cuMemcpyHtoDAsync_v2 2 / 0.03 ms; cuMemAllocAsync 4 / 0.02 ms; cuKernelGetName 1 / 0.00 ms
in the cell: cuMemHostAlloc 0 calls / 0.0 ms; cuMemAllocAsync of >= 1 ms 0 calls / 0.0 ms; host-to-device copies >= 1 MB: 50.3 MB pinned dev 1 0.88 ms, 50.3 MB pinned dev 0 0.88 ms, 1.6 MB pinned dev 1 0.03 ms, 1.6 MB pinned dev 0 0.03 ms, 50.3 MB pinned dev 0 0.88 ms, 50.3 MB pinned dev 1 0.88 ms, 1.6 MB pinned dev 0 0.03 ms, 1.6 MB pinned dev 1 0.03 ms, 50.3 MB pinned dev 1 0.88 ms, 50.3 MB pinned dev 0 0.88 ms, 1.6 MB pinned dev 0 0.03 ms, 1.6 MB pinned dev 1 0.03 ms, 50.3 MB pinned dev 1 0.88 ms, 50.3 MB pinned dev 0 0.88 ms, 1.6 MB pinned dev 0 0.03 ms, 1.6 MB pinned dev 1 0.03 ms, 50.3 MB pinned dev 1 0.88 ms, 50.3 MB pinned dev 0 0.88 ms, 1.6 MB pinned dev 1 0.03 ms, 1.6 MB pinned dev 0 0.03 ms, 50.3 MB pinned dev 1 0.88 ms, 50.3 MB pinned dev 0 0.88 ms, 1.6 MB pinned dev 1 0.03 ms, 1.6 MB pinned dev 0 0.03 ms, 50.3 MB pinned dev 1 0.88 ms, 50.3 MB pinned dev 0 0.88 ms, 1.6 MB pinned dev 1 0.03 ms, 1.6 MB pinned dev 0 0.03 ms, 50.3 MB pinned dev 1 0.88 ms, 50.3 MB pinned dev 0 0.88 ms, 1.6 MB pinned dev 1 0.03 ms, 1.6 MB pinned dev 0 0.03 ms
