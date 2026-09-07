# DSV4 attention FP8 packing component

This is a standalone storage adapter, not an attention runtime path or a performance claim.

- `wq_b` (32768 x 1024) and `wo_a` (8192 x 4096) split by contiguous output rows.
- `wo_b` (4096 x 8192) splits input columns and repacks each row to the local physical stride.
- FP8 code bytes and decoded FP32 128 x 128 scale grids remain separate and bit-preserved.
- Device packing admits only source allocations on the copy stream. With event tracking disabled, that keeps source deallocation ordered after asynchronous copies; foreign code or scale streams refuse.
- The eventual `wo_b` partial-sum reduction is a separate numeric class. This component neither chooses its tolerance nor claims equivalence to a full-width accumulation.

## Review and evidence

The source was isolated from component commit `dddb0eed789eb00a867e8d23b0c7089fb8e48c7d` onto main `f4d73e0252929dfb0f0ee831768b1d60e0431544`. Review checked partition alignment/ranges, packed scale strides, unsafe copy guards and ownership, and the independent full-plane oracle. Delivery correction `b168ba4e4477f3ca765304bb4a186a6668ea3aa4` adds the same-stream ownership refusal, its GPU negative cases, and test-only Clippy cleanup without changing copy arithmetic.

CPU fixtures cover both logical ranks, real projection dimensions, reconstruction, wrong-rank data and changed FP32 scales. The ignored GPU fixture covers all three shapes and both logical ranks, source immutability, destination canaries, and foreign-stream refusal.

The final component, including the new ownership guard, passed on two RTX PRO 6000 Blackwell devices on 2026-09-07. The remote build used Rust 1.97.1, CUDA 13.1.115 and `sm_120a`. Four CPU component tests passed; the ignored GPU fixture passed once normally and once under memcheck on each physical device. Both memcheck runs reported zero errors. Each GPU execution includes both logical ranks, all three projection shapes and the foreign code/scale stream refusal cases. The GPU controller completed at 16:14:58 UTC; its process was gone, no GPU process remained and the shared lock was free before handoff.

| Evidence binding | Value |
| --- | --- |
| Tested source commit | `f72dc0f884c0b4a46e2b4c2f08259be43419c7b2` |
| Test binary SHA256 | `5447bf094aab9ba4d43855a7b1dc4fe4d5547a8ca5bf350ae1c672efd1072060` |
| Packer module SHA256, local and remote | `4b4952a9bb43d63e5b54abbb730997aec935fe821e6d4c258efcd187a950ddab` |
| Companion raw receipt namespace | `attention-pack-f72dc-20260907-r1` |

Raw CPU/GPU/memcheck logs, build log, source/module/lockfile hashes and binary hash are retained in the companion operations receipt. The subsequent documentation update does not change the tested source module or copy arithmetic. This is component correctness evidence, not full-model attention TP qualification or a throughput measurement.

No local Cargo, CI, build, or GPU tests were run. Pushes use `MEMRA_SKIP_PERF_CI=1` under the owner's temporary local-rig prohibition, with normal hooks and hosted CI retained. This remains a draft until the final checks and bound target evidence are reviewed. No serving default, fleet pin, or model support state changes.
