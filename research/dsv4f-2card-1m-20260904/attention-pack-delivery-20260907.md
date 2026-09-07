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

Target execution was reported for the original component on two RTX PRO 6000 Blackwell devices with memcheck. Exact remote-source/binary/receipt correlation is still pending, and the new ownership guard has not yet been re-gated. Those reports are not a substitute for final-head target evidence.

No local Cargo, CI, build, or GPU tests were run. Pushes use `MEMRA_SKIP_PERF_CI=1` under the owner's temporary local-rig prohibition, with normal hooks and hosted CI retained. This remains a draft until the target evidence is bound and reviewed. No serving default, fleet pin, or model support state changes.
