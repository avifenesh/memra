# Native FP4 pair-store R2 runnable gate, compile-only, 2026-09-07

## Runnable arm

`tools/dsv4-gu-native-fp4-pair-gate.cu` now has an actual harness, not a
print-only main. It allocates the existing bank128/hidden4096/inter2048
fixture, launches the real current GU-half2 and real GU-M1+half2 controls, and
launches native M1 and native non-M1 symbols. It warms all three, flushes 2x
L2 before every scored launch, runs three ABBA cycles, checks dispatch deltas,
finite live rows, guards, and bit-exact H outputs.

The basis kernel covers all 256 packed code bytes x 256 signed-E4M3 scale
bytes against the actual current `kq_store_variant<QT_NVFP4_MODELOPT,true>`
helper. Nibble 8 is explicitly normalized to the helper's +0 policy before
native conversion. Basis comparisons are raw half2 bit comparisons; no NaN
tolerance path is used; every packed pair writes/validates all 16 half outputs
and a tail guard. The exact active-group set is 1,3,4,6. The actual GU-M1
counter is the GU half2 enqueue counter, not the down counter. Native-vs-current
and native-vs-M1 each have separate current/native/native/current ABBA means.

## Compile receipt

```sh
/usr/local/cuda-13.1/bin/nvcc -std=c++17 -O3 -fmad=false \
  -Xcompiler=-ffp-contract=off -gencode arch=compute_120a,code=sm_120a \
  -lcublasLt -lcublas -lcuda \
  -o target/dsv4-gu-native-fp4-pair-gate-r2 \
  tools/dsv4-gu-native-fp4-pair-gate.cu
```

```text
binary sha256 6eb7116382affef4e9e505c2ef843dba65b2a6bfec4356af3a8fe1cc059d07c9
source sha256 ac472b4366ee04371784b3df057bf7485a745fdcab3dfa0a937686797ff6e178
native M1 resource 122 registers/thread, 26128 shared bytes
native non-M1 resource 138 registers/thread, 26128 shared bytes
current non-M1 resource 150 registers/thread, 26128 shared bytes
basis resource 40 registers/thread, 2048 shared bytes
```

SASS contains `F2FP.F16.E2M1.UNPACK_B`. No GPU/Cargo/production run was made.
