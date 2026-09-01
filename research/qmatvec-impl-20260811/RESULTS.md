# IQ4_XS qmatvec arm-1 results

Date: 2026-08-11

Branch: `lane/cx-qmatvec-impl`

Base: `49f5002d7a37291c9b551ac2f683ce2edb27d163`

## Verdict

**GO for orchestrator promotion.** The dense wide-load plus `byte_perm` arm is
bit-identical to the base kernel on every observed Step `(out_f, in_f)` pair,
including a `base+4` alignment fallback. Two local RTX 5090 Laptop GPU
interleaved windows independently measured about 28.1% lower synthetic
315-launch time, or 39.1% higher effective logical throughput.

This is local directional evidence, not a Step performance claim. The
orchestrator still owns the Step-shape Nsys N>=32, per-shape NCU, and 2x RTX PRO
6000 battery. No board value moves from this lane.

## Implementation

`qmatvec_iq4_XS_dp4a` now reuses the force-inlined
`expert_dot_iq4xs_g_v` recipe for each existing group assignment. That recipe
uses one aligned 64-bit header load, two aligned 64-bit quant loads, packed table
words, and `get_int_from_table_16_d`; it falls back to the original scalar group
dot for non-8-aligned bases.

The following are unchanged:

- IQ4_XS bytes, dtype, row layout, and `row_bytes`;
- `g = tid; g += blockDim.x` assignment;
- low/high dp4a issue order and integer accumulation order;
- the outer `acc +=` order;
- the CTA reduction tree; and
- the `(out_f, m)` grid with the 128-thread one-row block.

## Bit-identity gate

The lane-local harness generated deterministic weight, Q8 value, and Q8 scale
inputs for all 11 unique pairs in the committed Step semantic launch map:

`(8192,4096)`, `(12288,4096)`, `(1024,4096)`, `(64,4096)`,
`(96,4096)`, `(4096,8192)`, `(4096,12288)`, `(11264,4096)`,
`(4096,11264)`, `(1280,4096)`, and `(4096,1280)`.

Each binary emitted both the naturally aligned output and the same bytes at
`base+4`. Within each binary, aligned versus fallback had zero bit mismatches
for every shape. Across binaries, the complete raw dumps were byte-identical:

```text
baseline  e52f8369ce62dd250f4d203033bc5e8f6e6c1c9ba4e262ce84565d22eb92accf
candidate e52f8369ce62dd250f4d203033bc5e8f6e6c1c9ba4e262ce84565d22eb92accf
cmp exit  0
```

The base binary embeds `qmatvec.cu` from `49f5002d`; the candidate embeds the
working-tree rewrite. Both were compiled by CUDA 13.1 with
`-gencode arch=compute_120a,code=sm_120a -O3`.

## SASS receipt

The candidate aligned arm contains the expected three `LDG.E.EF.64` streaming
loads plus packed `PRMT` and `IDP.4A.S8.S8` instructions. The scalar instructions
remain in the same kernel for the alignment fallback. Register allocation moved
from 42 registers/thread at base to 40 registers/thread in the candidate;
shared memory stayed 1,152 bytes/CTA.

## 5090 correctness battery

Pre-change:

- `kernel-check`: `ALL GREEN (101 cells, 1 skipped)`.

Post-change, through a fresh full release build:

- raw output comparison: byte-identical on all 11 shapes and both alignments;
- `kernel-check`: `ALL GREEN (101 cells, 1 skipped)`;
- `run-gen`: argmax MATCH on the standing 31B and depth-12B cells;
- `run-spec`: Qwen 35B K=1..8 self-consistency PASS, 8/8;
- KAT-Coder dense-IQ4_XS probe: argmax MATCH and golden token-identical; and
- complete `tools/local-ci.sh`: exit 0, including the additional graph, serving,
  stress, and acceptance gates.

`local-ci` printed its existing non-fatal uncovered-flag warning for
`MEMRA_MOESD_CACHE_CAP` and `MEMRA_MOESD_DEPTH`; this lane adds no flag.

## Directional micro-timing

The harness allocates unique matrix bytes for the exact 315-launch two-device
semantic mix from the feasibility study: 2,872,946,688 weight bytes and
2,879,039,040 logical bytes per synthetic token sweep. It times aligned kernels
only. Runs were `nice -n 10`, `ionice -c 3`, serialized on an otherwise idle GPU,
and interleaved in both orders. The machine stayed on its `balanced` platform
profile; clocks were not locked or reset.

| Window | Repetitions/sample | N/arm | Thermal range | Base median | Candidate median | Time change | Effective-throughput change |
|---|---:|---:|---:|---:|---:|---:|---:|
| ABBA | 256 | 8 | 58-83 C | 6.217861 ms / 463.027 GB/s | 4.468865 ms / 644.244 GB/s | -28.129% | +39.137% |
| position-balanced AB/BA | 128 | 6 | 59-78 C | 5.910983 ms / 487.066 GB/s | 4.248757 ms / 677.619 GB/s | -28.121% | +39.123% |

The GB/s values divide the fixed logical byte bill by CUDA-event time; they are
not direct DRAM-counter measurements. The second window gives each arm three
first and three second positions, and reproduces the first window within 0.02
percentage point despite the unlocked-clock regime.

## Raw evidence

- `baseline-kernel-check.log`
- `baseline-exactness.log`
- `exactness-comparison.log`
- `baseline-sass.log` and `candidate-sass.log`
- `candidate-local-ci.log`
- `candidate-run-gen-kat.log`
- `interleaved-microtiming.log` and `interleaved-thermal.csv`
- `confirmation-microtiming.log` and `confirmation-thermal.csv`
