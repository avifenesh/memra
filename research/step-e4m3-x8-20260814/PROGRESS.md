# Step E4M3 short-width tactic

## Verdict

Promote Memra's own 128-output-row x 8-token tactic for block-128 E4M3 calls with
`m <= 8`. Keep `MEMRA_FP8_MMQ_X8=0` as the literal rollback to the previous
128/256-token padded launch.

This is a GEMM-kernel result. It is not an end-to-end Step throughput claim.

## Mechanism

The existing kernel already places weights in MMA-A and the short token axis in
MMA-N. The change retains the existing dynamic E4M3 activation quantizer, scale
law, weight bytes and scale grid, MMA instruction, and epilogue. It stops
computing padded token columns for decode and short verification.

External implementations were read only to identify the operand-orientation and
short-N tactic family. No external headers, code, library, runtime dependency, or
submodule are used by the implementation.

## Gates

- RTX 5090 full ragged kernel battery: PASS.
- RTX 5090 Step gate/up/down widths `m=1..8`: PASS, zero exact-arm bit
  mismatches, all 254 legal E4M3 codes covered.
- Three RTX PRO 6000 cards, widths `m=1/4/8`: PASS independently on every card
  with the same exactness and code-coverage requirements.
- Naked-default admission: two independent boots on the RTX 5090 and two boots
  on each PRO card reproduced the forced-X8 timing band; literal
  `MEMRA_FP8_MMQ_X8=0` reproduced the previous padded-launch band.
- Performance: balanced adjacent order, synchronized kernel timings, N=6 per
  arm on 5090 and N=5 per arm per PRO card, with nine interleaved FP8/Q8 samples
  inside each process.

The structured results are in `RESULTS.json`.

## Reproduction

Build the Memra binaries for Blackwell:

```bash
MEMRA_CUDA_ARCH=120a cargo build --release -p memra-engine \
  --bin fp8_mmq_check --bin fp8_mmq_bench
```

Run the exactness controls:

```bash
MEMRA_FP8_MMQ_X8=1 target/release/fp8_mmq_check 4096 1280 1
MEMRA_FP8_MMQ_X8=1 target/release/fp8_mmq_check 1280 4096 1
```

Run one internally interleaved timing cell:

```bash
MEMRA_FP8_MMQ_X8=1 target/release/fp8_mmq_bench 1 9 step37
MEMRA_FP8_MMQ_X8=0 target/release/fp8_mmq_bench 1 9 step37
```

Repeat in balanced outer order. The public numbers require at least five
independent processes per arm.
