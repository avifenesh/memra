# One-shot out-of-place transport: RTX PRO 6000 correctness gate

On 2026-09-07, the current `cu/tp_ar.cu` one-shot primitive passed a bounded
native gate on two RTX PRO 6000 Blackwell Max-Q cards, driver 595.71.05. This
is a transport prerequisite for TP/EP, not model support, a CUDA-graph or
queued-hot-loop qualification, or a throughput measurement.

`tools/tp-ar-one-shot-gate.cu` is compiled against the unchanged engine CUDA
source. Both ranks produce the bitwise rank-0-plus-rank-1 sum out of place.
Each of 32 generations has fresh finite inputs, full input/output canaries,
input-immutability checks and zero barrier-error words. The tested payloads
are 4096 floats with one CTA and 24576 floats with 48 CTAs. A separate
missing-peer case refuses with error 40043 in 2 ms and leaves both inputs
and the entire output untouched. Memcheck and synccheck both report zero
errors; the native run also passes.

The gate deliberately synchronizes between generations. It does not prove
back-to-back producer/reduce reuse without host drains, graph replay,
end-to-end latency, or integration with DSV4 cache commit. Those need their
own receipts. No driver configuration or serving default changed.

## Pins

- Gate source SHA256: `c8d454f196d0b9e51d3e4c3b2e37552be89e8fb927246912ed95f7e31cd0e69b`.
- Engine CUDA source SHA256: `86c6be8d8b3ad54b197d45d1ef9f8c768803ef5394fd4123f795268c3143d809`.
- Executable SHA256: `c47b8fba9d4b1b7e2a7752a0f7595149fc8b51f0a07332a691531f393af327f3`.
- Native log SHA256: `9e0bddea0402a5fa64ad7b9e8513872e70a8739e5fc9df846be679f74613e1e7`.
- Memcheck and synccheck logs each SHA256: `90e40049b06ea5fc0241b8c50917283926d8af7ba033058a1682214a9673c641`.
- Receipt window: 2026-09-07T11:42:02Z through 11:42:08Z.

Build on the development host with CUDA 13.1:

```sh
nvcc -std=c++17 --expt-relaxed-constexpr -O3 \
  -gencode arch=compute_120a,code=sm_120a \
  tools/tp-ar-one-shot-gate.cu crates/memra-engine/cu/tp_ar.cu \
  -o tp-ar-one-shot-gate
```

Run only on an explicitly authorized non-production pair while holding its
shared GPU lock. This standalone executable does not own fleet scheduling.

Local CI was owner-waived on 2026-09-07 while the local machine was in use.
The source was independently reviewed and the concrete shape/initialization
findings were corrected before the remote gate. The native driver is not
built or run by the ordinary CPU-only Cargo suite.
