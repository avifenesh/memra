# GB10 prime: GDN traffic, fused K4+K5, fused norms and activation quantize, gate/up in one launch (2026-11-07)

**Status:** decided on one device class (unified-memory 120a family, DGX Spark): prime bench,
exactness battery and served A/B on both units. The levers keyed on the F8F4 v2 route (the
norm and gate/up fusions) reach only GB10. The fused K4+K5 and the tiled dp4a matvec change the
kernels' parallelism, so they are keyed on GB10 and wait for a discrete-card A/B. The rest (the GDN
compact layout at any dividing head ratio, the K2/K3/K5 grid order, the dropped f32 W store, the
q8_1 quantize-once scope) remove bytes with the same values on every card; their discrete-card
evidence is the RTX 5090's cell record (battery and perf A/B against the push base).

## Question

After F8F4 GEMM v2 and the h2p prime attention, a 131,051-token served prime on GB10 spent 157.6 s
in TTFT: 59.7 s in the F8F4 GEMMs, 40.8 s in attention, and about 50 s in the GDN chunk scan, the
GDN conv / l2 / gates, the e4m3 activation quantize, the norm/add glue and the beta/alpha matvecs.
Those stream kernels already ran at the device's DRAM rates (215-242 GB/s), and the GDN state pass
was bound by its own stores. Which bytes can go with the output unchanged?

## Decision

Exact rewrites, default ON, each with a seam (`docs/FLAGS.md` section 3):

- `MEMRA_GDN_TRAFFIC`: the compact q/k layout (`[T, num_k, 128]`, read as head `h % hk`) at any
  value:key head ratio that divides (it required num_v == 2 num_k; Qwen3.5/3.8 dense GDN has 16
  key and 48 value heads, so q and k were written, normed and read three times over); K2/K3/K5 run
  heads fastest so the value heads of one key head share its rows in L2; K3 no longer stores the
  f32 W that only the f32 K4 reads.
- `MEMRA_GDN_K45` (GB10): the state pass computes the chunk output (`gdn_chunk_state_output_mma`),
  so Y and the chunk-start states never reach memory.
- `MEMRA_PRIME_NORM_FUSE`: one kernel per residual add + RMSNorm site of the eager prime, which also
  writes the F8F4 v2 activation (and the q8_1 pair before a GDN mixer) for the GEMM group that
  reads the norm output; the dense FFN's f32 norm output is not written; the GDN out-norm writes
  the ssm_out GEMM's v2 activation the same way. Seeds are take-once, keyed by the buffer and
  cleared by a guard around the prime walk and by every non-seeding write.
- The q8_1 pair of a GEMM group's activation is quantized once (beta and alpha shared it and each
  quantized it); no seam, the F8F4 quantize-once precedent.
- `MEMRA_DP4A_TILED` (GB10): the NVFP4 dp4a matvec at prefill rows and under 128 outputs runs one
  block per four tokens over all output rows instead of one block per (row, token).
- `MEMRA_FFN_GATEUP_FUSE`: where gate, up and down take the F8F4 v2 tile, one launch runs gate and
  up over a paired weight image and writes silu(gate) * up as the down GEMM's v2 activation; the
  two [T, n_ff] f32 outputs are never written.

## Evidence

`research/gb10-prefill-next-20261107/` (README, LEVER-LEDGER): prime bench at 33,890 tokens, main
32.19 s -> 25.70 s with every cell's last-row prime logits identical; each lever measured on its
own; the fused K4+K5 probe (4.288 -> 1.703 ms per call at T 4,096, bit-identical); kernel-check bit
cells for every new kernel, run-gen argmax, run-spec, prime-path logits and the batched-prime gate
on both units.

Served cold TTFT, unit 2, B L L B B L, main `6f5e03e74a` against the lane (medians of three boots):

| prompt tokens | main | lane | change |
|---:|---:|---:|---:|
| 32,784 | 31.887 s | 25.382 s | -20.40% |
| 131,051 | 157.706 s | 131.689 s | -16.50% |
| 259,951 | 391.238 s | 340.886 s | -12.87% |

Every lane boot is faster than every main boot at every length, the answers are identical, and the
MemAvailable floor is 66.2 GiB or more in every lane boot. Unit 1, B L L B: -20.45% / -16.40% /
-12.85%, the same answers.

## Rejected

- K4 alone at 64 state columns per CTA (one wave of 96 CTAs, exact, 1.14x on the probe) and with
  the next chunk's U / gate / beta prefetched into registers (exact, 0.94x): not built. The probe
  showed K4 bound by its Y and Ssnap stores (1.1 ms of 2.4), which only the fused pass removes.
