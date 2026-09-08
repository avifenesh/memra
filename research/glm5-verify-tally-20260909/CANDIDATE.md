# Candidate: exact batched E4M3 six-projection fusion

2026-09-09. Door `MEMRA_GLM5_VERIFY_E4M3_FUSED6`, default OFF,
decide-by: 2026-09-23. No deployment/default change.

The t4 p32k trace has 204 `qmatvec_e4m3_mmvq_b4` calls per round,
3811.757 us, streaming 3467.117 MB logical weight bytes. At 5600 GB/s,
the weight-only ideal is 619.128 us, leaving 3192.629 us of screening
headroom. This is the largest derivable weight-streaming saving in the
t4 table. The 7218.395 us gathered MLA attention row is larger in duration,
but its documented mechanism is serial online-softmax/barrier work on an
L2-resident selected cache, not an HBM-bandwidth target. Its already-known
warp-online numeric arm is not reopened by this cell.

Fuse the KDA q/k/v/f_a/g_a/b projections at t2..8 into one block-offset
batched MMVQ grid, sharing one Q8 quantization of the input and folding each
tensor's macro-scale into its final store. Use the current batched dot body,
which already reads each weight once for all t rows. Do not substitute F16
activations, change the weight layout, or change the per-row accumulation.
The existing t1 six-group is prior art; t1 and widths above 8 stay unchanged.

Per KDA layer: six quantizes + six matvecs + six scale launches become one
quantize + one fused matvec. Across 34 layers this removes 170 quantizes,
170 matvec launch boundaries and 204 scales, 544 launches in all. At the
observed t2 means, quantize and scale elimination alone estimates
170*1.869 + 204*1.635 = 651.270 us/round. At t4 it estimates
170*2.133 + 204*1.715 = 712.470 us/round. These are screening estimates using
family means; the complete-six ABBA cell, not this arithmetic, decides KEEP.
Additional matvec launch savings are not pre-credited. Register pressure,
block-offset dispatch and loss of small-call locality can eat the estimate.

Numeric class: byte-exact. Preserve each row's weight decode, activation
Q8 bytes and scales, per-block fmaf chain, and warp reduction. The final
scale uses an explicitly rounded multiply matching the separate scale
kernel's store boundary. Compare all six output tensors, every row, on
captured real KDA inputs at t2/4/7. Require finite values, zero differing f32
bits, identical row argmaxes and zero error band, with engagement counts.
Only after all three width oracles pass, warm each layer with 40 arm pairs
and run ABBA x5, 100 complete-six calls per position, synchronizing at
position boundaries. Report 34-layer totals and fixed conditional weights
48/162, 103/162, 11/162 from the prior component protocol. KEEP iff weighted
paired saving >=0.5 ms/round; otherwise remove the door, kernels, dispatch
and candidate-only harness in this lane and record NEGATIVE with rev:.
