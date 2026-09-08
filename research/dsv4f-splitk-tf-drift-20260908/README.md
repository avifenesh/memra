# Native B/E teacher-forced split-K drift instrument

Report-only incremental evidence, not quality certification or promotion.
`dsv4_tp_ep_tf_drift_gate <model> <tf.json> <source-bank.json> <new-output-dir>`
is a separate binary; the legacy CPU compatibility gate is unchanged.

One load runs B,E,B,E with fresh states. Both arms use the composed expert-ID
TP/EP, attention TP2, matrix MoE, RefFp8Round, native experts, FP8 dense, f32x
and small-kernel diet. Only adaptive split-K changes. The external pinned
32-token prompt and 160-token CPU tape drive every position. No generated
pick is fed back. Device sampling is configured but draws are deliberately
zero: teacher forcing does not qualify the sampler or sampled serving.

The first B/E rows are retained in host memory. Every repeated-arm logit is
compared bitwise and final cache/hidden/join digests repeat. Actual rank walks,
attention joins, diet and split-K submissions, finite logits, state positions
and zero refusal words are asserted. The existing rank-order GPU join is
compared to the f32 sum of the actual partials.

Each position reports max absolute/RMS/relative-L2 error (relative to B),
selected-token margins, top-1/5/8/20 overlap, bank-token log-probability delta,
and directional KL at temperature 1.0. Log-softmax is centered in f64 and
sums use Neumaier compensation. A zero B norm is represented as null, without
an arbitrary epsilon. Raw signed floating KL is retained, not clamped.
Three CPU tests cover shift invariance, analytic asymmetric KL and tie/NaN
behavior. These tests run remotely or in hosted CI, never on the owner's rig.

Historical CPU compatibility uses the existing native quantization band and
is distinctly report-only. Neither it nor within-arm determinism supplies an
incremental split-K quality threshold. No threshold has been introduced.

The pinned teacher bank is SHA256
`63a6e13500b3a3b843595f2b6f55e27d55ea299006a6c6ec085d3b845847946b`;
its source bank is
`920901870d3a0925d595c3d1f0190f5c9aa443426d074455a02660efbc3eb030`.
The instrument validates both hashes and their prompt/token/variant linkage.
The scope is one 0731 REF proof prompt, not a broad quality corpus.

Base main: `df1273928a7369acf8cc943131382cccbf2f9044`.
Compared with measured composition source `6a5bc257b20874da83080e4a861505ad420809bb`,
the DSV4 GPU, grouped, sampler, attention-TP, FFI and CUDA compute sources are
unchanged. The composition experiment driver differs; it is not reused or
modified. No prior throughput is transferred to this new binary.

Private operations own raw receipts and the target execution report. No
serving flag, fleet change, HTTP adaptation or promotion is included.
Pushes use MEMRA_SKIP_PERF_CI=1 under the owner's no-local-rig instruction.
