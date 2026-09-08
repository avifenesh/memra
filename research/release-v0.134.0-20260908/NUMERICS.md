# Gemma reference fixture portability correction

The release validation exposed an old four-logit bit pin in
`dense_gemma_executes_scaled_parallel_residual_and_k_as_v`.

The unmodified v0.133.0 baseline (`9a447366877d0d5be74f8902de95815aecabb7d2`)
and candidate both fail on the same RTX 5090 host. The baseline fails in debug
and release modes, each with exactly one test executed. The earlier run with
zero tests and 30 filtered out is retained as `reference-baseline-vacuous.log.gz`
and is excluded from evidence.

Observed bits: `[3198203366, 1057194683, 3185247703, 3204119262]`.
Original pin: `[3198203366, 1057194687, 3185247713, 3204119266]`.
The differences are 0/4/10/4 ULP. This is not a release-introduced discrepancy.

## Decisive primitive control

On the SAME baseline test executable, the test fails with native libc `tanhf`
and passes the ORIGINAL bit pin when only `tanhf(x)` is interposed as
`(float)tanh((double)x)`. The control executes one test and reports 192 intercepted
calls. Its source is `tanh-control.c`; compile/run commands and SHA256 hashes of
source, shared object, and test binary are in `tanh-control-manifest.log.gz`.
The pass is in `reference-tanh-control.log.gz`. No runtime source changes or
model/backend substitutions are used by this diagnostic.

Rust documents `f32::tanh` precision as platform-dependent and currently delegated
to libc `tanhf`: https://doc.rust-lang.org/std/primitive.f32.html#method.tanh.
The control identifies the sensitive primitive. The original machine/compiler
that generated the pin is not known, and is not needed to establish that the pin
cannot be a portable exact-arithmetic contract.

## Regression-preserving correction

Only the existing test body changes. The general fixture still checks embedding
scale, absent global V weights, finite outputs, and sliding/global state windows.
A sparse fixture then has an independently hand-derived scalar program: token 0
occupies coordinate 0, layer 0 scales a zero-branch residual, and layer 1 routes
its K-as-V branch into coordinate 1 before its residual scale and output norm.
The two surviving logits and every zero logit are checked BIT-FOR-BIT, both before
and after softcap. No executor/norm/attention helper computes the expected values.

`black_box` keeps the expected tanhf call at runtime, matching the executor's
primitive boundary instead of allowing compile-time transcendental folding to
select a different implementation. No tolerance, accepted-bit list, runtime
arithmetic, or GPU byte-identity gate changes.

The exact cached value also proves K-as-V precedes the learned K norm. Red controls
change the first layer's scale, select an independent zero V projection, and remove
softcap; each must discriminate against the expected program. The corrected test
passes natively and under the alternate tanhf control (448 intercepted calls),
including all red controls. The full corrected model-plan suite passes with 341
passed, 12 declared artifact skips, zero failures; corrected workspace clippy passes.

## GPU receipt binding

The memra-engine tree and Cargo manifests/lock are unchanged from the versioned
GPU candidate. Every byte before the reference crate's `#[cfg(test)]` module is
identical, with SHA256 in `RECEIPTS.json`. These byte identities preserve the existing GPU receipt binding for the
test-only correction. GPU gates were run
with native runtime arithmetic; the tanhf interposer is only a CPU-test diagnostic.
