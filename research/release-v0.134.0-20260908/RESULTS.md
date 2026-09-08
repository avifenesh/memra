# v0.134.0 release candidate

Base: `df1273928a7369acf8cc943131382cccbf2f9044`.
Prior published tag: `v0.133.0`.
Version claimed atomically at the base before changing manifests.

## Complete change inventory

- #339, `f80553700`: DSV4 small-kernel diet. The only new environment door is
  `MEMRA_DSV4_SMALL_KERNEL_DIET`, default OFF. No default promotion or performance
  claim is made by this release. The original component/model receipts retain
  their two-rank scope; the release battery does not extend that scope.
- #353, `dd8cc9c74`: retry transient startup GPU-canary timeouts before latching
  a fault, with the existing deadline and unchanged steady-state behavior.
- #354, code `6adf6bbc9`, receipt `801a21ca9`, merge `df1273928`: refuse MTP verify
  capture for a non-resident or host-routed MoE MTP head and name eager verification.

The release also corrects one platform-dependent Gemma reference test pin; see
[NUMERICS.md](NUMERICS.md). Runtime arithmetic is unchanged.

The release changes workspace and internal pinned versions, Cargo.lock, the README
serving row, the release ledger, and this receipt namespace. No engine math,
serving default, published performance number, or fleet pin changes in this lane.

## Qualification

Versioned candidate `9d669417e58c59f02f3647a050c4707a0c7191e5` was built and
qualified on the authorized non-production RTX 5090, sm_120a, CUDA 13.0. Every
qualification log records source HEAD/status and the four release-gate binary hashes.

- Build: PASS; memra-server 0.134.0, source fingerprint
  `memra-0.134.0-1e3db796b617`, git SHA `9d669417e58c`.
- Full release battery: PASS, kernel-check 95 cells/21 skips; Ornith margin
  `flips=1 bad=0`, Qwen `flips=0 bad=0`, both K=1..8 self-consistency PASS.
- 10240 MiB squeeze: named eager verify fallback, K=1..8 PASS, exit 0.
- Engine suite: 446 passed, zero runtime skips (budget 0), zero failures.
- Server suite: 629 passed, zero failures, one ignored.
- Corrected model-plan/reference suite: 341 passed, 12 declared artifact skips
  (budget 12), zero failures across five binaries.
- Corrected Gemma exact oracle and wrong-semantics controls: PASS natively and
  with the isolated alternate-tanh diagnostic.
- DSV4 gate binary unit tests, formatting, release guard, and clippy: PASS.
- Raw model, binary, source, and log hashes are retained in the compressed logs
  and `RECEIPTS.json`. The initial ownership-refused guard and zero-test baseline
  run are preserved but excluded from qualification.

The initial local version/docs commit triggered the existing pre-commit formatting
check. That violated the no-local-gates instruction; no local CUDA, build, unit test,
battery, or server ran. All subsequent commits and validation/push hooks run on the
5090. The initial rsync ownership issue was corrected only in the isolated checkout;
the rebuilt binary reports the known Git SHA and the release guard passes.

The follow-up reference change is entirely inside `#[cfg(test)]`; its production
prefix, the engine tree, and versioned Cargo inputs match the GPU candidate exactly.
These byte identities preserve the existing GPU receipt binding. No performance claim is inferred
from correctness-only runs.

## Publication scope

Publicity: skipped (maintenance release). No HN, social, or blog publication.
No model-support or performance claims are added for the DSV4 experiment.
