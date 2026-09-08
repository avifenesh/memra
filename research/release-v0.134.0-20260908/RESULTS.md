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

The release changes workspace and internal pinned versions, Cargo.lock, the README
serving row, the release ledger, and this receipt namespace. No engine math,
serving default, published performance number, or fleet pin changes in this lane.

## Qualification

Pending on the authorized non-production RTX 5090, sm_120a. Every run will record
source HEAD, status, and binary hashes. No local rig build, test, gate, or server.

## Publication scope

Publicity: skipped (maintenance release). No HN, social, or blog publication.
No model-support or performance claims are added for the DSV4 experiment.
Owner review of the exact release PR head is required before merge or tag.
