# Source archive boundary

The simple-helper study used the buildable runtime source archive with SHA-256
`53cfab8e8a4c4a01362d54d9d96fd69100f99ab445b62ba81be1c096a17634bf`,
prepared from Memra `bdf9f4305b1093c4ac35a8e387c689047c938c5e`.
The source bytes remain unchanged so they can be matched to the measured native
binaries. The archive is historical measurement source, not current rental or
serving guidance.

A private expanded scan against the public policy from
`9717e8d57cf30bb2f9b2ae1e84994b6e719dedf5` found three new rule hits, all
in that runtime source archive:

- The compressed stream matched `provider_name_aws` at three deflate byte
  offsets. The nearby bytes are non-text binary data. The outer archive
  allowance is limited to its exact path, SHA-256 and that one rule.
- `crates/memra-engine/src/lib.rs:2907` matched `provider_bare_id` once in a
  dated 2026-09-10 development-pair receipt comment. The archived member
  SHA-256 is
  `a68609d31b1bc3d5f54d7db1598c460b89c01e5653a9501fbab7484b8ee281fc`.
- `docs/FLAGS.md:7`, `:945`, `:968` and `:1197` matched the same rule in dated
  development-pair receipt references. That archived member SHA-256 is
  `9612dc5e15c4d652a784031e4d8d1e0211df3fe28dcb214bc0683f32cb11b023`.

At the policy checkout named above, both live files had removed those bare
provider identifiers. The archived references are retained solely as exact
source provenance.
This study used one non-production Lium development pod; it did not rent or
reuse Vast or RunPod capacity. The expanded scan found no new rule hit in the
scientific data or harness archives; twelve other source hits matched prior
exact-file approvals.

Hosted private run `35786468365`, job `106944365854`, passed the expanded
scan with zero unapproved matches. Its report digest is
`28a45b7c60c7c372d0b455945cb16f8467ddd8afcd36ba433cf2eeeb64970e06`.

The scoped private review is
`darklanes/research/prompt-depth-router-20260922/prefix-publication-review-v2.json`.
It pins the policy, scanner, source member bytes, rules and compressed archive
bytes. Public CI must still enforce the outer archive boundary and reproduce
the measured report before any claim is accepted.
