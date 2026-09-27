# Qwen confidence science archive boundary

`receipts-v1/` publishes a scientific projection of the measured research
run: complete native prompt/output/round/timing/telemetry records, exact
measured harness source, and the buildable runtime source archive. The
external `manifest.sha256` pins `manifest.json`; the manifest pins each
archive and every expanded data and harness member. The projection
excludes model weights and this task's provider API records, pod
identifier and runtime credentials. The measured source snapshot
includes dated provider identifiers in separately reviewed source blobs.
A source recipe and its one-file fixed-C guard patch identify
the measured binary without publishing that binary.

The private publication scan checks all three compressed archives and
every expanded member against the same pinned public scanner and policy
that govern Memra. The current candidate has compressed-only matches
in two archives: provider-name byte coincidences in both the data and
runtime gzip streams, plus a fourteen-character pattern in the data gzip
stream
assembled by UTF-8 error skipping across non-text bytes. The latter
string is absent as contiguous raw bytes. The expanded scientific data
and harness members produced no findings. Expanded runtime source
members produced fourteen previously reviewed exact-blob matches under
other rules; their hashes and rule scopes are unchanged from the prior
source review. The private review pins the two affected archive digests
and each matching rule; it does not grant a blanket path exception.
Private workflow run `35808836373`, `confidence-scan` job
`107015610525`, passed at private head
`33501dc112e81cd2d0b7f1d8fe4e9a7532de44a1` with zero unapproved
matches.

The runtime source archive differs from the sealed prefix-study source
at `crates/memra-engine/src/spec.rs` only.
`verify_source.py` reconstructs and checks the exact guard substitution.
The measured phase analyzer is retained in `harness-source.tar.gz`.
Its public replay copy changes only two file-existence checks so that
numeric replay can use the public archives without the private executable
or its separately stored source copy; `reproduce.py` compares both source
texts and checks the measured analyzer hash before replaying raw records.

`reproduce.py` then re-audits every native request, the first failed
all-format qualification, the separate code-only qualifier, the fixed-C
grid, the offline policy and the post-result phase diagnostic. Hosted CI
compares its generated JSON and Markdown byte-for-byte with the
publication. Neither this archive nor replay qualifies a serving policy.
