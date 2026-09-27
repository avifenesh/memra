# Reviewed identity dependency intake

This issue-local merge joins the frozen #541 disk-consumer candidate
`c3aaab083469d5cc2b102df6c8ed6779ad4dd407` with the reviewed #542 stack
`8c514a910405d340dfb96d409a04f1974f2133d9`, in a separate integration branch.
The prior #541 worktree and its uncommitted GGUF backing work remain preserved.
This is dependency intake, not main integration, bound-root activation or native qualification.

Conflicts preserve both contracts: source traits retain fallible/authorized access alongside the
opened-artifact digest; Step schema fixtures retain complete auxiliaries and vision inventory;
CI retains one CLI job, memory fixtures, and incoming identity controls; decision/research indexes
retain both lanes. Native NVFP4 disk repacking uses the reviewed private unlinked backing and exact
canonical reconstruction. Its per-expert reads now propagate the fallible source API, and macro
reads propagate errors. No size-only cache shortcut replaces the strict path.

Validation:

- `cargo test --locked -p memra-gguf -p memra-cli`: 354 GGUF pass, 2 ignored; 13 CLI,
  1 Step integration, and 7 inspector pass. Artifact-dependent skips are not native evidence.
- `cargo clippy --locked -p memra-gguf -p memra-cli --all-targets -- -D warnings`: pass.
- The existing actual-source identity/snapshot host harness: 18 pass.
- An actual-source host harness importing `model/repack.rs` and its existing tests: 9 pass,
  including private inode/cache lifecycle cases. No implementation was copied into the harness.
- Linux-target engine/server lib, bin and test check passes with `DOCS_RS=1` and documentation
  stubs. CUDA execution was not performed.
- Formatting passes. The net delta against the reviewed dependency passes `git diff --check`.
  A first-parent check reports whitespace in inherited raw logs and SSE traces. Every reported
  file is byte-identical to the reviewed dependency; hashes and the full diagnostic are retained.
  Their bytes were not trimmed, rewritten, or hidden by a whitespace override.

The bound adapter's composite identity and handle-free identity metadata are the next separate
change. Final source/native composition review, pending storage corrections, and remaining scoped
consumers still gate activation. Existing native Step vision remains unchanged.
