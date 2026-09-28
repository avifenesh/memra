# MiMo development bank

This branch banks the MiMo work when development stopped on 2026-09-28. It is
an archive, not a tested merge candidate or a serving build. MiMo support and
customer placement remain unqualified. Do not merge this branch to `main`
without integrating current `main` and running the gates that apply then.

The branch starts at PR #879 head `96b209023b771258303c2740bf8ccca73fc7d33b`.
The earlier source trunk, PR #855, is already in `main`. Every head in
`heads.tsv` and `historical-heads.tsv` is an ancestor of this branch, so its
exact code remains available through Git history after the draft PRs close.
Together they cover all 165 local MiMo branch heads present at banking time.

Normal merges retained the visible changes from PRs #879, #880, #884, #883,
#866, #865, #870, #863, #872, and #629, plus the local audio conv2 bias GEMM
branch. Two conflicts needed explicit joins:

- The #880 FFI merge keeps both selected-expert reuse declarations.
- The #872 KV merge keeps the batched local attention path and selects the
  global V decoder by format. Both kernel inventory sections remain.

Seven older diagnostics or alternate implementations conflicted with the
selected source tree. They were merged with Git's `ours` strategy. Their
commits are ancestors, while their source edits are not active in this tree.
`heads.tsv` marks them `history-only`. Inspect a variant with
`git show <sha>:<path>` before deciding how to reintroduce it.

The 42 older local branch heads in `historical-heads.tsv` were also merged as
history-only parents. That step kept the selected tree unchanged. No claim is
made that these alternate implementations work together.

This bank does not establish text, vision, audio, MTP, long-context,
concurrency, or served API support. The #879 current-base CI run was canceled
when development stopped. Historical two-card GPU receipts remain scoped to
the source heads and request shapes named in their PR bodies.
