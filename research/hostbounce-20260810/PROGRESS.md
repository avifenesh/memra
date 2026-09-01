# Host-staged PP boundary fallback — progress

Date: 2026-08-10  
Branch: `lane/cx-hostbounce`  
Base: `3f8ca2ef`

## Scope

- Add opt-in `MEMRA_PP_HOST_BOUNCE=1` for PP cross-device transfers.
- Preserve the existing peer-copy path as the default.
- Allocate pinned staging storage once from runtime model geometry.
- Verify local compile/tests, Vast byte restoration and latency, and box1 default-path identity.
- Leave the Vast server running with sharded PP-2 plus host bounce enabled.

## Initial state

- Worktree clean at `3f8ca2ef` on the required dedicated branch.
- `~/.lanectl/inbox/cx-hostbounce.md` was absent at lane start.
- The referenced P2P receipt is on sibling worktree `wt-cx-p2pvast`: custom and NVIDIA
  probes both show successful API returns with wrong bytes on Vast; the live pre-fix server
  reproduced repeated-BOS output and is captured under `raw/vast-pre/` here.

## Live checklist

- [x] Inventory every PP-2 cross-device copy and any peer-read path.
- [x] Implement pinned D2H/event/H2D boundary fallback and unit coverage.
- [x] Local build and Cargo tests green.
- [x] Vast content, repeatability, short TTFT/decode, and 4k TTFT receipts captured.
- [x] box1 peer-default no-regression receipt captured under `/tmp/memra-gpu.lock`.
- [x] `RESULTS.md` complete with raw logs staged beside it.
- [x] Vast fixed-arm server left running.

## Work log

- 2026-08-10: Lane opened; instructions and branch/base checked. No engine changes yet.
- 2026-08-10: Central `PpNRt::tx/rx` boundary now selects default peer or opt-in
  host-staged transport. Two geometry-sized pinned slots are allocated once per cross
  boundary; Step-3.7 resolves to 64 MiB per slot from `4096 * n_embd * sizeof(f32)`.
- 2026-08-10: Peer-read audit closed unsafe side doors under the flag: sharding is
  mandatory; unsplit prime/decode paths refuse; serving spec, prefix snapshots, and
  affinity snapshots are disabled; stream-mode spec and Gemma4 PP reject explicitly.
- 2026-08-10: Local `cargo check`, focused host-bounce tests (3/3), server build, and
  full `cargo test --workspace` pass. The workspace run has zero failures; its two
  hardware-only tests remain explicitly ignored by their existing annotations.
- 2026-08-10: The first Vast live serve found a mapped peer read outside the activation
  boundary: Step-3.7 `rope_freqs.weight` was primary-only. It is now replicated once per
  distinct PP device and resolved through the stage-local engine.
- 2026-08-10: Vast transport smoke passed four alternating 1 MiB roundtrips with zero
  differing bytes. Exact hello, one-hash x3, matching short N=3, sustained decode N=3,
  and one 4k receipt all returned coherent content.
- 2026-08-10: box1 default-peer `decode-batch-gate` ran under `/tmp/memra-gpu.lock`;
  every B=1/2/4/8 split repetition and unsplit control was bit-identical, exit 0.
- 2026-08-10: Final Vast receipt confirms the sharded host-bounce server remains ready
  as PID 16734 with no current fatal-error match.
