Author's review of the full diff `main..lane/spill-integ71-20260927`, posted as a PR comment per the owner rule.

## What the diff is
- Lane B's work since #730, with main merged in the lane first (`c2c32539e`). About 7,700 lines across `memra-engine`
  (spec and prime, decode_batch, the step guard, grid capture, the kv-tier-gate grow mode), `memra-kv` (the VMM plane)
  and `memra-server` (worker, the VMM allocator, admission), plus `tools/health-fault-gate.sh`.
- Every new behavior is behind a default-off door with its FLAGS row and decide-by date: the exact resume, the spec
  budget clamp, the batch-OOM recovery, the W release, the admission reclaim and defer budget, the VMM KV allocator and
  its grow and fault seams, grid rewind and affinity.
- 223 MB of raw receipts, and one `.gitattributes` for the lane's receipt dir (whitespace exempt; HTTP captures `-text`
  so their CRLF bytes survive `core.autocrlf=input`).

## What I checked
- The in-lane merge: one textual conflict (the server's module list, both kept) and two adjacent hunks in `worker.rs`,
  both kept. Main's L', design Q, the purge scrub, R1 and P2 meet no lane code path otherwise.
- Doors off equals main's program: the continuation gate, the rewind probe (4 of 4 EXACT), prime-gate, run-spec K=1..8
  on two MTP models and run-gen on the 27B all pass on the merged tree, and every serving gate reads ALL GREEN.
- Doors on: the VMM allocator under every serving gate (ALL GREEN, `door=ON` in each log), the VMM grow series with
  bitwise tokens and logits, O14's arm j aimed at a 3-session chunk (and under VMM, with its reap line), the exact
  resume's E1 0 differences against cold on both routes, the W release through the admission burst gate.
- The fixture pin holds (`moe_cache.rs` untouched). No provider name, host, id, price or city in the lane's added lines.
- One substitution, recorded: cell 5's registered 35B draft is on no rig, box or repo, so run-spec ran on the 27B's
  and the 9B's own MTP heads (the same Qwen hybrid MTP walker, B's call).

## Batteries
- CPU battery 15 of 15 on the first tree and on the final head (server lib 985, engine lib 596).
- GPU battery on an RTX PRO 6000 (9950X host): integ70's cells, the same gates under the VMM allocator, and lane B's
  twelve cells, all green; serve-smoke's Q35 arm is main's own #777 in both serve-smoke runs.

**Hygiene:** no em dash in authored lines.

## Push regime
Engine and server source changed, so the branch goes up with `MEMRA_RELEASE_QUALIFICATION_MODE=development`. No tag.
Revuto: if capped or unavailable, this comment is the review.
