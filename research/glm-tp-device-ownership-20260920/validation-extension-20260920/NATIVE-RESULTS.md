# Indexed MLA / KDA native allocation results

4/4 exact native stages passed at `fcb1b986b742a91cf960e5a6fea671779cce79c7`.
The specific indexed-MLA/KDA future-state allocation gap is closed for this source and
the recorded native test binaries. This is not qualification of a later main merge.

The run used RTX PRO 6000 Blackwell Server Edition hardware, sm_120, CUDA 13.1.115,
driver 595.58.03 and Rust 1.97.1. The indexed-KDA, paired and worker stages each held
exactly two physical GPUs through the verified per-card wrapper; same-device held one.
Each test ran alone in a fresh process and reported one executed pass, no failures and
no ignored tests. There were no timeouts or interruptions.

## Indexed-MLA/KDA result

The real four-layer F32 fixture contains two TP2-sharded KDA layers and two independent
MLA k-pool indexers. It has 106 tensors and 2,771,952 bytes, SHA-256
`f186baf6671362e3f288e1e057eac611bcdeb305a614e23f811c5e1ab49c1e14`.
Cache capacity is 8,192 and the physical index tail ring is instantiated.

| Phase | Actual allocated primary / peer bytes | Remaining primary / peer bytes |
|---|---|---|
| Cold prediction | 0 / 0 | 271,360 / 2,106,376 |
| One KDA layer and peer MLA base planes materialized | 135,680 / 1,839,624 | 135,680 / 266,752 |
| First peer key plane materialized | 135,680 / 1,905,160 | 135,680 / 201,216 |
| Second KDA layer materialized | 271,360 / 2,040,840 | 0 / 65,536 |
| Second peer key plane materialized — warm | 271,360 / 2,106,376 | 0 / 0 |
| Further indexer append | 271,360 / 2,106,376 | 0 / 0 |

At every phase the test asserted `allocated + remaining == cold` independently for each
rank, counting actual cache-owned `CudaSlice` extents and element sizes. The partial
key obligation was **65,536 bytes**, not zero. KDA allocation used `ensure_kda_tp_state`;
MLA replicas used `ensure_mla_peer_latent`; lazy keys were allocated by the production
`mla_kpool_indices` path, not assigned by a test-side zero buffer. This does not measure
allocator reservation granularity or peak transient workspace.

## Same-source regressions and identities

The original same-device owner, paired reclaim/refill and worker admission/pinned-source
stages all passed again on the same freshly built source. Pair reclaim released 64 MiB
with live allocation counts unchanged; the worker retained byte-identical pinned source
bytes through all three pressure/reclaim/refill cycles and restored eviction after unpin.

- Engine test binary: `d2fe8a6ac41c5465189a8ec4f289ebc33cdba2c46da4f68ed907a822bb46865f`.
- Server test binary: `2d3d678499dda6ac753336d04dfa137b53a0f476a554133772736e25b9939128`.
- Wrapper: `073518dc842cfa0ec5d1cfe5f3439f378e3e0b0cec29835fae21b28dd0764e55`.
- [Build receipt and compiler logs](native-pro6000/compilation/build.json).
- [Raw stage summary](native-pro6000/summary.json) and
  [canonical payload manifest](native-pro6000/files.sha256.json).
- [Final lease release](native-pro6000/lock-release.json): every wrapper exited zero,
  no lingering compute, no surviving task wrapper PID or owned FLOCK.

Raw stdout is stored losslessly as `test.log.gz`; decompress it before checking the
canonical hashes. All other payloads retain their canonical filenames. Every payload
was boundary-checked as plaintext where applicable. All receipts and both exact binaries
were copied off-box and verified. The private archive is retained independently of this
worktree. The historical `d413747d` records remain byte-identical and still describe their
original, non-indexed fixture.

## Remaining scope

The short append does not qualify ring wraparound or indexer numerical quality. Complete
HTTP/checkpoint behavior, actual CUDA OOM recovery, token-program identity and performance
remain unclaimed. The coordinator owns applicable release checks and the serialized merge
queue; these source/binary-bound results do not transfer to a subsequently merged binary.
No model support-state promotion or independent merge is asserted.
