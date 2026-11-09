# DeepSeek-V4.1-Flash onboarding, step 5 (GPU, N ranks): the TP/EP partition, gated at N=2 (2026-11-09)

Verdict: the V4.1 layer now runs on N ranks (`crates/memra-engine/src/dsv41_gpu/tp.rs`, generic in
N): attention heads, output groups, `wo_b` rows, routed experts and head rows split across the
ranks; everything else replicated. Every cross-rank step is a gather and no sum is split, so the
N-rank program writes the one-card bytes. On two RTX PRO 6000 cards every checked tensor of six
real layers (Engram, ratio-2 source, ratio-1 source with the candidate pool, Reuse, Reindex) is
byte-identical to the one-card `block_forward` at prefill and at two decode steps, and the whole
trunk on two cards gives the one-card logits bit for bit at 16, 18,343, 73,520 and 128,745 tokens.
The same partition at N=4 (four ranks on one card) is byte-identical too. The 4-card stage changes N
and keeps the shares resident; the program stays the same. The support state stays as it is: this
is a multi-card surface, not a qualified serving path.

Hardware: two RTX PRO 6000 Blackwell Max-Q (sm_120a, 96 GB, 250 W cap), GPU0-GPU1 link NODE (PCIe
through the host bridge, same NUMA node), `nvidia-smi topo -p2p r` OK in every pair. CUDA 13.4,
rustc 1.97.1. All times are single runs, not timed for publication.

## The partition

`Part { rank, n }` gives each rank one contiguous range of each split dimension. `Part::check`
refuses a degree that breaks a law instead of forcing it: whole 16-head sparse-attention tiles per
rank, whole output groups, `wo_b` rows in 32-row scale blocks, whole experts, an even vocabulary.
For V4.1 (64 heads, 8 groups, 5,120 rows, 384 experts, 129,280 vocab) that admits N = 1, 2 and 4;
N = 8 would leave 8 heads per rank and is refused.

| tensor | per rank | why |
|---|---|---|
| `attn.wq_b` (64 heads x 512) | `64/N` heads | column parallel: a head's q is complete on its rank |
| `attn.attn_sink` | `64/N` heads | per head |
| `attn.wo_a` (8 groups of 8 heads) | `8/N` groups | a group reads only its 8 heads |
| `attn.wo_b` (5,120 x 8,192) | `5120/N` output rows | column parallel over the gathered `wo_a` output |
| routed experts (384) | `384/N` experts | expert parallel |
| `head.weight` | `129280/N` rows | vocab parallel |
| `wq_a`, `wkv` (1 KV head), norms, mHC, compressor, indexer, router, shared expert, Engram `wkv` | replicated | small (2.5 GB for the 40 layers), and every rank then holds the full stream and CSA2 state |
| `embed`, Engram tables | host | gathered per token on the host |

Per-rank device bytes of the 40-layer trunk (`rank_bytes.py` over the safetensors headers,
`receipts/rank-bytes.jsonl`): 298.6 GB at N=1, 150.6 GB at N=2, 76.5 GB (71.3 GiB) at N=4, of
which 2.5 GB is replicated. N=4 fits a 96 GB card with about 24 GiB left for the CSA2 state and
the workspace; N=2 does not fit, so the two-card runs below stream each layer's weights from the
page cache like the one-card runner.

## The collectives

Per layer, every rank runs the replicated steps on its own copy, then:

1. Indexer (prefill only): each rank selects for its `s/N` query rows (all 32 heads, replicated
   weights); all-gather the selection rows, and on the candidate source the candidate-pool rows.
   A decode row is scored on every rank, nothing moves.
2. Attention, per query slice: each rank runs its heads through sparse attention, inverse RoPE,
   the fp8 quantize-dequantize and its groups' `wo_a`; all-gather those `wo_a` output columns; each
   rank computes its `wo_b` output columns over the full row; all-gather the output columns.
3. MoE, per slice: the router runs on rank 0 and the host builds one dispatch plan (pairs sorted by
   expert, tokens ascending inside an expert, the per-expert loop's order). Each rank runs its
   experts' pairs through one grouped GEMM launch per projection (`memra_v41_gemm_f8_grouped`),
   all-gather the bf16 expert output rows, and every rank adds each token's rows in ascending
   expert id into an fp32 zero, then the replicated shared expert, then rounds
   (`memra_v41_moe_combine`: the per-expert scatter-add order).
4. Head: each rank's vocab rows of the fp32 head, gathered on the host.

Why bytes and not a band: the single-card gates bound summation-order effects (twice the vendor's
own CPU-vs-GPU spread). Here no summation crosses a rank. Every output element of `wo_b` sums its
K blocks on one rank in the one-card order, and every token's expert rows add in the one-card
order on every rank. The band is therefore zero, and the gates check byte identity. The vendor's
own TP code paths (`model.py` with `world_size > 1`) instead all-reduce fp32 partials after `wo_b`
and the MoE, and bf16 partials of the index score: that is a different summation order, which this
route does not take.

Bytes each rank receives per layer and token: `wo_a` columns 16 KiB x (N-1)/N, output columns
10 KiB x (N-1)/N, expert rows about 6 x 10 KiB x (N-1)/N (the share follows the routing),
selection rows 2 KiB x (N-1)/N at prefill; summed over the ranks, each term times N. A
3,761-token layer moved 331 to 341 MB in total at N=2.

## Transport

The route takes the fleet flag `MEMRA_TP_TRANSPORT` (`tp_transport.rs`) and follows its laws:
`peer-pull` copies on the consumer's stream after the producer's publication event, the producer
waits on a release event recorded after the copy before it reuses or frees the block, no atomics,
no peer flag polling. Peer access is granted with memra's `tp::grant_peer_access`
(`cuCtxEnablePeerAccess` and `cuMemPoolSetAccess`). Arming runs a byte-integrity ladder through the
exact gather primitive in every ordered pair (flat 4 KiB and 2 MiB, strided 33 rows of 5 KiB); one
differing byte refuses the group. `host-canonical` (the build's default) bounces every block
through host memory. `device-push` is refused on this route.

Both arms write identical bytes (gathers only). On this pair, `peer-pull` moved 1.35 GB blocks at
about 49 GB/s per direction (both directions at once: 98 GB/s summed), and a 3,761-token layer ran
in 0.03-0.04 s on `peer-pull` against 0.11-0.12 s on `host-canonical`. The 4-card runs set
`MEMRA_TP_TRANSPORT=peer-pull`; flipping the route's default needs the interleaved A/B the repo
asks for, not these single runs.

## Layer gate: N=2 on two cards against the one-card `block_forward`

`dsv41-gate tp CKPT captures/needle --layers 1,2,14,20,21,24 --devices 0,1 --decode 2`: each layer
from the vendor capture's inputs (stream, carried pre-mix, the CSA2 state it reads), the one-card
`block_forward` on GPU0 and the 2-rank program on GPU0+GPU1, then two decode steps (positions 3,761
and 3,762) from each side's own window ring, compressor pair and caches. Checked per rank and step:
the output stream, the output pre-mix, the attention output, the MoE output, the routing (ids and
weights), the CSA2 state the layer leaves (compressed KV, index keys, selection, candidate pool)
and the layer's own state (window ring, the open ratio-2 pair), plus the Engram output.

| run | checks | failed | rule |
|---|---|---|---|
| N=2, two cards, `peer-pull` (`receipts/tp2-needle-peer.*`) | 340 | 0 | byte-identical |
| N=2, two cards, `host-canonical` (`receipts/tp2-needle-host.*`) | 340 | 0 | byte-identical |
| N=4, four ranks on one card (`receipts/tp4-onecard-needle.*`) | 650 | 0 | byte-identical |
| N=1 (the grouped dispatch alone, `receipts/tp1-needle.*`) | 185 | 0 | byte-identical |
| N=2, two cards, 18,343 tokens, layers 20, 21, 24, 25, 28 (the candidate pool prunes; prefill only, the capture has no decode steps; `receipts/tp2-needle1200.*`) | 96 | 0 | byte-identical |

The N-rank stream against the vendor capture equals the one-card one at every layer (rel L2 4.12e-3,
7.41e-3, 7.71e-3, 5.02e-4, 3.35e-4 and 1.00e-4 for layers 1, 2, 14, 20, 21, 24, the numbers of the
one-card gate in `../gpu/README.md`), which follows from the byte identity. Four ranks on one card
prove the partition and the collectives at N=4 (ranks own disjoint ranges, each gather crosses
four streams), not the fabric.

Layer times at 3,761 tokens: one card 0.12-0.16 s, two cards 0.03-0.04 s. Most of that is the
grouped dispatch (the one-card loop launches and synchronizes per expert), not the second card.

## The whole trunk on two cards

`dsv41-run CKPT --ids FILE --devices 0,1` (MEMRA_FULL_PREC=1, `MEMRA_TP_TRANSPORT=peer-pull`): the
embedding on the host, each layer's two shares loaded in parallel from the safetensors mmap, run and
freed, the vocab-parallel head on the last token. Logits compared byte for byte with the one-card
runner's `--last-logits` files of the one-card step 4 (`../gpu/README.md`; sha256 of both sides in
`receipts/logits.sha256`) and scored against the vendor tapes (`trunk_vs_tape.py`, the one-card
gate's script). One-card times are those runs' (the capital one includes a cold page-in):

| prompt | tokens | logits vs one card | next token | top-5 | KL(vendor, two cards) | max abs diff | one card | two cards |
|---|---|---|---|---|---|---|---|---|
| capital | 16 | identical | equal (51119) | 5/5 | 2.86e-08 | 2.14 | 30.5 s | 14.3 s |
| needle1200 | 18,343 | identical | equal (17227) | 4/5 | 3.46e-06 | 2.53 | 41.7 s | 23.3 s |
| needle4800 | 73,520 | identical | equal (21391) | 4/5 | 6.02e-07 | 2.21 | 79.5 s | 49.1 s |
| needle8400 | 128,745 | identical | equal (20982) | 4/5 | 2.20e-04 | 2.99 | 120.8 s | 78.3 s |

The capital prompt also ran on `host-canonical`: identical logits. The one-card runner itself, after
the refactor this step made to the one-card layer (attention in keys / heads / `wo_b` phases, the
mHC helpers, row-sliced loaders), writes its earlier logits bit for bit on capital and needle1200
(`receipts/onecard-*.log`). The two-card logits are the
one-card runner's bytes, so the one-card CheckpointParity receipt on the capital probe (max abs
2.14 against the 6.4 band, `../gpu/README.md`) covers them; `verify checkpoint` itself was not rerun
with the two-card runner.

## The indexer at N ranks

`dsv41-gate idx CKPT CAPTURE --layer L --devices 0,1` on the real layer input (the capture's
stream through the collapse and `attn_norm`, the layer's own `qr` and index keys). Three arms:

- replicated: the model's indexer, all 32 heads, every row, on one rank (what each rank does when
  nothing is split);
- rows split: the same indexer with the query rows split across the ranks, then the selection rows
  gathered (2 KiB a row);
- heads split: each rank scores 16 of the 32 heads into fp32 partials (`memra_v41_index_score_part`),
  the partials gathered (4 bytes per row and key), added in rank order, rounded and masked
  (`memra_v41_index_score_finish`), then the top-k on every rank: the vendor's TP indexer shape.

Medians of 3 timed runs (5 for one row) after a warm-up, wall time with device syncs, `peer-pull`:

| layer | rows | keys | replicated | rows split | heads split (gather) | moved, rows / heads split | selection vs replicated |
|---|---|---|---|---|---|---|---|
| 20 (ratio 1) | 18,343 | 18,343 | 30.9 ms | 16.0 ms | 51.5 ms (27.3 ms) | 37.6 MB / 2.69 GB | rows: identical; heads: 18,343/18,343 rows same set |
| 20 (ratio 1) | 73,520 | 73,520 | 537.1 ms | 261.6 ms | 801.2 ms (437.4 ms) | 151 MB / 43.2 GB | rows: identical; heads: 73,520/73,520 same set |
| 20, one decode row | 1 | 18,343 | 0.21 ms | 0.22 ms | 0.25 ms | 2 KB / 147 KB | identical |
| 20, one decode row | 1 | 73,520 | 0.46 ms | 0.48 ms | 0.47 ms | 2 KB / 588 KB | identical |
| 2 (ratio 2) | 3,761 | 1,880 | 1.30 ms | 0.78 ms | 1.61 ms (0.60 ms) | 7.7 MB / 56.6 MB | rows: identical; heads: 3,761/3,761 same set |

The heads split loses at N=2: gathering the fp32 partials costs about as much as the whole
replicated score. At N=4 it gets worse. Each rank's 8 heads still fill one 16-row MMA tile, so its
score costs about the N=2 per-rank score, and each rank must receive 3 x rows x keys x 4 bytes
(64.9 GB at 73,520 tokens, at least 1.3 s at the measured 49 GB/s per link). The rows split costs
the replicated score over N plus a 2 KiB-per-row gather: projected 0.14 s at 73,520 tokens and
N=4, against 0.54 s replicated. The rows split is exact by construction (the same per-row
program), and the gate confirms it byte for byte; the heads split adds the head sum in another
order, which happened to select the same sets on every row measured here but is a different
program.

Decision for N=4: replicated indexer weights; at prefill, the query rows split across the ranks
with the selection (and candidate-pool) rows gathered; at decode, every rank scores its row with no
exchange. The N-rank prefill path does this now (step 1 of the collectives), and the layer and trunk
gates above ran with it.

Related precedent: the glm5 TP prime (`MEMRA_GLM5_TP_INDEXER_SPLIT_PRIME`) splits the key range
and merges exact top-k candidates; for V4.1 the query-row split needs no merge step and moves less
than the score.

## Left for the 4-card stage

- Full residency at N=4: hold all 40 layers' shares on the cards (76.5 GB per rank) instead of
  streaming them per layer, with the CSA2 state (compressed KV, index keys, window rings, candidate
  pool) sized for the target context inside the remaining ~24 GiB per card; then the same
  `dsv41-gate tp` and `dsv41-run --devices 0,1,2,3` gates on four cards.
- Engine and server integration: a `dsv41` engine entry over `tp::Group` (prefill then decode in
  one session, the decode loop that `dsv41-run` lacks, sampler on the gathered logits, request
  state per rank), the model pack's multi-card capability in the execution manifest, and the
  serve gates of the support record.
- CUDA graphs for decode: the per-layer gathers are event-ordered copies today (3 to 5 per layer,
  host-issued), with host round trips for the routing plan and the window index rows; a graph
  needs the routing plan built on the device and fixed-address buffers.
- Performance: the host thread issues ranks one after another and several steps drain a stream
  (Engram slices, the MoE plan); the expert-row gather could become a token-sharded combine (each
  rank combines s/N tokens, then gathers MoE output rows: about 2.4x fewer bytes at N=4, still
  exact); the GEMM tiles are the plain one-card tiles; nothing here is tuned or timed for
  publication.

## Commands

```bash
# per-layer gates: one card vs N ranks, prefill + 2 decode steps
MEMRA_TP_TRANSPORT=peer-pull dsv41-gate tp CKPT captures/needle --layers 1,2,14,20,21,24 \
  --devices 0,1 --decode 2 --token-map TOKEN_MAP --out tp2-needle-peer.json
dsv41-gate tp CKPT captures/needle --layers 1,2,14,20,21,24 --devices 0,0,0,0 --decode 2 \
  --token-map TOKEN_MAP --out tp4-onecard-needle.json
# the whole trunk on two cards
MEMRA_FULL_PREC=1 MEMRA_TP_TRANSPORT=peer-pull dsv41-run CKPT --ids needle1200.ids.i32 \
  --token-map TOKEN_MAP --devices 0,1 --last-logits trunk2-needle1200.f32
cmp trunk2-needle1200.f32 trunk-needle1200.f32   # the one-card runner's logits
python3 ../gpu/trunk_vs_tape.py TAPES/needle1200.pt trunk2-needle1200.f32 --json trunk2-needle1200.json
# the indexer choice
dsv41-gate idx CKPT captures/needle4800 --layer 20 --devices 0,1 --slice 2048 --reps 3
dsv41-gate idx CKPT captures/needle4800 --layer 20 --devices 0,1 --last 1 --reps 5
python3 rank_bytes.py CKPT 1 2 4
```

Every GPU step ran as one lease per run through `memra-gpu-run` holding both cards and both box
locks. Paths in the receipts are relative to the lane's run directory.

## Files

- `receipts/tp2-needle-{peer,host}.*`, `receipts/tp4-onecard-needle.*`, `receipts/tp1-needle.*`,
  `receipts/tp2-needle1200.*`: layer gates (JSON + log).
- `receipts/trunk2-*.{json,log}`: whole-trunk runs and their scores against the vendor tapes.
- `receipts/idx2-*.{json,log}`: the indexer arms.
- `receipts/onecard-*.log`: the refactored one-card runner against its earlier logits.
- `receipts/rank-bytes.jsonl`, `rank_bytes.py`: per-rank device bytes at N = 1, 2, 4.
