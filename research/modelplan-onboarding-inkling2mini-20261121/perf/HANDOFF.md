# Lane P handoff: Inkling 2 Mini GPU route performance (2026-11-22)

Branch perf/inkling2mini-decode-20261121, draft PR #1399, worktree ~/.cache/inkling2mini/wt-perf, head
ef692d8e7f. The current landing is frozen at 38925f1ac5; everything after it goes to a follow-up landing.

## Where it stands

nvfp4-rig (the default class) on the RTX 5090 Laptop, 3 interleaved takes, no voids:

| | c=1 | 4k | 32k (single run, a18) |
|---|---|---|---|
| nvfp4-rig | 76.9 tok/s (p50 12.4 ms) | 69.8 | 27.3 (pre-GQA attention; re-time) |
| shipped | 66.5 | 57.5 (38925f1ac5) | 26.1 |

The c=1 floor is 3.58 ms a token (3.10 GB at 865 GB/s). One token at c=1 (s12-breakdown, s13): expert GEMM
4.2 ms (52% of DRAM), dense NVFP4 GEMV + unembed 2.6 ms (about 41%), BF16 GEMVs 0.6 ms, about 5.5 ms of
latency-bound small kernels, 1493 launches a token.

## The precision program (Prec::default in crates/memra-engine/src/inkling2_gpu.rs)

controller f32, mix f32, norm f32, conv f32c, router f32c, epilogue f32, attention f32t (f32 tiles of 256
keys merged in f64). Record: docs/decisions/INKLING2-PRECISION-ARMS.md (rounds 1-5). The gate for any
program change (coordinator ruling): teacher-forced dNLL not worse on rawdoc, chatwrap, L1, L2 (bootstrap
CI not entirely above 0; L1/L2 by 64-token blocks), short-context KL <= 1e-4 (p1, p2, the corpus documents),
p2 argmax equal, route flips per 100k reported per corpus. Single arms often fail where pairs pass: report
both. --prec spec parses on top of the default ("--prec all=f64" is the #1397 program;
"controller=f32,norm=f32,conv=f64,router=f64,attn=f64g" the round-1 program).

## Bit identity

Every kernel change must keep the digests. Commands (scratch under ~/.cache/inkling2mini/perf, P below):

- Gate: `$B/ink2-gpu-gate` (36 cells, ALL OK). Cells cover the multi-row GEMV, f32t vs f64, the adopted
  program, attention calls of 8 vs 1.
- Digests: `ink2-gpu MODEL --class CLS --tape TAPE --score --chunk N --out f.json` on shipped and
  nvfp4-rig at chunk 1, 4 and 512 (p1_chat for 1/4, p2_window for 512) and nvfp4-rig p3 at chunk 1024
  (tapes-var/p3.exact_f64.chunk1024.as_f32, --context 8448); compare base vs new with P/dcmp.py (prints
  differing attn slots, ffn slots, logits equal). Multi-row work: also chunk 2, 3, 5, 8.
- Pins of the program: perf/pin-mixepi (ef692d8e7f's default; digests at chunk 1, 4, 8, 512, p3).
- Lane H runs serve/land/batch_gate.sh (batched == solo, past-load-context sessions) on each head it carries.
- Lane S runs run-spec 28/28 and its verify-cost bench on each head.

## Running on the GPU

- Leases only: `gpulock run --ttl 45m --wait 700m --for "..." -- bash script.sh > out 2>&1`, queued from the
  REPL as a background task (`$\`...\`.nothrow().timeout(4000)`). Lease ids are request times, not grant times.
- Clock check at the start of every lease (P/lf5.sh shows the block): ceiling/clampcheck.sh; when it is not OK,
  `sudo -n systemctl reset-failed nvidia-powerd; sudo -n systemctl restart nvidia-powerd` and wait for it.
- Timed takes: P/cls2.sh NAME CLASS BASEBIN "BASEFLAGS" NEWBIN "NEWFLAGS" [PROMPTLEN] (3 interleaved
  takes, 250 ms telemetry, voids counted); summarize with P/summn.py NAME NAME.log.
- Long context: ink2-gpu --prompt-len 32768 --context 33792 --gen 16 (single runs; P/lw.sh).
- Profiles: nsys with --cuda-graph-trace=node, export sqlite, P/bk.py (per decode token, graphed) and
  P/rows3.py (per call by row count, --score --chunk R --eager). ncu needs sudo -n (P/lncu.sh).
- Build: P/build.sh (waitfor --capacity 20,8 then cargo at low weight, target ~/.cache/inkling2mini/target-perf);
  copy the two bins into P/aNN-bin with SHA256SUMS before a lease.

## Measured and rejected (deleted)

- Depth mix dot and output in one kernel (last block forms the outputs): 1.53 -> 2.22 ms a token.
- Expert GEMM streaming two 16-row tiles a warp through one ring: 4.24 -> 5.33 ms.
- Expert GEMM 4 K-steps a copy group (128 B a row): 4.24 -> 4.22 ms, flat.
- Multi-row GEMV in one pass of 5..8 rows (4 warps a block): slower than two passes of 4.
- Fold loop unroll in the BF16 jobs GEMV: slower (removed; kept in the NVFP4 and multi-row folds).
- Earlier rounds: controller norms formed once, f64 widening by field moves (perf/README.md).

## ncu findings (s14-rowjobs, s12-breakdown)

- Expert GEMM, decode groups: 8 one-warp blocks an SM (shared-memory bound, 16% occupancy), 90% of cycles
  no eligible warp, stalls long scoreboard 39% and MIO throttle 34%, 43-54% of DRAM.
- NVFP4 jobs GEMV: 2-3 blocks an SM (registers / shared memory), 31-36% of DRAM, issue 28-43% busy; the
  branch-free e2m1 decode already halved its instructions.

## Lane K (2026-11-22, after this handoff)

See README step 15 and docs/decisions/INKLING2-EXPERT-FRAGMENT-ORDER.md: the expert GEMM in fragment order (84% of
DRAM), the dense NVFP4 GEMV re-tiled (projections 67%, unembed 75%), a side-stream L2 prefetch (inside noise),
PDL and the smaller ring rejected. nvfp4-rig c=1 about 92 tok/s, shipped about 78. A grid barrier costs 0.58 us and
a graph node 0.40 us on this card.

## Open items, in order (lane P's list)

1. Smaller ring for decode expert groups (RG = RING instead of 2*RING for ONE): twice the warps an SM; one
   A/B.
2. The per-slot persistent kernel for the latency-bound stages: perf/PERSISTENT-SLOT-DESIGN.md (stage
   functions with today's mappings, single-block chains without barriers, about 10 grid barriers a slot;
   keep the GEMVs and the expert GEMMs as their own nodes first).
3. A TMA bulk-copy weight stream for the expert GEMM (and the dense GEMV), toward 80% of 865 GB/s.
4. The dense NVFP4 GEMV toward 80% (occupancy: registers and shared memory per block).
5. Prefill (32k prefill 110 tok/s on nvfp4-rig): the sgemm path and the grouped attention at long rows.

## Traps hit

- Never edit a .cu while a build runs: build.rs compiles every .cu early, and an edit lands or not by
  timing. Check the object's mtime (target-perf/release/build/memra-engine-*/out/*.o) against the edit.
- The REPL can lose its globals: re-define the helpers (L, W, CU, RS, pyPairs) at the top of a script.
- Corpus mode at chunk 512 and 1024 is bit-equal (the route is chunk-invariant): the second chunk size only
  matters against a CPU f64 tape (p3 alone has one).
- p1 is one 57-token sequence: its block-bootstrap CI has zero width; judge p1 by KL only.
- Public-boundary check scans binary blobs: per-token JSON receipts go in as .json.gz (gzip -9n), not .xz.
- KERNELS.md census: every new memra_* host entry needs its exact name in a row's Symbol cell.
- The pre-push hook needs MEMRA_RELEASE_QUALIFICATION_MODE=development; confirm pushes with git ls-remote.
- --force-with-lease with the exact remote SHA; never stash (stash is repo-wide).
- The batched decode path and new_seq sessions exist only on #1400 (lane H); keep serve code out of #1399.
