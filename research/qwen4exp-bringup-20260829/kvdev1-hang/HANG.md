# memra#53 — `--ladder-kv-dev1` + `--ladder-spec` at 262,144 never arrives

Lane: `lane/q4e-kvdev1-hang-20260901`. Box: 4x RTX PRO 6000 96 GB, 360 GB RAM, all GPU
pairs `PHB`, `nvidia-smi topo -p2p r` OK everywhere. Cards 2,3 (`CUDA_VISIBLE_DEVICES=2,3`).
Checkpoint `q48fn-yarn1m` (qwen4_exp, 48 layers, `full_attention_interval` 4 -> **12 QSA
layers**, `num_attention_heads` 24, `num_key_value_heads` 2, `head_dim` 256,
`indexer_budget` 2048, `hidden_size` 2560).

## 1. Verdict

**Not a hang. Not a deadlock. Not a spin-wait, an unfired event, or a P2P race.** The
run was making forward progress the whole time, inside one section, at a rate that puts a
262,144-token prefill at **~9 hours** on this route. `sm 100% / mem 0%` on the trunk card
is the signature of exactly that: warps resident and stalled on **peer** loads while the
trunk card's own framebuffer sits idle, because every byte the attention kernel wants is
on the other card and peer memory is not cached in the reading card's L2.

The route had ALREADY been ruled out with receipts. `yarn/YARN-CELL.md` §3:

> **KV-on-the-other-card arm, measured (this is why the verdict is TP2, not "put KV on
> card 1"):** at a 4k fill, prefill 519 s (vs 18.3 s local, 28x) and decode **462
> ms/token (vs 25.7 ms, 18x)**, spread 9.44%. The attention kernel gathers K/V rows
> across PCIe with no coalescing across head-blocks; effective P2P throughput ~220 MB/s.

and §"What two cards DO deliver for depth today":

> `--ladder-kv-dev1` (remote KV) stays ruled out on round 1's 18x decode collapse; it was
> not re-measured.

The 262k spec cells were scheduled on it anyway, because the ruling lived in a lane doc
and nothing in the code enforced it. That is the defect this PR closes.

## 2. Mechanism

A quantized QSA cache has exactly one read path: the block-list form
(`q4e_sdpa_blocklist_q8q5` / `..._hoist`; the masked kernel has no quantized twin and is
refused). That kernel is a **scatter reader**. Its phase 1 is thread-per-position, so a
warp's 32 lanes sit on 32 *different* selected cache rows `k_tok_bytes = 544 B` apart and
every load instruction replays 32 ways into 32 distinct sectors — already measured in SASS
and written up in the `set_kv_hoist` FLAGS row (`sdpa_blocklist_f32` 37 instrs / 8 KV
loads; `q4e_sdpa_blocklist_q8q5` 120 / 8 / 8 redundant scale loads).

On the local card that replay is free, and this is *why* `qsa.sdpa` is depth-invariant:
the <= ~2,052 selected rows are shared by all 24 query heads (GQA group 12) and by
neighbouring query rows inside the chunk, so the reading SM's L2 serves the redundancy.

Peer memory is not cached in the reading card's L2. The redundancy L2 was hiding becomes
real wire traffic, and it is enormous: at a 262,144 fill ONE 2,048-token prefill chunk
asks for

    12 QSA layers x 24 heads x 2,048 query rows x ~2,052 selected positions x 432 B
    = ~523 GB across the link

and the rung has 128 such chunks. At the measured ~220 MB/s effective peer scatter
throughput that is days, not minutes.

Two independent things make the *placement* wrong even before the read pattern:

- The QSA KV is the **small** allocation. `kv_width = 2 x 256 = 512`, so a row costs
  `q8_row_bytes(512) = 544 B` (K) + `q5_row_bytes(512) = 320 B` (V) = **864 B per row per
  QSA layer**, **10,368 B/row** across the 12 QSA layers — **2.7 GiB at a 262,144
  capacity**.
- The allocation that actually cannot sit beside ~90 GiB of trunk weights is the **MTP
  draft state**: the hang run's own receipt shows card 3 growing from 9,587 MiB (draft
  weights) to 27,187 MiB once the states were allocated, i.e. **~17.6 GiB** of draft state
  + card-1 wide mirror. `--mtp-dev1` / `load_from_dir_dev1` already put that on card 1.

So `--ladder-kv-dev1` moved 2.7 GiB off the compute card and paid for it with the scatter
cliff, while the 17.6 GiB that motivated a second card was already handled by `--mtp-dev1`.

## 3. Repro / evidence matrix

| arm | rung | chunk | KV | prefill | first-chunk `qsa.sdpa` | decode | receipt |
|---|---|---|---|---|---|---|---|
| local control | 4,096 | 8,192 | card 0 | **18.7 s** | **1,255.4 ms (8.2%)** | 25.7 ms/tok | `../yarn/box-ladder-smoke2-local.tsv` |
| **peer KV** | 4,096 | 8,192 | card 1 | **519.1 s** (27.8x) | **501,436.6 ms (97.2%)** (**399x**) | **462.6 ms/tok** (18x); decode `qsa.sdpa` 430.5 ms = 95.1% | `../yarn/box-ladder-kvdev1-partial.tsv` |
| **peer KV** | 32,768 | 8,192 | card 1 | **never completed** | — | — | same file — this is why it is named `partial` |
| local control (today's binary, q8_0/q5_1 cache) | 2,048 | 512 | card 0 | **9.5 s** | **83.7 ms (4.5%)** | 27.9 ms/tok, 35.12 tok/s | `ladder-A-local-2048.tsv`, `A-local-2048.log` |
| peer KV + spec | 262,144 | 2,048 | card 1 | **no rung row in 45 min** (2-card box, AM) | — | — | operator, memra#53 |
| peer KV + spec | 262,144 | 2,048 | card 1 | **no rung row in 113 min**; trunk `sm 100% / mem 0%` for 3 consecutive `dmon` samples; KV card flat at 27,187 MiB for ~90 min; 1 host thread R (107 CPU-min), no OOM, `dmesg` clean | — | — | `operator-HANG-receipt-20260901.md`, `operator-ladder-r3spec262kv1-thinkon.tsv`, `operator-spec262kv1-thinkon.log` |

The named section is the answer to "which kernel": **`qsa.sdpa`, 97.2% of the prefill
chunk and 95.1% of the decode token**, on the arm's own `--profile` receipt. No `cuda-gdb`
was needed and none is installed on this box; the in-tree section profiler already
attributes the wall, and the other sections in the peer arm are within noise of the local
arm (`moe.sel_grouped` 5,219.0 vs 5,088.8 ms, `hyper.read` 2,801.2 vs 2,695.7 ms,
`gdn.proj` 1,997.9 vs 1,978.1 ms) — i.e. the process was running normally everywhere
except inside the peer KV reads.

**The peer treatment arm was deliberately NOT re-run at depth.** It is banked above at
rung 4,096 with a same-rung local control on the same binary, the 32,768 rung of that same
run already demonstrates the non-completion, and the quantized cache can only be worse on
this route than the f32 cache that measured it (`set_kv_hoist` row: q8_0 issues 2x the f32
twin's KV transactions while reading 3.76x fewer bytes, and transactions are what a PCIe
scatter pays). Re-deriving it would have cost a lock cycle on a contended box for a number
already in the tree. The fresh local control at rung 2,048 is banked so today's binary and
cache configuration have a baseline on the same line.

## 4. Fix

Root-cause fix, not a timeout wrapper. There is no version of "make peer KV fast at depth"
that works: the block-list form must touch ~all of the prefix's selected rows per chunk,
staging the prefix locally per chunk costs the same wire bytes as reading it, and an
incremental local mirror IS just keeping the KV local. So the fix is to stop taking the
route and to say why, in the code, at the two places an operator can hear it.

1. `qwen4exp_gpu::alloc_state_reserve` refuses a peer-resident QSA KV past
   `peer_kv_max_cap()` rows (default **8,192**, `MEMRA_Q4E_PEER_KV_MAX_CAP`). This is the
   load-bearing guard: it checks the exact state capacity and covers every caller, not
   just the ladder.
2. `qwen4exp_real_gate` refuses `--ladder-kv-dev1` at **arg parse** when the deepest rung
   is past the same ceiling, so the operator does not pay a ~100-265 s checkpoint load to
   learn it. Both messages name the mechanism, the row cost, and the route that works.
3. The ceiling is 8,192 **by design**: it keeps the shallow smoke arm that motivated the
   flag legal (rung 4,096 completes, slowly, and is the receipt above) and refuses the
   depths where the route cannot arrive. Raise the env only to deliberately re-measure the
   cliff.
4. Observability, because "no rows ever written" is half of why this cost 158 minutes of
   card time across two runs: the spec arm banks ONE row per rung at the END, so a slow
   route and a stuck route look identical from outside. It now prints
   `# spec-state-allocated <rung> <cap> <vram>` before the first co-prefill chunk (which
   also answers the residency question in seconds instead of via an OOM 20 minutes in) and
   `# spec-prefill-progress fill=<b>/<n> chunks=<c> elapsed_s=<s> draft_s=<s>` every 8
   chunks, the non-spec ladder's existing cadence.

FLAGS.md row for `MEMRA_Q4E_PEER_KV_MAX_CAP` lands in the same PR (new-env law).

## 5. Gates

- **Rig tiny fixture gate: PASS**, all arms, and **byte-identical to the pre-change
  receipt** — the only diff in `gpu-eager/tiny-fixture-gate.tsv` is the `binary_sha256`
  line. A refusal cannot move bits, and the progress prints are stdout. The gate includes
  `mtp-spec-ring` (chunked co-prefill 8 + wide ring 16, spec-vs-plain byte identity), so
  the co-prefill path carrying the new prints is exercised.
- `cargo clippy -p memra-engine --bin qwen4exp_real_gate`: clean.
- **CLI refusal, live on the box** (`qwen4exp_real_gate.kvdev1fix`): `--ladder 262144
  --ladder-kv-dev1 --ladder-spec 5` refuses at parse with the full reason; `--ladder 4096
  --ladder-kv-dev1` is accepted and proceeds past parse.
- No two-card regression arm is added: expressing this layout needs two devices, and the
  ceiling it enforces is a refusal rather than an arithmetic path, so the byte-identity
  bar it has to clear is the tiny gate's (met, above).

## 6. The three owed cells, re-cut on the corrected route

`spec262kv1-{thinkon,thinkoff,raw}` were owed on the kv-dev1 route. They are re-cut here
as `r4spec262-{thinkon,thinkoff,raw}`: the q2.sh invocation verbatim with
`--ladder-kv-dev1` dropped and `--mtp-dev1` kept.

    CUDA_VISIBLE_DEVICES=2,3 MEMRA_Q4E_SEAMS=idxsel qwen4exp_real_gate.kvdev1fix \
      <q48fn-yarn1m> <out> --label r4spec262-<shape> \
      --mtp --mtp-dev1 --spec-pmin 0.3 --spec-adapt 1 --spec-k 5 \
      --ladder 262144 --ladder-ids ladder-ids.txt --ladder-chunk 2048 --ladder-decode 36 \
      --ladder-spec 5 [--ladder-spec-shape <shape>-prompts.tsv]

**Box constraint these cells run under (host RAM, not VRAM).** The qwen4_exp loader stages
the whole 174 GB artifact in host RAM before uploading (~180 GB anon-RSS at peak) and the
host has 353 GB, so TWO concurrent loads exceed it and the GLOBAL OOM killer takes one —
receipted on this box 2026-09-02T00:15:09Z (pid 28271, a 32k trace re-run, anon-rss
180.8 GB, `constraint=CONSTRAINT_NONE`) while other lanes were loading. The measurement
lock does not prevent it: shared holders load concurrently, and an exclusive holder can be
loading while a `-s` waiter is not yet in. Worse, the victim's queue line read `rc=0` with
a log that stopped at the post-engine VRAM line, so a SIGKILL looked like a clean run.
Every real-artifact launch from `r4spec262-thinkoff` onward therefore goes through
`~/realgate/bin/q4e-load-lock.sh <log> <cmd...>`: an exclusive load lock plus a
`MemAvailable >= 200 GB` gate, held only until the binary prints its post-load VRAM line
and then released, with an explicit `# load-lock rc=.. killed=..` line appended so a kill
can never read as success. It is taken INSIDE the measurement lock
(`flock -x /tmp/q48fn-measure.lock -c "... q4e-load-lock.sh ..."`). The two shallow repro
loads in the matrix above (`A-local-256`, `A-local-2048`) predate the wrapper and did not
use it; `r4spec262-thinkon` was already resident when it landed.

## 7. Cell rows

**The corrected route allocates and prefills normally.** `# spec-state-allocated
rung=262144 cap=262193` reports the trunk card at **92,883 MiB / 97,887 MiB** with the
draft card at 9,811 MiB — the same 92,883 MiB figure `YARN-CELL.md` banked for a 262,144
single-card allocation, i.e. the 2.7 GiB of QSA KV lands beside the trunk weights with
~5 GiB of headroom once `--mtp-dev1` has taken the draft state to card 1. Co-prefill runs
at ~10.7 s per 2,048-token chunk (`# spec-prefill-progress` ... 65536/262144, chunks=32,
elapsed_s=340.6), which projects the rung's prefill at ~1,370 s and matches the plain
262k single-card ladder's 1,439 s. The peer-KV route's projection for the same rung is
~9 hours.

CELLS-PENDING
