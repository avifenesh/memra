# glm5 TP TRANSPORT-TRUTH lane (lane/glm5-tp-transport, 2026-09-01)

Owner bar: **GLM-5.3-Flash does not serve under 100 tok/s single-stream in any scenario**
(`LAW:glm53-100toks-serving-bar`). Ship today is 62.43 tok/s on the PP-3 shape
(`matvec-20260831/LANE.md`), and `100/62.4 = 1.60x` is the gap.

This lane exists because the deep-research pass
(darklanes `research/pro6000-multicard-research-20260901/RESEARCH.md`, on `origin/main`)
found that our banked `VERDICT:glm53:tp2-v1-bare-does-not-pay` was measured on a transport we
chose for CORRECTNESS and never intended to serve — and that the world's receipts on our exact
model and card class put TP-4 no-spec at **139.4 tok/s** (`RESEARCH.md` §1.3c) against our
PP-3 served 35.36 at the time of that battery.

Base: `origin/lane/glm53-flash-bringup` @ `92ea07376`. Worktree `~/projects/wt-glm5-tp-transport`.
Branch `lane/glm5-tp-transport`. Pushed; no self-merge.

---

## Stage 1a — THE RECEIPT-CORRECTION NOTE

> **No banked glm5 TP number has ever measured fabric P2P. Not one. The v1 transport is a
> host bounce with a full stream drain per leg, by design and by our own written decision —
> and the code confirms it with no ambiguity left.**

### The code receipt (what transport every banked glm5 TP number used)

Every cross-rank hop in the armed `MEMRA_GLM5_TP` seam was, before this lane, an inline
`Engine::dtoh` -> host `Vec<f32>` -> `Engine::htod` pair. Enumerated at HEAD `92ea07376`:

| site | file:line (pre-lane) | hops |
|---|---|---|
| KDA walk `kda_tp_cached` | `glm5_tp.rs:762-763` (x fan-out), `:816-826` (gated gather), `:832-841` (wo concat) | 5 draining `dtoh` + 4 `htod` per layer-call |
| MLA walk `mla_tp_attn_cached` | `hybrid_forward.rs:7383-7386` (h + pos fan-out), `:7416-7428` (head gather), `:7433-7442` (wo concat) | 6 draining `dtoh` + 5 `htod` per layer-call |
| MoE EP v1 sequential | `hybrid_forward.rs:10154` (z stage), `:10160` (per-token row), `:10243-10244` (per-slot row return) | 1 + one-per-peer-slot draining `dtoh`, one-per-token + one-per-peer-slot `htod` |
| MoE EP diet, swap point 1 | `hybrid_forward.rs:10422-10423` | 1 draining `dtoh` + 1 `htod` per layer-call |
| MoE EP diet, swap point 2 | `hybrid_forward.rs:10479-10480` | 1 draining `dtoh` + 1 `htod` per layer-call |
| MoE EP grouped prime | `hybrid_forward.rs:10748-10749`, `:10770-10771` | same two shapes |
| Load-time weight staging | `glm5_tp.rs:490` `shard_rows`, `:590` `replicate`, `:649/:657` conv | `dtoh` -> slice -> `htod`, per tensor |

Zero occurrences of `memcpy_peer_async`, `memcpy_dtod`, `cuDeviceCanAccessPeer`,
`cuCtxEnablePeerAccess` or any peer kernel in `glm5_tp.rs` or in the glm5 walk of
`hybrid_forward.rs`. `configure_native_p2p` appeared only inside a doc comment
(`hybrid_forward.rs:10285`: *"the glm5 seam inherits `configure_native_p2p` but does NOT wire
it on the rig"*) and had exactly **one** caller in the whole workspace, `tp.rs:1661` — the
STEP seam, not glm5.

### The written receipts that already said so, and were right

- `tp2-20260831/LANE.md:98`: *"v1 transport is host-canonical staging (the step seam's
  correctness transport). NOT built, deliberately: native P2P engagement (machinery inherited:
  `configure_native_p2p` ladder)"*.
- `tp2-battery-20260831/RESULTS.md` cell 2: *"v1 transport = host-canonical (every boot
  announces `transport=host-canonical` on all four seams). Native P2P is NOT wired to the glm5
  seam (LANE stage-3 decision)"*.
- `docs/FLAGS.md`, the `MEMRA_GLM5_TP` row: *"v1 transport is host-canonical staging — a
  correctness program, NEVER a serving or performance claim (`performance_claim=false` on
  every `[glm5-tp...]` marker)"*.

So the verdict's scope line carried all the weight, exactly as `RESEARCH.md`'s correction #1
says. **`VERDICT:glm53:tp2-v1-bare-does-not-pay` must never be read as "TP loses to PP on
PCIe", and it must never be read as "PCIe P2P is too slow for our TP shape". It was never a
measurement of either.**

### FIVE corrections this lane forces, three of them on OUR docs

1. **`SHARD-MAP.md:25` says the v1 hops "bounce through pinned host buffers". They do not.**
   `Engine::dtoh` (`lib.rs:9770`) is `stream().clone_dtoh(d)` into a fresh **pageable**
   `Vec<f32>` followed by `stream().synchronize()`; `Engine::htod` (`lib.rs:9748`) is
   `clone_htod` into a fresh `CudaSlice`. There is no pinned staging anywhere in the glm5 seam
   and no persistent staging buffer at all — `Glm5TpRt` carried only `peer: Engine`, two
   ordinals and a bool. `pp.rs` has real pinned staging (`PinnedHostBounce`, `pp.rs:1240`);
   glm5 never touched it. So the v1 arm paid **pageable DMA bandwidth AND a full device
   synchronize per leg**, which is worse than the doc's own description of it. Corrected in
   this lane's transport module doc; `SHARD-MAP.md` should be amended when that lane's owner
   next touches it (flagged, not silently edited — it is another lane's receipt).
2. **`glm5_tp.rs:52-53` said TP-3 is refused because "64 attention heads and the 32-head
   indexer do not divide by 3". The indexer clause is wrong.** The DSA indexer is
   REPLICATED per rank by this seam's own shard map (`shard_mla_layer`'s `replicate_indexer`),
   so `index_n_heads = 32` imposes no divisibility constraint at any rank count. Fixed in
   place this lane. The consequence matters: see "TP-4 divisibility" below.
3. **`RESEARCH.md` shortlist item 1's FIRST bullet does not apply to us at all.** The
   `NCCL_P2P_LEVEL` finding (upstream NCCL grants AMD-host P2P only at <=2 GPUs, selecting
   `SHM/direct/direct` = device->host->device at 3-4 GPUs) is a real and important finding —
   **for stacks that link NCCL. Memra links none.** Two greps: `nccl` appears twice in all of
   `crates/`, both times inside a `pp.rs` comment comparing our own `cudaMemcpyPeerAsync`
   against it. Our host-canonical transport was OUR choice, recorded in three places before
   the research existed. There is nothing to fix here and no `+24-42%` to collect from an env
   var. **The lead finding of the research does not explain our result; our own design
   decision does.** (This is the good outcome: no hidden config defect, and the fix is ours.)
4. **`RESEARCH.md` §2.6's byte estimate for our walk is low by ~10-40x, and the correction
   strengthens its conclusion.** §2.6 estimated *"4096 x 2 B = 8 KiB/layer = 0.35 MiB/token;
   charging mHC 4 streams gives 1.41 MiB/token"* and derived an effective **0.02-0.11 GB/s**.
   Both inputs are wrong for our program: our TP activations are **f32 (4 B), not bf16**, our
   join is a **column-parallel-over-gather** that materialises the FULL tensor on BOTH ranks
   (so it moves ~2x full width, not one all-reduce payload), a host bounce crosses PCIe
   **twice** per hop, and **mHC's 4 residual streams are root-owned** — they never cross a
   rank boundary, so there is no `x4`. Measured-from-code census below: **14.63 MiB/token**,
   i.e. an effective **0.81-1.13 GB/s**. Still **47-65x below** the 53 GB/s this card
   sustains, so §2.6's verdict ("counts, not bytes") holds — on better numbers.
5. **`RESEARCH.md` §2.2b's pull-vs-push 20x is scoped to SM-ISSUED traffic, not to us.**
   b12x#139's *"52 GB/s for peer reads and 2.6 GB/s for peer writes"* and FlashInfer#4393's
   write collapse are kernel `load`/`store` measurements. Copy-engine `cuMemcpyDtoDAsync`
   measures ~54-56 GB/s in BOTH directions (§2.2's `p2pBandwidthLatencyTest` "P2P Writes"
   rows; §3.2's `nvbandwidth` `read_ce`/`write_ce` matrices agree at 54.0-54.3 GB/s). We use
   the copy engine. **We still built pull-only** — for the ordering reason (below), and so
   that the day a fused collective is built here the shape is already right. Stated so nobody
   later cites "20x" as this lane's justification and gets caught by a reviewer.

### And one correction on our own topology law

`tp2-20260831/LANE.md:9-11` bans TP-3/TP-4 on this host class by citing
`docs/decisions/PRO6000-MULTICARD.md` (*"the box topology law is TP-2 inside a NUMA pair,
PP-2 across pairs, never TP-3/4 on this host class"*). `RESEARCH.md` §1.3 refutes the
premise directly: on two different PRO 6000 hosts **TP4 beat TP2 in every measured decode and
prefill cell** (direct-root server C1 149.8 vs 130.4; switch workstation 182.6 vs 150.4), and
the mechanism is *"TP adds a second memory system"* (§1.5c) — a MEMORY-bandwidth argument, not
a communication one, so it does not weaken with card count the way "shard narrower to cut PCIe
traffic" assumed. The topology law needs re-deriving with TP-4 in scope; that is a named
box-window item, not a code change, and it is listed under "Named for the box window" below.

---

## Stage 1b — THE MOVEMENT CENSUS (zero GPU-hours, derived from code + pinned config)

Geometry, from `research/glm53-flash-bringup-20260827/glm-config.json` (not from memory):
`hidden_size 4096`, `num_hidden_layers 45`, `linear_attn_config.num_heads 64` x
`head_dim 128` (34 KDA layers), `num_attention_heads 64` x `v_head_dim 256` (11 full-attn/MLA
layers), `first_k_dense_replace 3` (so 42 routed-MoE layers), `n_routed_experts 288`,
`num_experts_per_tok 8`, `index_n_heads 32` (REPLICATED), `hc_mult 4` (root-owned),
`num_nextn_predict_layers 1`. TP activations are **f32**.

Derived hop widths at TP-2: KDA gather half `8192/2 = 4096` elems, full 8192; MLA gather half
`64*256/2 = 8192` elems, full 16384; `wo` output half `4096/2 = 2048` elems.

### Per-token census, DECODE (t=1), host-canonical, EP diet OFF (= the banked v1 walk)

| layer class | calls | host legs / call | draining syncs / call | crossed bytes / call |
|---|---:|---:|---:|---:|
| KDA (fan-out 32 KiB + gather 96 KiB + concat 32 KiB) | 34 | 9 | 5 | 160 KiB |
| MLA (fan-out 32 KiB + gather 192 KiB + concat 32 KiB) | 11 | 11 | 6 | 256 KiB |
| MoE EP v1 (z 32 KiB + 4 peer slots x 32 KiB) | 42 | 10 | 5 | 160 KiB |
| **per token** | **87** | **847** | **446** | **14.63 MiB** |

The MoE row charges the expected peer-slot count, `E[peer slots] = 4` of top-8 under the even
split (`SHARD-MAP.md` §3's own arithmetic: the peer is touched on ~99.3% of layer-tokens).

### What that census reconstructs, and it is the whole tax

```
measured v1 residual join+dispatch tax   13-18 ms/token   (tp2-battery RESULTS.md cell 3)
draining host syncs per token                       446   (census above, from code)
=> implied cost per draining sync         29.1-40.4 us
```

**29-40 us per host-staged round trip is exactly the class `RESEARCH.md` §2.6 predicts
(20-40 us) and exactly the P2P-disabled peer latency §2.2 measures (14.15-14.44 us one way,
so ~30 us for a drain + round trip).** The tax reconstructs from COUNTS, with a residual of
zero. Bytes contribute `14.63 MiB / 53 GB/s = 0.276 ms/token` — **2% of the measured tax.**

Independent cross-check on the diet arm: with `MEMRA_GLM5_EP_DIET=1` the MoE row falls to 4
legs / 2 syncs per layer-call, giving **320 syncs/token**. `ep-diet-20260831/LANE.md:34`
independently states the diet folds *"the v1 per-slot round-trip dribble (~170-210/token)"*
into *"42 bulk returns/token"* — our census puts the v1 MoE per-slot syncs at `42 x 4 = 168`
(or 181 at the doc's 0.54 peer share). The two derivations agree to within the peer-share
assumption, from opposite directions.

### The all-reduce-size-histogram equivalent for our joins

`RESEARCH.md` §3.5 asks for our collective size histogram, because *"any knob whose active
band sits above 512 KB cannot touch our decode"*. Our joins are not all-reduces, so the
equivalent is the per-hop payload distribution. At TP-2 decode, every hop is one of exactly
**five** sizes:

| payload | hop | count / token |
|---:|---|---:|
| 4 B | MLA position fan-out | 11 |
| 8 KiB | `wo` half return (2048 f32) | 45 |
| 16 KiB | mixer-input fan-out, KDA gather half, MoE z, MoE expert row | 34 + 34 + 42 + 168 = 278 |
| 32 KiB | KDA full-width gather delivery, MLA gather half | 68 + 22 = 90 |
| 64 KiB | MLA full-width gather delivery | 22 |

**100% of our traffic is 4 B .. 64 KiB, and 63% of it is exactly 16 KiB.** Consequences,
stated so no future lane re-derives them:

- Every published PCIe-collective win in the 4-64 KiB band applies to our shape
  (`RESEARCH.md` §6.1: NCCL 45 -> 7.3 us at 4 KB, 344.5 -> 38.5 us at 64 KB on 8x SM120;
  §6.2: one-shot beats NCCL from 1 KB to ~512 KB on 4x PRO 6000).
- Every knob whose band starts above 512 KB is dead for us by construction: the b12x DMA-ring
  tier (>= 6-24 MiB, §6.12), FP8 wire compression (§6.10 — at 16 KiB, halving bytes saves
  0.15 us against a 6-14 us latency floor), the NCCL tuner plugin (+50% at 2 MB, *"zero
  measurable impact on decode throughput"*), Tree-vs-Ring (crossover >= 32 MB).
- The 16-byte per-rank stride alignment finding (§2.5, worth 1.5-2.8x) is a NON-issue for
  every hop above: 8 KiB / 16 KiB / 32 KiB / 64 KiB are all `0 mod 16`. The 4 B position hop
  is not, and it is 11 copies of one integer per token — noted, not fixed. **The shape §2.5
  warns about (variable-tokens-per-expert x MoE dim) does not exist in our transport: our EP
  moves whole `n_embd`-wide rows, never a ragged per-expert payload.**

### The `tensor.copy_()` latency-instrument trap, recorded before anyone measures with it

`RESEARCH.md` §2.10, verbatim: a CPU-driven `tensor.copy_()` ping-pong measured
**10.36-10.86 us** on links whose true hardware latency is **633-727 ns**, because *"each
round-trip includes two CUDA stream launches, each ~5 us of CPU-side overhead, which utterly
dominates the actual hardware latency. ... No conclusion can be drawn about the fabric from
this test."* Our exact exposure: this lane's peer-pull hop **is** a stream-launched copy, so a
naive per-hop timing harness would measure ~5 us of CPU issue and report it as fabric latency,
concluding "peer-pull is only 6x better than a 30 us host sync" when the fabric part is
~0.4 us and the 5 us is ours to remove with graph capture. **Any latency number for this
transport comes from a GPU-resident flag ping-pong kernel (1,000,000 rounds x 3, +-1 ns per
§2.10) or from an end-to-end tok/s A/B — never from timing a `memcpy_dtod` from the host.**
Corollary for the box window: the peer-pull arm's win is read from **tok/s and the movement
census**, not from a microbench of the primitive.

---

## Stage 2 — NATIVE-P2P TRANSPORT (built)

New module `crates/memra-engine/src/glm5_tp_transport.rs`. The design in one paragraph: the
glm5 TP program's cross-rank hops are PURE MOVEMENT (the single arithmetic site, the MoE
slot-ordered `fmaf` combine, stays on root), so the transport is swappable without changing a
bit. The module names the four hop SHAPES that cover every crossing, and puts both arms behind
each one. Nothing else in the walk knows which arm is live.

### The four hop shapes

| shape | used by | host-canonical | peer-pull |
|---|---|---|---|
| `fanout_f32` / `fanout_i32` | mixer input `x`/`h`, MLA positions, MoE `z` | 1 draining `dtoh` + 1 `htod` | 1 peer read + 4 event primitives (publish + release) |
| `gather_halves` | KDA gated halves, MLA head halves -> FULL tensor on BOTH ranks | 2 draining `dtoh` + host interleave + 2 `htod` | 2 peer reads + 8 event primitives + 2 local copies (`t=1`); + 4 `place_rows_strided` (`t>1`) |
| `concat_halves_on_root` | the two column-`wo` halves -> mixer output on root | 2 draining `dtoh` + host concat + 1 `htod` | 1 peer read + 4 event primitives + 1 local copy |
| `return_block_to_root` / `return_row_to_root` | EP diet bulk return, EP grouped-prime partial, EP v1 per-slot row | 1 draining `dtoh` + 1 `htod` | 1 peer read + 4 event primitives |

FOUR event primitives per peer read, not two, and the fourth is not optional — see "What the
gate caught" below.

### Three design decisions, each with its reason

**1. PULL, not push.** The copy is issued on the CONSUMER's stream reading the producer's
buffer. Two reasons, and the second is the load-bearing one on our stack:

- `RESEARCH.md` §2.2b/§6.12: on this fabric SM-issued peer writes collapse (2.6 vs 52 GB/s;
  110 us push vs 26 us pull), and b12x's own dead-end table records *"Push / peer-store
  collectives"* as a receipted negative. Our copies are copy-engine, so this does not bind
  us today — it binds any future fused collective, and building the push twin now would be
  building the shape the world already measured as wrong.
- **Ordering.** A consumer-issued copy is ordered against the consuming kernel on that same
  stream for free. A producer-issued push (`pp.rs`'s `BoundaryTransport::Peer`, which enqueues
  `memcpy_peer_async` on the TX stream at `pp.rs:2636`) needs a separate event to publish into
  the consumer's stream. Our tax is a round-trip and launch COUNT tax (§2.6, §6.3c, and three
  upstream maintainers in §1.5g), so a primitive removed per hop is the lever.

**2. EVENT ordering, not a host fence — and this is the delta that makes the arm worth
building.** The step seam's native-P2P pull sites already exist and are already pull-shaped
(`tp.rs:2650`, `:3552`, `:5441`), but every one of them fences the producer with a host
`engine.stream().synchronize()` — the *"PRODUCER FENCE (2026-08-20 flake fix)"* comments at
`tp.rs:2636` and `tp.rs:3543`. That is correct and it is **a full stream drain**, i.e. it
retains the exact cost class this lane exists to delete. A glm5 native-P2P arm copied from
that pattern would have removed the host *bytes* and kept the host *syncs*, measured as ~no
gain, and banked a false "native P2P does not help on this seam".

This lane instead takes the ordering contract from `pp.rs`: one publication event per rank,
`ev.record(producer_stream)` then `consumer_stream.wait(&ev)` — the live `BoundarySlot`
contract (`pp.rs:2647`/`:2678`) that the hy3 PP-4 qualification closed at **50/50 fresh
processes, 200/200 runtime probes, zero byte mismatch on this exact RTX PRO 6000 card class**,
and that `tp2-20260831/LANE.md` stage 1 already cherry-picked onto this line
(`18cfaf774` / `610425804` / `57a6b16d8`). **PULL direction from `tp.rs`, EVENT ordering from
`pp.rs`, host syncs from neither.** Two events total per runtime, not per hop: `cuEventRecord`
overwrites and `cuStreamWaitEvent` captures state at call time, so a single issuing host
thread gets correct ordering without putting a driver allocation on the per-token path.

**3. ATOMICS-FREE, sentinel-free.** Every SM120 PCIe pair reports
`NativeAtomicSupported=0` (§2.1) and *"CAS on peer memory silently loses barrier tokens under
PCIe load"* — load-dependently, so microbenchmarks pass. This transport uses only copy-engine
copies and CUDA events: no peer flag polling, no compare-and-swap, no payload-value sentinels
(§6.13: a payload-sentinel prototype caused a launch-day production stall, vLLM#479 — *"It
generates about 400 tokens, then stops generating... GPUS no longer respond to nvidia-smi"*).
There is also no slot counter to get wrong, so b12x#97's signed-`slot_`-aliases-the-barrier-
region class (*"fully silent cross-rank corruption... first appearing on call 2147483650"*)
has no analogue here — recorded because it WILL have one the day a double-buffered fused
collective lands.

### What is NOT built, and why (the pre-registered bug list applies to the thing we did not build)

**No fused device-resident collective, no CUDA-IPC handle path, no graph capture of the peer
copies — yet.** `RESEARCH.md` §6.13 hands over a complete specification for a one-shot,
pull-only, double-buffered, generation-lane collective at *exactly width 4096, our hidden
size*, with its bug list pre-registered (§6.7's eight b12x issues; §6.6's
`custom_all_reduce.cuh:455 'invalid argument'` at graph capture; §6.3's "IPC handles must be
opened peer-accessible to the IMPORTING device before any kernel dereferences them —
`copy_()` working proves nothing"). We did not build it this stage for a reason that is
itself a finding: **our program has no reduce to fuse.** Column-parallel-over-gather means
every hop is a point-to-point dense block move, which the copy engine already does at
~54 GB/s. A one-shot collective buys nothing over `cuMemcpyDtoDAsync` for a pure gather. What
it WOULD buy is launch-count amortisation at TP-4 — see the TP-4 arithmetic.

Also not built: CUDA-IPC across processes (we are single-process multi-device — `Glm5TpRt`
holds a second `Engine` in the same process, `PpNRt` holds N contexts and streams, no
fork/MPI/NCCL anywhere), and the `ForceP2P` modprobe change (see the driver note below).

### Fail-closed arming

`Glm5TpRt::arm_transport` runs at preflight, AFTER the rank engines exist and BEFORE any layer
is sharded:

1. `MEMRA_GLM5_TP_TRANSPORT` parse; any unknown spelling refuses by name.
2. On the peer-pull arm and a REAL device pair: `tp::grant_peer_access` in both directions —
   extracted this lane from `configure_native_p2p` so the two seams share one grant sequence
   instead of drifting copies. It does BOTH halves: `cuCtxEnablePeerAccess` (legacy
   allocations) **and** `cuMemPoolSetAccess` on the owner's default memory pool, because
   `cuCtxEnablePeerAccess` does NOT map stream-ordered pool allocations and every normal memra
   buffer is one (`pp.rs:1543`). Refuses by name when `cuDeviceCanAccessPeer` reports no path —
   the honest answer per §3.2, which records host classes of this card presenting **peer islands
   of two**, every cross-island cell of the peer-transfer matrix reading `N/A`. Which hosts those
   are is fleet data and stays in the private deployment repo; the engine refuses by name rather
   than knowing the host.
3. A byte-integrity **pull ladder through the exact primitive the walk uses**, both
   directions, at 16 KiB / 64 KiB / 1 MiB / 64 MiB (the step seam's
   `NATIVE_P2P_PROBE_WORDS` values, so a glm5 refusal and a step refusal name the same fabric
   with the same numbers), against a poisoned destination, compared **bit-for-bit on
   `f32::to_bits`**. One differing word refuses the load. Bottom rung is deliberately below
   our smallest real hop and the top above any prime block, because §2.4 records that LL and
   Simple paths *"can fail differently"* by size.
4. Announce `[glm5-tp-transport] armed transport=... shape=... same_device_gate=...
   performance_claim=false`, and all four pre-existing seam announces
   (`[glm5-tp-preflight]`, `[glm5-tp-kda]`, `[glm5-tp-mla]`, `[glm5-tp-ep]`) now print the
   LIVE transport instead of a hardcoded `transport=host-canonical`. That string was
   hardcoded before this lane, which would have made a transport A/B unreadable from the boot
   log — the tp2-battery greps exactly those four lines.

### What the ladder proves, and what it does not (stated, because §2.4 is emphatic)

By `RESEARCH.md` §2.4's ladder the pull ladder is a **transfer-tier** check. It proves our
copy path moves the right bytes both ways at four sizes. It does **not** prove that a KERNEL
dereference of a peer pointer works, and §2.3b is explicit that on direct-attach (`NODE`)
hosts — which is our box class, `RESULTS.md`: *"ALL pairs NODE, one NUMA node"* — the driver
serves **SM-issued** peer access through SysMem staging by default, making a custom PCIe
collective *"~15x slower than NCCL"*, while *"`nvidia-smi topo -p2p r` returns OK and
`cudaMemcpy` looks healthy, so neither detects it"*. Only a `simpleP2P`-class kernel peer read
does.

**Two things follow, and they are the honest reading of the driver question in the brief:**

- **The `ForceP2P` 3-key note does not gate this lane.** §2.3b's own matrix says the override
  is needed for *"PCIe oneshot / custom all-reduce: Yes on NODE / direct-attach"* and
  *"`cudaMemcpy` P2P: usually no — uses copy-engine path, so it can look fast even without
  this override"*. Our transport is copy-engine only. It never dereferences a peer pointer
  from a kernel. So the SysMem-staging default cannot bite it, and we do not need a
  reboot-scoped modprobe change to ship it. That is a deliberate scope choice: it makes this
  arm deployable with zero host configuration, and it is the reason the arm is worth shipping
  before the fused-collective arm.
- **We still measure the kernel path, because the NEXT arm depends on it.** Banked in this
  lane as `peer-read-probe.cu` (a self-contained `simpleP2P`-class byte-validating kernel
  peer read, 4 B .. 64 MiB, both directions, `Test passed` / non-zero exit) and run by
  `HEALTH.sh` at every box window. If it returns zeros or garbage on a box, the fused-collective
  arm is blocked on the 3-key `ForceP2P` form
  (`ForceP2P=0x11;GrdmaPciTopoCheckOverride=1;EnableResizableBar=1` — **never** the 5-key form:
  adding `RMForceP2PType=1;RMPcieP2PType=2` broke real peer copies with *"invalid device
  ordinal"* on driver **580.167.08** *while `cudaDeviceCanAccessPeer` still read 1*).
  **Driver check on the rig, done:** `/proc/driver/nvidia/version` = **595.84** open kernel
  module, i.e. above the 580.167.08 caveat, and `/proc/driver/nvidia/params` shows
  `RegistryDwords: ""` with `EnableResizableBar: 0`, `DmaRemapPeerMmio: 1` — no override
  applied. The rig is a single RTX 5090 Laptop GPU so it has no peer to probe; the box
  window's `HEALTH.sh` captures the same three facts per serving box and that is where the
  question is actually answerable.

### The FLAGS decision, written (per the 2026-08-25 owner rule)

`MEMRA_GLM5_TP_TRANSPORT` defaults to **`host-canonical` (OFF)**. Not an accident of
implementation order: on the day it landed the peer-pull arm had **zero receipts on real peer
hardware**, because the rig is one card (`LAW:rig-exactness-only`) and its gate arms run the
code over two contexts on ONE device — proving bit-preservation and refusing nothing about the
fabric. Unmeasured behaviour does not default ON. Both arms, the rollback seam
(`MEMRA_GLM5_TP_TRANSPORT=0`) and the receipts pointer are in the `docs/FLAGS.md` row in this
same PR. The default flips in the same commit as the box window's interleaved re-price
receipt, never before.

---

## Stage 3 — GATES

### The bar does not move, and that is the point

Transport moves the same bytes by the same layout rules in both arms, so the class bars from
`tp2-20260831/LANE.md` stage 4 stand UNCHANGED:

- **DECODE (t=1), non-MoE classes: BYTE IDENTITY** vs the door-OFF plain walk.
- **PRIME (t>=2): the calibrated 2e-4 band** + tape identity + repetition byte-identity
  (batched GEMM widths select shape-dependent K-reduction splits; the documented
  `Engine::linear` m-dependence class).
- **MoE EP: the pre-registered band** (the sequential per-slot walk does not bit-reproduce the
  fused NVFP4 epilogue kernels; `RESULTS.md` cell-1 item 2).
- **Reds bite orders above green.**

Plus one bar this lane adds: **transport-vs-transport byte identity**, decode AND prime, on
every arm. The two arms must agree bit-for-bit with each other, not merely each with plain —
because a transport that changed prime bits inside the calibrated band would hide inside it.

### What the gate caught on its FIRST run (the most valuable output of this stage)

Three defects, all in code written this lane, all found by the rig gate before any box time was
spent. Recorded here because two of them are reusable lessons and one is a corpus-grade trap.

**1. A write-after-read race — the `BoundarySlot` contract's second half, missing.** Arm X1
(peer-pull, undieted sequential EP walk) failed **DECODE SELF-CONSISTENCY at step 17**: two
repetitions of the same greedy walk diverged, and the prime band read `2.732e1` against a `2e-4`
bar. Cause: every per-slot expert row is a fresh stream-ordered allocation on the peer, and its
async free is enqueued on the **peer** stream while the **root** stream's pull is still reading
it. The publication event (`producer records -> consumer waits`) orders read-after-write; it does
nothing for write-after-read. Fix: a **release event per rank** — the consumer records after its
copy, the producer waits before proceeding. That is exactly `pp.rs`'s `ev_rx`
(`pp.rs:715`, recorded `:2705`, awaited `:2588`), and the step seam banks the same hazard as a
keepalive comment (`tp.rs:7831-7834`: *"otherwise async free can recycle the owner's allocation
while cuMemcpyPeerAsync is still reading it"*). **Cost of the fix: 4 event primitives per peer
read instead of 2** — folded into the arithmetic below. After the fix, X1 is byte-identical to
plain at decode, `3.637e-5` at prime, and self-consistent.

**2. The dieted arm was a VACUOUS GREEN on the broken build.** On the same run that X1 failed,
**X2 (peer-pull composed with the EP dispatch diet) PASSED every bar.** The diet issues ONE bulk
return per layer-call instead of ~168 per-slot returns per token, so it had ~168x fewer chances
to lose the race and simply did not lose it. A transport correctness claim read off the dieted
arm alone would have shipped a silent cross-rank corruption to the box.

Two things follow, both worth carrying:

- **A transport arm must be gated on the walk with the MOST hops, not the fastest one.** The
  count-cutting lane and the cost-cutting lane are orthogonal, and the undieted walk is the
  higher-resolution instrument for transport correctness precisely because it is worse. That is
  a reason NOT to retire the sequential EP walk.
- The failure presented as **run-to-run non-determinism**, not as a wrong answer. The gate's
  two-repetitions self-consistency arm is what saw it; a single-repetition identity arm would
  have passed roughly as often as not. `RESEARCH.md` §2.4's *"silent corruption is a real
  failure mode and is strictly worse than a hang"* has a sibling: **silent corruption that is
  also intermittent is strictly worse than either, and only a repetition arm catches it.**

**3. The arm-time validation ladder polluted the per-token census.** First run's census read
`xfer_bytes` on the peer-pull arm at **25x** the host-canonical arm's — the opposite of the
truth, since peer-pull crosses PCIe *half* as much (a host bounce crosses it twice). Cause: the
byte-integrity pull ladder drives the same instrumented primitive the walk does, and 4 rungs x 2
directions is ~130 MiB of arm-time traffic against ~15 MiB per decode token of real traffic. A
box window reading `xfer_bytes` deltas would have derived a bytes/token figure that was ~97%
arming noise. Fixed: `arm_transport` snapshots the counters before the ladder and restores them
after, and the ladder's PASS line now carries `census_excluded=arm-time-ladder-traffic`.
**The general shape: an instrument that measures its own self-test reports the self-test.**

### Gate table

All on the rig (single RTX 5090 Laptop, driver 595.84), `flock /tmp/memra-5090.lock`,
`NVIDIA_TF32_OVERRIDE=0`, exactness only — no timing number is read out of any of them.

| gate | invocation | result |
|---|---|---|
| unit: transport flag law + existing glm5_tp laws | `cargo test -p memra-engine --lib glm5_tp` | **8/8 PASS** (4 transport: literal/fail-closed parse incl. 7 rejected misspellings, the written default, rank involution, census-line field coverage; 4 pre-existing) |
| `glm5-tp-gate` FULL matrix, both transports, **on the merged tree** | `run-transport-gate.sh` P=16 N=12 | **ALL ARMS PASS — 80 verdicts, exit 0, zero FAIL** (`gates/01-tp-gate-transport-p16-n12.log`; the banked log is the re-run on the tree that carries the merge and the `[glm5-tp-ep]` argument-order fix, so every receipt matches the pushed tree) |
| X0 transport pin held | census flat across the whole `=0` battery | PASS: `host_legs=42930 host_syncs=22960 xfer_bytes=21901504 peer_pulls=0` |
| X1 tp-all peer-pull | decode byte identity + prime band + self-consistency | PASS: decode **BYTE-IDENTICAL** to plain (28 t=1 steps x 32 logits + tape), prime `max_rel=3.637e-5` (band 2e-4), two repetitions bit-identical |
| X1 transport engagement | census non-vacuity | PASS: `peer_pulls=1674 (>0) host_legs=0 (==0) host_syncs=0 (==0) pub_events=6696` — 4 event primitives per read, as designed |
| X2 tp-all peer-pull + EP diet | the two axes composed | PASS: decode BYTE-IDENTICAL, prime `3.637e-5` |
| X3 red skip-peer-combine through peer-pull | the red must still bite | PASS: `max_rel=6.742e1`, tape diverges — identical magnitude to the host-canonical R3, i.e. the red is transport-independent |
| **XT transport-vs-transport byte identity** | decode AND prime, arm vs arm | **PASS: decode BYTE-IDENTICAL, prime BYTE-IDENTICAL, `max_rel=0.000e0`, both tapes match** — the transport moves bytes and does not compute, proven rather than argued |
| XF unknown transport spelling | must refuse by name | PASS: `MEMRA_GLM5_TP_TRANSPORT="peer_pull" is not a known transport (host-canonical \| 0 ...; peer-pull \| 1 ...)` |
| peer-pull byte-integrity ladder | 4 rungs x 2 directions, poisoned destination, `f32::to_bits` compare | PASS x4 armings, `byte_ladder=[16384, 65536, 1048576, 67108864] mismatches=0` |
| every pre-existing arm (A/B/C/C0/C2/D/E/F/G/M/H1-H5/T/R1-R4/B2/B3/M2/R2D/R3D) | unchanged bars | PASS — verdicts identical to the banked `tp2-20260831` run |
| standing suites | `glm5_tparallel_verify_gpu` / `glm5_ep_diet_doors_gpu` / `hyper_connections_gpu`, `--ignored` | **3/3, 9/9, 6/6 PASS** (`gates/03-suites.log`) |
| spec-ppn 8 arms | `ppn-verify-20260830/run-spec-ppn-gate.sh` | **ALL ARMS PASS**, exit 0 (P=24 N=20 K=7, fence=[0,1,2,4]) — `gates/04-spec-ppn.log` |
| hyper-ppn 10 arms | `ppn-hyper-gate/run-ppn-hyper-gate.sh` | **ALL ARMS PASS**, exit 0 (25 comparisons BIT-IDENTICAL vs the unsplit hc walk per arm) — `gates/05-hyper-ppn.log` |
| clippy zero | `cargo clippy --workspace --all-targets`, merged tree, CPU-capped | **zero warnings** (the two `nvcc`/`MEMRA_CUDA_ARCH` build-script info lines are not warnings on our code) |
| fmt | `cargo fmt --all` | clean |
| check-flags | `tools/check-flags.sh`, merged tree | **755 runtime literal reads, no uncovered names, no grandfather list** |
| `tools/local-ci.sh --perf` | before push | correctness stage GREEN through the decode-batch gates (server HTTP unit suite, drafter-attach wiring gate ALL GREEN, `test_local_ci_lock` ALL GREEN, `kernel-check ALL GREEN (105 cells, 11 skipped)`, `sample-check OK`, `decode-batch-gate` x4 ALL GREEN on 9B NVFP4 + 9B Q8_0); then the run WEDGED in `graph-warmup-stress-gate.sh` — see the incident below — `gates/06-local-ci.log` |

#### Two receipt-plumbing defects found while banking these gates

Neither changes a verdict; both would have corrupted a receipt, which is worse than a red gate
because it is quiet.

1. **`... | tee "$LOG"` truncates `$LOG` before `flock` returns.** A queued re-run of
   `run-transport-gate.sh` on a busy rig destroyed the previously banked 237-line log and left a
   **zero-byte receipt** for as long as it sat waiting for another lane's rig lock. Fixed: write
   to `mktemp`, `mv -f` into place at the end. **A receipt file must never be shorter than the
   last run that produced it**, and a pipeline built before a blocking lock is acquired violates
   that by construction.
2. **A swapped `eprintln!` argument printed `[glm5-tp-ep] transport=even-split`.** The transport
   token had been inserted one position early and was landing on `ownership={}`. Only reading the
   rendered receipt caught it (the receipt-extract in `run-transport-gate.sh`, which is why that
   extract exists). Fixed with the incident recorded at the call site.

#### local-ci incident: `graph-warmup-stress` wedged at cycle 6 (recorded, not disowned)

`tools/graph-warmup-stress-gate.sh` (qwen35-9B-NVFP4-MTP, `--cycles 10`) logged cycles 1-5 in
the first ~5 minutes and then burned **100% CPU on its main thread with 0% GPU utilisation for
37 minutes** without logging cycle 6, ignoring `SIGTERM` (killed with `SIGKILL`). Evidence
captured before the kill: `State: R`, `nonvoluntary_ctxt_switches 79587`, four sibling threads
all sleeping in `poll_schedule_timeout`/`futex_do_wait`, and `utilization.gpu` sampled `0 %`
ten times across 30 s while the process held 7,684 MiB. The banked comparison run
(`tp2-20260831/fwd-merge-gates/local-ci.log:699-711`) logs cycles 1-10 consecutively and then
`ALL GREEN: graph-warmup-stress (10 cycles x 4 arms + overlap...)`, so this gate does complete
normally on this rig.

**Reachability, established rather than assumed:** the stage runs qwen35 with `MEMRA_GLM5_TP`
and `MEMRA_STEP_TP` unset, and the whole local-ci log contains **zero** `[glm5-tp-...]` markers,
so no line of this lane's transport executed. The transport module is only constructed from
`prepare_glm5_tp_load`, which returns `None` when the door is cold. The one file this lane
touches outside the glm5 seam is `tp.rs`'s `grant_peer_access` extraction, which is a pure
call-for-call move reached only from `configure_native_p2p` (one caller: the step seam).

**That is a reachability argument, not a receipt, and this lane does not close it by asserting
its own innocence.** The environment was also contended (another lane's `local-ci --perf` queued
on the same rig lock, plus an unrelated python job holding 4,258 MiB), which local-ci itself
flagged: *"WARNING — other GPU compute apps present (numbers not window-valid)"*. Two named
follow-ups, in order:

1. **The free control is already running.** A second lane's `local-ci --perf` on a different
   tree (`consol-bankfix`) took the rig lock immediately after this run was killed and will hit
   the same gate. If it wedges there too, the wedge is tree-independent and environmental; if it
   passes, the next step is a bisect on this tree. Watch
   `/tmp/bankfix-tools/logs/local-ci-perf.log` for `ALL GREEN: graph-warmup-stress`.
2. **Then re-run this lane's `local-ci --perf` UNDER A CPU QUOTA** and bank the full log. The
   first attempt ran **uncapped**, which violates the owner's standing *no uncapped local CPU
   saturation* rule and is also the contention this lane suspects; the re-run is queued as
   `systemd-run --user --scope -p CPUQuota=600% -p CPUWeight=20 nice -n 19 env
   CARGO_BUILD_JOBS=6 ... tools/local-ci.sh --perf`, i.e. the exact shape of the control run
   that PASSED the same gate.
   `MEMRA_CI_GWSTRESS=0` exists and would let the rest of the run complete, but it is NOT used:
   skipping the stage that failed and calling the gate green is the
   `exception-lists-need-expiry` / `cert lines carry invocations` failure shape. The row above
   states exactly how far the run got and stops there.

**The push is HOOK-GATED, and that is the right answer.** `git push` on this repo refuses
without a completed `tools/local-ci.sh --perf` (*"Override knowingly with
MEMRA_SKIP_PERF_CI=1"*). `MEMRA_SKIP_PERF_CI` is never set by this lane: an override that exists
to make a push succeed is the same failure shape as the skip above. Both commits therefore sit
on the local branch until the capped re-run exits 0, at which point the queued watcher banks the
log and pushes. 

#### The rig deadlock this lane caused, mis-attributed, and then cleared

**`TRAP:sccache-inherits-the-flock-fd` is NOT fixed. It recurred here, and this lane caused it.**
`research/INDEX.md`'s extract-general row banks it as *"the sccache-flock rig deadlock found+fixed
(daemon inherits the lock fd; ~80 min idle 5090 across three queued lanes)"*. Same shape, one
lane worse: **four** queued lanes, ~1 h of idle 5090.

Mechanism, nailed with evidence rather than inferred:

```
/proc/locks, matched on the lock file's dev:inode (00:30:931400):
  181: FLOCK ADVISORY WRITE 2581277 00:30:931400   <- HOLDER ... and `ps -p 2581277` is EMPTY
  181: -> FLOCK ... 2773246    (another lane's accrace-inner, waiting 1h01m)
  181:  -> FLOCK ... 2863671   (this lane's gate, waiting 52m)
  181:   -> FLOCK ... 2950992  (a second lane's local-ci)
  181:    -> FLOCK ... 3329190 (a third lane's local-ci)

/proc/2773346/fd/9 -> /tmp/memra-5090.lock (inode 931400)      <- sccache, started 22:32:11
```

`fd 9` is the fd number `local-ci.sh` uses for its whole-run lock. A build inside this lane's
first (uncapped) `local-ci --perf` spawned the `sccache` daemon, which **inherited fd 9**; when
that run was SIGKILLed the daemon survived — it is not in the run's process tree (PPID 3353, a
long-lived shell) — so the kernel never released the flock, and the lock record stayed attributed
to a PID that no longer exists.

**Three diagnostic readings, two of which lied to me:**

| reading | verdict |
|---|---|
| `pgrep -af flock` | **misleading** — every visible `flock` was a CHILDLESS waiter, which reads as "nobody holds it" |
| `fuser -v <lock>` | **misleading** — `F....` means only "has the file open for writing", NOT "holds the flock". I read `F` on another lane's `local-ci` and **wrongly blamed that lane for holding the lock for 45 minutes**; its own lock self-test (`test_local_ci_lock: ALL GREEN`, including the `MEMRA_GPU_LOCK == whole-run lock is redirected (the self-deadlock value)` case) was passing the whole time |
| **`/proc/locks` matched on `dev:inode`** | **the only one that names the holder** — and a holder PID absent from `ps` is the signature of the inherited-fd case |

Remedy applied: confirm no live compile with an **exact `comm` match** (`ps -eo comm | awk '$2=="rustc"`)
— pattern-matching `rustc|cargo` against command strings returns false positives from shell
wrapper command lines, which nearly stopped me from clearing it — then `kill -9` the sccache
daemon. On-disk cache survives; cargo respawns it. The lock passed to the head waiter
immediately and all four lanes drained in queue order.

**Prevention, a named engine-tooling follow-up (not this lane's file to change):** `local-ci.sh`
should open its whole-run lock fd `O_CLOEXEC`, or spawn cargo with that fd closed, so no daemon
can inherit it. Until then, **SIGKILLing a `local-ci` run leaves the rig locked**, and the
recovery is the `/proc/locks` + sccache sequence above. That belongs in the gpu corpus as a TRAP
with this receipt attached, since the banked "found+fixed" note is what made me look elsewhere
for an hour.

Separately and still true, but NOT the lock blocker: an unrelated third-party GPU job
(`run_local_candidates.py --model llmlingua`, 4,258 MiB, 3 h old) makes every perf window
`window_clean=false` for every lane. That degrades perf rows; it does not hold the lock.

The pinned-`=0` arm is not decoration: per the house pattern the OFF arm sets the flag
explicitly to `0` rather than leaving it unset, and the census counters must read FLAT
(`peer_pulls == 0`) across the whole banked OFF battery. A pin that is not holding reads
exactly like a passing gate otherwise.

### The engagement receipts a box window greps

```
[glm5-tp-transport] armed transport=peer-pull shape=consumer-issued cuMemcpyDtoDAsync, event-published, atomics-free same_device_gate=false performance_claim=false
[glm5-tp-transport] peer-pull byte-integrity ladder PASS: directions=2 byte_ladder=[16384, 65536, 1048576, 67108864] mismatches=0 same_device_gate=false
[glm5-tp-preflight] armed ranks=2 devices=[0, 1] ... transport=peer-pull weights_loaded=false performance_claim=false
[glm5-tp-kda] head shard armed: ... transport=peer-pull performance_claim=false
[glm5-tp-mla] head shard armed: ... transport=peer-pull performance_claim=false
[glm5-tp-ep] expert-parallel armed: ... transport=peer-pull performance_claim=false
[glm5-tp-transport] census transport=peer-pull host_legs=0 host_syncs=0 peer_pulls=N pub_events=2N local_copies=M xfer_bytes=B
```

`transport=` on all four pre-existing seams plus the two new lines.

**Reading a GATE log vs reading a BOX BOOT log, because they differ and the difference bit
once.** `[glm5-tp-kda]`, `-mla` and `-ep` latch on a once-per-process `AtomicBool`, so a gate
binary that loads 24 models in one process prints them ONCE, naming whichever transport armed
first. `[glm5-tp-preflight]` and `[glm5-tp-transport]` print per load and do carry both arms
(the banked gate log shows 18 `transport=host-canonical` and 4 `transport=peer-pull`
preflights). On a box each boot is one process with one transport, so all six lines agree —
but do not read a gate log as if it were a boot log.

And the receipt-extract earned its keep immediately: the first pass printed
`[glm5-tp-ep] transport=even-split`, because the transport argument had been inserted one
position early in that `eprintln!` and was landing on `ownership={}`. **A receipt line is worth
exactly what its argument order is**, and the only thing that catches a swapped format
argument is reading the rendered output. Fixed, with the incident recorded at the call site.

`bytes/token` is `xfer_bytes` delta divided by tokens decoded — the counter `RESEARCH.md` shortlist item 2
asked for and that this tree did not have (the only prior instrumentation was copy/dispatch
COUNTS: `PEER_BOUNDARY_COPIES` at `pp.rs:791` and the five `GLM5_EP_DIET_*`; every
`bytes_per_token` symbol in the tree is KV-cache accounting).

---

## Stage 4 — NAMED FOR THE BOX WINDOW (not this lane's to run)

Ordered by what the arithmetic below says. Every arm carries the standing measurement laws:
interleaved x5 fresh boots (`interleaved-ab-protocol-law`; §2.4 independently measures
*"a cold card measures ~4% faster on decode"*), arm identity by boot nonce not health-200,
vendor-default sampled twin + the 8-turn larger-prompt cache-on twin
(`multiturn-cache-measurement-law`), P8 wait before measurement (§6.14: an idle transition
moves the same graph 54.2 -> 56.4 tok/s), and `HEALTH.sh` at boot.

1. **TP-2 peer-pull re-price vs PP-3.** `MEMRA_GLM5_TP_TRANSPORT` 0 vs `peer-pull`, EP diet
   on both sides, `MEMRA_GLM5_EP_GROUPED_PRIME=1` for the TTFT arm. Reads: pool/deep decode
   tok/s, TTFT at 0.5k and 3.7k, the census deltas, spreads. Bar: does peer-pull TP-2 beat
   PP-3 served 35.36 (and the current ship 62.43 on the PP-3 shape) once TTFT is unblocked.
2. **TP-4 (all four cards)** if arm 1 holds. Geometry is legal with no padding (below). This
   is a real engine increment (`GLM5_TP_RANKS` is a const 2 and the preflight's laws are
   written against it), not a flag flip — sized in the arithmetic below.
3. **Spec x TP composition gate.** Currently co-refused (`glm5_spec_session_new` refuses
   while the TP door is armed; tp-gate arm F). This is the unlock: spec is worth **1.6-2.4x**
   (§1.3c: 139.4 -> 228.0 at MTP:3, accepted length 2.52; §1.5d: 45.37 -> 105.69 at
   `tp=4 dcp=4`, 2.33x; our own vrest DFlash2+PMIN loop is 1.77x on PP-3) — and the reason
   TP is the right axis at all is that **PP forecloses spec on both public stacks** (§1.4b:
   sglang forces `--disable-overlap-schedule`, *"speculative decoding incompatible with PP"*;
   vLLM's MTP-under-PP was broken because the accounting *"is computed on the last PP rank
   only"*). On OUR stack spec+PP works and spec+TP is co-refused — the exact inverse of the
   public constraint, and the single highest-value refusal to lift.
4. **Re-derive `docs/decisions/PRO6000-MULTICARD.md`'s topology law with TP-4 in scope**
   (§1.3 refutes its premise; see the correction above).
5. **Blue/green TP deploy is a new risk class, per §2.9 of the shortlist's below-the-line
   list:** stopping a `--tensor-parallel-size 2` container *"reproduces reliably in a single
   start/stop cycle"* and leaves *"every GPU in the TP group permanently unrecoverable until a
   full system reboot"*. Per `Incident fixes get a bench gate`, replay a TP stop/start cycle
   on a bench box before any TP cutover, and confirm unattended hard power-cycle exists
   (warm reboot is documented as insufficient).

### TP-4 divisibility — legal with NO padding on this geometry

| tensor / axis | value | / 4 | verdict |
|---|---:|---:|---|
| KDA heads (`linear_attn_config.num_heads`) | 64 | 16 | OK |
| KDA `head_dim` | 128 | n/a | OK — `KDA_HEAD_DIM` is rank-count independent; the preflight law is `== 128`, not `/ ranks` |
| KDA qkv plane (`64 * 128`) | 8192 | 2048 | OK |
| MLA heads (`num_attention_heads`) | 64 | 16 | OK |
| MLA attention width (`64 * v_head_dim 256`) | 16384 | 4096 | OK |
| `wo` out rows (`hidden_size`) | 4096 | 1024 | OK |
| routed experts | 288 | 72 | OK |
| DSA indexer heads (`index_n_heads`) | 32 | **replicated** | no constraint — the correction above |
| `kv_lora_rank` / latent | 512 | replicated | no constraint |

**TP-3 is the only shape that needs padding** (`64 -> 66` heads, 22/rank; 288/3 = 96 experts
already divides). Per the brief that is a **DESIGN NOTE ONLY, do not build**: `RESEARCH.md`
§1.5d shows the padding trick is real and automatic in one public stack (*"TP6 virtual
sharding remains automatic ... the padded 66-head / 2,112-expert layout"*), and also shows the
counter-evidence on the same host (`tp=8 dcp=1` 146.86 beat `tp=6 dcp=3` 97.90) — non-power-of-two
TP is not free even when legal. With TP-4 legal at zero padding cost on our geometry, TP-3 has
no reason to exist for us. Recorded and closed.

### Predicted arithmetic — nothing here is a claim; the box window prices it

Priors, all from `RESEARCH.md`, all on our exact model class and card class:

| prior | value | source |
|---|---|---|
| the bar: our model, 4x PRO 6000, TP4/DCP1, **spec OFF** | **139.4 tok/s** decode, 15,549 tok/s at 32k prefill | §1.3c (`llm-decode-bench` 0.4.29, greedy, median of 3, raw samples 139.53/139.41/139.38) |
| same, plain PyNCCL instead of a custom collective | **117.2 tok/s** | §1.3d |
| so the custom collective is worth | +17.1% prefill / +19.0% decode | §1.3d vs §1.3c |
| TP4 over TP2 at C1, direct-root host (our class) | **1.149x** (149.8/130.4) | §1.3 |
| TP4 over TP2 at C1, switched host | 1.214x (182.6/150.4) | §1.3 |
| spec multiplier on our model | 1.64x (MTP:3) .. 2.33x (`tp=4 dcp=4`) | §1.3c, §1.5d |
| peer bandwidth / latency budget, same root complex | ~55-56 GB/s uni, ~111 bidi, **~0.4 us** | §2.2 |
| the diagnostic threshold | **14 us peer latency = P2P is not engaged** | §2.2 |

Our own banked terms (`tp2-battery-20260831/RESULTS.md` cell 3, `ATTRIBUTION.md` §4):
table terms 22.0-24.1 ms/token x driver tax 1.2-1.3 = 26-31 ms; measured v1 transport residual
13-18 ms; engine wall 44.15 ms = 22.65 tok/s engine twin; served-class projection ~27-30.

**TP-2 peer-pull, predicted:**

```
peer-pull per-token movement (decode t=1, EP diet ON):
  peer reads          275      (KDA 34x4 + MLA 11x5 + MoE-diet 42x2)
  event primitives    1100     (4 per read: publish record+wait, release record+wait)
  local copies        ~145
  crossed bytes       6.09 MiB (half of host-canonical: a host bounce crosses PCIe TWICE)
  fabric time         6.09 MiB / 53 GB/s = 0.115 ms          <- 1% of the v1 tax
  host syncs          0        (gate-verified: X1 host_legs=0 host_syncs=0)
  exposed cost        275 copies x ~3 us CPU issue + 1100 event prims x ~1.5 us
                      ~= 2.0-3.5 ms/token                     <- now LAUNCH-bound, not sync-bound
  reclaimed           13-18  ->  2.0-3.5  =  -10 to -16 ms/token

  engine wall         26-31 (table x driver) + 2.0-3.5  =  28.0-34.5 ms/token
                      => 29.0-35.7 tok/s engine twin
  served-class        x1.19-1.32 (the battery's own single-engine driver factor: plain1
                      engine 19.87 vs 24-26 served; TP-2 v1's own 22.65 -> ~27-30 projection)
                      => 34.5-47.1 tok/s served-class
```

So peer-pull TP-2 is predicted to land **at parity to ~1.35x over PP-3's 35.36** — a real
crossing of the arm that beat it, and **still far below the 62.43 tok/s the PP-3 shape ships
today**, because the ship number carries the matvec + spec + diet stack that TP-2 does not.
**Transport is worth ~1.4-1.6x on the TP-2 arm and it is an ENABLER, not the answer.** Said
plainly so no one reads a crossing as a cutover.

**TP-4 peer-pull, predicted — and the term that flips:**

```
bandwidth term        15.2 ms (1 card) -> 7.6 (TP-2) -> 3.8 (TP-4 ideal)
                      EP-4 slowest-rank haircut: E[max share] of 8 slots over 4 ranks
                      ~= 3.4/8 = 42.5%, i.e. ~1.7x vs 1/4  =>  ~4.85 ms effective
latency class ~10 and drain ~4.4 do NOT scale with cards
table terms           4.85 + 10 + 4.4 = 19.25 ms  x1.2-1.3  =  23.1-25.0 ms

transport at TP-4     the gather becomes an ALL-GATHER: each rank pulls 3 peers' quarters,
                      so peer reads per gather go 2 -> 12 and crossed gather bytes x3
                      => ~750 peer reads + ~3000 event primitives per token
                      => ~4-8 ms/token exposed issue cost (bytes still ~0.03 ms: irrelevant)

engine wall           23.1-25.0 + 4-8  =  27.1-33.0 ms  =>  30.3-36.9 tok/s engine twin
served-class          x1.19-1.32                        =>  36.1-48.7 tok/s
vs peer-pull TP-2                                       =>  1.03-1.05x
```

**And that is the finding, not the number: at TP-4 the transport goes back to being
launch-count-bound and eats nearly ALL of the 1.15x that TP-4's second-and-third memory
systems buy.** The bottom-up arithmetic (1.03-1.05x) sits just under the public prior (1.149x),
which is the expected direction: the public TP4 arm runs a fused device collective, we would be
running ~3750 driver primitives per token. That gap IS the unlock, and it is the same gap the
research's own count model describes. Two unlocks, in order:

- **CUDA-graph capture of the peer copies.** Our primitives are copy-engine `memcpy_dtod`
  plus `cuEventRecord`/`cuStreamWaitEvent` — no host synchronize, no mutex, no lazy
  allocation, no atomics. That is exactly the capturable set, and exactly what the `pp.rs`
  `BoundarySlot` path is NOT (host `synchronize()` at `pp.rs:2603`, mutexes at `:2589`, lazy
  alloc at `:2593`). `b12x#246`'s *"graph peer-push"* (§6.3) is the upstream analogue. If the
  ~1000 primitives/token replay from a captured graph at ~0 CPU cost, TP-4's transport term
  collapses to the fabric term (~0.03 ms) and the 1.15x lands intact. **Hazard, pre-registered:
  §6.6 — graph capture is precisely where the IPC/peer-mapping path aborts
  (`custom_all_reduce.cuh:455 'invalid argument'`), and `vllm#54331` reports sm_120 NVFP4
  dying under sustained load with graphs on.** Capture the copies only; do not import IPC
  handles.
- **A fused pull collective at hidden 4096**, per §6.13's handed-over protocol, if capture is
  blocked. This is where the pre-registered bug list becomes ours: unsigned slot counter,
  generation lanes with all four validated, 32 incoming bytes per 16 payload, grid capped at
  36 blocks, 16 blocks x 512 threads to cover the PCIe round trip, no payload sentinels, and
  routing by MEASURED row count with the two-part accept test (beat the baseline by 1% AND
  0.25 us) because *"a static topology label is insufficient"* (§6.4). And this is the arm
  that DOES need the 3-key `ForceP2P` override, because it dereferences peer pointers from a
  kernel — which is why `peer-read-probe.cu` is banked now rather than then.

**The route to the 100 bar, with every multiplier named:**

```
peer-pull TP-4 served-class            36.1-48.7
x spec composition 1.64-2.33 (arm 3)   59.2-113
x matvec efficiency ~1.25              74.0-142     (census: bf16-mmv + moe = 65.3% of GPU
                                                     time at 57-70% efficiency vs q38's 87%,
                                                     matvec-20260831/LANE.md)
with graph-captured peer copies         add back most of the 1.15x TP-4 prior
```

**The bar is reachable, and not by transport alone.** Transport is the term that unblocks
TP at all; TP-4 is the term that makes the card count pay; spec composition is the biggest
single multiplier and is currently a REFUSAL, not a missing capability. Sequence accordingly.

### One prefill note, kept honest

`RESEARCH.md` §6.5 measures sequence-parallel/async-TP at **1.35-1.66x above ~640-896 tokens**
on 4x PRO 6000 at hidden 4096 — our exact configuration — and §6.9 settles overlap as a **NO
for decode** (`K >= sR/(2B)` gives `K >~ 19,000` tokens on PCIe against `K = 1` at batch-1
decode; three measured thresholds agree; a direct trial on 4x PRO 6000 measured
*"overlap 0.000 ms"* and **+14.4% E2E regression**). And §1.3d's largest single lever in that
whole corpus was not a collective at all: a **cache-page geometry** change worth **+30.5%
prefill** (512/512 split vs automatic 2304). `LAW:prefill-decode-ratio-diagnostic` says our
prefill runs in decode's launch-structure class. **So: EP grouped prime first (already built,
unblocks TTFT), then page geometry, then SP/async-TP — never overlap at decode.**

---

## Stage 5 — HEALTH.sh, banked for every future box window

`HEALTH.sh` in this directory. It captures, per box, at boot, before any measurement — every
item is a documented case of 100% reported utilisation with clean logs and no error:

| check | why | source |
|---|---|---|
| power limit ACTUAL vs max, per card | a card sat at **400 W of 600 W** from two persistent config sources, throttling dense-GEMM prefill **25.3%** with memory bandwidth unaffected | §6.14 |
| `clocks.sm` under load + temp + throttle reasons, alarmed on the COMBINATION (power at cap AND `clocks.sm < 1 GHz` AND temp < 50 C) | a 600 W WS card can drop to **577-675 MHz at 34-37 C** with `sw_power_cap` asserted, ~1/10 to 1/20 of its FP32 throughput, across three machines, both OSes, four driver branches. **Never flash VBIOS in-fleet** — it does not fix it | §5.4 |
| PCIe link gen + width, NEGOTIATED vs MAX | a card sat at **Gen2 x16 for 3.5 hours of production** — ~8 GB/s instead of 64 — while *"nothing logged an error. The card reported 100% utilisation the whole time"*. Lock links only AFTER verifying maximum | §3.7 |
| P-state, and a P8 wait before any timing | an idle transition moves the same executable graph **54.2 -> 56.4 tok/s**; *"P8 waiting is a required benchmark-normalization condition"* | §6.14 |
| CPU/NUMA affinity per GPU, flagged when a mask falls outside the host's CPU range | a `strtok()` race cost **25%** of all-reduce bandwidth (32.47 -> 24.23 GB/s) with the only signature an out-of-range affinity mask; *"the out-of-range 128-159 on a 128-CPU host is the clearest signature"* | §3.5 |
| BAR1 total (expect 98304 MiB) | some BIOS defaults to 256 MB BAR1, *"which cripples P2P performance"* | §2.10 |
| IOMMU mode from `/proc/cmdline` + `dmesg`, and ACS `ReqRedir` count from `lspci -vv` | NVIDIA: CUDA *"does not support IOMMU-enabled PCIe peer-to-peer memory transfer... the IOMMU must be disabled on Linux bare-metal systems to prevent silent device memory corruption"*. ACS ON forces P2P through the root port: measured **~50 -> ~103 GB/s** with it off. **We have never captured either per box**, and our fleet is VMs where *"disabling ACS is not an option"* | §2.8 |
| loaded driver params (`/proc/driver/nvidia/params`) + version | verify the LOADED params, never the file; the sysfs node *"reads empty even when active"*. Also pins the 580.167.08 5-key caveat against the box's actual driver | §2.3b |
| `topo -m` and `topo -p2p r|w|a` | recorded as a TIER-1 signal only. `-p2p a` returning `NS` is EXPECTED (`NativeAtomicSupported=0` on every SM120 pair), not a fault | §2.1, §2.4 |
| `peer-read-probe` — a `simpleP2P`-class byte-validating KERNEL peer read, both directions, 4 B .. 64 MiB | the ONLY check that detects the SysMem-staging default and the *"peer access reports Yes, `cudaMemcpyPeer` runs at 26 GB/s, but kernel peer-reads return zeros"* class. `topo -p2p r` = OK and a healthy `cudaMemcpy` in BOTH broken cases | §2.3b, §2.4, §2.10 |

`HEALTH.sh` writes one JSON-ish receipt per box into the window's directory and exits non-zero
on any hard failure, so a window cannot open on a degraded box and bank the degradation as a
result. `ncu` is deliberately absent: §6.14 records that *"profiling every rank deadlocks
(the profiler serialises the observed kernel while its peers wait)"* and that reading the
source found all four defects the profiler found none of.

---

## Cross-references

- Research: darklanes `research/pro6000-multicard-research-20260901/RESEARCH.md` (`origin/main`)
- Prior TP lane: `../tp2-20260831/{LANE.md,SHARD-MAP.md}`
- The banked box battery this lane corrects the reading of: `../tp2-battery-20260831/RESULTS.md`
- The orthogonal count-cutting lane: `../ep-diet-20260831/LANE.md`
- The efficiency lever in the 100-bar arithmetic: `../matvec-20260831/LANE.md`
- Flags: `docs/FLAGS.md` rows `MEMRA_GLM5_TP`, `MEMRA_GLM5_TP_TRANSPORT`
