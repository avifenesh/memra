# kv-vmm on the RTX PRO 6000 Blackwell Max-Q: pre-registration of the Max-Q deciding cell (DAY37 O1 replay)

Written 2026-11-04, before any Max-Q boot. Nothing below changes after a number is seen. This is the kit-side text of
the dated DAY37 addendum the extension needs (`research/spill-b-20260919/DAY37.md`, "a new dated addendum before any
Max-Q boot"); it has to be committed and pushed into DAY37.md before the first boot, as DAY37 section 1 requires of
every addendum.

## M.1 What the cell decides

The `kv-vmm` serving default (`docs/decisions/KV-VMM-SERVING-DEFAULT.md`) arms only on the full-power RTX PRO 6000
Blackwell Workstation Edition, because its deciding cell ran on that card (DAY37 2.5 and 2.6, PROMOTE-ELIGIBLE at
DAY37.md:704-708). This cell reads the same rule, 1.7, for the **RTX PRO 6000 Blackwell Max-Q Workstation Edition**
class (device name `NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition`). It decides the Max-Q class only. No
timing crosses cards or boxes (1.7), and nothing in it is compared with a Workstation number.

## M.2 The tree and the arms (unchanged from the Workstation reading)

- Lane binary from the r4 source `c6f9282c26e71349c0263b88dc7beb59c4a57797` (`memra-server`, `kv-tier-gate`,
  `vmm-call-cost`, one cargo invocation as `pro-single-day37/chain.sh:48-49`). At r4 the door reads
  `MEMRA_KV_ALLOCATOR=vmm` and `MEMRA_KV_VMM_GROW` on any card, and the Max-Q and the Workstation are one class for every
  device-keyed default in that tree (`HardwareTarget::from_device_name`), so the r4 binary runs the same program on
  both editions.
- A6's main binary from `17dceb9817d050d0960a8e2d0be372921bfbee80`.
- Harness and reader from `3476ea4dca27c4c3c738017b8429f507389f1223` (the second Workstation sitting's tree), with the
  portability ports listed in `DEVIATIONS.md`. Arms `pooled` (door unset) and `vmm` (`MEMRA_KV_ALLOCATOR=vmm`), the fault
  arms, `main`, the boot names, orders, N, prompts and clients are the registered ones.
- Builds: release, default features, `MEMRA_CUDA_ARCH=120a`, nvcc 13.1 (the release the Workstation box auto-detected),
  rustc 1.97.1 (the tree's pin). Hashes in `PROVENANCE.txt`.

Why this tree and not HEAD: HEAD deleted `MEMRA_KV_ALLOCATOR` and `MEMRA_KV_VMM_GROW`, so the vmm arm cannot be armed on
the Max-Q at HEAD without a temporary seam. The r4 replay is the evidence the Workstation default rests on.

## M.3 Placement: inline, pinned

Every vmm boot and gate cell of this cell runs with `MEMRA_KV_VMM_GROW=inline`, as addendum F pinned it for the second
Workstation sitting. A3 (i) is therefore read on 1.6's inline branch (per grow event `owner_us` over the stream vmm
boots, N >= 20, p99 <= 500), and A5-MAPPER prints `N/A (inline: no mapper)` with its digest comparison (addendum F
item 2). Stage 0 (`vmm-call-cost --reps 20 --queue-ms 60 --extents 1,4,16,64`) runs as a recorded reading in the timed
slot; what it prints does not change this cell's placement. Reasons: the Workstation class default in code is inline;
the extension this cell can justify keys the Max-Q to inline with its own source label; stage 0 read 55.3 and 271.2 us
busy p95 on two boxes of one class, on both sides of the 100 us line, so it is reported, not used to choose.

## M.4 The card and its regime

- One card, chosen at run time (`CARD=<nvidia-smi index>`), pinned with `CUDA_DEVICE_ORDER=PCI_BUS_ID
  CUDA_VISIBLE_DEVICES=$CARD` for every step. The runner refuses unless that card reports the Max-Q name, under
  2000 MiB used and no compute process.
- Its `power.limit` is recorded with the receipts (`card.csv`, `preflight/card-*.csv`, every 250 ms telemetry file).
  On the box the caps read 250, 250, 280 and 280 W (indices 0 to 3; index 2 reads 300 W max and default). The owner has not changed
  them and the kit never does. A3's timing is read at the limit the card had.
- The box has four cards. The untimed half runs while the other three train; that regime is recorded once at the
  start and end of each part (`preflight/box-*.csv`, compute processes by bus id and memory). Its clauses are pass/fail
  and are read as they read.
- The timed half (stage 0, the ten stream pairs) runs only when no card of the box has a compute process (the
  registered box-wide idle check, before the part and before every boot). GPU 0 is avoided for it (airflow-limited).
- The Max-Q guard stops a part at >= 92 C with fan >= 95 % on 3 consecutive 10 s samples, or at >= 95 C. A stopped
  part has no outcome for the step it cut; that step is asked again whole (DEVIATIONS.md).

## M.5 Which clause runs where

| part | steps | clauses |
|---|---|---|
| `run.sh` (a gap, beside training) | A2 `grow-32768-maxq`; gate set `gates-maxq-pooled`, `gates-maxq-vmm`; the 19 untimed boots (mix spec and plain, O1 and O2; `off-lane`, `fault-mapper`, `off-main`; `burst-g2/l64/boff` pooled and vmm; `fault-ensure`, `fault-build1`, `fault-build64`) | A1-GATE, A1-MIX, A2, A4, A5, A6, A7 |
| `run-timed.sh` (a quiet box) | stage 0 (a reading); `stream-O1-k` pooled then vmm and `stream-O2-k` vmm then pooled, k = 1..5 | A1-STREAM, A3 (i), A3 (ii) |

Both parts write one receipt root; the reader (`day37-read.py pro6000-maxq <root> gates-maxq`, the corrected reader of
addendum F) reads it whole, and `day37-verdict.py` applies 1.7 to its lines. Until the timed half has run the reading
is `INCOMPLETE (owed: A1-STREAM,A3-i,A3-ii:both)`, unless a correctness clause already failed, which makes it
`FAIL (no reading)` whatever A3 reads.

## M.6 The VERDICT line

One line, in the reader's own form:

```
DAY37 VERDICT card=pro6000-maxq rule=1.7 A1=<PASS|FAIL|MISSING> A2=.. A5=.. A6=.. A7=.. A3-i=.. A3-ii=.. A4=..(reading) lines=<n> reader_cards=pro6000-maxq -> <PROMOTE-ELIGIBLE|DELETE|FAIL (no reading)|INCOMPLETE (owed: ...)>
```

A1 is every A1-GATE cell (`admit-mem-burst`, `fault-default`, `fault-plain`, `hit-off`, `hit-on`,
`identity-default-on`, `serve-smoke`, `spec-ctx-edge`, `twin27-off`, `twin27-on`), the four A1-MIX lines and
A1-STREAM. A5 is A5-MAPPER (met by `N/A (inline: no mapper)`), A5-ENSURE, A5-BUILD1 and A5-BUILD64. A7 is the g2, l64
and boff lines. A3-ii is the `order=both` line; the per-order lines are reported. A4 is printed and not ruled on (1.7).
Dry-run against the Workstation receipts (both sittings, assembled as DAY37 2.6 assembled them) prints
`-> PROMOTE-ELIGIBLE`, the hand reading of DAY37.md:704-708.

## M.7 What follows from the reading

- PROMOTE-ELIGIBLE: the Max-Q extension lands as its own change (a Max-Q predicate beside
  `is_pro6000_workstation`, which stays as it is; `kv_vmm.rs` arms on either; `kv_vmm_placement_for_device` returns
  inline with a Max-Q source label; the tests and doc rows of the scope entry), with this cell's receipts.
- DELETE or FAIL (no reading): the Max-Q keeps the generic (pooled) program; the decision record names this cell as
  the rejection. A FAIL quotes its cause per 1.9; a rerun is a new cell under a new name.

## M.8 Quoted from the Workstation pre-registration (DAY37.md 1.6 to 1.9, unchanged at HEAD)

> ### 1.6 Acceptance clauses (the deciding cell, both cards)
>
> Workloads: the day-26 mix (`run-day26-cell.sh`, arms (i) open, (ii) bounded, (iii) continuation, three lengths,
> N=5 per arm per length) on the spec path and on the plain path (`MEMRA_SERVE_SPEC=0`), and the day-34 burst cell
> (`day34-client.py`, door `MEMRA_ADMIT_BY_MEMORY` OFF and ON at 32768). Arms: `pooled` (door unset) and `vmm`, one
> binary, one boot per arm, both orders (pooled then vmm, vmm then pooled). Models and cards as DAY36 1.2 (the 9B at
> `MEMRA_CTX=65536` on the 5090; the 27B at the checkpoint's 262,144 on the target card).
>
> - **A1 identity.** Completion digests `vmm == pooled` on every request of every mix cell, both orders, both paths.
>   Each of these gates reads its ALL GREEN line with the same ok count on both allocators: `serve-smoke`,
>   `KV-HOST-SPILL IDENTITY GATE`, `KV-HOST-CONTRACT-FAULT GATE` (default and plain), `SPEC-ON-CACHE-HIT GATE` (OFF and
>   ON), the twin gate, `prefix-newest-turn-fits`, `ADMIT-MEM BURST GATE`, `SPEC-CTX-EDGE GATE`. The CLI binaries
>   (`kernel-check`, `run-gen`, `run-spec`) do not reach the serving door (1.3), so their lines are the pooled program
>   on both arms by construction; they run once on the lane binary as the battery requires and prove nothing about
>   the arm, and the serving-shape gates above carry the identity proof. (This refines `KV-RESIDENCY-DESIGN.md` (b)'s
>   byte-cell list, which named them before the arm's scope was known.)
> - **A2 G1 grow series.** `kv-tier-gate --case grow --kv-allocator vmm-ondemand` (new case, one cache, the gate's
>   tokenwise program, collector-locked): one session decoding across at least 5 granule boundaries of every K/V plane
>   in one process. At every crossing, read after a stream sync with no other allocation between the two reads,
>   `driver_free_before - driver_free_after == extent_bytes` exactly; after the cache drops and the graveyard reaps,
>   `driver_free == driver_free_before_the_cache` exactly (drift 0); tokens and logits bit-identical to the same case
>   on `--kv-allocator pooled`.
> - **A3 stall.** (i) Per grow event, `owner_us` p99 <= 500 over at least 20 events per card (inline placement), or the
>   ensure point's wait p99 <= 500 us (helper placement). (ii) Serving, same-window interleaved, both orders, N >= 5 per
>   arm, the day-34 burst at open output 8192 and the day-26 mix: TTFT p50 (vmm / pooled) <= 1.05, ITL p99
>   (vmm / pooled) <= 1.05, TPOT p50 (vmm / pooled) <= 1.02, output token throughput (vmm / pooled) >= 0.98; every
>   median with its N and the 250 ms regime.
> - **A4 accounting.** Per request, from the retire line: `booked` equal on both arms; `mapped_at_retire - used <=
>   planes x (granularity + SLACK x tok_bytes)` rounded up to whole granules, where `used = (P + G) x tok_bytes` per
>   plane. Reported, no bound (these are the benefit readings): retained device bytes at idle after each mix
>   (`cuda_driver_free_bytes`, `cuda_pool_cached_bytes`, parked sessions' mapped bytes) per arm, and the admitted
>   concurrency of the burst per arm.
> - **A5 fault.** `MEMRA_KV_VMM_FAULT=grow:<n>` on the day-34 burst and on the mix: the reclaim-retry line is present;
>   with the retry forced to fail too (`grow:<n>,<n+1>`), the session outcome is the registered one (park when it has
>   emitted nothing, the typed error when it has); peers' digests equal the no-fault boot; zero
>   `CUDA_ERROR_ILLEGAL_ADDRESS` lines, zero worker panics, zero process exits.
> - **A6 door OFF.** An OFF boot of the lane binary against the main binary (`17dceb981`) on the mix: digests equal on
>   every request, zero `[kv-vmm]` lines apart from the boot line reading `door=OFF`.
> - **A7 booking.** On every armed burst boot: zero `CUDA_ERROR_OUT_OF_MEMORY` lines, zero 503s, zero un-injected grow
>   failures, and at every `verdict=admit` line the reading after `vmm_owed` covers the request's `required`.
>
> CPU (before any card): unit tests per arm of the mechanism (extent arithmetic from rows and tail pad; grow, trim,
> cancel-on-resume and reap state machine against a fake driver; the owed-growth reduction is the identity when
> unarmed; the ambient scope count; the census tests of 1.2's ensure points); `cargo test -p memra-kv`, `-p
> memra-server --tests`, `-p memra-engine --lib`, clippy `-D warnings`, fmt, check-flags. The spill review patterns
> (move-then-match, parked requests and the idle wait, gate literals, happy-path release, borrowed sources, budgeted
> bookings, guard identity) are read against every new state machine before the target-card sitting, and the reading
> is written into section 2.
>
> ### 1.7 The decision rule (stated, not chosen) and what each card decides
>
> Per card class, the rule reads:
>
> - **PROMOTE-ELIGIBLE** when A1, A2, A5, A6 and A7 PASS and A3 (i) and (ii) PASS. The benefit readings of A4 go with it.
> - **DELETE** when every correctness clause (A1, A2, A5, A6, A7) passes and A3 fails: the arm is correct and costs
>   more than the bound. The owner may weigh A4's retention against the stall instead; the rule does not.
> - **FAIL (no reading)** when a correctness clause fails: the cause is quoted and the arm is revised under a new
>   pre-registration before the decide-by, or recorded as failed at the decide-by.
>
> The 5090 reading decides the 5090 class only; the target card's decides the target class (the per-hardware rule in
> `CLAUDE.md`). Neither card's reading is compared with the other's, and no timing crosses cards or boxes. Lead ruling
> 7's 8k cycles control is revisited only if the owner promotes the door, as ruling 7 says; it is not a clause here.
>
> ### 1.8 Order of work
>
> 1. Stage 0 probe, 5090 (this rig, `/tmp/memra-5090.lock`).
> 2. The mechanism in `memra-kv` and the engine seam, with the CPU tests.
> 3. The server wiring, the owed-growth reduction, receipts, the fault door, the census tests.
> 4. The grow case in `kv-tier-gate`.
> 5. 5090 cells: A2, A1, A5, A6, A7, A3, A4, in that order, each under the lock.
> 6. The target-card sitting: stage 0 on the card, then A2, A1, A5, A6, A7, A3, A4. Prepared in full (scripts
>    committed and dry-checked, the expected duration and the box needs) before the lane reports `NEED TARGET CARD`.
>
> ### 1.9 Failures
>
> Causes are quoted from captured stderr, never inferred. An OOM is a captured line plus `nvidia-smi` compute-apps at
> the time. A rerun happens only as a whole cell under a new name, with the reason in section 2.

## M.9 Quoted from the addenda this cell reads under

Addendum B (DAY37.md 1.11), A3 (ii)'s instrument and the burst cells, verbatim:

> A3 (ii) is read from `day37-stream-client.py` (streaming, `stream_options.include_usage`): per boot 32 greedy
> requests over 5000-char slices of `docs/SERVING.md`, 8 in flight on the 5090 and 16 on the target card,
> `max_tokens` alternating 256 and 2600 (a 2600-token request crosses at least one granule boundary of every plane on
> both models), 250 ms telemetry. Same-window interleaved: per order 5 boot pairs (O1 pooled first, O2 vmm first), one
> binary; each boot prints `DAY37 STREAM ...`. The A3 (ii) ratios use the per-arm medians of the per-boot readings
> over the 10 boots of each arm (N = 10 per arm, 5 per order), and are also stated per order. The mix and burst boots
> contribute E2E and throughput as readings, outside the bound. The stream rows' text digests join A1: equal across
> arms, row for row.

> The A7 burst boots use `day33-client.py --chars 5000 --skip-long --burst 64` at open output 8192 (the day-35 `G2`)
> and 32768 (`L64`) with `MEMRA_ADMIT_BY_MEMORY=1`, and the same client with the admission door off (`--burst 64`),
> one boot per arm and shape

> **A5's fault lists.** `mapper:all` alone on the spec mix (the owner-behind path must meet A1 on the mix);
> `mapper:all,ensure:1,ensure:2` on `G2` (the first owner ensure grow and its retry both fail); `build:1` on `G2` (one
> construction grow fails: the retry line, then the request succeeds); `build:1` to `build:64` on `G2` (every early
> construction fails: the admission refusal).

> **A2's command.** `kv-tier-gate --case grow --kv-allocator vmm-ondemand --context 32768 --tiers host
> --same-program` with the 27B artifact on both cards

> **A6's baseline binary.** `memra-server` built from main `17dceb981` in a detached worktree.

Addendum F (DAY37.md 1.15), the reader corrections this cell's reader carries, verbatim:

> 1. A3 (i) reads each vmm boot's placement from its boot line (`grow=inline` or `grow=helper`). Inline: per grow event
>    `owner_us` over the stream vmm boots, N >= 20, p99 <= 500 (1.6's inline branch). Helper: the ensure walls and the
>    owner grows that waited, as before.
> 2. A5-MAPPER under the inline placement prints `N/A (inline: no mapper)` with the digest comparison: the fault
>    injects nothing there and the owner-behind path it targets is every grow, which A1 reads. Helper: as before.
> 3. A5-ENSURE compares only rows whose `prompt_sha256` is equal in both boots, prints how many rows it excluded for a
>    changed prompt (a later turn of a conversation whose earlier turn took the injected error carries that turn's
>    different answer, so it is a different request), and prints the outcomes whole.
> 4. A5-MAPPER's and A6's pooled side is `off-lane` when that boot exists (the lane binary, door unset, the mix).

> **The `main` arm clears every lane-only name.** `boots.sh` runs the main binary with `MEMRA_KV_ALLOCATOR`,
> `MEMRA_KV_VMM_GROW` and `MEMRA_KV_VMM_FAULT` unset, whatever the sitting exported.

Addendum H (DAY37.md 1.17, registered for the 5090 r5 cell after the 3476ea4dc tree; quoted for its rule only), the
pair unit this kit keeps when a window cuts a pair:

> A stream pair (its two boots in the registered order) is one unit: the yield falls between pairs, not inside one.

## M.10 Inputs

- Model: `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`, 15,705,922,304 B, sha256
  `1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a` (checked by the runner before the first boot of
  each part; the source copy was rehashed on 2026-11-04 and matches). `MEMRA_CTX` unset (the checkpoint's 262,144),
  `MODEL_KEY=q38`, `HOSTGATE_MB=256`, `STREAM_CONC=16`, the twin gate on the same 27B.
- serve-smoke's spec, gemma4 and Q35 arms stay absent at their default paths, as on both Workstation sittings (the
  runner refuses if any exists).
