# WP-B day 13: prefix eviction must credit admission and the driver (#346, #445, #523 item 4)

Repository: **avifenesh/memra**, branch `lane/spill-b-20260919`, merged with `origin/main` `ea08bc7f8` at
`1cfdac6eb`. Commits this day: `b351d7db9` (the gate), `f4350c241` (the fix), `a1a2239e3` and
`678cc83ea` (gate corrections after the first two target-card cells refused), `ebda75396` (TESTING.md),
plus this record and the receipts. **Not merged, released, deployed, or a default promotion.** Every
target-card cell is N=1, `executed-not-qualified` in the collector's vocabulary; the card ran at its
600 W cap; no timing is compared between binaries or boots. Serving-shape evidence for the G1 work; the
VMM door (`--kv-allocator vmm`, decide-by 2026-10-04) is untouched.

## The defect, read from the code and the issues

- `[admit-oom] reclaim-on-defer` (`worker.rs`, the admission phase) evicts every unleased prefix entry
  when a request does not fit, then re-reads `admission_headroom` in the SAME tick. A `PrefixEntry`'s
  planes are `CudaSlice`s; dropping them is a stream-ordered `cuMemFreeAsync` into the caching pool. The
  pool's `USED_MEM_CURRENT` keeps counting the bytes until the owning stream is fenced, and after the fence
  they sit mapped in the pool where `cuMemGetInfo` cannot see them (the boot pins `RELEASE_THRESHOLD` to
  `u64::MAX` for graph-launch speed). `effective_free = driver_free + (reserved - used)` therefore moves by
  nothing in the tick that evicted.
- The receipt line compounded it: `free_before_reclaim` was captured AFTER the prefix eviction and its
  re-read, so `effective free A -> B` could only ever describe the parked-session loop that follows. That is
  the literal shape of #346, `effective free 69204MB -> 69204MB` on a 43.5 GB eviction.
- An idle-box arm already existed (`oom_teardown_fence` + `trim_model_device_pools("admission-drain")`,
  `[admission] reject averted`), but only when `active.is_empty()`. A busy box never reaches it: the
  deferred request requeues every tick behind an entry that is already gone, which is the #445 shape
  ("every later prefill dies Overloaded" on a 24 GB card after one 2.53 GB entry).
- `/admin/trim` is a deployment-binary surface (`MEMRA_ADMIN_ADDR` is fatal in this build), so the stock
  server has no trim endpoint; the only serving-shape eviction trigger is admission pressure itself.

## The fix (`f4350c241`, `crates/memra-server/src/worker.rs`), the first of the two options

Implemented: **eviction returns the bytes to the driver in the tick that credits them.** Not implemented:
the alternative (admission accounting for pool-held bytes as reclaimable); `effective_free_bytes` already
counts pool-cached bytes, so that alternative was the status quo for the pool-allocation gate and could
never help the driver-side allocations (#445's admit-trim rung, cuBLAS workspaces, graph instantiation).

- `pool_readings` snapshots every model-owned device's `(reserved, used, driver_free)` BEFORE the eviction;
  `free_before_reclaim` and `px.total_bytes` are taken at the same point, so the reclaim-on-defer line now
  reports the eviction it names.
- `settle_reclaimed_prefix_bytes` runs after the eviction and before the re-read: `synchronize_model_devices`
  (the fence `oom_teardown_fence` already uses), then per device `cuMemPoolTrimTo(used + cached_before)`
  (`reclaim_settle_keep`, `None` when the reclaim added nothing to that device's cache). Exactly the
  reclaim's cached gain goes back to the driver; the blocks parked before the reclaim stay parked, so the
  graph-launch pin is not disturbed. One line per device, bytes only:
  `[admit-oom] reclaim settle (reclaim-on-defer): dev0 evicted_prefix_bytes=... pool_cached_gain_bytes=...
  trim_released_bytes=... driver_free_bytes A -> B pool_reserved_bytes A -> B pool_used_bytes A -> B`, with a
  `pool_retained_bytes=...` tail when the trim released less than the gain (a live neighbour shares the
  chunk); a shortfall is printed, never inferred away, and stays counted as pool-cached headroom.
- No numeric program change (memory accounting only, no kernel, stream program or byte layout touched),
  no new `MEMRA_*` read (flags census clean), the VMM door untouched. CPU arm:
  `worker::tests::reclaim_settle_returns_only_the_reclaims_gain`; the existing wiring tests around the
  shed and the memra#365 flush still pass.

## The gate (`tools/prefix-evict-reclaim-gate.py`, `678cc83ea`)

Serving shape on the real `memra-server`, one card, two boots per cell, plain path (`MEMRA_SERVE_SPEC=0`),
`MEMRA_PREFIX_CACHE_MB=6144`, `MEMRA_CTX=65536`, greedy, under the collector's inherited canonical lock
(`--external-lock @COLLECTOR_LOCK_FD@`, lead ruling 5; `tier-lock-proof.py` verified before any port binds).

- Calibration boot (no pressure): P1 (text A, 22,000 pseudo-random words) seeds entry E1; P0 (a short
  prompt, 400 greedy tokens) is the busy peer; P2 (text B) runs beside it. Records effective free after P1,
  E1, E0 (the peer's own seed), the two long prompts' `[admission] request cost` lines, and the message
  digests. `/metrics` publishes only after the first retire, so F0 is defined as effective free after P1
  plus E1.
- Measured boot: a ballast child (one plain `cuMemAlloc` through `libcuda` ctypes, started before the boot,
  held until the server stops, footprint measured with `nvidia-smi` before and after) sized into the window
  `(f2c - required(P2), f2c - required(P2) + E1 + E0]`, so that P2 does not fit beside the busy peer and fits
  after a true credit of E1 + E0. With spec disabled the boot prints the static-floor line (asserted) and the
  plain reserve is `min(cost, 1536 MiB)`, so `required(P2)` is known from the calibration. No admission door
  is set in either boot. P2 is sent while P0 is still decoding (`busy_overlap_s` recorded), so the idle-box
  drain arm cannot mask the tick.
- Verdicts, bytes from the server's own lines and `/metrics`: V1 the reclaim-on-defer line credits at least
  E1 minus 8 MiB; V2 P2 is admitted in that tick (no `VRAM defer` after the reclaim line, no
  `reject averted`, HTTP 200); V3 the settle line moves driver free and `trim_released_bytes` by at least
  E1 minus one 2 MiB granule with no `pool_retained_bytes`; V4 the (P1, P0, P2) message digests of the two
  boots are identical. Exit 0 PASS, 1 FAIL, 2 `REFUSED: ...`.

## Target-card cells (one RTX PRO 6000 Blackwell, 96 GB, 600/600 W, `--rig pro-single`, `/tmp/memra-gpu.lock`)

Binaries built natively in `/root/wt-b` (receipts `build-main/`, `build-fix/`):

| Arm | Source | `memra-server` SHA-256 |
| --- | --- | --- |
| base | `ea08bc7f8` (`main`) | `6fc3ec03e708951e34172ae53cf0e14a20fbb41b170df85ce1101f9ccf124e57` |
| fix | `f4350c241` | `24d6b453e930defec64d7f415e20b89b18a85a1211d2922b9711973adfc06da6` |

Artifact `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` (the same one every earlier B cell used). Prompts realized at
48,332 (P1) and 48,266 (P2) tokens; E1 = 1,592,160,256 B; E0 = 159,001,600 B (Qwen3.8's fixed recurrent
state makes even a 71-token entry 159 MB).

Two cells REFUSED before any verdict and stay on disk as refused, never relabelled:

- `gate-main`: `REFUSED: no [admission] request cost line for the long prompt in the calibration log`. The
  server's admission cap is prompt + max_tokens + 64, so the gate's exact-ctx lookup found nothing. Fixed
  in `a1a2239e3` (nearest cost line at or above the prompt length; F0 from the post-P1 publish; identity
  over the whole message object because 8 greedy tokens land in the thinking channel and leave `content`
  empty).
- `gate-main-rerun`: `REFUSED: the measured boot never reached reclaim-on-defer; the pressure arithmetic
  did not bite`. `MEMRA_ADMIT_RESERVE_MB=72627` produced no pressure because the plain path charges
  `min(cost, floor)`: the reserve door cannot make a 96 GB card tight for a plain request. Fixed in
  `678cc83ea` (ballast). Its two boots did produce identical message digests
  (`aa6cc3291b9816468b7cd1d5b03b08032f2e20aa873112eb15068d09231b79c9`), the first V4 receipt on the base.

## Verdicts, same card, same artifact, same prompts, same gate source (`678cc83ea`)

### Base, `main` `ea08bc7f8` (`gate-main-rerun2`, collector status `failed`, gate exit 1)

Ballast footprint 74,850,500,608 B (requested 74,262,450,904 B plus the child's CUDA context), inside the
calibration window `(73,386,869,976, 75,138,031,832]`; `required(P2)` = 4,701,000,000 + 1,610,612,736 =
6,311,612,736 B. The measured boot's reclaim-on-defer line and the gate's verdict, verbatim:

```text
[admit-oom] reclaim-on-defer: evicted 2 prefix entries + 0 plain + 0 spec + 0 dspark parked sessions (global LRU); effective free 6603MB -> 6603MB
PREFIX-EVICT-RECLAIM: entry_bytes=1592160256 reclaim_credit_bytes=0 driver_free_delta_bytes=none trim_released_bytes=none p2=admit-same-tick busy_overlap_s=21.655 identity=aa6cc3291b981646 V1=FAIL V2=ok V3=FAIL V4=ok -> FAIL
```

What the base actually did, from `/metrics` rows of the measured boot (bytes):

| Row | driver free | pool reserved | pool used | pool cached | prefix bytes (entries) |
| --- | ---: | ---: | ---: | ---: | ---: |
| after P1 | 5,129,961,472 | 21,407,727,616 | 21,045,950,420 | 361,777,196 | 1,592,160,256 (1) |
| before P2 (P0 active) | 4,827,971,584 | 21,709,717,504 | 21,685,840,360 | 23,877,144 | 1,751,161,856 (2) |
| final (E2 resident) | 2,980,380,672 | 23,555,211,264 | 21,522,796,392 | 2,032,414,872 | 1,590,200,320 (1) |

- V2 held on the base: P2 was admitted in the tick that evicted E1 and E0 (no `VRAM defer`, no
  `reject averted`, HTTP 200, 17.4 s). So on this card and driver the pool's `used` counter fell at the
  free call, and the pool-side gate saw the credit. The line printed `6603MB -> 6603MB` because its
  "before" was captured after the eviction: the #346 text is the receipt's capture point, not an
  unmoved gate, at least on this shape.
- V1 and V3 failed on the base: the credit the line claims to report is 0, and no byte reached the
  driver. The pool ended the boot holding 2,032,414,872 B of cached blocks (the evicted 1,751,161,856 B
  plus the peer's retired state) while it had grown by 1,847,483,648 B from the driver to serve P2's
  prime and E2; driver free fell from 4,827,971,584 to 2,980,380,672 B across a request that evicted
  more than it inserted. That is "frees to the caching pool, not to the driver" (#445) in bytes on the
  target card.
- V4 held: calibration and measured boots hash `aa6cc3291b9816468b7cd1d5b03b08032f2e20aa873112eb15068d09231b79c9`
  over the three messages (P1 8 tokens, P0 400 tokens, P2 8 tokens, all `finish_reason=length`).

### Fix, `f4350c241` (`gate-fix`, gate source `678cc83ea`: the first fix cell, FAILED on the gate's own extra clause)

Same ballast (requested 74,262,450,904 B; footprint 74,850,500,608 B), same window. The settle line, the
reclaim-on-defer line and the verdict, verbatim:

```text
[admit-oom] reclaim settle (reclaim-on-defer): dev0 evicted_prefix_bytes=1751161856 pool_cached_gain_bytes=1751161856 trim_released_bytes=1610612736 driver_free_bytes 4827971584 -> 6438584320 pool_reserved_bytes 21709717504 -> 20099104768 pool_used_bytes 21685840360 -> 19934678504 pool_retained_bytes=140549120 (live neighbour or fragmentation; counted as pool-cached headroom, not driver free)
[admit-oom] reclaim-on-defer: evicted 2 prefix entries + 0 plain + 0 spec + 0 dspark parked sessions (global LRU); effective free 4852MB -> 6603MB
PREFIX-EVICT-RECLAIM: entry_bytes=1592160256 reclaim_credit_bytes=1751000000 driver_free_delta_bytes=1610612736 trim_released_bytes=1610612736 p2=admit-same-tick busy_overlap_s=21.659 identity=aa6cc3291b981646 V1=ok V2=ok V3=FAIL V4=ok -> FAIL
```

- In the tick that evicted E1 and E0 (1,751,161,856 B): the fence made the pool's cached gain exactly the
  evicted bytes; the trim to `used + cached_before` released 1,610,612,736 B (768 granules) and driver free
  rose by exactly that, 4,827,971,584 to 6,438,584,320 B: more than E1 (1,592,160,256 B). The reclaim line
  reports the eviction it names, `4852MB -> 6603MB`, a 1,751 MB credit.
- The pool retained 140,549,120 B: the busy peer's own 159 MB seed E0 was allocated beside that peer's live
  session state, so its chunk has a live neighbour and `cuMemPoolTrimTo` cannot release it. The server
  prints those bytes as `pool_retained_bytes`; they stay counted by `effective_free_bytes` as pool-cached
  headroom (which is where they are usable) and are never reported as driver free.
- V3 failed here only because the first gate revision additionally demanded zero retained bytes. That clause
  went past the criterion the lead set ("eviction does not move driver free by the entry's bytes" is the red
  statement; the green one is that it does) and past the task's own allowance for the case the pool cannot
  give back ("the admission must account for pool-held bytes rather than pretend they are free", which the
  settle line does). `ec473770f` makes V3 the stated criterion and carries `pool_retained_bytes` in the
  verdict line as a reported figure. This cell stays on disk as failed; the final pair below reruns BOTH arms
  on the corrected gate so the two quoted verdicts share one gate source.
- V4: the fix's three messages hash `aa6cc3291b9816468b7cd1d5b03b08032f2e20aa873112eb15068d09231b79c9`,
  the same digest as the base's two boots. The tokens are byte-identical across binaries: the numeric
  program did not move.

### Which of the two fixes, with the numbers

Both halves of the lead's sentence, in the order the sentence gives them. The trim returned
1,610,612,736 B of the 1,751,161,856 B gain to the driver in the evicting tick: all of E1 and part of E0.
The 140,549,120 B the pool could not release (E0's chunk shares a live neighbour) are accounted as
pool-cached headroom on the settle line, never pretended to be driver free. `KvAllocator::Vmm` stays
behind its door; no VMM plane was needed.

FINAL-PLACEHOLDER



## Checks actually run

CHECKS-PLACEHOLDER

## Boundaries and record

- Every GPU command on the target card went through `tools/tier-battery.py --rig pro-single` with the
  canonical lock (`--external-lock`, inherited FD, proof in each cell's `LOCK.json`); the ballast smoke
  before the third cell ran under `flock /tmp/memra-gpu.lock`. No third lock name; no bare GPU run; no
  `--no-verify`; no skip variable; no touch of `/root/artifacts`, `/root/memra-spill`, other lanes'
  worktrees or `main`. Lane D built in its own tmux during the first cells and holds no lock. No local
  GPU cell (the local card carried the lead's perf battery). Local builds and tests under
  `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`.
- Native checkout `/root/wt-b` synced by git bundle (detached at the gate tip; the binaries record their
  own source in `build-*/source.txt`). No B tmux session remains. Receipts mirrored from
  `/root/spill-receipts/b-day13/` to `pro-single-day13/`.
- Push: refused by `tools/hooks/pre-push` (perf-ci freshness) on the merge commit's engine files
  (`banked_residency.rs`, `run_gen.rs`, `run_spec.rs`, `hybrid_forward.rs`, `lib.rs`, `moe_cache.rs`,
  `spec/prime.rs`, all from `origin/main`, none touched by this lane). No override used; the lead pushes.
