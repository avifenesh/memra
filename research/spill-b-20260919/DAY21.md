# WP-B day 21: memra#523 close-out map (item 3 held on both cards), memra#427 reproduced and classified, memra#372 verified landed

Repository: **avifenesh/memra**, branch `lane/spill-b-20260919`, on `origin/main` `fa5185d90` (day 20 merged through
#609). First action: `git merge --no-ff origin/main` (`f99971fce`). Every push of the day carried engine source in its
tree (main's, not the lane's: the merge brought `crates/` changes past the committed qualification pointer), so the
#589 hook refused each plain push `UNQUALIFIED: source inputs changed: ...` and the pushes ran as
`MEMRA_RELEASE_QUALIFICATION_MODE=development git push` (`UNQUALIFIED DEVELOPMENT`, logged in
`.git/memra-gate-skips.log`). **No qualification is claimed anywhere in this record**; every collector cell is
`executed-not-qualified`. No code changed on the lane today: cells, receipts, scripts and records only.

Binaries: the lane tip's release `memra-server` and `qwen-a4-continuation-gate` built from `f99971fce` on each card
(local `rtx5090-day21/build-tip/`: 45 s with sccache, SHA-256 `ad373700...` and `5efd1b98...`; target
`pro-single-day21/build-tip/` and `build-cont/`: 36 s and 44 s, SHA-256 `28c030b5...` and `f9d6a23e...`). Artifact
`Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` on both cards. Every GPU command went through `tools/tier-battery.py`
(`--rig rtx5090` with `/tmp/memra-5090.lock`; `--rig pro-single` with `/tmp/memra-gpu.lock`), no retry was needed
(every lock free on the first attempt), `nvidia-smi --query-compute-apps` read empty before and after every cell on
both cards, and the local card's work ran under `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`.
No timing is compared across the two boxes; the cells are pass/fail, not timed.

## 1. memra#523: the close-out map, and item 3 held today

The four-item map (mechanism, gate, receipt per item) is the comment posted on #523 today. In one line each:

| item | mechanism | gate | landed | receipt |
| --- | --- | --- | --- | --- |
| 1 newest turn fits | victim selection never picks the entry being inserted; typed byte refusals; no `snapshot skipped` | twin gate V2, V4 | #596 (`70038ed01`, fix `f42d921db`, day 14) | `DAY14.md`, `pro-single-day14/{gate-main-r3,gate-fix-r3}/` |
| 2 re-decide the default | plain LRU the naked default, the SLRU door deleted | `tools/prefix-policy-ab.py` (at `9466b8912`), twin gate on the landed binary | #598 (`0175e39d4`, day 15), #609 (`fa5185d90`, day 20) | `DAY15.md`, `DAY20.md`, `docs/decisions/PREFIX-CACHE-POLICY.md` |
| 3 the 8-turn twin gate | pre-seeded cohort, 8 growing turns, `cached_tokens` clause every turn k >= 2 | `tools/prefix-newest-turn-fits-gate.py` V1..V6 | gate in #596, capture law in #606 | today's cells below |
| 4 reclaim truth | the evicting tick fences and trims; `reclaim settle` reads driver free before and after | `tools/prefix-evict-reclaim-gate.py` V1, V3; twin gate V3 | #588 (`b013885ba`, fix `f4350c241`, day 13) | `DAY13.md`, `pro-single-day13/{gate-main-final,gate-fix-final}/` |

Item 3 as written asks for `cached_tokens >= previous prompt_tokens`. The gate's V1 asserts EQUALITY to the previous
turn's published entry instead, and that is deliberate: under the capture law (memra#602, #606) an entry publishes at
`capture_len(P)`, the largest grid-aligned length that leaves at least PRIME_MIN_T prompt tokens behind it, so
`cached_tokens` reports the restored aligned length and `>= P` would hold only for an off-grid prompt-end entry, which
is the second numeric program the law forbids. V6 checks the capture law on the server's own lines and V5 checks the
restored render against the cache-off render on every turn. The gate is a standalone serving-shape gate under
`tools/` rather than an arm of `cache-meter-gate.py` because it boots its own calibration and measured servers.

Since #602 landed, the current six-clause gate had been run only on the 12-turn day-16 shape (days 18 and 19). Today
its default shape (budget 1024 MiB, cohort 2800/3000/3200 ids, 8 turns 9,200 to 11,300 ids, plain path, the LRU
default) ran on the tip binary on both cards (`run-day21-gate.sh`; the target twin is `pro-single-day21/run-gate.sh`).
Verbatim:

```text
local RTX 5090 Laptop GPU (rtx5090-day21/gate-tip-default/):
PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns_after_1=0 cached_ok=7/7 lines_ok=8/8 evictions=9 cohort_evictions=3 self_evictions=0 refused_or_skipped=0 effective_free_ok=8/8 identity_ok=8/8 grid_ok=21/21 grid=32 off_grid_calls=0 V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS
RTX PRO 6000 Blackwell (pro-single-day21/gate-tip-default/):
PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns_after_1=0 cached_ok=7/7 lines_ok=8/8 evictions=9 cohort_evictions=3 self_evictions=0 refused_or_skipped=0 effective_free_ok=8/8 identity_ok=8/8 grid_ok=21/21 grid=32 off_grid_calls=0 V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS
```

| card | samples at 250 ms | temp C | peak `memory.used` MiB | peak draw W | power limit | collector | `--validate` |
| --- | ---: | --- | ---: | ---: | --- | --- | --- |
| RTX 5090 | 497 | 54..87 | 21,657 | 181.5 | none reported (laptop part) | `executed-not-qualified`, exit 0 | `CAPTURE INTEGRITY MATCH` |
| RTX PRO 6000 | 289 | 32..58 | 21,939 | 501.6 | 600 W | `executed-not-qualified`, exit 0 | `CAPTURE INTEGRITY MATCH` |

The same nine evictions, three of them cohort entries and none the entry itself, on both cards; every turn's restored
text equals the cache-off boot's; every published entry and every hit on the 32-token grid; `effective_free_ok=8/8`.
Item 3 is literally held on both card classes under the LRU default as of today's tip.

## 2. memra#427: the 16-row final segment, reproduced on the tip and classified

The issue's reproducer, `qwen-a4-continuation-gate`, is on `main` since #424
(`crates/memra-engine/src/bin/qwen_a4_continuation_gate.rs`; the branch named in the issue no longer exists on
origin). It primes the same tokens once in one `prime_cache` call and once as head + suffix on a fresh cache each,
no server, no prefix cache, no drafter, and compares the last-row logits digest. It refuses an unaligned head and
files a 16-row tail as "known" rather than a failure, so it prints `PASS` while reporting `DIFFERS` on that split. The
question of the day was whether the two capture-site fixes (#379 in the hit gate, #602 in #606) changed the outcome.
They did not: they move WHERE a seed lands; they do not touch the prime program of a 16-row call.

### 2.1 Reproduction on the tip, both cards, `MEMRA_PRIME_ROW_RECEIPT=1` (an existing diagnostic)

Prompt: `docs/SERVING.md` (tracked, sha256 in `cont-gate/prompt.sha256`), truncated by the gate to each total. Heads
are all multiples of 32 (`Engine::gdn_chunk_size()`). Cells `rtx5090-day21/cont-gate/` (`run-day21-cont.sh`) and
`pro-single-day21/cont-gate/` (`run-day21-cont-box3.sh`). Verbatim, local RTX 5090 (the target card printed
byte-identical digests on every line it ran, marked in the last column):

| total | split | one-call `logits_sha` | split `logits_sha` | verdict | PRO 6000 |
| ---: | --- | --- | --- | --- | --- |
| 9296 | 9280 + **16** | `14ab5f8b365dbd71` | `35bd15f063bfd5ba` | **DIFFERS** (known: a 16-row final segment is not bitwise on either artifact) | same digests, DIFFERS |
| 9296 | 9248 + 48 | `14ab5f8b365dbd71` | `14ab5f8b365dbd71` | ok | same, ok |
| 9296 | 9216 + 80 | `14ab5f8b365dbd71` | `14ab5f8b365dbd71` | ok | same, ok |
| 9296 | 9184 + 112, 9152 + 144, 9120 + 176, 9088 + 208 | `14ab5f8b365dbd71` | `14ab5f8b365dbd71` | ok (four splits) | not run |
| 9297 | 9280 + **17** | `e641952cac19e526` | `e641952cac19e526` | ok | same, ok |
| 9297 | 9248 + 49 | `e641952cac19e526` | `e641952cac19e526` | ok | not run |
| 9311 | 9280 + **31** | `b61719f294866b05` | `b61719f294866b05` | ok | not run |
| 9311 | 9248 + 63 | `b61719f294866b05` | `b61719f294866b05` | ok | not run |
| 9312 | 9280 + **32** | `736a88d7c7448dcb` | `736a88d7c7448dcb` | ok | same, ok |
| 9312 | 9248 + 64, 9216 + 96 | `736a88d7c7448dcb` | `736a88d7c7448dcb` | ok | not run |

The `[prime-row]` receipts of the 9296 run: the 48-row suffix call reproduces the one-call row exactly (`top1=318
margin=1.375317e0`); the 16-row suffix call keeps the argmax and moves the margin (`top1=318 margin=1.369547e0`), so
on this prompt the difference is below a token flip, which is why the issue saw it only on a near-tie transcript.
Same head (9280), tails 16, 17, 31, 32: only 16 differs. The defect is an exact-16 edge, not a short-tail or partial
WY-chunk effect, and it is deterministic across the two card classes (byte-identical digests).

### 2.2 Bisect on the same binary with existing rollback seams (`cont-bisect/`, `run-day21-cont-bisect.sh`)

Total 9296, tails 16 and 48, one arm per seam. An arm whose 16-row split turned `ok` would have named the kernel
family; none did:

| arm | one-call | 9280 + 16 | 9248 + 48 |
| --- | --- | --- | --- |
| baseline | `14ab5f8b365dbd71` | `35bd15f063bfd5ba` DIFFERS | ok |
| `MEMRA_GDN_CHUNKED=0` (sequential GDN scan) | `4749ef60887102ee` | `8c785e5d8fb582da` DIFFERS | ok |
| `MEMRA_F16OUT=0` | `14ab5f8b365dbd71` | `35bd15f063bfd5ba` DIFFERS | ok |
| `MEMRA_NOFA=1` (naive attention) | (moved) | `26eb8171bff49fa2` DIFFERS | ok |
| `MEMRA_MMQ_W4A8=0` (int8 GEMM class off) | (moved) | `bef04e7ce1485ee4` DIFFERS | ok |

Cleared by this table: the WY-chunked GDN scan, the f16 epilogue (inert here: identical digests to baseline), the
attention kernel family, and the int8 W4A8 MMQ class. Read from the tree and cleared as well: `GEMM_M_THRESHOLD` is 16
and the test is `m >= 16`, so a 16-row call takes the prefill GEMM like a 48-row call (the batched-MMVQ "b16 tier"
sits below it and is not reached at m = 16); the W4A8 MMQ launcher's tile is fixed at compile time (no `mmq_x` by
column count); the native f32 row GEMV that refuses `m > 16` sits behind `MEMRA_F32_GEMV_KERNEL=1`, default off.

### 2.3 The chunk-width arms (`cont-bisect2/`, `run-day21-cont-bisect2.sh`): the classification

`MEMRA_PRIME_CHUNK` (a documented memory knob) sets the one-call prime's own chunk schedule. Predictions written into
the runner before it ran: with `=16` every call in both arms is a 16-row chunk (ok expected); with `=32` the one-call
itself ends in a 16-row chunk (9296 = 290 x 32 + 16; the fold rule folds only tails strictly below PRIME_MIN_T), so
the split should equal it (ok expected); with `=48` only the split's head ends in a 16-row chunk (DIFFERS expected).
Verbatim:

| arm | one-call | 9280 + 16 | 9248 + 48 |
| --- | --- | --- | --- |
| `MEMRA_PRIME_CHUNK=16` | `290420a09bcbd2c2` | `290420a09bcbd2c2` ok | `290420a09bcbd2c2` ok |
| `MEMRA_PRIME_CHUNK=32` | **`35bd15f063bfd5ba`** | `35bd15f063bfd5ba` ok | `35bd15f063bfd5ba` ok |
| `MEMRA_PRIME_CHUNK=48` | `f317438ac6eed52e` | `493c452e6dd23f55` DIFFERS | `279a20e2101e6c44` DIFFERS |

The `=32` row is the classification: the one-call prime over 9296 tokens, when its own schedule ends in a 16-row
chunk, produces **exactly the digest the 16-row suffix call produced under the default schedule**
(`35bd15f063bfd5ba` in 2.1). The same rows computed as a 16-row chunk give the same bits whether the chunk is the tail
of a cold prime or a restored suffix, and they differ from the same rows computed inside a wider chunk (the default
schedule ends 9296 with an 80-row chunk, 9216..9296). The restore protocol is exact. **The seam is that a prime
chunk of exactly PRIME_MIN_T (16) rows is its own numeric program**, in the per-chunk layer walk, and every cold prime
whose schedule ends in a 16-row chunk carries it too. The `=48` row is a control that reads a different law: an
explicit `MEMRA_PRIME_CHUNK` returns the fixed ranges without `align_prime_ranges_to_gdn`, so the one-call's
boundary at 9264 is off the 32-token grid while the split's boundaries (9216, 9248) are on it, and both splits differ
by the GDN grid law (`research/multiturn-cache-20260821/LONGCTX-EXACTNESS-20260821.md`), not by the 16-row edge. That
row is reported as measured and not read as evidence about the edge.

Which kernel inside the layer walk changes program at width 16 is NOT identified today: the arms above clear the GDN
scan, attention, f16out, the W4A8 class, the MMQ tile, the GEMM threshold and the f32 GEMV door. What remains keyed
on the call width are the dense (non-quantized) matmuls that reach cuBLASLt, whose heuristic kernel choice is
shape-keyed, and the GDN prep/conv kernels' width tiling. Naming it needs a per-layer, per-tensor digest of a 16-row
chunk against a 17-row chunk on the same rows (the `[prime-row]` receipt is per call, not per layer), which is the
next cell.

### 2.4 Not fixed today, and the fix shape

The brief allows a fix only if it is one program for both sides with a bit-identity gate. Two shapes exist and both
change bytes outside the restore path, so neither fits inside today's budget with its battery:

1. **Engine side**: make a 16-row chunk take the wide-chunk program (whichever kernel the per-layer digest names). This
   changes the bits of every cold prime whose schedule ends in a 16-row chunk (t = 16 mod chunk) and needs
   `kernel-check`, the `run-gen` argmax gate, `run-spec` K=1..8, the chunkinv gate extended to width 16, and this
   gate's 16-row split turned from "known" into a verdict.
2. **Schedule side**: never emit a 16-row segment. The fold rule in `prime_cache` (`t - end < PRIME_MIN_T` folds the
   tail) would become `<= PRIME_MIN_T`, the worker's three capture sites (`seed_capture_boundary`, the LCP split,
   the spec `capture_at`) would leave MORE than PRIME_MIN_T prompt tokens behind a seed (the day-18 rule leaves at
   least 16, which is exactly the shape the issue names: P = 9296 publishes 9280 and restores a 16-row suffix),
   `DflashRestoreSuffix::new` and `check_mtp_prime_walker` would refuse a suffix of exactly 16 the way they refuse
   below 16, and the gate's "known" arm would become a verdict. This moves which rows a cold prime folds and where
   entries publish; it needs the twin gate, the restore gate and the #379 hit gate green on both cards plus the
   chunkinv row at the new fold.

Recommendation to the lead: shape 2 is the smaller numeric change (a fold boundary and a capture boundary, no kernel
change) and closes the served surface (a restored suffix can no longer be exactly 16 rows) while shape 1 is what
makes the cold prime chunk-invariant at width 16. They are independent; the per-layer digest cell decides whether
shape 1 is a one-line kernel-selection fix or a kernel rewrite. Until one lands, the 16-row segment stays a known
non-bitwise shape and `qwen-a4-continuation-gate` keeps reporting it as such.

Telemetry of the three local cells: `cont-gate` 579 samples, 56..88 C, peak draw 178.0 W, peak `memory.used`
19,961 MiB; `cont-bisect` 1,074 samples, 56..90 C, 179.8 W, 19,929 MiB; `cont-bisect2` 1,451 samples, 59..87 C,
173.6 W, 16,697 MiB. Target card `cont-gate`: 140 samples, 32..56 C, peak draw 499.6 W under the 600 W limit, peak
`memory.used` 20,211 MiB. All `executed-not-qualified`, exit 0, card alone before and after.

## 3. memra#372: verified landed; CPU tests green; the bounded cell pre-registered, not run

The issue's ask (include validated DFlash restores in retained-prefix admission) landed in #377 (`628521007`,
2026-09-08, v0.137.0), before this lane. On today's tree: `AdmissionRestoreRoute::{Native, Dflash { ctx_cap, suffix }}`,
`DflashRestoreSuffix::new(rows, prompt, partial)` (`FullCover` at exact cover, `Prime(n)` only under the partial door
with `n >= PRIME_MIN_T`, else `None`), `PrefixCache::admission_dflash_restore` (live lease, exact token boundary,
`tail.len == rows`, `validate_restore`, finite boundary logits of the vocab, `dspark_spec_prompt_fits`), the plan
revalidated at consumption and refused with a 429 if the route, the paid `ctx_cap` or the rows moved, and
`finish_planned_dflash_conversion` releasing the serving lease and failing closed when a planned conversion fails.
The cost rule is the native restore's `cost_after_prefix_restore` with `workspace_rows = rows.min(prompt_len -
PRIME_MIN_T)` for DFlash, so a full cover keeps a minimum-prime workspace. The `!dspark_drafts.contains_key(&model_key)`
exclusion the issue cites is gone. #377's own record holds the required 128k receipt (warm `cached_tokens=131040`,
`cost 7250MB -> 4129MB; reserve 1720MB unchanged; cold fallback forbidden`, base control `HTTP 429`).

CPU gate on the arithmetic, today, on the merged tree (`rtx5090-day21/cpu-tests-372/`, release profile, under the
CPU quota), verbatim:

```text
test worker::tests::dflash_retained_consumed_carrier_failure_releases_serving_lease_and_refuses_cold ... ok
test worker::tests::dflash_retained_suffix_distinguishes_full_boundary_and_unsupported_suffix ... ok
test worker::tests::dflash_retained_plan_checks_route_before_discounting ... ok
test result: ok. 3 passed; 0 failed; 1 ignored; 0 measured; 770 filtered out; finished in 0.00s
```

(`dflash_retained_gpu_plan_fault_matrix` is the ignored one: it needs an exclusively locked CUDA device and the
DFlash restore profile.)

**Pre-registration of the bounded serving-shape cell (not run today).** A DFlash-capable artifact and its draft exist
on this rig (`Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` and the `q38-dflash2` safetensors export). The cell: one
`memra-server` boot with `MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT=<export dir> MEMRA_DSPARK_PREFIX_RESTORE=1`
(`MEMRA_DSPARK_PARTIAL_RESTORE` at its default, so the suffix arm is `FullCover` or a boundary restore), a context
window and a ballast sized so that the warm turn's cold charge does not fit but its planned charge does (read from the
`[admission] request cost` line of a dry boot), a cold prompt then its continuation; verdicts: the warm turn is
admitted (HTTP 200) with `[admission] retained prefix plan: ... cold fallback forbidden` in the log and
`cached_tokens > 0`, DFlash engaged (`[dspark-acc]` rows), zero OOM or replay lines, and the warm completion's text
equal to the same continuation served cold in a cache-off boot (greedy; the greedy-penalty arm is default ON so the
route takes greedy requests on this profile). A 429 reproduction needs a pre-#377 binary and is not part of the cell.
Not run: the budget went to #427's classification; the cell is runnable on this rig when the lead schedules it.

## Checks actually run

| Check | Result |
| --- | --- |
| Native release build, lane tip, both cards (`build-tip/`, `build-cont/`) | exit 0 (local 45 s, target 36 s and 44 s), `dirty.txt` empty |
| Twin gate, default 8-turn shape, tip binary, both cards | `-> PASS` on both (above) |
| `qwen-a4-continuation-gate`, four totals, local; three totals, target | `9280 + 16: DIFFERS` on both cards; every other split `ok` |
| Bisect, five rollback-seam arms, local | 16-row split `DIFFERS` in every arm |
| Chunk-width arms, local | `=16` ok, `=32` ok with the one-call digest equal to the default run's 16-row suffix digest, `=48` DIFFERS (grid law) |
| `cargo test -p memra-server --lib -- dflash_retained cost_after_prefix_restore` (release, CPU quota) | 3 passed, 0 failed, 1 ignored |
| `tools/tier-battery.py --validate` on every collector cell (both cards) | `CAPTURE INTEGRITY MATCH; command status=executed-not-qualified; NOT qualification` |
| `cargo fmt --all -- --check`, `tools/check-flags.sh`, `git diff --check`, `python3 tools/check-public-boundary.py check` | PASS, PASS (no uncovered runtime names), clean, 0 new |
| Full GPU exactness battery | NOT RUN: no code change on the lane |

## Boundaries and record

- No code changed; no new `MEMRA_*` read (the scripts set existing documented variables in a cell's environment); no
  gate relaxed after a result (the reproduction gate still files the 16-row tail as "known", as on `main`); no
  `--no-verify`; no third lock name; no bare GPU run; no other lane's worktree touched; `/root/artifacts` read only;
  nothing of V4.1; no external dependency; no hosts, ids or costs in tracked files (the target card's receipts carry
  the collector's own `provider_instance_id: null, hourly_cost: null` keys, as every prior day's do).
- One shell mistake of the day, recorded: a first attempt to write the target card's #427 runner through a quoted
  ssh here-document broke on nested quotes; the tail of the script ran locally and was refused (`mkdir ''`, writes to
  `/total-*.log` denied), the truncated remote file was removed, and the runner was written locally
  (`run-day21-cont-box3.sh`, tracked) and copied over the socket instead. No process was started by the broken
  attempt and no file was left behind on either side.
- Every `/tmp` scratch file of the day (two comment drafts) was removed after use. The target card's `/root/wt-b`
  (this lane's own worktree there) stays detached at the lane tip with its build; the receipts were copied into
  `pro-single-day21/` by rsync over the socket (no host string in any tracked file).
- Agent time: about 3 h 40 min of the 4-hour budget.
