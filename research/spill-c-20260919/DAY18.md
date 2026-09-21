# Session C day eighteen: MoeSlotCache door items 1, 3, 4 and 6 as review inputs; the HOSTPREFIX door's review table and hash micro-cell

Scope (lead brief, day 18): (1) `MOE-SLOT-CACHE-DOOR.md` open items 1, 3 (second half), 4 and 6
(decide-by 2026-10-04), each with its question, pre-registered rule and shape written here before any GPU
run, executed on the target card through the collector, verdict verbatim with a replay; correctness items
pass/fail on both cards, the local RTX 5090 first; (2) the `HOSTPREFIX-DOOR.md` review table (every owed
cell, its receipt and verdict, what is missing) and the hash micro-cell named on day 17; (3) the records.
Tree: lane merge of main `5804cac6a` (`922927619`, #612 with the lead's two review rounds on memra#586:
`SubmitGuard` in `memra_cpu_expert_prefetch_v2`, fixture cells `submit-throw` and `submit-throw-claim`,
read, not re-derived; #586 closed), then `e16bc69e8` (the `hash-micro` diagnostic bin, below). Two earlier
day-18 attempts died on an API outage before any action; the worktree was clean at `a6b62c998` equal to
`origin/lane/spill-c-20260919` at the start of this one. Every push of this lane today is in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the #589 hook prints `UNQUALIFIED DEVELOPMENT ... no
GPU qualification claimed` and appends a `log_skip` row): no qualification is claimed by anything here;
every cell is `executed-not-qualified`. Nothing here is a support state.

## Push mode, stated

`git push origin lane/spill-c-20260919` at `922927619` ran with `MEMRA_RELEASE_QUALIFICATION_MODE=development`,
verbatim from the hook: `UNQUALIFIED DEVELOPMENT: refs/heads/lane/spill-c-20260919 at
922927619d6eeb059e43f818d44e06c9c3ea339f; no GPU qualification claimed` then `pre-push: skip recorded in
.../.git/memra-gate-skips.log`. The same shape at every later push today (the pre-registration push before
the runs, and the day-docs tip); each is listed in the Push section at the end.

## The instrument added today (`e16bc69e8`): `hash-micro`

`crates/memra-engine/src/bin/hash_micro.rs`, declared in the engine's `Cargo.toml` beside the other gate
and probe bins (`storage-bench`, `h2d-probe`, `tier-transfer-gate`). It binds a CUDA context, allocates the
same byte count three ways, cached pinned (`cuMemHostAlloc` flags 0, the call `PinnedHostBuf::new` and
`PinnedKind::Cached` make), write-combined pinned (`CU_MEMHOSTALLOC_WRITECOMBINED`, `PinnedKind::WriteCombined`)
and heap (`Vec<u8>`), fills all three from one fixed xorshift stream, reads the driver's flags back with
`cuMemHostGetFlags`, and times `memra_tier::contracts::checksum` (the door's hash: sha2 0.10 with the
domain prefix, the function the engine's completion checksum and bind's bundle checksum both call) over each
kind, N passes per kind per order in two orders (cached, wc, heap; then heap, wc, cached). It prints one line
per pass and one `HASH-MICRO rule` line; the digests must agree across kinds or it exits non-zero. No engine
path reads it, no default moves, no `MEMRA_*` read (so no `docs/FLAGS.md` row), no `.cu` file (so no
`docs/KERNELS.md` row); a diagnostic in the sense of the flags doctrine, run only through the collector.

## Pre-registration (written and pushed before any GPU run today)

Common shape. Every GPU cell runs through `tools/tier-battery.py --rig <rig> --external-lock --execute bash
research/spill-c-20260919/day18-cell.sh <cell> @COLLECTOR_LOCK_FD@` (one lock hold per cell, the lock proof
under `ev/LOCK.json`, the collector's 250 ms telemetry, `status: executed-not-qualified`), on BOX3 (one RTX
PRO 6000 Blackwell at its 600 W limit, lock `/tmp/memra-gpu.lock`, driver `day18-run-cell.sh` with the
day-17 bounded lock retries) and, for the correctness cells, first on the local RTX 5090 Laptop GPU (lock
`/tmp/memra-5090.lock`, `--rig rtx5090`). Binaries are built from this tree (`e16bc69e8`) on each rig; their
SHA-256 is in each cell's `ev/binary.sha256`. Every `run-gen` and `hash-micro` invocation's output is written
through a line stamper (UTC ms at line arrival; `2>&1` into the stamper, the exit status kept through
`PIPESTATUS`, nothing dropped), so phases can be read from line times without a code change. The replay is
`day18-replay.py <cell-dir>`; it applies the clauses below and prints `DAY18 REPLAY <cell>: PASS|FAIL (n
checks)` plus the cell's rule line. No clause or threshold moves after a run.

### Item 1, owner placement under PP (design item; census and CPU proof, no native cell exists)

**Question.** Does any path that the door could reach today admit through the bank from a thread other than
the one that registered the owner, and what happens when it does?

**What can be run.** The refusal mechanism has a CPU proof: `crates/memra-tier/tests/bank/owner_proxy.rs`
(the `WrongOwner` test: a proxy cloned to a spawned thread refuses `validate`, `demand`, `with_bytes` and
`finish`) and the unit tests in `owner_proxy.rs`. Rule: `cargo test -p memra-tier --test bank owner_proxy`
passes on this tree (pass/fail, CPU, both cards' hosts are the same program; the local rig runs it). No
native cell exists: the door lives in `run-gen` and `run-spec`, neither of which has a PP walk; the PP walks
(`prime_hyper_pp2_stage0_enqueue` and `prime_pp2_stage0_enqueue` behind `std::thread::scope` at
`hybrid_forward.rs:4721` and `:6011`, the prime PP head-stage walker at `:6179` whose stage threads own
their stage caches, `decode_step_h_ppn` and the batched and spec-verify stage splits behind `ppn-gate`,
`pp2-gate`, `decode-batch-gate --mode pp|ppspec` and `memra-server`) carry no installer. Adding the installer
to a PP gate binary is code, not a cell, and is not done today.

**Census, stated as the review input (source at `e16bc69e8`).** `Engine::with_moe_cache` (`lib.rs:6510`)
takes the `Mutex<Option<MoeSlotCache>>` from whatever thread calls it; its callers in `hybrid_forward.rs`
are the MoE layer functions (`moe_ffn_dev`, `moe_gdec_token`, `moe_gdec_token_q8`, `moe_fused_epi_token_q8`,
`moe_cached_gemm`, `moe_cached_gemm_q8`, `moe_frozen_gemm`, `moe_profile_admit_expert`,
`moe_prefetch_expert`, `moe_prefetch_disk_expert`, the batched MoE path and
`restore_cpu_expert_residency_profile`), so under a stage split a MoE layer's admission runs on the stage
thread that walks that layer. With the bank installed, `admit_banked` calls `bank.validate` first, and the
proxy's `access` compares `thread::current().id()` with the owner's: a stage thread gets `WrongOwner` before
any demand, H2D or publication, so the failure is a typed refusal, not a wrong-thread staging. That is the
whole of what item 1 can show today: refusal is the landed behavior, and the placement design (an owner per
CUDA owner thread, or a typed owner-thread hand-off keeping `ExpertBankOwner` `!Send`) is the lead's PP
placement call, as the door doc says.

### Item 3, second half: the hash lock (red arm + control, both cards) and scale admission (CPU proof)

**Question.** Does `install_expert_bank_gate` refuse a non-approved artifact on its SHA-256 lock before any
bank, catalog, CUDA slot or source read, with the typed text, and does the same artifact run natively
without the door (so the refusal is the lock's and not the model's)?

**Shape.** Cell `hashlock`: two `run-gen` runs in one lock hold on a non-approved single-shard GGUF with at
most one MTP head (the installer's preconditions hold, so the SHA pass is reached): `door` with
`--experts-via-tier` and `control` without, `MEMRA_NGEN=8`, prompt `55 88 13`. Local: the Qwen3.5-9B NVFP4
MTP GGUF the serve smoke uses. Target card: the Qwen3.8-27B NVFP4 Q5K MTP artifact under `/root/artifacts`
(the only non-approved artifact there). Rule, applied by the replay: PASS (`hash_lock_refuses`) iff the door
run exits 1 with the last line exactly `Error: "experts-via-tier artifact SHA256 mismatch"` and prints no
`[experts-via-tier]`, `[expert-host-slru]` or `[expert-gpu-slru]` line; and the control run exits 0 with
`MATCH` and no door line. Recorded, not a clause: the lock's cost, the time from the load-end line to the
refusal (N=1 per card, the SHA pass over the whole artifact; the installer runs after load by design).

**Scale admission.** No scale-bearing artifact (Hy3, Step FP8: `.scale` / `.input_scale` rows, `macros` /
`fp8_blk` planes) exists on either rig, so the native refusal cell is pre-registered and not run: its shape
is the same two-run cell on such an artifact with the expected last line `experts-via-tier expert catalog
refused: artifact carries expert scale planes the consumer does not declare: <names>` (exit 2). What runs
today is the CPU proof: `cargo test -p memra-gguf --lib expert_banks` (`a_scale_plane_on_a_bank_is_refused_by_name`
and the other catalog refusals) passes on this tree.

### Item 4, overlap: the synchronous miss path priced (timing pair, target card, N=5 per arm per order)

**Question.** With `max_pending = 1` and `prefetch_source` returning `false`, every GPU-slot miss under the
door is one synchronous demand, H2D and full compute-stream drain. On the target card, at the day-nine
8 GiB pressure budget (9,986 GPU slots, the shape that produced 12,091 GPU evictions), what does that cost
per decode token against the legacy SLRU miss path for the same tape, and is the door's decode slower by the
flags doctrine's margin?

**Shape.** Cell `overlap`: twenty `run-gen` runs in one lock hold on the approved artifact,
`MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986`, prompt `55 88 13` (the day-nine environment;
both arms take the same exact slot count through the untouched `MEMRA_MOE_SLOTS` clamp, so the GPU cache is
the same size in both and the budget flag, which refuses beside `MEMRA_MOE_SLOTS`, is not used). OFF = no
door; ON = `--experts-via-tier` (host budget default 256 MiB, 16 host slots). Order 1: (OFF, ON) x 5;
order 2: (ON, OFF) x 5; N=5 per arm per order, N=10 pooled. Observations per run from the stamped log:
the `generated 32 tokens in Xs` line (decode wall, gen-only, the observation the ratio is read on), `prime`
seconds, the `MoE cache: 9986 slots | cumulative hits= misses=` line, the STEADY-STATE line (hit-rate and
MB per decode token), `MATCH`, the `tokens:` tape; ON only: `[expert-gpu-slru] ... evictions=`,
`[experts-via-tier] physical_reads=`; derived from line times: `install_s` = the `installed` line minus the
`[q8rp] split-plane decode mirrors built` line (ON; the SHA lock, the catalog bind and the byte-for-byte
record compare), `forward_s` = the `generated` line minus the `loaded` line (both arms: prefill, warmup
and decode).

**Rule.** Integrity (a failure voids the verdict, the clause named): all twenty runs exit 0 and print
`MATCH`; one `tokens:` tape across the twenty; `slots=9986` in every cache line; one STEADY-STATE line
across the twenty (same routing, same steady miss traffic); the ON runs print `installed` and
`physical_reads=`; the OFF runs print no door line. Verdict on the decode gen-only seconds:
`decode_ratio = median(ON) / median(OFF)`, pooled and per order; `sync_miss_path_slower` if the ratio is
at least 1.10 pooled and in both orders; `sync_miss_path_faster` if at most 0.90 pooled and in both orders;
`sync_miss_path_flat` if the pooled ratio is inside (0.90, 1.10); `orders_disagree` otherwise. Reported
beside it, no clause: `door_cost_ms_per_decode_token = (median ON - median OFF) / 32`, that divided by the
STEADY-STATE MB per token as a per-staged-MB cost, the ON `install_s` median, `forward_s` medians and their
ratio, the cumulative miss counts per arm (expected to DIFFER: the bank returns `false` from
`prefetch_source`, so the legacy detached prefetch that turns demands into hits is off under ON; day nine
read 18,195 against 8,211) and the ON trace line count (`TracedDispatch` prints one `[expert-host-slru]`
line per host demand to stdout; that print is part of the door as landed and is inside the ON time).

**Against the door.** If `sync_miss_path_slower`: item 4 is confirmed as the promotion blocker at this
budget; the door cannot sit behind the materializer on speed without several leases in flight retired on
copy-stream events, and the synchronous path is the door's only miss path, so a delete decision deletes the
list in "Decision at decide-by" whole (there is no separate arm to remove). If `flat` or `faster`: item 4's
premise is not supported at this budget on this card and the item leaves the pending list as answered; no
code arm corresponds to it. The decision stays the review's.

### Item 6, serving shape (red arm, both cards)

**Question.** The door exists in `run-gen` and `run-spec` only. What happens when an operator hands the
door's flag to `memra-server`, and is any door line reachable in serving?

**Expectation from source, stated before the run.** `serve_with` (`crates/memra-server/src/lib.rs:5623`)
collects argv and consults it for `--version` / `-V` and the key-lifecycle flags (`auth::run_cli` acts only
when `--gen-key` or `--revoke-key` is present); no other argument is parsed and nothing rejects an unknown
one, so `--experts-via-tier` should be ignored silently and the server should serve the native program.

**Shape.** Cell `serverdoor`: one boot of `memra-server` from this tree with `--experts-via-tier` on its
argv (`MEMRA_COMPAT=openai`, `MEMRA_MODELS=gate=<artifact>`, `MEMRA_CTX=8192`, `MEMRA_MAX_SESSIONS=4`, a
lane-C port with the port guard), readiness polled on `/v1/models`, one completion (`max_tokens` 16,
temperature 0), TERM. Local: the Qwen3.5-9B NVFP4 MTP GGUF. Target card: the approved artifact (the door's
only artifact, so a serving installer, if one existed, would engage here). Rule: PASS
(`door_unreachable_in_serving`) iff the server becomes ready, the completion returns text, and the server
log carries zero `experts-via-tier`, `expert-host-slru` or `expert-gpu-slru` lines. Reported beside it:
`flag_refused` (any usage or unknown-argument line naming the flag) and `flag_silently_accepted` (ready
with no such line). Both are inputs: the first says the door has no serving surface (so the serving-shape
bit-identity gate item 6 requires cannot exist before an installer does); the second, if true, is a hygiene
finding for the review (the door's flag reaches the server's argv and produces neither the door nor a
refusal).

### HOSTPREFIX door: the hash micro-cell (both cards; review input, no verdict clause)

**Question** (`WC-DESTINATIONS.md` item 2, `HOSTPREFIX-DOOR.md` "what remains"). How long does the door's
hash take over the entry's byte count on this host, in the memory kind the door's destinations have on each
card (`PinnedKind::for_device`: cached on the target class, write-combined on the RTX 5090 class), so the
demote delta lane A measured with cached destinations (steady state 82 against 6.1 ms, day 15 pair) and the
day-16 write-combined delta (136 to 140 against 6 to 8 ms) can be split into the hashes and the ticket
lifecycle by arithmetic, and so the cached-versus-write-combined read cost of the hash is a number.

**Shape.** Cell `hashmicro`: `hash-micro --bytes 167772160 --n 5` (160 MiB, the entry is 159.9 MB; SHA-256
throughput is length-independent at this size, the per-plane hashing the engine does over 32 planes of
about 5 MB sums to the same bytes) through the collector on each card, N=5 per kind per order, two orders,
N=10 pooled per kind. Reported: per-kind medians, ranges and GB/s, per-order medians, `wc_over_cached`,
`cached_over_heap`, `two_hashes_cached_ms` (the two demote-side hashes) and `one_hash_cached_ms` (the
promote-side hash), the driver flags read back (the WC bit must be set on the WC buffer only) and digest
equality across kinds. Integrity checks in the replay: one rule line, ten passes per kind, the medians
recomputed from the pass lines agree with the rule line, digests equal, flags as expected. No verdict
clause: the review reads the split; the arithmetic against lane A's and day 16's deltas is written in the
results section as arithmetic across sittings on the same box, not as a same-window measurement.

**Budget.** About 2.5 agent-hours for the four GPU cells and the two local arms; the overlap pair is the
long one (ten ON runs at about 90 s each on day nine's shape). If the budget runs out the remaining cells
stay pre-registered here and the results section says so.

## Builds and CPU proofs

Local (`day18-local/build.log`, `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`, tree
`e16bc69e8`, rc=0): run-gen `ad113e31…`, run-spec `836ca8ee…`, hash-micro `73ca7d76…`, memra-server
`a492631b…`. Target card (`pro-single-day18/build.log`, `/root/wt-c` at `refs/bundle/c18` = `e16bc69e8`,
nvcc under `/usr/local/cuda`, rc=0 both): run-gen `293bada0…`, run-spec `3f2040cd…`, hash-micro `deac707c…`,
memra-server `c195b89e…`. The tree's `crates/`, `tools/`, `docs/` and workflows are byte-identical to main
`5804cac6a` except for the `hash-micro` bin and its `Cargo.toml` entry (`git diff origin/main -- crates/
tools/ docs/ .github/` before `e16bc69e8` was empty).

CPU proofs (`day18-local/cpu-proofs.log`, `CPUQuota=400%`, a separate target dir, all rc=0):
`cargo test -p memra-tier --test bank owner_proxy`: 4 passed, among them
`proxy_and_token_are_send_sync_but_staging_is_owner_only` (the spawned thread's `validate`, `demand`,
`with_bytes` and `finish` each return `WrongOwner`); `cargo test -p memra-tier --lib bank::owner_proxy`:
5 passed; `cargo test -p memra-gguf --lib expert_banks`: 10 passed, among them
`a_scale_plane_on_a_bank_is_refused_by_name`, `a_dense_plan_has_no_expert_projections`,
`a_missing_expert_tensor_is_the_contract_missing_verdict`, `a_duplicated_expert_tensor_is_ambiguous`,
`a_shape_incompatible_expert_tensor_is_refused`. These are the item 1 and item 3 (scale admission) proofs
named in the pre-registration; they are CPU tests of the refusal mechanisms, not native cells.

## Item 1 result: census and proof, no native cell

The census in the pre-registration stands as the review input (no code changed under it today). The proof
is the CPU test above. Item 1 is a design item: nothing here decides between an owner per CUDA owner thread
and a typed owner-thread hand-off; the landed behavior under a stage split is a typed `WrongOwner` refusal
before any demand or H2D.

## Item 3 result: the hash lock refuses on both cards (`hashlock`, replay `DAY18 REPLAY hashlock: PASS (7 checks)` on both)

Both cells through the collector, one lock hold each, `--validate` rc=0 (`*-validate.log`).

| Card | Artifact (sha256 prefix) | door run | control run | Rule line (verbatim) |
|---|---|---|---|---|
| RTX 5090 Laptop (`rtx5090-day18/hashlock/`, 55..60 C) | Qwen3.5-9B NVFP4 MTP GGUF, `52c9cceb190055e0` | exit 1, last line `Error: "experts-via-tier artifact SHA256 mismatch"`, no door line; the only other stdout line is `[q8rp] split-plane decode mirrors built: 92 tensors` 1.66 s before the refusal (the SHA pass over the page-cached artifact) | exit 0, `prefill argmax=198  decode argmax=198  logit maxdiff=2.199e-1  MATCH`, `generated 8 tokens in 0.055s` | `HASHLOCK rule door_exit=1 sha_mismatch_line=True door_lines=0 control_exit=0 control_match=True lock_cost_after_last_load_line_s=1.66 door_wall_s=2.9 control_wall_s=1.7 (N=1, not pooled) artifact=52c9cceb190055e0 -> hash_lock_refuses` |
| RTX PRO 6000 Blackwell (`pro-single-day18/hashlock/`, 37..42 C, 147 W peak under 600 W) | Qwen3.8-27B NVFP4 Q5K MTP GGUF, `1facf36c2db359dc` | exit 1, the refusal is the only stdout line (this artifact prints nothing during load), 12.3 s wall including the load and the SHA pass over an artifact the page cache no longer held | exit 0, `prefill argmax=271  decode argmax=271  logit maxdiff=2.591e-1  MATCH`, `generated 8 tokens in 0.109s`, 4.7 s wall | `HASHLOCK rule door_exit=1 sha_mismatch_line=True door_lines=0 control_exit=0 control_match=True lock_cost_after_last_load_line_s=nan door_wall_s=12.3 control_wall_s=4.7 (N=1, not pooled) artifact=1facf36c2db359dc -> hash_lock_refuses` |

Replay correction, stated: `day18-replay.py` as first committed counted any line containing the substring
`experts-via-tier` as a door line, and the refusal line itself contains it, so the first replay of both cells
printed `door_lines=1` and `FAIL` on that clause. The pre-registered clause names the bracketed tags
(`[experts-via-tier]`, `[expert-host-slru]`, `[expert-gpu-slru]`); the replay was corrected to the clause as
written (commit after `7050170ba`) before any other result was read, and the logs carry no bracketed tag
(`grep` shows the refusal line only). The `lock_cost_after_last_load_line_s` field is `nan` on the target
card because that artifact prints no load line; the wall from the driver's marks is reported instead. No
threshold moved.

Reading for the review: the lock refuses before the catalog, before any bank, CUDA slot or record read, with
the typed text, on both cards, and the same artifact runs natively without the door. The lock's cost is the
SHA-256 pass over the whole artifact after load (1.7 s for a 6 GB page-cached file here; the 19 GB approved
artifact on the target card is inside the ON `install_s` of the overlap pair below). The scale-admission
refusal has its CPU proof above and no native cell (no scale-bearing artifact on either rig; shape
pre-registered).

## HOSTPREFIX door: the hash micro-cell (`hashmicro`, both cards, replay `DAY18 REPLAY hashmicro: PASS (8 checks)` on both)

| Host | Rule line (verbatim) |
|---|---|
| RTX 5090 Laptop rig (`rtx5090-day18/hashmicro/`, 58..59 C) | `HASH-MICRO rule device="NVIDIA GeForce RTX 5090 Laptop GPU" bytes=167772160 n_per_order=5 pooled=10 orders=2 digest_equal=true wc_bit_cached=false wc_bit_wc=true cached_ms=37.339 wc_ms=1431.613 heap_ms=37.480 cached_range=37.244..39.231 wc_range=1424.480..1454.788 heap_range=37.227..37.912 cached_o1=37.357 cached_o2=37.263 wc_o1=1433.402 wc_o2=1428.212 heap_o1=37.542 heap_o2=37.443 cached_gbps=4.493 wc_gbps=0.117 heap_gbps=4.476 wc_over_cached=38.340 cached_over_heap=0.996 two_hashes_cached_ms=74.679 one_hash_cached_ms=37.339` |
| BOX3 host, RTX PRO 6000 Blackwell (`pro-single-day18/hashmicro/`, 36..38 C, 149 W peak under 600 W) | `HASH-MICRO rule device="NVIDIA RTX PRO 6000 Blackwell Server Edition" bytes=167772160 n_per_order=5 pooled=10 orders=2 digest_equal=true wc_bit_cached=false wc_bit_wc=true cached_ms=77.922 wc_ms=1698.063 heap_ms=77.990 cached_range=77.823..78.041 wc_range=1694.567..1700.849 heap_range=77.871..78.162 cached_o1=77.889 cached_o2=77.938 wc_o1=1696.512 wc_o2=1699.887 heap_o1=77.983 heap_o2=78.016 cached_gbps=2.153 wc_gbps=0.099 heap_gbps=2.151 wc_over_cached=21.792 cached_over_heap=0.999 two_hashes_cached_ms=155.843 one_hash_cached_ms=77.922` |

Observations: the engine's hash runs at the host CPU's SHA-256 speed over cached pinned and heap memory
alike (2.15 GB/s on the target box, 4.49 GB/s on the local rig; `cached_over_heap` 0.999 and 0.996) and at
0.10 to 0.12 GB/s over write-combined memory (21.8x and 38.3x slower), the driver confirming the WC bit on
the WC buffer only. The two orders agree within 0.1 ms per kind. The arithmetic against lane A's cached pair
and the day-16 write-combined pair, and the two census questions it raises (the promote side, and day 16's
130 ms delta against a 1698 ms write-combined pass), are written once in `HOSTPREFIX-DOOR.md` "Review table
for the decide-by" and not repeated here.

## Item 6 result, local arm: the server ignores the door's flag (`serverdoor`, RTX 5090, replay `DAY18 REPLAY serverdoor: PASS (4 checks)`)

`rtx5090-day18/serverdoor/`: `memra-server` from this tree with `--experts-via-tier` on its argv, the Qwen3.5-9B
NVFP4 MTP GGUF, ready in 4.0 s, one completion returned text (`spec-acc ctx=8 burst=13/15`), TERM, drain
complete; 59..72 C, 164 W peak. The server log carries zero `experts-via-tier`, `expert-host-slru` or
`expert-gpu-slru` lines and no usage or unknown-argument line. Verbatim: `SERVERDOOR rule ready=True
request_ok=True door_lines=0 flag_refused=False flag_silently_accepted=True -> door_unreachable_in_serving`.
As expected from source: the server consults argv for `--version` and the key-lifecycle flags only. The
target-card arm follows the overlap pair below.

## Item 4 result: the synchronous miss path priced on the target card (`overlap`, replay `DAY18 REPLAY overlap: PASS (9 checks)`)

`pro-single-day18/overlap/`: one collector lock hold of 976 s (`marks.tsv` 19:21:42.232Z to 19:37:57.781Z),
twenty `run-gen` runs, `293bada0…` built from `e16bc69e8`, the approved artifact (`df27a780…`), the day-nine
environment (`MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986`, prompt `55 88 13`), OFF = no door,
ON = `--experts-via-tier` with the default 256 MiB host budget (16 host slots). Regime over the window
(3,894 samples at 250 ms, `command.gpu.csv`): 36 to 40 C, power draw at most 189 W under the 600 W limit,
SM clock 2332 to 2377 MHz. The first run (`o1-off-r1`) re-read the artifact from disk (55.0 s wall; lane A's
sitting had dropped the page cache); every later run loaded from the page cache. Collector `--validate`
rc=0 (`overlap-validate.log`). `executed-not-qualified`.

Integrity, all held: twenty runs exit 0 with `MATCH`; one `tokens:` tape across the twenty; `slots=9986` in
every cache line; one STEADY-STATE line across the twenty (`hit-rate=90.4% | 43.5 MB/decode-token`); every ON
run prints `[experts-via-tier] installed ... host_slots=16 max_expert_bytes=860160` and
`physical_reads=22077 owner_close=Ok(())`; no OFF run prints a door line.

| Run | decode gen-only s (OFF) | decode gen-only s (ON) | cumulative misses OFF / ON | ON `install_s` | `forward_s` OFF / ON | wall s (marks) OFF / ON |
|---|---|---|---|---|---|---|
| o1 r1..r5 | 0.408, 0.407, 0.407, 0.408, 0.408 | 2.368, 2.323, 2.320, 2.323, 2.301 | 8,211 / 18,195 (every run) | 75.53, 74.55, 75.06, 75.16, 74.55 | 0.65 / 6.26, 6.07, 6.20, 6.16, 6.03 | 5.5 (r1 55.0, cold) / 87 |
| o2 r1..r5 | 0.408, 0.409, 0.409, 0.408, 0.408 | 2.522, 2.305, 2.363, 2.380, 2.528 | same | 74.48, 75.14, 75.37, 74.44, 74.62 | 0.65 / 6.54, 6.30, 6.38, 6.41, 6.56 | 5.5 / 87 |

Pooled (N=10 per arm): decode OFF **0.408 s** (0.407 to 0.409), ON **2.343 s** (2.301 to 2.528); ratio
**5.743** (order 1 5.694, order 2 5.833); door cost **60.5 ms per decode token**, **1.39 ms per staged MB**
at the steady 43.5 MB per token; ON `install_s` median **74.8 s** (the SHA-256 pass over the 19 GB artifact,
the plan-derived catalog bind and the byte-for-byte compare of every retained record against the loaded
`HostExps`); `forward_s` (loaded to generated: prefill, warmup, decode) 0.65 against 6.28 s, ratio 9.6;
whole-process wall 5.5 against 87.2 s (medians from the driver's marks). Every ON run carries 22,077
`[expert-host-slru]` trace lines (one per host demand) inside its time; the 12,091 GPU evictions and 22,077
physical reads of day nine are reproduced exactly in every ON run. ON `physical_reads` (22,077) exceeds ON
cumulative GPU misses (18,195): the 16-slot host tier misses on its own re-reads as well.

Verdict, verbatim from the replay of the pre-registered rule:

`OVERLAP-PAIR rule decode_off_s=0.408 decode_on_s=2.343 decode_ratio=5.743 ratio_o1=5.694 ratio_o2=5.833 off_range=0.407..0.409 on_range=2.301..2.528 door_cost_ms_per_decode_token=60.47 steady_mb_per_token=43.5 door_cost_ms_per_staged_MB=1.390 misses_off=[8211] misses_on=[18195] reads_on=[22077] evictions_on=[12091] install_on_s=74.84 forward_off_s=0.65 forward_on_s=6.28 forward_ratio=9.627 N=5/arm/order pooled=10 orders=2 temp_c=36..40 power_max_w=189 power_limit_w=600.00 W identity=ok integrity=ok -> sync_miss_path_slower`

Observations, stated as observations:

1. **The synchronous miss path is 5.7x slower at decode at this budget, in both orders, with the tape
   identical.** 60.5 ms per decode token for 43.5 MB of staged experts is 1.39 ms per MB, about 0.72 GB/s
   effective, against the legacy SLRU's own miss path at the same slot count (the OFF arm stages the same
   bytes per token in 0.408 s for 32 tokens). The door's miss is one demand, one pread into the 16-slot host
   tier when that misses, one H2D and a full compute-stream drain, serialized; the legacy path prefetches
   (its cumulative misses are 8,211 against 18,195 because its detached prefetch turns demands into hits,
   and `prefetch_source` returns `false` under the bank, as pre-registered).
2. **The install is 75 s per process at this artifact.** That is the hash lock's SHA-256 over 19 GB (at the
   2.15 GB/s this host hashes, about 9 s of it) plus the record compare (every retained record read and
   compared byte for byte) and the catalog bind; it is the door's per-boot cost as landed, not a per-request
   cost, and it is not inside the decode ratio.
3. **`forward_s` carries the prefill misses too**: 6.28 against 0.65 s, ratio 9.6, so the synchronous path
   costs relatively more where the working set is cold (prefill) than at the steady decode.

Against the door, as pre-registered: `sync_miss_path_slower` holds, so item 4 is confirmed as the promotion
blocker at this budget on the target card. The door cannot sit behind the tiered materializer on speed
without several leases in flight retired on copy-stream events (and without reintroducing the
publish-before-completion hole `d0acf6f03` closed). The synchronous path is the door's only miss path;
there is no separate arm to delete for this item, so a delete decision at the decide-by deletes the whole
list in the door doc's "Decision at decide-by". The decision stays the review's.

## Item 6 result, target-card arm (`serverdoor`, replay `DAY18 REPLAY serverdoor: PASS (4 checks)`)

`pro-single-day18/serverdoor/`: `memra-server` `c195b89e…` from this tree with `--experts-via-tier` on its
argv, the approved artifact as `gate`, `MEMRA_MOE_RESIDENT=0`, ready in 8.0 s (page-cached), one completion
returned text (`spec-acc ctx=8 burst=7/24`; the MTP verify-graph pool declined on the non-resident MoE MTP
head and the eager verify walk served, a serving detail outside the door), TERM, drain complete; 37 to 45 C,
314 W peak under 600 W. Zero `experts-via-tier`, `expert-host-slru` or `expert-gpu-slru` lines, no usage or
unknown-argument line. Verbatim: `SERVERDOOR rule ready=True request_ok=True door_lines=0 flag_refused=False
flag_silently_accepted=True -> door_unreachable_in_serving`, the same line as the local arm.

For the review: the door has no serving surface, so the serving-shape bit-identity gate item 6 requires
(banked against native, solo against batched) cannot exist before a serving installer does; and the door's
flag reaches the server's argv and produces neither the door nor a refusal on either card (the server scans
argv for `--version` and the key-lifecycle flags only). The second is a hygiene finding, not a door defect:
`memra-server` rejects no unknown argument of any kind.

## Local battery on the final tree (`day18-local/`) and hygiene

`cargo fmt --all -- --check` clean (the one new Rust file, `hash_micro.rs`, rustfmt-clean); `git diff
--check` clean; `tools/check-flags.sh` every runtime `MEMRA_*` name resolves (no new read: `hash-micro` reads
none; the cell scripts set existing names); `python3 tools/check-public-boundary.py check` 582
grandfathered, 0 new; `tools/docs-registry-census.sh` clean (58 tables, 905 rows); `shellcheck -x` clean on
`day18-cell.sh` and `day18-run-cell.sh`; `py_compile` clean on `day18-replay.py`; the em-dash scan over the
day's prose, scripts, the bin and the commit messages finds none. Every local GPU run went through the
collector on `/tmp/memra-5090.lock` (`rtx5090-day18/*/lock.json`); every CPU-heavy local step ran under
`systemd-run --user --scope` with a CPU quota. BOX3 after the sitting: no `tmux` session, no `memra-server`,
`/tmp/memra-gpu.lock` free, `/root/wt-c` detached at `refs/bundle/c18` (`e16bc69e8`), receipts
`/root/spill-receipts/c-day18/` mirrored to `pro-single-day18/` without the binaries (`bins/` stays there);
`/root/artifacts` and `/root/memra-spill` untouched; the shipped bundle deleted on both ends.

## Push

Every push today in the announced development mode, verbatim shape `UNQUALIFIED DEVELOPMENT:
refs/heads/lane/spill-c-20260919 at <sha>; no GPU qualification claimed` then `pre-push: skip recorded in
.../.git/memra-gate-skips.log`: `922927619` (the merge of main), `7050170ba` (the pre-registration, before
any GPU run), `a3e27505a` (the first receipts), `19b3ee36b` (interim results and the review table), and the
day-docs tip named in `STATE.md`. No commit on main or the lead branch; no PR; the lead integrates.

## Effort and what remains

About 2.5 agent-hours against the 4-hour budget (the four target-card cells took 17 minutes of card time,
the overlap pair 16 of them; both builds were warm). Done: items 1, 3 (second half), 4 and 6 have their
day-18 inputs with verbatim verdicts and replays (item 1 as census and CPU proof, item 3's scale half as CPU
proof, both stated); the HOSTPREFIX review table with the hash micro-cell on both cards. Not done, stated:
a native item-1 cell (needs the installer in a PP gate binary, code), a native scale-admission cell (needs a
scale-bearing artifact), the arena lease handoff pricing, the DFlash tail slice, verify digest v3, the
RTX 5090-class pair for the HOSTPREFIX door, and the two census questions the hash arithmetic raised.
