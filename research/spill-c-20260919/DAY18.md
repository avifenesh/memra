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
