# Graph-launch headroom guard sweep (lane/graph-launch-guard-sweep-20260831)

Extends the step37 admission lane's `GRAPH_LAUNCH_MIN_FREE` guard (256 MiB driver-free
floor; `research/step37-vram-admission-20260830/RESULTS.md` defect 3: `cuGraphLaunch`
segfaults inside libcuda at offset +0x27c87f, zero log lines, killing every session,
while eager arms fail recoverably on the same card) from the step35 spec-round arms to
EVERY remaining serving-reachable captured-graph launch in the engine. Base:
`b78b439bc` (origin/main); pre-land sync merge with `c4145956b`. Lane commits:
`d6701001e` (engine + worker guards and admission fixes), `838a0ee62` (docs +
coverage statements), `330321e49` (battery receipts), plus the perf-ci rows commit;
the landing merge SHA is the child of `c4145956b` on main.

Gate summary: (a) guard-fires squeeze N=5 PASS on the verify-graph route (m2..m6) and
N=5 PASS on the graph-session route (h4..h8, co-resident shape); (b) byte identity vs
base 6/6 cells 0/16 mismatches (greedy + seeded sampled, both serve shapes, x3
interleaved pairs); (c) `cargo test -p memra-engine -p memra-server` exit 0; (d) zero
ILLEGAL / #87 / panics / kernel faults in every run.

Every guarded route keeps the grep-stable `graph replay suspended:` key under its own
route tag, via `spec::graph_replay_suspended_note(route)` (once per process per route;
the original spec-round guard keeps its per-generation `[spec]` line).

## 1. Exhaustive launch-site enumeration and classification

Enumeration method: `git grep -n "\.launch()" crates/` plus the wrappers that pattern
misses: `TokenGraph::launch(&self, e)` call shape (`\.launch\(e\)`), raw
`sys::cuGraphLaunch` calls, and a `fn launch|GraphExec|launch_async` catch-all. All
line numbers are origin/main `b78b439bc`.

### Serving-reachable sites (guarded by this lane or already guarded)

| # | Site (b78b439bc) | Function | Route in SERVING (named in code) | Eager twin | Guarded before | Guarded now |
|---|---|---|---|---|---|---|
| 1 | spec.rs:13240 `sg.launch()` | `generate_spec_inner2` | MTP spec ROUND-STREAM burst (qwen35moe stream arm; `SpecSession` stepping in worker.rs `step_session`) | eager round path | YES (step37 lane, `graph_round_ok` @13220) | unchanged |
| 2 | spec.rs:13455/13475 `cg.interior/last[..].launch()` | `generate_spec_inner2` | MTP spec greedy chain draft | eager chain (`mtp_head_forward_dev` walk) | YES (@13426) | unchanged |
| 3 | spec.rs:13569/13571 | `generate_spec_inner2` | MTP spec sampled chain draft | eager sampled chain | YES (@13531) | unchanged |
| 4 | spec.rs:13679 `gr.launch()` | `generate_spec_inner2` | MTP spec greedy single-head draft (`dctx.graph`) | eager head forward | YES (@13651) | unchanged |
| 5 | spec.rs:13785 | `generate_spec_inner2` | MTP spec sampled single-head draft (`dctx.graph_s`) | eager sampled head | YES (@13758) | unchanged |
| 6 | spec.rs:3140 `self.full[&key].graph.launch()` | `DsparkVerifyGraphs::run_full` | dspark verify FULL trunk replay. THREE callers, all through `qwen35_verify_tparallel` (spec.rs:7367): (a) MTP spec vg door (`decode_step_t_core_vg`, spec.rs:14410, `MEMRA_SPEC_VERIFY_GRAPH` default ON for GDN+MoE: the ornith serve program); (b) dspark one-shot (dflash.rs:3004, `dspark_verify_graph_on`, opt-in bench arm); (c) dspark SERVE round (dflash.rs:4366, `dspark_verify_graph_serve_on`, DEFAULT ON since v0.108). MEASURED SCOPE CORRECTION (this lane, census runs 1-20 + code): caller (c) arms the pool ONLY for markov/plain-chain drafters; on a DFlash2 drafter (the box12 q38 shape) `deferred` is never set (dflash.rs serve burst: only the markov/plain-chain branch builds the device chain), the pool prints ENGAGED with ZERO captures, and these sites never launch. The serving reach of 3140/3299 today = the MTP vg door (ornith-class) + markov-drafter dspark deployments | eager cols-ckpt walk, same shared layer bodies (the existing `graphs=None` fallback the pool ceiling already takes) | NO | YES: guard inside `qwen35_verify_tparallel` (covers all 3 callers), `[dspark-vg]` line; plus the MTP vg door declines per round on `graph_round_ok` (covered by the `[spec]` line) |
| 7 | spec.rs:3299 `self.graphs[&key].graph.launch()` | `DsparkVerifyGraphs::run_segment` | dspark verify per-(segment, vt) linear-run replay; same three callers as #6 | same eager walk | NO | YES (same guard as #6) |
| 8 | spec.rs:4276 `dctx.graph...launch()` | `opti_graph_draft_step` via `opti_controller_draft_step` | optipipe controller probe draft (MTP spec route with the controller armed; spec.rs:14135/14158) | eager `mtp_head_forward_dev` arm inside `opti_controller_draft_step` (requires `eager_state`, which the round's eager main-draft arm seeds whenever `graph_round_ok` is false) | NO | YES: `round_graph_ok` param picks the eager arm; seed-unavailable stays a recoverable Err |
| 9 | decode.rs:75 `self.graph.launch()` | `GraphSession::step` | worker solo-interactive graph-replay session (worker.rs:9988 `g.step(&engine, &lm.model)`). REACHABILITY, measured this lane: OPT-IN `MEMRA_SERVE_GS=1` (default OFF, "load-stable default keeps every width on the generic batched body"), promotes RESTORED-HIT / POOL-RESUME solo sessions only (never cold prefill), and DEGRADES to batched-eager the moment a second session admits. No current prod launcher arms it | NONE by construction: the session IS the captured graph; no per-tick eager twin exists | NO | YES: refuses RECOVERABLY below the floor (`[graph-session]` line + session-scoped Err; the worker's step-FAILED error path ends the session, process and peer sessions live). FLAGGED: no eager twin, per the sweep contract nothing was invented. LIVE-FIRED in the co-resident cell (h4+: A refusals with A alive while co-resident B storms the shared driver) |
| 10 | hybrid_forward.rs:19856 `graph.launch(e)` (`TokenGraph::launch`, tp.rs:13767) | `step35_token_graph_step` | step35/step37 PLAIN decode whole-token graph on the TP stack, reached from `decode_step_h` (decode.rs:719) which serves plain rows and batch rows; armed only when `MEMRA_STEP_TP_GRAPH=1` + 4 sibling flags (all default OFF) | eager token step (`Ok(None)` early-out is the existing warmup/rebase fallback) | NO | YES: headroom early-out `Ok(None)`, `[step-tp-token]` line |
| 11 | tp.rs:11800 `sys::cuGraphLaunch(workspace.routes_graph...)` | `run_tensor_parallel_routes_nvfp4_device_routed_prejoin_add3` | TP MoE routed-expert prejoin graph door inside `moe_ffn_inner` (hybrid_forward.rs:5506) decode; armed by `MEMRA_STEP_TP_GRAPH=1` (default OFF) | the eager routes path below the door (the exact body the graph captures; stateless per call) | NO | YES: `step_tp_graph_headroom_ok(e)` in the door condition, `[step-tp-routes]` line |
| 12 | hybrid_forward.rs:2158/2283 `sm[il]/sg[il].launch()` | `prime_layers` | hybrid prefill S-mid/S-glue segment graphs; OPT-IN `MEMRA_PRIME_SEG=1` (default OFF, measured -0.7%-to-neutral) | eager fused `add_rms_norm_f16out` else-arm (byte-identical, the arm the opt-in was gated against) | NO | YES: `use_seg` requires headroom (probe only when the flag is armed), `[prime-seg]` line |

### Not serving-reachable (bench/gate binaries and offline generate paths; unguarded by design, verified no server caller)

| Site | Function | Reached from |
|---|---|---|
| decode.rs:118 | `GraphSession::prof_launch` | MEMRA_GS_PROF profiling (gates); guarding it would insert a driver call into the launch-cost measurement it exists for |
| decode.rs:2180 | `graph_decode_loop` | `decode::generate`/`generate_graph` only (bins: graph_decode_gate, decode_bench, fa_ab_bench, session_gate...); memra-server has zero `generate(`/`generate_graph` callers (verified by grep) |
| decode.rs:3075 | `gemma4_e4b_graph_exec_loop` | `decode::generate` only (offline E4B) |
| hybrid_forward.rs:14196 | `gemma4_generate_graph` | `decode::generate` + gemma_gate bin |
| hybrid_forward.rs:20349 | `step35_token_graph_chunk` | bin/run_gen only; guarded anyway (same `[step-tp-token]` early-out as #10, 3 lines, keeps step/chunk twins symmetric) |
| gemma_spec.rs:1254/1348/1456 | `generate_spec_gemma` (round graph; burst chain replay opt-in `MEMRA_GEMMA_BURST_GRAPH=1`; draft chain graph) | gates only (gemma_gate, gemma_spec_session_gate). The SERVING gemma spec route is `gemma_spec_session_burst` (worker.rs:16798) which is graph-free (verified: zero launch sites in it) |
| gemma_spec.rs:2266 | `gemma4_plain_graph_inner` | decode_bench only |
| prime_graph.rs:157 | `prime_graph_run` | bin/prime_graph_gate only (zero library consumers of `PrimeGraph` outside the module, verified by grep) |
| bins | launch_econ, prime_graph_smoke, tp_graph_probe, graph_allocfree_probe, memra-probe/pdl_probe | probe/bench binaries |

## 2. Admission-charging fixes folded in (fleet-peer refuted-read pass)

Scope addition from `darklanes research/admission-adoption-20260831/ASSESS.md` (two
gaps on the same surface):

1. **Boot calibration probes the SERVED route.** `run_boot_calibration` keyed on
   `mtp_spec_capable` and probed the MTP spec route on dspark boxes (whose MTP arm is
   DISABLED at drafter attach), then charged that probe's per-session draft-state
   figure at every spec admission for state the dspark route never allocates. Now the
   probe rides the dspark session for dspark-armed models, the receipt names
   `route=dspark|mtp`, MTP draft-state is never charged on dspark-armed models, and an
   MTP-skipped dspark boot is no longer skipped entirely (it used to keep the static
   floor unmeasured). Live receipt from this lane's cell (dev box, q38 + DFlash2):

   `[admit-cal] boot calibration done: model="q38" route=dspark transient floor 1824MB
   (static was 1536MB; measured 1824MB; probe kv charge 127MB, draft-state 0MB,
   drafted 69 accepted 41; ...)`

2. **Verify-graph pool debt charged BY STRUCT, not by route.** The MTP spec route's
   verify-graph door fills the same `dspark_vgraphs` pool (live receipt
   `[spec-vg] MTP verify-graph pool ENGAGED`) with the same monotonic growth, and
   escaped `dspark_vg_admission_debt` entirely behind a `dspark_drafts.contains_key`
   gate. The worker now charges the debt unconditionally and the debt fn consults both
   doors (`MEMRA_DSPARK_VERIFY_GRAPH` and `MEMRA_SPEC_VERIFY_GRAPH`/family default),
   still returning 0 when no door can engage or the pool has not captured.

Pool-magnitude prose note: the refuted-read pass killed the portable "8,852 MiB
high-water on q38" claim; pool sizes on other boxes are unknown until observed at boot
(the debt projection is self-measuring, which is why charging by struct is safe).

## 3. Gates

### (a) Guard-fires squeeze cell, VERIFY-GRAPH route

ROUTE EXERCISED, stated exactly: the cell drives spec.rs `run_full`/`run_segment`
(sites 3140/3299) through `decode_step_t_core_vg` into the guarded
`qwen35_verify_tparallel`, i.e. the MTP-route verify-graph door
(`MEMRA_SPEC_VERIFY_GRAPH=1` on the MTP-carrying q38 artifact; the ornith-class serve
program), with census receipts proving live captures (`[dspark-vg-census]`) before the
squeeze. SAID LOUDLY: the dflash-SERVE caller of the same two sites is NOT reachable
with on-box artifacts, and the reason is a finding, not a substitution: a DFlash2
drafter never arms the vgraphs ctx (zero captures across 20 census/debt observations;
dflash.rs sets `deferred` only on the markov/plain-chain branch on BOTH the serve and
bin arms), so no DFlash2 deployment can reach 3140/3299 at all. Both callers run the
SAME guard at the top of the SAME shared function; the cell exercises that guard on
the caller that exists in serving.

Squeeze mechanics that actually cross the floor (20 receipted failed approaches in
`raw/cell-squeeze-dsparkvg.sh` comments + banked logs): pre-ballast to a ~13GB world,
teeth door `MEMRA_ADMIT_RESERVE_MB=16`, solo MTP spec generation bursting, then a race
of concurrent multi-GB capacity-cache births (streamed penalized-greedy
max_tokens=125000) + a retrying chase ballast. Two instrument lessons for the fleet
peer's qualification bench: (1) nvidia-smi free is NOT driver free; the run-19 defer
receipts show driver-free 86MB while nvidia-smi read 717MB, and the crossing is only
visible to `mem_get_info` (the guard's own instrument) or the server's
`effective free (driver + pool-cached)` lines; (2) a foreign process's cudaMalloc is
refused well above the floor (~650MB nvidia-smi-free on this card), so external
ballast alone can NEVER fire the guard: the tail must be eaten by the server's own
allocations.

Run table (fresh server per run; the suspended note is per-generation on this route):
see section 4. FALSE-POSITIVE check on every run: zero `graph replay suspended:` lines
at healthy headroom, run killed otherwise (never tripped).

### (b) Byte identity vs base above the floor

Interleaved base/lane boot pairs x3, greedy (temp 0) + seeded sampled (temp 0.7,
top_p 0.9, seed 4242), 8 real agentic prompts x 128 tokens, on BOTH serve shapes
(dspark = box12 q38+DFlash2; mtp = the verify-graph-replaying shape). RESULT: 6/6
cells (pairs 1-3 x both shapes) at mismatches=0/16 with zero empty responses -
byte-identical base vs lane on every prompt in both modes, x3 default per the amended
A/B protocol (no anomaly, so no x5 escalation).

### (c) Existing test suites

`cargo test -p memra-engine -p memra-server` on the lane tree: exit 0, all suites
green (includes the spec unit suites: `dspark_vg_debt_projection` contract tests,
abort-park tests, worker admission tests).

### (d) Zero ILLEGAL / #87 / panics

Asserted per squeeze run from the server log (`illegal=0 sentinel87=0 panic=0` in
every run's receipts) and zero kernel fault lines (segfault/Xid/GPF) in per-run dmesg
windows. See section 4.

## 4. Battery (dev box, 2x RTX PRO 6000 Blackwell 96GB, CUDA 13.2, driver 595.91.07)

Artifacts (sha256 verified against `darklanes ops/serving/artifact-registry.tsv` pins
before any use):

- `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` = `1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a` (hf:Avifenesh/Qwen3.8-27B-NVFP4-MTP-GGUF)
- `q38-ranks-sxc32768.gguf.txt` = `2f7c2e94683ac552724c2275380de6de4035f0f597c71df0826c50826a7d1bb9` (pinned rev e4025e65)
- DFlash2 `model.safetensors` = `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`, `config.json` = `873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980` (hf:z-lab/Qwen3.8-27B-DFlash2 @ 50307d4c)

Binaries:

- lane `memra-server-lane-d6701001e` md5 `5966cee7fc8bf780f48e3cff75fbb8dd`
- base `memra-server-base-b78b439bc` md5 `1f3b51e6c3aa8624a854a8c331343ca7`

Driver scripts are BANKED in `raw/` (gl-lib.sh, cell-squeeze-vg-mtp.sh,
cell-squeeze-gsession.sh, cell-squeeze-dsparkvg.sh with the 20-run approach
archaeology in its comments, cell-identity-ab.sh, run-*-battery.sh, ballast.cu), not
just their outputs; the fleet peer reuses them for the box12/orn qualification bench.

### Guard-fires squeeze battery (cell-squeeze-vg-mtp.sh, lane binary, fresh server per run)

| run | suspended lines | fired at (nvidia-smi free; driver free is far lower) | vg captures pre-squeeze | loop completions under squeeze | recovery | alive | dmesg faults | verdict |
|---|---|---|---|---|---|---|---|---|
| m1 | 79 | 1162MB @ t=22s | 3 | 3/4 | OK | yes | 0 | all-green (verdict line said FAIL from a counting bug in the harness, fixed for m2+: `grep -c \|\| echo 0` emitted a two-line count) |
| m2 | 83 | 842MB @ t=22s | 3 | 3/4 | OK | yes | 0 | PASS |
| m3 | 40 | 1094MB @ t=208s | 3 | 14/25 | OK | yes | 0 | PASS |
| m4 | 1 | 814MB @ t=16s | 3 | 2/8 | OK | yes | 0 | PASS |
| m5 | 1 | 846MB @ t=16s | 3 | 2/8 | OK | yes | 0 | PASS |
| m6 | 81 | 1226MB @ t=22s | 3 | 3/4 | OK | yes | 0 | PASS |

N=5 clean PASS re-runs (m2..m6) + the all-green m1. Every run: `illegal=0
sentinel87=0 panic=0`, zero kernel fault lines, server alive through the squeeze,
post-release recovery request OK, zero suspended lines at healthy headroom
(false-positive check never tripped).

### Graph-session guard live-fire (cell-squeeze-gsession2.sh, co-resident shape)

The box10 two-stack shape: server A (plain q38, `MEMRA_SERVE_GS=1`, seed-and-rehit
loop so solo sessions PROMOTE to GraphSession) + co-resident server B (the MTP-vg
storm stack) on one card. B's birth-race exhausts the SHARED driver (B's own `[spec]`
line is the crossing's ground truth); A's promoted graph session refuses recoverably.
Run table (fresh servers per run; N=5 PASS):

| run | A [graph-session] line | A recoverable refusals | A step-FAILED endings | B ground-truth lines | fired after B | recovery | A alive | dmesg |
|---|---|---|---|---|---|---|---|---|
| h4 | 1 | 6 | 3 | 6 | t=2s | OK | yes | 0 |
| h5 | 1 | 6 | 3 | 5 | t=2s | OK | yes | 0 |
| h6 | 1 | 4 | 2 | 8 | t=2s | OK | yes | 0 |
| h7 | 1 | (run log) | (run log) | (run log) | t=2s | OK | yes | 0 |
| h8 | 1 | 8 | 4 | 5 | t=2s | OK | yes | 0 |

The exact banked worker line (raw/box-receipts/guard-line-excerpts.txt):
`[worker] graph session step FAILED (model q38): graph-session replay refused: driver
free below the 256MB launch floor (no eager twin for a captured session; ending the
session recoverably instead of segfaulting cuGraphLaunch)`.

Three findings
the earlier h/g runs banked (all receipted in raw/ and the cell comments): the route
is opt-in default-OFF; promotion is restored-hit/pool-resume only; and a single-server
storm can never squeeze a live graph session because concurrency demotes it, so the
co-resident stack is the only realistic firing shape (exactly the box10 two-stack
topology).

## 5. Fleet answer

One line for the fleet peer: after this lane, neither a box12 (qwen38 dspark) nor an
orn (ornith MTP + embed + rerank co-resident) deployment has ANY unguarded
captured-graph launch left in serving; box12's default shape reaches no captured
graph at all (two lane findings: a DFlash2 drafter never arms the verify-graph pool,
and GraphSession is opt-in default-OFF), while orn's live exposure was the default-ON
MTP verify-graph door plus the optipipe controller draft graph, both now guarded and
squeeze-receipted (N=5).

Per deployment, named in code:

- **box12 (q38 + DFlash2 dspark, single card).** dspark arming DISABLES the MTP spec
  arm (worker attach receipt), so the step37-guarded spec-round arms and the vg door
  never run; the DFlash2 drafter never sets `deferred`, so the dspark-serve vg caller
  (spec.rs:3140/3299) is unreachable (pool ENGAGED, zero captures, measured);
  GraphSession is default-OFF and unset in the launcher; `MEMRA_STEP_TP_GRAPH` /
  `MEMRA_PRIME_SEG` are off. Residual exposure was ALREADY nil-by-shape; every latent
  door (vg shared-walk, graph-session, TP token/routes, prime segments) is now
  guarded anyway. What box12 DOES gain from this lane is the admission pair: boot
  calibration now probes the SERVED dspark route (receipt `route=dspark`,
  `draft-state 0MB`) instead of charging a wrong-route MTP figure at every admission
  (the clean full-ctx session cut reverses), and the verify-graph debt term is
  charged by struct.
- **orn (ornith MTP + embed + rerank co-resident).** Ornith is GDN+MoE, so
  `MEMRA_SPEC_VERIFY_GRAPH` defaults ON and the MTP spec round replays
  run_full/run_segment through `decode_step_t_core_vg` (live `[spec-vg] pool ENGAGED`
  receipts predate this lane): that was THE unguarded serving route of the sweep, now
  gated per round (`graph_round_ok`) plus the shared-walk guard, and squeeze-receipted
  N=5 on the same program shape (GDN trunk + MTP + vg door). The optipipe controller
  draft graph (when armed) now follows the round's headroom snapshot. The step35
  spec-round arms were already guarded (step37 lane). Embed/rerank co-residents launch
  no captured graphs. Co-residency itself is the crossing amplifier (the h-runs prove
  one stack's storm exhausts the shared driver for its neighbor), and the by-struct
  vg debt charge now prices the MTP pool into orn's admission.
