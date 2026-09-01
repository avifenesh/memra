# step37 bank-v2 removal: the two bad doors leave the engine

Lane: `lane/step37-bankv2-removal-20260829`, branched from `origin/main` @ `4d9cf5747f`.
Owner order: "if those are bad doors, should be removed." The doors are
`MEMRA_NVFP4_BANK_V2` and `MEMRA_SEL_DOWN8`.

Bisect citation (do not re-derive): `research/step37-reasoning-effort-20260829` (landed as
`75bf4ce76`) proved `MEMRA_NVFP4_BANK_V2=1` alone corrupts step37 generated text at prefill
(first token `Ass` vs `Got` at 25 prompt tokens, greedy, one binary, one boot recipe), that
the row's "outputs BIT-IDENTICAL to v1" claim was false, and that `MEMRA_SEL_DOWN8`
requires the v2 banks. That lane fenced the doors with a step37-only boot refusal; this
lane removes them.

## Removal shape chosen, and why

**Full deletion of the doors and every TP code path that read the v2 banks, with one
carve-out: the slot-major layout itself survives as the fixed, env-independent layout of
the EP2 whole-expert banks.**

Deleted (commit on this lane):

- The env reads `nvfp4_bank_v2_on()` and `sel_down8_on()` (tp.rs). The vars are now read
  NOWHERE in the engine.
- The TP-bank permute arm: `nvfp4_repack_bank_matrix` takes an explicit `slot_major: bool`
  and the TP column/row shard banks are always block_nvfp4 v1.
- TP decode kernels that read the v2 layout: `qmatvec_nvfp4_dp4a_sel_v2`, `_sel_v2s`
  (with the `MEMRA_SEL_V2S` probe door), the gate/up fusion family `_sel_v2_gu`,
  `_gu_r2`, `_gu_r4`, `_gu_wpr` (with the `MEMRA_SEL_GU_RPW` and `MEMRA_SEL_GU_WPR`
  sub-doors, receipts FLAT/-3% and never-gated respectively), and the `MEMRA_SEL_DOWN8`
  kernels `_down8` and `_down8_rows`, plus their Rust wrappers and dispatch arms.
- The two-column/t-row verify MoE program, which hard-required the v2 banks
  (`run_tensor_parallel_routes_nvfp4_device_routed_tn`/`_tn_prejoin`, `Nvfp4T2Workspace`,
  `step35_verify_moe_tn`, the `_gu_tcol` kernel) and the doors only it read:
  `MEMRA_TN_PREJOIN` (family-armed 2026-08-27, but unreachable in serving since the
  2026-08-29 refusal), `MEMRA_TN_TRACE`, `MEMRA_WALK_SCRATCH` (own cell already FLAT),
  `MEMRA_SWEEP_TRACE` (its subject was the down8 dispatch).
- The `moe-sel-census` bin (it priced the deleted gu fusion).

Kept, with reasons stated:

- **EP2 (`MEMRA_STEP_NVFP4_EP2`) and its `*_ep` kernels.** The EP2 kernels read the
  slot-major layout unconditionally by design, and the FLAGS row records an owner-visible
  keep decision ("stays as the EP substrate for a future t-row batched step-TP program").
  Removal FIXED a latent defect here: EP2's bank build permuted only when the env door was
  on, so `MEMRA_STEP_NVFP4_EP2=1` without `MEMRA_NVFP4_BANK_V2=1` built v1 bytes under v2
  readers (garbage). EP2 banks are now ALWAYS slot-major; the layout is a property of the
  bank, never of the environment. The grouped prime's dequant qtype is likewise keyed on
  `experts.ep2` instead of the env.
- **`qmatvec_nvfp4_fast_v2` + the `qmatvec_nvfp4_dp4a_v2` kernel**: EP2's host-canonical
  oracle reader for its slot-major banks.
- **`QT_NVFP4_V2` and the moe_f16_grouped GEMM instantiation**: EP2's prime path and the
  `moe-tp2-repro` offline harness (`MEMRA_TP2_QT=107`), which is the remaining offline
  diagnostic surface for the layout.
- **`nvfp4_matrix_v2_permute` and its layout unit test** (`bank_v2_layout_tests`): the
  permutation feeds the EP2 banks and stays pinned to the documented mapping.
- **`MEMRA_TCOL_FFN`** keeps only its implied `MEMRA_TCOL_OPROJ` defer. It is family-armed
  in the step37 serving defaults; deleting it would have silently dropped the o_proj defer
  from the qualified serving shape. Its named feature (the two-column FFN sweep) is gone;
  the FLAGS row states this. A future lane may fold it into `MEMRA_TCOL_OPROJ` with its
  own receipts.

**The refusal became the permanent guard, not dead code.**
`unqualified_bank_v2_refusal` (step37-only, one var) was replaced by
`removed_bank_v2_doors_refusal` (worker.rs): ANY model, EITHER var set to `1`, refuse at
boot with the removal and receipt pointers. Why this is the simpler honest code: the vars
are read nowhere else, so "inert" was available for free, but a recipe that still sets
them was written against a binary where they changed the serving program, and the bisect
lane's whole lesson is that serving quietly under a wrong assumption is the worst failure
shape (fluent wrong answers, every counter green). A loud boot refusal with the receipt
pointer costs one function; recipes on this very dev box (`/root/agentic8.sh`) still carry
both flags today. The new guard is simpler than the old one (no model-class parameter, no
family scoping) because the flags are now invalid for every family, not unqualified for
one.

**The named follow-up lane (device-side v1-vs-v2 bank oracle) is moot for serving.** The
mismatched v1-vs-v2 TP reader was never localized; with the TP v2 paths deleted there is
no TP v2 reader left to localize. The layout's only remaining consumers (EP2 `*_ep`
kernels, `fast_v2`, the `QT_NVFP4_V2` GEMM) always read the layout their banks are always
built in. If EP2 is ever promoted toward serving, its argmax gate + battery (DEV_ROUTES
acceptance class, already named in its row) is the gate, and `moe-tp2-repro` remains the
offline instrument.

## Other qualified consumers of the v2 banks: the answer

Checked by reading every reader/writer (grep receipts in the lane transcript):

- `MEMRA_STEP_NVFP4_DEV_ROUTES` / `MEMRA_STEP_TP_DEV_ROUTER` (qualified, family-armed
  serving doors): do NOT require the v2 banks; their t=1 program runs the v1 `sel` sweep.
  Confirmed by prod, which serves the full corrected env correctly today.
- `MEMRA_TN_PREJOIN` (family-armed, +8.06% receipt) and `MEMRA_TCOL_FFN`'s FFN arm rode
  the v2-only tn program. Their receipts were measured WITH the v2 banks armed, the same
  configuration the bisect discredited; both arms had been unreachable in serving since
  the refusal, so removal changes no serving bytes (byte gate below).
- EP2: receipted NEGATIVE (-4.5% at B=1), default off, kept as substrate (see above). Its
  gates were necessarily run with the v2 layout, which is now its unconditional layout, so
  those receipts keep their meaning.
- No other flag, test, tool, or bin consumed the TP v2 banks.

## Gates

All cells on the step37 dev box (2x RTX PRO 6000 Blackwell Server, TP2), model
artifact `/root/models/step37-flash-nvfp4` (stepfun-ai Step-3.7-Flash-NVFP4), GPU lock
held for the whole battery. Serving env = the box's `agentic8.sh` ENVV minus the two
removed doors, plus the qualified spec-on policy (`MEMRA_LOAD_MTP=1 MEMRA_MTP_HEADS=3
MEMRA_SPEC_K=3 MEMRA_SPEC_PMIN=0.5 MEMRA_SPEC_PMIN0=1 MEMRA_CTX=262144
MEMRA_SERVE_SPEC=1`), the corrected-prod shape. Probe = the bisect lane's own
`probe5.py` (4 real short prompts x 2 greedy reps, temp 0.0, max_tokens 320).

| gate | arm(s) | result |
|---|---|---|
| cargo tests | `-p memra-server --lib` on the rebased tip | **404 passed, 0 failed** (incl. all 11 reasoning-effort mapping tests and the new refusal test) |
| cargo tests | `-p memra-engine --lib` | **246 passed, 0 failed, 2 ignored** (incl. `bank_v2_layout_tests`) |
| warning parity | per-file warning fingerprints vs an origin/main baseline worktree | **IDENTICAL** (23 == 23, same files; no new warnings) |
| byte gate | baseline `4d9cf5747` (md5 `0a850b58...`) vs removal (md5 `34192b3f...`), corrected env, spec-on, greedy temp 0.0, 4 prompts x 2 reps | **BYTE_GATE_PASS: 8/8 rows bytes-identical across binaries; r0==r1 within every arm** (`raw/gates2.log`, `raw/v2-a1-baseline-rows.json`, `raw/v2-a2-removal-rows.json`) |
| correctness | removal arm answers | **4/4 sane, 3/4 exact-HIT** (`391`, `Paris`, `366`; the `desserts` reversal misses identically on BOTH arms and in the bisect lane's own clean arms: a model limitation, and the byte gate is the bar) |
| arm identity | listener pid == booted pid via `lsof -t -i :PORT` per boot | **ARM_IDENTITY_OK on both A arms** (added after this lane briefly reproduced the banked pkill-basename trap, below) |
| refusal | removal + `MEMRA_NVFP4_BANK_V2=1` | **REFUSED in <=2s, exit rc=1**, `[server] FATAL: worker init failed: MEMRA_NVFP4_BANK_V2=1 is set, but these flags were REMOVED from the engine on 2026-08-29...`, health `000`, ZERO model-load attempts (`raw/v2-c1-bankv2.log`) |
| refusal | removal + `MEMRA_SEL_DOWN8=1` | **REFUSED in <=2s, rc=1**, message names `MEMRA_SEL_DOWN8=1` (`raw/v2-c2-seldown8.log`) |
| refusal | removal + both | **REFUSED, rc=1**, message names both flags (`raw/v2-c3-both.log`) |
| sampled probe | vendor-default (NO sampling params) on the removal arm | **HTTP 200, spec ENGAGED: rounds=66, drafted=152, accepted=128, rate=0.842**, `[spec-acc]` lines in the server log (`raw/v2-d-sampled.json`, `raw/gates2.log`) |
| log hygiene | every cell | **ILLEGAL=0, #87=0, panic=0** in all boots; GPUs at 0 MiB after the battery |
| build attribution | both binaries | real rebuilds (257s / 240s), distinct md5s, `strings` markers: baseline carries the OLD step37-scoped refusal string, removal carries GEMM_PRIME's (kept) + the new REMOVED message (`raw/build-both.log`, `raw/rebuild.log`) |

### Two findings from running the gates, stated

1. **The banked pkill-basename trap fired again, and the arm-identity law caught it.** The
   first battery ran the arms as `memra-server-baseline`/`memra-server-removal`; `stop()`'s
   `pkill -x memra-server` matched neither (comm truncates at 15 chars), the baseline server
   survived its own stop, answered the "removal" probes (a perfect false byte-PASS: baseline
   vs itself) and starved every later boot into `CUDA_ERROR_OUT_OF_MEMORY`. The v2 battery
   keeps the basename and changes the directory, kills by recorded pid, asserts
   listener-pid == booted-pid after every health 200, and waits for VRAM drain between
   cells. The invalid first run is preserved in the transcript; only v2 receipts are cited.
2. **The pre-load refusal exits rc=1, not 139.** Moving the removed-doors check BEFORE model
   load (it is env-only, unlike the model-class-scoped refusals) means no CUDA context
   exists when it fires, so the known worker-teardown race cannot happen and the exit code
   is the deterministic 1 the code asks for. The 139 artifact remains for the post-load
   refusals (`dead_prime_kill_switch_refusal`) and stays with its own named lane.

## Known, pre-existing, out of scope

Both refusal paths exit 139: the GPU worker thread races `std::process::exit(1)` during
CUDA teardown. Attributed and named in the bisect lane (its `raw/cell14-*`); shared with
`dead_prime_kill_switch_refusal`; has its own lane. The operational contract holds
(nonzero exit, named FATAL on stderr, no service, no leaked process).
