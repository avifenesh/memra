# Hermes review remediation — 2026-08-25

Scope: re-validate the held Hermes security lane and every newer high/medium finding against
the current `origin/main`, carry forward only defects that still exist, and re-run a targeted
regression after each fix. Findings about unmerged scratch worktrees are not current-main defects;
research ideas and hardware-specific performance-policy proposals are not silently promoted into
generic defaults.

## Fixed

| Hermes fingerprint(s) | Result | Regression |
|---|---|---|
| `96a4273a44b3d41e` | DSV4 preserves its distinct `max` reasoning rung through all request surfaces. | `dsv4_reasoning_effort_max_survives_canonicalization`; Responses translation test |
| `272ee806bd66b3e4` | Empty stop elements are removed and can no longer match every decode. | `empty_stop_string_element_never_matches` |
| `e78d16ccf5e4fd4d`, `e7acb22291be4cef` | DSpark refuses unqualified Step TP/EP compositions at boot. | `refuse_list_covers_step_tp_and_ep` |
| `13bf6db5102bc2c8`, `0fd95dff2b1a2405` | `verify_exact` is an RAII scope, including every error path. | `error_path_restores_verify_exact` |
| `e8c630097486df02` | DFlash accept emission cannot cross `max_new`. | `accepted_run_never_exceeds_max_new` |
| `ee6465d077b47081` | Still-image dimensions are admitted before decode, and decode runs only after tenant admission. | `decode_bomb_refuses_pre_decode`; `vision_decode_is_deferred_and_grid_pinned` |
| `2628e6b8a6c53f95` | NVFP4 fused4 and sm_120a GDN-MMA gained kernel-check coverage. | kernel-check test target; full GPU battery below |
| `53f3d24b4566ce65` | Filter statistics use one deterministic cooperative program, chunked to the residency cap. | kernel-check target; full GPU battery below |
| `a8e42e60a313cdd0` | Filtered sampling now has mixed-batch isolation and survivor-admission teeth alongside the newer penalty-dispatch tooth. | decode-batch gate compiles; model-backed battery below |
| `30a15a935fab4c2b`, `ed2f8b593814b7ac` | DSV4 hc launchers chunk across CUDA's `grid.y` ceiling. | local RTX 5090: `hc_post` 4,194,304 values and `hc_collapse` 2,240,000 values, zero bit mismatches |
| `0d220d8c9a3eb634` | Both sigmoid-router entry points fall back from fixed-eight fast scratch when `n_used > 8`; dexp+fast composition is symmetric. | `fast_kernel_refuses_wide_topk_and_composes_with_dexp` |
| `da99e50ec4750599` | Host, sparse-device, and speculative sampling share one 8,192-token penalty window, so demotion cannot change penalty logits. | HTTP plumbing and cross-surface vendor-default tests |
| `c1346ff7164ac45a` | DFlash2 trim requires provenance from `MEMRA_FRSPEC_TRIM` gathering the target's own output head; an external student `d2t` is ignored and reported. | engine/server compile plus DSpark boot regression |
| `f42543cfdb2c8279` | FLAGS now records `MEMRA_SERVE_OVERLAP` as OFF/unwired; the substrate is no longer presented as served behavior. | flags census |
| `091da5e9d1d06ac4` | The DSpark row now distinguishes the default-off runtime from the explicit, hardware-specific 5090 recommended recipe and keeps older untrimmed-artifact results historical. | docs cross-check plus flags census |
| `64fa2b55baf0d887` | Live T-column dispatch uses one runtime-T program at every batch width; compile-time 2/4/8 twins remain research-only. | engine compile plus source dispatch census |
| `11339f5cd3c132a3` | FA row pointer tables are rebuilt from each live K/V/len/base tuple immediately before launch; the process-lifetime raw-pointer maps are gone. | engine compile plus stale-cache source census |
| `788210c5eae3555d` | The shared NVFP4 t-row workspace grows for larger shapes and never shrinks when spec, serving, and prefill widths alternate. | `workspace_grows_but_never_shrinks_between_spec_and_batch` |
| `58843bb6b924125b`, `7f35d87cd121dc11` | DSV4 and Step TP probes include production payload size classes instead of relying on a single 16 KiB transfer. | DSV4 ladder unit tooth plus engine compile |
| `b8eaacad622e696c`, `47476e821c68d14d` | Client-aborted sessions are billed/logged but never publish generated KV into reuse pools. | `aborted_sessions_never_publish_reusable_kv` |
| `155a3d2466a52184` | The process-wide park cap and oldest-first eviction now span plain, MTP-spec, and DFlash2 pools; failed eviction refuses insertion. | three-pool global-LRU ceiling tests |
| `f22a180d1638b95a` | A DFlash2 round clamps its committed prefix only at the request's true remaining-token boundary, then becomes terminal until a non-empty next-turn suffix resumes it. Crossing the scheduler's intermediate burst quantum keeps the full accepted surplus public and the session live. | `max_tokens_caps_the_committed_prefix_not_only_the_visible_slice` pins both the mid-request and final-boundary shapes; `spec_emission_keeps_intermediate_scheduler_surplus_public`; model-backed reuse gates |
| `be74abea27d26432` | FLAGS and `decode_step_chain` Rustdoc now document sampled-in-chain behavior, sampled chunk-boundary ids, and sampled `HEAD_SPLIT`; `run-gen` is named as the only caller and the false serving pin is removed. | flags/source caller census; exact-head Revuto review |
| `c5445b9e2343392f` | `MEMRA_PRIME_TROWS_T` truthfully ships width 8 and strictly refuses non-integer or out-of-range operator values instead of silently falling back. | `width_defaults_to_eight_and_refuses_invalid_operator_values`; flags census |
| `e42c696751dc02ab` | `MEMRA_SSE_COALESCE` is documented as off/unwired on current main; the unmerged buffering experiment is not presented as a serving default and requires ITL p50/p95/p99 before any future wiring. | runtime-reader census; flags census |

## Not current-main fixes

- Darklanes deploy-script findings remain in the private deployment repository and are not
  copied into this public engine change.
- Global device-penalty and Step-TP default-flip proposals remain hardware-specific policy
  decisions. Existing PRO receipts do not authorize a generic all-GPU default.
- Web-research and idea entries remain research inventory, not defects to merge without a
  native implementation and the repository's required correctness/performance gates.

## Final gate

The branch must be replayed on the latest `origin/main`, then pass formatting, diff/flags/public
boundary checks, engine and server suites, model-plan/compiler/reference suites, the DSV4 grid
gate, model-backed kernel/decode gates, and `tools/local-ci.sh --perf` before merge. The exact
final SHA and remote CI results are recorded in the pull request.
