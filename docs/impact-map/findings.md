# Findings

Marked while mapping the repo (owner's ask: mergeable, reusable, decoupling needed, errors). Nothing here is fixed by this change.
Defects (errors, broken gates, dead code, unreachable paths, stale flags and docs) were checked by an independent skeptic agent told to refute them; refuted ones are not listed. `verification` says what happened to each.
Every row's quote was checked against the tree at the map commit.

| category | count | high | medium | low |
|---|---|---|---|---|
| Errors (real bugs) | 67 | 4 | 26 | 37 |
| Broken gates | 18 | 0 | 8 | 10 |
| Dead code | 35 | 0 | 0 | 35 |
| Unreachable paths | 9 | 0 | 1 | 8 |
| Stale flags and doors | 59 | 0 | 12 | 47 |
| Stale docs | 80 | 0 | 0 | 80 |
| Decoupling needed | 55 | 2 | 26 | 27 |
| Mergeable | 42 | 0 | 4 | 38 |
| Reusable | 23 | 0 | 2 | 21 |

## Errors (real bugs)

### high: Gemma4 PP-2 prime of 256+ tokens runs the serial (non-gemma) layer program

- where: `crates/memra-engine/src/hybrid_forward.rs:5879` (S03, confirmed)
- detail: The loader admits gemma on PP-2 as qualified_gemma_pp2 regardless of MEMRA_PP_STREAMS (hybrid.rs:4440). prime_cache_overlaid_inner routes text-only gemma through the generic chunk driver, and gemma's ops all carry the chunked_prime receipt, so ranges come from prime_chunk_ranges. With the PP auto geometry (prime_pp2_auto_geometry is true by default on a 2-stage fence) a 300-token prompt gets chunk max(ceil(300/8),128)=128, so 3 ranges. The pipelined arm at 5879 has no uses_gemma_program() exclusion and calls prime_cache_pp2_pipelined. prime_pp2_stage0_enqueue then calls self.embed with no sqrt(n_embd) scale (7572) and prime_layers, the serial residual body: generic full_attn_prime with no SWA/K=V geometry, a SiLU dense FFN instead of GELU, and no post-ffw norm or layer scale. Stage 1 does the same (7603). Result: wrong KV and logits on a 2-card gemma4 31B/26B deployment, with no error. prime_chunk already excludes gemma from its ppN arm (6654), so the omission here is an oversight.
- check: The gemma PP-2 load is admitted whatever MEMRA_PP_STREAMS says. Text-only gemma with a chunked-prime receipt gets prime_chunk_ranges, so a prompt over the PP-2 auto geometry or the 4096 cap gets 2+ ranges. The pipelined arm at 5879 has no uses_gemma_program() exclusion. prime_pp2_stage0_enqueue/stage1_enqueue call self.embed with no sqrt(n_embd) scale and the generic prime_layers. prime_layers has no gemma branch; only prime_chunk switches to prime_layers_gemma and scales. prime_chunk's own ppN arm excludes gemma (6654), and ppn_gate skips the gemma pipelined arm for decode only. Nothing refuses this path, so the KV and logits come out wrong with no error.
- fix: Add `&& !self.uses_gemma_program()` to the PP-2 (and PP-3/4) pipelined arms so gemma takes the serial prime_chunk loop, and add a gemma PP-2 multi-range cell to the gemma4 chunked-prime gate.

### high: Gemma-4 E4B is documented as Supported, but no pack matches it, so HybridModel refuses to load it

- where: `crates/memra-gguf/src/model_packs/gemma4_dense/mod.rs:45` (scout:packs, confirmed)
- check: The config parsers set n_embd_per_layer (E4B: 256) and shared_kv_layers (E4B: 18). Both gemma4 packs require both to be 0, so for_config returns None for E4B. compile_for_load then returns NoMatchingModelPack, and the comment there says 'Pack refusal is final'. HybridModel::load calls compile_for_source, which calls compile_for_load (hybrid.rs:4245). gemma_gate uses HybridModel::load too. So E4B cannot load at all. Meanwhile docs/models/gemma4-e4b.md says Supported, and MODELS.md and current-board.json list it with a 199.9 tok/s row. The PLE and KV-share engine code (hybrid.rs:4726, memra-kv lib.rs:3096) is now unreachable. Severity: a documented, benchmarked supported model fails to load.
- fix: Add an E4B pack (or widen gemma4_dense to accept PLE and KV-share) with a tiny plan and gate. Otherwise mark E4B unsupported in docs/models/gemma4-e4b.md, MODELS.md and current-board.json, and delete the E4B-only engine arms.

### high: Metered deployments skip the embedding EOS pooling append

- where: `crates/memra-server/src/lib.rs:9981` (S19, confirmed)
- detail: When a deployment Metering enforces limits, admit_tenant_budget_choices calls prepare_budget_prompt, which renders and stores request.prepared_prompt on the HTTP side. The worker's prepare (worker.rs:33409) runs its whole render block only when prepared_prompt is None, and that block contains the hidden-capture EOS append (worker.rs:33478: Qwen3-Embedding pools on a trailing eos). Scenario: a deployment binary with enforcing metering, a limited tenant, and POST /v1/embeddings {input:"hello"} on Qwen3-Embedding-8B. embed_api.rs:212 runs admit_tenant_budget, prompt_text is tokenized without the eos, the worker skips the append, and the vector is pooled one position early. The worker's own comment measures that at about 0.3 cosine against the reference instead of 0.998. The stock binary (no metering) is unaffected.
- cut: Once render and post-render are one function, a change to it reaches embed/rerank/prime cells once, not via two copies.
- check: embed_api builds a capture Request with prepared_prompt None, chat false and the input in prompt_text, then calls admit_tenant_budget. With an enforcing Metering and a limited tenant, admit_tenant_budget_choices calls prepare_budget_prompt. That encodes prompt_text via render_prompt_ids, with no eos append, and stores prepared_prompt. The worker's prepare runs its render block, including the hidden-capture eos push, only when prepared_prompt.is_none(). So the eos is never appended. The worker comment says pooling one position early scores about 0.3 cosine against the reference. Nothing else appends it. Metering is a public trait that deployments plug in; the stock binary has no enforcing implementation, so only metered deployments are hit. On those, every limited tenant gets silently wrong embeddings.
- fix: Factor the post-render steps (capture eos append, vision guard) into one function applied to every prepared prompt, whether the HTTP budget path or the worker rendered it, or move them after the worker's is_none block.

### high: Dark-lane concat prime batches carried routed-MoE continuations the interactive path refuses

- where: `crates/memra-server/src/worker.rs:31765` (S18, confirmed)
- detail: The interactive prime batch admits a carried chunk only when carried_prime_batch_eligible(plan) && rewrite_allowed(CarriedPrime) (31049-31057, commented 'P0 coldhol guard: routed-MoE carried batches stay serial'). The dark-lane batch (31737-31791) admits FRESH or CONTINUATION sessions (cache.pos == fed.len(), fed non-empty) with no CARRIED_PRIME check. The engine's prime_cache_batch_inner checks only the CarriedPrime rewrite (hybrid_forward.rs:8336), not the CARRIED_PRIME trunk capability, and MoeMlp has no carried_prime surface in op_registry. Scenario: two harvest-lane requests on a qwen35_moe (or qwen3_moe/olmoe/hy3/minimax_m3) model. Both have a markerless prompt, so ckpt_at is None, and both have a prefix-cache or continuation hit, so fed > 0 with cache.pos == fed. They are concatenated into one prime_cache_batch at base > 0. That is the unqualified numeric class the interactive lane serializes. The same request primes under a different program depending on its lane and on which peers arrive in the same tick, which breaks the one-numeric-program-per-request rule.
- cut: After the fix, dark-lane carried prime touches only CARRIED_PRIME-supported plans (dense qwen3/qwen35/llama_dense) instead of every non-eager family.
- check: The interactive batch admits a continuation chunk only through cold_chunk, which requires n_cached==0, no prefix_pin, carried_prime_batch_eligible and the CarriedPrime rewrite. The dark-lane batch admits any session with cache.pos == fed.len() and has no carried_prime_batch_eligible check. In the engine, prime_cache_batch_inner checks only rewrite_allowed(CarriedPrime), and that returns true when rewrite_qualifications is None (the default). MoeMlp has no carried_prime surface. Continuations really occur on dark lanes: the default dark budget is 256 and the single-chunk dark path primes one chunk per tick, so any dark prompt over 256 tokens becomes a continuation that the next tick can concat with a peer. FLAGS.md records that this class (routed-MoE repeated continuation batches) picked EOS at 26/60 tokens where serial primes reached 60/60, which is wrong output.
- fix: Factor the interactive cold_chunk predicate into one shared fn and require it (carried_prime_batch_eligible plus the CarriedPrime rewrite, n_cached==0, no prefix_pin) for dark candidates whose fed is non-empty.

### medium: Header rerun list omits memra_pdl_chain.cuh and dsv4_dense_census.cuh: header edits ship stale kernels

- where: `crates/memra-engine/build.rs:553` (S22, confirmed)
- detail: Once a build script emits any rerun-if-changed, Cargo reruns it only for the listed paths. The static-archive loop lists mmq_common, dsv4_dense_m1_exact_tail, dsv4_dense_tile, mmq_mma_i8, sm100_blockscale_layout, moe_kq_prims and wgmma_common. It never lists memra_pdl_chain.cuh (included by dsv4_gpu.cu, dsv4_sampler.cu, moe_f16_grouped.cu, tp_ar.cu) or dsv4_dense_census.cuh (dsv4_gpu.cu). Failure: edit memra_pdl_chain.cuh, then run `cargo build --release` (fast-gate.sh:325, serve-smoke.sh, local battery). build.rs does not rerun, libmemra_mmq.a and MEMRA_MMQ_ARCHIVE_HASH keep the old bytes, and every gate runs the previous DSv4 decode chain, glm5_next hc kernels, MoE f16 grouped GEMM and TP all-reduce while reporting the new tree. That is the stale-kernel class the archive-hash block at build.rs:706-710 was written to stop.
- check: The static-archive loop lists seven headers, but not memra_pdl_chain.cuh (included by dsv4_gpu.cu, dsv4_sampler.cu, moe_f16_grouped.cu, tp_ar.cu, dsv4_dense_m1_exact_tail.cuh and dsv4_replay_control.cuh) or dsv4_dense_census.cuh (included by dsv4_gpu.cu:4304). Cargo watches only listed paths, not transitive includes, so listing dsv4_dense_m1_exact_tail.cuh does not cover the header it includes. mma_tile.cuh and dsv4_replay_control.cuh are also unlisted.
- fix: Emit cargo:rerun-if-changed for every cu/*.cuh (glob the directory once) instead of a hand list, in both compile loops.

### medium: MEMRA_DSV4_CUTLASS hardcodes an sm_120a gencode and has no arch assert. Unlike MEMRA_CUTLASS (build.rs:273), a 100a/90a/89 build with it set links sm_120a code into the wrong-arch binary. Its rerun-if-env-changed (line 757) is printed only inside the branch, so setting the var after a build without it does not rerun build.rs.

- where: `crates/memra-engine/build.rs:767` (scout:devices, confirmed)
- check: Both parts hold. The MEMRA_DSV4_CUTLASS block hardcodes arch=compute_120a,code=sm_120a with no arch assert; MEMRA_CUTLASS has one at line 273, and dsv4_gpu.cu calls the weak symbol whenever it is linked, so a non-120a build runs sm_120a code. Also, rerun-if-env-changed=MEMRA_DSV4_CUTLASS is printed only inside the if. The script emits other rerun-if lines, so cargo will not rerun build.rs when the var is set after a build without it, and the CUTLASS path is silently left out. The env is an opt-in build switch, so the impact is narrow.
- fix: Move the rerun-if-env-changed=MEMRA_DSV4_CUTLASS println next to line 253, and assert cuda_arch == "120a" (or use the build's gencode) when it is set.

### medium: H100 PDL launches run kernels whose MEMRA_PDL_ENTRY wait is compiled out

- where: `crates/memra-engine/cu/kernels.cu:15` (S15, confirmed)
- detail: build.rs defines MEMRA_PORTABLE_CUDA for 90a as well as 89 (build.rs:270/309). On the Hopper build the PDL entry macro is therefore empty in kernels.cu:15, flash_attn.cu:59 and qmatvec.cu:18. Engine::launch_pdl and launch_pdl_flash still set CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION (lib.rs:5211) with no arch check, and MEMRA_PDL defaults on (lib.rs:4945). lib.rs:5142 states the contract: only kernels that carry the macro may use the attribute. Scenario: on an H100, a decode step launches rms_norm_q8_1 (kernels.cu:700) or fa_decode_vec_q_rows_v4_w through launch_pdl right after a matvec. The dependent grid may be scheduled after the primary's implicit trigger, with no griddepcontrol.wait to guarantee the primary's stores are visible. The programming model allows the dependent to read stale activations, an intermittent numeric fault on h100 cells only. memra_pdl_chain.cuh:19 keys the same instruction on __CUDA_ARCH__ >= 900 alone, so dsv4/tp_ar kernels do wait on 90a.
- check: build.rs sets portable for both "89" and "90a" and passes -DMEMRA_PORTABLE_CUDA=1 to the kernels.cu, qmatvec.cu and flash_attn.cu fatbins. On 90a the entry macro is therefore empty in all three TUs. launch_pdl sets PROGRAMMATIC_STREAM_SERIALIZATION with no arch or cfg check, pdl_on defaults true, and nothing references memra_portable_cuda near PDL. That breaks the contract lib.rs states itself. memra_pdl_chain.cuh keys on __CUDA_ARCH__ >= 900 alone and issues launch_dependents at entry. A chain kernel followed by a launch_pdl kernel on H100 is therefore a real race, not only a missing memory flush.
- fix: Key MEMRA_PDL_ENTRY on __CUDA_ARCH__ >= 900 only (as memra_pdl_chain.cuh does), or make pdl_on() return false under cfg!(memra_portable_cuda) && !cfg!(memra_hopper_mma).

### medium: FP8 block checkpoints fail to load on h100/sm89: the NaN-scan stub refuses

- where: `crates/memra-engine/cu/mmq_fp8_blk_stub.cu:32` (S14, confirmed)
- detail: On 89/90a builds build.rs swaps in mmq_fp8_blk_stub.cu, whose memra_fp8_blk_count_nan returns 2905. Engine::fp8_blk_nan_count turns any rc!=0 into Err (mmq_ffi.rs:2511). MEMRA_ST_E4M3 defaults on, so loading any block-128 FP8 safetensors checkpoint (e.g. Qwen3.8-27B-FP8, hy3/glm5_next/step35 FP8) calls e.fp8_blk_nan_count(&bytes)? at model.rs:1022 (and hybrid.rs:2147 for stacked experts) and the `?` aborts the load with 'memra_fp8_blk_count_nan rc=2905'. The code comment at model.rs:1000 promises a non-zero count declines to the Q8_0 floor, but the refusal is an error, not a count, so neither the native e4m3 GEMV (portable qmatvec_e4m3_blk_mmvq) nor the Q8_0 floor is ever reached. The real counter (fp8_blk_count_nan_kernel, mmq_fp8_blk.cu:856) is a plain byte scan with no f8f6f4 instruction, so nothing requires stubbing it.
- check: build.rs swaps in mmq_fp8_blk_stub.cu on every arch other than 120a and 100a. In the stub, memra_fp8_blk_count_nan returns 2905. fp8_blk_nan_count turns rc!=0 into Err. st_e4m3_blk_enabled is default ON (MEMRA_ST_E4M3 != "0"), and nothing checks the arch before model.rs:1022. So on sm_89/sm_90a a block-128 FP8 tensor that passes the shape checks hits `?` and aborts the load. It never declines to the Q8_0 floor as the comment promises. hybrid.rs:2147-2149 and fp8_ffi.rs:429 behave the same way. The real kernel is a plain byte scan with atomicAdd and needs no f8f6f4. The failure is loud (a load error), not wrong output, so medium.
- fix: Move fp8_blk_count_nan_kernel and memra_fp8_blk_count_nan into a TU built on every arch (e.g. fp8_blk_dequant.cu) and drop it from the stub, or have the load sites treat a refused scan as 'decline to the Q8_0 floor'.

### medium: glm_dsa (MLA, serial residual) reaches the generic batched body and panics on every decode chunk

- where: `crates/memra-engine/src/decode_batch.rs:3176` (S05, confirmed)
- detail: eager_only_model diverts only the Gemma program, E4B and HyperConnections plans (worker.rs:38874-38884). A glm_dsa plan is none of these, so with default batching its decode ticks go to decode_step_batch_sampled_lean_masked. rewrite_allowed(DecodeBatch) is true when no rewrite bundle is installed (hybrid.rs:3974), b1-fast is off by default, so even a B=1 chunk falls through to decode_batch_layers. There the MLA layer calls mla_path_unimplemented, which panics (hybrid.rs:2660). The worker comment at worker.rs:39388 already says the generic body panics on these mixers; chunk_cap_for only narrows the width to 1, it does not reroute. Result: every glm_dsa request becomes a caught panic (engine_error) after prefill, while decode_step_h, which does support MLA through mla_attn_cached, is never used.
- check: eager_only_model covers Gemma, E4B and HyperConnections (decode_batch_unconverted = HyperConnections only). glm_dsa is a Serial-residual MLA pack, so it stays out of eager_decode and its sessions go to the batched chunks. chunk_cap_for only returns 1. In the engine, rewrite_allowed is true with no bundle, so the eager-row fallback is skipped. hyper is None, b1-fast is off by default and the program is Generic, so execution reaches decode_batch_layers, where Mixer::Mla panics. The chunk_cap_for comment itself says the generic body panics on these mixers.
- fix: Add serial-residual MLA/KDA plans to eager_only_model, or have the engine use its decode_step_h row fallback whenever DECODE_BATCH trunk support is false.

### medium: Verify-width ladder panics when MEMRA_DFLASH_VERIFY_T=2

- where: `crates/memra-engine/src/dflash.rs:6593` (S11, confirmed)
- detail: vt_cap is parsed from MEMRA_DFLASH_VERIFY_T and clamped with .clamp(2, nd + 1) (line 6053), so 2 is an accepted value. The adaptive ladder then calls (m + 2).clamp(3, vt_cap). With vt_cap = 2, Ord::clamp asserts min <= max and panics. Scenario: MEMRA_DSPARK_SPEC=1, a head-less DSPARK or DFlash2 drafter (ladder policy, MEMRA_DFLASH_ADAPT default on), MEMRA_DFLASH_VERIFY_T=2. The first served round panics inside the worker thread and the server dies. The same expression sits in the bin loops at 3818 and 4603.
- check: vt_cap is clamped to 2..=nd+1, and FLAGS.md documents 2 as valid ('clamp 2..block'). The ladder then calls (m + 2).clamp(3, vt_cap). Ord::clamp asserts min <= max in release builds too, so vt_cap = 2 panics on the first non-terminal round of the head-less ladder policy (MEMRA_DFLASH_ADAPT defaults on). A drafter with nd = 1 hits the same panic without any env. The same expression sits at 3818 and 4603.
- fix: Use (m + 2).clamp(vt_cap.min(3), vt_cap) at all three sites, or raise the parse floor to 3 and update FLAGS.md.

### medium: Capture failure after host KDA ping-pong swaps leaves odd parity, yet the walk declines to an eager walk it calls byte-identical

- where: `crates/memra-engine/src/glm5_tp_sym_graph.rs:547` (S09, confirmed)
- detail: Each KDA run is captured twice so the host-side recurrent-buffer swaps (kda_cached swaps on the HOST, header lines 29-32) cancel out. If capture_pair fails after phase 0 completed (phase 1 body error, or end_capture/instantiate refusal), or partway through phase 0, the layers already passed have swapped an odd number of times. walk_token then sets failed, returns Ok(false), and the caller runs the eager symmetric walk. For those layers the eager walk reads the state buffer the capture never wrote (stale by one token). Scenario: MEMRA_GLM5_TP_SYM_GRAPH=1, phase-1 instantiate refuses because of an allocation inside the capture. The message says the walk stays byte-identical, but the KDA layers of that run decode from the wrong recurrent state from that token on. Behind a default-OFF door. PLAUSIBLE: the swap mechanics were read from the module doc, not traced in kda_cached.
- cut: glm5_next x tp x graph cells only
- check: The defect is real, though the cited example is the one case that is safe. The capture body runs sym_layer_step, which swaps ssm_state/ssm_state_alt on the host (glm5_tp.rs:1265/1390) while the captured kernels do not execute. On failure, walk_graphed sets failed and returns Ok(false); there is no snapshot or parity restore, unlike glm5_decode_graph, which snapshots and restores the KDA state. The eager loop in hybrid_forward then runs on the swapped pointers. Correction to the scenario: a phase-1 instantiate failure after a full body leaves two swaps per layer, which is even and safe. The odd-parity cases are a phase-0 end_capture/instantiate failure (the most likely first failure) and a body error partway through either phase. In those cases the eager walk reads the previous token's recurrent state for the rest of the session, while the log says byte-identical. This is behind the default-OFF door.
- fix: Before the first capture, snapshot each KDA layer's ssm_state pointer per rank and swap back on any capture failure (as glm5_run_restore does), or return Err instead of Ok(false).

### medium: Per-model FA kernel defaults are process globals latched on first read

- where: `crates/memra-engine/src/hybrid.rs:5592` (S15, confirmed)
- detail: HybridModel load writes process-wide atomics that pick which flash_attn.cu kernel family runs: FA_V4_MAX_DEFAULT=0 for gqa>8, and for gemma FA_VEC_MIN_DEFAULT=1, FA_SPW_DEFAULT and FA_SP512_DEFAULT (hybrid.rs:5592-5615). fa_v4_at and fa_vec_min_tkv then latch those values in a OnceLock on first call (lib.rs:2222, lib.rs:1214). MEMRA_MODELS serves several models in one process. Scenario A: MEMRA_MODELS=g=gemma-4-31b,q=qwen3.6-27b. Loading gemma stores FA_VEC_MIN_DEFAULT=1, so qwen decode at t_kv<96 takes the vec kernel instead of the scalar fa_decode_f32 it runs alone: a different numeric program that per-model qualification never sees. Scenario B: a gqa<=8 model runs any FA dispatch (warmup, graph capture) before a gqa16 model (122B-A10B class) loads. M is then latched at usize::MAX, the gqa16 model takes the v4 kernels whose q_ints[8] array it overflows, and decode logits are all NaN (the failure hybrid.rs:5578-5590 documents). The guard lives only in this store, not in the launcher or the kernel; an explicit MEMRA_FA_V4_MAX also bypasses it.
- cut: Removes the cross-family edges: gemma and gqa>8 model loads would no longer reach the decode cells of qwen35/qwen3/llama models co-served in the same process.
- check: The load stores process-wide atomics: FA_V4_MAX_DEFAULT=0 for gqa>8, and FA_VEC_MIN_DEFAULT, FA_SPW and FA_SP512 for gemma. fa_vec_min_tkv and fa_v4_at latch them in OnceLocks on first read. memra-server routinely hosts several models in one process (the hybrid lane 'routinely has several models at once'). The pick therefore depends on load order and leaks across models: a qwen next to gemma gets vec_min=1. Scenario B (the NaN case) needs an FA dispatch before the gqa16 load. I could not prove one happens between loads, but nothing at the launcher prevents it, and an explicit MEMRA_FA_V4_MAX bypasses the guard.
- fix: Carry FA tuning in a per-model struct passed to the fa_decode_* dispatch and assert gqa<=8 in the v4 launchers.

### medium: Per-model numeric defaults are process-global, so co-loaded models change each other's numeric program

- where: `crates/memra-engine/src/hybrid.rs:5627` (S05, confirmed)
- detail: load_from_source_impl writes RMS_BLOCK_DEFAULT, FA_SP_GEMMA, FA_SPW_DEFAULT, FA_SP512_DEFAULT, FUSED_MR1_DEFAULT (gemma, 5599-5633), RMS_BLOCK_DEFAULT again for glm5 on 100a (5639), FA_V4_MAX_DEFAULT for any gqa>8 model (5592), MMQ_SK_FORCE under MEMRA_DRAFT (4307), and arms step37 doors (4273). All are process statics read by every model's kernel dispatch, and rms_block() latches the value in a OnceLock at first call (lib.rs:1719-1726). Failure: MEMRA_MODELS='chat=Qwen3.6-27B.gguf,g=gemma-4-31B.gguf'. If gemma loads before any rms_norm runs, the qwen model runs every norm at block 1024 instead of its qualified 256; the code says the block size changes the FP reduction order. If qwen runs a norm first, the gemma model gets 256 instead of 1024. A gqa-16 model loaded beside a 27B turns off FA v4 decode for both. Either way, one model's bytes depend on the roster and load order, which no single-model gate measures.
- cut: Keying the defaults per model would remove the cross-family cells: a gemma4/glm5/gqa>8 load would no longer reach qwen/llama/step numeric cells in a multi-model process.
- check: load_from_source_impl stores per-model tuning into process statics (RMS_BLOCK_DEFAULT, FA_SPW/SP512, FUSED_MR1, FA_SP_GEMMA, FA_V4_MAX_DEFAULT). rms_block() latches the first value it reads in a OnceLock. The server runs every MEMRA_MODELS entry in one process on worker threads, and no check refuses rosters whose defaults conflict. One model's kernel config, and so its FP reduction order, then depends on which models share the roster and on load order.
- fix: Carry these defaults on HybridModel and pass them into dispatch, or refuse at startup any roster whose load-time defaults conflict.

### medium: MEMRA_STEP35_PRIME_BATCH=0 makes every step35 single-session prime fail

- where: `crates/memra-engine/src/hybrid_forward.rs:5711` (S03, confirmed)
- detail: The B=1 GEMM prime entry (5691-5695) is keyed on step_gemm_prime_on() (armed by step37 defaults; the server refuses =0 for SlidingGatedMoe) but not on step35_prime_batch_on(). It calls step35_prime_cache_batch, which returns Err("step35 batched prime is disabled") when the flag is 0 (8081-8082). The error goes up through `?`, so with MEMRA_STEP35_PRIME_BATCH=0 every step35 prompt of 16+ tokens fails to prime. FLAGS.md describes the seam as restoring serialized single primes.
- check: Loading a SlidingGatedMoe plan arms the step37 defaults (hybrid.rs:4273), so step_gemm_prime_on() is true for step35/37. The B=1 entry checks step_gemm_prime_on() and uses_sliding_gated_moe_program() but not step35_prime_batch_on(). It calls step35_prime_cache_batch, which returns Err when MEMRA_STEP35_PRIME_BATCH=0, and the `?` propagates it. The FLAGS row says =0 'restores ... serialized single primes', and those single primes are exactly what fails.
- fix: Move the MEMRA_STEP35_PRIME_BATCH refusal under `prompts.len() > 1` (or add `&& Self::step35_prime_batch_on()` to the B=1 entry predicate).

### medium: MEMRA_MOE_EXPERT_RP=1: slab-fused pair and per-expert slab read pass the interleaved qtype to repacked V2 slabs

- where: `crates/memra-engine/src/hybrid_forward.rs:14625` (S03, confirmed)
- detail: With MEMRA_MOE_EXPERT_RP=1, DevExps slabs of plain NVFP4 layers are repacked to slot-major QT_NVFP4_V2 and d.rp is set (hybrid.rs:2101-2140). The fused-epi and vrows arms pass slab_rp, and pairs/dev call moe_rp_refuse. The per-expert slab read here (qmatvec_expert_q8 / qmatvec_view with gl.qtype at 14649) and the slab-fused gdec pair (moe_gate_up_silu8_q8 with m.gate_exps.qtype at 14397) neither convert through rp_qt nor refuse. On glm5_next NVFP4 with the door set and MEMRA_GLM5_DECODE_GRAPH=0, or any eager gap layer, T=1 decode skips vrows (capture closed) and fused-epi (off by default), so every routed expert is read with the interleaved walk over slot-major bytes: wrong MoE output with no error. A softmax NVFP4 MoE (qwen35_moe GGUF) under MEMRA_MOE_TRACE/STATS (observe_routes skips the dev arm) takes the slab_fused pair the same way. FLAGS.md claims unwired readers refuse by name; these two do not.
- check: With RP on, DevExps slabs are repacked to V2 and d.rp is set. The per-expert slab read passes gl.qtype from m.gate_exps.expert_layout(ex), which is the host tensor's QT_NVFP4, and calls neither rp_qt nor moe_rp_refuse. In the moe_ffn window only the vrows and fused-epi arms use slab_rp. glm5_next is sigmoid-routed, so dev_ok is false. At t=1 with capture closed, vrows_t1_dev is false; the deny trace's own text calls this the 'host-readback path ... eager gap layer'. That leaves the per-expert slab arm, which reads slot-major bytes as interleaved. The slab_fused gdec pair has the same gap, but its predicate needs no_exp_macros, so for macro-carrying glm5 NVFP4 the per-expert arm is the realistic hit. The B200 receipt was bit-exact only on the decode-graph posture.
- fix: Pass crate::rp_qt(slab_rp, qtype) for gate/up/down in the per-expert slab read and the slab-fused pair, or call crate::moe_rp_refuse(slab_rp, ...) before them.

### medium: Gemma SWA prefill changes numeric program at t > sliding_window, so prefix KV depends on total prompt length

- where: `crates/memra-engine/src/hybrid_forward.rs:22407` (S04, confirmed)
- detail: Below the window SWA layers attend unwindowed on f32 operands (fa_prefill); above it windowed on bf16 (fa_prefill_w_pre). Per the FLAGS row, rows [0..932) of a 932-row prime and a 1048-row prime differ in 59 of 60 layers. Scenario: request A (1048 tokens) primes and publishes its prefix to the device prefix cache; request B shares A's first 932 tokens with a shorter total. B restores KV that A wrote under the windowed bf16 program, while B primed solo would have used the unwindowed f32 program, so B's tokens depend on cache history. That breaks the one-numeric-program-per-request rule on the prime/prefix-restore pair. The fix door MEMRA_GEMMA_FA_ONE_PROGRAM stays default OFF pending an owner call.
- cut: none; this is a correctness defect on gemma4_dense/gemma4_moe prime+prefix_cache cells
- check: The switch at t > win is live code, and FLAGS.md itself calls it a live prefix-restore exactness violation: below the window, SWA layers run unwindowed on f32; above it, windowed on bf16. 59 of 60 layers differ for rows [0..932). The fix door is default OFF (gemma_fa_one_program reads Ok("1") only).
- fix: Make the windowed arm the only SWA prefill program (flip MEMRA_GEMMA_FA_ONE_PROGRAM, then delete the switch), or key prefix entries on the program class.

### medium: Per-model numeric defaults are process-global statics, and rms_block/fa_vec_min_tkv/fa_v4_at latch them in OnceLocks

- where: `crates/memra-engine/src/lib.rs:1725` (S01, confirmed)
- detail: hybrid.rs stores RMS_BLOCK_DEFAULT=1024, FA_VEC_MIN_DEFAULT=1, FA_SP_GEMMA, FUSED_MR1_DEFAULT, FA_SPW/SP512 when it loads a gemma plan, FA_V4_MAX_DEFAULT=0 for gqa>8 plans, and RMS 1024 for glm5_next on 100a. Nothing resets them for the next model. rms_block() (L1719), fa_vec_min_tkv() (L1212) and fa_v4_at() (L2220) cache the first value they see for the process. The server worker loads every MEMRA_MODELS entry into one process (worker.rs:26127). Concrete case: MEMRA_MODELS="g=gemma4-31b.gguf,q=qwen3.6-27b.gguf". The gemma load stores 1024 and the qwen load stores nothing, so every qwen rms_norm launches at blockDim 1024 instead of its battery-arbitrated 256. qwen FA decode then takes the vec path from t_kv>=1 instead of 96. Qwen greedy bytes differ from the same model served alone. In the reverse order, if qwen's first rms_block() call comes before gemma loads, gemma gets 256. One model always runs an unqualified numeric program. This also breaks family-local impact reasoning: a gemma loader change reaches qwen cells.
- cut: Removes cross-family edges from gemma4/glm5_next/large-GQA loaders into every co-loaded family's rms_norm and FA-decode cells.
- check: The gemma load stores RMS_BLOCK_DEFAULT=1024, FA_VEC_MIN_DEFAULT=1, FA_SPW/SP512, FUSED_MR1 and FA_SP_GEMMA into process-global atomics. The gqa>8 load stores FA_V4_MAX_DEFAULT=0, and glm5_next on 100a stores RMS 1024. Nothing resets them. rms_block(), fa_vec_min_tkv() and fa_v4_at() latch the first value they read in OnceLocks. The worker loads every MEMRA_MODELS entry in one loop in one process, so a qwen model co-loaded with gemma runs at rms block 1024 and vec-FA from t_kv>=1, not 256 and 96. Its bytes then differ from qwen served alone. That makes it a different numeric program, not garbage, so medium rather than high. Worse case, unverified: if fa_v4_at latches before a gqa16 model loads, the v4 NaN guard is lost.
- fix: Carry these per-model numeric defaults on the HybridModel/plan and pass them to the launchers, or refuse multi-model rosters whose loaders store conflicting values.

### medium: Arch guard admits cc 12.1 (GB10) for fatbins that hold only sm_120a SASS

- where: `crates/memra-engine/src/lib.rs:3755` (S01, confirmed)
- detail: build.rs maps a 12.1 GPU to 120a (build.rs:174) and emits code=sm_{arch} SASS only, no PTX (build.rs:276). Arch-specific 'a' SASS is bound to its exact compute capability, so on a GB10 the guard passes and the first load_module fails with an opaque CUDA error, the failure the guard exists to name. PLAUSIBLE: not run on GB10 hardware. The device table already flags a GB10 load issue.
- check: build.rs maps cc 12.1 to 120a and emits code=sm_120a SASS only, with no PTX. Arch-specific 'a' SASS loads only on its exact compute capability. The guard explicitly admits ("120a", 12, 1), so on a GB10 it passes and the first module load fails with an opaque CUDA error, the exact case the guard exists to name. The repo's own hardware assessment already says 'our sm_120a SASS does not load on CC 12.1'. Not run on hardware here.
- fix: Drop ("120a", 12, 1) from the guard and stop mapping 12.1 to 120a in build.rs, until a 121a (or 120f family) target is built.

### medium: MEMRA_MLA_DECODE_SPLIT documented DEFAULT ON since 2026-09-03 but code is default OFF

- where: `crates/memra-engine/src/mla_ffi.rs:57` (S13, confirmed)
- detail: docs/FLAGS.md:953 records a receipt-backed flip to default ON (+4.35% glm5_next decode on 2x B200, greedy tape identical), and tools/check-no-remove-var-gates.sh cites that flip. The code still arms the door only on a literal 1, and the doc comment at mla_ffi.rs:21 still says default OFF. git log -S shows the line last touched by the 'catch-up sync (purge re-applied)' commit, so the flip was likely lost. A naked glm5_next/glm_dsa launch runs the unsplit absorb/decompress kernels on every device, and any qualification that assumed the split is default measures a different launch geometry.
- cut: glm5_next, glm_dsa decode speed on every device
- check: git log -S pins it down. #137 (0c7a303ab9) flipped the parse to `!= Ok("0")` with the FLAGS receipt (+4.35% on 2x B200, identical tape). #140 (e44f842759, MLA_COALESCE, the same day) reverted both the parse and the doc to `== Ok("1")` / 'default OFF', and its commit message gives no reason for the revert. FLAGS.md:953 and tools/check-no-remove-var-gates.sh still describe the door as default ON. No other code sets the var. Output is unchanged (split-invariant bytes), but naked glm5 decode loses the measured win and runs a different launch geometry from the one qualified.
- fix: Restore `!= Ok("0")`, the default-ON doc and the announce text from #137 (and the 'default OFF' note in cu/mla_attn.cu:517), and add a unit test on a pure parse helper.

### medium: Quantize-once MMQ activation window never closes, so the cache claim is false

- where: `crates/memra-engine/src/mmq_ffi.rs:1836` (S13, confirmed)
- detail: mmq_act_begin bumps the epoch, but nothing closes the window. MMQ_ACT_SLOT keeps (epoch, ptr, m, in_f) until the next begin. Any later Q4_0 MMQ matmul that did not open a window still matches the stale epoch. Example: after gemma4_layer_tail_core_pn opens the gate/up window, a pool-recycled activation at the same address with the same (m, in_f) reuses the old quantized scratch and computes on stale activations. Today it is safe only because the following projections differ in in_f; the doc's guarantee does not hold by construction.
- cut: gemma4 Q4_0 prefill
- check: It is worse than the doc admits. Every qmatvec_mmq_q4_0_raw call writes the slot with the current epoch, and only mmq_act_begin bumps the epoch. So outside a window, any consecutive pair of Q4_0 MMQ calls with the same (ptr, m, in_f) hits the cache, and 'Paths that never call this never hit the cache' is false. Only the two gemma4 sites call begin, but mmq_supports admits every Q4_0 tensor (MEMRA_PP_Q4MMQ is default ON) whatever the family. When a freed activation's address is recycled for a same-shape input on the next call, the GEMM reuses the old quantized activations and produces wrong output. Today this is avoided only by in_f differences and the order of allocations and frees, not by construction.
- fix: Make mmq_act_begin return a guard that clears the slot on drop, and only consult or populate the slot while a guard is live.

### medium: Block-128 FP8 checkpoints appear to fail to load on 89/90a builds. Native residency (MEMRA_ST_E4M3 default ON) calls fp8_blk_nan_count, which goes to mmq_fp8_blk_stub.cu on non-120a/100a builds; the stub returns 2905 and mmq_ffi.rs:2511 turns that into Err, which `?` propagates. On 90a, fp8_blk_mmq_native_policy is also default ON and try_fp8_blk_mmq is gated only by portable_mma_gated, so prefill reaches the stub too.

- where: `crates/memra-engine/src/model.rs:1022` (scout:devices, confirmed)
- check: On every build other than 120a/100a, build.rs swaps in mmq_fp8_blk_stub.cu, whose memra_fp8_blk_count_nan always returns 2905. fp8_blk_nan_count turns a non-zero rc into Err. st_e4m3_blk_enabled is just MEMRA_ST_E4M3 != "0" (default ON) with no arch check, and model.rs:1022 propagates with `?`. A block-128 FP8 safetensors load on 89/90a therefore errors instead of falling back to the Q8_0 floor. model.rs:866 and hybrid.rs:2147 call fp8_blk_nan_count the same way. fp8_blk_mmq_native_policy is default ON for non-100a builds, so the prefill arm would reach the stub too.
- fix: Gate st_e4m3_blk_enabled and fp8_blk_mmq_native_enabled on matches!(BUILT_CUDA_ARCH, "120a"|"100a") through one capability helper, and add an sm_89/sm_90a block-FP8 load cell.

### medium: Greedy serving requests on the MTP route read MEMRA_SPEC_TEMP and the other CLI sampling envs

- where: `crates/memra-engine/src/spec.rs:24` (S10, confirmed)
- detail: The worker passes sampling=None for every temperature-0 request (spec_sampling_for at worker.rs:1056). generate_spec_inner2 (12114) and mtp_prime_start then call resolve_spec_sampling(None), which reads MEMRA_SPEC_TEMP, MEMRA_SEED, MEMRA_TOP_K/P, MEMRA_MIN_P and MEMRA_PENALTY_* from the process env. Scenario: a server boots with MEMRA_SPEC_TEMP=0.7 left over from a run-spec sampled cell. Every greedy (temperature 0) request admitted to MTP spec on qwen35, qwen35_moe, step35 or hy3 is then rejection-sampled at T=0.7 with seed 42. The same request served plain (concurrency above the spec gate HIGH, or a vision or capture request) stays greedy, so one request gets two numeric programs. That breaks the API temperature contract and the one-numeric-program rule. The server never reads or refuses MEMRA_SPEC_TEMP.
- check: spec_sampling_for returns None whenever temperature is 0. generate_spec_inner2 (12114) and the prime paths (prime.rs 370/814) call resolve_spec_sampling(None), which reads MEMRA_SPEC_TEMP, MEMRA_SEED, MEMRA_TOP_K/P, MEMRA_MIN_P and MEMRA_PENALTY_* from the process env. memra-server never reads or refuses MEMRA_SPEC_TEMP. The boot env audit treats it as legal because it has a FLAGS.md row. A server started with MEMRA_SPEC_TEMP>0 left in its env therefore rejection-samples greedy requests on the MTP route, while the same request served plain stays greedy.
- fix: Have the server pass an explicit greedy SpecSampling (temp 0, identity penalties) instead of None, keep the env fallback for bins only, and refuse MEMRA_SPEC_TEMP at server boot.

### medium: Shared-expert gate dropped for qwen3next-named MoE GGUFs

- where: `crates/memra-gguf/src/model_plan.rs:2470` (S20, confirmed)
- detail: Arch::parse maps the upstream llama.cpp arch string "qwen3next" to Arch::Qwen35 even when the file carries experts (config.rs:55, and from_gguf warns about exactly this at config.rs:1374). qwen35_moe's matcher accepts Qwen35+experts, but compile_mlp keys SharedMlpPlan.gated on Arch::Qwen35Moe|Qwen4Exp only, so the plan marks the shared expert UNGATED. tensor_contract::add_moe_mlp (tensor_contract.rs:2352) then declares no ffn_gate_inp_shexp, and TensorContract::bind refuses extra tensors (TensorContractError::Extra). Failure: loading a llama.cpp Qwen3-Next-80B-A3B GGUF (arch qwen3next, sigmoid-gated shared expert) fails at bind_source with Extra{blk.N.ffn_gate_inp_shexp.weight}. If the binding is relaxed, memra-reference (lib.rs:1940/7679) runs the shared expert without its gate while HybridModel detects the tensor, so the reference and tuned programs diverge.
- cut: qwen35_moe GGUF cells only
- check: The failure chain holds. Arch::parse maps qwen3next to Qwen35, and from_gguf keeps the experts (there is a test for this). The qwen35 dense pack refuses expert configs, the qwen35_moe pack accepts Qwen35+experts, and expert_shared_ff_length is read from GGUF metadata, so a shared expert is planned. compile_mlp sets gated only for Qwen35Moe|Qwen4Exp, so it is false. tensor_contract then declares ffn_gate_inp_shexp only when shared.gated, bind refuses undeclared tensors with Extra, and hybrid.rs binds before upload. Impact is a refused load for llama.cpp qwen3next MoE GGUFs, not silent wrong output.
- fix: Key gated on the Qwen-hybrid MoE class (Qwen35|Qwen35Moe|Qwen4Exp with moe.expert_count > 0), or normalize qwen3next+experts to Arch::Qwen35Moe in from_gguf.

### medium: Step35 SWA-ring default leaks to every family in the process

- where: `crates/memra-kv/src/lib.rs:133` (S21, confirmed)
- detail: arm_step37_serving_defaults (engine lib.rs:1584, called at step35 load from hybrid.rs:4273) stores a process-global AtomicBool. Cache::new_inner then gives every SlidingKvCache layer a KvRing (lib.rs:3133-3141) whatever the family. The doc at lib.rs:125 says the default arms per loaded family because Gemma4's row-0-addressed window kernels cannot read a rebased ring view, and the gemma4 forward in hybrid_forward.rs has no ring guard. The server's ring_prefix_excluded (worker.rs:34258) also only excludes SlidingGatedMoe plans. Scenario: MEMRA_MODELS loads a step37 GGUF and a gemma4 GGUF in one server with MEMRA_CTX above about 5.7k. Every gemma4 session after boot allocates ring-backed SWA planes of min(max_ctx, 1024+4096+512+31) rows. Once a prompt plus decode passes the ring size, the ring rebases, but the gemma window kernels still address absolute rows, so gemma output diverges or reads past the plane. Verdict PLAUSIBLE: the multi-model co-load was not run.
- cut: With a per-plan flag, KvRing/swa_* changes reach only step35 cells by default; gemma4_dense, gemma4_moe and mimo_v2 x {plain,batch,prime,prefix_cache} drop out of the ring's impact set except under MEMRA_SWA_RING=1.
- check: Each step in the chain is in the code. Step35 load stores a process-global SWA-ring default. Cache::new_inner always resolves a plan (it compiles one when None) and gives a KvRing to every StatePlan::SlidingKvCache layer when swa_ring_on() is true. Gemma4 SWA layers are planned as SlidingKvCache via attention_window. The gemma decode append writes at absolute kvl.len without prepare_kv_append, and append_kv_quantized has no bounds check. memra-server's worker::run loads all MEMRA_MODELS on one thread in one process. A step37 + gemma4 co-load with max_ctx above the ring size would therefore give gemma ring-sized planes it addresses absolutely: an out-of-bounds write or wrong output. The co-load was not run, so the exact symptom is unobserved.
- fix: Gate the ring per cache on the plan's DecodeBatchProgram::SlidingGatedMoe, not the process-global default, or refuse a co-load of a SlidingGatedMoe model with any other SlidingKvCache family.

### medium: Malformed MEMRA_MODELS silently falls back to the hard-coded qwen35 test pair

- where: `crates/memra-server/src/lib.rs:6647` (S19, confirmed)
- detail: If MEMRA_MODELS is set but every entry is malformed (for example a ':' typo instead of '='), each entry only prints a WARN, out stays empty, and the function returns the BASE-4 default paths. Scenario: MEMRA_MODELS="q38:/models/q38.gguf" on a box where /data/ai-ml/hf-models/... exist. The server boots and serves Qwen3.6-27B and Qwen3.5-9B under names main/judge instead of failing, and clients asking for q38 get model_not_found. Elsewhere the worker load fails late with a confusing path.
- check: parse_models_config prints only a WARN for each entry without '='. It returns out only when it is non-empty, and otherwise falls through to the hard-coded BASE-4 main/judge paths. So a set but fully malformed MEMRA_MODELS boots the default pair (or fails late at worker load) instead of refusing at parse. Malformed entries mixed with valid ones are also skipped with only a WARN.
- fix: When MEMRA_MODELS is set, exit FATAL on any malformed entry (or at least when no entry is valid); keep the default pair only for the unset case.

### medium: Budget-path render bypasses the vision special-id intake guard

- where: `crates/memra-server/src/lib.rs:9985` (S19, confirmed)
- detail: vision_special_id_guard (worker.rs:33487) is called only inside the worker's prepared_prompt-is-None block. On a metered deployment the HTTP budget path fills prepared_prompt first, so the guard never runs. Scenario: a limited tenant sends a text-only chat to a glm5_next or qwen-VL model with literal "<|image|>" / "<|image_pad|>" strings in the message text. They tokenize to vision special ids and are not refused, while the same request is refused (400) on the stock binary. The documented law (lib.rs:4884) is that faked pad tokens fail validation.
- check: vision_special_id_guard has one production caller, inside the worker's prepared_prompt.is_none() block. The metered budget path fills prepared_prompt first through render_prompt_ids, which runs no guard. The guard's own doc says that without image units, literal special-token text reaches the trunk as a raw placeholder embedding and is 'served 200, fluent, invisible'. So text-only requests from limited tenants on metered deployments skip the refusal.
- fix: Run vision_special_id_guard on every prepared prompt, in the worker after the is_none block or in a shared prepare step.

### medium: Spec-gate PP-2 placement is frozen before MEMRA_PARALLEL=auto sets MEMRA_PP_STAGES

- where: `crates/memra-server/src/worker.rs:6653` (S17, confirmed)
- detail: spec_gate_thresholds() is a OnceLock whose PP-2 test (spec_gate_pp2_placement) reads MEMRA_PP_STAGES and MEMRA_PP_DEVICES from the env. run() calls log_spec_gate_policy() at worker.rs:26076, before HybridModel::load_from_source at 26201, and that call initializes the OnceLock whenever the gate is on and no K pin is set (the default). MEMRA_PARALLEL=auto writes MEMRA_PP_STAGES and MEMRA_PP_DEVICES only during load (hybrid.rs:856-857). Failure scenario: step35 or hy3 (MTP-capable and PIPELINE-admitted) launched with MEMRA_PARALLEL=auto on two cards, where the auto planner picks a 2-stage cross-device pipeline. The gate has already frozen at single-card LOW=2/HIGH=4 instead of LOW=0/HIGH=1, so serve_spec_enabled() stays true and choose_spec_k admits MTP spec at c<=2. That is the placement the doc measured as a loss in every PP-2 cell (36 vs 86-122 tok/s). Admission also pays SPEC_SHRINK_RESERVE, and the boot log prints placement=single-or-non-pp2. A hand-set MEMRA_PP_STAGES=2 is unaffected.
- cut: none
- check: spec_gate_thresholds is a OnceLock that reads MEMRA_PP_STAGES through spec_gate_pp2_placement and pp_sharded_cross_device. run() calls log_spec_gate_policy at 26076, before HybridModel::load_from_source at 26201, and with no K pin and the gate on (the defaults) it initializes the OnceLock. MEMRA_PARALLEL=auto sets MEMRA_PP_STAGES only inside load (prepare_auto_parallel refuses a hand-set value), so the thresholds stay frozen at the single-card 2/4. serve_spec_enabled then stays true, and choose_spec_k never returns Placement on an auto PP-2 placement. MEMRA_PARALLEL is opt-in, which limits the scope. The impact is lost performance, not wrong output.
- fix: Move log_spec_gate_policy() after the model load loop (and make sure nothing reads spec_gate_thresholds before then), or derive pp2 from the loaded PpNRt placement.

### medium: MEMRA_SERVE_BATCH=0 with a drafter route armed sends gemma/dspark/glm5 sessions to plain step_session with no cache

- where: `crates/memra-server/src/worker.rs:30192` (S18, confirmed)
- detail: admit sets gspec_k (36880), dspark_on (37083) and glm5_on (37209) without consulting serve_batching(), and builds those sessions with cache None (37288-37293). The non-batching loop (30161-30219) runs step_session_async_chain and then step_session for every session; no route dispatch exists there. step_session's prefill arm then calls prime_cache(.., s.cache.as_mut().unwrap()) (40135) on None and panics. request_fault_guard turns that into a worker_fault for the request. Scenario: MEMRA_SERVE_BATCH=0 MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT=<dir>, Qwen3.8-27B, a greedy cold request: every DFlash-routed request fails. MEMRA_DRAFT=<assistant> on gemma fails the same way. No boot verdict refuses the combination (dspark_spec_boot_conflict does not read MEMRA_SERVE_BATCH).
- check: step_dspark_spec, step_glm5_spec and step_gemma_spec are called only from the batching branch (30728-30732). The !batching loop calls step_session_async_chain, which returns None for gspec/dspark, and then step_session. A dspark or glm5 session has s.cache None at admission and its spec session is created only at the first spec tick, so step_session's s.dspark/s.glm5 refusals do not fire. The prefill arm then calls prime_cache(.., s.cache.as_mut().unwrap()) and panics. guard_request catches the panic, so the result is a per-request fault on every routed request, not a crash. Neither the dspark/gemma drafter attach nor dspark_spec_boot_conflict reads MEMRA_SERVE_BATCH. Minor correction: the gemma case also needs MEMRA_GEMMA4_SPEC=K, not only MEMRA_DRAFT.
- fix: AND serve_batching() into the gspec_k, dspark_on and glm5_on admissions (or refuse MEMRA_SERVE_BATCH=0 with an armed drafter route at boot).

### medium: Sub-floor capture boundary moves prompt tokens to tokenwise decode_step (W1, still open)

- where: `crates/memra-server/src/worker.rs:38581` (S18, confirmed)
- detail: When snapshot_at, ckpt_at or seed_at sits less than PRIME_MIN_T past fed, bound_rem vetoes the prime branch (38604). The remaining prompt tokens then walk decode_step one at a time (38740-38804), a second numeric program for the same prompt bytes. The comment names the remedy (drop the capture) and leaves it undone. It reaches every non-eager family on the prime and prefix_cache routes whenever a restored prefix ends within 16 tokens of the next armed boundary.
- check: prefill_tick_take never creates a sub-floor gap itself, but boundaries can be armed sub-floor at admission. On the default ckpt_at path, plain_checkpoint_boundary is filtered only by b > seed_fed.len(), with no PRIME_MIN_T floor. It is turn-marker based, while restored prefixes are grid-aligned, so a restored prefix ending 1-15 tokens before the checkpoint is reachable. The same holds for a ckpt_at within 16 tokens of a grid-aligned snapshot_at. bound_rem < PRIME_MIN_T then vetoes the prime branch, and those prompt tokens go through decode_step. The code labels this an OPEN DEFECT and names the remedy without applying it.
- fix: Before the branch, clear any boundary with 0 < b - fed < PRIME_MIN_T (and floor ckpt_at against seed_fed at admit) so prompt tokens always take the prime program.

### medium: The devpenalty default is keyed on the build arch, but its only receipt is from one RTX 5090 (worker.rs:39113). It therefore defaults ON for qwen35 on RTX PRO 6000 and GB10 with no receipt, which breaks the per-hardware arm rule (one-rig evidence sets a one-rig default at most).

- where: `crates/memra-server/src/worker.rs:39128` (scout:devices, confirmed)
- check: The default is ON for any 120a build with Arch::Qwen35, which covers the RTX PRO 6000 and (through the 12.1 mapping) GB10. The only qualification cited is one RTX 5090. AGENTS.md says: 'One-rig evidence sets a one-rig default at most... key the arm on the device'. The doc's own reasoning ('the win is a property of ... the board's host') argues against inheriting the 5090 receipt. Greedy output is byte-identical, so the impact is an unqualified default, not wrong output.
- fix: Key the default on the RTX 5090 device class (not the build arch), or add a PRO 6000 qwen35 interleaved receipt.

### low: hc dot split returns raw cudaError codes, breaking the TU error band

- where: `crates/memra-engine/cu/dsv4_dense_m1_exact_tail.cuh:654` (S16, confirmed)
- detail: The house pattern is 0 ok / 10000+cudaError / 400xx refusals (dsv4_gpu.cu:25-26). dsv4_hc_dot_split_launch and memra_dsv4_hc_dot_split_partial return raw cudaError_t. A launch failure such as cudaErrorInvalidConfiguration (9) reaches Rust as rc=9, which cannot be told apart from a small refusal code. Diagnosis then names the wrong band.
- check: dsv4_hc_dot_split_launch and memra_dsv4_hc_dot_split_partial return raw cudaError_t, against the TU's stated 0 / 10000+cudaError band. dsv4_ffi::ck prints rc= with no band decoding, so a launch failure shows as rc=9 and not 10009. One part is overstated: it cannot be mistaken for a refusal, since refusals are 400xx. The impact is diagnostic only.
- fix: Return err ? 10000 + (int)err : 0 at all three return sites (or use DSV4_ERR()).

### low: MEMRA_PRINT_TEXT ignores tok_dir and fails on Hy3 repack directories

- where: `crates/memra-engine/src/bin/run_spec.rs:422` (S22, confirmed)
- detail: main resolves tok_dir from the repack manifest's source_dir (line 146) because repack dirs carry no tokenizer.json, and the prompt paths use it. The two MEMRA_PRINT_TEXT decoders (lines 422 and 533) pass `path` instead. Scenario: MEMRA_PRINT_TEXT=1 run-spec <hy3 repack dir> finishes the plain oracle, then errors 'HF tokenizer init failed' before the K sweep, so the gate run aborts.
- check: main resolves tok_dir from the Hy3 repack manifest's source_dir because the repack dir may have no tokenizer.json, and the prompt-dir path uses tok_dir. Both MEMRA_PRINT_TEXT branches pass the raw path to from_hf_dir, which reads path/tokenizer.json and returns an error that main propagates with ?.
- fix: Use &tok_dir in both MEMRA_PRINT_TEXT branches.

### low: Sampled-mode failure report diffs the greedy gold against the sampled stream and can panic

- where: `crates/memra-engine/src/bin/run_spec.rs:518` (S22, confirmed)
- detail: With MEMRA_SPEC_TEMP>0, pass means spec == spec2 (seeded rerun). On a failure the report still compares gold (greedy) with spec. That diff is meaningless, and if the sampled stream happens to equal the greedy gold, first_divergence returns None and unwrap() panics, hiding the actual spec-vs-spec2 divergence.
- check: gold comes from plain_oracle, which is documented as the greedy oracle. In sampled mode pass = first_divergence(&spec, &spec2).is_none(), but the !pass branch unwraps first_divergence(&gold, &spec). If the sampled spec stream equals the greedy gold (equal length and ids), that returns None and unwrap panics. Otherwise the printed diff is greedy-vs-sampled, not the spec-vs-spec2 divergence that failed. This affects only reporting on an already-failing run.
- fix: In sampled mode, keep spec2 and report first_divergence(&spec, &spec2).

### low: generate_with's qwen dc door admits MLA and hyper plans that decode_step_dc cannot run

- where: `crates/memra-engine/src/decode.rs:497` (S05, confirmed)
- detail: The qwen dc gate (decode.rs:3290-3295) excludes only SlidingGatedMoe. For a greedy GGUF glm_dsa run-gen, decode_step_dc reaches attn_in_norm_mixer_dc. mixer_in_q8_1_fast is false for MLA, so the unfused arm panics at line 497. For a glm5_next GGUF, decode_step_dc returns the refuse_hyper error (line 1829). decode_step_h serves both. Bin-only today; the safetensors run_gen branch uses its own loop.
- check: The qwen dc door in generate_with excludes only uses_sliding_gated_moe_program. For an MLA plan, decode_step_dc reaches attn_in_norm_mixer_dc, and both the fused and unfused arms hit mla_path_unimplemented. For a hyper plan, refuse_hyper returns Err. decode_step_h supports both. Only bins reach this path.
- fix: Add hyper.is_none() and an only-Full/Linear-mixers check to the qwen dc door so those plans take the host-logits loop.

### low: Captured dc step lacks decode_step_dc's gemma delegate and PP refusal

- where: `crates/memra-engine/src/decode.rs:1976` (S05, confirmed)
- detail: decode_step_dc routes the Gemma program to gemma4_decode_step_dc and refuses remote unsplit PP (1845-1861). decode_step_dc_cap_masked, which graph_capture_segment records, has neither: it always walks the qwen-class stack. graph_session_new on a gemma checkpoint primes through the gemma dc twin (via decode_step_dc), then captures the generic stack over gemma weights. The decode_step_dc comment says this produced the argmax-INIT passthrough the g12 gate caught. Reached only by graph session bins and gates.
- check: decode_step_dc_cap_masked has refuse_hyper and ensure_usable but no uses_gemma_program delegate and no refuse_unsplit_if_remote. graph_session_new primes through decode_step_dc (which delegates gemma) and then captures the generic stack through graph_capture_segment_masked. Its callers are graph bins and gates. The server keeps gemma out of graph promotion through eager_only_model.
- fix: Factor the dc guards into one helper and call it from decode_step_dc_cap_masked, refusing the gemma program and remote unsplit PP.

### low: MEMRA_DSV4_DRAFTER accepts any value silently

- where: `crates/memra-engine/src/dsv4_gpu.rs:4272` (S06, confirmed)
- detail: MEMRA_DSV4_DRAFTER is read three times (3675, 3704, 4272) with an exact == "dspark" compare. MEMRA_DSV4_DRAFTER=DSpark, =1 or ='dspark ' loads trunk-only and logs 'GPU drafter path off'. The dflash route then serves plain with no error, a throughput loss nothing reports as a fault. Every other dsv4 door in load_placed refuses unknown values by name.
- check: All three engine reads compare exactly against "dspark" and nothing validates the value. Any other value, including typos and case variants, loads trunk-only and prints only an informational line. The server never parses the variable (worker.rs only names it in a message). Gates rely on 'off' as an accepted value, so the parser must accept off and unset.
- fix: Parse MEMRA_DSV4_DRAFTER once into {unset|off|dspark}, refuse other values by name at load, and pass the result to the three sites.

### low: f16 mirror budget is one process-global counter across devices and models

- where: `crates/memra-engine/src/f16_ffi.rs:807` (S13, confirmed)
- detail: SPENT and BUDGET are function statics, so a PP-2 load or a multi-model server charges every card and model against one 32 GiB budget in load order. Scenario: two models loaded in one server, the first spends the budget. The second model's Q8_0/Q4_K tensors silently skip the f16 mirror and prefill on MMQ, a different numeric class than the same model gets when loaded alone. Which tensors carry mirrors then depends on boot order, not on the model.
- cut: multi-model or pp boots of Q8_0/Q4_0/Q4_K/Q5_K/Q6_K models
- check: SPENT and BUDGET are function-local statics. Every engine and device in the process charges against one 32 GiB budget in load order. Unloading a model never refunds SPENT. The skip path returns silently. Under PP-2 or a model reload, which tensors get f16 mirrors (and so which prefill numeric class runs) depends on load history.
- fix: Keep the spent counter on the Engine (per device), refund it on drop, and log once when a mirror is skipped for budget.

### low: Gemma FR trim indexes ids[0] without checking the ranks file

- where: `crates/memra-engine/src/gemma_spec.rs:356` (S11, confirmed)
- detail: If MEMRA_GEMMA_DRAFT_RANKS names an empty file, or one whose ids are all >= n_vocab (they are filtered out at 313), ids is empty. The spare-slot loop indexes ids[0] (329) and spare_fill = d2t[0] (356) panics with index out of bounds at boot, with no message naming the flag.
- check: ids is filtered to < n_vocab with no empty check. If n_spare > 0, ids[0] in the spare loop panics. If n_spare = 0, d2t[0] panics. Either way the panic message does not name MEMRA_GEMMA_DRAFT_RANKS.
- fix: After collecting ids, return Err("MEMRA_GEMMA_DRAFT_RANKS=<path>: no valid token ids") when ids is empty.

### low: shard_cols silently drops a CUTLASS prefill operand that shard_rows refuses

- where: `crates/memra-engine/src/glm5_tp.rs:846` (S09, confirmed)
- detail: On a memra_cutlass build, shard_rows refuses a tensor carrying a cutlass operand (line 982), but shard_cols ignores the field and emits a shard with cutlass: None. A column-sharded tensor then quietly leaves the CUTLASS FP4 prefill program, while a row-sharded or unsharded one refuses or keeps it. That is a different prefill numeric class with no refusal. Today glm5_next ships FP8/BF16, so the operand is None.
- check: Under cfg(memra_cutlass), shard_rows refuses a tensor whose cutlass operand is set. shard_cols binds it as `cutlass: _` and emits `cutlass: None` with no refusal, so it is asymmetric and silently drops the operand. It is latent while glm5_next ships without CUTLASS operands.
- fix: Bind cutlass in shard_cols and refuse under cfg(memra_cutlass) when cutlass.is_some(), mirroring shard_rows.

### low: shard_cols uses the byte-divisibility test that tp_expert_split documents as unable to prove block alignment

- where: `crates/memra-engine/src/glm5_tp.rs:867` (S09, confirmed)
- detail: The comment above (line 865) says the check counts blocks, but the code tests (cols.start*row_bytes) % in_f and (take*row_bytes) % in_f. tp_expert_split.rs:196-198 shows this passes a cut through a block: Q8_0 with in_f 96 and a 48-column half gives 48*102 % 96 == 0, yet 48 columns is 1.5 blocks. For Q8_0 the test passes for every half of an in_f that is a multiple of 32, so a K-slice can split a block and the shard decodes wrong scales. Latent on GLM-5.3-Flash geometry (halves of 4096/2048 are block-aligned), but nothing in the preflight enforces it.
- check: shard_cols tests (cols.start*row_bytes) % inner and (take*row_bytes) % inner, which is byte divisibility. For Q8_0 (32-column block, 34 bytes) that only requires a multiple of 16 columns, so a half-block cut passes. tp_expert_split.rs documents this exact counterexample and uses qtype_block instead. The shard_cols comment claims to count blocks, but the code does not. Latent: the current callers slice la.wo at GLM-5.3 geometry, which is block-aligned.
- fix: Use tp_expert_split's qtype_block(qtype) and require cols.start and take to be multiples of the block (refuse an unknown qtype).

### low: Known panic and tear in the device-staged drafter prime (door ON only)

- where: `crates/memra-engine/src/glm_spec.rs:259` (S11, confirmed)
- detail: With MEMRA_GLM5_DRAFT_TAPS_DEVICE=1 and the MEMRA_B200_PRIME_V2 arm-2 prime, glm5_hc_tap_slot slices past the slot buffer from the second range on, and slice_mut panics the worker. The same-ordinal ingest also races the next range's writes, so drafter ctx KV can tear silently. The doc names both defects. They are unreachable at default but reachable with a documented door.
- check: The doc names both defects. The arm-2 branch (b200_prime_v2_on, prime_cache_hyper_pp2_pipelined) never calls glm5_taps_range_begin/done, and nothing refuses the combination with MEMRA_GLM5_DRAFT_TAPS_DEVICE=1. The panic and the tear are reachable only with two default-OFF doors both set.
- fix: Refuse (named Err) device-staged taps when the arm-2 pipelined prime would engage, until both defects are fixed and gated.

### low: Resident-expert budget reserves zero trunk bytes for safetensors sources

- where: `crates/memra-engine/src/hybrid.rs:264` (S05, confirmed)
- detail: ResidentPlan::from_layout computes trunk_bytes only from the GGUF header; any non-GGUF source gets 0. should_reside then admits experts when per_layer * layer_count <= free - 2 GB, measured at the first MoE layer, with nothing reserved for later layers' attention, shared experts, dense prefix, MTP blocks or the KV cache. Failure: a safetensors MoE checkpoint (for example a glm5_next FP8 or hy3 BF16 dir) whose experts fit in free-2GB but experts + remaining trunk do not. The plan picks RESIDENT, and the load later OOMs on a dense tensor upload or on cache allocation instead of falling back to the SLRU path.
- check: from_layout computes trunk_bytes only from src.gguf(). Non-GGUF sources get (None, 0), so should_reside's budget is free - 2 GB with no reserve for later layers' non-expert tensors or the KV cache. A safetensors MoE whose experts fit but whose experts plus remaining trunk do not would pick RESIDENT and could OOM later. Partly offset: projected counts every layer on the device, dense ones included.
- fix: For non-GGUF sources, sum non-expert tensor bytes from the source's tensor census (as residency_bytes_by_device does for GGUF) into trunk_bytes.

### low: MEMRA_B200_PRIME_V2 is the only B200-named door that is not build-gated. The others (B200_MATVEC_ARM, B200_GEMV_V2, B200_BF16_GEMV_LT, B200_MLA_DECODE_ARM) are no-ops off sm_100a, so this one can change the glm5 PP-2 prime schedule on a 120a build.

- where: `crates/memra-engine/src/hybrid_forward.rs:954` (scout:devices, confirmed)
- check: b200_prime_v2_on is a bare env read with no MEMRA_BUILT_CUDA_ARCH check. Its siblings b200_matvec_arm_on, b200_gemv_v2_level and b200_bf16_gemv_lt_on require "100a", and the MLA decode arm requires cfg!(memra_sm100_tcgen05). So MEMRA_B200_PRIME_V2=1 changes the glm5 PP-2 prime schedule and pipeline on a 120a build. Severity stays low: the door is opt-in only (no default arm), it is a host-side schedule door rather than a B200 ISA kernel, and FLAGS.md:512 does not claim build gating. The problem is a name and contract mismatch, not wrong output by default.
- fix: Either add env!("MEMRA_BUILT_CUDA_ARCH") == "100a" to b200_prime_v2_on like its siblings, or, since it is a rig-generic schedule door, rename it (for example MEMRA_HC_PRIME_V2) and say so in FLAGS.md.

### low: PP prime chunk loops skip the memra#536 cancellation point

- where: `crates/memra-engine/src/hybrid_forward.rs:2353` (S03, confirmed)
- detail: The single-engine hc loop (2408), the step35 GEMM loop (5734) and the serial loop (5955) call crate::progress::prime_cancel_point after each chunk. The hc ppN chunk loop (2353-2372), prime_cache_hyper_pp2_pipelined (4716-4813), prime_cache_pp2_pipelined (6013-6110) and the PP-3/4 wavefront never do. A client that disconnects during a long PP prime (the glm5 1M-context and 262k PP cells) keeps both cards busy until the whole prompt is primed.
- check: Partly deliberate. progress.rs says the pipelined walks (the PP-2 split with next_slot in flight, the ppN wave) are excluded on purpose because of the drain and tainted-cache seam, so that part is a known design choice. The hyper ppN range loop at 2353 is different. It is a sequential per-range loop: each prime_cache_hyper_ppn returns host logits and calls note_prime_rows, but there is no prime_cancel_point. progress.rs does not name it as an exclusion. This is the loop the glm5 long-context PP cells run, so a client that disconnects there keeps both cards priming until the prompt ends.
- fix: Add crate::progress::prime_cancel_point after note_prime_rows in the hyper ppN range loop. Keep the pipelined walks excluded as documented, or give them a drain-then-cancel seam.

### low: Cold-prompt first-token logits depend on whether another cold prompt was co-batched

- where: `crates/memra-engine/src/hybrid_forward.rs:8667` (S03, confirmed)
- detail: For B>=2 the batched prime takes the f16 m=B lm_head GEMM (8668) and m=sum(T) trunk GEMMs. B=1 uses the m=1 matmul head. The same request primes under two numeric programs depending on arrival timing (the 'batched vs solo' pair AGENTS.md says to watch). It is acknowledged and gated by prime_batch_gate argmax only, so greedy text for one prompt is not reproducible across load.
- check: In prime_cache_batch_inner, B>=2 takes try_f16_gemm(output, hcat, b) for the last-row logits. B=1, and the per-seq fallback, use e.matmul(output, hlast, 1). The comment itself calls this the f16-vs-mmvq numeric class, accepted on the argmax battery. So one request's first-token logits depend on whether a peer was co-batched. AGENTS.md lists batched vs solo as a pair that must be forbidden or proven bit-identical. The trunk part of the claim is weaker: the repo's own receipt says the prefill GEMM is m-invariant.
- fix: Compute each sequence's last-row logits with the same m=1 head the solo prime uses (or prove bit-identity in a serving-shape gate).

### low: fa_sm_count always reads CUDA ordinal 0, and Engine::sm_count caches the first engine's device in a process-global static (lib.rs:23644); both fall back to 82. In a process whose engines sit on cards with different SM counts, the FA split ladder (a numeric class) and the grid-fill policies follow the wrong card. Only PP/TP topology enforces uniform hardware, and only between 5090 and PRO.

- where: `crates/memra-engine/src/lib.rs:1839` (scout:devices, confirmed)
- check: fa_sm_count hard-codes cudarc device::get(0). Engine::sm_count reads self.gpu.ctx but stores it in a function-local static, so whichever engine calls it first sets the value for the whole process. batched_variant has the same pattern. Engine::new takes an ordinal (lib.rs:3731), and detect_uniform_hardware checks only the listed PP devices, not ordinal 0. So an engine on ordinal 1 of a box whose device 0 is a different card class follows device 0's FA split ladder. Uniformity is enforced only by name (5090 vs PRO 6000) in PP/TP topology. This needs a heterogeneous box, so the case is narrow, and the wrong ladder is still a valid program, not a crash.
- fix: Store sm_count on Engine from self.gpu.ctx at construction (no process-global static), and pass it into fa_split_keys instead of reading ordinal 0.

### low: ILP engagement counter is bumped even when the token-major down kernel runs instead

- where: `crates/memra-engine/src/lib.rs:9157` (S01, confirmed)
- detail: moe_down8_fma_q8_rows computes ilp (L9149) and increments MOE_VROWS_ILP_DISPATCHES (L9156-9158) before choosing the kernel. When tmaj is on (L9159) it launches moe_down8_fma_q8_rows_tmaj, which has no ILP form. On a 100a build (ILP default ON) with MEMRA_MOE_VROWS_DOWN_TMAJ=1, the counter rises although no _ilp down kernel ran. Gates that assert moe_vrows_ilp_dispatches() >= d0 + 2 pass vacuously for the down half.
- check: In moe_down8_fma_q8_rows, ilp is computed and MOE_VROWS_ILP_DISPATCHES is bumped before the tmaj branch. When tmaj is on, the launch is moe_down8_fma_q8_rows_tmaj and no _ilp kernel runs. The glm5_matvec_doors_gpu asserts (>= d0 + 2) would then count a down ILP launch that never happened. This only bites when MEMRA_MOE_VROWS_DOWN_TMAJ=1 is set with ILP on, hence low.
- fix: Bump MOE_VROWS_ILP_DISPATCHES only in the non-tmaj branch when it selects moe_down8_fma_q8_rows_ilp.

### low: matmul_pre sends small-out_f tensors to the int8 GEMM; matmul keeps them off it (out_f >= 128 floor)

- where: `crates/memra-engine/src/lib.rs:18532` (S01, confirmed)
- detail: matmul requires out_f >= GEMM_MIN_OUT_F (128) before both the MMQ and qmatvec_gemm arms (L18156, L18159). matmul_pre checks out_f >= 128 for MMQ only (L18513); its qmatvec_gemm arm has no floor. Example: a Q8_0 or Q4_K tensor with out_f < 128, such as qwen35 GDN ssm_alpha/ssm_beta with out_f = num_v_heads. At m >= 16 it runs the int8 MMA GEMM via matmul_pre but the dp4a per-m kernel via matmul. Which numeric program a prime chunk gets therefore depends on which entry the call site uses, not on (dtype, m). The small_m_tier_max doc (L4838-4840) reasons about the out_f >= 128 floor that only matmul has. PLAUSIBLE: not tied here to a path where both entries prime the same tensor, but a prime path that uses matmul_pre next to one that uses matmul (chunk tail, prefix restore) breaks the one-numeric-program rule for that tensor.
- check: matmul gates both MMQ and qmatvec_gemm on out_f >= GEMM_MIN_OUT_F (128). matmul_pre has the floor on MMQ only; its qmatvec_gemm arm has none. Small-out_f quantized tensors at m>=16 therefore take the int8 MMA GEMM through matmul_pre and the dp4a/mmvq path through matmul. One real reach: decode_batch's ssm_beta/ssm_alpha at b_n>=16. The small_m_tier_max doc reasons from the floor only matmul has. No same-request crossing of the two entries on one tensor was shown, so the impact stays low.
- fix: Add `&& w.out_features() >= 128` (shared GEMM_MIN_OUT_F) to matmul_pre's qmatvec_gemm arm.

### low: MEMRA_BVAR can force an _rp kernel onto GGUF-layout bytes

- where: `crates/memra-engine/src/lib.rs:23850` (S02, confirmed)
- detail: The guard prevents a non-rp variant on rp bytes, but any forced value is accepted when rp is false. MEMRA_BVAR=rpr2 on a plain NVFP4 weight therefore dispatches qmatvec_nvfp4_mmvq_b4_rpr2 over interleaved bytes (NaN or garbage). The debug_assert at 23953 checks only the opposite direction.
- check: The guard `Some(v) if !rp || v.contains("rp")` accepts any forced value when rp is false, so MEMRA_BVAR=rpr2 on a GGUF-layout weight dispatches {base}_rpr2 over interleaved bytes. The comment and FLAGS row both claim the layout stays safe. The debug_assert checks only the rp-weight-to-non-rp-kernel direction. MEMRA_BVAR is a tuning seam, so the impact is limited to diagnostic runs.
- fix: Accept the forced variant only when v.contains("rp") == rp; otherwise fall back to batched_variant.

### low: w8_act activation cache mixes two key schemes, so a t=1 lookup can return an undersized buffer

- where: `crates/memra-engine/src/lib.rs:25757` (S02, confirmed)
- detail: matvec_bf16_via_q8_mirror (t=1) keys the shared w8_act map by `in_f` and allocates in_f bytes (25897-25898). matvec_bf16_via_q8_mirror_t and kda_proj_fused6_q8rp_raw_pre key the same map by `in_f*64 + t.min(32)` and allocate 32*in_f (25757-25761, 23330). The keys collide when a t-wide walk ran with t>=32 on in_f_B and a t=1 call later arrives with in_f_A = 64*in_f_B + 32 (in_f_A is 32 mod 64, so it is a legal multiple of 32, e.g. 4128 vs 64 or 6176 vs 96). contains_key(&in_f_A) is then true, and the t=1 path quantizes in_f_A elements into a 32*in_f_B (about in_f_A/2) buffer. quantize_q8_1_into guards only with debug_assert (16166), so a release build writes out of bounds on the device and the W8 decode reads garbage. It is reachable only under MEMRA_GLM5_W8 or MEMRA_STEP_TP_W8+MEMRA_W8_HYBRID, and only for those geometries.
- check: The two key schemes do share one map. The t-wide sites key by in_f*64+min(t,32) and allocate 32*in_f; the t=1 sites (25897 and the view twin at 26162) key by bare in_f and allocate in_f. A t=1 lookup with in_f_A = 64*in_f_B+32 after a t>=32 walk on in_f_B would reuse an undersized buffer, and only a debug_assert guards it. But in_f_A must be 32 mod 64 (for example 2080, 4128, 6176, 8224), and no shipped W8 geometry has that shape. The defect is a real latent memory-safety hazard, not a reachable failure today, so I lowered it from medium to low.
- fix: Key w8_act by a (in_f, t_cap) tuple at all four sites (or a separate map per scheme), and check the buffer length before reuse in release builds.

### low: fa_decode_rows picks v4 from the last row's key bound, so a v4 threshold inside the batch breaks verify/decode parity

- where: `crates/memra-engine/src/lib.rs:30820` (S02, confirmed)
- detail: Rows are grouped by split rung (30803-30816), but the v4/v3 family is chosen once from fa_v4_at(base_len+t). With a finite MEMRA_FA_V4_MAX = N and base_len+1 < N <= base_len+t, the early rows run v3 in verify while eager decode at the same t_kv runs v4: two numeric programs for one request. In production FA_V4_MAX_DEFAULT is only usize::MAX or 0 (hybrid.rs:5592), so this is reachable only through the diagnostic env.
- check: Rows are grouped by split rung, but v4 is chosen once from fa_v4_at(base_len + t). FA_V4_MAX_DEFAULT is only ever usize::MAX or 0 (hybrid.rs stores 0 for gqa > 8). A threshold inside the batch therefore needs an explicit MEMRA_FA_V4_MAX, which is diagnostic-only. Separately, the doc at lib.rs:2208 says gemma sets 1024 at load; no such store exists anymore.
- fix: Include fa_v4_at(base_len + r + 1) in the row-grouping key (or evaluate v4 per group), and fix the stale gemma-1024 doc.

### low: fa_decode_dcw_rows caches the combine kernel choice but re-reads the env for its smem size

- where: `crates/memra-engine/src/lib.rs:32452` (S02, confirmed)
- detail: The kernel pick goes through a OnceLock (32452), while combine_shared, which sizes the dynamic shared memory, is a fresh env read on each call (32458). If MEMRA_FA_COMBINE_S changes after the first call in the process (gates toggle env per call), fa_decode_combine_gate_f32_s can launch with 0 bytes of dynamic smem: an out-of-bounds shared-memory read and wrong attention output. The reverse case gives the plain kernel extra smem, which is harmless. fa_decode_dcw (32649-32670) derives both from the same value and is correct.
- check: The inconsistency is real: the kernel pick goes through a OnceLock, while combine_shared re-reads the env on every call, unlike fa_decode_dcw, which derives both from one value. But no code in crates sets MEMRA_FA_COMBINE_S at runtime (no set_var), so the mismatch cannot happen today. It is a latent hazard.
- fix: Set combine_shared from the cached bool that selects fc, as fa_decode_dcw does.

### low: Q4_0 stream-K fixup scratch is one process-global buffer shared by every engine and device

- where: `crates/memra-engine/src/mmq_ffi.rs:25` (S13, unsure)
- detail: The fixup slot is allocated once on the first engine that runs a Q4_0 SK GEMM (line 1931) and reused by all later engines. Scenario: Gemma-4 Q4_0 on its PP2 split (two Engines in one process, SK default ON on 120a). Stage 1's SK GEMM on dev1 passes a dev0 pool pointer; it works only through the PP pool-access grant and pays remote writes. Two SK GEMMs in flight at once share the same partial-sum scratch: overlapping stage streams, or two models on different cards in one server. Their fixup partials then race and the prefill outputs are wrong. The buffer is also sized from the first device's SM count.
- cut: gemma4_dense, gemma4_moe Q4_0 prime under pp or multi-model multi-device
- check: The structure is as described: MMQ_FIXUP_SLOT is a process static allocated through whichever engine first runs an SK GEMM, and mmq_nsm() in mmq_q4_0.cu is also a static taken from the first device. But I found no live path where a second engine runs the Q4_0 SK GEMM. gemma4_prime walks every layer on the single engine `e` (pp.rs: the PP scope is plain eager decode only), so the claimed 'Gemma-4 PP2 stage-1 SK GEMM on dev1' scenario is not shown. Multi-model-per-process serving was not verified either. This is a latent hazard, not a demonstrated race.
- fix: If multi-engine Q4_0 SK use is ever enabled, move the fixup scratch onto the Engine and size it from that engine's device SM count.

### low: MEMRA_MOE_SLOTS / MEMRA_MOE_VRAM_FRAC fail open: typos are ignored, the forced count is silently clamped, and the frac is unbounded

- where: `crates/memra-engine/src/moe_cache.rs:805` (S12, confirmed)
- detail: FLAGS.md:449 promises 'force an exact SLRU slot count', but build() computes requested = n * slot_bytes and then budget_bytes = requested.min(hard_bytes) (line 821), and n = budget / slot (line 847). Scenario: MEMRA_MOE_SLOTS=4096 on a 24 GB RTX 5090 serving Qwen3.6-35B-A3B in the spill regime gets roughly the 0.80 hard ceiling's worth of slots (fewer than 4096) with no message, so a spill experiment reports numbers under a slot count it never had. MEMRA_MOE_SLOTS=4k or MEMRA_MOE_VRAM_FRAC=0,85 silently falls back to auto. MEMRA_MOE_VRAM_FRAC=-1 gives (free * -1) as usize = 0, so the .max(8) floor yields 8 slots: a thrashing, near stage-every-token cache with no warning. MEMRA_SPILL_IO, MEMRA_SPILL_PREAD_DEPTH and MEMRA_MOE_HARD_VRAM_FRAC already fail closed or bound their values for exactly this reason.
- check: A MEMRA_MOE_SLOTS value that fails to parse is dropped by .ok() and the auto size is used. A forced N is turned into bytes and capped at the hard ceiling with no log line; the only print is on the size-aware path, which a forced N disables. VRAM_FRAC has no bounds check. A negative frac casts to 0 usize, so .max(8) gives 8 slots. MEMRA_MOE_HARD_VRAM_FRAC, by contrast, is parsed with a warning. FLAGS.md promises an exact slot count.
- fix: Parse MEMRA_MOE_SLOTS and MEMRA_MOE_VRAM_FRAC strictly (refuse or warn), bound the frac to (0,1], and print when a forced N is clamped to the hard ceiling.

### low: PP guards parse MEMRA_PP_STAGES independently of pp_cuts, so a PP door that pp_cuts turned OFF can still trigger refusals

- where: `crates/memra-engine/src/pp.rs:306` (S09, confirmed)
- detail: pp_cuts returns None and only warns (door stays OFF) for a bad MEMRA_PP_SPLITS count, a bad MEMRA_PP_SPLIT with N>2, or N > n_layers. pp_sharded_cross_device and pp_multi_stream_same_device only test stages>=2 and the device list. Scenario: MEMRA_PP_STAGES=2, MEMRA_PP_DEVICES=0,1 and a typo in MEMRA_PP_SPLITS. The load runs unsplit with all weights on the primary (layer_engine returns e), then every batched/dc/graph/verify step fails in refuse_unsplit_if_remote with the 28x peer-read message, although nothing is remote. PLAUSIBLE: the batch entry was not traced to confirm it reaches the guard without its own pp_cuts check.
- check: pp_cuts returns None and only warns for a bad MEMRA_PP_SPLITS. The loader then skips the partition check, and layer_engine places every layer on the primary. pp_sharded_cross_device only checks MEMRA_PP_STAGES>=2 and two or more distinct MEMRA_PP_DEVICES, so it returns true. The batched entry falls through its pp_cuts().is_some() arms to refuse_unsplit_if_remote (decode_batch.rs:1326) and refuses with the 28x peer-read message, though nothing is remote. It fails closed, but with a misleading diagnosis.
- fix: Make an invalid PP fence (stages>=2 while pp_cuts is None) a hard load error in load_from_source_impl instead of a warning.

### low: set_idxq silently maps any unknown value to the F32 rollback arm

- where: `crates/memra-engine/src/qwen4exp_gpu.rs:2372` (S08, confirmed)
- detail: The match falls through to 0 (F32) for anything other than q8/1/bf16. MEMRA_Q4E_SEAMS=idxq=Q8 or idxq=bf-16 measures the f32 raw-key cache while the cell believes it armed a quantized arm, and seam_dispatch still reports success. selgroup's grammar was hardened against exactly this failure (malformed spec returns false).
- check: set_idxq maps anything except q8/1/bf16 to 0 (F32), and the seam arm always returns true, so idxq=Q8 silently measures the f32 rollback. idxq_mode_name() feeds receipt headers and would print f32, which limits the damage. selgroup already rejects malformed specs.
- fix: Accept only q8/1/bf16/f32/0 in set_idxq, return bool, and have seam_dispatch return false on anything else.

### low: MEMRA_Q4E_SEAMS=longatt=auto arms Force, not Auto

- where: `crates/memra-engine/src/qwen4exp_gpu.rs:3001` (S08, confirmed)
- detail: apply_env_seams decodes on = (value != "0"), and the longatt arm ignores value. MEMRA_Q4E_SEAMS=longatt=auto (set_longatt accepts "auto") therefore calls set_longatt("force"). With an f32 KV cache (kvq=0) below 12,288 KV rows, Force sends every chunk to sdpa_blocklist_f32 instead of the masked kernel. That is a different program from the historical byte-stable Auto receipts, and the run's header would still claim auto.
- check: apply_env_seams sets on = (v != "0"), and the longatt arm ignores value, so longatt=auto (or any non-0 value) calls set_longatt("force"). The documented seam grammar only covers bare longatt (force) and longatt=0, so this is a footgun for an undocumented spelling, not a broken documented path.
- fix: Pass the value through (set_longatt(value.unwrap_or("force"))) and have set_longatt reject unknown values so seam_dispatch returns false.

### low: qwen prime graph keys on exactly 170 SMs, but device id rtx5090 covers the 82-SM laptop too

- where: `crates/memra-engine/src/qwen_prime_graph.rs:101` (scout:routes, confirmed)
- check: The code keys on exactly 170 SMs on a 120a build. stage0/vocab.json has a single rtx5090 id that covers both the 82-SM laptop and the 170-SM desktop. This is a granularity gap in the impact map's vocabulary, not a memra defect. The scouts already carry per-SM truth notes elsewhere.
- fix: Record the SM-count predicate on the route (desktop 5090 only) or split rtx5090 into 82-SM and 170-SM ids in vocab.json.

### low: Unconditional per-burst stderr lines on the serving hot path

- where: `crates/memra-engine/src/spec.rs:11664` (S10, confirmed)
- detail: Every multi-head (step37) MTP burst prints the [mtp-chain] line (11663), and every constrained spec burst prints the [draft-mask] stats line (14583), which sits outside the spec_stats gate. Serving runs thousands of bursts per session, so stderr fills with noise that is not diagnostics-gated.
- check: generate_spec_inner2 runs once per burst. The [mtp-chain] eprintln fires on every call when mtp_extra is non-empty, with no diagnostics gate. The [draft-mask] eprintln fires on every constrained burst under `if constraint.is_some()`, outside the spec_stats-gated [spec-stats] block above it. Both are log noise on the serving path; neither changes behavior.
- fix: Print [mtp-chain] once per process with a std::sync::Once and gate [draft-mask] behind the spec_stats/debug_spec flag.

### low: Stage-0 graph marks a tests/ file attribute as a non-test use item and links server functions to it

- where: `crates/memra-engine/tests/dsv4_dense_fast_rows_gpu.rs:1` (scout:routes, confirmed)
- check: Part of the report holds and part is stale. Holds: nodes under crates/*/tests/ get test:false (2444 such nodes, including use@1 with idents allow/clippy/needless_range_loop), and server src has noise trait edges into tests/ files (18, e.g. worker HostTierLedger to memra-tier tests/contracts/support.rs). Stale: the current edges.tsv has no edge into dsv4_dense_fast_rows_gpu.rs::use@1, and step_gemma_spec has no edges into tests/.
- fix: In the stage-0 scanner, set test=true for every item under crates/*/tests/ and skip ref/trait edges from src into tests/ items.

### low: DSpark config refusal panics inside the serving boot path

- where: `crates/memra-gguf/src/dsv4_dspark.rs:63` (S07, confirmed)
- detail: With MEMRA_DSV4_DRAFTER=dspark, Dsv4Gpu::load_placed calls DsparkConfig::load (dsv4_gpu.rs:3707, 4281). A missing key or a census-pin drift panics (expect/assert_eq) instead of returning Err. Dsv4Model::open, in contrast, catches the config parser panic and returns a named error. The server boot gets an unwind, not the refusal string the other load errors produce.
- check: DsparkConfig::load returns Self and panics (expect, unwrap_or_else(panic!), assert_eq!) on a missing key or a census-pin drift. Dsv4Gpu::load_placed calls it directly when MEMRA_DSV4_DRAFTER=dspark (3707, 4281), outside the Res<> error path, so a drifted drafter config unwinds the load instead of returning a named Err. The panic text is still descriptive.
- fix: Return Result<DsparkConfig, String> with named refusals for each pin and propagate with ? in load_placed.

### low: gemma4_moe only allows tokenizer.json, but its served artifact is the QAT Q4_0 GGUF (tokenizer in GGUF metadata); gemma4_dense allows both

- where: `crates/memra-gguf/src/model_packs/gemma4_moe/mod.rs:10` (scout:packs, confirmed)
- check: gemma4_moe lists only TokenizerJson, while gemma4_dense also accepts GgufMetadata, and MODELS.md lists the 26B-A4B as QAT Q4_0 (GGUF). The CLI onboarding gate (memra-cli lib.rs:1307, 2104) checks pack.tokenizer_sources, so GGUF-metadata onboarding of the 26B reports a tokenizer error. Only the onboarding gate is affected; serving load does not consult tokenizer_sources.
- fix: Add TokenizerSource::GgufMetadata to gemma4_moe's tokenizer_sources, as in gemma4_dense.

### low: glm_dsa matches on arch alone, but the HF path always leaves cfg.mla None, so an HF glm_moe_dsa config compiles a full-attention plan instead of MLA

- where: `crates/memra-gguf/src/model_packs/glm_dsa/mod.rs:34` (scout:packs, confirmed)
- check: The HF path maps glm_moe_dsa to Arch::GlmDsa (config.rs:96) and always sets mla: None (config.rs:2582). validate_plan_semantics has no GlmDsa or MLA check, and glm_dsa matches on arch alone, so the canonical plan compiles without the MLA branch. The tensor census bound right after compile_for_source (hybrid.rs:4245) should refuse, because the full-attention tensor set will not match GLM-5 safetensors. So this gives a misleading refusal, not wrong output. qwen4_exp and minimax_m3 do require their config blocks.
- fix: matches_config: |config| matches!(config.arch, Arch::GlmDsa) && config.mla.is_some()

### low: BF16 head Q5_K re-encode keyed on total elements, not row length

- where: `crates/memra-gguf/src/source.rs:2977` (S20, confirmed)
- detail: The comment says Q5_K needs in_f % 256 == 0, but the guard checks the total element count. par_encode_bf16 encodes the flat plane in 256-element blocks, so a head with hidden % 256 != 0 but vocab*hidden % 256 == 0 gets Q5_K blocks that straddle rows. Example: hidden 2880, vocab 201088. The engine then reads in_f/256 blocks per row and decodes garbage logits. No current board model has such a hidden size, so this is latent.
- cut: none
- check: The guard checks the total element count (n_el % 256), not in_f % 256, while the comment says Q5_K needs in_f % 256 == 0. par_encode_bf16 encodes the flat buffer in 256-element blocks, so a row length that is not a multiple of 256 produces blocks that straddle rows. Example: 2880 x 201088 is divisible by 256 although 2880 is not. This is latent for current board models, and the same total-count pattern applies to the Q8_0 % 32 check.
- fix: Gate on info.shape[1] % 256 == 0 (and info.shape[1] % 32 == 0 for Q8_0); otherwise keep BF16 or fall back to Q8_0 by row.

### low: is_dsv4_dir is a substring test over all of config.json

- where: `crates/memra-server/src/dsv4_serve.rs:57` (S07, confirmed)
- detail: Any model dir whose config.json has a model_type key and the text deepseek_v4 anywhere (a distill whose _name_or_path or base_model mentions deepseek_v4, or a nested text_config) is routed to the dsv4 thread. Dsv4Model::open then refuses because mc.arch is not dsv4, so the model fails to boot instead of loading through HybridModel.
- check: is_dsv4_dir matches only two raw substrings anywhere in config.json, "model_type" and "deepseek_v4" (quoted). A non-dsv4 config with an exact "deepseek_v4" string value anywhere (a nested text_config, a base-model field) routes to the dsv4 thread, where the strict loader refuses it. The probe is also narrower than the pack aliases (deepseek-v4, deepseek_v4_preview): a config with model_type deepseek_v4_preview does not contain the exact "deepseek_v4" token, so it goes to HybridModel. No current checkpoint is known to hit either case, so this is low.
- fix: Parse config.json and compare the top-level model_type string against the deepseek_v4 pack aliases.

### low: KV flex grant is defeated by the snapshot preflight, which budgets against the floor

- where: `crates/memra-server/src/worker.rs:5943` (S17, confirmed)
- detail: insert_demoting budgets with kv_flex_effective_budget(), which is floor plus grant. prefix_insert_from_session first calls px.prepare_snapshot(..., prefix_cache_budget_bytes(), ...) at worker.rs:24291, the floor only. prepare_snapshot then evicts until total_bytes <= floor - bytes (9386-9396). Failure scenario: MEMRA_KV_FLEX=1 on a co-resident box. The chat model borrows headroom above its floor, and the next prompt-end seed or affinity publish through prefix_insert_from_session evicts every borrowed entry before inserting. Each publish leaves the cache at or below the floor, so flex never retains extra entries on this path. borrowed_bytes reads near 0 and sheds stay 0, so a flex A/B on this traffic measures nothing.
- cut: none
- check: Both insert wrappers use kv_flex_effective_budget() (floor plus grant). The only prepare_snapshot call, in prefix_insert_from_session, passes prefix_cache_budget_bytes() (the floor), and prepare_snapshot evicts until total_bytes <= budget - bytes. Under MEMRA_KV_FLEX=1, every publish through this path sheds the borrowed entries first. Flex is default OFF, so this affects only the flex A/B.
- fix: Pass kv_flex_effective_budget() to prepare_snapshot at 24291 and extend the source test at 48688 to cover that call.

### low: gemma4 boot line says monolithic prefill, but chunk-capable gemma plans chunk

- where: `crates/memra-server/src/worker.rs:26645` (scout:routes, confirmed)
- check: The gemma4 dense batched-decode boot line hardcodes 'monolithic prefill, no graph promotion'. monolithic_prime_model excludes gemma plans with chunked_prime_supported, and its doc says dense 12B/31B and the 26B-A4B prime in tick-budget chunks. 'graph promotion' appears only in these log strings and in comments.
- fix: Print the prefill mode from monolithic_prime_model(lm) and drop 'no graph promotion' from the three boot lines.

## Broken gates

### medium: CUDA-side MEMRA_* getenv reads escape the flags census

- where: `crates/memra-engine/cu/mimo_mixed_attn.cu:530` (S16, confirmed)
- detail: tools/check-flags.sh scans only *.rs files under crates/*/src, so a getenv in cu/ is never checked against docs/FLAGS.md. MEMRA_MIMO_SPLIT_STAGE_TIMING has no FLAGS.md row. The dsv4 doors MEMRA_DSV4_DENSE_EXACT_TAIL, MEMRA_DSV4_DENSE_FAST and MEMRA_DSV4_HC_DOT_SPLIT are also read only in cu/dsv4_dense_m1_exact_tail.cuh:15/41/588; their rows exist only because someone wrote them by hand. A new numeric door added in a .cu file would pass the pre-push hook with no row and no decide-by date.
- check: check-flags.sh scans only `--glob '*.rs'` under crates/*/src and src/bin, so getenv reads in cu/ never reach the FLAGS.md census. MEMRA_MIMO_SPLIT_STAGE_TIMING (mimo_mixed_attn.cu:530) has no FLAGS.md row. One part is overstated: the three dsv4 cu reads are covered by a second check, dsv4_doors::door_name_source_files, which scans cu/dsv4*, so their declaration does not rest only on hand-written rows. Non-dsv4 .cu reads have no check.
- fix: Extend check-flags.sh pass A to scan crates/*/cu/*.cu and *.cuh for getenv("MEMRA_...") and add the MEMRA_MIMO_SPLIT_STAGE_TIMING diagnostic row.

### medium: Release battery never gates the batched serving prime (prime_cache) against tokenwise decode

- where: `crates/memra-engine/src/bin/argmax_margin_probe.rs:145` (S22, confirmed)
- detail: release-battery.sh replaced raw run-gen with argmax-margin-gate.sh (release-battery.sh:259). That gate compares forward_last with decode_step only. run_gen's gap #46 cell (run_gen.rs:1269-1300) exists because forward_last and tokenwise can both be green while the batched prime_cache flips the first token (Qwen3.6-35B pp512). run-spec's oracle primes through prime_cache but compares spec against plain on the same prime, so a prime_cache-only regression passes the whole release battery. Scenario: a hybrid_forward.rs change that moves prime_cache's last-row logits beyond the calibrated bounds on the Ornith-1.5 roster model ships with a PASS receipt.
- check: The release battery runs only argmax-margin-gate.sh, run-spec and kernel-check. argmax-margin-probe compares forward_last with decode_step and never calls prime_cache. The prime_cache-vs-tokenwise gate exists only in run_gen.rs, which the battery no longer calls. run-spec's plain oracle and generate_spec both prime through prime_cache, so a prime-only drift is shared by reference and candidate. kernel-check's BATCHED PROMPT PRIME cell is a synthetic KV-append byte check, not a model-level logits comparison.
- fix: Add a prime_cache arm to argmax-margin-probe judged by prime_gate_verdict, or run run-gen's batched-prime gate per roster model in release-battery.sh.

### medium: run-gen argmax gate does not exercise the served decode program

- where: `crates/memra-engine/src/decode.rs:3429` (S05, confirmed)
- detail: run-gen calls generate_with. For greedy runs this decodes through decode_step_dc or graph_decode_loop, which is the decode_step_h numeric class. Serving decodes every non-eager-only model through the generic batched body at every width, B=1 included, because MEMRA_SERVE_B1FAST is off. decode_batch.rs:796-800 states that the eager program and the batched body carry a decode-config FP-composition gap that changed live token streams. So a change to decode_batch_layers or to the epilogue can pass run-gen MATCH and still move served tokens, and a run-gen MISMATCH can come from code serving never runs. The AGENTS.md gate list treats run-gen as the per-family numeric gate.
- check: generate_with (run-gen) decodes greedy runs through decode_step_dc or graph_decode_loop, which is the decode_step_h numeric class. Serving defaults B=1 to the generic batched body, and decode_batch.rs states the two programs carry an FP-composition gap that changed live token streams. run_gen.rs has no batched arm. AGENTS.md also asks for serve-smoke/identity on serving-path changes, which softens this, but run-gen as the per-family numeric gate cannot see changes to decode_batch_layers.
- fix: Add a run-gen arm (default for gating) that decodes through decode_step_batch_sampled_lean_masked at B=1, and keep the dc arm as a second cell.

### medium: Gate entry Dsv4Gpu::load defaults to PP-2 while serving defaults to TP/EP + attention TP

- where: `crates/memra-engine/src/dsv4_gpu.rs:3541` (S06, confirmed)
- detail: dsv4_serve resolve_topology maps unset MEMRA_DSV4_TOPOLOGY to TpEp{attention_tp:true}, but Dsv4Gpu::load (3527-3544) picks PP unless a gate set set_tp_ep_topology_for_gate. 33 of the 46 dsv4 gate bins that call Dsv4Gpu::load never arm TP/EP, among them dsv4_program_accuracy, dsv4_gpu_gate, dsv4_gpu_tf_gate, dsv4_plain_perf_gate, dsv4_host_cache_gate and dsv4_long_decode_gate. They qualify decode_matrix_step, the PP verify walk (20833-21044) and the grouped no-EP MoE arm (20316). The served code (decode_step_tp_ep, verify_batch_tp_ep, moe_tp_ep_fused, full-token replay) is not in their reach, so a TP/EP regression passes them. The reverse also holds: a change to PP-only code fails gates for a placement nobody serves by default.
- cut: Lets the push gate map TP/EP-only regions to TP/EP gate cells and PP regions to PP cells, instead of running every dsv4 gate for both.
- check: Dsv4Gpu::load resolves to Dsv4Placement::Pp unless dsv4_topology::tp_ep_for_gate() was armed. dsv4_serve::resolve_topology maps unset to TpEp{attention_tp:true}. Of the 47 files that call Dsv4Gpu::load(, 33 never call set_tp_ep_topology_for_gate, tp_ep_for_gate or load_placed. These include dsv4_program_accuracy, dsv4_gpu_gate, dsv4_gpu_tf_gate, dsv4_plain_perf_gate, dsv4_host_cache_gate and dsv4_long_decode_gate, so they qualify only the PP placement. The PP default is documented on load(), so this is a misleading gate rather than a hidden bug. TP/EP coverage exists in 14 other gates.
- fix: Make Dsv4Gpu::load default to TpEp{attention_tp:true} like dsv4_serve (or take a Dsv4Placement), and pin the PP-only gates to Pp explicitly as PP cells.

### medium: Gemma pre-staged FA prefill skips the portable_mma_gated fallback

- where: `crates/memra-engine/src/lib.rs:28429` (S15, confirmed)
- detail: fa_prefill, fa_prefill_w, fa_prefill_hd512, fa_prefill_view and the view_ws twins all return to sdpa_naive when portable_mma_gated() (lib.rs:28205, 28395, 28855, 29443, 29642). fa_prefill_w_pre, fa_prefill_hd512_pre, gemma_prefill_view_bf16 and mla_attn_gathered_tc have no such check. Gemma prime calls the _pre variants whenever emit holds (t>=16, rows>=64, MEMRA_FA_EMIT default on, hybrid_forward.rs:22206). Scenario: on the sm89 portable eval build, a gemma4 31B prompt of 4k tokens primes its SWA and hd512 layers through the bf16-MMA fa_prefill_w_bf16_p1h2 and fa_prefill_bf16_hd512_sp16h2 kernels, while a qwen prompt on the same build takes naive SDPA. The portable lane meant to be the correctness-first class silently runs the tuned numeric class for gemma (and glm5 MLA TC prefill), so its receipts are not comparable across families.
- check: fa_prefill and the other non-_pre variants return to sdpa_naive when portable_mma_gated(). fa_prefill_w_pre has no such check and launches fa_prefill_w_bf16_p1h2/p1 directly. Gemma's emit gate (t>=16, qkvnorm_w_on_prefill: rows>=64, MEMRA_FA_EMIT default on) has no portable term either. So on sm_89, gemma SWA and hd512 primes take the bf16-MMA class while other families take naive SDPA.
- fix: Force emit=false when portable_mma_gated(), and add the same early return to fa_prefill_w_pre, fa_prefill_hd512_pre, gemma_prefill_view_bf16 and mla_attn_gathered_tc.

### medium: The literal crate-input census counts any mention of a path under crates/, so a README.md edit reaches numeric on every build target

- where: `tools/change_impact.py:509` (map, checked by replay (history.md row c5853ae546))
- detail: literal_inputs() treats a non-crate path as a crate input when any non-comment line under crates/ spells it. crates/memra-gguf/src/model_packs/whisper/fixtures/source.lock.json:10 lists "path": "README.md" (a file of a model artifact), so a one-line README.md fix (c5853ae546) classifies as numeric on sm_120a, sm_100a, sm_90a and sm_89 and the push gate asks for the device battery. Replayed in docs/impact-map/history.md.
- fix: count only include_str!/include_bytes!/include! arguments and build-script reads (rerun-if-changed), not string mentions in data files

### medium: Fast-gate DEFAULT/expansion is described as conservative but only reaches gemma4 and qwen35-class models

- where: `tools/fast-gate/map.tsv:148` (S22, confirmed)
- detail: Unmatched paths, and every dependency-graph expansion (plan.py adds DEFAULT whenever native-checkpoint expands), plan g12, q9, q35, gwstress and q35spec. No map row covers cu/dsv4_gpu.cu, dsv4_sampler.cu, mla_attn.cu, tp_ar.cu, mimo_*.cu, mmq_fp8_blk.cu or fp8_blk_dequant.cu, and no models.tsv probe loads deepseek_v4, glm5_next, glm_dsa, hy3, minimax_m3, qwen3, qwen3_moe, olmoe, llama_dense or any FP8_BLOCK checkpoint. Scenario: an edit to dsv4_gpu.cu's hc kernels (used by glm5_next via hyper.rs) gets plan=expand with probes that cannot execute that code, and every one passes.
- check: No map.tsv row matches dsv4_gpu.cu, dsv4_sampler.cu, mla_attn.cu, tp_ar.cu, mimo_*.cu or the fp8_blk sources, so they fall to DEFAULT (g12,q9,q35,gwstress / q35spec). models.tsv has no dsv4, glm5_next, hy3 or FP8_BLOCK probe, and kernel_check.rs has no dsv4/hc/mla cell. The header still calls DEFAULT 'conservative ... one probe per model class'. Mitigation: AGENTS.md says fast-gate GPU selection is shadow-only and only informs the scoped battery, so this misleads the planner rather than gating merges.
- fix: Make the plan print the families no selected probe reaches, and add dsv4/glm5_next/hy3/FP8_BLOCK probes, or map those TUs to an explicit 'no fast probe' verdict.

### medium: Release battery covers two qwen35-class models; AGENTS.md's full battery promises every board model

- where: `tools/release-roster.tsv:18` (S22, confirmed)
- detail: AGENTS.md:118 defines the full battery as run-gen argmax on every board model plus run-spec K=1..8. release-battery.sh and release_qualification.py:394 bind the expected cells to this two-row roster (Ornith-1.5-35B-A3B and Qwen3.8-27B, both NVFP4+Q5_K GDN plans). A tagged release therefore carries no native evidence for gemma4, step35 (the blocking PRO 6000 target, gated separately only by the optional step-pro hook), glm5_next, deepseek_v4, hy3 or FP8_BLOCK/Q4_0/IQ4_XS formats. For the impact map, a release PASS says nothing outside {qwen35, qwen35_moe} x {NVFP4, Q5_K}.
- check: AGENTS.md defines the full battery, required before any tag, as run-gen argmax on every board model. release_qualification.py builds the expected cell set from the roster, and the roster has two rows (Ornith-1.5-35B-A3B and Qwen3.8-27B, both NVFP4+Q5K qwen35-class). A tag PASS therefore carries no evidence for gemma4, step35, glm5_next, dsv4 or hy3. RELEASING.md says the roster is the model set, so AGENTS.md and RELEASING.md contradict each other.
- fix: Add roster rows for every board family a release claims, or narrow AGENTS.md/RELEASING.md to say the tag gate covers the qwen35 class only and require per-family cells separately.

### low: MEMRA_DSV4_CUTLASS has no 120a build-arch assert

- where: `crates/memra-engine/build.rs:756` (S16, confirmed)
- detail: The DSV4 dense CUTLASS block always compiles with -gencode arch=compute_120a,code=sm_120a (build.rs:767) and links it whole-archive, whatever MEMRA_CUDA_ARCH is. MEMRA_CUTLASS asserts a 120a build; this block does not. On a 100a or 90a build with MEMRA_DSV4_CUTLASS set, the weak memra_dsv4_dense_cutlass_fp8 is non-null. memra_dsv4_gemv_fp8_m then skips the exact FP8 tile for m>32 (dsv4_gpu.cu:4922) and calls CUTLASS for every m==32 chunk. The launch has no sm_100 SASS, so gemm(stream) fails and returns 40082. dsv4_gpu.cu:4956 treats that as a real failure, and every deepseek_v4 prefill on B200 errors.
- cut: Removes b200/h100/sm89 x deepseek_v4 x prime from the 120a-MEMRA_DSV4_CUTLASS variant's reach.
- check: The MEMRA_DSV4_CUTLASS block always compiles sm_120a and links whole-archive, with no arch assert (MEMRA_CUTLASS has one at build.rs:273). g_armed defaults to true. Once linked, the non-null weak symbol disables the exact FP8 tile for m>32, and an sm_100 launch fails initialize/gemm (40081/40082), which the dispatcher returns as a real failure. The scenario is real but needs a deliberate opt-in: no tool, CI or release script sets MEMRA_DSV4_CUTLASS, and KERNELS.md lists the path as sm_120a. Severity lowered to low.
- fix: Add assert!(cuda_arch == "120a", "MEMRA_DSV4_CUTLASS is sm_120a-only") at the top of the MEMRA_DSV4_CUTLASS block.

### low: MEMRA_DFLASH_FA door routes to a kernel the code calls unfixed

- where: `crates/memra-engine/src/dflash.rs:3325` (S11, confirmed)
- detail: Both forward_block (2764) and forward_round (3325) keep a fa_prefill arm behind MEMRA_DFLASH_FA. The comment says that arm returns only once its kernel is fixed and parity-gated. The door has no decide-by date and selects a known-unqualified program for the drafter. It also costs one env read per drafter layer per round.
- check: The forward_block comment says fa_prefill's non-causal T != T_kv arm is BROKEN (maxdiff 0.34) and returns only once fixed. Both forward_block and forward_round still route to e.fa_prefill whenever MEMRA_DFLASH_FA is set to any value, and the FLAGS row has no decide-by. The door is opt-in, so default output is unaffected.
- fix: Delete the MEMRA_DFLASH_FA reads and both fa_prefill arms and move the FLAGS row to Removed doors.

### low: SERVED_PROGRAM assumes a resident drafter that the naked default does not load

- where: `crates/memra-engine/src/dsv4_doors.rs:239` (S07, confirmed)
- detail: MEMRA_DSV4_DRAFTER is unset by default, so a naked launch is drafter-less TP/EP, and replay cadence engages there. The registry nevertheless declares it OffProgram and counts it in inert_default_on_doors. door_receipt_lines prints resolved= for the loaded program but disposition= from served_disposition(SERVED_PROGRAM), so a naked boot logs replay cadence as resolved On with a not-engaged disposition.
- check: SERVED_PROGRAM hardcodes drafter_resident: true, but MEMRA_DSV4_DRAFTER is unset by default (FLAGS: 'unset (trunk-only, plain routes)'). door_program() derives drafter_resident from the loaded model, so on a naked TP/EP boot resolve_replay_cadence returns On. door_receipt_lines still computes disposition from served_disposition, which resolves against SERVED_PROGRAM (OffProgram, not engaged). One boot-log line then reports resolved=On with a non-engaged disposition, and inert_default_on_doors counts a door that engages on the naked default. Impact is limited to receipts and the registry claim; the kernels are unaffected.
- fix: Have served_disposition take the program (door_program() at load), or declare two served programs (drafter / no drafter) and count inert doors per program.

### low: MEMRA_STEP_TP_PREFILL=1 refuses every step35 GEMM prime, including the per-session path its message recommends

- where: `crates/memra-engine/src/hybrid_forward.rs:150` (S03, confirmed)
- detail: validate_step_prime_batch_modes errors whenever tp_prefill is set and tells the operator to use per-session grouped prefill. It runs at the top of step35_prime_cache_batch (8068), and that is also the B=1 single-session entry (5711). With the door on, no step35 prime of 16+ tokens succeeds, so the door's grouped-prefill arm (12187-12203) is reachable only with MEMRA_STEP_GEMM_PRIME=0, which the server refuses for SlidingGatedMoe.
- check: validate_step_prime_batch_modes errors whenever tp_prefill is true. It runs at the top of step35_prime_cache_batch with no prompts.len() check, and that function is also the B=1 single-session entry. So with MEMRA_STEP_TP_PREFILL=1 every step37 prime of 16+ tokens fails, including the per-session path the message and the FLAGS row point to. The grouped-prefill arm is reachable only through the walk (MEMRA_STEP_GEMM_PRIME=0), which the server refuses for SlidingGatedMoe. The door is default OFF, so the impact is low.
- fix: Run validate_step_prime_batch_modes only when prompts.len() > 1, or delete the TP prefill doors per the door hygiene rule.

### low: MEMRA_Q8_ROW_ILP's b200 default is unreachable without MEMRA_B200_GEMV_V2

- where: `crates/memra-engine/src/lib.rs:1408` (S14, confirmed)
- detail: q8_row_ilp_on() defaults ON on 100a builds, but its only consumers are qmatvec_q8_0_rp_v2_raw (called only under b200_gemv_v2_on(), lib.rs:25921), rows_tw_v2 (under b200_gemv_v2_on(), lib.rs:25783) and kda_proj_fused6_q8rp (gated by b200_gemv_v2_on(), kda.rs:1851). MEMRA_B200_GEMV_V2 defaults off, so a naked b200 launch never runs q8_0_mmvq_row1_rp_v2_ilp / qmatvec_kda6_q8f32_rp_v2_ilp although the device table and docs list it as a b200 default.
- check: q8_row_ilp_on is consumed by three paths, all behind b200_gemv_v2_on(). qmatvec_q8_0_rp_v2_raw is called in production only at lib.rs:25929 under b200_gemv_v2_on(). kda_proj_fused6_q8rp is reached only after kda.rs returns early unless b200_gemv_v2_on(). rows_tw_v2 does not take the ILP arm at all. MEMRA_B200_GEMV_V2 defaults off, so on a bare sm_100a launch the 'ON on sm_100a builds' default never engages. The FLAGS row does not say it depends on B200_GEMV_V2.
- fix: Document MEMRA_Q8_ROW_ILP as a sub-arm of MEMRA_B200_GEMV_V2 (effective only when that door is >= 1) in FLAGS.md and the lib.rs doc, or route the shipped qmatvec_q8_0_mmvq_rp path through the ILP twin.

### low: MEMRA_FA3=1 on sm_120a or sm_100a reaches the FA3 stub instead of refusing at the door

- where: `crates/memra-engine/src/lib.rs:28220` (S02, confirmed)
- detail: refuse_portable_force refuses only on portable builds. On a 120a or 100a build a forced MEMRA_FA3=1 calls memra_fa3_prefill, which the device table describes as a stub off 90a, and fails mid-request with rc!=0. It does not fail at the switch.
- check: build.rs compiles fa3_prefill.cu with -DMEMRA_FA3_STUB whenever cuda_arch != 90a, and the stub returns 3. refuse_portable_force panics only on portable builds, and it is unreachable anyway, because the earlier portable_mma_gated() check returns sdpa_naive first. A forced MEMRA_FA3=1 on 120a or 100a therefore does three bf16 conversions and then returns Err(rc=3) at request time. That is fail-closed (no wrong output), but it fails mid-request instead of at the switch.
- fix: In the Ok("1") arm, assert cfg!(memra_hopper_mma) with a message naming MEMRA_FA3=1 and the 90a requirement (in place of, or alongside, refuse_portable_force).

### low: Step PP topology validation is never called

- where: `crates/memra-engine/src/parallel.rs:1119` (S05, confirmed)
- detail: The doc says serving requires both this topology gate and the loader census. Nothing in the repo calls it, so a hand-set MEMRA_PP_STAGES is never checked against HardwareTarget (card class, max_cards: 5090 = 1) or the contract's stage legality on the loader path. Only the ModelPlan partition check in load_from_source_impl (4433-4457) runs.
- check: validate_step_pp_request has exactly one occurrence repo-wide, its definition. Its own doc says serving requires both gates, but only the ModelPlan partition check in load_from_source_impl runs on the loader path. The EP/TP preflights in hybrid.rs call detect_uniform_hardware for EP groups, not for PP stages.
- fix: Call validate_step_pp_request from load_from_source_impl before init_model_transport, or delete it and the 'serving requires both gates' doc line.

### low: glm5 cold-short depth 6 comes from a B200-only ladder and applies on every card

- where: `crates/memra-server/src/worker.rs:33830` (S18, confirmed)
- detail: The K ladder is from a 2x B200 pair (33821). The constant applies to glm5_next spec on PRO 6000 and any other card. The route is default-off (MEMRA_GLM5_SPEC), so no naked default cell is affected; armed fleets on other cards get an unmeasured depth.
- check: GLM5_SPEC_K_COLD_SHORT=6 comes from a 2x B200 ladder and applies to every build. Exposure is narrow because MEMRA_GLM5_SPEC is default off, so only armed fleets on non-B200 cards get the unmeasured depth.
- fix: Return 6 only on 100a builds (else the shared table's K), or record a PRO 6000 ladder.

### low: Device-penalty sampling default extends a single-5090 receipt to every 120a card

- where: `crates/memra-server/src/worker.rs:39128` (S18, confirmed)
- detail: The only cited receipt is one RTX 5090. The default is keyed on the build arch, so a PRO 6000 or GB10 serving qwen35 also takes the device penalty path by default. Sampled-with-penalty requests on those cards change numeric class (the device draw) with no receipt on that hardware, against the per-hardware arm rule (one-rig evidence sets a one-rig default).
- check: The default keys on built_arch == "120a", which covers the RTX 5090 and the RTX PRO 6000 alike. The only receipt cited for Qwen35 is one RTX 5090. AGENTS.md says one-rig evidence sets a one-rig default at most and asks for card-class keying. The doc says a PRO 6000 result exists for Qwen3Moe only. Rows with a penalty change from host sampling to a device draw. The draw is distributionally equal but a different stream.
- fix: Key the default on the RTX 5090 card class until a PRO 6000 Qwen35 receipt lands.

### low: Fast-gate maps five rows to the e4b probe, but both gemma packs refuse E4B configs

- where: `tools/fast-gate/map.tsv:60` (S22, confirmed)
- detail: gemma4_dense and gemma4_moe match_config only when n_embd_per_layer == 0 && shared_kv_layers == 0, so Gemma-4 E4B gets NoMatchingModelPack (per the scout family table). The e4b probe (models.tsv:32) is selected for kernels.cu, decode/hybrid_forward, hybrid/model/lib and memra-gguf changes. It can only fail at load, or show as SKIP when the file is missing, so it gives no coverage for the E4B PLE/KV-share code it claims to gate. Not run here (no GPU).
- check: Both gemma4 packs match only when n_embd_per_layer == 0 && shared_kv_layers == 0. compile_for_load turns a miss into NoMatchingModelPack, and HybridModel load calls it before any tensor read. An E4B GGUF therefore cannot load, yet map.tsv rows 52, 60, 103, 113 and 122 select e4b, and the models.tsv comment says e4b is the only arm reaching PLE/KV-share. docs/MODELS.md:107 also still lists E4B as served.
- fix: Drop e4b from map.tsv/models.tsv (and the MODELS.md served row) until a pack admits E4B.

## Dead code

### low: memra_dsv4_fp4_gemm_sel_g has no caller

- where: `crates/memra-engine/cu/dsv4_gpu.cu:2286` (S16, confirmed)
- detail: Only the dsv4_ffi.rs declaration references it. Callers use memra_dsv4_fp4_gemm_sel_g_arm directly.
- check: memra_dsv4_fp4_gemm_sel_g appears only at its definition and in the dsv4_ffi.rs extern declaration. Nothing in crates/ or tools/ calls it.
- fix: Delete the wrapper and its dsv4_ffi.rs declaration.

### low: fattn_vendor.cu is an empty skeleton since 2026-07-22

- where: `crates/memra-engine/cu/fattn_vendor.cu:28` (S15, confirmed)
- detail: 38 comment lines, no code, not in build.rs. MEMRA_FA_F16ACC, its door, is unimplemented. It sits in cu/ like a kernel TU and adds noise to kernel censuses.
- check: The file is 38 lines with zero non-comment lines and is not in build.rs; KERNELS.md already says it is NOT BUILT. It is still referenced by tools/fast-gate/map.tsv:52 and THIRD-PARTY.md:11, and both need updating if it is removed.
- fix: Move the plan to research/ and delete the file, updating map.tsv and THIRD-PARTY.md.

### low: HOIST_ALIGN/CAST_LD branches of fa_dec_v3_walk_u are never instantiated

- where: `crates/memra-engine/cu/flash_attn.cu:6164` (S15, confirmed)
- detail: The only instantiations are <4>, <4,false,false,true> and <8> (flash_attn.cu:6146, 9542, 9602). The MEMRA_FA_HOIST door was removed (FLAGS.md:2184), but the HOIST_ALIGN/CAST_LD code paths and the comment naming MEMRA_FA_HOIST=1 (6179) remain.
- check: There are exactly three instantiations: the macro's <4>, <4,false,false,true> (PROF) and <8>. No call sets HOIST_ALIGN or CAST_LD to true, and the comment still names the removed MEMRA_FA_HOIST=1.
- fix: Remove the HOIST_ALIGN/CAST_LD template parameters, their branches and the stale comment.

### low: Dead decode kernels reg_w_i2 and rows_dpl16_kv, plus dead kernels.cu argmax

- where: `crates/memra-engine/cu/flash_attn.cu:8617` (S15, confirmed)
- detail: No crate, bin or tool names fa_decode_vec_q_rows_reg_w_i2 or fa_decode_vec_q_rows_dpl16_kv (the latter is the K=V twin retired 2026-07-11, lib.rs:30822), and none looks up argmax_logits_f32_to_u32 (kernels.cu:29). All three are still compiled, and the first two twice. The reg_w_i2 comment claims a kf8vf8-module launch that no longer exists.
- check: Outside research/, the only hits for fa_decode_vec_q_rows_reg_w_i2 and fa_decode_vec_q_rows_dpl16_kv are their definitions in flash_attn.cu. lib.rs picks only rows_dpl16_i2/rows_dpl16. argmax_logits_f32_to_u32 appears only in its definition and in a doc comment in bin/argmax_gate.rs. That gate calls argmax_token_device, which launches argmax_partial_f32/argmax_final_f32.
- fix: Delete the three kernels and their KERNELS.md rows; fix the argmax_gate doc comment.

### low: gdn_p_bf16_masked is never loaded

- where: `crates/memra-engine/cu/hybrid.cu:2374` (S16, confirmed)
- detail: It was replaced by the K4/K5 wgmma contract (hybrid.cu:2643), and no Rust func() load names it. It still compiles into every non-portable hybrid fatbin.
- check: The kernel is defined at hybrid.cu:2374. No Rust func() load or string names gdn_p_bf16_masked. lib.rs:34684 and ARCHITECTURE-H100.md:1648 both say the pass was deleted. KERNELS.md still lists it.
- fix: Delete the kernel and drop it from the KERNELS.md row.

### low: Unused FFI entry points memra_mmq_nvfp4, memra_mmq_nvfp4_ex, memra_mmq_q4_0

- where: `crates/memra-engine/cu/mmq_fp4.cu:1528` (S14, confirmed)
- detail: Rust declares them (mmq_ffi.rs:749, 765, 1006) but never calls them: the W4A4 path uses memra_mmq_nvfp4_ex2 and Q4_0 uses memra_mmq_q4_0_gemm/_sk. The comment calls memra_mmq_nvfp4 the default entry, which is stale. They still have to be mirrored in mmq_fp4_stub.cu.
- check: memra_mmq_nvfp4, memra_mmq_nvfp4_ex and memra_mmq_q4_0 appear in Rust only as extern declarations in mmq_ffi.rs (749, 765, 1006), with no call sites in crates, tools or probe. The two nvfp4 entries are mirrored in mmq_fp4_stub.cu. memra_mmq_q4_0 is not in a stub, so that part of the report applies only to the nvfp4 pair.
- fix: Delete the three C wrappers, their extern decls and the two stub mirrors, and update the 'same contract as memra_mmq_nvfp4' doc references.

### low: memra_moe_f16g_w_bytes / _act_bytes unused

- where: `crates/memra-engine/cu/moe_f16_grouped.cu:2226` (S16, confirmed)
- detail: No FFI declaration and no caller in any crate.
- check: memra_moe_f16g_w_bytes and memra_moe_f16g_act_bytes appear only at their definitions. The whole repo has no FFI declaration or caller for either.
- fix: Delete both helpers.

### low: Kernels reachable only through Rust launchers that nothing calls

- where: `crates/memra-engine/cu/qmatvec.cu:10911` (S14, confirmed)
- detail: qmatvec_nvfp4_bf16_sel_down_fma (10871), qmatvec_nvfp4_bf16_ep_down_fma (10911), qmatvec_nvfp4_q8_ep_down_fma (11008), axpy_rows_seq_md_off_f32 (12635) and moe_w_exscale (7299) are launched only by Engine::qmatvec_nvfp4_bf16_sel_down_fma_into, qmatvec_nvfp4_bf16_ep_down_fma_into/_raw, qmatvec_nvfp4_q8_ep_down_fma_raw, axpy_rows_seq_md_off_into and moe_w_exscale; a repo-wide search (crates, tools, tests) finds no caller of any of those Rust functions. qmatvec_nvfp4_bf16_ep_dual_pairs (10598) has no reference at all.
- check: A word search over crates, tools and probe (excluding .cu and .md) finds only the definitions of moe_w_exscale, qmatvec_nvfp4_bf16_sel_down_fma_into, qmatvec_nvfp4_bf16_ep_down_fma_into/_raw, qmatvec_nvfp4_q8_ep_down_fma_raw and axpy_rows_seq_md_off_into, and no call sites. moe_router_topk_scaled's doc says it replaces moe_w_exscale. qmatvec_nvfp4_bf16_ep_dual_pairs appears only at its own definition.
- fix: Delete the six kernels, their Rust launchers and the KERNELS.md mention of moe_w_exscale.

### low: Spill-free compile-time-T tcol twins exist but are never launched; Rust uses the spilling runtime-t kernels

- where: `crates/memra-engine/cu/qmatvec.cu:11627` (S14, confirmed)
- detail: The comment records that runtime-t matvec_bf16_b4_tcol spills accumulators to local memory (283us vs 33us at t=8) and that the _t2/_t4/_t8 twins (11602-11707) fix it bit-identically. No Rust, test or doc names matvec_bf16_qkvg_tcol_t*, matvec_bf16_b4_tcol_t* or matvec_bf16_f32acc_tcol; lib.rs launches only the runtime-t matvec_bf16_qkvg_tcol (25330) and matvec_bf16_b4_tcol (25492) on the step TP decode-v2 walk. Either a measured win was never wired or the twins are leftovers of a closed lane.
- check: The kernels really are never launched: no Rust, tool or test names matvec_bf16_qkvg_tcol_t*, matvec_bf16_b4_tcol_t* or matvec_bf16_f32acc_tcol. But the report's framing is wrong. The twins were not a win someone forgot to wire. lib.rs says they are deliberately not dispatched: switching kernels by live batch width is a load-history change of numeric program, and a Hermes review verdict marks them research-only. Since no research tool launches them either, they are dead weight in the fatbin. Lowered to low.
- fix: Delete the six _t{2,4,8} twins and matvec_bf16_f32acc_tcol (no research harness launches them), or add the harness that uses them as controls.

### low: matvec_bf16_dual_silu_ilv and its door comment outlived MEMRA_DUAL_ILV

- where: `crates/memra-engine/cu/qmatvec.cu:11766` (S14, confirmed)
- detail: No env read of MEMRA_DUAL_ILV and no FLAGS.md row exist; the kernel has no launcher. The comment still documents a door that was removed.
- check: MEMRA_DUAL_ILV and matvec_bf16_dual_silu_ilv appear only in qmatvec.cu, at the comment and the kernel. Nothing reads the env var, nothing launches the kernel, and FLAGS.md has neither a row nor a removed-door ledger entry for it.
- fix: Delete the kernel and its comment and record the verdict in the FLAGS.md removed-doors ledger.

### low: Overlap scheduler pending path is dead

- where: `crates/memra-engine/src/decode_batch.rs:1089` (S05, confirmed)
- detail: decode_step_batch_sampled_lean_masked_pending, decode_step_overlap_eligible, PendingBatchStep::new/wait and the pending_out branches in decode_batch_epilogue (4805, 4907-4925) have no caller in memra-server, the engine or bins. The pending_out parameter on the shared schedule is always None in production.
- cut: Removing it shrinks the shared epilogue branch set that every batched family's cell depends on.
- check: Outside decode_batch.rs, decode_step_batch_sampled_lean_masked_pending, decode_step_overlap_eligible and PendingBatchStep appear only in a research RECEIPTS.md. The only Some(&mut pending) passed into the schedule comes from the dead entry itself, so pending_out is always None in production.
- fix: Delete the pending entry, PendingBatchStep, decode_step_overlap_eligible and the pending_out parameter and branches.

### low: DsparkHarvest::resolve is dead and its doc claims to be the served resolution

- where: `crates/memra-engine/src/dflash.rs:666` (S11, confirmed)
- detail: No caller exists in any crate. Served code resolves through for_draft (711), which adds the DFlash2 family key. A reader trusting the doc at 658 would edit the wrong function.
- check: A repo-wide search finds no caller of DsparkHarvest::resolve. Served code uses for_draft. The 'served resolution' doc on resolve, and from_env_value's note that round arms resolve through Self::resolve, are both stale.
- fix: Delete resolve, move its doc to for_draft, and fix from_env_value's reference.

### low: Column-partition and wo_b partial-sum machinery has no production caller

- where: `crates/memra-engine/src/dsv4_attention_split.rs:368` (S07, confirmed)
- detail: Production builds only Partition::Rows (dsv4_gpu.rs:4483-4501), and dsv4_attention_tp.rs documents an exact row gather with no cross-rank sum. The Columns arms of from_device (L368, L407), WoBPartialSumPlan, wo_b_rank_order_sum_f32, PackedFp8Host (rows/columns/half_*), ExistingGemvHalfPlan and DeviceFp8Plane::upload are reached only from tests. The module doc (L3) still claims the module selects no runtime path.
- check: The only production user is dsv4_gpu.rs, which imports DeviceFp8Plane and Partition and builds wq_b, wo_a and wo_b all with Partition::Rows through from_device. No file outside dsv4_attention_split.rs references WoBPartialSumPlan, wo_b_rank_order_sum_f32, PackedFp8Host, ExistingGemvHalfPlan or Partition::Columns. The module doc says it 'does not select a runtime path' and that wo_b uses column partitions, but production does use it and partitions wo_b by rows.
- fix: Move the Columns variant, WoBPartialSumPlan, the rank-order sum and the host packers under #[cfg(test)] (or delete them), and rewrite the module doc to say it is the TP2 attention row-plane loader.

### low: Ten uncalled accessors and gate getters

- where: `crates/memra-engine/src/dsv4_gpu.rs:1423` (S06, confirmed)
- detail: No caller in crates/, tools/, bins or tests for ep_device_scratch_bytes (1423), c4_device_scratch_bytes (1438), dsv4_phase_reset (2201), index_topk_radix_for_gate (2346), clear_grouped_m1_tc_for_gate (2591), c4_host_copy_elision_for_gate (2646), Dsv4Gpu::tp_ep_topology_for_gate (2693), set_indexer_score_for_gate (2990), set_matrix_ep_graph_for_gate (14877) or matrix_ep_graph_dispatches (14883).
- check: All ten names appear only at their definitions in dsv4_gpu.rs (word-boundary search of crates/ and tools/, and repo-wide outside research/), apart from FLAGS.md mentions. Nothing calls them.
- fix: Delete the ten accessors (the two matrix-EP graph ones together with S06:F1).

### low: trim_mtp_requested is computed and discarded

- where: `crates/memra-engine/src/hybrid.rs:4919` (S05, confirmed)
- detail: Lines 4906-4919 read MEMRA_FRSPEC_TRIM and full_prec only to throw the result away; the comment says the branch that used it was removed.
- check: trim_mtp_requested reads MEMRA_FRSPEC_TRIM and full_prec_enabled(), then is discarded with let _ =. The comment above it says the branch that used it was removed.
- fix: Delete the binding, its env read and the history comment.

### low: moe_ffn_pairs shared-expert t==1 and verify_t arms are unreachable, and their comment is stale

- where: `crates/memra-engine/src/hybrid_forward.rs:18272` (S04, confirmed)
- detail: moe_ffn_pairs is entered only when t > MOE_DEV_MAX_T (16), and PRIME_MIN_T is 16. So `step_exact && t == 1` and `t < PRIME_MIN_T` are always false: shexp_gate_up_t1, the spec_fused_t fused2 arm and matmul_decode_exact (18274-18292, 18298-18299) never run. The comment says these forms are required by the new Step resident arm, but step35 is denied entry to pairs by the sigmoid_router gate.
- cut: none
- check: The only caller (13753) requires t > MOE_DEV_MAX_T (16), and PRIME_MIN_T is 16. So t == 1 and t < PRIME_MIN_T can never hold, and the t1/verify_t arms are dead. The caller also requires sigmoid_router().is_none() and m3.is_none(), which shuts out step35 and M3. That makes both the 'Step resident arm' comment and the MiniMax-M3 comment stale.
- fix: Replace the epilogue with the plain matmul arm (or call moe_shexp_add) and drop the stale Step/M3 comments.

### low: DSV4 fused-MoE gate override is never called, though KERNELS.md documents it as the A/B knob

- where: `crates/memra-engine/src/lib.rs:573` (S01, confirmed)
- detail: set_dsv4_moe_fused_for_gate and clear_dsv4_moe_fused_for_gate have no caller, so DSV4_MOE_FUSED_OVERRIDE stays at -1 (fused on) forever. docs/KERNELS.md:704 says set_dsv4_moe_fused_for_gate(false) runs the chain, but no gate does that. The fused-vs-chain identity is only checked through dsv4_grouped's own unit tests.
- check: set_dsv4_moe_fused_for_gate and clear_dsv4_moe_fused_for_gate have no reference anywhere outside their definitions, so DSV4_MOE_FUSED_OVERRIDE stays -1 and fused is always on. KERNELS.md:704 and the lib.rs comment present the override as the way a gate runs the unfused chain, but no gate uses it.
- fix: Have the dsv4 gate run a fused-vs-chain arm through the override, or delete the pair and the KERNELS.md sentence.

### low: Fourteen Engine launchers have no caller

- where: `crates/memra-engine/src/lib.rs:13253` (S01, confirmed)
- detail: No caller in any crate, test or bin: nvfp4_ep_stage_inputs_raw (L8146), i32_add_k (L10343), moe_w_exscale (L10487), moe_pairs_weighted_scatter (L11521), qmatvec_nvfp4_fast_prequant_into (L12109), qmatvec_nvfp4_bf16_sel_down_fma_into (L13128), qmatvec_nvfp4_bf16_ep_down_fma_into (L13187), qmatvec_nvfp4_bf16_ep_down_fma_raw (L13253), qmatvec_nvfp4_q8_ep_down_fma_raw (L13385), dtoh_f32_into_pinned (L13710), htod_f32_from_pinned_async (L13744), rms_norm3 (L14919), row_softmax (L15747), add_q8_1 (L17259). The kernels behind them (for example i32_add_k in kernels.cu:1940 and the *_ep_down_fma kernels) stay in the fatbins and in kernel census tooling.
- check: A search of the whole repo (crates incl. tests/bins, tools, docs, research) finds no callers for any of the 14 launchers. The only hits are their definitions, their own func("...") kernel-name strings, the bounds-check messages inside them, and doc-comment mentions. That makes it dead code, a maintenance and kernel-census hazard but no runtime fault.
- fix: Delete the 14 launchers and the kernels only they reach, and update the KERNELS.md rows.

### low: B200 MoE v2 twins exist only for a bench and are never dispatched

- where: `crates/memra-engine/src/lib.rs:23415` (S02, confirmed)
- detail: moe_gate_up_preclamp8_q8_v2 and moe_down8_fma_q8_v2 are called only by src/bin/b200_matvec_bench.rs. FLAGS.md:595 states they are built and benched but NOT dispatched, which makes them a door with no dispatch branch.
- check: The only callers are in src/bin/b200_matvec_bench.rs. FLAGS.md (line 693, not 595) records the receipt: gate/up is inside noise, down regressed 20%, and the kernels stay in the fatbin and the bench as measured arms. AGENTS.md says a neutral or negative receipt deletes the door and every kernel only it reaches.
- fix: Delete both launchers, their kernels and the bench arms; leave the verdict in the Removed doors ledger.

### low: Dead launchers and the kernels only they reach

- where: `crates/memra-engine/src/lib.rs:35623` (S02, confirmed)
- detail: No caller anywhere in crates/ for: transpose, repeat_heads, conv_left_pad, add_rms_norm_zq8 (refuted fuse, decode.rs:630), gated_rmsnorm_f16out, ssm_conv1d_tm, ssm_conv1d_view, gdn_scan_s128_view, axpy_rows_seq_md_off_into, matmul_group_multi, qmatvec_nvfp4_batched_raw, bf16_to_f16, addr_f32_mut, slice_range, matvec_bf16_dual. Per kernels.tsv, transpose_f32, repeat_heads_f32, conv_left_pad_f32, ssm_conv1d_tm_f32, add_rms_norm_zq8 and axpy_rows_seq_md_off_f32 are launched only from these dead functions.
- check: A search of all .rs files found no call to any of the 15 Engine methods. The transpose( hits in tests are a local host helper, and the remaining .transpose() hits are Option::transpose. The matvec_bf16_dual hits are kernel names in other launchers; the fn itself carries #[allow(dead_code)] as a kept reference. The six kernels named in the item have func("...") lookups only inside these dead functions.
- fix: Delete the 15 launchers, the six kernels only they reach, and their KERNELS.md rows.

### low: Unreferenced wrappers in the slice

- where: `crates/memra-engine/src/mla_ffi.rs:2537` (S13, confirmed)
- detail: These have no caller in any crate, bin or test: mla_dsa_attn_warp_online (mla_ffi 2537), Engine::kda_gate (kda 1427), quantize_fp4_act_cutlass (cutlass_ffi 237), a4_prefill_launches and a4_stats_armed (mmq_ffi 3178, 3248), moe_sel_dump::flush, gemma_prep_data_uri and glm5_prep_data_uri, and fp8_ffi::blk_native_nan_refused. Several more are test-only; see candidate_verdicts.
- check: This is only partly correct. Repo-wide search finds no caller for quantize_fp4_act_cutlass, a4_prefill_launches, a4_stats_armed, gemma_prep_data_uri, glm5_prep_data_uri and fp8_ffi::blk_native_nan_refused. Three of the named items do have callers. mla_dsa_attn_warp_online is called from tests/mla_kpool_live_gpu.rs:167,421. Engine::kda_gate is the reference in tests/kda_small_folds_gpu.rs:84. moe_sel_dump::flush is called by its own unit test at L263 and is the documented flush-now API. A4_PREFILL_LAUNCHES also has a live reader, a4_prefill_launches_reset (used by the qwen_a4_dispatch_gate bin), so the counter must stay.
- fix: Delete only the six uncalled functions; keep mla_dsa_attn_warp_online, kda_gate, moe_sel_dump::flush and the A4_PREFILL_LAUNCHES counter.

### low: Unused GGUF-era wrappers in model.rs, hybrid.rs and decode.rs

- where: `crates/memra-engine/src/model.rs:808` (S05, confirmed)
- detail: GpuTensor::scale, GpuTensor::load (826), EmbedHost::from_gguf (1567), HostExps::load (2105), HybridModel::step_tp_rank_engine (hybrid.rs:4148), has_plan_operation (hybrid.rs:4187), GraphSession::upload_mask (decode.rs:105), decode_step_dc_cap (decode.rs:1930) and graph_session_from_cache (decode.rs:2315) have no callers repo-wide.
- check: No caller in any crate, bin or test for GpuTensor::scale (every .scale() call site is on MlaDims or the MLA config), GpuTensor::load, EmbedHost::from_gguf, HostExps::load (the loaders use load_tiered, load_from_source and load_stacked_from_source), step_tp_rank_engine, has_plan_operation, upload_mask, decode_step_dc_cap or graph_session_from_cache. The only hits are definitions, their own error strings, and docs.
- fix: Delete the nine wrappers, and the docs that cite graph_session_from_cache and upload_mask.

### low: GroupedParallelPlan and several ParallelPlan range accessors are test-only

- where: `crates/memra-engine/src/parallel.rs:874` (S05, confirmed)
- detail: plan_grouped and every GroupedParallelPlan method, plus ParallelPlan::global_rank, query_feature_range, kv_feature_range, dense_ffn_range, routed_expert_range and routed_expert_ffn_range, are called only from #[cfg(test)].
- check: Every use of plan_grouped, GroupedParallelPlan and the named range accessors (global_rank, query/kv_feature_range, dense_ffn_range, routed_expert_range, routed_expert_ffn_range) outside their definitions is in parallel.rs's #[cfg(test)] mod tests (line 1556 onward). No other crate, bin or test refers to them.
- fix: Move plan_grouped, GroupedParallelPlan and the range accessors under cfg(test), or delete them with their tests.

### low: qsa_indexer_mask is dead and its justification comment is false

- where: `crates/memra-engine/src/qwen4exp_gpu.rs:18389` (S08, confirmed)
- detail: No caller exists in the engine, bins or tests; the #[allow(dead_code)] comment says in-flight lanes still call it. Same for the unused pub helpers kv_hoist_is_on (L2306), pool_t_is_on (L2337), MtpDraftState::rows (L13898), mtp_on_dev1 (L13922), spec_disarm (L14890), and the runtime-unused LayerPlacement::is_even (L2506, tests only).
- check: A repo-wide search (excluding research/) finds no caller outside the definitions for qsa_indexer_mask, kv_hoist_is_on, pool_t_is_on, mtp_on_dev1 or spec_disarm. MtpDraftState has no external user and no .rows() call on it. is_even is used only in the #[cfg(test)] block. The 'in-flight lanes still call' comment is not true on main.
- fix: Delete the unused helpers (keep is_even under cfg(test) or with its tests), or wire kv_hoist_is_on/pool_t_is_on into the receipt headers their docs name.

### low: Uncalled public spec wrappers and trait method

- where: `crates/memra-engine/src/spec.rs:10589` (S10, confirmed)
- detail: generate_spec_session_constrained (10589) and spec_session_from_restored (9393) have no callers in any crate, bin or test. SpecConstraint::mask_words (784) forces every implementor to provide a method that production code never calls; only constrained.rs tests (1486, 1498) use it.
- check: Word-exact searches across the engine, server, cli, bins and tests find no call to generate_spec_session_constrained or spec_session_from_restored. Only comments and docs/SERVING.md mention them; the worker uses spec_session_from_restored_ready/_deferred. SpecConstraint::mask_words is called only in constrained.rs tests (1486, 1498); the worker builds device masks from its own words.
- fix: Delete the two wrappers (update docs/SERVING.md:145 to the _ready/_deferred names) and move mask_words off the trait to an inherent test helper.

### low: SpillBlock and Tiered are unused structural aliases

- where: `crates/memra-engine/src/spill.rs:203` (S12, confirmed)
- detail: Nothing in the workspace constructs or names SpillBlock or spill::Tiered; the doc at spill.rs:219-223 calls Tiered 'a documentation/structural alias, not a new hot path'.
- check: A search of every .rs file in the repo finds SpillBlock and Tiered only at their definitions in spill.rs: no constructor, no type use, no re-export (pub mod spill exports only the module). The doc itself calls Tiered a documentation alias.
- fix: Delete SpillBlock, its bytes() impl and Tiered (spill.rs 199-227).

### low: Unreferenced tp.rs entry points

- where: `crates/memra-engine/src/tp.rs:3836` (S09, confirmed)
- detail: No caller in src, bins or tests for: upload_bf16_column_parallel (3057), upload_bf16_row_parallel (3836), bf16_row_parallel_resident (3869), upload_expert_parallel_nvfp4 (10792), Nvfp4DeviceRoutesWorkspace::out_stage_mut (9932), graph_section_absorbing/graph_section_detached (15877/15886), PreparedPeerWeightedRouteCombine::tokens/pairs (1641/1645). E4m3ExpertBank::expert (1226) is test-only. Each widens the file's surface, and therefore its impact hash, for nothing.
- check: upload_bf16_column_parallel, upload_bf16_row_parallel, bf16_row_parallel_resident, out_stage_mut, upload_expert_parallel_nvfp4, graph_section_absorbing and graph_section_detached appear only at their definitions. PreparedPeerWeightedRouteCombine::tokens/pairs have no caller: the gate's prepared.pairs() is on PreparedStepGroupedExpertParallelGate. The only E4m3ExpertBank::expert call found is in the tp.rs test block.
- fix: Delete the unreferenced functions and move E4m3ExpertBank::expert under cfg(test).

### low: ArLink broadcast/all_gather/ranks and glm5 ar_1stage_into have no production caller

- where: `crates/memra-engine/src/tp_ar.rs:432` (S09, confirmed)
- detail: broadcast and all_gather are exercised only by tests/tp_ar_gpu.rs. ArLink::ranks (405), ArPhaseRecord::nanos (272, unit test only), Glm5TpRt::ar_1stage_into (glm5_tp.rs:344), Glm5TpKda::ranks (glm5_tp.rs:1050), glm5_ep_diet_fanout_uploads_avoided (glm5_tp.rs:1697) and tp_transport::move_i32 (746) are unreferenced. The tp_ar module doc still says broadcast/all_gather matter most for the walk.
- check: ArLink::broadcast and all_gather are called only from tests/tp_ar_gpu.rs, and ArPhaseRecord::nanos only from the tp_ar unit test. ar_1stage_into, glm5_ep_diet_fanout_uploads_avoided and move_i32 have no callers. No ArLink or Glm5TpKda receiver appears among the .ranks() call sites (those are on the runtime, the hop and MLA TP). The module doc still says broadcast and all_gather matter more than the reduce for the walk.
- fix: Delete the unreferenced items, mark broadcast/all_gather test-only (or delete them with their GPU test), and fix the tp_ar module doc.

### low: n_layer_total is unread and disagrees between GGUF and HF

- where: `crates/memra-gguf/src/config.rs:1765` (S20, confirmed)
- detail: GGUF block_count already includes NextN blocks, so from_gguf double-counts (block_count + nextn), while from_hf gives num_hidden_layers + nextn (config.rs:2595). Its doc says 'includes appended MTP layers'. No production code reads it (only tests and the source.rs:1381 write).
- cut: none
- check: The only reads are test asserts. Production code only writes the field (config.rs, source.rs:1381, parallel.rs:1712). from_gguf sets n_layer = block_count, which already includes the NextN blocks, then n_layer_total = n_layer + nextn, so NextN is counted twice. from_hf sets n_layer = num_hidden_layers + nextn and n_layer_total to the same value, so the two loaders disagree.
- fix: Delete the field and its writes and asserts, or set it to n_layer in from_gguf.

### low: Dead cfg-only sizing wrappers re-compile a ModelPlan

- where: `crates/memra-kv/src/lib.rs:2419` (S21, confirmed)
- detail: cache_bytes_per_token, cache_bytes_per_token_for_layers, cache_ring_bytes_per_token(_for_layers), cache_ring_row_cap and Cache::new_pp2 have no callers. The sizing wrappers recompile a ModelPlan from cfg (lib.rs:2420, 2528, 2569). A future caller would size admission from a plan that can differ from the loaded one (rewrite bundles, packs), the mismatch the _for_plan variants exist to avoid.
- check: A workspace-wide search for cache_bytes_per_token, cache_bytes_per_token_for_layers, cache_ring_bytes_per_token(_for_layers), cache_ring_row_cap and new_pp2 finds only their definitions and doc cross-references in memra-kv/src/lib.rs. The remaining mentions are research notes. No re-export or other crate uses them. Each wrapper recompiles a ModelPlan from cfg (ModelPlan::compile(cfg)) instead of taking the loaded plan, which is the hazard described.
- fix: Delete the cfg-only wrappers and Cache::new_pp2, keep the _for_plan forms, and update the doc references that point at the deleted names (lib.rs:2318, 2516, 2522, 3016).

### low: tiered Scheduler/Hierarchy/ActiveEpoch/policy/integration have no installed caller

- where: `crates/memra-kv/src/tiered/scheduler.rs:45` (S21, confirmed)
- detail: About 1,000 lines (scheduler.rs, policy.rs, integration.rs, Hierarchy and ActiveEpoch in mod.rs, SealedImage in hostprefix.rs) are reached only from tiered/tests.rs and day4/day5 tests. They compile into the server binary but no binary constructs them.
- check: Outside memra-kv/src/tiered/, nothing names Hierarchy, ActiveEpoch, the tiered Scheduler, the policy functions (write_action, prefetch_decision, eviction_victim), reserve_peer_source, KvBacking or SealedImage. The only PrefetchPolicy hit is its definition in memra-tier contracts. The day4 and day5 suites are mounted only from tiered/tests.rs, which is cfg(test). The server uses only hostprefix's governor, charge and identity types plus the re-exported contracts. The roughly 1,000-line estimate is consistent with the file sizes (scheduler.rs 312, policy.rs 188, integration.rs 110, plus parts of mod.rs and hostprefix.rs).
- fix: Put scheduler, policy, integration, Hierarchy, ActiveEpoch and SealedImage behind cfg(test) or a feature until a serving path uses them, or delete them if the lane is closed.

### low: memra-probe Phase-0 spike is unreferenced and hardcodes sm_120 (12,0)

- where: `crates/memra-probe/src/main.rs:21` (S21, confirmed)
- detail: No crate depends on memra-probe. Its assert fails on GB10 (12.1), which the engine itself admits for 120a.
- check: The facts hold. No crate depends on memra-probe. It hardcodes assert_eq!((cc_major, cc_minor), (12, 0)), and the engine's arch check admits ('120a', 12, 1), so the probe panics on a 12.1 GB10 device that the engine accepts. Calling it 'dead' overstates it: it is a deliberate, documented standalone dev binary (README: 'Unpublished dev spike'; CONTRIBUTING: 'Standalone hardware microbenches'). It is still a workspace member, so it runs nvcc in workspace builds, and tools/validation_plan.py lists it as NATIVE. It costs build time and gates nothing at runtime.
- fix: Delete the crate, or move it out of the workspace into research/. If it stays, accept cc_minor 0 or 1 and fix the CONTRIBUTING description so it matches the README.

### low: kv_hat survives only as a test oracle under allow(dead_code)

- where: `crates/memra-server/src/admit_predict.rs:606` (S19, confirmed)
- detail: Production uses kv_hat_ring; kv_hat has no non-test caller and is kept alive with #[allow(dead_code)].
- check: kv_hat carries #[allow(dead_code)]. Its only callers are in the tests module (from line 753), at lines 1018, 1020 and 1146. Nothing outside tests calls it. The rest of the 'kv_hat' hits in crates are field names (shadow_kv_hat, request_kv_hat), not calls.
- fix: Move kv_hat into mod tests as a #[cfg(test)] helper and drop the allow(dead_code).

### low: Test-only PrefixCache seams compiled into production, and a stale allow on touch

- where: `crates/memra-server/src/worker.rs:9189` (S17, confirmed)
- detail: PrefixCache::insert (9190) and insert_with_budget (9250) are test-only, yet they are compiled into release under allow(dead_code). PrefixCache::touch carries the same attribute (9014) although production calls it through touch_exact_without_credit (8989), so that allow hides nothing and misleads.
- cut: none
- check: PrefixCache::insert and insert_with_budget are called only from #[cfg(test)] code (worker.rs tests, and host_glm.rs inside its mod tests starting at line 499). They are compiled into release under cfg_attr(not(test), allow(dead_code)). touch has the same attribute even though production calls it: touch_exact_without_credit -> touch at 8989, called from non-test code at 37094. So that allow hides nothing.
- fix: Gate insert and insert_with_budget with #[cfg(test)] and drop the allow attribute on touch.

### low: evaluate_decoupled_admission and EvaluatedAdmissionOutcome are test-only

- where: `crates/memra-server/src/worker.rs:25973` (S18, confirmed)
- detail: No production caller exists (private module, #[allow(dead_code)]). The memra#187 test claims to drive 'the production' predicate (49093), so it pins logic the admission loop never runs.
- check: evaluate_decoupled_admission and EvaluatedAdmissionOutcome are referenced only by the test module (the import at 42390 and calls at 49126-49216). They carry #[allow(dead_code)], and no other crate or tool names them. The test doc calls it 'the production' predicate, which it is not.
- fix: Delete the fn, the enum and the memra#187 test, or route the live admission decision through the fn.

## Unreachable paths

### medium: Gemma-4 E4B path (~880 lines here, plus is_gemma4_e4b branches in 23 callers) cannot be reached

- where: `crates/memra-engine/src/hybrid_forward.rs:29348` (S04, confirmed)
- detail: Every load goes through compile_for_source (hybrid.rs:4245). Both gemma packs admit a config only if n_embd_per_layer == 0 && shared_kv_layers == 0, so an E4B artifact refuses with NoMatchingModelPack. gemma4_aux.e4b is built only when per_layer_token_embd.weight is present. A config that passes the pack check with that tensor present would fail the tensor-census contract as an unexpected tensor. So is_gemma4_e4b() is false on every loaded model. That makes gemma4_e4b_* (29338-30218) dead, along with every is_gemma4_e4b branch in worker::run/admit/eager_only_model, spec.rs and decode.rs.
- cut: removes x:gemma4_e4b cells and ~880 lines from the gemma impact set; worker branches stop dragging E4B into gemma4 admission
- check: Every HybridModel load goes through load_from_source_impl, which calls compile_for_source. Both gemma packs match only if n_embd_per_layer == 0 && shared_kv_layers == 0. If the metadata were missing (unwrap_or(0)) while the tensor was present, bind_source would refuse the unexpected per_layer_token_embd tensor. So e4b is never Some. Worse, tools/fast-gate/models.tsv still registers an 'e4b' argmax cell, described as 'the only arm that reaches Gemma-4's per-layer embeddings'. Its golden dates from 2026-08-02, before the pack refusal landed on 2026-09-01. Where the model file exists, that cell now fails at load.
- fix: Add an E4B pack with its tensor contract and qualify it, or delete the E4B path, the is_gemma4_e4b branches and the fast-gate e4b cell.

### low: Q4_0 b16 r2 twins can never be selected

- where: `crates/memra-engine/cu/qmatvec.cu:3498` (S14, confirmed)
- detail: qmatvec_mmvq_batched overrides the variant to base/rp whenever mcols==16 (lib.rs:23859), after any MEMRA_BVAR force, and Q4_0 has no other b16 call site, so qmatvec_q4_0_mmvq_b16_r2 (3498) and qmatvec_q4_0_mmvq_b16_r2_rp (9542) are compiled but unreachable.
- check: batched_kernel_name has a single caller, qmatvec_mmvq_batched. After the MEMRA_BVAR force and the auto pick, it overrides every mcols==16 variant to rp/base, and its comment says only base/_rp b16 kernels are compiled. No other code builds a q4_0 b16 name. So qmatvec_q4_0_mmvq_b16_r2 and _r2_rp cannot be selected.
- fix: Delete qmatvec_q4_0_mmvq_b16_r2 and qmatvec_q4_0_mmvq_b16_r2_rp, or let the mcols==16 rule admit r2 for Q4_0 if a receipt exists.

### low: Preview (ClampOnly) DeepSeek-V4 artifacts cannot load under the only serving program

- where: `crates/memra-engine/src/dsv4_gpu.rs:2946` (S06, confirmed)
- detail: dsv4_serve maps the Preview encoding to ActQuantVariant::ClampOnly. validate_matrix_program refuses any variant other than RefFp8Round, and the matrix program is the only one a serving process loads (3575). So a deepseek_v4_preview checkpoint fails at boot. The ClampOnly device arms (17890), the NextN MTP load (4245-4270) and mtp_logits_last are reachable only from gates. The family table still lists the preview alias as servable.
- check: dsv4_serve maps the Preview encoding to ClampOnly. The matrix program is the only one a serving process can load, because the reference program needs a gate arm. load_placed calls validate_matrix_program unconditionally, and it refuses any variant other than RefFp8Round. A Preview checkpoint therefore fails at boot with a generic matrix-program error, even though the pack lists the deepseek_v4_preview alias.
- fix: Refuse Preview early in dsv4_serve::load with a named 'preview encoding is gate-only' error (or qualify the matrix program for ClampOnly) and say so in the pack and docs.

### low: E4B arms in the gemma spec loop cannot run

- where: `crates/memra-engine/src/gemma_spec.rs:752` (S11, confirmed)
- detail: generate_spec_gemma is reached only by gate bins, and no model pack matches an E4B config, so compile_for_load returns NoMatchingModelPack. The served burst also refuses E4B. The E4B prime and verify arms (752-771, 1374) are dead code.
- check: Both gemma packs match only configs with n_embd_per_layer == 0 && shared_kv_layers == 0. compile_for_load returns NoMatchingModelPack otherwise, and the load path calls compile_for_source first. An E4B checkpoint therefore never loads, so is_gemma4_e4b() is always false at runtime and the E4B arms in generate_spec_gemma are dead.
- fix: Remove the E4B arms from gemma_spec (they return with the E4B pack and its receipts).

### low: Always-false term in the qwen35 rowwise predicate

- where: `crates/memra-engine/src/spec.rs:6735` (S10, confirmed)
- detail: qwen35_verify_batch_layers has a single caller, which checks batched_serving_numeric_class() first (7865), so this disjunct is always false.
- check: qwen35_verify_batch_layers has one caller (7866), reached only after `if self.batched_serving_numeric_class()`. The `|| !self.batched_serving_numeric_class()` disjunct is therefore always false.
- fix: Drop the term, or turn it into a debug_assert at function entry.

### low: Generic verify_layers Mixer::Linear arm and step35 guard cannot be reached

- where: `crates/memra-engine/src/spec.rs:7984` (S10, confirmed)
- detail: Mixer::Linear is built only from AttentionPlan::GatedDeltaNet (hybrid.rs:438), and verify_layers returns into qwen35_verify_batch_layers whenever the plan has GatedDeltaNet (7865). It also returns earlier for SlidingGatedMoe (7855). So the Linear arm (7972-8036, including its per-column ckpt clones) and the 'Mixer::Full(_) if sliding_gated_moe_batch_program' guard (7915) are dead.
- check: Mixer::Linear is built only from AttentionPlan::GatedDeltaNet. batched_serving_numeric_class() is true when trunk_operations contains GatedDeltaNet, and verify_layers returns into qwen35_verify_batch_layers in that case, after first returning into step35_verify_batch_layers for SlidingGatedMoe. The generic loop therefore never sees a Linear mixer, and the 'Mixer::Full(_) if sliding_gated_moe_batch_program()' guard is also dead.
- fix: Replace the generic Linear arm with a typed refusal/unreachable and drop the sliding_gated guard.

### low: step35_verify has no serving path

- where: `crates/memra-engine/src/spec.rs:8966` (S10, confirmed)
- detail: For step35, verify_layers returns into step35_verify_batch_layers at 7863 before full_attn_verify. The only other route into full_attn_verify's sliding-gated arm is decode_step_t_aux2, which is reached only from the eagle lane and the probe bins (verify_probe, t2probe, verify_shape_bisect). So step35_verify (8880-8933) is probe-only code that still shapes the shared full_attn_verify.
- check: For step35, verify_layers returns into step35_verify_batch_layers (7863) before any full_attn_verify call. The other full_attn_verify caller (8736) is inside decode_step_t_aux2, which is called only by eagle.rs (an EAGLE3 qwen35-9b lane with no worker reference) and the probe bins verify_probe, t2probe and verify_shape_bisect. step35_verify is reachable from probes only.
- fix: Delete step35_verify and its full_attn_verify arm, and make the step35 aux2 probe refuse (or route it through step35_verify_batch_layers).

### low: Dead else-arm in the rows_step TP/EP branch

- where: `crates/memra-server/src/dsv4_serve.rs:2074` (S07, confirmed)
- detail: Inside the TP/EP multi-row branch, `if !full` returns through decode_rows_draw at L2051-2068. The later `if full {...} else { decode_rows_greedy }` therefore never takes its else, and the Option wrapping of rows is always Some.
- check: In the TP/EP branch, `if !full { return ... decode_rows_draw ... }` exits before the later `if full {..} else { decode_rows_greedy }`, and `full` is not reassigned in between. So the else arm is dead and `rows` is always Some.
- fix: Bind `let (rows, am) = m.gpu.decode_rows_full(...)?;` directly and drop the Option wrapping.

### low: Cold spec drivers and monolithic prefill are unreachable on serving

- where: `crates/memra-server/src/dsv4_serve.rs:2995` (S07, confirmed)
- detail: The matrix expert program is always loaded on serving (matrix_expert_program_resolved is false only after the gate-only arm), and load refuses prefill_chunk 0 under matrix (L916). So short_monolithic (L2894) is always false and initial_logits is always Some. The non-restored spec_greedy_batched_stream / spec_sampled_batched_pen calls (L2994-3024), the plain prefill_with_cache arm (L3076-3081), the tokenwise restore arm (L2498-2508) and the non-chunked dspark_continue_prefix arm run only in a gate process. The module doc (L13) still names spec_greedy_batched_stream as the T=0 route.
- check: dsv4_serve is a private mod of memra-server. matrix_expert_program_resolved() is false only after arm_reference_expert_program_for_gate, which only gate bins call. The load refuses prefill_chunk==0 under matrix. So short_monolithic is always false and initial_logits is always Some on serving. That leaves the non-restored spec drivers, the plain prefill_with_cache arm and the tokenwise arm in continue_plain_prefix unreachable from the server. The module doc still names spec_greedy_batched_stream as the T=0 route.
- fix: Remove the unreachable server arms (or make them a refusing error) and update the L12-16 route table to the _restored drivers.

## Stale flags and doors

### medium: set_matrix_ep_graph_for_gate is past its decide-by date with no caller, so its graph arm is dead

- where: `crates/memra-engine/src/dsv4_gpu.rs:14877` (S06, confirmed)
- detail: The FLAGS.md row says OFF with decide-by 2026-09-21, and nothing calls this setter (or dsv4_ep_graph::set_for_gate). dsv4_ep_graph::enabled() is therefore always false. The whole graph_eligible arm in moe_verify_dev (20151-20234: capture, MatrixEpGraphSlot launch, builder) is unreachable, along with the matrix_ep_graph slot plumbing in verify_batch_dev_output (20965) and post_attention_moe_verify_dev, and the matrix_ep_graph_dispatches counter.
- cut: Takes about 80 lines of moe_verify_dev and the slot parameters out of the hot MoE verify path.
- check: The only caller of dsv4_ep_graph::set_for_gate is the uncalled set_matrix_ep_graph_for_gate, and the module has no test that sets the static. MATRIX_EP_GRAPH_FOR_GATE is therefore always false, and the graph_eligible arm in moe_verify_dev, with its capture, builder and slot launch, can never run. The matrix_ep_graphs slots are still allocated and threaded through the verify callers. The FLAGS row is OFF, decide-by 2026-09-21, unqualified.
- fix: Delete set_matrix_ep_graph_for_gate, dsv4_ep_graph's flag, counter and slot, the graph arm and matrix_ep_graph parameters in moe_verify_dev and its callers, and the FLAGS row (add a Removed-doors entry).

### medium: MEMRA_TP_AR_1STAGE, MEMRA_GLM5_TP_SYMMETRIC, MEMRA_GLM5_TP_EXPERT_SPLIT are past decide-by 2026-09-20 with no model-scale receipt

- where: `crates/memra-engine/src/glm5_tp.rs:2105` (S09, confirmed)
- detail: FLAGS.md rows 213, 217 and 218 read 'DEFAULT OFF (2026-09-06, decide-by 2026-09-20), pending its model-scale row'. Today is 2026-11-03. Per the door doctrine, a door past decide-by without a receipt is removed with its dispatch branches (kda_tp_partials_sym, shard_moe_layer_split, the one-shot reduce sites) and its kernels.
- check: FLAGS.md rows 213, 217 and 218 still read DEFAULT OFF, decide-by 2026-09-20, with no later receipt. AGENTS.md says a door past its decide-by date without a receipt is removed with its env read, dispatch branches and kernels. All three env reads and their arms remain.
- fix: Land the model-scale receipts and decide, or delete MEMRA_TP_AR_1STAGE, MEMRA_GLM5_TP_SYMMETRIC and MEMRA_GLM5_TP_EXPERT_SPLIT with their arms and kernels, and record them in the Removed doors ledger.

### medium: MEMRA_GLM5_TP_SYM_GRAPH past decide-by 2026-09-21 (pending its pair row)

- where: `crates/memra-engine/src/glm5_tp_sym_graph.rs:57` (S09, confirmed)
- detail: FLAGS.md row 215 carries decide-by 2026-09-21. The door, the whole module and its MLA sub-door MEMRA_GLM5_TP_SYM_GRAPH_MLA are still present with no receipt.
- check: FLAGS.md row 215 has decide-by 2026-09-21 and says 'pending its pair row', with no receipt. The module, its env read, the MLA sub-door and the hybrid_forward call site are all still present. This also carries the unfixed S09:F0 parity defect.
- fix: Record a pair receipt and decide, or delete glm5_tp_sym_graph.rs, the hybrid_forward call and both FLAGS rows.

### medium: MEMRA_SPEC_STREAM door kept after a negative receipt

- where: `crates/memra-engine/src/spec.rs:338` (S10, confirmed)
- detail: FLAGS.md:1093 records MEASURED NEGATIVE 2026-07-10 (35B p2 -16%) and keeps the door as an 'experimental seam'. The door rules say a negative receipt deletes the door with every kernel only it reaches. This door alone keeps linear_attn_verify_t (8245-8491, whose other caller is unreachable), commit_verified_prefix_stream, the stream arms in generate_spec_inner2 (12614-12697, 12903-13014), the stream arm in qwen35_tparallel_fa_layer (7105) and the stream_ctr arms in full_attn_verify and the step35 refusals in the verify-funnel blast radius.
- cut: removes about 700 lines of default-unreached verify code from every qwen35, qwen35_moe and hy3 MTP cell
- check: The FLAGS.md row records MEASURED NEGATIVE 2026-07-10 and keeps the door 'as the experimental seam'. AGENTS.md says a negative receipt deletes the door in the same PR, with every kernel only it reaches. The door still gates d2t_dev allocation, stream_on in generate_spec_inner2, commit_verified_prefix_stream and the stream arm that calls linear_attn_verify_t (7116), plus refusal arms in verify_layers/full_attn_verify. That is a sizable maintenance surface kept alive by a refuted lever.
- fix: Delete MEMRA_SPEC_STREAM/_M, the stream arms, commit_verified_prefix_stream and the stream-only helpers. Move the row to the Removed doors ledger.

### medium: MEMRA_RANK0_MERGE has a negative receipt but its door and runtime redirect remain

- where: `crates/memra-runtime/src/lib.rs:226` (S21, confirmed)
- detail: docs/FLAGS.md:2354 records MEMRA_RANK0_MERGE as 'RECEIPTED NEGATIVE (-2.7)'. The doors doctrine says a negative receipt deletes the env read, dispatch branch and machinery in the same PR. The tp.rs:164 env read, the two decode.rs call sites (824, 1119) and memra-runtime's RANK0_REDIRECT thread-local, rank0_redirect_scope and the enter_main arm (lib.rs:81-118, 226) are all still live. The arm also adds a TLS lookup to every enter_main.
- check: FLAGS.md records MEMRA_RANK0_MERGE as 'RECEIPTED NEGATIVE (-2.7)' in the LIVE Step TP doors section. Door hygiene says a negative receipt deletes the door in the same PR. The env read (tp.rs:164), both decode.rs scopes (824, 1119), the RANK0_REDIRECT thread-local and the redirect lookup at the top of enter_main are all still present, so every enter_main pays the lookup.
- fix: Delete rank0_merge_on, both decode.rs scopes and RANK0_REDIRECT/set_rank0_redirect/rank0_redirect_scope plus the enter_main branch, and move the row to the Removed doors ledger.

### medium: WP-B resume measurement doors are past decide-by with no default decision

- where: `crates/memra-server/src/worker.rs:3588` (S17, confirmed)
- detail: MEMRA_RESUME_EXACT (decide-by 2026-10-10), MEMRA_RESUME_GRID_REWIND (2026-10-09), MEMRA_SPEC_BUDGET_CLAMP (2026-10-10) and MEMRA_KV_PARK_COMPACT (2026-10-06) are all still default OFF. Each keeps a full alternate resume program: exact_resume_plain/spec, grid_rewind_plain/spec, exact_settle_*, settle_one, compact_parked_plain_cache, grid_rewind_checkpoint_boundary.
- cut: Removes about 500 lines of default-OFF resume arms from the prefix_cache and mtp cells.
- check: All four rows are still default OFF with decide-by dates between 2026-10-06 and 2026-10-10 and no recorded decision. Two are labelled MEASUREMENT DOOR. Every cited alternate program exists in worker.rs.
- fix: For each door, flip it on with a receipt or delete the env read, its arms (exact_resume_*, grid_rewind_*, settle_one, compact_parked_plain_cache, etc.) and its tests, and log the verdict in the Removed doors ledger.

### medium: MEMRA_KV_HOST_CONTRACTS is past its decide-by (2026-10-05), still OFF, and gates about 11k lines

- where: `crates/memra-server/src/worker.rs:4921` (S17, confirmed)
- detail: The FLAGS.md row reads 'OFF (unset or 0), decide-by: 2026-10-05'. Today is 2026-11-03 and the door is still default OFF with no flip recorded. It gates HostTierContext, the off-tick capture, restore and promote routes and the three largest functions in the slice (host_kv_planes_settle_promote 503 lines, host_kv_planes_submit_promote 406, host_capture_submit 403).
- cut: Deleting it removes the kv_host_tier contract arms from every impact set; it already affects no default cell.
- check: The FLAGS.md row still reads OFF with decide-by 2026-10-05, and the parse still defaults to OFF. No decision or extension is recorded, and AGENTS.md says a door past decide-by without a receipt is removed. The row does name gates (kv-host-contract-fault-gate.sh, kv-host-spill-identity-gate.sh), but those gates test the door, which does not exempt it as a feature. The cited routes exist: HostTierContext, host_kv_planes_submit_promote, host_kv_planes_settle_promote and host_capture_submit.
- fix: Decide it: flip ON with an interleaved A/B receipt, or delete the door and the contract/off-tick code only it reaches, and record the verdict in the Removed doors ledger.

### medium: MEMRA_PREFIX_PARTIAL_RESTORE kept after a NO-GO receipt

- where: `crates/memra-server/src/worker.rs:6197` (S17, confirmed)
- detail: The lane returned NO-GO on 2026-08-13 (restored text byte-diverges from cold at splits 512 and 2048). The doctrine says a no-go receipt deletes the door in the same PR, yet the env read, partial_prefix_decision, PartialPrefixDecision, routed_moe_prefix_split and the admit-side mid-entry restore arm remain, labelled 'for measurement only'.
- cut: Removes a default-OFF numeric arm from the prefix_cache cell for every HybridModel family.
- check: The FLAGS row and the code doc both record a NO-GO receipt (output byte-diverges from cold at splits 512 and 2048) and keep the door 'for measurement only' until cx-eosclass. The 2026-09-05 doctrine says a no-go receipt deletes the door. The env read, partial_prefix_decision, PartialPrefixDecision and the two call sites (27304, 35359) remain. routed_moe_prefix_split is used only inside this door's arm in production (its other references are tests), so it goes with it.
- fix: Delete MEMRA_PREFIX_PARTIAL_RESTORE, partial_prefix_decision, PartialPrefixDecision, the admit-side mid-entry restore arms, routed_moe_prefix_split (if nothing else uses it) and their tests, and move the verdict to the Removed doors ledger.

### medium: Admission and OOM doors past decide-by: RECLAIM_OFFTICK, BATCH_OOM_RECOVER, ADMIT_W_RELEASE, KV_ALLOCATOR

- where: `crates/memra-server/src/worker.rs:15723` (S17, confirmed)
- detail: The FLAGS.md rows give decide-by 2026-10-09 (MEMRA_ADMIT_RECLAIM_OFFTICK), 2026-10-10 (MEMRA_BATCH_OOM_RECOVER, MEMRA_ADMIT_W_RELEASE) and 2026-10-04 (MEMRA_KV_ALLOCATOR, whose on-demand VMM helpers vmm_scope..vmm_reap_for live at 7315-7564). All four are still default OFF.
- cut: none
- check: All four rows are still default OFF, with decide-by dates of 2026-10-09, 2026-10-10, 2026-10-10 and 2026-10-04, and no recorded decision. The door functions and the VMM helpers (vmm_scope at 7338, vmm_reap_for at 7554) exist.
- fix: Decide each door: promote it with an A/B receipt, or delete the door and the code only it reaches, and update the ledger.

### medium: MEMRA_KV_ALLOCATOR=vmm door is a month past its decide-by date

- where: `docs/FLAGS.md:318` (S21, confirmed)
- detail: The on-demand VMM KV allocator (memra-kv plane.rs:247-983 mapper/graveyard/fault machinery, lib.rs:3142-3166, server kv_vmm.rs) is default-OFF with decide-by 2026-10-04. docs/decisions/KV-PHYSICAL-RECLAIM.md still says gate-only door, decide-by 2026-10-04. Today is 2026-11-03, and no promotion or removal was found.
- check: The MEMRA_KV_ALLOCATOR row and the decision record both say default OFF, decide-by 2026-10-04. The decision record's Decide-by section says promote with gates or delete the door. The worktree HEAD is dated 2026-11-02, git log since 2026-10-01 has no vmm/allocator decision commit, and the env read in memra-server kv_vmm.rs plus the memra-kv VMM machinery are still live.
- fix: Decide the door now: promote it with the interleaved receipt and serving-shape gates, or delete the vmm arm, its memra-kv/server machinery, the companion VMM_GROW/VMM_FAULT doors and the gate cells.

### medium: MEMRA_MOE_COLD_BYPASS is past decide-by with no recorded verdict

- where: `docs/FLAGS.md:450` (S12, confirmed)
- detail: The cold read-once bypass (moe_cache.rs ColdBypass/ColdGhost 105-187, serve_cold_bypass 2087-2135, dispatch_source_once/consumed, plus the spill_pread mapped buffers) is still in the spill hot path behind a default-off door. The row says the decision is the registered interleaved B3-shape cell on both rigs, and no verdict is recorded.
- cut: Deleting it removes the bypass arms from moe_cache.rs/spill_pread.rs, so their edits stop reaching every MoE spill cell.
- check: The FLAGS row is still 'off (default), decide-by: 2026-10-10' with no verdict. The research ledger has a partial result: on the PRO 6000, staged is flat and mapped loses, while the 5090 cell is 'queued'. By the row's own rule ('a flat or losing door is deleted whole') the evidence points toward deletion, but the door, ColdGhost and the mapped arm all remain.
- fix: Delete the mapped arm now per the PRO loser row, then run the 5090 cell or delete the whole door with ColdGhost and the FLAGS row.

### medium: MEMRA_KV_HOST_CONTRACTS is past its decide-by date with only N=1 executed-not-qualified receipts

- where: `docs/FLAGS.md:1462` (S12, confirmed)
- detail: The door gates the whole CudaTransfers engine (tier_transfer.rs, about 4300 non-test lines) and the worker's contract capture/restore/demote/promote paths. Its FLAGS row records one RTX PRO 6000, OFF and ON, N=1, executed-not-qualified. The decide-by date passed four weeks ago without a decision. AGENTS.md door hygiene says a door past decide-by without a receipt is removed.
- cut: Deleting it removes kv_host_tier contract cells from every family's impact set and shrinks tier_transfer.rs to the device-class helpers.
- check: The FLAGS row is OFF with decide-by 2026-10-05. Its only receipt is 'one RTX PRO 6000, OFF and ON, N=1, executed-not-qualified', and no October lane records a decision. AGENTS.md says a door past decide-by without a receipt is removed. The door gates a large surface: worker host_tier_context, CudaTransfers and the receipt kernels.
- fix: Run the decision cell on both rigs and promote, or delete the door with its CudaTransfers classes, gates and FLAGS rows.

### low: Build-time doors read in build.rs escape the flags census; MEMRA_DSV4_CUTLASS has no FLAGS.md row

- where: `crates/memra-engine/build.rs:756` (S22, confirmed)
- detail: tools/check-flags.sh scans only crates/*/src, so env reads in build.rs are not censused. MEMRA_DSV4_CUTLASS appears in docs/KERNELS.md but nowhere in docs/FLAGS.md, so there is no default, no decide-by date and no owner row. It is also absent from the generated boot-audit registry: it is a build env and never read at runtime, so the audit does not matter here, but the door-hygiene rule does. MEMRA_DSV4_FMAD is only mentioned in prose. It also has no rerun-if-env-changed outside its own branch: the line sits inside the if, so unsetting the var does not trigger a rebuild.
- check: check-flags.sh scans only crates/*/src and src/bin, so build.rs reads are not censused. MEMRA_DSV4_CUTLASS has no individual FLAGS.md row and no decide-by. The census passes only through the MEMRA_DSV4_* prefix scope note, which describes the remaining names as bench/gate instrument seams; a build switch that links a tensor-core dense path is not one. The rerun-if-env-changed line sits inside the if. One correction: unsetting after a set build does rerun, because the previous run emitted the directive. The broken direction is setting it after an unset build, which does not rerun.
- fix: Move rerun-if-env-changed=MEMRA_DSV4_CUTLASS above the if, add FLAGS.md rows with decide-by for MEMRA_DSV4_CUTLASS and MEMRA_DSV4_FMAD, and add build.rs files to the census scan.

### low: Full-token replay control component past decide-by 2026-09-22, outside every build

- where: `crates/memra-engine/cu/dsv4_replay_control.cuh:33` (S16, confirmed)
- detail: The header is included only by tools/dsv4-full-token-control-gate.cu. The lane receipt lists decide-by 2026-09-22 ('Remove with a negative/flat full-model verdict; do not promote from the fixture'). Nothing in the engine reaches it.
- check: Only tools/dsv4-full-token-control-gate.cu includes dsv4_replay_control.cuh. No engine .cu, Rust file or build.rs references it. The served full-token replay is implemented elsewhere in dsv4_gpu.rs. The FLAGS row is decide-by 2026-09-22 and records no full-model verdict. One extra reference: tools/test_public_ci_integration.py uses the header path as a CI-plan fixture, so deleting the header needs that test updated.
- fix: Delete the header, the tool, the FLAGS and KERNELS rows, and repoint the public_ci fixture at another header.

### low: MEMRA_SHEXP_SWIGLU_ZQ8 default-OFF door past its 2026-09-20 decide-by

- where: `crates/memra-engine/cu/hybrid.cu:1074` (S16, confirmed)
- detail: The FLAGS row says DEFAULT OFF (decide-by 2026-09-20) and names no box verdict. swiglu_preclamped_mul_scaled_q8_1_f32 and its dispatch branch in hybrid_forward.rs survive only behind it.
- check: FLAGS row still says DEFAULT OFF, decide-by 2026-09-20, and names a decision receipt (composed-posture c1 row) that is not recorded anywhere. The research tree only holds a copy of the FLAGS text. The kernel, the lib.rs door, the hybrid_forward branch, the swiglu_zq8_gpu test and the KERNELS.md row all survive behind the door. AGENTS.md:89 says a door past decide-by with no receipt is removed. The path is bit-identical by construction and default off, so this is maintenance, not wrong output.
- fix: Run the registered c1 row or delete the kernel, the lib.rs door and counter, the hybrid_forward branch, tests/swiglu_zq8_gpu.rs, and the FLAGS and KERNELS rows.

### low: set_tp_ep_topology_for_gate / set_attention_tp_for_gate are past decide-by 2026-10-07 and their docs are stale

- where: `crates/memra-engine/src/dsv4_gpu.rs:2687` (S06, confirmed)
- detail: The FLAGS rows (1360, 1364) say OFF, decide-by 2026-10-07, 'the server has no selector'. Since #710, TP/EP with attention TP is the served default through MEMRA_DSV4_TOPOLOGY, so these are no longer experimental doors. They are the only way gate bins select the served placement. The doc comment at 2687-2688 still says an armed request refuses before the PP loader.
- check: Both FLAGS rows say OFF, decide-by 2026-10-07, and 'The server has no selector'. That is no longer true: since #710 MEMRA_DSV4_TOPOLOGY defaults to tp_ep with attention TP. The doc comment at 2687-2688 says an armed request refuses before the PP loader, but load() now maps an armed request to TpEp. The 14 gates that need the served placement depend on these setters, so they cannot just be deleted.
- fix: Replace the statics with a placement argument on load (or keep them as gate placement selectors), fix the 2687 comment, and rewrite the FLAGS rows without the 'no selector' and decide-by text.

### low: MEMRA_DSV4_SAMPLER past decide-by 2026-10-08

- where: `crates/memra-engine/src/dsv4_sampler.rs:18` (S07, confirmed)
- detail: The FLAGS.md row carries decide-by 2026-10-08 and has no later receipt. The override seam is still read at load.
- check: The FLAGS row carries decide-by 2026-10-08, the override is still read, and I found no later receipt. The module header also still says 'CPU radix remains the oracle and default', which is stale because TP/EP defaults to the device sampler.
- fix: Remove the override (keep the placement default) or record the decision, and fix the module header.

### low: MEMRA_GLM5_TP_DEVICE_SAMPLE past decide-by 2026-09-22

- where: `crates/memra-engine/src/glm5_tp_sampler.rs:29` (S09, confirmed)
- detail: FLAGS.md row 1326 states decide-by: 2026-09-22 for this default-OFF door. No receipt flipped or removed it.
- check: FLAGS.md row 1326 states decide-by: 2026-09-22 for this default-OFF door, and the row has no later receipt or flip.
- fix: Decide with a receipt, or remove glm5_tp_sampler.rs, the worker arm and the FLAGS row.

### low: Three glm5 doors past their 2026-09-19 decide-by date

- where: `crates/memra-engine/src/glm_spec.rs:1454` (S11, confirmed)
- detail: MEMRA_GLM5_VERIFY_GRAPH (glm_spec.rs 1454/3611), MEMRA_GLM5_SPEC_DEV_IO (glm_spec.rs 1815) and MEMRA_GLM5_GRAPH_MLA_MID (glm5_decode_graph.rs 277) are still default-OFF doors. Their FLAGS.md rows say decide-by 2026-09-19, and today is 2026-11-03. Under the door-hygiene rule they should have been removed or decided with a receipt.
- check: All three env reads are still strict-'1' default-OFF doors in lib.rs. Their FLAGS rows say decide-by 2026-09-19 and give no extension or receipt, and today is 2026-11-03. Under AGENTS.md door hygiene they should already be decided or removed. This is a policy and maintenance issue with no wrong output.
- fix: Bank receipts and flip each door, or delete the env reads, arms, kernels, tests and FLAGS rows and log them in Removed doors.

### low: MEMRA_MOE_GROUPED research arm kept although FLAGS calls it blocked and unsafe

- where: `crates/memra-engine/src/hybrid_forward.rs:973` (S03, confirmed)
- detail: The FLAGS row says STILL BLOCKED and known UNSAFE after the 2026-08-13 PRO-pair re-sweep, with no decide-by. The arm (13187-13224) and moe_ffn_grouped remain reachable.
- check: The FLAGS row says 'STILL BLOCKED, and now known UNSAFE' (serve-smoke early-EOS failure) and has no decide-by. That is a no-go receipt, yet moe_grouped_enabled still reads the variable and opens the arm.
- fix: Delete the MEMRA_MOE_GROUPED door, its dispatch arm and moe_ffn_grouped (where nothing else reaches them), and record the verdict in the Removed-doors ledger.

### low: MEMRA_MOE_PREFETCH is past its decide-by date (2026-10-04)

- where: `crates/memra-engine/src/hybrid_forward.rs:1009` (S03, confirmed)
- detail: The FLAGS row is experimental, default OFF, decide-by 2026-10-04. Today is 2026-11-03 and no flip receipt is recorded.
- check: The FLAGS row carries decide-by 2026-10-04 and says the target card qualifies, pending 'the owner's call'. No flip or removal is recorded, and the read is still default OFF in code. AGENTS.md says a door past decide-by without a receipt is removed.
- fix: Get the owner's decision: flip it as the target-card default with its receipt, or delete the door and its prefetch paths.

### low: MEMRA_GLM5_TP_SYMMETRIC is past its decide-by date (2026-09-20)

- where: `crates/memra-engine/src/hybrid_forward.rs:2860` (S03, confirmed)
- detail: The FLAGS row says DEFAULT OFF, decide-by 2026-09-20, pending a model-scale row. The symmetric walk (3283-3783) and its routing here remain.
- check: The FLAGS row says DEFAULT OFF, decide-by 2026-09-20, 'pending its model-scale row', and nothing is recorded since. glm5_tp_gate sets the variable, but only to test the door itself, so the launcher/gate exemption does not apply.
- fix: Land the model-scale receipt and decide, or delete the symmetric walk, tp_glue replication and the door.

### low: MEMRA_PRIME_SEG kept after a neutral verdict, with no decide-by

- where: `crates/memra-engine/src/hybrid_forward.rs:6939` (S03, confirmed)
- detail: The code comment records the 2026-07-26 interleaved verdict as net -0.7% to neutral, and the FLAGS row lists it as an open hypothesis with no decide-by. The doctrine deletes neutral doors and requires a decide-by on default-OFF doors.
- check: The code comment records the 2026-07-26 interleaved verdict as net -0.7% to neutral. The FLAGS row calls it an 'open hypothesis' with no decide-by. Door hygiene deletes neutral doors and requires a decide-by on default-OFF doors.
- fix: Delete the MEMRA_PRIME_SEG capture arm and its read (also in qwen_prime_graph.rs:152), or give the row a decide-by and the named missing gate.

### low: MEMRA_SHEXP_DEV1 door still live after a receipted NEGATIVE verdict

- where: `crates/memra-engine/src/hybrid_forward.rs:12622` (S03, confirmed)
- detail: FLAGS.md marks it RECEIPTED NEGATIVE (~-5, structural) and keeps the replica machinery. The door doctrine requires a negative receipt to delete the env read, the branch and its workspace statics (SHEXP_D1_REPS/WS at 1809-1822).
- check: The FLAGS row reads 'RECEIPTED NEGATIVE (~-5, structural)', yet the env read and its dispatch branch are still live. AGENTS.md door hygiene says a negative receipt deletes the env read, the dispatch branch and every kernel only it reaches. No launcher or gate exemption applies.
- fix: Delete the MEMRA_SHEXP_DEV1 read, its branch and the SHEXP_D1 statics, and add a Removed-doors row.

### low: MEMRA_GLM5_TP_SPLIT_GROUPED_PRIME past its 2026-09-21 decide-by with the receipt still owed, but armed by a served launcher

- where: `crates/memra-engine/src/hybrid_forward.rs:16417` (S04, confirmed)
- detail: FLAGS.md:1026 says decide-by 2026-09-21 and 'Receipt owed'. FLAGS.md:1027 says the served 1M TP-2 launcher arms MEMRA_GLM5_TP_SPLIT_PRIME_HOSTDIET, which only acts inside this door's grouped prime. So the deployed configuration depends on an undecided default-off door, and naked commands do not get it. Today is 2026-11-03.
- cut: none until decided; on deletion removes 375 lines from glm5_next tp/prime cells
- check: The FLAGS row still says decide-by 2026-09-21 and 'Receipt owed'. The served launcher arms HOSTDIET, which acts only inside this door (hybrid_forward.rs:15533), so the launcher has to arm this door as well. Under the launcher exemption the door may stay (the memra#886 decision on HOSTDIET uses the same rule). What is left is stale row text and an unrecorded receipt, not a doctrine breach.
- fix: Update the row: record that the launcher sets the door (exemption), cite the tpwalk27/28 receipt, and drop the past decide-by/'Receipt owed' text.

### low: MEMRA_SHEXP_SWIGLU_ZQ8 is past its 2026-09-20 decide-by with no decision receipt

- where: `crates/memra-engine/src/hybrid_forward.rs:16938` (S04, confirmed)
- detail: FLAGS.md:960 says default OFF with decide-by 2026-09-20; the decision receipt (composed-posture c1 row) is not recorded. Under door doctrine the door, the swiglu_preclamped_mul_scaled_q8_1 kernel, its gate and the rows go unless a receipt lands.
- cut: removes the fused shexp arm from glm5_next decode cells
- check: The row says default OFF with decide-by 2026-09-20 and names a decision receipt that is still to be produced. No launcher or gate in the repo sets the flag (only the code, its test and its docs mention it).
- fix: Land the c1 A/B and promote, or delete the door, the kernel, tests/swiglu_zq8_gpu.rs and the FLAGS/KERNELS rows.

### low: MEMRA_GRAPH_DRAIN only admits values documented to crash

- where: `crates/memra-engine/src/hybrid_forward.rs:25230` (S04, confirmed)
- detail: DRAIN is pinned at 1. FLAGS.md:1056 says relaunching the same graph exec before its prior launch completes reproduced ILLEGAL_ADDRESS at chunk=4. The only behaviour the door adds is the known-bad multi-launch window, and the chunk>1 loop (25234-25250) plus the multi-token ring drain exist only for it.
- cut: trims the gemma graph route
- check: DRAIN is pinned at 1. Values 0 or 1 are the default behaviour, and any N>1 enables the multi-launch window that FLAGS.md documents as ILLEGAL (ILLEGAL_ADDRESS at chunk=4). The chunk loop exists only for N>1.
- fix: Delete the env read and the chunk>1 loop (chunk = 1); move the verdict to Removed doors.

### low: MEMRA_STEP_TP_GRAPH door survives a -20% re-measured receipt and carries a documented shadow-cache hazard

- where: `crates/memra-engine/src/hybrid_forward.rs:30855` (S04, confirmed)
- detail: FLAGS.md:2341: on the current stack graph replay measured 59-61 tok/s against eager 75.8-76.0 (-20%). The 'multi-token replay loop' it was kept for is run-gen only, and step35_token_graph_chunk is already removed. The door keeps step35_token_graph_step (290 lines), step35_token_graph_build (630 lines) and Step35TokenGraphState. Its own doc (30843) says local shadow caches advance length but not content under the door, so a session that saves or rolls back its cache (prefix-cache publish, spec rollback) would carry stale KV if the door were armed while serving.
- cut: removes ~950 lines from step35 tp/graph cells
- check: The door has a -20% re-measured receipt and no launcher or gate in the repo sets it, so the doctrine says delete it. Two qualifications. First, the same env read also gates the routed-prejoin graph arm in tp.rs:13689, so deleting the door removes that arm too. Second, the shadow-cache hazard is partly guarded: pp.rs refuses TP restore while the door is open (TokenGraphDoorOpen). Default OFF, so there is no live wrong output.
- fix: Delete MEMRA_STEP_TP_GRAPH, step35_token_graph_step/build, Step35TokenGraphState and the tp.rs routed-prejoin graph arm (or re-receipt that arm separately); record in Removed doors.

### low: Default-OFF KDA/MLA doors without decide-by dates

- where: `crates/memra-engine/src/kda.rs:979` (S13, confirmed)
- detail: MEMRA_KDA_NARROW_Q8 (2026-09-05, 'pending its model-scale row'), MEMRA_MLA_COALESCE (2026-09-03) and MEMRA_KDA_FUSED_PROJ (2026-08-30) are default OFF, and their FLAGS rows carry no decide-by date. The 14-day rule makes them overdue.
- check: All three are default-OFF env reads. Their FLAGS rows (1040, 954, 941) carry no decide-by date, while AGENTS.md requires one, 14 days after landing unless the row says why. MLA_COALESCE's row gives an open-ended A/B condition, not a date. The #140 commit message even records a serving gain (coalesce + split +1.9%), so it can be decided now.
- fix: Decide each door with a receipt (promote or delete), or add a dated decide-by to each FLAGS row.

### low: launch-collapse doors past decide-by 2026-09-20/21: SHEXP_SWIGLU_ZQ8, ROUTER_FUSED, HC_PRE_NORM_FUSE, GLM5_TP_SPLIT_GROUPED_PRIME

- where: `crates/memra-engine/src/lib.rs:630` (S01, confirmed)
- detail: docs/FLAGS.md:960, 214, 1035 and 1026 all still say DEFAULT OFF pending a row, with decide-by between 2026-09-20 and 2026-09-21. MEMRA_GLM5_TP_SPLIT_PRIME_HOSTDIET is armed in an external launcher but still default OFF in code. That case wants a decision (promote or delete), not a permanent door.
- check: SHEXP_SWIGLU_ZQ8, ROUTER_FUSED, HC_PRE_NORM_FUSE and GLM5_TP_SPLIT_GROUPED_PRIME are all still read, default OFF and past decide-by (2026-09-20/21) with no receipt. The HOSTDIET part is refuted: the FLAGS row says the served 1M TP-2 launcher arms it, and launcher-set flags are exempt under 'What stays'. It still deserves a decision, since it is default OFF in code while served ON.
- fix: Delete the four expired doors with their kernels and rows, or promote with receipts. Promote HOSTDIET to the code default, since the served launcher already arms it.

### low: glm5 graph/spec doors past decide-by 2026-09-19 with no receipt

- where: `crates/memra-engine/src/lib.rs:777` (S01, confirmed)
- detail: MEMRA_GLM5_SPEC_DEV_IO, MEMRA_GLM5_VERIFY_GRAPH and MEMRA_GLM5_GRAPH_MLA_MID are still DEFAULT OFF 'pending' in docs/FLAGS.md:950-952 with decide-by 2026-09-19. Today is 2026-11-03. Door hygiene says a door past decide-by with no receipt is removed. VERIFY_GRAPH also widens the verify-ws pool caps (L2175) and forces verify_ws_on (L3467).
- check: All three variables are still read in lib.rs. Their FLAGS rows say DEFAULT OFF, decide-by 2026-09-19, pending, and no receipt has landed. They are set only by their own door tests, so the gate exemption does not apply. AGENTS.md says a door past decide-by is removed.
- fix: Delete the three doors with their branches, kernels, tests and FLAGS rows, or land the receipts.

### low: MEMRA_ROUTER_FUSED default-OFF door past its 2026-09-20 decide-by

- where: `crates/memra-engine/src/lib.rs:1688` (S16, confirmed)
- detail: The FLAGS.md row still says 'box row pending' and decide-by 2026-09-20. Today is 2026-11-03. The door keeps memra_router_fused_f32/_dexp (moe_router.cu:724-885) and a branch in moe_split_rank_half alive with no receipt. Per the door doctrine, it should be deleted.
- check: The FLAGS.md row still says DEFAULT OFF, decide-by 2026-09-20, pending its box row. Today is 2026-11-03, and AGENTS.md removes a door past decide-by without a receipt. The door gates hybrid_forward.rs:15421 and the two fused kernels, with tests/router_fused_gpu.rs as its gate. It is a stale-door maintenance item with no served impact, so severity is lowered to match the same-class S10:F3.
- fix: Delete the door, memra_router_fused_f32/_dexp, ROUTER_FUSED_DISPATCHES, tests/router_fused_gpu.rs and the FLAGS row (ledger entry), or land the box receipt.

### low: verify-tier twins past decide-by 2026-09-19: MEMRA_TOPK_WARP, MEMRA_E4M3_BATCH_R2, MEMRA_Q80_BATCH_R2

- where: `crates/memra-engine/src/lib.rs:3041` (S01, confirmed)
- detail: All three are DEFAULT OFF with decide-by 2026-09-19 (docs/FLAGS.md:206, 405, 406) and no promotion or removal. Each keeps kernel twins (topk_rows_wshard_*, *_mmvq_b{2,4,8}_r2) and counters reachable only through the door.
- check: MEMRA_TOPK_WARP, MEMRA_E4M3_BATCH_R2 and MEMRA_Q80_BATCH_R2 are still read and still DEFAULT OFF with decide-by 2026-09-19. No promotion or removal is recorded. Only their own door tests set them.
- fix: Remove the three doors and the kernel twins only they reach, or land the box receipts and promote.

### low: MEMRA_Q80_BATCH_R2 and MEMRA_E4M3_BATCH_R2 are past their decide-by date (2026-09-19) and still in code

- where: `crates/memra-engine/src/lib.rs:23932` (S02, confirmed)
- detail: Both rows in docs/FLAGS.md (405-406) say DEFAULT OFF, decide-by 2026-09-19. Today is 2026-11-03 and both doors (23914, 23932) plus their _r2 kernels are still dispatched. Door doctrine says to remove a door whose decide-by passes without a receipt.
- check: Both rows say DEFAULT OFF, decide-by 2026-09-19, and both end with 'Box row owed', so no deciding receipt exists. Both dispatch arms are still in qmatvec_mmvq_batched. Under the AGENTS.md door rule they should have been removed. This is door hygiene with no wrong output, so I lowered it from medium to low.
- fix: Run the owed box rows and flip, or delete the env reads, the r2 arms, the _r2 kernels, their gates and the FLAGS rows.

### low: MEMRA_PRIME_ATTN_FA2 and MEMRA_PRIME_KV_T3 are past decide-by 2026-09-23 and still default OFF

- where: `crates/memra-engine/src/lib.rs:29867` (S02, confirmed)
- detail: FLAGS.md rows 31 and 63 carry decide-by 2026-09-23. KV_T3 has receipts but stays OFF pending another gate; FA2 has no flip. Both doors and their kernels (fa_prefill_qw_fa2*, fa_prefill_qw_t3*) remain.
- check: Both rows still read OFF with decide-by 2026-09-23. Both have positive receipts: FA2 says 'proposed enablement is owner-batched', and KV_T3 is held back by a separate accumulated-state failure. Because receipts exist, the delete-without-receipt rule does not strictly apply, but the decision is six weeks overdue and the doors are still live.
- fix: Get the owner decision: flip FA2 (and KV_T3 once its blocker clears) with the receipts, or delete them and log them under Removed doors; at minimum set a new justified decide-by date.

### low: MEMRA_KPOOL_SELECT_V2 is past its decide-by date with no verdict

- where: `crates/memra-engine/src/mla_ffi.rs:482` (S13, confirmed)
- detail: Default-OFF door with decide-by 2026-09-20 in both the code comment and its FLAGS.md row (959). Today is 2026-11-03, and no receipt promoted or removed it. Per AGENTS.md door hygiene it should be deleted with its kernels, or decided.
- cut: glm5_next/glm_dsa k-pool select variants
- check: The code doc and FLAGS.md:959 both say default OFF with decide-by 2026-09-20. The row names a decision receipt that would come from a box run, but no verdict was ever recorded. AGENTS.md: 'A door whose decide-by date passes without a receipt is removed'. This is policy debt with no behavior impact, so I rate it low.
- fix: Run the selab receipt or delete the door, the v2 kernels, tests/mla_kpool_select_v2_gpu.rs and the rows, and log it under Removed doors.

### low: MEMRA_GLM5_SPEC_DEV_IO is past its decide-by date and leaks into the qwen, step and hy3 MTP head

- where: `crates/memra-engine/src/spec.rs:3838` (S10, confirmed)
- detail: FLAGS.md:951 gives decide-by 2026-09-19, pending a box row; today is 2026-11-03. The door is named for glm5 DFlash2, yet mtp_head_forward_dev_at, which the qwen35, qwen35_moe, step35 and hy3 MTP heads use, reads it at 3838 (and the glm5 MLA head at 4118). This drags the glm5 experiment into every MTP family's draft program.
- cut: removes qwen35, qwen35_moe, step35 and hy3 MTP cells from the glm5 dev-IO door's reach
- check: The FLAGS row says DEFAULT OFF with decide-by 2026-09-19, pending a box row. Today is 2026-11-03, and AGENTS.md says a door past decide-by without a receipt is removed. The read sits inside mtp_head_forward_dev_at (fn at 3785), the generic MTP head, not only the glm5 paths.
- fix: Delete the door and its glm_spec/dflash/spec arms, or land its receipt and make the device iota the default.

### low: qwen-path MEMRA_SPEC_ADAPT kept after a REFUTED receipt

- where: `crates/memra-engine/src/spec.rs:12726` (S10, confirmed)
- detail: FLAGS.md:902: ported + REFUTED 2026-08-01 on the tuned qwen configs. The adaptive-K block (12726-12763, and the kc updates in the loop) is still in generate_spec_inner2. The gemma arm of the same name lives in gemma_spec and is unaffected.
- check: FLAGS.md marks the qwen-path MEMRA_SPEC_ADAPT=1 as 'ported + REFUTED 2026-08-01', but the opt-in block still lives in generate_spec_inner2. By the door rule a refuted receipt deletes it. The gemma arm (gemma_spec.rs 861) is separate. MEMRA_SPEC_ADAPT_FLOOR/FLOOR_CTX/CAPMAX are shared with gemma and must stay.
- fix: Remove the qwen adapt/kc block from generate_spec_inner2 and the FLAGS.md:902 row, and keep the shared floor/cap envs for gemma.

### low: MEMRA_RANK0_MERGE has a NEGATIVE receipt but the door and its decode.rs dispatch remain

- where: `crates/memra-engine/src/tp.rs:164` (S09, confirmed)
- detail: FLAGS.md:2354 says 'RECEIPTED NEGATIVE (-2.7)'. The doctrine says a negative receipt deletes the door in the same PR. It is still read in decode_step_h and decode_step_chain (decode.rs:824, 1119). MEMRA_FENCE_MEMOPS (tp.rs:174) is likewise 'RECEIPTED FLAT on PCIe' and kept only for an unmeasured NVLink case.
- check: FLAGS.md marks MEMRA_RANK0_MERGE RECEIPTED NEGATIVE (-2.7), yet rank0_merge_on is still read and dispatched in decode.rs. MEMRA_FENCE_MEMOPS is RECEIPTED FLAT, which the doctrine also lists as a deleting verdict, and is kept for an unmeasured NVLink case.
- fix: Delete MEMRA_RANK0_MERGE (env read, both decode.rs redirects, FLAGS row) and MEMRA_FENCE_MEMOPS, and log both in Removed doors.

### low: MEMRA_TP_TRANSPORT non-default arms past decide-by 2026-09-20

- where: `crates/memra-engine/src/tp_transport.rs:150` (S09, confirmed)
- detail: FLAGS.md row 212 lists decide-by 2026-09-20 for the peer-pull/device-push arms. Both arms and the alias MEMRA_GLM5_TP_TRANSPORT remain without a recorded flip. The module doc itself says peer-pull measured worse than host bounce.
- check: The FLAGS row gives device-push decide-by 2026-09-20 and says pull measured worse than host bounce on the B200 pair, which is a negative receipt that the doctrine says deletes the arm. Both non-default arms and the parser are still present.
- fix: Delete the peer-pull arm on its negative receipt, and re-price device-push or delete it and the MEMRA_GLM5_TP_TRANSPORT alias.

### low: MEMRA_FP8_FOLD research door has no decide-by and no receipt

- where: `crates/memra-gguf/src/source.rs:2312` (S20, confirmed)
- detail: Default-OFF lossy arm (block-128 to per-tensor e4m3 fold) from lane fp8-gemm-arm 2026-08-03. The FLAGS.md row (docs/FLAGS.md:424) carries no decide-by date, and three months later no verdict exists. Door hygiene says a door past decide-by without a receipt is removed with its dispatch arm (source.rs:2498-2520) and its tests.
- cut: Removes an FP8_BLOCK numeric arm from every safetensors FP8 family's possible cells.
- check: The FLAGS.md row carries no decide-by, and the env read and fold arm are live in source.rs. One correction: a receipt does exist. research/fp8st-20260804/mmq/SLICE3-MODEL-VERDICT.md shows ARM A flips top-1 and drifts more than the MMQ path, which superseded it as the default. Under door hygiene a superseded receipt requires deletion, so the door should already be gone.
- fix: Delete the MEMRA_FP8_FOLD read, the fold arm in find_fp8_native and its tests, and move the row to Removed doors with the SLICE3 verdict as its receipt.

### low: MEMRA_SPEC_RESUME_SAMPLER rollback seam kept past the two-week winner rule

- where: `crates/memra-sampling/src/lib.rs:173` (S21, confirmed)
- detail: The predicate has been default ON since 2026-08-20 (docs/FLAGS.md:1481). Doctrine says a winner's rollback seam is deleted once it has served two weeks unused. legacy_admits (always true) and the server's =0 branch remain.
- check: AGENTS.md:90 says to delete a winner's rollback seam once the default has served two weeks unused. MEMRA_SPEC_RESUME_SAMPLER has defaulted ON since 2026-08-20, more than ten weeks ago, and the server still reads =0. The FLAGS row gives a reason for the seam ('the within-binary A/B arm for the refusal's cost') but no decide-by date and no recorded measurement, so the doctrine applies. One part of the suggested fix is wrong: legacy_admits is also the 'teeth' oracle used by the memra-sampling refusal tests (lib.rs:1128, 1230) and by dsv4_moe_program_class_gate. It cannot simply be deleted; only the server's env door should go.
- fix: Remove the MEMRA_SPEC_RESUME_SAMPLER=0 branch in worker.rs and its FLAGS row, and keep legacy_admits as test-only teeth, with its doc no longer naming a runtime door. Alternatively, give the row a decide-by date for the pending cost A/B.

### low: MEMRA_DSV4_SESSIONS / MEMRA_DSV4_ROWS past decide-by 2026-10-11

- where: `crates/memra-server/src/dsv4_serve.rs:249` (S07, confirmed)
- detail: Both rows in FLAGS.md say decide-by 2026-10-11. The defaults (16 lanes, rows = lanes on TP/EP) serve and the env seams remain.
- check: Both FLAGS rows (1342, 1343) set decide-by 2026-10-11 and both env seams are still read. One caveat: lane and row counts could legitimately be reclassified as runtime parameters, but then the rows should drop the decide-by.
- fix: Either reclassify both as runtime parameters (drop the decide-by and say so in the rows) or remove the seams.

### low: MEMRA_DSV4_TOPOLOGY pp rollback seam past its decide-by date

- where: `crates/memra-server/src/dsv4_serve.rs:263` (S07, confirmed)
- detail: FLAGS.md gives the pp seam decide-by 2026-10-09 ('goes once TP/EP has served two weeks without it'). On 2026-11-03 the seam is still live, with its PP-only code (rows_step PP arm, pipelined greedy_step, PP+EP in dsv4_ep.rs, the PP plan in dsv4_topology.rs). Under the door rule this is either removal or a receipt.
- check: The FLAGS row sets decide-by 2026-10-09 for the `pp` rollback seam, and resolve_topology still accepts `pp` with no later receipt. The proposed fix is too broad, though. Dsv4Gpu::load, the gate entry point, defaults to Dsv4Placement::Pp unless a gate arms TP/EP, so the PP engine paths are still used by gates. Only the server's `pp` env arm is past its date. That is door hygiene, not broken behavior, so I rate it low.
- fix: Remove the server `pp` arm, or record a receipt and a new decide-by; delete PP engine paths only after moving the gates that load PP by default.

### low: DFlash restore doors have no decide-by date

- where: `crates/memra-server/src/worker.rs:873` (S17, confirmed)
- detail: MEMRA_DSPARK_PREFIX_RESTORE (2026-08-27), MEMRA_DSPARK_PARTIAL_RESTORE (2026-09-05) and MEMRA_DSPARK_BOUNDARY_CAPTURE (2026-09-06) are default OFF. Their FLAGS.md rows carry no decide-by, and the 14-day default has long passed.
- cut: none
- check: Correct for two of the three doors, with one correction. MEMRA_DSPARK_PREFIX_RESTORE (FLAGS.md:1232) and MEMRA_DSPARK_PARTIAL_RESTORE (1233) have no decide-by. MEMRA_DSPARK_BOUNDARY_CAPTURE (1234) does have one, 'decide-by: 2026-09-20', but that date is long past. All three are still default OFF in code. The worker doc still says the FLAGS row 'lands with the receipts'.
- fix: Add decide-by dates to the PREFIX_RESTORE and PARTIAL_RESTORE rows and decide all three (BOUNDARY_CAPTURE is past 2026-09-20); drop the 'row lands with the receipts' sentence.

### low: Winner rollback seams kept long past their two-week window

- where: `crates/memra-server/src/worker.rs:39892` (S18, confirmed)
- detail: MEMRA_ADMIT_YIELD (default on since 2026-08-06) and MEMRA_SSE_PER_BURST (2026-08-05, 39882) are rollback seams for shipped winners. The door rule deletes a winner's seam once it has served two weeks unused. No receipt shows they were used.
- check: Only half holds. MEMRA_SSE_PER_BURST is a pure rollback seam: content is byte-identical, it has been on since 2026-08-05, and no launcher or gate in the repo sets it, so the two-week winner rule applies. MEMRA_ADMIT_YIELD is not just a rollback seam. Its FLAGS row documents a real trade-off (-3.4% aggregate at c=8 for 3.8x better p50) and tells operators to 'Set 0 on pure-batch tiers that want lockstep fairness back'. That makes it a runtime policy parameter the doctrine allows.
- fix: Delete MEMRA_SSE_PER_BURST and its per-burst arm. Keep MEMRA_ADMIT_YIELD as a documented fairness parameter, or reword its row from 'rollback seam' to a tier knob.

### low: MEMRA_ADMIT_BY_MEMORY door past decide-by

- where: `docs/FLAGS.md:17` (S19, confirmed)
- detail: Receipts exist (research/INDEX.md spill-b day 36: V-DOOR PASS on both cards), but no decision is recorded and the door remains default OFF almost four weeks after its date.
- check: The row is still OFF with decide-by 2026-10-07 and still says the flip waits on the lane B cell. research/INDEX.md day36 records 'the owner's decision cell ... both cards: V-DOOR PASS', but FLAGS.md records no decision.
- fix: Record the owner decision in the FLAGS.md row: promote to default ON, or delete the door and its admit_memory arms.

### low: MEMRA_PRIME_ATTN_FA2 and MEMRA_PRIME_KV_T3 are past their 2026-09-23 decide-by

- where: `docs/FLAGS.md:31` (S15, confirmed)
- detail: Both doors carry positive receipts but are still default OFF six weeks after decide-by. The rows defer to an owner batch (FA2) or a separately recorded failure (T3). They keep fa_prefill_qw_fa2(_prime_table) (flash_attn.cu:9926-10092) and fa_prefill_qw_t3(_prime_table) (5117, 9918) alive behind opt-in reads.
- check: Both rows still read 'decide-by: 2026-09-23', with default OFF. FA2's enablement is 'owner-batched'. T3 is held on 'the separately recorded accumulated-state failure'. Nothing resolves either one. Only the door's own margin-gate binary and its test set them.
- fix: Get the owner decision: promote with a per-device key, or delete both doors and kernels.

### low: MEMRA_Q80_BATCH_R2 and MEMRA_E4M3_BATCH_R2 are past their decide-by date

- where: `docs/FLAGS.md:406` (S14, confirmed)
- detail: Both doors landed 2026-09-05 with decide-by 2026-09-19 (FLAGS.md:405-406) and are still DEFAULT OFF on 2026-11-03 with no promotion or removal. They keep nine Q8_0 r2/r2_rp kernels (qmatvec.cu 3240-3322, 9032-9052) and three E4M3 r2 kernels (3886-3993) plus the two dispatch branches in qmatvec_mmvq_batched alive. Doctrine: a door past decide-by without a receipt is removed.
- check: Both FLAGS rows and both code readers (lib.rs:3053, 3072) still say DEFAULT OFF with decide-by 2026-09-19, and the date has passed with no promotion or removal recorded. Only the GPU identity tests set them. Nothing is broken, but they keep r2 kernels and dispatch branches alive against the decide-by law. Severity lowered to low.
- fix: Run the speed A/B and promote, or delete both doors, their r2 kernels, the dispatch branches and the FLAGS rows.

### low: Default-off doors in this slice carry no decide-by date

- where: `docs/FLAGS.md:493` (S14, confirmed)
- detail: MEMRA_MMQ_F8F4 (FLAGS.md:493, the 450-line F8F4 tile in mmq_nvfp4_w4a8.cu 960-1406), MEMRA_B200_GEMV_V2 (595; qmatvec.cu 13470-14516), MEMRA_PP_BF16 (1419; f16_prefill.cu bf16 GEMMs) and MEMRA_PP_FP8 (402; all of fp8_prefill.cu) are default OFF with no decide-by, against the rule that every default-OFF door lands with one.
- check: Partly right. None of the four rows has a decide-by. MEMRA_B200_GEMV_V2 (595) gives no reason for staying off at all. The other three explain why they are off: MEMRA_MMQ_F8F4 has a 2026-08-06 re-verdict ('still NO global flip', per-model adoption), MEMRA_PP_FP8 is opt-in for its VRAM cost and partly superseded, and MEMRA_PP_BF16 records NO-GO verdicts and owner calls. Those three arguably meet the 'unless the row says why' clause in AGENTS.md. So the clear gap is B200_GEMV_V2.
- fix: Add a decide-by to MEMRA_B200_GEMV_V2, and give the other three rows an explicit 'no decide-by: per-model/owner product switch' note.

### low: MEMRA_MOE_PREFETCH is past decide-by; the row says it qualifies as the PRO card's naked default but no flip happened

- where: `docs/FLAGS.md:861` (S12, confirmed)
- detail: The FLAGS row records G1-G3 PASS and pftime -> pf_wins on the target card, then waits on the owner's call and the RTX 5090's cells. The door still gates moe_cache prefetch (2314-2545), hybrid_forward moe_prefetch_expert and the cpu_experts prediction worker. Per the flags doctrine a measured winner becomes a per-device default; an unmeasured arm is removed.
- cut: none
- check: The FLAGS row is 'experimental, default OFF; decide-by: 2026-10-04'. It records G1-G3 PASS and pftime -> pf_wins on the target card, then says the flip is the owner's call and the 5090 cells are waiting. There is no flip or update since. hybrid_forward still enables prefetch only on =1 or the opt-in spill worker. So the door is stale but waiting on an owner decision, not neglected.
- fix: Get the owner's call and flip it to a device-keyed default on rtxpro6000 per DAY59, finish the 5090 cell or delete the door, and update the row's date.

### low: MEMRA_KPOOL_SELECT_V2 is past decide-by with no decision receipt

- where: `docs/FLAGS.md:959` (S15, confirmed)
- detail: The row names the decision receipt still to be produced (box selab row); none is recorded, and the date passed 2026-09-20. Per doctrine the door, the v2 kernels (mla_attn.cu:2028-2250), the gate and the rows go.
- check: The row's decide-by of 2026-09-20 has passed, and its decision receipt (box selab row) is still described as pending. Only the door's own test sets the flag; the read is live in mla_ffi.rs.
- fix: Record the selab row and promote, or delete the door, both v2 kernels, the gate and the rows.

### low: Windowed FA rollback seams (MEMRA_FAW_P1/G4/O2) outlived the two-week rule

- where: `docs/FLAGS.md:1005` (S15, confirmed)
- detail: P1 became default 2026-07-22. The rollback still keeps three kernels (fa_prefill_w_bf16_pp, _g4, _g4o2, about 600 lines at flash_attn.cu:1327-1507/2050-2503) compiled into both flash fatbins. Doctrine: winners lose their door after two weeks unused.
- check: p1 = !floor && !f32_stage && P1_ON, and g4 needs the same !floor && !f32_stage. So the g4, g4o2 and bf16_pp kernels run only when MEMRA_FAW_P1=0 (or G4=0). P1 became the default on 2026-07-22, well past the two-week rule, and no launcher in the repo sets any of the three.
- fix: Delete MEMRA_FAW_P1/G4/O2 and the g4, g4o2 and bf16_pp kernels plus their rows.

### low: MEMRA_BVAR keeps forced-only batched kernels alive without a decide-by

- where: `docs/FLAGS.md:1291` (S14, confirmed)
- detail: Auto batched_variant never picks NVFP4 b2_pf, b8_pf, b8_r2, b2/b4_pfr2, b4_rpsc, b8_rp, b8_rpr2 or k-quant b4_r2w8 (lib.rs:23708-23813); they are reachable only via MEMRA_BVAR, described as an H100 re-tune seam, and the comment says they are 'kept behind the force seam for the corpus'. The doors doctrine deletes kernels only a measurement seam reaches once the verdict is in.
- check: The auto batched_variant never picks NVFP4 b2 pf (b2 non-rp gives r2 or base), b8 r2/pf/rp/rpr2 (b8 gives rpsc/rpr2w8/r2w8) or b4 rpsc, and it never picks k-quant r2w8. The comment says these kernels are 'kept behind the force seam'. The MEMRA_BVAR row calls it a re-tune seam but gives no decide-by and no explicit no-decide-by reason, as AGENTS.md asks. It is a seam, not a feature door, so this is low.
- fix: Give the MEMRA_BVAR row a decide-by tied to the sm_90a re-tune (or a stated no-decide-by reason), then delete the forced-only kernels once the verdict is in.

### low: MEMRA_KV_HOST_HANDOFF_IO door past decide-by

- where: `docs/FLAGS.md:1461` (S19, confirmed)
- detail: The direct-I/O arm in handoff_io.rs is still selectable past 2026-10-10 with no verdict in its row.
- check: The row still says buffered (default), decide-by 2026-10-10, and 'Decision needs the registered interleaved cell on both rigs', with no verdict. handoff_io.rs:54 still parses the env var, so the direct arm can still be selected.
- fix: Run the registered interleaved cell and promote direct, or delete the direct arm and the env parse.

### low: MEMRA_FA_COMBINE_S default-OFF door has no decide-by date

- where: `docs/FLAGS.md:2357` (S15, confirmed)
- detail: The row says default OFF pending the battery while the step37 serving env pins it on. No decide-by is recorded, which the doors rule requires for every default-OFF door. It selects fa_decode_combine_gate_f32_s (flash_attn.cu:6559).
- check: The row is default 'off', says 'Default OFF pending the battery', and has no decide-by. AGENTS.md requires a decide-by on every default-OFF door. The launcher exemption (step37 env) allows keeping the door but does not supply the missing date.
- fix: Add a decide-by, or mark the row as launcher-set and promote it for step35.

### low: MEMRA_FA_UNROLL and MEMRA_FA_HSPLIT are receipted negative but their doors and kernels remain

- where: `docs/FLAGS.md:2359` (S15, confirmed)
- detail: Door doctrine says a negative receipt deletes the env read, the dispatch branch and every kernel only it reaches. Both rows record negatives (-2.2% and -2.7%, FLAGS.md:2359/2362), yet lib.rs:32562/32569 still read them and flash_attn.cu keeps fa_decode_vec_q_v3_dcw_u8 (9555) and fa_decode_vec_q_v3_dcw_hs2 (9426). They widen the step35 dcw blast radius for nothing.
- check: Both rows record negative receipts (-2.2%, -2.7%), yet lib.rs still reads both envs and selects the _u8 and _hs2 kernels. No launcher or gate in the repo sets them. Under the doctrine a negative receipt deletes the door in the same PR.
- fix: Delete both env reads, fa_decode_vec_q_v3_dcw_u8/_hs2, the <8> instantiation and their rows; record them in Removed doors.

### low: default-OFF qwen4exp perf seams have no decide-by date, 64 days after landing

- where: `docs/FLAGS.md:2385` (S08, confirmed)
- detail: kvhoist and poolT (introduced 2026-08-31) and vgraph (mtp9) are default-OFF perf levers whose FLAGS.md rows carry no decide-by date and no flip receipt. AGENTS.md door hygiene requires a decide-by 14 days after landing and removal when it passes without a receipt. They are reachable only through the gate-only MEMRA_Q4E_SEAMS, which may qualify as a gate-set flag, but their rows describe them as pending perf flips, not gate instruments.
- check: The set_kv_hoist, set_pool_t and set_verify_graphs FLAGS rows are default-OFF perf levers (introduced 2026-08-31 or earlier) with no decide-by date. AGENTS.md requires a decide-by for every default-OFF door and removal once it passes without a receipt. MEMRA_Q4E_SEAMS itself is gate-set, but these rows describe pending perf flips, not gate instruments.
- fix: Run the interleaved A/B for kvhoist, poolT and vgraph and flip or delete each, or relabel them as dated gate instruments.

### low: MEMRA_BACKGROUND_RESPONSES door past decide-by

- where: `docs/FLAGS.md:2737` (S19, confirmed)
- detail: The background delivery door is read by background_jobs::validate and responses_api past its decide-by, and the row does not say it is an owner product switch.
- check: The row is OFF with decide-by 2026-10-12 and is still read by background_jobs.rs:60 and responses_api.rs:120. It says only that 'The owner retains the retention/cap decision'. FLAGS.md never uses the phrase 'product switch', so the row does not mark the door as one. The decide-by has passed with no verdict.
- fix: Either drop the decide-by and label the row an owner product switch, or record the decision.

## Stale docs

### low: build.rs claims mla_attn.cu has no tensor-core intrinsics

- where: `crates/memra-engine/build.rs:503` (S15, confirmed)
- detail: mla_attn.cu has a bf16 mma.sync/ldmatrix scorer (mla_attn.cu:1391-1401, memra_mla_kpool_score_tc_kernel). It is sm_80-class, so it builds on every target, but the comment is false.
- check: mla_attn.cu contains an ldmatrix.sync plus mma.sync.m16n8k16 bf16 tensor-core scorer (the DSA score TC arm), so 'no tensor-core intrinsics' is false. Since it is sm_80-class it still needs no arch stub, so the build itself is fine.
- fix: Say the TU carries a bf16 mma.sync/ldmatrix scorer (sm_80+, present on every target) so no arch stub is needed.

### low: dsv4_dense_cutlass.cu names the wrong build switch

- where: `crates/memra-engine/cu/dsv4_dense_cutlass.cu:4` (S16, confirmed)
- detail: build.rs compiles this file only under MEMRA_DSV4_CUTLASS and deliberately not under MEMRA_CUTLASS (build.rs:748-753).
- check: build.rs compiles cu/dsv4_dense_cutlass.cu only when MEMRA_DSV4_CUTLASS is set, for sm_120a only, and says outright that this is independent of the MEMRA_CUTLASS block.
- fix: Say 'Compiled only under MEMRA_DSV4_CUTLASS (sm_120a, needs MEMRA_CUTLASS_ROOT)'.

### low: fp4_gemm_sel wrappers claim a process-env behavior that no longer exists

- where: `crates/memra-engine/cu/dsv4_gpu.cu:2269` (S16, confirmed)
- detail: dsv4_gpu.cu has no getenv at all. Both legacy wrappers hard-code reduce_arm=0.
- check: dsv4_gpu.cu calls getenv nowhere. The only getenv in the TU comes from dsv4_dense_m1_exact_tail.cuh, which is included later at line 4826 and has nothing to do with these wrappers. Both legacy wrappers pass a literal reduce_arm=0 to _g_arm, and _g_arm reads no env.
- fix: Drop 'and their process-env behavior' from the comment.

### low: FA3 header still says opt-in; it is default ON on 90a

- where: `crates/memra-engine/cu/fa3_prefill.cu:4` (S15, confirmed)
- detail: lib.rs:28214-28229 promoted FA3 default-ON on Hopper (2026-07-27, MEMRA_FA3=0 reverts). The TU header (lines 4 and 16-17) still says opt-in, which misleads impact readers into treating h100 prime cells as unaffected by edits here.
- check: The TU header says 'opt-in MEMRA_FA3=1' and 'the dispatch arm is opt-in'. lib.rs defaults fa3_on to cfg!(memra_hopper_mma), promoted 2026-07-27, with MEMRA_FA3=0 as the revert. The comment just above the dispatch also still says OPT-IN.
- fix: Header and lib.rs:28210 comment: default ON under memra_hopper_mma, MEMRA_FA3=0 rolls back.

### low: Copy-pasted hd512 banner sits over the hd256 g4o2 windowed kernel

- where: `crates/memra-engine/cu/flash_attn.cu:2301` (S15, confirmed)
- detail: Lines 2300-2318 repeat the hd512 banner and the N_WARPS_512/BLOCK_Q_512 defines verbatim from 2505-2523, directly above fa_prefill_w_bf16_g4o2 (hd256 MQA windowed). The real hd512 kernel starts at 2524.
- check: Lines 2300-2318 are a verbatim copy of the hd512 banner and the N_WARPS_512/BLOCK_Q_512 defines at 2505-2523, but the kernel under them is fa_prefill_w_bf16_g4o2, which has HEAD_DIM=256 and launch bounds N_WARPS*WARP_SZ. Redefining an identical macro is legal C, so it compiles. Only the text is wrong.
- fix: Replace the banner above fa_prefill_w_bf16_g4o2 with a g4o2 (hd256 MQA windowed) description and drop the duplicate N_WARPS_512/BLOCK_Q_512 defines there.

### low: mmq_fp4_stub.cu header names the wrong target

- where: `crates/memra-engine/cu/mmq_fp4_stub.cu:1` (S14, confirmed)
- detail: build.rs swaps the stub in only when the arch is neither 120a nor 100a (89, 90a); sm_100a now compiles the real mmq_fp4.cu with the tcgen05 twin (-DMEMRA_SM100_TCGEN05=1).
- check: build.rs swaps in mmq_fp4_stub.cu only when the arch is neither 120a nor 100a, and sm_100a gets the real mmq_fp4.cu with the tcgen05 path. The stub header still says it is for B200 (sm_100a).
- fix: Retitle the header 'Portable (sm_89 / sm_90a) fail-closed ABI stubs for the W4A4 MMQ path'.

### low: F8F4 header still says the mainloop is pending

- where: `crates/memra-engine/cu/mmq_nvfp4_f8f4.cu:12` (S14, confirmed)
- detail: The F8F4 loader, MMA, pipelined tile and launcher exist in mmq_nvfp4_w4a8.cu (960-1406, memra_mmq_nvfp4_f8f4); this TU only holds the quantizer and act_bytes.
- check: The header still says the tile mainloop 'lands next'. The launcher memra_mmq_nvfp4_f8f4 is defined in mmq_nvfp4_w4a8.cu:1358, and this TU exports only the quantizer and size helpers.
- fix: Replace the STATUS lines with a pointer to the mainloop and launcher in mmq_nvfp4_w4a8.cu.

### low: Header calls the grouped f16 MoE path an experimental door

- where: `crates/memra-engine/cu/moe_f16_grouped.cu:13` (S16, confirmed)
- detail: Lines 13-14 say 'Experimental door MEMRA_MOE_F16G=1 until gated', but moe_f16g_mode() defaults to 2 on every arch (lib.rs:268, re-arbitrated 2026-08-02). A reader would wrongly treat this file as default-off.
- check: The header calls the path an experimental door that needs MEMRA_MOE_F16G=1. moe_f16g_mode() returns 2 when the variable is unset, on both arches re-arbitrated 2026-08-02, and 0 is the rollback.
- fix: Rewrite the NUMERICS paragraph: default mode 2 on every arch, MEMRA_MOE_F16G=0 is the rollback.

### low: MEMRA_HYPER_BATCH refusal text says unset = eager; unset is batched

- where: `crates/memra-engine/src/decode_batch.rs:2640` (S05, confirmed)
- detail: Line 2638 maps unset to true (batched mHC chunks, default ON since 2026-08-31). The panic text, and the FLAGS.md row description, still say unset/0 = eager per-session decode. The scout route table also lists the door as opt-in.
- check: hyper_batch_on maps Err(_) (unset) to true, but the panic text says unset/0 = eager. The FLAGS.md row description also says 'Unset/`0` = eager per-session decode (the current serving behavior...)' and still has a stale 'NOT default-ON because' sentence, while its status column says DEFAULT ON.
- fix: Change the message to 'unset/1 = batched mHC decode chunks, 0 = eager', and fix the FLAGS.md description and remove its stale 'NOT default-ON' sentence.

### low: decode_step_batch_hyper doc says B <= 8; the cap is 15

- where: `crates/memra-engine/src/decode_batch.rs:2698` (S05, confirmed)
- detail: hyper_batch_cap() derives PRIME_MIN_T - 1 = 15, and the FLAGS row says B<=15. The doc on the entry still states B <= 8.
- check: PRIME_MIN_T is 16, so hyper_batch_cap() defaults to 15. The FLAGS row says B<=15 and glm_spec.rs cites cap 15, but the decode_step_batch_hyper doc still says B <= 8.
- fix: Change the doc to 'Width: B <= hyper_batch_cap() (= PRIME_MIN_T - 1 = 15)'.

### low: Attention TP geometry marked gate-only but built for every served TP/EP load

- where: `crates/memra-engine/src/dsv4_attention_tp.rs:1` (S07, confirmed)
- detail: AttentionTpGeometry::new is called from Dsv4Gpu::load_placed for Dsv4Placement::TpEp { attention_tp: true }, the served default.
- check: The server's default placement (unset or tp_ep) is TpEp { attention_tp: true }, and load_placed builds AttentionTpGeometry for exactly that placement. So the module serves; it is not gate-only. Only the set_for_gate control in the file is gate-only.
- fix: Reword line 1 to 'Geometry for DSV4 head-parallel attention (served TP/EP default)' and keep 'gate-only' for set_for_gate alone.

### low: Door registry docs disagree with the registry

- where: `crates/memra-engine/src/dsv4_doors.rs:738` (S07, confirmed)
- detail: The doc says THREE inert default-ON doors, but the test pins one (replay cadence). L500 says 'nine merged DSV4F doors' over a 7-row table. L29-31 says a TP/EP-only door is unreachable until TP/EP serves, but it does serve. L70 says Dsv4Gpu::load calls the policy functions, while load calls only matrix_expert_program_resolved and door_receipt_lines.
- check: The inert_default_on_doors doc says THREE, but the test pins exactly one, ["replay cadence"]. DSV4_DOORS is introduced as 'The nine merged DSV4F doors' and has 7 DoorRow entries (lines 507-579). Lines 29-31 say a TP/EP-only door is unreachable 'until a TP/EP program wins its served A/B and the server gains a selector', which the paragraph just above says happened on 2026-09-25. The load-path claim is also loose: dsv4_gpu.rs calls only matrix_expert_program_resolved and door_receipt_lines from this module.
- fix: Change THREE to ONE and nine to seven; drop or past-tense the L29-31 TP/EP-unreachable paragraph; say load reaches the policy functions through door_receipt_lines/matrix_expert_program_resolved.

### low: Module header describes PP + EP-pair bring-up, not the served TP/EP route

- where: `crates/memra-engine/src/dsv4_gpu.rs:21` (S06, confirmed)
- detail: Lines 18-22 say the module is 'PP layer split plus the matrix expert-ID EP pair' and 'not a general serving path'. It is the dsv4_serve engine, the default placement is all-layer TP/EP with attention TP2, and plain decode runs on full-token replay graphs.
- check: The module header describes PP plus an EP pair and says 'not a general serving path'. This module is what dsv4_serve loads, and its default placement is TpEp with attention TP (#710). Line 2930 of the same file says 'TP/EP is the served two-card default since memra #710'.
- fix: Rewrite the header to name the served TP/EP attention-TP2 matrix program and list PP and reference executors as secondary or gate arms.

### low: TP_EP_RANK_ORDER_NUMERIC_CLASS still called gate-only

- where: `crates/memra-engine/src/dsv4_gpu.rs:291` (S06, confirmed)
- detail: The TP/EP expert partial join is the default served program.
- check: TP/EP is the served default (#710, dsv4_serve resolve_topology), yet the doc comment still calls its rank-order partial join gate-only.
- fix: Drop 'gate-only' from the comment.

### low: Serving-path dispatch uses *_for_gate names and hides an env door

- where: `crates/memra-engine/src/dsv4_gpu.rs:10127` (S06, confirmed)
- detail: memra_dsv4_hc_dot_split_slices_for_gate() is read on every device decode/verify/prefill step (10127, 16499, 16607). It returns the value of MEMRA_DSV4_HC_DOT_SPLIT, which the CUDA host code reads with getenv (default 16 slices), so the HC24 numeric class depends on an env var that no Rust env scan of this file sees. enable_tp_ep_local_banks_for_gate (4382) and enable_ep_pair_for_gate (4384) also run on the default load path. Tooling or reviewers that treat '_for_gate' as gate-only would wrongly exclude these from serving cells.
- check: The naming half holds. memra_dsv4_hc_dot_split_slices_for_gate() picks the HC24 class on the served decode path, and load_placed calls enable_tp_ep_local_banks_for_gate and enable_ep_pair_for_gate on the default path. The 'hidden env door' half is refuted: MEMRA_DSV4_HC_DOT_SPLIT is registered in dsv4_doors.rs and has a FLAGS row.
- fix: Rename the serving-path functions without _for_gate (hc_dot_split_slices, enable_tp_ep_local_banks, enable_ep_pair).

### low: dsv4_graph says replay is gate-only; it serves the default plain route

- where: `crates/memra-engine/src/dsv4_graph.rs:1` (S07, confirmed)
- detail: ReplayPair is armed by dsv4_serve::arm_replay for every unpenalized greedy (and vendor-default sampled) TP/EP plain request, and RowsReplay runs the default B-row steps.
- check: The module doc says replay is gate-only. In fact dsv4_serve::arm_replay calls arm_full_token_replay, which builds ReplayPair, on every eligible TP/EP plain request, and the B-row path builds RowsReplay.
- fix: Rewrite the module doc to name the served full-token replay (ReplayPair) and the B-row replay (RowsReplay).

### low: Sampler doc claims host is the default

- where: `crates/memra-engine/src/dsv4_sampler.rs:1` (S07, confirmed)
- detail: The plain sampler defaults to device on TP/EP (FLAGS.md row, serve L3155), and dsv4_sampler() returns host only for the PP path.
- check: dsv4_sampler() falls back to Host when the env is unset, but its own doc comment says a loaded model takes Dsv4Gpu::plain_sampler, which defaults by topology. FLAGS.md says device on TP/EP (the default placement) and host on PP-2. The module header's 'default' claim is stale.
- fix: Header: 'CPU radix remains the oracle; the plain default is device on TP/EP, host on PP-2 (Dsv4Gpu::plain_sampler).'

### low: Topology doc still says PP+EP is shipped

- where: `crates/memra-engine/src/dsv4_topology.rs:3` (S07, confirmed)
- detail: TP/EP is the served default since 2026-09-25 and PP-2 is the rollback.
- check: The module doc still calls PP+EP the shipped reference and TP/EP a 'requested experiment'. FLAGS.md says tp_ep is the default since #710 and pp is the rollback.
- fix: State that TP/EP is the served default and PP-2 (`MEMRA_DSV4_TOPOLOGY=pp`) is the rollback program.

### low: Module header says MLA layers are never captured

- where: `crates/memra-engine/src/glm5_decode_graph.rs:23` (S11, confirmed)
- detail: The header (and the hyper_range_decode comment in hybrid_forward.rs) states that MLA/DSA layers stay eager. plan_runs captures MLA PRE/FFN halves under MEMRA_GLM5_GRAPH_MLA, which is default ON on 100a builds via b200_posture_door_from, so on B200 the captured program includes MLA halves.
- check: The header says MLA/DSA layers stay EAGER. plan_runs captures MLA halves whenever glm5_graph_mla_on() && mla_seg_ws_on(), and both are b200_posture_door_from defaults that are true on 100a builds.
- fix: Update the module header (and the hybrid_forward comment) to describe the 100a-default MLA-halves capture and the MLA_MID door.

### low: MEMRA_GLM5_TP_SYMMETRIC doc says the walk half has not landed and the replicated glue is unread

- where: `crates/memra-engine/src/glm5_tp.rs:2045` (S09, confirmed)
- detail: The symmetric walk exists: kda_tp_partials_sym (line 1360) branches on glm5_tp_symmetric_on() and hyper_range_decode_ws_sym runs it. FLAGS.md records the routing fix from 2026-09-07. The doc misstates what the door changes.
- check: The doc says the flag arms only the loader half and that the replicated glue is unread. In fact glm5_tp_symmetric_on gates the walk: kda_tp_partials_sym requires it, and hybrid_forward branches on it at 2860, 3029 and 11652.
- fix: Rewrite the doc: the door replicates the glue at load time and routes the walk through the symmetric path.

### low: Misattached doc comments in the spec and graph files

- where: `crates/memra-engine/src/glm_spec.rs:1789` (S11, confirmed)
- detail: Several rustdoc blocks attach to the wrong item. In glm_spec.rs:1789-1806 the ppN verify-walk doc sits on glm5_rows_embed, and glm5_verify_rows_ppn has none. In dflash.rs:5631 the burst doc is prepended to dspark_vg_admission_debt, and dflash.rs:5689 puts the multi-turn resume doc on dspark_spec_session_from_restored. In glm5_decode_graph.rs, line 85 puts the RunGraph doc on enum RunPiece, 215 puts the LayerSig doc on StageBufs, and 266 puts the kda_runs doc on plan_runs. Generated docs describe the wrong functions.
- check: Each cited block is stacked onto the next item's doc. The ppN verify-walk doc and its #[allow] sit on glm5_rows_embed. The burst doc is prepended to the admission-debt doc. The resume doc sits on dspark_spec_session_from_restored. The RunGraph doc sits on RunPiece, the LayerSig doc on StageBufs, and the KDA runs doc on plan_runs.
- fix: Move each orphaned block (and the stray #[allow]) onto the item it describes.

### low: graph fa_plan matches a kernel name that does not exist

- where: `crates/memra-engine/src/graph_update.rs:253` (S15, confirmed)
- detail: flash_attn.cu defines no fa_decode_vec_q_dc kernel; the dc mains are dpl16_dc, v2_dc, v3_dc, v4_dc and v4_deep_dc. The arm is unreachable.
- check: No .cu file defines fa_decode_vec_q_dc. The only hits are this match arm, a comment in flash_attn.cu and old research ptxas logs, so the kernel was removed. The dc kernels that exist are dpl16_dc, rows_v3_dc, v2_dc, v3_dc, v4_dc, v4_deep_dc and rows_v4_dc. The arm can never match.
- fix: Drop the arm and update the flash_attn.cu:5976 comment so it names a kernel that exists.

### low: The Q8RP comment (and the kqrp_enabled doc at lib.rs:10162 and the pp_f16_enabled doc at f16_ffi.rs:109) say Hopper-only or OFF elsewhere. The code is capacity-keyed off-Hopper (free >= need + 8 GiB), and FLAGS.md rows 899/900/1417 say capacity-keyed.

- where: `crates/memra-engine/src/hybrid.rs:5810` (scout:devices, confirmed)
- check: Two of the three comments are stale. The hybrid.rs:5810 comment says 'DEFAULT ON only on the Hopper lane', and the Err arm about 10 lines below contradicts it with the capacity-keyed owner ruling. The kqrp_enabled doc (lib.rs:10162) says OFF elsewhere: the fn itself returns false off-Hopper, but its only caller (hybrid.rs:5896) ORs in the capacity rule, so the seam's real default is capacity-keyed (FLAGS.md:900). The third claim does not hold as stated. The pp_f16_enabled doc ('opt-in elsewhere') is accurate: the Q8_0/Q6_K/Q4_K f16 mirror sites (hybrid.rs:5985, 6049) and the runtime sites use pp_f16_enabled() alone, which stays Hopper-only. pp_f16_capacity_ok is used only by the Q4F16 lane (hybrid.rs:6283). If anything, FLAGS.md:1417 overstates the capacity keying for MEMRA_PP_F16 itself.
- fix: Rewrite the hybrid.rs:5810 line and the kqrp_enabled doc to the capacity-keyed rule (env first, Hopper ON, else free >= need + 8 GiB). Leave pp_f16_enabled's doc. Narrow FLAGS.md:1417 so the capacity auto is credited to the Q4F16 lane only.

### low: Doc comments attached to the wrong items

- where: `crates/memra-engine/src/hybrid_forward.rs:2441` (S03, confirmed)
- detail: The 2414-2437 block written for hyper_prime_workspace_shape ('`None` for every non-hyper trunk') is the rustdoc of prime_workspace_shape, which returns None for the hyper trunk, the opposite. The same happens elsewhere: the MEMRA_B200_PRIME_V2 header (865-908) attaches to enum HcTapArm, not b200_prime_v2_on; sigmoid_router_enabled's doc (1105-1106) attaches to enum VrowsSel; hyper_prime_call_rows's doc (1599-1602) attaches to PrimeWorkspaceShape; the HC_DECODE_WS counter doc (806-807) attaches to GLM5_TP_SYM_DISPATCHES.
- check: Consecutive /// lines merge into one rustdoc on the next item. The hyper workspace block ('None for every non-hyper trunk') runs straight into the memra#144 lines that document prime_workspace_shape, which returns None for the hyper trunk. The B200_PRIME_V2 header runs into the HcTapArm doc. The sigmoid-router doc runs into the VrowsSel doc. The hyper_prime_call_rows doc runs into the PrimeWorkspaceShape doc. The HC_DECODE_WS counter doc sits on GLM5_TP_SYM_DISPATCHES.
- fix: Move each orphaned block onto the item it describes (hyper_prime_workspace_shape, b200_prime_v2_on, sigmoid_router_enabled, hyper_prime_call_rows, the HC_DECODE_WS counter).

### low: MLA TC prefill comment says default OFF; the door is default ON

- where: `crates/memra-engine/src/hybrid_forward.rs:10102` (S03, confirmed)
- detail: mla_tc_prefill_enabled() returns true when unset (862) and its header (846) records the 2026-08-30 default-ON acceptance. The call-site comment still says default OFF.
- check: mla_tc_prefill_enabled() returns true when the variable is unset, and its header records the DEFAULT ON acceptance. The call-site comment still says default OFF.
- fix: Change the call-site comment to 'default ON; =0 is the rollback seam'.

### low: Hy3 CPU-expert comments still say llama.cpp; the backend is memra's own companion

- where: `crates/memra-engine/src/hybrid_forward.rs:13657` (S12, confirmed)
- detail: cpu_experts.rs:3-6 says the companion (tools/memra_cpu_experts.cpp) owns its own decoders and dots, with no external inference runtime linked or loaded. hybrid_forward.rs:13655-13660 and 13669 still describe 'llama.cpp CPU quant dots' and a llama.cpp/OpenMP dependency, which misleads anyone checking the numeric class of the CPU arm.
- check: cpu_experts.rs and the companion .cpp both say no external inference runtime is compiled, linked or loaded. The hybrid_forward comments still name llama.cpp CPU quant dots. One partial correction: the companion does include omp.h, so the OpenMP part of the comment is still accurate.
- fix: Replace the llama.cpp references with the memra CPU-expert companion (v2 C ABI); keep the OpenMP note.

### low: Doc blocks attached to the wrong functions in six places

- where: `crates/memra-engine/src/hybrid_forward.rs:16409` (S04, confirmed)
- detail: Doc comments and attributes slid onto neighbouring items. The moe_shexp_add doc sits on moe_ffn_glm5_tp_split_grouped_prime (16409). The moe_ffn_dev doc is split: its start is on moe_ffn_pairs (17863) and the tail begins mid-sentence at 18318. The moe_gdec_token doc is on moe_gdec_token_q8 (18730). The gemma4_attn_prime doc is on gemma_fa_one_program (22125). The gemma4_verify_trunk doc is on dflash_tap (25468). The v2 decode driver doc is on step35_verify_qkv_precompute (27807). Rustdoc and readers get wrong contracts for hot-path functions.
- cut: none
- check: All six cases checked. The shexp doc sits above the TP-split grouped prime doc. moe_ffn_pairs carries the head of the STAGE 3 dev doc fused into its own. moe_ffn_dev's doc begins mid-sentence ('kernel headers);') and has a duplicate allow attribute. The STAGE-2 gdec doc sits on moe_gdec_token_q8. The gemma4 attention doc sits on gemma_fa_one_program. The verify trunk doc sits on dflash_tap. The v2 decode doc sits on step35_verify_qkv_precompute.
- fix: Move each doc block (and the attributes) back onto its own function; remove the duplicate allow on moe_ffn_dev.

### low: moe_ffn_grouped says the clamp is step35-only; glm5_next is PRE-clamped on every layer

- where: `crates/memra-engine/src/hybrid_forward.rs:21114` (S04, confirmed)
- detail: clamp_exp_at returns Some(Pre) on glm5_next, which this function can reach under MEMRA_MOE_GROUPED. The code is correct (ffn_act_lim handles both forms) but the comment invites a POST-only shortcut.
- cut: none
- check: clamp_exp_at returns SwigluClamp::pre_if_live(g5.swiglu_limit) for glm5 before reaching step35's Post. moe_grouped_enabled ignores cfg, so glm5 can reach moe_ffn_grouped when the earlier sigmoid grouped-prefill arm declines. The comment is wrong, but the code is correct.
- fix: Comment: 'step35 POST on its clamped layers, glm5_next PRE on every layer; None elsewhere.'

### low: MEMRA_HC_FUSED_PRE doc says default OFF and unset stays Off; code defaults V2 on sm_100a

- where: `crates/memra-engine/src/hyper.rs:317` (S05, confirmed)
- detail: hc_fused_pre_arm_from maps unset or unrecognized values to V2 on 100a (line 353). The enum doc (317-321) and the hc_fused_pre_arm doc (329) still say default OFF and that unset stays Off. A reader would scope a fused-kernel change away from B200 default cells.
- check: The enum doc and the hc_fused_pre_arm doc say default OFF and that unset stays Off. hc_fused_pre_arm_from maps unset and unrecognized values to V2 when built for 100a. Its own doc states the arch-keyed default correctly.
- fix: Change both doc comments to: unset or unrecognized = V2 on a 100a build, the unfused chain elsewhere.

### low: The hc_fused_pre_arm doc says default OFF, but unset on 100a gives V2 (hyper.rs:353).

- where: `crates/memra-engine/src/hyper.rs:329` (scout:devices, confirmed)
- check: The hc_fused_pre_arm doc says '(default OFF, both 1 and 2 opt in)'. The HcFusedPreArm enum doc at hyper.rs:317-321 also says unset and unrecognized values stay Off. hc_fused_pre_arm_from returns V2 for unset or unrecognized values when built_arch == "100a". Both docs are stale; the _from doc is correct.
- fix: Update both docs: unset or unrecognized means V2 on sm_100a builds and the unfused chain elsewhere; =0 forces unfused.

### low: KDA_ONORM_ZQ8 doc says default OFF; it is the B200 posture default

- where: `crates/memra-engine/src/kda.rs:951` (S13, confirmed)
- detail: kda_onorm_zq8_on goes through b200_posture_door_from, so it is ON for 100a builds when unset (FLAGS.md: B200 posture flip 2026-09-06).
- check: kda_onorm_zq8_on delegates to b200_posture_door_from, which returns `built_arch == "100a"` when unset. So the door is ON by default on 100a builds, while the doc says default OFF.
- fix: State 'default ON on sm_100a builds (B200 posture), OFF elsewhere; =0/=1 override'.

### low: The glm5_graph_mla_on doc says default OFF, but the code is ON on 100a through b200_posture_door_from.

- where: `crates/memra-engine/src/lib.rs:750` (scout:devices, confirmed)
- check: The glm5_graph_mla_on doc ends with 'Default OFF pending its model-scale row' and says it needs MEMRA_MLA_SEG_WS=1. The body is b200_posture_door_from (ON on 100a), and SEG_WS is also ON by default on 100a. FLAGS.md:949 says DEFAULT ON on sm_100a builds.
- fix: State: default ON on sm_100a builds (where MEMRA_MLA_SEG_WS also defaults ON), OFF elsewhere; =0 rollback.

### low: FA split doc says the default is a fixed 64 and sits on FA_VEC_MIN_TKV, not fa_split_keys

- where: `crates/memra-engine/src/lib.rs:1201` (S01, confirmed)
- detail: fa_split_keys (L1729) runs a SM-keyed, ctx-adaptive 8/16/32/64/128 ladder. The 'FIXED 64' doc is attached to an unrelated constant 500 lines earlier.
- check: The FA split 'TUNE SEAM ... default stays FIXED 64' block runs straight into the FA_VEC_MIN_TKV doc with no item between them, so rustdoc attaches it to that constant. fa_split_keys at L1729 is undocumented and actually runs a ctx-adaptive, SM-aware ladder (16/32/64/128, plus the FA_SP_GEMMA rung). The 'FIXED 64' text is stale.
- fix: Move the block onto fa_split_keys, rewrite it to describe the adaptive SM-keyed ladder, and leave FA_VEC_MIN_TKV with only its own doc.

### low: fa_v4_at doc says gemma stores 1024; no such store exists

- where: `crates/memra-engine/src/lib.rs:2208` (S15, confirmed)
- detail: The only writer of FA_V4_MAX_DEFAULT is the gqa>8 guard (hybrid.rs:5592, stores 0). The gemma load block sets FA_VEC_MIN/SPW/SP512 only. Readers mapping gemma decode to the v4 vs v3 kernels get the wrong arm.
- check: The only store to FA_V4_MAX_DEFAULT is the gqa>8 guard, which stores 0. The gemma load block sets FUSED_MR1, RMS_BLOCK and FA_SP_GEMMA, and its comment now says v4 measured the depth winner, so there is no 1024 store. The lib.rs doc is stale.
- fix: Rewrite the doc: default usize::MAX, the gqa>8 guard stores 0 at load, and gemma rides v4 at every depth.

### low: Doc comments glued to the wrong items in the door section

- where: `crates/memra-engine/src/lib.rs:3179` (S01, confirmed)
- detail: Several /// blocks run into the next door's doc and attach to it. MEMRA_VERIFY_WS doc (L3179) is on hyper_batch_solo_on. MEMRA_HC_PRE_BLOCK doc (L3209) is on hc_pre_sink_reg. The PDL doc (L4809) is on set_verify_exact. MEMRA_Q8T_WONCE doc (L1651) is on step_gemm_prime_on. MEMRA_GLM5_VROWS_T1_DEV doc (L712, still saying 'Default OFF') is on glm5_graph_recapture_on. The router-GEMV doc (L216) is on moe_f16g_mode. The devq8 geometry doc (L9347) is on moe_pairs_matvec_q8. MEMRA_STEP_TP_W8 doc (L1554) is on the STEP37 static. rustdoc and readers attribute each policy to the wrong door.
- check: I checked the cited sites and each doc block runs into the next door's doc with no item in between, so it attaches to the wrong item: VERIFY_WS on hyper_batch_solo_on, HC_PRE_BLOCK on hc_pre_sink_reg, the PDL doc on set_verify_exact, Q8T_WONCE on step_gemm_prime_on, VROWS_T1_DEV on glm5_graph_recapture_on, the router-GEMV doc on the MOE_F16G block, the devq8 geometry doc on moe_pairs_matvec_q8, and STEP_TP_W8 on STEP37_SERVING_DEFAULTS. The VROWS_T1_DEV doc still says 'Default OFF', but glm5_vrows_t1_dev_forced_from defaults ON for 100a builds.
- fix: Give each block its own item (move each doc above its own fn or static) and update the VROWS_T1_DEV text to 'ON on 100a builds when unset'.

### low: The hc_pre_v4_on doc says default OFF, but hc_pre_v4_on_from defaults ON on 100a (FLAGS.md agrees with the code).

- where: `crates/memra-engine/src/lib.rs:3298` (scout:devices, confirmed)
- check: The hc_pre_v4_on doc ends with 'Default OFF until its model-scale row'. The function delegates to hc_pre_v4_on_from, whose unset arm is built_arch == "100a". FLAGS.md:1036 says DEFAULT ON on sm_100a builds. The doc on the public function is stale; the _from doc below it is correct.
- fix: Replace the last doc line with: Default ON on sm_100a builds, OFF elsewhere; `=0` is the rollback.

### low: with_moe_cache / moe_cache_enabled docs still describe the pre-2026-07-08 default-off cache

- where: `crates/memra-engine/src/lib.rs:6602` (S12, confirmed)
- detail: The MoE cache has been default ON since 2026-07-08 (moe_cache.rs:29-34, lib.rs:6693 `!= Ok("0")`). lib.rs:6602 says the function is never called when the var is unset, and lib.rs:6691 says 'enabled (MEMRA_MOE_CACHE set)'. moe_cache.rs:31-34 records the same stale sentence being quoted as the live default in a placement plan.
- check: moe_cache_enabled returns true unless MEMRA_MOE_CACHE is "0", so the cache is on by default. The two doc comments still describe the old opt-in cache, and moe_cache.rs's own header records the same stale wording being misquoted once before.
- fix: Reword both comments to: enabled unless MEMRA_MOE_CACHE=0 (rollback).

### low: The mla_seg_ws_on doc says default OFF, but it routes through b200_posture_door_from (ON on 100a).

- where: `crates/memra-engine/src/lib.rs:13936` (scout:devices, confirmed)
- check: The mla_seg_ws_on doc says 'default OFF', but the body calls b200_posture_door_from, which defaults ON when built_arch == "100a". FLAGS.md:947 says ON on sm_100a builds.
- fix: Change 'default OFF' to 'default ON on sm_100a builds, OFF elsewhere; =0 rollback'.

### low: The hc_mixes_kernel_on doc says default OFF pending a model-scale row, but the code is ON on 100a.

- where: `crates/memra-engine/src/lib.rs:16008` (scout:devices, confirmed)
- check: The hc_mixes_kernel_on doc says 'Default OFF pending its model-scale row', but the body uses b200_posture_door_from (ON on 100a). FLAGS.md:1037 says DEFAULT ON on sm_100a builds.
- fix: State: default ON on sm_100a builds, OFF elsewhere; =0 rollback.

### low: FloatBf16 rows-path comments disagree with the m <= 32 code; ILP door comment says default OFF

- where: `crates/memra-engine/src/lib.rs:18388` (S01, confirmed)
- detail: The matmul comment says 2<=m<=8 and 'Prefill (m>8) keeps GEMM', but the code takes the rows path for 1..=32 (L18394, same at L18709). bf16_mmv_on's doc says 'decode-time (m=1)'. The moe_gate_up_preclamp8_q8_rows comment says MEMRA_MOE_VROWS_ILP is 'default OFF' (L9059), but it is ON on 100a builds. The ILP doc names QT_NVFP4 only (L2591), but the code also admits QT_NVFP4_V2.
- check: Both FloatBf16 rows guards take m in 1..=32, but the comment says 2<=m<=8 and 'Prefill (m>8) keeps GEMM'. bf16_mmv_on's doc says decode-time m=1 and default OFF. The ILP site comment says 'default OFF, interleaved-NVFP4 only', yet moe_vrows_ilp_on_from defaults ON for 100a and the site also admits QT_NVFP4_V2. The moe_vrows_ilp_on doc names only QT_NVFP4.
- fix: Rewrite the rows comments to 1 <= m <= 32, the bf16_mmv_on doc to the m<=32 rows tier, and the ILP comments to 'default ON on 100a, QT_NVFP4 and QT_NVFP4_V2'.

### low: Q8_0 triple doc block is attached to matmul_q4_fused3

- where: `crates/memra-engine/src/lib.rs:20129` (S02, confirmed)
- detail: Lines 20129-20131 describe the Q8_0 triple (matmul_q8_fused3 at 20809), but they precede the Q4_0 launcher's own doc.
- check: The Q8_0 triple doc lines come right before the Q4_0 doc and pub fn matmul_q4_fused3. matmul_q8_fused3 at 20809 has no doc of its own.
- fix: Move lines 20129-20131 onto matmul_q8_fused3.

### low: Orphaned mmvq_supports doc sits above mmvq_fast_eligible and contradicts the code

- where: `crates/memra-engine/src/lib.rs:22188` (S02, confirmed)
- detail: MMVQ is default ON (MEMRA_MMVQ=0 disables it), covers six qtypes plus E4M3, and the comment is attached to the wrong function.
- check: The two-line mmvq_supports doc sits at the top of mmvq_fast_eligible's doc block. It says MMVQ needs MEMRA_MMVQ set and covers four dtypes. In fact mmvq_supports is default ON (=0 reverts), covers six qtypes, and always admits E4M3.
- fix: Remove the two lines from mmvq_fast_eligible and put a corrected doc on mmvq_supports.

### low: Dispatch comment says MEMRA_NVFP4_ROW_ILP is default OFF; it is ON for 100a builds

- where: `crates/memra-engine/src/lib.rs:22514` (S14, confirmed)
- detail: nvfp4_row_ilp_on_from returns built_arch=="100a" when unset (doc at the function says 'default ON on sm_100a builds since 2026-09-04'), so the comment at the only dispatch site misstates the b200 default.
- check: The dispatch-site comment says 'default OFF', but nvfp4_row_ilp_on_from returns built_arch == "100a" when the variable is unset, so it is ON on sm_100a builds.
- fix: Change the comment to 'default ON on sm_100a builds, OFF elsewhere'.

### low: gemm_supports doc lists 4 dtypes; the code admits 6

- where: `crates/memra-engine/src/lib.rs:24390` (S02, confirmed)
- detail: The match at 24409 also admits Q5_K and Q4_0.
- check: The doc lists four dtypes; the match admits Q8_0, Q4_K, Q6_K, Q5_K and Q4_0, plus NVFP4 when in_f % 64 == 0.
- fix: List Q8_0, Q4_K, Q5_K, Q6_K, Q4_0 and NVFP4 (in_f % 64 == 0).

### low: fa_decode_kvmod_view comment says head_dim<=256 for every shipped model

- where: `crates/memra-engine/src/lib.rs:30366` (S02, confirmed)
- detail: The code admits head_dim<=512 (30368) with a dedicated hd512 dpl16 arm for gemma globals (30377), and hd128 families (qwen3, step35, minimax_m3) also reach this path.
- check: The comment says head_dim <= 256 and that all shipped models use 256, but the next line admits head_dim <= 512, and there is a dedicated hd512 DPL16 arm for the gemma4 globals.
- fix: Update the comment to the real coverage: hd128/256 on the vec path and hd512 on the DPL16 twin.

### low: f32_gemv_kernel_on doc says default OFF but the code is default ON on sm_100a

- where: `crates/memra-engine/src/lib.rs:35838` (S02, confirmed)
- detail: The function returns b200_posture_door_from(env, built arch), which is true when the env is unset on 100a builds (3340). FLAGS.md:1038 agrees with the code.
- check: f32_gemv_kernel_on returns b200_posture_door_from(env, built arch), which is true when the env is unset on a 100a build. The FLAGS row matches the code; the doc comment does not.
- fix: Rewrite the doc: default ON on sm_100a builds, OFF elsewhere, =0 is the rollback.

### low: KvDev doc says 7 ops; the impl has 11

- where: `crates/memra-engine/src/lib.rs:35896` (S02, confirmed)
- detail: alloc_vmm_u8, alloc_vmm_on_demand_u8, kv_vmm_granularity and the delegations are 11 methods.
- check: The impl defines 11 methods: alloc_vmm_u8, alloc_vmm_on_demand_u8, kv_vmm_granularity, zeros, uninit, alloc_u8, htod_i32, clone_dtod, copy_into, copy_range_into and set_i32_one. The doc says 7.
- fix: Drop the count from the doc.

### low: MLA B200 door docs contradict their own defaults

- where: `crates/memra-engine/src/mla_ffi.rs:246` (S13, confirmed)
- detail: Line 236 and the code (mla_dsa_decode_level returns 2 when unset) make DSA decode level 2 the B200 default. Line 246 says 0 is the default, and line 353 says 'The door is default OFF'. Line 464 says MEMRA_B200_DSA_SELECT is default OFF, while line 516 and mla_dsa_select_on_from say default ON. A reader sizing B200 qualification from these comments would skip the dsa-warp-online-f32 numeric class that runs by default at t_q=1.
- check: mla_dsa_decode_level returns 2 when the var is unset on sm100 builds, and L236 says 'DEFAULT 2', yet L246 says '`0` (default)'. L353 says 'The door is default OFF'. L464 says DSA_SELECT is 'default OFF', while mla_dsa_select_on_from disarms only on an explicit 0, and its own doc at L516 says DEFAULT ON.
- fix: Rewrite L246 (2 is the sm100 default), L353 (level 2 is default on sm100) and L464 (default ON on sm100, =0 rollback).

### low: Q8_0 MMQ described as opt-in/default OFF although it is default ON

- where: `crates/memra-engine/src/mmq_ffi.rs:1516` (S13, confirmed)
- detail: mmq_q8_enabled defaults ON (unwrap_or(true), 'DEFAULT ON since 2026-07-09'). The same function's doc (line 1418, 'Default OFF until the battery is green') and the mmq_supports comment at 1516 still say opt-in.
- check: mmq_q8_enabled uses unwrap_or(true), and its doc opens with 'DEFAULT ON since 2026-07-09', then ends with 'Default OFF until the battery is green'. The mmq_supports Q8_0 arm still says 'opt-in only (MEMRA_PP_Q8MMQ=1)'.
- fix: Drop the 'Default OFF until...' sentence and change the 1516 comment to 'default ON, MEMRA_PP_Q8MMQ=0 reverts'.

### low: moe_f16_grouped doc comment and allow attribute are attached to bind_runtime_device

- where: `crates/memra-engine/src/mmq_ffi.rs:2701` (S13, confirmed)
- detail: Lines 2687-2696 (the MEMRA_MOE_F16G mode 1/2 contract plus #[allow(clippy::too_many_arguments)]) sit above bind_runtime_device, so rustdoc shows the grouped-GEMM contract on the wrong item and moe_f16_grouped has no doc.
- check: The grouped-f16 contract doc and `#[allow(clippy::too_many_arguments)]` at L2687-2696 run into bind_runtime_device's doc with no item between them, so both attach to bind_runtime_device, which has a single argument. moe_f16_grouped already carries its own allow attribute but has no doc.
- fix: Move the L2687-2695 doc block onto moe_f16_grouped and delete the stray allow above bind_runtime_device.

### low: MoeSlotCache::new doc claims a 256-slot floor and an exact forced N; the code floors at 8 and clamps forced N

- where: `crates/memra-engine/src/moe_cache.rs:777` (S12, confirmed)
- detail: build() uses .max(8) (line 847) and clamps a forced N to the hard ceiling (line 821). The 256 floor and 'exact N' are both false.
- check: build() has no 256-slot floor and no hot-set upper clamp. The auto path floors at 8, and a forced MEMRA_MOE_SLOTS goes through requested_bytes.min(hard_bytes), so the result is not the exact N.
- fix: Document the 8-slot floor and the hard-ceiling clamp of a forced N.

### low: pinned_host module doc says portable and startup-only; PinnedHostBuf allocates at runtime with flags 0

- where: `crates/memra-engine/src/pinned_host.rs:1` (S12, confirmed)
- detail: Only PinnedHostArena uses CU_MEMHOSTALLOC_PORTABLE (line 104). PinnedHostBuf::new and new_unwritten call malloc_host(len, 0) (lines 190, 209), which is non-portable, and run mid-serve (worker staging_take allocates per demote at worker.rs:10796; dsv4 allocates per load). The doc misstates both the attribute and the lifetime.
- check: Only PinnedHostArena passes CU_MEMHOSTALLOC_PORTABLE. PinnedHostBuf::new and new_unwritten call malloc_host(len, 0). The server worker allocates new_unwritten on the first demote that needs a buffer, which is mid-serve. dsv4_c4 and tier_transfer also allocate PinnedHostBuf outside startup.
- fix: Scope the module doc to PinnedHostArena; describe PinnedHostBuf as owned, cached, non-portable allocations that may be made at runtime.

### low: stale line reference to launch_nvfp4_sel_matvec_pack

- where: `crates/memra-engine/src/qwen4exp_gpu.rs:1555` (S08, confirmed)
- detail: The launcher is at L20348, not ~L19840.
- check: The comment points to ~L19840, but launch_nvfp4_sel_matvec_pack is defined at L20348.
- fix: Drop the line number and reference the function by name.

### low: verify_mt doc says prefill is untouched; linear_trunk_into engages the mt kernel for any 2..=12-row forward

- where: `crates/memra-engine/src/qwen4exp_gpu.rs:1843` (S08, confirmed)
- detail: linear_trunk_into and linear_trunk_stacked_into test only (2..=12).contains(&t) && verify_mt_on(), not the verify state. A 2-12 token prompt prefill or prefill_extend tail chunk therefore runs qmatvec_bf16w_mt_f32 on every trunk linear, lm_head included. The kernel is documented bit-identical per row, so the output should not move, but the doc misstates which program prefill runs.
- check: linear_trunk_into and linear_trunk_stacked_into check only (2..=12).contains(&t) && verify_mt_on(). prefill() calls forward(), which calls linear_trunk_into for the trunk linears, so a 2-12 token prefill runs the mt kernel. One part of the claim is overstated: on the LastRow head path, lm_head runs with head_rows = 1 and does not take the mt arm. The kernel is documented bit-identical, so this is doc drift only.
- fix: Correct the doc to say the mt arm covers every 2..=12-row trunk forward, prefill included, or gate it on an explicit verify flag.

### low: decode_step_tp2 doc says graphs stay off; the body captures per-rank segment graphs

- where: `crates/memra-engine/src/qwen4exp_gpu.rs:18499` (S08, confirmed)
- detail: L18589 enables segment graphs whenever decode_graphs_on && step_ws_on (both default ON), and replays them from the second step.
- check: The doc says decode graphs stay off in TP2. The body arms segment-graph mode whenever decode_graphs_on() && step_ws_on(), and both atomics start true, with captures from the second step.
- fix: Rewrite the doc to describe the per-rank segment-graph mode (first step eager warm-up, lazy captures after).

### low: MEMRA_SPEC_HEAD_ROWS comments say default OFF; it is family-armed ON

- where: `crates/memra-engine/src/spec.rs:5470` (S10, confirmed)
- detail: head_rows_on uses step37_door (237), so it is ON whenever a step35 model armed the defaults. FLAGS.md:1080 says DEFAULT ON (family-armed) since 2026-08-27. The doc at 234 and the call-site comment at 5470 still say default off.
- check: head_rows_on goes through step37_door, so it is ON when the step37 family defaults are armed. FLAGS.md:1080 says DEFAULT ON (family-armed) since 2026-08-27. The function doc ('Default off until the byte tape says so') and the call-site comment ('DEFAULT OFF for exactly one turn of the crank') are stale.
- fix: Update both comments to 'default ON when step37 serving defaults are armed; =0 kills'.

### low: qwen35_verify_batch_layers doc says MoE takes the rowwise arm

- where: `crates/memra-engine/src/spec.rs:6708` (S10, confirmed)
- detail: The body admits Qwen35Moe to the t-parallel arm (6728, 2026-08-20). Only t>16 or ROWWISE=1 take rowwise. A reader sizing impact would wrongly treat the MoE verify as the rowwise program.
- check: The doc lists MoE as a rowwise trigger, but the body admits Qwen35Moe to the t-parallel arm (comment at 6728). rowwise is now only ROWWISE=1 or t>16; the predicate has no MoE term.
- fix: Rewrite the arm list: t-parallel for dense and MoE GDN at t<=16; rowwise for t>16 or MEMRA_SPEC_VERIFY_ROWWISE=1.

### low: Detached rustdoc blocks attach to the wrong items

- where: `crates/memra-engine/src/tp.rs:38` (S09, confirmed)
- detail: The SwiGLU doc (38-41) and the OPROJ_DIRECT/MOE_DIRECT/ROUTES_PRESTAGE docs all become the rustdoc of routes_prestage_on (line 60). The FENCE_MEMOPS/LEN_MIRROR_LAZY/RANK0_MERGE/OPROJ_TAIL docs land on oproj_tail_on (82). The SEL_MIRROR/SPEC_FA2/ROWS_TAB docs land on rows_tab_host. In tp_expert_split.rs:95 the split_rows doc attaches to for_each_expert_slab. Readers and rustdoc see the wrong contract on each item.
- check: Lines 38-59 of tp.rs are one unbroken /// run (SwiGLU, raw copy, OPROJ_DIRECT, MOE_DIRECT, ROUTES_PRESTAGE) and attach to routes_prestage_on. The FENCE_MEMOPS, LEN_MIRROR_LAZY, RANK0_MERGE and OPROJ_TAIL block attaches to oproj_tail_on, leaving rank0_merge_on and fence_memops_on undocumented. The SPEC_FA2 and ROWS_TAB block sits above rows_tab_host. The split_rows doc in tp_expert_split.rs runs straight into the for_each_expert_slab doc.
- fix: Move each doc block onto the item it describes (the SwiGLU fn, each *_on door, split_rows).

### low: Hard-coded line references to tp.rs and pp.rs have drifted

- where: `crates/memra-engine/src/tp_transport.rs:395` (S09, confirmed)
- detail: The PRODUCER FENCE comments are now at tp.rs:3309/4015/6032, and the BoundarySlot ev_tx record/wait cited as pp.rs:2647/2678 (line 399) are at pp.rs:3135/3168.
- check: tp.rs:3543 and tp.rs:2636 are now unrelated lines (a weight field and a rank loop). The PRODUCER FENCE comments are at 3309, 4015 and 6032, among others. pp.rs:2647/2678 hold boundary filter code, not the ev_tx record and wait.
- fix: Cite the functions (the tp.rs PRODUCER FENCE sites by function name, PpNRt's BoundarySlot tx/rx path) instead of line numbers.

### low: DeepSeekV4 called loader-lane-only although dsv4_serve serves it

- where: `crates/memra-gguf/src/config.rs:29` (S20, confirmed)
- detail: The comments at config.rs:29-30 and config.rs:181 say there is no forward pass, but memra-server dsv4_serve runs Dsv4Gpu with plain, spec and dspark drivers (route table). The same applies to qwen4_exp at config.rs:186, which has Qwen4ExpGpu.
- cut: none
- check: config.rs says DeepSeekV4 has no forward pass and is loader-lane only, and is_dsv4/is_qwen4exp say 'loader lane only today'. Dsv4Gpu (dsv4_gpu.rs) is served by memra-server dsv4_serve.rs, and Qwen4ExpGpu exists in qwen4exp_gpu.rs.
- fix: Rewrite the comments to name the dedicated engines (Dsv4Gpu via dsv4_serve, Qwen4ExpGpu).

### low: PIPELINE note names step35 as a GatedDeltaNet family

- where: `crates/memra-gguf/src/execution_manifest.rs:624` (S20, confirmed)
- detail: step35 has no GDN layers (SWA + full attention with a separate head gate). It is in fact PIPELINE-admitted, so the note misleads anyone reading why a family lacks PP.
- cut: none
- check: Step35 layer geometry sets every layer to LayerKind::FullAttention (SWA through a window, SeparateHead gate), so its plans never contain GatedDeltaNet, yet the PIPELINE note lists step35 as a GDN recurrent-mixer family.
- fix: Name the GDN families (qwen35/qwen35_moe/qwen4_exp) and drop step35 from the note.

### low: gkv_on doc says windowed layers stay q8_0/q5_1, but wkv_on makes them e4m3 by default

- where: `crates/memra-kv/src/lib.rs:99` (S21, confirmed)
- detail: wkv_on (lib.rs:112-119) defaults ON whenever MEMRA_DRAFT is unset, so in plain serving the gemma windowed layers are e4m3 too (full_attention_kv_layout lib.rs:2389).
- check: The gkv_on doc says windowed layers stay q8_0/q5_1. wkv_on() defaults ON whenever MEMRA_DRAFT is unset, and full_attention_kv_layout gives GemmaWindowed layers 32-byte (e4m3) blocks when wkv_on() is true. In plain serving the windowed layers are therefore e4m3, which docs/FLAGS.md:889 confirms.
- fix: Replace the sentence with: 'Windowed layers follow MEMRA_GEMMA_WKV (wkv_on: e4m3 unless MEMRA_DRAFT is set).'

### low: tiered module says no server path, but the host prefix tier uses it

- where: `crates/memra-kv/src/tiered/mod.rs:2` (S21, confirmed)
- detail: The server builds a tiered::hostprefix::SharedGovernor and implements tiered::BudgetGovernor (worker.rs:5303-5347, 5524), and admission reserves tier images through ResidentCharge::reserve_for_program (admit_memory.rs:396) when MEMRA_KV_HOST_MB>0.
- check: The module header says no server path is enabled. memra-server uses tiered::hostprefix::SharedGovernor, ResidentCharge, IdentitySlot and the BudgetGovernor trait in worker.rs and admit_memory.rs, so the header is stale for that part of the module.
- fix: Say that hostprefix and the BudgetGovernor/TierBudget contracts are installed by the host prefix tier, and that Hierarchy, ActiveEpoch, scheduler, policy and integration are CPU-tested scaffolding only.

### low: hidden_trace claims it changes no arm selection; three call sites prove otherwise

- where: `crates/memra-reference/src/hidden_trace.rs:19` (S21, confirmed)
- detail: enabled() is a conjunct of the device-table MoE arms (hybrid_forward.rs:13887 vrows_t1_dev, :13916 vrows_dev), so tracing forces the host readback arm. glm5_decode_graph.rs:794 also refuses the glm5 decode graph while it is set. The arms are claimed bit-identical, so output should not move, but the timing and program path do, and a bisect under MEMRA_HYPER_TRACE does not exercise the default arms.
- check: The module doc says the trace changes no arm selection and contrasts itself with MoE traces that move observation_mode. That is false. hidden_trace::enabled() is a negated conjunct of both device-table MoE arms (vrows_t1_dev and vrows_dev), so with MEMRA_HYPER_TRACE set the host readback arm runs. The glm5 decode graph also refuses to capture while it is set. The 'when off' part of the doc is true; the contrast with the MoE traces and 'cannot change what the model computes' imply that turning it on keeps the same program path, and it does not.
- fix: Reword the doc: when on, the trace forces the host route readback in place of the device-table MoE arms (vrows_t1_dev, vrows_dev) and turns the glm5 decode graph off. Those arms are claimed bit-identical, but the program path differs.

### low: dsv4_serve doc describes a one-request PP stack

- where: `crates/memra-server/src/dsv4_serve.rs:2` (S07, confirmed)
- detail: The default is TP/EP with up to 16 lanes and B-row coalescing (L230, L262). The doc says PP stack, one request at a time.
- check: The module doc describes a 2-card PP stack serving one request at a time. resolve_topology defaults to TpEp, default_tp_ep_sessions gives 16 lanes, and resolve_rows coalesces B-rows to the lane count on TP/EP.
- fix: Rewrite the header for the TP/EP default: up to 16 lanes, B-row coalesced steps, PP-2 as the rollback.

### low: lane_cap doc points at capacity_for, which nothing calls

- where: `crates/memra-server/src/lib.rs:2319` (S19, confirmed)
- detail: The comment names RouteRegistry::capacity_for as the dsv4 per-model cap, but only route_contract tests call capacity_for, while route_contract.rs marks #501 as implemented through RouteLoad. lane_cap stays lane-scoped from hybrid_interactive_cap.
- check: capacity_for is called only in route_contract tests (622-625). The dsv4 per-model cap is wired as the contract's capacity.concurrency() into route_telemetry::register, i.e. RouteLoad (worker.rs:26160). route_contract marks #501 as done. The lane_cap doc still points at capacity_for as the #501 mechanism.
- fix: Update the lane_cap doc to say the dsv4 per-model cap is the route's own RouteLoad (route_telemetry::register from the contract capacity), and delete capacity_for if only tests use it.

### low: Doc names ServerWiring::reference(), which does not exist

- where: `crates/memra-server/src/lib.rs:5924` (scout:routes, confirmed)
- check: ServerWiring has no reference() constructor; the only constructor is stock() at line 5940, and serve_main calls serve_with(ServerWiring::stock()).
- fix: Change the doc to ServerWiring::stock().

### low: Module header describes the retired one-token round-robin scheduler

- where: `crates/memra-server/src/worker.rs:12` (S17, confirmed)
- detail: The header says each session advances one decode_step per iteration (one prefill token or one decode token) and that admits fill slots up to MAX_ACTIVE. The default scheduler is batched: prefill is chunked prime_cache of PREFILL_TICK_T=1024 tokens (8192 solo), decode is batched chunks, and the cap is MEMRA_MAX_SESSIONS. MAX_ACTIVE=4 applies only to MEMRA_SERVE_BATCH=0.
- cut: none
- check: The module header still describes one decode_step per session per iteration and a MAX_ACTIVE slot cap. MAX_ACTIVE's own doc says it is the legacy round-robin cap (MEMRA_SERVE_BATCH=0). Batched scheduling is the default (serve_batching unwrap_or(true)) and is capped by MEMRA_MAX_SESSIONS. Prefill is chunked by PREFILL_TICK_T=1024.
- fix: Rewrite the SCHEDULER LOOP paragraph for the batched tick (chunked prefill, batched decode, MEMRA_MAX_SESSIONS) and name MAX_ACTIVE as the serial-mode cap.

### low: Doc comments attached to the wrong items

- where: `crates/memra-server/src/worker.rs:3557` (S17, confirmed)
- detail: rustdoc joins each misplaced block onto the next item. The MEMRA_RESUME_GRID_REWIND doc lands on spec_budget_clamp_on (3557). The MEMRA_AFFINITY rollback doc lands on grid_rewind_plain (3323). An orphan 'Can this process admit a spec session...' paragraph, describing a function that no longer exists, plus the GEMMA SPEC seam doc land on dspark_draft_without_spec (6220-6233); the gemma doc belongs to gemma4_spec_k_env. The MEMRA_KV_HOST_MB doc lands on glm5_tp_kv_host_on (4903). The MEMRA_PREFIX_LATENT doc lands on glm5_tp_prefix_on (6083). The is_cuda_oom doc lands on engine_client_message (8081). The MEMRA_STEP_OOM_FAULT doc lands on StepOomFaultSite (8405). The dspark_load_admits doc lands on resumed_carrier_spec_k_floor (589). The dspark_hit_is_restorable doc lands on dspark_partial_restore_on (902). Separately, the glm5_tp_prefix_on doc still says 'default OFF, decide-by 2026-09-21' (6089), while FLAGS.md says the served launcher arms it.
- cut: none
- check: Each cited block was checked: each doc block runs straight into the next item's doc with no item between them. MEMRA_RESUME_GRID_REWIND doc + SPEC_BUDGET_CLAMP doc -> fn spec_budget_clamp_on. MEMRA_AFFINITY rollback doc -> fn grid_rewind_plain. The 'Can this process admit a spec session' orphan + GEMMA SPEC doc + #478 doc -> fn dspark_draft_without_spec. KV_HOST_MB doc -> fn glm5_tp_kv_host_on. PREFIX_LATENT doc -> fn glm5_tp_prefix_on. is_cuda_oom doc -> the engine_client_message doc. STEP_OOM_FAULT doc -> enum StepOomFaultSite. The dspark_hit_is_restorable doc (896-901) -> fn dspark_partial_restore_on, while the real fn dspark_hit_is_restorable at 965 has no doc. The glm5_tp_prefix_on doc says 'default OFF, decide-by 2026-09-21', but the FLAGS.md row says the served 1M TP-2 launcher arms it.
- fix: Move each doc block onto its own item (kv-host-MB onto its reader, is_cuda_oom onto fn is_cuda_oom, the restorable doc onto fn dspark_hit_is_restorable, and so on), delete the orphan 'Can this process admit' paragraph, and note in the glm5_tp_prefix_on doc that the served launcher arms it.

### low: Three unrelated doc blocks are attached to dspark_draft_without_spec

- where: `crates/memra-server/src/worker.rs:6226` (scout:routes, confirmed)
- check: Three doc blocks are stacked on dspark_draft_without_spec: the 'Can this process admit a spec session' text (which belongs on serve_spec_enabled), the GEMMA SPEC seam semantics (which describe gemma4_spec_k_env's unset/0/K logic), and the memra #478 text. Only the last one describes the function.
- fix: Move the spec-admission block to serve_spec_enabled and the GEMMA SPEC block to gemma4_spec_k_env.

### low: prefill_tick doc still describes step_session's tokenwise loop

- where: `crates/memra-server/src/worker.rs:38457` (S18, confirmed)
- detail: The header says the prefill phase consumes ONE prompt token via decode_step. prefill_tick primes tick-budget chunks and reaches decode_step only on its fallback branch.
- check: The doc on prefill_tick opens with step_session's header: one scheduler tick returning Ok(true)/Ok(false), with a prefill phase that consumes ONE prompt token through decode_step. prefill_tick returns tokens consumed, primes budget-sized chunks, and uses tokenwise decode_step only as a fallback. The real prefill_tick doc is indented under the stale text.
- fix: Move the step_session header lines to step_session (or delete them) and keep only the prefill_tick doc: chunked prime under the budget, tokenwise fallback.

### low: stage_grammar_mask's doc block is attached to eager_only_model

- where: `crates/memra-server/src/worker.rs:38861` (S18, confirmed)
- detail: The six-line grammar-mask doc sits directly above the eager-only doc, so rustdoc for eager_only_model opens with mask staging and stage_grammar_mask (38932) has no doc.
- check: The six-line GRAMMAR MASK STAGING doc runs straight into the EAGER-ONLY doc above eager_only_model, so rustdoc attaches both to eager_only_model. stage_grammar_mask has no doc comment.
- fix: Move lines 38861-38866 directly above stage_grammar_mask.

### low: advance_token_emit doc says greedy-only, but the async chain feeds it sampled device draws

- where: `crates/memra-server/src/worker.rs:39047` (S18, confirmed)
- detail: step_session_async_chain (39644) passes seeded device-sampled ids from decode_step_chain through advance_token_emit.
- check: step_session_async_chain builds a device sample meta (async_chain_devsample) and passes it to decode_step_chain. FLAGS.md says sampled decode draws inside the chain. Those sampled ids reach advance_token_emit, so 'greedy, no sampling' is wrong for this caller.
- fix: Say the token is device-produced (argmax or seeded device draw) and the host does not sample it again.

### low: pre_tokenizer_is_byte_level's doc sits above collect_split_regexes

- where: `crates/memra-tokenizer/src/lib.rs:1544` (S21, confirmed)
- detail: Lines 1544-1545 document pre_tokenizer_is_byte_level (defined at 1594 with no doc) but are attached to collect_split_regexes (1549), whose rustdoc now starts with the wrong sentence.
- check: Lines 1544-1545 describe pre_tokenizer_is_byte_level but sit in the same /// block directly above fn collect_split_regexes. collect_split_regexes's rustdoc therefore opens with the wrong sentence, and pre_tokenizer_is_byte_level at line 1594 has no doc.
- fix: Move lines 1544-1545 to just above fn pre_tokenizer_is_byte_level.

### low: FLAGS.md gives MEMRA_MMQ_X_Q45K a default of 64; the kernel default is 128

- where: `docs/FLAGS.md:786` (S22, confirmed)
- detail: mmq_q45k.cu defines MMQ_X 128 when unset, and build.rs:385-386 describes =64 as the rebuild that changes the tile. The catalog row states the opposite default.
- check: mmq_q45k.cu defines MMQ_X 128 when it is not set, and build.rs passes -DMMQ_X only when the env is set, describing =64 as the rebuild. The FLAGS.md row says the default is 64.
- fix: Set the row's default to 128 (unset = kernel default).

### low: mma_tile.cuh is not dead: a tool includes it

- where: `docs/KERNELS.md:211` (S14, confirmed)
- detail: No engine TU includes mma_tile.cuh, but tools/dsv4-dense-tc-gate.cu:27 does, so deleting it per the KERNELS note would break that gate.
- check: KERNELS.md calls mma_tile.cuh dead. That is literally true for cu/ and src/, but tools/dsv4-dense-tc-gate.cu includes it, so deleting the header per the note would break that gate.
- fix: Say 'engine-dead; included by tools/dsv4-dense-tc-gate.cu', or move the header next to the tool.

### low: The board labels Qwen3.5-9B and Qwen3.6-27B 'dense', but both compile as qwen35 GDN hybrids with FusedQ gates (Qwen3.8-27B in the same table is labeled 'dense hybrid')

- where: `research/tune-data/current-board.json:99` (scout:packs, confirmed)
- check: The board labels Qwen3.5-9B, Qwen3.6-27B and Ornith-1.0-9B 'dense', while Qwen3.8-27B and Ornith-1.5 use 'dense/MoE hybrid (GDN + gated attention)' for the same qwen35 GDN class. The labels are inconsistent. tools/update-perf-board.py exists.
- fix: Set class to 'dense hybrid (GDN + gated attention)' for Qwen3.5-9B, Qwen3.6-27B and Ornith-1.0-9B and regenerate with tools/update-perf-board.py.

### low: Fast-gate goldens predate the 120a GDN-MMA default flip that changed the qwen35 prefill numeric class

- where: `tools/fast-gate/goldens/q9.tokens:1` (S22, unsure)
- detail: Every golden was minted 2026-08-02..05. gdn_mma_default_on() turned the bf16-HMMA GDN scan on for sm_120a on 2026-08-21 (lib.rs:1122), and that path is a distinct numeric class from the f32 scan. The q9/k27/q35/aw35/kat/o9/o35 goldens therefore pin a pre-flip prefill stream. Either they now false-fail, or the 20-token window is too short to see the class. Both cases make the token pin a weak signal. Not run here.
- check: The dates hold: every golden header is 2026-08-02..05, and gdn_mma_default_on() became ON for sm_120a on 2026-08-21, so the bf16 MMA GDN chunk scan (c == 32) is the default on 120a. Whether the flip changes the first 20 generated ids on these ~100-token probes was not measured, and the finding itself hedges between false-fail and too-weak. No post-flip fast-gate run with these goldens was found to settle it.
- fix: Run the q9/k27/q35-class probes once on current main; refresh goldens at the next full-battery green point and record the defaults generation in the header.

### low: Comment says largest present roster model; code takes the first present row

- where: `tools/release-battery.sh:199` (S22, confirmed)
- detail: Line 204 breaks on the first roster row whose file exists, so with the current roster it is the own Ornith row, not a size-ranked pick. The 9B-backed required cell DUAL-BATCHED-AUX resolves through kc_model's own search, independent of this choice.
- check: The loop breaks on the first non-comment roster row whose file exists. There is no size ranking; with today's roster the first row (Ornith 35B) may also be the largest, but only by row order.
- fix: Change the comment to 'first present roster model', or sort the rows by file size.

### low: serve-smoke header says it defaults to E4B; it defaults to the 9B NVFP4 pair

- where: `tools/serve-smoke.sh:8` (S22, confirmed)
- detail: Line 34 defaults MODEL to Qwen3.5-9B NVFP4 with its draft, and E4B no longer loads through any pack.
- check: The header says it defaults to E4B, but line 34 defaults MODEL to the Qwen3.5-9B NVFP4 GGUF (with its own comment explaining why), and E4B no longer passes pack admission.
- fix: Update the header to the 9B NVFP4 default pair.

## Decoupling needed

### high: moe_ffn_sequential_zq8 puts every MoE family in one 1366-line blast radius

- where: `crates/memra-engine/src/hybrid_forward.rs:13623` (S03, evidence-checked)
- detail: One function holds the softmax pairs/dev arms (qwen3_moe/qwen35_moe/olmoe), the sigmoid host-oracle routing (step35/hy3/m3/glm_dsa), glm5_next-only arms (door D, T=1 device tables, vrows, fused epilogue, trace ledgers), the Hy3 CPU-expert hybrid, and SLRU/slab/worker-I/O spill dispatch. Any edit, even inside a glm5-only arm, makes the impact map mark every MoE family on every decode/verify/prime route and every device.
- cut: Edits to the glm5-only arms (13872-13985, 14181-14316, 14418-14500) would then drop qwen3_moe, qwen35_moe, olmoe, minimax_m3, hy3, glm_dsa and step35 cells. Edits to the softmax pairs/dev arms would drop glm5_next, step35, hy3, minimax_m3 and glm_dsa. Edits to the CPU-expert arm would shrink to hy3/spill.
- fix: Split it into a router prologue, a softmax device-dispatch module, a glm5 sigmoid/PRE-clamp module (vrows, door D, fused epi) and the generic per-expert loop, each a separate item.

### high: Process-global statics carry per-model numeric defaults across families

- where: `crates/memra-engine/src/lib.rs:1719` (map, mechanical (assemble.py static-writer pass))
- detail: Mapping found 459 items that write a process-global static (or call a small setter of one). The static is shared by every model the process serves, so a change in one family's loader that writes it reaches every family the process can serve; the map widens those writers to the static's whole reach. The S01 finding below shows the concrete numeric consequence for multi-model processes.
- cut: would remove the cross-family widening of the 459 writer items
- fix: carry per-model defaults on the model (or its plan) instead of process statics; the writers then become family-local and their change reach shrinks to their own families

### medium: Serving engine links the whole reference executor for three small seams

- where: `crates/memra-engine/Cargo.toml:21` (S21, evidence-checked)
- detail: memra-engine's production code uses only ReferenceTensor/ReferenceWeights (qwen4exp_gpu.rs:65), the hidden_trace module (hybrid_forward.rs, glm5_decode_graph.rs) and MiMo constants/helpers. Every other memra-reference item (10k-line executor, speech, oracles) is reached only from tests, bins and memra-cli. The regular dependency makes every reference-executor edit a rebuild of the serving binary, and a file-level impact map links it to all families and routes.
- cut: Edits to memra-reference/src/lib.rs, speech/* and mimo_* oracles leave every serving cell (all families x all routes x all GPUs) and keep only CPU oracle/gate cells; qwen4_exp plain/tp stays reached only through the container crate.
- fix: Move hidden_trace and the ReferenceTensor/Weights containers into a small shared crate (or memra-gguf), and make memra-reference a dev-dependency plus a feature for the mimo diagnostic modules.

### medium: docs/FLAGS.md is a compiled input of memra-engine, so every flag-catalog prose edit reaches numeric on every target

- where: `crates/memra-engine/build.rs:899` (map, checked by replay (history.md row 5ba8a3b11a))
- detail: build.rs generates the boot-time env registry from docs/FLAGS.md (generate_env_registry), which makes the whole catalog a source input; 5ba8a3b11a (Hugging Face ids in prose) classifies as numeric + perf on all four targets. Only the set of MEMRA_* names and the retired-door rows reach the binary.
- cut: removes every FLAGS.md prose-only change from all device cells
- fix: have the gate compare the generated registry (names and retired doors) at base and head instead of the file bytes, or generate the registry into a small tracked file that FLAGS.md edits only touch when a name changes

### medium: glm5_next hyper-connection kernels live in the dsv4 TU and inherit its build flags

- where: `crates/memra-engine/cu/dsv4_gpu.cu:1146` (S16, evidence-checked)
- detail: hyper.rs (the glm5_next HC trunk) calls rowsq_scale, hc_collapse, hc_post, hc_mean, hc_expand, hc_head_pre_m, hc_sinkhorn_m and every hc_pre_fused v1 to v4 kernel from dsv4_gpu.cu. That TU is compiled with -fmad=false, and the research build MEMRA_DSV4_FMAD=1 lifts that for the whole TU (build.rs:668). The build warning says the flag 'forks the CPU oracle' for dsv4, but it also silently forks glm5_next HC numerics. Every dsv4_gpu.cu edit also has to be treated as glm5-reaching unless it is attributed region by region. qwen4_exp (nvfp4_deq_bf16) and mimo_v2 (fp4_gemm, act_quant_fp8) are coupled the same way.
- cut: Edits elsewhere in dsv4_gpu.cu stop reaching glm5_next x {plain,batch,graph,prime,glm5_spec,dflash,tp,pp,vision,prefix_cache}, and MEMRA_DSV4_FMAD stops reaching glm5_next.
- fix: Move the hc_* kernels (1146-1259, 4008-4058, 5718-5779, 6031-7426) to cu/hyper_hc.cu with its own explicit -fmad policy, and the two dequant/GEMM entries to a shared quant TU.

### medium: 45 qwen4exp-only kernels live in the shared kernels.cu unit

- where: `crates/memra-engine/cu/kernels.cu:3293` (S08, evidence-checked)
- detail: stage0 kernels.tsv shows 50 kernels launched from qwen4exp_gpu.rs, 45 of them launched from nowhere else (qsa_*, q4e_*, hc_diet_*, hc_inject_*, qmatvec_nvfp4_modelopt_sel_*, qmatvec_bf16w_*, qwen4exp_route_topk_f32, gdn_scan_step_f32, rms_sigmul_f32 ...), and all 45 sit in cu/kernels.cu, which every build variant compiles and every family's fatbin carries. AGENTS.md routes any .cu change to the full battery, so a qwen4exp-only kernel edit drags every family, route and device into the impact set even though only the qwen4exp gate bins can launch those symbols.
- cut: Edits to those kernels reach only {qwen4_exp} x gate routes instead of every family x route x device; all serving cells of the other 18 families drop out.
- fix: Move the qwen4exp-exclusive kernels to their own unit (cu/qwen4exp.cu) and key the validation plan on that unit.

### medium: Gate-only dc/graph/generate machinery shares the serving decode file

- where: `crates/memra-engine/src/decode.rs:2248` (S05, evidence-checked)
- detail: decode.rs ~1640-3461 (lockstep, decode_step_dc*, graph sessions, generate, generate_with, E4B graph-exec) and graph_update fa_plan/fa_apply are reached only by bins and gates, never by memra-server. It shares decode.rs with decode_step_h, residual_norm_ffn and ffn_swiglu_decode, which serving and spec use. A file-level impact rule therefore pulls every serving route into changes that only touch gate drivers, and the reverse: serving changes in the shared helpers also move these gate paths.
- cut: Removes plain/batch/mtp/dflash/http serving cells from changes to the dc/graph/generate drivers; those changes then select only run-gen and the graph gates per family.
- fix: Move the bin-only drivers into a gate module (for example decode_gate.rs) that depends on the shared helpers.

### medium: Generic sampled-spec accept walk lives in the DFlash file

- where: `crates/memra-engine/src/dflash.rs:1225` (S11, evidence-checked)
- detail: dspark_accept_sampled (with DsparkDraftSample, penalized_verify_rows and rejection_accept_len) is the shared rejection-sampling contract. gemma_spec's sampled burst and glm_spec's DFlash2 accept call it. Because it sits in dflash.rs, any edit to the 8.6k-line DFlash file pulls the gemma4 gemma_spec cells into the impact set, and any edit to the walk forces the whole dflash surface to be re-qualified.
- cut: Removes gemma4_dense and gemma4_moe from the gemma_spec route in dflash.rs's impact set. The walk's own changes reach only dflash, glm5_spec and gemma_spec sampled cells.
- fix: Move the accept walk and DsparkDraftSample into spec.rs or a spec_accept module that both drafters import.

### medium: dsv4_ffi is a shared FFI module for glm5_next, qwen4_exp and mimo_v2

- where: `crates/memra-engine/src/dsv4_ffi.rs:47` (S07, evidence-checked)
- detail: hyper.rs (HybridModel hyper-connection trunk, glm5_next) imports dsv4_ffi as k and calls hc_pre_fused/v2/v3/v4, hc_post, hc_collapse, rowsq_scale and sinkhorn. mla_ffi.rs calls hc_pre_fused_v2/v3. qwen4exp_gpu.rs calls nvfp4_deq_bf16. mimo_source_moe.rs calls act_quant_fp8 and fp4_gemm. ck() is shared too. So any edit to this 2064-line dsv4-named file has to be qualified on glm5_next and qwen4_exp as well.
- cut: Edits to dsv4-only declarations would stop pulling glm5_next (all hyper routes), qwen4_exp and mimo_v2 into the impact set.
- fix: Split the shared HC, nvfp4-deq and act-quant declarations plus ck() into a generic hc_ffi/quant_ffi module; keep dsv4-only kernels in dsv4_ffi.rs.

### medium: Spec-route argmax lives in the bin-only dense forward module

- where: `crates/memra-engine/src/forward.rs:170` (S13, evidence-checked)
- detail: forward.rs holds Model::forward (dense f32 reference prefill, bins only) next to argmax/top2, which the MTP, dflash, gemma_spec, glm5_spec, eagle and decode generate loops call. Any edit to the reference forward therefore maps to every greedy spec route.
- cut: Edits to Model::forward/prime_gate_verdict would reach no serving cell (bins only) instead of plain+mtp+dflash+gemma_spec+glm5_spec for all families
- fix: Move argmax/top2 into a small sampling util module; leave Model::forward and prime_gate_verdict in forward.rs.

### medium: qwen prime graph's safety census lives in the glm5 decode-graph module

- where: `crates/memra-engine/src/glm5_decode_graph.rs:643` (S11, evidence-checked)
- detail: qwen_prime_graph::run gates its carried-prime capture on glm5_decode_graph::graph_census. glm5_tp_sym_graph also imports CapCtx, step and capture_error from here. An edit to the glm5 decode-graph file therefore reaches qwen35 27B prime on the desktop RTX 5090 (120a, 170 SMs), and the glm5 TP path.
- cut: Drops qwen35 x prime/graph x rtx5090(desktop) from glm5_decode_graph.rs. The file then reaches only glm5_next plain/graph.
- fix: Move GraphCensus/graph_census and the capture helpers (CapCtx, step, capture_one, capture_error) into a shared graph_util module.

### medium: glm5_next hyper walks and glm5 TP/EP MoE share a file with the serial trunk

- where: `crates/memra-engine/src/hybrid_forward.rs:2171` (S03, evidence-checked)
- detail: Lines 2167-5175 (mHC prime/decode/ppN/TP walks) and 15006-16407 (glm5 TP/EP MoE) are glm5_next-only, but the file default must cover every HybridModel family. Any path-level gate or reviewer reading the file sees a 13-family blast radius for glm5-only edits.
- cut: glm5-only edits stop marking qwen3/qwen35/olmoe/minimax_m3/hy3/gemma4_*/step35/llama_dense/glm_dsa at file granularity.
- fix: Move the hc walks to hyper_forward.rs and the glm5 TP/EP MoE to glm5_moe_tp.rs.

### medium: Step TP/EP MoE runtime (about 960 lines) is inlined in the shared MoE entry

- where: `crates/memra-engine/src/hybrid_forward.rs:12174` (S03, evidence-checked)
- detail: The step_ep/step_tp arm (12174-13135) only runs for sigmoid-router models loaded with MEMRA_STEP_TP/EP or auto-parallel on PRO 6000 pairs, but it sits inside moe_ffn_inner, which every MoE family's decode, verify and prime call. Item-level impact cannot separate it from the softmax families.
- cut: Edits to the Step TP/EP runtime would no longer reach qwen3_moe, qwen35_moe, olmoe or glm5_next, and devices would narrow to rtxpro6000 and routes to tp/ep.
- fix: Move the arm into a step_parallel MoE function called from a one-line guard.

### medium: step37 serving defaults arm W8 mirror kernels for every model in the process

- where: `crates/memra-engine/src/lib.rs:1577` (S14, evidence-checked)
- detail: arm_step37_serving_defaults sets a process-global atomic; step_tp_w8_on()/w8_hybrid_on() read it as their unset default. The W8 routing sits in generic Engine::matvec_bf16_rows_into (lib.rs:26515-26533), so in a multi-model process that also loads a step35 model, every BF16 matvec of another family (glm5_next, hy3, qwen35 BF16, ...) with in_f%32==0 and out_f>=64 is rerouted through encode_q8_0_rows_from_bf16 + qmatvec_q8_0_rows_t*/qmatvec_q8_0_mmvq_rp, a different numeric class (int8 dp4a vs bf16 fma), depending on load order. This also forces the impact map to put every BF16 family on the W8 kernels (qmatvec.cu 12932-13469).
- cut: Removes all non-step35 BF16 families x {encode_q8_0_rows_from_bf16, qmatvec_q8_0_rows_t/tw/tw32, W8 mirror qmatvec_q8_0_mmvq_rp} cells from the default impact set.
- fix: Key the W8 defaults on the model's plan (store the armed flag on the HybridModel/plan and pass it to the matvec call), not on a process-global atomic.

### medium: step37 serving defaults arm process-wide and reach generic bf16 GEMV and glm5 KDA paths

- where: `crates/memra-engine/src/lib.rs:1583` (S01, evidence-checked)
- detail: arm_step37_serving_defaults (called at hybrid.rs:4273 for any SlidingGatedMoe plan) sets a process AtomicBool. It also flips memra_kv's SWA ring default. step37_door then turns on MEMRA_W8_HYBRID, MEMRA_STEP_TP_W8, MEMRA_STEP_GEMM_PRIME and MEMRA_Q8T_WONCE for every model in the process. The consumers are family-agnostic: the generic FloatBf16 GEMV routes through a q8_0 mirror when step_tp_w8_on() && w8_hybrid_on() (lib.rs:26220), and glm5 KDA checks the same pair (kda.rs:1850). A step35 model co-loaded with a BF16 hy3/glm5_next/qwen35 model therefore changes the other model's decode numeric class (int8 dp4a mirrors) and VRAM use. The doc admits the per-process scope.
- cut: Every non-step35 family drops the step35-loader and step37-door cells (W8 mirror numeric class, SWA ring) unless it shares a process with step35.
- fix: Keep the armed state on the loaded model (plan flag on HybridModel) and pass it to the door checks, instead of a static.

### medium: B200 f32-GEMV posture door sits in the generic linear_device_into, so every family's f32 linears change class on b200

- where: `crates/memra-engine/src/lib.rs:27522` (S02, evidence-checked)
- detail: The door was motivated by the glm5 MLA/DSA indexer projections (doc 35838-35845). Placing it in linear_device_into (and linear_decode_exact) means every Engine::linear caller takes gemv_f32_rows instead of cuBLASLt by default on sm_100a builds: MoE routers via sigmoid_dot_rows, hyper.rs hc mixes, qwen4exp linears, mimo, vision towers and Float-weight decode. That is a different numeric class. Any change to the door or to gemv_f32_rows therefore reaches every family's b200 cells, not only glm5_next.
- cut: removes b200 x {every family except glm5_next, glm_dsa, deepseek_v4 (hc)} from the impact set of gemv_f32_rows / MEMRA_F32_GEMV_KERNEL changes
- fix: Opt into the native GEMV at the MLA/DSA call sites (a linear_rows variant) and keep linear_device_into on cuBLASLt.

### medium: Qualification-only banked-expert machinery sits inside the default spill file

- where: `crates/memra-engine/src/moe_cache.rs:1281` (S12, evidence-checked)
- detail: About 560 lines of moe_cache.rs (1273-1835, plus the banked arms in dispatch_source_unclocked, admit, restage_block and Drop) serve only the experts-via-tier gate, which is pinned to one qwen35_moe IQ4_XS artifact. They share a file with the default SLRU dispatch that every spill-regime MoE family takes, so a file-level impact map charges every MoE family's spill cell for a gate-only edit. A moe_cache/banked_native.rs submodule already exists (moe_cache.rs:3326).
- cut: Edits to the banked code then reach only qwen35_moe x spill x IQ4_XS gate cells instead of 10 MoE families x spill on every GPU.
- fix: Move the banked methods into moe_cache/banked_native.rs behind an impl block and keep only the one-line state check in dispatch_source_unclocked.

### medium: Generic cache allocation and checkpoint restore live in the PP module and read step-TP state

- where: `crates/memra-engine/src/pp.rs:3661` (S09, evidence-checked)
- detail: new_cache/new_cache_planned/new_cache_inner, layer_engine, sync_stages_after_load, tp_restore_plan and restore_cache_checkpoint run for every family on every admit and prefix restore. restore_cache_checkpoint also reads MEMRA_STEP_TP_GRAPH (line 3694) and step TP mirrors. So any edit to pp.rs, including PP-only wave/dual/peer-probe code, hashes into every family x prime/prefix_cache/batch/mtp cell.
- cut: Edits to the remaining pp.rs would reach only route pp (PP-capable families: step35, glm5_next, hy3, qwen3, llama_dense, gemma4_dense, gemma4_moe) instead of all families x all routes.
- fix: Move restore_cache_checkpoint, tp_restore_plan and new_cache* into cache.rs (or a cache_restore module) that calls a narrow PP hook for stage engines.

### medium: The 16k-line step TP runtime also hosts thread-local seams read on every decode for every family

- where: `crates/memra-engine/src/tp.rs:60` (S09, evidence-checked)
- detail: decode_step_h and decode_step_chain call tp::oproj_tail_scope and tp::rank0_merge_on (decode.rs:820/824/1119), and residual_norm_ffn calls take_oproj_tail for all families. Any tp.rs edit therefore looks like it reaches plain/batch decode on every family and device, although the runtime itself is PRO-6000 multi-card only.
- cut: tp.rs edits drop from all families x plain/batch x all devices to the tp/ep routes on rtxpro6000 only.
- fix: Move the TLS seams (oproj tail, verify tcol, spec_fa2 defer, tcol oproj) and the door parsers into a small tp_seams.rs. Keep tp.rs for the TpE4m3HostBounce runtime.

### medium: Family-agnostic overlay machinery shares a file with the qwen-VL tower

- where: `crates/memra-engine/src/vision.rs:98` (S13, evidence-checked)
- detail: EmbedOverlay, splice_into, new_published and the publish door serve gemma4, glm5_next, step35 and qwen vision. VisionTower is qwen35/qwen35_moe only. File-level mapping forces all six vision families to re-qualify on a qwen tower edit.
- cut: VisionTower edits drop gemma4_dense, gemma4_moe, glm5_next, step35 vision cells
- fix: Split EmbedOverlay plus OverlayPublish into vision_overlay.rs.

### medium: 825-line from_hf holds every family's HF normalization

- where: `crates/memra-gguf/src/config.rs:1773` (S20, evidence-checked)
- detail: Per-family sub-config parsing (m3, hy3, step35, dsv4, qwen4_exp, glm5, mimo, gemma4) lives in one function and shares locals (nextn is mutated for dsv4, n_head is overridden for step35). AGENTS.md assigns config normalization to the pack, yet a family-local fix here is textually a change to the config builder of every family.
- cut: A pack-local config fix would stop reaching other safetensors families' cells.
- fix: Give each pack a normalize(HfConfig) -> sub-config hook and keep from_hf to the shared scalar fields.

### medium: Serving loader lives inside the 2188-line CPU oracle file

- where: `crates/memra-gguf/src/dsv4_forward.rs:501` (S07, evidence-checked)
- detail: Dsv4Gpu (the served dsv4 route) imports Dsv4Model, precompute_freqs_cis, window/compress_topk_idxs, hc_split_sinkhorn, ActQuantVariant and expert_arm_native from dsv4_forward.rs. Everything from L736 on, and the CPU math helpers, reaches only memra-reference and gate bins. Because they share a file, a file-level impact rule must treat any oracle edit as touching every served dsv4 cell.
- cut: Oracle edits (L96-268, L319-451, L635-647, L736-2188) would map to devices [cpu] and routes [] instead of deepseek_v4 on all serving routes and devices.
- fix: Move Dsv4Model, the rope/index builders, hc_split_sinkhorn, ActQuantVariant and expert_arm_native into a dsv4_model.rs; leave dsv4_forward.rs as the oracle.

### medium: plan_sha256 hashes the whole ModelPlan Debug output

- where: `crates/memra-gguf/src/execution_manifest.rs:475` (S20, evidence-checked)
- detail: The receipt-binding hash is the pretty Debug of the full plan, including every family's Option sub-plans (None fields print too) and the hand-written Debug impls (model_plan.rs:49, 370). Adding a field for one family, renaming a variant, or touching a Debug impl changes plan_sha256 for EVERY family, so every installed MEMRA_REWRITE_BUNDLE goes stale and must be re-qualified, even when the program did not change.
- cut: A pack-local plan field change stops invalidating rewrite receipts (policy cells) of every other family.
- fix: Hash a canonical, versioned serialization that omits absent sub-plans, or hash only the fields the rewrite surface consumes.

### medium: Family special cases inside the shared SafetensorsSource::find

- where: `crates/memra-gguf/src/source.rs:2910` (S20, evidence-checked)
- detail: The generic find, which fixes the numeric program (Q8_0/Q5_K/NVFP4 re-encode) for every safetensors family, also holds hy3 key aliasing, MiMo sharded-QKV refusal and the qwen35_moe MTP stacked-expert slice. Any edit for one family is a diff in the function every safetensors cell reads, so the impact map must widen it to all families.
- cut: Edits to hy3/mimo/qwen35_moe source quirks would reach only those families' safetensors cells instead of all of them.
- fix: Move family name/alias hooks into the pack (a pack-provided alias fn), leaving find family-agnostic.

### medium: Vision walker choice is process-global, not per model

- where: `crates/memra-server/src/lib.rs:4899` (S19, evidence-checked)
- detail: step_vision_enabled / gemma_vision_enabled / glm5_vision_enabled are process-wide switches that pick the content walker for every model in MEMRA_MODELS. With MEMRA_STEP_VISION_DIR set, a second (non-step) model's multi-part text content is joined with ' ' (step separator law, 4816-4820) and its image parts are planned as step tiles. With MEMRA_GEMMA_VISION=1, a qwen-VL model's images get gemma placeholders. The worker-side refusal relies on the one-vision-family-per-process convention. Default launches serve two models (main+judge), so multi-model processes are a normal shape.
- cut: step35, gemma4 and glm5 vision door changes stop reaching the prompt-render cells of every other family served in the same process (removes {qwen35, qwen35_moe, all others} x vision/prime from the step/gemma walker impact sets).
- fix: Key the walker on the target model's ModelCaps (a per-model vision-kind cap published by the worker), not on process env.

### medium: Every family dialect lives inside one 429-line chat builder

- where: `crates/memra-server/src/lib.rs:9258` (S19, evidence-checked)
- detail: build_chat_request_with_trace holds the glm5, hy3, dsv4 and gemma parser, stop and effort arms next to the generic path. At function or file granularity the impact map has to treat any edit as reaching all families' chat cells.
- cut: An edit to the glm5/hy3/dsv4/gemma arms then reaches only that family's tools/http_api cells instead of all families.
- fix: Move per-dialect parser/stop/effort selection into a ChatDialect value derived once from ModelCaps (one small impl per dialect), and keep the builder generic.

### medium: The pinned-host KV tier and handoff (about 13k lines) sit inline in worker.rs

- where: `crates/memra-server/src/worker.rs:9536` (S17, evidence-checked)
- detail: Lines 9536-22673 (HostPrefixCache, demote/promote/capture/restore contracts, the hash helper, the handoff codec) are all behind MEMRA_KV_HOST_MB>0, which is default 0. Because they share a file with the default-ON scheduler, the impact map cannot narrow an edit there below the worker.rs file default, so every family, route and device cell lands in the push gate's delta.
- cut: Edits to the moved modules map to kv_host_tier (door default OFF), which removes every default plain/batch/mtp/prefix_cache cell for all 14 HybridModel families from their impact set.
- fix: Move the host tier into worker/host_tier.rs and the handoff codec into worker/handoff.rs, as host_glm.rs already is. Leave only the armed()-gated call sites in worker.rs.

### medium: run() hosts every family's boot attach, so item-level impact is always all families x all routes

- where: `crates/memra-server/src/worker.rs:26651` (S18, evidence-checked)
- detail: Gemma drafter attach (26651-26705), DFlash attach (26730-26819), the four vision tower loads (26870-27060), the host tier (27061-27255) and dsv4 routing all sit inside the one 7k-line run item, which every family's serving reaches. An edit to the gemma attach can only change gemma4 gemma_spec, yet an item-granular gate marks every family.
- cut: A gemma-drafter edit maps to {gemma4_dense, gemma4_moe} x gemma_spec instead of * x *. A dspark-attach edit maps to {qwen35, qwen35_moe, qwen4_exp} x dflash. Vision tower edits map to their own family x vision.
- fix: Move each attach and tower load into its own fn (or route module) that returns the loaded state to run.

### low: dsv4_gpu.cu hosts the hyper-connection kernels glm5_next uses, tying DSv4 kernel and flag edits to glm5_next

- where: `crates/memra-engine/build.rs:514` (S22, evidence-checked)
- detail: hyper.rs (the HybridModel hc trunk for glm5_next) calls memra_dsv4_hc_expand/collapse/post/mean through dsv4_ffi. So the TU-wide -fmad=false / MEMRA_DSV4_FMAD choice, the PDL chain header and any dsv4_gpu.cu edit reach glm5_next as well as deepseek_v4 (and mimo_source_moe.rs).
- cut: A dsv4_gpu.cu-only edit would drop glm5_next (and mimo_v2) cells from its impact set, and an hc kernel edit would drop deepseek_v4's dsv4-only dense/MoE/C4 cells.
- fix: Move the shared hc_* kernels into their own TU (hc_kernels.cu) with an explicit numeric-flag contract.

### low: Process-global PDL switch defined in the dsv4 TU drives every static-lib launcher

- where: `crates/memra-engine/cu/dsv4_gpu.cu:8515` (S16, evidence-checked)
- detail: Dsv4Gpu toggles memra_pdl_chain_on (dsv4_gpu.rs:3559), and memra_chain_launch reads it in tp_ar.cu (glm5 TP transport), moe_f16_grouped.cu (all-MoE grouped prefill) and dsv4_sampler.cu. With dsv4 and a HybridModel in one server process, the dsv4 setting changes launch scheduling of the other model's kernels. Bodies are unchanged by contract (speed only), but the coupling widens dsv4's speed blast radius.
- cut: Removes glm5_next/MoE-family x speed cells from dsv4_gpu.rs PDL changes.
- fix: Move the flag to a neutral TU and make it per-stream or per-engine, not process-global.

### low: qwen4_exp's 46-kernel eager program lives inside the shared engine fatbin

- where: `crates/memra-engine/cu/kernels.cu:3228` (S15, evidence-checked)
- detail: Lines 3227-5792 of kernels.cu are qwen4_exp-only, yet they share MEMRA_ENGINE_FATBIN, which every family loads. A file-granular gate must treat any kernels.cu edit as touching every family. The line-level map already narrows it, but tools/validation_plan.py and fast-gate key on files.
- cut: kernels.cu edits stop mapping to qwen4_exp cells, and qwen4exp.cu edits map to qwen4_exp (plain, tp) only, removing about 18 families x all routes from the file-level impact of either side.
- fix: Move them to cu/qwen4exp.cu with its own fatbin env, loaded by Qwen4ExpGpu.

### low: The dsv4 PDL switch is one process global that also drives tp_ar and moe_f16_grouped launches

- where: `crates/memra-engine/cu/memra_pdl_chain.cuh:31` (S15, evidence-checked)
- detail: memra_pdl_chain_on is set by Dsv4Gpu::load_placed (dsv4_gpu.rs:3559) but read by every memra_chain_launch, including the tp_ar.cu all-reduce kernels used by glm5/step TP and moe_kq_m1_stream_kernel. When a dsv4 model is co-served, PDL turns on for other families' TP/MoE kernels. That changes scheduling only (the entry waits), but it couples dsv4 config to other families' speed cells.
- cut: dsv4 PDL configuration edits would no longer reach glm5_next/step35 tp and ep speed cells.
- fix: Pass the PDL flag per launch from the owning runtime (dsv4 vs TP runtime) instead of a shared global.

### low: kernel-check re-derives gdn_mma_default_on inline because the helper is pub(crate). If the engine default changes (for example to add 100a), the gate's GDN mma pin silently stops matching the served default.

- where: `crates/memra-engine/src/bin/kernel_check.rs:1889` (scout:devices, confirmed)
- check: kernel_check.rs computes mma_default_arch inline with the same expression as lib.rs gdn_mma_default_on (pub(crate) const fn, so a bin cannot reach it). Both read the same package-wide cfg and env, so they agree today. If the engine default gains an arch, the gate's mma pin will not follow it. The comment at kernel_check.rs:1882 records that this drift already happened once (Hopper-only pin vs 120a default).
- fix: Make gdn_mma_default_on pub (doc-hidden if needed) and call memra_engine::gdn_mma_default_on() in kernel_check.

### low: SM-count card-class thresholds are scattered and disagree by design: fa_sm_count()>=128 (lib.rs:1767, hybrid_forward.rs:27915/28071, lib.rs:32396), sm_count()>=180 (decode.rs:3356) and sm_count()==170 (qwen_prime_graph.rs:101, lib.rs:29866). They are read through four separate caches: fa_sm_count on ordinal 0, Engine::sm_count's process-global OnceLock, batched_variant's own static, and tier_transfer's attribute read. Card class by name is a fifth mechanism (HardwareTarget).

- where: `crates/memra-engine/src/decode.rs:3356` (scout:devices, confirmed)
- check: Every threshold and cache named in the item exists as described: fa_sm_count()>=128/<128 (lib.rs:1767, 32396, hybrid_forward.rs:27915, 28071), sm_count()>=180 (decode.rs:3356), sm_count()==170 (qwen_prime_graph.rs:101, lib.rs:29866). The SM count is read in four places: fa_sm_count's own OnceLock on ordinal 0, Engine::sm_count's process-global OnceLock, a second static SMS inside batched_variant (lib.rs:23680), and tier_transfer.rs:1244. Each threshold is documented as measured on purpose (decode.rs explains the 180 gate), so the different values are not a bug. What is real is the scattering: nothing names the classes, so the rules can drift apart. That is a maintenance hazard, not wrong output today.
- fix: Read the SM count once per Engine into a DeviceClass field and give each threshold a named const (FA_BIG_RIG_SMS=128, GEN_GRAPH_48_SMS=180, QWEN_PRIME_SMS=170).

### low: A step35-named driver is the generic TP attention decode for every ungated plan

- where: `crates/memra-engine/src/decode.rs:3527` (S04, evidence-checked)
- detail: full_attn_decode_pre sends any ungated full-attention plan with step_tp attention (MEMRA_PARALLEL=auto + MEMRA_PARALLEL_TP_ATTENTION) into step35_tp_decode_attn_resident_v2. A change made for Step-3.7 therefore reaches hy3/m3/qwen3_moe/olmoe/dense TP decode, and a name-based reviewer or gate would scope it to step35 only.
- cut: lets step35-only mixer edits (26341-29336 minus the v2 driver) drop the non-step35 tp cells
- fix: Move the v2 driver to tp.rs under a generic name and keep the step35-specific gate handling as a thin wrapper.

### low: dsv4 admission and replay gating depend on generic MoE f16g doors in lib.rs

- where: `crates/memra-engine/src/dsv4_gpu.rs:2949` (S06, evidence-checked)
- detail: validate_matrix_program (2949-2951), full_token_moe_program_admitted (1496-1505) and rows_steps_supported (2674) read crate::moe_f16g_mode, moe_f16g_direct_on, moe_f16g_gu_fuse_on, moe_f16g_tail_on and related functions. Those doors also steer other families' MoE prefill. A default change made for a qwen/step MoE program can make dsv4 refuse to boot, lose full-token replay or lose B-row batching, with no dsv4 file in the diff.
- cut: lib.rs MoE door edits would stop reaching deepseek_v4 policy cells, or the dependency becomes explicit.
- fix: Resolve dsv4's MoE program constants in dsv4_doors (pinned for the matrix program) instead of the shared lib.rs doors, or have the push gate map those lib.rs doors to deepseek_v4 cells.

### low: dsv4 load flips a process-global PDL flag that glm5 TP all-reduce kernels also read

- where: `crates/memra-engine/src/dsv4_gpu.rs:3559` (S06, evidence-checked)
- detail: memra_pdl_chain_on is one C global (dsv4_gpu.cu). It gates the programmatic-serialization attribute for every memra_chain_launch, including tp_ar.cu (used by glm5_tp.rs) and dsv4_sampler.cu (used by glm5_tp_sampler). When a multi-model server loads a dsv4 checkpoint next to a glm5_next TP model, MEMRA_DSV4_PDL changes the glm5 launch schedule. The effect is speed only if every chained kernel opens with MEMRA_PDL_CHAIN_ENTRY.
- cut: Removes glm5_next x tp (speed) from the impact set of dsv4 PDL changes and of load_placed.
- fix: Carry the PDL switch per stream or per model (pass it to the launcher), or key it on MEMRA_PDL alone.

### low: About 5k lines of reference/legacy oracle code share the serving file

- where: `crates/memra-engine/src/dsv4_gpu.rs:6974` (S06, evidence-checked)
- detail: forward_impl, block_forward, compressor, populate_cmp_cache, mtp_logits_last, alloc_step_ws, legacy cmp_decode/block_decode and the non-matrix device step (sink_attn_dec_arm, hc_pre_dev, cmp_decode_dev, block_decode_dev, moe_forward_dev, head_logits_dev) run only after arm_reference_expert_program_for_gate or MEMRA_DSV4_DECODE_PATH=legacy, and the matrix default refuses both. A diff there changes no serving cell, but its file, and the shared kernels it calls, put it in the same blast radius as the hot path.
- cut: Takes the deepseek_v4 serving routes (plain, batch, graph, prime, dflash, tp, ep) out of the impact set of reference-program edits.
- fix: Move the reference and legacy programs into a dsv4_gpu_reference.rs submodule so the impact map can label the whole file gate-only.

### low: Shared sampler types live in the 26k-line DSv4 engine file

- where: `crates/memra-engine/src/dsv4_gpu.rs:24409` (S06, evidence-checked)
- detail: glm5_tp_sampler (glm5_next TP device sampling) and dsv4_sampler import Dsv4SampleCfg, Dsv4PenaltyCfg, dsv4_pos_uniform and full_token_profile_phase from dsv4_gpu.rs. That is the only reason this file reaches glm5_next at all.
- cut: dsv4_gpu.rs becomes deepseek_v4-only; edits to it stop reaching glm5_next x tp.
- fix: Move Dsv4SampleCfg, splitmix64/dsv4_pos_uniform, Dsv4PenaltyCfg and the phase bracket into dsv4_sampler.rs (or a shared sampler module).

### low: glm5_next TP sampling depends on the dsv4 sampler

- where: `crates/memra-engine/src/dsv4_sampler.rs:72` (S07, evidence-checked)
- detail: glm5_tp_sampler.rs builds Dsv4DeviceSampler for its sampled scratch. Any dsv4 sampler change (penalty upload, replay draw, canary) therefore reaches the glm5_next tp route.
- cut: dsv4-specific sampler edits (dsv4_sampler_env, replay arm) would stop pulling glm5_next/tp.
- fix: Move Dsv4DeviceSampler to a model-neutral device_sampler module that both dsv4 and glm5 TP import.

### low: The capacity-keyed mirror admission rule (free >= need + 8 GiB) is duplicated for Q8RP (hybrid.rs:5882), KQRP (hybrid.rs:5948) and PP_F16/Q4F16. It reads free VRAM at load rather than a device class, so the Q4F16 prefill numeric program (labelled UNPINNED, hybrid.rs:6295) depends on free memory at load time.

- where: `crates/memra-engine/src/f16_ffi.rs:135` (scout:devices, confirmed)
- check: The rule `free >= need + (8usize << 30)` appears literally at hybrid.rs:5882 (Q8RP), hybrid.rs:5948 (KQRP) and f16_ffi.rs:135 (pp_f16_capacity_ok, used only by the Q4F16 auto path at hybrid.rs:6283). The boot log at 6316 rebuilds the threshold a fourth time. Q4F16 picks the prefill arithmetic, and the code itself says so ('SELECTS PREFILL ARITHMETIC', 'capacity-keyed auto (UNPINNED)'). So the same card can run a different numeric program depending on free VRAM at load, for example when another process holds memory. This is acknowledged and logged, which keeps it low.
- fix: Add const MIRROR_HEADROOM: usize = 8 << 30 and fn mirror_capacity_ok(free, need) used by all three sites and the log; key the numeric Q4F16 choice on total VRAM or card class rather than free memory.

### low: The "is sm_100a build" predicate has two encodings: env!(MEMRA_BUILT_CUDA_ARCH)=="100a" (about 20 sites) and cfg!(memra_sm100_tcgen05) (mla_ffi, mmq_ffi, fp8_ffi). Only a unit test (fp8_ffi.rs:304) keeps them equal. Likewise "120a" is tested by string while Hopper and Ada use cfgs.

- where: `crates/memra-engine/src/fp8_ffi.rs:307` (scout:devices, confirmed)
- check: There really are two encodings, but drift is close to impossible. build.rs emits both from the same cuda_arch variable a few lines apart: the cfg memra_sm100_tcgen05 when cuda_arch == "100a" (line 283), and rustc-env MEMRA_BUILT_CUDA_ARCH={cuda_arch} (line 288). The docs.rs path sets 120a with no cfg, which is also consistent. The unit test pins them. This is a cosmetic consistency issue.
- fix: Optionally add one const fn per arch class (is_sm100_build etc.) and use it everywhere.

### low: Default glm5 plain prime walker sits inside the spec module

- where: `crates/memra-engine/src/glm_spec/prime.rs:587` (S11, evidence-checked)
- detail: Glm5PlainPrimeState/Walker is the worker's default text-only hc prime (saved_hyper), but it lives in glm_spec/prime.rs next to the spec preparation state. A spec-only change to this file therefore marks the default glm5_next prime cells, and a reader of glm_spec assumes default-OFF reach.
- cut: glm_spec/prime.rs becomes glm5_spec-only (default_reached false). Default glm5_next prime/chunked_prefill cells leave its impact set.
- fix: Move Glm5PlainPrimeState/Walker beside Glm5TrunkPrime (hybrid_forward/glm5_prime.rs).

### low: The Step distributed runtime re-implements the PRO 6000 name match inline instead of using parallel::HardwareTarget::from_device_name, a third copy of the card-class rule.

- where: `crates/memra-engine/src/hybrid.rs:225` (scout:devices, confirmed)
- check: hybrid.rs:225 repeats the same rule as HardwareTarget::from_device_name ("RTX PRO 6000" && "Blackwell") inline instead of calling it. Today the logic is equivalent, so there is no behavior bug. It is a duplicate of the card-class rule; tier_transfer's PinnedKind::for_device is a separate name-keyed copy too.
- fix: Replace the inline check with HardwareTarget::from_device_name(name) != Ok(HardwareTarget::RtxPro6000Blackwell) (after making the error type comparable or using matches!).

### low: Prefill arithmetic is keyed on free VRAM at load

- where: `crates/memra-engine/src/hybrid.rs:5985` (S05, evidence-checked)
- detail: The f16 prefill mirrors (a different numeric class, labeled UNPINNED) build only when q8rp_on, and the unset q8rp_on is free-VRAM keyed (free >= mirror mass + 8 GiB, line 5880). The gemma Q4F16 auto arm (6280-6295) is keyed the same way. The same model on the same card therefore runs two prefill programs depending on what else holds VRAM at load: co-loaded models, a vision tower, another process. The impact map cannot name a device cell for these arms.
- cut: Makes the prefill numeric class a function of (family, format, device), so the device column alone selects the cells.
- fix: Key the mirror decisions on total VRAM / card class (deterministic per device), keep free VRAM only as a refusal guard, and print the chosen program as today.

### low: FA split numeric class is keyed on CUDA ordinal 0, not on the engine's device

- where: `crates/memra-engine/src/lib.rs:1839` (S01, evidence-checked)
- detail: fa_sm_count reads the SM count of device 0 for every engine. Engine::sm_count is also a process OnceLock taken from the first engine. On a host with two card classes, the second card picks the first card's FA split ladder. Example: a PRO 6000 at ordinal 1 next to a laptop-class 5090 at 0 gets the 82-SM ladder (a different decode numeric class) while running a single-card model on device 1.
- cut: Lets device cells be keyed on the serving card rather than on whatever sits at ordinal 0.
- fix: Query the SM count from the engine's own context and cache it per ordinal.

### low: b200_posture_door_from is the declared shared B200 default, yet the same parse is copied in q8_row_ilp_on_from (1408), nvfp4_row_ilp_on_from (1450), moe_vrows_ilp_on_from (2609), hc_pre_sink_reg_from (3266), hc_pre_v4_on_from (3317) and kda_conv3_on_from (kda.rs:933), with variants in glm5_vrows_t1_dev_forced_from, hc_pre_block_default and hc_fused_pre_arm_from.

- where: `crates/memra-engine/src/lib.rs:3340` (scout:devices, confirmed)
- check: The same Some("1")/Some("0")/_ => built_arch == "100a" match is repeated in several *_from functions in lib.rs (match arms near 1408, 1450, 2609, 3266, 3317) and in kda.rs around 933, next to the shared b200_posture_door_from. Each door may own its own receipt, so the duplication is a maintenance hazard only, not a behavior bug.
- fix: Route these parses through b200_posture_door_from.

### low: The MEMRA_GDN_MMA env parse plus the portable/c==32 conjunction is copied three times (gdn_mma_enabled at 34003, gdn_scan_chunked at 34675 and 34762). The comment says the three sites must agree, but only the default is shared.

- where: `crates/memra-engine/src/lib.rs:34762` (scout:devices, confirmed)
- check: The full predicate (!portable_mma_gated() && c == 32 && MEMRA_GDN_MMA match {1,0,default}) is written out three times: gdn_mma_enabled (lib.rs:34002), gdn_mma_pre (34670) and gdn_mma (34757). The comment at 34755 says all three must agree. If the two reads in one call ever disagree (a predicate edit at one site, or an env change between reads), the .expect("mma path pre-allocates wb16") at the gdn_mma branch panics. Today they are textually identical.
- fix: Compute let gdn_mma = self.gdn_mma_enabled(c); once at the top of gdn_scan_chunked and use it for both the wb16 pre-allocation and the scan branch.

### low: mmq_ffi mixes dense prefill MMQ with step TP/EP grouped FP8 and MoE grouped lanes

- where: `crates/memra-engine/src/mmq_ffi.rs:437` (S13, evidence-checked)
- detail: ExpertCsr/DeviceExpertCsr/Fp8GroupedWorkspace (tp.rs only), the moe_f16_grouped lane and the dsv4 grouped externs share the file with the dense MMQ dispatch every family uses. A file-level edit maps to all families and routes.
- cut: Grouped-expert edits drop dense families (qwen3, qwen35, gemma4_dense, llama_dense) and non-tp/ep routes
- fix: Move the expert-CSR/grouped-FP8 workspace and moe_f16_grouped into a moe_grouped_ffi.rs module.

### low: Step-TP KV restore runs (and syncs) on every MTP accept for every family

- where: `crates/memra-engine/src/spec.rs:3589` (S10, evidence-checked)
- detail: restore_step_tp_kv_verified_prefix is called on every full accept (14281) and every partial commit (8628). It validates and synchronizes the stream before looking at tp_kv, so non-TP qwen35, qwen35_moe and hy3 rounds pay a host sync and sit in the step-TP restore's blast radius. In the full-accept path the sync lands before the draft-KV refresh is enqueued.
- cut: removes qwen35, qwen35_moe and hy3 MTP cells from step-TP restore changes; only step35 TP (and TP-attention) remains
- fix: Return early when cache.tp_kv has no Some layer, before the synchronize.

### low: Draft-KV fill chunking is keyed on a process-global step37 flag

- where: `crates/memra-engine/src/spec.rs:12532` (S10, evidence-checked)
- detail: swa_ring_on() is armed process-wide by arm_step37_serving_defaults (lib.rs:1584) when any SlidingGatedMoe model loads. In a multi-model process, the qwen35, qwen35_moe and hy3 MTP draft-KV fill (here and prime.rs:338) then switches from MEMRA_PRIME_CHUNK (4096) to prime_chunk_tokens. The fill is exact under chunking, so the effect is speed and memory only, but the step37 default now reaches other families' cells.
- cut: removes non-step35 MTP cells from step37 serving-default changes
- fix: Key the fill chunk on this model's plan (SlidingGatedMoe) instead of the global ring flag.

### low: Device-class policy helpers live in the host-tier contracts transfer file

- where: `crates/memra-engine/src/tier_transfer.rs:102` (S12, evidence-checked)
- detail: kv_vmm_placement_for_device serves the unrelated MEMRA_KV_ALLOCATOR=vmm door (plain/mtp on one device), and PinnedKind::for_device is a card-class policy. Both sit at the top of the transfer engine file, which otherwise belongs to the kv_host_tier door, so edits to either drag the other into the impact set.
- cut: tier_transfer.rs then narrows to kv_host_tier only, and VMM placement edits stop listing host-tier cells.
- fix: Move both into parallel.rs next to HardwareTarget::from_device_name.

### low: Generic MEMRA_PARALLEL=auto W4A16 EP backend is interleaved with step37-specific TP code

- where: `crates/memra-engine/src/tp.rs:11385` (S09, evidence-checked)
- detail: The auto EP path (hy3-class W4A16 artifacts, any routed MoE plan) shares tp.rs with step37 recipes (STEP37_TRUNK_LAYERS, step expert clamp, decode-v2 attention). A step TP attention change marks every auto-EP MoE family cell, and an auto-EP change marks step35 TP cells.
- cut: Step-attention edits stop reaching hy3/other MoE auto-EP cells, and auto-EP edits stop reaching step35 TP attention cells.
- fix: Split the auto EP W4A16/W4A8 walk (11087-12169) and its workspace into ep_w4a16.rs.

### low: Generic host-tier helper lives in the GLM-only module

- where: `crates/memra-server/src/worker/host_glm.rs:13` (S19, evidence-checked)
- detail: read_f32 is the door-OFF heap demote for every family's recurrent/KV planes (worker.rs:9621 HostF32::down), but it sits in host_glm.rs. A file-level map therefore cannot narrow host_glm.rs to glm5_next/glm_dsa.
- cut: host_glm.rs becomes {glm5_next, glm_dsa} x {kv_host_tier, prefix_cache} only, which removes every other family from its impact set.
- fix: Move read_f32 next to HostF32 in worker.rs or a host_tier module.

### low: Any .cu edit, including MiMo diagnostic TUs with no serving route, demands the full battery

- where: `tools/validation_plan.py:1166` (S22, evidence-checked)
- detail: native_scope maps every CUDA source to 'full', so an edit to mimo_audio_codec_rvq.cu or mmq_q8_0_f32acc.cu (research instrument, never linked into serving) requires the 2x PRO 6000 full battery. stage0/kernels.tsv already has per-kernel launch sites that would let a TU-to-family owner table narrow this.
- cut: A mimo_*.cu or accprobe TU edit would remove every serving family/route cell (only mimo_v2 probes remain), and a dsv4-only TU would remove the non-dsv4/glm5 families.
- fix: Add a TU owner table (TU -> families/routes, generated from kernels.tsv/ffi edges) and scope native obligations from it, falling back to full for shared TUs and headers.

## Mergeable

### medium: Two copies of the PP-2 scoped-thread prime pipeline

- where: `crates/memra-engine/src/hybrid_forward.rs:4639` (S03, evidence-checked)
- detail: prime_cache_hyper_pp2_pipelined and prime_cache_pp2_pipelined (5966) have the same fence/prewarm, PrimeCacheStages split, thread::scope stage-0 spawn, stage-1 + tail, publish, odometer and fence-behind order. They differ only in the stage enqueue closures, the tail (hyper_prime_tail vs prime_chunk_epilogue) and the payload width. A fix to one, such as the missing cancel point, has to be made twice.
- fix: Extract one generic pp2 pipelined driver that takes stage0/stage1/tail closures and the payload width.

### medium: forward/forward_last/prime_chunk_captured repeat the serial layer and dense FFN, and have drifted

- where: `crates/memra-engine/src/hybrid_forward.rs:5389` (S03, evidence-checked)
- detail: The dense FFN block appears three times (5227-5258, 5362-5392, 7701-7734) next to dense_ffn_with (2076), which already implements it with the prefill-phase A4 program. The copies disagree: forward uses matmul_prefill for down but plain matmul_group for gate/up, forward_last and prime_chunk_captured use plain matmul, and prime_layers uses matmul_group_prefill_xh + matmul_prefill. For an A4-stamped qwen35 artifact, forward_last (the run_gen oracle) and prime_cache compute different programs, while the doc at 5276 claims forward_last is bit-identical to forward()[last].
- fix: Have the three stateless walks call dense_ffn_with(.., prefill=true) and drop the duplicated blocks; fix or remove the bit-identity sentence.

### medium: Two concat-prime batch formers (interactive and dark) duplicate filters and error handling and have already drifted

- where: `crates/memra-server/src/worker.rs:31728` (S18, evidence-checked)
- detail: 31004-31223 and 31728-31861 each build a candidate list, call prime_cache_batch, split outputs and handle the tainted vs pre-mutation refusal. The dark copy lost the CARRIED_PRIME guard (high finding above) and the choice exclusion.
- fix: One candidate predicate plus one batch-prime executor parameterized by lane budget.

### medium: step_session's prefill arm is a hand-kept twin of prefill_tick

- where: `crates/memra-server/src/worker.rs:40090` (S18, evidence-checked)
- detail: Both implement the PRIME_MIN_T floor, the tail merge, the monolithic/carried veto, the suffix-prime receipts and the capture fallback (comments say 'the prefill_tick twin'). The twin already lacks the bound_rem boundary stops, the vision overlay, the glm5 plain prime walker and the cancel scope. A change to one prime law needs two edits, and the impact map must mark both paths.
- fix: Have step_session's prefill arm call prefill_tick with budget PREFILL_TICK_T.

### low: Three gated RMSNorm kernels and two q8_1 epilogues repeat the same bodies

- where: `crates/memra-engine/cu/hybrid.cu:1190` (S16, evidence-checked)
- detail: gated_rmsnorm_f32, _f16out_f32 and _q8_1 copy the same block reduce. swiglu_preclamped_mul_scaled_q8_1_f32 and gated_rmsnorm_q8_1 copy quantize_q8_1's warp amax chain. kda.cu has a sigmoid-gated twin. One templated reduce plus an epilogue functor would keep the bytes and remove the drift risk between copies.
- fix: Template the shared reduce and epilogue; keep the bitwise gate tests.

### low: Ten RMS-norm + quantize-epilogue variants share one prologue

- where: `crates/memra-engine/cu/kernels.cu:700` (S15, evidence-checked)
- detail: rms_norm_q8_1, rms_norm_zq8_f32(_v2), add_rms_norm_q8_1, add_rms_norm_zq8, rms_pre_add_rms_norm_q8z_f32, rms_pre_add_q8_1_f32, add_scale_rms_norm_q8_1, rms_pre_add_scale_rms_norm_q8_1 and add_rms_norm3_q8z_f32 repeat the add/reduce/scale prologue with q8_1 or zq8 epilogues (shingle similarity 0.75-1.00).
- fix: One templated kernel over {pre-norm, add, scale, n_outputs, epilogue} with the current names as thin instantiations.

### low: Five private copies of the q8_1 MMQ activation quantizer

- where: `crates/memra-engine/cu/mmq_nvfp4_w4a8.cu:754` (S14, evidence-checked)
- detail: quantize_mmq_q8_1_d4_kernel (mmq_nvfp4_w4a8.cu:754, mmq_iq_experts.cu:570), quantize_mmq_q8_1_d4_q8_0 (mmq_q8_0.cu:322), quantize_mmq_q8_1_d4_q4_0 (mmq_q4_0.cu:515) and the ds4 variant (mmq_q45k.cu:433) are near-identical. A quantizer fix has to be made five times and each copy is its own impact cell.
- fix: Host one templated quantizer (d4/ds4 layout parameter) in mmq_common.cuh.

### low: vec_dot_q8_0_q8_1_mma is token-identical in mmq_q4_0.cu and mmq_q8_0.cu

- where: `crates/memra-engine/cu/mmq_q4_0.cu:147` (S14, evidence-checked)
- detail: Both TUs already include mmq_mma_i8.cuh; the q45k and w4a8 dot bodies differ only by scale type (half2 dm vs float d).
- fix: Move vec_dot_q8_0_q8_1_mma into mmq_mma_i8.cuh (SASS-identity gated like the earlier dedup).

### low: tw/tw32 and tcols/tcols16 pairs differ only in the accumulator bound

- where: `crates/memra-engine/cu/qmatvec.cu:13216` (S14, evidence-checked)
- detail: qmatvec_q8_0_qkv_rp_tw vs _tw32, qmatvec_q8_0_b4_rp_tw vs _tw32, matvec_bf16_f32acc_x4_tcols vs tcols16 differ only in MEMRA_Q8T_TMAX vs _WIDE / TCOLS_MAX vs TCOLS16_MAX (diffed). The split exists for register footprint, which a template<int TMAX> keeps.
- fix: One template body per pair, instantiated at both bounds.

### low: attn_in_norm_mixer triplet and the TG_PROBE dump block are copied across eager, dc and dc_cap walks

- where: `crates/memra-engine/src/decode.rs:942` (S05, evidence-checked)
- detail: attn_in_norm_mixer/_dc/_dc_cap share the norm+fuse skeleton, and the same MEMRA_TG_PROBE_LAYER block that appends to /root/eager-probe-*.bin is pasted into decode_step_h, decode_step_dc and decode_step_dc_cap_masked. The copies have already drifted: the dc_cap entry lost the gemma and PP guards.
- fix: One helper parameterized by the full-attn callee, plus one probe helper.

### low: DFlash2 greedy and sampled proposals duplicate their prelude

- where: `crates/memra-engine/src/dflash.rs:2397` (S11, evidence-checked)
- detail: dflash2_propose_greedy_q (2367) and dflash2_propose_sampled (2422) repeat the geometry check, d2t length check, top-k, hidden_proj matmul, two DtoH reads, the candidate guard and the d2t remap verbatim. Only the final walk differs.
- fix: Extract a dflash2_candidates(e, dl, rows, nd, n_vocab, d2t, ctx) helper that returns (unary, cand, hproj).

### low: Admission-debt and its peek twin duplicate the door logic

- where: `crates/memra-engine/src/dflash.rs:5672` (S11, evidence-checked)
- detail: Both functions recompute the dspark/MTP door predicate and the reserved-bytes read, and differ only in admission_debt vs admission_debt_peek on the pool.
- fix: Use one fn with an observe: bool parameter, or share a door-check helper.

### low: dots_m_dev and dspark_head_dots are the same function

- where: `crates/memra-engine/src/dsv4_gpu.rs:16397` (S06, evidence-checked)
- detail: Both call memra_dsv4_dots_f32acc_mrow or memra_dsv4_dots_f32_mrow with identical arguments; only the selecting bool differs (dots_f32 vs dspark_head_f32).
- fix: Make one helper that takes the f32acc bool and have both callers use it.

### low: MEMRA_DSV4_SPEC_DEPTH parsed twice with different rules

- where: `crates/memra-engine/src/dsv4_gpu.rs:23328` (S06, evidence-checked)
- detail: The engine bench entry (spec_greedy_batched_with) quietly maps a non-numeric value to 'no cap'. dsv4_serve (line 159) refuses it, so the same env value gives different policies on the bench and serving paths.
- fix: Share dsv4_serve's strict resolver (move it into the engine) and use it in spec_greedy_batched_with.

### low: Four copies of the matrix projection launcher

- where: `crates/memra-engine/src/dsv4_grouped.rs:860` (S07, evidence-checked)
- detail: project, project_stream, project_mrow and project_m1 repeat the same table-length/output/mode guard (moe_f16g_mode, dsv4_moe_f16g_sk_params, moe_f16g_direct_on) and the same pointer marshalling. They differ only in the kernel symbol and a few arguments. A guard fix applied to one copy can silently miss another.
- fix: One project_with(kind: ProjectKind) that does the guard once and dispatches the kernel.

### low: upload_step_bf16_column and upload_step_bf16_row are copies

- where: `crates/memra-engine/src/hybrid.rs:1595` (S05, evidence-checked)
- detail: The two functions differ only in return type and the two runtime upload calls.
- fix: One generic helper taking the two upload closures.

### low: Shared-expert epilogue copied three times beside moe_shexp_compute_with/apply

- where: `crates/memra-engine/src/hybrid_forward.rs:18676` (S04, evidence-checked)
- detail: moe_ffn_pairs (18132-18153, 18265-18314) and moe_ffn_dev (18676-18725) each re-implement gate/up matmul, SiLU, down and sigmoid_dot_rows gate/add_scaled_rows. That is the same program as moe_shexp_compute_with + moe_shexp_apply (16899-17033), whose doc claims byte identity with these arms. Only moe_ffn_dev's copy has the spec_fused_t fused2 arm (t=2..4), which compute_with lacks. Three copies mean a change to the shexp numeric class has to be made three times or the arms drift apart, against the one-numeric-program law.
- cut: none
- fix: Move the spec_fused_t arm into moe_shexp_compute_with and call moe_shexp_add from pairs and dev.

### low: Resident slot-pointer collection repeated in gdec f32, gdec q8 and the fused-epilogue pass 2

- where: `crates/memra-engine/src/hybrid_forward.rs:19724` (S04, evidence-checked)
- detail: moe_gdec_token, moe_gdec_token_q8 and moe_fused_epi_token_q8 each walk sel, look up three BlockIds, take device_ptr of each slot into g/u/d [u64;8], and bail on a miss. moe_cached_gemm and moe_cached_gemm_q8 are likewise identical except for the matvec call.
- cut: none
- fix: Extract resident_ptrs8(c, il, sel) -> Option<([u64;8];3)> and a cached_gemm(kernel closure) helper.

### low: gemma4_moe_q8 verify-rows arm duplicates gemma4_moe's

- where: `crates/memra-engine/src/hybrid_forward.rs:22514` (S04, evidence-checked)
- detail: 22514-22567 and 22657-22709 are the same CSR/rows gate_up GELU + rows_g down chain, differing only in where zq/zd come from (pre-quantized vs quantize_q8_1(moe_in)). The dispatch-parity law relies on both staying identical.
- cut: none
- fix: Have gemma4_moe quantize and call gemma4_moe_q8 for t < PRIME_MIN_T.

### low: step35 verify vs batch FA rows join/pass are near-copies

- where: `crates/memra-engine/src/hybrid_forward.rs:28017` (S04, evidence-checked)
- detail: step35_verify_fa_rows_join is step35_batch_fa_rows_join with one cache and len_back = t-1-r (28017 vs 28479); the verify/batch rope_fa_pass pair likewise. The shingle score is 0.71.
- cut: none
- fix: One rows_join(caches, row_to_cache, len_back fn) used by both walks.

### low: Five copies of the 100a tri-state door parser

- where: `crates/memra-engine/src/lib.rs:1404` (S01, evidence-checked)
- detail: q8_row_ilp_on_from, nvfp4_row_ilp_on_from (L1446), moe_vrows_ilp_on_from (L2605), hc_pre_sink_reg_from (L3262) and hc_pre_v4_on_from (L3313) have the same body as b200_posture_door_from (L3336). A per-device default change has to be made five times, and drift between them changes B200 cells silently.
- fix: Make them call b200_posture_door_from (or one arch_default_door(v, arch, armed_on)).

### low: Launch wrappers that differ only in the kernel name

- where: `crates/memra-engine/src/lib.rs:9479` (S01, evidence-checked)
- detail: moe_pairs_matvec_q8_dec and moe_pairs_matvec_q8_em (L9419) differ in the kernel string only. So do qmatvec_nvfp4_q8_ep_dual_slots_into (L12603) and qmatvec_nvfp4_q8_ep_paired_slots_into (L12679), apart from one error string. The gemma gelu dev_q8_rows/csr pair (L9660/L9715) differs by one index buffer.
- fix: One wrapper per shape taking the kernel name (or an enum).

### low: absorb_q and decompress_v repeat the same four-door dispatch

- where: `crates/memra-engine/src/mla_ffi.rs:1576` (S13, evidence-checked)
- detail: mla_absorb_q (1553) and mla_decompress_v (1664) are ~100% similar. Each runs coalesce, B200 split, generic split, then shipped, differing only in the kernel symbols and out_dim. Any door fix must be applied twice.
- fix: One helper taking the three kernel fn pointers plus out_dim and kind name.

### low: Block-FP8 MMQ slice and view launchers duplicate the B200 refusal and shape checks

- where: `crates/memra-engine/src/mmq_ffi.rs:2357` (S13, evidence-checked)
- detail: qmatvec_mmq_fp8_blk_scaled and qmatvec_mmq_fp8_blk_view repeat the same refusal text, asserts and launch. A third copy of the B200 MEMRA_FP8_MMQ policy lives in fp8_ffi.rs fp8_blk_mmq_native_policy.
- fix: Have the slice form build views and delegate to the view form (with a scale parameter); read the B200 policy from one function.

### low: FP8 and NVFP4 expert checkpoint validators share their walk

- where: `crates/memra-engine/src/parallel.rs:1149` (S05, evidence-checked)
- detail: Same per-routed-layer census loop and error shape; only the native finder and the shard rule differ.
- fix: One validator generic over a small format trait.

### low: _at GDN scan launchers copy the base launchers on column views

- where: `crates/memra-engine/src/qwen4exp_gpu.rs:4776` (S08, evidence-checked)
- detail: launch_gdn_scan_step_at equals launch_gdn_scan_step and launch_gdn_scan_at equals launch_gdn_scan (t=1) after slicing the token column; guards, grids and arguments are copied verbatim.
- fix: Slice the column views in one helper and call the base launchers.

### low: three qmatvec_bf16w_f32 launchers differ only in which operands are views

- where: `crates/memra-engine/src/qwen4exp_gpu.rs:5767` (S08, evidence-checked)
- detail: launch_qmatvec_bf16w, launch_qmatvec_bf16w_off and launch_qmatvec_bf16w_off_into all launch qmatvec_bf16w_f32 with the same argument order; _off and _off_into only pre-slice W (and x/y) and pin strides.
- fix: Keep launch_qmatvec_bf16w and have the two wrappers slice views and call it.

### low: Restored-session constructors duplicate their validation preamble

- where: `crates/memra-engine/src/spec.rs:9593` (S10, evidence-checked)
- detail: spec_session_from_restored_deferred (9489-9519) and spec_session_from_restored_ready (9593-9628) repeat the same ensure_usable, MTP-present, pos==0, cache.pos and ctx-room checks with identical messages before calling the shared _scratch tail.
- fix: Factor a validate_restored(cache, pos, suffix_len, max_ctx) helper used by both.

### low: submit_d2d_capture and submit_d2d_restore duplicate their whole batch skeleton

- where: `crates/memra-engine/src/tier_transfer.rs:2386` (S12, evidence-checked)
- detail: Fault take, empty/overflow checks, receipt scratch, governor reserve, ticket and Entry construction, and the per-item wait/zeroed-wait/delay/digest/copy/digest/early-reader closure are repeated (2386-2440 vs 2590-2629). Only destination resolution, the length rule and fence-at-submit differ. A fix to one (for example the per-batch delay change both comments cite) has to be made twice.
- fix: Extract a shared submit_d2d(ops, epochs, DestKind) helper that takes a destination slice and a fence-at-submit flag.

### low: Three host-canonical routed-expert oracle loops

- where: `crates/memra-engine/src/tp.rs:8082` (S09, evidence-checked)
- detail: run_routed_experts (E4M3 EP), run_routed_experts_nvfp4 and run_tensor_parallel_routes_nvfp4 each implement token x slot -> gate, up, step_expert_activation_host, down, weighted sum, differing only in the bank accessor. Divergence between oracles weakens every gate that compares against them.
- fix: One generic oracle over a trait providing gate/up/down per expert.

### low: One-shot collective family repeats one skeleton four times

- where: `crates/memra-engine/src/tp_ar.rs:636` (S09, evidence-checked)
- detail: all_reduce_1stage, all_reduce_1stage_into, gather_i32 and all_reduce_1stage_hcpost each repeat: two-rank check, per-rank enter_main pointer resolution, alias check, blocks choice and a two-launch loop with rc mapping. all_reduce_1stage is a stage copy plus _into.
- fix: Factor a resolve-and-launch helper taking a per-rank closure for the FFI call, and implement the in-place form as stage copy + _into.

### low: gather_parts / concat_parts_on_root and push_2d / push_2d_i32 are near-copies

- where: `crates/memra-engine/src/tp_transport.rs:952` (S09, evidence-checked)
- detail: concat_parts_on_root is gather_parts with a root-only consumer, with the same three transport arms. push_2d_i32 is push_2d at rows=1 over reinterpreted 4-byte words.
- fix: Express concat as gather with a consumer set {root}, and push_2d_i32 as a generic 4-byte push_2d.

### low: Three copies of the norm +1 fold arm

- where: `crates/memra-gguf/src/hf_mapping.rs:916` (S20, evidence-checked)
- detail: step35 (916-924), qwen35 (944-951) and minimax_m3 gemma-norm (958-966) each do is_plusone_norm + ggml_to_hf -> Transform NormPlusOne. Only the predicate differs; model_plan::norm_weight_transform already states the same family set.
- cut: none
- fix: One arm keyed on a shared 'norm +1 family' predicate that norm_weight_transform also uses.

### low: step35 fixture writers near-duplicated

- where: `crates/memra-gguf/src/micro_gguf.rs:408` (S20, evidence-checked)
- detail: write_step35_meta_only and write_step35_mtp_meta_only share about 92% of their text (test fixtures only).
- cut: none
- fix: One writer parameterized by block range and MTP keys.

### low: LatentKvLayer::snapshot_plane duplicates snapshot_tail + snapshot_plane_at

- where: `crates/memra-kv/src/lib.rs:839` (S21, evidence-checked)
- detail: Lines 843-939 repeat snapshot_tail's len/pool/pools_ready checks and tail-ring copy (957-1021) verbatim, then copy rows/keys as snapshot_plane_at does. Two copies of the drain-boundary invariant can drift; the glm5 kpool GPU test already asserts the composed path equals the eager one.
- fix: Implement snapshot_plane as self.snapshot_plane_at(e, self.snapshot_tail(e)?).

### low: Duplicated nvcc resolver in memra-probe drifts from the engine copy

- where: `crates/memra-probe/build.rs:18` (S22, evidence-checked)
- detail: The copy is deliberate (packaging boundary, documented at lines 13-15), but it already diverged. It reads CUDA_PATH and CUDA_ROOT (line 22) without rerun-if-env-changed for them, and falls back without the engine's warning, so changing CUDA_PATH does not rebuild the probe fatbin.
- fix: Add the two missing rerun-if-env-changed lines, or move both copies into a small shared build-helper file included by path from each crate root.

### low: image_url extraction and data-URI check repeated in four walker arms

- where: `crates/memra-server/src/lib.rs:4924` (S19, evidence-checked)
- detail: The url extraction (string or {url}) and the data: check are copied in the step, glm5, gemma and qwen arms (4826, 4914, 4958, 4996) plus video. The glm5 arm also differs: it omits vision_placement_admits because the published flag folds it in, and that difference is easy to lose in a copy.
- fix: One helper fn media_data_uri(part, key) -> Result<&str, String>, with the placement check passed explicitly.

### low: StopScrubber and SurfaceScrubber are near-copies

- where: `crates/memra-server/src/lib.rs:11988` (S19, evidence-checked)
- detail: lib.rs StopScrubber (11988-12040) and surfaces.rs SurfaceScrubber (652-700) share the same holdback algorithm via partial_stop_suffix. SurfaceScrubber only adds matched-stop reporting. A fix to one surface's stop law must be repeated in the other.
- fix: Keep SurfaceScrubber (a superset) and use it on the chat/completions SSE path too.

### low: Three hand-written SpecSampling builders

- where: `crates/memra-server/src/worker.rs:1078` (S17, evidence-checked)
- detail: spec_sampling_for, glm5_spec_sampling_for and dspark_spec_sampling_for each build the nine-field SpecSampling, and the last two repeat the same penalized predicate (1079-1082 and 1102-1105). The doc of spec_sampling_for names hand-written copies as the way a restored session ends up sampling from a different distribution.
- cut: none
- fix: One builder taking a greedy_penalized flag; the three become thin wrappers.

### low: grid_rewind_* and exact_resume_* are near-copies of one rewind

- where: `crates/memra-server/src/worker.rs:3336` (S17, evidence-checked)
- detail: grid_rewind_plain and exact_resume_plain (and the spec pair) share the checkpoint take, ring-lap check, prompt-prefix check, restore and fed truncation. They differ only in the PRIME_MIN_T floor and the settled arm. Both doors are stale (see the stale_flag finding).
- cut: none
- fix: Keep one resume program once the doors are decided; delete the loser.

### low: Spec burst epilogue and demotion arms are repeated per drafter

- where: `crates/memra-server/src/worker.rs:40539` (S18, evidence-checked)
- detail: spec_visible_prefix + emit_spec_token_events + sampler/generated/fed bookkeeping + stop/context checks repeat in step_session's MTP arm, step_gemma_spec, step_dspark_spec and step_glm5_spec. The three demotion arms (30377-30639) repeat the same greedy/unconstrained/threshold policy.
- fix: Extract one burst-commit helper and one demotion-policy predicate.

### low: d2h_span_batch and h2d_span_batch conformance schedules are 0.98-similar mirrors

- where: `crates/memra-tier/src/conformance/d2h_span.rs:59` (S12, evidence-checked)
- detail: The two test schedules differ only in copy direction. Any rule change must be applied to both.
- fix: Parameterize one schedule by direction.

## Reusable

### medium: The batched small-m tier admission is copied in four dispatch entries

- where: `crates/memra-engine/src/lib.rs:18218` (S01, evidence-checked)
- detail: matmul (L18218-18235), matmul_pre (L18607-18616), matmul_decode_exact (L18753-18759) and matmul_decode_exact_pre (L18824-18836) each restate batched_supports && mmvq_supports && !MEMRA_NO_BATCHED && (m<=4 || b8) && the b16 qtype list. The decode-exact pair uses 2..=16 and the general pair uses small_m_tier_max. The parity law says the m>1 class must equal the m=1 class, and that depends on these copies staying identical. The out_f drift between matmul and matmul_pre (previous finding) shows they already diverge.
- fix: Extract one batched_tier_ok(qtype, m, exact) predicate and one prefill_gemm_ok(w, m) predicate, and call them from all four entries.

### medium: render_prompt_ids is a third copy of the worker prompt render

- where: `crates/memra-server/src/lib.rs:9933` (S19, evidence-checked)
- detail: render_prompt_ids duplicates the worker's prepare render (worker.rs:33415-33465) and has already drifted: it does not publish req.prompt_text = rendered (worker.rs:33453), so the reuse pools' text-extension tier is dead for chat on metered deployments (speed). It also omits the capture EOS append and the vision guard (findings above). The comment at 9930 already records one incident caused by a third copy.
- cut: Removes the duplicate numeric render from lib.rs, so a lib.rs change outside it no longer has to be treated as a prompt-render change for every family.
- fix: Extract one worker::render_and_tokenize(req, tok) -> (ids, rendered) with every post-step, and have both paths call it.

### low: Hand-written 16/32-case M switch tables repeated in six launchers

- where: `crates/memra-engine/cu/dsv4_gpu.cu:4373` (S16, evidence-checked)
- detail: gemv_bf16_m, gemv_fp8_m (twice), gemv_fp8_m_pair, _gated and _pair_gated each spell out DSV4_*_CASE(1..32). A constexpr dispatch helper over std::integer_sequence would generate them.
- fix: Add one dispatch_m<MAX>(m, lambda) helper and use it in all six.

### low: PDL entry macro is copied in three TUs with a predicate different from the shared chain header

- where: `crates/memra-engine/cu/flash_attn.cu:59` (S15, evidence-checked)
- detail: kernels.cu:15, flash_attn.cu:59 and qmatvec.cu:18 each define MEMRA_PDL_ENTRY with the portable exclusion; memra_pdl_chain.cuh:19 keys only on the arch. The divergence is what produces the H100 contract gap above.
- fix: Put one PDL entry header next to memra_pdl_chain.cuh and include it in all four TUs.

### low: Model-source and prompt resolution is reimplemented in run_gen (twice), run_spec and argmax_margin_probe; the probe lacks the repack arm

- where: `crates/memra-engine/src/bin/argmax_margin_probe.rs:87` (S22, evidence-checked)
- detail: run_gen.rs:162-185 and run_spec.rs:134-160 both pick Hy3RepackSource when manifest.json exists, else SafetensorsSource, plus the tokenizer source dir. argmax_margin_probe opens SafetensorsSource unconditionally, so the release argmax gate cannot run a Hy3 repack/overlay directory. Prompt and chat-template handling is likewise copied in run_gen (213-247 and 1091-1131) and run_spec (298-332).
- fix: Add one memra_engine helper (open_model_source -> (source, tok_dir)) plus a shared prompt resolver, and call it from all three bins.

### low: kernel_check re-derives gdn_mma_default_on inline

- where: `crates/memra-engine/src/bin/kernel_check.rs:1889` (S22, evidence-checked)
- detail: The engine's gdn_mma_default_on() is pub(crate), so the gate copies its predicate. If the engine default changes (for example adding 100a), the gate's mma pin silently stops matching the shipped default. That drift was the 2026-08-23 hermes finding the comment records.
- fix: Make gdn_mma_default_on pub and call it from kernel_check.

### low: MEMRA_DFLASH_VERIFY_T parse repeated five times

- where: `crates/memra-engine/src/dflash.rs:6049` (S11, evidence-checked)
- detail: The same parse-default-clamp chain appears at 3685, 4057, 5605, 5811 and 6049, and the 3685 doc says default 8 while the others default to nd+1. A fix to the clamp bug above has to be repeated at each site.
- fix: Add one dflash_vt_cap(nd) helper and use it at every site.

### low: Inline card-name check duplicates HardwareTarget::from_device_name

- where: `crates/memra-engine/src/hybrid.rs:225` (S05, evidence-checked)
- detail: The Step distributed runtime gate re-derives the PRO 6000 card test that parallel.rs:50 already encodes. If one changes (a new edition string), the two disagree.
- fix: Call HardwareTarget::from_device_name and match RtxPro6000Blackwell.

### low: Prime range walker reimplemented per walk

- where: `crates/memra-engine/src/hybrid_forward.rs:2396` (S03, evidence-checked)
- detail: Five loops each re-implement 'walk ranges, copy hidden rows, keep last logits, stamp progress, maybe cancel, maybe rebase taps': hc single-engine (2396), hc ppN (2353), step35 GEMM (5704), serial (5938) and step35 batch. They already disagree on cancellation and on tap rebasing.
- fix: One helper taking a per-range closure that owns hidden-stack assembly, progress, cancellation and tap base.

### low: gdn_hk re-derives the engine's compact-head GDN predicate

- where: `crates/memra-engine/src/hybrid_forward.rs:6532` (S03, evidence-checked)
- detail: The doc says the engine-side hint mirrors this exact formula (chunked && t>=16 && gdn_mma_enabled && num_k*2==num_v). Two copies of a device-keyed predicate is the 'MEMRA_GDN_MMA three read sites' defect class this file cites at 56-57. If they drift, prep emits a layout the scan does not expect.
- fix: Expose one Engine::gdn_compact_heads(t, num_v, num_k) and call it from both sides.

### low: Expert-major CSR build reimplemented in five MoE arms

- where: `crates/memra-engine/src/hybrid_forward.rs:17914` (S04, evidence-checked)
- detail: moe_ffn_pairs (17914-17928), gemma4_moe pairs (22737-22751), moe_ffn_grouped_resident_q8 clamped arm (20149), the glm5 TP split grouped prime (16503-16529, with a counting-sort variant) and the grouped sigmoid prefill each build ex_ids/ex_off/ex_pairs from sel. Their bucket orders must agree for the cross-arm bit-identity claims, and today nothing enforces it.
- cut: none
- fix: One expert_csr(sel, n_expert) -> (ex_ids, ex_off, ex_pairs) helper with a unit test that pins order.

### low: FA big-rig split ladder copied four times as host closures

- where: `crates/memra-engine/src/hybrid_forward.rs:27973` (S04, evidence-checked)
- detail: step35 rows joins and passes (27973, 28169, 28322, 28427) hard-code 16/64/128 at 2048/16384 to size max_ns. That mirrors fa_split_keys's big_rig branch in lib.rs. If the lib.rs ladder changes, these copies silently size the partial pool for the old partition while the kernel uses the new one.
- cut: none
- fix: Call crate::fa_split_keys(t_kv, nkv) (the prechecks already refuse the env overrides) or share a const fn.

### low: kda_conv3_on_from duplicates crate::b200_posture_door_from

- where: `crates/memra-engine/src/kda.rs:929` (S13, evidence-checked)
- detail: Its body is identical to lib.rs:3336 b200_posture_door_from, which kda_onorm_zq8_on already uses. A future change to the B200 posture parse would miss this copy.
- fix: Call crate::b200_posture_door_from and keep kda_conv3_on_from as a thin alias or drop it.

### low: batched_variant keeps its own SM-count OnceLock instead of calling sm_count()

- where: `crates/memra-engine/src/lib.rs:23681` (S02, evidence-checked)
- detail: This duplicates Engine::sm_count (23643), including the 82 fallback. Two process-global caches of the same attribute.
- fix: Use self.sm_count().

### low: GDN MMA predicate is copied three times; the comment says all three must agree

- where: `crates/memra-engine/src/lib.rs:34670` (S02, evidence-checked)
- detail: gdn_mma_enabled (34002), gdn_mma_pre (34670) and gdn_mma (34757) each re-derive portable && c==32 && MEMRA_GDN_MMA || gdn_mma_default_on. If one copy drifts, wb16 pre-allocation and the scan arm disagree: expect("mma path pre-allocates wb16") panics, or the scalar path runs with mma pre-work.
- fix: Compute one `let gdn_mma = self.gdn_mma_enabled(c);` and use it for both.

### low: launch_push duplicates tp.rs launch_tp2_peer_push (same kernel, same args)

- where: `crates/memra-engine/src/qwen4exp_gpu.rs:17265` (S08, evidence-checked)
- detail: Both launch q4e_push_f32 with LaunchConfig::for_num_elems and (src, dst_raw, n as i64). The generic step TP2 join (tp.rs) reuses a qwen4exp-named kernel, so that kernel is a shared mechanism under a family-specific name.
- fix: Keep one launcher in tp.rs (or a p2p module), rename the kernel to a neutral name and call it from both sites.

### low: tp2_enable_p2p reimplements configure_native_p2p

- where: `crates/memra-engine/src/qwen4exp_gpu.rs:17281` (S08, evidence-checked)
- detail: The doc says it holds the configure_native_p2p essentials (cuDeviceCanAccessPeer, cuCtxEnablePeerAccess, pool access); tp.rs:9449 already implements that sequence.
- fix: Expose configure_native_p2p (or a two-engine wrapper) and call it from the qwen4exp TP2 arm.

### low: graph_draft and fill-chunk predicates are re-derived in three places

- where: `crates/memra-engine/src/spec/prime.rs:349` (S10, evidence-checked)
- detail: The five-term graph_draft conjunction is written out in prime.rs:349, spec.rs:10660 and spec.rs:10727. The fill-chunk derivation (swa_ring_on, MEMRA_PRIME_CHUNK, 0 means monolithic, CUDA_GRID_YZ_MAX cap) is repeated in prime.rs:338 and spec.rs:12532. A future edit to one copy changes the walker and inline prime programs differently.
- fix: Add HybridModel::spec_graph_draft_allowed(k) and a shared draft_fill_chunk(tp) helper.

### low: select_activation is tested but the engine re-implements the lookup

- where: `crates/memra-gguf/src/model_packs/qwen35/activation.rs:195` (S20, evidence-checked)
- detail: The doc claims runtime dispatch uses select_activation, but only its unit tests call it. GpuTensor load paths in model.rs:760 and model.rs:781 repeat the scales().get(name) + slot() lookup by hand, twice. So the policy the pack tests prove is not the code that stamps weights.
- cut: none
- fix: Have both model.rs stamping sites call one pack helper (or select_activation) and delete the duplicate.

### low: About 30 copies of the read-once env-door OnceLock pattern

- where: `crates/memra-server/src/worker.rs:693` (S17, evidence-checked)
- detail: Each door repeats `static ON: OnceLock<bool>; get_or_init(|| env::var(X).as_deref() == Ok("1")` or `!= Ok("0")`, and some add a one-time eprintln (691, 832, 873, 931, 960, 1035, 1045, 2584, 2652, 2745, 3568, 3588, 3612, 3659, 3672, 6095, 6145, 6169, 6197, 6204, 6597, 15723, 15904, 15952...). A shared helper would make the door census mechanical and keep strict 0/1 parsing consistent; today some doors refuse a typo and others treat it as OFF.
- cut: none
- fix: Add a door_on!/door_off! helper with a default and a strictness flag.

### low: Spec-program membership predicate copied with drift

- where: `crates/memra-server/src/worker.rs:31392` (S18, evidence-checked)
- detail: `spec || gspec_k>0 || dspark_on || glm5_on` is spelled out at 31254, 31872 and 30661. The eager decode loop (31392-31396), the batched decoding filter (31434-31438) and the async-chain eligibility (39557-39559) omit glm5_on and rely on glm5 sessions having cache None. A future glm5 carrier that keeps s.cache would fall into plain decode.
- fix: Add Session::on_spec_program() and use it at every site.

### low: MEMRA_SPEC_BURST parsed separately in each spec step

- where: `crates/memra-server/src/worker.rs:39728` (S18, evidence-checked)
- detail: Each burst reparses the env in step_session (39728), step_gemma_spec (40493), step_glm5_spec (41030) and step_dspark_spec, while MEMRA_ADMIT_YIELD is cached at 30671 and reread per burst at 39892, so the two can disagree mid-process.
- fix: Use one spec_burst_t() helper, and pick cached or per-burst reads deliberately for each door.

### low: memra-validate's median/run_n_median/GateTally unused while bins hand-roll them

- where: `crates/memra-validate/src/lib.rs:43` (S21, evidence-checked)
- detail: The crate exists to de-duplicate the N=5 median and ALL-GREEN tally, but rel_of, run_n_median and GateTally have zero callers. Bins reimplement median: hash_micro.rs:71, b200_matvec_bench.rs:143, gemv_e4m3_bench.rs:121, tier_transfer_gate.rs:883 and the glm5_kpool_indexer_bench test. Even-N conventions may differ between copies.
- fix: Switch the bins to memra_validate::median/run_n_median, or delete the unused API.

## Mechanical candidates

Stage 0 listed candidates by script (items with no caller in the graph, identical or near-identical bodies, functions of 400+ lines, flags past their decide-by date, files with many device checks). The slice annotators judged them; these verdicts were not re-checked by a skeptic.

| candidate kind | confirmed | rejected | unsure |
|---|---|---|---|
| arch_check_scatter | 6 | 25 | 0 |
| big_fn | 56 | 0 | 0 |
| dead_candidate | 377 | 176 | 9 |
| exact_duplicate | 1 | 0 | 0 |
| near_duplicate | 159 | 79 | 51 |

### Functions of 400+ lines (decoupling candidates) (56)

- `crates/memra-engine/build.rs:191`: 689-line build-script main. fan_in=0 is normal for a build script entry. It mixes arch selection, nine fatbins, a 36-TU static archive with per-TU flag arms, two CUTLASS archives and linking. Splitting it into per-artifact functions would make per-variant impact reviewable (and would have exposed th
- `crates/memra-engine/src/decode_batch.rs:3597`: 639 lines; step35-only batched row walk (decode + MTP verify via spec.rs). Internal doors MEMRA_STEP_TP_BATCH and MEMRA_ROPE_ROWS labeled.
- `crates/memra-engine/src/dflash.rs:1548`: DflashDraft::load 673 lines: config parse, tensor validation, precision seam, DFlash2/markov/confidence heads, harvest resolution. Internal precision arm labeled 1803-1857.
- `crates/memra-engine/src/dflash.rs:3947`: generate_spec_dspark 674 lines, bin-only twin of the served burst; duplicates the burst's arm structure (sampled/DFlash2/markov, ckpt, vt policy).
- `crates/memra-engine/src/dflash.rs:6020`: dspark_spec_session_burst_inner 585 lines; arms labeled (vg pool 6110, sampled 6191, DFlash2 6210, markov 6218, confidence 6254, ckpt 6189, ladder 6590).
- `crates/memra-engine/src/dsv4_gpu.rs:3548`: load_placed, 883 lines: door parsing, placement, upload, drafter, EP and pack; internal arms labeled as regions 3556-4420
- `crates/memra-engine/src/dsv4_gpu.rs:5377`: block_forward, 613 lines, reference monolithic prefill reachable only under the gate-armed reference program
- `crates/memra-engine/src/dsv4_gpu.rs:8642`: block_decode, 550 lines, legacy path only (MEMRA_DSV4_DECODE_PATH=legacy, refused by matrix)
- `crates/memra-engine/src/dsv4_gpu.rs:10363`: block_decode_dev, 692 lines, non-matrix device step (reference program only)
- `crates/memra-engine/src/dsv4_gpu.rs:11544`: decode_step_tp_ep, 543 lines, default plain decode on TP/EP incl. replay capture
- `crates/memra-engine/src/dsv4_gpu.rs:15613`: alloc_batched_state_for, 485 lines of workspace allocation
- `crates/memra-engine/src/dsv4_gpu.rs:17809`: attention_rows_dev, 1238 lines; hot path of every device route; arms labeled (TP2 shard, replay, radix, tiled knee, C4 host, wo_a)
- `crates/memra-engine/src/dsv4_gpu.rs:19896`: moe_verify_dev, 537 lines; arms labeled (host_math, EP/TP-EP, dead matrix-EP graph, grouped, reference)
- `crates/memra-engine/src/gemma_spec.rs:726`: generate_spec_gemma 801 lines, bin-only (gemma_gate, gemma_spec_session_gate); carries the graph/round-graph doors not reachable from serving.
- `crates/memra-engine/src/glm5_decode_graph.rs:1639`: glm5_selfcheck_run 452 lines: snapshot, eager reference, replay, bitwise compare of output and KDA state.
- `crates/memra-engine/src/glm_spec.rs:3383`: glm5_spec_round 464 lines; arms labeled (draft source 3436, verify/penalty 3611-3680, accept 3683).
- `crates/memra-engine/src/hybrid.rs:4220`: 2207 lines; every HybridModel load. Arms labeled: PP admission, glm5 TP, spill, E4B, gemma bits, MTP chain, FRSPEC, glm5 DFlash, process-global numeric defaults, mirror passes, consumption audit.
- `crates/memra-engine/src/hybrid_forward.rs:6762`: prime_layers is 497 lines and the serial-trunk prime body for 10+ families. Its internal arms (slabs, PRIME_SEG, f16fuse, mixer/FFN dispatch, trace, taps) are labelled as regions.
- `crates/memra-engine/src/hybrid_forward.rs:12134`: moe_ffn_inner is 1093 lines. About 960 of them (12174-13135) are the Step TP/EP arm, which only sigmoid-router families on PRO 6000 reach, inlined in the entry every MoE family takes (decoupling finding).
- `crates/memra-engine/src/hybrid_forward.rs:13620`: moe_ffn_sequential_zq8 is 1366 lines and mixes softmax arms (pairs/dev), sigmoid arms, glm5-only vrows/door-D/fused-epi arms, Hy3 CPU experts and spill I/O (decoupling finding). The rp slab-reader bug lives here.
- `crates/memra-engine/src/hybrid_forward.rs:17871`: moe_ffn_pairs 446 lines; arms labeled: MMA/f16g (17996-18155), dp4a dec/em (18157-18260), dead small-t shexp arms (18272-18295). The two shexp epilogues duplicate moe_shexp_add.
- `crates/memra-engine/src/hybrid_forward.rs:18319`: moe_ffn_dev 410 lines; arms labeled: CSR (18409-18455), rows, serial resident loop, SLRU arm (18584-18670), shexp epilogue copy.
- `crates/memra-engine/src/hybrid_forward.rs:21100`: moe_ffn_grouped 556 lines, reachable only behind MEMRA_MOE_GROUPED (default off, receipt says unsafe). Deletion candidate rather than a split.
- `crates/memra-engine/src/hybrid_forward.rs:22573`: gemma4_moe 523 lines; arms labeled: fast small-t (22607-22710), prefill pairs (22720-22985), f16g door (22763-22834), sequential fallback.
- `crates/memra-engine/src/hybrid_forward.rs:26843`: step35_tp_prefill_attn_resident 483 lines, behind MEMRA_STEP_TP_PREFILL (default off), PRO 6000 Step TP only.
- `crates/memra-engine/src/hybrid_forward.rs:27364`: step35_tp_decode_attn_resident_inner 442 lines (v1 TP decode driver, default when MEMRA_STEP_TP set and DECODE_V2 unset).
- `crates/memra-engine/src/hybrid_forward.rs:28502`: step35_tp_decode_attn_resident_v2 635 lines; misnamed: also serves generic MEMRA_PARALLEL TP attention for ungated plans (decode.rs:3527).
- `crates/memra-engine/src/hybrid_forward.rs:29445`: gemma4_e4b_attn 422 lines, unreachable: no model pack admits E4B configs.
- `crates/memra-engine/src/hybrid_forward.rs:31386`: step35_token_graph_build 630 lines, behind MEMRA_STEP_TP_GRAPH whose receipt is negative (-20%).
- `crates/memra-engine/src/lib.rs:30735`: fa_decode_rows is 461 lines: split-rung grouping, hd512 split override, tb512/i2/v4/v3/v2 picks and the module pick in one body. Its arms are labeled as regions; the per-group partial-pool block is a third copy of the retire-on-grow code (also in fa_decode_kvmod_view and fa_part_pool_grow).
- `crates/memra-engine/src/model.rs:892`: 496 lines; the per-tensor residency choice (NVFP4 direct, FP8 per-tensor/blk, ARM B', GGUF quant, CUTLASS, FP8 stash, FULL_PREC/BF16, Float). Regions labeled per arm.
- `crates/memra-engine/src/model.rs:2675`: 400 lines; safetensors expert gather with repack disk cache, pinned tiers, mixed layouts. MoE families only.
- `crates/memra-engine/src/qwen4exp_gpu.rs:7100`: 449-line kernel oracle gate; gate-only, observability effect
- `crates/memra-engine/src/qwen4exp_gpu.rs:12832`: 723 lines; holds router (device/host), grouped NVFP4, merged verify, DeviceBf16 and per-expert arms, all labeled as regions
- `crates/memra-engine/src/qwen4exp_gpu.rs:15124`: 663-line spec loop; arms labeled (co-prefill, targets, adapt/dynk)
- `crates/memra-engine/src/qwen4exp_gpu.rs:18503`: 743-line TP2 decode; gate-bin only
- `crates/memra-engine/src/qwen4exp_gpu.rs:19376`: 730-line TP2 forward; gate-bin only
- `crates/memra-engine/src/spec.rs:5977`: step35_verify_batch_layers is about 503 lines (5978-6479) with trows chunking, a t-column walk, fa2/oproj deferral through crate::tp ambient state, and a batch-layer fallback. Its internal arms are labeled as regions.
- `crates/memra-engine/src/spec.rs:10754`: prepare_spec_draft is about 845 lines. It holds four near-parallel capture blocks (greedy chain, greedy single, sampled chain, sampled single), each with the same OOM-trim / mark_failed / dcw-headroom boilerplate. That boilerplate can be factored.
- `crates/memra-engine/src/spec.rs:11600`: generate_spec_inner2 is about 3169 lines covering prime, setup, the round loop (stream burst, 5 draft arms, verify, greedy/sampled accept, grammar, budget clamp, 4 commit arms) and the tail. All arms are labeled as regions.
- `crates/memra-engine/src/tp.rs:11438`: 400 lines; holds the auto W4A16 EP walk with W4A8 and pair-down sub-arms plus timing instrumentation. Timing/phase-event bookkeeping could move out.
- `crates/memra-engine/src/tp.rs:13060`: 483 lines; TP prefill grouped experts with three determinism/profiling probes inline (MEMRA_PRIME_PROF, MEMRA_MOE_DETERM_STAGE/BANK).
- `crates/memra-engine/src/tp.rs:13601`: 507 lines; decode routes with graph, sel-mirror, memops, moe-direct sub-arms.
- `crates/memra-gguf/src/config.rs:1773`: 825 lines with one arm per family (m3, hy3, step35, dsv4, qwen4exp, glm5, mimo, gemma4); any edit outside an arm reaches every safetensors family. Splitting per-family builders into the packs would narrow impact.
- `crates/memra-gguf/src/config.rs:3642`: HfConfig::apply, 556-line key dispatcher shared by every HF family.
- `crates/memra-gguf/src/model_packs/qwen4_exp/mod.rs:316`: 528-line qwen4_exp census builder; family-local, gate/census only.
- `crates/memra-server/src/dsv4_serve.rs:2811`: serve_one is 461 lines and carries every request arm (spec restored/cold, plain greedy/sampled, replay, B-row, park). Its arms are labeled as regions. The cold spec drivers and the monolithic prefill inside it are unreachable on serving.
- `crates/memra-server/src/lib.rs:6107`: serve_with is 440 lines. Its internal arms (env audit, metering claim gate, ttft layer, drain) are labeled; it is boot and policy only.
- `crates/memra-server/src/lib.rs:8947`: build_chat_request_with_trace is 429 lines and carries every family dialect arm (parsers, stops, effort, grammar x think). Arms are labeled as family regions; see the decoupling finding.
- `crates/memra-server/src/lib.rs:11050`: chat_completions_with_admission is 469 lines; its n-choice, remote fetch, constraint wait, background and stream arms are labeled.
- `crates/memra-server/src/worker.rs:14285`: 406 lines. Contract H2D promote submit, reached only with MEMRA_KV_HOST_MB>0 and MEMRA_KV_HOST_CONTRACTS=1 (no default cell). Its door passed decide-by 2026-10-05.
- `crates/memra-server/src/worker.rs:14698`: 503 lines. Contract promote settle, same door as the submit. It is the largest item in the slice, and the stale contracts door keeps it alive.
- `crates/memra-server/src/worker.rs:19319`: 403 lines. Off-tick capture submit, reached only after prefix_capture_off_tick passes its tier/transfer-engine gate (19181).
- `crates/memra-server/src/worker.rs:26012`: run spans 26013-32987 (~6976 lines): boot for every family, drafter and vision attach, admission, spec demotion, prime batching, batched decode, dark-lane prefill and retire in one body. Its fan-out is why any edit here maps to every family x route.
- `crates/memra-server/src/worker.rs:34355`: admit spans 34356-37799 (~3445 lines) and holds the MTP, dspark, gemma and glm5 route decisions plus every reuse tier (continuation, affinity, prefix, host promote).
- `crates/memra-server/src/worker.rs:39666`: step_session is 633 lines: an MTP burst arm (39713-40069), a prefill arm duplicating prefill_tick (40077-40217) and the eager decode arm (40219-40297).

### No caller (confirmed dead) (377)

- `crates/memra-engine/cu/flash_attn.cu:8617`: fa_decode_vec_q_rows_reg_w_i2: no string literal names it in any crate, bin or tool. Its comment claims a kf8vf8-module launch that no longer exists.
- `crates/memra-engine/cu/flash_attn.cu:8826`: fa_decode_vec_q_rows_dpl16_kv: the K=V twin retired 2026-07-11 (lib.rs:30822). The dispatcher discards kv_shared; no launch site.
- `crates/memra-engine/cu/hybrid.cu:2374`: No string-keyed func("gdn_p_bf16_masked") load in any crate or tool; only comments (hybrid.cu:2643, lib.rs:34684) name it as replaced.
- `crates/memra-engine/cu/kernels.cu:29`: argmax_logits_f32_to_u32: no func() lookup. argmax_gate.rs mentions it only in a doc comment and calls argmax_token_device (partial/final kernels).
- `crates/memra-engine/cu/moe_f16_grouped.cu:2226`: rg -w finds only the definition: no mmq_ffi.rs declaration, no C++ caller.
- `crates/memra-engine/cu/moe_f16_grouped.cu:2229`: rg -w finds only the definition.
- `crates/memra-engine/cu/qmatvec.cu:3498`: unreachable: qmatvec_mmvq_batched forces base/rp at mcols 16 (lib.rs:23859), even over MEMRA_BVAR
- `crates/memra-engine/cu/qmatvec.cu:9542`: unreachable: qmatvec_mmvq_batched forces base/rp at mcols 16 (lib.rs:23859), even over MEMRA_BVAR
- `crates/memra-engine/cu/qmatvec.cu:10598`: no Rust, test, tool or doc reference to the kernel name
- `crates/memra-engine/cu/qmatvec.cu:11434`: no reference outside qmatvec.cu; Rust launches only the runtime-t tcol kernels (lib.rs:25330, 25492)
- `crates/memra-engine/cu/qmatvec.cu:11602`: no reference outside qmatvec.cu; Rust launches only the runtime-t tcol kernels (lib.rs:25330, 25492)
- `crates/memra-engine/cu/qmatvec.cu:11610`: no reference outside qmatvec.cu; Rust launches only the runtime-t tcol kernels (lib.rs:25330, 25492)
- `crates/memra-engine/cu/qmatvec.cu:11618`: no reference outside qmatvec.cu; Rust launches only the runtime-t tcol kernels (lib.rs:25330, 25492)
- `crates/memra-engine/cu/qmatvec.cu:11690`: no reference outside qmatvec.cu; Rust launches only the runtime-t tcol kernels (lib.rs:25330, 25492)
- `crates/memra-engine/cu/qmatvec.cu:11696`: no reference outside qmatvec.cu; Rust launches only the runtime-t tcol kernels (lib.rs:25330, 25492)
- `crates/memra-engine/cu/qmatvec.cu:11702`: no reference outside qmatvec.cu; Rust launches only the runtime-t tcol kernels (lib.rs:25330, 25492)
- `crates/memra-engine/cu/qmatvec.cu:11770`: no reference anywhere; its door MEMRA_DUAL_ILV has no env read or FLAGS row
- `crates/memra-engine/src/cutlass_ffi.rs:237`: quantize_fp4_act_cutlass: only its definition matches in the whole workspace
- `crates/memra-engine/src/decode.rs:105`: GraphSession::upload_mask: no call site anywhere (engine, bins, server); only docs mention it.
- `crates/memra-engine/src/decode.rs:1929`: decode_step_dc_cap (fn at 1930): no caller; capture paths call decode_step_dc_cap_masked directly.
- `crates/memra-engine/src/decode.rs:2315`: graph_session_from_cache: no caller; only the _masked variant and docs reference it.
- `crates/memra-engine/src/decode_batch.rs:419`: PendingBatchStep::wait is only obtainable from decode_step_batch_sampled_lean_masked_pending, which has no caller.
- `crates/memra-engine/src/decode_batch.rs:1089`: decode_step_batch_sampled_lean_masked_pending: no caller in memra-server, engine or bins; decode_step_overlap_eligible is only used by it.
- `crates/memra-engine/src/dflash.rs:666`: DsparkHarvest::resolve has no caller in any crate; superseded by for_draft (711), yet its doc still calls it the served resolution.
- `crates/memra-engine/src/dflash.rs:769`: trained_offset_of_row is called only from #[cfg(test)] modules in dflash.rs (7598+, test mod starts 7568).
- `crates/memra-engine/src/dflash.rs:3219`: DflashKv::floor() accessor is read only by tests/glm5_dflash_session_gpu.rs; production reads the field directly.
- `crates/memra-engine/src/dflash.rs:4741`: DsparkSpecSession::draft_tail_rows is referenced only in doc comments.
- `crates/memra-engine/src/dflash.rs:4993`: DflashKv::n_layer has no caller; .n_layer() hits elsewhere are cfg fields.
- `crates/memra-engine/src/dsv4_attention_split.rs:40`: ExistingGemvHalfPlan::q_b: only callers are in the #[cfg(test)] module (L898, L905). No other crate or bin uses it.
- `crates/memra-engine/src/dsv4_attention_split.rs:53`: ExistingGemvHalfPlan::wo_a: test-only callers (L914, L921).
- `crates/memra-engine/src/dsv4_attention_split.rs:79`: WoBPartialSumPlan::rank: no caller outside tests. It belongs to the superseded column partial-sum design.
- `crates/memra-engine/src/dsv4_attention_split.rs:94`: wo_b_rank_order_sum_f32: test-only (L954, L1274). The served attention TP joins by an exact row gather.
- `crates/memra-engine/src/dsv4_attention_split.rs:119`: Partition::start: no .start() call outside tests in this file or in dsv4_gpu.rs.
- `crates/memra-engine/src/dsv4_attention_split.rs:131`: Partition::is_empty: no caller. It exists for the clippy len/is_empty pair.
- `crates/memra-engine/src/dsv4_attention_split.rs:250`: PackedFp8Host::half_rows: test-only (L719-720).
- `crates/memra-engine/src/dsv4_attention_split.rs:269`: PackedFp8Host::half_columns: test-only (L769, L852).
- `crates/memra-engine/src/dsv4_attention_split.rs:288`: PackedFp8Host::code_bytes: no caller anywhere.
- `crates/memra-engine/src/dsv4_attention_split.rs:292`: PackedFp8Host::scale_bytes: test-only (L783).
- `crates/memra-engine/src/dsv4_attention_split.rs:314`: DeviceFp8Plane::upload: test-only. Production uses DeviceFp8Plane::from_device.
- `crates/memra-engine/src/dsv4_doors.rs:409`: doors_without_evidence_in_either_direction: only the #[cfg(test)] module calls it (L1375-1438). It is a registry test helper with no runtime caller.
- `crates/memra-engine/src/dsv4_doors.rs:430`: engaged_but_below_the_floor: test-only (L1389, L1397).
- `crates/memra-engine/src/dsv4_doors.rs:609`: door_name_source_files: test-only (L1065). env_audit.rs only mentions it in a comment.
- `crates/memra-engine/src/dsv4_doors.rs:635`: door_names_in_source: test-only (L1073, L1120, L1129). env_audit.rs:454 is a doc comment, not a call.
- `crates/memra-engine/src/dsv4_doors.rs:665`: undeclared_door_names: test-only.
- `crates/memra-engine/src/dsv4_doors.rs:681`: stale_exemptions: test-only (L1098).
- `crates/memra-engine/src/dsv4_doors.rs:716`: declaration_violations: test-only (L845, L876, L901).
- `crates/memra-engine/src/dsv4_doors.rs:740`: inert_default_on_doors: test-only (L954, L1411).
- `crates/memra-engine/src/dsv4_gpu.rs:1423`: DecodeState::ep_device_scratch_bytes: no caller in crates/, tools/, bins or tests; only FLAGS.md prose
- `crates/memra-engine/src/dsv4_gpu.rs:1438`: c4_device_scratch_bytes: only a research note mentions it
- `crates/memra-engine/src/dsv4_gpu.rs:2201`: dsv4_phase_reset: no caller anywhere
- `crates/memra-engine/src/dsv4_gpu.rs:2346`: getter unused; the setter set_index_topk_radix_for_gate is used by gates
- `crates/memra-engine/src/dsv4_gpu.rs:2591`: clear_grouped_m1_tc_for_gate: no caller
- `crates/memra-engine/src/dsv4_gpu.rs:2646`: getter unused; setter used by dsv4_decode_rate_gate
- `crates/memra-engine/src/dsv4_gpu.rs:2693`: Dsv4Gpu::tp_ep_topology_for_gate unused; Dsv4Gpu::load reads dsv4_topology::tp_ep_for_gate directly
- `crates/memra-engine/src/dsv4_gpu.rs:2990`: set_indexer_score_for_gate: no caller in crates/ or tools/
- `crates/memra-engine/src/dsv4_gpu.rs:14877`: set_matrix_ep_graph_for_gate: no caller; FLAGS.md decide-by 2026-09-21 passed; makes the moe_verify_dev graph arm unreachable
- `crates/memra-engine/src/dsv4_gpu.rs:14883`: matrix_ep_graph_dispatches: no caller
- `crates/memra-engine/src/env_audit.rs:82`: audit_vars is used only by env_audit unit tests; production uses audit_os_vars
- `crates/memra-engine/src/fp8_ffi.rs:97`: blk_native_nan_refused getter has no reader; only note_blk_native_nan_refused is called (model.rs:1044)
- `crates/memra-engine/src/gemma_spec.rs:488`: gemma4_draft_step has no caller in crates/ (bins and tests included).
- `crates/memra-engine/src/gemma_spec.rs:701`: gemma4_draft_step_greedy has no caller in crates/; only its own definition matches.
- `crates/memra-engine/src/gemma_spec.rs:1581`: GemmaSpecSession::emitted_len has no reference anywhere.
- `crates/memra-engine/src/glm5_tp.rs:344`: Glm5TpRt::ar_1stage_into has no caller in crates/ (src, bins, tests); the walk uses ar_1stage/ar_1stage_hcpost.
- `crates/memra-engine/src/glm5_tp.rs:1050`: Glm5TpKda::ranks: every .ranks() receiver found is Glm5TpRt, the MLA sidecar, the EP/split structs or Hop; KDA code uses rt.ranks().
- `crates/memra-engine/src/glm5_tp.rs:1697`: glm5_ep_diet_fanout_uploads_avoided: no reader anywhere; the counter static is still bumped.
- `crates/memra-engine/src/glm_spec.rs:585`: glm5_rank_trimmed_draft_rounds read only by tests/glm5_dflash_session_gpu.rs (gate accessor).
- `crates/memra-engine/src/glm_spec.rs:811`: VerifyGraphPool::replays read only by tests/glm5_decode_graph_capture_gpu.rs.
- `crates/memra-engine/src/glm_spec.rs:1142`: Glm5VerifyCkpt::kda_stash_kinds used only by tests/glm5_tparallel_verify_gpu.rs.
- `crates/memra-engine/src/glm_spec.rs:1182`: Glm5VerifyCkpt::kda_ssm_snap_buffers used only by tests/glm5_decode_graph_capture_gpu.rs.
- `crates/memra-engine/src/glm_spec.rs:2885`: glm5_spec_session_from_restored: worker mentions it only in comments and restores through glm5_prime_restored_start; callers are tests/glm5_dflash_session_gpu.rs.
- `crates/memra-engine/src/glm_spec.rs:4459`: Glm5SpecSession::round_log_cap referenced only in a worker doc comment.
- `crates/memra-engine/src/glm_spec.rs:4463`: draft_kv_len used only by tests/glm5_dflash_session_gpu.rs.
- `crates/memra-engine/src/glm_spec.rs:4471`: draft_kv_rows_host used only by tests/glm5_dflash_session_gpu.rs.
- `crates/memra-engine/src/glm_spec/prime.rs:226`: Glm5PrimeState::boundary_logits read only by tests/glm5_dflash_session_gpu.rs.
- `crates/memra-engine/src/grid_capture.rs:59`: capture_point is referenced only by tests/grid_capture_gpu.rs and the unit test; worker computes its own point
- `crates/memra-engine/src/hybrid.rs:4148`: step_tp_rank_engine: no reference repo-wide.
- `crates/memra-engine/src/hybrid.rs:4187`: has_plan_operation: no reference repo-wide.
- `crates/memra-engine/src/hybrid_forward.rs:558`: No production caller. Its only consumer is tests/mla_seg_ws_gpu.rs:63/65 (engagement receipt for the MEMRA_MLA_SEG_WS gate), and the pub static MLA_SEG_WS_DISPATCHES could be read there directly. Removal is harmless but would need the test edited.
- `crates/memra-engine/src/kda.rs:942`: kda_conv3_dispatches used only by tests/kda_conv3_gpu.rs
- `crates/memra-engine/src/kda.rs:1426`: Engine::kda_gate has no caller; kda_gate_beta replaced it
- `crates/memra-engine/src/kda.rs:1662`: kda_proj_fused6 wrapper called only by tests/kda_fused_proj*_gpu.rs; production calls kda_proj_fused6_pre
- `crates/memra-engine/src/lib.rs:553`: No production caller. Only in-crate unit tests (dsv4_grouped.rs mod tests at L1373; calls at 1694/1784/1979) use it. Its setter has a bin caller (dsv4_gpu_dspark_gate). Test-only gate seam, so no serving reach.
- `crates/memra-engine/src/lib.rs:573`: No caller in any .rs file (tests and bins included). docs/KERNELS.md:704 still names it as the gate knob.
- `crates/memra-engine/src/lib.rs:579`: No caller anywhere. Pair of the unused setter above.
- `crates/memra-engine/src/lib.rs:1417`: Snapshot has no reader. Q8_ROW_ILP_DISPATCHES is written by q8_row_ilp_note only.
- `crates/memra-engine/src/lib.rs:1459`: Snapshot has no reader. NVFP4_ROW_ILP_DISPATCHES is bumped at lib.rs:22518 only.
- `crates/memra-engine/src/lib.rs:2508`: Only reader is tests/glm5_tparallel_verify_gpu.rs. A test hook with no serving reach.
- `crates/memra-engine/src/lib.rs:2531`: Only reader is tests/glm5_matvec_doors_gpu.rs (test hook).
- `crates/memra-engine/src/lib.rs:2551`: Only reader is tests/glm5_matvec_doors_gpu.rs (test hook).
- `crates/memra-engine/src/lib.rs:2577`: Only reader is tests/glm5_matvec_doors_gpu.rs (test hook).
- `crates/memra-engine/src/lib.rs:2618`: Only reader is tests/glm5_matvec_doors_gpu.rs (test hook).
- `crates/memra-engine/src/lib.rs:2654`: Only reader is tests/glm5_moe_loc_doors_gpu.rs (test hook).
- `crates/memra-engine/src/lib.rs:2665`: No reader in code. The static is bumped at hybrid_forward.rs:13936, but no test or bin reads the snapshot. docs/FLAGS.md:225 claims a gate asserts it.
- `crates/memra-engine/src/lib.rs:2735`: Test hook by name. Only reader is tests/glm5_moe_loc_doors_gpu.rs.
- `crates/memra-engine/src/lib.rs:2774`: Only reader is tests/glm5_dedup_sched_gpu.rs (test hook).
- `crates/memra-engine/src/lib.rs:2890`: Only reader is tests/glm5_moe_loc_doors_gpu.rs (test hook).
- `crates/memra-engine/src/lib.rs:3027`: Only reader is tests/glm5_matvec_doors_gpu.rs (test hook).
- `crates/memra-engine/src/lib.rs:3049`: Only reader is tests/glm5_matvec_doors_gpu.rs (test hook).
- `crates/memra-engine/src/lib.rs:3068`: Only reader is tests/e4m3_batch_r2_gpu.rs (test hook).
- `crates/memra-engine/src/lib.rs:3087`: Only reader is tests/q80_batch_r2_gpu.rs (test hook).
- `crates/memra-engine/src/lib.rs:3488`: Only reader is tests/glm5_spec_session_gpu.rs (test hook).
- `crates/memra-engine/src/lib.rs:3504`: Only reader is tests/mla_tc_prefill_gpu.rs (test hook).
- `crates/memra-engine/src/lib.rs:8145`: No caller anywhere. A raw-pointer twin of nvfp4_ep_stage_inputs (tp.rs uses the slice form).
- `crates/memra-engine/src/lib.rs:10343`: No Rust caller. The i32_add_k kernel in cu/kernels.cu:1940 is then also unreached.
- `crates/memra-engine/src/lib.rs:10487`: No Rust caller. A comment at L7816 says the fold replaced this launch. The kernel stays in moe_router.cu/qmatvec.cu.
- `crates/memra-engine/src/lib.rs:11521`: No caller anywhere.
- `crates/memra-engine/src/lib.rs:12108`: No caller. Only named in a doc at L12304 and in a research note.
- `crates/memra-engine/src/lib.rs:13127`: No caller anywhere.
- `crates/memra-engine/src/lib.rs:13186`: No caller. Only named in the doc of its raw twin (L13250).
- `crates/memra-engine/src/lib.rs:13252`: No caller anywhere.
- `crates/memra-engine/src/lib.rs:13384`: No caller anywhere.
- `crates/memra-engine/src/lib.rs:13709`: dtoh_f32_into_pinned: no caller. Only its own error string names it.
- `crates/memra-engine/src/lib.rs:13744`: htod_f32_from_pinned_async: no caller.
- `crates/memra-engine/src/lib.rs:13994`: Only reader is tests/glm5_decode_graph_capture_gpu.rs (test hook).
- `crates/memra-engine/src/lib.rs:14918`: rms_norm3: no Rust caller. add_rms_norm3 launches its own fused kernel.
- `crates/memra-engine/src/lib.rs:15747`: row_softmax: no caller. Only listed in docs/KERNELS.md:237.
- `crates/memra-engine/src/lib.rs:17259`: add_q8_1: no caller. rms_pre_add_q8_1 replaced it (doc L17292, hybrid_forward.rs:29932).
- `crates/memra-engine/src/lib.rs:21877`: qmatvec_e4m3_fused6_raw: the only caller is tests/kda_fused6_e4m3_gpu.rs. No production caller; it is a test harness entry.
- `crates/memra-engine/src/lib.rs:24012`: qmatvec_nvfp4_batched_raw: rg finds no reference outside its own definition. Its doc says older gates used it.
- `crates/memra-engine/src/lib.rs:24358`: matmul_group_multi: no reference anywhere in crates/.
- `crates/memra-engine/src/lib.rs:25186`: axpy_rows_seq_md_off_into: no reference. Its kernel axpy_rows_seq_md_off_f32 is launched only from here (kernels.tsv).
- `crates/memra-engine/src/lib.rs:26853`: matvec_bf16_tcols_gate_kernel_into: the only caller is tests/glm5_matvec_doors_gpu.rs. It is a test entry.
- `crates/memra-engine/src/lib.rs:27093`: matvec_bf16_dual is pub(crate) with #[allow(dead_code)] and has no caller. The _into and _view_into twins carry the kernel.
- `crates/memra-engine/src/lib.rs:29381`: bf16_to_f16: no caller. bf16_to_f16_into is the live form.
- `crates/memra-engine/src/lib.rs:32973`: gdn_scan_s128_view: no caller. The gdn_scan_s128 kernel stays live through gdn_scan_s128.
- `crates/memra-engine/src/lib.rs:33018`: ssm_conv1d_view: no caller. ssm_conv1d_silu_f32 is still reached through ssm_conv1d, which only kernel_check calls.
- `crates/memra-engine/src/lib.rs:33054`: ssm_conv1d_tm: no caller. Its kernel ssm_conv1d_tm_f32 is launched only here, so the kernel is dead too.
- `crates/memra-engine/src/lib.rs:34516`: addr_f32_mut: no caller. addr_f32, addr_f32v and addr_u8 are the live forms.
- `crates/memra-engine/src/lib.rs:35457`: gated_rmsnorm_f16out: no caller. The kernel stays live through gated_rmsnorm_f16out_zv.
- `crates/memra-engine/src/lib.rs:35489`: add_rms_norm_zq8: no Rust caller. decode.rs:630 records the z-quantize fuse measured NEGATIVE, so this is a refuted door's leftover. The add_rms_norm_zq8 kernel is launched only here.
- `crates/memra-engine/src/lib.rs:35623`: Engine::transpose: no method-call site (rg on .transpose( finds only Option::transpose). transpose_f32 is launched only here.
- `crates/memra-engine/src/lib.rs:35643`: repeat_heads: no caller. repeat_heads_f32 is launched only here.
- `crates/memra-engine/src/lib.rs:35750`: conv_left_pad: no caller. Comments in hybrid.cu:30 and hybrid_forward.rs:11909 say the fused conv replaced it. conv_left_pad_f32 is launched only here.
- `crates/memra-engine/src/lib.rs:35826`: slice_range: no caller. It also does a host round-trip with synchronize, which no hot path should use.
- `crates/memra-engine/src/mimo_audio_codec_rvq.rs:194`: encode_one_rvq_step used only at line 468 inside the cfg(test) module (starts 271)
- `crates/memra-engine/src/mimo_audio_codec_weights.rs:163`: in_channels: no call site anywhere
- `crates/memra-engine/src/mimo_audio_codec_weights.rs:167`: out_channels: no call site anywhere
- `crates/memra-engine/src/mimo_audio_codec_weights.rs:395`: tensor() used only at mimo_audio_codec_layer.rs:752, inside its cfg(test) module (starts 522)
- `crates/memra-engine/src/mimo_compressed_text_forward.rs:113`: has_modal_payload read only by this file's tests (390, 413); memra-reference hits are a different type
- `crates/memra-engine/src/mimo_text_chunk_ffi.rs:37`: mimo_text_chunk_attention called only by its own cfg(test) module (lines 171, 192)
- `crates/memra-engine/src/mimo_vision_block_forward.rs:587`: forward_one_block: only the definition matches
- `crates/memra-engine/src/mla.rs:270`: mla::rope_neox used only by mla.rs unit tests (512, 528); other rope_neox hits are Engine::rope_neox
- `crates/memra-engine/src/mla.rs:288`: norm_to_neox_perm used only at mla.rs:494 inside cfg(test)
- `crates/memra-engine/src/mla_ffi.rs:497`: used only by tests/mla_kpool_select_v2_gpu.rs
- `crates/memra-engine/src/mla_ffi.rs:1813`: mla_attn_absorbed_live called only by tests/mla_live_len_gpu.rs
- `crates/memra-engine/src/mla_ffi.rs:2218`: mla_kpool_score_ref called only by tests (glm5_kpool_indexer_gpu/bench); retained oracle
- `crates/memra-engine/src/mla_ffi.rs:2536`: mla_dsa_attn_warp_online has no caller at all; mla_attn_gathered calls memra_mla_dsa_attn_split_f32 directly and only the _live twin is used
- `crates/memra-engine/src/mla_ffi.rs:2711`: mla_kpool_select_ref called only by tests; dsa_select_gate uses the extern directly
- `crates/memra-engine/src/mmq_ffi.rs:3178`: a4_prefill_launches: only the definition; the counter is still incremented but never read
- `crates/memra-engine/src/mmq_ffi.rs:3248`: a4_stats_armed: only the definition
- `crates/memra-engine/src/model.rs:808`: GpuTensor::scale: every .scale() call in the repo is MlaDims/MlaConfig::scale; no GpuTensor caller.
- `crates/memra-engine/src/model.rs:826`: GpuTensor::load (GGUF wrapper): no 'GpuTensor::load(' or Self::load call anywhere.
- `crates/memra-engine/src/model.rs:1567`: EmbedHost::from_gguf: no caller; loaders use EmbedHost::from_source. Tokenizer::from_gguf hits are a different type.
- `crates/memra-engine/src/model.rs:2105`: HostExps::load: no caller; load_ffn uses load_stacked_from_source/load_from_source/load_tiered.
- `crates/memra-engine/src/moe_sel_dump.rs:217`: moe_sel_dump::flush has no caller in any crate, test or bin
- `crates/memra-engine/src/o2_band.rs:62`: prob_band_worst used only by o2_band unit tests; the module has no consumer
- `crates/memra-engine/src/o2_band.rs:93`: tie_flip used only by o2_band unit tests
- `crates/memra-engine/src/parallel.rs:874`: plan_grouped: only called from #[cfg(test)] module (parallel.rs:2125-2236); tests are not callers.
- `crates/memra-engine/src/parallel.rs:991`: ParallelPlan::global_rank: test-only uses.
- `crates/memra-engine/src/parallel.rs:1015`: query_feature_range: test-only.
- `crates/memra-engine/src/parallel.rs:1022`: kv_feature_range: test-only.
- `crates/memra-engine/src/parallel.rs:1028`: dense_ffn_range: test-only.
- `crates/memra-engine/src/parallel.rs:1036`: ParallelPlan::routed_expert_range: test-only.
- `crates/memra-engine/src/parallel.rs:1042`: routed_expert_ffn_range: test-only.
- `crates/memra-engine/src/parallel.rs:1079`: GroupedParallelPlan::group_for_global_rank: the whole grouped plan is test-only.
- `crates/memra-engine/src/parallel.rs:1085`: GroupedParallelPlan::global_rank: test-only.
- `crates/memra-engine/src/parallel.rs:1090`: GroupedParallelPlan::query_head_range: production calls (hybrid.rs:1862, parallel.rs:831, tp_step_attention_gate) are on ParallelPlan from contract.plan(), not the grouped type.
- `crates/memra-engine/src/parallel.rs:1099`: GroupedParallelPlan::kv_head_range: same as above; production calls resolve to ParallelPlan::kv_head_range.
- `crates/memra-engine/src/parallel.rs:1108`: GroupedParallelPlan::routed_expert_range: test-only.
- `crates/memra-engine/src/parallel.rs:1119`: validate_step_pp_request: no caller anywhere although its doc says serving requires this gate.
- `crates/memra-engine/src/parity_geometry.rs:214`: exact_div is #[allow(dead_code)] and used only by its unit test; the parity bins that #[path]-include the module do not call it
- `crates/memra-engine/src/ple_rows_tier.rs:86`: submit_device is explicitly #[allow(dead_code)] (adapter installed by a future gate); memra-tier test hits are a different type
- `crates/memra-engine/src/qwen4exp_gpu.rs:2306`: pub kv_hoist_is_on has no caller in crates/ or tools/ (bins never read it for receipt headers)
- `crates/memra-engine/src/qwen4exp_gpu.rs:2337`: pub pool_t_is_on has no caller anywhere
- `crates/memra-engine/src/qwen4exp_gpu.rs:2506`: LayerPlacement::is_even is called only from this file's cfg(test) tests (L20749..L21013); no runtime caller
- `crates/memra-engine/src/qwen4exp_gpu.rs:13898`: MtpDraftState::rows() has no caller; the code reads the private field directly (dstate.rows)
- `crates/memra-engine/src/qwen4exp_gpu.rs:13922`: mtp_on_dev1 unused; spec_generate_ext tests self.mtp_dev1.is_some() inline
- `crates/memra-engine/src/qwen4exp_gpu.rs:14890`: spec_disarm has no caller in engine, bins or tests
- `crates/memra-engine/src/qwen4exp_gpu.rs:18391`: qsa_indexer_mask is private, #[allow(dead_code)], and has no caller; its comment claiming in-flight lanes call it is false
- `crates/memra-engine/src/spec.rs:784`: SpecConstraint::mask_words is a required trait method implemented at constrained.rs:918. It is called only from constrained.rs tests (1486, 1498) and never from engine or server production code.
- `crates/memra-engine/src/spec.rs:9391`: spec_session_from_restored (9393) has no caller in any crate, bin or test. The worker calls spec_session_from_restored_deferred and spec_session_from_restored_ready; the remaining mentions are doc comments.
- `crates/memra-engine/src/spec.rs:10587`: generate_spec_session_constrained (10589) has no caller. The worker calls generate_spec_session_constrained_prime_split directly; worker.rs:34992 mentions it only in a comment.
- `crates/memra-engine/src/tp.rs:1226`: E4m3ExpertBank::expert is called only from a #[cfg(test)] module (tp.rs test at ~15130). Other .expert() calls hit the NVFP4 bank.
- `crates/memra-engine/src/tp.rs:1641`: PreparedPeerWeightedRouteCombine::tokens: the ep_step_fp8_gate bin calls tokens() on the gate-prepared types, not on the combine.
- `crates/memra-engine/src/tp.rs:1645`: PreparedPeerWeightedRouteCombine::pairs: same as tokens; the bin uses owner_pair_counts on the combine.
- `crates/memra-engine/src/tp.rs:3057`: upload_bf16_column_parallel wrapper has no caller; the f32-mirror variant reaches the _inner directly.
- `crates/memra-engine/src/tp.rs:3835`: upload_bf16_row_parallel: no caller; the step_ variant replaced it.
- `crates/memra-engine/src/tp.rs:3869`: bf16_row_parallel_resident: no caller; step_bf16_row_parallel_resident is used by hybrid_forward and the gate.
- `crates/memra-engine/src/tp.rs:9931`: Nvfp4DeviceRoutesWorkspace::out_stage_mut: no caller.
- `crates/memra-engine/src/tp.rs:10791`: upload_expert_parallel_nvfp4: no caller; upload_expert_parallel_nvfp4_normalized is the live path.
- `crates/memra-engine/src/tp.rs:15877`: graph_section_absorbing: pub, no caller in src/bins/tests.
- `crates/memra-engine/src/tp.rs:15886`: graph_section_detached: pub, no caller.
- `crates/memra-engine/src/tp_ar.rs:272`: ArPhaseRecord::nanos is used only by the in-file unit test.
- `crates/memra-engine/src/tp_ar.rs:405`: ArLink::ranks: no caller.
- `crates/memra-engine/src/tp_ar.rs:432`: ArLink::broadcast is called only by tests/tp_ar_gpu.rs; glm5 TP uses tp_transport device-push for movement. The module doc still says it matters more than the reduce.
- `crates/memra-engine/src/tp_ar.rs:480`: ArLink::all_gather: tests/tp_ar_gpu.rs only.
- `crates/memra-engine/src/tp_expert_split.rs:249`: ep_busier_rank_experts: only the in-file test; glm5_tp.rs cites it in a doc comment.
- `crates/memra-engine/src/tp_transport.rs:746`: move_i32: pub, no caller anywhere.
- `crates/memra-engine/src/vision_gemma.rs:202`: gemma_prep_data_uri: only the definition
- `crates/memra-engine/src/vision_glm5.rs:229`: glm5_prep_data_uri: only the definition
- `crates/memra-engine/src/vision_pre.rs:174`: prep_data_uri used only in vision_pre unit test (727); server uses decode_data_uri + prep_image_bytes
- `crates/memra-engine/src/vision_step.rs:221`: StepImagePlan::n_rows used only in vision_step tests (886, 942); worker.rs:24751 calls the prepped-unit n_rows at line 289
- `crates/memra-engine/src/vision_step.rs:225`: n_prompt_tokens used only by vision_step tests and a #[test] fn in memra-server lib.rs:29758
- `crates/memra-gguf/src/config.rs:506`: ArchGeometryTable::layer_classes is called only from tests (config.rs:4876, 5555; micro_gguf.rs:970 in cfg(test)).
- `crates/memra-gguf/src/config.rs:766`: Step35Config::n_full_attn: only micro_gguf.rs:964 (cfg(test) from line 818).
- `crates/memra-gguf/src/config.rs:1082`: Glm5NextConfig::has_own_indexer: only config.rs:5695 (test).
- `crates/memra-gguf/src/config.rs:1147`: MlaConfig::v_view_dim: only micro_gguf.rs:865 (test).
- `crates/memra-gguf/src/config.rs:2733`: attn_out_gate_at: only config.rs:4728 (test). Engine uses layer_geometry/attn_out_gate directly.
- `crates/memra-gguf/src/config.rs:2739`: attn_gate_separate_at: no caller anywhere in crates/ or tools/.
- `crates/memra-gguf/src/config.rs:2802`: n_head_at: only micro_gguf.rs:985/1180 (tests).
- `crates/memra-gguf/src/config.rs:2814`: n_head_kv_at: only micro_gguf.rs:986 (test).
- `crates/memra-gguf/src/config.rs:2831`: is_swa_at: only micro_gguf.rs:976 (test). The family-table predicate naming cfg.is_swa_at has no production reader.
- `crates/memra-gguf/src/dsv4.rs:377`: CensusReport::is_clean: no caller.
- `crates/memra-gguf/src/execution_manifest.rs:42`: multimodal_prefill_capabilities: only mimo_v2/mod.rs:632 inside cfg(test) (from 474).
- `crates/memra-gguf/src/execution_manifest.rs:130`: RewriteQualifications::all_eligible: no caller.
- `crates/memra-gguf/src/micro_gguf.rs:267`: write_glm52_meta_only: only micro_gguf.rs:832 (cfg(test)).
- `crates/memra-gguf/src/micro_gguf.rs:316`: write_step35_rope_contract_fixture: callers are memra-cli/tests/step_rope_contract.rs, semantics_tests.rs and parallel.rs:1581 (cfg(test) from 1556). Test support only; prod-dead.
- `crates/memra-gguf/src/micro_gguf.rs:478`: write_step35_mtp_meta_only: only test callers (model_plan.rs:3247, execution_manifest.rs:801, semantics_tests.rs).
- `crates/memra-gguf/src/model_packs/mimo_v2/dflash.rs:233`: verify_pinned_auxiliary (dflash variant): only dflash.rs tests (297-329, cfg(test) from 274); the audio_tokenizer.rs function of the same name is a different item.
- `crates/memra-gguf/src/model_packs/mimo_v2/mod.rs:66`: tiny_text_plan: memra-reference/tests/mimo_tiny_text.rs and mod.rs tests only.
- `crates/memra-gguf/src/model_packs/mimo_v2/mod.rs:358`: inspect_pinned_mint_headers: no caller.
- `crates/memra-gguf/src/model_packs/nemotron_rnnt/mod.rs:204`: state_elements: nemotron_rnnt/tests.rs:117 only.
- `crates/memra-gguf/src/model_packs/nemotron_rnnt/mod.rs:215`: is_causal: nemotron_rnnt/tests.rs:99/123 only.
- `crates/memra-gguf/src/model_packs/qwen35/activation.rs:197`: select_activation is called only from activation.rs tests (cfg(test) from 216). Its doc says 'Used by runtime dispatch', but model.rs:760/781 reimplements the lookup (see finding).
- `crates/memra-gguf/src/model_packs/qwen3_tts/mod.rs:210`: AudioContract::frame_rate_hz: qwen3_tts/tests.rs only.
- `crates/memra-gguf/src/model_packs/qwen3_tts/mod.rs:214`: frame_ms: tests only (audio_api.rs uses its own FRAME_MS const).
- `crates/memra-gguf/src/model_packs/qwen3_tts/mod.rs:218`: self_consistent: tests only.
- `crates/memra-gguf/src/model_packs/qwen3_tts/mod.rs:298`: speaker_id: tests only (mod.rs:382 is a parameter name).
- `crates/memra-gguf/src/model_packs/qwen3_tts/mod.rs:318`: resolve_language_id: tests only.
- `crates/memra-gguf/src/model_packs/qwen3_tts/mod.rs:344`: is_suppressed: tests only.
- `crates/memra-gguf/src/model_packs/qwen3_tts/mod.rs:382`: prefill_rows: tests only.
- `crates/memra-gguf/src/model_packs/qwen3_tts/mod.rs:420`: prefill_width: tests only.
- `crates/memra-gguf/src/model_packs/qwen3_tts/mod.rs:426`: streaming_prefill_width: tests only.
- `crates/memra-gguf/src/model_packs/qwen3_tts/mod.rs:434`: trailing_text_is_constant_on_non_streaming: no caller at all.
- `crates/memra-gguf/src/model_packs/qwen3_tts/mod.rs:503`: frames_for_audible: tests only.
- `crates/memra-gguf/src/model_packs/qwen3_tts/mod.rs:819`: codec_total_upsample: tests only.
- `crates/memra-gguf/src/model_packs/qwen3_tts/mod.rs:932`: bind_talker: tests only; the pack executes nothing by design.
- `crates/memra-gguf/src/model_packs/qwen3_tts/mod.rs:945`: bind_codec_decoder: tests only.
- `crates/memra-gguf/src/model_packs/qwen3_tts/mod.rs:962`: codec_encoder_is_dead_weight: tests only.
- `crates/memra-gguf/src/model_packs/qwen4_exp/mod.rs:305`: expected_census wrapper: only tests (1124, 1234; cfg(test) from 1033). expected_census_for is the live entry.
- `crates/memra-gguf/src/model_packs/whisper/mod.rs:306`: WhisperCheckpoint::open: only whisper/tests.rs:224.
- `crates/memra-gguf/src/model_packs/whisper/mod.rs:362`: inspect_metadata: whisper/tests.rs only.
- `crates/memra-gguf/src/model_plan.rs:2780`: OperationSupport::all: only model_plan.rs:3333/3358/3365 (cfg(test) from 2836).
- `crates/memra-gguf/src/nvfp4_repack.rs:337`: swizzle_blockscale: only tests (nvfp4_repack.rs:1151+, source.rs:3597 in cfg(test) from 3114). Production uses unswizzle_blockscale.
- `crates/memra-gguf/src/op_registry.rs:146`: Surfaces::is_none: only mimo_v2/mod.rs test asserts.
- `crates/memra-gguf/src/op_registry.rs:520`: render_table: only the doc-pin test (op_registry.rs:590, 604). Intentional docs generator; prod-dead.
- `crates/memra-gguf/src/safetensors.rs:822`: parse_header_json: no caller; the _checked variant is the live one.
- `crates/memra-gguf/src/safetensors.rs:944`: parse_index_weight_map_json: no caller; _checked variant is live.
- `crates/memra-gguf/src/source.rs:1150`: Hy3RepackSource::dir: only source.rs:4290 (test).
- `crates/memra-gguf/src/source.rs:1164`: Hy3RepackSource::tensor_count: only source.rs:4291 (test).
- `crates/memra-gguf/src/source.rs:1168`: Hy3RepackSource::expert_stride: the only `.expert_stride(` call is source.rs:4308 (test); the other 200 hits are unrelated fields.
- `crates/memra-gguf/src/source.rs:1639`: SafetensorsSource::open_with_config: memra-gguf/tests/canonical_config.rs and source.rs unit tests only.
- `crates/memra-kv/src/lib.rs:240`: index_ring_default_rows: only in-crate tests call it (lib.rs:3476, 3616); production uses index_ring_rows.
- `crates/memra-kv/src/lib.rs:826`: LatentTailCapture::bytes: no caller; the server charges latent snapshots via LatentPlaneSnapshot::bytes (worker.rs:24045).
- `crates/memra-kv/src/lib.rs:2101`: ResidentTpKvCache::append_target: tp.rs uses prepare_append/commit_target; the hits are the private TpKvTransactionState method.
- `crates/memra-kv/src/lib.rs:2109`: ResidentTpKvCache::publish_append: same as append_target, no external caller.
- `crates/memra-kv/src/lib.rs:2412`: cache_bytes_per_token: callers use cache_bytes_per_token_for_plan (worker.rs:4674); cache_bytes_per_token_for_layers is dead with it.
- `crates/memra-kv/src/lib.rs:2518`: cache_ring_bytes_per_token and its _for_layers wrapper have zero callers.
- `crates/memra-kv/src/lib.rs:2568`: cache_ring_row_cap: server uses cache_ring_row_cap_for_plan.
- `crates/memra-kv/src/lib.rs:3000`: Cache::new_pp2: superseded by new_ppn/new_ppn_planned; only a doc mention remains.
- `crates/memra-kv/src/tiered/hostprefix.rs:102`: SealedImage::copy_committed: only tiered/tests.rs calls it.
- `crates/memra-kv/src/tiered/hostprefix.rs:117`: SealedImage::bundle: no caller (kv_tier_gate's .bundle() is a memra-tier type).
- `crates/memra-kv/src/tiered/hostprefix.rs:120`: SealedImage::payloads: no caller.
- `crates/memra-kv/src/tiered/hostprefix.rs:199`: ResidentCharge::lease: no .lease() call in server or engine.
- `crates/memra-kv/src/tiered/integration.rs:53`: reserve_peer_source: only day4/day5 tests.
- `crates/memra-kv/src/tiered/mod.rs:41`: Hierarchy has no non-test constructor; phase() unreachable.
- `crates/memra-kv/src/tiered/mod.rs:495`: ActiveEpoch::current: tests only.
- `crates/memra-kv/src/tiered/mod.rs:498`: ActiveEpoch::fork: tests only.
- `crates/memra-kv/src/tiered/mod.rs:506`: ActiveEpoch::commit: tests only.
- `crates/memra-kv/src/tiered/mod.rs:514`: ActiveEpoch::rollback: tests only.
- `crates/memra-kv/src/tiered/mod.rs:523`: ActiveEpoch::reallocate: tests only.
- `crates/memra-kv/src/tiered/policy.rs:17`: write_action: tiered/tests.rs only.
- `crates/memra-kv/src/tiered/policy.rs:44`: prefetch_decision: tests only.
- `crates/memra-kv/src/tiered/policy.rs:79`: recompute_vs_load: tests only (scheduler uses calibrated_recompute_vs_load).
- `crates/memra-kv/src/tiered/policy.rs:130`: eviction_victim: tests only.
- `crates/memra-kv/src/tiered/scheduler.rs:45`: Scheduler is not installed anywhere (doc says so); enqueue tests only.
- `crates/memra-kv/src/tiered/scheduler.rs:50`: enqueue_with_costs: day4 tests only.
- `crates/memra-kv/src/tiered/scheduler.rs:116`: restore_decision: day4 tests only.
- `crates/memra-kv/src/tiered/scheduler.rs:161`: Scheduler::tick: tests only.
- `crates/memra-kv/src/tiered/scheduler.rs:281`: Scheduler::consume: tests only.
- `crates/memra-kv/src/tiered/scheduler.rs:299`: Scheduler::release: tests only (other .release hits are governor/backing methods).
- `crates/memra-lanes/src/audio_stream.rs:225`: AudioSession::depth: no caller in server or tests outside the crate.
- `crates/memra-lanes/src/audio_stream.rs:328`: AudioScheduler::session: no caller.
- `crates/memra-lanes/src/audio_stream.rs:522`: replay_realtime: pub, but only the crate's own tests call it; could be cfg(test).
- `crates/memra-net-guard/src/lib.rs:73`: is_blocked: server calls classify_ip directly (image_fetch.rs:230); only unit tests use it.
- `crates/memra-reference/src/mimo_audio_codec_rvq.rs:125`: encode_20: only its own unit tests; the engine twin uses encode_one.
- `crates/memra-reference/src/mimo_audio_pcm_mel.rs:89`: compute_diagnostic: in-file tests only.
- `crates/memra-reference/src/mimo_modal_overlay.rs:35`: rows(): engine production uses into_parts(); only an engine test reads rows().
- `crates/memra-reference/src/mimo_modal_overlay.rs:39`: token_count(): engine calls are on its own MiMoGpuEmbeddingChunk type; reference tests only.
- `crates/memra-reference/src/mimo_modal_overlay.rs:43`: modal_counts(): reference tests only.
- `crates/memra-reference/src/mimo_modal_overlay.rs:47`: has_modal_payload(): engine has its own field/method; reference tests only.
- `crates/memra-reference/src/mimo_vision.rs:17`: patch_merger_bf16: only in-file tests; no engine twin consumes it.
- `crates/memra-reference/src/mimo_vision.rs:100`: column_patch_indices: in-file tests only.
- `crates/memra-reference/src/speech/decoder.rs:229`: WhisperDecoderSession::position: decoder tests only; bins do not call it.
- `crates/memra-reference/src/speech/decoder.rs:233`: self_kv: decoder tests only.
- `crates/memra-reference/src/speech/decoder.rs:239`: cross_kv: decoder tests only.
- `crates/memra-reference/src/speech/fastconformer.rs:153`: StreamState::contract: fastconformer tests only; rnnt-stage calls FastConformerEncoder::contract (line 319), a different method.
- `crates/memra-reference/src/speech/rnnt_head.rs:38`: GreedySession::last_token: no caller.
- `crates/memra-sampling/src/lib.rs:107`: SamplerIdentity::seed: no caller (doc: logging only).
- `crates/memra-sampling/src/lib.rs:257`: is_spec_sampling: only memra-server lib.rs test module (from line 12611) calls it.
- `crates/memra-server/src/admit_predict.rs:605`: pub(crate) under #[allow(dead_code)]. Only callers are in admit_predict tests (lines 1018, 1146); production uses kv_hat_ring.
- `crates/memra-server/src/build_id.rs:199`: pub(crate); every caller (lib.rs 15243, 17980, 29789+) is inside cfg(test) modules. It is a test oracle, not production code.
- `crates/memra-server/src/worker.rs:9189`: PrefixCache::insert has no production caller; it is a test seam (#[cfg_attr(not(test), allow(dead_code))], doc says unit tests only). Production inserts go through insert_demoting/insert_pinned_demoting.
- `crates/memra-server/src/worker.rs:9249`: insert_with_budget is called only from worker tests and from the host_glm.rs fixture at line 821, which sits inside #[cfg(test)] mod tests (host_glm.rs:499-500). It is production-dead by design.
- `crates/memra-server/src/worker.rs:25972`: Only the #[cfg(test)] module calls evaluate_decoupled_admission (worker.rs:49126 and later). worker is a private module (lib.rs:140 `mod worker;`), and #[allow(dead_code)] sits on the fn and on EvaluatedAdmissionOutcome. The admission loop never calls it. The test doc at 49093 wrongly calls it 'the 
- `crates/memra-tier/src/bank/adapters.rs:240`: PleTable::last_chunk: only tests/bank/main.rs calls it; ple_rows_tier.rs does not.
- `crates/memra-tier/src/bank/owner_proxy.rs:380`: ExpertBankProxy::with_bytes_at: native.rs:2029 is a source-text assertion, and 2449/2491 sit inside #[cfg(test)] mods (native.rs:2001, 2081). Otherwise only tests/bank/day79.rs calls it.
- `crates/memra-tier/src/bank/source.rs:22`: BankSource::install: called only from memra-tier tests (tests/contracts/v12_bindings.rs:218, tests/bank/day4.rs:300). The common-name hits were unrelated.
- `crates/memra-tier/src/bank/types.rs:165`: prefetch_batch: only memra-tier tests call it (tests/bank/integration.rs, tests/bank/main.rs).
- `crates/memra-tier/src/conformance/d2d_capture.rs:67`: d2d_capture_publish: test-only conformance schedule. worker.rs cites it in doc comments only.
- `crates/memra-tier/src/conformance/d2d_capture.rs:159`: d2d_capture_draft_publish: test-only conformance schedule (tests/contracts/d2d_capture_bindings.rs).
- `crates/memra-tier/src/conformance/d2d_receipt.rs:147`: d2d_receipt_witnessed: called only by other test-only schedules (d2d_restore.rs:104, d2d_capture.rs:90) and by tests. The production oracle in this file is receipt_digest, not this function.
- `crates/memra-tier/src/conformance/d2d_receipt.rs:182`: d2d_receipt_refused: test-only.
- `crates/memra-tier/src/conformance/d2d_restore.rs:82`: d2d_restore_ready: test-only conformance schedule. worker.rs mentions it in comments only.
- `crates/memra-tier/src/conformance/d2d_restore.rs:157`: d2d_restore_primed_before_its_wait_is_unordered: test-only.
- `crates/memra-tier/src/conformance/d2d_restore.rs:217`: d2d_restore_draft_ready: test-only conformance schedule (tests/contracts/d2d_restore_bindings.rs); no bin/src caller.
- `crates/memra-tier/src/conformance/d2d_restore.rs:316`: d2d_restore_draft_read_before_its_wait_is_unordered: test-only.
- `crates/memra-tier/src/conformance/d2h_device_receipt.rs:39`: d2h_device_receipt_lands_with_the_source_digest: tier_transfer.rs:5285 is inside the test mod; otherwise only memra-tier tests.
- `crates/memra-tier/src/conformance/d2h_span.rs:59`: d2h_span_batch: tier_transfer.rs:7059-7060 is inside the test mod (from 4298); otherwise only memra-tier tests.
- `crates/memra-tier/src/conformance/d2h_span.rs:120`: d2h_span_enqueue_failure_quarantines: test-only.
- `crates/memra-tier/src/conformance/h2d_deferred_checksum.rs:38`: h2d_deferred_checksum_lands_with_its_digests: test-only. worker.rs cites it in a comment.
- `crates/memra-tier/src/conformance/h2d_span.rs:83`: h2d_span_batch: the engine references (tier_transfer.rs:7441-7442) are inside the #[cfg(test)] mod that starts at tier_transfer.rs:4298; otherwise only memra-tier tests call it.
- `crates/memra-tier/src/conformance/h2d_span.rs:156`: h2d_span_enqueue_failure_quarantines: test-only.
- `crates/memra-tier/src/conformance/h2d_span.rs:187`: h2d_span_read_before_its_wait_is_unordered: test-only.
- `crates/memra-tier/src/conformance/h2d_span.rs:228`: h2d_span_fill_ordered_before_its_copy: test-only.
- `crates/memra-tier/src/conformance/mod.rs:37`: tier_cancel: test-only (tests/contracts/services.rs, memra-kv tiered/tests.rs).
- `crates/memra-tier/src/conformance/mod.rs:87`: peer_cancel: test-only.
- `crates/memra-tier/src/conformance/reader_fence.rs:81`: h2d_reader_fence: test-only (reader_fence_bindings.rs). worker.rs cites it in comments only.
- `crates/memra-tier/src/conformance/reader_fence.rs:140`: h2d_reader_issued_before_its_wait_is_unordered: test-only.
- `crates/memra-tier/src/conformance/recovery.rs:152`: required_resident_continuation: memra-kv lib.rs:4054/4103 are inside the continuation_gate_tests mod (lib.rs:4048); otherwise only memra-tier tests.
- `crates/memra-tier/src/conformance/revision_v11.rs:25`: budget_governor: production-dead; referenced only by memra-kv tiered/tests.rs and memra-tier tests. It is a conformance schedule by design (conformance/mod.rs:1-4), so moving it means moving its test bindings with it.
- `crates/memra-tier/src/conformance/revision_v11.rs:127`: governor_priority: test-only (memra-kv tiered/tests.rs).
- `crates/memra-tier/src/conformance/revision_v11.rs:173`: pinned_lease: test-only (tests/storage/day2.rs).
- `crates/memra-tier/src/conformance/revision_v11.rs:201`: pinned_quarantine: test-only.
- `crates/memra-tier/src/conformance/revision_v11.rs:255`: bank_complete_cancel: test-only.
- `crates/memra-tier/src/conformance/revision_v11.rs:279`: rows_complete_cancel: test-only.
- `crates/memra-tier/src/conformance/revision_v11.rs:303`: peer_complete_cancel: test-only.
- `crates/memra-tier/src/conformance/revision_v11.rs:326`: tier_identity: the worker.rs hits are variable/field names, not calls. Called only from memra-kv tiered/tests.rs.
- `crates/memra-tier/src/conformance/revision_v11.rs:400`: bank_uniform: test-only (tests/bank/main.rs).
- `crates/memra-tier/src/conformance/revision_v11.rs:456`: peer_lifetime: test-only.
- `crates/memra-tier/src/conformance/revision_v11.rs:499`: peer_capacity: test-only (tests/peer/fake.rs).
- `crates/memra-tier/src/conformance/revision_v11.rs:549`: peer_directed_grants: test-only.
- `crates/memra-tier/src/conformance/revision_v11.rs:652`: rows_bytes: test-only (tests/bank/main.rs).
- `crates/memra-tier/src/conformance/revision_v11.rs:700`: bundle_integrity: no reference anywhere (test, bin or src). Fully dead.
- `crates/memra-tier/src/conformance/revision_v11.rs:761`: peer_submit_epochs: test-only (tests/peer/fake.rs).
- `crates/memra-tier/src/conformance/revision_v11.rs:813`: bank_failed_sibling: test-only.
- `crates/memra-tier/src/conformance/revision_v11.rs:832`: rows_failed_sibling: test-only.
- `crates/memra-tier/src/conformance/revision_v11.rs:852`: tier_unknown: test-only (memra-kv tiered/tests.rs).
- `crates/memra-tier/src/conformance/revision_v11.rs:891`: bank_lifetime: test-only.
- `crates/memra-tier/src/conformance/revision_v11.rs:915`: rows_lifetime: test-only.
- `crates/memra-tier/src/conformance/revision_v12.rs:93`: ready_owner: test-only (tests/contracts/v12_bindings.rs).
- `crates/memra-tier/src/conformance/revision_v12.rs:190`: bank_source_install: test-only (v12_bindings.rs).
- `crates/memra-tier/src/conformance/span_receipt.rs:53`: span_receipt_lands_with_its_digests: test-only (span_receipt_bindings.rs).
- `crates/memra-tier/src/conformance/span_receipt.rs:98`: span_receipt_landed_flip_refuses_the_demote: test-only.
- `crates/memra-tier/src/conformance/span_receipt.rs:120`: span_receipt_resident_flip_refuses_the_promote: test-only.
- `crates/memra-tier/src/io/retirement.rs:33`: FakeTransferLease::observe: the fake is used only by tests/storage/retirement.rs.
- `crates/memra-tier/src/object_store/catalog.rs:84`: CatalogStore::open: only tests/storage/day4.rs calls it. The common-name hits were unrelated.
- `crates/memra-tier/src/object_store/catalog.rs:198`: CatalogStore::install_index: only tests/storage/day4.rs calls it.
- `crates/memra-tier/src/object_store/catalog.rs:355`: CatalogStore::lease_extent: the object_store/mod.rs:214 hit is a different method (ExtentStore::lease_extent). Only tests call the catalog version.
- `crates/memra-tier/src/object_store/catalog.rs:398`: CatalogStore::read_extent: object_store/mod.rs:274 is ExtentStore::read_extent, a different method. Only tests call the catalog version.
- `crates/memra-tier/src/object_store/catalog.rs:424`: CatalogStore::put_extent: only tests/storage/day4.rs calls it.
- `crates/memra-tier/src/telemetry/join.rs:99`: StorageTelemetry::observe: StorageTelemetry is constructed only in tests/storage/telemetry.rs. The other 'observe' hits are unrelated methods.
- `crates/memra-tier/src/telemetry/join.rs:123`: StorageTelemetry::json_line: only tests/storage/telemetry.rs calls it.
- `crates/memra-tier/src/tier/governor.rs:130`: Governor::dispatch: callers are governor.rs:244/325 inside #[cfg(test)] (from 225), memra-kv tiered/tests.rs and memra-tier tests/bank/integration.rs. hostprefix builds a Governor but never dispatches its queue.
- `crates/memra-tokenizer/src/chat.rs:471`: free fn chat::apply_chat_template_tools: chat.rs tests only; the Tokenizer method of the same name calls _ex directly.
- `crates/memra-tokenizer/src/detokenize.rs:103`: Detokenizer::vocab_size: tests only.
- `crates/memra-tokenizer/src/detokenize.rs:107`: Detokenizer::is_special: tests only.
- `crates/memra-tokenizer/src/detokenize.rs:214`: SpmDetokenizer::vocab_size: tests only.
- `crates/memra-tokenizer/src/detokenize.rs:218`: SpmDetokenizer::piece: no caller.
- `crates/memra-tokenizer/src/detokenize.rs:222`: SpmDetokenizer::kind: no caller.
- `crates/memra-tokenizer/src/lib.rs:999`: normalization_program: normalizer_tests only.
- `crates/memra-tokenizer/src/lib.rs:1005`: eos_ids(): no caller; eog_ids reads the field.
- `crates/memra-tokenizer/src/lib.rs:1040`: Tokenizer::split(): no caller (all .split hits are str::split).
- `crates/memra-validate/src/lib.rs:24`: rel_of: zero callers anywhere.
- `crates/memra-validate/src/lib.rs:51`: run_n_median: zero callers; bins reimplement median.
- `crates/memra-validate/src/lib.rs:71`: GateTally::check: only the crate's own unit test.
- `crates/memra-validate/src/lib.rs:80`: GateTally::finish: only the crate's own unit test.

### Near-identical bodies (159)

- `crates/memra-engine/cu/dsv4_dense_m1_exact_tail.cuh:109`: Deliberate verbatim body copy of dsv4_gemv_fp8_m_kernel so the gated arm's generated code cannot move (dsv4_gpu.cu:4297-4298). Merging is possible only with a SASS-identity gate.
- `crates/memra-engine/cu/dsv4_dense_m1_exact_tail.cuh:216`: dense_fast_dots_body is the exact-tail dots program with four-iteration load scheduling; FLAGS row says same numeric class. Once the DENSE_FAST seam is retired, the exact-tail twin can be deleted.
- `crates/memra-engine/cu/dsv4_gpu.cu:4373`: gemv_bf16_m and gemv_fp8_m share the stride/tiling preamble and 32-case switch; mergeable through one templated launcher table without touching kernels.
- `crates/memra-engine/cu/dsv4_gpu.cu:6443`: v3_stamped is v3 plus clock stamps, used only by bin hc_fused_gate.rs; intentional instrument twin.
- `crates/memra-engine/cu/dsv4_gpu.cu:8150`: f32acc twin of dsv4_sink_out_mq_kernel (different accumulation class); a template on the accumulator type would merge them.
- `crates/memra-engine/cu/flash_attn.cu:502`: Cross-TU copy of kernels.cu rms_norm_qkv_rope_f32 math. The bit-identity contract between fused append and split kernels depends on the copies staying textually equal. Move it to a shared header.
- `crates/memra-engine/cu/flash_attn.cu:502`: Same as above, against rms_norm_qkv_rope_cat_f32.
- `crates/memra-engine/cu/flash_attn.cu:2075`: g4 and g4o2 differ only in smem layout (occupancy-2). Both are reachable only behind the MEMRA_FAW_P1=0 rollback, so delete them with that seam, or template on the layout.
- `crates/memra-engine/cu/flash_attn.cu:4284`: fa_prefill_q_body (inline dequant, MEMRA_PRIME_DEQW=0 rollback) vs fa_prefill_qw_body (workspace, default): the MMA/softmax/PV code is byte-identical per lib.rs doc. Delete q with its rollback seam.
- `crates/memra-engine/cu/flash_attn.cu:6964`: Shells of v4 and v4_deep are near-identical (same partition, partO write). v4 survives only for the kf8vf8 gemma module; one template parameterized on the smem/walk would remove a copy.
- `crates/memra-engine/cu/flash_attn.cu:8181`: rows_v4_w and rows_v4_w_sp differ in split geometry (n_splits_max vs split_keys). A template flag would merge them.
- `crates/memra-engine/cu/flash_attn.cu:8918`: rows_dpl16_i2 vs rows_reg_w_i2: the latter is dead (no launch site). Resolve by deleting reg_w_i2, not by merging.
- `crates/memra-engine/cu/flash_attn.cu:9366`: v3_dcw vs v3_dcw_u8: identical shells calling fa_dec_v3_walk_u<4> vs <8>. u8 is the receipted-negative MEMRA_FA_UNROLL=8 twin; delete it.
- `crates/memra-engine/cu/flash_attn.cu:9366`: v3_dcw vs v3_dcw_prof: the same shell (PROF template flag).
- `crates/memra-engine/cu/flash_attn.cu:9495`: dcw_prof vs dcw_u8: the same shell with different fa_dec_v3_walk_u template args. One templated kernel would do.
- `crates/memra-engine/cu/kernels.cu:519`: rms_pre_add_rms_norm_q8z_f32 vs rms_pre_add_scale_rms_norm_q8_1 differ by the layer-scale multiply and the q8z vs q8_1 epilogue. A shared prologue helper would serve both.
- `crates/memra-engine/cu/kernels.cu:700`: rms_norm_q8_1 vs rms_norm_zq8_f32: the same norm with different quant epilogues (q8_1 vs zq8). Template on the epilogue.
- `crates/memra-engine/cu/kernels.cu:700`: rms_norm_q8_1 vs rms_pre_add_rms_norm_q8z_f32: a shared prologue.
- `crates/memra-engine/cu/kernels.cu:700`: rms_norm_q8_1 vs rms_norm_zq8_f32_v2: the same norm family.
- `crates/memra-engine/cu/kernels.cu:700`: rms_norm_q8_1 vs rms_pre_add_scale_rms_norm_q8_1: a shared prologue and epilogue.
- `crates/memra-engine/cu/kernels.cu:776`: rms_norm_zq8_f32 vs rms_pre_add_rms_norm_q8z_f32: the pre-add variant of the same norm+quant.
- `crates/memra-engine/cu/kernels.cu:776`: rms_norm_zq8_f32 vs _v2: two block forms selected by Engine::rms_norm_zq8_f32_arm. Keep one after A/B.
- `crates/memra-engine/cu/kernels.cu:776`: rms_norm_zq8_f32 vs rms_pre_add_scale_rms_norm_q8_1: the same norm family.
- `crates/memra-engine/cu/kernels.cu:941`: rms_norm_zq8_f32_v2 vs rms_pre_add_rms_norm_q8z_f32: the same norm family.
- `crates/memra-engine/cu/kernels.cu:941`: rms_norm_zq8_f32_v2 vs rms_pre_add_scale_rms_norm_q8_1: the same norm family.
- `crates/memra-engine/cu/kernels.cu:1389`: rms_norm_qkv_rope_cat_f32 vs rms_norm_qkv_rope_f32 differ only in output layout (concatenated vs split). A template on the store would merge them.
- `crates/memra-engine/cu/kernels.cu:1439`: rms_norm_qkv_rope_f32 vs qmatvec.cu qk_norm_rope_f32: a cross-TU copy of the qk-norm+rope body. Reusable header.
- `crates/memra-engine/cu/kernels.cu:1759`: add_rms_norm3_q8z_f32 and add_rms_norm_q8_1 share the add+row-reduce prologue verbatim. A device helper (add_reduce_row) would serve both.
- `crates/memra-engine/cu/kernels.cu:2413`: topk_rows_f32 vs topk_rows_shard_f32: per-thread top-k insertion is duplicated; shard adds a column range. A shared device helper would serve both.
- `crates/memra-engine/cu/kernels.cu:2769`: sdpa_naive_island_f32 vs sdpa_naive_w_f32: the windowed naive SDPA plus an island mask. Template on the visibility predicate.
- `crates/memra-engine/cu/kernels.cu:3240`: sdpa_naive_mask_f32 (qwen4_exp) vs sdpa_naive_w_lo_f32 (dflash): the same naive SDPA loop with different visibility predicates. Template on a visibility functor.
- `crates/memra-engine/cu/kernels.cu:3683`: sdpa_blocklist_f32 vs sdpa_naive_mask_f32: both qwen4_exp QSA attention oracles. One kernel with a source-list or mask mode would do.
- `crates/memra-engine/cu/kernels.cu:3683`: sdpa_blocklist_f32 vs sdpa_naive_w_lo_f32: the naive SDPA loop family.
- `crates/memra-engine/cu/kernels.cu:5397`: q4e_gu_g_body vs qmatvec_nvfp4_modelopt_sel_gu_silu_f32: NVFP4 sel gate/up twins (qwen4_exp only). Template on the group form.
- `crates/memra-engine/cu/kernels.cu:5624`: q4e_sdpa_blocklist_q8q5 and _hoist are bit-identical by construction (scale-load site only). Keep the hoist form and drop the original after its seam closes (qwen4_exp only).
- `crates/memra-engine/cu/kernels.cu:5624`: Same as the hoist pair: loader template.
- `crates/memra-engine/cu/kernels.cu:5693`: q4e_sdpa_blocklist_q8q5_hoist vs sdpa_blocklist_f32: a deliberate quantized-read twin (gated bit-identical to dequant+f32). Template on the K/V loader.
- `crates/memra-engine/cu/mimo_sink_attn.cu:25`: Same geometry (qk 192, v 128, 256 threads, learned sink) as mimo_swa_ring_attn decode_ring; ring addressing is the delta. Both are mimo_v2 probe components.
- `crates/memra-engine/cu/mmq_nvfp4_w4a8.cu:419`: int8 MMA dot body differing only in scale type (float d vs half2 dm); template in mmq_mma_i8.cuh
- `crates/memra-engine/cu/mmq_nvfp4_w4a8.cu:419`: int8 MMA dot body differing only in scale type (float d vs half2 dm); template in mmq_mma_i8.cuh
- `crates/memra-engine/cu/mmq_nvfp4_w4a8.cu:419`: int8 MMA dot body differing only in scale type; template in mmq_mma_i8.cuh
- `crates/memra-engine/cu/mmq_nvfp4_w4a8.cu:754`: per-TU copy of the q8_1 MMQ activation quantizer; one templated copy fits mmq_common.cuh
- `crates/memra-engine/cu/mmq_nvfp4_w4a8.cu:754`: per-TU copy of the q8_1 MMQ activation quantizer; one templated copy fits mmq_common.cuh
- `crates/memra-engine/cu/mmq_nvfp4_w4a8.cu:754`: per-TU copy of the q8_1 MMQ activation quantizer; one templated copy fits mmq_common.cuh
- `crates/memra-engine/cu/mmq_nvfp4_w4a8.cu:1178`: W4A8 and F8F4 pipelined tile loops share the schedule; policy template
- `crates/memra-engine/cu/mmq_q45k.cu:227`: int8 MMA dot body differing only in scale type (half2 dm vs float d, diffed); template in mmq_mma_i8.cuh
- `crates/memra-engine/cu/mmq_q45k.cu:227`: int8 MMA dot body differing only in scale type (half2 dm vs float d, diffed); template in mmq_mma_i8.cuh
- `crates/memra-engine/cu/mmq_q45k.cu:433`: ds4 variant of the per-TU q8_1 MMQ activation quantizer; one templated copy fits mmq_common.cuh
- `crates/memra-engine/cu/mmq_q45k.cu:433`: ds4 variant of the per-TU q8_1 MMQ activation quantizer; one templated copy fits mmq_common.cuh
- `crates/memra-engine/cu/mmq_q4_0.cu:515`: per-TU copy of the q8_1 MMQ activation quantizer; one templated copy fits mmq_common.cuh
- `crates/memra-engine/cu/qmatvec.cu:4265`: same-format schedule twin of one numeric program; mergeable by a template parameter
- `crates/memra-engine/cu/qmatvec.cu:6627`: differs only in nvfp4_load_g<false/true> (diffed); bench-only probes
- `crates/memra-engine/cu/qmatvec.cu:7559`: differs only in expert_dot_g vs expert_dot_g_v (diffed); template on the dot
- `crates/memra-engine/cu/qmatvec.cu:10074`: same-format schedule twin of one numeric program; mergeable by a template parameter
- `crates/memra-engine/cu/qmatvec.cu:10817`: same-format schedule twin of one numeric program; mergeable by a template parameter
- `crates/memra-engine/cu/qmatvec.cu:11308`: same-format schedule twin of one numeric program; mergeable by a template parameter
- `crates/memra-engine/cu/qmatvec.cu:11477`: runtime-t qkvg_tcol vs the never-launched compile-time-T body; wire the template and delete the runtime-t copy
- `crates/memra-engine/cu/qmatvec.cu:11631`: compile-time-T b4_tcol body vs the launched runtime-t kernel; one template
- `crates/memra-engine/cu/qmatvec.cu:12152`: differs only in acc bound TCOLS_MAX vs TCOLS16_MAX (diffed); template keeps both footprints
- `crates/memra-engine/cu/qmatvec.cu:13216`: differs only in MEMRA_Q8T_TMAX vs _WIDE acc bound (diffed); template<int TMAX>
- `crates/memra-engine/cu/qmatvec.cu:13312`: differs only in MEMRA_Q8T_TMAX vs _WIDE acc bound (diffed); template<int TMAX>
- `crates/memra-engine/cu/qmatvec.cu:13713`: same-format schedule twin of one numeric program (kda6 v2 vs v3 staging); mergeable by a template parameter
- `crates/memra-engine/cu/qmatvec.cu:14146`: same-format schedule twin (row1 rp v2 vs its ILP twin); mergeable by a template parameter
- `crates/memra-engine/src/decode.rs:417`: attn_in_norm_mixer vs _dc: same structure; eager uses position/host pos, dc uses device pos_d. One generic over the full-attn callee would cover all three.
- `crates/memra-engine/src/decode.rs:417`: attn_in_norm_mixer vs _dc_cap: third member of the same triplet.
- `crates/memra-engine/src/decode.rs:461`: attn_in_norm_mixer_dc vs _dc_cap: same norm/fuse skeleton, differ in the full-attn callee (dc vs dc_cap with bucket_max) and linear persistence flag.
- `crates/memra-engine/src/decode.rs:1817`: decode_step_dc vs decode_step_dc_cap_masked share the layer loop and the duplicated TG_PROBE block, but cap_masked lacks the gemma delegate and PP refusal that decode_step_dc has (see findings).
- `crates/memra-engine/src/dflash.rs:2365`: dflash2_propose_greedy_q and dflash2_propose_sampled share the same 30-line prelude (geometry check, d2t length check, topk_rows, hidden_proj matmul, dtoh, candidate guard, d2t remap); only the final walk differs.
- `crates/memra-engine/src/dsv4_gpu.rs:8483`: cmp_decode (legacy host-bounce) vs cmp_decode_dev (device); real overlap, but they are separate numeric programs kept as a byte-identity instrument; deleting the legacy arm is the better cut than merging
- `crates/memra-engine/src/dsv4_gpu.rs:16396`: dots_m_dev and dspark_head_dots are the same two FFI calls keyed on a different bool (dots_f32 vs dspark_head_f32); merge with a flag argument
- `crates/memra-engine/src/dsv4_grouped.rs:807`: project vs project_mrow: same table/output/mode guard. project adds cross, tail and host_offsets args.
- `crates/memra-engine/src/dsv4_grouped.rs:807`: project vs project_stream: same guard and pointer marshalling.
- `crates/memra-engine/src/dsv4_grouped.rs:906`: project_mrow is project_stream with memra_moe_kq_mrow_stream in place of memra_moe_kq_m1_stream. The precondition block and argument list are identical.
- `crates/memra-engine/src/dsv4_grouped.rs:952`: project_m1 repeats the same precondition block and marshalling, minus the projection id and with a half2 fn-pointer pick.
- `crates/memra-engine/src/dsv4_grouped.rs:952`: project_m1 vs project_stream: same shared prologue, one finding covers it.
- `crates/memra-engine/src/f16_ffi.rs:525`: try_f16_gemm_pre_into is try_f16_gemm_pre_into_off with off=0 except the post-scale; one body with an offset suffices
- `crates/memra-engine/src/f16_ffi.rs:656`: try_f16_gemm_pre = allocate y + try_f16_gemm_pre_into
- `crates/memra-engine/src/hybrid.rs:1595`: upload_step_bf16_column and upload_step_bf16_row differ only in return type and the two runtime upload methods (diff: 4 lines).
- `crates/memra-engine/src/hybrid_forward.rs:4637`: prime_cache_hyper_pp2_pipelined (4639-4820) and prime_cache_pp2_pipelined (5966-6117) are the same scoped-thread two-stage scheduler. They differ only in the stage-0/stage-1/tail closures and the payload width (streams*n_embd vs n_embd). Mergeable into one generic driver.
- `crates/memra-engine/src/hybrid_forward.rs:19700`: moe_gdec_token vs moe_gdec_token_q8: identical residency check, pointer collection and profile/hit bookkeeping; differ only in the f32 vs q8 kernel pair. moe_fused_epi_token_q8 pass 2 repeats the same pointer collection.
- `crates/memra-engine/src/hybrid_forward.rs:19830`: moe_cached_gemm vs moe_cached_gemm_q8: same dispatch_source_once/payload/consumed; only the matvec call differs (qmatvec_view vs qmatvec_expert_q8).
- `crates/memra-engine/src/hybrid_forward.rs:28405`: step35_batch_fa_rows_join vs step35_verify_fa_rows_join: verify is the single-cache case with len_back = t-1-r instead of 0; ladder closure and table build are copies.
- `crates/memra-engine/src/lib.rs:9477`: vs moe_pairs_matvec_q8_em L9417 (1.00). Bodies differ only in the kernel name (diff: 2 lines).
- `crates/memra-engine/src/lib.rs:9714`: vs moe_gate_up_gelu8_dev_q8_rows L9660 (0.92, 16 diff lines). Gemma gelu twins differ only by kernel and an index buffer.
- `crates/memra-engine/src/lib.rs:12602`: vs qmatvec_nvfp4_q8_ep_paired_slots_into L12678 (1.00). Bodies differ only in the kernel name and one error string.
- `crates/memra-engine/src/lib.rs:12873`: vs qmatvec_nvfp4_bf16_ep_down_slots_raw L12813 (0.96, 8 diff lines). Pairs vs slots, kernel-name-level twin.
- `crates/memra-engine/src/lib.rs:12988`: vs silu_mul_scaled_host_expf_bf16_sel_into L12933 (0.96, 28 diff lines). Same activation epilogue with EP-slot vs selected-row indexing. Mergeable behind an index-mode argument.
- `crates/memra-engine/src/lib.rs:13186`: vs qmatvec_nvfp4_bf16_sel_down_fma_into L13127 (0.94, 28 diff lines). Both have no callers: delete both.
- `crates/memra-engine/src/lib.rs:13186`: vs qmatvec_nvfp4_bf16_ep_down_fma_raw L13252 (0.86, 16 diff lines). Slice vs raw-pointer twin. Both dead.
- `crates/memra-engine/src/lib.rs:13252`: vs qmatvec_nvfp4_q8_ep_down_fma_raw L13384 (0.89, 18 diff lines). bf16 vs q8 activation twin. Both are dead (no callers): delete rather than merge.
- `crates/memra-engine/src/lib.rs:13252`: vs qmatvec_nvfp4_bf16_sel_down_fma_into L13127 (0.83). Both dead.
- `crates/memra-engine/src/lib.rs:15238`: vs rms_pre_add_scale_rms_norm_q8_1 L15380 (0.98). Same body plus one wa operand. The PDL gate also differs (pdl_wb_on is missing in the rms_pre variant).
- `crates/memra-engine/src/lib.rs:19419`: matmul_nvfp4_fused3 vs matmul_nvfp4_fused4: the same tiering (m=1 fused, 2..8 batched, 9..16 group delegate) with a different tensor count.
- `crates/memra-engine/src/lib.rs:19982`: matmul_q8_fused2 vs matmul_q8_fused3: both wrap q8_fused_params::<N> and the fused-core launch.
- `crates/memra-engine/src/lib.rs:20024`: q8_fused2_core vs q8_fused3_core: arity twins.
- `crates/memra-engine/src/lib.rs:20024`: q8_fused2_core vs q8_fused2_t_core: m=1 vs batched-column twin.
- `crates/memra-engine/src/lib.rs:20134`: matmul_q4_fused3 vs matmul_q4_fused3_into: allocating vs caller-owned output.
- `crates/memra-engine/src/lib.rs:20404`: matmul_q4_fused2 vs matmul_q4_fused3: N-tensor variants of one Q4_0 fused launcher. q8_fused_params already shows the const-generic N pattern.
- `crates/memra-engine/src/lib.rs:20404`: matmul_q4_fused2 vs matmul_q4_fused2_into: allocating vs caller-owned output.
- `crates/memra-engine/src/lib.rs:20525`: matmul_q4_fused2_into vs matmul_q4_fused3_into: N-ary variants (same pattern as #6).
- `crates/memra-engine/src/lib.rs:20645`: matmul_q4_fused2_batched vs matmul_q4_fused3_batched: arity twins.
- `crates/memra-engine/src/lib.rs:20854`: q8_fused3_core vs q8_fused3_t_core: m=1 vs batched-column twin of the triple.
- `crates/memra-engine/src/lib.rs:20939`: matmul_q8_fused2_t vs matmul_q8_fused3_t: arity twins.
- `crates/memra-engine/src/lib.rs:20990`: q8_fused2_t_core vs q8_fused3_t_core: arity twins.
- `crates/memra-engine/src/lib.rs:21299`: e4m3_fused2_core vs q8_fused2_core: same block-offset fused-pair launch. Only the kernel name and the e4m3 per-tensor scale args differ.
- `crates/memra-engine/src/lib.rs:21299`: e4m3_fused2_core vs q8_fused3_core: members of one fused-core family (dtype x arity x t). Eight launchers could be one parameterized builder.
- `crates/memra-engine/src/lib.rs:21299`: e4m3_fused2_core vs e4m3_fused3_core: arity twins.
- `crates/memra-engine/src/lib.rs:21299`: e4m3_fused2_core vs q8_fused2_t_core: the same fused-core family.
- `crates/memra-engine/src/lib.rs:21346`: e4m3_fused3_core vs q8_fused3_core: the same triple-launch body, dtype-parameterizable.
- `crates/memra-engine/src/lib.rs:21346`: e4m3_fused3_core vs q8_fused2_core: the same fused-core family.
- `crates/memra-engine/src/lib.rs:21346`: e4m3_fused3_core vs q8_fused3_t_core: the same fused-core family.
- `crates/memra-engine/src/lib.rs:21346`: e4m3_fused3_core vs e4m3_fused3_t_core: m=1 vs batched twin.
- `crates/memra-engine/src/lib.rs:21513`: e4m3_fused2_t_core vs q8_fused2_t_core: a dtype twin of the batched fused pair.
- `crates/memra-engine/src/lib.rs:21513`: e4m3_fused2_t_core vs e4m3_fused3_t_core: arity twins.
- `crates/memra-engine/src/lib.rs:21576`: e4m3_fused3_t_core vs q8_fused3_t_core: a dtype twin of the batched fused triple.
- `crates/memra-engine/src/lib.rs:21576`: e4m3_fused3_t_core vs q8_fused3_core: the same fused-core family.
- `crates/memra-engine/src/lib.rs:22714`: qmatvec_nvfp4_rp_arm_raw vs _ilp: bench-only twins that differ only in the kernel-name pick.
- `crates/memra-engine/src/lib.rs:23414`: moe_gate_up_preclamp8_q8_v2 is a copy of moe_gate_up_preclamp8_q8_w4 with another kernel name. The v2 copy is bench-only (never dispatched).
- `crates/memra-engine/src/lib.rs:25234`: axpy_rows_seq_md_into vs axpy_rows_seq_md_off_into: the _off twin is dead. Delete it, or merge it as an offset parameter.
- `crates/memra-engine/src/lib.rs:26043`: qmatvec_q8_0_b4_rp_t_into vs qmatvec_q8_0_qkv_rp_t_into: the same t<=8 / tw / t-loop dispatch ladder for two Step TP projection kernels.
- `crates/memra-engine/src/lib.rs:27043`: matvec_bf16_dual_into vs matvec_bf16_dual_view_into: an owned/view twin of the same matvec_bf16_dual launch.
- `crates/memra-engine/src/lib.rs:27559`: sdpa_naive vs sdpa_naive_view: the same sdpa_naive_f32 kernel, owned vs view.
- `crates/memra-engine/src/lib.rs:27689`: sdpa_naive_island vs sdpa_naive_w: the diff is only span_id vs causal and the kernel name.
- `crates/memra-engine/src/lib.rs:28025`: sdpa_naive_quantized_view_fmt and sdpa_naive_w_quantized_view_fmt differ only by the window argument. One launcher with window=0 would do.
- `crates/memra-engine/src/lib.rs:33344`: gdn_scan_s128 vs gdn_scan_s128_view: the same gdn_scan_s128 kernel. The view form is dead.
- `crates/memra-engine/src/lib.rs:33425`: gdn_prep_decode_b vs gdn_prep_decode_b_view: only the argument types differ (CudaSlice vs CudaView).
- `crates/memra-engine/src/lib.rs:33483`: gdn_scan_s128_batched vs _batched_view: owned-slice vs CudaView twin of one launch. Make it generic over DevicePtr, as linear_device_into is.
- `crates/memra-engine/src/mimo_audio_patch_forward.rs:162`: epilogue copied from mimo_vision_block_forward; mimo-only diagnostics, low value
- `crates/memra-engine/src/mla_ffi.rs:1552`: mla_absorb_q and mla_decompress_v carry the identical 4-door dispatch chain (coalesce, B200 split, generic split, shipped) differing only in kernel symbols and out_dim
- `crates/memra-engine/src/mla_ffi.rs:2536`: non-live twin is dead; delete rather than merge
- `crates/memra-engine/src/mmq_ffi.rs:1637`: iq4xs/q45k raw launchers repeat alloc-scratch, quantize, launch, rc-check; one helper over the FFI fn would do
- `crates/memra-engine/src/mmq_ffi.rs:1637`: same pattern as qmatvec_mmq_nvfp4_inner; mergeable with the trio
- `crates/memra-engine/src/mmq_ffi.rs:2040`: nvfp4_inner vs q45k_raw: same launcher skeleton
- `crates/memra-engine/src/mmq_ffi.rs:2346`: qmatvec_mmq_fp8_blk_scaled and _view duplicate the B200 refusal and all shape asserts; the slice form can delegate to the view form
- `crates/memra-engine/src/mmq_ffi.rs:3000`: kernel_check raw entries moe_f16g_gemm_sk_raw and moe_kq_gemm_sk_raw share the argument marshalling; gate-only
- `crates/memra-engine/src/parallel.rs:1149`: validate_fp8_expert_checkpoint vs validate_nvfp4_expert_checkpoint: same per-layer walk, different native finder and shape rules.
- `crates/memra-engine/src/qwen4exp_gpu.rs:4679`: launch_gdn_scan_at is launch_gdn_scan (gdn_scan_naive_f32) on column views with t=1; same kernel, grid and smem
- `crates/memra-engine/src/qwen4exp_gpu.rs:4728`: launch_gdn_scan_step_at is launch_gdn_scan_step on per-column views (same kernel, grid, args); _at can slice and call the base launcher
- `crates/memra-engine/src/qwen4exp_gpu.rs:4828`: launch_gdn_scan_at and launch_gdn_scan_step_at share the column-view prologue (conv/g/beta/o slicing); kernels differ, so only the view helper is mergeable
- `crates/memra-engine/src/qwen4exp_gpu.rs:5713`: launch_qmatvec_bf16w_off and _off_into launch the same qmatvec_bf16w_f32 with the same args on row-offset views; one view-taking launcher (or launch_qmatvec_bf16w on views) covers both
- `crates/memra-engine/src/spec.rs:4241`: mtp_step35_attn is the host-len twin of mtp_step35_attn_dcw. It is kept as the MEMRA_STEP35_DRAFT_DCW=0 rollback seam and also serves dcw-ineligible shapes (MEMRA_NO_FA_VEC, small window, head dim), so it can only be removed with the door, and only once the ineligible shapes get another arm.
- `crates/memra-engine/src/spec.rs:9451`: _deferred and _ready share an identical validation preamble (ensure_usable, mtp present, pos==0, cache.pos, ctx room) before the shared _scratch tail. Only the draft-length check and the alloc/copy differ.
- `crates/memra-engine/src/tier_transfer.rs:2275`: submit_d2d_capture and submit_d2d_restore duplicate the fault take, admission, receipt scratch, charge, Entry construction and the per-item wait/digest/copy/digest closure (2386-2440 vs 2590-2629). They differ only in destination resolution, the length rule and fence-at-submit. One helper taking the
- `crates/memra-engine/src/tp.rs:8081`: Host-oracle routed loops for E4M3 EP and NVFP4 TP (and run_routed_experts_nvfp4) are the same token x slot x (gate, up, act, down, weighted sum) program over different bank accessors.
- `crates/memra-engine/src/tp_ar.rs:480`: all_gather and all_reduce share the wait_for_consumer, push and record loop. Moderate overlap.
- `crates/memra-engine/src/tp_ar.rs:555`: all_reduce_1stage is a staging copy followed by the out-of-place body of all_reduce_1stage_into.
- `crates/memra-engine/src/tp_ar.rs:555`: Same launch skeleton as gather_i32.
- `crates/memra-engine/src/tp_ar.rs:636`: all_reduce_1stage_into and gather_i32 share the per-rank pointer resolution, alias check and launch loop. Only the FFI symbol and element type differ.
- `crates/memra-engine/src/tp_ar.rs:766`: hcpost repeats the 1stage_into pointer/launch skeleton with extra operands.
- `crates/memra-engine/src/tp_ar.rs:766`: Same skeleton as gather_i32 (pair 2).
- `crates/memra-engine/src/tp_transport.rs:456`: push_2d_i32 is push_2d with rows=1 on reinterpreted words. A generic over a 4-byte element would merge them.
- `crates/memra-engine/src/tp_transport.rs:952`: concat_parts_on_root is gather_parts restricted to the root consumer. Both carry the same three transport arms.
- `crates/memra-gguf/src/micro_gguf.rs:408`: write_step35_meta_only and write_step35_mtp_meta_only differ only in block range and MTP keys; test-fixture code, so mergeable at low value.
- `crates/memra-kv/src/lib.rs:839`: snapshot_plane repeats snapshot_tail's precondition and tail-copy block verbatim; it equals snapshot_plane_at(e, snapshot_tail(e)?), which the glm5 kpool GPU test already compares.
- `crates/memra-tier/src/conformance/d2d_capture.rs:67`: d2d_capture_publish and d2d_restore_ready share the landing/publish rule skeleton (0.86). They differ in the fence-at-submit and reader-wait rules. Test-only.
- `crates/memra-tier/src/conformance/d2d_capture.rs:159`: The draft variants (capture_draft_publish vs restore_draft_ready, 0.82) mirror each other in the same way as their trunk twins.
- `crates/memra-tier/src/conformance/d2d_capture.rs:159`: capture_draft_publish is capture_publish plus a draft plane (0.78). An optional draft argument would merge them.
- `crates/memra-tier/src/conformance/d2d_restore.rs:217`: restore_draft_ready is restore_ready plus a draft plane (0.76). Same merge as the capture pair.
- `crates/memra-tier/src/conformance/d2h_span.rs:59`: d2h_span_batch and h2d_span_batch are the same schedule mirrored by direction (0.98). They could become one direction-parameterized schedule. Test-only code, low value.

### Identical bodies (1)

- `crates/memra-engine/cu/mmq_q4_0.cu:147`: token-identical in mmq_q4_0.cu and mmq_q8_0.cu; both include mmq_mma_i8.cuh, one copy can live there

