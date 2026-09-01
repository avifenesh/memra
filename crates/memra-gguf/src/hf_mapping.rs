//! HF (transformers) tensor-name -> ggml tensor-name mapping.
//!
//! The memra engine ONLY ever asks for ggml-style names (e.g. `blk.3.attn_q.weight`,
//! hardcoded in model.rs / hybrid.rs). The safetensors source resolves on demand by
//! translating a requested **ggml name** into the **HF name** to look up in the file.
//!
//! Direction is ggml -> HF (one lookup per requested tensor; no full-table build needed).
//!
//! Coverage:
//!   * dense transformer (Qwen3 / Llama / OLMoE-attention style) — `ggml_to_hf`, zero-copy.
//!   * MoE stacked experts — per-expert ggml name `blk.{il}.ffn_{proj}_exps.{e}.weight` ->
//!     `model.layers.{il}.mlp.experts.{e}.{proj}_proj.weight` (gathered by HostExps::load_from_source).
//!   * hybrid (qwen35) linear-attn SSM tensors — `resolve_ggml`, with the value transforms
//!     (`-exp(A_log)`, norm `+1`, conv1d squeeze, V-reorder) materialized as owned buffers.
//!
//! The SSM name map + transforms are cited against llama.cpp's converter (conversion/qwen.py,
//! gguf-py/gguf/tensor_mapping.py) — see `resolve_ggml` and `TransformKind`.

use crate::config::{Arch, ModelConfig};

/// Translate a requested ggml tensor name into the HF safetensors name (DENSE tensors only —
/// a plain rename with no value transform). Returns `None` if the ggml name has no known HF
/// equivalent for this arch (e.g. SSM tensors, which go through `resolve_ggml`).
pub fn ggml_to_hf(ggml: &str, arch: &Arch) -> Option<String> {
    // Top-level tensors (arch-independent across Llama/Qwen dense).
    match ggml {
        "token_embd.weight" => return Some("model.embed_tokens.weight".into()),
        "output_norm.weight" => return Some("model.norm.weight".into()),
        "output.weight" => return Some("lm_head.weight".into()),
        _ => {}
    }

    // Per-layer: "blk.{il}.{suffix}".
    let rest = ggml.strip_prefix("blk.")?;
    let (il, suffix) = rest.split_once('.')?;

    // MiniMax-M3 MoE-side names live under `block_sparse_moe.` (Mixtral-style), with DeepSeek-V3
    // routing extras (e_score_correction_bias) and `shared_experts.{p}_proj` (plural, no gate_inp).
    // Dense FFN layers (moe_layer_freq==0) keep the standard `mlp.{p}_proj` names below.
    if arch.is_minimax() {
        let m3_suffix: Option<&str> = match suffix {
            "ffn_gate_inp.weight" => Some("block_sparse_moe.gate.weight"),
            // llama.cpp ggml name for the DeepSeek-V3 selection bias is `exp_probs_b.bias`.
            "exp_probs_b.bias" => Some("block_sparse_moe.e_score_correction_bias"),
            "ffn_gate_shexp.weight" => Some("block_sparse_moe.shared_experts.gate_proj.weight"),
            "ffn_up_shexp.weight" => Some("block_sparse_moe.shared_experts.up_proj.weight"),
            "ffn_down_shexp.weight" => Some("block_sparse_moe.shared_experts.down_proj.weight"),
            _ => None,
        };
        if let Some(s) = m3_suffix {
            return Some(format!("model.layers.{il}.{s}"));
        }
    }

    // Hy3 REAP tensors use dense layer-0 `mlp.{gate,up,down}_proj`, then routed MoE layers with
    // `mlp.router.*`, `mlp.shared_mlp.*`, and stacked `mlp.switch_mlp.*` projections.
    if arch.is_hy3() {
        let hy3_suffix: Option<&str> = match suffix {
            "ffn_gate_inp.weight" => Some("mlp.router.gate.weight"),
            // Current tencent/Hy3 checkpoints store the selection-only correction beside the
            // router module. Preview checkpoints used `mlp.router.expert_bias`; the safetensors
            // source keeps that older spelling as a lookup fallback.
            "exp_probs_b.bias" => Some("mlp.expert_bias"),
            "ffn_gate_shexp.weight" => Some("mlp.shared_mlp.gate_proj.weight"),
            "ffn_up_shexp.weight" => Some("mlp.shared_mlp.up_proj.weight"),
            "ffn_down_shexp.weight" => Some("mlp.shared_mlp.down_proj.weight"),
            "ffn_gate_exps.weight" => Some("mlp.switch_mlp.gate_proj.weight"),
            "ffn_up_exps.weight" => Some("mlp.switch_mlp.up_proj.weight"),
            "ffn_down_exps.weight" => Some("mlp.switch_mlp.down_proj.weight"),
            _ => None,
        };
        if let Some(s) = hy3_suffix {
            return Some(format!("model.layers.{il}.{s}"));
        }
    }

    // Step-3.7's official HF checkpoint uses stacked 3D experts under `moe.*`, a singular
    // `share_expert`, and a separate head-wise attention gate named `g_proj`. Keep this mapping
    // arch-scoped: the generic Qwen map uses `mlp.*` and would resolve plausible but absent names.
    if arch.is_step35() {
        let step_suffix: Option<&str> = match suffix {
            "attn_gate.weight" => Some("self_attn.g_proj.weight"),
            "ffn_gate_inp.weight" => Some("moe.gate.weight"),
            "exp_probs_b.bias" => Some("moe.router_bias"),
            "ffn_gate_shexp.weight" => Some("share_expert.gate_proj.weight"),
            "ffn_up_shexp.weight" => Some("share_expert.up_proj.weight"),
            "ffn_down_shexp.weight" => Some("share_expert.down_proj.weight"),
            "ffn_gate_exps.weight" => Some("moe.gate_proj.weight"),
            "ffn_up_exps.weight" => Some("moe.up_proj.weight"),
            "ffn_down_exps.weight" => Some("moe.down_proj.weight"),
            _ => None,
        };
        if let Some(s) = step_suffix {
            return Some(format!("model.layers.{il}.{s}"));
        }
    }

    // glm5_next KDA (Kimi Delta Attention) mixer tensors. Arch-scoped because these live under
    // `self_attn.` and would collide with the dense attention map below (`self_attn.q_proj` is a
    // KDA projection here, not a head-wise query). Every pair below is the same pair
    // `tensor_contract::add_kda` declares, and `glm5_next_kda_names_match_the_contract` asserts
    // that against the contract itself so a typo cannot survive.
    //
    // ALL PLAIN RENAMES, no value transform. The 2D matrices are ggml `ne = {in, out}` — row-major
    // [out][in] — which is byte-for-byte HF's `[out, in]`. The conv1d weights are HF
    // `[qkv, 1, kernel]` row-major, byte-identical to the contract's `[kernel, qkv]` ggml ne; the
    // extra unit axis is ne bookkeeping only, `GpuTensor::Float` carries whatever ne the source
    // reports, and `KdaAttnLayer::load` refuses loudly on any element count but `qkv * kernel`.
    if arch.is_glm5_next() {
        let kda_suffix: Option<&str> = match suffix {
            "kda_q.weight" => Some("self_attn.q_proj.weight"),
            "kda_k.weight" => Some("self_attn.k_proj.weight"),
            "kda_v.weight" => Some("self_attn.v_proj.weight"),
            "kda_f_a.weight" => Some("self_attn.f_a_proj.weight"),
            "kda_f_b.weight" => Some("self_attn.f_b_proj.weight"),
            "kda_g_a.weight" => Some("self_attn.g_a_proj.weight"),
            "kda_g_b.weight" => Some("self_attn.g_b_proj.weight"),
            "kda_b.weight" => Some("self_attn.b_proj.weight"),
            "kda_out.weight" => Some("self_attn.o_proj.weight"),
            "kda_a_log" => Some("self_attn.A_log"),
            "kda_dt.bias" => Some("self_attn.dt_bias"),
            "kda_o_norm.weight" => Some("self_attn.o_norm.weight"),
            "kda_q_conv1d.weight" => Some("self_attn.q_conv1d.weight"),
            "kda_k_conv1d.weight" => Some("self_attn.k_conv1d.weight"),
            "kda_v_conv1d.weight" => Some("self_attn.v_conv1d.weight"),
            // mHC hyper-connection parameters. The checkpoint spells these flat on the
            // layer (no module prefix) and the contract's `add_hyper_connections` uses the
            // same spelling for both dialects, so the map is an identity -- but it must be
            // WRITTEN, because the ggml->HF path returns None for anything unlisted and the
            // tensor then reads as absent. Found by the first real-artifact load, which
            // refused on `blk.0.hc_attn_fn is absent` rather than falling back to a serial
            // residual: the refusal is what made a missing NAME visible instead of a
            // silently different model.
            "hc_attn_base" => Some("hc_attn_base"),
            "hc_attn_fn" => Some("hc_attn_fn"),
            "hc_attn_scale" => Some("hc_attn_scale"),
            "hc_ffn_base" => Some("hc_ffn_base"),
            "hc_ffn_fn" => Some("hc_ffn_fn"),
            "hc_ffn_scale" => Some("hc_ffn_scale"),
            // noaux_tc SELECTION BIAS, nested under the router module. glm5_next spells it
            // `mlp.gate.e_score_correction_bias`, not the DeepSeek-V3 `mlp.e_score_correction_bias`
            // the generic map below would need, and the generic map carries no `exp_probs_b.bias`
            // row at all — so without this line the name resolved to None, `MoeWeights::load`
            // substituted a ZERO bias, and every routed layer selected its top-8 by raw sigmoid
            // score. Measured on the real artifact 2026-08-28: layer 3 chose experts
            // {41,253,255} where the reference chose {10,16,24}, 5/8 overlap, and the FFN branch
            // came out at cos 0.878 of the reference. The router LOGITS matched to cos 0.9999994,
            // which is why nothing upstream looked wrong.
            "exp_probs_b.bias" => Some("mlp.gate.e_score_correction_bias"),
            // SHARED EXPERT, spelled PLURAL (`shared_experts`, the DeepSeek-V3 spelling the
            // census recorded for this checkpoint). The generic map below spells it SINGULAR
            // (`mlp.shared_expert.*`, qwen3moe), so without these three rows the names resolved
            // to tensors that do not exist, `load_opt` answered None, and the always-on shared
            // expert was dropped from every MoE layer — a whole branch of the FFN, silently.
            // Found by `glm5_next_every_contract_tensor_resolves_through_the_engine_map` in the
            // same change that fixed the selection bias, not by a load failure.
            "ffn_gate_shexp.weight" => Some("mlp.shared_experts.gate_proj.weight"),
            "ffn_up_shexp.weight" => Some("mlp.shared_experts.up_proj.weight"),
            "ffn_down_shexp.weight" => Some("mlp.shared_experts.down_proj.weight"),
            _ => None,
        };
        if let Some(hf) = kda_suffix {
            return Some(format!("model.layers.{il}.{hf}"));
        }
    }

    let hf_suffix: &str = match suffix {
        // attention block
        "attn_norm.weight" => "input_layernorm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight", // qwen3 / qwen3moe
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        // some HF checkpoints carry attention biases (Qwen2 style) — map them too.
        "attn_q.bias" => "self_attn.q_proj.bias",
        "attn_k.bias" => "self_attn.k_proj.bias",
        "attn_v.bias" => "self_attn.v_proj.bias",
        // FFN (dense SwiGLU)
        "ffn_norm.weight" => "post_attention_layernorm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        // MoE router + shared expert (qwen3_moe dense-side tensors; experts handled separately).
        "ffn_gate_inp.weight" => "mlp.gate.weight",
        "ffn_gate_shexp.weight" => "mlp.shared_expert.gate_proj.weight",
        "ffn_up_shexp.weight" => "mlp.shared_expert.up_proj.weight",
        "ffn_down_shexp.weight" => "mlp.shared_expert.down_proj.weight",
        "ffn_gate_inp_shexp.weight" => "mlp.shared_expert_gate.weight",
        _ => return None,
    };
    Some(format!("model.layers.{il}.{hf_suffix}"))
}

/// HF name for a single expert's projection weight (for the MoE gather+concat path).
/// ggml stacks all experts into one 3D tensor; HF stores them separately. `proj` is one of
/// "gate", "up", "down". Two layouts:
///   * qwen3moe / olmoe: `mlp.experts.{e}.{gate,up,down}_proj.weight`
///   * MiniMax-M3 (Mixtral-style): `block_sparse_moe.experts.{e}.{w1,w2,w3}.weight`
///     (w1=gate, w2=down, w3=up — the Mixtral convention)
pub fn hf_expert_name(il: u32, e: u32, proj: &str, arch: &Arch) -> String {
    if arch.is_minimax() {
        let w = match proj {
            "gate" => "w1",
            "down" => "w2",
            "up" => "w3",
            other => panic!("unknown expert proj {other}"),
        };
        return format!("model.layers.{il}.block_sparse_moe.experts.{e}.{w}.weight");
    }
    let p = match proj {
        "gate" => "gate_proj",
        "up" => "up_proj",
        "down" => "down_proj",
        other => panic!("unknown expert proj {other}"),
    };
    format!("model.layers.{il}.mlp.experts.{e}.{p}.weight")
}

/// Resolved HF target for a requested ggml name.
pub enum HfTarget {
    /// A plain rename — borrow the on-disk bytes zero-copy.
    Plain(String),
    /// A rename PLUS a value transform that must materialize an owned f32 buffer (§2.1).
    Transform { hf: String, kind: TransformKind },
}

/// The value transforms the qwen35 HF->GGUF converter applies (llama.cpp conversion/qwen.py).
/// All operate on the dequantized-to-f32 tensor and return the GGUF-equivalent owned bytes.
#[derive(Clone, Copy)]
pub enum TransformKind {
    /// `ssm_a` <- `A_log`: elementwise `-exp(x)`, then V-head reorder (qwen.py:296-297, 496-503).
    NegExpReorderHeads,
    /// `ssm_dt.bias` <- `dt_bias`: V-head reorder only (qwen.py:496-503).
    ReorderHeads,
    /// `ssm_norm.weight` <- `linear_attn.norm.weight`: NO +1, NO reorder (qwen.py:302-303 carve-out).
    /// Materialized only to keep a single owned-arm code path; value is unchanged.
    Identity,
    /// qwen35 norm `+1` (every `*norm.weight` EXCEPT linear_attn.norm) (qwen.py:302-303).
    NormPlusOne,
    /// `ssm_conv1d.weight` <- `conv1d.weight`: squeeze `[C,1,K]->[C,K]`, V-channel reorder (qwen.py:300-301,505-512).
    Conv1dSqueezeReorder,
    /// `attn_qkv.weight` <- `in_proj_qkv.weight`: reorder ONLY the V row-block (rows 4096:8192) (qwen.py:478-486).
    QkvVReorderRows,
    /// `attn_gate.weight` <- `in_proj_z.weight`: reorder ALL rows (qwen.py:488-490).
    ZReorderRows,
    /// `ssm_alpha`/`ssm_beta` <- `in_proj_a`/`in_proj_b`: reorder ALL rows, head_dim=1 (qwen.py:492-494).
    AbReorderRows,
    /// `ssm_out.weight` <- `out_proj.weight`: reorder COLUMNS (in-dim), head_dim=128 (qwen.py:514-516).
    OutReorderCols,
    /// `attn_k_b.weight` <- the fused `kv_b_proj.weight` (contract transform `SplitMlaKv`, key half).
    /// Emits the 3D absorb operand `ne = [nope, kv_rank, head]`, TRANSPOSED per head.
    MlaKeyUpSplit,
    /// `attn_v_b.weight` <- the fused `kv_b_proj.weight` (`SplitMlaKv`, value half). Emits the 3D
    /// decompress operand `ne = [kv_rank, v, head]` — already the consumer's order, no transpose.
    MlaValueUpSplit,
}

/// Row split of the fused MLA `kv_b_proj` weight into the key-up / value-up planes the absorbed
/// MLA core consumes. ONE function so the two halves cannot drift apart.
///
/// Convention, byte-pinned against `micro_gguf::write_glm_dsa_micro` (which generates `attn_kv_b`
/// as the source of truth and derives `attn_k_b`/`attn_v_b` from it) and against
/// `memra_reference`'s `split_mla_kv` (whose `self_test_split_mla_kv` pins the HF `expand_kv`
/// lineage): the fused weight is row-major `[head*(nope+v), kv_rank]`, per-head contiguous blocks
/// with the `nope` key rows first and the `v` value rows after.
///
///   k_b element (h, r, p) at `h*rank*nope + r*nope + p` = fused[(h*(nope+v) + p)*rank + r]
///   v_b element (h, j, r) at `h*v*rank    + j*rank + r` = fused[(h*(nope+v) + nope + j)*rank + r]
///
/// A wrong convention preserves element counts, so the shape asserts in `MlaAttnLayer::load`
/// cannot catch it — `mla_kv_split_matches_the_micro_fixture_convention` can.
///
/// DTYPE-AGNOSTIC BY CONSTRUCTION: the caller hands this f32 (`deq_f32` dequantizes BF16, F16,
/// F32, F8-E4M3 and modelopt NVFP4 alike), and the output is F32 with a 3-D `ne`. That is
/// deliberate — the absorb/decompress kernels (`mla_absorb_q`, `mla_decompress_v`) take
/// `&CudaSlice<f32>` and there is no quantized 3-D resident layout in the engine (see the
/// `ne.len() != 2` refusal in `GpuTensor::load_from_source_inner`). A checkpoint that ships
/// `kv_b_proj` quantized — the vendor FP8 artifact keeps it BF16, but a third-party NVFP4 mint
/// or a tighter re-mint of ours need not — therefore loads through exactly this path, at the
/// cost of one f32 materialization per MLA layer and no kernel change.
fn split_mla_kv_plane(fused: &[f32], key: bool, nope: usize, v: usize, rank: usize) -> Vec<f32> {
    let per_head = (nope + v) * rank;
    let heads = fused.len() / per_head;
    let rows = if key { nope } else { v };
    let mut out = vec![0f32; heads * rows * rank];
    for h in 0..heads {
        let base = h * per_head + if key { 0 } else { nope * rank };
        for row in 0..rows {
            for r in 0..rank {
                let src = fused[base + row * rank + r];
                if key {
                    // [h][rank][nope] — the transpose the absorb GEMM operand needs.
                    out[h * rank * nope + r * nope + row] = src;
                } else {
                    // [h][v][rank] — already the decompress order.
                    out[h * v * rank + row * rank + r] = src;
                }
            }
        }
    }
    out
}

impl TransformKind {
    /// Apply the transform to `data` (dequantized f32, row-major matching the HF shape whose ggml
    /// `ne` is `ne_in`). Returns the new ggml `ne` and the owned little-endian f32 bytes.
    pub fn apply(
        &self,
        data: &mut [f32],
        ne_in: Vec<u64>,
        cfg: &ModelConfig,
    ) -> (Vec<u64>, Vec<u8>) {
        // Identity and norm folding are geometry-free and are shared by non-SSM architectures
        // (Step and MiniMax). Do not demand Qwen's hybrid head metadata before reaching them.
        if matches!(self, TransformKind::Identity) {
            return (ne_in, f32_to_le(data));
        }
        if matches!(self, TransformKind::NormPlusOne) {
            for x in data.iter_mut() {
                *x += 1.0;
            }
            return (ne_in, f32_to_le(data));
        }
        // MLA kv_b split: glm5_next geometry, NOT the qwen hybrid head metadata `head_params`
        // demands (it `expect`s an ssm config this family does not have). Must precede that call.
        if let TransformKind::MlaKeyUpSplit | TransformKind::MlaValueUpSplit = self {
            let glm5 = cfg
                .glm5
                .as_ref()
                .expect("MLA kv_b split needs the glm5_next config geometry");
            let (nope, v) = (glm5.qk_nope_head_dim as usize, glm5.v_head_dim as usize);
            // ne_in = [kv_rank, head*(nope+v)] (ggml reverses the HF row-major shape).
            let rank = ne_in[0] as usize;
            let total_out = ne_in[1] as usize;
            let per_head = nope + v;
            assert_eq!(
                total_out % per_head,
                0,
                "MLA kv_b out-features {total_out} is not a multiple of nope+v {per_head} — the \
                 checkpoint geometry disagrees with the glm5_next config"
            );
            let heads = total_out / per_head;
            assert_eq!(
                data.len(),
                heads * per_head * rank,
                "MLA kv_b has {} elements, geometry requires {heads} x {per_head} x {rank}",
                data.len()
            );
            let key = matches!(self, TransformKind::MlaKeyUpSplit);
            let out = split_mla_kv_plane(data, key, nope, v, rank);
            let ne = if key {
                vec![nope as u64, rank as u64, heads as u64]
            } else {
                vec![rank as u64, v as u64, heads as u64]
            };
            return (ne, f32_to_le(&out));
        }
        let (nk, nv, hk, hv) = head_params(cfg);
        match self {
            TransformKind::Identity
            | TransformKind::NormPlusOne
            | TransformKind::MlaKeyUpSplit
            | TransformKind::MlaValueUpSplit => unreachable!(),
            TransformKind::NegExpReorderHeads => {
                for x in data.iter_mut() {
                    *x = -x.exp();
                }
                // 1D [nv]: per-head reorder, head_dim=1.
                let out = reorder_v_heads_axis0(data, nv, nk, 1);
                (ne_in, f32_to_le(&out))
            }
            TransformKind::ReorderHeads => {
                let out = reorder_v_heads_axis0(data, nv, nk, 1);
                (ne_in, f32_to_le(&out))
            }
            TransformKind::AbReorderRows => {
                // in_proj_a/b: HF [out=nv, in=hidden] -> ne [hidden, nv]. Reorder the `nv` out-rows
                // (head_dim=1). data is row-major [nv][hidden]; row index = V-head.
                let in_f = ne_in[0] as usize;
                let out = reorder_rows_v(data, nv, nk, 1, in_f, 0, nv);
                (ne_in, f32_to_le(&out))
            }
            TransformKind::ZReorderRows => {
                // in_proj_z: HF [out=value_dim, in=hidden] -> ne [hidden, value_dim]. Reorder ALL
                // value_dim out-rows, head_dim=hv. data row-major [value_dim][hidden].
                let in_f = ne_in[0] as usize;
                let total_out = ne_in[1] as usize; // value_dim = nv*hv
                let out = reorder_rows_v(data, nv, nk, hv, in_f, 0, total_out);
                (ne_in, f32_to_le(&out))
            }
            TransformKind::QkvVReorderRows => {
                // in_proj_qkv: HF [out=conv_dim, in=hidden] -> ne [hidden, conv_dim].
                // q rows [0, qk), k rows [qk, 2qk) UNTOUCHED; V rows [2qk, conv_dim) reordered (head_dim=hv).
                let in_f = ne_in[0] as usize;
                let conv_dim = ne_in[1] as usize;
                let qk = nk * hk; // q_dim = k_dim = num_k_heads*head_k_dim
                let out = reorder_rows_v(data, nv, nk, hv, in_f, 2 * qk, conv_dim);
                (ne_in, f32_to_le(&out))
            }
            TransformKind::Conv1dSqueezeReorder => {
                // conv1d HF [C, 1, K] -> ne_in reversed = [K, 1, C]. Squeeze -> ggml [K, C] (ne[K,C]).
                // data row-major [C][1][K] == [C][K]; channels [0,2qk) untouched, V channels reordered (hv).
                let k = ne_in[0] as usize; // kernel taps (4)
                // ne_in could be [K,1,C] or [K,C]; channel count is the product of the rest.
                let c: usize = ne_in[1..].iter().map(|&d| d as usize).product();
                let qk = nk * hk;
                let out = reorder_rows_v(data, nv, nk, hv, k, 2 * qk, c);
                (vec![k as u64, c as u64], f32_to_le(&out))
            }
            TransformKind::OutReorderCols => {
                // out_proj: HF [out=hidden, in=value_dim] -> ne [value_dim, hidden]. Reorder the
                // `value_dim` INPUT columns (head_dim=hv). data row-major [hidden][value_dim].
                let value_dim = ne_in[0] as usize; // in-features
                let hidden = ne_in[1] as usize; // out-features
                let out = reorder_cols_v(data, hidden, value_dim, nv, nk, hv);
                (ne_in, f32_to_le(&out))
            }
        }
    }
}

impl TransformKind {
    /// Apply this transform to a REPACKED GGUF NVFP4 weight WITHOUT dequantizing, when it is a pure
    /// structural V-head permutation (qkv/z/a/b out-row reorder, out_proj in-column reorder). Keeps
    /// the weight NVFP4 (no ~8x f32 blow-up). Returns `Some((ne, packed_bytes))` for the permutable
    /// kinds, `None` for the value transforms (`-exp`, `+1`, conv1d squeeze, identity) which must go
    /// through f32. `packed`/`out_f`/`in_f` describe the repacked weight (`ne = [in_f, out_f]`).
    pub fn apply_nvfp4(
        &self,
        packed: &[u8],
        out_f: usize,
        in_f: usize,
        cfg: &ModelConfig,
    ) -> Option<(Vec<u64>, Vec<u8>)> {
        use crate::nvfp4_repack::{reorder_cols_nvfp4, reorder_rows_nvfp4};
        // The MLA kv_b split is a RESHAPE, not a permutation of 2D rows: its output is 3D and its
        // consumers are f32-only kernels. There is no packed fast path, and `head_params` would
        // panic on this family's config, so refuse before it is called.
        if matches!(
            self,
            TransformKind::MlaKeyUpSplit | TransformKind::MlaValueUpSplit
        ) {
            return None;
        }
        let (nk, nv, hk, hv) = head_params(cfg);
        let row_bytes = (in_f / 64) * 36;
        let ne = vec![in_f as u64, out_f as u64];
        match self {
            TransformKind::QkvVReorderRows => {
                // V band = out-rows [2*qk, conv_dim); q/k rows untouched (copied through).
                let qk = nk * hk;
                Some((
                    ne,
                    reorder_rows_nvfp4(packed, out_f, row_bytes, nv, nk, hv, 2 * qk, out_f),
                ))
            }
            TransformKind::ZReorderRows => {
                // all value_dim=out_f out-rows reordered (head_dim=hv).
                Some((
                    ne,
                    reorder_rows_nvfp4(packed, out_f, row_bytes, nv, nk, hv, 0, out_f),
                ))
            }
            TransformKind::AbReorderRows => {
                // out-rows reordered, head_dim=1 (a/b: out_f == nv).
                Some((
                    ne,
                    reorder_rows_nvfp4(packed, out_f, row_bytes, nv, nk, 1, 0, out_f),
                ))
            }
            TransformKind::OutReorderCols => {
                // in-columns (value_dim) reordered, head_dim=hv (block-aligned: hv % 64 == 0).
                Some((ne, reorder_cols_nvfp4(packed, out_f, in_f, nv, nk, hv)))
            }
            // value transforms (operate on tiny BF16 tensors) — no NVFP4 fast path.
            _ => None,
        }
    }
}

impl TransformKind {
    /// Apply this transform to a BLOCK-128 FP8 weight WITHOUT dequantizing — permuting the e4m3
    /// codes and the `[ceil(out/128), ceil(in/128)]` scale grid TOGETHER. Returns
    /// `Some((ne, codes, scales, rows, cols))`, or `None` when the permutation is not expressible on
    /// the grid (see the alignment proof) or the kind is not a pure permutation.
    ///
    /// WHY THIS EXISTS (lane/fp8-blk128-decode, 2026-08-05). Qwen3.5/3.6 hybrids reach their
    /// linear-attn projections through a V-head permutation, and `find_fp8_native` previously
    /// refused block-128 on every Transform target with "block grid does not survive a row
    /// permutation". On the 27B that is **144 of 208** FP8 projections — every `in_proj_qkv`,
    /// `in_proj_z` and `out_proj` in all 48 linear-attn layers, i.e. 69% of the class's bytes and
    /// EVERY layer of the flagship shape. With that refusal standing, a block-128 Qwen3.6/3.8-FP8
    /// checkpoint gets native residency on its 64 full-attn projections only and keeps the Q8_0
    /// slab (and its 1.0625 B/weight) for the rest — the day-one gap this lane exists to close
    /// would remain two-thirds open.
    ///
    /// THE ALIGNMENT PROOF (why this is EXACT, not approximate). A V-head reorder is a pure index
    /// permutation `p`: the post-transform element at (o, e) is the pre-transform element at
    /// (p[o], e) for a row reorder, (o, p[e]) for a column reorder. Its value is
    /// `code[p[o]][e] * s[p[o] >> 7][e >> 7]`. So permuting the CODES by `p` reproduces the
    /// numerator exactly, and the result is exact iff a permuted grid can reproduce the scale — i.e.
    /// iff `p[o] >> 7` depends only on `o >> 7`. `block128_perm` tests exactly that and returns the
    /// induced block permutation, so a `Some` return is bit-exact by construction and a non-aligned
    /// permutation is refused rather than approximated. No dequantization happens, so nothing is
    /// re-rounded: this is strictly better than the f32 hop the Q8_0 fallback takes.
    ///
    /// The condition HOLDS on the real shapes because `linear_value_head_dim == 128` (Qwen3.5/3.6
    /// alike): one V-head is exactly one 128-row scale block, and every reorder band starts at a
    /// multiple of 128 (`in_proj_qkv`'s V band at `2*nk*hk`). It also holds trivially for
    /// `in_proj_a`/`in_proj_b` (out_f = nv <= 128 = one grid row, so every row shares one scale and
    /// the grid is unchanged). The test is on the permutation itself, not on those coincidences, so
    /// a future model with a different head_dim gets a correct refusal instead of wrong numbers.
    pub fn apply_fp8_blk(
        &self,
        codes: &[u8],
        out_f: usize,
        in_f: usize,
        scales: &[f32],
        cols: usize,
        cfg: &ModelConfig,
    ) -> Option<(Vec<u64>, Vec<u8>, Vec<f32>, usize, usize)> {
        // See `apply_nvfp4`: the MLA kv_b split has no packed arm, and `head_params` panics here.
        if matches!(
            self,
            TransformKind::MlaKeyUpSplit | TransformKind::MlaValueUpSplit
        ) {
            return None;
        }
        let (nk, nv, hk, hv) = head_params(cfg);
        if codes.len() != out_f * in_f || cols != in_f.div_ceil(128) {
            return None;
        }
        let rows = out_f.div_ceil(128);
        if scales.len() != rows * cols {
            return None;
        }
        let ne = vec![in_f as u64, out_f as u64];
        // (axis_len, permutation) for the axis this kind reorders; `true` = out-rows.
        let (row_axis, perm) = match self {
            // V band = out-rows [2*qk, out_f); q/k rows identity.
            TransformKind::QkvVReorderRows => {
                (true, v_perm(out_f, nv, nk, hv, 2 * nk * hk, out_f)?)
            }
            TransformKind::ZReorderRows => (true, v_perm(out_f, nv, nk, hv, 0, out_f)?),
            TransformKind::AbReorderRows => (true, v_perm(out_f, nv, nk, 1, 0, out_f)?),
            TransformKind::OutReorderCols => (false, v_perm(in_f, nv, nk, hv, 0, in_f)?),
            // value transforms (-exp, +1, conv1d squeeze, identity) are not permutations.
            _ => return None,
        };
        let bperm = block128_perm(&perm)?;
        if row_axis {
            let mut nc = vec![0u8; codes.len()];
            for (d, &s) in perm.iter().enumerate() {
                nc[d * in_f..(d + 1) * in_f].copy_from_slice(&codes[s * in_f..(s + 1) * in_f]);
            }
            let mut ns = vec![0f32; scales.len()];
            for (db, &sb) in bperm.iter().enumerate() {
                ns[db * cols..(db + 1) * cols].copy_from_slice(&scales[sb * cols..(sb + 1) * cols]);
            }
            Some((ne, nc, ns, rows, cols))
        } else {
            let mut nc = vec![0u8; codes.len()];
            for o in 0..out_f {
                let base = o * in_f;
                for (d, &s) in perm.iter().enumerate() {
                    nc[base + d] = codes[base + s];
                }
            }
            let mut ns = vec![0f32; scales.len()];
            for ob in 0..rows {
                for (db, &sb) in bperm.iter().enumerate() {
                    ns[ob * cols + db] = scales[ob * cols + sb];
                }
            }
            Some((ne, nc, ns, rows, cols))
        }
    }
}

/// The V-head reorder as an explicit `dst -> src` index permutation over an axis of `n` indices,
/// identity outside `[lo, hi)`. Mirrors `reorder_v_axis`/`reorder_rows_v`/`reorder_cols_v` exactly
/// (they all write `out[dst] = data[src]` with `dst_head = j*nk + g`, `src_head = g*vpk + j`), so a
/// permute driven by this is the same relabeling those functions perform. `None` when the band does
/// not match `nv*head_dim` — the same debug_assert the primitives carry, promoted to a refusal.
fn v_perm(
    n: usize,
    nv: usize,
    nk: usize,
    head_dim: usize,
    lo: usize,
    hi: usize,
) -> Option<Vec<usize>> {
    if hi > n || hi.checked_sub(lo)? != nv * head_dim || nv % nk != 0 {
        return None;
    }
    let vpk = nv / nk;
    let mut p: Vec<usize> = (0..n).collect();
    for j in 0..vpk {
        for g in 0..nk {
            let (dst_head, src_head) = (j * nk + g, g * vpk + j);
            for d in 0..head_dim {
                p[lo + dst_head * head_dim + d] = lo + src_head * head_dim + d;
            }
        }
    }
    Some(p)
}

/// The permutation `perm` induces on 128-element blocks, or `None` if it does not act blockwise.
/// `Some(b)` means `perm[i] >> 7 == b[i >> 7]` for every `i` — exactly the condition under which a
/// block-128 scale grid can be permuted alongside the data (see `apply_fp8_blk`'s proof).
fn block128_perm(perm: &[usize]) -> Option<Vec<usize>> {
    let mut out = vec![usize::MAX; perm.len().div_ceil(128)];
    for (d, &s) in perm.iter().enumerate() {
        let slot = &mut out[d >> 7];
        if *slot == usize::MAX {
            *slot = s >> 7
        } else if *slot != s >> 7 {
            return None;
        }
    }
    Some(out)
}

/// (num_k_heads, num_v_heads, head_k_dim, head_v_dim) from cfg.ssm / cfg fields (qwen35).
fn head_params(cfg: &ModelConfig) -> (usize, usize, usize, usize) {
    let ssm = cfg.ssm.as_ref().expect("ssm config for hybrid transform");
    let nk = ssm.group_count as usize; // linear_num_key_heads = 16
    let nv = ssm.time_step_rank as usize; // linear_num_value_heads = 32 (stored here, see config.rs)
    let hk = ssm.state_size as usize; // linear_key_head_dim = 128
    let hv = hk; // linear_value_head_dim == key (qwen35: both 128)
    (nk, nv, hk, hv)
}

/// Core V-head permutation primitive for a flat axis of `num_v_heads*head_dim` elements
/// (qwen.py:366-376 `_reorder_v_heads`). Output enumerates `j` (within-K-group) OUTER, `g`
/// (K-group) INNER: output V-head slot `j*num_k_heads + g` <- source HF V-head `g*num_v_per_k + j`.
/// `block` is one contiguous V axis (length num_v_heads*head_dim); returns the permuted copy.
fn reorder_v_axis(
    block: &[f32],
    num_v_heads: usize,
    num_k_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let num_v_per_k = num_v_heads / num_k_heads;
    debug_assert_eq!(block.len(), num_v_heads * head_dim);
    let mut out = vec![0f32; block.len()];
    for j in 0..num_v_per_k {
        for g in 0..num_k_heads {
            let dst_head = j * num_k_heads + g;
            let src_head = g * num_v_per_k + j;
            let dst = dst_head * head_dim;
            let src = src_head * head_dim;
            out[dst..dst + head_dim].copy_from_slice(&block[src..src + head_dim]);
        }
    }
    out
}

/// 1D tensor [num_v_heads*head_dim] (here head_dim is the per-head element count). Thin wrapper.
fn reorder_v_heads_axis0(
    data: &[f32],
    num_v_heads: usize,
    num_k_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    reorder_v_axis(data, num_v_heads, num_k_heads, head_dim)
}

/// Reorder the V-head rows of a row-major [out_rows][in_f] matrix in-place over the row-band
/// `[row_lo, row_hi)` (which must span `num_v_heads*head_dim` rows). Rows outside the band copy
/// through unchanged. Used for in_proj_qkv (V band), in_proj_z / a / b (whole band), conv1d (V band).
fn reorder_rows_v(
    data: &[f32],
    num_v_heads: usize,
    num_k_heads: usize,
    head_dim: usize,
    in_f: usize,
    row_lo: usize,
    row_hi: usize,
) -> Vec<f32> {
    let num_v_per_k = num_v_heads / num_k_heads;
    let mut out = data.to_vec();
    debug_assert_eq!(row_hi - row_lo, num_v_heads * head_dim);
    for j in 0..num_v_per_k {
        for g in 0..num_k_heads {
            let dst_head = j * num_k_heads + g;
            let src_head = g * num_v_per_k + j;
            for d in 0..head_dim {
                let dst_row = row_lo + dst_head * head_dim + d;
                let src_row = row_lo + src_head * head_dim + d;
                out[dst_row * in_f..dst_row * in_f + in_f]
                    .copy_from_slice(&data[src_row * in_f..src_row * in_f + in_f]);
            }
        }
    }
    out
}

/// Reorder the V-head COLUMNS of a row-major [out_rows][in_f=value_dim] matrix (out_proj, dim=1).
/// Column index = V-head*head_dim + d over the whole `value_dim` input axis.
fn reorder_cols_v(
    data: &[f32],
    out_rows: usize,
    in_f: usize,
    num_v_heads: usize,
    num_k_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let num_v_per_k = num_v_heads / num_k_heads;
    debug_assert_eq!(in_f, num_v_heads * head_dim);
    let mut out = data.to_vec();
    for r in 0..out_rows {
        let base = r * in_f;
        for j in 0..num_v_per_k {
            for g in 0..num_k_heads {
                let dst_head = j * num_k_heads + g;
                let src_head = g * num_v_per_k + j;
                for d in 0..head_dim {
                    out[base + dst_head * head_dim + d] = data[base + src_head * head_dim + d];
                }
            }
        }
    }
    out
}

fn f32_to_le(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

/// Resolve a requested ggml name to an HF target (+ optional transform), arch-aware.
/// Order: per-expert MoE name -> dense plain map -> qwen35 SSM map / norm `+1`.
pub fn resolve_ggml(ggml: &str, cfg: &ModelConfig) -> Option<HfTarget> {
    // 1. Per-expert MoE: blk.{il}.ffn_{gate,up,down}_exps.{e}.weight (gathered one expert at a time).
    if let Some(rest) = ggml.strip_prefix("blk.") {
        if let Some((il, suffix)) = rest.split_once('.') {
            for (tag, proj) in [
                ("ffn_gate_exps.", "gate"),
                ("ffn_up_exps.", "up"),
                ("ffn_down_exps.", "down"),
            ] {
                if let Some(e_part) = suffix.strip_prefix(tag) {
                    if let Some(e) = e_part.strip_suffix(".weight") {
                        if let Ok(eid) = e.parse::<u32>() {
                            if let Ok(ilid) = il.parse::<u32>() {
                                let n_trunk = cfg.n_layer - cfg.nextn_predict_layers;
                                if ilid < n_trunk {
                                    return Some(HfTarget::Plain(hf_expert_name(
                                        ilid, eid, proj, &cfg.arch,
                                    )));
                                }
                                // Qwen stores appended MTP blocks under a separate `mtp.layers`
                                // namespace. Hy3 appends its MTP block to `model.layers` instead.
                                if matches!(cfg.arch, Arch::Qwen35 | Arch::Qwen35Moe) {
                                    let mtp_il = ilid - n_trunk;
                                    let projection = match proj {
                                        "gate" => "gate_proj",
                                        "up" => "up_proj",
                                        "down" => "down_proj",
                                        _ => unreachable!(),
                                    };
                                    return Some(HfTarget::Plain(format!(
                                        "mtp.layers.{mtp_il}.mlp.experts.{eid}.{projection}.weight"
                                    )));
                                }
                                return Some(HfTarget::Plain(hf_expert_name(
                                    ilid, eid, proj, &cfg.arch,
                                )));
                            }
                        }
                    }
                }
            }
        }
    }

    // Hy3 appends its MTP block as model.layers.{n_trunk}, in the same namespace and tensor
    // layout as the trunk. Only the four glue tensors need special ggml names; ordinary attention,
    // FFN, router, shared-expert, and per-expert names continue through the standard Hy3 maps.
    if cfg.arch.is_hy3() && cfg.nextn_predict_layers > 0 {
        let n_trunk = cfg.n_layer - cfg.nextn_predict_layers;
        if let Some((il, suffix)) = ggml
            .strip_prefix("blk.")
            .and_then(|rest| rest.split_once('.'))
        {
            if il.parse::<u32>().ok().is_some_and(|il| il >= n_trunk) {
                let hf_suffix = match suffix {
                    "nextn.enorm.weight" => Some("enorm.weight"),
                    "nextn.hnorm.weight" => Some("hnorm.weight"),
                    "nextn.eh_proj.weight" => Some("eh_proj.weight"),
                    "nextn.shared_head_norm.weight" => Some("final_layernorm.weight"),
                    _ => None,
                };
                if let Some(hf_suffix) = hf_suffix {
                    return Some(HfTarget::Plain(format!("model.layers.{il}.{hf_suffix}")));
                }
            }
        }
    }

    // 2a. gemma-4 (Arch::Gemma4): dense hybrid, tied embeddings.
    //    NORM-FOLD LAW, MEASURED not assumed (the coordinator's checked-invariant directive,
    //    2026-08-16): a byte compare of the shipped GGUF norms against the raw HF safetensors
    //    (research/gemma-vision-20260816/normcmp receipts) proved gemma-4-31B stores its norm
    //    weights RAW and IDENTICAL in both formats (attn_norm 4.69≈4.66, q_norm 1.0234 exact,
    //    ffn_norm 3.03≈3.0 — deltas are pure Q4_0 rounding). The 31B does NOT use the (1+w)
    //    gemma-2/3 convention (weights are ~O(1..5), not ~0), so its RMSNorm is PLAIN and
    //    every norm is a plain passthrough rename. An earlier +1 fold here inflated every
    //    norm ~21% and saturated the logits at the softcap (wrong argmax). If a future gemma
    //    checkpoint reintroduces the (1+w) convention, the load-time parity gate against the
    //    GGUF twin catches it — the fold is verified per artifact, never assumed.
    if matches!(cfg.arch, Arch::Gemma4) {
        if ggml == "output_norm.weight" {
            return Some(HfTarget::Plain("model.norm.weight".into()));
        }
        // rope_freqs.weight is a GGUF-only synthesized tensor (freq factors: 1.0 for the
        // first partial_rotary_factor fraction of dim pairs, ~1e30 beyond — verified
        // against the shipped GGUF tensor bytes). The engine synthesizes it when the
        // source misses it; there is no HF tensor to map.
        if ggml == "rope_freqs.weight" {
            return None;
        }
        if let Some(rest) = ggml.strip_prefix("blk.") {
            if let Some((il, suffix)) = rest.split_once('.') {
                let plain = |hf_suffix: &str| {
                    Some(HfTarget::Plain(format!("model.layers.{il}.{hf_suffix}")))
                };
                return match suffix {
                    // norms — PLAIN passthrough (measured: raw, no +1 on this checkpoint)
                    "attn_norm.weight" => plain("input_layernorm.weight"),
                    "post_attention_norm.weight" => plain("post_attention_layernorm.weight"),
                    "ffn_norm.weight" => plain("pre_feedforward_layernorm.weight"),
                    "post_ffw_norm.weight" => plain("post_feedforward_layernorm.weight"),
                    "attn_q_norm.weight" => plain("self_attn.q_norm.weight"),
                    "attn_k_norm.weight" => plain("self_attn.k_norm.weight"),
                    // projections — plain renames (v_proj absent on the K=V globals in HF
                    // exactly as in GGUF; the loader's wv:=wk dedup handles the miss)
                    "attn_q.weight" => plain("self_attn.q_proj.weight"),
                    "attn_k.weight" => plain("self_attn.k_proj.weight"),
                    "attn_v.weight" => plain("self_attn.v_proj.weight"),
                    "attn_output.weight" => plain("self_attn.o_proj.weight"),
                    "ffn_gate.weight" => plain("mlp.gate_proj.weight"),
                    "ffn_up.weight" => plain("mlp.up_proj.weight"),
                    "ffn_down.weight" => plain("mlp.down_proj.weight"),
                    // per-layer output scalar: HF stores it WITHOUT a .weight suffix
                    "layer_output_scale.weight" => plain("layer_scalar"),
                    // NORM backstop (the fold-sensitive class): an unrecognized *norm.weight
                    // must fail loudly rather than fall through to a wrong/absent mapping —
                    // this is the checked invariant that guards the fold decision above.
                    other if other.ends_with("norm.weight") => panic!(
                        "gemma-4 norm blk.N.{other:?} has no native mapping — add it to the \
                         exhaustive norm arm and decide its fold from a GGUF-vs-HF compare"
                    ),
                    // Optional/absent tensors (MoE expert stacks + router on the dense 31B,
                    // any future extra): return None so the loader's presence probe reads
                    // 'absent' and takes the dense path. Not a norm, so no fold risk.
                    _ => None,
                };
            }
        }
        // token_embd falls through to the arch-independent top-level map (tied embeddings:
        // gemma-4 ships no lm_head; the loader's tied-output path reads token_embd).
    }

    // Step's appended NextN blocks stay under model.layers.{45,46,47}. Ordinary attention and
    // dense FFN tensors use the same layer names as the trunk; only the glue and owned draft head
    // need explicit ggml-name translations. Standard transformer norms take the Step +1 fold
    // below; this block only resolves the explicit NextN glue names.
    if cfg.arch.is_step35() && cfg.nextn_predict_layers > 0 {
        let n_trunk = cfg.n_layer - cfg.nextn_predict_layers;
        if let Some((il, suffix)) = ggml
            .strip_prefix("blk.")
            .and_then(|rest| rest.split_once('.'))
        {
            if il.parse::<u32>().ok().is_some_and(|il| il >= n_trunk) {
                let target = match suffix {
                    "nextn.enorm.weight" => {
                        Some(("enorm.weight", Some(TransformKind::NormPlusOne)))
                    }
                    "nextn.hnorm.weight" => {
                        Some(("hnorm.weight", Some(TransformKind::NormPlusOne)))
                    }
                    "nextn.eh_proj.weight" => Some(("eh_proj.weight", None)),
                    "nextn.shared_head_norm.weight" => Some((
                        "transformer.shared_head.norm.weight",
                        Some(TransformKind::NormPlusOne),
                    )),
                    "nextn.shared_head_head.weight" | "nextn.shared_head.weight" => {
                        Some(("transformer.shared_head.output.weight", None))
                    }
                    _ => None,
                };
                if let Some((hf_suffix, transform)) = target {
                    let hf = format!("model.layers.{il}.{hf_suffix}");
                    return Some(match transform {
                        Some(kind) => HfTarget::Transform { hf, kind },
                        None => HfTarget::Plain(hf),
                    });
                }
            }
        }
    }

    // Step3p7RMSNorm computes `normed * (weight + 1)` for input, post-attention, q/k, and
    // output norms. Fold the +1 at HF load so the engine's plain RMSNorm remains the one numeric
    // program used by eager, prime, verify, and PP stage walkers. This is independent of Qwen3.5's
    // identically-shaped convention and deliberately excludes Hy3's verbatim norm weights.
    if cfg.arch.is_step35() && is_plusone_norm(ggml) {
        if let Some(hf) = ggml_to_hf(ggml, &cfg.arch) {
            return Some(HfTarget::Transform {
                hf,
                kind: TransformKind::NormPlusOne,
            });
        }
    }

    // 2. qwen35 norm +1: every `*norm.weight` EXCEPT linear_attn (ssm_norm). The dense map below
    //    renames them; for hybrid we must additionally add +1 (qwen.py:302-303). Catch the dense
    //    norm names here so they take the Transform arm.
    // Dense-attention MoE families also reuse HybridModel, but the MTP/SSM maps and +1 norm
    // convention here are Qwen3.5-specific.
    // MiniMax applies its independently configured Gemma-norm fold in the next arm; Hy3 uses the
    // checkpoint norm weights verbatim.
    if matches!(cfg.arch, Arch::Qwen35 | Arch::Qwen35Moe) {
        // MTP/NextN block FIRST: blk.{n_trunk}.* maps into the HF `mtp.*` namespace (NVIDIA 27B /
        // qwen3.6 text ckpts), NOT model.layers.{n_trunk}.* (which does not exist — HF
        // num_hidden_layers excludes the MTP block). Must precede the SSM/dense arms.
        if let Some(t) = resolve_mtp_block(ggml, cfg) {
            return Some(t);
        }
        if let Some(t) = resolve_hybrid_ssm(ggml, cfg) {
            return Some(t);
        }
        // norm +1 on the dense-mapped norms (attn_norm / ffn_norm / q_norm / k_norm / output_norm).
        if is_plusone_norm(ggml) {
            if let Some(hf) = ggml_to_hf(ggml, &cfg.arch) {
                return Some(HfTarget::Transform {
                    hf,
                    kind: TransformKind::NormPlusOne,
                });
            }
        }
    }

    // 3. MiniMax-M3 gemma-norm: out = normed*(1+w) — fold the +1 into the weight at load so the
    //    engine's plain RMSNorm runs unchanged (exact; same fold the qwen35 converter applies).
    //    Applies to EVERY norm in the text model (input/post_attention layernorm, q/k_norm,
    //    model.norm — and index_q/k_norm when the MSA arc lands).
    if cfg.m3.as_ref().is_some_and(|m| m.use_gemma_norm) && is_plusone_norm(ggml) {
        if let Some(hf) = ggml_to_hf(ggml, &cfg.arch) {
            return Some(HfTarget::Transform {
                hf,
                kind: TransformKind::NormPlusOne,
            });
        }
    }

    // 3b. glm5_next MLA/DSA layers. Arch-scoped and placed AFTER the KDA map inside `ggml_to_hf`
    // (which the plain arm below reaches): the two mixer kinds share the `self_attn.` namespace on
    // this checkpoint, and only the ggml spelling distinguishes them.
    //
    // `attn_k_b`/`attn_v_b` are the CONVERSION SPLIT of the fused `kv_b_proj` — llama.cpp's GGUF
    // mint pre-splits them, HF ships only the fused weight, and `tensor_contract::add_mla_hf`
    // registers it with `TensorTransform::SplitMlaKv`. Doing the split here (rather than in the
    // engine's MLA loader) is what makes the operands dtype-agnostic: `deq_f32` handles BF16, F16,
    // F32, F8-E4M3 and modelopt NVFP4, and the 3-D output `ne` keeps the result on the F32 arm
    // (both the Q8_0 and the F8 re-encodes in `source.rs` gate on `ne.len() == 2`), which is the
    // only layout `mla_absorb_q`/`mla_decompress_v` can read. `attn_output` and the norms fall
    // through to the generic dense map below, which already spells them correctly.
    if cfg.arch.is_glm5_next() {
        if let Some((il, suffix)) = ggml
            .strip_prefix("blk.")
            .and_then(|rest| rest.split_once('.'))
        {
            let name = |s: &str| format!("model.layers.{il}.{s}");
            let plain: Option<&str> = match suffix {
                "attn_q_a.weight" => Some("self_attn.q_a_proj.weight"),
                "attn_q_b.weight" => Some("self_attn.q_b_proj.weight"),
                "attn_kv_a_mqa.weight" => Some("self_attn.kv_a_proj_with_mqa.weight"),
                "attn_q_a_norm.weight" => Some("self_attn.q_a_layernorm.weight"),
                "attn_kv_a_norm.weight" => Some("self_attn.kv_a_layernorm.weight"),
                // The fused source itself: censused and loadable, unused by the absorbed v1 core.
                "attn_kv_b.weight" => Some("self_attn.kv_b_proj.weight"),
                // DSA k-pool indexer. Every one of these is REQUIRED on a layer whose plan
                // declares `SparseIndexPlan::Own { kpool: Some(..) }`: a missing name makes
                // `MlaAttnLayer::load` refuse the layer by name rather than fall back to dense
                // attention, which above `index_topk` is a different function.
                "indexer.attn_q_b.weight" => Some("self_attn.indexer.wq_b.weight"),
                "indexer.attn_k.weight" => Some("self_attn.indexer.wk.weight"),
                "indexer.proj.weight" => Some("self_attn.indexer.weights_proj.weight"),
                "indexer.k_norm.weight" => Some("self_attn.indexer.k_norm.weight"),
                "indexer.k_norm.bias" => Some("self_attn.indexer.k_norm.bias"),
                "indexer.kpool_gate.weight" => Some("self_attn.indexer.index_kpool_compress_gate"),
                "indexer.kpool_ape.weight" => Some("self_attn.indexer.index_kpool_compress_ape"),
                _ => None,
            };
            if let Some(hf) = plain {
                return Some(HfTarget::Plain(name(hf)));
            }
            let split: Option<TransformKind> = match suffix {
                "attn_k_b.weight" => Some(TransformKind::MlaKeyUpSplit),
                "attn_v_b.weight" => Some(TransformKind::MlaValueUpSplit),
                _ => None,
            };
            if let Some(kind) = split {
                return Some(HfTarget::Transform {
                    hf: name("self_attn.kv_b_proj.weight"),
                    kind,
                });
            }
        }
    }

    // 4. Dense plain rename (zero-copy).
    ggml_to_hf(ggml, &cfg.arch).map(HfTarget::Plain)
}

/// qwen3.5/3.6 MTP (NextN) block map: the engine asks for `blk.{n_trunk}.*` (GGUF numbering, where
/// block_count includes the MTP block) but the HF checkpoint stores the head under a separate
/// `mtp.*` namespace (NVIDIA 27B + qwen3.6 text ckpts; HF num_hidden_layers EXCLUDES the block):
///   nextn.enorm  -> mtp.pre_fc_norm_embedding   (+1 fold, verified vs GGUF twin blk.64)
///   nextn.hnorm  -> mtp.pre_fc_norm_hidden      (+1)
///   nextn.eh_proj -> mtp.fc                     (plain, [2*n_embd, n_embd])
///   nextn.shared_head_norm -> mtp.norm          (+1)
///   attn/ffn/norm tensors -> mtp.layers.{il-n_trunk}.{hf dense names}  (norms +1, matrices plain)
/// `nextn.shared_head.weight` has no mtp.* equivalent (head reuses the model's lm_head) -> None.
fn resolve_mtp_block(ggml: &str, cfg: &ModelConfig) -> Option<HfTarget> {
    if cfg.nextn_predict_layers == 0 {
        return None;
    }
    let n_trunk = cfg.n_layer - cfg.nextn_predict_layers;
    let rest = ggml.strip_prefix("blk.")?;
    let (il, suffix) = rest.split_once('.')?;
    let il: u32 = il.parse().ok()?;
    if il < n_trunk {
        return None;
    }
    let mtp_il = il - n_trunk; // mtp.layers.{k} index (27B: always 0)
    // NextN glue tensors (top-level mtp.* names).
    let glue: Option<(&str, TransformKind)> = match suffix {
        "nextn.enorm.weight" => Some((
            "mtp.pre_fc_norm_embedding.weight",
            TransformKind::NormPlusOne,
        )),
        "nextn.hnorm.weight" => Some(("mtp.pre_fc_norm_hidden.weight", TransformKind::NormPlusOne)),
        "nextn.eh_proj.weight" => Some(("mtp.fc.weight", TransformKind::Identity)),
        "nextn.shared_head_norm.weight" => Some(("mtp.norm.weight", TransformKind::NormPlusOne)),
        _ => None,
    };
    if let Some((hf, kind)) = glue {
        return Some(match kind {
            TransformKind::Identity => HfTarget::Plain(hf.into()),
            k => HfTarget::Transform {
                hf: hf.into(),
                kind: k,
            },
        });
    }
    // Full transformer block tensors under mtp.layers.{k}.*
    let (hf_suffix, plus_one): (&str, bool) = match suffix {
        "attn_norm.weight" => ("input_layernorm.weight", true),
        "post_attention_norm.weight" | "ffn_norm.weight" => {
            ("post_attention_layernorm.weight", true)
        }
        "attn_q_norm.weight" => ("self_attn.q_norm.weight", true),
        "attn_k_norm.weight" => ("self_attn.k_norm.weight", true),
        "attn_q.weight" => ("self_attn.q_proj.weight", false),
        "attn_k.weight" => ("self_attn.k_proj.weight", false),
        "attn_v.weight" => ("self_attn.v_proj.weight", false),
        "attn_output.weight" => ("self_attn.o_proj.weight", false),
        "ffn_gate.weight" => ("mlp.gate_proj.weight", false),
        "ffn_up.weight" => ("mlp.up_proj.weight", false),
        "ffn_down.weight" => ("mlp.down_proj.weight", false),
        // MoE MTP block (qwen3.6-35B-A3B ST class): router + shared expert are plain 2D
        // tensors under mtp.layers.{k}.mlp.*; the fused stacked experts
        // (mlp.experts.{gate_up,down}_proj, 3D) are sliced source-side, not name-mapped.
        "ffn_gate_inp.weight" => ("mlp.gate.weight", false),
        "ffn_gate_shexp.weight" => ("mlp.shared_expert.gate_proj.weight", false),
        "ffn_up_shexp.weight" => ("mlp.shared_expert.up_proj.weight", false),
        "ffn_down_shexp.weight" => ("mlp.shared_expert.down_proj.weight", false),
        "ffn_gate_inp_shexp.weight" => ("mlp.shared_expert_gate.weight", false),
        _ => return None, // nextn.shared_head.weight etc: absent -> head reuses lm_head
    };
    let hf = format!("mtp.layers.{mtp_il}.{hf_suffix}");
    Some(if plus_one {
        HfTarget::Transform {
            hf,
            kind: TransformKind::NormPlusOne,
        }
    } else {
        HfTarget::Plain(hf)
    })
}

/// True for the ggml norm names that get qwen35 `+1` (all norms except ssm_norm/linear_attn.norm).
fn is_plusone_norm(ggml: &str) -> bool {
    match ggml {
        "output_norm.weight" => true,
        _ => {
            let Some(rest) = ggml.strip_prefix("blk.") else {
                return false;
            };
            let Some((_, suffix)) = rest.split_once('.') else {
                return false;
            };
            matches!(
                suffix,
                "attn_norm.weight"
                    | "ffn_norm.weight"
                    | "attn_q_norm.weight"
                    | "attn_k_norm.weight"
                    | "post_attention_norm.weight"
                    | "post_ffw_norm.weight"
            )
        }
    }
}

/// qwen35 linear-attn SSM tensor map (the `blk.{il}.{ssm/attn_qkv/attn_gate}` family). HF prefix is
/// `model.layers.{il}.linear_attn.` (the `model.language_model.` wrapper is handled by the source's
/// prefix-fallback). Cited against llama.cpp gguf-py/gguf/tensor_mapping.py + conversion/qwen.py.
fn resolve_hybrid_ssm(ggml: &str, _cfg: &ModelConfig) -> Option<HfTarget> {
    let rest = ggml.strip_prefix("blk.")?;
    let (il, suffix) = rest.split_once('.')?;
    let la = |t: &str| format!("model.layers.{il}.linear_attn.{t}");
    let (hf, kind) = match suffix {
        // matrices with a V-reorder transform (owned f32 buffer)
        "attn_qkv.weight" => (la("in_proj_qkv.weight"), TransformKind::QkvVReorderRows), // tensor_mapping.py:251
        "attn_gate.weight" => (la("in_proj_z.weight"), TransformKind::ZReorderRows),     // :386
        "ssm_beta.weight" => (la("in_proj_b.weight"), TransformKind::AbReorderRows),     // :909
        "ssm_alpha.weight" => (la("in_proj_a.weight"), TransformKind::AbReorderRows),    // :885
        "ssm_a" => (la("A_log"), TransformKind::NegExpReorderHeads),                     // :846
        "ssm_dt.bias" => (la("dt_bias"), TransformKind::ReorderHeads),                   // :831
        "ssm_conv1d.weight" => (la("conv1d.weight"), TransformKind::Conv1dSqueezeReorder), // :816
        "ssm_norm.weight" => (la("norm.weight"), TransformKind::Identity), // :871 (NO +1)
        "ssm_out.weight" => (la("out_proj.weight"), TransformKind::OutReorderCols), // :880
        _ => return None,
    };
    Some(HfTarget::Transform { hf, kind })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Arch;

    /// The glm5_next KDA name map may not drift from the tensor contract: for every KDA tensor
    /// the contract declares, the ggml spelling must resolve to exactly the HF spelling the
    /// contract names. Written AGAINST the contract rather than against a copied literal table,
    /// so a typo on either side fails here instead of at load time on the real safetensors
    /// artifact — the KDA GPU fixture serves ggml names and cannot see this surface at all.
    #[test]
    fn glm5_next_kda_names_match_the_contract() {
        use crate::model_plan::{
            ActivationPlan, AttentionPlan, DenseMlpPlan, DraftSourcePlan, KimiDeltaNetPlan,
            LayerPlan, MlpPlan, ModelPlan, NormKind, NormPlan, ResidualTopology, StatePlan,
            WeightTransform,
        };
        use crate::tensor_contract::{
            CheckpointDialect, ContractOptions, OutputHead, TensorContract, TensorId,
        };

        let norm = NormPlan {
            kind: NormKind::Rms,
            epsilon: 1e-5,
            weight_transform: WeightTransform::Identity,
        };
        let plan = ModelPlan {
            arch: Arch::Glm5Next,
            hidden_size: 256,
            vocab_size: 32,
            context_length: 512,
            embedding_scale: 1.0,
            vision: None,
            multimodal: None,
            layers: vec![LayerPlan {
                index: 0,
                pre_attention_norm: norm,
                attention: AttentionPlan::KimiDeltaNet(KimiDeltaNetPlan {
                    num_heads: 2,
                    head_dim: 128,
                    conv_kernel: 4,
                    gate_lower_bound: -5.0,
                }),
                pre_mlp_norm: norm,
                mlp: MlpPlan::Dense(DenseMlpPlan {
                    intermediate_size: 32,
                    activation: ActivationPlan::Silu,
                }),
                // HyperConnections, not Serial: with a serial residual the contract emits no
                // hc_* rows, so this pin could not see that the ggml->HF map was missing all
                // six of them. It did not, and the gap reached the first real-artifact load.
                residual: ResidualTopology::HyperConnections {
                    streams: 4,
                    epsilon: 1e-6,
                    sinkhorn_iterations: 20,
                    collapse: crate::model_plan::HcCollapse::Mean,
                },
                state: StatePlan::Recurrent {
                    conv_width: 768,
                    conv_kernel: 4,
                    state_width: 32768,
                },
            }],
            output_norm: norm,
            logits: Vec::new(),
            mtp_blocks: Vec::new(),
            drafter: None,
            draft_source: DraftSourcePlan::Embedded,
            sampling_defaults: None,
            partition_boundaries: Vec::new(),
        };
        let gguf = TensorContract::for_plan(
            &plan,
            CheckpointDialect::Gguf,
            ContractOptions {
                output_head: OutputHead::TiedToEmbedding,
            },
        )
        .expect("gguf contract");
        let hf = TensorContract::for_plan(
            &plan,
            CheckpointDialect::HfSafetensors,
            ContractOptions {
                output_head: OutputHead::TiedToEmbedding,
            },
        )
        .expect("hf contract");

        let mut checked = 0;
        for g in &gguf.requirements {
            // Quant-scale sidecars are a dialect-specific surface (the ggml side names them,
            // the HF side carries them inside the checkpoint's own quant metadata) and are not
            // part of the ggml->HF rename map.
            if matches!(g.id, TensorId::QuantAux { .. }) {
                continue;
            }
            let Some(gname) = g.names.first() else {
                continue;
            };
            // hc_* as well as kda_*: the mHC parameters ride the same arch-scoped map, and
            // a missing name there reads as an ABSENT tensor. Excluding them is how all six
            // went unmapped until the first real-artifact load refused on blk.0.hc_attn_fn.
            if !gname.starts_with("blk.0.kda") && !gname.starts_with("blk.0.hc_") {
                continue;
            }
            let want = hf
                .requirements
                .iter()
                .find(|h| h.id == g.id)
                .and_then(|h| h.names.first())
                .unwrap_or_else(|| panic!("HF contract declares no {:?}", g.id));
            let got = ggml_to_hf(gname, &Arch::Glm5Next)
                .unwrap_or_else(|| panic!("no glm5_next HF mapping for {gname}"));
            assert_eq!(&got, want, "{gname} maps to the wrong HF name");
            checked += 1;
        }
        assert_eq!(
            checked, 21,
            "expected all 15 contract KDA + 6 mHC tensors to be name-gated, saw {checked}"
        );
    }

    /// A shape-faithful miniature glm5_next config with BOTH mixer kinds: layer 0 KDA, layer 1
    /// MLA/DSA. Parsed through the real `HfConfig`/`ModelConfig` path, so the geometry the MLA
    /// split reads (`qk_nope_head_dim`, `v_head_dim`) is the parser's, not a hand-built struct.
    /// The MLA dims are deliberately tiny and PAIRWISE DISTINCT (nope 2, v 1, rank 3, heads 2) —
    /// a split that confuses two axes cannot pass by coincidence the way equal dims would let it.
    fn mla_mini_config() -> ModelConfig {
        glm5_mini_config(r#"["dense", "dense"]"#, 2)
    }

    /// The same miniature, with a MoE layer: `mlp_layer_types` and `first_k_dense_replace` are
    /// the only knobs, so the router, its selection bias, the expert banks and the shared expert
    /// all appear in the compiled contract.
    fn glm5_mini_config(mlp_layer_types: &str, first_k_dense_replace: u32) -> ModelConfig {
        let json = mla_mini_config_json()
            .replace(
                r#""mlp_layer_types": ["dense", "dense"]"#,
                &format!(r#""mlp_layer_types": {mlp_layer_types}"#),
            )
            .replace(
                r#""first_k_dense_replace": 2"#,
                &format!(r#""first_k_dense_replace": {first_k_dense_replace}"#),
            );
        ModelConfig::from_hf(&crate::config::HfConfig::parse(&json))
    }

    fn mla_mini_config_json() -> &'static str {
        r#"{
          "model_type": "glm5_next_text",
          "num_hidden_layers": 2,
          "num_nextn_predict_layers": 0,
          "hidden_size": 128,
          "intermediate_size": 64,
          "vocab_size": 32,
          "max_position_embeddings": 512,
          "rms_norm_eps": 1e-05,
          "hidden_act": "silu",
          "swiglu_limit": 1e30,
          "tie_word_embeddings": true,
          "hc_mult": 4,
          "hc_eps": 1e-06,
          "hc_sinkhorn_iters": 20,
          "mhc": true,
          "layer_types": ["linear_attention", "deepseek_sparse_attention"],
          "mlp_layer_types": ["dense", "dense"],
          "first_k_dense_replace": 2,
          "indexer_types": ["full", "full"],
          "linear_attn_config": {
            "num_heads": 1,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "kda_layers": [0],
            "full_attn_layers": [1]
          },
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "q_lora_rank": 4,
          "kv_lora_rank": 3,
          "qk_head_dim": 2,
          "qk_nope_head_dim": 2,
          "qk_rope_head_dim": 0,
          "v_head_dim": 1,
          "mla_use_nope": true,
          "index_n_heads": 1,
          "index_head_dim": 8,
          "index_topk": 8,
          "index_kpool": 4,
          "index_kpool_always_select_tail": true,
          "index_kpool_compress": true,
          "indexer_rope_interleave": true,
          "index_share_for_mtp_iteration": true,
          "n_routed_experts": 4,
          "num_experts_per_tok": 2,
          "moe_intermediate_size": 32,
          "n_shared_experts": 1,
          "scoring_func": "sigmoid",
          "topk_method": "noaux_tc",
          "routed_scaling_factor": 2.5,
          "norm_topk_prob": true,
          "n_group": 1,
          "topk_group": 1,
          "head_dim": 0,
          "attention_bias": false,
          "moe_router_dtype": "float32",
          "dtype": "bfloat16"
        }"#
    }

    /// COMPLETENESS, not a sample: EVERY tensor the compiled glm5_next contract declares must
    /// resolve through `resolve_ggml` — the function the ENGINE's loader calls — onto a name the
    /// HF dialect of the SAME contract declares for the SAME `TensorId`.
    ///
    /// The property no other gate in this lane covers. `TensorCensus` proves the CONTRACT's HF
    /// names bind the artifact; the reference executor loads through the contract too, so both
    /// stay green while the engine's separate ggml->HF resolver is missing a row. Every fixture
    /// gate (`glm5_routed_router_gpu`, `glm5_moe_residency_gpu`, `swiglu_preclamp_gpu`,
    /// `kda_fixture_gpu`, `mla_gpu_forward`) serves its own synthetic tensors under names the
    /// test itself chose, so none of them touches name resolution at all. Three name gaps reached
    /// a real-artifact load in this lane before this pin existed: the six mHC parameters, the MLA
    /// family, and `exp_probs_b.bias`.
    ///
    /// Written against the contract, with NO per-name allowlist. The predecessor pin
    /// (`glm5_next_kda_names_match_the_contract`) filtered to `blk.0.kda*`/`blk.0.hc_*` and built
    /// a DENSE-MLP plan, so it was blind to the router bias twice over — an allowlist is exactly
    /// how a missing row stays missing (LAW: exception lists need expiry). The two structural
    /// exceptions below are asserted as facts about the contract, not skipped.
    #[test]
    fn glm5_next_every_contract_tensor_resolves_through_the_engine_map() {
        use crate::tensor_contract::{
            CheckpointDialect, ContractOptions, LayerTensor, OutputHead, TensorContract, TensorId,
        };

        // Layer 0 KDA + dense MLP, layer 1 MLA/DSA + MoE: both mixer kinds and both MLP kinds,
        // so the router, its selection bias, the expert banks and the shared expert are all in
        // the compiled contract.
        let cfg = glm5_mini_config(r#"["dense", "sparse"]"#, 1);
        let plan = crate::model_packs::for_config(&cfg)
            .expect("glm5_next model pack matches the mini config")
            .compile_plan(&cfg)
            .expect("mini glm5_next plan compiles");
        assert!(
            plan.layers.iter().any(|l| matches!(
                l.mlp,
                crate::model_plan::MlpPlan::Moe(ref m)
                    if matches!(m.router, crate::model_plan::RouterPlan::Sigmoid { selection_bias: true, .. })
            )),
            "the fixture must compile a MoE layer with a selection-bias router, or this pin \
             cannot see the router bias at all"
        );
        let opts = ContractOptions {
            output_head: OutputHead::TiedToEmbedding,
        };
        let gguf =
            TensorContract::for_plan(&plan, CheckpointDialect::Gguf, opts).expect("gguf contract");
        let hf = TensorContract::for_plan(&plan, CheckpointDialect::HfSafetensors, opts)
            .expect("hf contract");

        // The two structural exceptions, asserted rather than assumed.
        //
        // 1. The MLA conversion split: HF ships only the fused `kv_b_proj`, so the HF contract
        //    carries no MlaKeyUp/MlaValueUp requirement and both ggml halves must Transform onto
        //    the fused source. `glm5_next_mla_names_match_the_contract` pins that in detail.
        // 2. Expert banks: ggml stacks the experts into one tensor, HF stores them one per
        //    expert, and the engine asks per expert (`resolve_ggml` step 1). The stacked ggml
        //    spelling is therefore never resolved; the per-expert spelling is what must land.
        let split_ids: Vec<TensorId> = plan
            .layers
            .iter()
            .flat_map(|l| {
                [LayerTensor::MlaKeyUp, LayerTensor::MlaValueUp].map(|tensor| TensorId::Layer {
                    index: l.index,
                    tensor,
                })
            })
            .collect();
        for id in &split_ids {
            assert!(
                !hf.requirements.iter().any(|h| &h.id == id),
                "{id:?} must NOT be an HF-dialect requirement — HF ships only the fused kv_b_proj"
            );
        }

        let expert_bank = |tensor: LayerTensor| -> Option<&'static str> {
            match tensor {
                LayerTensor::MoeExpertGateBank => Some("gate"),
                LayerTensor::MoeExpertUpBank => Some("up"),
                LayerTensor::MoeExpertDownBank => Some("down"),
                _ => None,
            }
        };

        let mut checked = 0usize;
        for requirement in &gguf.requirements {
            // Quant-scale sidecars are a dialect-specific surface: the ggml side names them,
            // the HF side carries them inside the checkpoint's own quant metadata.
            if matches!(requirement.id, TensorId::QuantAux { .. }) {
                continue;
            }
            if split_ids.contains(&requirement.id) {
                // Covered by exception 1 above and by the MLA pin.
                continue;
            }
            let ggml = requirement
                .names
                .first()
                .unwrap_or_else(|| {
                    panic!("GGUF contract requirement {:?} has no name", requirement.id)
                })
                .clone();
            let want = hf
                .requirements
                .iter()
                .find(|h| h.id == requirement.id)
                .unwrap_or_else(|| {
                    panic!(
                        "HF contract declares no {:?} (GGUF names it {ggml})",
                        requirement.id
                    )
                });

            // Exception 2: ask the per-expert spelling the engine actually asks for.
            let ask = match requirement.id {
                TensorId::Layer { index, tensor } if expert_bank(tensor).is_some() => {
                    format!(
                        "blk.{index}.ffn_{}_exps.0.weight",
                        expert_bank(tensor).unwrap()
                    )
                }
                _ => ggml.clone(),
            };

            let got = match resolve_ggml(&ask, &cfg) {
                Some(HfTarget::Plain(hf)) | Some(HfTarget::Transform { hf, .. }) => hf,
                None => panic!(
                    "no glm5_next ggml->HF mapping for {ask} ({:?}). The engine's loader resolves \
                     names through resolve_ggml; an unmapped name reads as an ABSENT tensor, and \
                     absent is silently substituted for at several load sites. The HF contract \
                     declares this tensor as {:?}",
                    requirement.id, want.names
                ),
            };
            assert!(
                want.names.contains(&got),
                "{ask} ({:?}) resolves to {got}, which is not one of the HF contract's names \
                 {:?} for the same tensor",
                requirement.id,
                want.names
            );
            checked += 1;
        }

        assert_eq!(
            checked, 58,
            "the fixture's compiled contract changed size. Raise this number when the plan grows \
             a tensor; NEVER lower it to make a red pin green — a shrinking count is the surface \
             this gate exists to cover going unwatched"
        );
    }

    /// The glm5_next MLA name map may not drift from the tensor contract — the same law
    /// `glm5_next_kda_names_match_the_contract` enforces for the other mixer, and the reason the
    /// MLA layers of an HF checkpoint could not be resolved AT ALL before this lane: the engine
    /// asks for ggml spellings and `hf_mapping` carried no MLA entry for any family.
    ///
    /// `attn_k_b`/`attn_v_b` are the exception the assertion names explicitly: the HF dialect
    /// declares only the FUSED `MlaKvSource`, so both must resolve to that one HF tensor through
    /// a Transform, never to an invented `k_b_proj`/`v_b_proj` name that no checkpoint ships.
    #[test]
    fn glm5_next_mla_names_match_the_contract() {
        use crate::tensor_contract::{
            CheckpointDialect, ContractOptions, LayerTensor, OutputHead, TensorContract, TensorId,
        };

        let cfg = mla_mini_config();
        let plan = crate::model_packs::for_config(&cfg)
            .expect("glm5_next model pack matches the mini config")
            .compile_plan(&cfg)
            .expect("mini glm5_next plan compiles");
        let opts = ContractOptions {
            output_head: OutputHead::TiedToEmbedding,
        };
        let gguf =
            TensorContract::for_plan(&plan, CheckpointDialect::Gguf, opts).expect("gguf contract");
        let hf = TensorContract::for_plan(&plan, CheckpointDialect::HfSafetensors, opts)
            .expect("hf contract");

        let hf_name = |tensor: LayerTensor| -> String {
            let id = TensorId::Layer { index: 1, tensor };
            hf.requirements
                .iter()
                .find(|h| h.id == id)
                .and_then(|h| h.names.first())
                .unwrap_or_else(|| panic!("HF contract declares no {id:?}"))
                .clone()
        };
        let ggml_name = |tensor: LayerTensor| -> String {
            let id = TensorId::Layer { index: 1, tensor };
            gguf.requirements
                .iter()
                .find(|g| g.id == id)
                .and_then(|g| g.names.first())
                .unwrap_or_else(|| panic!("GGUF contract declares no {id:?}"))
                .clone()
        };

        // Plain renames: the ggml spelling must land on the HF contract's own name for the SAME
        // TensorId, so neither side can be edited alone.
        for tensor in [
            LayerTensor::MlaQueryDown,
            LayerTensor::MlaQueryUp,
            LayerTensor::MlaKvDown,
            LayerTensor::MlaOutput,
            LayerTensor::MlaQueryDownNorm,
            LayerTensor::MlaKvDownNorm,
            LayerTensor::MlaKvSource,
        ] {
            let g = ggml_name(tensor);
            let want = hf_name(tensor);
            match resolve_ggml(&g, &cfg) {
                Some(HfTarget::Plain(got)) => {
                    assert_eq!(got, want, "{g} maps to the wrong HF name")
                }
                other => panic!(
                    "{g} must be a plain rename to {want}, got {}",
                    match other {
                        Some(HfTarget::Transform { hf, .. }) => format!("a transform on {hf}"),
                        Some(HfTarget::Plain(p)) => p,
                        None => "no mapping".into(),
                    }
                ),
            }
        }

        // The conversion split: the HF dialect has NO MlaKeyUp/MlaValueUp requirement — both
        // halves come out of the fused MlaKvSource tensor.
        for tensor in [LayerTensor::MlaKeyUp, LayerTensor::MlaValueUp] {
            let id = TensorId::Layer { index: 1, tensor };
            assert!(
                !hf.requirements.iter().any(|h| h.id == id),
                "{id:?} must NOT be an HF-dialect requirement — HF ships only the fused kv_b_proj"
            );
        }
        let fused = hf_name(LayerTensor::MlaKvSource);
        for (tensor, want_kind) in [
            (LayerTensor::MlaKeyUp, "key"),
            (LayerTensor::MlaValueUp, "value"),
        ] {
            let g = ggml_name(tensor);
            match resolve_ggml(&g, &cfg) {
                Some(HfTarget::Transform { hf, kind }) => {
                    assert_eq!(hf, fused, "{g} must split the fused kv_b_proj");
                    let is_key = matches!(kind, TransformKind::MlaKeyUpSplit);
                    assert_eq!(
                        is_key,
                        want_kind == "key",
                        "{g} took the wrong half of the split"
                    );
                }
                _ => panic!("{g} must resolve to a SplitMlaKv transform on {fused}"),
            }
        }
    }

    /// Executed pin of the conversion-split convention. Element counts survive every wrong
    /// convention (transposed key, swapped halves, head-major vs rank-major), so only a
    /// hand-derived table can catch one. Mirrors `micro_gguf::write_glm_dsa_micro`, which
    /// generates `attn_kv_b` as the source of truth and derives k_b/v_b from it — and
    /// `memra_reference`'s `self_test_split_mla_kv`, which pins the HF `expand_kv` lineage.
    ///
    /// Encoding: fused element (head, row, rank) = head*100 + row*10 + rank.
    #[test]
    fn mla_kv_split_matches_the_micro_fixture_convention() {
        let (heads, nope, v, rank) = (2usize, 2usize, 1usize, 3usize);
        let mut fused = Vec::new();
        for h in 0..heads {
            for row in 0..nope + v {
                for r in 0..rank {
                    fused.push((h * 100 + row * 10 + r) as f32);
                }
            }
        }
        // k_b ne [nope, rank, head] -> row-major [head][rank][nope]: the TRANSPOSED nope slice.
        let key = split_mla_kv_plane(&fused, true, nope, v, rank);
        assert_eq!(
            key,
            vec![
                0.0, 10.0, 1.0, 11.0, 2.0, 12.0, // head 0: rank-major over the 2 nope rows
                100.0, 110.0, 101.0, 111.0, 102.0, 112.0, // head 1
            ],
            "attn_k_b must be the per-head nope slice TRANSPOSED to [head][rank][nope]"
        );
        // v_b ne [rank, v, head] -> row-major [head][v][rank]: already the decompress order.
        let value = split_mla_kv_plane(&fused, false, nope, v, rank);
        assert_eq!(
            value,
            vec![20.0, 21.0, 22.0, 120.0, 121.0, 122.0],
            "attn_v_b must be the per-head value slice in [head][v][rank] order, untransposed"
        );

        // And through the real transform entry point, with the real parsed geometry: the emitted
        // `ne` is what keeps these operands off every quantized arm. Both the Q8_0 re-encode and
        // the F8 re-encode in `source.rs` gate on `ne.len() == 2`, and the engine's loader refuses
        // a quantized tensor whose ne is not 2-D, so a 3-D `ne` here is load-bearing, not cosmetic.
        let cfg = mla_mini_config();
        let ne_in = vec![rank as u64, (heads * (nope + v)) as u64];
        let mut data = fused.clone();
        let (ne_k, bytes_k) = TransformKind::MlaKeyUpSplit.apply(&mut data, ne_in.clone(), &cfg);
        assert_eq!(ne_k, vec![nope as u64, rank as u64, heads as u64]);
        assert_eq!(bytes_k, f32_to_le(&key));
        let mut data = fused.clone();
        let (ne_v, bytes_v) = TransformKind::MlaValueUpSplit.apply(&mut data, ne_in, &cfg);
        assert_eq!(ne_v, vec![rank as u64, v as u64, heads as u64]);
        assert_eq!(bytes_v, f32_to_le(&value));
    }

    /// The split has NO packed fast path, and must not acquire one by accident. `apply_nvfp4` and
    /// `apply_fp8_blk` exist for pure 2-D row/column permutations that keep a weight quantized;
    /// this transform changes rank and feeds f32-only kernels, so both must refuse and let the
    /// f32 (`deq_f32`) route run. That refusal is what makes the operands dtype-agnostic: a
    /// modelopt NVFP4 or block-128 FP8 `kv_b_proj` dequantizes here instead of staying packed.
    #[test]
    fn mla_kv_split_has_no_packed_fast_path() {
        let cfg = mla_mini_config();
        for kind in [TransformKind::MlaKeyUpSplit, TransformKind::MlaValueUpSplit] {
            assert!(
                kind.apply_nvfp4(&[0u8; 36], 1, 64, &cfg).is_none(),
                "the MLA kv_b split must not claim an NVFP4 packed path"
            );
            assert!(
                kind.apply_fp8_blk(&[0u8; 64], 1, 64, &[1.0], 1, &cfg)
                    .is_none(),
                "the MLA kv_b split must not claim a block-128 FP8 packed path"
            );
        }
    }

    #[test]
    fn top_level_names() {
        let a = Arch::Qwen3;
        assert_eq!(
            ggml_to_hf("token_embd.weight", &a).unwrap(),
            "model.embed_tokens.weight"
        );
        assert_eq!(
            ggml_to_hf("output_norm.weight", &a).unwrap(),
            "model.norm.weight"
        );
        assert_eq!(ggml_to_hf("output.weight", &a).unwrap(), "lm_head.weight");
    }

    #[test]
    fn per_layer_dense() {
        let a = Arch::Qwen3;
        assert_eq!(
            ggml_to_hf("blk.0.attn_q.weight", &a).unwrap(),
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            ggml_to_hf("blk.7.attn_k.weight", &a).unwrap(),
            "model.layers.7.self_attn.k_proj.weight"
        );
        assert_eq!(
            ggml_to_hf("blk.3.attn_output.weight", &a).unwrap(),
            "model.layers.3.self_attn.o_proj.weight"
        );
        assert_eq!(
            ggml_to_hf("blk.0.attn_norm.weight", &a).unwrap(),
            "model.layers.0.input_layernorm.weight"
        );
        assert_eq!(
            ggml_to_hf("blk.0.ffn_norm.weight", &a).unwrap(),
            "model.layers.0.post_attention_layernorm.weight"
        );
        assert_eq!(
            ggml_to_hf("blk.0.ffn_gate.weight", &a).unwrap(),
            "model.layers.0.mlp.gate_proj.weight"
        );
        assert_eq!(
            ggml_to_hf("blk.0.ffn_up.weight", &a).unwrap(),
            "model.layers.0.mlp.up_proj.weight"
        );
        assert_eq!(
            ggml_to_hf("blk.0.ffn_down.weight", &a).unwrap(),
            "model.layers.0.mlp.down_proj.weight"
        );
        assert_eq!(
            ggml_to_hf("blk.5.attn_q_norm.weight", &a).unwrap(),
            "model.layers.5.self_attn.q_norm.weight"
        );
    }

    #[test]
    fn unmapped_returns_none() {
        let a = Arch::Qwen35; // SSM tensors are NOT in the plain dense map (they go through resolve_ggml)
        assert!(ggml_to_hf("blk.0.ssm_a", &a).is_none());
        assert!(ggml_to_hf("blk.0.attn_qkv.weight", &a).is_none());
        assert!(ggml_to_hf("totally.unknown", &a).is_none());
    }

    #[test]
    fn expert_names() {
        assert_eq!(
            hf_expert_name(2, 13, "gate", &Arch::Qwen3Moe),
            "model.layers.2.mlp.experts.13.gate_proj.weight"
        );
        assert_eq!(
            hf_expert_name(0, 0, "down", &Arch::Qwen3Moe),
            "model.layers.0.mlp.experts.0.down_proj.weight"
        );
        // MiniMax-M3 Mixtral-style layout: w1=gate, w2=down, w3=up.
        assert_eq!(
            hf_expert_name(5, 63, "gate", &Arch::MinimaxM3),
            "model.layers.5.block_sparse_moe.experts.63.w1.weight"
        );
        assert_eq!(
            hf_expert_name(5, 63, "down", &Arch::MinimaxM3),
            "model.layers.5.block_sparse_moe.experts.63.w2.weight"
        );
        assert_eq!(
            hf_expert_name(5, 63, "up", &Arch::MinimaxM3),
            "model.layers.5.block_sparse_moe.experts.63.w3.weight"
        );
    }

    #[test]
    fn hy3_moe_names() {
        let a = Arch::Hy3;
        assert_eq!(
            ggml_to_hf("blk.0.ffn_gate.weight", &a).unwrap(),
            "model.layers.0.mlp.gate_proj.weight"
        );
        assert_eq!(
            ggml_to_hf("blk.1.ffn_gate_inp.weight", &a).unwrap(),
            "model.layers.1.mlp.router.gate.weight"
        );
        assert_eq!(
            ggml_to_hf("blk.1.exp_probs_b.bias", &a).unwrap(),
            "model.layers.1.mlp.expert_bias"
        );
        assert_eq!(
            ggml_to_hf("blk.1.ffn_gate_shexp.weight", &a).unwrap(),
            "model.layers.1.mlp.shared_mlp.gate_proj.weight"
        );
        assert_eq!(
            ggml_to_hf("blk.1.ffn_up_shexp.weight", &a).unwrap(),
            "model.layers.1.mlp.shared_mlp.up_proj.weight"
        );
        assert_eq!(
            ggml_to_hf("blk.1.ffn_down_shexp.weight", &a).unwrap(),
            "model.layers.1.mlp.shared_mlp.down_proj.weight"
        );
        assert_eq!(
            ggml_to_hf("blk.1.ffn_gate_exps.weight", &a).unwrap(),
            "model.layers.1.mlp.switch_mlp.gate_proj.weight"
        );
        assert_eq!(
            ggml_to_hf("blk.1.ffn_up_exps.weight", &a).unwrap(),
            "model.layers.1.mlp.switch_mlp.up_proj.weight"
        );
        assert_eq!(
            ggml_to_hf("blk.1.ffn_down_exps.weight", &a).unwrap(),
            "model.layers.1.mlp.switch_mlp.down_proj.weight"
        );
    }

    #[test]
    fn hy3_appended_mtp_names_use_layer_namespace() {
        let json = r#"{
            "model_type":"hy_v3",
            "num_hidden_layers":80,
            "num_nextn_predict_layers":1,
            "hidden_size":256,
            "num_attention_heads":8,
            "num_key_value_heads":2,
            "intermediate_size":512,
            "vocab_size":1024,
            "max_position_embeddings":2048
        }"#;
        let cfg = ModelConfig::from_hf(&crate::config::HfConfig::parse(json));
        assert_eq!(cfg.n_layer, 81);
        for (ggml, expected) in [
            ("blk.80.nextn.enorm.weight", "model.layers.80.enorm.weight"),
            ("blk.80.nextn.hnorm.weight", "model.layers.80.hnorm.weight"),
            (
                "blk.80.nextn.eh_proj.weight",
                "model.layers.80.eh_proj.weight",
            ),
            (
                "blk.80.nextn.shared_head_norm.weight",
                "model.layers.80.final_layernorm.weight",
            ),
            (
                "blk.80.ffn_gate_exps.7.weight",
                "model.layers.80.mlp.experts.7.gate_proj.weight",
            ),
            (
                "blk.80.attn_q.weight",
                "model.layers.80.self_attn.q_proj.weight",
            ),
        ] {
            match resolve_ggml(ggml, &cfg) {
                Some(HfTarget::Plain(hf)) => assert_eq!(hf, expected),
                _ => panic!("Hy3 MTP tensor {ggml} did not resolve as a plain layer tensor"),
            }
        }
    }

    fn step_mapping_config() -> ModelConfig {
        let json = r#"{
          "model_type":"step3p5","num_hidden_layers":3,"num_nextn_predict_layers":1,
          "hidden_size":16,"intermediate_size":32,"num_attention_heads":2,
          "num_attention_groups":1,"head_dim":8,"vocab_size":64,
          "max_position_embeddings":2048,"moe_num_experts":6,"moe_top_k":2,
          "moe_intermediate_size":12,"share_expert_dim":12,"moe_layers_enum":"1,2",
          "moe_router_activation":"sigmoid","layer_types":["full_attention",
          "sliding_attention","sliding_attention","sliding_attention"],
          "rope_theta":[5000000,10000,10000,10000],
          "partial_rotary_factors":[0.5,1,1,1],"sliding_window":512,
          "attention_other_setting":{"num_attention_heads":4,"num_attention_groups":1},
          "swiglu_limits":[0,0,0,0],"swiglu_limits_shared":[0,0,0,0]
        }"#;
        ModelConfig::from_hf(&crate::config::HfConfig::parse(json))
    }

    #[test]
    fn step37_stacked_moe_gate_and_appended_mtp_names() {
        let cfg = step_mapping_config();
        for (ggml, expected) in [
            (
                "blk.1.attn_gate.weight",
                "model.layers.1.self_attn.g_proj.weight",
            ),
            (
                "blk.1.ffn_gate_inp.weight",
                "model.layers.1.moe.gate.weight",
            ),
            ("blk.1.exp_probs_b.bias", "model.layers.1.moe.router_bias"),
            (
                "blk.1.ffn_gate_shexp.weight",
                "model.layers.1.share_expert.gate_proj.weight",
            ),
            (
                "blk.1.ffn_gate_exps.weight",
                "model.layers.1.moe.gate_proj.weight",
            ),
            (
                "blk.1.ffn_up_exps.weight",
                "model.layers.1.moe.up_proj.weight",
            ),
            (
                "blk.1.ffn_down_exps.weight",
                "model.layers.1.moe.down_proj.weight",
            ),
            (
                "blk.3.nextn.eh_proj.weight",
                "model.layers.3.eh_proj.weight",
            ),
            (
                "blk.3.nextn.shared_head_head.weight",
                "model.layers.3.transformer.shared_head.output.weight",
            ),
            (
                "blk.3.attn_q.weight",
                "model.layers.3.self_attn.q_proj.weight",
            ),
            (
                "blk.3.ffn_gate.weight",
                "model.layers.3.mlp.gate_proj.weight",
            ),
        ] {
            match resolve_ggml(ggml, &cfg) {
                Some(HfTarget::Plain(hf)) => assert_eq!(hf, expected, "{ggml}"),
                _ => panic!("Step tensor {ggml} did not resolve as a plain HF tensor"),
            }
        }
        for (ggml, expected) in [
            ("blk.3.nextn.enorm.weight", "model.layers.3.enorm.weight"),
            ("blk.3.nextn.hnorm.weight", "model.layers.3.hnorm.weight"),
            (
                "blk.3.nextn.shared_head_norm.weight",
                "model.layers.3.transformer.shared_head.norm.weight",
            ),
        ] {
            match resolve_ggml(ggml, &cfg) {
                Some(HfTarget::Transform {
                    hf,
                    kind: TransformKind::NormPlusOne,
                }) => assert_eq!(hf, expected, "{ggml}"),
                _ => panic!("Step MTP norm {ggml} lost its required +1 transform"),
            }
        }
    }

    #[test]
    fn qwen35_moe_appended_mtp_names_use_mtp_namespace() {
        let json = r#"{
            "model_type":"qwen3_5_moe",
            "num_hidden_layers":2,
            "num_nextn_predict_layers":1,
            "hidden_size":256,
            "num_attention_heads":8,
            "num_key_value_heads":2,
            "intermediate_size":512,
            "vocab_size":1024,
            "max_position_embeddings":2048
        }"#;
        let cfg = ModelConfig::from_hf(&crate::config::HfConfig::parse(json));
        assert_eq!(cfg.n_layer, 3);
        for (ggml, expected) in [
            (
                "blk.2.ffn_gate_exps.7.weight",
                "mtp.layers.0.mlp.experts.7.gate_proj.weight",
            ),
            ("blk.2.ffn_gate_inp.weight", "mtp.layers.0.mlp.gate.weight"),
            (
                "blk.2.ffn_gate_shexp.weight",
                "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
            ),
            (
                "blk.2.ffn_gate_inp_shexp.weight",
                "mtp.layers.0.mlp.shared_expert_gate.weight",
            ),
            (
                "blk.2.attn_q.weight",
                "mtp.layers.0.self_attn.q_proj.weight",
            ),
        ] {
            match resolve_ggml(ggml, &cfg) {
                Some(HfTarget::Plain(hf)) => assert_eq!(hf, expected),
                _ => panic!("Qwen3.5 MTP tensor {ggml} did not resolve in the mtp namespace"),
            }
        }
        match resolve_ggml("blk.2.attn_norm.weight", &cfg) {
            Some(HfTarget::Transform {
                hf,
                kind: TransformKind::NormPlusOne,
            }) => assert_eq!(hf, "mtp.layers.0.input_layernorm.weight"),
            _ => panic!("Qwen3.5 MTP input norm lost its mtp namespace or +1 transform"),
        }
    }

    fn mapping_config(model_type: &str) -> ModelConfig {
        let json = format!(
            r#"{{
            "model_type":"{model_type}",
            "num_hidden_layers":2,
            "hidden_size":256,
            "num_attention_heads":8,
            "num_key_value_heads":2,
            "intermediate_size":512,
            "vocab_size":1024,
            "max_position_embeddings":2048
        }}"#
        );
        ModelConfig::from_hf(&crate::config::HfConfig::parse(&json))
    }

    #[test]
    fn norm_transform_is_arch_specific() {
        let qwen35 = mapping_config("qwen3_5");
        match resolve_ggml("blk.0.attn_norm.weight", &qwen35) {
            Some(HfTarget::Transform {
                kind: TransformKind::NormPlusOne,
                ..
            }) => {}
            _ => panic!("Qwen3.5 attn norm lost its required +1 transform"),
        }

        let step35 = step_mapping_config();
        for ggml in [
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.attn_q_norm.weight",
            "blk.0.attn_k_norm.weight",
            "output_norm.weight",
        ] {
            match resolve_ggml(ggml, &step35) {
                Some(HfTarget::Transform {
                    kind: TransformKind::NormPlusOne,
                    ..
                }) => {}
                _ => panic!("Step norm {ggml} lost its required +1 transform"),
            }
        }
        let mut step_weights = [-0.5f32, 0.0, 1.5];
        let (ne, bytes) = TransformKind::NormPlusOne.apply(&mut step_weights, vec![3], &step35);
        assert_eq!(ne, vec![3]);
        let folded: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(folded, vec![0.5, 1.0, 2.5]);

        let hy3 = mapping_config("hy_v3");
        for ggml in [
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.attn_q_norm.weight",
            "blk.0.attn_k_norm.weight",
            "output_norm.weight",
        ] {
            match resolve_ggml(ggml, &hy3) {
                Some(HfTarget::Plain(_)) => {}
                _ => panic!("Hy3 norm {ggml} must use the raw checkpoint weight"),
            }
        }
    }

    /// The V-head reorder permutation: HF grouped [0,1, 2,3, ...] -> ggml tiled [0,2,...,1,3,...].
    /// For nv=4, nk=2, head_dim=1: src heads [0,1,2,3] (grouped g*2+j) -> dst slots j*2+g.
    /// dst0=src0(g0,j0), dst1=src2(g1,j0), dst2=src1(g0,j1), dst3=src3(g1,j1) => [v0,v2,v1,v3].
    #[test]
    fn v_reorder_permutation() {
        let block = vec![10.0f32, 11.0, 12.0, 13.0]; // 4 heads, head_dim=1
        let out = reorder_v_axis(&block, 4, 2, 1);
        assert_eq!(out, vec![10.0, 12.0, 11.0, 13.0]);
        // identity when nv == nk (num_v_per_k == 1): order unchanged.
        let id = reorder_v_axis(&block, 4, 4, 1);
        assert_eq!(id, block);
    }

    /// reorder_rows_v over a band: a [4 rows x 2 cols] matrix, reorder all 4 rows (nv=4,nk=2,hd=1).
    #[test]
    fn v_reorder_rows() {
        // rows: r0=[0,1] r1=[2,3] r2=[4,5] r3=[6,7]; expect dst order [r0,r2,r1,r3].
        let data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let out = reorder_rows_v(&data, 4, 2, 1, 2, 0, 4);
        assert_eq!(out, vec![0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]);
    }

    /// `block128_perm` accepts exactly the permutations that act blockwise on 128-element blocks.
    #[test]
    fn block_perm_alignment() {
        // head_dim == 128: each V-head IS one scale block -> aligned, and the induced block
        // permutation is the head permutation itself ([0,2,1,3] for nv=4, nk=2).
        let p = v_perm(512, 4, 2, 128, 0, 512).unwrap();
        assert_eq!(block128_perm(&p).unwrap(), vec![0, 2, 1, 3]);
        // head_dim == 64: two heads share a scale block and the permutation splits them -> refused.
        let p = v_perm(256, 4, 2, 64, 0, 256).unwrap();
        assert!(block128_perm(&p).is_none());
        // out_f <= 128 (the a/b projections): one block, identity.
        let p = v_perm(48, 48, 16, 1, 0, 48).unwrap();
        assert_eq!(block128_perm(&p).unwrap(), vec![0]);
    }

    /// A minimal qwen35 cfg — `apply_fp8_blk` reads only `cfg.ssm` (via `head_params`), so the rest
    /// of the config is deliberately left at whatever the parser defaults produce.
    fn hybrid_cfg(nk: u32, nv: u32, hd: u32) -> ModelConfig {
        let dir = std::env::temp_dir().join(format!("memra_hfmap_cfg_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("config.json");
        std::fs::write(
            &p,
            br#"{"architectures":["Qwen3_5ForConditionalGeneration"],
            "model_type":"qwen3_5","text_config":{"hidden_size":512,"num_hidden_layers":2,
            "num_attention_heads":4,"num_key_value_heads":2,"head_dim":128,
            "intermediate_size":1024,"vocab_size":1000,"rms_norm_eps":1e-6,
            "full_attention_interval":4}}"#,
        )
        .unwrap();
        let mut cfg = ModelConfig::from_config_json(&p).unwrap();
        cfg.ssm = Some(crate::config::SsmConfig {
            conv_kernel: 4,
            inner_size: nv * hd,
            state_size: hd,
            time_step_rank: nv,
            group_count: nk,
        });
        cfg
    }

    /// THE EXACTNESS GATE for the block-128 V-reorder arm: permuting (codes, grid) together must
    /// reproduce, value for value, the permutation of the dequantized weight. This is the property
    /// `find_fp8_native`'s Transform arm relies on to keep 144 of the 27B's 208 FP8 projections on
    /// native residency instead of the Q8_0 slab.
    #[test]
    fn fp8_blk_v_reorder_is_exact() {
        let (nk, nv, hd) = (2usize, 4usize, 128usize);
        let (out_f, in_f) = (nv * hd, 256usize); // [512, 256], grid [4, 2]
        let cols = in_f.div_ceil(128);
        let codes: Vec<u8> = (0..out_f * in_f)
            .map(|i| {
                let m = (i % 0x7F) as u8;
                if m == 0x7F {
                    0x30
                } else {
                    m | (((i >> 7) & 1) as u8) << 7
                }
            })
            .collect();
        let scales: Vec<f32> = (0..out_f.div_ceil(128) * cols)
            .map(|i| 2f32.powi(i as i32 - 3))
            .collect();
        let deq = |c: u8, s: f32| crate::nvfp4_repack::fp8_e4m3_to_f32(c) * s;

        let mut cfg = hybrid_cfg(nk as u32, nv as u32, hd as u32);
        cfg.arch = Arch::Qwen35;
        let (ne, nc, ns, rows2, cols2) = TransformKind::ZReorderRows
            .apply_fp8_blk(&codes, out_f, in_f, &scales, cols, &cfg)
            .expect("grid-aligned reorder must be accepted");
        assert_eq!(ne, vec![in_f as u64, out_f as u64]);
        assert_eq!((rows2, cols2), (out_f.div_ceil(128), cols));

        // Reference: dequant, then permute the f32 rows with the existing primitive.
        let mut f32w: Vec<f32> = (0..out_f * in_f)
            .map(|i| deq(codes[i], scales[(i / in_f >> 7) * cols + ((i % in_f) >> 7)]))
            .collect();
        let want = reorder_rows_v(&f32w, nv, nk, hd, in_f, 0, out_f);
        f32w.clear();
        for o in 0..out_f {
            for e in 0..in_f {
                let got = deq(nc[o * in_f + e], ns[(o >> 7) * cols + (e >> 7)]);
                assert_eq!(
                    got.to_bits(),
                    want[o * in_f + e].to_bits(),
                    "permuted (codes,grid) != permuted dequant at [{o}][{e}]"
                );
            }
        }
    }

    /// A non-aligned permutation must be REFUSED (the caller then falls to the Q8_0 arm, which
    /// dequants with the pre-permutation grid and is correct) rather than silently approximated.
    #[test]
    fn fp8_blk_non_aligned_refused() {
        let (nk, nv, hd) = (2usize, 4usize, 64usize); // head_dim 64 < 128 -> heads share a block
        let (out_f, in_f) = (nv * hd, 256usize);
        let cols = in_f.div_ceil(128);
        let codes = vec![0x30u8; out_f * in_f];
        let scales = vec![1.0f32; out_f.div_ceil(128) * cols];
        let mut cfg = hybrid_cfg(nk as u32, nv as u32, hd as u32);
        cfg.arch = Arch::Qwen35;
        assert!(
            TransformKind::ZReorderRows
                .apply_fp8_blk(&codes, out_f, in_f, &scales, cols, &cfg)
                .is_none()
        );
        // value transforms are not permutations at all
        assert!(
            TransformKind::NormPlusOne
                .apply_fp8_blk(&codes, out_f, in_f, &scales, cols, &cfg)
                .is_none()
        );
    }
}

#[cfg(test)]
mod glm5_hc_resolve {
    use super::*;
    use crate::config::{HfConfig, ModelConfig};

    /// `TensorSource::has` goes through `resolve_ggml`, NOT `ggml_to_hf` directly: the engine's
    /// hyper-connection loader asks `blk.{il}.hc_*` and a `None` here reads as an ABSENT tensor.
    /// The sibling name pin exercises `ggml_to_hf`, so it cannot see a break in the dispatch that
    /// reaches it. Config comes from the real banked checkpoint config, not a hand-built struct.
    #[test]
    fn glm5_next_hc_names_resolve_through_the_source_path() {
        let json = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../research/glm53-flash-bringup-20260827/glm-config.json"
        ))
        .expect("banked glm5_next config");
        let cfg = ModelConfig::from_hf(&HfConfig::parse(&json));
        for site in ["hc_attn", "hc_ffn"] {
            for part in ["base", "fn", "scale"] {
                let ggml = format!("blk.0.{site}_{part}");
                let got = resolve_ggml(&ggml, &cfg)
                    .unwrap_or_else(|| panic!("resolve_ggml returned None for {ggml}"));
                let hf = match got {
                    HfTarget::Plain(hf) => hf,
                    HfTarget::Transform { hf, .. } => hf,
                };
                assert_eq!(hf, format!("model.layers.0.{site}_{part}"), "{ggml}");
            }
        }
    }
}
