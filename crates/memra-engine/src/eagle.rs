//! EAGLE3.1 greedy-chain speculative decode (research/basics/EAGLE-PLAN.md, N1-N7).
//!
//! Greedy spec decode is MATHEMATICALLY EXACT: the accepted+bonus token stream is token-for-token
//! identical to plain greedy `generate` (decode.rs). EAGLE differs from MTP (spec.rs) ONLY in the
//! DRAFT step: instead of the trunk-coupled NextN head, EAGLE drafts with a SEPARATE 1-layer model
//! (own vocab, own RoPE, untied lm_head) fed the trunk's hidden states from 3 aux layers [1,15,28]
//! fused through an encoder `fc`. The verify / accept-prefix / snapshot / rollback are REUSED
//! VERBATIM from spec.rs (decode_step_t, the greedy accept walk, cache.snapshot/rollback).
//!
//! On-disk draft (`eagle3-qwen35-9b/model.safetensors`, bf16, ground-truthed at impl time):
//!   fc.weight                            [4096, 12288]  (3*n_embd -> n_embd encoder)
//!   midlayer.input_layernorm.weight      [4096]         (RMSNorm of the prev-token EMBED)
//!   midlayer.hidden_norm.weight          [4096]         (RMSNorm of the recurrent hidden g)
//!   midlayer.self_attn.{q,k,v}_proj      q[4096,8192] k/v[1024,8192]  (in = 2*n_embd!)
//!   midlayer.self_attn.o_proj            [4096, 4096]
//!   midlayer.post_attention_layernorm    [4096]
//!   midlayer.mlp.{gate,up}_proj          [12288,4096]   down [4096,12288]
//!   norm.weight                          [4096]         (final RMSNorm before lm_head)
//!   lm_head.weight                       [32000, 4096]  (DRAFT vocab)
//!   d2t                                  [32000] i64    target_id = draft_id + d2t[draft_id]
//!   t2d                                  [248320] bool  (unused on the chain-greedy decode path)
//!
//! Op-sequence (authoritative: vLLM `llama_eagle3.py` LlamaDecoderLayer layer_idx==0, this ckpt's
//! flags norm_before_residual=false, norm_before_fc=false, fc_norm=false, norm_output=false):
//!   ENCODE (once/round): g = fc @ concat(aux[1], aux[15], aux[28])                 -> [n_embd]
//!   DRAFT step (T=1):
//!     e   = embed(prev_tok)                          (TARGET embedding; EAGLE3 shares it)
//!     eN  = RMSNorm(e,  input_layernorm)
//!     res = g                                         (_norm_after_residual: residual is PRE-norm g)
//!     gN  = RMSNorm(g,  hidden_norm)
//!     cat = [eN ; gN]                                 -> [2*n_embd]
//!     attn= o_proj @ SDPA( q,k,v = {q,k,v}_proj @ cat ; partial RoPE 64/256 @ theta 1e7 ; GQA16:4 )
//!     x1  = attn + res
//!     z   = RMSNorm(x1, post_attention_layernorm)
//!     mlp = down @ silu(gate @ z) * (up @ z)
//!     gsum= mlp + x1                                  (the model's final fused-add residual)
//!     dl  = lm_head @ RMSNorm(gsum, norm)             -> draft_logits[32000]
//!     g_next = gsum                                   (EAGLE recurrence: pre-norm residual)

use crate::Engine;
use crate::cache::{Cache, KvLayer};
use crate::forward::argmax;
use crate::hybrid::HybridModel;
use crate::model::GpuTensor;
use cudarc::driver::CudaSlice;
use memra_gguf::dequant;
use memra_gguf::safetensors::StModel;
use std::path::Path;

/// The EAGLE3 draft model: encoder `fc` + ONE Llama-style decoder layer + untied lm_head + d2t.
/// All weights are bf16 -> dequant to f32 GpuTensor::Float (the draft is ~0.8 GB; the matmuls go
/// through cuBLASLt `linear`). The draft attention is PLAIN Llama (no QK-norm, no output gate),
/// distinct from the trunk's gated/QK-normed full-attn.
pub struct Eagle3Draft {
    pub fc: GpuTensor,              // [3*n_embd, n_embd]  encoder
    pub input_layernorm: GpuTensor, // [n_embd]  norm of prev-token embedding
    pub hidden_norm: GpuTensor,     // [n_embd]  norm of recurrent g
    pub q_proj: GpuTensor,          // [2*n_embd, n_head*head_dim]
    pub k_proj: GpuTensor,          // [2*n_embd, n_head_kv*head_dim]
    pub v_proj: GpuTensor,          // [2*n_embd, n_head_kv*head_dim]
    pub o_proj: GpuTensor,          // [n_head*head_dim, n_embd]
    pub post_attention_layernorm: GpuTensor,
    pub gate_proj: GpuTensor,
    pub up_proj: GpuTensor,
    pub down_proj: GpuTensor,
    pub norm: GpuTensor,    // [n_embd]  final RMSNorm before lm_head
    pub lm_head: GpuTensor, // [n_embd, draft_vocab]
    pub d2t: Vec<i64>,      // [draft_vocab]  target_id = draft_id + d2t[draft_id]

    // shape / rope params (from the draft config.json, NOT the trunk cfg)
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub draft_vocab: usize,
    pub rope_dim_count: usize, // resolve_rope_dim_count (shared with GGUF/HF readers): 64 of 256
    pub rope_theta: f32,       // 1e7
    pub eps: f32,
    pub aux_layers: Vec<usize>, // [1, 15, 28]
}

fn validate_eagle_tensor(
    name: &str,
    info: &memra_gguf::safetensors::StInfo,
    expected_ne: &[u64],
) -> Result<Vec<u64>, String> {
    let ne = info.ne();
    if !matches!(info.dtype.as_str(), "BF16" | "F32") {
        return Err(format!(
            "EAGLE3 tensor {name} has dtype {}, expected BF16 or F32",
            info.dtype
        ));
    }
    if ne != expected_ne {
        return Err(format!(
            "EAGLE3 tensor {name} has shape {ne:?}, expected {expected_ne:?}"
        ));
    }
    Ok(ne)
}

fn validate_aux_layers(aux_layers: &[usize]) -> Result<(), String> {
    if aux_layers.len() != 3 {
        return Err(format!(
            "EAGLE3 fc.weight consumes exactly three auxiliary hidden states; config declares {}",
            aux_layers.len()
        ));
    }
    if aux_layers.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("EAGLE3 auxiliary layer ids must be strictly increasing".into());
    }
    Ok(())
}

fn validate_eagle_attention_geometry(
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
) -> Result<(), String> {
    if n_head == 0 || n_head_kv == 0 || head_dim == 0 || !n_head.is_multiple_of(n_head_kv) {
        return Err(format!(
            "EAGLE3 attention geometry requires nonzero n_head divisible by n_head_kv; got n_head={n_head}, n_head_kv={n_head_kv}, head_dim={head_dim}"
        ));
    }
    n_head
        .checked_mul(head_dim)
        .ok_or("EAGLE3 query-head geometry overflow")?;
    n_head_kv
        .checked_mul(head_dim)
        .ok_or("EAGLE3 key/value-head geometry overflow")?;
    Ok(())
}

fn validate_d2t_map(
    d2t: &[i64],
    draft_vocab: usize,
    target_vocab: Option<usize>,
) -> Result<(), String> {
    if d2t.len() != draft_vocab {
        return Err(format!(
            "EAGLE3 d2t has {} entries, expected draft_vocab_size {draft_vocab}",
            d2t.len()
        ));
    }
    for (draft_id, delta) in d2t.iter().copied().enumerate() {
        let target = i64::try_from(draft_id)
            .ok()
            .and_then(|id| id.checked_add(delta))
            .filter(|target| (0..=i64::from(u32::MAX)).contains(target))
            .filter(|target| {
                target_vocab.is_none_or(|vocab| usize::try_from(*target).is_ok_and(|id| id < vocab))
            });
        if target.is_none() {
            let limit = target_vocab
                .map(|vocab| format!(" target vocabulary of {vocab} entries"))
                .unwrap_or_else(|| " target u32 vocabulary".to_string());
            return Err(format!(
                "EAGLE3 d2t[{draft_id}]={delta} maps outside the{limit}"
            ));
        }
    }
    Ok(())
}

/// Load a single bf16 (or f32) tensor from the draft safetensors into a GpuTensor::Float.
/// `name` is the raw HF/EAGLE name in the file (e.g. "fc.weight", "midlayer.self_attn.q_proj.weight").
fn load_float(
    e: &Engine,
    m: &StModel,
    name: &str,
    expected_ne: &[u64],
) -> Result<GpuTensor, Box<dyn std::error::Error>> {
    let (info, bytes) = m
        .raw(name)
        .ok_or_else(|| format!("EAGLE3 draft missing tensor {name}"))?;
    let ne = validate_eagle_tensor(name, info, expected_ne)?;
    let n = ne.iter().try_fold(1u64, |total, extent| {
        total
            .checked_mul(*extent)
            .ok_or_else(|| format!("EAGLE3 tensor {name} element count overflow"))
    })?;
    let f32v = dequant::dequantize(info.ggml_type()?, bytes, n as usize);
    Ok(GpuTensor::Float {
        data: e.htod(&f32v)?,
        ne,
    })
}

impl Eagle3Draft {
    /// Load the EAGLE3 draft from a checkpoint directory (config.json + model.safetensors) or a
    /// direct path to the .safetensors. Reads the geometry/rope params from the sibling config.json.
    /// `aux_layers` is the trunk layer-id list from `eagle_config.eagle_aux_hidden_state_layer_ids`.
    pub fn load(e: &Engine, path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let dir = if path.is_file() {
            path.parent().unwrap_or(Path::new("."))
        } else {
            path
        };
        let cfg = EagleConfig::from_json(&dir.join("config.json"))?;
        validate_aux_layers(&cfg.aux_layers)?;
        validate_eagle_attention_geometry(cfg.n_head, cfg.n_head_kv, cfg.head_dim)?;
        let m = StModel::open(path)?;

        let d2t = read_i64(&m, "d2t")?;
        validate_d2t_map(&d2t, cfg.draft_vocab, None)?;

        let n = cfg.hidden_size as u64;
        let two_n = n.checked_mul(2).ok_or("EAGLE3 hidden geometry overflow")?;
        let three_n = n.checked_mul(3).ok_or("EAGLE3 hidden geometry overflow")?;
        let ff = cfg.intermediate_size as u64;
        let q = cfg
            .n_head
            .checked_mul(cfg.head_dim)
            .ok_or("EAGLE3 q projection geometry overflow")? as u64;
        let kv = cfg
            .n_head_kv
            .checked_mul(cfg.head_dim)
            .ok_or("EAGLE3 kv projection geometry overflow")? as u64;
        let vocab = cfg.draft_vocab as u64;

        let draft = Eagle3Draft {
            fc: load_float(e, &m, "fc.weight", &[three_n, n])?,
            input_layernorm: load_float(e, &m, "midlayer.input_layernorm.weight", &[n])?,
            hidden_norm: load_float(e, &m, "midlayer.hidden_norm.weight", &[n])?,
            q_proj: load_float(e, &m, "midlayer.self_attn.q_proj.weight", &[two_n, q])?,
            k_proj: load_float(e, &m, "midlayer.self_attn.k_proj.weight", &[two_n, kv])?,
            v_proj: load_float(e, &m, "midlayer.self_attn.v_proj.weight", &[two_n, kv])?,
            o_proj: load_float(e, &m, "midlayer.self_attn.o_proj.weight", &[q, n])?,
            post_attention_layernorm: load_float(
                e,
                &m,
                "midlayer.post_attention_layernorm.weight",
                &[n],
            )?,
            gate_proj: load_float(e, &m, "midlayer.mlp.gate_proj.weight", &[n, ff])?,
            up_proj: load_float(e, &m, "midlayer.mlp.up_proj.weight", &[n, ff])?,
            down_proj: load_float(e, &m, "midlayer.mlp.down_proj.weight", &[ff, n])?,
            norm: load_float(e, &m, "norm.weight", &[n])?,
            lm_head: load_float(e, &m, "lm_head.weight", &[n, vocab])?,
            d2t,
            n_embd: cfg.hidden_size,
            n_head: cfg.n_head,
            n_head_kv: cfg.n_head_kv,
            head_dim: cfg.head_dim,
            n_ff: cfg.intermediate_size,
            draft_vocab: cfg.draft_vocab,
            rope_dim_count: cfg.rope_dim_count(),
            rope_theta: cfg.rope_theta,
            eps: cfg.rms_eps,
            aux_layers: cfg.aux_layers,
        };
        Ok(draft)
    }

    /// Map a DRAFT-vocab id to a TARGET-vocab id (d2t is a DELTA: target = draft + d2t[draft]).
    #[inline]
    pub fn d2t_map(&self, draft_id: u32) -> u32 {
        (draft_id as i64 + self.d2t[draft_id as usize]) as u32
    }

    /// ENCODE (once per round, EAGLE-PLAN N3): g = fc @ concat(aux0, aux1, aux2). `aux` are the 3
    /// trunk residual hiddens of the just-committed token (decode_step_aux / decode_step_t_aux),
    /// in ascending-layer order. Returns the recurrent draft hidden `g` [n_embd].
    pub fn encode(
        &self,
        e: &Engine,
        aux: &[CudaSlice<f32>],
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        if aux.len() != self.aux_layers.len() {
            return Err(format!(
                "EAGLE3 encode received {} auxiliary states, expected {}",
                aux.len(),
                self.aux_layers.len()
            )
            .into());
        }
        let n = self.n_embd;
        let mut cat = e.zeros(self.aux_layers.len() * n)?;
        for (i, a) in aux.iter().enumerate() {
            e.copy_into(&mut cat, i * n, a, n)?;
        }
        e.matmul(&self.fc, &cat, 1) // [3*n_embd] @ fc[3n_embd,n_embd] -> [n_embd]
    }

    /// One DRAFT-token forward (EAGLE-PLAN N4, T=1). `prev_tok` = the TARGET token id to predict
    /// from (last committed or previous draft). `g` = the recurrent draft hidden (encode() output
    /// on round entry, then the previous step's g_next). Returns (draft_logits[draft_vocab] host,
    /// g_next dev). Mirrors the vLLM op-sequence documented at the top of this file.
    pub fn draft_token(
        &self,
        e: &Engine,
        target: &HybridModel,
        prev_tok: u32,
        g: &CudaSlice<f32>,
        scratch: &mut Eagle3Scratch,
        pos: usize,
    ) -> Result<(Vec<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let n = self.n_embd;
        let eps = self.eps;
        let pos_d = e.htod_i32(&[pos as i32])?;

        // e = TARGET embedding of prev_tok (EAGLE3 shares the target's token embedding).
        // eN = input_layernorm(e); gN = hidden_norm(g); residual = PRE-norm g (norm_after_residual).
        let e_emb = e.htod(&target.embd.gather(n, &[prev_tok]))?;
        let mut e_norm = e.zeros(n)?;
        e.rms_norm(
            &e_emb,
            self.input_layernorm.float_data(),
            &mut e_norm,
            n,
            1,
            eps,
        )?;
        let res = e.clone_dtod(g)?;
        let mut g_norm = e.zeros(n)?;
        e.rms_norm(g, self.hidden_norm.float_data(), &mut g_norm, n, 1, eps)?;
        // cat = [eN ; gN] -> [2*n_embd]  (vLLM llama_eagle3: torch.cat([embeds, hidden_states])).
        let mut cat = e.zeros(2 * n)?;
        e.copy_into(&mut cat, 0, &e_norm, n)?;
        e.copy_into(&mut cat, n, &g_norm, n)?;

        // attention from the 2*n_embd concat (plain Llama: no QK-norm, no output gate).
        let attn = self.attn(e, &cat, &pos_d, scratch)?;
        // x1 = attn + residual(g)
        let mut x1 = e.zeros(n)?;
        e.add(&attn, &res, &mut x1, n)?;
        // z = post_attention_layernorm(x1)
        let mut z = e.zeros(n)?;
        e.rms_norm(
            &x1,
            self.post_attention_layernorm.float_data(),
            &mut z,
            n,
            1,
            eps,
        )?;
        // mlp = down @ (silu(gate@z) * (up@z))
        let gate = e.matmul(&self.gate_proj, &z, 1)?;
        let up = e.matmul(&self.up_proj, &z, 1)?;
        let mut act = e.zeros(self.n_ff)?;
        e.silu_mul(&gate, &up, &mut act, self.n_ff)?;
        let mlp = e.matmul(&self.down_proj, &act, 1)?;
        // g_next = mlp + x1  (final fused-add residual; this is the aux_output recurrence)
        let mut g_next = e.zeros(n)?;
        e.add(&mlp, &x1, &mut g_next, n)?;
        // dl = lm_head @ norm(g_next)
        let mut hn = e.zeros(n)?;
        e.rms_norm(&g_next, self.norm.float_data(), &mut hn, n, 1, eps)?;
        let logits = e.matmul(&self.lm_head, &hn, 1)?;
        let host = e.dtoh(&logits)?;
        Ok((host, g_next))
    }

    /// Plain Llama attention over the [2*n_embd] concat input, T=1, on the draft's own scratch KV.
    /// q/k/v project from 2*n_embd; partial RoPE (rope_dim_count of head_dim) at the draft theta;
    /// GQA broadcast in fa_decode; o_proj back to n_embd. No QK-norm, no output gate.
    fn attn(
        &self,
        e: &Engine,
        cat: &CudaSlice<f32>,
        pos_d: &CudaSlice<i32>,
        scratch: &mut Eagle3Scratch,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (nh, nhkv, hd) = (self.n_head, self.n_head_kv, self.head_dim);
        let scale = 1.0 / (hd as f32).sqrt();
        let mut q = e.matmul(&self.q_proj, cat, 1)?; // [nh*hd]
        let mut k = e.matmul(&self.k_proj, cat, 1)?; // [nhkv*hd]
        let v = e.matmul(&self.v_proj, cat, 1)?; // [nhkv*hd]

        // partial RoPE: rope_dim_count from resolve_rope_dim_count (= 64 of 256), draft theta.
        e.rope_neox(
            &mut q,
            pos_d,
            hd,
            self.rope_dim_count,
            nh,
            1,
            self.rope_theta,
            1.0,
        )?;
        e.rope_neox(
            &mut k,
            pos_d,
            hd,
            self.rope_dim_count,
            nhkv,
            1,
            self.rope_theta,
            1.0,
        )?;

        let kv = &mut scratch.kv;
        e.append_kv_quantized(
            &k,
            &v,
            &mut kv.k,
            &mut kv.v,
            kv.len,
            kv.kv_dim_k,
            kv.kv_dim_v,
            kv.k_tok_bytes,
            kv.v_tok_bytes,
            false,
        )?;
        kv.len += 1;
        let t_kv = kv.len;
        let (ktb, vtb) = (kv.k_tok_bytes, kv.v_tok_bytes);
        let k_view = e.view_u8(&kv.k, t_kv * ktb);
        let v_view = e.view_u8(&kv.v, t_kv * vtb);
        let mut attn = e.zeros(nh * hd)?;
        e.fa_decode(
            &q, &k_view, &v_view, &mut attn, hd, nh, nhkv, t_kv, scale, ktb, vtb,
        )?;
        e.matmul(&self.o_proj, &attn, 1)
    }
}

/// Tiny scratch KV for the EAGLE3 draft layer (one full-attn layer). Reset each draft round. Uses
/// the SAME q8_0-K / q5_1-V quantized layout as the trunk KV (head_dim%32==0 holds: 256).
pub struct Eagle3Scratch {
    pub kv: KvLayer,
}
impl Eagle3Scratch {
    pub fn new(
        e: &Engine,
        draft: &Eagle3Draft,
        cap: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let (nhkv, hd) = (draft.n_head_kv, draft.head_dim);
        assert!(
            hd % 32 == 0,
            "KVQUANT requires head_dim%32==0 (EAGLE3 scratch)"
        );
        let kv_dim_k = hd * nhkv;
        let kv_dim_v = hd * nhkv;
        let (kbb, vbb) = crate::kv_blk_bytes(); // env-selected KV formats (default 34/24)
        let k_tok_bytes = (kv_dim_k / 32) * kbb;
        let v_tok_bytes = (kv_dim_v / 32) * vbb;
        Ok(Eagle3Scratch {
            kv: KvLayer {
                k: e.alloc_u8(cap * k_tok_bytes)?,
                v: e.alloc_u8(cap * v_tok_bytes)?,
                kv_dim_k,
                kv_dim_v,
                k_tok_bytes,
                v_tok_bytes,
                len: 0,
                ring: None,
                len_d: e.htod_i32(&[0])?,
                base_d: None,
            },
        })
    }
    pub fn reset(&mut self) {
        self.kv.len = 0;
    }
}

impl HybridModel {
    /// Greedy EAGLE3 speculative decode (EAGLE-PLAN N6). Token-identical to `generate(prompt,n)`
    /// but drafts K tokens with the separate EAGLE3 draft, then verifies them in ONE batched target
    /// forward. Verify/accept/snapshot/rollback are REUSED from the MTP path (decode_step_t,
    /// cache.snapshot/rollback). Returns (tokens, total_drafted, total_accepted).
    pub fn generate_spec_eagle(
        &self,
        e: &Engine,
        draft: &Eagle3Draft,
        prompt: &[u32],
        max_new: usize,
        k: usize,
    ) -> Result<(Vec<u32>, usize, usize), Box<dyn std::error::Error>> {
        self.refuse_hyper("generate_spec_eagle")?;
        assert!(k >= 1, "k must be >= 1");
        assert!(!prompt.is_empty(), "prompt must be non-empty");
        let n_vocab = self.output.out_features();
        validate_d2t_map(&draft.d2t, draft.draft_vocab, Some(n_vocab))?;
        let n_embd = self.cfg.n_embd as usize;
        assert_eq!(n_embd, draft.n_embd, "draft n_embd != target n_embd");
        let aux = &draft.aux_layers;
        let max_ctx = prompt.len() + max_new + k + 8;
        let mut cache = Cache::new(e, &self.cfg, max_ctx)?;

        // prime: feed the prompt; capture the LAST token's aux hiddens (seed for round-1 encode).
        let mut prime_logits = Vec::new();
        let mut prime_aux: Vec<CudaSlice<f32>> = Vec::new();
        for &tok in prompt {
            let (l, a) = self.decode_step_aux(e, tok, &mut cache, aux)?;
            prime_logits = l;
            prime_aux = a;
        }

        let mut scratch = Eagle3Scratch::new(e, draft, k + 1)?;
        let mut out: Vec<u32> = Vec::with_capacity(max_new);
        let mut total_drafted = 0usize;
        let mut total_accepted = 0usize;

        // EAGLE3 token/hidden alignment (vLLM `llama_eagle3.py`/`cnets.py`): the draft pairs the
        // aux hidden of position p with the EMBEDDING of the token at position p+1 (input_ids are the
        // target tokens shifted left by one). So drafting the token after `last_token` (at pos p)
        // uses g = encode(aux of the token BEFORE last_token, at pos p-1) and embed(last_token).
        // MEMRA_EAGLE_ALIGN=0 forces the un-shifted MTP-style pairing (aux & embed both = last_token)
        // for A/B comparison; default (1) is the EAGLE shift. The prime loop already gave us the
        // aux of the prompt's last token (= the predecessor of `last_token`), so we keep it as
        // `prev_aux` and roll it forward by one each round.
        let shift = std::env::var("MEMRA_EAGLE_ALIGN")
            .ok()
            .map(|s| s != "0")
            .unwrap_or(true);
        let mut last_token = argmax(&prime_logits) as u32;
        out.push(last_token);
        // prev_aux = aux of the token at the position whose forward predicted `last_token`
        // (= the prompt's last token for round 1). g_aux = aux of `last_token` itself.
        let mut prev_aux = prime_aux;
        let (mut last_logits, mut g_aux) = self.decode_step_aux(e, last_token, &mut cache, aux)?;

        while out.len() < max_new {
            let pos = cache.pos;
            let snap = cache.snapshot(e)?;

            // --- 1. ENCODE once: g0 = fc @ concat(aux). With the EAGLE shift, the seed aux is the
            //        PREDECESSOR token's (paired with embed(last_token)); else last_token's own. ---
            let seed_aux = if shift { &prev_aux } else { &g_aux };
            let g0 = draft.encode(e, seed_aux)?;

            // --- 2. DRAFT k tokens with the EAGLE3 draft (autoregressive, T=1 each) ---
            scratch.reset();
            let mut draft_toks: Vec<u32> = Vec::with_capacity(k);
            let mut prev = last_token;
            let mut g = g0;
            for j in 0..k {
                let (dl, g_next) = draft.draft_token(e, self, prev, &g, &mut scratch, pos + j)?;
                let d_draft = argmax(&dl) as u32;
                let d_target = draft.d2t_map(d_draft); // map draft-vocab id -> target-vocab id
                draft_toks.push(d_target);
                prev = d_target;
                g = g_next;
            }

            // --- 3. VERIFY: one batched target forward over draft_toks (T=k). REUSED from MTP. ---
            let tlogits = self.decode_step_t(e, &draft_toks, pos, &mut cache)?;

            // --- 4. GREEDY ACCEPT (walk prefix, stop at first mismatch). REUSED logic. ---
            let t_pred = |j: usize| -> u32 {
                if j == 0 {
                    argmax(&last_logits) as u32
                } else {
                    argmax(&tlogits[(j - 1) * n_vocab..j * n_vocab]) as u32
                }
            };
            let mut n_acc = 0usize;
            #[allow(clippy::needless_range_loop)]
            // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
            for j in 0..k {
                if t_pred(j) == draft_toks[j] {
                    n_acc += 1;
                } else {
                    break;
                }
            }
            let bonus = t_pred(n_acc);
            total_drafted += k;
            total_accepted += n_acc;

            // --- 5. COMMIT draft[0..n_acc] then bonus ---
            #[allow(clippy::needless_range_loop)]
            // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
            for j in 0..n_acc {
                if out.len() >= max_new {
                    break;
                }
                out.push(draft_toks[j]);
            }
            let bonus_emitted = out.len() < max_new;
            if bonus_emitted {
                out.push(bonus);
            }
            last_token = bonus;

            // --- 6. ROLLBACK + advance to pos + n_acc + 1 committed tokens (REUSED from MTP). The
            //        next round's EAGLE seed needs TWO auxs: g_aux = aux(bonus) and prev_aux =
            //        aux(bonus's predecessor). bonus's predecessor is the last committed token BEFORE
            //        bonus = draft[n_acc-1] if n_acc>=1, else this round's `last_token` (its aux is
            //        the CURRENT g_aux). We always replay [committed-tail.. , bonus] aux-capturing so
            //        the predecessor's aux is the second-to-last column; this keeps both exact.
            let pred_is_prev_round = n_acc == 0; // bonus's predecessor = old last_token
            let old_g_aux = std::mem::take(&mut g_aux); // = aux(old last_token)
            // Unified exact path (also covers full-accept n_acc==k): restore the pre-round snapshot
            // then replay the committed prefix draft[0..n_acc] ++ [bonus] as ONE T=(n_acc+1) aux-
            // capturing forward — single weight read, bit-identical to greedy (verify-all-columns
            // math). Captures aux at the last column (bonus) and, when the predecessor of bonus is a
            // replayed token (n_acc>=1), the second-to-last column.
            cache.rollback(e, &snap, 0)?;
            let mut replay: Vec<u32> = draft_toks[0..n_acc].to_vec();
            replay.push(bonus);
            let pred_col = if pred_is_prev_round {
                None
            } else {
                Some(replay.len() - 2)
            };
            let (rl, mut a_last, a_pred) =
                self.decode_step_t_aux2(e, &replay, pos, &mut cache, aux, pred_col)?;
            last_logits = rl[(replay.len() - 1) * n_vocab..replay.len() * n_vocab].to_vec();
            prev_aux = if pred_is_prev_round {
                old_g_aux
            } else {
                a_pred.unwrap()
            };
            g_aux = std::mem::take(&mut a_last);
        }
        out.truncate(max_new);
        Ok((out, total_drafted, total_accepted))
    }
}

// ============================ draft config.json (geometry + rope) ============================

struct EagleConfig {
    hidden_size: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    intermediate_size: usize,
    draft_vocab: usize,
    /// Explicit rotary dim count (`rotary_dim`, the MiniMax-M3 spelling). `None` on every
    /// published EAGLE3 draft today; read anyway because the trunk readers honour it and a
    /// draft config that declares it must not be silently ignored here.
    rotary_dim: Option<u32>,
    /// Fraction of `head_dim` that rotates (`partial_rotary_factor`, the Qwen3.5-family
    /// spelling; eagle3-qwen35-9b declares 0.25 both top-level and under `rope_parameters`).
    /// `None` means the config declares no partial rotary — full rope, resolved by
    /// `resolve_rope_dim_count`, NOT defaulted to 1.0 here so the absent/malformed arms take
    /// the same path the GGUF and HF/safetensors readers take.
    partial_rotary_factor: Option<f32>,
    rope_theta: f32,
    rms_eps: f32,
    aux_layers: Vec<usize>,
}

impl EagleConfig {
    /// Rotary width for the draft attention: `resolve_rope_dim_count`, the ONE derivation the
    /// GGUF and HF/safetensors readers already share (explicit dims > fraction > full width;
    /// malformed fractions take the full width instead of a silently odd rotation). This used
    /// to be a third, parallel implementation — `partial_rotary_factor.unwrap_or(1.0) *
    /// head_dim`, no `rotary_dim`, no malformed-factor refusal — which is exactly the
    /// two-implementations-drift class that gave the HF trunk path full rope on qwen3_5*
    /// while its GGUF twin was correct (hermes finding d3a9414b560416b5).
    fn rope_dim_count(&self) -> usize {
        memra_gguf::config::resolve_rope_dim_count(
            self.rotary_dim,
            self.partial_rotary_factor,
            self.head_dim as u32,
        ) as usize
    }

    fn from_json(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Self::from_json_str(&std::fs::read_to_string(path)?)
    }

    fn from_json_str(txt: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Minimal field extraction (avoid a serde dep here; the draft config.json is flat-ish).
        let num = |key: &str| -> Option<f64> {
            let pat = format!("\"{key}\"");
            let i = txt.find(&pat)? + pat.len();
            let rest = &txt[i..];
            let c = rest.find(':')? + 1;
            let tail = rest[c..].trim_start();
            let end = tail.find([',', '}', '\n']).unwrap_or(tail.len());
            tail[..end].trim().parse::<f64>().ok()
        };
        let aux_layers: Vec<usize> = {
            // eagle_aux_hidden_state_layer_ids: [1, 15, 28]
            let pat = "\"eagle_aux_hidden_state_layer_ids\"";
            match txt.find(pat) {
                Some(i) => {
                    let rest = &txt[i + pat.len()..];
                    let lb = rest.find('[').ok_or("no [ after aux ids")?;
                    let rb = rest.find(']').ok_or("no ] after aux ids")?;
                    rest[lb + 1..rb]
                        .split(',')
                        .filter_map(|s| s.trim().parse::<usize>().ok())
                        .collect()
                }
                None => vec![1, 15, 28], // fall back to the known EAGLE3-qwen35-9b layers
            }
        };
        Ok(EagleConfig {
            hidden_size: num("hidden_size").ok_or("hidden_size")? as usize,
            n_head: num("num_attention_heads").ok_or("num_attention_heads")? as usize,
            n_head_kv: num("num_key_value_heads").ok_or("num_key_value_heads")? as usize,
            head_dim: num("head_dim").ok_or("head_dim")? as usize,
            intermediate_size: num("intermediate_size").ok_or("intermediate_size")? as usize,
            draft_vocab: num("draft_vocab_size").ok_or("draft_vocab_size")? as usize,
            rotary_dim: num("rotary_dim").map(|v| v as u32),
            partial_rotary_factor: num("partial_rotary_factor").map(|v| v as f32),
            rope_theta: num("rope_theta").unwrap_or(10000.0) as f32,
            rms_eps: num("rms_norm_eps").unwrap_or(1e-6) as f32,
            aux_layers,
        })
    }
}

/// Read an i64 1-D tensor (d2t) from the draft safetensors.
fn read_i64(m: &StModel, name: &str) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
    let (info, bytes) = m
        .raw(name)
        .ok_or_else(|| format!("EAGLE3 draft missing {name}"))?;
    if info.dtype != "I64" {
        return Err(format!("{name} dtype must be I64, found {}", info.dtype).into());
    }
    let ne = info.ne();
    if ne.len() != 1 {
        return Err(format!("{name} must be rank-1, found shape {ne:?}").into());
    }
    let n = usize::try_from(ne[0]).map_err(|_| format!("{name} length does not fit usize"))?;
    let expected = n
        .checked_mul(8)
        .ok_or_else(|| format!("{name} byte length overflow"))?;
    if bytes.len() != expected {
        return Err(format!("{name} has {} bytes, expected {expected}", bytes.len()).into());
    }
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        v.push(i64::from_le_bytes(
            bytes[i * 8..i * 8 + 8].try_into().unwrap(),
        ));
    }
    Ok(v)
}

/// The draft loader's rope width shares `resolve_rope_dim_count` with the GGUF and HF readers —
/// these tests pin that it stays ONE derivation (hermes d3a9414b560416b5, the lane that fixed the
/// trunk HF path getting n_rot=256 where its GGUF twin said 64). CPU-only: config text in, width
/// out, no device, no checkpoint.
///
/// The fixture is the REAL `eagle3-qwen35-9b/config.json` — the exact checkpoint this loader
/// serves — verbatim, not a hand-written approximation. The trunk lane's postmortem: a fixture
/// unrepresentative of every real instance of the arch it claims to model is how the suite came
/// to bless full rope. Variant shapes below are derived from the real text by asserted edits, so
/// a drifted fixture fails loudly instead of testing a config that no longer exists.
#[cfg(test)]
mod tensor_contract_tests {
    use super::{
        validate_aux_layers, validate_d2t_map, validate_eagle_attention_geometry,
        validate_eagle_tensor,
    };
    use memra_gguf::safetensors::StInfo;

    #[test]
    fn every_eagle_weight_is_shape_and_dtype_checked_before_cuda() {
        let valid = StInfo {
            dtype: "BF16".into(),
            shape: vec![8, 4],
            data_offsets: [0, 64],
        };
        assert_eq!(validate_eagle_tensor("q", &valid, &[4, 8]).unwrap(), [4, 8]);
        let mut bad = valid.clone();
        bad.dtype = "I8".into();
        assert!(
            validate_eagle_tensor("q", &bad, &[4, 8])
                .unwrap_err()
                .contains("dtype")
        );
        bad = valid.clone();
        bad.shape = vec![32];
        assert!(
            validate_eagle_tensor("q", &bad, &[4, 8])
                .unwrap_err()
                .contains("shape")
        );
    }

    #[test]
    fn auxiliary_and_vocabulary_maps_are_closed_before_cuda() {
        assert!(validate_aux_layers(&[1, 15, 28]).is_ok());
        for invalid in [&[1, 2][..], &[1, 1, 2], &[2, 1, 3], &[1, 2, 3, 4]] {
            assert!(
                validate_aux_layers(invalid).is_err(),
                "accepted {invalid:?}"
            );
        }
        assert!(validate_d2t_map(&[0, 1, -1], 3, Some(4)).is_ok());
        assert!(validate_d2t_map(&[0], 2, None).is_err());
        assert!(validate_d2t_map(&[-1], 1, None).is_err());
        assert!(validate_d2t_map(&[i64::MAX], 1, None).is_err());
        let err = validate_d2t_map(&[4], 1, Some(4)).unwrap_err();
        assert!(err.contains("target vocabulary of 4 entries"), "{err}");
        assert!(validate_eagle_attention_geometry(32, 8, 128).is_ok());
        assert!(validate_eagle_attention_geometry(7, 8, 128).is_err());
        assert!(validate_eagle_attention_geometry(8, 0, 128).is_err());
        assert!(validate_eagle_attention_geometry(usize::MAX, 1, 2).is_err());
    }
}

#[cfg(test)]
mod draft_rope_width_tests {
    use super::EagleConfig;
    use memra_gguf::config::{HfConfig, resolve_rope_dim_count};

    /// Verbatim `~/ai-ml/hf-models/eagle3-qwen35-9b/config.json` (banked shape also in the lane
    /// receipts, darklanes research/ornith-prep-20260819/N-ROT-FIX.md): `partial_rotary_factor`
    /// 0.25 declared BOTH top-level and under `rope_parameters` (the Ornith spelling spread),
    /// `head_dim` 256, and — like every published qwen3_5-family config — NO `rotary_dim`.
    const EAGLE3_QWEN35_9B_CONFIG: &str = r#"{
  "architectures": [
    "LlamaForCausalLMEagle3"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 248040,
  "draft_vocab_size": 32000,
  "dtype": "bfloat16",
  "eos_token_id": 248044,
  "head_dim": 256,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 12288,
  "max_position_embeddings": 262144,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 16,
  "num_hidden_layers": 1,
  "num_key_value_heads": 4,
  "pad_token_id": null,
  "partial_rotary_factor": 0.25,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000000,
    "rope_type": "default"
  },
  "tie_word_embeddings": false,
  "transformers_version": "5.3.0",
  "use_cache": true,
  "vocab_size": 248320,
  "eagle_config": {
    "use_aux_hidden_state": true,
    "eagle_aux_hidden_state_layer_ids": [1, 15, 28]
  }
}"#;

    /// Edit the fixture, refusing to no-op: a variant built by a replace that matched nothing
    /// would silently test the unmodified shape.
    fn edited(from: &str, to: &str) -> String {
        assert!(
            EAGLE3_QWEN35_9B_CONFIG.contains(from),
            "fixture drifted: {from:?} not found — the variant below would test the wrong shape"
        );
        EAGLE3_QWEN35_9B_CONFIG.replace(from, to)
    }

    /// Both readers of one config must extract the same two rope facts. This is the divergence
    /// gate — the same shape as the trunk lane's `n_rot_agrees_across_the_gguf_and_hf_loader_paths`
    /// — because the draft reader is a hand-rolled scanner and `HfConfig::parse` is the structured
    /// parser, and nothing else forces them to agree on what a config declares.
    fn assert_reader_parity(json: &str) -> usize {
        let draft = EagleConfig::from_json_str(json).expect("draft reader must parse the fixture");
        let hf = HfConfig::parse(json);
        assert_eq!(
            draft.rotary_dim, hf.rotary_dim,
            "draft scanner and HfConfig::parse disagree on rotary_dim for the same config"
        );
        assert_eq!(
            draft.partial_rotary_factor, hf.partial_rotary_factor,
            "draft scanner and HfConfig::parse disagree on partial_rotary_factor for the same config"
        );
        let expected = resolve_rope_dim_count(
            hf.rotary_dim,
            hf.partial_rotary_factor,
            hf.head_dim.expect("fixture declares head_dim"),
        ) as usize;
        assert_eq!(
            draft.rope_dim_count(),
            expected,
            "draft rope width diverged from the shared derivation on the same facts"
        );
        draft.rope_dim_count()
    }

    /// The teeth: a mutation that reintroduces full-rope derivation (ignoring the factor, or
    /// multiplying an unwrap_or(1.0) default) fails HERE, on the real checkpoint's own config,
    /// with the corrupted band named.
    #[test]
    fn real_eagle3_qwen35_9b_config_derives_partial_rope_64_of_256() {
        let cfg = EagleConfig::from_json_str(EAGLE3_QWEN35_9B_CONFIG).expect("real config parses");
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(
            cfg.rotary_dim, None,
            "no published EAGLE3 draft declares rotary_dim"
        );
        assert_eq!(
            cfg.partial_rotary_factor,
            Some(0.25),
            "the declared factor must be READ, not defaulted — unwrap_or(1.0) is the bug class"
        );
        assert_eq!(
            cfg.rope_dim_count(),
            64,
            "eagle3-qwen35-9b rotates 64 of 256 head dims; full rope silently corrupts the \
             pass-through band 64..256 — no shape error, fluent output, wrecked long context"
        );
        assert_eq!(assert_reader_parity(EAGLE3_QWEN35_9B_CONFIG), 64);
    }

    /// The Qwen3.5-122B spelling: the factor ONLY under `rope_parameters`, nothing top-level.
    /// A rewrite of the scanner that reads only the top-level key regresses exactly here.
    #[test]
    fn nested_only_partial_rotary_spelling_is_still_partial_rope() {
        let json = edited("\n  \"partial_rotary_factor\": 0.25,", "");
        let cfg = EagleConfig::from_json_str(&json).expect("nested-only config parses");
        assert_eq!(
            cfg.partial_rotary_factor,
            Some(0.25),
            "rope_parameters spelling must be read"
        );
        assert_eq!(cfg.rope_dim_count(), 64);
        assert_reader_parity(&json);
    }

    /// The honest default, isolated (the trunk lane's
    /// `qwen35_hf_without_a_partial_rotary_declaration_is_full_rope` twin): no declaration at
    /// all means full rope, and this case must never be conflated with the partial answer.
    #[test]
    fn no_rope_declaration_is_full_rope() {
        let json = edited("\n  \"partial_rotary_factor\": 0.25,", "")
            .replace("\n    \"partial_rotary_factor\": 0.25,", "");
        assert!(
            !json.contains("partial_rotary_factor"),
            "variant edit failed: a factor spelling survived"
        );
        let cfg = EagleConfig::from_json_str(&json).expect("undeclared-rope config parses");
        assert_eq!(cfg.partial_rotary_factor, None);
        assert_eq!(
            cfg.rope_dim_count(),
            256,
            "absent declaration = every head dim rotates"
        );
        assert_reader_parity(&json);
    }

    /// Explicit dims beat the fraction — the shared precedence. The old draft code read ONLY the
    /// fraction, so a draft config carrying `rotary_dim` (the MiniMax-M3 spelling, what a
    /// converter writes once it has resolved the fraction) was silently ignored. A mutation back
    /// to factor-only arithmetic fails here.
    #[test]
    fn explicit_rotary_dim_wins_over_the_fraction() {
        let json = edited(
            "\n  \"partial_rotary_factor\": 0.25,",
            "\n  \"partial_rotary_factor\": 0.25,\n  \"rotary_dim\": 32,",
        );
        let cfg = EagleConfig::from_json_str(&json).expect("explicit-dims config parses");
        assert_eq!(cfg.rotary_dim, Some(32));
        assert_eq!(
            cfg.rope_dim_count(),
            32,
            "explicit rotary_dim is the more specific declaration and must win over the fraction"
        );
        assert_reader_parity(&json);
    }

    /// Malformed fractions refuse to truncate — same posture as the trunk readers. The OLD draft
    /// arithmetic multiplied the raw factor: 2.0 * 256 = a 512-dim rotation over a 256-dim head
    /// (writing past the head), and 0.0 * 256 -> max(2) = a 2-dim rotation that silently
    /// disables rope while looking like a plausible model.
    #[test]
    fn malformed_factor_takes_full_width_not_a_wider_than_head_rotation() {
        let over = EAGLE3_QWEN35_9B_CONFIG.replace(
            "\"partial_rotary_factor\": 0.25",
            "\"partial_rotary_factor\": 2.0",
        );
        let cfg = EagleConfig::from_json_str(&over).expect("factor-2.0 config parses");
        assert_eq!(cfg.partial_rotary_factor, Some(2.0));
        assert_eq!(
            cfg.rope_dim_count(),
            256,
            "factor 2.0 must take the FULL head width (256), never 512 — the old \
             factor*head_dim arithmetic rotated past the head allocation"
        );
        assert_reader_parity(&over);

        let zero = EAGLE3_QWEN35_9B_CONFIG.replace(
            "\"partial_rotary_factor\": 0.25",
            "\"partial_rotary_factor\": 0.0",
        );
        let cfg = EagleConfig::from_json_str(&zero).expect("factor-0.0 config parses");
        assert_eq!(
            cfg.rope_dim_count(),
            256,
            "factor 0.0 is malformed and takes the full width, not the old max(2) stub rotation"
        );
        assert_reader_parity(&zero);
    }
}
