//! The DFly2 drafter on the GPU route, in the route's numeric class
//! (`hy-v4-flash-bf16x2-activations`): every weight GEMM's input as two BF16 terms, the weights as
//! stored (BF16, FP8 128x128 blocks, NVFP4 with its E4M3 group scales and `weight_scale_2`), the
//! norms, RoPE, softmax and the head in F32. The program is the reference's
//! (`memra_reference::hy_v4_flash_dfly`, gated against the pinned vLLM program in
//! research/modelplan-onboarding-hy4flash-20261119/dfly/).
//!
//! A round runs right after the target call that scored its source position `p` (the call's last
//! accepted row): it reads that call's final normed hidden row and, per target index layer, the
//! keys that row's attention read (the indexer's recall list past the budget, every slot up to
//! `p` below it), which the call keeps for its last [`SNAP_ROWS`] rows. The block's own latent and
//! rope key rows are written into the target KV layer's cache at `p + 1 ..= p + 1 + k`, as the
//! vendor's shared storage does; the verify call that follows writes those rows again before any
//! query reads them. Rejected drafts roll back with [`Route::rollback_in`], which restores the
//! sliding rings' rows the verify call overwrote ([`Route::ring_backup_in`]).

use super::*;

/// Drafts a round at most (vLLM `DFlySpeculatorConfig.resolve_num_speculative_tokens`).
pub const MAX_DRAFTS: usize = 16;
/// The last rows of a call whose recall lists the drafter keeps: a verify call's rows (at most
/// `MAX_DRAFTS + 1`) or a prefill chunk's last.
pub const SNAP_ROWS: usize = MAX_DRAFTS + 1;

/// Wrong drafter programs the drafter gate must refuse.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DraftRed {
    #[default]
    None,
    /// draft layer `i` reads draft layer `i + 1`'s target layer (rotated)
    Layers,
    /// the MLP at half its width (the second half of the activations zeroed)
    Width,
    /// hidden correction off
    NoCorrection,
    /// the history keeps the first half of the target's top-k row
    TopK,
}

impl DraftRed {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "" | "none" => Self::None,
            "layers" => Self::Layers,
            "width" => Self::Width,
            "no-correction" => Self::NoCorrection,
            "topk" => Self::TopK,
            _ => return None,
        })
    }
}

struct DLayer {
    in_norm: DBuf,
    post_attn_norm: DBuf,
    gate: W,
    qa: W,
    qa_norm: DBuf,
    qb: W,
    kva: W,
    kva_norm: DBuf,
    kb_t: DBuf,
    kvb: DBuf,
    o: W,
    post_norm: DBuf,
    sinks: DBuf,
    mg: W,
    mu: W,
    md: W,
}

/// One target index layer's recall lists of a session's last call's last rows (the session
/// keeps them: `Session::snaps`).
pub(super) struct Snap {
    /// `[rows, ldl]` slots, `lens[rows]`; `None`: the layer attended densely in that call
    lists: Option<(DBuf, DBuf, usize)>,
    /// the absolute position of the first kept row
    pos0: usize,
    rows: usize,
}

#[derive(Default)]
struct DScratch {
    tok: Grow,
    x: Grow,
    res: Grow,
    xn: Grow,
    cat: Grow,
    fused: Grow,
    cs: Grow,
    posrow: Grow,
    gate: Grow,
    qa: Grow,
    qr: Grow,
    q: Grow,
    ckv: Grow,
    lat: Grow,
    qcat: Grow,
    olat: Grow,
    o: Grow,
    og: Grow,
    ao: Grow,
    a: Grow,
    list: Grow,
    lens: Grow,
    rows: Grow,
    po: Grow,
    pml: Grow,
    mg: Grow,
    mu: Grow,
    ma: Grow,
    out: Grow,
    /// a draft step's hidden rows of the rounds still drafting, gathered
    hrows: Grow,
    hcat: Grow,
    hg: Grow,
    hu: Grow,
    ha: Grow,
    hd: Grow,
    logits: Grow,
}

/// The drafter, resident beside the target's weights.
pub struct Drafter {
    layers: Vec<DLayer>,
    target_index: Vec<usize>,
    target_kv: Vec<usize>,
    topk: usize,
    /// `dfly_context_local_window`
    window: usize,
    final_norm: DBuf,
    eh: W,
    enorm: DBuf,
    hnorm: DBuf,
    fusion_gate: Vec<f32>,
    hcg: W,
    hcu: W,
    hcd: W,
    hc_embed_norm: DBuf,
    hc_hidden_norm: DBuf,
    ffn: usize,
    correction: usize,
    mask: u32,
    eps: f32,
    base: usize,
    /// drafts a round the artifact declares (`speculative_config.num_speculative_tokens`)
    pub k: Option<usize>,
    pub red: DraftRed,
    /// the drafter gate's composite runs: every call row's history per target index layer,
    /// `(layer, position) -> slots`, kept on the host when set
    pub record: Option<HashMap<(usize, usize), Vec<i32>>>,
    sc: DScratch,
    /// device bytes the drafter's weights hold
    pub weight_bytes: usize,
}

/// How a draft step turns its logits row into a draft: `pick(step, row)` answers the draft and
/// whether the round drafts on after it. Unset, every draft is the first maximum and the round
/// runs its `k` steps. The serve route's draft gate ends a round early once another draft cannot
/// pay (`hy4f_serve`); the lane's replay tools draw drafts in other ways to compare them. A pick
/// decides only which drafts a verify call scores, never a served token.
pub type DraftPick<'p> = &'p mut dyn FnMut(usize, &[f32]) -> Result<(u32, bool), String>;

/// A round's drafts and, when asked for, every step's logits (`[k, vocab]`).
#[derive(Debug, Clone, Default)]
pub struct DraftOut {
    pub ids: Vec<u32>,
    pub logits: Vec<f32>,
}

/// One request's draft round of a batched draft (`Route::draft_batch`): right after its
/// session's last call, from that call's row `row` (its own rows' index), the anchor chosen there,
/// up to `k` drafts and its `pick` (unset: the first maximum, all `k`).
pub struct DraftReq<'a, 'p> {
    pub sess: &'a mut Session,
    pub row: usize,
    pub anchor: u32,
    pub k: usize,
    pub pick: Option<DraftPick<'p>>,
}

/// A round's inputs, resolved (`draft_rounds`).
struct Round<'f, 'p> {
    hidden_p: u64,
    kvp: Vec<u64>,
    hist: HashMap<usize, Vec<i32>>,
    p: usize,
    anchor: u32,
    k: usize,
    forced: Option<&'f [u32]>,
    all_logits: bool,
    pick: Option<DraftPick<'p>>,
}

/// The sliding rings' rows a verify call will overwrite, saved before it.
pub struct RingBackup {
    from: usize,
    count: usize,
    /// every sliding layer's saved rows `[count, 576]`, one after another
    buf: Option<DBuf>,
    /// per sliding layer: (layer, its rows' first value in `buf`)
    rows: Vec<(usize, usize)>,
}

/// The ring slots of `positions` for each ring width among `layers`, one table: the table and
/// each width's first slot in it.
fn ring_slots(
    sess: &Session,
    layers: &[usize],
    positions: std::ops::Range<usize>,
) -> (Vec<i32>, HashMap<usize, usize>) {
    let mut table = Vec::new();
    let mut at = HashMap::new();
    for &il in layers {
        let w = sess.caches[il].ring;
        at.entry(w).or_insert_with(|| {
            let o = table.len();
            table.extend(positions.clone().map(|p| (p % w) as i32));
            o
        });
    }
    (table, at)
}

/// What the drafter reads of a target held outside a session (the drafter gate's oracle inputs):
/// the hidden row at `p` and, per target layer, its cache rows `[p + 1, 576]` (F32) and the slots
/// its attention read for `p`.
pub struct ExternalTarget<'t> {
    pub hidden: &'t [f32],
    pub kv: &'t HashMap<usize, Vec<f32>>,
    pub history: &'t HashMap<usize, Vec<i32>>,
}

/// One weight GEMM of the class with the NVFP4 tensor scale (`weight_scale_2`) applied.
fn dgemm(
    dev: &Dev,
    x: *const f32,
    ldx: usize,
    w: &W,
    y: *mut f32,
    ldy: usize,
    m: usize,
) -> Result<(), String> {
    cu!(memra_hy4r_gemm(
        x,
        ldx as i64,
        0,
        w.data.v(),
        0,
        w.scales.as_ref().map_or(std::ptr::null_mut(), |s| s.v()),
        0,
        w.scale2
            .as_ref()
            .map_or(std::ptr::null(), |s| s.f() as *const f32),
        w.fmt,
        w.grid_cols,
        y,
        ldy as i64,
        0,
        m as i32,
        w.n as i32,
        w.k as i32,
        1,
        std::ptr::null(),
        0,
        0,
        i32::from(m <= 16),
        0,
        dev.sp
    ))
}

impl Ckpt {
    /// One NVFP4 matrix `[n, k]` (ModelOpt: e2m1 pairs, an E4M3 scale per 16 inputs, a scalar F32
    /// `weight_scale_2`), or BF16 / FP8 as [`Ckpt::weight`] reads them, by the stored dtype.
    fn dfly_weight(&self, dev: &Dev, name: &str, n: usize, k: usize) -> Result<W, String> {
        match self.dtype(name)?.as_str() {
            "U8" => {
                let stem = name.strip_suffix(".weight").unwrap_or(name);
                // the codes in the GEMMs' tile layout, as `Ckpt::weight` keeps every GEMM weight
                let codes = self.upload_staged(dev, name, "U8", &[n, k / 2])?;
                Ok(W {
                    data: dev.tiled(codes, n, k / 2, k / 2, 0, 1)?.0,
                    scales: Some(self.upload(
                        dev,
                        &format!("{stem}.weight_scale"),
                        "F8_E4M3",
                        &[n, k / 16],
                    )?),
                    scale2: Some(self.upload(
                        dev,
                        &format!("{stem}.weight_scale_2"),
                        "F32",
                        &[],
                    )?),
                    fmt: FMT_NVFP4,
                    n,
                    k,
                    grid_cols: 0,
                    wbs: 0,
                    scbs: 0,
                })
            }
            "F8_E4M3" => self.weight(dev, name, n, k, HyV4WeightFormat::Fp8Block { block: 128 }),
            _ => self.weight(dev, name, n, k, HyV4WeightFormat::Bf16),
        }
    }
}

impl<'a> Route<'a> {
    /// Load the DFly2 drafter (`model.dfly_layers.*`) beside the resident target. The MLP width is
    /// the pack's (`dfly_ffn_width`: the declared width, else the artifact's); a checkpoint whose
    /// tensors are another width is refused here by shape.
    pub fn load_drafter(&mut self, ck: &Ckpt) -> Result<(), String> {
        let hy = self.hy;
        let d = &hy.dfly;
        if d.num_hidden_layers == 0 {
            return Err("the config declares no DFly2 drafter".into());
        }
        let dev = &self.dev;
        let before = dev.live_bytes();
        let h = self.plan.hidden_size as usize;
        let g = hy.indexed;
        let heads = g.heads as usize;
        let (nope, rd, vd) = (
            g.qk_nope_head_dim as usize,
            g.qk_rope_head_dim as usize,
            g.v_head_dim as usize,
        );
        let (ql, kvl) = (hy.q_lora_rank as usize, hy.kv_lora_rank as usize);
        if (nope, rd, kvl, vd) != (192, 64, 512, 256) || !heads.is_multiple_of(8) {
            return Err("the drafter's heads are not the route's geometry".into());
        }
        let ffn = memra_gguf::model_packs::hy_v4_flash::dfly_ffn_width(hy) as usize;
        let correction = d.hidden_correction_intermediate_size as usize;
        let mut layers = Vec::new();
        for i in 0..d.num_hidden_layers as usize {
            let p = format!("model.dfly_layers.{i}.");
            let at = |s: &str| format!("{p}self_attn.{s}");
            ck.check(&at("kv_b_proj.weight"), "BF16", &[heads * (nope + vd), kvl])?;
            let kvb_host = ck.host(&at("kv_b_proj.weight"))?;
            let mut kb_t = vec![0u8; heads * kvl * nope * 2];
            for hd in 0..heads {
                for dd in 0..nope {
                    let src = (hd * (nope + vd) + dd) * kvl;
                    for c in 0..kvl {
                        let dst = (hd * kvl + c) * nope + dd;
                        kb_t[2 * dst] = kvb_host[2 * (src + c)];
                        kb_t[2 * dst + 1] = kvb_host[2 * (src + c) + 1];
                    }
                }
            }
            layers.push(DLayer {
                in_norm: ck.vec(dev, &format!("{p}input_layernorm.weight"), &[h])?,
                post_attn_norm: ck.vec(
                    dev,
                    &format!("{p}post_attention_layernorm.weight"),
                    &[h],
                )?,
                gate: ck.dfly_weight(dev, &at("linear_gate.weight"), heads * vd, h)?,
                qa: ck.dfly_weight(dev, &at("q_a_proj.weight"), ql, h)?,
                qa_norm: ck.vec(dev, &at("q_a_layernorm.weight"), &[ql])?,
                qb: ck.dfly_weight(dev, &at("q_b_proj.weight"), heads * (nope + rd), ql)?,
                kva: ck.dfly_weight(dev, &at("kv_a_proj_with_mqa.weight"), kvl + rd, h)?,
                kva_norm: ck.vec(dev, &at("kv_a_layernorm.weight"), &[kvl])?,
                // tiled per head as the target's (`Route::load`): the absorbed-query rows, and
                // the value rows of each head (192..448 of its 448)
                kb_t: {
                    let rows = ck.stage_host(dev, &kb_t)?;
                    dev.tiled_dense(rows, kvl, nope * 2, nope * 2, kvl * nope * 2, heads)?
                },
                kvb: {
                    let rows = ck.stage_host(dev, &kvb_host)?;
                    dev.tiled_dense(
                        (rows as u64 + (nope * kvl * 2) as u64) as *const _,
                        vd,
                        kvl * 2,
                        kvl * 2,
                        (nope + vd) * kvl * 2,
                        heads,
                    )?
                },
                o: ck.dfly_weight(dev, &at("o_proj.weight"), h, heads * vd)?,
                post_norm: ck.vec(dev, &at("post_proj_norm.weight"), &[h])?,
                sinks: ck.vec(dev, &at("learnable_sink_param"), &[heads])?,
                mg: ck.dfly_weight(dev, &format!("{p}mlp.gate_proj.weight"), ffn, h)?,
                mu: ck.dfly_weight(dev, &format!("{p}mlp.up_proj.weight"), ffn, h)?,
                md: ck.dfly_weight(dev, &format!("{p}mlp.down_proj.weight"), h, ffn)?,
            });
        }
        let gl = "model.dfly_layers.";
        let base_val = g.rope_theta;
        let base = match self.bases.iter().position(|(b, _)| *b == base_val) {
            Some(b) => b,
            // a route without the trunk (the drafter gate's): the indexed layers' base
            None if self.layers.is_empty() => {
                let inv: Vec<f64> = (0..rd / 2)
                    .map(|i| 1.0 / (base_val as f64).powf((2 * i) as f64 / rd as f64))
                    .collect();
                self.bases.push((base_val, self.dev.upload(&inv)?));
                self.bases.len() - 1
            }
            None => return Err("the drafter's RoPE base is no target layer's".into()),
        };
        let dev = &self.dev;
        let drafter = Drafter {
            layers,
            target_index: d.target_index_layers.iter().map(|l| *l as usize).collect(),
            target_kv: d.target_kv_layers.iter().map(|l| *l as usize).collect(),
            topk: d.target_index_topk as usize,
            window: d.context_local_window as usize,
            final_norm: ck.vec(dev, &format!("{gl}final_norm.weight"), &[h])?,
            eh: ck.dfly_weight(dev, &format!("{gl}mtp_fusion.eh_proj.weight"), h, 2 * h)?,
            enorm: ck.vec(dev, &format!("{gl}mtp_fusion.enorm.weight"), &[h])?,
            hnorm: ck.vec(dev, &format!("{gl}mtp_fusion.hnorm.weight"), &[h])?,
            fusion_gate: ck.f32s(&format!("{gl}mtp_fusion.gate"), &[h])?,
            hcg: ck.dfly_weight(
                dev,
                &format!("{gl}hidden_correction.gate_proj.weight"),
                correction,
                2 * h,
            )?,
            hcu: ck.dfly_weight(
                dev,
                &format!("{gl}hidden_correction.up_proj.weight"),
                correction,
                2 * h,
            )?,
            hcd: ck.dfly_weight(
                dev,
                &format!("{gl}hidden_correction.down_proj.weight"),
                h,
                correction,
            )?,
            hc_embed_norm: ck.vec(
                dev,
                &format!("{gl}hidden_correction.embed_norm.weight"),
                &[h],
            )?,
            hc_hidden_norm: ck.vec(
                dev,
                &format!("{gl}hidden_correction.hidden_norm.weight"),
                &[h],
            )?,
            ffn,
            correction,
            mask: d.mask_token_id,
            eps: hy.rms_norm_eps,
            base,
            k: d.num_speculative_tokens.map(|k| k as usize),
            red: DraftRed::None,
            record: None,
            sc: DScratch::default(),
            weight_bytes: 0,
        };
        for &l in drafter.target_index.iter().chain(&drafter.target_kv) {
            if !self.layers.is_empty()
                && self
                    .layers
                    .get(l)
                    .and_then(|x| x.attn.indexer.as_ref())
                    .is_none()
            {
                return Err(format!(
                    "drafter target layer {l} is not an indexed layer of the route"
                ));
            }
        }
        // the tiling's staging buffer is not the drafter's
        ck.release_staging(dev);
        dev.sync()?;
        let mut drafter = drafter;
        drafter.weight_bytes = dev.live_bytes().saturating_sub(before);
        self.dfly = Some(drafter);
        Ok(())
    }

    /// The vocabulary: a verify call's logits are `[rows, vocab]`.
    pub fn vocab_size(&self) -> usize {
        self.plan.vocab_size as usize
    }

    /// Whether a drafter is loaded.
    pub fn has_drafter(&self) -> bool {
        self.dfly.is_some()
    }

    /// The loaded drafter (its declared `k`, red arms, weight bytes).
    pub fn drafter_mut(&mut self) -> Option<&mut Drafter> {
        self.dfly.as_mut()
    }

    /// After part `j`'s attention at layer `il` (its rows `p0..p0 + t`): keep the recall lists of
    /// its last rows in its session when `il` is a drafter target index layer. Called from
    /// `attention`, before the next part's indexer writes the lists again.
    pub(super) fn dfly_after_seg(&mut self, il: usize, j: usize) -> Result<(), String> {
        let Some(dr) = self.dfly.as_mut() else {
            return Ok(());
        };
        if !dr.target_index.contains(&il) {
            return Ok(());
        }
        let ix = self.layers[il]
            .attn
            .indexer
            .as_ref()
            .expect("checked at load")
            .plan;
        let (top_k, pool) = (ix.top_k as usize, ix.pool as usize);
        let sg = &mut self.segs[j];
        let (p0, t) = (sg.p0, sg.t());
        let kv_len = p0 + t;
        let rows = t.min(SNAP_ROWS);
        let pos0 = p0 + t - rows;
        let snap = sg.sess.snaps.entry(il).or_insert(Snap {
            lists: None,
            pos0,
            rows,
        });
        snap.pos0 = pos0;
        snap.rows = rows;
        if kv_len <= top_k {
            snap.lists = None;
            if let Some(rec) = dr.record.as_mut() {
                for q in p0..p0 + t {
                    rec.insert((il, q), (0..=q as i32).collect());
                }
            }
            return Ok(());
        }
        // the part's recall lists, as `attend_seg` built them (`memra_hy4r_recall`)
        let ng = kv_len / pool;
        let ks = (top_k / pool - 1).min(ng) + 1;
        let ldl = ks * pool;
        let dev = &self.dev;
        let list_src = self.sc.ilist.get(dev, t * ldl * 4)?.p;
        let lens_src = self.sc.ilens.get(dev, t * 4)?.p;
        let reuse = snap
            .lists
            .as_ref()
            .is_some_and(|(l, _, w)| *w == ldl && l.bytes >= SNAP_ROWS * ldl * 4);
        if !reuse {
            // the old lists go first; the new ones are admitted against MemAvailable
            snap.lists = None;
            admit(SNAP_ROWS * (ldl + 1) * 4, "the drafter's recall lists")?;
            snap.lists = Some((
                dev.alloc(SNAP_ROWS * ldl * 4)?,
                dev.alloc(SNAP_ROWS * 4)?,
                ldl,
            ));
        }
        if let Some(rec) = dr.record.as_mut() {
            let lists = dev.down::<i32>(list_src as *const _, t * ldl)?;
            let lens = dev.down::<i32>(lens_src as *const _, t)?;
            for r in 0..t {
                let q = p0 + r;
                let mut v: Vec<i32> = lists[r * ldl..r * ldl + lens[r] as usize]
                    .iter()
                    .copied()
                    .filter(|s| *s >= 0 && *s as usize <= q)
                    .collect();
                v.sort_unstable();
                v.dedup();
                rec.insert((il, q), v);
            }
        }
        let (l, n, _) = snap.lists.as_ref().unwrap();
        // the lists of the part's last rows: one copy each (rows `t - rows ..`, contiguous)
        cu!(memra_hy4r_copy(
            (list_src as *const f32).wrapping_add((t - rows) * ldl),
            ldl as i64,
            l.v(),
            ldl as i64,
            rows as i32,
            ldl as i32,
            0,
            std::ptr::null(),
            dev.sp
        ))?;
        cu!(memra_hy4r_copy(
            (lens_src as *const f32).wrapping_add(t - rows),
            1,
            n.v(),
            1,
            rows as i32,
            1,
            0,
            std::ptr::null(),
            dev.sp
        ))?;
        Ok(())
    }

    /// Save the sliding rings' rows positions `from .. from + count` will overwrite: one buffer
    /// for every sliding layer's rows, the slots of each ring width uploaded once.
    pub fn ring_backup_in(
        &mut self,
        sess: &Session,
        from: usize,
        count: usize,
    ) -> Result<RingBackup, String> {
        let layers: Vec<usize> = (0..sess.caches.len())
            .filter(|&il| sess.caches[il].ring > 0)
            .collect();
        if count == 0 || layers.is_empty() {
            return Ok(RingBackup {
                from,
                count,
                buf: None,
                rows: Vec::new(),
            });
        }
        if layers.iter().any(|&il| count > sess.caches[il].ring) {
            return Err("a verify call longer than the window".into());
        }
        let dev = &self.dev;
        let (table, at) = ring_slots(sess, &layers, from..from + count);
        let rb = self.sc.ring_rows.get(dev, table.len() * 4)?.i();
        dev.put(rb as *mut _, &table)?;
        let row = count * 576;
        admit(layers.len() * row * 4, "a verify call's ring backup")?;
        let buf = dev.alloc(layers.len() * row * 4)?;
        let mut rows = Vec::with_capacity(layers.len());
        for (k, &il) in layers.iter().enumerate() {
            let c = &sess.caches[il];
            cu!(memra_hy4r_gather_bf(
                c.kv.v(),
                576,
                rb.wrapping_add(at[&c.ring]),
                buf.at(k * row * 4),
                576,
                count as i32,
                576,
                dev.sp
            ))?;
            rows.push((il, k * row));
        }
        Ok(RingBackup {
            from,
            count,
            buf: Some(buf),
            rows,
        })
    }

    /// Keep the session's first `keep` positions after a verify call: the history is cut and the
    /// sliding rings' rows of the dropped positions get back what they held before the call
    /// (`backup`). Every other cache row past `keep` is written again before a query reads it.
    pub fn rollback_in(
        &mut self,
        sess: &mut Session,
        keep: usize,
        backup: &RingBackup,
    ) -> Result<(), String> {
        let end = sess.history.len();
        if keep > end || keep < backup.from || end > backup.from + backup.count {
            return Err(format!(
                "rollback to {keep} of {end} positions outside the saved {}..{}",
                backup.from,
                backup.from + backup.count
            ));
        }
        let Some(buf) = backup.buf.as_ref().filter(|_| keep < end) else {
            sess.history.truncate(keep);
            return Ok(());
        };
        let dev = &self.dev;
        let layers: Vec<usize> = backup.rows.iter().map(|r| r.0).collect();
        let (table, at) = ring_slots(sess, &layers, keep..end);
        let rb = self.sc.ring_rows.get(dev, table.len() * 4)?.i();
        dev.put(rb as *mut _, &table)?;
        for &(il, off) in &backup.rows {
            let c = &sess.caches[il];
            cu!(memra_hy4r_copy(
                (buf.f() as *const f32).wrapping_add(off + (keep - backup.from) * 576),
                576,
                c.kv.v(),
                576,
                (end - keep) as i32,
                576,
                0,
                rb.wrapping_add(at[&c.ring]),
                dev.sp
            ))?;
        }
        sess.history.truncate(keep);
        Ok(())
    }

    /// A draft round after `sess`'s last call: the source is that call's row `row`, which must be
    /// the session's last position `p`; `anchor` is the token chosen at `p`. `forced` replaces each
    /// step's previous token after step 0 (the gate's forced runs); `pick` chooses each draft from
    /// its logits row and may end the round after it (unset: the first maximum, all `k`;
    /// `DraftPick`).
    #[allow(clippy::too_many_arguments)]
    pub fn draft_in(
        &mut self,
        sess: &mut Session,
        row: usize,
        anchor: u32,
        k: usize,
        forced: Option<&[u32]>,
        all_logits: bool,
        pick: Option<DraftPick<'_>>,
    ) -> Result<DraftOut, String> {
        let (hidden_p, kvp, hist, p) = self.draft_inputs(sess, row, k)?;
        self.draft_core(
            hidden_p, &kvp, &hist, p, anchor, k, forced, all_logits, pick,
        )
    }

    /// The last call's first final normed hidden rows (`[rows, hidden]`), the head's input: a
    /// one-part call's rows (a batched call's first part's).
    pub fn last_hidden(&mut self, rows: usize) -> Result<Vec<f32>, String> {
        let h = self.plan.hidden_size as usize;
        let p = self.sc.hn.get(&self.dev, rows * h * 4)?.p;
        self.dev.down::<f32>(p as *const _, rows * h)
    }

    /// A draft round from position `p` of `sess`'s cache with the hidden row and histories given
    /// (the drafter gate's composite runs, after a teacher-forced pass: rounds run from the last
    /// anchor down, so a round's block rows are never a later round's history).
    #[allow(clippy::too_many_arguments)]
    pub fn draft_over_session(
        &mut self,
        sess: &Session,
        hidden: &[f32],
        hist: &HashMap<usize, Vec<i32>>,
        p: usize,
        anchor: u32,
        k: usize,
        forced: Option<&[u32]>,
    ) -> Result<DraftOut, String> {
        let dr = self.dfly.as_ref().ok_or("no drafter loaded")?;
        if p + 2 + k > sess.cap + 1 {
            return Err("the round passes the session's cache".into());
        }
        let kvp: Vec<u64> = dr.target_kv.iter().map(|l| sess.caches[*l].kv.p).collect();
        let hb = self.dev.upload(hidden)?;
        let out = self.draft_core(hb.p, &kvp, hist, p, anchor, k, forced, true, None);
        self.dev.sync()?;
        drop(hb);
        out
    }

    /// A draft round over a target held outside a session (the drafter gate: the oracle's target
    /// inputs rounded to F32).
    pub fn draft_external(
        &mut self,
        target: &ExternalTarget,
        p: usize,
        anchor: u32,
        k: usize,
        forced: Option<&[u32]>,
    ) -> Result<DraftOut, String> {
        let dr = self.dfly.as_ref().ok_or("no drafter loaded")?;
        let dev = &self.dev;
        let hb = dev.upload(target.hidden)?;
        let mut held = Vec::new();
        let mut kvp = Vec::new();
        for &l in &dr.target_kv {
            let rows = target
                .kv
                .get(&l)
                .ok_or(format!("no target KV for layer {l}"))?;
            if rows.len() < (p + 1) * 576 {
                return Err(format!(
                    "the target KV of layer {l} holds fewer than {} rows",
                    p + 1
                ));
            }
            admit((p + 2 + k) * 576 * 4, "the gate's target rows")?;
            let b = dev.alloc((p + 2 + k) * 576 * 4)?;
            dev.put(b.v(), &rows[..(p + 1) * 576])?;
            kvp.push(b.p);
            held.push(b);
        }
        let hist: HashMap<usize, Vec<i32>> = dr
            .target_index
            .iter()
            .map(|l| {
                target
                    .history
                    .get(l)
                    .cloned()
                    .map(|v| (*l, v))
                    .ok_or(format!("no target history for layer {l}"))
            })
            .collect::<Result<_, _>>()?;
        let out = self.draft_core(hb.p, &kvp, &hist, p, anchor, k, forced, true, None);
        self.dev.sync()?;
        drop(held);
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn draft_core(
        &mut self,
        hidden_p: u64,
        kvp: &[u64],
        hist: &HashMap<usize, Vec<i32>>,
        p: usize,
        anchor: u32,
        k: usize,
        forced: Option<&[u32]>,
        all_logits: bool,
        pick: Option<DraftPick<'_>>,
    ) -> Result<DraftOut, String> {
        let mut r = [Round {
            hidden_p,
            kvp: kvp.to_vec(),
            hist: hist.clone(),
            p,
            anchor,
            k,
            forced,
            all_logits,
            pick,
        }];
        Ok(self.draft_rounds(&mut r)?.pop().expect("one round"))
    }

    /// Draft rounds of several requests together: each round after its own session's last call,
    /// its drafts those of its own `draft_in` bit for bit (every row computes on its own). The
    /// weights are read once for all the rounds: the block rows of every round run through the
    /// drafter's layers in one pass, each round's attention over its own cache, then each draft
    /// step scores the rounds still drafting in one pass of the correction and the head. A round
    /// whose inputs are refused (no scored position, a round past its cache, no kept recall
    /// lists for its position) is answered by its own error and runs nothing; the outer error is
    /// the pass's, which every drafted round shares.
    #[allow(clippy::type_complexity)]
    pub fn draft_batch(
        &mut self,
        reqs: &mut [DraftReq<'_, '_>],
    ) -> Result<Vec<Result<DraftOut, String>>, String> {
        let mut out: Vec<Result<DraftOut, String>> = Vec::with_capacity(reqs.len());
        let mut rounds = Vec::with_capacity(reqs.len());
        let mut at = Vec::with_capacity(reqs.len());
        for (j, rq) in reqs.iter_mut().enumerate() {
            let inputs = if rq.k == 0 || rq.k > MAX_DRAFTS {
                Err(format!(
                    "{} drafts a round; DFly drafts 1 to {MAX_DRAFTS}",
                    rq.k
                ))
            } else {
                self.draft_inputs(rq.sess, rq.row, rq.k)
            };
            match inputs {
                Ok((hidden_p, kvp, hist, p)) => {
                    rounds.push(Round {
                        hidden_p,
                        kvp,
                        hist,
                        p,
                        anchor: rq.anchor,
                        k: rq.k,
                        forced: None,
                        all_logits: false,
                        pick: rq.pick.take(),
                    });
                    at.push(j);
                    out.push(Ok(DraftOut::default()));
                }
                Err(e) => out.push(Err(e)),
            }
        }
        for (j, d) in at.into_iter().zip(self.draft_rounds(&mut rounds)?) {
            out[j] = Ok(d);
        }
        Ok(out)
    }

    /// A round's inputs after `sess`'s last call (`draft_in`): its source hidden row, the target
    /// KV layers' caches, the target index layers' histories of `p`, and `p`.
    #[allow(clippy::type_complexity)]
    fn draft_inputs(
        &self,
        sess: &Session,
        row: usize,
        k: usize,
    ) -> Result<(u64, Vec<u64>, HashMap<usize, Vec<i32>>, usize), String> {
        let p = sess
            .history
            .len()
            .checked_sub(1)
            .ok_or("a draft round needs a scored position")?;
        if p + 2 + k > sess.cap + 1 {
            return Err(format!(
                "a round of {k} drafts at {p} passes the cache's {} positions",
                sess.cap
            ));
        }
        let dr = self.dfly.as_ref().ok_or("no drafter loaded")?;
        if self.layers.is_empty() {
            return Err("a route without the trunk drafts only from external targets".into());
        }
        // histories from the last call's kept recall lists
        let mut hist: HashMap<usize, Vec<i32>> = HashMap::new();
        for &l in &dr.target_index {
            let s = sess
                .snaps
                .get(&l)
                .ok_or("the drafter's target layers ran no call")?;
            if p < s.pos0 || p >= s.pos0 + s.rows {
                return Err(format!(
                    "position {p} is not among the last call's kept rows"
                ));
            }
            let r = p - s.pos0;
            let keys: Vec<i32> = match &s.lists {
                None => (0..=p as i32).collect(),
                Some((lists, lens, ldl)) => {
                    let n = (self.dev.down::<i32>(lens.at(r * 4), 1)?[0].max(0) as usize).min(*ldl);
                    let mut v = self.dev.down::<i32>(lists.at(r * ldl * 4), n)?;
                    v.retain(|s| *s >= 0 && (*s as usize) <= p);
                    v.sort_unstable();
                    v.dedup();
                    v
                }
            };
            hist.insert(l, keys);
        }
        let h = self.plan.hidden_size as usize;
        // the session's rows in its last call start at its `row0` (0 unless batched)
        let hidden_p =
            self.sc.hn.0.as_ref().ok_or("no call ran")?.p + ((sess.row0 + row) * h * 4) as u64;
        let kvp: Vec<u64> = dr.target_kv.iter().map(|l| sess.caches[*l].kv.p).collect();
        Ok((hidden_p, kvp, hist, p))
    }

    fn draft_rounds(&mut self, rounds: &mut [Round<'_, '_>]) -> Result<Vec<DraftOut>, String> {
        let nr = rounds.len();
        if nr == 0 {
            return Ok(Vec::new());
        }
        for r in rounds.iter() {
            if r.k == 0 || r.k > MAX_DRAFTS {
                return Err(format!(
                    "{} drafts a round; DFly drafts 1 to {MAX_DRAFTS}",
                    r.k
                ));
            }
            if r.forced.is_some_and(|f| f.len() < r.k) {
                return Err("fewer forced tokens than drafts".into());
            }
        }
        let h = self.plan.hidden_size as usize;
        let v = self.plan.vocab_size as usize;
        let dev = &self.dev;
        let embed = &self.embed;
        let lm_head = self.lm_head.p;
        let inv = self.bases[self.dfly.as_ref().unwrap().base].1.p;
        let dr = self.dfly.as_mut().unwrap();
        let red = dr.red;
        let sc = &mut dr.sc;
        let sp = dev.sp;
        // the rounds' block rows, in round order: round `j` at rows `r0[j] ..`
        let mut r0 = Vec::with_capacity(nr);
        let mut rows = 0usize;
        for r in rounds.iter() {
            r0.push(rows);
            rows += r.k + 1;
        }
        let topk_cap = dr.topk;
        let window = dr.window;
        let g = self.hy.indexed;
        let heads = g.heads as usize;
        let (ql, kvl) = (self.hy.q_lora_rank as usize, self.hy.kv_lora_rank as usize);
        let hd = 256usize;
        let eps = dr.eps;
        // each block: [anchor, mask x k] at p + 1 ..= p + 1 + k
        let mut ids = Vec::with_capacity(rows);
        for r in rounds.iter() {
            ids.push(r.anchor as i32);
            ids.extend(std::iter::repeat_n(dr.mask as i32, r.k));
        }
        let tok = sc.tok.get(dev, rows * 4)?.i();
        dev.put(tok as *mut _, &ids)?;
        let x = sc.x.get(dev, rows * h * 4)?.f();
        cu!(memra_hy4r_embed(
            embed.v(),
            tok,
            x,
            rows as i32,
            1,
            h as i32,
            sp
        ))?;
        // slot 0 of each block: e + gate * eh_proj(cat(enorm(e), hnorm(h_p)))
        let cat = sc.cat.get(dev, nr * 2 * h * 4)?.f();
        for (j, r) in rounds.iter().enumerate() {
            cu!(memra_hy4r_rms(
                x.wrapping_add(r0[j] * h),
                h as i64,
                dr.enorm.f(),
                cat.wrapping_add(j * 2 * h),
                2 * h as i64,
                1,
                h as i32,
                eps,
                sp
            ))?;
            cu!(memra_hy4r_rms(
                r.hidden_p as *const f32,
                h as i64,
                dr.hnorm.f(),
                cat.wrapping_add(j * 2 * h + h),
                2 * h as i64,
                1,
                h as i32,
                eps,
                sp
            ))?;
        }
        let fused = sc.fused.get(dev, nr * h * 4)?.f();
        dgemm(dev, cat, 2 * h, &dr.eh, fused, h, nr)?;
        let f_host = dev.down::<f32>(fused as *const _, nr * h)?;
        for (j, f_row) in f_host.chunks(h).enumerate() {
            let xr = x.wrapping_add(r0[j] * h);
            let mut x0 = dev.down::<f32>(xr as *const _, h)?;
            for ((xv, gv), fv) in x0.iter_mut().zip(&dr.fusion_gate).zip(f_row) {
                *xv += gv * fv;
            }
            dev.put(xr as *mut _, &x0)?;
        }
        // RoPE rows of the blocks' positions: round `j`'s sub-table at its rows
        let cs = sc.cs.get(dev, rows * 32 * 8)?;
        for (j, r) in rounds.iter().enumerate() {
            cu!(memra_hy4r_rope_table(
                cs.at(r0[j] * 32 * 8),
                (r.p + 1) as i32,
                (r.k + 1) as i32,
                32,
                inv as *const f64,
                sp
            ))?;
        }
        let cs = cs.p as *const std::os::raw::c_void;
        // the rows' RoPE rows 0 .. rows; the attention reads its `qpos` only off the list mode
        let pr: Vec<i32> = (0..rows as i32).collect();
        let posrow = sc.posrow.get(dev, rows * 4)?.i();
        dev.put(posrow as *mut _, &pr)?;
        let nl = dr.layers.len();
        // every round's block rows and, per drafter layer, its key lists (the target index layer's
        // history of p, then block rows 0..=j), one table uploaded once
        let limit = if red == DraftRed::TopK {
            topk_cap / 2
        } else {
            topk_cap
        };
        let mut tab: Vec<i32> = Vec::new();
        let mut dst_at = Vec::with_capacity(nr);
        for r in rounds.iter() {
            dst_at.push(tab.len());
            tab.extend((r.p + 1..=r.p + 1 + r.k).map(|q| q as i32));
        }
        // per (layer, round): (list offset, lens offset, ldl)
        let mut lists: Vec<Vec<(usize, usize, usize)>> = Vec::with_capacity(nl);
        for i in 0..nl {
            let at = if red == DraftRed::Layers {
                (i + 1) % nl
            } else {
                i
            };
            let il_index = dr.target_index[at];
            let mut per = Vec::with_capacity(nr);
            for r in rounds.iter() {
                let (p, nq) = (r.p, r.k + 1);
                let mut history = r.hist.get(&il_index).cloned().ok_or("no history")?;
                // `_reference_reuse_indices`: the row's first `target_index_topk` slots, those
                // before `anchor - window`, then the window's slots, then the block
                history.truncate(limit.min(history.len()));
                let start = (p + 1).saturating_sub(window);
                history.retain(|s| (*s as usize) < start);
                history.extend((start..p + 1).map(|s| s as i32));
                let ldl = history.len() + nq;
                let lo = tab.len();
                for b in 0..nq {
                    tab.extend_from_slice(&history);
                    tab.extend((0..=b).map(|c| (p + 1 + c) as i32));
                    tab.extend(std::iter::repeat_n(-1, nq - 1 - b));
                }
                let no = tab.len();
                tab.extend((0..nq).map(|b| (history.len() + b + 1) as i32));
                per.push((lo, no, ldl));
            }
            lists.push(per);
        }
        let tb = sc.list.get(dev, tab.len() * 4)?.i();
        dev.put(tb as *mut _, &tab)?;
        let res = sc.res.get(dev, rows * h * 4)?.f();
        let xn = sc.xn.get(dev, rows * h * 4)?.f();
        let mut have_res = false;
        for (i, per) in lists.iter().enumerate() {
            let at = if red == DraftRed::Layers {
                (i + 1) % nl
            } else {
                i
            };
            // the target KV layer this drafter layer reads (rotated under the `layers` red arm)
            let kv_slot = at;
            let l = &dr.layers[i];
            // hidden (+ residual) -> residual; input norm
            if have_res {
                cu!(memra_hy4r_add(
                    x,
                    x,
                    res,
                    std::ptr::null(),
                    (rows * h) as i64,
                    h as i32,
                    sp
                ))?;
            }
            cu!(memra_hy4r_copy(
                x,
                h as i64,
                res as *mut _,
                h as i64,
                rows as i32,
                h as i32,
                0,
                std::ptr::null(),
                sp
            ))?;
            cu!(memra_hy4r_rms(
                x,
                h as i64,
                l.in_norm.f(),
                xn,
                h as i64,
                rows as i32,
                h as i32,
                eps,
                sp
            ))?;
            // gated MLA over each round's target layer cache and its block
            let gate = sc.gate.get(dev, rows * heads * hd * 4)?.f();
            let qa = sc.qa.get(dev, rows * ql * 4)?.f();
            let qr = sc.qr.get(dev, rows * ql * 4)?.f();
            let q = sc.q.get(dev, rows * heads * hd * 4)?.f();
            let ckv = sc.ckv.get(dev, rows * 576 * 4)?.f();
            let lat = sc.lat.get(dev, rows * 576 * 4)?.f();
            let qcat = sc.qcat.get(dev, rows * heads * 576 * 4)?.f();
            let olat = sc.olat.get(dev, rows * heads * 512 * 4)?.f();
            let o = sc.o.get(dev, rows * heads * hd * 4)?.f();
            let og = sc.og.get(dev, rows * heads * hd * 4)?.f();
            let ao = sc.ao.get(dev, rows * h * 4)?.f();
            let aout = sc.a.get(dev, rows * h * 4)?.f();
            dgemm(dev, xn, h, &l.gate, gate, heads * hd, rows)?;
            dgemm(dev, xn, h, &l.qa, qa, ql, rows)?;
            cu!(memra_hy4r_rms(
                qa,
                ql as i64,
                l.qa_norm.f(),
                qr,
                ql as i64,
                rows as i32,
                ql as i32,
                eps,
                sp
            ))?;
            dgemm(dev, qr, ql, &l.qb, q, heads * hd, rows)?;
            cu!(memra_hy4r_rope(
                q,
                (heads * hd) as i64,
                hd as i64,
                rows as i32,
                heads as i32,
                192,
                32,
                cs,
                posrow,
                sp
            ))?;
            dgemm(dev, xn, h, &l.kva, ckv, 576, rows)?;
            cu!(memra_hy4r_rms(
                ckv,
                576,
                l.kva_norm.f(),
                lat,
                576,
                rows as i32,
                kvl as i32,
                eps,
                sp
            ))?;
            cu!(memra_hy4r_copy(
                ckv.wrapping_add(512),
                576,
                lat.wrapping_add(512) as *mut _,
                576,
                rows as i32,
                64,
                0,
                std::ptr::null(),
                sp
            ))?;
            cu!(memra_hy4r_rope(
                lat,
                576,
                576,
                rows as i32,
                1,
                512,
                32,
                cs,
                posrow,
                sp
            ))?;
            cu!(memra_hy4r_gemm(
                q,
                (heads * hd) as i64,
                hd as i64,
                l.kb_t.v(),
                (kvl * 192 * 2) as i64,
                std::ptr::null(),
                0,
                std::ptr::null(),
                FMT_BF16,
                0,
                qcat,
                (heads * 576) as i64,
                576,
                rows as i32,
                kvl as i32,
                192,
                heads as i32,
                std::ptr::null(),
                0,
                0,
                i32::from(rows <= 16),
                0,
                sp
            ))?;
            cu!(memra_hy4r_copy(
                q.wrapping_add(192),
                hd as i64,
                qcat.wrapping_add(512) as *mut _,
                576,
                (rows * heads) as i32,
                64,
                0,
                std::ptr::null(),
                sp
            ))?;
            for (j, r) in rounds.iter().enumerate() {
                let nq = r.k + 1;
                let lat_j = lat.wrapping_add(r0[j] * 576);
                // the block's latent rows into the target KV layer's cache at p + 1 ..= p + 1 + k
                let cache = r.kvp[kv_slot];
                cu!(memra_hy4r_copy(
                    lat_j,
                    576,
                    cache as *mut _,
                    576,
                    nq as i32,
                    576,
                    0,
                    tb.wrapping_add(dst_at[j]),
                    sp
                ))?;
                let (lo, no, ldl) = per[j];
                let (lb, nb) = (tb.wrapping_add(lo), tb.wrapping_add(no));
                let chunks = ldl.div_ceil(ATTN_CHUNK).max(1);
                let po = sc.po.get(dev, nq * heads * chunks * 512 * 4)?.f();
                let pml = sc.pml.get(dev, nq * heads * chunks * 2 * 4)?.f();
                cu!(memra_hy4r_attn(
                    qcat.wrapping_add(r0[j] * heads * 576),
                    cache as *const _,
                    lb,
                    ldl as i32,
                    nb,
                    posrow,
                    1,
                    0,
                    0,
                    0,
                    nq as i32,
                    heads as i32,
                    chunks as i32,
                    (256f64).powf(-0.5) as f32,
                    l.sinks.f(),
                    po,
                    pml,
                    olat.wrapping_add(r0[j] * heads * 512),
                    sp
                ))?;
            }
            cu!(memra_hy4r_gemm(
                olat,
                (heads * 512) as i64,
                512,
                l.kvb.v(),
                (256 * kvl * 2) as i64,
                std::ptr::null(),
                0,
                std::ptr::null(),
                FMT_BF16,
                0,
                o,
                (heads * hd) as i64,
                hd as i64,
                rows as i32,
                256,
                kvl as i32,
                heads as i32,
                std::ptr::null(),
                0,
                0,
                i32::from(rows <= 16),
                0,
                sp
            ))?;
            cu!(memra_hy4r_act(
                gate,
                o,
                og,
                (rows * heads * hd) as i64,
                -1.0,
                1,
                sp
            ))?;
            dgemm(dev, og, heads * hd, &l.o, ao, h, rows)?;
            cu!(memra_hy4r_rms(
                ao,
                h as i64,
                l.post_norm.f(),
                aout,
                h as i64,
                rows as i32,
                h as i32,
                eps,
                sp
            ))?;
            // + residual -> residual; post-attention norm; dense SwiGLU
            cu!(memra_hy4r_add(
                res,
                aout,
                res,
                std::ptr::null(),
                (rows * h) as i64,
                h as i32,
                sp
            ))?;
            cu!(memra_hy4r_rms(
                res,
                h as i64,
                l.post_attn_norm.f(),
                xn,
                h as i64,
                rows as i32,
                h as i32,
                eps,
                sp
            ))?;
            let ffn = dr.ffn;
            let mg = sc.mg.get(dev, rows * ffn * 4)?.f();
            let mu = sc.mu.get(dev, rows * ffn * 4)?.f();
            let ma = sc.ma.get(dev, rows * ffn * 4)?.f();
            dgemm(dev, xn, h, &l.mg, mg, ffn, rows)?;
            dgemm(dev, xn, h, &l.mu, mu, ffn, rows)?;
            cu!(memra_hy4r_act(mg, mu, ma, (rows * ffn) as i64, -1.0, 0, sp))?;
            if red == DraftRed::Width {
                for r in 0..rows {
                    cu!(memra_hy4r_memset(
                        (ma as *mut u8).wrapping_add((r * ffn + ffn / 2) * 4) as *mut _,
                        0,
                        (ffn - ffn / 2) * 4,
                        sp
                    ))?;
                }
            }
            dgemm(dev, ma, ffn, &l.md, x, h, rows)?;
            have_res = true;
        }
        // final_norm(hidden + residual)
        cu!(memra_hy4r_add(
            x,
            x,
            res,
            std::ptr::null(),
            (rows * h) as i64,
            h as i32,
            sp
        ))?;
        let out = sc.out.get(dev, rows * h * 4)?.f();
        cu!(memra_hy4r_rms(
            x,
            h as i64,
            dr.final_norm.f(),
            out,
            h as i64,
            rows as i32,
            h as i32,
            eps,
            sp
        ))?;
        // the drafts: corrected hidden[s + 1], the target's F32 head, the first maximum; each
        // step scores the rounds still drafting, in round order
        let ci = dr.correction;
        let hrows = sc.hrows.get(dev, nr * h * 4)?.f();
        let hcat = sc.hcat.get(dev, nr * 2 * h * 4)?.f();
        let hg = sc.hg.get(dev, nr * ci * 4)?.f();
        let hu = sc.hu.get(dev, nr * ci * 4)?.f();
        let ha = sc.ha.get(dev, nr * ci * 4)?.f();
        let hdl = sc.hd.get(dev, nr * h * 4)?.f();
        let lg = sc.logits.get(dev, nr * v * 4)?.f();
        let pe = sc.tok.get(dev, rows * 4)?.i();
        let emb_row = sc.fused.get(dev, nr * h * 4)?.f();
        let mut prev: Vec<u32> = rounds.iter().map(|r| r.anchor).collect();
        let mut going = vec![true; nr];
        let mut outs: Vec<DraftOut> = vec![DraftOut::default(); nr];
        let max_k = rounds.iter().map(|r| r.k).max().unwrap();
        for s in 0..max_k {
            let act: Vec<usize> = (0..nr).filter(|&j| going[j] && s < rounds[j].k).collect();
            if act.is_empty() {
                break;
            }
            let m = act.len();
            // the active rounds' hidden rows `s + 1`, gathered in order
            for (a, &j) in act.iter().enumerate() {
                cu!(memra_hy4r_copy(
                    out.wrapping_add((r0[j] + s + 1) * h),
                    h as i64,
                    hrows.wrapping_add(a * h) as *mut _,
                    h as i64,
                    1,
                    h as i32,
                    0,
                    std::ptr::null(),
                    sp
                ))?;
            }
            let corrected: *const f32 = if red == DraftRed::NoCorrection {
                hrows
            } else {
                let pv: Vec<i32> = act.iter().map(|&j| prev[j] as i32).collect();
                dev.put(pe as *mut _, &pv)?;
                cu!(memra_hy4r_embed(
                    embed.v(),
                    pe,
                    emb_row,
                    m as i32,
                    1,
                    h as i32,
                    sp
                ))?;
                cu!(memra_hy4r_rms(
                    hrows,
                    h as i64,
                    dr.hc_hidden_norm.f(),
                    hcat,
                    2 * h as i64,
                    m as i32,
                    h as i32,
                    eps,
                    sp
                ))?;
                cu!(memra_hy4r_rms(
                    emb_row,
                    h as i64,
                    dr.hc_embed_norm.f(),
                    hcat.wrapping_add(h),
                    2 * h as i64,
                    m as i32,
                    h as i32,
                    eps,
                    sp
                ))?;
                dgemm(dev, hcat, 2 * h, &dr.hcg, hg, ci, m)?;
                dgemm(dev, hcat, 2 * h, &dr.hcu, hu, ci, m)?;
                cu!(memra_hy4r_act(hg, hu, ha, (m * ci) as i64, -1.0, 0, sp))?;
                dgemm(dev, ha, ci, &dr.hcd, hdl, h, m)?;
                cu!(memra_hy4r_add(
                    hdl,
                    hdl,
                    hrows,
                    std::ptr::null(),
                    (m * h) as i64,
                    h as i32,
                    sp
                ))?;
                hdl
            };
            Gm {
                dev,
                red: RedArms::default(),
            }
            .gemm_f32_raw(corrected, lm_head, v, h, lg, m)?;
            let all = dev.down::<f32>(lg as *const _, m * v)?;
            for (a, &j) in act.iter().enumerate() {
                let row = &all[a * v..(a + 1) * v];
                if row.iter().any(|x| !x.is_finite()) {
                    return Err(format!("draft step {s}: non-finite logits"));
                }
                let r = &mut rounds[j];
                let (best, go_on) = match r.pick.as_mut() {
                    Some(f) => {
                        let (d, go_on) = f(s, row)?;
                        (d as usize, go_on)
                    }
                    None => {
                        // `greedy_argmax`: the first of equal maxima
                        let mut best = 0usize;
                        for (i, x) in row.iter().enumerate() {
                            if *x > row[best] {
                                best = i;
                            }
                        }
                        (best, true)
                    }
                };
                if best >= v {
                    return Err(format!(
                        "draft step {s}: picked id {best} past the vocabulary"
                    ));
                }
                outs[j].ids.push(best as u32);
                if r.all_logits {
                    outs[j].logits.extend_from_slice(row);
                }
                prev[j] = match r.forced {
                    Some(f) => f[s],
                    None => best as u32,
                };
                if !go_on {
                    going[j] = false;
                }
            }
        }
        Ok(outs)
    }

    /// Device bytes the drafter's call scratch holds now (grown on its first rounds, then kept):
    /// the share of the drafter's state every session uses one at a time.
    pub fn drafter_scratch_bytes(&self) -> usize {
        let Some(dr) = &self.dfly else { return 0 };
        let s = &dr.sc;
        [
            &s.tok, &s.x, &s.res, &s.xn, &s.cat, &s.fused, &s.cs, &s.posrow, &s.gate, &s.qa, &s.qr,
            &s.q, &s.ckv, &s.lat, &s.qcat, &s.olat, &s.o, &s.og, &s.ao, &s.a, &s.list, &s.lens,
            &s.rows, &s.po, &s.pml, &s.mg, &s.mu, &s.ma, &s.out, &s.hrows, &s.hcat, &s.hg, &s.hu,
            &s.ha, &s.hd, &s.logits,
        ]
        .iter()
        .map(|g| g.0.as_ref().map_or(0, |b| b.bytes))
        .sum::<usize>()
    }

    /// Device bytes a session's recall-list snapshots take at most (`Session::snaps`): per
    /// drafter target index layer, [`SNAP_ROWS`] lists of up to the budget's slots and their
    /// lengths, each at least the allocator's 16 bytes.
    pub(super) fn snap_bytes(&self) -> usize {
        let Some(dr) = &self.dfly else { return 0 };
        dr.target_index
            .iter()
            .map(|&l| {
                let top_k = self
                    .layers
                    .get(l)
                    .and_then(|ly| ly.attn.indexer.as_ref())
                    .map_or(0, |ix| ix.plan.top_k as usize);
                (SNAP_ROWS * top_k * 4).max(16) + (SNAP_ROWS * 4).max(16)
            })
            .sum()
    }
}
