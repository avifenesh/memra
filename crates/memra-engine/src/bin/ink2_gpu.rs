//! Inkling 2 Mini on one GPU (`memra_engine::inkling2_gpu`): the checkpoint runner of the native
//! sm_120a route. Weights resident as stored, a carried cache per slot.
//!
//! usage: ink2-gpu CKPT_DIR [--class declared|float] MODE... [--red ARM]... [--chunk N]
//!                 [--context N] [--out FILE] [--device D]
//!        ink2-gpu CKPT_DIR --numeric-class [--class ...]
//!   --class declared|float   the plan: every layer on its declared activation quantizer (the
//!                            family's default class), or every layer on float expert inputs (the
//!                            bring-up class, which has an independent oracle)
//!   --tape DIR               score against an oracle tape directory (tokens, per-slot residuals,
//!                            router picks, logits, greedy ids): the tape's prefill, then one
//!                            teacher-forced decode call per later position
//!   --force-router           run the tape's router picks and weights (the route's own picks are
//!                            still compared)
//!   --force-moe-in           feed every MoE the tape's MoE input and score its output alone (the
//!                            per-slot gate of a class that is not continuous)
//!   --gen N                  greedy: the tape's prompt, then N - 1 decode calls on the argmax,
//!                            scored against the tape's greedy ids; timed, launches counted
//!   --prec SPEC              precision arms, group=f32|f32c|f64 (groups: controller, mix, norm, conv,
//!                            router, epilogue, attn, all)
//!   --corpus FILE --arms A;B  teacher-forced KL and NLL of each arm against all-f64 (lane F's corpora)
//!   --eager                  no decode graph: every step eager (the graph's check arm)
//!   --gen-digest             with --gen: an FNV digest of every step's logits row (untimed check)
//!   --prompt-len N           with --gen: the prompt is the tape's ids repeated to N tokens (a
//!                            context length for timing; no oracle ids past the tape's)
//!   --chunkings A,B --limit N   the first N tape ids as calls of A rows and again as calls of B
//!                            rows (1 = one token per call); the logits and every slot's output
//!                            must be bit-identical
//!   --red ARM                a wrong program for the gate's red arms: dequant-bf16 (float
//!                            class only), conv-state, ties-high, commit-on-error (with
//!                            --inject-failure)
//!   --contract-check         CPU only: bind the tensor contract over the census and check that
//!                            its red arms (extra, missing, reshaped, retyped tensor; a read of a
//!                            name it does not bind) are refused
//!   --attn-probe S,S..       the tape's prompt, then the last 4 query rows of the last call of
//!                            each full-attention slot S recomputed on the host from the route's
//!                            own queries and caches, in f64 and as an f32 eager program
//!   --inject-failure SLOT    the atomicity gate: the tape's first --limit ids, half as a prefill,
//!                            then one decode call that fails after SLOT, then one decode call per
//!                            later id; every slot output and logit row must equal a clean run
//!   --chunk N                rows one call may hold (default 512)

use memra_engine::inkling2_gpu::{
    Ckpt, Dev, Head, Observer, Prec, RedArms, Route, Tap, numeric_class,
};
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::{ExpertActivationMode, ModelPlan};
use memra_gguf::safetensors::StShard;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("ink2-gpu: {e}");
        std::process::exit(1);
    }
}

#[derive(Default)]
struct Args {
    ckpt: PathBuf,
    class: String,
    tape: Option<PathBuf>,
    force_router: bool,
    force_moe_in: bool,
    gen_tokens: usize,
    /// --prompt-len N: the gen prompt is the tape's tokens repeated to N (a context length for timing)
    prompt_len: usize,
    /// --gen-digest: fold every decode step's logits row into a digest (an untimed check)
    gen_digest: bool,
    /// --eager: no decode graph (the check arm of a graph's bits)
    eager: bool,
    /// --prec SPEC: precision arms (group=f32|f32c|f64,...; see Prec)
    prec: String,
    /// --corpus FILE --arms SPEC;SPEC: teacher-forced KL / NLL of each arm against all-f64
    corpus: Option<PathBuf>,
    arms: String,
    chunkings: Option<(usize, usize)>,
    limit: usize,
    red: RedArms,
    red_names: Vec<String>,
    chunk: usize,
    context: usize,
    out: Option<PathBuf>,
    device: usize,
    numeric_class: bool,
    score: bool,
    inject_failure: Option<u32>,
    contract_check: bool,
    attn_probe: Vec<u32>,
}

fn parse() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let mut a = Args {
        ckpt: PathBuf::from(it.next().ok_or("usage: ink2-gpu CKPT_DIR ...")?),
        class: "declared".into(),
        chunk: 512,
        ..Args::default()
    };
    while let Some(arg) = it.next() {
        let mut val = || it.next().ok_or(format!("{arg} needs a value"));
        let num = |v: String| v.parse::<usize>().map_err(|e| format!("{e}"));
        match arg.as_str() {
            "--class" => a.class = val()?,
            "--tape" => a.tape = Some(PathBuf::from(val()?)),
            "--score" => a.score = true,
            "--force-router" => a.force_router = true,
            "--force-moe-in" => a.force_moe_in = true,
            "--gen" => a.gen_tokens = num(val()?)?,
            "--prompt-len" => a.prompt_len = num(val()?)?,
            "--gen-digest" => a.gen_digest = true,
            "--eager" => a.eager = true,
            "--prec" => a.prec = val()?,
            "--corpus" => a.corpus = Some(PathBuf::from(val()?)),
            "--arms" => a.arms = val()?,
            "--limit" => a.limit = num(val()?)?,
            "--chunk" => a.chunk = num(val()?)?,
            "--context" => a.context = num(val()?)?,
            "--device" => a.device = num(val()?)?,
            "--out" => a.out = Some(PathBuf::from(val()?)),
            "--numeric-class" => a.numeric_class = true,
            "--inject-failure" => a.inject_failure = Some(num(val()?)? as u32),
            "--contract-check" => a.contract_check = true,
            "--attn-probe" => {
                for v in val()?.split(',') {
                    a.attn_probe.push(num(v.into())? as u32);
                }
            }
            "--chunkings" => {
                let v = val()?;
                let (x, y) = v.split_once(',').ok_or("--chunkings A,B")?;
                a.chunkings = Some((num(x.into())?, num(y.into())?));
            }
            "--red" => {
                let v = val()?;
                match v.as_str() {
                    "dequant-bf16" => a.red.dequant_bf16 = true,
                    "conv-state" => a.red.conv_state_dropped = true,
                    "ties-high" => a.red.ties_high = true,
                    "commit-on-error" => a.red.commit_on_error = true,
                    _ => return Err(format!("unknown red arm {v}")),
                }
                a.red_names.push(v);
            }
            _ => return Err(format!("unknown argument {arg}")),
        }
    }
    if a.class != "declared" && a.class != "float" {
        return Err("--class is declared or float".into());
    }
    if a.red.commit_on_error && a.inject_failure.is_none() {
        return Err("--red commit-on-error is an arm of --inject-failure".into());
    }
    Ok(a)
}

fn compile(
    a: &Args,
) -> Result<
    (
        ModelPlan,
        &'static memra_gguf::model_packs::ModelPack,
        ModelConfig,
    ),
    String,
> {
    let read =
        |name: &str| std::fs::read_to_string(a.ckpt.join(name)).map_err(|e| format!("{name}: {e}"));
    let mut config = ModelConfig::from_hf(&HfConfig::try_parse(&read("config.json")?)?);
    config.attach_quant_groups(&read("hf_quant_config.json")?)?;
    // the pack is the one whose config contract matches, never a name
    let pack = memra_gguf::model_packs::for_config(&config).ok_or("no pack matches this config")?;
    let default = pack
        .compile_plan(&config)
        .map_err(|e| format!("plan: {e:?}"))?;
    let program = default.inkling2.as_ref().ok_or("not an inkling2 plan")?;
    let plan = if a.class == "float" {
        let modes = vec![ExpertActivationMode::Float; program.numerics.experts.len()];
        memra_gguf::model_packs::inkling2::plan_with_expert_activation(&config, &modes)
            .map_err(|e| format!("plan: {e:?}"))?
    } else {
        default
    };
    Ok((plan, pack, config))
}

fn clocks() -> String {
    std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu",
            "--format=csv,noheader",
        ])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map_or("unavailable".into(), |s| s.trim().to_string())
}

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg").map_or("unavailable".into(), |s| s.trim().to_string())
}

// ---- oracle tape ---------------------------------------------------------------------------------

struct Tensor<'a> {
    bytes: &'a [u8],
    shape: Vec<usize>,
}

impl Tensor<'_> {
    fn f32s(&self, at: usize, n: usize) -> Vec<f32> {
        self.bytes[at * 4..(at + n) * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
    fn i32s(&self, at: usize, n: usize) -> Vec<i32> {
        self.bytes[at * 4..(at + n) * 4]
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

fn tensor<'a>(shard: &'a StShard, name: &str, dtype: &str) -> Result<Tensor<'a>, String> {
    let (info, bytes) = shard
        .raw(name)
        .ok_or_else(|| format!("the tape has no tensor {name}"))?;
    if info.dtype != dtype {
        return Err(format!("tape tensor {name} is {}, not {dtype}", info.dtype));
    }
    Ok(Tensor {
        bytes,
        shape: info.shape.iter().map(|v| *v as usize).collect(),
    })
}

fn i64s(shard: &StShard, name: &str) -> Result<Vec<i64>, String> {
    Ok(tensor(shard, name, "I64")?
        .bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().expect("8 bytes")))
        .collect())
}

struct Tape {
    all_ids: Vec<u32>,
    prompt: usize,
    greedy: Vec<u32>,
    resid_attn: StShard,
    resid_ffn: StShard,
    router: StShard,
    logits: StShard,
    moe_in: Option<StShard>,
    moe_out: Option<StShard>,
    /// position -> kept row, for the residual files and for the logits file
    resid_rows: BTreeMap<usize, usize>,
    logit_rows: BTreeMap<usize, usize>,
    arm: String,
}

impl Tape {
    fn open(dir: &Path) -> Result<Self, String> {
        let shard = |name: &str| {
            StShard::open(dir.join(name)).map_err(|e| format!("{}/{name}: {e}", dir.display()))
        };
        let tokens = shard("tokens.safetensors")?;
        let ids = |name: &str| -> Result<Vec<u32>, String> {
            i64s(&tokens, name)?
                .into_iter()
                .map(|v| u32::try_from(v).map_err(|_| format!("tape id {v}")))
                .collect()
        };
        let all_ids = ids("all_ids")?;
        let prompt = ids("input_ids")?.len();
        let greedy = ids("greedy_tokens")?;
        let resid_attn = shard("resid_attn.safetensors")?;
        let logits = shard("logits_post.safetensors")?;
        let rows = |s: &StShard| -> Result<BTreeMap<usize, usize>, String> {
            Ok(i64s(s, "positions")?
                .into_iter()
                .enumerate()
                .map(|(row, pos)| (pos as usize, row))
                .collect())
        };
        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("meta.json"))
                .map_err(|e| format!("meta.json: {e}"))?,
        )
        .map_err(|e| format!("meta.json: {e}"))?;
        let optional = |name: &str| {
            let path = dir.join(name);
            path.exists()
                .then(|| StShard::open(path))
                .transpose()
                .map_err(|e| format!("{name}: {e}"))
        };
        Ok(Self {
            resid_rows: rows(&resid_attn)?,
            logit_rows: rows(&logits)?,
            all_ids,
            prompt,
            greedy,
            resid_attn,
            resid_ffn: shard("resid_ffn.safetensors")?,
            router: shard("router.safetensors")?,
            logits,
            moe_in: optional("moe_in.safetensors")?,
            moe_out: optional("moe_out.safetensors")?,
            arm: meta["arm"].as_str().unwrap_or("unknown").to_string(),
        })
    }
}

#[derive(Default, Clone)]
struct Dist {
    diff2: f64,
    ref2: f64,
    max_abs: f64,
    /// worst relative L2 of one row
    max_row_rel: f64,
    rows: usize,
}

impl Dist {
    fn add(&mut self, got: &[f32], want: &[f32]) {
        let (mut d2, mut r2) = (0f64, 0f64);
        for (g, w) in got.iter().zip(want) {
            let d = f64::from(*g) - f64::from(*w);
            d2 += d * d;
            r2 += f64::from(*w) * f64::from(*w);
            self.max_abs = self.max_abs.max(d.abs());
        }
        self.diff2 += d2;
        self.ref2 += r2;
        self.max_row_rel = self.max_row_rel.max((d2 / r2.max(1e-300)).sqrt());
        self.rows += 1;
    }
    fn rel(&self) -> f64 {
        (self.diff2 / self.ref2.max(1e-300)).sqrt()
    }
}

struct Compare<'a> {
    tape: &'a Tape,
    hidden: usize,
    top_k: usize,
    /// position of row 0 of the current call
    base: usize,
    force_router: bool,
    force_moe_in: bool,
    attn: Vec<Dist>,
    ffn: Vec<Dist>,
    moe_out: Vec<Dist>,
    route_rows: Vec<(usize, usize)>,
    min_margin: f64,
    error: Option<String>,
    /// FNV-1a 64 of the bits of every tap row a slot emitted, in call order: two runs of the
    /// same program on the same tokens and chunking hold equal digests
    digest_attn: Vec<u64>,
    digest_ffn: Vec<u64>,
}

fn fnv_fold(mut h: u64, data: &[f32]) -> u64 {
    for v in data {
        for b in v.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}
const FNV0: u64 = 0xcbf2_9ce4_8422_2325;

impl Compare<'_> {
    fn rows_of(&mut self, shard: &StShard, name: &str, slot: u32, data: &[f32], into: usize) {
        let result = (|| -> Result<(), String> {
            let t = tensor(shard, name, "F32")?;
            let (kept, h) = (t.shape[1], t.shape[2]);
            for (row, got) in data.chunks_exact(self.hidden).enumerate() {
                let Some(&tr) = self.tape.resid_rows.get(&(self.base + row)) else {
                    continue;
                };
                let want = t.f32s((slot as usize * kept + tr) * h, h);
                match into {
                    0 => self.attn[slot as usize].add(got, &want),
                    1 => self.ffn[slot as usize].add(got, &want),
                    _ => self.moe_out[slot as usize].add(got, &want),
                }
            }
            Ok(())
        })();
        if let Err(e) = result {
            self.error.get_or_insert(e);
        }
    }
}

impl Observer for Compare<'_> {
    fn wants(&self, tap: Tap) -> bool {
        match tap {
            Tap::ResidAttn | Tap::ResidFfn | Tap::RouteMargin => true,
            Tap::MoeOut => self.tape.moe_out.is_some(),
            _ => false,
        }
    }
    fn floats(&mut self, slot: u32, tap: Tap, data: &[f32]) {
        let tape = self.tape;
        match tap {
            Tap::ResidAttn => {
                self.digest_attn[slot as usize] = fnv_fold(self.digest_attn[slot as usize], data)
            }
            Tap::ResidFfn => {
                self.digest_ffn[slot as usize] = fnv_fold(self.digest_ffn[slot as usize], data)
            }
            _ => {}
        }
        match tap {
            Tap::ResidAttn => self.rows_of(&tape.resid_attn, "resid_attn", slot, data, 0),
            Tap::ResidFfn => self.rows_of(&tape.resid_ffn, "resid_ffn", slot, data, 1),
            Tap::MoeOut => {
                if let Some(shard) = &tape.moe_out {
                    self.rows_of(shard, "moe_out", slot, data, 2);
                }
            }
            Tap::RouteMargin => {
                for m in data {
                    self.min_margin = self.min_margin.min(f64::from(*m));
                }
            }
            _ => {}
        }
    }
    fn picks(&mut self, slot: u32, ids: &[u32]) {
        let Ok(t) = tensor(&self.tape.router, "route_ids", "I32") else {
            self.error.get_or_insert("the tape has no route_ids".into());
            return;
        };
        let (positions, k) = (t.shape[1], t.shape[2]);
        for (row, got) in ids.chunks_exact(self.top_k).enumerate() {
            let pos = self.base + row;
            if pos >= positions {
                continue;
            }
            let want = t.i32s((slot as usize * positions + pos) * k, k);
            let mut mine: Vec<i32> = got.iter().map(|e| *e as i32).collect();
            mine.sort_unstable();
            let entry = &mut self.route_rows[slot as usize];
            entry.1 += 1;
            if mine == want {
                entry.0 += 1;
            }
        }
    }
    fn forced(&mut self, slot: u32, rows: usize) -> Option<(Vec<u32>, Vec<f32>)> {
        if !self.force_router {
            return None;
        }
        let ids = tensor(&self.tape.router, "route_ids", "I32").ok()?;
        let wts = tensor(&self.tape.router, "route_weights", "F32").ok()?;
        let (positions, k) = (ids.shape[1], ids.shape[2]);
        let at = (slot as usize * positions + self.base) * k;
        Some((
            ids.i32s(at, rows * k)
                .into_iter()
                .map(|e| e as u32)
                .collect(),
            wts.f32s(at, rows * k),
        ))
    }
    fn forced_moe_in(&mut self, slot: u32, rows: usize) -> Option<Vec<f32>> {
        if !self.force_moe_in {
            return None;
        }
        let t = tensor(self.tape.moe_in.as_ref()?, "moe_in", "F32").ok()?;
        let (kept, h) = (t.shape[1], t.shape[2]);
        let mut out = Vec::with_capacity(rows * h);
        for row in 0..rows {
            let tr = *self.tape.resid_rows.get(&(self.base + row))?;
            out.extend(t.f32s((slot as usize * kept + tr) * h, h));
        }
        Some(out)
    }
}

// ---- attention probe -------------------------------------------------------------------------

#[derive(Default)]
struct AttnProbe {
    slots: Vec<u32>,
    taps: BTreeMap<(u32, u8), Vec<f32>>,
}

fn tap_id(tap: Tap) -> u8 {
    match tap {
        Tap::AttnQ => 0,
        Tap::AttnRel => 1,
        Tap::AttnSink => 2,
        Tap::AttnTable => 3,
        Tap::AttnKeys => 4,
        Tap::AttnValues => 5,
        Tap::AttnOut => 6,
        _ => 255,
    }
}

impl Observer for AttnProbe {
    fn wants(&self, _tap: Tap) -> bool {
        false
    }
    fn floats(&mut self, slot: u32, tap: Tap, data: &[f32]) {
        self.taps.insert((slot, tap_id(tap)), data.to_vec());
    }
    fn picks(&mut self, _slot: u32, _ids: &[u32]) {}
    fn forced(&mut self, _slot: u32, _rows: usize) -> Option<(Vec<u32>, Vec<f32>)> {
        None
    }
    fn attn_probe(&self, slot: u32) -> bool {
        self.slots.contains(&slot)
    }
}

/// One query row in f32 the way an f32 eager program computes it: f32 dot products, f32 softmax
/// with the max subtracted and the sink in the denominator, f32 value sum (keys oldest first).
#[allow(clippy::too_many_arguments)]
fn attend_f32(
    q: &[f32],
    bias: &[f64],
    sink: f32,
    keys: &[f32],
    values: &[f32],
    nkeys: usize,
    pos: usize,
    kvw: usize,
    kvh: usize,
    dim: usize,
    extent: usize,
    mask_beyond: bool,
) -> Vec<f32> {
    let scale = 1.0f32 / (dim as f32).sqrt();
    let mut s = vec![f32::NEG_INFINITY; nkeys];
    for (d, sd) in s.iter_mut().enumerate() {
        if d >= extent && mask_beyond {
            continue;
        }
        let k = &keys[(pos - d) * kvw + kvh * dim..][..dim];
        let dot: f32 = q.iter().zip(k).map(|(a, b)| a * b).sum();
        *sd = dot * scale + if d < extent { bias[d] as f32 } else { 0.0 };
    }
    let top = s.iter().copied().fold(sink, f32::max);
    let e: Vec<f32> = s.iter().map(|v| (v - top).exp()).collect();
    let den: f32 = e.iter().sum::<f32>() + (sink - top).exp();
    let mut out = vec![0f32; dim];
    for d in (0..nkeys).rev() {
        let v = &values[(pos - d) * kvw + kvh * dim..][..dim];
        for (o, x) in out.iter_mut().zip(v) {
            *o += e[d] / den * x;
        }
    }
    out
}

fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let (mut n, mut d) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        n += (x - y).powi(2);
        d += y.powi(2);
    }
    (n / d.max(1e-300)).sqrt()
}

/// The tape's prompt through the route, then for each probed slot the last `rows` query rows of
/// the last call recomputed on the host from the route's own inputs and caches: in f64 (the
/// reference attention) and in f32 (an f32 eager program). Returns the receipt section.
fn attn_probe(
    route: &mut Route,
    plan: &ModelPlan,
    tokens: &[u32],
    slots: &[u32],
    rows: usize,
) -> Result<serde_json::Value, String> {
    use memra_reference::inkling2::ops;
    route.reset()?;
    let last = tokens.len() - ((tokens.len() - 1) % route.chunk + 1);
    if last > 0 {
        route.call(&tokens[..last], Head::None, None)?;
    }
    let mut probe = AttnProbe {
        slots: slots.to_vec(),
        ..AttnProbe::default()
    };
    route.forward(&tokens[last..], Head::None, Some(&mut probe))?;
    let call_rows = tokens.len() - last;
    let mut out = serde_json::Map::new();
    for &slot in slots {
        let layer = &plan.layers[slot as usize];
        let memra_gguf::model_plan::AttentionPlan::RelativeBias(a) = &layer.attention else {
            return Err(format!("slot {slot} has no relative-bias attention"));
        };
        let (heads, kvh, dim, rank, extent) = (
            a.query_heads as usize,
            a.kv_heads as usize,
            a.head_dim as usize,
            a.relative.rank as usize,
            a.relative.extent as usize,
        );
        let (qw, kvw) = (heads * dim, kvh * dim);
        let mask = a.relative.beyond_extent == memra_gguf::model_plan::BeyondExtent::Masked;
        let get = |t: Tap| {
            probe
                .taps
                .get(&(slot, tap_id(t)))
                .ok_or(format!("slot {slot}: no probe tap"))
        };
        let (q, rel, sink, table, keys, values, att) = (
            get(Tap::AttnQ)?,
            get(Tap::AttnRel)?,
            get(Tap::AttnSink)?,
            get(Tap::AttnTable)?,
            get(Tap::AttnKeys)?,
            get(Tap::AttnValues)?,
            get(Tap::AttnOut)?,
        );
        let cap = keys.len() / kvw;
        let shape = ops::AttentionShape {
            query_heads: heads,
            kv_heads: kvh,
            head_dim: dim,
            window: a.window.map(|w| w as usize),
            extent,
            mask_beyond_extent: mask,
        };
        let mut rows_out = Vec::new();
        for r in call_rows.saturating_sub(rows)..call_rows {
            let pos = last + r;
            if a.window.is_some() || cap <= pos {
                return Err(format!(
                    "slot {slot}: the probe reads a full-attention cache"
                ));
            }
            let nkeys = pos + 1;
            let wide = |x: &[f32]| x.iter().map(|v| f64::from(*v)).collect::<Vec<f64>>();
            let bias = ops::relative_bias(
                &wide(&rel[r * heads * rank..][..heads * rank]),
                table,
                rank,
                extent,
            );
            let positions: Vec<u64> = (0..nkeys as u64).collect();
            let want = ops::attend(
                &shape,
                &wide(&q[r * qw..][..qw]),
                &bias,
                &wide(&sink[r * heads..][..heads]),
                pos as u64,
                ops::CachedRows {
                    positions: &positions,
                    keys: &wide(&keys[..nkeys * kvw]),
                    values: &wide(&values[..nkeys * kvw]),
                },
            );
            let gpu = wide(&att[r * qw..][..qw]);
            let mut f32_rows = Vec::with_capacity(qw);
            for h in 0..heads {
                f32_rows.extend(attend_f32(
                    &q[r * qw + h * dim..][..dim],
                    &bias[h * extent..][..extent],
                    sink[r * heads + h],
                    keys,
                    values,
                    nkeys,
                    pos,
                    kvw,
                    h / (heads / kvh),
                    dim,
                    extent,
                    mask,
                ));
            }
            let f32w = wide(&f32_rows);
            rows_out.push(serde_json::json!({
                "position": pos,
                "keys": nkeys,
                "gpu_vs_f64_rel_l2": rel_l2(&gpu, &want),
                "f32_eager_vs_f64_rel_l2": rel_l2(&f32w, &want),
                "gpu_vs_f32_eager_rel_l2": rel_l2(&gpu, &f32w),
            }));
            eprintln!(
                "ink2-gpu: attn probe slot {slot} position {pos} ({nkeys} keys): gpu vs f64 {:.3e}, f32 eager vs f64 {:.3e}",
                rel_l2(&gpu, &want),
                rel_l2(&f32w, &want)
            );
        }
        out.insert(slot.to_string(), serde_json::Value::Array(rows_out));
    }
    Ok(serde_json::Value::Object(out))
}

/// --corpus: teacher-forced runs of every document of a corpus (lane F's rawdoc / chatwrap JSON:
/// ids, sampled) under the reference precision (all f64) and each arm. Per position: KL(ref || arm)
/// over the full vocabulary in f64, the NLL of the next token under both, top-1 agreement.
fn corpus_run(
    route: &mut Route,
    path: &Path,
    arms: &str,
    out: Option<&PathBuf>,
) -> Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let docs: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("corpus: {e}"))?;
    let docs = docs.as_array().ok_or("corpus: a list of documents")?;
    let specs: Vec<String> = arms
        .split(';')
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();
    let precs: Vec<Prec> = specs
        .iter()
        .map(|s| Prec::parse(s))
        .collect::<Result<_, _>>()?;
    let threads = std::thread::available_parallelism()
        .map_or(8, |n| n.get())
        .min(16);
    // per row: log-softmax in f64, stored as f32 deltas from the row's max is not needed: keep f64 lse
    let lse_rows = |logits: &[f32], keep: usize| -> Vec<f64> {
        let rows = logits.len() / keep;
        let mut out = vec![0f64; rows];
        std::thread::scope(|sc| {
            for (ci, chunk) in out.chunks_mut(rows.div_ceil(threads).max(1)).enumerate() {
                let base = ci * rows.div_ceil(threads).max(1);
                sc.spawn(move || {
                    for (i, o) in chunk.iter_mut().enumerate() {
                        let x = &logits[(base + i) * keep..(base + i + 1) * keep];
                        let top = x
                            .iter()
                            .fold(f64::NEG_INFINITY, |a, v| a.max(f64::from(*v)));
                        *o = top
                            + x.iter()
                                .map(|v| (f64::from(*v) - top).exp())
                                .sum::<f64>()
                                .ln();
                    }
                });
            }
        });
        out
    };
    let mut per_arm: Vec<BTreeMap<&str, Vec<f64>>> = vec![BTreeMap::new(); precs.len()];
    let t0 = Instant::now();
    for (di, d) in docs.iter().enumerate() {
        let ids: Vec<u32> = d["ids"]
            .as_array()
            .ok_or("a document without ids")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as u32)
            .collect();
        let sampled: Vec<u8> = d["sampled"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as u8).collect())
            .unwrap_or_else(|| vec![0; ids.len()]);
        let n = ids.len();
        route.dev.prec.set(Prec::default());
        route.reset()?;
        let rl = route.call(&ids, Head::All, None)?.logits;
        let keep = rl.len() / n;
        let rz = lse_rows(&rl, keep);
        for (ai, p) in precs.iter().enumerate() {
            route.dev.prec.set(*p);
            route.reset()?;
            let al = route.call(&ids, Head::All, None)?.logits;
            let az = lse_rows(&al, keep);
            let mut kl = vec![0f64; n];
            std::thread::scope(|sc| {
                let per = n.div_ceil(threads).max(1);
                for (ci, chunk) in kl.chunks_mut(per).enumerate() {
                    let (rl, al, rz, az) = (&rl, &al, &rz, &az);
                    sc.spawn(move || {
                        for (i, o) in chunk.iter_mut().enumerate() {
                            let t = ci * per + i;
                            let (x, y) =
                                (&rl[t * keep..(t + 1) * keep], &al[t * keep..(t + 1) * keep]);
                            *o = x
                                .iter()
                                .zip(y)
                                .map(|(a, b)| {
                                    let lp = f64::from(*a) - rz[t];
                                    lp.exp() * (lp - (f64::from(*b) - az[t]))
                                })
                                .sum();
                        }
                    });
                }
            });
            let m = &mut per_arm[ai];
            for t in 0..n {
                let has = t + 1 < n;
                let lab = if has { ids[t + 1] as usize } else { 0 };
                let (rr, aa) = (&rl[t * keep..(t + 1) * keep], &al[t * keep..(t + 1) * keep]);
                m.entry("seq").or_default().push(di as f64);
                m.entry("kl").or_default().push(kl[t]);
                m.entry("has_label")
                    .or_default()
                    .push(f64::from(u8::from(has && lab < keep)));
                m.entry("sampled").or_default().push(f64::from(if has {
                    sampled[t + 1]
                } else {
                    0
                }));
                m.entry("nll_ref").or_default().push(if has && lab < keep {
                    rz[t] - f64::from(rr[lab])
                } else {
                    0.0
                });
                m.entry("nll_arm").or_default().push(if has && lab < keep {
                    az[t] - f64::from(aa[lab])
                } else {
                    0.0
                });
                m.entry("top1")
                    .or_default()
                    .push(f64::from(u8::from(argmax(rr) == argmax(aa))));
            }
        }
        eprintln!(
            "ink2-gpu corpus: doc {}/{} ({n} tokens) done at {:.0}s",
            di + 1,
            docs.len(),
            t0.elapsed().as_secs_f64()
        );
    }
    let arms_json: Vec<serde_json::Value> = specs
        .iter()
        .zip(&per_arm)
        .map(|(s, m)| serde_json::json!({"spec": s, "arrays": m}))
        .collect();
    let receipt = serde_json::json!({"corpus": path.display().to_string(), "docs": docs.len(), "arms": arms_json});
    let text = serde_json::to_string(&receipt).map_err(|e| e.to_string())?;
    match out {
        Some(p) => std::fs::write(p, text).map_err(|e| format!("{}: {e}", p.display()))?,
        None => println!("{text}"),
    }
    Ok(())
}

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0;
    for (i, v) in row.iter().enumerate() {
        if v.total_cmp(&row[best]).is_gt() {
            best = i;
        }
    }
    best as u32
}

/// KL(p || q) in nats of two logit rows.
fn kl(p: &[f32], q: &[f32]) -> f64 {
    let lse = |x: &[f32]| {
        let top = x
            .iter()
            .fold(f64::NEG_INFINITY, |a, v| a.max(f64::from(*v)));
        top + x
            .iter()
            .map(|v| (f64::from(*v) - top).exp())
            .sum::<f64>()
            .ln()
    };
    let (zp, zq) = (lse(p), lse(q));
    p.iter()
        .zip(q)
        .map(|(a, b)| {
            let lp = f64::from(*a) - zp;
            lp.exp() * (lp - (f64::from(*b) - zq))
        })
        .sum()
}

#[derive(Default)]
struct LogitStats {
    positions: usize,
    max_abs: f64,
    sum_abs: f64,
    kl_sum: f64,
    kl_max: f64,
    argmax_equal: usize,
}

fn fnv(bytes: impl Iterator<Item = u8>) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// Digests of every slot's output rows, for the chunking gate.
#[derive(Default)]
struct Digest {
    hidden: usize,
    rows: BTreeMap<u32, Vec<u64>>,
}

impl Observer for Digest {
    fn wants(&self, tap: Tap) -> bool {
        tap == Tap::ResidFfn
    }
    fn floats(&mut self, slot: u32, _tap: Tap, data: &[f32]) {
        let list = self.rows.entry(slot).or_default();
        for row in data.chunks_exact(self.hidden) {
            list.push(fnv(row.iter().flat_map(|v| v.to_le_bytes())));
        }
    }
    fn picks(&mut self, slot: u32, ids: &[u32]) {
        self.rows
            .entry(slot + 1_000_000)
            .or_default()
            .push(fnv(ids.iter().flat_map(|v| v.to_le_bytes())));
    }
    fn forced(&mut self, _slot: u32, _rows: usize) -> Option<(Vec<u32>, Vec<f32>)> {
        None
    }
}

fn run() -> Result<(), String> {
    let a = parse()?;
    let (plan, pack, config) = compile(&a)?;
    let class = numeric_class(&plan)?;
    if a.numeric_class {
        println!("{class}");
        return Ok(());
    }
    if a.contract_check {
        let ck = Ckpt::open(&a.ckpt, pack, &config, &plan)?;
        // CPU only: the binding of the real census passed above; now its red arms
        let (lines, fails) = ck.contract_red_arms(&config)?;
        for line in &lines {
            println!("{line}");
        }
        println!(
            "ink2-gpu: tensor contract bound over {} tensors; red arms {}",
            ck.binding.bound.tensors.len(),
            if fails == 0 {
                "all refused"
            } else {
                "NOT all refused"
            }
        );
        return if fails == 0 {
            Ok(())
        } else {
            Err(format!("{fails} contract red arm(s) accepted"))
        };
    }
    let tape = match &a.tape {
        Some(dir) => Some(Tape::open(dir)?),
        None => None,
    };
    let tape = tape.as_ref().ok_or("a mode needs --tape DIR")?;
    let need = tape
        .all_ids
        .len()
        .max(tape.prompt.max(a.prompt_len) + a.gen_tokens)
        + 8;
    let context = if a.context > 0 {
        a.context
    } else {
        need.next_power_of_two().max(2048)
    };
    let ck = Ckpt::open(&a.ckpt, pack, &config, &plan)?;
    let t_load = Instant::now();
    let dev = Dev::new(a.device)?;
    let mut route = Route::load(dev, &ck, &plan, context, a.chunk)?;
    route.set_red(a.red)?;
    route.dev.prec.set(Prec::parse(&a.prec)?);
    if let Some(cp) = &a.corpus {
        return corpus_run(&mut route, cp, &a.arms, a.out.as_ref());
    }
    eprintln!("ink2-gpu: {}", route.account.describe());
    let load_s = t_load.elapsed().as_secs_f64();
    eprintln!(
        "ink2-gpu: class {class}; resident weights {:.3} GiB, with state {:.3} GiB, loaded in {load_s:.1}s; context {context}, chunk {}; auxiliary files {:?}",
        route.weight_bytes as f64 / (1u64 << 30) as f64,
        route.dev.allocated.get() as f64 / (1u64 << 30) as f64,
        a.chunk,
        ck.auxiliary
            .iter()
            .map(|(p, m)| format!("{p}* ({} tensors)", m.n_tensors()))
            .collect::<Vec<_>>(),
    );
    let program = plan.inkling2.as_ref().ok_or("not an inkling2 plan")?;
    let hidden = plan.hidden_size as usize;
    let nslots = plan.layers.len();
    let top_k = match &plan.layers[0].mlp {
        memra_gguf::model_plan::MlpPlan::Moe(m) => m.experts_per_token as usize,
        _ => return Err("not an MoE plan".into()),
    };
    let mut receipt = serde_json::json!({
        "runner": "ink2-gpu",
        "numeric_class": class,
        // the experts' activation class from the plan; outside the experts this route runs an f32
        // stream in both classes (numeric_class names it)
        "expert_activation_class": format!("{:?}", program.numerics.class),
        "stream_outside_experts": "f32",
        "memory": {
            "device_total": route.account.device_total,
            "driver_free_at_open": route.account.driver_free_at_open,
            "weights": route.account.weights,
            "per_position": route.account.per_position,
            "kv_full": route.account.kv_full,
            "kv_rings": route.account.kv_rings,
            "conv": route.account.conv,
            "work": route.account.work,
            "declared_work": route.account.declared_work,
            "driver_free_before_kv": route.account.driver_free_before_kv,
            "driver_free_after_load": route.account.driver_free_after_load,
            "max_context": route.account.max_context,
        },
        "tape": a.tape.as_ref().map(|p| p.display().to_string()),
        "tape_arm": tape.arm,
        "red_arms": a.red_names,
        "force_router": a.force_router,
        "force_moe_in": a.force_moe_in,
        "chunk": a.chunk,
        "context": context,
        "weight_gib": route.weight_bytes as f64 / (1u64 << 30) as f64,
        "resident_gib": route.dev.allocated.get() as f64 / (1u64 << 30) as f64,
        "load_s": load_s,
        "clocks_at_start": clocks(),
        "loadavg_at_start": loadavg(),
    });

    if a.score {
        let vocab = route.vocab();
        // A forced MoE input exists only at the tape's kept positions; a chunk holding any other
        // position would run unforced and score the drift of the whole stream instead of the MoE.
        if a.force_moe_in {
            let missing = (0..tape.all_ids.len())
                .filter(|p| !tape.resid_rows.contains_key(p))
                .count();
            if tape.moe_in.is_none() || missing > 0 {
                return Err(format!(
                    "--force-moe-in needs the tape's MoE input at every position: {} has {} of {} positions{}",
                    a.tape
                        .as_ref()
                        .map_or(String::new(), |p| p.display().to_string()),
                    tape.all_ids.len() - missing,
                    tape.all_ids.len(),
                    if tape.moe_in.is_none() {
                        " and no moe_in file"
                    } else {
                        ""
                    }
                ));
            }
        }
        let mut cmp = Compare {
            tape,
            hidden,
            top_k,
            base: 0,
            force_router: a.force_router,
            force_moe_in: a.force_moe_in,
            attn: vec![Dist::default(); nslots],
            ffn: vec![Dist::default(); nslots],
            moe_out: vec![Dist::default(); nslots],
            route_rows: vec![(0, 0); nslots],
            min_margin: f64::INFINITY,
            error: None,
            digest_attn: vec![FNV0; nslots],
            digest_ffn: vec![FNV0; nslots],
        };
        let mut digest_logits = FNV0;
        let lt = tensor(&tape.logits, "logits_post", "F32")?;
        if lt.shape[1] != vocab {
            return Err(format!(
                "the tape holds {} logits a row, the route {vocab}",
                lt.shape[1]
            ));
        }
        let mut stats = LogitStats::default();
        let mut greedy_equal = 0usize;
        let mut greedy_first_miss = None;
        let total = tape.all_ids.len();
        let mut calls: Vec<(usize, usize)> = (0..tape.prompt)
            .step_by(a.chunk)
            .map(|s| (s, (s + a.chunk).min(tape.prompt)))
            .collect();
        calls.extend((tape.prompt..total).map(|p| (p, p + 1)));
        let t0 = Instant::now();
        for (s0, s1) in calls {
            cmp.base = s0;
            let out = route.forward(&tape.all_ids[s0..s1], Head::All, Some(&mut cmp))?;
            digest_logits = fnv_fold(digest_logits, &out.logits);
            for (row, got) in out.logits.chunks_exact(vocab).enumerate() {
                let pos = s0 + row;
                if let Some(&tr) = tape.logit_rows.get(&pos) {
                    let want = lt.f32s(tr * vocab, vocab);
                    stats.positions += 1;
                    for (g, w) in got.iter().zip(&want) {
                        let d = (f64::from(*g) - f64::from(*w)).abs();
                        stats.max_abs = stats.max_abs.max(d);
                        stats.sum_abs += d;
                    }
                    let k = kl(&want, got);
                    stats.kl_sum += k;
                    stats.kl_max = stats.kl_max.max(k);
                    if argmax(got) == argmax(&want) {
                        stats.argmax_equal += 1;
                    }
                }
                if pos + 1 >= tape.prompt {
                    let index = pos + 1 - tape.prompt;
                    if let Some(want) = tape.greedy.get(index) {
                        if argmax(got) == *want {
                            greedy_equal += 1;
                        } else if greedy_first_miss.is_none() {
                            greedy_first_miss = Some(index);
                        }
                    }
                }
            }
            if let Some(e) = cmp.error.take() {
                return Err(e);
            }
        }
        let rel = |d: &[Dist]| {
            d.iter()
                .map(|x| (x.rel() * 1e9).round() / 1e9)
                .collect::<Vec<_>>()
        };
        let worst = |d: &[Dist]| d.iter().fold(0f64, |m, x| m.max(x.rel()));
        let (eq, all) = cmp
            .route_rows
            .iter()
            .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        let hex = |d: &[u64]| d.iter().map(|v| format!("{v:016x}")).collect::<Vec<_>>();
        receipt["digest"] = serde_json::json!({
            "chunk": a.chunk,
            "resid_attn_per_slot": hex(&cmp.digest_attn),
            "resid_ffn_per_slot": hex(&cmp.digest_ffn),
            "logits": format!("{digest_logits:016x}"),
        });
        receipt["score"] = serde_json::json!({
            "positions": total,
            "seconds": t0.elapsed().as_secs_f64(),
            "resid_attn_rel_l2_per_slot": rel(&cmp.attn),
            "resid_ffn_rel_l2_per_slot": rel(&cmp.ffn),
            "resid_attn_rel_l2_worst_slot": worst(&cmp.attn),
            "resid_ffn_rel_l2_worst_slot": worst(&cmp.ffn),
            "resid_ffn_worst_row_rel_l2": cmp.ffn.iter().fold(0f64, |m, x| m.max(x.max_row_rel)),
            "moe_out_rel_l2_per_slot": if tape.moe_out.is_some() { serde_json::json!(rel(&cmp.moe_out)) } else { serde_json::Value::Null },
            "moe_out_rel_l2_worst_slot": if tape.moe_out.is_some() { serde_json::json!(worst(&cmp.moe_out)) } else { serde_json::Value::Null },
            "moe_out_worst_row_rel_l2": cmp.moe_out.iter().fold(0f64, |m, x| m.max(x.max_row_rel)),
            "route_rows_equal": eq,
            "route_rows": all,
            "route_rows_equal_frac_per_slot": cmp.route_rows.iter().map(|(e, n)| *e as f64 / (*n).max(1) as f64).collect::<Vec<_>>(),
            "own_min_route_margin": cmp.min_margin,
            "logits": {
                "positions": stats.positions,
                "max_abs": stats.max_abs,
                "mean_abs": stats.sum_abs / (stats.positions.max(1) * vocab) as f64,
                "kl_oracle_route_mean": stats.kl_sum / stats.positions.max(1) as f64,
                "kl_oracle_route_max": stats.kl_max,
                "argmax_equal": stats.argmax_equal,
            },
            "greedy_teacher_forced_equal": greedy_equal,
            "greedy_tokens": tape.greedy.len(),
            "greedy_first_miss": greedy_first_miss,
        });
        eprintln!(
            "ink2-gpu: score {} positions in {:.1}s: resid attn worst slot {:.3e}, ffn worst slot {:.3e}, router rows {eq}/{all}, logits max abs {:.3e} KL mean {:.3e}, argmax {}/{}, greedy {greedy_equal}/{}",
            total,
            t0.elapsed().as_secs_f64(),
            worst(&cmp.attn),
            worst(&cmp.ffn),
            stats.max_abs,
            stats.kl_sum / stats.positions.max(1) as f64,
            stats.argmax_equal,
            stats.positions,
            tape.greedy.len()
        );
    }

    if a.gen_tokens > 0 {
        route.reset()?;
        route.graph_off = a.eager;
        route.dev.sync()?;
        let prompt: Vec<u32> = if a.prompt_len > 0 {
            (0..a.prompt_len)
                .map(|i| tape.all_ids[i % tape.all_ids.len()])
                .collect()
        } else {
            tape.all_ids[..tape.prompt].to_vec()
        };
        let prompt = &prompt[..];
        let tp = Instant::now();
        let mut next = *route
            .call(prompt, Head::LastArgmax, None)?
            .argmax
            .last()
            .ok_or("no argmax")?;
        let prefill_s = tp.elapsed().as_secs_f64();
        let mut ids = vec![next];
        let mut steps = Vec::new();
        let mut digest = FNV0;
        if a.gen_digest {
            digest = fnv_fold(digest, &route.last_logits()?);
        }
        route.dev.reset_counters();
        route.graph_steps.set(0);
        let before = clocks();
        for _ in 1..a.gen_tokens {
            let ts = Instant::now();
            next = route.forward(&[next], Head::LastArgmax, None)?.argmax[0];
            steps.push(ts.elapsed().as_secs_f64());
            ids.push(next);
            if a.gen_digest {
                digest = fnv_fold(digest, &route.last_logits()?);
            }
        }
        let after = clocks();
        let n = steps.len().max(1) as f64;
        let mut sorted = steps.clone();
        sorted.sort_by(f64::total_cmp);
        let equal = ids
            .iter()
            .zip(&tape.greedy)
            .take_while(|(x, y)| x == y)
            .count();
        receipt["gen"] = serde_json::json!({
            "ids": ids,
            "oracle_greedy_equal_prefix": equal,
            "oracle_greedy_tokens": tape.greedy.len(),
            "prompt_tokens": prompt.len(),
            "prefill_s": prefill_s,
            "prefill_tok_s": prompt.len() as f64 / prefill_s,
            "decode_calls": steps.len(),
            "logits_digest": if a.gen_digest { serde_json::json!(format!("{digest:016x}")) } else { serde_json::Value::Null },
            "graph_steps": route.graph_steps.get(),
            "decode_step_p50_ms": sorted.get(sorted.len() / 2).map(|s| s * 1e3),
            "decode_tok_s": n / steps.iter().sum::<f64>().max(1e-12),
            "launches_per_token": route.dev.launches.get() as f64 / n,
            "device_copies_per_token": route.dev.copies.get() as f64 / n,
            "host_round_trips_per_token": route.dev.round_trips.get() as f64 / n,
            "clocks_before": before,
            "clocks_after": after,
            "loadavg": loadavg(),
        });
        eprintln!(
            "ink2-gpu: gen {} ids, {equal}/{} equal to the oracle's greedy prefix; {:.1} launches a token; {:.2} tok/s (clocks {after})",
            ids.len(),
            tape.greedy.len().min(ids.len()),
            route.dev.launches.get() as f64 / n,
            n / steps.iter().sum::<f64>().max(1e-12),
        );
        println!(
            "gen {}",
            ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
        );
    }

    if let Some((x, y)) = a.chunkings {
        let limit = if a.limit > 0 {
            a.limit.min(tape.all_ids.len())
        } else {
            tape.all_ids.len()
        };
        let tokens = &tape.all_ids[..limit];
        let mut runs = Vec::new();
        for step in [x, y] {
            if step == 0 || step > a.chunk {
                return Err(format!(
                    "a chunking of {step} rows; the route holds 1..={}",
                    a.chunk
                ));
            }
            route.reset()?;
            let mut digest = Digest {
                hidden,
                ..Digest::default()
            };
            let mut logits = Vec::new();
            let mut best = Vec::new();
            let t0 = Instant::now();
            for chunk in tokens.chunks(step) {
                let out = route.forward(chunk, Head::All, Some(&mut digest))?;
                // rows are compared through digests: a chunk's logits are too many to keep
                for row in out.logits.chunks_exact(route.vocab()) {
                    logits.push(fnv(row.iter().flat_map(|v| v.to_le_bytes())));
                }
                best.extend(out.argmax);
            }
            eprintln!(
                "ink2-gpu: {limit} rows as calls of {step}: {:.1}s",
                t0.elapsed().as_secs_f64()
            );
            runs.push((digest.rows, logits, best));
        }
        let slot_rows_differ: usize = runs[0]
            .0
            .iter()
            .filter(|(slot, _)| **slot < 1_000_000)
            .map(|(slot, rows)| {
                runs[1].0.get(slot).map_or(rows.len(), |other| {
                    rows.iter().zip(other).filter(|(p, q)| p != q).count()
                })
            })
            .sum();
        // router picks are digested per call, so they are compared through the rows they produce
        let logit_rows_differ = runs[0]
            .1
            .iter()
            .zip(&runs[1].1)
            .filter(|(p, q)| p != q)
            .count();
        let argmax_differ = runs[0]
            .2
            .iter()
            .zip(&runs[1].2)
            .filter(|(p, q)| p != q)
            .count();
        receipt["chunkings"] = serde_json::json!({
            "rows": limit,
            "a": x,
            "b": y,
            "slot_output_rows_compared": limit * nslots,
            "slot_output_rows_differ": slot_rows_differ,
            "logit_rows_differ": logit_rows_differ,
            "argmax_differ": argmax_differ,
            "bit_identical": slot_rows_differ == 0 && logit_rows_differ == 0 && argmax_differ == 0,
        });
        eprintln!(
            "ink2-gpu: chunkings {x} vs {y} over {limit} rows: slot rows differ {slot_rows_differ}/{}, logit rows differ {logit_rows_differ}, argmax differ {argmax_differ}",
            limit * nslots
        );
    }
    if !a.attn_probe.is_empty() {
        let tokens = &tape.all_ids[..tape.prompt];
        receipt["attn_probe"] = attn_probe(&mut route, &plan, tokens, &a.attn_probe, 4)?;
    }
    if let Some(fail_slot) = a.inject_failure {
        let limit = if a.limit > 0 {
            a.limit.min(tape.all_ids.len())
        } else {
            tape.all_ids.len()
        };
        let tokens = &tape.all_ids[..limit];
        let split = limit / 2;
        if split == 0 || split + 1 >= limit {
            return Err("--inject-failure needs a --limit of at least 4 rows".into());
        }
        // the same request twice: clean, and with one failed decode call before the decode
        let mut runs = Vec::new();
        for faulted in [false, true] {
            route.reset()?;
            route.forward(&tokens[..split], Head::None, None)?;
            let mut failure = None;
            if faulted {
                route.fail_after_slot = Some(fail_slot);
                match route.forward(&tokens[split..split + 1], Head::All, None) {
                    Ok(_) => return Err("the injected failure did not fire".into()),
                    Err(e) => failure = Some(e),
                }
                if route.position != split as u64 {
                    return Err(format!(
                        "a failed call moved the position to {}",
                        route.position
                    ));
                }
            }
            let mut digest = Digest {
                hidden,
                ..Digest::default()
            };
            let mut logits = Vec::new();
            for t in &tokens[split..] {
                let out = route.forward(std::slice::from_ref(t), Head::All, Some(&mut digest))?;
                logits.push(fnv(out.logits.iter().flat_map(|v| v.to_le_bytes())));
            }
            runs.push((digest.rows, logits, failure));
        }
        let rows_differ: usize = runs[0]
            .0
            .iter()
            .map(|(slot, rows)| {
                runs[1].0.get(slot).map_or(rows.len(), |other| {
                    rows.iter().zip(other).filter(|(p, q)| p != q).count()
                })
            })
            .sum();
        let logits_differ = runs[0]
            .1
            .iter()
            .zip(&runs[1].1)
            .filter(|(p, q)| p != q)
            .count();
        let compared: usize = runs[0].0.values().map(Vec::len).sum();
        receipt["atomicity"] = serde_json::json!({
            "fail_after_slot": fail_slot,
            "slot_rows_compared": compared,
            "prefix_rows": split,
            "decode_rows": limit - split,
            "failure": runs[1].2,
            "slot_rows_differ": rows_differ,
            "logit_rows_differ": logits_differ,
            "bit_identical": rows_differ == 0 && logits_differ == 0,
        });
        eprintln!(
            "ink2-gpu: one failed call after slot {fail_slot} at row {split}, then {} decode rows: slot rows differ {rows_differ}/{compared}, logit rows differ {logits_differ}",
            limit - split
        );
    }
    receipt["clocks_at_end"] = serde_json::json!(clocks());
    let text = serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())?;
    match &a.out {
        Some(path) => {
            std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))?
        }
        None => println!("{text}"),
    }
    Ok(())
}
