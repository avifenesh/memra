//! The DFly2 drafter's reference runs (memra-reference `hy_v4_flash_dfly`), writing the
//! `memra-hy4f-dfly-tape-v1` tapes the lane's oracle writes
//! (research/modelplan-onboarding-hy4flash-20261119/dfly/make_dfly_oracle.py).
//!
//! - `tiny-image OUT`: the tiny drafter's checkpoint image for the oracle: the tiny trunk
//!   (`model.safetensors`, the main request's), the drafter (`dfly.safetensors`), the drafter's
//!   tiny config, `fixture.json` and `request.json`.
//! - `tiny OUT [--anchors SPEC] [--red ARM]`: the reference trunk on the tiny request, then a
//!   draft round from every anchor.
//! - `checkpoint CKPT REQUEST TARGET OUT [--anchors SPEC] [--k K] [--red ARM] [--force-drafts TAPE]`:
//!   the drafter on the artifact's own tensors, every tensor dequantized exactly in f64, from the
//!   target inputs the f64 target oracle dumped (`--dump-dfly`) for that request.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use memra_gguf::config::{HfConfig, HyV4FlashConfig, ModelConfig};
use memra_gguf::dsv41_forward::{e2m1, e4m3};
use memra_gguf::model_packs::hy_v4_flash as pack;
use memra_gguf::safetensors::StModel;
use memra_reference::hy_v4_flash_dfly::{
    DflyGeometry, DflyRed, DflyTarget, DflyWeights, DraftRound, draft_round, tiny_rounds,
};
use memra_reference::hy_v4_flash_trunk as trunk;
use serde_json::{Value, json};

type Fail = Box<dyn std::error::Error>;

fn main() {
    if let Err(e) = run() {
        eprintln!("hy4f-dfly: {e}");
        std::process::exit(1);
    }
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn run() -> Result<(), Fail> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("tiny-image") => tiny_image(Path::new(args.get(1).ok_or("tiny-image OUT")?)),
        Some("tiny") => tiny(&args[1..]),
        Some("checkpoint") => checkpoint(&args[1..]),
        _ => Err(
            "usage: hy4f-dfly tiny-image OUT | tiny OUT [--anchors SPEC] [--red ARM] | \
                  checkpoint CKPT REQUEST TARGET OUT [--anchors SPEC] [--k K] [--red ARM] \
                  [--force-drafts TAPE]"
                .into(),
        ),
    }
}

fn red_arm(args: &[String]) -> Result<DflyRed, Fail> {
    let name = flag(args, "--red").unwrap_or("");
    Ok(DflyRed::parse(name).ok_or_else(|| format!("unknown red arm {name:?}"))?)
}

/// `all`, a comma list, or `stride:S,range:A-B,at:P` (the oracle's `--anchors`).
fn anchors(spec: &str, tokens: usize) -> Result<Vec<usize>, Fail> {
    let last = tokens - 2;
    if spec == "all" {
        return Ok((0..=last).collect());
    }
    let mut out = std::collections::BTreeSet::new();
    for part in spec.split(',') {
        if let Some(s) = part.strip_prefix("stride:") {
            out.extend((0..=last).step_by(s.parse()?));
        } else if let Some(r) = part.strip_prefix("range:") {
            let (lo, hi) = r.split_once('-').ok_or("range:A-B")?;
            let hi: usize = hi.parse()?;
            out.extend(lo.parse::<usize>()?..=hi.min(last));
        } else if let Some(p) = part.strip_prefix("at:") {
            out.insert(p.parse()?);
        } else {
            out.insert(part.parse()?);
        }
    }
    Ok(out.into_iter().filter(|p| *p <= last).collect())
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Draft every anchor and write the tape.
#[allow(clippy::too_many_arguments)]
fn write_tape(
    out: &Path,
    engine: &str,
    request: &str,
    tokens: &[u32],
    g: &DflyGeometry,
    k: usize,
    list: &[usize],
    red: DflyRed,
    forced: Option<&[Vec<u32>]>,
    mut round: impl FnMut(usize, Option<&[u32]>) -> Result<DraftRound, String>,
) -> Result<(), Fail> {
    std::fs::create_dir_all(out)?;
    let t0 = Instant::now();
    let mut logits = Vec::with_capacity(list.len() * k * g.vocab * 4);
    let mut drafts = Vec::new();
    let mut accepted = Vec::new();
    for (n, &p) in list.iter().enumerate() {
        let r = round(p, forced.map(|f| f[n].as_slice()))?;
        for v in &r.logits {
            logits.extend_from_slice(&v.to_le_bytes());
        }
        let mut m = 0;
        while m < k && p + 2 + m < tokens.len() && r.drafts[m] == tokens[p + 2 + m] {
            m += 1;
        }
        accepted.push(m);
        if n % 25 == 0 || n + 1 == list.len() {
            eprintln!(
                "[{:8.1}s] anchor {}/{} p={p} drafts={:?}",
                t0.elapsed().as_secs_f64(),
                n + 1,
                list.len(),
                r.drafts
            );
        }
        drafts.push(r.drafts);
    }
    std::fs::write(out.join("logits.f32"), &logits)?;
    let tape = json!({
        "format": "memra-hy4f-dfly-tape-v1",
        "engine": engine,
        "backend_dtype": "f64",
        "red_arm": if red == DflyRed::None { Value::Null } else { json!(format!("{red:?}")) },
        "request": request,
        "token_ids": tokens,
        "k": k,
        "vocab": g.vocab,
        "mlp_width": g.ffn,
        "mask_token": g.mask_token,
        "target_index_layers": g.target_index_layers,
        "target_kv_layers": g.target_kv_layers,
        "target_index_topk": g.target_index_topk,
        "anchors": list,
        "drafts": drafts,
        "accepted_teacher_forced": accepted,
        "logits": {"file": "logits.f32", "sha256": sha256_hex(&logits), "shape": [list.len(), k, g.vocab]},
        "forced_drafts": forced.is_some(),
        "seconds": t0.elapsed().as_secs_f64(),
    });
    std::fs::write(out.join("tape.json"), serde_json::to_string_pretty(&tape)?)?;
    let mean = accepted.iter().sum::<usize>() as f64 / accepted.len().max(1) as f64;
    println!(
        "{}",
        json!({"request": request, "anchors": list.len(), "k": k, "red": tape["red_arm"],
               "logits_sha256": tape["logits"]["sha256"], "mean_accepted_teacher_forced": mean})
    );
    Ok(())
}

// ---------------------------------------------------------------- tiny

type TinyParts = (
    trunk::DflyFixture,
    memra_reference::ReferenceFixture,
    memra_gguf::model_plan::ModelPlan,
);

fn tiny_parts() -> Result<TinyParts, Fail> {
    let plan = memra_gguf::model_packs::by_alias("hy_v4_flash")
        .ok_or("no hy_v4_flash pack")?
        .compile_tiny_plan()
        .map_err(|e| format!("{e:?}"))?;
    let dfly = trunk::dfly_fixture().map_err(|e| format!("{e:?}"))?;
    let fixture = trunk::fixture(&plan).map_err(|e| format!("{e:?}"))?;
    Ok((dfly, fixture, plan))
}

fn tiny_image(out: &Path) -> Result<(), Fail> {
    let (dfly, fixture, _) = tiny_parts()?;
    std::fs::create_dir_all(out)?;
    for (name, bytes) in &fixture.checkpoint_files {
        if name == "config.json" {
            continue;
        }
        std::fs::write(out.join(name), bytes)?;
    }
    std::fs::write(out.join("config.json"), pack::TINY_DFLY_CONFIG)?;
    std::fs::write(out.join("dfly.safetensors"), &dfly.safetensors)?;
    let request = json!({"name": "main", "token_ids": fixture.token_ids,
                         "decode_from": fixture.decode_from});
    std::fs::write(out.join("request.json"), serde_json::to_string(&request)?)?;
    println!("tiny drafter image written to {}", out.display());
    Ok(())
}

fn tiny(args: &[String]) -> Result<(), Fail> {
    let out = Path::new(args.first().ok_or("tiny OUT")?);
    let red = red_arm(args)?;
    let (_, fixture, _) = tiny_parts()?;
    let list = anchors(
        flag(args, "--anchors").unwrap_or("all"),
        fixture.token_ids.len(),
    )?;
    let (tokens, k, g, rounds) = tiny_rounds(&list, red)?;
    let mut it = rounds.into_iter();
    write_tape(
        out,
        "memra-reference-hy_v4_flash_dfly-f64",
        "main",
        &tokens,
        &g,
        k,
        &list,
        red,
        None,
        |_, _| Ok(it.next().expect("one round an anchor")),
    )
}

// ---------------------------------------------------------------- checkpoint

/// A tensor of the artifact, dequantized exactly in f64: BF16 and F32 as stored; FP8 E4M3 times
/// its 128x128 block `weight_scale_inv`; NVFP4 (ModelOpt: e2m1 pairs, element `2i` in the low
/// nibble, an E4M3 scale per 16 inputs, the tensor's F32 `weight_scale_2`).
fn dequant(st: &StModel, name: &str) -> Result<Vec<f64>, Fail> {
    let (info, bytes) = st.raw(name).ok_or_else(|| format!("no tensor {name}"))?;
    let shape: Vec<usize> = info.shape.iter().map(|v| *v as usize).collect();
    let stem = name.strip_suffix(".weight").unwrap_or(name);
    let f32s = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    Ok(match info.dtype.as_str() {
        "BF16" => bytes
            .chunks_exact(2)
            .map(|c| {
                f64::from(f32::from_bits(
                    u32::from(u16::from_le_bytes([c[0], c[1]])) << 16,
                ))
            })
            .collect(),
        "F32" => f32s(bytes).into_iter().map(f64::from).collect(),
        "F8_E4M3" => {
            let (rows, cols) = (shape[0], shape[1]);
            let (_, gb) = st
                .raw(&format!("{stem}.weight_scale_inv"))
                .ok_or_else(|| format!("{name}: no weight_scale_inv"))?;
            let grid = f32s(gb);
            let gc = cols.div_ceil(128);
            (0..rows * cols)
                .map(|i| {
                    let (r, c) = (i / cols, i % cols);
                    f64::from(e4m3(bytes[i])) * f64::from(grid[(r / 128) * gc + c / 128])
                })
                .collect()
        }
        "U8" => {
            let (rows, half) = (shape[0], shape[1]);
            let cols = half * 2;
            let (_, sb) = st
                .raw(&format!("{stem}.weight_scale"))
                .ok_or_else(|| format!("{name}: no weight_scale"))?;
            let (_, s2) = st
                .raw(&format!("{stem}.weight_scale_2"))
                .ok_or_else(|| format!("{name}: no weight_scale_2"))?;
            let s2 = f64::from(f32s(s2)[0]);
            (0..rows * cols)
                .map(|i| {
                    let (r, c) = (i / cols, i % cols);
                    let byte = bytes[r * half + c / 2];
                    let nibble = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                    f64::from(e2m1(nibble)) * f64::from(e4m3(sb[r * (cols / 16) + c / 16])) * s2
                })
                .collect()
        }
        other => return Err(format!("{name}: dtype {other}").into()),
    })
}

fn read_f64(path: &Path) -> Result<Vec<f64>, Fail> {
    Ok(std::fs::read(path)?
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn checkpoint(args: &[String]) -> Result<(), Fail> {
    let [ckpt, request, target, out, ..] = args else {
        return Err("checkpoint CKPT REQUEST TARGET OUT".into());
    };
    let red = red_arm(args)?;
    let (ckpt, target, out) = (Path::new(ckpt), Path::new(target), Path::new(out));
    let config = std::fs::read_to_string(ckpt.join("config.json"))?;
    let mc = ModelConfig::from_hf(&HfConfig::parse(&config));
    let hy: &HyV4FlashConfig = mc.hy_v4_flash.as_ref().ok_or("not a hy_v4_flash config")?;
    let st = StModel::open(ckpt)?;
    let ffn = pack::dfly_ffn_width(hy) as usize;
    let g = DflyGeometry::of(hy, ffn);
    let t0 = Instant::now();
    let w = DflyWeights::read(hy.dfly.num_hidden_layers as usize, |name| {
        dequant(&st, name).map_err(|e| e.to_string())
    })?;
    for (i, l) in w.layers.iter().enumerate() {
        if l.mlp_gate.len() != ffn * g.hidden || l.mlp_down.len() != ffn * g.hidden {
            return Err(format!("drafter layer {i}: the MLP is not {ffn} wide").into());
        }
    }
    let head_t = dequant(&st, "lm_head.weight")?;
    eprintln!(
        "drafter and head dequantized in {:.1}s",
        t0.elapsed().as_secs_f64()
    );
    let (_, emb) = st.raw("model.embed_tokens.weight").ok_or("no embedding")?;
    let h = g.hidden;
    let embed = |id: u32| -> Vec<f64> {
        emb[id as usize * h * 2..(id as usize + 1) * h * 2]
            .chunks_exact(2)
            .map(|c| {
                f64::from(f32::from_bits(
                    u32::from(u16::from_le_bytes([c[0], c[1]])) << 16,
                ))
            })
            .collect()
    };
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let vocab = g.vocab;
    let head = |row: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; vocab];
        let chunk = vocab.div_ceil(threads);
        std::thread::scope(|s| {
            for (c, part) in out.chunks_mut(chunk).enumerate() {
                let head_t = &head_t;
                s.spawn(move || {
                    for (i, o) in part.iter_mut().enumerate() {
                        let v = c * chunk + i;
                        *o = head_t[v * h..(v + 1) * h]
                            .iter()
                            .zip(row)
                            .map(|(a, b)| a * b)
                            .sum();
                    }
                });
            }
        });
        out
    };
    let req: Value = serde_json::from_str(&std::fs::read_to_string(request)?)?;
    let tokens: Vec<u32> = req["token_ids"]
        .as_array()
        .ok_or("request token_ids")?
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let name = req["name"].as_str().unwrap_or("request").to_string();
    let tj: Value = serde_json::from_str(&std::fs::read_to_string(target.join("target.json"))?)?;
    if tj["positions"].as_u64() != Some(tokens.len() as u64) {
        return Err("the target dump scored another request".into());
    }
    let hidden = read_f64(&target.join(tj["hidden"]["file"].as_str().ok_or("hidden file")?))?;
    let mut kv: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
    let mut hist: BTreeMap<u32, (Vec<i32>, Vec<i64>)> = BTreeMap::new();
    for l in tj["layers"].as_array().ok_or("layers")? {
        let layer = l["layer"].as_u64().ok_or("layer")? as u32;
        kv.insert(
            layer,
            read_f64(&target.join(l["file"].as_str().ok_or("file")?))?,
        );
        let flat: Vec<i32> = std::fs::read(target.join(l["history_file"].as_str().ok_or("hf")?))?
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let offs: Vec<i64> =
            std::fs::read(target.join(l["history_offsets_file"].as_str().ok_or("ho")?))?
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
        hist.insert(layer, (flat, offs));
    }
    let k = flag(args, "--k")
        .map(str::parse)
        .transpose()?
        .or(hy.dfly.num_speculative_tokens.map(|v| v as usize))
        .ok_or("no k")?;
    let list = anchors(flag(args, "--anchors").unwrap_or("all"), tokens.len())?;
    let forced: Option<Vec<Vec<u32>>> = match flag(args, "--force-drafts") {
        None => None,
        Some(dir) => {
            let ft: Value =
                serde_json::from_str(&std::fs::read_to_string(Path::new(dir).join("tape.json"))?)?;
            if ft["anchors"] != json!(list) || ft["k"] != json!(k) {
                return Err("the forcing tape drafted other anchors".into());
            }
            Some(serde_json::from_value(ft["drafts"].clone())?)
        }
    };
    write_tape(
        out,
        "memra-reference-hy_v4_flash_dfly-f64-checkpoint",
        &name,
        &tokens,
        &g,
        k,
        &list,
        red,
        forced.as_deref(),
        |p, forced| {
            let kvf = |l: u32| kv.get(&l).map(Vec::as_slice);
            let hf = |l: u32| {
                hist.get(&l).map(|(flat, offs)| {
                    flat[offs[p] as usize..offs[p + 1] as usize]
                        .iter()
                        .map(|v| *v as usize)
                        .collect()
                })
            };
            let target = DflyTarget {
                hidden: &hidden[p * h..(p + 1) * h],
                kv: &kvf,
                history: &hf,
            };
            draft_round(
                &g,
                &w,
                &embed,
                &head,
                &target,
                p,
                tokens[p + 1],
                k,
                forced,
                red,
            )
        },
    )
}
