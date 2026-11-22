//! ink2-mint: the `nvfp4-rig` overlay of Inkling 2 Mini (memra_gguf::model_packs::inkling2::nvfp4_rig).
//! CPU only and deterministic: the overlay's bytes do not depend on the thread count or the
//! machine, and no external engine is involved.
//!
//! usage:
//!   ink2-mint mint CKPT_DIR OUT_DIR --artifact HF_ID@REVISION [--threads N]
//!       converts the class's tensors (memra_gguf::model_packs::inkling2::nvfp4_rig::rig_tensors),
//!       writes OUT_DIR/overlay.safetensors and OUT_DIR/manifest.json (the source file's sha256,
//!       size and header hash, the hf_quant_config.json hash, every source and overlay tensor's
//!       sha256) and makes both read-only
//!   ink2-mint verify CKPT_DIR OVERLAY_DIR [--threads N]
//!       the loader's binding check, the full-file hashes of the source and the overlay, and a
//!       re-mint of every tensor compared byte for byte
//!   ink2-mint drafter CKPT_DIR OUT_FILE [--threads N]
//!       NVFP4 of every drafter projection (`mtp.safetensors`: each layer's q, k, v, o, k_ctx, v_ctx,
//!       gate, up, down; markov_w1; rnn_proj; hidden_out) and of the fold `unembed @ hidden_out`
//!       (f64 sums over the BF16 rows), with the class's quantizer: NAME (codes), NAME_scale (e4m3)
//!       and NAME_scale_2 (f32). A measurement overlay: the drafter reads the components it is told
//!       to (`ink2-spec --draft-nv`).

use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_packs::inkling2::nvfp4_rig as rig;
use memra_gguf::safetensors::StModel;
use std::path::Path;
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("ink2-mint: {e}");
        std::process::exit(1);
    }
}

fn config(dir: &Path) -> Result<(ModelConfig, String), String> {
    let read = |n: &str| std::fs::read_to_string(dir.join(n)).map_err(|e| format!("{n}: {e}"));
    let mut config = ModelConfig::from_hf(&HfConfig::try_parse(&read("config.json")?)?);
    let quant = read("hf_quant_config.json")?;
    config.attach_quant_groups(&quant)?;
    Ok((config, rig::sha256_hex(quant.as_bytes())))
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let mut artifact = None;
    let mut pos = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--threads" => {
                threads = it
                    .next()
                    .ok_or("--threads N")?
                    .parse()
                    .map_err(|e| format!("--threads: {e}"))?
            }
            "--artifact" => artifact = Some(it.next().ok_or("--artifact HF_ID@REVISION")?.clone()),
            _ if a.starts_with("--") => return Err(format!("unknown argument {a}")),
            _ => pos.push(a.clone()),
        }
    }
    match pos.as_slice() {
        [cmd, ckpt, out] if cmd == "drafter" => drafter(Path::new(ckpt), Path::new(out), threads),
        [cmd, ckpt, out] if cmd == "mint" => mint(
            Path::new(ckpt),
            Path::new(out),
            artifact.as_deref().ok_or("mint needs --artifact HF_ID@REVISION")?,
            threads,
        ),
        [cmd, ckpt, dir] if cmd == "verify" => verify(Path::new(ckpt), Path::new(dir), threads),
        _ => Err("usage: ink2-mint mint CKPT_DIR OUT_DIR --artifact HF_ID@REV [--threads N] | verify CKPT_DIR OVERLAY_DIR".into()),
    }
}

fn drafter(ckpt: &Path, out: &Path, threads: usize) -> Result<(), String> {
    use std::collections::BTreeMap;
    let t0 = Instant::now();
    let mtp = StModel::open(&ckpt.join("mtp.safetensors"))
        .map_err(|e| format!("open mtp.safetensors: {e}"))?;
    let base = StModel::open(ckpt).map_err(|e| format!("open {}: {e}", ckpt.display()))?;
    let bf16 = |m: &StModel, name: &str| -> Result<(Vec<usize>, Vec<f32>), String> {
        let (info, bytes) = m.raw(name).ok_or_else(|| format!("no tensor {name}"))?;
        if info.dtype != "BF16" {
            return Err(format!("{name} is {}, not BF16", info.dtype));
        }
        Ok((
            info.shape.iter().map(|&d| d as usize).collect(),
            rig::bf16_values(bytes),
        ))
    };
    let mut tensors: BTreeMap<String, (&'static str, Vec<usize>, Vec<u8>)> = BTreeMap::new();
    let mut put = |name: &str, shape: &[usize], w: &[f32]| {
        let k = shape[1];
        let t = rig::tensor_scale(rig::amax(w));
        let mut codes = vec![0u8; w.len() / 2];
        let mut scales = vec![0u8; w.len() / 16];
        rig::quantize_rows(w, k, t, &mut codes, &mut scales, threads);
        let deq = rig::dequant_rows(&codes, &scales, k, t);
        let (num, den) = w.iter().zip(&deq).fold((0f64, 0f64), |(n, d), (a, b)| {
            (n + f64::from(a - b).powi(2), d + f64::from(*a).powi(2))
        });
        eprintln!(
            "ink2-mint: {name} {shape:?} rel_mse {:.3e}",
            num / den.max(1e-300)
        );
        tensors.insert(name.into(), ("U8", vec![shape[0], k / 2], codes));
        tensors.insert(
            format!("{name}_scale"),
            ("F8_E4M3", vec![shape[0], k / 16], scales),
        );
        tensors.insert(
            format!("{name}_scale_2"),
            ("F32", vec![], t.to_le_bytes().to_vec()),
        );
    };
    let mut names: Vec<String> = Vec::new();
    for l in 0.. {
        let p = format!("layers.{l}.");
        if mtp.raw(&format!("{p}self_attn.q_proj.weight")).is_none() {
            break;
        }
        for s in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "self_attn.k_proj_ctx",
            "self_attn.v_proj_ctx",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ] {
            names.push(format!("{p}{s}.weight"));
        }
    }
    for s in ["markov_w1", "rnn_proj", "hidden_out"] {
        names.push(format!("markov_head.{s}.weight"));
    }
    for n in &names {
        let (shape, w) = bf16(&mtp, n)?;
        put(n, &shape, &w);
    }
    // the fold: fold[v, j] = sum_h unembed[v, h] * hidden_out[h, j], f64 sums (the GPU fold's)
    let (us, head) = bf16(&base, "model.llm.unembed.weight")?;
    let (hs, how) = bf16(&mtp, "markov_head.hidden_out.weight")?;
    let (vocab, h, r) = (us[0], us[1], hs[1]);
    if hs[0] != h {
        return Err("hidden_out does not match the unembed's width".into());
    }
    let mut fold = vec![0f32; vocab * r];
    let per = vocab.div_ceil(threads.max(1));
    std::thread::scope(|s| {
        for (c, rows) in fold.chunks_mut(per * r).enumerate() {
            let (head, how) = (&head, &how);
            s.spawn(move || {
                for (i, row) in rows.chunks_mut(r).enumerate() {
                    let v = c * per + i;
                    let hr = &head[v * h..(v + 1) * h];
                    let mut acc = vec![0f64; r];
                    for (k, &hv) in hr.iter().enumerate() {
                        let hv = f64::from(hv);
                        for (a, &w) in acc.iter_mut().zip(&how[k * r..(k + 1) * r]) {
                            *a += hv * f64::from(w);
                        }
                    }
                    for (o, a) in row.iter_mut().zip(acc) {
                        *o = a as f32;
                    }
                }
            });
        }
    });
    put("fold", &[vocab, r], &fold);
    let mut meta = BTreeMap::new();
    meta.insert("tool".to_string(), "ink2-mint drafter".to_string());
    rig::write_safetensors(out, &tensors, &meta)?;
    eprintln!(
        "ink2-mint: drafter overlay {} ({} tensors) in {:.1}s",
        out.display(),
        tensors.len(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn mint(ckpt: &Path, out: &Path, artifact: &str, threads: usize) -> Result<(), String> {
    let t0 = Instant::now();
    let manifest = rig::mint_overlay_dir(ckpt, out, artifact, threads, &mut |line| {
        eprintln!("ink2-mint: {line}")
    })?;
    eprintln!(
        "ink2-mint: {} tensors, overlay {} bytes sha256 {}, in {:.1}s",
        manifest.tensors.len(),
        manifest.overlay_bytes,
        manifest.overlay_sha256,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn verify(ckpt: &Path, dir: &Path, threads: usize) -> Result<(), String> {
    let t0 = Instant::now();
    let (config, quant_sha) = config(ckpt)?;
    let ink = config.inkling2.as_ref().ok_or("not an inkling2 config")?;
    let base = StModel::open(ckpt).map_err(|e| format!("open {}: {e}", ckpt.display()))?;
    let bound = rig::bind_overlay(ckpt, dir, &base, ink, &quant_sha, true, threads)?;
    eprintln!(
        "ink2-mint: binding check OK in {:.1}s",
        t0.elapsed().as_secs_f64()
    );
    let m = &bound.manifest;
    let (sha, _) = rig::sha256_file(&ckpt.join(&m.source_file))?;
    if sha != m.source_sha256 {
        return Err(format!(
            "{}: sha256 {sha} is not the manifest's {}",
            m.source_file, m.source_sha256
        ));
    }
    let (sha, _) = rig::sha256_file(&dir.join(rig::OVERLAY_FILE))?;
    if sha != m.overlay_sha256 {
        return Err(format!(
            "{}: sha256 {sha} is not the manifest's {}",
            rig::OVERLAY_FILE,
            m.overlay_sha256
        ));
    }
    eprintln!("ink2-mint: full-file hashes OK");
    let mut differ = 0usize;
    for t in rig::rig_tensors(ink)? {
        let (planes, _, _) = rig::mint_tensor(&base, &t, threads)?;
        for (name, _, _, bytes) in planes {
            let (_, have) = bound
                .overlay
                .raw(&name)
                .ok_or_else(|| format!("no overlay tensor {name}"))?;
            let d = have.iter().zip(&bytes).filter(|(a, b)| a != b).count()
                + have.len().abs_diff(bytes.len());
            if d != 0 {
                eprintln!("ink2-mint: {name}: {d} bytes differ from a re-mint");
            }
            differ += d;
        }
    }
    if differ != 0 {
        return Err(format!("{differ} overlay bytes differ from a re-mint"));
    }
    eprintln!(
        "ink2-mint: re-mint byte-identical; verify OK in {:.1}s",
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
