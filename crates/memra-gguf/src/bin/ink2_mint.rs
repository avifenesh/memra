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
