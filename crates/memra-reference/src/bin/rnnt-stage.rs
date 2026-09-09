//! CPU-only stage runner for the native cache-aware FastConformer encoder.
//!
//! It reads chunk log-mel fixtures produced by the pinned NeMo capture, runs the native
//! streaming encoder over them in order, and writes one encoder file per chunk. It opens no
//! audio device, runs no decoding and touches no GPU.
use memra_gguf::model_packs::nemotron_rnnt::{
    HEBREW_GEOMETRY, MappedNemo, QUALIFIED_CONTEXT, bind,
};
use memra_reference::speech::fastconformer::FastConformerEncoder;
use memra_reference::speech::rnnt_frontend::RnntFrontend;
use std::path::{Path, PathBuf};

fn read_f32(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if !bytes.len().is_multiple_of(4) {
        return Err("f32 input byte length is not divisible by four".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 5 && args[1] == "frontend" {
        let started = std::time::Instant::now();
        let archive = MappedNemo::open(Path::new(&args[2]))?;
        let bound = bind(HEBREW_GEOMETRY, &archive.checkpoint.census)?;
        let frontend = RnntFrontend::from_bound(
            HEBREW_GEOMETRY,
            &bound,
            archive.bytes(),
            &archive.checkpoint.storages,
        )?;
        let pcm = read_f32(Path::new(&args[3]))?;
        let (mel, frames) = frontend.compute(&pcm)?;
        let bytes: Vec<u8> = mel.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(Path::new(&args[4]), bytes)?;
        println!(
            "frontend samples={} frames={frames} valid={} elapsed={:.3}",
            pcm.len(),
            frontend.valid_frames(pcm.len()),
            started.elapsed().as_secs_f64()
        );
        return Ok(());
    }
    if args.len() != 6 || args[1] != "encoder" {
        return Err(
            "usage: rnnt-stage encoder ARCHIVE.nemo ORACLE_DIR OUTPUT_DIR \
             DROP_EXTRA_PRE_ENCODED | rnnt-stage frontend ARCHIVE.nemo PCM.f32 OUT.f32"
                .into(),
        );
    }
    let oracle = PathBuf::from(&args[3]);
    let out = PathBuf::from(&args[4]);
    let drop_extra: usize = args[5].parse()?;
    std::fs::create_dir(&out)?;

    let started = std::time::Instant::now();
    let archive = MappedNemo::open(Path::new(&args[2]))?;
    let bound = bind(HEBREW_GEOMETRY, &archive.checkpoint.census)?;
    println!(
        "bound {} tensors, {} elements, elapsed={:.3}",
        bound.tensors.len(),
        bound.elements,
        started.elapsed().as_secs_f64()
    );
    let encoder = FastConformerEncoder::load(
        HEBREW_GEOMETRY,
        QUALIFIED_CONTEXT,
        &bound,
        archive.bytes(),
        &archive.checkpoint.storages,
    )?;
    println!(
        "loaded encoder elapsed={:.3}",
        started.elapsed().as_secs_f64()
    );

    let bins = HEBREW_GEOMETRY.mel_bins as usize;
    let mut state = encoder.new_state();
    let mut index = 0usize;
    loop {
        let mel_path = oracle.join(format!("chunk-{index:03}-mel.f32.bin"));
        if !mel_path.exists() {
            break;
        }
        let mel = read_f32(&mel_path)?;
        if !mel.len().is_multiple_of(bins) {
            return Err(format!("chunk {index} mel is not a whole number of mel rows").into());
        }
        let frames = mel.len() / bins;
        // The first step has no cache to overlap with, so it drops nothing.
        let drop = if index == 0 { 0 } else { drop_extra };
        let (encoded, emitted) = encoder.stream_step(&mel, frames, drop, &mut state)?;
        let bytes: Vec<u8> = encoded.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(out.join(format!("chunk-{index:03}-encoder.f32")), bytes)?;
        println!(
            "chunk={index} mel_frames={frames} drop={drop} emitted={emitted} \
             valid_cache={} elapsed={:.3}",
            state.valid_cache_frames(),
            started.elapsed().as_secs_f64()
        );
        index += 1;
    }
    if index == 0 {
        return Err("oracle directory holds no chunk mel fixtures".into());
    }
    std::fs::write(out.join("chunks.txt"), format!("{index}\n"))?;
    println!(
        "chunks={index} elapsed={:.3}",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
