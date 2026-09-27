//! Dump pinned MiMo PCM-to-mel rows for a same-host publisher comparison.
//! This is a diagnostic example, not an audio serving path.

use std::error::Error;
use std::path::Path;

use memra_reference::mimo_audio_pcm_mel::{MEL_BINS, MiMoPcmMelFrontend, SAMPLE_RATE};
use memra_reference::mimo_audio_pcm_mel_mkl::OneMklFft960;

type Fail = Box<dyn Error>;

fn floats(bytes: &[u8]) -> Result<Vec<f32>, Fail> {
    if !bytes.len().is_multiple_of(4) {
        return Err("MiMo PCM fixture byte extent differs from F32".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn run() -> Result<(), Fail> {
    let mut args = std::env::args().skip(1);
    let mkl_path = args
        .next()
        .ok_or("usage: mimo_mel_same_host <absolute-mkl-path> <output-dir>")?;
    let output_dir = args
        .next()
        .ok_or("usage: mimo_mel_same_host <absolute-mkl-path> <output-dir>")?;
    if args.next().is_some() || !Path::new(&output_dir).is_absolute() {
        return Err("usage: mimo_mel_same_host <absolute-mkl-path> <absolute-output-dir>".into());
    }
    let output_dir = Path::new(&output_dir);
    if !output_dir.is_dir() {
        return Err("MiMo mel diagnostic output directory is missing".into());
    }
    let frontend = MiMoPcmMelFrontend::new();
    let mut fft = OneMklFft960::open(Path::new(&mkl_path))?;
    let cases: [(&str, &[u8]); 9] = [
        (
            "1680",
            include_bytes!("../src/fixtures/mimo-pcm-mel-1680.pcm.f32"),
        ),
        (
            "721",
            include_bytes!("../src/fixtures/mimo-pcm-mel-721.pcm.f32"),
        ),
        (
            "chirp-1201",
            include_bytes!("../src/fixtures/mimo-pcm-mel-chirp-1201.pcm.f32"),
        ),
        (
            "noise-1537",
            include_bytes!("../src/fixtures/mimo-pcm-mel-noise-1537.pcm.f32"),
        ),
        (
            "quiet-tone-1680",
            include_bytes!("../src/fixtures/mimo-pcm-mel-quiet-tone-1680.pcm.f32"),
        ),
        (
            "silence-960",
            include_bytes!("../src/fixtures/mimo-pcm-mel-silence-960.pcm.f32"),
        ),
        (
            "sparse-1680",
            include_bytes!("../src/fixtures/mimo-pcm-mel-sparse-1680.pcm.f32"),
        ),
        (
            "tone-1680",
            include_bytes!("../src/fixtures/mimo-pcm-mel-tone-1680.pcm.f32"),
        ),
        (
            "voiced-2048",
            include_bytes!("../src/fixtures/mimo-pcm-mel-voiced-2048.pcm.f32"),
        ),
    ];
    println!("format\tmimo-mel-same-host-diagnostic-v1");
    for (name, bytes) in cases {
        let pcm = floats(bytes)?;
        let features = frontend.compute_with_mkl(&pcm, SAMPLE_RATE, &mut fft)?;
        if features.shape != [MEL_BINS, pcm.len() / 240 + 1] {
            return Err(format!("MiMo mel {name} source frame geometry changed").into());
        }
        let output = features
            .data
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>();
        std::fs::write(
            output_dir.join(format!("memra-remote-{name}.f32le")),
            output,
        )?;
        println!("case\t{name}\t{}\t{}", pcm.len(), features.shape[1]);
    }
    let pcm = (0..2048)
        .map(|index| ((index * 37 % 257) as f32 - 128.0) / 256.0)
        .collect::<Vec<_>>();
    let features = frontend.compute_with_mkl(&pcm, SAMPLE_RATE, &mut fft)?;
    if features.shape != [MEL_BINS, 9] {
        return Err("MiMo raw-modal probe PCM frame geometry changed".into());
    }
    let output = features
        .data
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<u8>>();
    std::fs::write(output_dir.join("memra-remote-probe.f32le"), output)?;
    println!("case\tprobe\t{}\t{}", pcm.len(), features.shape[1]);
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_mel_same_host: {error}");
        std::process::exit(1);
    }
}
