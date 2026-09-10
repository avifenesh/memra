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
use memra_reference::speech::rnnt_head::RnntHead;
use memra_tokenizer::detokenize::SpmDetokenizer;
use std::path::{Path, PathBuf};

fn write_f32(path: &Path, values: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes)?;
    Ok(())
}

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
    if args.len() == 6 && args[1] == "stream" {
        // usage: stream ARCHIVE PCM.f32 OUT_DIR PROMPT_SLOT
        let out = PathBuf::from(&args[4]);
        let slot: u32 = args[5].parse()?;
        std::fs::create_dir(&out)?;
        let started = std::time::Instant::now();
        let archive = MappedNemo::open(Path::new(&args[2]))?;
        let bound = bind(HEBREW_GEOMETRY, &archive.checkpoint.census)?;
        let frontend = RnntFrontend::from_bound(
            HEBREW_GEOMETRY,
            &bound,
            archive.bytes(),
            &archive.checkpoint.storages,
        )?;
        let encoder = FastConformerEncoder::load(
            HEBREW_GEOMETRY,
            QUALIFIED_CONTEXT,
            &bound,
            archive.bytes(),
            &archive.checkpoint.storages,
        )?;
        let head = RnntHead::load(
            HEBREW_GEOMETRY,
            &bound,
            archive.bytes(),
            &archive.checkpoint.storages,
        )?;
        let contract = encoder.contract();
        let bins = HEBREW_GEOMETRY.mel_bins as usize;
        let width = HEBREW_GEOMETRY.encoder_width as usize;

        let pcm = read_f32(Path::new(&args[3]))?;
        let (mel, frames) = frontend.compute(&pcm)?;
        let valid = frontend.valid_frames(pcm.len());

        let mut state = encoder.new_state();
        let mut session = head.new_greedy();
        let mut emitted: Vec<u32> = Vec::new();
        let mut summary = String::from(
            "chunk	mel_start	mel_frames	drop	emitted	total
",
        );
        let mut cursor = 0usize;
        let mut index = 0usize;
        while cursor < valid {
            let first = index == 0;
            let take = if first {
                contract.first_chunk_mel_frames as usize
            } else {
                contract.chunk_mel_frames as usize
            };
            let carry = if first {
                0
            } else {
                contract.pre_encode_carry_mel_frames as usize
            };
            let drop = if first {
                0
            } else {
                contract.drop_extra_pre_encoded as usize
            };
            let start = cursor.saturating_sub(carry);
            let lead = carry - (cursor - start);
            let end = (cursor + take).min(frames);
            let window_frames = lead + (end - start);
            // Frames before the audio began are zero, which is what the reference's own
            // streaming buffer pads with.
            let mut window = vec![0.0f32; bins * window_frames];
            for bin in 0..bins {
                for frame in start..end {
                    window[bin * window_frames + lead + (frame - start)] =
                        mel[bin * frames + frame];
                }
            }
            let (encoded, count) = encoder.stream_step(&window, window_frames, drop, &mut state)?;
            // The encoder banks [width, frames]; the head consumes [frames, width].
            let mut rows = vec![0.0f32; count * width];
            for frame in 0..count {
                for channel in 0..width {
                    rows[frame * width + channel] = encoded[channel * count + frame];
                }
            }
            let prompted = head.prompt(&rows, count, slot)?;
            let mut chunk_tokens = 0usize;
            for frame in 0..count {
                let step = head.greedy_frame(
                    &prompted[frame * width..(frame + 1) * width],
                    &mut session,
                    10,
                )?;
                chunk_tokens += step.len();
                emitted.extend(step);
            }
            summary.push_str(&format!(
                "{index}	{start}	{window_frames}	{drop}	{chunk_tokens}	{}
",
                emitted.len()
            ));
            std::fs::write(
                out.join(format!("chunk-{index:03}-tokens.txt")),
                emitted
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
            )?;
            cursor += take;
            index += 1;
        }
        // The transcript, in the engine. The vocabulary is the archive's own SentencePiece
        // model, read from the same mapped bytes the weights came from.
        let spm = archive
            .checkpoint
            .members
            .iter()
            .find(|(name, _, _)| name.ends_with("_tokenizer.model"))
            .ok_or("archive has no tokenizer model")?;
        let vocabulary = SpmDetokenizer::from_proto(&archive.bytes()[spm.1..spm.1 + spm.2])?;
        let transcript = vocabulary.decode(&emitted);
        std::fs::write(out.join("transcript.txt"), &transcript)?;
        std::fs::write(out.join("chunks.tsv"), summary)?;
        std::fs::write(
            out.join("tokens.txt"),
            emitted
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(" "),
        )?;
        println!(
            "stream chunks={index} samples={} tokens={} text={transcript:?} elapsed={:.3}",
            pcm.len(),
            emitted.len(),
            started.elapsed().as_secs_f64()
        );
        return Ok(());
    }
    if args.len() == 6 && args[1] == "head" {
        // usage: head ARCHIVE ENCODER_ORACLE OUT_DIR PROMPT_SLOT
        let oracle = PathBuf::from(&args[3]);
        let out = PathBuf::from(&args[4]);
        let slot: u32 = args[5].parse()?;
        std::fs::create_dir(&out)?;
        let started = std::time::Instant::now();
        let archive = MappedNemo::open(Path::new(&args[2]))?;
        let bound = bind(HEBREW_GEOMETRY, &archive.checkpoint.census)?;
        let head = RnntHead::load(
            HEBREW_GEOMETRY,
            &bound,
            archive.bytes(),
            &archive.checkpoint.storages,
        )?;
        let width = HEBREW_GEOMETRY.encoder_width as usize;

        // Encoder frames come from the already-gated capture, banked one chunk per file as
        // [width, frames]; the head consumes [frames, width].
        let mut frames = Vec::new();
        let mut index = 0usize;
        loop {
            let path = oracle.join(format!("chunk-{index:03}-encoder.f32.bin"));
            if !path.exists() {
                break;
            }
            let chunk = read_f32(&path)?;
            let count = chunk.len() / width;
            for frame in 0..count {
                for channel in 0..width {
                    frames.push(chunk[channel * count + frame]);
                }
            }
            index += 1;
        }
        if frames.is_empty() {
            return Err("encoder oracle holds no chunk outputs".into());
        }
        let count = frames.len() / width;
        let prompted = head.prompt(&frames, count, slot)?;
        write_f32(&out.join("prompted-encoder.f32"), &prompted)?;

        // The pinned predictor walk, banked step by step with its state.
        let walk: [u32; 7] = [0, 5, 61, 137, 900, 4321, 13086];
        let mut state = head.new_state();
        let mut last_row = Vec::new();
        for (step, token) in walk.iter().enumerate() {
            let (row, next) = head.predict(Some(*token), &state)?;
            write_f32(&out.join(format!("predictor-{step:02}-out.f32")), &row)?;
            for layer in 0..HEBREW_GEOMETRY.predictor_layers as usize {
                write_f32(
                    &out.join(format!("predictor-{step:02}-hidden-{layer}.f32")),
                    next.hidden(layer),
                )?;
                write_f32(
                    &out.join(format!("predictor-{step:02}-cell-{layer}.f32")),
                    next.cell(layer),
                )?;
            }
            last_row = row;
            state = next;
        }
        let (start_row, _) = head.predict(None, &head.new_state())?;
        write_f32(&out.join("predictor-start-out.f32"), &start_row)?;

        for (label, row) in [("start", &start_row), ("walk", &last_row)] {
            let mut logits = Vec::new();
            for frame in 0..count {
                logits.extend(head.joint(&prompted[frame * width..(frame + 1) * width], row)?);
            }
            write_f32(&out.join(format!("joint-{label}.f32")), &logits)?;
        }

        let emitted = head.greedy(&prompted, count, 10)?;
        std::fs::write(
            out.join("greedy.txt"),
            emitted
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(" "),
        )?;
        println!(
            "head frames={count} prompt_slot={slot} blank={} greedy={} elapsed={:.3}",
            head.blank(),
            emitted.len(),
            started.elapsed().as_secs_f64()
        );
        return Ok(());
    }
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
