//! One pinned MiMo source composition on exactly two GPUs.
//! Prepared pixels and mels reach model-owned text through vision, the
//! bundled 20-depth audio codec, and the grouped-code audio patch encoder.
//! An explicit RGB8 mode also runs the source image processor before vision.
//! A separate position-zero MTP3 draft runs while the 1M text KV stays
//! allocated. This is synthetic offline evidence, not request serving.

use std::path::Path;
use std::sync::Arc;

use memra_engine::Engine;
use memra_engine::mimo_audio_codec_weights::MiMoAudioCodecEncoderWeights;
use memra_engine::mimo_audio_patch_load::MiMoAudioPatchWeights;
use memra_engine::mimo_mtp_weights::Mtp3Weights;
use memra_engine::mimo_text_weights::MiMoTextWeights;
use memra_engine::mimo_vision_load::MiMoVisionWeights;
use memra_gguf::model_packs::mimo_v2::bind_pinned_text_source;
use memra_gguf::model_packs::mimo_v2::vision::MiMoVisionGrid;
use memra_gguf::source::SafetensorsSource;
use memra_reference::mimo_modal_overlay::{AUDIO_TOKEN_ID, IMAGE_TOKEN_ID, VIDEO_TOKEN_ID};
use memra_reference::mimo_pixel_prepare::prepare_mimo_rgb8_frames;
use memra_reference::mimo_vision_patchify::patchify_prepared_frames;
use sha2::{Digest, Sha256};

type Fail = Box<dyn std::error::Error>;
const HIDDEN: usize = 4_096;
const VOCAB: usize = 152_576;
const FOUR_GIB: usize = 4 * 1024 * 1024 * 1024;
const FULL_CONTEXT: usize = 1_048_576;

fn digest(values: &[f32]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn digest_bytes(values: &[u8]) -> String {
    format!("{:x}", Sha256::digest(values))
}

fn argmax(logits: &[f32]) -> Result<usize, &'static str> {
    if logits.len() != VOCAB || logits.iter().any(|value| !value.is_finite()) {
        return Err("MiMo combined source logits are incomplete");
    }
    let mut best = 0;
    for index in 1..logits.len() {
        if logits[index].total_cmp(&logits[best]).is_gt() {
            best = index;
        }
    }
    Ok(best)
}

fn memory(engine: &Engine) -> Result<(usize, usize), Fail> {
    engine.gpu.ctx.bind_to_thread()?;
    engine.stream().synchronize()?;
    Ok(engine.stream().context().mem_get_info()?)
}

fn record(cards: [&Engine; 2], phase: &str, enforce_guard: bool) -> Result<(), Fail> {
    for (gpu, engine) in cards.into_iter().enumerate() {
        let (free, total) = memory(engine)?;
        println!("memory\t{phase}\t{gpu}\t{free}\t{total}");
        if enforce_guard && free < FOUR_GIB {
            return Err(format!("MiMo combined source GPU {gpu} fell below 4 GiB headroom").into());
        }
    }
    Ok(())
}

fn run() -> Result<(), Fail> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .ok_or("usage: mimo_combined_modal_source_probe <source_dir> [--rgb8]")?;
    let rgb8 = match args.next().as_deref() {
        None => false,
        Some("--rgb8") => true,
        _ => return Err("usage: mimo_combined_modal_source_probe <source_dir> [--rgb8]".into()),
    };
    if args.next().is_some() {
        return Err("usage: mimo_combined_modal_source_probe <source_dir> [--rgb8]".into());
    }
    let source = Arc::new(SafetensorsSource::open(Path::new(&dir))?);
    let (config, _, binding) = bind_pinned_text_source(&source)?;
    let cards = [Engine::new(0)?, Engine::new(1)?];
    let engines = [&cards[0], &cards[1]];
    if cards[0].stream().context().ordinal() == cards[1].stream().context().ordinal() {
        return Err("MiMo combined source needs two distinct GPU devices".into());
    }
    println!(
        "format\t{}",
        if rgb8 {
            "memra-mimo-combined-modal-source-rgb8-v1"
        } else {
            "memra-mimo-combined-modal-source-v1"
        }
    );
    println!("source\tXiaomiMiMo/MiMo-V2.6-Flash-RL@3b38d063180c3e4aed9691fdc735f3d10b266ee4");
    record(engines, "empty", false)?;

    let text = MiMoTextWeights::load(engines, source.clone())?;
    let vision = MiMoVisionWeights::load(&cards[0], source.as_ref(), &binding, &config)?;
    let audio_patch = MiMoAudioPatchWeights::load(&cards[1], source.as_ref(), &binding, &config)?;
    let mtp = Mtp3Weights::load(&cards[0], source)?;
    let codec = MiMoAudioCodecEncoderWeights::load(&cards[0], Path::new(&dir))?;
    if vision.blocks.len() != 28
        || audio_patch.local_layers.len() != 6
        || codec.tensors().len() != 449
        || text.layers.len() != 48
    {
        return Err("MiMo combined source has incomplete resident weight owners".into());
    }
    record(engines, "all_five_weights", false)?;
    cards[0].gpu.ctx.bind_to_thread()?;
    let workspace0 = cards[0].alloc_u8_uninit(FOUR_GIB)?;
    cards[1].gpu.ctx.bind_to_thread()?;
    let workspace1 = cards[1].alloc_u8_uninit(FOUR_GIB)?;
    record(engines, "workspace4g", false)?;
    let mut text_sequence = text.compressed_text_forward(engines, FULL_CONTEXT, [FOUR_GIB; 2])?;
    record(engines, "kv1m", true)?;

    let (pixels, grids, image_tokens, video_tokens) = if rgb8 {
        const HEIGHT: usize = 33;
        const WIDTH: usize = 65;
        let image_rgb = (0..HEIGHT * WIDTH * 3)
            .map(|index| ((index * 17 + 29) % 256) as u8)
            .collect::<Vec<_>>();
        let video_rgb = (0..3 * HEIGHT * WIDTH * 3)
            .map(|index| ((index * 23 + 71) % 256) as u8)
            .collect::<Vec<_>>();
        println!("rgb_image_sha256\t{}", digest_bytes(&image_rgb));
        println!("rgb_video_sha256\t{}", digest_bytes(&video_rgb));
        let (image_prepared, image_shape) =
            prepare_mimo_rgb8_frames(&image_rgb, [1, HEIGHT, WIDTH, 3])?;
        let (video_prepared, video_shape) =
            prepare_mimo_rgb8_frames(&video_rgb, [3, HEIGHT, WIDTH, 3])?;
        let (mut image_patches, image_grid) =
            patchify_prepared_frames(&image_prepared, image_shape)?;
        let (video_patches, video_grid) = patchify_prepared_frames(&video_prepared, video_shape)?;
        let image_tokens = (image_grid.frames * image_grid.height * image_grid.width / 4) as usize;
        let video_tokens = (video_grid.frames * video_grid.height * video_grid.width / 4) as usize;
        image_patches.extend(video_patches);
        (
            image_patches,
            [image_grid, video_grid],
            image_tokens,
            video_tokens,
        )
    } else {
        (
            (0..8 * 3 * 2 * 16 * 16)
                .map(|index| ((index * 17 % 251) as f32 - 125.0) / 256.0)
                .collect::<Vec<_>>(),
            [
                MiMoVisionGrid {
                    frames: 1,
                    height: 2,
                    width: 2,
                },
                MiMoVisionGrid {
                    frames: 1,
                    height: 2,
                    width: 2,
                },
            ],
            1,
            1,
        )
    };
    if rgb8 {
        println!("rgb_image_tokens\t{image_tokens}");
        println!("rgb_video_tokens\t{video_tokens}");
    }
    let vision_output =
        vision.forward_patchified_vision(&cards[0], &config, &grids, &cards[0].htod(&pixels)?)?;
    let visual_rows = cards[0].dtoh(&vision_output)?;
    if visual_rows.len() != (image_tokens + video_tokens) * HIDDEN {
        return Err("MiMo combined source vision rows are incomplete".into());
    }
    let image_rows = visual_rows[..image_tokens * HIDDEN]
        .chunks_exact(HIDDEN)
        .map(<[f32]>::to_vec)
        .collect::<Vec<_>>();
    let video_rows = visual_rows[image_tokens * HIDDEN..]
        .chunks_exact(HIDDEN)
        .map(<[f32]>::to_vec)
        .collect::<Vec<_>>();
    println!("image_row_sha256\t{}", digest(&image_rows[0]));
    println!("video_row_sha256\t{}", digest(&video_rows[0]));
    if rgb8 {
        println!(
            "image_rows_sha256\t{}",
            digest(&visual_rows[..image_tokens * HIDDEN])
        );
        println!(
            "video_rows_sha256\t{}",
            digest(&visual_rows[image_tokens * HIDDEN..])
        );
    }
    record(engines, "vision_forward", true)?;

    let mel = (0..4 * 128)
        .map(|index| ((index * 19 % 53) as f32 - 26.0) / 64.0)
        .collect::<Vec<_>>();
    let conv = codec.encode_prepared_mel_conv(&cards[0], &cards[0].htod(&mel)?, 4)?;
    let stack = codec.encode_transformer_stack(&cards[0], &conv, 2)?;
    let downsample = codec.downsample_post_stack(&cards[0], &stack, 2)?;
    let rvq = codec.encode_20_rvq(&cards[0], &downsample, 1)?;
    let grouped = rvq.grouped_patch_codes()?;
    if grouped.groups != 1 || grouped.codes.len() != 4 * 20 {
        return Err("MiMo combined source grouped audio codes are incomplete".into());
    }
    let audio_output =
        audio_patch.forward_grouped_codes(&cards[1], &grouped.codes, grouped.groups)?;
    let audio_row = cards[1].dtoh(&audio_output)?;
    if audio_row.len() != HIDDEN {
        return Err("MiMo combined source audio row is incomplete".into());
    }
    let audio_rows = vec![audio_row];
    println!("audio_row_sha256\t{}", digest(&audio_rows[0]));
    record(engines, "audio_forward", true)?;

    let tokens = if rgb8 {
        let mut tokens = vec![42];
        tokens.extend(std::iter::repeat_n(IMAGE_TOKEN_ID, image_tokens));
        tokens.extend(std::iter::repeat_n(VIDEO_TOKEN_ID, video_tokens));
        tokens.push(AUDIO_TOKEN_ID);
        tokens
    } else {
        vec![42, IMAGE_TOKEN_ID, VIDEO_TOKEN_ID, AUDIO_TOKEN_ID]
    };
    let prepared =
        text.modal_embedding_gpu_chunk(&cards[0], &tokens, &image_rows, &video_rows, &audio_rows)?;
    if !prepared.requires_payload_identity() || prepared.token_count() != tokens.len() {
        return Err("MiMo combined source lost modal payload identity".into());
    }
    if rgb8
        && (prepared.modal_counts().image != image_tokens
            || prepared.modal_counts().video != video_tokens
            || prepared.modal_counts().audio != 1)
    {
        return Err("MiMo RGB8 source modal placeholder counts drifted".into());
    }
    let step = text_sequence.consume_embedding_chunk(&prepared)?;
    if step.position != tokens.len() - 1 || text_sequence.position() != tokens.len() {
        return Err("MiMo combined source text and KV cursors drifted".into());
    }
    println!("modal_text_argmax\t{}", argmax(&step.logits)?);
    println!("modal_text_logits_sha256\t{}", digest(&step.logits));
    let continuing = text_sequence.token(220)?;
    println!("modal_continuation_argmax\t{}", argmax(&continuing)?);
    record(engines, "modal_text_continuation", true)?;

    // MTP3 runs on a separate position-zero text sequence because its
    // per-depth prefix priming and speculative rollback are still missing.
    let mut mtp_target = text.compressed_text_forward(engines, 1, [FOUR_GIB; 2])?;
    let target = mtp_target.token_with_hidden(42)?;
    let mut draft = mtp.draft_forward(&text, &cards[0], &cards[1])?;
    let mut token = 42u32;
    let mut hidden = target.hidden_before_norm;
    for depth in 0..3 {
        let next = draft.token(depth, 0, token, &hidden)?;
        token = argmax(&next.logits)? as u32;
        hidden = next.hidden_before_norm;
        println!("mtp_depth\t{depth}\tproposed_token\t{token}");
    }
    record(engines, "all_components_executed", true)?;
    std::hint::black_box((
        &text_sequence,
        &mtp_target,
        &vision_output,
        &audio_output,
        &conv,
        &stack,
        &downsample,
        &rvq,
        &workspace0,
        &workspace1,
    ));
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_combined_modal_source_probe: {error}");
        std::process::exit(1);
    }
}
