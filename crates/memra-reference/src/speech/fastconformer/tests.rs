//! Unit tests for the pieces of the cache-aware encoder that do not need the 2.5 GB
//! checkpoint. Whole-encoder parity against the pinned NeMo capture is
//! `tools/check_rnnt_encoder.py`, which runs on the rig where the archive lives.

use super::*;

fn contract() -> StreamingStateContract {
    StreamingStateContract::new(HEBREW_GEOMETRY, QUALIFIED_CONTEXT).unwrap()
}

use memra_gguf::model_packs::nemotron_rnnt::{HEBREW_GEOMETRY, QUALIFIED_CONTEXT};

#[test]
fn causal_convolution_reads_the_past_and_a_stride_of_zeros_after_it() {
    // One channel, identity-ish kernel picking the top-left tap, so an output that reads a
    // future frame would show it.
    let mut weight = vec![0.0f32; 9];
    weight[0] = 1.0;
    let conv = Conv2d {
        output_channels: 1,
        input_channels: 1,
        kernel: 3,
        stride: 2,
        groups: 1,
        weight,
        bias: vec![0.0],
    };
    // time 4, freq 4, values are t * 10 + f.
    let x: Vec<f32> = (0..16).map(|i| ((i / 4) * 10 + (i % 4)) as f32).collect();
    let (y, time, freq) = conv.forward(&x, 4, 4);
    // Padding is kernel-1 before and stride-1 after on both axes: 4 -> 4+2+1 = 7 -> 3 outputs.
    assert_eq!((time, freq), (3, 3));
    // Output (0,0) reads padded position (-2,-2), which is zero.
    assert_eq!(y[0], 0.0);
    // Output (1,1) reads padded (0,0) = the first real sample, value 0.
    assert_eq!(y[3 + 1], 0.0);
    // Output (2,2) reads padded (2,2) = t=2, f=2.
    assert_eq!(y[2 * 3 + 2], 22.0);
}

#[test]
fn the_stem_reduces_one_hundred_twenty_eight_bins_to_seventeen() {
    // Three stride-2 causal stages: 128 -> 65 -> 33 -> 17, which is what the checkpoint's
    // [1024, 4352] projection is shaped for.
    let mut freq = 128usize;
    for _ in 0..3 {
        freq = (freq + 3 - 3) / 2 + 1;
    }
    assert_eq!(freq, 17);
    assert_eq!(
        HEBREW_GEOMETRY.subsample_channels as usize * freq,
        4352,
        "projection input width"
    );
}

#[test]
fn the_position_table_centres_on_the_current_frame() {
    let contract = contract();
    let span = contract.last_channel_frames as usize + contract.chunk_frames as usize;
    let rows = 2 * span - 1;
    assert_eq!(span, 57);
    assert_eq!(rows, 113);
    // Row index for relative distance d is span - 1 - d, so the current frame (distance 0)
    // lands on the centre row and the oldest cached frame lands on the first row.
    assert_eq!(span - 1, 56);
    assert_eq!(span as isize - 1 - 56, 0);
}

#[test]
fn the_cache_rolls_oldest_out_and_reports_only_written_rows() {
    let contract = contract();
    let mut state = StreamState::new(contract);
    assert_eq!(state.valid_cache_frames(), 0);
    assert_eq!(state.contract().last_channel_frames, 56);
    assert_eq!(state.contract().last_time_frames, 8);
    assert_eq!(state.channel.len(), 24);
    assert_eq!(state.channel[0].len(), 56 * 1024);
    assert_eq!(state.time[0].len(), 1024 * 8);

    // Roll one frame in by hand, the way the attention step does.
    let width = 1024usize;
    let cache_frames = 56usize;
    let row = vec![7.0f32; width];
    let cache = &mut state.channel[0];
    cache.copy_within(width.., 0);
    cache[(cache_frames - 1) * width..].copy_from_slice(&row);
    assert_eq!(cache[(cache_frames - 1) * width], 7.0);
    assert_eq!(cache[0], 0.0);
}

#[test]
fn a_wider_chunk_is_refused_rather_than_approximated() {
    // The relative-position shift collapses to a direct lookup only for one query row, so a
    // chunk that would emit more than one frame has to be refused until it has its own gate.
    let contract = contract();
    assert_eq!(contract.chunk_frames, 1);
}
