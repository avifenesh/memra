//! Decode-policy tests. The token tails and window boundaries here are copied out of the
//! pinned rental oracle, so a rule that drifts stops matching real recorded Hebrew audio.

use super::*;
use memra_gguf::model_packs::whisper::PACK;
use memra_gguf::model_plan::speech::WhisperPlan;

fn plan() -> WhisperPlan {
    let config =
        include_str!("../../../../memra-gguf/src/model_packs/whisper/fixtures/config.json");
    let frontend = include_str!(
        "../../../../memra-gguf/src/model_packs/whisper/fixtures/preprocessor_config.json"
    );
    PACK.compile_plan(config, frontend)
        .expect("pinned whisper config")
        .speech
        .expect("speech plan")
}

/// A floor low enough that only the ids a test names carry weight. Flat zeros would put
/// 1501 equal timestamps against one text id and force a timestamp in every case.
fn flat(plan: &WhisperPlan) -> Vec<f32> {
    vec![-30.0f32; plan.vocab_size as usize]
}

#[test]
fn first_step_suppresses_blank_eos_and_late_timestamps() {
    let plan = plan();
    let mut logits = flat(&plan);
    // Keep the text side ahead so the force-timestamp rule stays out of this case.
    logits[1000] = 10.0;
    apply_decode_policy(&plan, &mut logits, &[]).unwrap();
    let begin = plan.decode.timestamp_begin as usize;
    assert!(!logits[plan.decode.blank_token as usize].is_finite());
    assert!(!logits[plan.decode.eos_token as usize].is_finite());
    assert!(logits[begin].is_finite());
    assert!(logits[begin + plan.decode.max_initial_timestamp_index as usize].is_finite());
    assert!(!logits[begin + plan.decode.max_initial_timestamp_index as usize + 1].is_finite());
    assert!(logits[1000].is_finite());
}

#[test]
fn suppressed_ids_are_masked_at_every_step() {
    let plan = plan();
    let mut logits = flat(&plan);
    logits[1000] = 10.0;
    apply_decode_policy(&plan, &mut logits, &[1000, 2000]).unwrap();
    for &id in plan.decode.suppress_tokens {
        assert!(
            !logits[id as usize].is_finite(),
            "suppressed id {id} survived"
        );
    }
    assert!(logits[3000].is_finite());
}

#[test]
fn open_segment_admits_only_a_closing_timestamp_or_the_end() {
    let plan = plan();
    let begin = plan.decode.timestamp_begin;
    let mut logits = flat(&plan);
    logits[1000] = 10.0;
    logits[plan.decode.eos_token as usize] = 0.0;
    // text then one timestamp: the segment is open.
    apply_decode_policy(&plan, &mut logits, &[1000, begin + 30]).unwrap();
    assert!(!logits[1000].is_finite());
    assert!(logits[plan.decode.eos_token as usize].is_finite());
    // Time cannot move backwards, but the open segment may repeat its own timestamp.
    assert!(!logits[(begin + 29) as usize].is_finite());
    assert!(logits[(begin + 30) as usize].is_finite());
    assert!(logits[(begin + 31) as usize].is_finite());
}

#[test]
fn closed_segment_admits_only_text_and_moves_time_forward() {
    let plan = plan();
    let begin = plan.decode.timestamp_begin;
    let mut logits = flat(&plan);
    logits[1000] = 10.0;
    apply_decode_policy(&plan, &mut logits, &[1000, begin + 30, begin + 40]).unwrap();
    assert!(logits[1000].is_finite());
    for id in begin..=begin + 40 {
        assert!(!logits[id as usize].is_finite(), "timestamp {id} survived");
    }
    // A closed segment bans every timestamp for the next step, not only the past ones.
    assert!(!logits[(begin + 41) as usize].is_finite());
}

#[test]
fn timestamp_mass_over_the_best_text_id_forces_a_timestamp() {
    let plan = plan();
    let begin = plan.decode.timestamp_begin as usize;
    let mut logits = flat(&plan);
    logits[1000] = 1.0;
    for slot in logits[begin..begin + 40].iter_mut() {
        *slot = 0.9;
    }
    apply_decode_policy(&plan, &mut logits, &[1000, 2000]).unwrap();
    // No single timestamp beats 1.0, but forty of them together do.
    assert!(!logits[1000].is_finite());
    assert!(logits[begin].is_finite());

    // Red arm: one weak timestamp does not take the step away from the text.
    let mut logits = flat(&plan);
    logits[1000] = 1.0;
    logits[begin] = -5.0;
    // every other timestamp stays on the floor
    apply_decode_policy(&plan, &mut logits, &[1000, 2000]).unwrap();
    assert!(logits[1000].is_finite());
}

#[test]
fn seek_advance_reproduces_pinned_oracle_window_starts() {
    let plan = plan();
    let eos = plan.decode.eos_token;
    // d1-000 windows 0 and 1, whatsapp-030 window 9: real tails, real advances.
    // Window 0 ends on one open timestamp and consumes the whole field.
    assert_eq!(seek_advance(&plan, &[13, 51377, eos], 3000), 3000);
    // Window 1 ends on a closed segment at 51724 and resumes there: 2 * (51724 - 50365).
    assert_eq!(seek_advance(&plan, &[13, 51724, 51780, eos], 3000), 2718);
    // A capped window has no EOS and no timestamp pair at its tail.
    assert_eq!(seek_advance(&plan, &[31175, 14507, 24622], 3000), 3000);
    // A short final window advances by its own extent, not by a full field.
    assert_eq!(seek_advance(&plan, &[13, 51136, eos], 1550), 1550);
}

#[test]
fn window_mel_zero_fills_past_the_audio() {
    let mel = ReferenceTensor::new(vec![2, 10], (0..20).map(|i| i as f32 + 1.0).collect()).unwrap();
    let window = window_mel(&mel, 7, 3).unwrap();
    assert_eq!(window.len(), 2 * WINDOW_FRAMES);
    assert_eq!(&window[..3], &[8.0, 9.0, 10.0]);
    assert_eq!(window[3], 0.0);
    assert_eq!(
        &window[WINDOW_FRAMES..WINDOW_FRAMES + 3],
        &[18.0, 19.0, 20.0]
    );
    assert_eq!(window[WINDOW_FRAMES + 3], 0.0);
    assert!(window_mel(&mel, 8, 3).is_err());
    assert_eq!(clip_content_frames(&mel).unwrap(), 9);
}
