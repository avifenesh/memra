//! G1 for the TTS family: the census IS the refusal, against the pinned artifact's own headers.
//!
//! The two fixtures are the safetensors headers of `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` at
//! revision `0c0e3051f131929182e2c023b9537f8b1c68adfe`, taken by HTTP range request so no weight
//! byte was ever downloaded to produce them. They are header-only, which is why this gate runs in
//! hosted CI with no artifact and no GPU.
//!
//! Every red arm below has a named wrong answer it refuses. A gate that has never failed proves
//! nothing, so each one asserts the specific refusal and not merely "an error happened".

use super::*;
use crate::safetensors::StInfo;
use std::collections::HashMap;

const TALKER_HEADER: &str = include_str!("fixtures/talker-header.json");
const CODEC_HEADER: &str = include_str!("fixtures/codec-header.json");

fn header(json: &str) -> HashMap<String, StInfo> {
    crate::safetensors::parse_header_json_checked(json).expect("fixture header must parse")
}

// --------------------------------------------------------------------------- the census gate

#[test]
fn the_talker_contract_binds_the_pinned_artifact_exactly() {
    let bound = bind_talker(TALKER_HEADER).expect("the pinned talker must bind");
    assert_eq!(bound.tensors.len(), 404);
    assert_eq!(bound.elements, 1_916_676_352);
    // BF16 throughout: 2 bytes per element and nothing else.
    assert_eq!(bound.payload_bytes, 1_916_676_352 * 2);
    assert_eq!(talker_tensor_contract(TALKER).len(), 404);
}

#[test]
fn the_codec_decoder_contract_binds_exactly_and_the_encoder_half_is_dead_weight() {
    let bound = bind_codec_decoder(CODEC_HEADER).expect("the pinned codec decoder must bind");
    assert_eq!(bound.tensors.len(), 271);
    assert_eq!(bound.elements, 114_323_137);
    assert_eq!(
        bound.payload_bytes,
        114_323_137 * 4,
        "the codec is F32 on disk"
    );
    assert_eq!(codec_decoder_tensor_contract().len(), 271);

    // The encoder half is counted so its size is a receipt and not an estimate, and it is NOT in
    // the decoder contract: 0.22 GB of weights synthesis can never run.
    let (tensors, elements) = codec_encoder_is_dead_weight(CODEC_HEADER).unwrap();
    assert_eq!(tensors, 225);
    assert_eq!(elements, 56_234_304);
    assert!(
        !codec_decoder_tensor_contract()
            .iter()
            .any(|r| r.name.starts_with("encoder."))
    );

    // What a serving box holds is the talker plus the codec's decoder half, not the whole codec.
    assert_eq!(RESIDENT_ELEMENTS, 1_916_676_352 + 114_323_137);
    assert_eq!(RESIDENT_ELEMENTS, 2_030_999_489);
}

#[test]
fn the_generated_contract_reproduces_the_census_groups_the_artifact_reports() {
    let h = header(TALKER_HEADER);
    let groups: &[(&str, usize, u64)] = &[
        ("talker.model.", 311, 1_726_866_432),
        ("talker.code_predictor.", 88, 175_125_760),
        ("talker.text_projection.", 4, 8_392_704),
        ("talker.codec_head.", 1, 6_291_456),
    ];
    for (prefix, tensors, elements) in groups {
        let got: Vec<&StInfo> = h
            .iter()
            .filter(|(k, _)| k.starts_with(*prefix))
            .map(|(_, v)| v)
            .collect();
        // `talker.model.` also matches nothing else, but `talker.codec_head.` must not be caught by
        // the `talker.code_predictor.` prefix, so each group is counted from the header directly.
        let n = got.len();
        let e: u64 = got.iter().map(|i| i.shape.iter().product::<u64>()).sum();
        assert_eq!((n, e), (*tensors, *elements), "group {prefix} disagrees");
    }
}

// --------------------------------------------------------------------------- red arms
//
// These mutate the PARSED header and call `bind` directly. `memra-gguf` has no JSON serializer
// (it has its own reader), and re-serializing a mutated map to feed the string entry point would
// mean adding a dependency to test a contract that already takes a map.

/// The same contract `bind_talker` runs, on a header the test has already mutated.
fn bind_talker_map(h: HashMap<String, StInfo>) -> Result<BoundTts, TtsError> {
    bind(
        "talker",
        TALKER_TENSORS,
        TALKER_ELEMENTS,
        talker_tensor_contract(TALKER),
        h,
    )
}

fn bind_codec_map(h: HashMap<String, StInfo>) -> Result<BoundTts, TtsError> {
    let decoder = h
        .into_iter()
        .filter(|(k, _)| k.starts_with("decoder."))
        .collect();
    bind(
        "codec-decoder",
        CODEC_DECODER_TENSORS,
        CODEC_DECODER_ELEMENTS,
        codec_decoder_tensor_contract(),
        decoder,
    )
}

/// `BoundTts` is not `PartialEq` (its `StInfo` values are not), so a red arm asserts on the error,
/// which IS `PartialEq`, rather than on the result.
fn refused<T>(r: Result<T, TtsError>) -> TtsError {
    match r {
        Ok(_) => panic!("the contract accepted something it must refuse"),
        Err(e) => e,
    }
}

#[test]
fn a_missing_tensor_is_refused_by_name() {
    let mut h = header(TALKER_HEADER);
    h.remove("talker.model.layers.27.mlp.down_proj.weight")
        .unwrap();
    assert_eq!(
        refused(bind_talker_map(h)),
        TtsError::MissingTensor("talker.model.layers.27.mlp.down_proj.weight".into())
    );
}

#[test]
fn an_extra_tensor_is_refused_by_name() {
    let mut h = header(TALKER_HEADER);
    h.insert(
        "talker.model.layers.28.input_layernorm.weight".into(),
        StInfo {
            dtype: "BF16".into(),
            shape: vec![2048],
            data_offsets: [9_000_000_000, 9_000_004_096],
        },
    );
    assert_eq!(
        refused(bind_talker_map(h)),
        TtsError::UnexpectedTensor("talker.model.layers.28.input_layernorm.weight".into())
    );
}

#[test]
fn a_reshaped_tensor_is_refused_with_both_shapes() {
    let mut h = header(TALKER_HEADER);
    // The codec head is [3072, 2048]; a port that assumed a [2048, 2048] head would silently
    // emit codes from the wrong space, so the shape is checked and not just the name.
    let info = h.get_mut("talker.codec_head.weight").unwrap();
    info.shape = vec![2048, 2048];
    info.data_offsets = [0, 2048 * 2048 * 2];
    assert_eq!(
        refused(bind_talker_map(h)),
        TtsError::Shape {
            name: "talker.codec_head.weight".into(),
            expected: vec![3072, 2048],
            found: vec![2048, 2048],
        }
    );
}

#[test]
fn a_wrong_storage_dtype_is_refused() {
    let mut h = header(TALKER_HEADER);
    let info = h.get_mut("talker.model.norm.weight").unwrap();
    let start = info.data_offsets[0];
    info.dtype = "F32".into();
    info.data_offsets = [start, start + 2048 * 4];
    assert_eq!(
        refused(bind_talker_map(h)),
        TtsError::Dtype {
            name: "talker.model.norm.weight".into(),
            expected: "BF16",
            found: "F32".into(),
        }
    );
}

#[test]
fn a_tensor_that_does_not_read_its_own_storage_densely_is_refused() {
    let mut h = header(TALKER_HEADER);
    // Right name, right shape, right dtype, and a byte span that is not elements x dtype size.
    // This is the shape a truncated or strided archive presents, and it is invisible to a census
    // that only counts names and shapes.
    let info = h
        .get_mut("talker.model.layers.0.self_attn.q_proj.weight")
        .unwrap();
    let start = info.data_offsets[0];
    info.data_offsets = [start, start + 2048 * 2048];
    assert!(matches!(
        refused(bind_talker_map(h)),
        TtsError::Shape {
            name,
            ..
        } if name == "talker.model.layers.0.self_attn.q_proj.weight"
    ));
}

#[test]
fn aliased_storage_is_refused_before_any_shape_is_checked() {
    let mut h = header(CODEC_HEADER);
    // Two tensors claiming one byte range is how a loader substitutes one tensor's data for
    // another's while every shape and name still looks correct.
    let taken = h["decoder.pre_conv.conv.bias"].data_offsets;
    let victim = h.get_mut("decoder.decoder.6.conv.bias").unwrap();
    victim.data_offsets = taken;
    match refused(bind_codec_map(h)) {
        TtsError::AliasedStorage { offsets, .. } => assert_eq!(offsets, taken),
        other => panic!("aliased storage must be refused first, got {other:?}"),
    }
}

#[test]
fn a_geometry_that_disagrees_with_the_artifact_fails_rather_than_binding_a_shorter_model() {
    let mut g = TALKER;
    g.layers = 27;
    let contract = talker_tensor_contract(g);
    assert_eq!(contract.len(), 404 - 11, "one layer is eleven tensors");
    // The contract is now short, so the census refuses: a 27-layer read of a 28-layer artifact
    // leaves the last layer's eleven tensors unexpected.
    let err = refused(bind(
        "talker",
        404,
        TALKER_ELEMENTS,
        contract,
        header(TALKER_HEADER),
    ));
    assert!(
        matches!(err, TtsError::UnexpectedTensor(_)),
        "a short geometry must surface as unexpected tensors, got {err:?}"
    );

    let mut g2 = TALKER;
    g2.predictor_codebooks = 14;
    let err2 = refused(bind(
        "talker",
        404,
        TALKER_ELEMENTS,
        talker_tensor_contract(g2),
        header(TALKER_HEADER),
    ));
    assert!(matches!(err2, TtsError::UnexpectedTensor(_)));

    // And a geometry that asks for MORE than the artifact has is refused as missing, which is the
    // other direction of the same mistake.
    let mut g3 = TALKER;
    g3.layers = 29;
    let err3 = refused(bind(
        "talker",
        404,
        TALKER_ELEMENTS,
        talker_tensor_contract(g3),
        header(TALKER_HEADER),
    ));
    assert!(
        matches!(err3, TtsError::MissingTensor(ref n) if n == "talker.model.layers.28.input_layernorm.weight"),
        "got {err3:?}"
    );
}

#[test]
fn a_census_whose_total_moves_is_refused_even_when_every_name_binds() {
    // The census is the refusal, so a contract that binds every name but totals the wrong element
    // count must still fail. Asking for one more element than the artifact has does that.
    let decoder: HashMap<String, StInfo> = header(CODEC_HEADER)
        .into_iter()
        .filter(|(k, _)| k.starts_with("decoder."))
        .collect();
    match refused(bind(
        "codec-decoder",
        CODEC_DECODER_TENSORS,
        CODEC_DECODER_ELEMENTS + 1,
        codec_decoder_tensor_contract(),
        decoder,
    )) {
        TtsError::Census {
            expected_elements,
            found_elements,
            ..
        } => {
            assert_eq!(found_elements, 114_323_137);
            assert_eq!(expected_elements, 114_323_138);
        }
        other => panic!("expected a census refusal, got {other:?}"),
    }
}

#[test]
fn the_codec_encoder_half_is_refused_by_the_decoder_contract() {
    // Binding the WHOLE codec archive against the decoder contract must fail, because the encoder
    // half is present in the file and absent from synthesis.
    match refused(bind(
        "codec",
        CODEC_DECODER_TENSORS,
        CODEC_DECODER_ELEMENTS,
        codec_decoder_tensor_contract(),
        header(CODEC_HEADER),
    )) {
        TtsError::UnexpectedTensor(name) => assert!(name.starts_with("encoder."), "{name}"),
        other => panic!("expected the encoder half to be refused, got {other:?}"),
    }
}

// ------------------------------------------------------------------- contract arithmetic

#[test]
fn the_audio_contract_is_read_off_the_artifact_and_closes_on_itself() {
    assert_eq!(AUDIO.sample_rate, 24000);
    assert_eq!(AUDIO.samples_per_frame, 1920);
    // 12.5 Hz, not the 12 Hz in the model's name. Neither 83.33 ms nor 2000 samples exists here.
    assert_eq!(AUDIO.frame_rate_hz(), 12.5);
    assert_eq!(AUDIO.frame_ms(), 80.0);
    assert!(AUDIO.self_consistent());
    assert_eq!(AUDIO.codebooks, 16);
    assert_eq!(AUDIO.semantic_codebooks, 1);
    assert_eq!(AUDIO.codebook_size, 2048);
    assert_eq!(AUDIO.sliding_window, 72);
}

#[test]
fn the_codec_upsample_chain_multiplies_to_exactly_one_frame() {
    assert_eq!(CODEC_UPSAMPLE_STRIDES, [8, 5, 4, 3]);
    assert_eq!(CODEC_UPSAMPLE_KERNELS, [16, 10, 8, 6]);
    // Each kernel is twice its stride, which is what the header's four DIFFERENT conv shapes say.
    for (stride, kernel) in CODEC_UPSAMPLE_STRIDES.iter().zip(CODEC_UPSAMPLE_KERNELS) {
        assert_eq!(kernel, 2 * stride);
    }
    assert_eq!(codec_upsample_product(), 480);
    assert_eq!(codec_total_upsample(), 1920);
    assert_eq!(codec_total_upsample(), AUDIO.samples_per_frame);
    // Channels halve at every stage and end at one output channel.
    assert_eq!(CODEC_CHANNELS, [1536, 768, 384, 192, 96]);
}

#[test]
fn the_prefill_widths_close_against_three_measured_values() {
    // Fixture 1: 16 text-body tokens, measured prefill [1, 27, 2048].
    assert_eq!(prefill_width(16), 27);
    // Fixture 2: 6 text-body tokens, measured prefill [1, 17, 2048].
    assert_eq!(prefill_width(6), 17);
    // The streaming path, measured [1, 10, 2048] for the SAME 24-token input as fixture 1.
    assert_eq!(streaming_prefill_width(), 10);

    // The decomposition, so the width is not a fitted constant: three role positions, then the
    // rows this function generates, then the text body plus tts_eos.
    let rows = prefill_rows(Some(2050), Some(3061));
    assert_eq!(rows.len(), 7, "six tts_pad rows plus the tts_bos row");
    for t in [16usize, 6, 1] {
        assert_eq!(3 + rows.len() + t + 1, prefill_width(t), "T={t}");
    }
}

#[test]
fn the_prefill_rows_are_the_exact_codec_sequence_in_order() {
    let rows = prefill_rows(Some(2050), Some(3061));
    let codec_ids: Vec<i64> = rows.iter().map(|r| r.codec_id).collect();
    assert_eq!(
        codec_ids,
        vec![2154, 2156, 2050, 2157, 3061, 2148, 2149],
        "think_id, think_bos, language, think_eos, speaker row, codec_pad, codec_bos"
    );
    // Positions 3..=8 sum a tts_pad row; position 9 sums tts_bos. Getting this backwards is
    // silent and the prefill is a bit-identity gate.
    let text: Vec<Option<TextSide>> = rows.iter().map(|r| r.text).collect();
    assert_eq!(
        text,
        vec![
            Some(TextSide::PadRow),
            Some(TextSide::PadRow),
            Some(TextSide::PadRow),
            Some(TextSide::PadRow),
            Some(TextSide::PadRow),
            Some(TextSide::PadRow),
            Some(TextSide::BosRow),
        ]
    );
    assert_eq!(rows.iter().filter(|r| r.speaker_row).count(), 1);
    assert!(rows[4].speaker_row, "the speaker row is the fifth position");

    // With no language the think block collapses to three ids, so the whole prefill is one
    // position narrower per removed id.
    let auto = prefill_rows(None, Some(3061));
    assert_eq!(auto.len(), 6);
    assert_eq!(auto[0].codec_id, IDS.codec_nothink);
    // With no speaker row either (a voice-design or clone variant) it collapses again.
    assert_eq!(prefill_rows(None, None).len(), 5);
}

#[test]
fn suppression_bans_every_control_and_speaker_id_except_codec_eos() {
    // The codebook ceiling: 0..2047 are codes, everything above is an input-only token.
    assert!(!is_suppressed(0));
    assert!(!is_suppressed(2047));
    assert!(is_suppressed(2048));
    assert!(is_suppressed(3071));
    assert!(!is_suppressed(3072), "outside the head's width entirely");
    // codec_eos is the ONLY id in the suppressed band the head may emit, and it is how the talker
    // stops. Suppressing it would make every generation run to the frame cap.
    assert!(!is_suppressed(IDS.codec_eos));
    assert_eq!(IDS.codec_eos, 2150);
    // Every speaker row and every language id is inside the suppressed band, which is what keeps
    // the head from emitting a voice or a language mid-utterance.
    for (_, id) in SPEAKERS {
        assert!(is_suppressed(*id), "speaker id {id} must be suppressed");
    }
    for (_, id) in LANGUAGES {
        assert!(is_suppressed(*id), "language id {id} must be suppressed");
    }
}

#[test]
fn the_speaker_and_language_tables_match_the_artifact_and_the_dialect_rule_holds() {
    assert_eq!(speaker_id("Ryan").unwrap(), 3061);
    assert_eq!(speaker_id("ryan").unwrap(), 3061, "case-insensitive");
    assert_eq!(speaker_id("serena").unwrap(), 3066);
    assert!(matches!(
        speaker_id("nobody"),
        Err(TtsError::UnknownSpeaker(_))
    ));
    assert_eq!(language_id("English").unwrap(), 2050);
    assert_eq!(language_id("chinese").unwrap(), 2055);
    assert!(matches!(
        language_id("klingon"),
        Err(TtsError::UnknownLanguage(_))
    ));

    // The dialect override: eric and dylan force a dialect whenever the language is chinese, and
    // `auto` resolves straight to the dialect for those two speakers.
    assert_eq!(resolve_language_id("chinese", "eric").unwrap(), Some(2062));
    assert_eq!(resolve_language_id("chinese", "dylan").unwrap(), Some(2074));
    assert_eq!(resolve_language_id("auto", "eric").unwrap(), Some(2062));
    assert_eq!(resolve_language_id("auto", "dylan").unwrap(), Some(2074));
    // A non-dialect speaker leaves chinese alone, and auto yields NO id, which collapses the think
    // block to three positions rather than four.
    assert_eq!(resolve_language_id("chinese", "ryan").unwrap(), Some(2055));
    assert_eq!(resolve_language_id("auto", "ryan").unwrap(), None);
    assert_eq!(resolve_language_id("english", "eric").unwrap(), Some(2050));
    // The nine built-in speakers and twelve language ids the artifact carries.
    assert_eq!(SPEAKERS.len(), 9);
    assert_eq!(LANGUAGES.len(), 12);
}

#[test]
fn the_served_decode_is_sampled_on_both_loops_and_greedy_is_only_the_instrument() {
    // G7: a default inherited by silence is a violation. The vendor recommendation for this
    // artifact is SAMPLED, and there are TWO sampled shapes, not one.
    const {
        assert!(SERVED_DECODE.talker_do_sample);
        assert!(
            SERVED_DECODE.predictor_do_sample,
            "a port that samples the talker but argmaxes the nested predictor serves a decode nobody \
             recommended"
        );
    }
    assert_eq!(SERVED_DECODE.talker_temperature, 0.9);
    assert_eq!(SERVED_DECODE.predictor_temperature, 0.9);
    assert_eq!(SERVED_DECODE.talker_top_k, 50);
    assert_eq!(SERVED_DECODE.predictor_top_k, 50);
    assert_eq!(SERVED_DECODE.talker_top_p, 1.0);
    assert_eq!(SERVED_DECODE.talker_repetition_penalty, 1.05);
    assert_eq!(SERVED_DECODE.max_new_frames, 8192);
    // 8192 frames at 12.5 Hz is 655.36 s of audio.
    assert!(
        (f64::from(SERVED_DECODE.max_new_frames) / AUDIO.frame_rate_hz() - 655.36).abs() < 1e-9
    );
    // Greedy differs on exactly the two sampling switches and nothing else.
    const {
        assert!(!GREEDY_INSTRUMENT.talker_do_sample);
        assert!(!GREEDY_INSTRUMENT.predictor_do_sample);
    }
    assert_eq!(
        GREEDY_INSTRUMENT.talker_temperature,
        SERVED_DECODE.talker_temperature
    );
    assert_eq!(
        GREEDY_INSTRUMENT.max_new_frames,
        SERVED_DECODE.max_new_frames
    );
}

#[test]
fn the_emit_contract_is_one_whole_frame_and_declares_its_silence() {
    assert_eq!(EMIT.samples_per_chunk, AUDIO.samples_per_frame);
    assert_eq!(EMIT.sample_rate, AUDIO.sample_rate);
    assert_eq!(EMIT.silent_lead_in_frames, 1);
    // The threshold sits above BOTH measured silent lead-ins (5.72e-05 and 4.816e-05) and below
    // the quietest audible frame measured (0.032), so it separates them with margin on both sides.
    const {
        assert!(EMIT.audible_peak > 5.72e-05);
        assert!(EMIT.audible_peak > 4.816e-05);
        assert!(EMIT.audible_peak < 0.032);
    }
    assert_eq!(frames_for_audible(2), 3);
    assert_eq!(frames_for_audible(1), 2);
    // A chunk is never a partial frame.
    assert_eq!(EMIT.samples_per_chunk % AUDIO.samples_per_frame, 0);
}

#[test]
fn the_pin_and_licence_are_the_artifacts_own_and_not_a_family_inheritance() {
    assert_eq!(
        SOURCE_REVISION.len(),
        40,
        "an immutable commit, never a tag"
    );
    assert_eq!(LICENCE, "Apache-2.0");
    assert_eq!(TALKER_WEIGHTS_SHA256.len(), 64);
    assert_eq!(CODEC_WEIGHTS_SHA256.len(), 64);
    assert_eq!(VENDOR_SDIST_SHA256.len(), 64);
    // The vendor source this contract was derived from is byte-identified, so a reader can check
    // the reading against the same bytes.
    assert_eq!(VENDOR_REFERENCE, "qwen-tts==0.1.1");
}
