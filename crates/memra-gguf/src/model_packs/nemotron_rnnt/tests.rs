use super::*;
use crate::nemo::pickle_census;
use std::path::Path;

const DATA_PKL: &[u8] = include_bytes!("fixtures/model_weights.data.pkl");

/// The Hebrew successor archive. Present on the lane rig, absent in hosted CI, which is why
/// the census that gates the contract runs on the checked-in `data.pkl` instead.
const HEBREW_ARCHIVE: &str =
    "/home/avifenesh/hebrew-asr-data/models/campaign-20260908/clean-step-21959.nemo";

#[test]
fn the_contract_binds_the_pinned_hebrew_checkpoint_exactly() {
    let census = pickle_census(DATA_PKL).unwrap();
    let bound = bind(HEBREW_GEOMETRY, &census).unwrap();
    assert_eq!(bound.tensors.len(), 657);
    assert_eq!(bound.elements, 638_030_384);
    assert_eq!(bound.payload_bytes, 2_552_121_536);
    assert_eq!(tensor_contract(HEBREW_GEOMETRY).len(), census.len());
}

#[test]
fn a_missing_extra_or_reshaped_tensor_fails_the_bind() {
    let census = pickle_census(DATA_PKL).unwrap();

    let mut short = census.clone();
    let dropped = short.remove(300).name;
    assert_eq!(
        bind(HEBREW_GEOMETRY, &short),
        Err(RnntError::MissingTensor(dropped))
    );

    let mut extra = census.clone();
    let mut spare = extra[0].clone();
    spare.name = "encoder.layers.24.norm_out.weight".into();
    spare.storage_key = "99999".into();
    extra.push(spare);
    assert_eq!(
        bind(HEBREW_GEOMETRY, &extra),
        Err(RnntError::UnexpectedTensor(
            "encoder.layers.24.norm_out.weight".into()
        ))
    );

    let mut reshaped = census.clone();
    let index = reshaped
        .iter()
        .position(|t| t.name == "joint.enc.weight")
        .unwrap();
    reshaped[index].shape = vec![640, 1023];
    assert_eq!(
        bind(HEBREW_GEOMETRY, &reshaped),
        Err(RnntError::Shape {
            name: "joint.enc.weight".into(),
            expected: vec![640, 1024],
            found: vec![640, 1023],
        })
    );

    // A geometry that disagrees with the artifact is a bind failure, not a silent accept.
    let mut wider = HEBREW_GEOMETRY;
    wider.encoder_layers = 25;
    assert!(matches!(
        bind(wider, &census),
        Err(RnntError::MissingTensor(_))
    ));
}

#[test]
fn aliased_storage_fails_the_bind_before_any_shape_is_checked() {
    let mut census = pickle_census(DATA_PKL).unwrap();
    let key = census[10].storage_key.clone();
    census[11].storage_key = key.clone();
    assert_eq!(
        bind(HEBREW_GEOMETRY, &census),
        Err(RnntError::AliasedStorage(key))
    );
}

#[test]
fn the_prompt_slot_is_the_encoder_row_plus_a_language_slot() {
    // 1152 is not a free parameter: it is the 1024-wide encoder row concatenated with the
    // 128-way prompt representation that carries he-IL.
    assert_eq!(
        HEBREW_GEOMETRY.prompt_input,
        HEBREW_GEOMETRY.encoder_width + HEBREW_GEOMETRY.prompt_slots
    );
    let contract = tensor_contract(HEBREW_GEOMETRY);
    let projection = contract
        .iter()
        .find(|r| r.name == "prompt_kernel.0.weight")
        .unwrap();
    assert_eq!(projection.shape, vec![2048, 1152]);
}

#[test]
fn the_qualified_streaming_arm_never_reads_future_audio() {
    let state = StreamingStateContract::new(HEBREW_GEOMETRY, QUALIFIED_CONTEXT).unwrap();
    assert!(state.is_causal());
    assert_eq!(state.context.left, 56);
    assert_eq!(state.last_channel_frames, 56);
    // The causal depthwise convolution carries kernel minus one frames.
    assert_eq!(state.last_time_frames, 8);
    assert_eq!(state.layers, 24);
    assert_eq!(state.chunk_frames, 1);
    // The driver constants, measured from the reference's streaming configuration rather than
    // derived: the first step is one mel frame with no carry, later steps are eight new frames
    // on top of a nine-frame carry and drop the two rows the carry already produced.
    assert_eq!(state.first_chunk_mel_frames, 1);
    assert_eq!(state.chunk_mel_frames, 8);
    assert_eq!(state.pre_encode_carry_mel_frames, 9);
    assert_eq!(state.drop_extra_pre_encoded, 2);
    // 24 layers x 56 frames x 1024 of cached layer input, plus 24 x 8 x 1024 convolution
    // history, plus two LSTM layers of hidden and cell state. The attention term is not
    // doubled: the reference caches the layer input, not separate keys and values, which the
    // pinned capture shows as a [24, 1, 56, 1024] tensor.
    assert_eq!(state.state_elements(), 1_376_256 + 196_608 + 2_560);

    // The other published arms stay expressible so the schema is not [56,0]-shaped, but they
    // are not causal and this lane does not admit them.
    for context in EXPRESSIBLE_CONTEXTS.iter().filter(|c| c.right > 0) {
        let other = StreamingStateContract::new(HEBREW_GEOMETRY, *context).unwrap();
        assert!(!other.is_causal());
    }
    assert!(
        StreamingStateContract::new(HEBREW_GEOMETRY, AttentionContext { left: 70, right: 0 })
            .is_err()
    );
}

#[test]
fn the_real_archive_census_matches_the_fixture_when_the_artifact_is_present() {
    let path = Path::new(HEBREW_ARCHIVE);
    if !path.exists() {
        eprintln!("skipping: {HEBREW_ARCHIVE} is not on this machine");
        return;
    }
    let file = std::fs::File::open(path).unwrap();
    let map = unsafe { memmap2::Mmap::map(&file) }.unwrap();
    let archive: &[u8] = &map;
    assert_eq!(archive.len() as u64, HEBREW_SUCCESSOR_BYTES);

    let checkpoint = read_archive(archive).unwrap();
    let names: Vec<&str> = checkpoint
        .members
        .iter()
        .map(|(name, _, _)| name.as_str())
        .collect();
    assert_eq!(names.len(), 4);
    assert!(names.iter().any(|n| n.ends_with("_tokenizer.model")));
    assert!(names.iter().any(|n| n.ends_with("_vocab.txt")));
    assert_eq!(names[2], "./model_config.yaml");
    assert_eq!(names[3], "./model_weights.ckpt");
    assert_eq!(checkpoint.config_yaml.1, 218_028);

    // The census the loader takes off the real archive is the census the fixture pins.
    assert_eq!(checkpoint.census, pickle_census(DATA_PKL).unwrap());
    let bound = bind(HEBREW_GEOMETRY, &checkpoint.census).unwrap();
    assert_eq!(bound.elements, 638_030_384);

    // Every bound tensor's payload is a real, exactly sized member of the archive.
    for tensor in bound.tensors.values() {
        let (at, len) = checkpoint.storages.get(&tensor.storage_key).unwrap();
        assert_eq!(*len as u64, tensor.elements() * 4, "{}", tensor.name);
        assert!(at + len <= archive.len());
    }
    assert_eq!(checkpoint.storages.len(), 657);
}
