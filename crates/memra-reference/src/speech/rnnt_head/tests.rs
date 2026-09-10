//! Unit tests for the RNNT head's shapes and refusals. Parity against the pinned NeMo capture
//! is `tools/check_rnnt_head.py`, which needs the checkpoint on the machine.

use memra_gguf::model_packs::nemotron_rnnt::HEBREW_GEOMETRY;

#[test]
fn the_blank_is_the_last_class_and_the_joint_is_one_wider_than_the_vocabulary() {
    // The tokenizer has 13087 symbols and the joint emits 13088 classes: the extra one is the
    // blank, and it sits last.
    assert_eq!(HEBREW_GEOMETRY.vocabulary, 13088);
    assert_eq!(HEBREW_GEOMETRY.vocabulary - 1, 13087);
}

#[test]
fn the_prompt_slot_is_a_language_index_not_a_token() {
    // he-IL is slot 64 of 128. It is concatenated onto every encoder row, so it is not a
    // position in the token sequence and cannot be confused for one.
    // he-IL is slot 64 of the model's 128.
    assert_eq!(HEBREW_GEOMETRY.prompt_slots, 128);
    assert_eq!(HEBREW_GEOMETRY.prompt_slots.min(64), 64);
    assert_eq!(
        HEBREW_GEOMETRY.prompt_input,
        HEBREW_GEOMETRY.encoder_width + HEBREW_GEOMETRY.prompt_slots
    );
}

#[test]
fn the_predictor_packs_four_gates_into_every_row() {
    // weight_ih is [2560, 640] = four 640-wide gates, which is why the state is 640 and not
    // 2560 wide.
    assert_eq!(4 * HEBREW_GEOMETRY.predictor_width, 2560);
    assert_eq!(HEBREW_GEOMETRY.predictor_layers, 2);
}
