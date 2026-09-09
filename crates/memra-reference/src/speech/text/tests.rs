use super::*;
use memra_gguf::model_packs::whisper::PACK;

fn plan() -> WhisperPlan {
    let config =
        include_str!("../../../../memra-gguf/src/model_packs/whisper/fixtures/config.json");
    let frontend = include_str!(
        "../../../../memra-gguf/src/model_packs/whisper/fixtures/preprocessor_config.json"
    );
    PACK.compile_plan(config, frontend).unwrap().speech.unwrap()
}

const CHECKPOINT: &str = "/home/avifenesh/hebrew-asr-data/models/whisper-large-v3-ivrit-766847c9";

#[test]
fn a_window_of_pinned_oracle_ids_reads_as_hebrew() {
    let dir = std::path::Path::new(CHECKPOINT);
    if !dir.exists() {
        eprintln!("skipping: whisper checkpoint is not on this machine");
        return;
    }
    let plan = plan();
    let detokenizer = Detokenizer::from_hf_dir(dir).unwrap();
    // d1-000 window 0, the pinned oracle's own beam-1 ids: a timestamp, six text ids, and EOS.
    let ids = [50413u32, 6044, 97, 2986, 8573, 8817, 29526, 17333, 50257];
    assert_eq!(
        whisper_transcript(&plan, &detokenizer, &ids),
        " בפרק של היום"
    );
    let aligned = whisper_transcript_with_timestamps(&plan, &detokenizer, &ids);
    assert!(aligned.starts_with("<|0.96|>"), "{aligned}");
    assert!(aligned.ends_with("היום"), "{aligned}");
}

#[test]
fn timestamps_and_the_terminator_never_reach_the_transcript() {
    let dir = std::path::Path::new(CHECKPOINT);
    if !dir.exists() {
        eprintln!("skipping: whisper checkpoint is not on this machine");
        return;
    }
    let plan = plan();
    let detokenizer = Detokenizer::from_hf_dir(dir).unwrap();
    let only_markers = [plan.decode.eos_token, plan.decode.timestamp_begin, 51865];
    assert_eq!(whisper_transcript(&plan, &detokenizer, &only_markers), "");
    // The prefix ids are flagged in the vocabulary, so they drop even without the id filter.
    let prefix = [
        plan.decode.start_token,
        plan.decode.language_token,
        plan.decode.task_token,
    ];
    assert_eq!(whisper_transcript(&plan, &detokenizer, &prefix), "");
}
