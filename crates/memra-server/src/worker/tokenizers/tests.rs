use super::*;

mod fixture;
use fixture::Fixture;

#[test]
fn source_replacement_keeps_worker_http_and_respawn_on_one_snapshot() {
    let fixture = Fixture::new();
    let snapshots = TokenizerSnapshots::default();
    let worker = snapshots
        .load("m", "checkpoint", || Tokenizer::from_hf_dir(&fixture.0))
        .unwrap();
    let ids = worker.encode("hello", true);
    let rendered = worker.apply_chat_template(&[("user", "hello")], true);
    fixture.replace_config(false);
    let reopened = Tokenizer::from_hf_dir(&fixture.0).unwrap();
    assert_ne!(
        ids,
        reopened.encode("hello", true),
        "replacement must change BOS behavior"
    );
    assert_ne!(
        rendered,
        reopened.apply_chat_template(&[("user", "hello")], true),
        "replacement must change the template"
    );

    let http = snapshots.snapshot(&["m".into()]).unwrap();
    let respawn = snapshots
        .load("m", "checkpoint", || {
            panic!("respawn must not reopen tokenizer files")
        })
        .unwrap();
    assert!(Arc::ptr_eq(&worker, &http["m"]));
    assert!(Arc::ptr_eq(&worker, &respawn));
    assert_eq!(http["m"].encode("hello", true), ids);
    assert_eq!(
        http["m"].apply_chat_template(&[("user", "hello")], true),
        rendered
    );
}

#[test]
fn failed_load_does_not_publish_a_partial_snapshot() {
    let snapshots = TokenizerSnapshots::default();
    assert!(
        snapshots
            .load("m", "checkpoint", || Err("missing tokenizer".into()))
            .is_err()
    );
    assert!(snapshots.snapshot(&["m".into()]).is_err());
    let fixture = Fixture::new();
    let loaded = snapshots
        .load("m", "checkpoint", || Tokenizer::from_hf_dir(&fixture.0))
        .unwrap();
    assert!(Arc::ptr_eq(
        &loaded,
        &snapshots.snapshot(&["m".into()]).unwrap()["m"]
    ));
    assert!(snapshots.snapshot(&["m".into(), "missing".into()]).is_err());
}

#[test]
fn model_alias_cannot_reuse_another_source_snapshot() {
    let snapshots = TokenizerSnapshots::default();
    let fixture = Fixture::new();
    snapshots
        .load("m", "first", || Tokenizer::from_hf_dir(&fixture.0))
        .unwrap();
    let result = snapshots.load("m", "second", || {
        panic!("different source must refuse before opening")
    });
    assert!(result.err().unwrap().contains("source changed"));
}
