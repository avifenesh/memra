use std::process::Command;

/// Black-box check that `memra model qualify` is a real caller of
/// `validate_qualification_record`, not dead library code: the binary must exit 0 on a fully
/// exercised NativeQualified record and refuse (nonzero) a readiness-only one that claims the
/// same promotion (memra#543 acceptance criteria 3 and 4).
#[test]
fn qualify_cli_accepts_a_full_record_and_refuses_a_readiness_only_one() {
    let root = std::env::temp_dir().join(format!("memra-cli-qualify-cli-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();

    let good = root.join("good.tsv");
    std::fs::write(
        &good,
        "format\tmemra-qualification-record-v1\n\
         family\tqwen3\n\
         promote_to\tNativeQualified\n\
         cell.streaming\tpassed\n\
         cell.cache\tpassed\n\
         cell.concurrency\tpassed\n\
         cell.cancellation\tpassed\n\
         cell.long_context\tpassed\n",
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_memra"))
        .args(["model", "qualify"])
        .arg(&good)
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("internally consistent"),
        "{output:?}"
    );

    let stub = root.join("stub.tsv");
    std::fs::write(
        &stub,
        "format\tmemra-qualification-record-v1\n\
         family\tqwen3\n\
         promote_to\tNativeQualified\n\
         cell.streaming\tnot_exercised\n\
         cell.cache\tnot_exercised\n\
         cell.concurrency\tnot_exercised\n\
         cell.cancellation\tnot_exercised\n\
         cell.long_context\tnot_exercised\n",
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_memra"))
        .args(["model", "qualify"])
        .arg(&stub)
        .output()
        .unwrap();
    assert!(!output.status.success(), "{output:?}");
    let err = String::from_utf8_lossy(&output.stderr);
    assert!(err.contains("has not passed"), "{err}");

    std::fs::remove_dir_all(root).unwrap();
}
