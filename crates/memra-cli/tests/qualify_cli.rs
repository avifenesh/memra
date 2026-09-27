use std::process::Command;

/// Black-box check that `memra model qualify` is a real caller of
/// `validate_qualification_record`, not dead library code: the binary must exit 0 on a fully
/// exercised NativeQualified record backed by real evidence files, and refuse (nonzero) a
/// readiness-only one that claims the same promotion (memra#543 acceptance criteria 3 and 4).
#[test]
fn qualify_cli_accepts_a_full_record_and_refuses_a_readiness_only_one() {
    let root = std::env::temp_dir().join(format!("memra-cli-qualify-cli-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();

    let cells = [
        "cell.streaming",
        "cell.cache",
        "cell.concurrency",
        "cell.cancellation",
        "cell.long_context",
    ];

    // A record whose "passed" claims are backed by real evidence files on disk, hashed
    // correctly, in the same directory the CLI is told to read evidence from.
    let good = root.join("good.tsv");
    let mut good_text =
        "format\tmemra-qualification-record-v1\nfamily\tqwen3\npromote_to\tNativeQualified\n"
            .to_string();
    for cell in cells {
        let receipt_name = format!("{}.receipt", cell.replace('.', "-"));
        let content = format!("evidence for {cell}\n");
        std::fs::write(root.join(&receipt_name), &content).unwrap();
        let hash = sha256_hex(content.as_bytes());
        good_text.push_str(&format!(
            "{cell}\tpassed\n{cell}.evidence\t{receipt_name}\n{cell}.evidence_sha256\t{hash}\n"
        ));
    }
    std::fs::write(&good, &good_text).unwrap();
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

    // A readiness-only record: every cell honestly not_exercised, claiming the same promotion.
    let stub = root.join("stub.tsv");
    let mut stub_text =
        "format\tmemra-qualification-record-v1\nfamily\tqwen3\npromote_to\tNativeQualified\n"
            .to_string();
    for cell in cells {
        stub_text.push_str(&format!("{cell}\tnot_exercised\n"));
    }
    std::fs::write(&stub, &stub_text).unwrap();
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

/// A record that hand-types `passed` with no evidence file must not pass, even though the
/// record is otherwise well-formed. This is the exact gap revuto flagged on memra#543 PR #897.
#[test]
fn qualify_cli_refuses_passed_cells_with_no_evidence_on_disk() {
    let root = std::env::temp_dir().join(format!(
        "memra-cli-qualify-cli-no-evidence-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let record = root.join("no-evidence.tsv");
    std::fs::write(
        &record,
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
        .arg(&record)
        .output()
        .unwrap();
    assert!(!output.status.success(), "{output:?}");
    let err = String::from_utf8_lossy(&output.stderr);
    assert!(err.contains("evidence"), "{err}");
    std::fs::remove_dir_all(root).unwrap();
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
