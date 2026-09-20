//! AD-556-1: exercise the actual `model inspect` CLI and its placement/census receipts.
use std::process::Command;

#[test]
fn step_rope_compact_and_full_shapes_pass_cli_census_and_bad_shapes_fail() {
    let root = std::env::temp_dir().join(format!("memra-cli-step-rope-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    for (case, shape, accepted) in [
        ("compact", vec![32], true),
        ("full", vec![64], true),
        ("short", vec![1], false),
        ("intermediate", vec![48], false),
        ("oversized", vec![65], false),
        ("matrix", vec![2, 16], false),
        ("compact_rank2", vec![32, 1], false),
        ("full_rank2", vec![64, 1], false),
    ] {
        let model = root.join(format!("{case}.gguf"));
        memra_gguf::micro_gguf::write_step35_rope_contract_fixture(&model, &shape).unwrap();
        let out = root.join(case);
        let result = Command::new(env!("CARGO_BIN_EXE_memra"))
            .args(["model", "inspect"])
            .arg(&model)
            .args(["--against", "step35", "--out"])
            .arg(&out)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&result.stderr);
        if accepted {
            assert!(result.status.success(), "{case}: {stderr}");
            assert!(String::from_utf8_lossy(&result.stdout).contains("family=step35"));
            assert!(out.join("placement-checkpoint-2.tsv").is_file(), "{case}");
            assert!(!out.join("contract-error.txt").exists(), "{case}");
        } else {
            assert_eq!(result.status.code(), Some(2), "{case}: {stderr}");
            assert!(
                stderr.contains("RopeFactors") && stderr.contains("shape mismatch"),
                "{case}: {stderr}"
            );
            assert!(!out.join("placement-checkpoint-2.tsv").exists(), "{case}");
            assert!(out.join("contract-error.txt").is_file(), "{case}");
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}
