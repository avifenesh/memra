//! Day 12 red arms for lead ruling 6: the series-level `g1_reclaim_qualified` and the printed
//! label. Bytes come from the committed day-11 series receipts of both card classes on disk
//! (`reclaim-cycles.tsv`, `cycle-1/active-reclaim.txt`, `ACTIVE.txt`), not from constants.
//! These tests replay the pure rule; they do not label the day-11 cells, which keep the
//! verdicts recorded in `DAY11.md`. The label exists only when a gate rerun prints it.
use super::reclaim_contract::{
    self, Cycle, SERIES_MIN_CYCLES, SeriesVerdict, classify_cycles, observe, series_verdict,
};
use std::collections::BTreeMap;
use std::path::PathBuf;

const LABEL_32K: &str =
    "ACTIVE-32K G1 PASS (classified one-time-driver-mapping-metadata, 5 cycles)";

fn day11(rig: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../research/spill-b-20260919")
        .join(format!("{rig}-day11/cycles-32768/receipt"))
}

fn kv(path: PathBuf) -> BTreeMap<String, String> {
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .filter_map(|line| line.split_once('='))
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect()
}

/// The committed day-11 series of one card: cycles, granule, per-cycle restore identity
/// against the suspended prefix manifest, and the recorded (day-11 rule) series flag.
fn series(rig: &str) -> (Vec<Cycle>, usize, Vec<bool>, BTreeMap<String, String>) {
    let receipt = day11(rig);
    let tsv = std::fs::read_to_string(receipt.join("reclaim-cycles.tsv")).unwrap();
    let mut lines = tsv.lines();
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let column = |name: &str| header.iter().position(|h| *h == name).unwrap();
    let prefix = kv(receipt.join("ACTIVE.txt"))["prefix_state_manifest_sha256"].clone();
    let mut cycles = Vec::new();
    let mut identical = Vec::new();
    for line in lines {
        let f: Vec<&str> = line.split('\t').collect();
        let get = |name: &str| f[column(name)].parse::<usize>().unwrap();
        cycles.push(Cycle {
            free_before: get("free_before_bytes"),
            free_after_demote: get("free_after_demote_bytes"),
            free_after_restore: get("free_after_restore_bytes"),
            released: get("vmm_released_chunk_bytes"),
        });
        identical.push(f[column("restored_prefix_state_manifest_sha256")] == prefix);
        // The per-cycle line recorded on disk is what the unchanged rule computes.
        let o = observe(
            cycles[cycles.len() - 1].free_before,
            cycles[cycles.len() - 1].free_after_demote,
            cycles[cycles.len() - 1].free_after_restore,
            cycles[cycles.len() - 1].released,
            kv(receipt.join("cycle-1/active-reclaim.txt"))["vmm_granularity_bytes"]
                .parse()
                .unwrap(),
        );
        assert_eq!(f[column("g1_reclaim_qualified")], "false");
        assert_eq!(
            f[column("residual_bytes")].parse::<i128>().unwrap(),
            o.residual
        );
        assert!(o.bounded_no_leak && !o.exact && o.residual > 0);
    }
    let granule = kv(receipt.join("cycle-1/active-reclaim.txt"))["vmm_granularity_bytes"]
        .parse()
        .unwrap();
    (
        cycles,
        granule,
        identical,
        kv(receipt.join("reclaim-cycles.txt")),
    )
}

#[test]
fn both_card_classes_day11_series_satisfy_ruling_six_offline_and_nothing_shorter_does() {
    for rig in ["pro-single", "rtx5090"] {
        let (cycles, granule, identical, recorded) = series(rig);
        assert_eq!(cycles.len(), 5, "{rig}");
        assert!(
            identical.iter().all(|&same| same),
            "{rig}: restore identity"
        );
        assert_eq!(
            recorded["residual_series_class"],
            reclaim_contract::ONE_TIME_DRIVER_MAPPING_METADATA
        );
        // Day 11 recorded the AND of the per-cycle lines; the day-11 cells keep that record.
        assert_eq!(recorded["g1_reclaim_qualified"], "false", "{rig}");
        assert_eq!(recorded["free_before_drift_last_bytes"], "0", "{rig}");
        let v = series_verdict(&cycles, granule, &identical, 32768);
        assert_eq!(
            v,
            SeriesVerdict {
                class: reclaim_contract::ONE_TIME_DRIVER_MAPPING_METADATA,
                g1_reclaim_qualified: "true",
                label: Some(LABEL_32K.to_string()),
            },
            "{rig}"
        );
        // Every prefix shorter than five cycles of the SAME bytes stays false, unlabelled.
        for n in 1..SERIES_MIN_CYCLES {
            let v = series_verdict(&cycles[..n], granule, &identical[..n], 32768);
            assert_eq!(v.g1_reclaim_qualified, "false", "{rig} n={n}");
            assert_eq!(v.label, None, "{rig} n={n}");
        }
        // The per-cycle line is untouched by the ruling: still false in every cycle.
        for c in &cycles {
            let o = observe(
                c.free_before,
                c.free_after_demote,
                c.free_after_restore,
                c.released,
                granule,
            );
            assert!(!(granule != 0 && o.bounded_no_leak && o.residual == 0));
        }
    }
}

#[test]
fn ruling_six_red_arms_on_the_committed_bytes() {
    let (cycles, granule, identical, _) = series("pro-single");
    let base = cycles[0];
    // One cycle drifting (baseline moved by one granule): false, and the class leaves one-time.
    for k in 0..5 {
        let mut drifted = cycles.clone();
        drifted[k] = Cycle {
            free_before: base.free_before + granule,
            free_after_demote: base.free_after_demote + granule,
            free_after_restore: base.free_after_restore + granule,
            ..base
        };
        let v = series_verdict(&drifted, granule, &identical, 32768);
        assert_eq!(
            (v.g1_reclaim_qualified, v.label),
            ("false", None),
            "drift at {k}"
        );
        assert_ne!(v.class, reclaim_contract::ONE_TIME_DRIVER_MAPPING_METADATA);
    }
    // Growing residual: false.
    let growing: Vec<Cycle> = (0..5)
        .map(|k| Cycle {
            free_after_demote: base.free_after_demote - k * granule,
            ..base
        })
        .collect();
    let v = series_verdict(&growing, granule, &identical, 32768);
    assert_eq!(v.class, reclaim_contract::GROWING_RESIDUAL);
    assert_eq!((v.g1_reclaim_qualified, v.label), ("false", None));
    // Any other class: two granules every cycle, or one granule never returned on restore.
    for shape in [
        Cycle {
            free_after_demote: base.free_after_demote - granule,
            ..base
        },
        Cycle {
            free_after_restore: base.free_after_restore - granule,
            ..base
        },
    ] {
        let v = series_verdict(&[shape; 5], granule, &identical, 32768);
        assert_eq!(v.class, reclaim_contract::UNCLASSIFIED);
        assert_eq!((v.g1_reclaim_qualified, v.label), ("false", None));
    }
    // One cycle whose restored prefix differs: false, even with the class intact.
    for k in 0..5 {
        let mut restored = identical.clone();
        restored[k] = false;
        let v = series_verdict(&cycles, granule, &restored, 32768);
        assert_eq!(v.class, reclaim_contract::ONE_TIME_DRIVER_MAPPING_METADATA);
        assert_eq!(
            (v.g1_reclaim_qualified, v.label),
            ("false", None),
            "restore {k}"
        );
    }
    // One byte moved anywhere in the series: the label disappears.
    for k in 0..5 {
        for field in 0..4 {
            let mut mutated = cycles.clone();
            match field {
                0 => mutated[k].free_before += 1,
                1 => mutated[k].free_after_demote += 1,
                2 => mutated[k].free_after_restore += 1,
                _ => mutated[k].released += 1,
            }
            let v = series_verdict(&mutated, granule, &identical, 32768);
            assert_eq!(v.label, None, "cycle {k} field {field}");
            assert_eq!(v.g1_reclaim_qualified, "false", "cycle {k} field {field}");
        }
    }
    // Pooled: not-applicable-pooled, no label, whatever the bytes.
    let pooled = series_verdict(&cycles, 0, &identical, 32768);
    assert_eq!(pooled.class, reclaim_contract::NOT_APPLICABLE_POOLED);
    assert_eq!(pooled.g1_reclaim_qualified, "not-applicable-pooled");
    assert_eq!(pooled.label, None);
    assert_eq!(
        classify_cycles(&cycles, 0),
        reclaim_contract::NOT_APPLICABLE_POOLED
    );
}

#[test]
fn an_exact_series_is_true_without_a_new_label_and_the_label_names_its_context() {
    // The day-10 8k target-card roundtrip, repeated five times: class none, true, no label.
    let receipt = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../research/spill-b-20260919/pro-single-day10/vmm-8192/receipt/active-reclaim.txt",
    );
    let fields = kv(receipt);
    let get = |key: &str| fields[key].parse::<usize>().unwrap();
    let exact = Cycle {
        free_before: get("free_before_bytes"),
        free_after_demote: get("free_after_demote_bytes"),
        free_after_restore: get("free_after_restore_bytes"),
        released: get("vmm_released_chunk_bytes"),
    };
    let granule = get("vmm_granularity_bytes");
    let v = series_verdict(&[exact; 5], granule, &[true; 5], 8192);
    assert_eq!(
        v,
        SeriesVerdict {
            class: reclaim_contract::NONE,
            g1_reclaim_qualified: "true",
            label: None,
        }
    );
    // The label's tag follows the committed context, so a 16k series can never print 32K.
    let (cycles, g32, identical, _) = series("pro-single");
    assert_eq!(
        series_verdict(&cycles, g32, &identical, 16384)
            .label
            .as_deref(),
        Some("ACTIVE-16K G1 PASS (classified one-time-driver-mapping-metadata, 5 cycles)")
    );
    assert_eq!(
        series_verdict(&cycles, g32, &identical, 32768)
            .label
            .as_deref(),
        Some(LABEL_32K)
    );
}
