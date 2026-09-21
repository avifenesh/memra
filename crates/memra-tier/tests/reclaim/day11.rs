//! Day 11 red arms. The `--reclaim-cycles N` door refuses everything but a diagnostic VMM
//! series, and a residual series replayed from the committed target-card receipts classifies
//! without ever turning a nonzero residual into a G1 pass. Bytes come from the day-10
//! receipts on disk, not from constants that could drift away from them.
use super::cli;
use super::reclaim_contract::{self, Cycle, classify_cycles, observe};
use std::collections::BTreeMap;
use std::path::PathBuf;

fn receipt(label: &str) -> BTreeMap<String, String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../research/spill-b-20260919/pro-single-day10")
        .join(label)
        .join("receipt/active-reclaim.txt");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .filter_map(|line| line.split_once('='))
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect()
}

fn day10_cycle(label: &str) -> (Cycle, usize) {
    let fields = receipt(label);
    let get = |key: &str| fields[key].parse::<usize>().unwrap();
    (
        Cycle {
            free_before: get("free_before_bytes"),
            free_after_demote: get("free_after_demote_bytes"),
            free_after_restore: get("free_after_restore_bytes"),
            released: get("vmm_released_chunk_bytes"),
        },
        get("vmm_granularity_bytes"),
    )
}

fn unchanged_g1_line(c: Cycle, granule: usize) -> bool {
    let o = observe(
        c.free_before,
        c.free_after_demote,
        c.free_after_restore,
        c.released,
        granule,
    );
    granule != 0 && o.bounded_no_leak && o.residual == 0
}

#[test]
fn target_card_receipts_replayed_as_series_classify_without_promotion() {
    let (diag, granule) = day10_cycle("diagnostic-32768");
    let (vmm32, g32) = day10_cycle("vmm-32768");
    assert_eq!(granule, g32);
    assert_eq!(
        receipt("diagnostic-32768")["residual_class"],
        "unclassified"
    );
    // The day-10 single roundtrips: one granule short on both 32k cells, not a series yet.
    for c in [diag, vmm32] {
        let o = observe(
            c.free_before,
            c.free_after_demote,
            c.free_after_restore,
            c.released,
            granule,
        );
        assert_eq!(o.residual, granule as i128);
        assert!(o.bounded_no_leak && !o.exact);
        assert_eq!(
            classify_cycles(&[c], granule),
            reclaim_contract::UNCLASSIFIED
        );
        assert!(!unchanged_g1_line(c, granule));
    }
    // Repeating the identical one-granule roundtrip is the lead-ruled metadata class,
    // and the per-cycle G1 line stays false in every cycle. The class alone never promotes;
    // the series-level lift of lead ruling 6 (N >= 5) is `series_verdict`, tested in day12.rs.
    for n in 2..=5 {
        assert_eq!(
            classify_cycles(&vec![diag; n], granule),
            reclaim_contract::ONE_TIME_DRIVER_MAPPING_METADATA
        );
        assert!(
            !vec![diag; n]
                .into_iter()
                .any(|c| unchanged_g1_line(c, granule))
        );
    }
    // The 8k cells (both PASS on day 10) repeated: no residual anywhere.
    for label in ["vmm-8192", "injected-8192"] {
        let (c, g) = day10_cycle(label);
        assert_eq!(receipt(label)["g1_reclaim_qualified"], "true");
        assert_eq!(classify_cycles(&[c; 5], g), reclaim_contract::NONE);
        assert!(unchanged_g1_line(c, g));
    }
    // The pooled control has no chunk to cycle.
    let pooled = receipt("pooled-8192");
    assert_eq!(pooled["vmm_granularity_bytes"], "0");
    assert_eq!(
        classify_cycles(&[diag; 2], 0),
        reclaim_contract::NOT_APPLICABLE_POOLED
    );
}

#[test]
fn a_growing_or_unreturned_residual_is_never_the_one_time_class() {
    let (diag, granule) = day10_cycle("diagnostic-32768");
    let one_more_short = |k: usize| Cycle {
        free_after_demote: diag.free_after_demote - k * granule,
        ..diag
    };
    let growing: Vec<Cycle> = (0..5).map(one_more_short).collect();
    assert_eq!(
        classify_cycles(&growing, granule),
        reclaim_contract::GROWING_RESIDUAL
    );
    // Every red arm: mutate one byte of the constant series and the class must move.
    for mutated in [
        vec![diag, one_more_short(1)],
        vec![diag, diag, one_more_short(1)],
    ] {
        assert_eq!(
            classify_cycles(&mutated, granule),
            reclaim_contract::GROWING_RESIDUAL
        );
    }
    let never_returned = Cycle {
        free_after_restore: diag.free_after_restore - 1,
        ..diag
    };
    assert_eq!(
        classify_cycles(&[never_returned; 3], granule),
        reclaim_contract::UNCLASSIFIED
    );
    let shrinking = [one_more_short(1), diag];
    assert_eq!(
        classify_cycles(&shrinking, granule),
        reclaim_contract::UNCLASSIFIED
    );
    let (vmm8, g8) = day10_cycle("vmm-8192");
    let leaking: Vec<Cycle> = (0..3)
        .map(|k| Cycle {
            free_before: vmm8.free_before - k * g8,
            free_after_demote: vmm8.free_after_demote - k * g8,
            free_after_restore: vmm8.free_before - (k + 1) * g8,
            released: vmm8.released,
        })
        .collect();
    assert_eq!(
        classify_cycles(&leaking, g8),
        reclaim_contract::GROWING_RESIDUAL
    );
}

#[test]
fn the_reclaim_cycles_door_refuses_outside_a_diagnostic_vmm_series() {
    let series = |extra: &[&str]| -> Vec<String> {
        let mut args: Vec<String> = "--artifact fixture.gguf --case active --context 32768 --tiers host --same-program --kv-allocator vmm --reclaim-diagnostic --out receipts"
            .split_whitespace()
            .map(str::to_owned)
            .collect();
        args.extend(extra.iter().map(|s| s.to_string()));
        args
    };
    let accepted = cli::parse(series(&["--reclaim-cycles", "5"])).unwrap();
    assert_eq!(accepted.reclaim_cycles, Some(5));
    assert!(accepted.reclaim_diagnostic);
    assert_eq!(accepted.kv_allocator, cli::KvAllocator::Vmm);
    assert_eq!(cli::parse(series(&[])).unwrap().reclaim_cycles, None);
    let refusals: [(&str, Vec<String>); 6] = [
        ("count below two", series(&["--reclaim-cycles", "1"])),
        ("junk count", series(&["--reclaim-cycles", "five"])),
        ("missing count", series(&["--reclaim-cycles"])),
        (
            "duplicate",
            series(&["--reclaim-cycles", "2", "--reclaim-cycles", "2"]),
        ),
        ("without the diagnostic", {
            let mut args = series(&["--reclaim-cycles", "2"]);
            args.retain(|a| a != "--reclaim-diagnostic");
            args
        }),
        ("pooled allocator", {
            let mut args = series(&["--reclaim-cycles", "2"]);
            let index = args.iter().position(|a| a == "vmm").unwrap();
            args[index] = "pooled".into();
            args
        }),
    ];
    for (name, args) in refusals {
        let error = cli::parse(args).unwrap_err();
        assert!(error.starts_with("REFUSED: "), "{name}: {error}");
        // Exit-2 refusal contract: the collector reads the unwrapped last line.
        assert_eq!(cli::diagnostic(&error), error, "{name}");
    }
    assert!(cli::USAGE.contains("[--reclaim-diagnostic [--reclaim-cycles N]]"));
}
