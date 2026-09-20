//! Pure G1 driver-accounting rule, made explicit by the lead before residual diagnosis.
#[derive(Debug, PartialEq, Eq)]
pub struct Observation {
    pub exact: bool,
    /// Criteria (a)-(c) only. A nonzero residual still needs a diagnostic class.
    pub bounded_no_leak: bool,
    pub residual: i128,
}
pub fn observe(
    before: usize,
    demoted: usize,
    restored: usize,
    released: usize,
    granule: usize,
) -> Observation {
    let gain = demoted as i128 - before as i128;
    let reacquired = demoted as i128 - restored as i128;
    let residual = released as i128 - gain;
    Observation {
        exact: released > 0 && gain == released as i128 && reacquired == gain,
        bounded_no_leak: released > 0
            && granule > 0
            && gain > 0
            && gain >= released.saturating_sub(granule) as i128
            && restored == before
            && reacquired == gain,
        residual,
    }
}
/// One demote/restore cycle as the gate reads it from `cuMemGetInfo` (free bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cycle {
    pub free_before: usize,
    pub free_after_demote: usize,
    pub free_after_restore: usize,
    pub released: usize,
}

pub const NOT_APPLICABLE_POOLED: &str = "not-applicable-pooled";
pub const NONE: &str = "none";
pub const ONE_TIME_DRIVER_MAPPING_METADATA: &str = "one-time-driver-mapping-metadata";
pub const GROWING_RESIDUAL: &str = "growing-residual";
pub const UNCLASSIFIED: &str = "unclassified";

/// Lead ruling 5 (day 11): a residual is classified by repeating the SAME demote/restore
/// program N >= 2 times in ONE process on ONE cache. Exactly one granule after cycle 1 and
/// identical through cycle N, with (a)-(c) holding every cycle and no drift of the process
/// free baseline, is `one-time-driver-mapping-metadata`; a residual (or the bytes still
/// unreturned against the first baseline) that grows across cycles is `growing-residual`;
/// zero everywhere is `none`. Every other shape stays `unclassified`. The class never
/// promotes: tightening (e) keeps any nonzero residual `g1_reclaim_qualified=false`.
pub fn classify_cycles(cycles: &[Cycle], granule: usize) -> &'static str {
    if granule == 0 {
        return NOT_APPLICABLE_POOLED;
    }
    let Some(first) = cycles.first() else {
        return UNCLASSIFIED;
    };
    if cycles.len() < 2 {
        return UNCLASSIFIED;
    }
    let observations: Vec<Observation> = cycles
        .iter()
        .map(|c| {
            observe(
                c.free_before,
                c.free_after_demote,
                c.free_after_restore,
                c.released,
                granule,
            )
        })
        .collect();
    let residuals: Vec<i128> = observations.iter().map(|o| o.residual).collect();
    let steady = observations.iter().all(|o| o.bounded_no_leak)
        && cycles.iter().all(|c| c.free_before == first.free_before);
    if steady && residuals.iter().all(|&r| r == 0) {
        return NONE;
    }
    if steady && residuals.iter().all(|&r| r == granule as i128) {
        return ONE_TIME_DRIVER_MAPPING_METADATA;
    }
    // Bytes not returned after restore k, measured against the first cycle's baseline.
    let unreturned: Vec<i128> = cycles
        .iter()
        .map(|c| first.free_before as i128 - c.free_after_restore as i128)
        .collect();
    let grows = |series: &[i128]| {
        series.windows(2).all(|w| w[1] >= w[0]) && series[series.len() - 1] > series[0]
    };
    if grows(&residuals) || grows(&unreturned) {
        return GROWING_RESIDUAL;
    }
    UNCLASSIFIED
}

#[cfg(test)]
mod tests {
    use super::*;
    // Target-card day-10 receipts (one RTX PRO 6000 Blackwell), bytes verbatim.
    const PRO_32K: Cycle = Cycle {
        free_before: 84_743_421_952,
        free_after_demote: 85_647_294_464,
        free_after_restore: 84_743_421_952,
        released: 905_969_664,
    };
    const PRO_8K: Cycle = Cycle {
        free_before: 86_098_182_144,
        free_after_demote: 86_299_508_736,
        free_after_restore: 86_098_182_144,
        released: 201_326_592,
    };
    const GRANULE: usize = 2_097_152;
    #[test]
    fn series_classes_follow_the_lead_ruling_and_never_promote() {
        assert_eq!(classify_cycles(&[PRO_8K; 5], GRANULE), NONE);
        for n in 2..=6 {
            assert_eq!(
                classify_cycles(&vec![PRO_32K; n], GRANULE),
                ONE_TIME_DRIVER_MAPPING_METADATA
            );
        }
        // One cycle is a roundtrip, not a series; a pooled cache has no chunk to cycle.
        assert_eq!(classify_cycles(&[PRO_32K], GRANULE), UNCLASSIFIED);
        assert_eq!(classify_cycles(&[], GRANULE), UNCLASSIFIED);
        assert_eq!(classify_cycles(&[PRO_32K; 2], 0), NOT_APPLICABLE_POOLED);
        // The unchanged per-cycle G1 line: any nonzero residual is false.
        let o = observe(
            PRO_32K.free_before,
            PRO_32K.free_after_demote,
            PRO_32K.free_after_restore,
            PRO_32K.released,
            GRANULE,
        );
        assert_eq!(o.residual, GRANULE as i128);
        assert!(o.bounded_no_leak && !o.exact);
        assert!(!(GRANULE != 0 && o.bounded_no_leak && o.residual == 0));
    }
    #[test]
    fn growing_shrinking_and_drifting_series_are_not_one_time_metadata() {
        // Residual k granules on cycle k: the demote shortfall grows.
        let growing: Vec<Cycle> = (1..=4)
            .map(|k| Cycle {
                free_after_demote: PRO_32K.free_after_demote - (k - 1) * GRANULE,
                ..PRO_32K
            })
            .collect();
        assert_eq!(classify_cycles(&growing, GRANULE), GROWING_RESIDUAL);
        // Zero shortfall but one granule never comes back after each restore.
        let leaking: Vec<Cycle> = (0..4)
            .map(|k| Cycle {
                free_before: PRO_8K.free_before - k * GRANULE,
                free_after_demote: PRO_8K.free_after_demote - k * GRANULE,
                free_after_restore: PRO_8K.free_before - (k + 1) * GRANULE,
                released: PRO_8K.released,
            })
            .collect();
        assert_eq!(classify_cycles(&leaking, GRANULE), GROWING_RESIDUAL);
        // First cycle clean, later cycles short: still growing, never the one-time class.
        assert_eq!(classify_cycles(&[PRO_8K, PRO_8K, PRO_8K], GRANULE), NONE);
        let exact_then_short = [
            Cycle {
                free_after_demote: PRO_32K.free_after_demote + GRANULE,
                ..PRO_32K
            },
            PRO_32K,
        ];
        assert_eq!(
            classify_cycles(&exact_then_short, GRANULE),
            GROWING_RESIDUAL
        );
        // Shrinking, oscillating, two granules, or a residual that restore never returns.
        let shrinking = [PRO_32K, exact_then_short[0]];
        assert_eq!(classify_cycles(&shrinking, GRANULE), UNCLASSIFIED);
        let oscillating = [PRO_32K, exact_then_short[0], PRO_32K];
        assert_eq!(classify_cycles(&oscillating, GRANULE), UNCLASSIFIED);
        let two_granules = Cycle {
            free_after_demote: PRO_32K.free_after_demote - GRANULE,
            ..PRO_32K
        };
        assert_eq!(classify_cycles(&[two_granules; 3], GRANULE), UNCLASSIFIED);
        let unreturned = Cycle {
            free_after_restore: PRO_32K.free_after_restore - GRANULE,
            ..PRO_32K
        };
        assert_eq!(classify_cycles(&[unreturned; 3], GRANULE), UNCLASSIFIED);
        // A constant one-granule shortfall with an upward baseline drift is not steady.
        let drifted = Cycle {
            free_before: PRO_32K.free_before + GRANULE,
            free_after_demote: PRO_32K.free_after_demote + GRANULE,
            free_after_restore: PRO_32K.free_after_restore + GRANULE,
            ..PRO_32K
        };
        assert_eq!(classify_cycles(&[PRO_32K, drifted], GRANULE), UNCLASSIFIED);
    }
    #[test]
    fn exact_and_bounded_remain_distinct() {
        assert_eq!(
            observe(100, 120, 100, 20, 2),
            Observation {
                exact: true,
                bounded_no_leak: true,
                residual: 0
            }
        );
        assert_eq!(
            observe(100, 118, 100, 20, 2),
            Observation {
                exact: false,
                bounded_no_leak: true,
                residual: 2
            }
        );
        assert!(!observe(100, 117, 100, 20, 2).bounded_no_leak);
        assert!(!observe(100, 118, 101, 20, 2).bounded_no_leak);
        assert!(observe(100, 121, 100, 20, 2).bounded_no_leak);
        assert!(!observe(100, 99, 100, 20, 2).bounded_no_leak);
        assert!(!observe(100, 100, 100, 0, 2).bounded_no_leak);
        assert!(!observe(100, 120, 100, 20, 0).bounded_no_leak);
    }
}
