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
            && gain <= released as i128
            && restored == before
            && reacquired == gain,
        residual,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
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
        assert!(!observe(100, 121, 100, 20, 2).bounded_no_leak);
        assert!(!observe(100, 99, 100, 20, 2).bounded_no_leak);
        assert!(!observe(100, 100, 100, 0, 2).bounded_no_leak);
        assert!(!observe(100, 120, 100, 20, 0).bounded_no_leak);
    }
}
