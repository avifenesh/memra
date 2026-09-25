//! Day 61 (`research/spill-c-20260919/DAY61.md`, I11): the lease protocol's repeated work
//! removed without changing what it charges, leases or refuses.
use super::*;

/// Change 1: the allowance `Catalog::new` memoizes equals the per-ticket formula it replaced
/// (the two canonical encodings plus 1024), for every retained record of both layout classes;
/// a masked record refuses as `record` does.
#[test]
fn the_memoized_allowance_is_the_recomputed_one() {
    for class in [LayoutClass::Uniform, LayoutClass::PerRecord] {
        let (catalog, ids) = catalog(class);
        let mut retained = 0;
        for id in &ids {
            match catalog.record(id) {
                Ok(record) => {
                    let formula = (id.encode().unwrap().len()
                        + record.layout.encode().unwrap().len()
                        + 1024) as u64;
                    assert_eq!(catalog.metadata_allowance(id).unwrap(), formula);
                    retained += 1;
                }
                Err(err) => {
                    assert_eq!(err, Error::MaskedId);
                    assert_eq!(catalog.metadata_allowance(id), Err(Error::MaskedId));
                }
            }
        }
        assert_eq!(retained, 3);
    }
}
