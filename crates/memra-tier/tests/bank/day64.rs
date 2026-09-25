//! Day 64 (`research/spill-c-20260919/DAY64.md`): I14 hashes the record lookups and keeps every answer, refusal and
//! order the ordered maps gave.
use super::*;

fn fixture_entries(class: LayoutClass) -> Vec<(BankId, Option<CatalogRecord>)> {
    let mut entries = Vec::new();
    for (n, q, len) in [
        (9u32, 2, 16u64),
        (47, 3, 16),
        (83, 4, 32),
        (5, 2, 16),
        (61, 3, 16),
    ] {
        let (q, len) = if class == LayoutClass::Uniform {
            (2, 16)
        } else {
            (q, len)
        };
        let l = layout(u64::from(n), q, len);
        entries.push((bank_id(n, &l), Some(record(l))));
    }
    let masked = BankId {
        record: RecordId::Expert {
            layer: 2,
            original_id: 21,
            projection: Projection::Gate,
        },
        ..entries[0].0.clone()
    };
    entries.push((masked, None));
    entries
}

/// I14 change 1: every id's `record` and `metadata_allowance`, and every refusal (an unknown id, a masked one, an id
/// that fails validation), equal what a `BTreeMap` of the same entries answers.
#[test]
fn the_hashed_catalog_answers_as_the_ordered_one() {
    for class in [LayoutClass::Uniform, LayoutClass::PerRecord] {
        let entries = fixture_entries(class);
        let reference: BTreeMap<BankId, Option<CatalogRecord>> = entries.iter().cloned().collect();
        let catalog = Catalog::new(class, entries.clone()).unwrap();
        for (id, want) in &reference {
            match want {
                Some(r) => {
                    let got = catalog.record(id).unwrap();
                    assert_eq!(got.layout, r.layout);
                    assert_eq!(got.checksums, r.checksums);
                    let formula = (id.encode().unwrap().len()
                        + r.layout.encode().unwrap().len()
                        + 1024) as u64;
                    assert_eq!(catalog.metadata_allowance(id).unwrap(), formula);
                }
                None => assert_eq!(catalog.record(id).err(), Some(Error::MaskedId)),
            }
        }
        let unknown = BankId {
            record: RecordId::Expert {
                layer: 3,
                original_id: 9,
                projection: Projection::Gate,
            },
            ..entries[0].0.clone()
        };
        assert_eq!(catalog.record(&unknown).err(), Some(Error::NotFound));
        let invalid = BankId {
            version: 99,
            ..entries[0].0.clone()
        };
        assert_eq!(catalog.record(&invalid).err(), invalid.validate().err());
        // Construction refuses a duplicate id as before.
        let mut twice = entries.clone();
        twice.push(entries[1].clone());
        assert_eq!(Catalog::new(class, twice).err(), Some(Error::Conflict));
    }
}
