//! Load-time, exact source installation against the frozen ObjectStore API.
use super::*;
use crate::{contracts::*, pool::FakePinnedPool};
use std::collections::{BTreeMap, BTreeSet};

/// Authoritative loader expectation, from the locked source byte manifest, NOT
/// inferred from a successful lookup. `key.layout` describes the source object's
/// encoding; per-record layout digests remain independently bound by Catalog.
#[derive(Clone, Debug)]
pub struct BankSourceSpec {
    pub tensor: TensorId,
    pub key: ObjectKey,
    pub valid_bytes: u64,
}
/// Installed bank and byte source are created together; no missing-source fallback.
/// This is CPU loader-contract evidence, not native HostExps loader wiring.
pub struct BankSource<S: ObjectStore> {
    catalog: Catalog,
    reader: ObjectReader<S>,
}
impl<S: ObjectStore> BankSource<S> {
    pub fn install(
        catalog: Catalog,
        expected: Vec<BankSourceSpec>,
        store: S,
        supplied: Vec<(TensorId, ObjectKey)>,
        pool: FakePinnedPool,
        request: BudgetRequest,
        queue_charge: &ChargedLease,
    ) -> Result<Self> {
        let mut specs = BTreeMap::new();
        for spec in expected {
            spec.tensor.validate()?;
            spec.key.validate()?;
            if spec.key.artifact != spec.tensor.artifact
                || spec.key.semantic_id != spec.tensor.identity()?
                || spec.valid_bytes == 0
            {
                return Err(Error::InvalidLayout);
            }
            if specs.insert(spec.tensor.clone(), spec).is_some() {
                return Err(Error::Conflict);
            }
        }
        let mut required = BTreeSet::new();
        for record in catalog.records() {
            for segment in &record.layout.segments {
                let tensor = segment.tensor.as_ref().ok_or(Error::InvalidLayout)?;
                let spec = specs.get(tensor).ok_or(Error::NotFound)?;
                if segment
                    .offset
                    .checked_add(segment.storage_bytes)
                    .ok_or(Error::Overflow)?
                    > spec.valid_bytes
                {
                    return Err(Error::InvalidLayout);
                }
                required.insert(tensor.clone());
            }
        }
        // Masked records have no sources; unexpected sources cannot silently revive them.
        if required.len() != specs.len() {
            return Err(Error::Conflict);
        }
        let mut seen = BTreeSet::new();
        for (tensor, key) in &supplied {
            if !seen.insert(tensor.clone()) {
                return Err(Error::Conflict);
            }
            let spec = specs.get(tensor).ok_or(Error::NotFound)?;
            if &spec.key != key {
                return Err(Error::InvalidLayout);
            }
            let manifest = store.lookup(key)?.ok_or(Error::NotFound)?;
            manifest.validate()?;
            if &manifest.key != key || manifest.valid_bytes != spec.valid_bytes {
                return Err(Error::InvalidLayout);
            }
        }
        if seen != required {
            return Err(Error::NotFound);
        }
        let reader = ObjectReader::new(store, supplied, pool, request, queue_charge)?;
        // Lookup is advisory: reject changed lengths even if a faulty backend
        // changes its answer between validation and reader construction.
        for spec in specs.values() {
            if reader.storage_bytes(&spec.tensor)? != spec.valid_bytes {
                return Err(Error::InvalidLayout);
            }
        }
        Ok(Self { catalog, reader })
    }
    pub fn into_parts(self) -> (Catalog, ObjectReader<S>) {
        (self.catalog, self.reader)
    }
}

/// Exact HostExps source bound: split storage already addresses this expert, so
/// its logical layout offset MUST NOT be added again. No slicing or byte fallback.
pub fn validate_bank_source_extent(
    offset: u64,
    len: u64,
    available: u64,
    split: bool,
) -> Result<()> {
    if len == 0 || (split && len != available) {
        return Err(Error::InvalidLayout);
    }
    let start = if split { 0 } else { offset };
    if start.checked_add(len).ok_or(Error::Overflow)? > available {
        return Err(Error::InvalidLayout);
    }
    Ok(())
}
