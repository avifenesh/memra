//! Hints select existing immutable records only. They do not mutate demand heat,
//! accepted-token history, router top-k, or the mandatory demand path.
use super::*;
use crate::contracts::*;
use std::collections::{BTreeMap, BTreeSet};

struct BoundedHint<D: BankDomain> {
    bytes: BTreeMap<BankId, u64>,
    max_items: usize,
    max_bytes: u64,
    _domain: std::marker::PhantomData<D>,
}
impl<D: BankDomain> BoundedHint<D> {
    fn new(
        catalog: &Catalog,
        candidates: &[BankId],
        max_items: usize,
        max_bytes: u64,
    ) -> Result<Self> {
        let mut bytes = BTreeMap::new();
        for id in candidates {
            if !D::accepts(&id.record) {
                return Err(Error::InvalidLayout);
            }
            bytes.insert(id.clone(), catalog.record(id)?.layout.storage_bytes()?);
        }
        Ok(Self {
            bytes,
            max_items,
            max_bytes,
            _domain: std::marker::PhantomData,
        })
    }
    fn predict<'a>(&self, ids: impl Iterator<Item = &'a BankId>, limit: usize) -> Vec<BankId> {
        let cap = limit.min(self.max_items);
        let mut remaining = self.max_bytes;
        let mut seen = BTreeSet::new();
        let mut output = Vec::new();
        // Scan is bounded too: duplicate or invalid hint storms cannot monopolize owner.
        for id in ids.take(self.max_items) {
            if output.len() == cap {
                break;
            }
            if let Some(&bytes) = self.bytes.get(id)
                && bytes <= remaining
                && seen.insert(id)
            {
                remaining -= bytes;
                output.push(id.clone());
            }
        }
        output
    }
}

/// Context is the router's already-ranked original expert/projection IDs.
/// The hook never runs top-k itself and cannot alter mandatory selected IDs.
pub struct RouterTopKHint(BoundedHint<ExpertDomain>);
impl RouterTopKHint {
    pub fn new(
        catalog: &Catalog,
        candidates: &[BankId],
        max_items: usize,
        max_bytes: u64,
    ) -> Result<Self> {
        Ok(Self(BoundedHint::new(
            catalog, candidates, max_items, max_bytes,
        )?))
    }
}
impl PrefetchHook<ExpertDomain> for RouterTopKHint {
    type Context = Vec<BankId>;
    fn predict(&self, context: &Self::Context, limit: usize) -> Vec<BankId> {
        self.0.predict(context.iter(), limit)
    }
}

/// Context is a bounded lookahead of row IDs from the native n-gram adapter.
/// Hashing, accepted history and rollback stay outside the residency policy.
pub struct NgramLookaheadHint(BoundedHint<RowDomain>);
impl NgramLookaheadHint {
    pub fn new(
        catalog: &Catalog,
        candidates: &[BankId],
        max_items: usize,
        max_bytes: u64,
    ) -> Result<Self> {
        Ok(Self(BoundedHint::new(
            catalog, candidates, max_items, max_bytes,
        )?))
    }
}
impl PrefetchHook<RowDomain> for NgramLookaheadHint {
    type Context = Vec<BankId>;
    fn predict(&self, context: &Self::Context, limit: usize) -> Vec<BankId> {
        self.0.predict(context.iter(), limit)
    }
}
