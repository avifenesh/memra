//! Serial native-cache demand adapter. This changes byte provenance only; the
//! native cache still owns fixed CUDA slots, its intrusive SLRU and numeric code.
//! Optional lookahead is deliberately refused until bank transfers own its fences.
use super::*;
use crate::contracts::*;
use std::collections::BTreeMap;

pub type ExpertDispatchId = (u16, u8, u16);
/// The one mapping between a native cache key and the semantic record id: `(layer, proj,
/// expert)` with `proj` 0/1/2 for Gate/Up/Down. The native cache keys its MTP block at
/// `layer = u16::MAX`, which is the largest layer this key can name. A record that is not a
/// routed expert, or whose layer or original id does not fit the native key, has no
/// dispatch id.
pub fn dispatch_id(record: &RecordId) -> Result<ExpertDispatchId> {
    let RecordId::Expert {
        layer,
        original_id,
        projection,
    } = record
    else {
        return Err(Error::InvalidLayout);
    };
    let proj = match projection {
        Projection::Gate => 0u8,
        Projection::Up => 1,
        Projection::Down => 2,
        Projection::Other(_) => return Err(Error::InvalidLayout),
    };
    Ok((
        u16::try_from(*layer).map_err(|_| Error::InvalidLayout)?,
        proj,
        u16::try_from(*original_id).map_err(|_| Error::InvalidLayout)?,
    ))
}
/// A host ticket remains open across native H2D. Only a physically observed
/// transfer completion may call finish; unknown completion retains this object.
#[derive(Debug)]
pub struct ExpertDemand {
    pub ticket: TransferTicket,
    pub lease: BankLease,
}
/// Day 64 (I15, `research/spill-c-20260919/DAY64.md`): the most records one grouped demand leases, an expert's gate,
/// up and down blocks.
pub const MAX_GROUP: usize = 3;
/// Day 64 (I15): one ticket's host leases for up to `MAX_GROUP` records, in the order demanded. The same rule as
/// `ExpertDemand`: only an observed completion of every transfer that reads them may finish it.
#[derive(Debug)]
pub struct ExpertDemands {
    pub ticket: TransferTicket,
    pub leases: Vec<BankLease>,
}
/// Typed default-OFF native installation point. No synthetic artifact identities,
/// lazy source registration or implicit fallback is permitted here.
pub trait ExpertDispatchBank {
    fn validate(&self, local: ExpertDispatchId, bytes: usize) -> Result<()>;
    fn demand(&mut self, local: ExpertDispatchId, bytes: usize) -> Result<ExpertDemand>;
    fn finish(&mut self, demand: ExpertDemand) -> Result<()>;
    /// The log-only stage clock's `key=value` line, `None` when no clock is installed
    /// (the `--expert-bank-stages` diagnostic; `research/spill-c-20260919/DAY40.md`).
    fn stage_report(&self) -> Option<String> {
        None
    }
    /// Day 50: whether the record behind `local` is resident in the host tier now (a prefetch
    /// takes only host-resident records, so it never reads storage on the owner thread).
    fn host_resident(&self, _local: ExpertDispatchId) -> Result<bool> {
        Ok(false)
    }
    /// Day 64 (I15): one ticket leasing every record of `blocks`, in order. A bank without it refuses.
    fn demand_many(&mut self, _blocks: &[(ExpertDispatchId, usize)]) -> Result<ExpertDemands> {
        Err(Error::Unsupported)
    }
    /// Day 64 (I15): finish a grouped demand's ticket, every lease at once.
    fn finish_many(&mut self, _demands: ExpertDemands) -> Result<()> {
        Err(Error::Unsupported)
    }
}

pub struct SlruExpertDispatch<H: Hotness<ExpertDomain>, R: ExactReader> {
    bank: BankService<ExpertDomain, H, R>,
    ids: BTreeMap<ExpertDispatchId, BankId>,
    request: BudgetRequest,
    epochs: Epochs,
}
impl<H: Hotness<ExpertDomain>, R: ExactReader> SlruExpertDispatch<H, R> {
    pub fn new(
        bank: BankService<ExpertDomain, H, R>,
        ids: BTreeMap<ExpertDispatchId, BankId>,
        request: BudgetRequest,
        epochs: Epochs,
    ) -> Result<Self> {
        if bank.slru_policy().is_none() || ids.is_empty() {
            return Err(Error::InvalidLayout);
        }
        request.validate()?;
        for (&local, id) in &ids {
            if dispatch_id(&id.record)? != local {
                return Err(Error::InvalidLayout);
            }
            let layout = bank.layout(id)?;
            // Scale-bearing records require a multi-plane dispatch adapter first;
            // never discard macro/block scales to qualify the payload-only slice.
            if layout.segments.len() != 1
                || layout.segments[0].role != Role::Payload
                || layout.segments[0].valid_bytes != layout.segments[0].storage_bytes
            {
                return Err(Error::Unsupported);
            }
        }
        Ok(Self {
            bank,
            ids,
            request,
            epochs,
        })
    }
    pub fn bank(&self) -> &BankService<ExpertDomain, H, R> {
        &self.bank
    }
    pub fn into_bank(self) -> BankService<ExpertDomain, H, R> {
        self.bank
    }
    /// Offer one record the host fill read and checksummed off the owner thread (day 45):
    /// `BankService::admit_filled` under this dispatch's request.
    pub fn admit_filled(
        &mut self,
        local: ExpertDispatchId,
        bytes: HostBytes,
        digest: Digest,
    ) -> Result<FillOutcome> {
        let id = self.ids.get(&local).ok_or(Error::NotFound)?.clone();
        self.bank.admit_filled(&id, bytes, digest, &self.request)
    }
}
impl<H: Hotness<ExpertDomain>, R: ExactReader> SlruExpertDispatch<H, R> {
    /// The record behind `local` when its payload holds exactly `bytes`: the one validation
    /// `validate` and `demand` share.
    fn validated(&self, local: ExpertDispatchId, bytes: usize) -> Result<&BankId> {
        let id = self.ids.get(&local).ok_or(Error::NotFound)?;
        let layout = self.bank.layout(id)?;
        if layout.segments[0].valid_bytes != bytes as u64 {
            return Err(Error::InvalidLayout);
        }
        Ok(id)
    }
}
impl<H: Hotness<ExpertDomain>, R: ExactReader> ExpertDispatchBank for SlruExpertDispatch<H, R> {
    fn validate(&self, local: ExpertDispatchId, bytes: usize) -> Result<()> {
        self.validated(local, bytes).map(|_| ())
    }
    fn demand(&mut self, local: ExpertDispatchId, bytes: usize) -> Result<ExpertDemand> {
        // Day 61 (I11 change 4): the id validated is the id staged; one lookup.
        let ids = vec![self.validated(local, bytes)?.clone()];
        let ticket = self.bank.stage(BankBatch {
            ids,
            epochs: self.epochs,
            request: self.request.clone(),
        })?;
        let result = (|| {
            while !self.bank.progress(&ticket)? {}
            let mut leases = self.bank.publish(&ticket, self.epochs)?;
            if leases.len() != 1 {
                return Err(Error::Incomplete);
            }
            Ok(ExpertDemand {
                ticket,
                lease: leases.remove(0),
            })
        })();
        if result.is_err() {
            self.bank.cancel(&ticket)?;
            while !self.bank.progress(&ticket)? {}
            self.bank.finish_host_use(&ticket)?;
            if !self.bank.retire(&ticket)? {
                return Err(Error::NotReady);
            }
            self.bank.acknowledge(&ticket)?;
            self.bank.collect_evicted()?;
        }
        result
    }
    fn stage_report(&self) -> Option<String> {
        self.bank.stage_times().map(BankStageTimes::line)
    }
    fn host_resident(&self, local: ExpertDispatchId) -> Result<bool> {
        let id = self.ids.get(&local).ok_or(Error::NotFound)?;
        Ok(self
            .bank
            .slru_policy()
            .is_some_and(|policy| policy.resident(id).is_some()))
    }
    fn finish(&mut self, demand: ExpertDemand) -> Result<()> {
        // Day 63 (I13 change 3): the three retire-side calls on one pending lookup.
        self.bank.finish_ticket(&demand.ticket)?;
        drop(demand);
        self.bank.collect_evicted()
    }
    fn demand_many(&mut self, blocks: &[(ExpertDispatchId, usize)]) -> Result<ExpertDemands> {
        if blocks.is_empty() {
            return Err(Error::EmptyBatch);
        }
        if blocks.len() > MAX_GROUP {
            return Err(Error::Capacity);
        }
        let ids = blocks
            .iter()
            .map(|&(local, bytes)| self.validated(local, bytes).cloned())
            .collect::<Result<Vec<_>>>()?;
        let ticket = self.bank.stage(BankBatch {
            ids,
            epochs: self.epochs,
            request: self.request.clone(),
        })?;
        let result = (|| {
            while !self.bank.progress(&ticket)? {}
            let leases = self.bank.publish(&ticket, self.epochs)?;
            if leases.len() != blocks.len() {
                return Err(Error::Incomplete);
            }
            Ok(ExpertDemands { ticket, leases })
        })();
        if result.is_err() {
            // The single demand's unwind, unchanged: revoke, pump, retire, acknowledge, collect.
            self.bank.cancel(&ticket)?;
            while !self.bank.progress(&ticket)? {}
            self.bank.finish_host_use(&ticket)?;
            if !self.bank.retire(&ticket)? {
                return Err(Error::NotReady);
            }
            self.bank.acknowledge(&ticket)?;
            self.bank.collect_evicted()?;
        }
        result
    }
    fn finish_many(&mut self, demands: ExpertDemands) -> Result<()> {
        self.bank.finish_ticket(&demands.ticket)?;
        drop(demands);
        self.bank.collect_evicted()
    }
}
