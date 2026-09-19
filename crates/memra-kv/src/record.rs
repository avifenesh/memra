//! Provisional day-1 tier contracts; shared definitions move to memra-tier after lead freeze.
//! Metadata describes bytes, never a codec or an alternate attention program.
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;

pub type Digest = [u8; 32];
pub fn digest(bytes: &[u8]) -> Digest {
    Sha256::digest(bytes).into()
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProgramIdentity {
    pub artifact: Digest,
    pub serialized_plan: Digest,
    pub numeric: Digest,
    pub stream: Digest,
    pub tokenizer: Digest,
    pub template: Digest,
    pub adapter: Digest,
    pub modality: Digest,
    pub position: Digest,
    pub tenant_salt: Digest,
}
impl ProgramIdentity {
    pub fn namespace(&self) -> Digest {
        let mut b = b"memra.kv.namespace.v1\0".to_vec();
        for d in [
            self.artifact,
            self.serialized_plan,
            self.numeric,
            self.stream,
            self.tokenizer,
            self.template,
            self.adapter,
            self.modality,
            self.position,
            self.tenant_salt,
        ] {
            b.extend(d);
        }
        digest(&b)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KvBlockId {
    pub namespace: Digest,
    pub parent: Digest,
    pub tokens: Digest,
    pub start: u64,
    pub end: u64,
    pub group: u32,
    pub owner: u32,
    pub epoch: u64,
}
impl KvBlockId {
    pub fn new(
        program: &ProgramIdentity,
        parent: Digest,
        tokens: &[u32],
        start: u64,
        group: u32,
        owner: u32,
        epoch: u64,
    ) -> Result<Self, TierError> {
        if tokens.is_empty() {
            return Err(TierError::InvalidLayout);
        }
        let mut b = b"memra.kv.tokens.v1\0".to_vec();
        b.extend((tokens.len() as u64).to_le_bytes());
        for t in tokens {
            b.extend(t.to_le_bytes());
        }
        Ok(Self {
            namespace: program.namespace(),
            parent,
            tokens: digest(&b),
            start,
            end: start
                .checked_add(tokens.len() as u64)
                .ok_or(TierError::Overflow)?,
            group,
            owner,
            epoch,
        })
    }
    pub fn hash(&self) -> Digest {
        let mut b = b"memra.kv.block.v1\0".to_vec();
        b.extend(self.namespace);
        b.extend(self.parent);
        b.extend(self.tokens);
        b.extend(self.start.to_le_bytes());
        b.extend(self.end.to_le_bytes());
        b.extend(self.group.to_le_bytes());
        b.extend(self.owner.to_le_bytes());
        b.extend(self.epoch.to_le_bytes());
        digest(&b)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TierError {
    InvalidLayout,
    Overflow,
    Incomplete,
    ProgramMismatch,
    StaleEpoch,
    Capacity,
    Missing,
    Busy,
    WrongState,
    Rejected,
    ShortRead,
    Corrupt,
    Cancelled,
    Backend,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ByteSegment {
    pub group: u32,
    pub page: u64,
    /// Canonical physical owner. Shared consumers are aliases, not extra allocations.
    pub owner: u32,
    pub role: u32,
    pub valid_bytes: u64,
    pub storage_bytes: u64,
    pub alignment: u32,
    pub encoding: Digest,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PageRequirement {
    AllPages,
    TrailingPages(u64),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupRequirement {
    pub group: u32,
    pub owner: u32,
    pub role: u32,
    pub pages: PageRequirement,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordLayout {
    pub version: u32,
    pub page_count: u64,
    pub segments: Vec<ByteSegment>,
    pub requirements: Vec<GroupRequirement>,
}
impl RecordLayout {
    pub fn validate(&self) -> Result<(), TierError> {
        if self.version != 1 || self.page_count == 0 || self.requirements.is_empty() {
            return Err(TierError::InvalidLayout);
        }
        let mut keys = BTreeSet::new();
        for s in &self.segments {
            if s.valid_bytes == 0
                || s.valid_bytes > s.storage_bytes
                || !s.alignment.is_power_of_two()
                || s.storage_bytes % u64::from(s.alignment) != 0
                || s.page >= self.page_count
                || !keys.insert((s.group, s.owner, s.role, s.page))
            {
                return Err(TierError::InvalidLayout);
            }
        }
        let mut roles = BTreeSet::new();
        for r in &self.requirements {
            if !roles.insert((r.group, r.owner, r.role)) {
                return Err(TierError::InvalidLayout);
            }
            let first = match r.pages {
                PageRequirement::AllPages => 0,
                PageRequirement::TrailingPages(n) if n > 0 => self.page_count.saturating_sub(n),
                _ => return Err(TierError::InvalidLayout),
            };
            // Bound validation by provided segments, not a malicious page_count-sized loop.
            let count = keys
                .iter()
                .filter(|&&(g, o, k, p)| (g, o, k) == (r.group, r.owner, r.role) && p >= first)
                .count() as u64;
            if count != self.page_count - first {
                return Err(TierError::Incomplete);
            }
        }
        if self
            .segments
            .iter()
            .any(|s| !roles.contains(&(s.group, s.owner, s.role)))
        {
            return Err(TierError::InvalidLayout);
        }
        self.storage_bytes()?;
        Ok(())
    }
    /// No TP degree parameter: explicit owner/replica inventory is authoritative.
    pub fn storage_bytes(&self) -> Result<u64, TierError> {
        self.segments.iter().try_fold(0u64, |n, s| {
            n.checked_add(s.storage_bytes).ok_or(TierError::Overflow)
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateBundle {
    pub id: KvBlockId,
    pub program: ProgramIdentity,
    pub layout: RecordLayout,
    pub committed_high_water: u64,
    /// (consumer, canonical owner); aliases never add physical bytes.
    pub owner_aliases: Vec<(u32, u32)>,
    /// Hash of each segment's VALID bytes, in layout order. Padding is not state.
    pub checksums: Vec<Digest>,
}
impl StateBundle {
    pub fn validate(&self) -> Result<(), TierError> {
        self.layout.validate()?;
        if self.id.namespace != self.program.namespace() {
            return Err(TierError::ProgramMismatch);
        }
        if self.id.start >= self.id.end || self.committed_high_water < self.id.end {
            return Err(TierError::Incomplete);
        }
        if self.checksums.len() != self.layout.segments.len() {
            return Err(TierError::Incomplete);
        }
        let owners: BTreeSet<_> = self.layout.segments.iter().map(|s| s.owner).collect();
        let mut consumers = BTreeSet::new();
        for &(consumer, owner) in &self.owner_aliases {
            if !owners.contains(&owner) || !consumers.insert(consumer) {
                return Err(TierError::InvalidLayout);
            }
        }
        Ok(())
    }
    pub fn verify(&self, payloads: &[Vec<u8>]) -> Result<(), TierError> {
        self.validate()?;
        if payloads.len() != self.layout.segments.len() {
            return Err(TierError::Incomplete);
        }
        for ((s, hash), bytes) in self
            .layout
            .segments
            .iter()
            .zip(&self.checksums)
            .zip(payloads)
        {
            if bytes.len() as u64 != s.storage_bytes {
                return Err(TierError::ShortRead);
            }
            let n = usize::try_from(s.valid_bytes).map_err(|_| TierError::Overflow)?;
            if digest(&bytes[..n]) != *hash {
                return Err(TierError::Corrupt);
            }
        }
        Ok(())
    }
}
