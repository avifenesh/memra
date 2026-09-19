//! CPU decision model of moe_cache.rs default SLRU (not opt-in LFU/frozen mode).
//! Fixed size classes, free-first admission, pending exclusion, full-class hits.
use crate::contracts::*;
use std::collections::{BTreeMap, VecDeque};

#[derive(Clone, Debug)]
struct Class {
    capacity: u64,
    free: Vec<usize>,
    probation: VecDeque<usize>,
    protected: VecDeque<usize>,
    protected_cap: usize,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlruDecision {
    pub slot: usize,
    pub evicted: Option<BankId>,
}
/// Metadata-only policy. Slots are NOT allocations or GPU permission. The caller
/// retains and charges backing independently until last use, even after eviction.
/// VecDeque is intentionally a readable CPU oracle, not a tuned O(1) replacement.
#[derive(Clone, Debug)]
pub struct SlruPolicy {
    classes: Vec<Class>,
    slot_class: Vec<usize>,
    occupants: Vec<Option<BankId>>,
    reserved: BTreeMap<BankId, usize>,
    table: BTreeMap<BankId, usize>,
}
impl SlruPolicy {
    /// Ascending, unique (capacity, count) classes. Caller uses the native size
    /// class planner; this does not infer a hardware budget or change that plan.
    pub fn new(classes: &[(u64, usize)]) -> Result<Self> {
        if classes.windows(2).any(|w| w[0].0 >= w[1].0)
            || classes.iter().any(|&(cap, count)| cap == 0 || count == 0)
        {
            return Err(Error::InvalidLayout);
        }
        let mut out = Self {
            classes: Vec::new(),
            slot_class: Vec::new(),
            occupants: Vec::new(),
            reserved: BTreeMap::new(),
            table: BTreeMap::new(),
        };
        for &(capacity, count) in classes {
            let start = out.slot_class.len();
            let end = start.checked_add(count).ok_or(Error::Overflow)?;
            out.slot_class.resize(end, out.classes.len());
            out.occupants.resize(end, None);
            out.classes.push(Class {
                capacity,
                free: (start..end).rev().collect(),
                probation: VecDeque::new(),
                protected: VecDeque::new(),
                protected_cap: ((count as f64 * 0.8) as usize).max(1),
            });
        }
        Ok(out)
    }
    pub fn capacity_bytes(&self) -> Result<u64> {
        self.slot_class.iter().try_fold(0u64, |n, &class| {
            n.checked_add(self.classes[class].capacity)
                .ok_or(Error::Overflow)
        })
    }
    pub fn is_empty(&self) -> bool {
        self.table.is_empty() && self.reserved.is_empty()
    }
    pub fn slots(&self) -> usize {
        self.occupants.len()
    }
    pub fn resident(&self, id: &BankId) -> Option<usize> {
        self.table.get(id).copied()
    }
    pub fn pending(&self, id: &BankId) -> bool {
        self.reserved.contains_key(id)
    }
    pub fn hit(&mut self, id: &BankId) -> bool {
        let Some(slot) = self.resident(id) else {
            return false;
        };
        let class = &mut self.classes[self.slot_class[slot]];
        if !class.free.is_empty() {
            return true;
        }
        if let Some(pos) = class.probation.iter().position(|&s| s == slot) {
            class.probation.remove(pos);
        } else if let Some(pos) = class.protected.iter().position(|&s| s == slot) {
            class.protected.remove(pos);
        }
        class.protected.push_back(slot);
        while class.protected.len() > class.protected_cap {
            class
                .probation
                .push_back(class.protected.pop_front().unwrap());
        }
        true
    }
    /// Pending slots are outside both queues. `keep` is native prefetch's current
    /// expert set; demand passes []. No ghost filter: first miss reserves a slot.
    pub fn reserve(
        &mut self,
        id: &BankId,
        required: u64,
        keep: &[BankId],
    ) -> Result<Option<SlruDecision>> {
        id.validate()?;
        if required == 0 {
            return Err(Error::InvalidLayout);
        }
        if self.table.contains_key(id) || self.reserved.contains_key(id) {
            return Err(Error::Conflict);
        }
        let mut selected = None;
        for class in &mut self.classes {
            if class.capacity >= required
                && let Some(slot) = class.free.pop()
            {
                selected = Some(slot);
                break;
            }
        }
        if selected.is_none() {
            for class in &mut self.classes {
                if class.capacity < required {
                    continue;
                }
                for queue in [&mut class.probation, &mut class.protected] {
                    if let Some(pos) = queue.iter().position(|&s| {
                        self.occupants[s]
                            .as_ref()
                            .is_some_and(|id| !keep.contains(id))
                    }) {
                        selected = queue.remove(pos);
                        break;
                    }
                }
                if selected.is_some() {
                    break;
                }
            }
        }
        let Some(slot) = selected else {
            return Ok(None);
        };
        let evicted = self.occupants[slot].take();
        if let Some(old) = &evicted {
            self.table.remove(old);
        }
        self.reserved.insert(id.clone(), slot);
        Ok(Some(SlruDecision { slot, evicted }))
    }
    /// CPU producer completion only. Native caller must establish consumer wait
    /// before this metadata publication; calling this cannot create a ReadyView.
    pub fn publish(&mut self, id: &BankId) -> Result<usize> {
        let slot = self.reserved.remove(id).ok_or(Error::NotFound)?;
        self.occupants[slot] = Some(id.clone());
        self.table.insert(id.clone(), slot);
        self.classes[self.slot_class[slot]]
            .probation
            .push_back(slot);
        Ok(slot)
    }
    /// Only a proven retired producer may return a reserved slot to free.
    /// Unknown completion must leave it reserved/quarantined, not call this.
    pub fn abort_retired(&mut self, id: &BankId) -> Result<()> {
        let slot = self.reserved.remove(id).ok_or(Error::NotFound)?;
        self.classes[self.slot_class[slot]].free.push(slot);
        Ok(())
    }
    pub fn remove(&mut self, id: &BankId) -> bool {
        let Some(slot) = self.table.remove(id) else {
            return false;
        };
        self.occupants[slot] = None;
        let class = &mut self.classes[self.slot_class[slot]];
        class.probation.retain(|&s| s != slot);
        class.protected.retain(|&s| s != slot);
        class.free.push(slot);
        true
    }
    /// Recorded state is LRU→MRU, per class (free is pop-from-end).
    pub fn orders(&self) -> Vec<(Vec<usize>, Vec<usize>, Vec<usize>)> {
        self.classes
            .iter()
            .map(|c| {
                (
                    c.free.clone(),
                    c.probation.iter().copied().collect(),
                    c.protected.iter().copied().collect(),
                )
            })
            .collect()
    }
}
