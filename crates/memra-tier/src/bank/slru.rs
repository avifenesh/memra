//! CPU decision model of moe_cache.rs default SLRU (not opt-in LFU/frozen mode).
//! Fixed size classes, free-first admission, pending exclusion, full-class hits.
//!
//! Day 43 (`research/spill-c-20260919/DAY43.md`): every operation is O(1) apart from the
//! `keep` skip of a prefetch eviction (O(skipped slots), the old scan's visit order). The
//! segments are intrusive doubly-linked lists over slot indices (the `moe_cache.rs` `SlruList`
//! shape) and the maps are `HashMap`s. Decisions and `orders()` equal the VecDeque version this
//! replaced, which the tier's tests keep as the oracle (`tests/bank/slru_oracle.rs`) and drive
//! against this one on the recorded synthetic trace and a randomized one.
use crate::contracts::*;
use std::collections::HashMap;

const NIL: usize = usize::MAX;
const SEG_NONE: u8 = 0;
const SEG_PROBATION: u8 = 1;
const SEG_PROTECTED: u8 = 2;

#[derive(Clone, Copy, Debug)]
struct Link {
    prev: usize,
    next: usize,
    seg: u8,
}
impl Link {
    const NONE: Self = Self {
        prev: NIL,
        next: NIL,
        seg: SEG_NONE,
    };
}

/// One segment, front = LRU, back = MRU.
#[derive(Clone, Debug)]
struct List {
    head: usize,
    tail: usize,
    len: usize,
}
impl List {
    const fn new() -> Self {
        Self {
            head: NIL,
            tail: NIL,
            len: 0,
        }
    }
    fn push_back(&mut self, slot: usize, seg: u8, links: &mut [Link]) {
        debug_assert_eq!(
            links[slot].seg, SEG_NONE,
            "slot {slot} already in a segment"
        );
        links[slot] = Link {
            prev: self.tail,
            next: NIL,
            seg,
        };
        if self.tail == NIL {
            self.head = slot;
        } else {
            links[self.tail].next = slot;
        }
        self.tail = slot;
        self.len += 1;
    }
    fn unlink(&mut self, slot: usize, links: &mut [Link]) {
        let l = links[slot];
        if l.prev == NIL {
            self.head = l.next;
        } else {
            links[l.prev].next = l.next;
        }
        if l.next == NIL {
            self.tail = l.prev;
        } else {
            links[l.next].prev = l.prev;
        }
        links[slot] = Link::NONE;
        self.len -= 1;
    }
    fn pop_front(&mut self, links: &mut [Link]) -> Option<usize> {
        let slot = self.head;
        if slot == NIL {
            return None;
        }
        self.unlink(slot, links);
        Some(slot)
    }
    fn iter<'a>(&self, links: &'a [Link]) -> impl Iterator<Item = usize> + 'a {
        let mut cur = self.head;
        std::iter::from_fn(move || {
            if cur == NIL {
                return None;
            }
            let slot = cur;
            cur = links[slot].next;
            Some(slot)
        })
    }
}

#[derive(Clone, Debug)]
struct Class {
    capacity: u64,
    free: Vec<usize>,
    probation: List,
    protected: List,
    protected_cap: usize,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlruDecision {
    pub slot: usize,
    pub evicted: Option<BankId>,
}
/// Metadata-only policy. Slots are NOT allocations or GPU permission. The caller
/// retains and charges backing independently until last use, even after eviction.
#[derive(Clone, Debug)]
pub struct SlruPolicy {
    classes: Vec<Class>,
    slot_class: Vec<usize>,
    links: Vec<Link>,
    occupants: Vec<Option<BankId>>,
    reserved: HashMap<BankId, usize>,
    table: HashMap<BankId, usize>,
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
            links: Vec::new(),
            occupants: Vec::new(),
            reserved: HashMap::new(),
            table: HashMap::new(),
        };
        for &(capacity, count) in classes {
            let start = out.slot_class.len();
            let end = start.checked_add(count).ok_or(Error::Overflow)?;
            out.slot_class.resize(end, out.classes.len());
            out.links.resize(end, Link::NONE);
            out.occupants.resize(end, None);
            out.classes.push(Class {
                capacity,
                free: (start..end).rev().collect(),
                probation: List::new(),
                protected: List::new(),
                protected_cap: ((count as f64 * 0.8) as usize).max(1),
            });
        }
        out.table.reserve(out.occupants.len());
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
    /// Free slots over every class (day 45: the host fill stops when this reaches zero).
    pub fn free_slots(&self) -> usize {
        self.classes.iter().map(|c| c.free.len()).sum()
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
        match self.links[slot].seg {
            SEG_PROBATION => class.probation.unlink(slot, &mut self.links),
            SEG_PROTECTED => class.protected.unlink(slot, &mut self.links),
            _ => {}
        }
        class
            .protected
            .push_back(slot, SEG_PROTECTED, &mut self.links);
        while class.protected.len > class.protected_cap {
            let demoted = class
                .protected
                .pop_front(&mut self.links)
                .expect("protected is longer than its cap");
            class
                .probation
                .push_back(demoted, SEG_PROBATION, &mut self.links);
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
            let occupants = &self.occupants;
            let links = &mut self.links;
            'classes: for class in &mut self.classes {
                if class.capacity < required {
                    continue;
                }
                for list in [&mut class.probation, &mut class.protected] {
                    let victim = list.iter(links).find(|&s| {
                        occupants[s]
                            .as_ref()
                            .is_some_and(|occupant| !keep.contains(occupant))
                    });
                    if let Some(slot) = victim {
                        list.unlink(slot, links);
                        selected = Some(slot);
                        break 'classes;
                    }
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
    /// Reserve a FREE slot only (day 45, the host fill): the first class, in order, that holds
    /// `required` and has a free slot, exactly `reserve`'s free-first choice; `None` instead of
    /// an eviction when every fitting class is full. The fill never displaces a resident record.
    pub fn reserve_free(&mut self, id: &BankId, required: u64) -> Result<Option<usize>> {
        id.validate()?;
        if required == 0 {
            return Err(Error::InvalidLayout);
        }
        if self.table.contains_key(id) || self.reserved.contains_key(id) {
            return Err(Error::Conflict);
        }
        for class in &mut self.classes {
            if class.capacity >= required
                && let Some(slot) = class.free.pop()
            {
                self.reserved.insert(id.clone(), slot);
                return Ok(Some(slot));
            }
        }
        Ok(None)
    }
    /// CPU producer completion only. Native caller must establish consumer wait
    /// before this metadata publication; calling this cannot create a ReadyView.
    pub fn publish(&mut self, id: &BankId) -> Result<usize> {
        let slot = self.reserved.remove(id).ok_or(Error::NotFound)?;
        self.occupants[slot] = Some(id.clone());
        self.table.insert(id.clone(), slot);
        self.classes[self.slot_class[slot]].probation.push_back(
            slot,
            SEG_PROBATION,
            &mut self.links,
        );
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
        match self.links[slot].seg {
            SEG_PROBATION => class.probation.unlink(slot, &mut self.links),
            SEG_PROTECTED => class.protected.unlink(slot, &mut self.links),
            _ => {}
        }
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
                    c.probation.iter(&self.links).collect(),
                    c.protected.iter(&self.links).collect(),
                )
            })
            .collect()
    }
}
