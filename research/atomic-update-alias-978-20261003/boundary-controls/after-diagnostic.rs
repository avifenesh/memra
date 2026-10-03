use std::sync::atomic::{AtomicU64,AtomicUsize,Ordering};
const MAX_EVENT_QUEUE_EVENTS: usize = 256;
const MAX_EVENT_QUEUE_BYTES: usize = 8 * 1024 * 1024;
pub fn site0(counter: &AtomicU64) -> Result<u64,u64> { counter.try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1)) }
pub fn site1(counter: &AtomicU64) -> Result<u64,u64> { counter.try_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1)) }
pub fn site2(counter: &AtomicU64) -> Result<u64,u64> { counter.try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1)) }
pub fn site3(counter: &AtomicU64) -> Result<u64,u64> { counter.try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1)) }
pub fn site4(counter: &AtomicU64) -> Result<u64,u64> { counter.try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1)) }
pub fn site5(counter: &AtomicU64) -> Result<u64,u64> { counter.try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1)) }
pub fn site6(counter: &AtomicUsize) -> Result<usize,usize> { counter.try_update(Ordering::AcqRel, Ordering::Acquire, |v| v.checked_sub(1)) }
pub fn site7(counter: &AtomicUsize) -> Result<usize,usize> { counter.try_update(Ordering::AcqRel, Ordering::Acquire, |v| v.checked_sub(1)) }
pub fn site8(counter: &AtomicUsize) -> Result<usize,usize> { counter.try_update(Ordering::AcqRel, Ordering::Acquire, |events| {
                (events < MAX_EVENT_QUEUE_EVENTS).then_some(events + 1)
            }) }
pub fn site9(counter: &AtomicUsize, retained: usize) -> Result<usize,usize> { counter.try_update(Ordering::AcqRel, Ordering::Acquire, |bytes| {
                bytes
                    .checked_add(retained)
                    .filter(|next| *next <= MAX_EVENT_QUEUE_BYTES)
            }) }
pub fn site10(counter: &AtomicUsize) -> Result<usize,usize> { counter.try_update(
        std::sync::atomic::Ordering::AcqRel,
        std::sync::atomic::Ordering::Acquire,
        |value| value.checked_sub(1),
    ) }
pub fn site11(counter: &AtomicUsize) -> Result<usize,usize> { counter.try_update(
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
            |_| Some(0),
        ) }
