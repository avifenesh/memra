use std::sync::atomic::{AtomicU64,AtomicUsize,Ordering};
#[derive(Debug,PartialEq)] enum Error {Overflow}
const MAX_EVENT_QUEUE_EVENTS: usize = 256;
const MAX_EVENT_QUEUE_BYTES: usize = 8 * 1024 * 1024;
fn site0(counter: &AtomicU64) -> Result<u64,u64> { counter.try_update(Ordering::Relaxed, Ordering::Relaxed, |n| Some(n.wrapping_add(1))) }
fn site0_adapted(counter: &AtomicU64) -> Result<u64,Error> { counter.try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| Error::Overflow) }
fn site1(counter: &AtomicU64) -> Result<u64,u64> { counter.try_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1)) }
fn site1_adapted(counter: &AtomicU64) -> Result<u64,Error> { counter.try_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
            .map_err(|_| Error::Overflow) }
fn site2(counter: &AtomicU64) -> Result<u64,u64> { counter.try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1)) }
fn site2_adapted(counter: &AtomicU64) -> Result<u64,Error> { counter.try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| Error::Overflow) }
fn site3(counter: &AtomicU64) -> Result<u64,u64> { counter.try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1)) }
fn site3_adapted(counter: &AtomicU64) -> u64 { counter.try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .expect("issuer space exhausted") }
fn site4(counter: &AtomicU64) -> Result<u64,u64> { counter.try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1)) }
fn site4_adapted(counter: &AtomicU64) -> u64 { counter.try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                .expect("materializer id exhausted") }
fn site5(counter: &AtomicU64) -> Result<u64,u64> { counter.try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1)) }
fn site5_adapted(counter: &AtomicU64) -> u64 { counter.try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                .expect("materializer id exhausted") }
fn site6(counter: &AtomicUsize) -> Result<usize,usize> { counter.try_update(Ordering::AcqRel, Ordering::Acquire, |v| v.checked_sub(1)) }
fn site7(counter: &AtomicUsize) -> Result<usize,usize> { counter.try_update(Ordering::AcqRel, Ordering::Acquire, |v| v.checked_sub(1)) }
fn site8(counter: &AtomicUsize) -> Result<usize,usize> { counter.try_update(Ordering::AcqRel, Ordering::Acquire, |events| {
                (events < MAX_EVENT_QUEUE_EVENTS).then_some(events + 1)
            }) }
fn site9(counter: &AtomicUsize, retained: usize) -> Result<usize,usize> { counter.try_update(Ordering::AcqRel, Ordering::Acquire, |bytes| {
                bytes
                    .checked_add(retained)
                    .filter(|next| *next <= MAX_EVENT_QUEUE_BYTES)
            }) }
fn site10(counter: &AtomicUsize) -> Result<usize,usize> { counter.try_update(
        std::sync::atomic::Ordering::AcqRel,
        std::sync::atomic::Ordering::Acquire,
        |value| value.checked_sub(1),
    ) }
fn site11(counter: &AtomicUsize) -> Result<usize,usize> { counter.try_update(
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
            |_| Some(0),
        ) }
#[test] fn site0_overflow_error_adapter() {let c=AtomicU64::new(u64::MAX);assert_eq!(site0_adapted(&c),Err(Error::Overflow));assert_eq!(c.load(Ordering::Relaxed),u64::MAX);}
#[test] fn site0_overflow_and_contention() {
let c=AtomicU64::new(u64::MAX-1); assert_eq!(site0(&c),Ok(u64::MAX-1)); assert_eq!(site0(&c),Err(u64::MAX)); assert_eq!(c.load(Ordering::Relaxed),u64::MAX);
let c=AtomicU64::new(0); std::thread::scope(|s|{for _ in 0..4{s.spawn(||{for _ in 0..256{site0(&c).unwrap();}});}}); assert_eq!(c.load(Ordering::Relaxed),1024);
}
#[test] fn site1_overflow_error_adapter() {let c=AtomicU64::new(u64::MAX);assert_eq!(site1_adapted(&c),Err(Error::Overflow));assert_eq!(c.load(Ordering::Relaxed),u64::MAX);}
#[test] fn site1_overflow_and_contention() {
let c=AtomicU64::new(u64::MAX-1); assert_eq!(site1(&c),Ok(u64::MAX-1)); assert_eq!(site1(&c),Err(u64::MAX)); assert_eq!(c.load(Ordering::Relaxed),u64::MAX);
let c=AtomicU64::new(0); std::thread::scope(|s|{for _ in 0..4{s.spawn(||{for _ in 0..256{site1(&c).unwrap();}});}}); assert_eq!(c.load(Ordering::Relaxed),1024);
}
#[test] fn site2_overflow_error_adapter() {let c=AtomicU64::new(u64::MAX);assert_eq!(site2_adapted(&c),Err(Error::Overflow));assert_eq!(c.load(Ordering::Relaxed),u64::MAX);}
#[test] fn site2_overflow_and_contention() {
let c=AtomicU64::new(u64::MAX-1); assert_eq!(site2(&c),Ok(u64::MAX-1)); assert_eq!(site2(&c),Err(u64::MAX)); assert_eq!(c.load(Ordering::Relaxed),u64::MAX);
let c=AtomicU64::new(0); std::thread::scope(|s|{for _ in 0..4{s.spawn(||{for _ in 0..256{site2(&c).unwrap();}});}}); assert_eq!(c.load(Ordering::Relaxed),1024);
}
#[test] fn site3_overflow_panic_adapter() {let c=AtomicU64::new(u64::MAX);let p=std::panic::catch_unwind(||site3_adapted(&c)).expect_err("must retain exhausted panic");let msg=p.downcast_ref::<String>().map(String::as_str).or_else(||p.downcast_ref::<&str>().copied()).unwrap();assert!(msg.contains("issuer space exhausted"));assert_eq!(c.load(Ordering::Relaxed),u64::MAX);}
#[test] fn site3_overflow_and_contention() {
let c=AtomicU64::new(u64::MAX-1); assert_eq!(site3(&c),Ok(u64::MAX-1)); assert_eq!(site3(&c),Err(u64::MAX)); assert_eq!(c.load(Ordering::Relaxed),u64::MAX);
let c=AtomicU64::new(0); std::thread::scope(|s|{for _ in 0..4{s.spawn(||{for _ in 0..256{site3(&c).unwrap();}});}}); assert_eq!(c.load(Ordering::Relaxed),1024);
}
#[test] fn site4_overflow_panic_adapter() {let c=AtomicU64::new(u64::MAX);let p=std::panic::catch_unwind(||site4_adapted(&c)).expect_err("must retain exhausted panic");let msg=p.downcast_ref::<String>().map(String::as_str).or_else(||p.downcast_ref::<&str>().copied()).unwrap();assert!(msg.contains("materializer id exhausted"));assert_eq!(c.load(Ordering::Relaxed),u64::MAX);}
#[test] fn site4_overflow_and_contention() {
let c=AtomicU64::new(u64::MAX-1); assert_eq!(site4(&c),Ok(u64::MAX-1)); assert_eq!(site4(&c),Err(u64::MAX)); assert_eq!(c.load(Ordering::Relaxed),u64::MAX);
let c=AtomicU64::new(0); std::thread::scope(|s|{for _ in 0..4{s.spawn(||{for _ in 0..256{site4(&c).unwrap();}});}}); assert_eq!(c.load(Ordering::Relaxed),1024);
}
#[test] fn site5_overflow_panic_adapter() {let c=AtomicU64::new(u64::MAX);let p=std::panic::catch_unwind(||site5_adapted(&c)).expect_err("must retain exhausted panic");let msg=p.downcast_ref::<String>().map(String::as_str).or_else(||p.downcast_ref::<&str>().copied()).unwrap();assert!(msg.contains("materializer id exhausted"));assert_eq!(c.load(Ordering::Relaxed),u64::MAX);}
#[test] fn site5_overflow_and_contention() {
let c=AtomicU64::new(u64::MAX-1); assert_eq!(site5(&c),Ok(u64::MAX-1)); assert_eq!(site5(&c),Err(u64::MAX)); assert_eq!(c.load(Ordering::Relaxed),u64::MAX);
let c=AtomicU64::new(0); std::thread::scope(|s|{for _ in 0..4{s.spawn(||{for _ in 0..256{site5(&c).unwrap();}});}}); assert_eq!(c.load(Ordering::Relaxed),1024);
}
#[test] fn site6_underflow_and_contention() {let c=AtomicUsize::new(0);assert_eq!(site6(&c),Err(0));assert_eq!(c.load(Ordering::Acquire),0);c.store(1024,Ordering::Release);std::thread::scope(|s|{for _ in 0..4{s.spawn(||{for _ in 0..256{site6(&c).unwrap();}});}});assert_eq!(c.load(Ordering::Acquire),0);assert_eq!(site6(&c),Err(0));}
#[test] fn site7_underflow_and_contention() {let c=AtomicUsize::new(0);assert_eq!(site7(&c),Err(0));assert_eq!(c.load(Ordering::Acquire),0);c.store(1024,Ordering::Release);std::thread::scope(|s|{for _ in 0..4{s.spawn(||{for _ in 0..256{site7(&c).unwrap();}});}});assert_eq!(c.load(Ordering::Acquire),0);assert_eq!(site7(&c),Err(0));}
#[test] fn site8_queue_bound_and_contention() {let c=AtomicUsize::new(MAX_EVENT_QUEUE_EVENTS-1);assert_eq!(site8(&c),Ok(MAX_EVENT_QUEUE_EVENTS-1));assert_eq!(site8(&c),Err(MAX_EVENT_QUEUE_EVENTS));assert_eq!(c.load(Ordering::Acquire),MAX_EVENT_QUEUE_EVENTS);c.store(0,Ordering::Release);std::thread::scope(|s|{for _ in 0..4{s.spawn(||{for _ in 0..128{let _=site8(&c);}});}});assert_eq!(c.load(Ordering::Acquire),MAX_EVENT_QUEUE_EVENTS);}
#[test] fn site9_byte_bound_and_arithmetic_overflow() {let c=AtomicUsize::new(MAX_EVENT_QUEUE_BYTES-1);assert_eq!(site9(&c, 1),Ok(MAX_EVENT_QUEUE_BYTES-1));assert_eq!(site9(&c, 1),Err(MAX_EVENT_QUEUE_BYTES));assert_eq!(c.load(Ordering::Acquire),MAX_EVENT_QUEUE_BYTES);c.store(usize::MAX,Ordering::Release);assert_eq!(site9(&c, 1),Err(usize::MAX));assert_eq!(c.load(Ordering::Acquire),usize::MAX);assert_eq!(site9(&c,0),Err(usize::MAX));}
#[test] fn site10_underflow_and_contention() {let c=AtomicUsize::new(0);assert_eq!(site10(&c),Err(0));assert_eq!(c.load(Ordering::Acquire),0);c.store(1024,Ordering::Release);std::thread::scope(|s|{for _ in 0..4{s.spawn(||{for _ in 0..256{site10(&c).unwrap();}});}});assert_eq!(c.load(Ordering::Acquire),0);assert_eq!(site10(&c),Err(0));}
#[test] fn site11_reset_returns_previous_value() {for n in [0,1,usize::MAX] {let c=AtomicUsize::new(n);assert_eq!(site11(&c),Ok(n));assert_eq!(c.load(Ordering::Acquire),0);}}
