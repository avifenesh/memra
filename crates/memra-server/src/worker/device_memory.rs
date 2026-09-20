//! Ordering for model-owned streams and per-device default allocator pools.
//!
//! Kept independent of CUDA so asymmetric ownership/reclaim transitions can be tested
//! without treating a mock allocator as hardware qualification.
use std::collections::{HashMap, HashSet};

/// A shared model runtime may appear once per layer or model. Distinct engines on the
/// same physical device still own different streams and must each be fenced.
pub(super) fn unique_owners<'a, E: 'a>(owners: impl IntoIterator<Item = &'a E>) -> Vec<&'a E> {
    let mut seen = HashSet::new();
    owners
        .into_iter()
        .filter(|owner| seen.insert(std::ptr::from_ref(*owner)))
        .collect()
}

/// Fence all owning streams before trimming any device's default pool. CUDA's default
/// pool is per physical device, so its occupancy/released bytes must be reported once.
/// Fence errors are retained in the report; they never masquerade as a successful fence.
pub(super) fn reclaim<E, R>(
    owners: &[&E],
    device: impl Fn(&E) -> usize,
    synchronize: impl Fn(&E) -> Result<(), String>,
    mut trim: impl FnMut(usize, &E, Vec<String>) -> R,
) -> Vec<R> {
    let owners = unique_owners(owners.iter().copied());
    let mut errors: HashMap<usize, Vec<String>> = HashMap::new();
    for owner in &owners {
        if let Err(error) = synchronize(owner) {
            errors.entry(device(owner)).or_default().push(error);
        }
    }
    let mut seen = HashSet::new();
    let mut reports = Vec::new();
    for owner in owners {
        let ordinal = device(owner);
        if seen.insert(ordinal) {
            reports.push(trim(
                ordinal,
                owner,
                errors.remove(&ordinal).unwrap_or_default(),
            ));
        }
    }
    reports
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[derive(Default)]
    struct Pool {
        used: usize,
        cached: usize,
    }
    struct Owner {
        device: usize,
        pool: Rc<RefCell<Pool>>,
        pending_free: RefCell<usize>,
    }
    impl Owner {
        fn new(device: usize, used: usize, cached: usize) -> Self {
            Self {
                device,
                pool: Rc::new(RefCell::new(Pool { used, cached })),
                pending_free: RefCell::new(0),
            }
        }
        fn sync(&self) -> Result<(), String> {
            let freed = self.pending_free.replace(0);
            let mut pool = self.pool.borrow_mut();
            pool.used -= freed;
            pool.cached += freed;
            Ok(())
        }
    }

    #[test]
    fn peer_only_reclaim_and_refill_preserves_live_owners() {
        let root = Owner::new(0, 100, 0);
        // An ordinal above the old worker's bounded 0..16 scan must be visited.
        let peer = Owner::new(19, 300, 0);
        for _ in 0..3 {
            peer.pending_free.replace(200);
            let rows = reclaim(
                &[&root, &peer, &peer],
                |o| o.device,
                Owner::sync,
                |dev, o, errors| {
                    assert!(errors.is_empty());
                    let mut pool = o.pool.borrow_mut();
                    let released = std::mem::take(&mut pool.cached);
                    (dev, released, pool.used)
                },
            );
            assert_eq!(rows, vec![(0, 0, 100), (19, 200, 100)]);
            // Live state (including pinned leases) survives; the next request refills.
            assert_eq!(root.pool.borrow().used, 100);
            peer.pool.borrow_mut().used += 200;
        }
    }

    #[test]
    fn all_streams_fenced_before_any_shared_pool_trim() {
        let root = Owner::new(0, 100, 0);
        let peer = Owner::new(1, 300, 0);
        let second_stream = Owner {
            device: 1,
            pool: peer.pool.clone(),
            pending_free: RefCell::new(50),
        };
        peer.pending_free.replace(100);
        let events = RefCell::new(Vec::new());
        let rows = reclaim(
            &[&root, &peer, &second_stream, &peer, &root],
            |o| o.device,
            |o| {
                events.borrow_mut().push(("sync", o.device));
                o.sync()
            },
            |dev, o, _| {
                events.borrow_mut().push(("trim", dev));
                let mut pool = o.pool.borrow_mut();
                (dev, std::mem::take(&mut pool.cached), pool.used)
            },
        );
        assert_eq!(rows, vec![(0, 0, 100), (1, 150, 150)]);
        assert_eq!(
            *events.borrow(),
            vec![
                ("sync", 0),
                ("sync", 1),
                ("sync", 1),
                ("trim", 0),
                ("trim", 1)
            ]
        );
    }

    #[test]
    fn failed_peer_fence_is_reported_without_suppressing_other_owners() {
        let root = Owner::new(0, 1, 0);
        let peer = Owner::new(1, 1, 0);
        let rows = reclaim(
            &[&root, &peer],
            |o| o.device,
            |o| {
                if o.device == 1 {
                    Err("peer sync failed".into())
                } else {
                    Ok(())
                }
            },
            |dev, _, errors| (dev, errors),
        );
        assert_eq!(
            rows,
            vec![(0, vec![]), (1, vec!["peer sync failed".into()])]
        );
    }
}
