use memra_tier::peer::topology::*;
#[allow(dead_code)]
#[path = "../contracts/conformance.rs"]
mod conformance;
mod fake;
#[allow(dead_code)]
#[path = "../contracts/support.rs"]
mod support;

fn topology() -> TopologySnapshot {
    use Observation::Known;
    TopologySnapshot {
        devices: (0..4)
            .map(|device| DeviceTopology {
                device,
                numa_node: Known(device / 2),
                cpu_affinity: vec![device * 2],
                current_generation: Known(5),
                max_generation: Known(5),
                current_width: Known(16),
                max_width: Known(16),
                idle_p8: Known(false),
            })
            .collect(),
        edges: (0..4)
            .flat_map(|source| {
                (0..4)
                    .filter(move |&d| d != source)
                    .map(move |destination| PeerEdge {
                        source,
                        destination,
                        can_access: Known(true),
                        context_granted: Known(true),
                        pool_granted: Known(true),
                        direct_copy_qualified: Known(true),
                        shared_fabric: Some("synthetic-shared-uplink".into()),
                    })
            })
            .collect(),
    }
}
#[test]
fn topology_is_directed_grant_bound_and_never_host_bounce() {
    let mut t = topology();
    assert_eq!(t.edges.len(), 12);
    assert!(t.observations_allow(0, 1));
    t.edges
        .iter_mut()
        .find(|e| e.source == 0 && e.destination == 1)
        .unwrap()
        .pool_granted = Observation::Known(false);
    assert!(!t.observations_allow(0, 1));
    assert!(t.observations_allow(1, 0));
    t.edges[0].pool_granted = Observation::Known(true);
    t.edges[0].direct_copy_qualified = Observation::Unknown;
    assert!(!t.observations_allow(0, 1));
    assert!(!t.observations_allow(0, 0));
    assert!(!t.observations_allow(0, 7));
}
#[test]
fn idle_downshift_is_deferred_active_downgrade_refuses() {
    let mut t = topology();
    t.devices[0].current_generation = Observation::Known(1);
    t.devices[0].idle_p8 = Observation::Known(true);
    assert_eq!(t.devices[0].link_health(), LinkHealth::IdleDeferred);
    assert!(!t.observations_allow(0, 1));
    t.devices[0].idle_p8 = Observation::Known(false);
    assert_eq!(t.devices[0].link_health(), LinkHealth::Downgraded);
    t.devices[0].current_generation = Observation::Unknown;
    assert_eq!(t.devices[0].link_health(), LinkHealth::Unknown);
}

#[test]
fn observation_expires_with_context_topology_or_binary() {
    let scope = RouteScope {
        source_device: 0,
        destination_device: 1,
        source_context: 1,
        destination_context: 2,
        topology_digest: [1; 32],
        binary_digest: [2; 32],
    };
    let scoped = ScopedTopology::new(topology(), scope.clone());
    assert!(scoped.peer_available(0, 1, &scope));
    assert!(!scoped.peer_available(1, 0, &scope));
    for field in 0..4 {
        let mut changed = scope.clone();
        match field {
            0 => changed.source_context += 1,
            1 => changed.destination_context += 1,
            2 => changed.topology_digest[0] ^= 1,
            _ => changed.binary_digest[0] ^= 1,
        };
        assert!(!scoped.peer_available(0, 1, &changed));
    }
}

#[test]
fn exported_capacity_is_fail_closed_sized_and_shared() {
    use memra_tier::{contracts::*, peer::test_support::FakePeerCapacity};
    use std::cell::RefCell;
    let gov = support::shared();
    let mut capacity = FakePeerCapacity::new(gov.clone(), 2, 7);
    let mut request = support::request(0, Priority::MandatoryActive);
    request.bytes.device[0] = 8;
    request.bytes.peer[0] = 8;
    let plan = PeerPlan {
        owner_device: 0,
        consumer_device: 1,
        bytes: 8,
        alignment: 4,
        epochs: support::epochs(),
        request,
    };
    assert!(matches!(
        capacity.reserve(plan.clone()),
        Err(Error::Unsupported)
    ));
    capacity
        .set_route(0, 1, true, true, LinkHealth::AtMaximum)
        .unwrap();
    let lease = capacity.reserve(plan.clone()).unwrap();
    assert_eq!(
        capacity.owners[0]
            .resolve::<RefCell<Vec<u8>>>(&lease.device)
            .unwrap()
            .borrow()
            .len(),
        8
    );
    assert_eq!(gov.borrow().used().peer[0], 8);
    let retained = capacity.owners[0].retain(&lease.device).unwrap();
    assert_eq!(capacity.release(&lease), Err(Error::Busy));
    drop(retained);
    capacity.state += 1;
    assert!(matches!(capacity.reserve(plan), Err(Error::StaleEpoch)));
    capacity.release(&lease).unwrap();
    assert_eq!(capacity.release(&lease), Err(Error::ForeignLease));
    assert_eq!(gov.borrow().used(), TierBudget::zero(2));
}
