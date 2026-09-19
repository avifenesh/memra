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
    assert!(t.peer_available(0, 1));
    t.edges
        .iter_mut()
        .find(|e| e.source == 0 && e.destination == 1)
        .unwrap()
        .pool_granted = Observation::Known(false);
    assert!(!t.peer_available(0, 1));
    assert!(t.peer_available(1, 0));
    t.edges[0].pool_granted = Observation::Known(true);
    t.edges[0].direct_copy_qualified = Observation::Unknown;
    assert!(!t.peer_available(0, 1));
    assert!(!t.peer_available(0, 0));
    assert!(!t.peer_available(0, 7));
}
#[test]
fn idle_downshift_is_deferred_active_downgrade_refuses() {
    let mut t = topology();
    t.devices[0].current_generation = Observation::Known(1);
    t.devices[0].idle_p8 = Observation::Known(true);
    assert_eq!(t.devices[0].link_health(), LinkHealth::IdleDeferred);
    assert!(!t.peer_available(0, 1));
    t.devices[0].idle_p8 = Observation::Known(false);
    assert_eq!(t.devices[0].link_health(), LinkHealth::Downgraded);
    t.devices[0].current_generation = Observation::Unknown;
    assert_eq!(t.devices[0].link_health(), LinkHealth::Unknown);
}

#[test]
fn observation_expires_with_context_topology_or_binary() {
    let scope = RouteScope {
        source_context: 1,
        destination_context: 2,
        topology_digest: [1; 32],
        binary_digest: [2; 32],
    };
    let scoped = ScopedTopology::new(topology(), scope.clone());
    assert!(scoped.peer_available(0, 1, &scope));
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
