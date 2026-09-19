use memra_tier::placement::*;

// 07 §3 sizing-only fixture; opaque byte census, NOT a model adapter or runtime program.
fn fixture(record_bytes: u64, layers: [u64; 4]) -> PlacementPlan {
    let index_layers = [2, 1, 3, 2];
    let devices = layers.into_iter().enumerate().map(|(i, n)| DevicePlan {
        device: i as u32,
        capacity_bytes: 96_000_000_000,
        weights: vec![
            WeightAllocation { bytes: n * 7_219_445_760, estimate: None },
            WeightAllocation { bytes: n * 126_743_040 + index_layers[i] * 10_229_376,
                estimate: Some("07 ALLOC-4 attention/indexer residual allocation; not per-layer header census".into()) },
            WeightAllocation { bytes: n * 35_424_000, estimate: None },
            WeightAllocation { bytes: n * 7_888_088 + if i == 3 { 40_960 } else { 0 },
                estimate: Some("07 ALLOC-4 norm/router residual allocation".into()) },
            WeightAllocation { bytes: if i < 2 { 157_521_920 } else { 0 }, estimate: None },
            WeightAllocation { bytes: if i == 0 || i == 3 { 1_323_827_200 } else { 0 }, estimate: None },
            WeightAllocation { bytes: if i == 3 { 7_932_874_632 } else { 0 }, estimate: None },
            WeightAllocation { bytes: if i == 0 { 970_506_240 } else { 0 }, estimate: None },
        ],
        fixed_state_bytes_per_request: (n + if i == 3 { 3 } else { 0 }) * 128 * 528,
        staging_bytes: 0, loader_bytes: 0, scratch_bytes: 0, reserve_bytes: 0,
    }).collect();
    PlacementPlan {
        devices,
        owners: [(2, 0, 2), (8, 0, 2), (14, 1, 2), (20, 2, 1)]
            .into_iter()
            .map(|(owner, device, tokens_per_record)| RecordOwner {
                owner,
                devices: vec![device],
                tokens_per_record,
                record_bytes,
            })
            .collect(),
        routes: vec![],
        host_resident_bytes: 202_758_032_400,
        host_staging_bytes: 8_000_000_000,
        host_loader_and_os_bytes: 32_000_000_000,
    }
}

fn first_overflow(plan: &PlacementPlan, t: u64) -> (u64, u32) {
    for n in 1..=100 {
        let r = placement_report(plan, t, n).unwrap();
        if let Some(d) = r.per_device.iter().find(|d| d.remaining_bytes < 0) {
            return (n, d.device);
        }
    }
    panic!("fixture failed to reach overflow");
}

#[test]
fn equal_split_reproduces_census_slope_and_frontier() {
    let p = fixture(584 + 68, [10; 4]);
    let r = placement_report(&p, 1_048_576, 16).unwrap();
    assert_eq!(
        r.per_device
            .iter()
            .map(|d| d.weight_bytes)
            .collect::<Vec<_>>(),
        [
            76_367_322_992,
            74_062_760_176,
            73_925_697_008,
            83_172_210_424
        ]
    );
    assert_eq!(
        r.per_device.iter().map(|d| d.weight_bytes).sum::<u64>(),
        307_527_990_600
    );
    assert_eq!(r.host_bytes, 242_758_032_400);
    assert_eq!(r.per_device[0].global_kv_bytes / 16 / 1_048_576, 652);
    assert_eq!(
        r.per_device
            .iter()
            .map(|d| d.remaining_bytes)
            .collect::<Vec<_>>(),
        [
            8_683_118_736,
            16_457_053_968,
            11_124_744_720,
            12_813_732_104
        ]
    );
    assert!(r.per_device.iter().all(|d| d.estimates.len() == 2));
    assert_eq!(first_overflow(&p, 1_048_576), (29, 0));
    assert!(placement_report(&p, 1_037_289, 29).unwrap().per_device[0].remaining_bytes >= 0);
    assert!(placement_report(&p, 1_037_290, 29).unwrap().per_device[0].remaining_bytes < 0);
}

#[test]
fn alternate_bytes_and_explicit_reserve_are_not_format_fallbacks() {
    assert_eq!(
        first_overflow(&fixture(288 + 68, [10; 4]), 1_048_576),
        (53, 0)
    );
    let mut p = fixture(652, [10; 4]);
    for d in &mut p.devices {
        d.reserve_bytes = 4_000_000_000;
    }
    assert_eq!(first_overflow(&p, 1_048_576), (23, 0));
    let mut q = fixture(652, [9, 11, 9, 11]);
    assert_eq!(first_overflow(&q, 1_048_576), (40, 0));
    for d in &mut q.devices {
        d.reserve_bytes = 4_000_000_000;
    }
    assert_eq!(first_overflow(&q, 1_048_576), (31, 1));
}

#[test]
fn odd_tails_and_replicas_are_counted_not_quartered() {
    let mut p = fixture(652, [10; 4]);
    let r = placement_report(&p, 3, 1).unwrap();
    assert_eq!(
        r.per_device
            .iter()
            .map(|d| d.global_kv_bytes)
            .collect::<Vec<_>>(),
        [1304, 652, 1956, 0]
    );
    p.owners[1].devices.push(1);
    p.owners[3].devices.push(3);
    let r = placement_report(&p, 100, 1).unwrap();
    assert_eq!(
        r.per_device.iter().map(|d| d.global_kv_bytes).sum::<u64>(),
        260_800
    );
    p.owners.push(p.owners[0].clone());
    assert_eq!(placement_report(&p, 1, 1), Err(BudgetError::DuplicateOwner));
}

#[test]
fn invalid_geometry_and_integer_overflow_refuse() {
    let mut p = fixture(652, [10; 4]);
    assert_eq!(
        placement_report(&p, u64::MAX, 2),
        Err(BudgetError::Overflow)
    );
    p.owners[0].tokens_per_record = 0;
    assert_eq!(
        placement_report(&p, 1, 1),
        Err(BudgetError::InvalidGeometry)
    );
    p.owners[0].tokens_per_record = 2;
    p.owners[0].devices.push(0);
    assert_eq!(
        placement_report(&p, 1, 1),
        Err(BudgetError::DuplicateReplica)
    );
    p.owners[0].devices = vec![10];
    assert_eq!(placement_report(&p, 1, 1), Err(BudgetError::UnknownDevice));
}

#[test]
fn staging_loader_scratch_and_routes_are_explicit() {
    let mut p = fixture(652, [10; 4]);
    p.devices[0].staging_bytes = 10;
    p.devices[0].loader_bytes = 20;
    p.devices[0].scratch_bytes = 30;
    p.routes.push(RoutePlan {
        from: Endpoint::Device(0),
        to: Endpoint::Device(1),
        kind: RouteKind::PcieP2p,
        bytes_per_generated_token: 100,
        generated_tokens_per_second: 5,
        restore_bytes: 100,
        restores_per_second: 2,
        measured_bytes_per_second: None,
    });
    let r = placement_report(&p, 1, 1).unwrap();
    assert_eq!(r.routes[0].demand_bytes_per_second, 700);
    assert_eq!(r.routes[0].physical_bytes_per_second, 700);
    p.routes[0].kind = RouteKind::HostBounce;
    assert_eq!(
        placement_report(&p, 1, 1).unwrap().routes[0].physical_bytes_per_second,
        1400
    );
    p.routes[0].kind = RouteKind::PcieP2p;
    assert_eq!(r.routes[0].within_measured_envelope, None);
    assert_eq!(
        r.per_device[0].total_bytes,
        r.per_device[0].weight_bytes + 675_840 + 60
    );
    p.routes[0].measured_bytes_per_second = Some(1000);
    assert_eq!(
        placement_report(&p, 1, 1).unwrap().routes[0].within_measured_envelope,
        Some(true)
    );
    p.routes[0].restores_per_second = 3;
    assert_eq!(
        placement_report(&p, 1, 1).unwrap().routes[0].within_measured_envelope,
        Some(false)
    );
    p.routes[0].to = Endpoint::PinnedHost;
    assert_eq!(
        placement_report(&p, 1, 1),
        Err(BudgetError::InvalidGeometry)
    );
}

#[test]
fn report_is_frozen_wire_and_replica_breakdown_not_extra_charge() {
    use memra_tier::contracts::Wire;
    let mut p = fixture(652, [10; 4]);
    p.owners[0].devices.push(1);
    let r = placement_report(&p, 100, 2).unwrap();
    assert_eq!(r.per_device[1].replica_bytes, 65_200);
    assert_eq!(r.per_device[0].replica_bytes, 0);
    assert_eq!(PlacementReport::decode(&r.encode().unwrap()).unwrap(), r);
}
