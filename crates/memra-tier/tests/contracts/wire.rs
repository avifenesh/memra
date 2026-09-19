use super::support::*;
use memra_tier::contracts::*;
#[test]
fn identity_domains_full_program_and_canonical_roundtrip() {
    let p = program();
    let encoded = p.encode().unwrap();
    assert_eq!(ProgramIdentity::decode(&encoded).unwrap(), p);
    assert_ne!(digest("program", b"abc"), digest("valid-bytes", b"abc"));
    for field in 0..10 {
        let mut q = p.clone();
        let fields = [
            &mut q.artifact,
            &mut q.serialized_plan,
            &mut q.numeric,
            &mut q.stream,
            &mut q.tokenizer,
            &mut q.template,
            &mut q.adapter,
            &mut q.modality,
            &mut q.position,
            &mut q.tenant_salt,
        ];
        fields.into_iter().nth(field).unwrap()[0] ^= 1;
        assert_ne!(p.namespace().unwrap(), q.namespace().unwrap());
    }
    let mut noncanonical = encoded.clone();
    noncanonical.push(b' ');
    assert_eq!(ProgramIdentity::decode(&noncanonical), Err(Error::Corrupt));
    let text = String::from_utf8(encoded)
        .unwrap()
        .replace("\"version\":1", "\"version\":2");
    assert_eq!(
        ProgramIdentity::decode(text.as_bytes()),
        Err(Error::UnsupportedVersion(2))
    );
    assert_ne!(
        KvBlockId::new(&p, [0; 32], &[1, 2], 0, 0, 0, 0)
            .unwrap()
            .identity()
            .unwrap(),
        KvBlockId::new(&p, [1; 32], &[1, 2], 0, 0, 0, 0)
            .unwrap()
            .identity()
            .unwrap()
    );
}
#[test]
fn immutable_seal_and_committed_high_water_cannot_publish_rollback_tail() {
    let active = bundle();
    let sealed = active.seal().unwrap();
    assert_eq!(sealed.id.epoch, 0);
    assert_eq!(sealed.kind, StateKind::ImmutablePrefix);
    let mut next = active.clone();
    next.id.epoch = 999;
    assert_eq!(next.seal().unwrap(), sealed);
    assert_eq!(
        active.require(&active.program, &active.layout, 8, 3),
        Err(Error::StaleEpoch)
    );
    assert_eq!(
        active.require(&active.program, &active.layout, 7, 2),
        Err(Error::StaleEpoch)
    );
    let mut bad = active;
    bad.committed_high_water = 2;
    assert_eq!(bad.seal(), Err(Error::Incomplete));
    let encoded = sealed.encode().unwrap();
    assert_eq!(StateBundle::decode(&encoded).unwrap(), sealed);
}
#[test]
fn heterogeneous_all_trailing_groups_aliases_and_padding() {
    let mut b = bundle();
    b.layout.requirements[0].page_count = 2;
    b.layout.segments.push(b.layout.segments[0].clone());
    b.layout.segments[1].page = 1;
    b.checksums.push(b.checksums[0]);
    let mut tail = b.layout.segments[0].clone();
    tail.group = 1;
    tail.role = Role::Tail;
    tail.page = 1;
    b.layout.segments.push(tail);
    b.layout.requirements.push(GroupRequirement {
        version: 1,
        group: 1,
        owner: 0,
        role: Role::Tail,
        page_count: 2,
        pages: PageRequirement::TrailingPages(1),
    });
    b.checksums.push(b.checksums[0]);
    b.owner_aliases.push(OwnerAlias {
        version: 1,
        group: 0,
        consumer: 1,
        owner: 0,
    });
    b.validate().unwrap();
    assert_eq!(b.layout.storage_bytes().unwrap(), 12);
    let data = vec![vec![3, 20, 37, 0]; 3];
    b.verify(&data).unwrap();
    let mut bad = data;
    bad[0][3] = 1;
    assert_eq!(b.verify(&bad), Err(Error::Corrupt));
    for i in 0..3 {
        let mut bad = b.clone();
        bad.layout.segments.remove(i);
        bad.checksums.remove(i);
        assert_eq!(bad.validate(), Err(Error::Incomplete));
    }
    let mut wrong = b;
    wrong.owner_aliases[0].owner = 99;
    assert_eq!(wrong.validate(), Err(Error::InvalidLayout));
}
#[test]
fn layout_duplicate_overflow_and_group_specific_counts_refuse() {
    let mut l = layout();
    l.segments.push(l.segments[0].clone());
    assert_eq!(l.validate(), Err(Error::InvalidLayout));
    let mut l = layout();
    l.segments[0].offset = u64::MAX - 3;
    l.segments[0].storage_bytes = 8;
    assert_eq!(l.validate(), Err(Error::Overflow));
    let mut l = layout();
    l.requirements[0].page_count = u64::MAX;
    assert_eq!(l.validate(), Err(Error::Incomplete)); // bounded by actual segments
}
#[test]
fn golden_fixture_hashes_are_pinned() {
    let manifest: serde_json::Value = serde_json::from_str(include_str!("fixtures.json")).unwrap();
    for n in [0usize, 1, 264, 288, 4095, 4096, 4097, 1048576] {
        let hex = checksum(&bytes(n))
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        assert_eq!(manifest["payloads"][n.to_string()].as_str().unwrap(), hex);
    }
    for (name, b) in [
        ("program", program().encode().unwrap()),
        ("key", key().encode().unwrap()),
        ("bundle", bundle().encode().unwrap()),
    ] {
        let hex = digest("fixture-wire", &b)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        assert_eq!(manifest["wire"][name].as_str().unwrap(), hex);
    }
}
#[test]
fn placement_negative_headroom_route_labels_and_estimates_survive_wire() {
    let device = |id| DeviceBudget {
        version: 1,
        device: id,
        weight_bytes: 100,
        global_kv_bytes: 20,
        fixed_state_bytes: 3,
        staging_bytes: 4,
        loader_bytes: 5,
        scratch_bytes: 6,
        reserve_bytes: 7,
        replica_bytes: 10,
        total_bytes: 145,
        remaining_bytes: -1,
        estimates: vec!["scratch estimated".into()],
    };
    let r = RouteBudget {
        version: 1,
        from: Endpoint::Device(0),
        to: Endpoint::Device(1),
        kind: RouteKind::HostBounce,
        demand_bytes_per_second: 10,
        physical_bytes_per_second: 20,
        measured_bytes_per_second: None,
        within_measured_envelope: None,
        estimates: vec!["demand estimate".into()],
    };
    let mut p = PlacementReport {
        version: 1,
        per_device: vec![device(0), device(1)],
        routes: vec![r],
        host_bytes: 32,
        estimates: vec!["fixture, not admission evidence".into()],
    };
    assert_eq!(PlacementReport::decode(&p.encode().unwrap()).unwrap(), p);
    p.routes[0].kind = RouteKind::PcieP2p;
    assert_eq!(p.validate(), Err(Error::InvalidLayout));
}
