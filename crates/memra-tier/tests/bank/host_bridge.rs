//! API-shaped CPU fixture for the unexported engine bridge. This checks the
//! actual bridge source, not CUDA/engine compilation. Native wiring remains gated.
use super::*;
use crate::model::*;

#[test]
fn bridge_uses_per_record_metadata_and_preserves_existing_scale_planes() {
    let host = HostExps {
        n_expert: 3,
        qtype: 99,
        row_bytes: 999,
        expert_stride: 999,
        layouts: Some(vec![
            ExpertLayout {
                offset: 0,
                len: 16,
                qtype: 2,
                row_bytes: 8,
            },
            ExpertLayout {
                offset: 0,
                len: 0,
                qtype: 0,
                row_bytes: 0,
            },
            ExpertLayout {
                offset: 80,
                len: 24,
                qtype: 3,
                row_bytes: 8,
            },
        ]),
        tiers: Some(vec![(), (), ()]),
        macros: Some(vec![1.5, 0.0, 2.5]),
        fp8_blk: Some(HostExpertFp8BlockScales {
            scales: vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0],
            expert_stride: 2,
        }),
    };
    let mut sources = BTreeMap::new();
    for n in [0u32, 2] {
        let mut macro_plane = layout(n as u64, 2, 16).segments[1].clone();
        macro_plane.offset = n as u64 * 4;
        let block = ByteSegment {
            group: 2,
            role: Role::Scale,
            tensor: Some(TensorId {
                name: "expert.block_scales".into(),
                ..tensor()
            }),
            offset: n as u64 * 8,
            valid_bytes: 8,
            storage_bytes: 8,
            ..macro_plane.clone()
        };
        let index = n as usize;
        let bytes: Vec<_> = host.fp8_blk.as_ref().unwrap().scales[index * 2..index * 2 + 2]
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        sources.insert(
            n,
            ExpertSource {
                tensor: TensorId {
                    name: format!("expert.{n}.weight"),
                    ..tensor()
                },
                split: true,
                scales: vec![macro_plane, block],
                checksums: vec![
                    [0; 32],
                    checksum(&host.macros.as_ref().unwrap()[index].to_bits().to_le_bytes()),
                    checksum(&bytes),
                ],
            },
        );
    }
    let mapped = crate::engine_bridge::map_host_exps(
        &host,
        tensor(),
        2,
        Projection::Gate,
        &[true, false, true],
        &sources,
    )
    .unwrap();
    let r = mapped.catalog.record(&mapped.ids[2]).unwrap();
    assert_eq!(r.layout.segments[0].offset, 0); // split already positioned
    assert_eq!(r.layout.segments[0].valid_bytes, 24);
    assert_eq!(r.layout.segments[0].encoding.row_bytes, 8);
    assert_eq!(
        r.layout.segments[0].encoding.program,
        digest("host-exps-qtype-v1", &3i32.to_le_bytes())
    );
    assert_eq!(mapped.max_expert_bytes, 24);
    assert_eq!(r.layout.segments.len(), 3);
    sources.get_mut(&2).unwrap().scales.pop();
    assert!(matches!(
        crate::engine_bridge::map_host_exps(
            &host,
            tensor(),
            2,
            Projection::Gate,
            &[true, false, true],
            &sources
        ),
        Err(Error::Incomplete)
    ));
}
