use memra_engine::{
    Engine,
    latent_nvfp4_ffi::{Nvfp4LatentOps, Nvfp4LatentStorage},
};
use memra_kv::latent_nvfp4::PackedRow;

// Explicitly ignored without invocation: absence of CUDA must never look like a pass.
#[test]
#[ignore = "requires an admitted non-production CUDA device and canonical GPU lock"]
fn append_gather_matches_cpu_and_preserves_prefix() {
    let e = Engine::new(0).expect("CUDA required");
    let width = 512;
    let input: Vec<f32> = (0..width * 3)
        .map(|i| ((i * 37 % 1021) as f32 - 510.0) / 73.0)
        .collect();
    let mut storage = Nvfp4LatentStorage::new(&e, width, 8).unwrap();
    assert_eq!(storage.allocated_bytes(), 8 * 292 + 4);
    storage.append(&e, &e.htod(&input).unwrap(), 0).unwrap();
    storage.check(&e).unwrap();
    let encoded: Vec<_> = input
        .chunks_exact(width)
        .map(|r| PackedRow::encode(r).unwrap())
        .collect();
    assert_eq!(
        &e.stream().clone_dtoh(&storage.payload).unwrap()[..3 * width / 2],
        encoded
            .iter()
            .flat_map(|r| r.payload.iter().copied())
            .collect::<Vec<_>>()
            .as_slice()
    );
    assert_eq!(
        &e.stream().clone_dtoh(&storage.scales).unwrap()[..3 * width / 16],
        encoded
            .iter()
            .flat_map(|r| r.block_scales.iter().copied())
            .collect::<Vec<_>>()
            .as_slice()
    );
    assert_eq!(
        e.dtoh(&storage.macros).unwrap()[..3]
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        encoded
            .iter()
            .map(|r| r.macro_scale.to_bits())
            .collect::<Vec<_>>()
    );
    let selections = e.htod_i32(&[2, 0, -1, 1]).unwrap();
    let got = storage.gather(&e, &selections, 3).unwrap();
    storage.check(&e).unwrap();
    let got = e.dtoh(&got).unwrap();
    for (i, row) in [Some(2), Some(0), None, Some(1)].into_iter().enumerate() {
        let expected = row
            .map(|r| {
                PackedRow::encode(&input[r * width..(r + 1) * width])
                    .unwrap()
                    .decode()
                    .unwrap()
            })
            .unwrap_or(vec![0.; width]);
        assert_eq!(
            got[i * width..(i + 1) * width]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    let old_payload = e.stream().clone_dtoh(&storage.payload).unwrap();
    let snapshot = storage.snapshot(&e, 2).unwrap();
    let mut restored = Nvfp4LatentStorage::new(&e, width, 8).unwrap();
    restored.copy_prefix_from(&e, &snapshot, 2).unwrap();
    let restored_values = restored
        .gather(&e, &e.htod_i32(&[0, 1]).unwrap(), 2)
        .unwrap();
    restored.check(&e).unwrap();
    assert_eq!(
        e.dtoh(&restored_values).unwrap(),
        [
            got[width..2 * width].to_vec(),
            got[3 * width..4 * width].to_vec()
        ]
        .concat()
    );
    storage
        .append(&e, &e.htod(&vec![1000.; width]).unwrap(), 2)
        .unwrap();
    storage.check(&e).unwrap();
    let new_payload = e.stream().clone_dtoh(&storage.payload).unwrap();
    assert_eq!(&old_payload[..width], &new_payload[..width]);
    assert!(storage.append(&e, &e.htod(&input).unwrap(), 7).is_err());
    let bad = storage.gather(&e, &e.htod_i32(&[3]).unwrap(), 3).unwrap();
    assert!(storage.check(&e).is_err());
    assert!(e.dtoh(&bad).unwrap().iter().all(|v| *v == 0.));
    assert!(
        storage.snapshot(&e, 2).is_err(),
        "poisoned plane must not publish a clean snapshot"
    );
}

#[test]
#[ignore = "requires an admitted non-production CUDA device and canonical GPU lock"]
fn compressed_attention_matches_decode_then_f32_attention() {
    let e = Engine::new(0).expect("CUDA required");
    let (width, rows, heads, queries, slots) = (512, 31, 4, 3, 19);
    let input: Vec<f32> = (0..width * rows)
        .map(|i| ((i * 79 % 997) as f32 - 498.) / 101.)
        .collect();
    let mut plane = Nvfp4LatentStorage::new(&e, width, rows).unwrap();
    plane.append(&e, &e.htod(&input).unwrap(), 0).unwrap();
    let decoded: Vec<f32> = input
        .chunks_exact(width)
        .flat_map(|r| PackedRow::encode(r).unwrap().decode().unwrap())
        .collect();
    let bf16 = plane.bf16_history(&e, rows).unwrap();
    let bf16_reference = e
        .f32_to_bf16(&e.htod(&decoded).unwrap(), rows * width)
        .unwrap();
    assert_eq!(
        e.stream().clone_dtoh(&bf16).unwrap(),
        e.stream().clone_dtoh(&bf16_reference).unwrap()
    );
    let q: Vec<f32> = (0..width * heads * queries)
        .map(|i| ((i * 41 % 523) as f32 - 261.) / 157.)
        .collect();
    let indices: Vec<i32> = (0..queries * slots)
        .map(|i| {
            if i % 7 == 0 {
                -1
            } else {
                (i * 13 % rows) as i32
            }
        })
        .collect();
    let q = e.htod(&q).unwrap();
    let idx = e.htod_i32(&indices).unwrap();
    let actual = plane
        .attend(&e, &q, &idx, heads, queries, slots, rows, 0.0625)
        .unwrap();
    plane.check(&e).unwrap();
    let mut expected = e.uninit(width * heads * queries).unwrap();
    e.mla_attn_gathered(
        &q,
        &e.uninit(1).unwrap(),
        &e.htod(&decoded).unwrap(),
        &idx,
        &mut expected,
        heads,
        width,
        0,
        queries,
        slots,
        0.0625,
    )
    .unwrap();
    let (a, b) = (e.dtoh(&actual).unwrap(), e.dtoh(&expected).unwrap());
    assert!(a.iter().all(|x| x.is_finite()));
    assert_eq!(
        a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "requires an admitted non-production CUDA device and canonical GPU lock"]
fn latent_snapshot_carries_compressed_rows_and_unmodified_index_tail() {
    let e = Engine::new(0).expect("CUDA required");
    let make = || memra_kv::LatentKvLayer {
        rows: e.uninit(0).unwrap(),
        nvfp4: Some(Nvfp4LatentStorage::new(&e, 512, 8).unwrap()),
        width: 512,
        len: 0,
        len_d: e.htod_i32(&[0]).unwrap(),
        index_rows: Some(e.zeros(64 * 8).unwrap()),
        index_width: 8,
        index_ring_rows: Some(64),
        index_pool_keys: None,
        index_pools_ready: 0,
        index_pool: 4,
    };
    let mut source = make();
    source
        .nvfp4
        .as_mut()
        .unwrap()
        .append(&e, &e.htod(&vec![1.25; 512 * 5]).unwrap(), 0)
        .unwrap();
    source.index_rows = Some(
        e.htod(&(0..64 * 8).map(|i| i as f32).collect::<Vec<_>>())
            .unwrap(),
    );
    source.index_pool_keys = Some(e.htod(&[3., 4., 5., 6.]).unwrap());
    source.index_pools_ready = 1;
    source.len = 5;
    let snapshot = source.snapshot_plane(&e).unwrap();
    assert!(snapshot.rows.is_empty());
    assert_eq!(snapshot.bytes(), 5 * 292 + 4 + (8 + 4) * 4);
    let mut restored = make();
    restored.restore_plane(&e, &snapshot, 8).unwrap();
    assert_eq!(restored.len, 5);
    assert_eq!(restored.index_pools_ready, 1);
    assert_eq!(
        e.dtoh(restored.index_pool_keys.as_ref().unwrap()).unwrap()[..4],
        [3., 4., 5., 6.]
    );
    assert_eq!(
        e.dtoh(restored.index_rows.as_ref().unwrap()).unwrap()[32..40],
        [32., 33., 34., 35., 36., 37., 38., 39.]
    );
    let positions = e.htod_i32(&[0, 4]).unwrap();
    let got = restored
        .nvfp4
        .as_mut()
        .unwrap()
        .gather(&e, &positions, 5)
        .unwrap();
    let want = source
        .nvfp4
        .as_mut()
        .unwrap()
        .gather(&e, &positions, 5)
        .unwrap();
    assert_eq!(e.dtoh(&got).unwrap(), e.dtoh(&want).unwrap());
    // Restore may not silently reinterpret the same-width f32 format.
    let mut wrong = make();
    wrong.nvfp4 = None;
    wrong.rows = e.zeros(8 * 512).unwrap();
    assert!(wrong.restore_plane(&e, &snapshot, 8).is_err());
}

#[test]
#[ignore = "requires CUDA, MEMRA_GLM53_NVFP4_LATENT=1 and an explicit NVFP4_LATENT_TEST_CONFIG"]
fn raw_checkpoint_plan_allocates_only_compressed_latent_history() {
    assert!(
        memra_kv::latent_layout::nvfp4_enabled(),
        "explicit NVFP4 selection required"
    );
    let config = std::env::var("NVFP4_LATENT_TEST_CONFIG").expect("explicit model config required");
    let cfg =
        memra_gguf::config::ModelConfig::from_config_json(std::path::Path::new(&config)).unwrap();
    let plan = memra_gguf::model_plan::ModelPlan::compile(&cfg).unwrap();
    let e = Engine::new(0).expect("CUDA required");
    let capacity = 128;
    let mut cache = memra_kv::Cache::new_planned(&e, &cfg, &plan, capacity).unwrap();
    let headless = memra_kv::Cache::new_planned_active(&e, &cfg, &plan, capacity, false).unwrap();
    let trunk_latent = plan
        .layers
        .iter()
        .filter(|l| {
            matches!(
                l.state,
                memra_gguf::model_plan::StatePlan::LatentKvCache { .. }
            )
        })
        .count();
    assert_eq!(headless.latent.iter().flatten().count(), trunk_latent);
    for block in &plan.mtp_blocks {
        let index = block.layer.index as usize;
        assert!(
            headless.latent[index].is_none()
                && headless.kv[index].is_none()
                && headless.recur[index].is_none(),
            "unloaded MTP must reserve no cache state"
        );
    }
    println!("NVFP4_ALLOC_NO_MTP: trunk_latent_layers={trunk_latent} unloaded_head_planes=0");
    drop(headless);
    let mut latent_count = 0;
    for layer in cache.latent.iter().flatten() {
        assert!(
            layer.rows.is_empty(),
            "f32 history shadow must not be allocated"
        );
        let plane = layer.nvfp4.as_ref().expect("compressed plane missing");
        assert_eq!(plane.allocated_bytes(), capacity * 292 + 4);
        assert_eq!(layer.capacity(), capacity);
        plane.check(&e).unwrap();
        latent_count += 1;
    }
    assert!(latent_count > 0, "gate did not engage a latent plane");
    let statuses: std::collections::HashSet<_> = cache
        .latent
        .iter()
        .flatten()
        .map(|l| l.nvfp4.as_ref().unwrap().error.identity())
        .collect();
    assert_eq!(
        statuses.len(),
        1,
        "one shared status per owning stage, not per layer"
    );
    // Raw checkpoint metadata includes NextN; this allocator-only gate is not
    // evidence of the loaded serving recipe. Score only DFlash2/no-MTP serving.
    let trunk_index = plan
        .layers
        .iter()
        .find(|l| {
            matches!(
                l.state,
                memra_gguf::model_plan::StatePlan::LatentKvCache { .. }
            )
        })
        .unwrap()
        .index as usize;
    cache.latent[trunk_index]
        .as_mut()
        .expect("trunk latent plane missing")
        .nvfp4
        .as_mut()
        .unwrap()
        .append(&e, &e.htod(&vec![f32::NAN; 512]).unwrap(), 0)
        .unwrap();
    assert!(cache.check_latent_status().is_err());
    assert!(cache.ensure_usable("after invalid append").is_err());
    println!(
        "NVFP4_ALLOC_RAW_CHECKPOINT_PLAN: latent_layers={latent_count} capacity={capacity} resident_row_bytes=292 no_f32_shadow=true"
    );
}

#[test]
#[ignore = "requires an admitted non-production CUDA device and canonical GPU lock"]
fn captured_live_append_and_attention_follow_device_positions() {
    let e = Engine::new(0).expect("CUDA required");
    let (width, capacity, heads) = (512, 12, 4);
    let row = |p: usize| {
        (0..width)
            .map(|i| ((i * 41 + p * 17) % 251) as f32 / 37.0 - 3.0)
            .collect::<Vec<_>>()
    };
    let mut live = Nvfp4LatentStorage::new(&e, width, capacity).unwrap();
    let mut scalar = Nvfp4LatentStorage::new(&e, width, capacity).unwrap();
    let mut input = e.htod(&row(0)).unwrap();
    let query = e.htod(&vec![0.05; width * heads]).unwrap();
    let positions = e
        .htod_i32(&(0..capacity as i32).collect::<Vec<_>>())
        .unwrap();
    let mut pos = e.htod_i32(&[0]).unwrap();
    let mut slots = e.htod_i32(&[1]).unwrap();
    let mut output = e.zeros(width * heads).unwrap();
    let graph = e
        .capture_graph(|e| {
            live.append_live(e, &input, &pos)?;
            let result = live.attend_live(e, &query, &positions, &slots, &pos, heads, 1, 0.0625)?;
            e.copy_into(&mut output, 0, &result, width * heads)
        })
        .unwrap();
    for p in 0..capacity {
        e.copy_into(&mut input, 0, &e.htod(&row(p)).unwrap(), width)
            .unwrap();
        e.set_i32_one(&mut pos, p as i32).unwrap();
        e.set_i32_one(&mut slots, (p + 1) as i32).unwrap();
        graph.launch().unwrap();
        live.check(&e).unwrap();
        scalar.append(&e, &input, p).unwrap();
        let selected = e.htod_i32(&(0..=p as i32).collect::<Vec<_>>()).unwrap();
        let expected = scalar
            .attend(&e, &query, &selected, heads, 1, p + 1, p + 1, 0.0625)
            .unwrap();
        scalar.check(&e).unwrap();
        assert_eq!(
            e.dtoh(&output)
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            e.dtoh(&expected)
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            "replay position {p}"
        );
    }
    assert_eq!(
        e.stream().clone_dtoh(&live.payload).unwrap(),
        e.stream().clone_dtoh(&scalar.payload).unwrap()
    );
    let before = e.stream().clone_dtoh(&live.payload).unwrap();
    e.set_i32_one(&mut pos, capacity as i32).unwrap();
    graph.launch().unwrap();
    assert!(
        live.check(&e).is_err(),
        "live out-of-bounds append must poison the request"
    );
    assert_eq!(before, e.stream().clone_dtoh(&live.payload).unwrap());
}

#[test]
#[ignore = "requires an admitted non-production CUDA device and canonical GPU lock"]
fn captured_multirow_verify_rewinds_and_overwrites_only_rejected_suffix() {
    let e = Engine::new(0).expect("CUDA required");
    let (r, heads, t, slots, capacity) = (512, 2, 3, 8, 16);
    let make = |n: usize, salt: usize| {
        (0..n)
            .map(|i| ((i * 53 + salt) % 991) as f32 / 113.0 - 4.0)
            .collect::<Vec<_>>()
    };
    let mut history = make(4 * r, 11);
    let mut plane = Nvfp4LatentStorage::new(&e, r, capacity).unwrap();
    plane.append(&e, &e.htod(&history).unwrap(), 0).unwrap();
    let mut input = e.htod(&make(t * r, 17)).unwrap();
    let q = e.htod(&make(t * heads * r, 23)).unwrap();
    let indices = |base: usize| {
        (0..t)
            .flat_map(|i| (0..slots).map(move |s| if s <= base + i { s as i32 } else { -1 }))
            .collect::<Vec<_>>()
    };
    let mut idx = e.htod_i32(&indices(4)).unwrap();
    let width = e.htod_i32(&[slots as i32]).unwrap();
    let mut pos = e.htod_i32(&[4]).unwrap();
    let mut output = e.zeros(t * heads * r).unwrap();
    let graph = e
        .capture_graph(|e| {
            plane.append_live(e, &input, &pos)?;
            let result = plane.attend_live(e, &q, &idx, &width, &pos, heads, t, 0.0625)?;
            e.copy_into(&mut output, 0, &result, t * heads * r)
        })
        .unwrap();
    let mut preserved: Option<(Vec<u8>, Vec<u8>, Vec<u32>)> = None;
    for (base, salt) in [(4, 17), (5, 31)] {
        let proposal = make(t * r, salt);
        e.copy_into(&mut input, 0, &e.htod(&proposal).unwrap(), t * r)
            .unwrap();
        e.stream().memcpy_htod(&indices(base), &mut idx).unwrap();
        e.set_i32_one(&mut pos, base as i32).unwrap();
        graph.launch().unwrap();
        plane.check(&e).unwrap();
        history.truncate(base * r);
        history.extend_from_slice(&proposal);
        let decoded: Vec<_> = history
            .chunks_exact(r)
            .flat_map(|row| PackedRow::encode(row).unwrap().decode().unwrap())
            .collect();
        let mut expected = e.uninit(t * heads * r).unwrap();
        e.mla_attn_gathered(
            &q,
            &e.uninit(1).unwrap(),
            &e.htod(&decoded).unwrap(),
            &idx,
            &mut expected,
            heads,
            r,
            0,
            t,
            slots,
            0.0625,
        )
        .unwrap();
        assert_eq!(
            e.dtoh(&output)
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            e.dtoh(&expected)
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        let bytes = e.stream().clone_dtoh(&plane.payload).unwrap();
        let scales = e.stream().clone_dtoh(&plane.scales).unwrap();
        let macros = e
            .dtoh(&plane.macros)
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>();
        if let Some((b, s, m)) = &preserved {
            assert_eq!(&bytes[..5 * r / 2], b);
            assert_eq!(&scales[..5 * r / 16], s);
            assert_eq!(&macros[..5], m);
        } else {
            preserved = Some((
                bytes[..5 * r / 2].to_vec(),
                scales[..5 * r / 16].to_vec(),
                macros[..5].to_vec(),
            ));
        }
    }
}
