#[test]
fn review_digest_distinguishes_declared_scale_layout() {
    let cfg = QWEN.replace("\"hidden_size\":32", "\"hidden_size\":128")
        .replace("\"num_attention_heads\":2", "\"num_attention_heads\":8")
        .replace("\"intermediate_size\":64", "\"intermediate_size\":256");
    let f = fixture(&cfg, |ts| {
        let stem = "model.layers.0.self_attn.q_proj";
        ts.insert(format!("{stem}.weight"), Tensor { dtype: "U8", shape: vec![128,64], bytes: vec![0x22;8192] });
        ts.insert(format!("{stem}.weight_scale"), Tensor { dtype: "F8_E4M3", shape: vec![128,8], bytes: (0..1024).map(|i| 0x38+(i%8) as u8).collect() });
        ts.insert(format!("{stem}.weight_scale_2"), float(vec![1], 2.0));
    });
    let linear = SafetensorsSource::open(&f.dir).unwrap();
    std::fs::write(f.dir.join("LAYOUT.json"), r#"{"nvfp4_scale":"Swizzle32x4x4"}"#).unwrap();
    let swizzled = SafetensorsSource::open(&f.dir).unwrap();
    let a = BoundTensorSource::compile(&linear).unwrap();
    let b = BoundTensorSource::compile(&swizzled).unwrap();
    let id = layer(LayerTensor::Query);
    assert_ne!(a.nvfp4(&id).unwrap().unwrap().wscale, b.nvfp4(&id).unwrap().unwrap().wscale, "must exercise different interpretations");
    assert_ne!(a.binding_sha256(), b.binding_sha256(), "different scale-layout interpretation must change binding digest");
}

#[test]
fn review_input_scale_shape_rejected_at_compile() {
    let f = fixture(QWEN, |ts| {
        nvfp4(ts, 2.0);
        ts.insert("model.layers.0.self_attn.q_proj.input_scale".into(), float(vec![7], 2.0));
    });
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let compiled = BoundTensorSource::compile(&source);
    if let Ok(bound) = &compiled {
        let aux = bound.auxiliary(&layer(LayerTensor::Query), QuantAuxTensor::InputScale).unwrap().unwrap();
        eprintln!("accepted input_scale ne={:?}, bytes={}", aux.ne, aux.bytes.len());
    }
    assert!(compiled.is_err(), "invalid NVFP4 input-scale shape should fail compile");
}

#[test]
fn review_native_rejects_inverse_scale_overflow() {
    let f = fixture(QWEN, |ts| {
        nvfp4(ts, f32::from_bits(1));
        let stem = "model.layers.0.self_attn.q_proj";
        let packed = ts.remove(&format!("{stem}.weight")).unwrap();
        ts.insert(format!("{stem}.weight_packed"), packed);
        let scale = ts.remove(&format!("{stem}.weight_scale_2")).unwrap();
        ts.insert(format!("{stem}.weight_global_scale"), scale);
    });
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let id = layer(LayerTensor::Query);
    assert!(bound.auxiliary(&id, QuantAuxTensor::WeightScale).is_err());
    assert!(bound.nvfp4(&id).is_err(), "malformed macro scale must be an error for native access too");
}

#[test]
fn review_awq_input_scale_follows_bound_column_transform() {
    let cfg = HYBRID.replace("\"linear_num_key_heads\":1", "\"linear_num_key_heads\":2")
        .replace("\"linear_num_value_heads\":2", "\"linear_num_value_heads\":4");
    let f = fixture(&cfg, |ts| {
        let stem = "model.layers.0.linear_attn.out_proj";
        ts.insert(format!("{stem}.weight"), Tensor { dtype: "F32", shape: vec![32,64], bytes: (0..2048).flat_map(|i| ((i%64+1) as f32).to_le_bytes()).collect() });
        ts.insert(format!("{stem}.pre_quant_scale"), Tensor { dtype: "F32", shape: vec![64], bytes: (1..=64).flat_map(|i| (i as f32).to_le_bytes()).collect() });
    });
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let id = layer(LayerTensor::GdnOutput);
    assert_eq!(bound.binding().tensors[&id].transform, TensorTransform::OutReorderColumns);
    let weight = bound.tensor(&id).unwrap();
    let scale = bound.auxiliary(&id, QuantAuxTensor::PreQuantScale).unwrap().unwrap();
    let values = |bytes: &[u8]| bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>();
    eprintln!("bound weight first-row={:?}", values(&weight.bytes[..256]));
    eprintln!("bound input-axis scale={:?}", values(&scale.bytes));
    assert_eq!(&weight.bytes[..256], scale.bytes.as_ref(), "input-axis auxiliary must follow the owner's column permutation");
}

#[test]
fn review_transformed_nvfp4_bad_scale_is_not_optional_absence() {
    let f = fixture(HYBRID, |ts| {
        let stem = "model.layers.0.linear_attn.out_proj";
        ts.insert(format!("{stem}.weight"), Tensor { dtype: "U8", shape: vec![32,16], bytes: vec![0x22;512] });
        ts.insert(format!("{stem}.weight_scale"), Tensor { dtype: "F8_E4M3", shape: vec![32,2], bytes: vec![0x38;64] });
        ts.insert(format!("{stem}.weight_scale_2"), float(vec![1], f32::NAN));
    });
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let id = layer(LayerTensor::GdnOutput);
    assert!(bound.nvfp4(&id).is_err(), "NaN scale must not become Ok(None) just because the native view cannot represent a transform");
}
