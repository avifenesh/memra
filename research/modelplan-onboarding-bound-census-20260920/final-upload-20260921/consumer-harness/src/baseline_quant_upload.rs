// Exact reviewed57de constructor; only the method name changes for coexistence.
impl GpuTensor {
    pub fn from_quant_bytes_baseline(
        e: &Engine,
        bytes: &[u8],
        ty: GgmlType,
        ne0: u64,
        ne1: u64,
        scale: f32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let qt = match ty {
            GgmlType::Q8_0 => QT_Q8_0,
            GgmlType::Q4_K => QT_Q4_K,
            GgmlType::Q6_K => QT_Q6_K,
            GgmlType::Q5_K => QT_Q5_K,
            GgmlType::Q3_K => QT_Q3_K,
            GgmlType::IQ4_XS => QT_IQ4_XS,
            GgmlType::IQ3_S => QT_IQ3_S,
            GgmlType::NVFP4 => QT_NVFP4,
            GgmlType::Q4_0 => QT_Q4_0,
            other => panic!("from_quant_bytes: unsupported dtype {other:?}"),
        };
        let row_bytes = bytes.len() / ne1 as usize;
        // Same A6 repack as load_from_source: callers pass GGUF-layout host bytes (the FR-Spec
        // self-trim row-gathers from the source file bytes, which are always original layout).
        let rp = qt == QT_NVFP4
            && ne0.is_multiple_of(64)
            && row_bytes.is_multiple_of(36)
            && rp_enabled();
        let dev = if rp {
            e.htod_bytes(&repack_nvfp4_split(bytes, ne1 as usize))?
        } else {
            e.htod_bytes(bytes)?
        };
        Ok(GpuTensor::Quant {
            bytes: dev,
            qtype: qt,
            row_bytes,
            ne: vec![ne0, ne1],
            scale,
            rp,
            #[cfg(memra_cutlass)]
            cutlass: None,
            fp8: None,
            blk: None,
            f16: None,
            a4: None,
            rp4: None,
        })
    }

}
