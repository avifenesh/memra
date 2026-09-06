//! Checked geometry for append-only latent history. No runtime dispatch is enabled here.
//! NVFP4 rows use sequential low/high nibbles, one E4M3 scale per 16 values,
//! and one f32 macro scale per row. Row-local macro scales keep earlier rows
//! immutable when later tokens have a different range. Index/recurrent planes
//! are separate and must not be counted as compressed by this layout.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatentFormat {
    F32,
    Nvfp4,
}

/// Permanent basis identity. Rht512V1 fixes mask, order and normalization below;
/// any future change to those constants requires a NEW variant, never reinterpretation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum LatentBasis {
    Identity = 0,
    Rht512V1 = 1,
}

pub const RHT512_SEED: u64 = 0x243f6a8885a308d3;
pub const RHT512_NORMALIZER_BITS: u32 = 0x3d3504f3;
pub const RHT512_NORMALIZER: f32 = f32::from_bits(RHT512_NORMALIZER_BITS);
pub const RHT512_SIGN_MASK: [u64; 8] = [
    0x4e087330ad225fff,
    0xb94fae7082589876,
    0x61fcfa90a1809bba,
    0x8fc61edbac51fd4e,
    0x6e8f21bb0378f0a8,
    0xd6090fefe872ed9b,
    0x5d0def85c580fadf,
    0x4947d7c4be8581cb,
];

impl LatentBasis {
    pub fn validate(self, width: usize) -> Result<(), &'static str> {
        if self == Self::Rht512V1 && width != 512 {
            return Err("Rht512V1 requires exactly 512 latent columns");
        }
        Ok(())
    }

    pub fn validate_copy(self, source: Self) -> Result<(), &'static str> {
        if self != source {
            return Err("latent basis mismatch");
        }
        Ok(())
    }
}

/// Row-vector R = D H. Sign bit is the high bit of the SplitMix64 FINALIZER
/// applied to seed+column (wrapping), WITHOUT a golden-ratio state increment.
/// Butterfly spans increase 1,2,...,256; normalize once at the end.
/// This is a float32 reference, not a promise of exact real-arithmetic orthogonality.
pub fn rht512_forward(values: &[f32]) -> Result<Vec<f32>, &'static str> {
    rht512(values, false)
}

/// R^T = H D: inverse signs are AFTER the butterfly and normalization.
pub fn rht512_inverse(values: &[f32]) -> Result<Vec<f32>, &'static str> {
    rht512(values, true)
}

fn rht512(values: &[f32], inverse: bool) -> Result<Vec<f32>, &'static str> {
    if values.len() != 512 || values.iter().any(|v| !v.is_finite()) {
        return Err("RHT512 requires 512 finite values");
    }
    let sign = |i: usize, x: f32| {
        if RHT512_SIGN_MASK[i / 64] & (1u64 << (i % 64)) != 0 {
            -x
        } else {
            x
        }
    };
    let mut row = values.to_vec();
    if !inverse {
        for (i, v) in row.iter_mut().enumerate() {
            *v = sign(i, *v);
        }
    }
    for span in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
        for block in row.chunks_exact_mut(2 * span) {
            for i in 0..span {
                let (a, b) = (block[i], block[i + span]);
                block[i] = a + b;
                block[i + span] = a - b;
                if !block[i].is_finite() || !block[i + span].is_finite() {
                    return Err("RHT512 butterfly overflow");
                }
            }
        }
    }
    for (i, v) in row.iter_mut().enumerate() {
        *v *= RHT512_NORMALIZER;
        if inverse {
            *v = sign(i, *v);
        }
        if !v.is_finite() {
            return Err("RHT512 normalization overflow");
        }
    }
    Ok(row)
}

/// Experiment-only selection, shared by allocation and admission. This is a
/// NoPE DSA latent-plane program, not a generic full-attention KV dtype switch.
pub fn nvfp4_enabled() -> bool {
    std::env::var("MEMRA_GLM53_NVFP4_LATENT").as_deref() == Ok("1")
}

pub fn selected_layout(width: usize, index_width: usize) -> Result<LatentLayout, &'static str> {
    layout_for_selection(nvfp4_enabled(), width, index_width)
}

pub fn layout_for_selection(
    enabled: bool,
    width: usize,
    index_width: usize,
) -> Result<LatentLayout, &'static str> {
    if enabled {
        if width != 512 || index_width == 0 {
            return Err("NVFP4 latent storage requires 512-wide NoPE DSA history");
        }
        LatentLayout::new_with_basis(LatentFormat::Nvfp4, width, LatentBasis::Rht512V1)
    } else {
        LatentLayout::new(LatentFormat::F32, width)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatentLayout {
    pub format: LatentFormat,
    pub basis: LatentBasis,
    pub width: usize,
    pub payload_bytes: usize,
    pub block_scale_bytes: usize,
    pub row_scale_bytes: usize,
}

impl LatentLayout {
    pub fn new(format: LatentFormat, width: usize) -> Result<Self, &'static str> {
        Self::new_with_basis(format, width, LatentBasis::Identity)
    }

    pub fn new_with_basis(
        format: LatentFormat,
        width: usize,
        basis: LatentBasis,
    ) -> Result<Self, &'static str> {
        basis.validate(width)?;
        if format == LatentFormat::F32 && basis != LatentBasis::Identity {
            return Err("f32 latent history must use identity basis");
        }
        if width == 0 {
            return Err("latent width must be nonzero");
        }
        let (payload_bytes, block_scale_bytes, row_scale_bytes) = match format {
            LatentFormat::F32 => (width.checked_mul(4).ok_or("latent width overflow")?, 0, 0),
            LatentFormat::Nvfp4 => {
                if !width.is_multiple_of(16) {
                    return Err("NVFP4 latent width must be divisible by 16");
                }
                (width / 2, width / 16, 4)
            }
        };
        let layout = Self {
            format,
            basis,
            width,
            payload_bytes,
            block_scale_bytes,
            row_scale_bytes,
        };
        layout.allocation_bytes(1)?;
        Ok(layout)
    }

    /// Validate the actual declared attention program, not a model-family heuristic.
    pub fn for_attention(
        enabled: bool,
        attention: &memra_gguf::model_plan::AttentionPlan,
        width: usize,
        index_width: usize,
    ) -> Result<Self, &'static str> {
        use memra_gguf::model_plan::{AttentionPlan, MlaAttentionPlan, SparseIndexPlan};
        if enabled {
            match attention {
                AttentionPlan::Mla(MlaAttentionPlan::LatentKv {
                    kv_lora_rank: 512,
                    rope_head_dim: 0,
                    rope,
                    sparse_index:
                        SparseIndexPlan::Own {
                            heads,
                            head_dim,
                            top_k,
                            kpool: Some(pool),
                        },
                    ..
                }) if rope.dimensions == 0
                    && *heads > 0
                    && *head_dim > 0
                    && *top_k > 0
                    && pool.pool > 0
                    && index_width == (*head_dim as usize) * 2 => {}
                _ => {
                    return Err(
                        "rotated NVFP4 requires declared NoPE rank512 DSA k-pool attention",
                    );
                }
            }
        }
        layout_for_selection(enabled, width, index_width)
    }

    /// Storage planes only. Callers must additionally account for index state,
    /// device counters, workspace, prefix copies and draft caches.
    pub fn allocation_bytes(self, rows: usize) -> Result<usize, &'static str> {
        self.payload_bytes
            .checked_add(self.block_scale_bytes)
            .and_then(|n| n.checked_add(self.row_scale_bytes))
            .and_then(|n| n.checked_mul(rows))
            .ok_or("latent allocation overflow")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mix(mut x: u64) -> u64 {
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }

    // Independent dense matrix reference; not the production butterfly loop.
    fn reference(x: &[f32], inverse: bool) -> Vec<f64> {
        (0..512usize)
            .map(|j| {
                x.iter()
                    .enumerate()
                    .map(|(i, x)| {
                        let sign_column = if inverse { j } else { i };
                        let negative = (mix(RHT512_SEED.wrapping_add(sign_column as u64)) >> 63
                            != 0)
                            ^ ((i & j).count_ones() % 2 == 1);
                        *x as f64 * if negative { -1. } else { 1. }
                    })
                    .sum::<f64>()
                    / 512f64.sqrt()
            })
            .collect()
    }

    fn squared_norm(x: &[f32]) -> f64 {
        x.iter().map(|x| (*x as f64).powi(2)).sum()
    }
    fn dot(x: &[f32], y: &[f32]) -> f64 {
        x.iter().zip(y).map(|(x, y)| *x as f64 * *y as f64).sum()
    }

    #[test]
    fn frozen_mask_and_normalizer_identity() {
        assert_eq!(
            RHT512_NORMALIZER_BITS,
            (1. / 512f64.sqrt() as f32).to_bits()
        );
        for i in 0..512 {
            assert_eq!(
                (RHT512_SIGN_MASK[i / 64] >> (i % 64)) & 1,
                mix(RHT512_SEED.wrapping_add(i as u64)) >> 63
            );
        }
        assert_eq!(LatentBasis::Identity as u8, 0);
        assert_eq!(LatentBasis::Rht512V1 as u8, 1);
    }

    #[test]
    fn rht_roundtrip_norm_and_dense_f64_reference() {
        let rows = [
            vec![1.; 512],
            (0..512).map(|i| if i == 37 { 1. } else { 0. }).collect(),
            (0..512)
                .map(|i| (((i * 73 + 17) % 509) as f32 - 254.) / 254.)
                .collect(),
        ];
        for x in rows {
            let y = rht512_forward(&x).unwrap();
            let back = rht512_inverse(&y).unwrap();
            for (actual, expected) in y.iter().zip(reference(&x, false)) {
                assert!((*actual as f64 - expected).abs() < 4e-6);
            }
            for (actual, expected) in back.iter().zip(reference(&y, true)) {
                assert!((*actual as f64 - expected).abs() < 4e-6);
            }
            let error = back
                .iter()
                .zip(&x)
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum::<f64>();
            assert!(error.sqrt() <= 2e-6 * squared_norm(&x).sqrt());
            assert!((squared_norm(&y) - squared_norm(&x)).abs() <= 2e-6 * squared_norm(&x));
        }
    }

    #[test]
    fn score_and_weighted_value_cancel_but_wrong_inverse_does_not() {
        let q: Vec<_> = (0..512)
            .map(|i| (((i * 37 + 11) % 503) as f32 - 251.) / 251.)
            .collect();
        let rows: Vec<Vec<f32>> = (0..3)
            .map(|r| {
                (0..512)
                    .map(|i| (((i * (r * 20 + 13) + r * 17) % 509) as f32 - 254.) / 254.)
                    .collect()
            })
            .collect();
        let qr = rht512_forward(&q).unwrap();
        let rotated: Vec<_> = rows.iter().map(|x| rht512_forward(x).unwrap()).collect();
        for (z, zr) in rows.iter().zip(&rotated) {
            assert!(
                (dot(&q, z) - dot(&qr, zr)).abs()
                    < 2e-6 * (squared_norm(&q) * squared_norm(z)).sqrt()
            );
        }
        let weights = [0.25f32, 0.5, 0.25];
        let sum = |xs: &[Vec<f32>]| -> Vec<f32> {
            (0..512)
                .map(|i| (0..3).map(|r| weights[r] * xs[r][i]).sum())
                .collect()
        };
        let original = sum(&rows);
        let out_r = sum(&rotated);
        let correct = rht512_inverse(&out_r).unwrap();
        assert!(
            correct
                .iter()
                .zip(&original)
                .all(|(a, b)| (*a - *b).abs() < 3e-6)
        );
        let wrong = rht512_forward(&out_r).unwrap(); // deliberate D H instead of H D
        assert!(
            wrong
                .iter()
                .zip(&original)
                .any(|(a, b)| (*a - *b).abs() > 0.1)
        );
    }

    #[test]
    fn zero_subnormal_and_overflow_refusal() {
        for value in [
            0.,
            -0.,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::MIN_POSITIVE,
        ] {
            let x = vec![value; 512];
            let y = rht512_forward(&x).unwrap();
            assert!(y.iter().all(|x| x.is_finite()));
            let back = rht512_inverse(&y).unwrap();
            assert!(back.iter().all(|x| x.is_finite()));
            if value == 0. {
                assert!(back.iter().all(|x| *x == 0.));
            }
        }
        let original = vec![f32::MAX; 512];
        assert!(rht512_forward(&original).is_err());
        assert!(rht512_inverse(&original).is_err());
        assert_eq!(original, vec![f32::MAX; 512]); // caller's input remains intact
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(rht512_forward(&vec![value; 512]).is_err());
            assert!(rht512_inverse(&vec![value; 512]).is_err());
        }
        assert!(rht512_forward(&[0.; 511]).is_err());
        assert!(rht512_inverse(&[0.; 513]).is_err());
    }

    #[test]
    fn runtime_selection_and_basis_copy_are_explicit() {
        let compressed = layout_for_selection(true, 512, 256).unwrap();
        assert_eq!(compressed.basis, LatentBasis::Rht512V1);
        assert_eq!(compressed.allocation_bytes(1).unwrap(), 292);
        assert_eq!(
            layout_for_selection(false, 512, 256).unwrap().basis,
            LatentBasis::Identity
        );
        assert_eq!(
            LatentLayout::new(LatentFormat::Nvfp4, 512).unwrap().basis,
            LatentBasis::Identity
        );
        assert!(
            LatentBasis::Rht512V1
                .validate_copy(LatentBasis::Identity)
                .is_err()
        );
        assert!(
            LatentBasis::Identity
                .validate_copy(LatentBasis::Rht512V1)
                .is_err()
        );
        assert!(
            LatentBasis::Rht512V1
                .validate_copy(LatentBasis::Rht512V1)
                .is_ok()
        );
        assert!(
            LatentLayout::new_with_basis(LatentFormat::Nvfp4, 576, LatentBasis::Rht512V1).is_err()
        );
        assert!(
            LatentLayout::new_with_basis(LatentFormat::F32, 512, LatentBasis::Rht512V1).is_err()
        );
    }

    #[test]
    fn model_geometry_not_width_alone_controls_rotated_basis() {
        use memra_gguf::model_plan::{
            AttentionPlan, KpoolPlan, MlaAttentionPlan, RopeFactors, RopePlan, SparseIndexPlan,
        };
        let valid = AttentionPlan::Mla(MlaAttentionPlan::LatentKv {
            query_heads: 64,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_head_dim: 256,
            rope_head_dim: 0,
            value_head_dim: 256,
            rope: RopePlan {
                dimensions: 0,
                base: 10000.,
                factors: RopeFactors::None,
            },
            sparse_index: SparseIndexPlan::Own {
                heads: 32,
                head_dim: 128,
                top_k: 2048,
                kpool: Some(KpoolPlan {
                    pool: 4,
                    always_select_tail: true,
                }),
            },
        });
        assert_eq!(
            LatentLayout::for_attention(true, &valid, 512, 256)
                .unwrap()
                .basis,
            LatentBasis::Rht512V1
        );
        let mut wrong = valid.clone();
        if let AttentionPlan::Mla(MlaAttentionPlan::LatentKv { rope_head_dim, .. }) = &mut wrong {
            *rope_head_dim = 64;
        }
        assert!(LatentLayout::for_attention(true, &wrong, 512, 256).is_err());
        if let AttentionPlan::Mla(MlaAttentionPlan::LatentKv {
            kv_lora_rank,
            rope_head_dim,
            ..
        }) = &mut wrong
        {
            *kv_lora_rank = 448;
            *rope_head_dim = 0;
        }
        assert!(LatentLayout::for_attention(true, &wrong, 512, 256).is_err());
        assert!(LatentLayout::for_attention(true, &valid, 512, 128).is_err());
        assert!(LatentLayout::for_attention(false, &wrong, 512, 256).is_ok());
    }

    #[test]
    fn nvfp4_counts_both_scale_planes() {
        let layout = LatentLayout::new(LatentFormat::Nvfp4, 512).unwrap();
        assert_eq!(
            (
                layout.payload_bytes,
                layout.block_scale_bytes,
                layout.row_scale_bytes
            ),
            (256, 32, 4)
        );
        assert_eq!(layout.allocation_bytes(218_000), Ok(63_656_000));
        assert_eq!(
            LatentLayout::new(LatentFormat::F32, 512)
                .unwrap()
                .allocation_bytes(218_000),
            Ok(446_464_000)
        );
    }

    #[test]
    fn refuses_invalid_widths_and_overflow() {
        for format in [LatentFormat::F32, LatentFormat::Nvfp4] {
            assert!(LatentLayout::new(format, 0).is_err());
            assert!(
                LatentLayout::new(format, 512)
                    .unwrap()
                    .allocation_bytes(usize::MAX)
                    .is_err()
            );
        }
        assert!(LatentLayout::new(LatentFormat::Nvfp4, 511).is_err());
        assert!(LatentLayout::new(LatentFormat::F32, usize::MAX).is_err());
    }

    #[test]
    fn empty_history_has_no_plane_storage() {
        for format in [LatentFormat::F32, LatentFormat::Nvfp4] {
            assert_eq!(
                LatentLayout::new(format, 576).unwrap().allocation_bytes(0),
                Ok(0)
            );
        }
    }
}
