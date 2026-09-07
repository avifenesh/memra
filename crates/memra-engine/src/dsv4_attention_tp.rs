//! Gate-only geometry for DSV4 head-parallel attention and a rank-order output sum.
//!
//! KV latent, compressor and indexer state stay replicated. The output projection's
//! input split changes the accumulation program; it is not a full-width identity claim.

use std::sync::atomic::{AtomicBool, Ordering};

pub const ATTENTION_TP_NUMERIC_CLASS: &str = "dsv4_attention_wo_b_input_split_f32_rank_reduce";

static ENABLED_FOR_GATE: AtomicBool = AtomicBool::new(false);

/// Before-load diagnostic control only. Existing model instances never change program.
pub fn set_for_gate(enabled: bool) -> bool {
    ENABLED_FOR_GATE.swap(enabled, Ordering::AcqRel)
}

pub(crate) fn enabled_for_gate() -> bool {
    ENABLED_FOR_GATE.load(Ordering::Acquire)
}

/// Derived from the normalized model configuration, not a fixed layer/head allowlist.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionTpGeometry {
    pub full_heads: usize,
    pub local_heads: usize,
    pub head_dim: usize,
    pub q_lora: usize,
    pub full_groups: usize,
    pub local_groups: usize,
    pub o_lora: usize,
    pub hidden: usize,
    pub group_width: usize,
    pub full_output_width: usize,
    pub local_output_width: usize,
}

/// Actual final-layer device partials and their native GPU rank sums, read only by gates.
pub struct AttentionTpJoinSnapshot {
    pub partials: [Vec<f32>; 2],
    pub joined: [Vec<f32>; 2],
}

impl AttentionTpGeometry {
    pub fn new(
        heads: usize,
        head_dim: usize,
        q_lora: usize,
        groups: usize,
        o_lora: usize,
        hidden: usize,
    ) -> Result<Self, String> {
        if heads == 0
            || groups == 0
            || !heads.is_multiple_of(2)
            || !groups.is_multiple_of(2)
            || !heads.is_multiple_of(groups)
        {
            return Err(format!(
                "attention TP2 requires even head/group counts and whole heads per group: heads={heads} groups={groups}"
            ));
        }
        let product = |a: usize, b: usize| {
            a.checked_mul(b)
                .filter(|&n| n <= i32::MAX as usize)
                .ok_or_else(|| "attention TP2 projection geometry overflow".to_string())
        };
        let full_q_rows = product(heads, head_dim)?;
        let group_width = product(heads / groups, head_dim)?;
        let full_output_width = product(groups, o_lora)?;
        for value in [
            head_dim,
            q_lora,
            o_lora,
            hidden,
            full_q_rows / 2,
            group_width,
            full_output_width / 2,
        ] {
            if value == 0 || value > i32::MAX as usize || !value.is_multiple_of(128) {
                return Err(format!(
                    "attention TP2 FP8 geometry must be nonzero and 128-aligned: {value}"
                ));
            }
        }
        Ok(Self {
            full_heads: heads,
            local_heads: heads / 2,
            head_dim,
            q_lora,
            full_groups: groups,
            local_groups: groups / 2,
            o_lora,
            hidden,
            group_width,
            full_output_width,
            local_output_width: full_output_width / 2,
        })
    }

    pub fn head_start(self, rank: usize) -> Result<usize, String> {
        if rank >= 2 {
            return Err(format!("attention TP2 rank {rank} outside two ranks"));
        }
        Ok(rank * self.local_heads)
    }
}

#[cfg(test)]
mod tests {
    use super::AttentionTpGeometry;

    #[test]
    fn model_geometry_has_whole_head_and_group_partitions() {
        let plan = AttentionTpGeometry::new(64, 512, 1024, 8, 1024, 4096).unwrap();
        assert_eq!((plan.local_heads, plan.local_groups), (32, 4));
        assert_eq!((plan.group_width, plan.local_output_width), (4096, 4096));
        assert_eq!(plan.head_start(0).unwrap(), 0);
        assert_eq!(plan.head_start(1).unwrap(), 32);
        assert!(plan.head_start(2).is_err());
    }

    #[test]
    fn refuses_partial_groups_unaligned_scales_and_overflow() {
        for args in [
            (63, 512, 1024, 8, 1024, 4096),
            (64, 512, 1024, 6, 1024, 4096),
            (64, 512, 1000, 8, 1024, 4096),
            (64, 0, 1024, 8, 1024, 4096),
            (usize::MAX - 1, 512, 1024, 2, 1024, 4096),
        ] {
            assert!(
                AttentionTpGeometry::new(args.0, args.1, args.2, args.3, args.4, args.5).is_err()
            );
        }
    }
}
