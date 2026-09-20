//! CUDA-free arithmetic used by the loaded-model memory inventory.
use std::collections::BTreeMap;

/// Allocation owners are identities, not physical ordinals. Shared runtime references
/// collapse; independent engines on one card must each be synchronized.
pub(crate) fn unique_owners<'a, T: 'a>(owners: impl IntoIterator<Item = &'a T>) -> Vec<&'a T> {
    let mut out: Vec<&'a T> = Vec::new();
    for owner in owners {
        if !out.iter().any(|&seen| std::ptr::eq(seen, owner)) {
            out.push(owner);
        }
    }
    out
}

#[derive(Default)]
pub(crate) struct DeviceBytes(pub BTreeMap<usize, usize>);

impl DeviceBytes {
    pub fn add(&mut self, device: usize, bytes: usize) -> Result<(), String> {
        let total = self.0.entry(device).or_default();
        *total = total
            .checked_add(bytes)
            .ok_or("device state byte count overflow")?;
        Ok(())
    }
}

pub(crate) fn f32_bytes(elements: usize) -> Result<usize, String> {
    elements
        .checked_mul(4)
        .ok_or_else(|| "state byte count overflow".into())
}

/// Extra KDA core allocations not itemized by the generic prime-workspace shape:
/// q/k/v raw and conv (6Q), l2 pair (2Q), forget/g_log/core/gate/gated (5Q),
/// low-rank forget/gate (2D), beta raw/activated (2H). The scan itself has no
/// device scratch. Count all of them conservatively, even if a last-use frees early.
pub(crate) fn kda_workspace_row_bytes(heads: usize, dim: usize) -> Result<usize, String> {
    let elements = heads
        .checked_mul(dim)
        .and_then(|q| q.checked_mul(13))
        .and_then(|q| dim.checked_mul(2).and_then(|d| q.checked_add(d)))
        .and_then(|q| heads.checked_mul(2).and_then(|h| q.checked_add(h)))
        .ok_or("KDA workspace geometry overflow")?;
    f32_bytes(elements)
}

/// A peer's conservative workspace envelope is a floor, not a measured allocation.
/// The ordinary model cost can be zero/small, so it must not cap the caller's explicit
/// per-rank transient floor. A one-row call still covers decode on an empty suffix.
pub(crate) fn peer_workspace_bytes(
    primary_workspace: usize,
    extra_row_bytes: usize,
    call_rows: usize,
    transient_floor: usize,
) -> Result<usize, String> {
    let workspace = extra_row_bytes
        .checked_mul(call_rows.max(1))
        .and_then(|extra| primary_workspace.checked_add(extra))
        .ok_or("peer workspace byte count overflow")?;
    Ok(workspace.max(transient_floor))
}

/// Replicated k-pool score plane: one f32 per query row and complete context pool.
pub(crate) fn mla_score_bytes(
    call_rows: usize,
    prompt_rows: usize,
    pool: usize,
) -> Result<usize, String> {
    let elements = prompt_rows
        .checked_div(pool)
        .and_then(|pools| pools.checked_mul(call_rows))
        .ok_or("invalid MLA score geometry")?;
    f32_bytes(elements)
}

pub(crate) fn kda_elements(heads: usize, dim: usize, kernel: usize) -> Result<[usize; 3], String> {
    let conv = heads
        .checked_mul(dim)
        .and_then(|v| v.checked_mul(3))
        .and_then(|v| kernel.checked_sub(1).and_then(|k| v.checked_mul(k)))
        .ok_or("invalid KDA convolution geometry")?;
    let state = heads
        .checked_mul(dim)
        .and_then(|v| v.checked_mul(dim))
        .ok_or("KDA recurrent geometry overflow")?;
    Ok([conv, state, state])
}

/// Sum allocations still needed. A short live allocation remains driver-owned while
/// its replacement is allocated, so reserve the whole replacement, not only its delta.
pub(crate) fn missing_f32_bytes(required: &[usize], live: &[usize]) -> Result<usize, String> {
    required
        .iter()
        .enumerate()
        .try_fold(0usize, |sum, (i, &need)| {
            let missing = if live.get(i).copied().unwrap_or(0) >= need {
                0
            } else {
                f32_bytes(need)?
            };
            sum.checked_add(missing)
                .ok_or_else(|| "state byte count overflow".into())
        })
}

/// Mirrors Cache::new_inner and ensure_mla_peer_latent: latent + index rows + len_d,
/// followed by the lazy k-pool key plane (mla_kpool_indices, including max(1)).
pub(crate) fn mla_elements(
    capacity: usize,
    width: usize,
    index_width: usize,
    index_rows: usize,
    pool: usize,
) -> Result<[usize; 4], String> {
    let rows = capacity
        .checked_mul(width)
        .ok_or("MLA latent geometry overflow")?;
    let index = index_rows
        .checked_mul(index_width)
        .ok_or("MLA index geometry overflow")?;
    let keys = if index_width == 0 {
        0
    } else {
        capacity
            .checked_div(pool)
            .and_then(|n| n.checked_mul(index_width / 2))
            .ok_or("invalid MLA pool geometry")?
            .max(1)
    };
    Ok([rows, index, 1, keys])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_owners_collapse_but_same_ordinal_engines_survive() {
        let primary = (0, "primary");
        let pp = (1, "PP");
        let step = (0, "Step context");
        let glm = (1, "GLM context");
        let expert_only = (2, "expert-only context");
        let owners = unique_owners([&primary, &pp, &step, &glm, &glm, &expert_only, &step]);
        assert_eq!(owners, [&primary, &pp, &step, &glm, &expert_only]);
        let old_ordinal_dedup: std::collections::BTreeSet<_> = owners.iter().map(|o| o.0).collect();
        assert_eq!(old_ordinal_dedup.len(), 3);
        assert_eq!(
            owners.len(),
            5,
            "ordinal dedup would miss two synchronization owners"
        );
    }

    #[test]
    fn kda_reserves_convolution_and_both_recurrent_planes() {
        let rank = kda_elements(2, 128, 4).unwrap();
        assert_eq!(rank, [2304, 32768, 32768]);
        assert_eq!(missing_f32_bytes(&rank, &[]).unwrap(), 271360);
        assert_eq!(missing_f32_bytes(&rank, &[2304, 32768, 0]).unwrap(), 131072);
        assert_eq!(missing_f32_bytes(&rank, &rank).unwrap(), 0);
    }

    #[test]
    fn mla_ring_and_flat_layouts_include_lazy_keys_and_length_scalar() {
        let ring = mla_elements(8192, 576, 256, 5120, 4).unwrap();
        assert_eq!(ring, [4718592, 1310720, 1, 262144]);
        assert_eq!(missing_f32_bytes(&ring, &ring[..3]).unwrap(), 1048576);
        let flat = mla_elements(8192, 576, 256, 8192, 4).unwrap();
        assert_eq!(missing_f32_bytes(&flat, &[]).unwrap(), 28311556);
        assert_eq!(mla_elements(3, 16, 8, 3, 4).unwrap(), [48, 24, 1, 1]);
        assert_eq!(mla_elements(3, 16, 0, 3, 0).unwrap(), [48, 0, 1, 0]);
    }

    #[test]
    fn partial_layers_and_same_device_ranks_sum_without_charging_live_bytes() {
        let shape = mla_elements(8, 16, 8, 8, 4).unwrap();
        let mut bytes = DeviceBytes::default();
        bytes.add(0, 0).unwrap(); // canonical MLA plane is covered by base cache accounting
        bytes
            .add(1, missing_f32_bytes(&shape, &shape).unwrap())
            .unwrap();
        bytes
            .add(1, missing_f32_bytes(&shape, &shape[..3]).unwrap())
            .unwrap();
        bytes.add(2, 0).unwrap(); // fully materialized / expert-only peer retains its reserve
        assert_eq!(
            bytes.0.into_iter().collect::<Vec<_>>(),
            [(0, 0), (1, 32), (2, 0)]
        );
    }

    #[test]
    fn growing_allocations_reserve_the_replacement_while_old_bytes_are_live() {
        assert_eq!(missing_f32_bytes(&[256], &[128]).unwrap(), 1024);
        assert_eq!(missing_f32_bytes(&[128], &[256]).unwrap(), 0);
    }

    #[test]
    fn malformed_geometry_and_overflow_fail_closed() {
        assert!(kda_elements(1, 128, 0).is_err());
        assert!(kda_elements(usize::MAX, 128, 4).is_err());
        assert!(mla_elements(8, 16, 8, 8, 0).is_err());
        assert!(mla_elements(usize::MAX, 16, 8, 8, 4).is_err());
        assert!(missing_f32_bytes(&[usize::MAX], &[]).is_err());
        let mut bytes = DeviceBytes::default();
        bytes.add(1, usize::MAX).unwrap();
        assert!(bytes.add(1, 1).is_err());
    }

    #[test]
    fn peer_workspace_is_not_capped_by_small_primary_session_cost() {
        let row = kda_workspace_row_bytes(2, 128).unwrap();
        assert_eq!(row, 14352);
        assert_eq!(peer_workspace_bytes(0, 0, 0, 1 << 30).unwrap(), 1 << 30);
        assert_eq!(peer_workspace_bytes(1000, row, 512, 4096).unwrap(), 7349224);
        assert!(peer_workspace_bytes(usize::MAX, row, 1, 0).is_err());
    }

    #[test]
    fn mla_peer_score_workspace_scales_with_prompt_depth_and_call_rows() {
        assert_eq!(mla_score_bytes(512, 262144, 4).unwrap(), 134217728);
        assert_eq!(mla_score_bytes(512, 8192, 4).unwrap(), 4194304);
        assert_eq!(mla_score_bytes(1, 3, 4).unwrap(), 0);
        assert!(mla_score_bytes(512, 8192, 0).is_err());
        assert!(mla_score_bytes(usize::MAX, 8192, 4).is_err());
    }
}
