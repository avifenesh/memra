//! Split every expert across the TP ranks, instead of giving each rank whole experts.
//!
//! WHY (lane/tp-expert-split-20260906). glm5's TP arm is expert-PARALLEL: each rank OWNS half the
//! experts (144 of 288). A token routes to 8, so the split across two ranks is Binomial(8, 0.5),
//! and a memory-bound decode step is paced by the BUSIER rank, not the average. The arithmetic is
//! blunt: `E[max(X, 8-X)] = 5.094`, so expert-parallelism buys a **1.571x** expert-half speedup
//! where the TP sizing assumed 2x, and the even 4/4 split people picture happens on only 27.3% of
//! tokens while 28.9% land 6-2 or worse. Routing luck throws away a quarter of the win, every
//! token, and no amount of transport work touches it.
//!
//! Tensor-parallelism inside each expert removes the variance instead of averaging it: rank `r`
//! holds ROWS `[r*half, (r+1)*half)` of every expert's `gate` and `up`, and the matching COLUMNS
//! of its `down`. Each rank then streams exactly half of every routed expert, deterministically,
//! whatever the router picks.
//!
//! WHAT IS EXACT AND WHAT IS NOT, stated up front because the halves differ:
//!   * `gate`/`up` are ROW splits. Every output element stays one full-K dot by the same kernel
//!     over the same bytes, so this half is BIT-IDENTICAL to the unsharded expert, and so is the
//!     SwiGLU that follows it (elementwise, each rank on its own rows).
//!   * `down` is a COLUMN split, so each rank produces a PARTIAL SUM over half the K range and
//!     the ranks' partials are added. That is a 2-way split of a reduction the unsharded walk
//!     does in one pass, so it is NOT bit-identical and takes a named numeric class, exactly as
//!     `dsa-warp-online-f32` did for the DSA rewrite.
//!
//! THE COLUMN SPLIT MUST LAND ON A BLOCK BOUNDARY, and the check has to ask the block table
//! rather than infer. `staged_expert_row_bytes` is `row_bytes = in_f / block * type_size`
//! (`model.rs:1676`), linear in `in_f`, so every staged expert qtype lays its blocks ascending
//! along the row and a column prefix IS a byte prefix. The trap is that "the byte offset divides
//! evenly" does NOT imply "the half is a whole number of blocks": at `in_f` 96 with Q8_0 (block
//! 32, type_size 34, `row_bytes` 102) a 48-column half is one and a half blocks and still lands
//! on byte 51 exactly. This module's red arm caught that, so the guard counts BLOCKS, from
//! `GgmlType::block_and_type_size` in memra-gguf, which is the one place block sizes are
//! defined. Layouts where a prefix cannot reach the bytes refuse by name instead of slicing: a
//! per-expert `layouts` table, a native block scale plane (`fp8_blk`), or a tiered bank.
use crate::model::HostExps;
use memra_gguf::GgmlType;

/// Block size for a staged expert qtype, from the ONE table that defines them
/// (`GgmlType::block_and_type_size`). An unknown qtype returns `None` and its caller refuses,
/// because a split that guesses a block size slices through one.
fn qtype_block(qtype: i32) -> Option<u64> {
    let ty = match qtype {
        q if q == crate::QT_Q8_0 => GgmlType::Q8_0,
        q if q == crate::QT_Q2_K => GgmlType::Q2_K,
        q if q == crate::QT_Q3_K => GgmlType::Q3_K,
        q if q == crate::QT_Q4_K => GgmlType::Q4_K,
        q if q == crate::QT_Q5_K => GgmlType::Q5_K,
        q if q == crate::QT_Q6_K => GgmlType::Q6_K,
        q if q == crate::QT_IQ4_XS => GgmlType::IQ4_XS,
        q if q == crate::QT_IQ3_S => GgmlType::IQ3_S,
        q if q == crate::QT_NVFP4 => GgmlType::NVFP4,
        q if q == crate::QT_F32 => GgmlType::F32,
        q if q == crate::QT_BF16 => GgmlType::BF16,
        _ => return None,
    };
    Some(ty.block_and_type_size().0)
}

/// One rank's share of one projection: the raw bytes plus the dims the forward reads off
/// `HostExps`, so a sharded bank flows through the existing width-parameterized walk unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertShard {
    pub bytes: Vec<u8>,
    pub in_f: usize,
    pub out_f: usize,
    pub row_bytes: usize,
    pub expert_stride: usize,
}

fn refuse_unsplittable(h: &HostExps, what: &str) -> Result<(), Box<dyn std::error::Error>> {
    if h.tiers.is_some() {
        return Err(format!(
            "tp expert split: {what} is per-expert tiered (spilling plan); the split needs one \
             contiguous bank"
        )
        .into());
    }
    if h.layouts.is_some() {
        return Err(format!(
            "tp expert split: {what} carries a per-expert layouts table, so experts do not share \
             one row layout and a byte prefix is not a column prefix"
        )
        .into());
    }
    if h.fp8_blk.is_some() {
        return Err(format!(
            "tp expert split: {what} carries a native block-E4M3 scale plane, which is indexed \
             [expert, output_block, input_block] and does not split with the code bytes"
        )
        .into());
    }
    Ok(())
}

/// ROW split of a `[in_f, out_f, n_expert]` projection (`gate`, `up`): rank `r` takes out-rows
/// `[r*out_f/ranks, (r+1)*out_f/ranks)` of every expert. Contiguous inside each expert, so this
/// is a byte range and nothing is rewritten. BIT-IDENTICAL: each kept row is the row the
/// unsharded bank holds, and every output element remains one full-K dot.
pub fn split_rows(
    h: &HostExps,
    ranks: usize,
    rank: usize,
) -> Result<ExpertShard, Box<dyn std::error::Error>> {
    refuse_unsplittable(h, "a row-split projection")?;
    if ranks == 0 || rank >= ranks {
        return Err(format!("tp expert split: rank {rank} of {ranks}").into());
    }
    if !h.out_f.is_multiple_of(ranks) {
        return Err(format!(
            "tp expert split: out_f {} does not divide across {ranks} ranks",
            h.out_f
        )
        .into());
    }
    let half = h.out_f / ranks;
    let src = h.bytes.as_bytes();
    let stride = half * h.row_bytes;
    let mut bytes = Vec::with_capacity(h.n_expert * stride);
    for ex in 0..h.n_expert {
        let base = ex * h.expert_stride + rank * stride;
        bytes.extend_from_slice(&src[base..base + stride]);
    }
    Ok(ExpertShard {
        bytes,
        in_f: h.in_f,
        out_f: half,
        row_bytes: h.row_bytes,
        expert_stride: stride,
    })
}

/// COLUMN split of a `[in_f, out_f, n_expert]` projection (`down`, whose `in_f` is the expert's
/// intermediate width): rank `r` takes input columns `[r*in_f/ranks, (r+1)*in_f/ranks)` of every
/// row of every expert. Each row contributes one contiguous byte range, so the result is a gather
/// of `out_f` ranges per expert.
///
/// PARTIAL SUMS: a rank's output is the dot over ITS HALF of the K range, so the ranks' outputs
/// must be added to reproduce the unsharded row. That addition is the named numeric class this
/// arm carries; the bytes themselves are untouched.
pub fn split_cols(
    h: &HostExps,
    ranks: usize,
    rank: usize,
) -> Result<ExpertShard, Box<dyn std::error::Error>> {
    refuse_unsplittable(h, "a column-split projection")?;
    if ranks == 0 || rank >= ranks {
        return Err(format!("tp expert split: rank {rank} of {ranks}").into());
    }
    if !h.in_f.is_multiple_of(ranks) {
        return Err(format!(
            "tp expert split: in_f {} does not divide across {ranks} ranks",
            h.in_f
        )
        .into());
    }
    let half = h.in_f / ranks;
    // The only thing standing between this and slicing through a quantization block. Count
    // BLOCKS, not bytes: at in_f 96 with Q8_0 (block 32, row_bytes 102) a 48-column half is one
    // and a half blocks and STILL divides evenly into 51 bytes, so a byte-divisibility test
    // passes a slice that cuts a block in half. The red arm below is that exact case.
    let Some(block) = qtype_block(h.qtype) else {
        return Err(format!(
            "tp expert split: qtype {} has no known block size, so a column split cannot be \
             proven to land on a block boundary",
            h.qtype
        )
        .into());
    };
    if !(half as u64).is_multiple_of(block) {
        return Err(format!(
            "tp expert split: column half {half} of {} does not land on a block boundary \
             (qtype {} block {block})",
            h.in_f, h.qtype
        )
        .into());
    }
    debug_assert_eq!(
        (half * h.row_bytes) % h.in_f,
        0,
        "a block-aligned half must also divide the row bytes"
    );
    let keep = half * h.row_bytes / h.in_f;
    let src = h.bytes.as_bytes();
    let stride = h.out_f * keep;
    let mut bytes = Vec::with_capacity(h.n_expert * stride);
    for ex in 0..h.n_expert {
        let ebase = ex * h.expert_stride;
        for row in 0..h.out_f {
            let base = ebase + row * h.row_bytes + rank * keep;
            bytes.extend_from_slice(&src[base..base + keep]);
        }
    }
    Ok(ExpertShard {
        bytes,
        in_f: half,
        out_f: h.out_f,
        row_bytes: keep,
        expert_stride: stride,
    })
}

/// What whole-expert ownership costs at batch 1, as a closed form, so the sizing in the lane doc
/// can be re-derived rather than trusted: with `n_used` experts drawn over `ranks` equal owners,
/// a memory-bound step is paced by the busiest rank, so the cost is `E[max]` and not `n_used /
/// ranks`. Returns `(E[max], speedup)` for two ranks.
pub fn ep_busier_rank_experts(n_used: usize) -> (f64, f64) {
    fn comb(n: usize, k: usize) -> f64 {
        let mut v = 1.0;
        for i in 0..k {
            v = v * (n - i) as f64 / (i + 1) as f64;
        }
        v
    }
    let total: f64 = (0..=n_used).map(|k| comb(n_used, k)).sum();
    let e_max: f64 = (0..=n_used)
        .map(|k| comb(n_used, k) * k.max(n_used - k) as f64)
        .sum::<f64>()
        / total;
    (e_max, n_used as f64 / e_max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::HostBuf;

    /// A synthetic bank whose every byte encodes (expert, row, column-byte), so a misplaced slice
    /// is a wrong VALUE and not just a wrong length. Dims mirror the served glm5 shape's ratios:
    /// `gate`/`up` are `[n_embd, moe_inter]` per expert and `down` is `[moe_inter, n_embd]`.
    fn bank(in_f: usize, out_f: usize, n_expert: usize, row_bytes: usize) -> HostExps {
        let stride = out_f * row_bytes;
        let mut v = vec![0u8; n_expert * stride];
        for (i, b) in v.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        HostExps {
            bytes: HostBuf::Paged(v),
            tiers: None,
            qtype: crate::QT_Q8_0,
            in_f,
            out_f,
            n_expert,
            row_bytes,
            expert_stride: stride,
            layouts: None,
            macros: None,
            fp8_blk: None,
        }
    }

    /// The row halves must reassemble the original expert byte for byte, in rank order. This is
    /// the bit-identity claim for the `gate`/`up` half stated as a test.
    #[test]
    fn row_split_halves_reassemble_the_bank() {
        let h = bank(4096, 2048, 3, 34);
        let a = split_rows(&h, 2, 0).unwrap();
        let b = split_rows(&h, 2, 1).unwrap();
        assert_eq!((a.out_f, a.in_f, a.row_bytes), (1024, 4096, 34));
        assert_eq!(a.expert_stride, 1024 * 34);
        let src = h.bytes.as_bytes();
        for ex in 0..h.n_expert {
            let want = &src[ex * h.expert_stride..(ex + 1) * h.expert_stride];
            let mut got = Vec::new();
            got.extend_from_slice(&a.bytes[ex * a.expert_stride..(ex + 1) * a.expert_stride]);
            got.extend_from_slice(&b.bytes[ex * b.expert_stride..(ex + 1) * b.expert_stride]);
            assert_eq!(got, want, "expert {ex} row halves do not reassemble");
        }
    }

    /// The column halves must reassemble each ROW of each expert, which is a different claim from
    /// the row split: the halves interleave per row rather than concatenating per expert.
    #[test]
    fn column_split_halves_reassemble_every_row() {
        let h = bank(2048, 4096, 3, 68);
        let a = split_cols(&h, 2, 0).unwrap();
        let b = split_cols(&h, 2, 1).unwrap();
        assert_eq!((a.in_f, a.out_f, a.row_bytes), (1024, 4096, 34));
        assert_eq!(a.expert_stride, 4096 * 34);
        let src = h.bytes.as_bytes();
        for ex in 0..h.n_expert {
            for row in 0..h.out_f {
                let base = ex * h.expert_stride + row * h.row_bytes;
                let want = &src[base..base + h.row_bytes];
                let ab = ex * a.expert_stride + row * a.row_bytes;
                let bb = ex * b.expert_stride + row * b.row_bytes;
                let mut got = Vec::new();
                got.extend_from_slice(&a.bytes[ab..ab + a.row_bytes]);
                got.extend_from_slice(&b.bytes[bb..bb + b.row_bytes]);
                assert_eq!(
                    got, want,
                    "expert {ex} row {row} column halves do not reassemble"
                );
            }
        }
    }

    /// RED ARM, and it earned its keep: the first version of this guard tested that the byte
    /// offset divided evenly, which this case PASSES while cutting a block in half. in_f 96 with
    /// Q8_0 (block 32, type_size 34, row_bytes 102) has a 48-column half of one and a half
    /// blocks, and 48*102/96 is exactly 51. Counting blocks refuses it; counting bytes does not.
    /// The neighbouring width that does land on a boundary must still pass, so this proves the
    /// guard discriminates rather than that it always fires.
    #[test]
    fn column_split_refuses_a_half_that_lands_mid_block() {
        let bad = bank(96, 8, 1, 102);
        assert_eq!(
            (48 * 102) % 96,
            0,
            "the byte-divisibility test would have passed this"
        );
        let err = split_cols(&bad, 2, 0).expect_err("a mid-block half must refuse");
        assert!(err.to_string().contains("block boundary"), "{err}");
        // in_f 64 at the same block size: the half is 32 columns, exactly one block.
        let good = bank(64, 8, 1, 68);
        assert_eq!(split_cols(&good, 2, 0).unwrap().row_bytes, 34);
    }

    /// Layouts where a byte prefix is NOT a column prefix must refuse by name rather than slice.
    #[test]
    fn split_refuses_layouts_a_prefix_cannot_reach() {
        let mut h = bank(2048, 4096, 2, 68);
        h.fp8_blk = None;
        h.layouts = Some(Vec::new());
        let err = split_cols(&h, 2, 0).expect_err("a per-expert layouts table must refuse");
        assert!(err.to_string().contains("layouts table"), "{err}");
        let err = split_rows(&h, 2, 0).expect_err("a per-expert layouts table must refuse");
        assert!(err.to_string().contains("layouts table"), "{err}");
    }

    /// The sizing claim the lane doc rests on, as a test rather than an assertion in prose.
    #[test]
    fn whole_expert_ownership_pays_the_busier_rank() {
        let (e_max, speedup) = ep_busier_rank_experts(8);
        assert!((e_max - 5.09375).abs() < 1e-9, "E[max] = {e_max}");
        assert!((speedup - 1.5709).abs() < 1e-3, "speedup = {speedup}");
        // Splitting every expert removes the variance entirely: the deterministic half is 4.
        assert!(
            speedup < 2.0,
            "EP cannot reach the split's deterministic 2x"
        );
    }
}
