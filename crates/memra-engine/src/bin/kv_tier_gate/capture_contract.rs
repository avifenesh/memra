//! Gate-only, CPU-testable capture contract. No allocator or numerical policy changes.
use memra_gguf::model_plan::{ModelPlan, StatePlan};

#[derive(Debug, Clone, Copy)]
pub struct KvGeometry {
    pub len: usize,
    pub ring: bool,
    pub base: bool,
    pub key_width: usize,
    pub value_width: usize,
    pub key_row: usize,
    pub value_row: usize,
    pub key_allocation: usize,
    pub value_allocation: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Plane {
    Kv,
    Recurrent,
    AbsentUnexecutedMtp,
}

/// The caller MUST use load_without_mtp + trunk-only decode_step_h. Only explicitly
/// declared MTP indices may be absent; neither a zero-length trunk nor an unknown slot
/// is excused. Validate topology once, before KV allocation/prefill.
pub fn validate_plan(plan: &ModelPlan, slots: usize) -> Result<(), String> {
    if plan.layers.is_empty() || slots != plan.layers.len() + plan.mtp_blocks.len() {
        return Err("allocated slot count != plan trunk + MTP count".into());
    }
    for (index, layer) in plan.layers.iter().enumerate() {
        if layer.index as usize != index {
            return Err(format!("trunk plan index {} != slot {index}", layer.index));
        }
        if !matches!(
            layer.state,
            StatePlan::KvCache { .. } | StatePlan::Recurrent { .. }
        ) {
            return Err(format!("unsupported trunk state at slot {index}"));
        }
    }
    for (depth, block) in plan.mtp_blocks.iter().enumerate() {
        if block.layer.index as usize != plan.layers.len() + depth
            || block.depth as usize != depth
            || !matches!(block.layer.state, StatePlan::KvCache { .. })
        {
            return Err(format!(
                "unsupported/ambiguous unexecuted MTP slot at depth {depth}"
            ));
        }
    }
    Ok(())
}

pub fn classify(
    plan: &ModelPlan,
    index: usize,
    pos: usize,
    kv: Option<KvGeometry>,
    recurrent: bool,
) -> Result<Plane, String> {
    let fail = |predicate: &str| {
        format!(
            "REFUSED: capture layer={index} pos={pos} kv={kv:?} recurrent={recurrent}; failed predicate: {predicate}"
        )
    };
    if index >= plan.layers.len() {
        let block = plan
            .mtp_blocks
            .iter()
            .find(|b| b.layer.index as usize == index)
            .ok_or_else(|| fail("slot must be declared by trunk or MTP plan"))?;
        if !matches!(block.layer.state, StatePlan::KvCache { .. }) || recurrent {
            return Err(fail(
                "unexecuted MTP must be plan-declared KV with no recurrent plane",
            ));
        }
        if let Some(g) = kv {
            if g.len != 0 {
                return Err(fail("trunk-only execution requires MTP KV len == 0"));
            }
            validate_geometry(&block.layer.state, g).map_err(|p| fail(p))?;
        }
        return Ok(Plane::AbsentUnexecutedMtp);
    }
    match (&plan.layers[index].state, kv, recurrent) {
        (StatePlan::KvCache { .. }, Some(g), false) => {
            if g.len != pos {
                return Err(fail("executed trunk KV len == cache.pos"));
            }
            validate_geometry(&plan.layers[index].state, g).map_err(|p| fail(p))?;
            Ok(Plane::Kv)
        }
        (StatePlan::Recurrent { .. }, None, true) => Ok(Plane::Recurrent),
        _ => Err(fail(
            "exactly the plan-declared trunk state plane must be present",
        )),
    }
}

fn validate_geometry(state: &StatePlan, g: KvGeometry) -> Result<(), &'static str> {
    let StatePlan::KvCache {
        key_width,
        value_width,
    } = state
    else {
        return Err("state is full-history KV");
    };
    if g.ring || g.base {
        return Err("ring and base must both be absent");
    }
    if g.key_width != *key_width as usize || g.value_width != *value_width as usize {
        return Err("KV widths equal plan key_width/value_width");
    }
    if g.key_width == 0 || !g.key_width.is_multiple_of(32) || g.key_row != g.key_width / 32 * 34 {
        return Err("native q8_0 K: nonzero width % 32 == 0 and row == width / 32 * 34");
    }
    if g.value_width == 0
        || !g.value_width.is_multiple_of(32)
        || g.value_row != g.value_width / 32 * 24
    {
        return Err("native q5_1 V: nonzero width % 32 == 0 and row == width / 32 * 24");
    }
    if g.len
        .checked_mul(g.key_row)
        .is_none_or(|n| n > g.key_allocation)
    {
        return Err("checked K valid extent <= K allocation");
    }
    if g.len
        .checked_mul(g.value_row)
        .is_none_or(|n| n > g.value_allocation)
    {
        return Err("checked V valid extent <= V allocation");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::config::{HfConfig, ModelConfig};

    fn plan() -> ModelPlan {
        let cfg = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"qwen3_5","num_hidden_layers":4,
            "num_nextn_predict_layers":1,"hidden_size":4096,"num_attention_heads":32,
            "num_key_value_heads":8,"head_dim":256,"intermediate_size":12288,
            "vocab_size":151936,"max_position_embeddings":262144,"rms_norm_eps":0.000001,
            "rope_theta":5000000,"partial_rotary_factor":0.25,"full_attention_interval":4,
            "linear_conv_kernel_dim":4,"linear_key_head_dim":128,
            "linear_value_head_dim":128,"linear_num_key_heads":16,"linear_num_value_heads":32}"#,
        ));
        ModelPlan::compile(&cfg).unwrap()
    }
    fn geometry(plan: &ModelPlan, len: usize) -> KvGeometry {
        let StatePlan::KvCache {
            key_width,
            value_width,
        } = plan.layers[3].state
        else {
            panic!()
        };
        KvGeometry {
            len,
            ring: false,
            base: false,
            key_width: key_width as usize,
            value_width: value_width as usize,
            key_row: key_width as usize / 32 * 34,
            value_row: value_width as usize / 32 * 24,
            key_allocation: 1 << 24,
            value_allocation: 1 << 24,
        }
    }
    #[test]
    fn synthetic_trunk_plus_nextn_only_declared_unexecuted_plane_is_absent() {
        let p = plan();
        assert_eq!(p.layers.len(), 4);
        assert_eq!(p.mtp_blocks.len(), 1);
        validate_plan(&p, 5).unwrap();
        assert_eq!(classify(&p, 0, 272, None, true).unwrap(), Plane::Recurrent);
        assert_eq!(
            classify(&p, 3, 272, Some(geometry(&p, 272)), false).unwrap(),
            Plane::Kv
        );
        assert_eq!(
            classify(&p, 4, 272, Some(geometry(&p, 0)), false).unwrap(),
            Plane::AbsentUnexecutedMtp
        );
        assert_eq!(
            classify(&p, 4, 272, None, false).unwrap(),
            Plane::AbsentUnexecutedMtp
        );
        for len in [0, 271, 273] {
            let error = classify(&p, 3, 272, Some(geometry(&p, len)), false).unwrap_err();
            assert!(
                error.contains("layer=3") && error.contains("executed trunk KV len == cache.pos")
            );
        }
        for len in [1, 272] {
            assert!(
                classify(&p, 4, 272, Some(geometry(&p, len)), false)
                    .unwrap_err()
                    .contains("MTP KV len == 0")
            );
        }
        assert!(classify(&p, 5, 272, None, false).is_err());
    }
    #[test]
    fn format_geometry_presence_and_topology_remain_fail_closed() {
        let mut p = plan();
        let g = geometry(&p, 272);
        for bad in [
            KvGeometry { ring: true, ..g },
            KvGeometry { base: true, ..g },
            KvGeometry {
                key_width: g.key_width + 32,
                ..g
            },
            KvGeometry { key_row: 1, ..g },
            KvGeometry { value_row: 1, ..g },
            KvGeometry {
                key_allocation: 1,
                ..g
            },
            KvGeometry {
                value_allocation: 1,
                ..g
            },
            KvGeometry {
                len: usize::MAX,
                ..g
            },
        ] {
            assert!(classify(&p, 3, bad.len, Some(bad), false).is_err());
        }
        assert!(classify(&p, 3, 272, None, false).is_err());
        assert!(classify(&p, 3, 272, Some(g), true).is_err());
        assert!(classify(&p, 4, 272, None, true).is_err());
        assert!(validate_plan(&p, 4).is_err());
        p.mtp_blocks[0].layer.index = 3;
        assert!(validate_plan(&p, 5).is_err());
    }
}
