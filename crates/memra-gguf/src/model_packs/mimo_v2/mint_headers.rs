//! Strict metadata preflight for the published MiMo NVFP4 expert bank.
//! This checks header facts that the generic tensor census currently folds
//! into auxiliary names without retaining their dtype and shape.

use std::collections::{BTreeMap, BTreeSet};

use crate::config::Arch;
use crate::model_plan::{MlpPlan, ModelPlan};
use crate::safetensors::StInfo;
use crate::tensor_contract::TensorRequirement;

use super::mint_expert_requirements;

fn expect_header(
    headers: &BTreeMap<String, StInfo>,
    name: &str,
    dtype: &str,
    shape: &[u64],
) -> Result<(), String> {
    let info = headers
        .get(name)
        .ok_or_else(|| format!("MiMo mint is missing {name}"))?;
    if info.dtype != dtype || info.shape != shape {
        return Err(format!(
            "MiMo mint {name}: expected {dtype} {shape:?}, got {} {:?}",
            info.dtype, info.shape
        ));
    }
    Ok(())
}

fn verify_expert_row(
    requirement: &TensorRequirement,
    headers: &BTreeMap<String, StInfo>,
) -> Result<(), String> {
    let [output, input] = requirement.shape.as_slice() else {
        return Err("MiMo mint expert is not a matrix".to_owned());
    };
    if *input % 16 != 0 {
        return Err(format!(
            "MiMo mint expert input width {input} is not a 16-element block"
        ));
    }
    let name = &requirement.names[0];
    let stem = name
        .strip_suffix(".weight")
        .ok_or_else(|| format!("MiMo mint expert name is not a weight: {name}"))?;
    expect_header(headers, name, "U8", &[*output, *input / 2])?;
    expect_header(
        headers,
        &format!("{stem}.weight_scale"),
        "F8_E4M3",
        &[*output, *input / 16],
    )?;
    expect_header(headers, &format!("{stem}.weight_scale_2"), "F32", &[1])?;
    expect_header(headers, &format!("{stem}.input_scale"), "F32", &[1])?;
    Ok(())
}

fn is_expert_name(name: &str) -> bool {
    name.starts_with("model.layers.") && name.contains(".mlp.experts.")
}

/// Validate every expert weight and auxiliary in the pinned NVFP4 mint.
///
/// The caller must still run the full tensor contract against a source census.
/// Header admission does not establish executable W4A4 kernels or quality.
pub(crate) fn verify_mint_expert_headers(
    plan: &ModelPlan,
    headers: &BTreeMap<String, StInfo>,
) -> Result<(), String> {
    if plan.arch != Arch::MiMoV2 || plan.hidden_size != 4_096 || plan.layers.len() != 48 {
        return Err("MiMo mint header preflight requires the pinned 48-layer text plan".to_owned());
    }
    if !matches!(&plan.layers[0].mlp, MlpPlan::Dense(_)) {
        return Err("MiMo mint header preflight requires a dense first layer".to_owned());
    }
    let mut expected = BTreeSet::new();
    let mut weights = 0;
    for layer in plan.layers.iter().skip(1) {
        let MlpPlan::Moe(moe) = &layer.mlp else {
            return Err(format!("MiMo mint layer {} is not MoE", layer.index));
        };
        if moe.expert_count != 256
            || moe.experts_per_token != 8
            || moe.expert_intermediate_size != 2_048
        {
            return Err(format!(
                "MiMo mint layer {} has changed expert geometry",
                layer.index
            ));
        }
        for requirement in mint_expert_requirements(plan, layer.index, moe) {
            verify_expert_row(&requirement, headers)?;
            let name = &requirement.names[0];
            let stem = name.strip_suffix(".weight").unwrap();
            expected.insert(name.clone());
            expected.insert(format!("{stem}.weight_scale"));
            expected.insert(format!("{stem}.weight_scale_2"));
            expected.insert(format!("{stem}.input_scale"));
            weights += 1;
        }
    }
    if weights != 36_096 {
        return Err(format!(
            "MiMo mint has {weights} expert requirements, expected 36096"
        ));
    }
    let actual: BTreeSet<_> = headers
        .keys()
        .filter(|name| is_expert_name(name))
        .cloned()
        .collect();
    if expected != actual {
        let missing = expected.difference(&actual).next();
        let extra = actual.difference(&expected).next();
        return Err(format!(
            "MiMo mint expert namespace differs: missing {missing:?}, extra {extra:?}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{HfConfig, ModelConfig};
    use sha2::{Digest, Sha256};

    fn row_and_headers() -> (TensorRequirement, BTreeMap<String, StInfo>) {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!("fixtures/config.json")));
        let plan = ModelPlan::compile(&config).unwrap();
        let MlpPlan::Moe(moe) = &plan.layers[1].mlp else {
            unreachable!()
        };
        let row = mint_expert_requirements(&plan, 1, moe)
            .into_iter()
            .next()
            .unwrap();
        let stem = row.names[0].strip_suffix(".weight").unwrap();
        let info = |dtype: &str, shape| StInfo {
            dtype: dtype.to_owned(),
            shape,
            data_offsets: [0, 0],
        };
        let headers = BTreeMap::from([
            (row.names[0].clone(), info("U8", vec![2_048, 2_048])),
            (
                format!("{stem}.weight_scale"),
                info("F8_E4M3", vec![2_048, 256]),
            ),
            (format!("{stem}.weight_scale_2"), info("F32", vec![1])),
            (format!("{stem}.input_scale"), info("F32", vec![1])),
        ]);
        (row, headers)
    }

    #[test]
    fn pinned_expert_header_fields_are_all_required() {
        let (row, mut headers) = row_and_headers();
        verify_expert_row(&row, &headers).unwrap();
        let scale = row.auxiliaries.as_ref().unwrap()[0].clone();
        headers.get_mut(&scale).unwrap().shape[1] = 128;
        assert!(verify_expert_row(&row, &headers).is_err());
        headers.get_mut(&scale).unwrap().shape[1] = 256;
        headers.get_mut(&scale).unwrap().dtype = "U8".to_owned();
        assert!(verify_expert_row(&row, &headers).is_err());
        headers.get_mut(&scale).unwrap().dtype = "F8_E4M3".to_owned();
        let input = row.auxiliaries.as_ref().unwrap()[2].clone();
        headers.remove(&input);
        assert!(verify_expert_row(&row, &headers).is_err());
    }

    #[test]
    fn expert_namespace_excludes_modal_and_draft_tensors() {
        assert!(is_expert_name(
            "model.layers.47.mlp.experts.255.down_proj.weight"
        ));
        assert!(!is_expert_name("model.mtp.layers.0.mlp.down_proj.weight"));
        assert!(!is_expert_name("visual.blocks.0.mlp.down_proj.weight"));
    }

    #[test]
    fn full_mint_expert_headers_match_the_pinned_artifact_digest() {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!("fixtures/config.json")));
        let plan = ModelPlan::compile(&config).unwrap();
        let mut headers = BTreeMap::new();
        for layer in plan.layers.iter().skip(1) {
            let MlpPlan::Moe(moe) = &layer.mlp else {
                unreachable!()
            };
            for requirement in mint_expert_requirements(&plan, layer.index, moe) {
                let [output, input] = requirement.shape.as_slice() else {
                    unreachable!()
                };
                let name = requirement.names[0].clone();
                let stem = name.strip_suffix(".weight").unwrap();
                for (name, dtype, shape) in [
                    (name.clone(), "U8", vec![*output, *input / 2]),
                    (
                        format!("{stem}.weight_scale"),
                        "F8_E4M3",
                        vec![*output, *input / 16],
                    ),
                    (format!("{stem}.weight_scale_2"), "F32", vec![1]),
                    (format!("{stem}.input_scale"), "F32", vec![1]),
                ] {
                    headers.insert(
                        name,
                        StInfo {
                            dtype: dtype.to_owned(),
                            shape,
                            data_offsets: [0, 0],
                        },
                    );
                }
            }
        }
        assert_eq!(headers.len(), 144_384);
        let mut manifest = String::new();
        for (name, info) in &headers {
            manifest.push_str(name);
            manifest.push('\0');
            manifest.push_str(&info.dtype);
            manifest.push('\0');
            manifest.push_str(
                &info
                    .shape
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            );
            manifest.push('\n');
        }
        assert_eq!(
            format!("{:x}", Sha256::digest(manifest.as_bytes())),
            "dbbe48853e459202115d529165acd38ed8a128a85271d2f410fc033a867532f8"
        );
        verify_mint_expert_headers(&plan, &headers).unwrap();
        headers.remove("model.layers.47.mlp.experts.255.up_proj.input_scale");
        assert!(verify_mint_expert_headers(&plan, &headers).is_err());
    }
}
