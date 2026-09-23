//! Inventory-only Step perception encoder schema for the canonical binding migration.
//! The existing native Step vision route remains separate until bound-plan integration.
//! Shape defaults are pinned to configuration_step3p7.py/vision_encoder.py at revision
//! b3d7916fccac844cca050d7520f2aaa513f9a84f; no vendor implementation is used at runtime.
use crate::config::{JsonObj, ModelConfig};
use crate::surface_catalog::{ArtifactComponent, InventorySurface};
use crate::tensor_contract::{
    QuantConstraint, TensorId, TensorMatch, TensorOwner, TensorRequirement, TensorTransform,
};

fn integer(o: &JsonObj, key: &str, default: u32) -> Result<u64, String> {
    match o.raw(key) {
        None => Ok(u64::from(default)),
        Some(v) => v
            .trim()
            .parse::<u32>()
            .map(u64::from)
            .map_err(|_| format!("Step vision inventory {key} must be an unsigned integer")),
    }
}
fn boolean(o: &JsonObj, key: &str, default: bool) -> Result<bool, String> {
    match o.raw(key).map(str::trim) {
        None => Ok(default),
        Some("true") => Ok(true),
        Some("false") => Ok(false),
        _ => Err(format!("Step vision inventory {key} must be boolean")),
    }
}

pub(crate) fn compile(
    config: &ModelConfig,
    dialect: crate::tensor_contract::CheckpointDialect,
    raw: &str,
) -> Result<Vec<InventorySurface>, String> {
    let root = JsonObj::parse(raw);
    // Flat text artifacts may also use the Step3p7 alias. Do not invent a vision component
    // from that spelling alone; unclaimed vision tensors still fail the complete census.
    if root.raw("vision_config").is_none() {
        return Ok(Vec::new());
    }
    if dialect != crate::tensor_contract::CheckpointDialect::HfSafetensors {
        return Err(
            "Step perception inventory requires an explicitly declared safetensors layout".into(),
        );
    }
    let vision = match root.raw("vision_config").map(str::trim) {
        None | Some("null") => JsonObj::parse("{}"),
        Some(_) => root
            .object("vision_config")
            .ok_or("Step vision_config must be an object")?,
    };
    if vision.raw("model_type").is_some()
        && vision.string("model_type").as_deref() != Some("perception_encoder")
    {
        return Err("Step vision inventory does not recognize the declared encoder".into());
    }
    let width = integer(&vision, "width", 1536)?;
    let layers = integer(&vision, "layers", 47)?;
    // Match the reader's tensor-count ceiling before expanding artifact-controlled layer counts.
    if layers
        .checked_mul(14)
        .and_then(|n| n.checked_add(16))
        .is_none_or(|n| n > 1_000_000)
    {
        return Err("Step vision inventory exceeds the supported tensor-count bound".into());
    }
    let heads = integer(&vision, "heads", 16)?;
    let patch = integer(&vision, "patch_size", 14)?;
    let image = integer(&vision, "image_size", 728)?;
    let channels = integer(&vision, "num_channels", 3)?;
    if width == 0
        || layers == 0
        || heads == 0
        || patch == 0
        || channels == 0
        || image < patch
        || !width.is_multiple_of(heads)
    {
        return Err("invalid Step vision inventory geometry".into());
    }
    let ratio = match vision.raw("mlp_ratio") {
        None => 8960.0f64 / 1536.0,
        Some(v) => v
            .parse::<f64>()
            .map_err(|_| "Step vision mlp_ratio must be numeric")?,
    };
    let intermediate = width as f64 * ratio;
    if !intermediate.is_finite() || intermediate < 1.0 || intermediate > u32::MAX as f64 {
        return Err("Step vision intermediate width is invalid".into());
    }
    let ff = intermediate as u64;
    let cls = match vision.raw("use_cls_token").map(str::trim) {
        None | Some("null") => boolean(&vision, "ues_cls_token", false)?,
        _ => boolean(&vision, "use_cls_token", false)?,
    };
    let pre = boolean(&vision, "use_ln_pre", true)?;
    let post = boolean(&vision, "use_ln_post", false)?;
    let absolute = boolean(&vision, "use_abs_posemb", true)?;
    let projector_bias = boolean(&root, "projector_bias", false)?;
    let mut requirements = Vec::new();
    let mut add = |layer: Option<u32>, name: String, shape: Vec<u64>| {
        requirements.push(TensorRequirement {
            id: TensorId::Family {
                family: "step_perception_inventory",
                key: name.clone(),
            },
            names: vec![name],
            match_mode: TensorMatch::OneOf,
            shape,
            owner: TensorOwner::Vision(layer),
            transform: TensorTransform::Identity,
            quant: QuantConstraint::UnquantizedFloat,
            auxiliaries: Some(Vec::new()),
            required: true,
        });
    };
    add(
        None,
        "vision_model.conv1.weight".into(),
        vec![width, channels, patch, patch],
    );
    for (enabled, prefix) in [(pre, "ln_pre"), (post, "ln_post")] {
        if enabled {
            for suffix in ["weight", "bias"] {
                add(None, format!("vision_model.{prefix}.{suffix}"), vec![width]);
            }
        }
    }
    if cls {
        add(None, "vision_model.class_embedding".into(), vec![width]);
    }
    if absolute {
        let grid = image / patch;
        add(
            None,
            "vision_model.positional_embedding".into(),
            vec![grid * grid + u64::from(cls), width],
        );
    }
    for layer in 0..layers as u32 {
        let prefix = format!("vision_model.transformer.resblocks.{layer}");
        for (name, shape) in [
            ("attn.in_proj_weight", vec![3 * width, width]),
            ("attn.in_proj_bias", vec![3 * width]),
            ("attn.out_proj.weight", vec![width, width]),
            ("attn.out_proj.bias", vec![width]),
            ("ln_1.weight", vec![width]),
            ("ln_1.bias", vec![width]),
            ("ln_2.weight", vec![width]),
            ("ln_2.bias", vec![width]),
            ("ls_1.gamma", vec![width]),
            ("ls_2.gamma", vec![width]),
            ("mlp.c_fc.weight", vec![ff, width]),
            ("mlp.c_fc.bias", vec![ff]),
            ("mlp.c_proj.weight", vec![width, ff]),
            ("mlp.c_proj.bias", vec![width]),
        ] {
            add(Some(layer), format!("{prefix}.{name}"), shape);
        }
    }
    for (which, input, output) in [(1, width, 2 * width), (2, 2 * width, 4 * width)] {
        add(
            None,
            format!("vision_model.vit_downsampler{which}.weight"),
            vec![output, input, 3, 3],
        );
        add(
            None,
            format!("vision_model.vit_downsampler{which}.bias"),
            vec![output],
        );
    }
    add(
        None,
        "vit_large_projector.weight".into(),
        vec![u64::from(config.n_embd), 4 * width],
    );
    if projector_bias {
        add(
            None,
            "vit_large_projector.bias".into(),
            vec![u64::from(config.n_embd)],
        );
    }
    Ok(vec![InventorySurface {
        component: ArtifactComponent::Vision,
        execution_unavailable: "Step perception encoder is not yet represented in this canonical executable plan; its existing native vision route requires explicit integration",
        requirements,
    }])
}
