//! Reader for the `parameter_groups` form of `hf_quant_config.json`.
//!
//! The older ModelOpt form (`{"quantization": {"quant_algo": ..}}`) names an algorithm for the
//! whole file. This form partitions every tensor of every weight file into groups, and each group
//! declares how its weights are stored, the dtype its linear computes in and, where the
//! checkpoint program quantizes the linear's input, the activation quantizer it requires:
//!
//! ```json
//! {"quant_method": "inkling2",
//!  "model": {"parameter_groups": [
//!    {"compute_dtype": "nvfp4", "output_dtype": "uint8", "weight_transform": "...",
//!     "weight_block_shape": [1, 16],
//!     "required_activation_quantization": {"format": "nvfp4_e2m1", "block_size": 16,
//!        "amax_type": "per_token", "scale_dtype": "float8_e4m3fn", "rounding": "nearest",
//!        "scale_selection_rule": "fouroversix_mae"},
//!     "tensors": ["..."]}]},
//!  "mtp": {"parameter_groups": [...]}}
//! ```
//!
//! `required_activation_quantization` is part of the checkpoint's numeric program, not a hint:
//! a group that declares one computes a different function when its input is left in floating
//! point. It is typed here ([`ActivationQuant`]), an unknown format, rule or dtype is refused by
//! name, and the key must be present in every group (`null` for a float linear) so that an
//! omitted declaration can never read as "no quantizer".

use serde_json::{Map, Value};

/// The dtype a group's linear computes in (`compute_dtype`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeDtype {
    Float32,
    Bfloat16,
    /// e2m1 codes, one e4m3 scale per 16 inputs, a per-expert outer factor.
    Nvfp4,
    /// e4m3 values, one ue8m0 exponent per 32 inputs.
    Mxfp8,
}

impl ComputeDtype {
    pub fn is_quantized(self) -> bool {
        matches!(self, Self::Nvfp4 | Self::Mxfp8)
    }
}

/// Scale encoding of an activation quantizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationScale {
    /// `float8_e4m3fn`.
    E4m3,
    /// `ue8m0`: a power of two.
    Ue8m0,
}

/// How an NVFP4 activation block picks its scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleSelection {
    /// `fouroversix_mae`: of the scale that maps the block maximum to 6 and the one that maps
    /// it to 4, keep the one with the smaller mean absolute error.
    FourOverSixMae,
}

/// Over what an NVFP4 activation's outer maximum is taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationAmax {
    /// `per_token`: one maximum per input row.
    PerToken,
}

/// The activation quantizer a group requires on the input of its linear.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationQuant {
    /// `nvfp4_e2m1`: W4A4. e2m1 elements in blocks of `block_size`, an e4m3 block scale chosen
    /// by `scale_selection`, under an outer maximum per `amax`. Round to nearest.
    Nvfp4E2m1 {
        block_size: u32,
        amax: ActivationAmax,
        scale: ActivationScale,
        scale_selection: ScaleSelection,
    },
    /// `mxfp8_e4m3`: W8A8. e4m3 elements in blocks of `block_size` with a ue8m0 block scale.
    /// Round to nearest.
    Mxfp8E4m3 {
        block_size: u32,
        scale: ActivationScale,
    },
}

/// Storage of a quantized weight's outer scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightScaleFormat {
    /// `amax_denominator`: a per-expert `_amax` plane and a one-element
    /// `_outer_scale_denominator`; the factor is their quotient.
    AmaxDenominator,
}

/// One `parameter_groups` entry.
#[derive(Clone, PartialEq, Eq)]
pub struct ParameterGroup {
    pub compute_dtype: ComputeDtype,
    /// Stored dtype of the weight (`output_dtype`), verbatim.
    pub output_dtype: String,
    /// `fwd_output_dtype`: the dtype the linear's output is accumulated in when it differs.
    pub fwd_output_dtype: Option<String>,
    pub weight_transform: String,
    pub weight_block_shape: Vec<u32>,
    /// `weight_scale_format`: how a quantized weight's outer scale is stored.
    pub weight_scale_format: Option<WeightScaleFormat>,
    /// `None` is the declared `null`: the linear takes a floating-point input.
    pub required_activation_quantization: Option<ActivationQuant>,
    /// Sorted tensor names of the group.
    pub tensors: Vec<String>,
}

/// The tensor list prints as a count: a config's `Debug` text is a receipt
/// (`normalized-config.txt`), and the names are the census's, not the config's.
impl std::fmt::Debug for ParameterGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParameterGroup")
            .field("compute_dtype", &self.compute_dtype)
            .field("output_dtype", &self.output_dtype)
            .field("fwd_output_dtype", &self.fwd_output_dtype)
            .field("weight_transform", &self.weight_transform)
            .field("weight_block_shape", &self.weight_block_shape)
            .field("weight_scale_format", &self.weight_scale_format)
            .field(
                "required_activation_quantization",
                &self.required_activation_quantization,
            )
            .field("tensors", &self.tensors.len())
            .finish()
    }
}

/// A parsed `hf_quant_config.json` in the `parameter_groups` form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantGroups {
    pub quant_method: String,
    /// sha256 of the file's bytes, lowercase hex.
    pub sha256: String,
    /// Groups of the indexed weight file(s) (`model`).
    pub model: Vec<ParameterGroup>,
    /// Groups of the drafter file (`mtp`).
    pub mtp: Vec<ParameterGroup>,
}

impl QuantGroups {
    pub fn parse(text: &str) -> Result<Self, String> {
        let root = crate::strict_json::parse_object(text)
            .map_err(|error| format!("hf_quant_config.json: {error}"))?;
        for key in root.keys() {
            if !matches!(key.as_str(), "quant_method" | "model" | "mtp") {
                return Err(format!(
                    "hf_quant_config.json: unknown top-level key {key:?}"
                ));
            }
        }
        let quant_method = string(&root, "$", "quant_method")?;
        let part = |name: &str| -> Result<Vec<ParameterGroup>, String> {
            let path = format!("$.{name}");
            let object = root
                .get(name)
                .and_then(Value::as_object)
                .ok_or_else(|| format!("hf_quant_config.json: {path} must be an object"))?;
            for key in object.keys() {
                if key != "parameter_groups" {
                    return Err(format!("hf_quant_config.json: unknown key {path}.{key}"));
                }
            }
            let groups = object
                .get("parameter_groups")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    format!("hf_quant_config.json: {path}.parameter_groups must be an array")
                })?;
            let parsed = groups
                .iter()
                .enumerate()
                .map(|(index, group)| {
                    parse_group(group, &format!("{path}.parameter_groups[{index}]"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut seen = std::collections::BTreeSet::new();
            for group in &parsed {
                for tensor in &group.tensors {
                    if !seen.insert(tensor.as_str()) {
                        return Err(format!(
                            "hf_quant_config.json: {path} declares tensor {tensor} in two groups"
                        ));
                    }
                }
            }
            Ok(parsed)
        };
        Ok(Self {
            quant_method,
            sha256: sha256_hex(text.as_bytes()),
            model: part("model")?,
            mtp: part("mtp")?,
        })
    }

    /// The `model` group that declares `tensor`.
    pub fn model_group(&self, tensor: &str) -> Option<&ParameterGroup> {
        self.model.iter().find(|group| {
            group
                .tensors
                .binary_search_by(|name| name.as_str().cmp(tensor))
                .is_ok()
        })
    }

    /// The `mtp` group that declares `tensor`.
    pub fn mtp_group(&self, tensor: &str) -> Option<&ParameterGroup> {
        self.mtp.iter().find(|group| {
            group
                .tensors
                .binary_search_by(|name| name.as_str().cmp(tensor))
                .is_ok()
        })
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn string(object: &Map<String, Value>, path: &str, key: &str) -> Result<String, String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("hf_quant_config.json: {path}.{key} must be a string"))
}

fn unsigned(value: &Value, path: &str) -> Result<u32, String> {
    value
        .as_u64()
        .and_then(|number| u32::try_from(number).ok())
        .ok_or_else(|| format!("hf_quant_config.json: {path} must be an unsigned 32-bit integer"))
}

fn parse_group(value: &Value, path: &str) -> Result<ParameterGroup, String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("hf_quant_config.json: {path} must be an object"))?;
    for key in object.keys() {
        if !matches!(
            key.as_str(),
            "compute_dtype"
                | "output_dtype"
                | "fwd_output_dtype"
                | "weight_transform"
                | "weight_block_shape"
                | "weight_scale_format"
                | "required_activation_quantization"
                | "tensors"
        ) {
            return Err(format!("hf_quant_config.json: unknown key {path}.{key}"));
        }
    }
    let compute = string(object, path, "compute_dtype")?;
    let compute_dtype = match compute.as_str() {
        "float32" => ComputeDtype::Float32,
        "bfloat16" => ComputeDtype::Bfloat16,
        "nvfp4" => ComputeDtype::Nvfp4,
        "mxfp8" => ComputeDtype::Mxfp8,
        other => {
            return Err(format!(
                "hf_quant_config.json: {path}.compute_dtype {other:?} is not a known compute dtype"
            ));
        }
    };
    let weight_block_shape = object
        .get("weight_block_shape")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("hf_quant_config.json: {path}.weight_block_shape must be an array"))?
        .iter()
        .enumerate()
        .map(|(index, value)| unsigned(value, &format!("{path}.weight_block_shape[{index}]")))
        .collect::<Result<Vec<_>, _>>()?;
    // The key must be present: an explicit null is the declaration of a float input.
    let activation = object.get("required_activation_quantization").ok_or_else(|| {
        format!("hf_quant_config.json: {path}.required_activation_quantization is missing (null declares a float input)")
    })?;
    let required_activation_quantization = match activation {
        Value::Null => None,
        other => Some(parse_activation(
            other,
            &format!("{path}.required_activation_quantization"),
        )?),
    };
    if compute_dtype.is_quantized() != required_activation_quantization.is_some() {
        return Err(format!(
            "hf_quant_config.json: {path} pairs compute_dtype {compute:?} with {}",
            if required_activation_quantization.is_some() {
                "an activation quantizer"
            } else {
                "no activation quantizer"
            }
        ));
    }
    match (compute_dtype, required_activation_quantization) {
        (ComputeDtype::Nvfp4, Some(ActivationQuant::Mxfp8E4m3 { .. }))
        | (ComputeDtype::Mxfp8, Some(ActivationQuant::Nvfp4E2m1 { .. })) => {
            return Err(format!(
                "hf_quant_config.json: {path} activation format does not match compute_dtype {compute:?}"
            ));
        }
        _ => {}
    }
    let weight_scale_format = match object.get("weight_scale_format") {
        None => None,
        Some(Value::String(text)) if text == "amax_denominator" => {
            Some(WeightScaleFormat::AmaxDenominator)
        }
        Some(other) => {
            return Err(format!(
                "hf_quant_config.json: {path}.weight_scale_format {other} is not typed"
            ));
        }
    };
    // NVFP4 weights carry the amax/denominator outer scale; nothing else does.
    if (compute_dtype == ComputeDtype::Nvfp4) != weight_scale_format.is_some() {
        return Err(format!(
            "hf_quant_config.json: {path} pairs compute_dtype {compute:?} with weight_scale_format {weight_scale_format:?}"
        ));
    }
    let fwd_output_dtype = match object.get("fwd_output_dtype") {
        None => None,
        Some(Value::String(text)) => Some(text.clone()),
        Some(_) => {
            return Err(format!(
                "hf_quant_config.json: {path}.fwd_output_dtype must be a string"
            ));
        }
    };
    let mut tensors = object
        .get("tensors")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("hf_quant_config.json: {path}.tensors must be an array"))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("hf_quant_config.json: {path}.tensors holds a non-string"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    tensors.sort();
    if tensors.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(format!(
            "hf_quant_config.json: {path}.tensors names a tensor twice"
        ));
    }
    Ok(ParameterGroup {
        compute_dtype,
        output_dtype: string(object, path, "output_dtype")?,
        fwd_output_dtype,
        weight_transform: string(object, path, "weight_transform")?,
        weight_block_shape,
        weight_scale_format,
        required_activation_quantization,
        tensors,
    })
}

fn parse_activation(value: &Value, path: &str) -> Result<ActivationQuant, String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("hf_quant_config.json: {path} must be an object or null"))?;
    let format = string(object, path, "format")?;
    let allowed: &[&str] = match format.as_str() {
        "nvfp4_e2m1" => &[
            "format",
            "block_size",
            "amax_type",
            "scale_dtype",
            "rounding",
            "scale_selection_rule",
        ],
        "mxfp8_e4m3" => &["format", "block_size", "scale_dtype", "rounding"],
        other => {
            return Err(format!(
                "hf_quant_config.json: {path}.format {other:?} is not a known activation quantizer"
            ));
        }
    };
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("hf_quant_config.json: unknown key {path}.{key}"));
        }
    }
    let block_size = unsigned(
        object
            .get("block_size")
            .ok_or_else(|| format!("hf_quant_config.json: {path}.block_size is missing"))?,
        &format!("{path}.block_size"),
    )?;
    let rounding = string(object, path, "rounding")?;
    if rounding != "nearest" {
        return Err(format!(
            "hf_quant_config.json: {path}.rounding {rounding:?}; only \"nearest\" is typed"
        ));
    }
    let scale_dtype = string(object, path, "scale_dtype")?;
    let scale = match scale_dtype.as_str() {
        "float8_e4m3fn" => ActivationScale::E4m3,
        "ue8m0" => ActivationScale::Ue8m0,
        other => {
            return Err(format!(
                "hf_quant_config.json: {path}.scale_dtype {other:?} is not a known scale dtype"
            ));
        }
    };
    if format == "mxfp8_e4m3" {
        if scale != ActivationScale::Ue8m0 || block_size != 32 {
            return Err(format!(
                "hf_quant_config.json: {path} is mxfp8_e4m3 with block {block_size} and scale {scale_dtype:?}; MXFP8 is block 32 with a ue8m0 scale"
            ));
        }
        return Ok(ActivationQuant::Mxfp8E4m3 { block_size, scale });
    }
    if scale != ActivationScale::E4m3 || block_size != 16 {
        return Err(format!(
            "hf_quant_config.json: {path} is nvfp4_e2m1 with block {block_size} and scale {scale_dtype:?}; NVFP4 is block 16 with an e4m3 scale"
        ));
    }
    let amax_type = string(object, path, "amax_type")?;
    let amax = match amax_type.as_str() {
        "per_token" => ActivationAmax::PerToken,
        other => {
            return Err(format!(
                "hf_quant_config.json: {path}.amax_type {other:?} is not typed"
            ));
        }
    };
    let rule = string(object, path, "scale_selection_rule")?;
    let scale_selection = match rule.as_str() {
        "fouroversix_mae" => ScaleSelection::FourOverSixMae,
        other => {
            return Err(format!(
                "hf_quant_config.json: {path}.scale_selection_rule {other:?} is not typed"
            ));
        }
    };
    Ok(ActivationQuant::Nvfp4E2m1 {
        block_size,
        amax,
        scale,
        scale_selection,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn group(compute: &str, activation: &str, tensors: &str) -> String {
        let scale_format = if compute == "nvfp4" {
            r#""weight_scale_format":"amax_denominator","#
        } else {
            ""
        };
        format!(
            r#"{{"compute_dtype":"{compute}","output_dtype":"x","weight_transform":"cast",{scale_format}
            "weight_block_shape":[],"required_activation_quantization":{activation},
            "tensors":[{tensors}]}}"#
        )
    }

    fn file(model: &str) -> String {
        format!(
            r#"{{"quant_method":"inkling2","model":{{"parameter_groups":[{model}]}},
            "mtp":{{"parameter_groups":[]}}}}"#
        )
    }

    const NVFP4_ACT: &str = r#"{"amax_type":"per_token","block_size":16,"format":"nvfp4_e2m1",
        "rounding":"nearest","scale_dtype":"float8_e4m3fn","scale_selection_rule":"fouroversix_mae"}"#;
    const MXFP8_ACT: &str =
        r#"{"block_size":32,"format":"mxfp8_e4m3","rounding":"nearest","scale_dtype":"ue8m0"}"#;

    #[test]
    fn typed_activation_quantizers_parse_and_lookup_by_tensor() {
        let text = file(&format!(
            "{},{},{}",
            group("bfloat16", "null", r#""b","a""#),
            group("nvfp4", NVFP4_ACT, r#""w4""#),
            group("mxfp8", MXFP8_ACT, r#""w8""#)
        ));
        let parsed = QuantGroups::parse(&text).unwrap();
        assert_eq!(parsed.quant_method, "inkling2");
        assert_eq!(parsed.sha256.len(), 64);
        assert_eq!(parsed.model[0].tensors, ["a", "b"]);
        assert_eq!(
            parsed
                .model_group("a")
                .unwrap()
                .required_activation_quantization,
            None
        );
        assert_eq!(
            parsed
                .model_group("w4")
                .unwrap()
                .required_activation_quantization,
            Some(ActivationQuant::Nvfp4E2m1 {
                block_size: 16,
                amax: ActivationAmax::PerToken,
                scale: ActivationScale::E4m3,
                scale_selection: ScaleSelection::FourOverSixMae,
            })
        );
        assert_eq!(
            parsed
                .model_group("w8")
                .unwrap()
                .required_activation_quantization,
            Some(ActivationQuant::Mxfp8E4m3 {
                block_size: 32,
                scale: ActivationScale::Ue8m0,
            })
        );
        assert!(parsed.model_group("absent").is_none());
        // The Debug text carries the count, not the names.
        assert!(format!("{:?}", parsed.model[0]).contains("tensors: 2"));
    }

    #[test]
    fn an_omitted_or_mismatched_activation_declaration_is_refused() {
        let refuse = |text: String, needle: &str| {
            let error = QuantGroups::parse(&text).unwrap_err();
            assert!(error.contains(needle), "{error}");
        };
        // Omitted key: never read as "no quantizer".
        refuse(
            file(
                r#"{"compute_dtype":"nvfp4","output_dtype":"uint8","weight_transform":"t",
                "weight_block_shape":[],"tensors":["w"]}"#,
            ),
            "required_activation_quantization is missing",
        );
        // A quantized compute dtype with a declared float input, and the reverse.
        refuse(
            file(&group("nvfp4", "null", r#""w""#)),
            "no activation quantizer",
        );
        refuse(
            file(&group("bfloat16", NVFP4_ACT, r#""w""#)),
            "an activation quantizer",
        );
        refuse(
            file(&group("nvfp4", MXFP8_ACT, r#""w""#)),
            "does not match compute_dtype",
        );
        // Unknown rule, format, rounding, dtype and key.
        refuse(
            file(&group(
                "nvfp4",
                &NVFP4_ACT.replace("fouroversix_mae", "mse"),
                r#""w""#,
            )),
            "scale_selection_rule",
        );
        refuse(
            file(&group(
                "nvfp4",
                &NVFP4_ACT.replace("nvfp4_e2m1", "nvfp4_x"),
                r#""w""#,
            )),
            "not a known activation quantizer",
        );
        refuse(
            file(&group(
                "mxfp8",
                &MXFP8_ACT.replace("nearest", "floor"),
                r#""w""#,
            )),
            "rounding",
        );
        refuse(
            file(&group("mxfp8", &MXFP8_ACT.replace("32", "16"), r#""w""#)),
            "MXFP8 is block 32",
        );
        refuse(
            file(&group(
                "nvfp4",
                &NVFP4_ACT.replace("per_token", "per_tensor"),
                r#""w""#,
            )),
            "amax_type",
        );
        refuse(
            file(&group("int8", "null", r#""w""#)),
            "not a known compute dtype",
        );
        // An NVFP4 group must say how its outer scale is stored, and only it may.
        refuse(
            file(
                &group("nvfp4", NVFP4_ACT, r#""w""#)
                    .replace(r#""weight_scale_format":"amax_denominator","#, ""),
            ),
            "weight_scale_format None",
        );
        refuse(
            file(&group("mxfp8", MXFP8_ACT, r#""w""#).replace(
                r#""weight_transform""#,
                r#""weight_scale_format":"amax_denominator","weight_transform""#,
            )),
            "weight_scale_format Some",
        );
        refuse(
            file(&format!(
                "{},{}",
                group("bfloat16", "null", r#""w""#),
                group("float32", "null", r#""w""#)
            )),
            "in two groups",
        );
        refuse(
            r#"{"quant_method":"inkling2","quantization":{},"model":{"parameter_groups":[]},
                "mtp":{"parameter_groups":[]}}"#
                .to_owned(),
            "unknown top-level key",
        );
    }
}
