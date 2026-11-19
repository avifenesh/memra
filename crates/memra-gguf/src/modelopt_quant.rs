//! Per-layer NVIDIA ModelOpt quantization map: `hf_quant_config.json` (`quantization` object) or
//! the `quantization_config` block a ModelOpt export writes into `config.json`.
//!
//! A `MIXED_PRECISION` export names a `quant_algo` per module in `quantized_layers` and lists the
//! modules it left unquantized as wildcards (`exclude_modules` in `hf_quant_config.json`,
//! `ignore` in `config.json`). This reader keeps that per-module map exactly. It knows two
//! weight-only algorithms, `W4A16_NVFP4` and `FP8_PB_WO`; any other `quant_algo`, a missing
//! group size, an activation or KV-cache quantization, or a module listed twice is refused by
//! name, never mapped to a nearby format.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

/// A weight-only ModelOpt algorithm this reader accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ModeloptAlgo {
    /// `W4A16_NVFP4`: e2m1 codes, one e4m3 scale per `group_size` inputs, one F32 second-level
    /// scale per tensor (per expert when stacked). Activations stay in the engine's float type.
    W4a16Nvfp4,
    /// `FP8_PB_WO`: e4m3 weights with one scale per `group_size` x `group_size` block.
    Fp8PbWo,
}

impl ModeloptAlgo {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "W4A16_NVFP4" => Some(Self::W4a16Nvfp4),
            "FP8_PB_WO" => Some(Self::Fp8PbWo),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::W4a16Nvfp4 => "W4A16_NVFP4",
            Self::Fp8PbWo => "FP8_PB_WO",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerQuant {
    pub algo: ModeloptAlgo,
    pub group_size: u32,
}

/// The per-module map of one export.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerQuantMap {
    /// Module name (no `.weight`) to its algorithm.
    pub layers: BTreeMap<String, LayerQuant>,
    /// Wildcard patterns of modules left unquantized (`*` matches any run of characters).
    pub excluded: Vec<String>,
}

impl LayerQuantMap {
    /// Parse `hf_quant_config.json`.
    pub fn from_hf_quant_config(json: &str) -> Result<Self, String> {
        let root: Value = serde_json::from_str(json)
            .map_err(|e| format!("hf_quant_config.json is not JSON: {e}"))?;
        let quant = root
            .get("quantization")
            .and_then(Value::as_object)
            .ok_or("hf_quant_config.json has no `quantization` object")?;
        Self::from_object(
            quant,
            "hf_quant_config.json quantization",
            "exclude_modules",
        )
    }

    /// Parse the `quantization_config` object of a ModelOpt `config.json`.
    pub fn from_config_quantization(object: &Map<String, Value>) -> Result<Self, String> {
        match object.get("quant_method").and_then(Value::as_str) {
            Some("modelopt") => {}
            other => {
                return Err(format!(
                    "config.json quantization_config.quant_method {other:?}: the per-layer map \
                     is a ModelOpt export (\"modelopt\")"
                ));
            }
        }
        Self::from_object(object, "config.json quantization_config", "ignore")
    }

    fn from_object(
        object: &Map<String, Value>,
        origin: &str,
        exclude_key: &str,
    ) -> Result<Self, String> {
        match object.get("quant_algo").and_then(Value::as_str) {
            Some("MIXED_PRECISION") => {}
            other => {
                return Err(format!(
                    "{origin}.quant_algo {other:?}: a per-layer map needs \"MIXED_PRECISION\""
                ));
            }
        }
        if let Some(kv) = object.get("kv_cache_quant_algo")
            && !kv.is_null()
        {
            return Err(format!(
                "{origin}.kv_cache_quant_algo {kv}: KV-cache quantization is not part of a \
                 weight-only map"
            ));
        }
        let layers_value = object
            .get("quantized_layers")
            .and_then(Value::as_object)
            .ok_or_else(|| format!("{origin} has no `quantized_layers` object"))?;
        let mut layers = BTreeMap::new();
        for (module, entry) in layers_value {
            let entry = entry
                .as_object()
                .ok_or_else(|| format!("{origin}.quantized_layers.{module} is not an object"))?;
            let algo_name = entry
                .get("quant_algo")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("{origin}.quantized_layers.{module} has no quant_algo"))?;
            let algo = ModeloptAlgo::parse(algo_name).ok_or_else(|| {
                format!(
                    "{origin}.quantized_layers.{module}.quant_algo {algo_name:?} is not a \
                     weight-only algorithm this reader knows (W4A16_NVFP4, FP8_PB_WO)"
                )
            })?;
            let group_size = entry
                .get("group_size")
                .and_then(Value::as_u64)
                .and_then(|g| u32::try_from(g).ok())
                .filter(|g| *g > 0)
                .ok_or_else(|| {
                    format!("{origin}.quantized_layers.{module} has no positive group_size")
                })?;
            if let Some(extra) = entry
                .keys()
                .find(|key| !matches!(key.as_str(), "quant_algo" | "group_size"))
            {
                return Err(format!(
                    "{origin}.quantized_layers.{module} carries `{extra}`, which this reader does \
                     not interpret"
                ));
            }
            if module.ends_with(".weight") {
                return Err(format!(
                    "{origin}.quantized_layers key {module} names a tensor, not a module"
                ));
            }
            layers.insert(module.clone(), LayerQuant { algo, group_size });
        }
        let excluded = match object.get(exclude_key) {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(items)) => items
                .iter()
                .map(|item| {
                    item.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| format!("{origin}.{exclude_key} holds a non-string"))
                })
                .collect::<Result<_, _>>()?,
            Some(_) => return Err(format!("{origin}.{exclude_key} is not an array")),
        };
        let map = Self { layers, excluded };
        if let Some(module) = map.layers.keys().find(|module| map.is_excluded(module)) {
            return Err(format!(
                "{origin}: {module} is both quantized and matched by {exclude_key}"
            ));
        }
        Ok(map)
    }

    /// The algorithm a module was quantized with, if any.
    pub fn get(&self, module: &str) -> Option<LayerQuant> {
        self.layers.get(module).copied()
    }

    /// True when an exclusion wildcard matches the module.
    pub fn is_excluded(&self, module: &str) -> bool {
        self.excluded
            .iter()
            .any(|pattern| wildcard_match(pattern, module))
    }
}

/// `fnmatch`-style match with `*` only, as ModelOpt's exclusion lists use it.
fn wildcard_match(pattern: &str, text: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == text;
    }
    let (first, last) = (parts[0], parts[parts.len() - 1]);
    if !text.starts_with(first) || text.len() < first.len() + last.len() || !text.ends_with(last) {
        return false;
    }
    let mut rest = &text[first.len()..text.len() - last.len()];
    for part in &parts[1..parts.len() - 1] {
        match rest.find(part) {
            Some(at) => rest = &rest[at + part.len()..],
            None => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    const HF_QUANT: &str = r#"{"quantization": {"quant_algo": "MIXED_PRECISION",
        "kv_cache_quant_algo": null,
        "quantized_layers": {
            "model.layers.1.mlp.experts.gate_proj": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
            "model.layers.1.self_attn.o_proj": {"quant_algo": "FP8_PB_WO", "group_size": 128}},
        "exclude_modules": ["model.layers.0.*", "*.hc_pre", "lm_head"]}}"#;

    #[test]
    fn reads_the_per_layer_map_and_its_exclusions() {
        let map = LayerQuantMap::from_hf_quant_config(HF_QUANT).unwrap();
        assert_eq!(
            map.get("model.layers.1.mlp.experts.gate_proj"),
            Some(LayerQuant {
                algo: ModeloptAlgo::W4a16Nvfp4,
                group_size: 16
            })
        );
        assert_eq!(
            map.get("model.layers.1.self_attn.o_proj").unwrap().algo,
            ModeloptAlgo::Fp8PbWo
        );
        assert_eq!(map.get("model.layers.2.self_attn.o_proj"), None);
        assert!(map.is_excluded("model.layers.0.self_attn.o_proj"));
        assert!(map.is_excluded("model.layers.7.hc_attn_layer.hc_pre"));
        assert!(map.is_excluded("lm_head"));
        assert!(!map.is_excluded("lm_head_2"));
        assert!(!map.is_excluded("model.layers.10.self_attn.o_proj"));
    }

    #[test]
    fn refuses_what_it_does_not_interpret() {
        for (from, to, why) in [
            ("\"FP8_PB_WO\"", "\"NVFP4\"", "weight-only algorithm"),
            ("\"MIXED_PRECISION\"", "\"FP8\"", "MIXED_PRECISION"),
            (
                "\"group_size\": 128",
                "\"group_size\": 0",
                "positive group_size",
            ),
            (
                "\"kv_cache_quant_algo\": null",
                "\"kv_cache_quant_algo\": \"FP8\"",
                "KV-cache",
            ),
            (
                "\"lm_head\"",
                "\"model.layers.1.self_attn.*\"",
                "both quantized",
            ),
        ] {
            let json = HF_QUANT.replacen(from, to, 1);
            let error = LayerQuantMap::from_hf_quant_config(&json).unwrap_err();
            assert!(error.contains(why), "{from} -> {to}: {error}");
        }
    }

    #[test]
    fn wildcards_match_like_fnmatch_star() {
        assert!(wildcard_match("a*", "abc"));
        assert!(wildcard_match("*c", "abc"));
        assert!(wildcard_match("a*c", "abbc"));
        assert!(wildcard_match(
            "model.layers.*.mlp.gate",
            "model.layers.12.mlp.gate"
        ));
        assert!(!wildcard_match(
            "model.layers.*.mlp.gate",
            "model.layers.12.mlp.gate_proj"
        ));
        assert!(wildcard_match(
            "*.self_attn.compressed_indexer*",
            "m.self_attn.compressed_indexer.key_norm"
        ));
        assert!(!wildcard_match("a*b*c", "acb"));
    }
}
