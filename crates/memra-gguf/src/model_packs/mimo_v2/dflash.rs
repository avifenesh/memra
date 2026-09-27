//! Header-only contract for the separately stored MiMo V2.6 DFlash draft.
//!
//! This pins source metadata. It does not load weights, execute MTP, or change
//! the MiMo model pack's unsupported serving state. The architecture facts
//! below come from `dflash/config.json` and `dflash/dflash.py` at the pinned
//! revision, with tensor shapes from the sidecar safetensors header.

use crate::safetensors::{StInfo, parse_header_json_checked};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};

pub const SOURCE: &str = "XiaomiMiMo/MiMo-V2.6-Flash-RL@3b38d063180c3e4aed9691fdc735f3d10b266ee4";
pub const CONFIG_SHA256: &str = "29f18def0d74535771b2364b28107f4914ebb88872abb93189012f7573e10e4b";
pub const HEADER_SHA256: &str = "c72f0a6b6d24349daf558f2d3eb09d6a92dcf9dd9aab8b9dbf05bc7d15d94521";
/// Source-code provenance. The executable Python file is never a runtime dependency.
pub const DFLASH_PY_SHA256: &str =
    "da5ab1738b954800950405131f1d1d97c3345f37e32676d511d3a25dfddd9d75";

const DRAFT_LAYERS: usize = 5;
const HEADER_ROWS: usize = 63;
const HEADER_BYTES: usize = 2_936_114_304;

/// Pinned draft geometry for a future native executor. This receipt alone
/// grants no MTP or serving capability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DFlashAuxiliaryContract {
    pub target_tap_ids: [u32; DRAFT_LAYERS],
    pub draft_layers: usize,
    pub target_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub query_head_dim: usize,
    pub value_head_dim: usize,
    pub rotary_dims: usize,
    pub sliding_window: usize,
    pub block_size: usize,
    pub mask_token_id: u32,
    pub num_anchors: usize,
    pub attention_value_scale: f32,
    pub attention_sink_bias: bool,
    pub is_causal: bool,
    pub tensor_rows: usize,
    pub tensor_bytes: usize,
}

fn check_hash(bytes: &[u8], expected: &str, which: &str) -> Result<(), String> {
    let found = format!("{:x}", Sha256::digest(bytes));
    if found != expected {
        return Err(format!(
            "MiMo DFlash {which} changed: got {found}, expected {expected}"
        ));
    }
    Ok(())
}

/// The pinned source has exactly one invalid JSON token: a comma immediately
/// before the final object brace. This normalization is called only after the
/// raw byte digest matches. It cannot admit a changed source document.
fn normalize_pinned_config(raw: &[u8]) -> Result<Vec<u8>, String> {
    if serde_json::from_slice::<Value>(raw).is_ok() {
        return Err("MiMo DFlash config unexpectedly became strict JSON".into());
    }
    let end = raw
        .iter()
        .rposition(|byte| !byte.is_ascii_whitespace())
        .ok_or("MiMo DFlash config is empty")?;
    if raw[end] != b'}' {
        return Err("MiMo DFlash config has no terminal object brace".into());
    }
    let comma = raw[..end]
        .iter()
        .rposition(|byte| !byte.is_ascii_whitespace())
        .ok_or("MiMo DFlash config has no final field")?;
    if raw[comma] != b',' {
        return Err("MiMo DFlash config has no single terminal comma".into());
    }
    let mut normalized = raw.to_vec();
    normalized.remove(comma);
    Ok(normalized)
}

fn check_config_fields(config: &Value) -> Result<(), String> {
    for (path, expected) in [
        ("/architectures", json!(["DFlashDraftModel"])),
        ("/model_type", json!("qwen3")),
        ("/auto_map/AutoModel", json!("dflash.DFlashDraftModel")),
        ("/hidden_size", json!(4096)),
        ("/intermediate_size", json!(16384)),
        ("/num_hidden_layers", json!(5)),
        ("/num_attention_heads", json!(64)),
        ("/num_key_value_heads", json!(8)),
        ("/head_dim", json!(128)),
        ("/v_head_dim", json!(128)),
        ("/partial_rotary_factor", json!(0.5)),
        ("/block_size", json!(8)),
        (
            "/dflash_config/target_layer_ids",
            json!([0, 11, 23, 35, 47]),
        ),
        ("/dflash_config/mask_token_id", json!(151675)),
        ("/dflash_config/num_anchors", json!(4096)),
        ("/dflash_config/block_size", json!(8)),
        ("/dflash_config/loss_decay_gamma", json!(7.0)),
        ("/dflash_config/attention_value_scale", json!(0.612)),
        ("/dflash_config/attention_sink_bias", json!(true)),
        (
            "/layer_types",
            json!([
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention"
            ]),
        ),
        ("/sliding_window", json!(1024)),
        ("/use_sliding_window", json!(true)),
        ("/is_causal", json!(false)),
        ("/num_target_layers", json!(48)),
        ("/target_hidden_size", json!(4096)),
        ("/vocab_size", json!(152576)),
        ("/max_position_embeddings", json!(1048576)),
        ("/rope_theta", json!(10000.0)),
        (
            "/rms_norm_eps",
            serde_json::from_str("1e-06").expect("pinned epsilon is JSON"),
        ),
        ("/torch_dtype", json!("bfloat16")),
        ("/hidden_act", json!("silu")),
        ("/attention_bias", json!(false)),
        ("/attention_dropout", json!(0.0)),
        ("/add_swa_attention_sink_bias", json!(true)),
        ("/tie_word_embeddings", json!(false)),
        ("/use_cache", json!(true)),
    ] {
        if config.pointer(path) != Some(&expected) {
            return Err(format!("MiMo DFlash config field {path} differs from pin"));
        }
    }
    Ok(())
}

fn expected_headers() -> BTreeMap<String, Vec<u64>> {
    let mut rows = BTreeMap::from([
        ("fc.weight".into(), vec![4_096, 20_480]),
        ("hidden_norm.weight".into(), vec![4_096]),
        ("norm.weight".into(), vec![4_096]),
    ]);
    for layer in 0..DRAFT_LAYERS {
        for (suffix, shape) in [
            ("input_layernorm.weight", vec![4_096]),
            ("mlp.down_proj.weight", vec![4_096, 16_384]),
            ("mlp.gate_proj.weight", vec![16_384, 4_096]),
            ("mlp.up_proj.weight", vec![16_384, 4_096]),
            ("post_attention_layernorm.weight", vec![4_096]),
            ("self_attn.attention_sink_bias", vec![64]),
            ("self_attn.k_norm.weight", vec![128]),
            ("self_attn.k_proj.weight", vec![1_024, 4_096]),
            ("self_attn.o_proj.weight", vec![4_096, 8_192]),
            ("self_attn.q_norm.weight", vec![128]),
            ("self_attn.q_proj.weight", vec![8_192, 4_096]),
            ("self_attn.v_proj.weight", vec![1_024, 4_096]),
        ] {
            rows.insert(format!("layers.{layer}.{suffix}"), shape);
        }
    }
    rows
}

fn check_header_rows(headers: &HashMap<String, StInfo>) -> Result<(), String> {
    let expected = expected_headers();
    if headers.len() != HEADER_ROWS || expected.len() != HEADER_ROWS {
        return Err(format!(
            "MiMo DFlash header has {} tensors, expected {HEADER_ROWS}",
            headers.len()
        ));
    }
    let mut offsets = Vec::with_capacity(HEADER_ROWS);
    for (name, shape) in expected {
        let info = headers
            .get(&name)
            .ok_or_else(|| format!("MiMo DFlash header is missing {name}"))?;
        if info.dtype != "BF16" || info.shape != shape {
            return Err(format!(
                "MiMo DFlash {name}: expected BF16 {shape:?}, got {} {:?}",
                info.dtype, info.shape
            ));
        }
        let elements = shape
            .iter()
            .try_fold(1usize, |count, &dim| count.checked_mul(dim as usize));
        let bytes = elements
            .and_then(|count| count.checked_mul(2))
            .ok_or_else(|| format!("MiMo DFlash {name} byte extent overflows"))?;
        let [start, end] = info.data_offsets;
        if end.checked_sub(start) != Some(bytes) {
            return Err(format!("MiMo DFlash {name} BF16 byte extent differs"));
        }
        offsets.push((start, end, name));
    }
    let actual_names = headers.keys().collect::<std::collections::BTreeSet<_>>();
    let expected_names = offsets
        .iter()
        .map(|(_, _, name)| name)
        .collect::<std::collections::BTreeSet<_>>();
    if actual_names != expected_names {
        return Err("MiMo DFlash header contains an unexpected tensor".into());
    }
    offsets.sort_by_key(|(start, _, _)| *start);
    let mut cursor = 0;
    for (start, end, name) in offsets {
        if start != cursor {
            return Err(format!(
                "MiMo DFlash {name} has overlapping or missing bytes"
            ));
        }
        cursor = end;
    }
    if cursor != HEADER_BYTES {
        return Err(format!(
            "MiMo DFlash header spans {cursor} bytes, expected {HEADER_BYTES}"
        ));
    }
    Ok(())
}

/// Verify the exact source identity and raw metadata before returning a draft
/// geometry receipt. A later loader must separately verify full weights and
/// numerical execution.
pub fn verify_pinned_auxiliary(
    source: &str,
    config_bytes: &[u8],
    header_bytes: &[u8],
) -> Result<DFlashAuxiliaryContract, String> {
    if source != SOURCE {
        return Err(format!("MiMo DFlash source {source:?} is not {SOURCE}"));
    }
    check_hash(config_bytes, CONFIG_SHA256, "config")?;
    let normalized = normalize_pinned_config(config_bytes)?;
    let config: Value = serde_json::from_slice(&normalized)
        .map_err(|error| format!("MiMo DFlash normalized config is invalid: {error}"))?;
    check_config_fields(&config)?;
    check_hash(header_bytes, HEADER_SHA256, "header")?;
    let json = std::str::from_utf8(header_bytes)
        .map_err(|error| format!("MiMo DFlash header is not UTF-8: {error}"))?;
    let headers = parse_header_json_checked(json)?;
    check_header_rows(&headers)?;
    Ok(DFlashAuxiliaryContract {
        target_tap_ids: [0, 11, 23, 35, 47],
        draft_layers: DRAFT_LAYERS,
        target_layers: 48,
        hidden_size: 4_096,
        intermediate_size: 16_384,
        query_heads: 64,
        kv_heads: 8,
        query_head_dim: 128,
        value_head_dim: 128,
        rotary_dims: 64,
        sliding_window: 1_024,
        block_size: 8,
        mask_token_id: 151_675,
        num_anchors: 4_096,
        attention_value_scale: 0.612,
        attention_sink_bias: true,
        is_causal: false,
        tensor_rows: HEADER_ROWS,
        tensor_bytes: HEADER_BYTES,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // The source config has two spaces on its final blank line; the header
    // ends in four spaces. Reinsert those bytes so the checked fixtures are
    // exact while the tracked text has no Git whitespace errors.
    fn raw_fixtures() -> (Vec<u8>, Vec<u8>) {
        let mut config = include_bytes!("fixtures/dflash-config.json").to_vec();
        assert!(config.ends_with(b"\n\n}\n"));
        let insert_at = config.len() - b"\n}\n".len();
        config.splice(insert_at..insert_at, b"  ".iter().copied());
        let mut header = include_bytes!("fixtures/dflash-header.json").to_vec();
        assert!(header.ends_with(b"}}"));
        header.extend_from_slice(b"    ");
        assert_eq!(format!("{:x}", Sha256::digest(&config)), CONFIG_SHA256);
        assert_eq!(format!("{:x}", Sha256::digest(&header)), HEADER_SHA256);
        (config, header)
    }

    #[test]
    fn pinned_source_receipt_has_five_layers_and_63_bf16_rows() {
        let (config, header) = raw_fixtures();
        let contract = verify_pinned_auxiliary(SOURCE, &config, &header).unwrap();
        assert_eq!(contract.target_tap_ids, [0, 11, 23, 35, 47]);
        assert_eq!(contract.draft_layers, 5);
        assert_eq!(contract.block_size, 8);
        assert_eq!(contract.rotary_dims, 64);
        assert_eq!(contract.tensor_rows, 63);
        assert_eq!(contract.tensor_bytes, 2_936_114_304);
        assert!(!contract.is_causal);
        assert!(contract.attention_sink_bias);
    }

    #[test]
    fn only_exact_raw_config_can_use_the_single_comma_normalization() {
        let (config, header) = raw_fixtures();
        assert!(serde_json::from_slice::<Value>(&config).is_err());
        assert!(
            serde_json::from_slice::<Value>(&normalize_pinned_config(&config).unwrap()).is_ok()
        );
        let mut changed = config.clone();
        let offset = changed.windows(4).position(|part| part == b"4096").unwrap();
        changed[offset] = b'5';
        assert!(verify_pinned_auxiliary(SOURCE, &changed, &header).is_err());
        let strict = normalize_pinned_config(&config).unwrap();
        assert!(verify_pinned_auxiliary(SOURCE, &strict, &header).is_err());
        assert!(verify_pinned_auxiliary("other/revision", &config, &header).is_err());
    }

    #[test]
    fn changed_header_name_shape_dtype_or_offsets_are_refused() {
        let (config, header) = raw_fixtures();
        let mut changed = header.clone();
        changed[0] = b'x';
        assert!(verify_pinned_auxiliary(SOURCE, &config, &changed).is_err());

        let mut rows = parse_header_json_checked(std::str::from_utf8(&header).unwrap()).unwrap();
        rows.remove("norm.weight");
        assert!(check_header_rows(&rows).is_err());
        let mut rows = parse_header_json_checked(std::str::from_utf8(&header).unwrap()).unwrap();
        let renamed = rows.remove("norm.weight").unwrap();
        rows.insert("other.weight".into(), renamed);
        assert!(check_header_rows(&rows).is_err());
        let mut rows = parse_header_json_checked(std::str::from_utf8(&header).unwrap()).unwrap();
        rows.get_mut("fc.weight").unwrap().shape[0] += 1;
        assert!(check_header_rows(&rows).is_err());
        let mut rows = parse_header_json_checked(std::str::from_utf8(&header).unwrap()).unwrap();
        rows.get_mut("fc.weight").unwrap().dtype = "F16".into();
        assert!(check_header_rows(&rows).is_err());
        let mut rows = parse_header_json_checked(std::str::from_utf8(&header).unwrap()).unwrap();
        rows.get_mut("fc.weight").unwrap().data_offsets[1] -= 2;
        assert!(check_header_rows(&rows).is_err());
    }
}
