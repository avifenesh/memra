//! Cross-surface contracts for DeepSeek-class sigmoid routing.

use std::sync::OnceLock;

const SERVED_LOGIT_MAGIC: &[u8; 8] = b"MSIGRPL1";

pub struct ServedLogitRecord {
    pub layer: u32,
    pub tokens: usize,
    pub n_expert: usize,
    pub n_used: usize,
    pub scaling_factor: f32,
    pub route_norm: bool,
    pub active: Vec<u8>,
    pub bias: Vec<f32>,
    pub logits: Vec<f32>,
}

struct ServedLogitWriter {
    path: std::path::PathBuf,
    file: std::fs::File,
    seen_layers: std::collections::HashSet<u32>,
}

static SERVED_LOGIT_WRITER: OnceLock<std::sync::Mutex<Option<ServedLogitWriter>>> = OnceLock::new();

/// Reject an undersubscribed active set before sorting, slicing, or launching CUDA work.
pub fn validate_active_count(n_used: usize, active_count: usize) -> Result<(), String> {
    if active_count < n_used {
        return Err(format!(
            "sigmoid router requires active_count >= n_used: active_count={active_count}, n_used={n_used}",
        ));
    }
    Ok(())
}

/// Probe the runtime scalar expf against the bit patterns that froze the host/device oracle.
pub fn verify_host_expf() -> Result<(), String> {
    static RESULT: OnceLock<Result<(), String>> = OnceLock::new();
    RESULT
        .get_or_init(|| {
            const CASES: &[(u32, u32)] = &[
                (0xc0f4_bca4, 0x39fa_13b9), (0x4152_4c00, 0x48f9_5e97),
                (0xc100_cdee, 0x39a7_4153), (0xc07a_9e68, 0x3ca3_33fa),
                (0x412b_32ca, 0x472d_3f68), (0xc153_f532, 0x35ec_e4cd),
                (0x4031_a0d8, 0x4180_5da2), (0xc0e3_fbc0, 0x3a53_10be),
                (0x4141_525e, 0x482c_a0b2), (0x40f0_a9a8, 0x44e6_bc14),
                (0xc158_4cca, 0x35b4_96ea), (0xc123_1dda, 0x381c_b7e3),
                (0x3f72_ed60, 0x4025_4f28), (0x4179_f55a, 0x4ab9_e5a0),
                (0x415c_15c0, 0x4965_e07c), (0x3fa0_5800, 0x405f_fb8f),
                (0x416d_e91a, 0x4a2f_193b), (0x417c_082a, 0x4ad3_9e5b),
                (0x4133_a8f6, 0x4792_ffbd), (0xc17b_4c4a, 0x3422_1cc9),
                (0x4049_f238, 0x41bb_b376), (0x4122_c18a, 0x46cc_6dd0),
                (0x40c3_8904, 0x43e1_46c6), (0x411f_3fd2, 0x46a4_31c2),
            ];
            for (case, &(input_bits, expected_bits)) in CASES.iter().enumerate() {
                let input = std::hint::black_box(f32::from_bits(input_bits));
                let actual_bits = input.exp().to_bits();
                if actual_bits != expected_bits {
                    return Err(format!(
                        "host expf byte probe mismatch at case {case}: input=0x{input_bits:08x}, expected=0x{expected_bits:08x}, actual=0x{actual_bits:08x}",
                    ));
                }
            }
            Ok(())
        })
        .clone()
}

pub fn served_logit_trace_enabled() -> bool {
    std::env::var_os("MEMRA_SIG_ROUTER_LOGIT_TRACE").is_some()
}

/// Persist the first real decode router row for each layer. All floats are stored as raw f32 bits,
/// so replay tests the exact served inputs rather than a decimal serialization of them.
#[allow(clippy::too_many_arguments)]
pub fn capture_served_logits(
    layer: u32,
    tokens: usize,
    n_expert: usize,
    n_used: usize,
    scaling_factor: f32,
    route_norm: bool,
    active: &[u8],
    bias: &[f32],
    logits: &[f32],
) -> Result<(), String> {
    use std::io::Write as _;

    let Some(path) = std::env::var_os("MEMRA_SIG_ROUTER_LOGIT_TRACE") else {
        return Ok(());
    };
    if tokens != 1 {
        return Ok(());
    }
    if active.len() != n_expert || bias.len() != n_expert || logits.len() < n_expert {
        return Err(format!(
            "served sigmoid-logit trace shape mismatch at layer {layer}: active={} bias={} logits={} n_expert={n_expert}",
            active.len(),
            bias.len(),
            logits.len(),
        ));
    }
    let path = std::path::PathBuf::from(path);
    let state = SERVED_LOGIT_WRITER.get_or_init(|| std::sync::Mutex::new(None));
    let mut state = state
        .lock()
        .map_err(|_| "served sigmoid-logit trace writer lock is poisoned".to_string())?;
    if state.is_none() {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&path)
            .map_err(|error| {
                format!(
                    "cannot create served sigmoid-logit trace {}: {error}",
                    path.display()
                )
            })?;
        file.write_all(SERVED_LOGIT_MAGIC)
            .map_err(|error| format!("cannot write served sigmoid-logit trace header: {error}"))?;
        *state = Some(ServedLogitWriter {
            path: path.clone(),
            file,
            seen_layers: std::collections::HashSet::new(),
        });
    }
    let writer = state.as_mut().unwrap();
    if writer.path != path {
        return Err("MEMRA_SIG_ROUTER_LOGIT_TRACE changed after capture started".into());
    }
    if !writer.seen_layers.insert(layer) {
        return Ok(());
    }

    for value in [
        layer,
        tokens as u32,
        n_expert as u32,
        n_used as u32,
        scaling_factor.to_bits(),
        u32::from(route_norm),
    ] {
        writer
            .file
            .write_all(&value.to_le_bytes())
            .map_err(|error| format!("cannot write served sigmoid-logit trace row: {error}"))?;
    }
    writer
        .file
        .write_all(active)
        .map_err(|error| format!("cannot write served sigmoid-logit active mask: {error}"))?;
    for value in bias.iter().chain(logits[..n_expert].iter()) {
        writer
            .file
            .write_all(&value.to_bits().to_le_bytes())
            .map_err(|error| format!("cannot write served sigmoid-logit f32 row: {error}"))?;
    }
    writer
        .file
        .flush()
        .map_err(|error| format!("cannot flush served sigmoid-logit trace: {error}"))?;
    Ok(())
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, String> {
    let end = cursor.saturating_add(4);
    let raw: [u8; 4] = bytes
        .get(*cursor..end)
        .ok_or_else(|| "truncated served sigmoid-logit trace".to_string())?
        .try_into()
        .unwrap();
    *cursor = end;
    Ok(u32::from_le_bytes(raw))
}

pub fn read_served_logits(path: &std::path::Path) -> Result<Vec<ServedLogitRecord>, String> {
    let bytes = std::fs::read(path).map_err(|error| {
        format!(
            "cannot read served sigmoid-logit trace {}: {error}",
            path.display()
        )
    })?;
    if bytes.get(..SERVED_LOGIT_MAGIC.len()) != Some(SERVED_LOGIT_MAGIC) {
        return Err("served sigmoid-logit trace has wrong or missing v1 header".into());
    }
    let mut cursor = SERVED_LOGIT_MAGIC.len();
    let mut records = Vec::new();
    while cursor < bytes.len() {
        let layer = read_u32(&bytes, &mut cursor)?;
        let tokens = read_u32(&bytes, &mut cursor)? as usize;
        let n_expert = read_u32(&bytes, &mut cursor)? as usize;
        let n_used = read_u32(&bytes, &mut cursor)? as usize;
        let scaling_factor = f32::from_bits(read_u32(&bytes, &mut cursor)?);
        let route_norm = match read_u32(&bytes, &mut cursor)? {
            0 => false,
            1 => true,
            value => {
                return Err(format!(
                    "invalid route_norm={value} in served sigmoid-logit trace"
                ));
            }
        };
        if tokens != 1 || n_expert == 0 || n_used == 0 || n_expert > 1024 {
            return Err(format!(
                "invalid served sigmoid-logit record shape: layer={layer} tokens={tokens} n_expert={n_expert} n_used={n_used}",
            ));
        }
        let active_end = cursor.saturating_add(n_expert);
        let active = bytes
            .get(cursor..active_end)
            .ok_or_else(|| "truncated served sigmoid-logit active mask".to_string())?
            .to_vec();
        cursor = active_end;
        let mut read_f32_row = || -> Result<Vec<f32>, String> {
            (0..n_expert)
                .map(|_| read_u32(&bytes, &mut cursor).map(f32::from_bits))
                .collect()
        };
        let bias = read_f32_row()?;
        let logits = read_f32_row()?;
        records.push(ServedLogitRecord {
            layer,
            tokens,
            n_expert,
            n_used,
            scaling_factor,
            route_norm,
            active,
            bias,
            logits,
        });
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_count_error_quotes_both_counts() {
        assert_eq!(
            validate_active_count(8, 7).unwrap_err(),
            "sigmoid router requires active_count >= n_used: active_count=7, n_used=8",
        );
        assert!(validate_active_count(8, 8).is_ok());
    }

    #[test]
    fn pinned_host_expf_matches() {
        verify_host_expf().unwrap();
    }
}
