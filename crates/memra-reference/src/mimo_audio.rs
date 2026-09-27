//! Portable MiMo audio-code grouping and embedding sum.
//! The bundled codec, six-layer local transformer, and projection remain
//! separate execution and parity gates.

use memra_gguf::model_packs::mimo_v2::audio::MiMoAudioPatchPlan;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupedAudioCodes {
    pub groups: usize,
    /// Row-major `[groups, group_size, code_channels]` after final-row padding.
    pub codes: Vec<u16>,
}

/// Match the source's first-20-channel slice and repeated final-row padding.
pub fn group_audio_codes(
    plan: &MiMoAudioPatchPlan,
    codes: &[i64],
    channels_per_row: usize,
) -> Result<GroupedAudioCodes, String> {
    if plan.code_channels != 20 || plan.group_size != 4 || plan.code_vocab != 1_280 {
        return Err("MiMo audio code geometry differs from pinned source".into());
    }
    if channels_per_row < plan.code_channels
        || codes.is_empty()
        || !codes.len().is_multiple_of(channels_per_row)
    {
        return Err("MiMo audio codes need nonempty rows with at least 20 channels".into());
    }
    let rows = codes.len() / channels_per_row;
    let groups = rows.div_ceil(plan.group_size);
    let padded_rows = groups
        .checked_mul(plan.group_size)
        .ok_or("MiMo audio padded row count overflows")?;
    let capacity = padded_rows
        .checked_mul(plan.code_channels)
        .ok_or("MiMo audio code extent overflows")?;
    let mut grouped = Vec::with_capacity(capacity);
    for row in 0..padded_rows {
        let source_row = row.min(rows - 1);
        let offset = source_row * channels_per_row;
        for &code in &codes[offset..offset + plan.code_channels] {
            let id = usize::try_from(code).map_err(|_| "MiMo audio code is negative")?;
            if id >= plan.code_vocab {
                return Err(format!(
                    "MiMo audio code {id} exceeds the source vocabulary"
                ));
            }
            grouped.push(id as u16);
        }
    }
    Ok(GroupedAudioCodes {
        groups,
        codes: grouped,
    })
}

fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let bias = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(bias) >> 16) as u16
}

/// In-place BF16 summation in code-channel order, as in the source's `out.add_`.
/// `table_vocab` permits small synthetic tables; the pinned checkpoint contract
/// separately requires 1,280 rows in every physical speech embedding table.
pub fn sum_speech_embeddings_bf16(
    plan: &MiMoAudioPatchPlan,
    grouped: &GroupedAudioCodes,
    tables: &[&[u16]],
    table_vocab: usize,
) -> Result<Vec<u16>, String> {
    if plan.code_channels != 20
        || plan.group_size != 4
        || plan.local_hidden != 1_024
        || grouped.groups == 0
        || tables.len() != plan.code_channels
        || table_vocab == 0
    {
        return Err("MiMo audio speech embedding geometry changed".into());
    }
    let positions = grouped
        .groups
        .checked_mul(plan.group_size)
        .ok_or("MiMo audio group extent overflows")?;
    let codes_len = positions
        .checked_mul(plan.code_channels)
        .ok_or("MiMo audio grouped-code extent overflows")?;
    let table_len = table_vocab
        .checked_mul(plan.local_hidden)
        .ok_or("MiMo audio table extent overflows")?;
    let output_len = positions
        .checked_mul(plan.local_hidden)
        .ok_or("MiMo audio embedding extent overflows")?;
    if grouped.codes.len() != codes_len || tables.iter().any(|table| table.len() != table_len) {
        return Err("MiMo audio grouped codes or embedding table extents changed".into());
    }
    let mut output = vec![0u16; output_len];
    for (channel, table) in tables.iter().enumerate() {
        let table = *table;
        for position in 0..positions {
            let code = usize::from(grouped.codes[position * plan.code_channels + channel]);
            if code >= table_vocab {
                return Err(format!(
                    "MiMo audio code {code} exceeds the embedding table"
                ));
            }
            let source = &table[code * plan.local_hidden..(code + 1) * plan.local_hidden];
            let target =
                &mut output[position * plan.local_hidden..(position + 1) * plan.local_hidden];
            for (value, &weight) in target.iter_mut().zip(source) {
                let weight = bf16_to_f32(weight);
                let sum = bf16_to_f32(*value) + weight;
                if !weight.is_finite() || !sum.is_finite() {
                    return Err("MiMo audio speech embedding is not finite".into());
                }
                let rounded = f32_to_bf16_rne(sum);
                if !bf16_to_f32(rounded).is_finite() {
                    return Err("MiMo audio BF16 embedding sum overflowed".into());
                }
                *value = rounded;
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::config::{HfConfig, ModelConfig};
    use memra_gguf::model_packs::mimo_v2::audio::pinned_patch_plan;

    fn plan() -> MiMoAudioPatchPlan {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!(
            "../../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        )));
        pinned_patch_plan(&config).unwrap()
    }

    #[test]
    fn final_code_row_repeats_and_extra_channels_are_sliced() {
        let mut codes = Vec::new();
        for row in 0..3 {
            codes.extend(std::iter::repeat_n(row as i64, 20));
            codes.extend([10_000, -1]);
        }
        let grouped = group_audio_codes(&plan(), &codes, 22).unwrap();
        assert_eq!(grouped.groups, 1);
        assert_eq!(grouped.codes.len(), 80);
        assert_eq!(&grouped.codes[0..20], &[0u16; 20]);
        assert_eq!(&grouped.codes[20..40], &[1u16; 20]);
        assert_eq!(&grouped.codes[40..60], &[2u16; 20]);
        assert_eq!(&grouped.codes[60..80], &[2u16; 20]);
        assert!(group_audio_codes(&plan(), &[], 20).is_err());
        assert!(group_audio_codes(&plan(), &[0; 19], 19).is_err());
        assert!(group_audio_codes(&plan(), &[-1; 20], 20).is_err());
        assert!(group_audio_codes(&plan(), &[1_280; 20], 20).is_err());
    }

    #[test]
    fn source_order_rounds_after_each_bf16_channel_addition() {
        let grouped = group_audio_codes(&plan(), &[1; 20], 20).unwrap();
        let mut tables = vec![vec![0u16; 2 * 1_024]; 20];
        tables[0][1_024] = 0x3f80; // 1.0
        tables[1][1_024] = 0x3b80; // half a BF16 ULP at 1.0
        tables[2][1_024] = 0x3b80;
        let rows: Vec<&[u16]> = tables.iter().map(Vec::as_slice).collect();
        let output = sum_speech_embeddings_bf16(&plan(), &grouped, &rows, 2).unwrap();
        assert_eq!(output.len(), 4 * 1_024);
        for position in 0..4 {
            assert_eq!(output[position * 1_024], 0x3f80);
        }
        assert!(sum_speech_embeddings_bf16(&plan(), &grouped, &rows[..19], 2).is_err());
    }
}
