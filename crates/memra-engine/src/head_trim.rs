//! Upload only the immutable output of the bound head/rank materializer.
use crate::{Engine, model::GpuTensor};
use memra_gguf::{GgmlType, bound_source::head_trim::PreparedHeadTrim};

pub(crate) fn extra_head_ranks<'a>(
    input: &'a crate::trim_ranks::RankInput,
    extra_heads: usize,
    ids: Option<&[u32]>,
) -> Result<Option<&'a memra_gguf::bound_source::ranks::RankArtifact>, String> {
    if extra_heads == 0 {
        return Ok(None);
    }
    let Some(ids) = ids else {
        return Ok(None);
    };
    let ranks = input
        .captured()
        .ok_or("self-trim chain has no captured rank source")?
        .artifact();
    if ranks.ids() != ids {
        return Err("self-trim chain rank order differs from its captured source".into());
    }
    Ok(Some(ranks))
}

#[allow(clippy::type_complexity)] // allow: retain the existing payload/size receipt at the load seam
pub(crate) fn load(
    e: &Engine,
    trim: &PreparedHeadTrim,
) -> Result<(GpuTensor, Option<(usize, usize)>), Box<dyn std::error::Error>> {
    let ne = trim.shape().to_vec();
    let head = match trim.dtype() {
        GgmlType::BF16 => GpuTensor::FloatBf16 {
            data: e.htod_bytes(trim.bytes())?,
            ne,
        },
        GgmlType::F32 => GpuTensor::Float {
            data: e.htod(
                &trim
                    .bytes()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>(),
            )?,
            ne,
        },
        GgmlType::Q8_0
        | GgmlType::Q4_K
        | GgmlType::Q6_K
        | GgmlType::Q5_K
        | GgmlType::Q3_K
        | GgmlType::IQ4_XS
        | GgmlType::IQ3_S
        | GgmlType::NVFP4
        | GgmlType::Q4_0 => GpuTensor::from_quant_bytes(
            e,
            trim.bytes(),
            trim.dtype(),
            ne[0],
            ne[1],
            trim.macro_scale(),
        )?,
        other => return Err(format!("bound trim has no upload consumer for {other:?}").into()),
    };
    eprintln!(
        "[frspec-trim] bound head={} program={:?} materialization={}",
        trim.runtime_name(),
        trim.identity().program(),
        trim.identity().materialization_sha256()
    );
    Ok((head, trim.requant_sizes()))
}
