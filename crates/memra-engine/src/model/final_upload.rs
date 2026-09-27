//! Exact final host buffers submitted by the bound trim upload path. Device execution and
//! readback qualification remain separate; these receipts cannot install rewrite admission.
use super::GpuTensor;
use crate::Engine;
use memra_gguf::{
    GgmlType,
    bound_source::draft_pair::{
        DraftProposalIdentity, DraftSourcePairIdentity, PreparedPairedHeadTrim,
    },
    tensor_contract::{MtpTensor, TensorId},
};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UploadLayout {
    F32Native,
    Bf16LittleEndian,
    GgmlBlocks { qtype: i32, row_bytes: usize },
    Nvfp4SplitA6 { qtype: i32, row_bytes: usize },
}

/// Produced only after the actual backend copy returns and its tensor metadata is checked.
/// The digest is of the same final buffer supplied to H2D, not a source-path reread or a
/// callback's arbitrary output. It is not a device-readback or atomic-source-snapshot claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinalUploadIdentity {
    host_sha256: String,
    submitted_sha256: String,
    byte_len: usize,
    dtype: GgmlType,
    shape: [u64; 2],
    macro_bits: u32,
    device: usize,
    layout: UploadLayout,
    sha256: String,
}
impl FinalUploadIdentity {
    pub fn host_sha256(&self) -> &str {
        &self.host_sha256
    }
    pub fn submitted_sha256(&self) -> &str {
        &self.submitted_sha256
    }
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }
    pub fn dtype(&self) -> GgmlType {
        self.dtype
    }
    pub fn shape(&self) -> [u64; 2] {
        self.shape
    }
    pub fn macro_bits(&self) -> u32 {
        self.macro_bits
    }
    pub fn device(&self) -> usize {
        self.device
    }
    pub fn layout(&self) -> &UploadLayout {
        &self.layout
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub(crate) fn validate_assigned(&self, tensor: &GpuTensor) -> Result<(), String> {
        let layout = tensor_layout(
            tensor,
            self.dtype,
            self.shape,
            f32::from_bits(self.macro_bits),
            self.byte_len,
        )?;
        if layout != self.layout || tensor.ordinal() != self.device {
            return Err("assigned trim tensor layout/device differs from its actual upload".into());
        }
        Ok(())
    }

    pub(crate) fn after_upload(
        tensor: &GpuTensor,
        host: &[u8],
        submitted: &[u8],
        dtype: GgmlType,
        shape: [u64; 2],
        scale: f32,
    ) -> Result<Self, String> {
        validate_input(host, dtype, shape)?;
        if submitted.len() != host.len() {
            return Err("trim upload changed encoded byte count".into());
        }
        let layout = tensor_layout(tensor, dtype, shape, scale, submitted.len())?;
        validate_layout_bytes(host, submitted, &layout)?;
        let host_sha256 = digest(host);
        let submitted_sha256 = digest(submitted);
        let device = tensor.ordinal();
        let macro_bits = scale.to_bits();
        let sha256 = framed(
            "memra-final-trim-upload-v1",
            &[
                host_sha256.clone(),
                submitted_sha256.clone(),
                format!(
                    "{:?}",
                    (submitted.len(), dtype, shape, macro_bits, device, &layout)
                ),
            ],
        );
        Ok(Self {
            host_sha256,
            submitted_sha256,
            byte_len: submitted.len(),
            dtype,
            shape,
            macro_bits,
            device,
            layout,
            sha256,
        })
    }
}

pub(crate) fn validate_input(bytes: &[u8], dtype: GgmlType, shape: [u64; 2]) -> Result<(), String> {
    if shape.contains(&0) {
        return Err("trim upload shape must be nonempty".into());
    }
    let (block, _) = dtype.block_and_type_size();
    if !shape[0].is_multiple_of(block) {
        return Err("trim upload rows are not independently encoded".into());
    }
    let expected = dtype
        .checked_nbytes(&shape)
        .ok_or("trim upload shape/byte count overflow")?;
    if u64::try_from(bytes.len()).map_err(|_| "trim upload length exceeds u64")? != expected {
        return Err("trim upload bytes do not match dtype/shape".into());
    }
    Ok(())
}

fn tensor_layout(
    tensor: &GpuTensor,
    dtype: GgmlType,
    shape: [u64; 2],
    scale: f32,
    byte_len: usize,
) -> Result<UploadLayout, String> {
    match tensor {
        GpuTensor::Float { data, ne }
            if dtype == GgmlType::F32
                && *ne == shape
                && data.len().checked_mul(4) == Some(byte_len)
                && scale.to_bits() == 1.0f32.to_bits() =>
        {
            Ok(UploadLayout::F32Native)
        }
        GpuTensor::FloatBf16 { data, ne }
            if dtype == GgmlType::BF16
                && *ne == shape
                && data.len() == byte_len
                && scale.to_bits() == 1.0f32.to_bits() =>
        {
            Ok(UploadLayout::Bf16LittleEndian)
        }
        GpuTensor::Quant {
            bytes,
            qtype,
            row_bytes,
            ne,
            scale: actual_scale,
            rp,
            fp8,
            rp4,
            blk,
            f16,
            a4,
            ..
        } => {
            if *ne != shape
                || bytes.len() != byte_len
                || *qtype != super::quant_type_for_trim(dtype)?
                || actual_scale.to_bits() != scale.to_bits()
                || row_bytes.checked_mul(
                    usize::try_from(shape[1]).map_err(|_| "trim upload rows exceed usize")?,
                ) != Some(byte_len)
                || fp8.is_some()
                || rp4.is_some()
                || blk.is_some()
                || f16.is_some()
                || a4.is_some()
            {
                return Err("returned quant tensor differs from submitted trim operand".into());
            }
            #[cfg(memra_cutlass)]
            if let GpuTensor::Quant {
                cutlass: Some(_), ..
            } = tensor
            {
                return Err("trim upload cannot claim an unrecorded CUTLASS mirror".into());
            }
            if *rp {
                if dtype != GgmlType::NVFP4
                    || !shape[0].is_multiple_of(64)
                    || !row_bytes.is_multiple_of(36)
                {
                    return Err("trim upload split layout requires NVFP4 rows".into());
                }
                Ok(UploadLayout::Nvfp4SplitA6 {
                    qtype: *qtype,
                    row_bytes: *row_bytes,
                })
            } else {
                Ok(UploadLayout::GgmlBlocks {
                    qtype: *qtype,
                    row_bytes: *row_bytes,
                })
            }
        }
        _ => Err("returned tensor dtype/shape/bytes differ from submitted trim operand".into()),
    }
}

fn validate_layout_bytes(
    host: &[u8],
    submitted: &[u8],
    layout: &UploadLayout,
) -> Result<(), String> {
    let matches = match layout {
        UploadLayout::GgmlBlocks { .. } | UploadLayout::Bf16LittleEndian => host == submitted,
        UploadLayout::F32Native => {
            host.chunks_exact(4)
                .zip(submitted.chunks_exact(4))
                .all(|(source, target)| {
                    u32::from_le_bytes(source.try_into().unwrap()).to_ne_bytes() == target
                })
        }
        UploadLayout::Nvfp4SplitA6 { .. } => {
            let qplane = host.len() / 36 * 32;
            host.chunks_exact(36).enumerate().all(|(block, source)| {
                submitted[block * 32..(block + 1) * 32] == source[4..]
                    && submitted[qplane + block * 4..qplane + (block + 1) * 4] == source[..4]
            })
        }
    };
    if !matches {
        return Err("submitted trim bytes do not implement the declared upload layout".into());
    }
    Ok(())
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn framed(domain: &str, fields: &[String]) -> String {
    let mut hash = Sha256::new();
    for value in std::iter::once(domain).chain(fields.iter().map(String::as_str)) {
        hash.update((value.len() as u64).to_le_bytes());
        hash.update(value.as_bytes());
    }
    format!("{:x}", hash.finalize())
}

/// GPU tensor and receipt have one owner. There is no caller-supplied tensor/callback/hash
/// constructor and no mutable tensor accessor. A runtime integration must preserve this link.
pub struct UploadedHeadTrim {
    tensor: GpuTensor,
    source: DraftSourcePairIdentity,
    upload: FinalUploadIdentity,
    sha256: String,
}
impl UploadedHeadTrim {
    pub fn load(
        e: &Engine,
        paired: &PreparedPairedHeadTrim,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let trim = paired.materialization();
        let (tensor, upload) = load_recorded(e, trim)?;
        if upload.host_sha256() != trim.identity().output_sha256()
            || upload.dtype() != trim.dtype()
            || upload.shape() != trim.shape()
            || upload.macro_bits() != trim.macro_scale().to_bits()
        {
            return Err("submitted trim differs from source-owned host materialization".into());
        }
        if tensor.ordinal() != e.ctx().ordinal() {
            return Err("trim upload returned a tensor on the wrong device".into());
        }
        let source = paired.identity().clone();
        let sha256 = framed(
            "memra-source-submitted-trim-v1",
            &[source.source_pair_sha256().into(), upload.sha256().into()],
        );
        Ok(Self {
            tensor,
            source,
            upload,
            sha256,
        })
    }
    pub(crate) fn install_into(
        self,
        slot: &mut Option<GpuTensor>,
    ) -> Result<(DraftSourcePairIdentity, FinalUploadIdentity, String), String> {
        if slot.is_some() {
            return Err(
                "paired model slot contains a substituted tensor before installation".into(),
            );
        }
        *slot = Some(self.tensor);
        Ok((self.source, self.upload, self.sha256))
    }
    pub fn tensor(&self) -> &GpuTensor {
        &self.tensor
    }
    pub fn source(&self) -> &DraftSourcePairIdentity {
        &self.source
    }
    pub fn upload(&self) -> &FinalUploadIdentity {
        &self.upload
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
}

/// Ordered first/extra embedded-MTP heads. All slots must belong to one target and captured
/// rank artifact. This is upload provenance, not a qualified speculative execution surface.
pub struct UploadedTrimChain {
    heads: Vec<UploadedHeadTrim>,
    sha256: String,
}
impl UploadedTrimChain {
    pub fn load(
        e: &Engine,
        prepared: &[PreparedPairedHeadTrim],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let first = prepared.first().ok_or("trim upload chain is empty")?;
        for (slot, paired) in prepared.iter().enumerate() {
            if paired.identity().target() != first.identity().target()
                || paired.materialization().identity().rank()
                    != first.materialization().identity().rank()
            {
                return Err(
                    "trim upload chain target/rank provenance differs between slots".into(),
                );
            }
            let DraftProposalIdentity::TargetHeadTrim(identity) = paired.identity().proposal()
            else {
                return Err("trim upload chain requires target materializations".into());
            };
            let expected = u32::try_from(slot).map_err(|_| "trim upload slot exceeds u32")?;
            let valid = match identity.semantic_head() {
                TensorId::TokenEmbedding | TensorId::OutputProjection => slot == 0,
                TensorId::Mtp {
                    depth,
                    tensor: MtpTensor::OutputProjection,
                } => *depth == expected,
                _ => false,
            };
            if !valid {
                return Err(
                    "trim upload chain head roles are missing, duplicated or out of order".into(),
                );
            }
        }
        let heads = prepared
            .iter()
            .map(|paired| UploadedHeadTrim::load(e, paired))
            .collect::<Result<Vec<_>, _>>()?;
        let sha256 = framed(
            "memra-ordered-submitted-trim-chain-v1",
            &heads
                .iter()
                .map(|head| head.sha256().to_owned())
                .collect::<Vec<_>>(),
        );
        Ok(Self { heads, sha256 })
    }
    pub fn heads(&self) -> &[UploadedHeadTrim] {
        &self.heads
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
}

/// Proof-only upload. The actual submitted buffer stays alive through copy and hashing;
/// unknown modes/lengths refuse and failed backend copies produce no upload identity.
pub(crate) fn load_recorded(
    e: &Engine,
    trim: &memra_gguf::bound_source::head_trim::PreparedHeadTrim,
) -> Result<(GpuTensor, crate::model::final_upload::FinalUploadIdentity), Box<dyn std::error::Error>>
{
    let shape = trim.shape();
    validate_input(trim.bytes(), trim.dtype(), shape)?;
    match trim.dtype() {
        GgmlType::BF16 => {
            let bytes = trim.bytes().to_vec();
            let tensor = GpuTensor::FloatBf16 {
                data: e.htod_bytes(&bytes)?,
                ne: shape.to_vec(),
            };
            let identity = FinalUploadIdentity::after_upload(
                &tensor,
                &bytes,
                &bytes,
                trim.dtype(),
                shape,
                1.0,
            )?;
            Ok((tensor, identity))
        }
        GgmlType::F32 => {
            let values = trim
                .bytes()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            let tensor = GpuTensor::Float {
                data: e.htod(&values)?,
                ne: shape.to_vec(),
            };
            // SAFETY: f32 has no padding or invalid bit patterns. This borrows the same
            // immutable allocation passed to htod, in its actual native byte encoding.
            let submitted = unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(values.as_slice()),
                )
            };
            let identity = FinalUploadIdentity::after_upload(
                &tensor,
                trim.bytes(),
                submitted,
                trim.dtype(),
                shape,
                1.0,
            )?;
            Ok((tensor, identity))
        }
        GgmlType::Q8_0
        | GgmlType::Q4_K
        | GgmlType::Q6_K
        | GgmlType::Q5_K
        | GgmlType::Q3_K
        | GgmlType::IQ4_XS
        | GgmlType::IQ3_S
        | GgmlType::NVFP4
        | GgmlType::Q4_0 => GpuTensor::from_quant_bytes_recorded(
            e,
            trim.bytes(),
            trim.dtype(),
            shape[0],
            shape[1],
            trim.macro_scale(),
        ),
        other => Err(format!("bound trim has no recorded upload consumer for {other:?}").into()),
    }
}
