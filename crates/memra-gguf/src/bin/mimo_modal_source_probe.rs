//! Read selected bound MiMo modal tensors from the pinned full source.
//! This probes source custody, not modality execution or serving support.

use memra_gguf::checkpoint_binding::RecordingSource;
use memra_gguf::model_packs::mimo_v2::{bind_pinned_text_source, read_bound_modal_bf16};
use memra_gguf::source::SafetensorsSource;
use memra_gguf::tensor_contract::{MtpTensor, TensorId, VisionTensor};
use serde_json::json;
use sha2::{Digest, Sha256};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let dir = args
        .next()
        .ok_or("usage: mimo_modal_source_probe <pinned-source-dir>")?;
    if args.next().is_some() {
        return Err("usage: mimo_modal_source_probe <pinned-source-dir>".into());
    }
    let source = SafetensorsSource::open(std::path::Path::new(&dir))?;
    let (config, _, binding) = bind_pinned_text_source(&source)?;
    let recording = RecordingSource::new(&source);
    let targets = [
        (
            "vision_patch",
            TensorId::Vision {
                layer: None,
                tensor: VisionTensor::PatchProjection,
            },
        ),
        (
            "audio_encoder_norm",
            TensorId::Family {
                family: "mimo_v2_audio",
                key: "audio_encoder.input_local_transformer.norm.weight".into(),
            },
        ),
        (
            "mtp_fusion",
            TensorId::Mtp {
                depth: 0,
                tensor: MtpTensor::FusionProjection,
            },
        ),
    ];
    let mut rows = Vec::new();
    for (label, id) in &targets {
        let hf_name = binding.require_hf(id)?;
        let view = read_bound_modal_bf16(&recording, &binding, id)?;
        rows.push(json!({
            "label": label,
            "hf_name": hf_name,
            "ne": view.ne,
            "bytes": view.bytes.len(),
            "payload_sha256": format!("{:x}", Sha256::digest(view.bytes.as_ref())),
        }));
    }
    let missing = binding.audit_consumption(&recording.requested(), &config, |id, _| {
        !targets.iter().any(|(_, target)| id == target)
    });
    if !missing.is_empty() {
        return Err(format!("MiMo modal probe left selected tensors unread: {missing:?}").into());
    }
    println!(
        "{}",
        serde_json::to_string(&json!({
            "format": "memra-mimo-bound-modal-source-v1",
            "bound_tensors": binding.bound.tensors.len(),
            "selected_tensors": rows,
            "audit_unconsumed_selected": missing.len(),
            "scope": "source-payload-only",
        }))?
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_modal_source_probe: {error}");
        std::process::exit(1);
    }
}
