//! The F32 macro-scale reader used by resident NVFP4 loads, before device allocation.
use memra_gguf::source::TensorSource;

pub(crate) fn load(source: &dyn TensorSource, name: &str) -> Result<f32, String> {
    let stem = name.strip_suffix(".weight").unwrap_or(name);
    let scale_name = format!("{stem}.scale");
    match source.try_find(&scale_name)? {
        Some(view) => memra_gguf::bound_source::consumer::read_nvfp4_macro_scale(&view)
            .map_err(|e| format!("{scale_name}: {e}")),
        None => Ok(1.0),
    }
}
