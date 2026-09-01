//! Env-gated hidden-state trace, shared by the reference executor and the CUDA trunk.
//!
//! `MEMRA_HYPER_TRACE=<path>` makes BOTH sides append the same stage names, in the same
//! format, from the same token ids — so a bisect compares like with like instead of two
//! hand-rolled dumps that agree on nothing but the layer index. It lives in the reference
//! crate because `memra-engine` already depends on it, so one emitter serves both.
//!
//! Format (`memra-hidden-trace-v1`), one line per stage, the LAST token row only:
//!
//! ```text
//! stage\t<name>\t<layer|-1>\t<width>\t<f32 bits hex>,<f32 bits hex>,...
//! ```
//!
//! Only the last row is emitted: it is the row the logits come from, and it is the row every
//! banked oracle in this lane already pins. Width is `streams * hidden` at the residual-stream
//! stages and `hidden` at the branch stages.
//!
//! Off by default and, when off, costs one `OnceLock` read per call — no allocation, no
//! device work, and no arm selection changes anywhere (unlike the MoE traces, which move
//! `observation_mode`). Turning it on cannot change what the model computes.

use std::fmt::Write as _;
use std::fs::File;
use std::io::Write as _;
use std::sync::{Mutex, OnceLock};

static SINK: OnceLock<Option<Mutex<File>>> = OnceLock::new();

fn sink() -> Option<&'static Mutex<File>> {
    SINK.get_or_init(|| {
        let path = std::env::var_os("MEMRA_HYPER_TRACE")?;
        let mut file = File::create(&path)
            .unwrap_or_else(|error| panic!("MEMRA_HYPER_TRACE={path:?}: {error}"));
        writeln!(file, "format\tmemra-hidden-trace-v1").ok();
        Some(Mutex::new(file))
    })
    .as_ref()
}

/// True when `MEMRA_HYPER_TRACE` named a path. Callers use it to skip a device readback.
pub fn enabled() -> bool {
    sink().is_some()
}

/// Record the run's token ids once, so a trace file can be checked against its oracle TSV.
pub fn emit_tokens(token_ids: &[u32]) {
    let Some(sink) = sink() else { return };
    let ids = token_ids
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let mut file = sink.lock().expect("hidden-trace sink poisoned");
    writeln!(file, "tokens\t{ids}").ok();
}

static ROWS: OnceLock<Option<(std::path::PathBuf, Vec<i64>)>> = OnceLock::new();

/// `MEMRA_TRACE_LAYER_ROWS=<dir>` + `MEMRA_TRACE_LAYER_ROWS_LAYERS=5,14,24,33,42`:
/// dump the CONTRACTED (mean over hyper streams) residual-stream row of every token at the
/// named layers, one raw little-endian f32 file per layer (`<dir>/layer<il>.f32`,
/// `[rows, hidden]` row-major). The contraction is the same arithmetic the glm5_next DFlash2
/// integration pins for its aux-hidden capture (`hc_contract`: mean over the `hc_mult`
/// stream blocks of the completed layer output), so these files are drafter context
/// features, not a debugging trace. Off by default; one OnceLock read when off.
fn rows_cfg() -> Option<&'static (std::path::PathBuf, Vec<i64>)> {
    ROWS.get_or_init(|| {
        let dir = std::env::var_os("MEMRA_TRACE_LAYER_ROWS")?;
        let layers: Vec<i64> = std::env::var("MEMRA_TRACE_LAYER_ROWS_LAYERS")
            .ok()?
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir)
            .unwrap_or_else(|error| panic!("MEMRA_TRACE_LAYER_ROWS={dir:?}: {error}"));
        Some((dir, layers))
    })
    .as_ref()
}

/// True when the layer-rows dump wants this plan layer. Callers use it to skip a readback.
pub fn layer_rows_wanted(layer: i64) -> bool {
    rows_cfg().is_some_and(|(_, layers)| layers.contains(&layer))
}

/// Contract `[rows, streams, width]` token-major data to `[rows, width]` by the stream mean
/// and write it as raw little-endian f32. Truncates any previous file for the layer: one
/// forward per process is the supported shape (run-safetensors).
pub fn emit_layer_rows(layer: i64, rows: usize, streams: usize, width: usize, data: &[f32]) {
    let Some((dir, _)) = rows_cfg() else { return };
    if rows == 0 || streams == 0 || width == 0 || data.len() != rows * streams * width {
        panic!(
            "layer-rows {layer}: {} values is not rows {rows} x streams {streams} x width {width}",
            data.len()
        );
    }
    let mut out = Vec::with_capacity(rows * width * 4);
    let inv = 1.0f32 / streams as f32;
    for t in 0..rows {
        let tok = &data[t * streams * width..(t + 1) * streams * width];
        for i in 0..width {
            let mut sum = 0.0f32;
            for k in 0..streams {
                sum += tok[k * width + i];
            }
            out.extend_from_slice(&(sum * inv).to_le_bytes());
        }
    }
    let path = dir.join(format!("layer{layer}.f32"));
    std::fs::write(&path, &out)
        .unwrap_or_else(|error| panic!("layer-rows {}: {error}", path.display()));
}

/// Emit the last token row of a `[rows, width]` row-major activation.
///
/// `layer` is the plan layer index, or `-1` for the trunk-level stages (`expand`, `collapse`).
pub fn emit_last_row(stage: &str, layer: i64, rows: usize, width: usize, data: &[f32]) {
    let Some(sink) = sink() else { return };
    if rows == 0 || width == 0 || data.len() != rows * width {
        panic!(
            "hidden-trace {stage}[{layer}]: {} values is not rows {rows} x width {width}",
            data.len()
        );
    }
    let row = &data[(rows - 1) * width..];
    let mut line = String::with_capacity(width * 9 + 64);
    let _ = write!(line, "stage\t{stage}\t{layer}\t{width}\t");
    for (index, value) in row.iter().enumerate() {
        if index != 0 {
            line.push(',');
        }
        let _ = write!(line, "{:08x}", value.to_bits());
    }
    line.push('\n');
    let mut file = sink.lock().expect("hidden-trace sink poisoned");
    file.write_all(line.as_bytes()).ok();
}
