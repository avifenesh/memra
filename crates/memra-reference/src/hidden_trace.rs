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
