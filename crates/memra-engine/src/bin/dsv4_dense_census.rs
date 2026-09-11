//! Dense entry-point SHAPE CENSUS and traffic decomposition (memra #472).
//!
//! The question this answers. memra #471 cut the dense family's launch count 4x
//! and got about a tenth of the family, so roughly seven eighths of a dense
//! launch is per-row work that grouping rows does not touch. Naming that work
//! needs the launch shapes, and the shapes are not in any doc: the four entry
//! points are called from many sites with strides that make the visible tensor
//! dimensions a poor guide. So the engine counts its own calls and this prints
//! the arithmetic that follows from them.
//!
//! What the arithmetic says, stated up front so a reader can check it against
//! the kernels rather than take it. Each kernel's grid is `n` blocks, one per
//! OUTPUT row, 128 threads. Every block reads its own weight row (`k` elements),
//! so a launch reads `n * k` weight elements once. Every block ALSO reads the
//! whole activation row for each of its `M` accumulators, so a launch reads
//! `n * M * k` activation elements: the activation term carries a factor of `n`
//! and the weight term does not. If that is where the time is, then the per-row
//! work is a LOAD PATTERN (the same activation row pulled through cache once per
//! output-row block) rather than arithmetic or address generation, and the fix is
//! a two-dimensional tile that reuses an activation tile across a row tile, which
//! is a batched dense GEMM and is NOT what fusing the chain would buy.
//!
//! This binary does not settle that. It produces the byte counts; the durations
//! that turn them into bandwidths come from an nsys run over the same process,
//! and the ceilings they must be compared against come from the standalone probe
//! in the lane. Numbers printed here are counts and derived bytes only.
use memra_engine::dsv4_gpu::{
    Dsv4Gpu, dense_census_for_gate, dense_tile_counts_for_gate, reset_dense_census_for_gate,
    set_dense_census_for_gate,
};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::Instant;

const SERVED_WIDTH: usize = 64;

fn entry_name(entry: i32) -> &'static str {
    match entry {
        0 => "gemv_bf16",
        1 => "gemv_fp8",
        2 => "dots_f32",
        3 => "dots_f32acc",
        other => panic!("unknown dense entry {other}"),
    }
}

/// Bytes per weight element and per activation element, as the kernel reads them.
/// `gemv_fp8` reads e4m3 weights plus one f32 block scale per 128 columns.
fn element_bytes(entry: i32) -> (f64, f64) {
    match entry {
        0 => (2.0, 2.0),               // bf16 weights, bf16 activations
        1 => (1.0 + 4.0 / 128.0, 2.0), // e4m3 weights + f32 scale per 128, bf16 activations
        2 | 3 => (2.0, 4.0),           // bf16 or f32 weights (bf16 assumed), f32 activations
        other => panic!("unknown dense entry {other}"),
    }
}

fn main() {
    unsafe {
        std::env::set_var("MEMRA_DSV4_HC_DOT_SPLIT", "0");
        std::env::set_var("MEMRA_DSV4_DENSE_FAST", "0");
        std::env::set_var("MEMRA_DSV4_AR_PHASE", "0");
    }
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        3,
        "usage: dsv4_dense_census <model-dir> <real-source.txt>"
    );
    let dir = Path::new(&args[1]);
    let source = std::fs::read_to_string(&args[2]).expect("real source");
    let tokenizer = Tokenizer::from_hf_dir(dir).expect("tokenizer");
    let mut prompt = tokenizer.encode(&format!("Review this engine source:\n{source}"), true);
    assert!(prompt.len() >= 1025, "do not pad/repeat source");
    prompt.truncate(1025);
    println!(
        "SOURCE sha256={:x} tokens={} width={SERVED_WIDTH}",
        Sha256::digest(source.as_bytes()),
        prompt.len()
    );

    let mut gpu = Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, 4096).expect("load");
    gpu.set_prefill_grouped_for_gate(false).expect("arm");
    gpu.set_grouped_route_device_for_gate(false)
        .expect("host routes");

    // A cold first prefill is its own regime and its shapes are the same, so the
    // census is taken on a warm one and the cold one is thrown away.
    {
        let mut state = gpu
            .alloc_decode_state_for_transient(prompt.len() + 32, SERVED_WIDTH)
            .expect("cache");
        let mut draft = gpu.dspark_alloc_state().expect("draft");
        gpu.dspark_prefill_prime_chunked(&prompt, &mut state, &mut draft, SERVED_WIDTH)
            .expect("warmup prime");
    }

    reset_dense_census_for_gate().expect("reset");
    let tiles_before = dense_tile_counts_for_gate();
    set_dense_census_for_gate(true).expect("arm census");
    let mut state = gpu
        .alloc_decode_state_for_transient(prompt.len() + 32, SERVED_WIDTH)
        .expect("cache");
    let mut draft = gpu.dspark_alloc_state().expect("draft");
    let timer = Instant::now();
    gpu.dspark_prefill_prime_chunked(&prompt, &mut state, &mut draft, SERVED_WIDTH)
        .expect("census prime");
    let seconds = timer.elapsed().as_secs_f64();
    set_dense_census_for_gate(false).expect("disarm census");
    let tiles = dense_tile_counts_for_gate();
    let (rows, overflow) = dense_census_for_gate();

    assert!(
        !rows.is_empty(),
        "the census recorded nothing, so it is not armed"
    );
    assert_eq!(
        overflow, 0,
        "the census table overflowed by {overflow} calls, so every total below is a floor"
    );
    println!(
        "PREFILL seconds={seconds:.6} decompositions={} tiles={}",
        tiles[0] - tiles_before[0],
        tiles[1] - tiles_before[1]
    );

    // Sorted by the activation bytes each shape moves, which is the term under
    // test. A shape that is loud in call count and quiet here is not the bill.
    let mut table: Vec<_> = rows
        .iter()
        .map(|row| {
            let (wb, ab) = element_bytes(row.entry);
            let calls = row.calls as f64;
            let n = row.n as f64;
            let m = row.m as f64;
            let k = row.k as f64;
            // Per launch: weights read once per block, activations read once per
            // block PER accumulator, so the activation term carries the n factor.
            let weight_bytes = calls * n * k * wb;
            let activation_bytes = calls * n * m * k * ab;
            let output_bytes = calls * n * m * 4.0;
            let flops = calls * n * m * k * 2.0;
            (row, weight_bytes, activation_bytes, output_bytes, flops)
        })
        .collect();
    table.sort_by(|a, b| b.2.total_cmp(&a.2));

    let mut total_w = 0.0;
    let mut total_a = 0.0;
    let mut total_o = 0.0;
    let mut total_f = 0.0;
    let mut total_calls = 0u64;
    println!(
        "{:<12} {:>4} {:>7} {:>7} {:>9} {:>12} {:>14} {:>12}",
        "entry", "m", "n", "k", "calls", "weight_GB", "activation_GB", "GFLOP"
    );
    for (row, w, a, o, f) in &table {
        total_w += w;
        total_a += a;
        total_o += o;
        total_f += f;
        total_calls += row.calls;
        println!(
            "{:<12} {:>4} {:>7} {:>7} {:>9} {:>12.4} {:>14.4} {:>12.4}",
            entry_name(row.entry),
            row.m,
            row.n,
            row.k,
            row.calls,
            w / 1e9,
            a / 1e9,
            f / 1e9
        );
    }
    println!(
        "TOTALS shapes={} calls={total_calls} weight_GB={:.4} activation_GB={:.4} output_GB={:.4} GFLOP={:.4}",
        table.len(),
        total_w / 1e9,
        total_a / 1e9,
        total_o / 1e9,
        total_f / 1e9
    );
    let bytes = total_w + total_a + total_o;
    println!(
        "RATIO activation_over_weight={:.2}x arithmetic_intensity={:.4} flop_per_byte",
        total_a / total_w,
        total_f / bytes
    );
    // Per prompt row, which is the unit darklanes #596 priced the family in.
    let per_row = prompt.len() as f64;
    println!(
        "PER_ROW weight_MB={:.3} activation_MB={:.3} total_MB={:.3} MFLOP={:.3}",
        total_w / per_row / 1e6,
        total_a / per_row / 1e6,
        bytes / per_row / 1e6,
        total_f / per_row / 1e6
    );
    println!(
        "NOTE bytes here are what the kernels READ, not what leaves HBM: the activation term is \
         the same row pulled through cache once per output-row block. Compare against the L2 and \
         HBM ceilings measured by the lane's standalone probe, never against a spec sheet."
    );
    println!("PASS dense shape census");
}
