//! Causal instrument for the residual all-reduce pool, and the arms that catch it lying.
//!
//! WHAT THIS IS FOR. The replay map ranks the remaining 85 ARs at 1.977316 ms/step on rank 0 and
//! records them as the largest sink we cannot attribute: nsys graph-node tracing moves the first-AR
//! and launch-API rows by two orders of magnitude (darklanes #519), and per-AR CUDA events cannot
//! exist outside a captured graph. The stamps in `memra_tp_ar_1stage_kernel` decompose each join
//! into peer wait, reduce and tail wait; the null arm bounds how much of the reduce is transport.
//!
//! WHAT THIS IS NOT. No lever is chosen here and no saving is claimed. A phase number is an
//! attribution of time already spent, and darklanes VERDICT:dsv4-issue-interleave-nogo is the
//! standing reason to keep those apart: an arm that cut 3.197 ms/token of AR residence moved almost
//! all of it into kernel-free span and improved rank-0's span by 0.129 ms. Wait time is not a
//! budget. Every phase row printed here is diagnostic exposure.
//!
//! ONE ARM PER PROCESS. `MEMRA_DSV4_AR_PHASE` is read at load, so the controller runs a fresh
//! process per arm and each cell below states which arm it requires.
use memra_engine::dsv4_gpu::{ArPhaseDoor, DecodeState, Dsv4Gpu, Dsv4SampleCfg};
use memra_engine::dsv4_sampler::{Dsv4Sampler, dsv4_sampler};
use memra_engine::tp_ar::ArPhaseRecord;
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const PRIME: usize = 256;
const STEPS: usize = 32;
const CAPACITY: usize = PRIME + 256 + 8;
/// 43 layers x (attention join, expert join).
const ARS_PER_STEP: usize = 86;
/// One window's worth of records per rank, plus room for the arming step's own joins so a
/// mis-bracketed window is a REFUSAL from the drain rather than a silently wrapped ring.
const WINDOW: usize = ARS_PER_STEP * (STEPS + 2);
/// Plausible SM clock band for this card, in nanoseconds per cycle: 5.0 GHz down to 0.5 GHz. A
/// record calibrating outside it is not a slow all-reduce, it is a stamp that did not land where
/// the analysis believes, and the gate throws it out instead of reporting it.
const NS_PER_CYCLE: (f64, f64) = (0.2, 2.0);

fn sha_tokens(tokens: &[u32]) -> String {
    let mut h = Sha256::new();
    for v in tokens {
        h.update(v.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

fn sha_f32(row: &[f32]) -> String {
    let mut h = Sha256::new();
    for v in row {
        h.update(v.to_bits().to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

type Identity = (String, [u64; 2], [u64; 2]);

fn identity(gpu: &Dsv4Gpu, state: &DecodeState) -> Identity {
    let logits = gpu
        .read_decode_logits_for_gate(state)
        .expect("final logits");
    let cache = gpu
        .tp_ep_cache_digest_for_gate(state)
        .expect("cache digest");
    let hidden = gpu
        .tp_ep_hidden_digest_for_gate(state)
        .expect("hidden digest");
    (sha_f32(&logits), cache, hidden)
}

fn state(gpu: &Dsv4Gpu) -> DecodeState {
    gpu.alloc_decode_state_for_transient(CAPACITY, 1)
        .expect("independent state")
}

fn graph_state(gpu: &Dsv4Gpu, prefix: &DecodeState, cfg: Dsv4SampleCfg) -> DecodeState {
    let mut result = state(gpu);
    gpu.restore_full_token_prefix_for_gate(&mut result, prefix)
        .unwrap();
    unsafe {
        gpu.arm_full_token_replay_for_gate(&mut result, cfg)
            .unwrap();
    }
    result
}

/// Phase sums over one window, per rank, in nanoseconds. Every record is converted with its OWN
/// calibration ratio, so a clock that moved during the window rescales each record correctly
/// instead of applying one wrong constant to all of them.
#[derive(Clone, Copy, Debug, Default)]
struct PhaseSum {
    records: usize,
    wait_ns: f64,
    reduce_ns: f64,
    tail_ns: f64,
    span_ns: f64,
}

impl PhaseSum {
    fn add(&mut self, r: &ArPhaseRecord) {
        let scale = r.nanos_per_cycle().expect("calibrated record");
        self.records += 1;
        self.wait_ns += r.wait_cycles() as f64 * scale;
        self.reduce_ns += r.reduce_cycles() as f64 * scale;
        self.tail_ns += r.tail_cycles() as f64 * scale;
        self.span_ns += r.span_nanos() as f64;
    }
    fn per_step(&self, steps: usize) -> [f64; 4] {
        let steps = steps as f64;
        [
            self.wait_ns / steps / 1.0e6,
            self.reduce_ns / steps / 1.0e6,
            self.tail_ns / steps / 1.0e6,
            self.span_ns / steps / 1.0e6,
        ]
    }
}

/// NON-VACUITY, and it is the whole reason this function exists. A phase table computed from
/// records that do not close, cannot calibrate, or do not cover every join once per step describes
/// a program that did not run. Every one of these refusals has an arm below that provokes it.
fn validate(records: &[Vec<ArPhaseRecord>; 2], steps: usize) {
    for (rank, rows) in records.iter().enumerate() {
        assert_eq!(
            rows.len(),
            ARS_PER_STEP * steps,
            "rank {rank} wrote {} records for {steps} steps of {ARS_PER_STEP} joins",
            rows.len()
        );
        let mut per_site = vec![0usize; ARS_PER_STEP];
        for (index, r) in rows.iter().enumerate() {
            assert!(
                r.closes(),
                "rank {rank} record {index} does not close: {r:?}"
            );
            assert_eq!(r.rank as usize, rank, "rank {rank} record {index} rank tag");
            let scale = r
                .nanos_per_cycle()
                .unwrap_or_else(|| panic!("rank {rank} record {index} cannot calibrate: {r:?}"));
            assert!(
                (NS_PER_CYCLE.0..=NS_PER_CYCLE.1).contains(&scale),
                "rank {rank} record {index} calibrates at {scale} ns/cycle, outside the card's band"
            );
            assert!(
                (r.site as usize) < ARS_PER_STEP,
                "rank {rank} record {index} site {}",
                r.site
            );
            per_site[r.site as usize] += 1;
            // Block 0 is the only stamping block, and the barrier pairs block i with the peer's
            // block i, so a record from a grid that lost block 0 is not a record of this join.
            assert!(r.blocks >= 1, "rank {rank} record {index} blocks");
            assert!(r.elems > 0, "rank {rank} record {index} elems");
        }
        for (site, count) in per_site.iter().enumerate() {
            assert_eq!(
                *count, steps,
                "rank {rank} site {site} appears {count} times in {steps} steps"
            );
        }
        // The epoch tag is the SIGNAL BLOCK's counter, not a per-site one: one `TpEpArState` serves
        // all 86 joins, so `self_sg->seq[block]` advances once per join and a given site sees it
        // move by exactly the size of the pool between two consecutive steps. That makes this a
        // STRONGER statement than "this site ran once per step": it says the 85 other joins ran
        // between them, in this rank's own device state, so a skipped, doubled or reordered join
        // anywhere in the step shows up here. Measured 22103 -> 22189 on rank 0 site 0.
        for site in 0..ARS_PER_STEP as u32 {
            let flags: Vec<u32> = rows
                .iter()
                .filter(|r| r.site == site)
                .map(|r| r.flag)
                .collect();
            for pair in flags.windows(2) {
                assert_eq!(
                    pair[1].wrapping_sub(pair[0]),
                    ARS_PER_STEP as u32,
                    "rank {rank} site {site} epoch {} -> {}, the pool is {ARS_PER_STEP} joins",
                    pair[0],
                    pair[1]
                );
            }
        }
    }
}

/// Median of the per-record wait at one site, in cycles. Median rather than mean because one
/// scheduling outlier in 32 steps must not decide a red arm.
fn median_wait(rows: &[ArPhaseRecord], site: u32) -> f64 {
    let mut waits: Vec<u64> = rows
        .iter()
        .filter(|r| r.site == site)
        .map(|r| r.wait_cycles())
        .collect();
    waits.sort_unstable();
    assert!(!waits.is_empty(), "no records at site {site}");
    let mid = waits.len() / 2;
    if waits.len().is_multiple_of(2) {
        (waits[mid - 1] as f64 + waits[mid] as f64) / 2.0
    } else {
        waits[mid] as f64
    }
}

fn median_reduce(rows: &[ArPhaseRecord], site: u32) -> f64 {
    let mut cycles: Vec<u64> = rows
        .iter()
        .filter(|r| r.site == site)
        .map(|r| r.reduce_cycles())
        .collect();
    cycles.sort_unstable();
    assert!(!cycles.is_empty(), "no records at site {site}");
    let mid = cycles.len() / 2;
    if cycles.len().is_multiple_of(2) {
        (cycles[mid - 1] as f64 + cycles[mid] as f64) / 2.0
    } else {
        cycles[mid] as f64
    }
}

struct Walk {
    /// Held, not read: the armed graph state was restored from this prefix and the cells run for
    /// as long as the walk does, so dropping it early would retire state the graph still names.
    _prefix: DecodeState,
    graph: DecodeState,
    first: u32,
}

fn prime(gpu: &Dsv4Gpu, prompt: &[u32], cfg: Dsv4SampleCfg) -> Walk {
    let mut prefix = state(gpu);
    gpu.prefill_with_cache_chunked(&prompt[..1], &mut prefix, 1)
        .unwrap();
    for &token in &prompt[1..PRIME] {
        gpu.decode_step_device_logits(token, &mut prefix).unwrap();
    }
    let mut sampler = gpu.device_sampler().unwrap();
    let first = gpu
        .sample_device_logits(&prefix, &mut sampler, &cfg, &[], None)
        .unwrap();
    let graph = graph_state(gpu, &prefix, cfg);
    Walk {
        _prefix: prefix,
        graph,
        first,
    }
}

/// One measured window: `STEPS` replayed tokens with the instrument bracketed around them. The
/// graph is captured on the FIRST step of a fresh armed state, which is why the window is opened
/// after one warm step rather than at token zero: a capture step is not a replay step and its
/// records would describe a different program.
fn window(gpu: &Dsv4Gpu, walk: &mut Walk) -> (Vec<u32>, [Vec<ArPhaseRecord>; 2], Identity) {
    let mut carry = walk.first;
    let mut tokens = Vec::with_capacity(STEPS + 1);
    tokens.push(carry);
    carry = gpu
        .decode_sample_full_token_for_gate(carry, &mut walk.graph)
        .unwrap();
    let since = gpu.ar_phase_cursor_for_gate().unwrap();
    for _ in 0..STEPS {
        tokens.push(carry);
        carry = gpu
            .decode_sample_full_token_for_gate(carry, &mut walk.graph)
            .unwrap();
    }
    assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0, 0]);
    let records = gpu.ar_phase_records_for_gate(since).unwrap();
    (tokens, records, identity(gpu, &walk.graph))
}

fn report(gpu: &Dsv4Gpu, records: &[Vec<ArPhaseRecord>; 2], label: &str) -> [PhaseSum; 2] {
    let mut sums = [PhaseSum::default(); 2];
    for (rank, rows) in records.iter().enumerate() {
        for r in rows {
            sums[rank].add(r);
        }
        let [wait, reduce, tail, span] = sums[rank].per_step(STEPS);
        println!(
            "PHASE_POOL {label} rank={rank} arm={:?} records={} wait_ms_step={wait:.6} reduce_ms_step={reduce:.6} tail_ms_step={tail:.6} span_ms_step={span:.6}",
            gpu.ar_phase_door(),
            sums[rank].records
        );
    }
    // Per-site rows: the attention join is an even site, the expert join the odd one after it.
    for (rank, rows) in records.iter().enumerate() {
        for site in 0..ARS_PER_STEP as u32 {
            let wait = median_wait(rows, site);
            let reduce = median_reduce(rows, site);
            let scale = rows
                .iter()
                .find(|r| r.site == site)
                .and_then(ArPhaseRecord::nanos_per_cycle)
                .unwrap();
            println!(
                "PHASE_SITE {label} rank={rank} site={site} layer={} join={} wait_ns={:.1} reduce_ns={:.1} ns_per_cycle={scale:.4}",
                site / 2,
                if site.is_multiple_of(2) {
                    "attention"
                } else {
                    "expert"
                },
                wait * scale,
                reduce * scale
            );
        }
    }
    sums
}

/// Every record, raw, so the analysis is redoable from the bank without rerunning the box. Cycles
/// stay cycles and nanoseconds stay nanoseconds: the conversion is per record and belongs to
/// whoever reads the file, not to the writer.
fn records_csv(records: &[Vec<ArPhaseRecord>; 2]) -> String {
    let mut out = String::from(
        "rank,index,site,layer,join,flag,smid,blocks,elems,wait_cycles,reduce_cycles,tail_cycles,span_cycles,span_nanos\n",
    );
    for rows in records.iter() {
        for (index, r) in rows.iter().enumerate() {
            out.push_str(&format!(
                "{},{index},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                r.rank,
                r.site,
                r.site / 2,
                if r.site.is_multiple_of(2) {
                    "attention"
                } else {
                    "expert"
                },
                r.flag,
                r.smid,
                r.blocks,
                r.elems,
                r.wait_cycles(),
                r.reduce_cycles(),
                r.tail_cycles(),
                r.span_cycles(),
                r.span_nanos(),
            ));
        }
    }
    out
}

fn summary_json(
    arm: &str,
    sums: &[PhaseSum; 2],
    generated: &str,
    logits: &str,
    product_generated: Option<&str>,
) -> String {
    let ranks: Vec<String> = (0..2)
        .map(|rank| {
            let [wait, reduce, tail, span] = sums[rank].per_step(STEPS);
            format!(
                "{{\"rank\":{rank},\"records\":{},\"wait_ms_step\":{wait:.6},\"reduce_ms_step\":{reduce:.6},\"tail_ms_step\":{tail:.6},\"span_ms_step\":{span:.6}}}",
                sums[rank].records
            )
        })
        .collect();
    format!(
        "{{\"arm\":\"{arm}\",\"steps\":{STEPS},\"ars_per_step\":{ARS_PER_STEP},\
         \"generated_sha256\":\"{generated}\",\"final_logits_sha256\":\"{logits}\",\
         \"product_generated_sha256\":{},\"ranks\":[{}]}}\n",
        product_generated.map_or("null".to_string(), |s| format!("\"{s}\"")),
        ranks.join(",")
    )
}

/// Every other DSV4 gate bin pins this door `0`; this bin is the one that reads it, and it reads
/// it only after arming, which is what makes an exported `1` a refusal everywhere else.
fn door() -> String {
    std::env::var("MEMRA_DSV4_AR_PHASE").unwrap_or_else(|_| "unset".into())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() >= 5,
        "usage: dsv4_ar_phase_gate <model-dir> <source-tape> <out-dir> <cell> [args]"
    );
    for (name, value) in [
        ("MEMRA_DSV4_SAMPLER", "device"),
        ("MEMRA_DSV4_SMALL_KERNEL_DIET", "1"),
    ] {
        assert_eq!(
            std::env::var(name).as_deref(),
            Ok(value),
            "requires {name}={value}"
        );
    }
    // memra #433 door MEMRA_DSV4_TP_HEAD_SPLIT, default OFF, decide-by 2026-09-24.
    // Not this bin's arm: pin the resolved state rather than inherit whatever the caller
    // exported, and refuse an exported 1 instead of phasing all-reduces under a split head.
    assert_ne!(
        std::env::var("MEMRA_DSV4_TP_HEAD_SPLIT").as_deref(),
        Ok("1"),
        "MEMRA_DSV4_TP_HEAD_SPLIT=1 is not this bin's arm"
    );
    // SAFETY: single-threaded process start, before any engine state exists.
    unsafe { std::env::set_var("MEMRA_DSV4_TP_HEAD_SPLIT", "0") };
    // The instrument measures the DEFAULT program or it measures nothing anyone serves. Doors that
    // are ON by default must be unset or explicitly 1 here; nothing may be pinned OFF.
    for name in [
        "MEMRA_DSV4_DENSE_FAST",
        "MEMRA_DSV4_NORM_FUSE",
        "MEMRA_DSV4_NORM_FUSE2",
        "MEMRA_DSV4_NORM2_WIDE",
        "MEMRA_DSV4_DENSE_EXACT_TAIL",
        "MEMRA_DSV4_REPLAY_CADENCE",
    ] {
        assert!(
            std::env::var(name).is_err() || std::env::var(name).as_deref() == Ok("1"),
            "default ON required: {name}"
        );
    }
    assert!(memra_engine::moe_m1_graph_splitk_on());
    assert_eq!(dsv4_sampler().unwrap(), Dsv4Sampler::Device);

    // Arming BEFORE load is the whole contract: a process that does not call this and exports the
    // door refuses at load instead of running an instrumented or null collective.
    memra_engine::dsv4_gpu::arm_ar_phase_door_for_gate();

    let cfg = Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: 20260910,
    };
    let source = std::fs::read_to_string(&args[2]).expect("source tape");
    let tokenizer = Tokenizer::from_hf_dir(Path::new(&args[1])).expect("tokenizer");
    let prompt = tokenizer.encode(
        &format!("Review this inference engine source:\n\n{source}"),
        true,
    );
    assert!(prompt.len() >= PRIME);
    let prompt = &prompt[..PRIME];
    let output = PathBuf::from(&args[3]);
    std::fs::create_dir_all(&output).expect("output directory");
    println!(
        "DOOR MEMRA_DSV4_AR_PHASE={} source_sha256={:x} prompt_sha256={}",
        door(),
        Sha256::digest(source.as_bytes()),
        sha_tokens(prompt)
    );

    let cell = args[4].as_str();
    // The red arm's delay is keyed at arming time, so it is decided before the model loads too.
    let delay = if cell == "--red-delay" {
        let site: u32 = args[5].parse().expect("delay site");
        let rank: usize = args[6].parse().expect("delay rank");
        let ticks: i64 = args[7].parse().expect("delay ticks");
        Some((site, rank, ticks))
    } else {
        None
    };

    Dsv4Gpu::set_tp_ep_topology_for_gate(true);
    Dsv4Gpu::set_attention_tp_for_gate(true);
    let gpu = Box::new(
        Dsv4Gpu::load(
            Path::new(&args[1]),
            &[0, 1],
            ActQuantVariant::RefFp8Round,
            CAPACITY + 32,
        )
        .expect("pinned TP2 model"),
    );
    assert!(gpu.topology().is_tp_ep());
    assert_eq!(gpu.topology().layers, 43);
    assert!(gpu.attention_tp_geometry().is_some() && gpu.small_kernel_diet_enabled());
    gpu.set_grouped_route_validation_for_gate(false);
    gpu.set_grouped_mirror_validation_for_gate(false);
    gpu.set_grouped_gu_fuse_for_gate(true);
    gpu.set_grouped_m1_tc_for_gate(true);
    memra_engine::set_moe_f16g_gu_m1_tc_for_gate(true);
    memra_engine::set_moe_f16g_gu_half2_for_gate(true);
    memra_engine::set_moe_f16g_down_m1_half2_for_gate(true);
    gpu.set_dense_wo_a_grouped_for_gate(false);
    gpu.set_index_topk_radix_for_gate(true);

    match cell {
        // The instrument must not exist unless the door says so. This is the arm that proves the
        // OFF default is a real OFF: no allocation, no records, and an arming attempt refuses.
        "--disarmed" => {
            assert_eq!(
                gpu.ar_phase_door(),
                ArPhaseDoor::Off,
                "--disarmed requires MEMRA_DSV4_AR_PHASE unset or 0"
            );
            assert_eq!(
                gpu.ar_phase_instrument_armed_for_gate().unwrap(),
                (false, false)
            );
            let error = gpu
                .arm_ar_phase_instrument_for_gate(WINDOW, None)
                .unwrap_err();
            assert!(error.contains("requires MEMRA_DSV4_AR_PHASE"), "{error}");
            assert!(gpu.ar_phase_cursor_for_gate().is_err());
            let mut walk = prime(&gpu, prompt, cfg);
            let mut carry = walk.first;
            let mut tokens = Vec::new();
            for _ in 0..=STEPS {
                tokens.push(carry);
                carry = gpu
                    .decode_sample_full_token_for_gate(carry, &mut walk.graph)
                    .unwrap();
            }
            let ident = identity(&gpu, &walk.graph);
            assert!(gpu.ar_phase_records_for_gate([0, 0]).is_err());
            println!(
                "DISARMED instrument_allocated=false records=0 generated_sha256={} final_logits_sha256={} cache={:?} hidden={:?}",
                sha_tokens(&tokens),
                ident.0,
                ident.1,
                ident.2
            );
        }
        // Coverage, closure, calibration and the phase table. Requires the product arm: `null`
        // would report a pool that never crossed the fabric.
        "--report" => {
            assert_eq!(gpu.ar_phase_door(), ArPhaseDoor::Product);
            gpu.arm_ar_phase_instrument_for_gate(WINDOW, None).unwrap();
            assert_eq!(
                gpu.ar_phase_instrument_armed_for_gate().unwrap(),
                (true, false),
                "the product arm must allocate a product-arm instrument"
            );
            let mut walk = prime(&gpu, prompt, cfg);
            let (tokens, records, ident) = window(&gpu, &mut walk);
            validate(&records, STEPS);
            let sums = report(&gpu, &records, "product");
            std::fs::write(
                output.join("phase-product.json"),
                summary_json("product", &sums, &sha_tokens(&tokens), &ident.0, None),
            )
            .unwrap();
            std::fs::write(output.join("phase-product.csv"), records_csv(&records)).unwrap();
            println!(
                "REPORT arm=product steps={STEPS} generated_sha256={} final_logits_sha256={} cache={:?} hidden={:?}",
                sha_tokens(&tokens),
                ident.0,
                ident.1,
                ident.2
            );
        }
        // RED ARM. A known delay is injected on one rank at one site. The instrument is believed
        // only if that delay appears as the OTHER rank's start-barrier wait at that same site, does
        // not appear in its reduce phase, and does not appear at the neighbouring sites. Run it
        // with ticks=0 for the negative control: the same comparison must then NOT fire.
        "--red-delay" => {
            assert_eq!(gpu.ar_phase_door(), ArPhaseDoor::Product);
            let (site, rank, ticks) = delay.unwrap();
            let peer = 1 - rank;
            // A zero-tick request is the negative control and arms no delay at all.
            gpu.arm_ar_phase_instrument_for_gate(
                WINDOW,
                (ticks > 0).then_some((site, rank, ticks)),
            )
            .unwrap();
            let mut walk = prime(&gpu, prompt, cfg);
            let (_, records, _) = window(&gpu, &mut walk);
            validate(&records, STEPS);
            let peer_wait = median_wait(&records[peer], site);
            let peer_reduce = median_reduce(&records[peer], site);
            // Neighbours in the same step, one join either side, carry the delay only if the
            // instrument is attributing to the wrong record.
            let neighbours: Vec<u32> = [site.wrapping_sub(1), site + 1]
                .into_iter()
                .filter(|s| (*s as usize) < ARS_PER_STEP)
                .collect();
            let neighbour_wait: f64 = neighbours
                .iter()
                .map(|s| median_wait(&records[peer], *s))
                .sum::<f64>()
                / neighbours.len() as f64;
            let ticks = ticks as f64;
            println!(
                "RED_DELAY site={site} delayed_rank={rank} ticks={ticks:.0} peer_wait_cycles={peer_wait:.1} peer_reduce_cycles={peer_reduce:.1} peer_neighbour_wait_cycles={neighbour_wait:.1}"
            );
            if ticks > 0.0 {
                // 0.6 of the injected delay, because the peer's own arrival skew subtracts from
                // what it waits: the claim is that the delay is VISIBLE and attributed to the
                // wait phase, not that the two numbers match to the cycle.
                assert!(
                    peer_wait >= 0.6 * ticks,
                    "instrument lies: {ticks} injected cycles produced {peer_wait} cycles of peer wait at site {site}"
                );
                assert!(
                    peer_reduce < 0.2 * ticks,
                    "instrument lies: the injected delay landed in the reduce phase ({peer_reduce} cycles)"
                );
                assert!(
                    neighbour_wait < 0.3 * ticks,
                    "instrument lies: the injected delay is visible at neighbouring sites ({neighbour_wait} cycles)"
                );
                println!("RED_DELAY_VERDICT injected_delay_is_visible_in_the_wait_phase=true");
            } else {
                // NEGATIVE CONTROL. Nothing was injected, so the named site must not stand out
                // against the rest of the pool. The bound is the pool's own median rather than a
                // constant, because "this site waits like every other site" is the claim, and a
                // constant would pass on a card whose whole pool moved.
                let mut all: Vec<f64> = (0..ARS_PER_STEP as u32)
                    .map(|s| median_wait(&records[peer], s))
                    .collect();
                all.sort_by(f64::total_cmp);
                let pool_median = all[all.len() / 2];
                assert!(
                    peer_wait <= 4.0 * pool_median.max(1.0),
                    "negative control: an undelayed site waits {peer_wait} cycles against a pool median of {pool_median}"
                );
                println!(
                    "RED_DELAY_VERDICT negative_control_no_injected_delay=true pool_median_wait_cycles={pool_median:.1}"
                );
            }
        }
        // NULL ARM. Same launch shape, same barriers, same epochs, no peer operand. Its tokens MUST
        // differ from the product arm's: an identical stream would mean the arm did not actually
        // stop reading the peer, and every transport bound taken from it would be a fiction.
        "--null" => {
            assert_eq!(gpu.ar_phase_door(), ArPhaseDoor::Null);
            let product_tokens = args
                .get(5)
                .expect("--null needs the product arm's token sha");
            gpu.arm_ar_phase_instrument_for_gate(WINDOW, None).unwrap();
            assert_eq!(
                gpu.ar_phase_instrument_armed_for_gate().unwrap(),
                (true, true),
                "the null cell must allocate a null-arm instrument"
            );
            let mut walk = prime(&gpu, prompt, cfg);
            let (tokens, records, ident) = window(&gpu, &mut walk);
            validate(&records, STEPS);
            let generated = sha_tokens(&tokens);
            assert_ne!(
                &generated, product_tokens,
                "null arm produced the product arm's tokens: it did not remove the peer read"
            );
            let sums = report(&gpu, &records, "null");
            std::fs::write(
                output.join("phase-null.json"),
                summary_json(
                    "null",
                    &sums,
                    &generated,
                    &ident.0,
                    Some(product_tokens.as_str()),
                ),
            )
            .unwrap();
            std::fs::write(output.join("phase-null.csv"), records_csv(&records)).unwrap();
            println!("NULL arm=null tokens_differ=true generated_sha256={generated}");
        }
        // IDENTITY AND OVERHEAD. Run once per arm from the controller with the door unset and with
        // it at 1, then diff: the instrument may not move a single bit, and the wall cost of the
        // stamps is what the interleaved rows report. Identity is checked on the printed digests
        // rather than in-process, because one process holds one arm.
        "--identity" => {
            if gpu.ar_phase_door() != ArPhaseDoor::Off {
                gpu.arm_ar_phase_instrument_for_gate(WINDOW, None).unwrap();
            }
            let mut walk = prime(&gpu, prompt, cfg);
            let mut carry = walk.first;
            let mut tokens = Vec::new();
            let start = std::time::Instant::now();
            for _ in 0..=STEPS {
                tokens.push(carry);
                carry = gpu
                    .decode_sample_full_token_for_gate(carry, &mut walk.graph)
                    .unwrap();
            }
            let elapsed = start.elapsed().as_secs_f64();
            let ident = identity(&gpu, &walk.graph);
            assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0, 0]);
            println!(
                "IDENTITY door={} arm={:?} steps={} tok_s={:.6} generated_sha256={} final_logits_sha256={} cache={:?} hidden={:?}",
                door(),
                gpu.ar_phase_door(),
                tokens.len(),
                tokens.len() as f64 / elapsed,
                sha_tokens(&tokens),
                ident.0,
                ident.1,
                ident.2
            );
        }
        other => panic!("unknown cell {other}"),
    }
}
