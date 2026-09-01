//! MoE ROUTED-UNION cost probe (`moeu` lane, mtp13) — prices the union-of-experts gather
//! BEFORE any kernel exists, on the shipped kernels, with no checkpoint load.
//!
//! The lever: at a K=5 verify chunk (t=6) the qwen4_exp MoE dispatches `t*selected` = 60
//! (token, expert) slots through ONE `qmatvec_nvfp4_modelopt_sel_gu_silu_f32` launch and
//! ONE `qmatvec_nvfp4_modelopt_sel_f32_v3` launch, both with `grid.y = slots`. Each block
//! addresses its expert as `sel[slot]`, so an expert named by two columns has its NVFP4
//! bytes read TWICE — the kernel's own doc says "the weight banks are read once per
//! selected slot either way — the launch count is what drops". A union gather would read
//! each DISTINCT expert once per chunk.
//!
//! Why a probe and not a kernel: the union gather changes exactly one quantity — distinct
//! experts touched — leaving per-slot arithmetic, slot count and launch geometry alone. So
//! run the shipped kernels at a FIXED slot count and sweep only the number of distinct
//! experts those slots name. The `slots=60, union=U` row IS the idealised union gather's
//! cost, measured, with no rewrite. Per LAW:price-the-dispatch-first.
//!
//! Read the output as: if `union=60` and `union=10` cost the same at `slots=60`, the
//! duplicated bytes are not what the section is paying for, and the lever has no surface
//! whatever the real routed union sizes are — this card's L2 is 128 MiB and a whole chunk's
//! routed gate+up working set is ~105 MiB, so the hardware may already deduplicate what
//! the kernel re-reads.
//!
//! Env: MEMRA_MOEU_T (verify columns, default 6), MEMRA_MOEU_REPS (default 20),
//! MEMRA_MOEU_EXPERTS/HIDDEN/FF/SELECTED override the qwen4_exp serving geometry.
//!
//! TIMING ARM: take `flock -x /tmp/q48fn-measure.lock` around the WHOLE invocation, and
//! never quote a row measured on the rig (rig = exactness only).
use memra_engine::Engine;
use memra_engine::qwen4exp_gpu::moe_union_cost_probe;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // qwen4_exp serving geometry: experts fused 3D gate_up [512, 1280, 2560] (ff=640
    // gate + 640 up), down [512, 2560, 640]. SEMANTICS.md "MoE (L510-527)".
    let experts = env_usize("MEMRA_MOEU_EXPERTS", 512);
    let hidden = env_usize("MEMRA_MOEU_HIDDEN", 2560);
    let ff = env_usize("MEMRA_MOEU_FF", 640);
    let selected = env_usize("MEMRA_MOEU_SELECTED", 10);
    let t = env_usize("MEMRA_MOEU_T", 6);
    let reps = env_usize("MEMRA_MOEU_REPS", 20);

    let e = Engine::new(0)?;
    // Per-expert NVFP4 bytes, from the kernels' own row arithmetic: a row is in_f/2 code
    // bytes + in_f/16 ue4m3 scale bytes.
    let gu_expert_b = 2 * ff * (hidden / 2 + hidden / 16);
    let dn_expert_b = hidden * (ff / 2 + ff / 16);
    println!("# moe_union_probe (SYNTHETIC banks, no checkpoint — traffic/latency only)");
    println!(
        "# geometry experts={experts} hidden={hidden} ff={ff} selected={selected} t={t} reps={reps}"
    );
    println!(
        "# per-expert bytes gate+up={gu_expert_b} ({:.3} MiB) down={dn_expert_b} ({:.3} MiB) \
         total={} ({:.3} MiB)",
        gu_expert_b as f64 / 1048576.0,
        dn_expert_b as f64 / 1048576.0,
        gu_expert_b + dn_expert_b,
        (gu_expert_b + dn_expert_b) as f64 / 1048576.0
    );
    let rows = moe_union_cost_probe(&e, experts, hidden, ff, selected, t, reps)?;
    println!(
        "t\tslots\tunion\tgu_us\tdown_us\tsection_us\tgu_spread\tdn_spread\t\
         gu_BW_vs_union\tgu_BW_vs_slots\tvs_t6_full"
    );
    // Reference = the t == max, union == slots row (today's shape).
    let full = rows
        .iter()
        .filter(|r| r.t == t && r.union_size == r.slots)
        .map(|r| r.gu_us + r.down_us)
        .next_back();
    for r in &rows {
        let section = r.gu_us + r.down_us;
        let bw_union = (r.union_size * gu_expert_b) as f64 / r.gu_us / 1e6;
        let bw_slots = (r.slots * gu_expert_b) as f64 / r.gu_us / 1e6;
        let rel = match full {
            Some(f) if section > 0.0 => format!("{:.4}x", f / section),
            _ => "-".to_string(),
        };
        println!(
            "{}\t{}\t{}\t{:.1}\t{:.1}\t{:.1}\t{:.4}\t{:.4}\t{:.3}\t{:.3}\t{}",
            r.t,
            r.slots,
            r.union_size,
            r.gu_us,
            r.down_us,
            section,
            r.gu_spread_rel,
            r.down_spread_rel,
            bw_union,
            bw_slots,
            rel
        );
    }
    println!(
        "# arms INTERLEAVED rep by rep (LAW:interleaved-ab), rep 0 a warmed throwaway for \
         every arm; gu_spread/dn_spread = (max-min)/median over the arm's reps."
    );
    println!(
        "# gu_BW_vs_union = TB/s if only the distinct experts' bytes reach DRAM; \
         gu_BW_vs_slots = TB/s if every slot's bytes do. The true DRAM rate cannot exceed \
         the card's peak, so whichever column stays under peak is the traffic that is real."
    );
    println!(
        "# READ vs_t6_full AGAINST THE SPREAD COLUMNS: if an arm's gain is smaller than its \
         own spread, the lever is unmeasurable at that union, not merely small."
    );
    Ok(())
}
