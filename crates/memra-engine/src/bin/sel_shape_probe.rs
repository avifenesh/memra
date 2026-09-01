//! Sel-matvec SHAPE cost probe (`downsel` lane, mtp14) — prices the sub-warp pair-group
//! lane partitions on the serving geometry, with no checkpoint.
//!
//! The defect. `qmatvec_nvfp4_modelopt_sel_f32_v3` and its gufuse twin partition the pair
//! loop over all 32 lanes (`for p = lane; p < pairs; p += 32`, `pairs = in_f/32`). At the
//! qwen4_exp MoE geometry the DOWN launch has `in_f = expert ff = 640`, so `pairs = 20`:
//! lanes 20-31 hold no pair for the whole kernel and each active lane runs exactly ONE
//! iteration before a full 5-step shfl tree. The gate+up launch (`in_f = hidden = 2560`,
//! `pairs = 80`) instead pays 3 warp iterations for 2.5 iterations of work. 62.5% and 83.3%
//! lane occupancy respectively, in a section `moe_union_probe` already showed is per-SLOT-
//! WORK bound rather than weight-traffic bound (KNEE:q4e-sel-slots-not-bytes).
//!
//! The arms are lane partitions: `g` lanes cooperate on one output row, the warp carries
//! `32/g` groups of `rows` rows each, and the reduce is log2(g) shfl steps inside the group.
//! Rows per warp is `(32/g)*rows`, so a shape trades lane occupancy against WARP COUNT —
//! which is why the table prints grid.x next to the time. `g=32, rows=4` is the shipped
//! program exactly.
//!
//! READ THE `dn:32:4+gu:32:4` ROW FIRST, and read every other arm AGAINST IT. That arm runs
//! the shipped program through the new kernel at the shape where the two coincide —
//! bit-identical output, gated by `gate_nvfp4_sel_group`. It was put in as a noise floor and
//! it turned out not to be one: on the rig it reproducibly measures a few percent FASTER
//! than `off` across independent passes, because the source restructure (indexed arrays and
//! unrolled row loops instead of four named accumulators) lets nvcc schedule differently for
//! the same bits. So `arm / off` MIXES the source rewrite with the shape, and only
//! `arm / control` isolates the shape. Both columns are printed for that reason.
//!
//! Arms are interleaved rep by rep (TRAP:monotone-sweep-inflates-the-lever) and rep 0 is a
//! warmed throwaway. Per-arm spread is printed and never averaged away: on a throttling card
//! the spreads run 13-23% while the interleaved arm RATIOS reproduce to under 0.5% across
//! independent passes, which is exactly what interleaving buys and why the ratio is the
//! statistic.
//!
//! Env: MEMRA_SELSHAPE_T (verify columns, default 6 — the K=5 chunk), MEMRA_SELSHAPE_REPS
//! (default 25), MEMRA_SELSHAPE_ARMS (comma-separated `selgroup` specs),
//! MEMRA_SELSHAPE_EXPERTS/HIDDEN/FF/SELECTED override the serving geometry.
//!
//! TIMING ARM: take `flock -x /tmp/q48fn-measure.lock` around the WHOLE invocation, and
//! never quote a row measured on the rig (LAW:rig-gpu-exactness-only).
use memra_engine::Engine;
use memra_engine::qwen4exp_gpu::sel_shape_cost_probe;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // qwen4_exp serving geometry: experts fused 3D gate_up [512, 1280, 2560] (ff=640 gate +
    // 640 up), down [512, 2560, 640], top-10 (SEMANTICS.md "MoE (L510-527)", cross-checked
    // against the artifact census in ARCH.md).
    let experts = env_usize("MEMRA_SELSHAPE_EXPERTS", 512);
    let hidden = env_usize("MEMRA_SELSHAPE_HIDDEN", 2560);
    let ff = env_usize("MEMRA_SELSHAPE_FF", 640);
    let selected = env_usize("MEMRA_SELSHAPE_SELECTED", 10);
    let t = env_usize("MEMRA_SELSHAPE_T", 6);
    let reps = env_usize("MEMRA_SELSHAPE_REPS", 25);
    let arms: Vec<String> = std::env::var("MEMRA_SELSHAPE_ARMS")
        .unwrap_or_else(|_| {
            // `off` = the shipped kernels. `dn:32:4+gu:32:4` = the NEW kernel at the
            // shipped shape (bit-identical) — the control that separates the source
            // rewrite's scheduling from the shape effect; read every other arm against IT,
            // not against `off`. `auto` = rows-per-lane 4 with the lanes filled
            // (down g4r4, gate+up g16r4), the measured winner at t=6 and t=1. The last two
            // are the rows-per-lane-spending arms kept in the ladder because they are the
            // ones that LOSE, and a ladder without its losers cannot show the rule.
            "off,dn:32:4+gu:32:4,auto,dn:4:4+gu:off,dn:off+gu:16:4,dn:4:2+gu:16:2,\
             dn:8:1+gu:16:2"
                .replace(char::is_whitespace, "")
        })
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();

    let e = Engine::new(0)?;
    let gu_expert_b = 2 * ff * (hidden / 2 + hidden / 16);
    let dn_expert_b = hidden * (ff / 2 + ff / 16);
    println!("# sel_shape_probe (SYNTHETIC banks, no checkpoint — latency only, no oracle)");
    println!(
        "# geometry experts={experts} hidden={hidden} ff={ff} selected={selected} t={t} \
         reps={reps} slots={}",
        t * selected
    );
    println!(
        "# pairs: gate+up in_f={hidden} -> {} (32-lane loop: {:.1}% occupancy); \
         down in_f={ff} -> {} ({:.1}% occupancy)",
        hidden / 32,
        100.0 * (hidden / 32) as f64 / (32.0 * ((hidden / 32) as f64 / 32.0).ceil()),
        ff / 32,
        100.0 * (ff / 32) as f64 / (32.0 * ((ff / 32) as f64 / 32.0).ceil())
    );
    println!(
        "# per-expert bytes gate+up={gu_expert_b} down={dn_expert_b} \
         (chunk gate+up working set {:.1} MiB)",
        (t * selected * gu_expert_b) as f64 / 1048576.0
    );
    let rows = sel_shape_cost_probe(&e, experts, hidden, ff, selected, t, reps, &arms)?;
    println!(
        "arm\tt\tslots\tgu_shape\tgu_gridx\tdn_shape\tdn_gridx\tgu_us\tdown_us\tsection_us\t\
         gu_spread\tdn_spread\tsection_vs_off\tsection_vs_ctl\tgu_vs_ctl\tdn_vs_ctl"
    );
    let pick = |arm: &str| rows.iter().find(|r| r.arm == arm);
    let base = pick("off").map(|r| r.gu_us + r.down_us);
    // The control arm, if the ladder carries it. `*_vs_ctl` is the SHAPE effect with the
    // source rewrite divided out; `section_vs_off` is what the caller would actually ship.
    let ctl = pick("dn:32:4+gu:32:4");
    for r in &rows {
        let section = r.gu_us + r.down_us;
        let vs = base.map_or(f64::NAN, |b| b / section);
        let vs_ctl = ctl.map_or(f64::NAN, |c| (c.gu_us + c.down_us) / section);
        let gu_ctl = ctl.map_or(f64::NAN, |c| c.gu_us / r.gu_us);
        let dn_ctl = ctl.map_or(f64::NAN, |c| c.down_us / r.down_us);
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.1}\t{:.1}\t{:.1}\t{:.1}%\t{:.1}%\t{:.4}\t\
             {:.4}\t{:.4}\t{:.4}",
            r.arm,
            r.t,
            r.slots,
            r.gu_shape,
            r.gu_grid_x,
            r.dn_shape,
            r.dn_grid_x,
            r.gu_us,
            r.down_us,
            section,
            r.gu_spread_rel * 100.0,
            r.down_spread_rel * 100.0,
            vs,
            vs_ctl,
            gu_ctl,
            dn_ctl
        );
    }
    Ok(())
}
