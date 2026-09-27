//! WP-B day 37, clause A2 (`research/spill-b-20260919/DAY37.md` 1.6): the G1 grow series of
//! on-demand VMM K/V planes (`MEMRA_KV_ALLOCATOR=vmm`'s mechanism, `memra_kv::with_on_demand_kv`).
//!
//! One process runs the gate's tokenwise program twice at the same context: first on pooled
//! planes, then on on-demand planes that back rows only as the position grows. The pooled arm
//! goes first so the device pool is at the program's high-water before the on-demand arm's
//! baseline is read. Every grow of the on-demand arm is read on the driver: a stream sync, the
//! free bytes, the ensure, a stream sync, the free bytes again. No other allocation runs between
//! the two reads (grows run inline here; the mapper thread is not started).
//!
//! PASS needs every term: each grow moves driver free by exactly its extents' bytes; every
//! on-demand plane crosses at least `MIN_CROSSINGS` granule boundaries after construction; after
//! the on-demand cache drops and its graveyard is reaped, driver free equals the baseline read
//! before the cache was built (drift 0); and the tokens, the per-step logits rows, and the prefix
//! and final state manifests equal the pooled arm's.
use super::{Result, capture, f32_bytes, hash, u32_bytes};
use memra_engine::{Engine, forward::argmax, hybrid::HybridModel};
use memra_kv::Cache;
use std::{collections::BTreeMap, fs, io::Write, path::Path};

/// DAY37 1.6 A2: at least 5 boundary crossings of every plane in one process.
const MIN_CROSSINGS: usize = 5;
/// The rows backed past the position before each step (`memra_kv::ON_DEMAND_MIN_AHEAD_ROWS`).
const AHEAD: usize = memra_kv::ON_DEMAND_MIN_AHEAD_ROWS;

struct Row {
    pos: usize,
    planes: Vec<String>,
    extent_bytes: usize,
    free_before: usize,
    free_after: usize,
}

struct Arm {
    tokens: Vec<u32>,
    logit_rows: Vec<String>,
    prefix: String,
    final_state: String,
}

fn free_bytes(e: &Engine) -> Result<usize> {
    e.stream().synchronize()?;
    Ok(e.ctx().mem_get_info()?.0)
}

/// Back the rows the next step needs; when that grows anything, read the driver around it.
fn ensure_read(e: &Engine, cache: &mut Cache, series: &mut Vec<Row>) -> Result<()> {
    let rows = (cache.pos + 1 + AHEAD).min(cache.max_ctx);
    if cache.kv_mapped_rows().is_none_or(|m| m >= rows) {
        return Ok(());
    }
    let free_before = free_bytes(e)?;
    let events = cache.ensure_kv_rows(rows)?;
    let free_after = free_bytes(e)?;
    if events.is_empty() {
        return Err("on-demand cache reported short rows but grew nothing".into());
    }
    series.push(Row {
        pos: cache.pos,
        planes: events.iter().map(|(label, _)| label.clone()).collect(),
        extent_bytes: events.iter().map(|(_, ev)| ev.bytes).sum(),
        free_before,
        free_after,
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_arm(
    e: &Engine,
    model: &HybridModel,
    prompt: &[u32],
    generate: usize,
    cache: &mut Cache,
    out: &Path,
    name: &str,
    mut series: Option<&mut Vec<Row>>,
) -> Result<Arm> {
    let mut last = None;
    for (i, &token) in prompt.iter().enumerate() {
        if let Some(s) = series.as_deref_mut() {
            ensure_read(e, cache, s)?;
        }
        last = Some(model.decode_step_h(e, token, cache)?);
        if (i + 1).is_multiple_of(4096) {
            eprintln!("grow {name} prompt committed={}", cache.pos);
        }
    }
    let (mut logits, mut hidden) = last.ok_or("empty prompt")?;
    let prefix = capture(
        e,
        cache,
        &model.plan,
        &logits,
        &e.dtoh(&hidden)?,
        out,
        &format!("{name}-prefix"),
    )?;
    let mut tokens = Vec::with_capacity(generate);
    let mut logit_rows = Vec::with_capacity(generate + 1);
    for _ in 0..generate {
        if logits.len() != model.cfg.n_vocab as usize || logits.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite or wrong-sized logits".into());
        }
        logit_rows.push(format!("{}\t{}", cache.pos, hash(&f32_bytes(&logits))));
        let token = argmax(&logits) as u32;
        tokens.push(token);
        if let Some(s) = series.as_deref_mut() {
            ensure_read(e, cache, s)?;
        }
        (logits, hidden) = model.decode_step_h(e, token, cache)?;
    }
    logit_rows.push(format!("{}\t{}", cache.pos, hash(&f32_bytes(&logits))));
    let final_state = capture(
        e,
        cache,
        &model.plan,
        &logits,
        &e.dtoh(&hidden)?,
        out,
        &format!("{name}-final"),
    )?;
    Ok(Arm {
        tokens,
        logit_rows,
        prefix,
        final_state,
    })
}

/// The whole case: returns the status line the collector reads.
pub fn run(
    e: &Engine,
    model: &HybridModel,
    prompt: &[u32],
    context: usize,
    generate: usize,
    out: &Path,
) -> Result<String> {
    // Exact driver reads need the owner to make every grow; no mapper thread in this case.
    memra_kv::vmm_set_grow_placement(memra_kv::VmmGrowPlacement::Inline);
    memra_kv::vmm_set_faults(None);

    // Arm 1, pooled: the same program, which also brings the device pool to its high-water.
    let pooled = {
        let mut cache = memra_engine::pp::new_cache(e, &model.cfg, context)?;
        let arm = run_arm(e, model, prompt, generate, &mut cache, out, "pooled", None)?;
        drop(cache);
        arm
    };

    // Arm 2, on demand: the baseline is read with the pooled cache gone.
    let baseline_free = free_bytes(e)?;
    let (built, scoped) = memra_kv::with_on_demand_kv(1 + AHEAD, || {
        memra_engine::pp::new_cache(e, &model.cfg, context)
    });
    let mut cache = built?;
    if cache.on_demand_planes() != scoped || scoped == 0 {
        return Err(format!(
            "on-demand cache reaches {} planes, the scope allocated {scoped}",
            cache.on_demand_planes()
        )
        .into());
    }
    let (built_mapped, reserved) = cache.kv_on_demand_bytes();
    let after_build_free = free_bytes(e)?;
    let mut series = Vec::new();
    let ondemand = run_arm(
        e,
        model,
        prompt,
        generate,
        &mut cache,
        out,
        "ondemand",
        Some(&mut series),
    )?;
    let (final_mapped, _) = cache.kv_on_demand_bytes();
    let planes = cache.on_demand_planes();
    let end_pos = cache.pos;
    drop(cache);
    let (reaped, pending, pending_bytes) = memra_kv::vmm_reap_graveyard_blocking()?;
    let end_free = free_bytes(e)?;

    // The readings.
    let mut crossings: BTreeMap<String, usize> = BTreeMap::new();
    let mut unequal = 0usize;
    let mut tsv = String::from(
        "pos\tplanes\textent_bytes\tfree_before\tfree_after\tdelta\tequal\tplane_labels\n",
    );
    for r in &series {
        let delta = r.free_before as i128 - r.free_after as i128;
        let equal = delta == r.extent_bytes as i128;
        unequal += usize::from(!equal);
        for p in &r.planes {
            *crossings.entry(p.clone()).or_default() += 1;
        }
        tsv.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{delta}\t{equal}\t{}\n",
            r.pos,
            r.planes.len(),
            r.extent_bytes,
            r.free_before,
            r.free_after,
            r.planes.join(",")
        ));
    }
    fs::write(out.join("grow-series.tsv"), tsv)?;
    let min_crossings = crossings.values().copied().min().unwrap_or(0);
    let every_plane_crossed = crossings.len() == planes;
    let drift = baseline_free as i128 - end_free as i128;
    let tokens_equal = pooled.tokens == ondemand.tokens;
    let logits_equal = pooled.logit_rows == ondemand.logit_rows;
    let prefix_equal = pooled.prefix == ondemand.prefix;
    let final_equal = pooled.final_state == ondemand.final_state;
    let mut rows = fs::File::create(out.join("logits.tsv"))?;
    writeln!(
        rows,
        "committed\tpooled_logits_sha256\tondemand_logits_sha256"
    )?;
    for (a, b) in pooled.logit_rows.iter().zip(&ondemand.logit_rows) {
        let b_hash = b.split('\t').nth(1).unwrap_or("");
        writeln!(rows, "{a}\t{b_hash}")?;
    }
    fs::write(out.join("tokens.u32le"), u32_bytes(&ondemand.tokens))?;
    let pass = unequal == 0
        && every_plane_crossed
        && min_crossings >= MIN_CROSSINGS
        && drift == 0
        && pending == 0
        && tokens_equal
        && logits_equal
        && prefix_equal
        && final_equal;
    let status = format!(
        "GROW-G1 {} (grows={} unequal={unequal} planes={planes} planes_crossed={} \
         min_crossings_per_plane={min_crossings} rule>={MIN_CROSSINGS} drift={drift} \
         tokens_equal={tokens_equal} logits_equal={logits_equal} prefix_state_equal={prefix_equal} \
         final_state_equal={final_equal}) committed={end_pos} generated={generate}",
        if pass { "PASS" } else { "FAIL" },
        series.len(),
        crossings.len(),
    );
    fs::write(
        out.join("GROW.txt"),
        format!(
            "{status}\nbaseline_free={baseline_free}\nafter_build_free={after_build_free}\n\
             built_mapped={built_mapped}\nreserved={reserved}\nfinal_mapped={final_mapped}\n\
             end_free={end_free}\nreaped={reaped}\npending_graves={pending}\npending_bytes={pending_bytes}\n\
             pooled_prefix_manifest={}\nondemand_prefix_manifest={}\npooled_final_manifest={}\n\
             ondemand_final_manifest={}\npooled_tokens_sha256={}\nondemand_tokens_sha256={}\n\
             crossings_per_plane={crossings:?}\n",
            pooled.prefix,
            ondemand.prefix,
            pooled.final_state,
            ondemand.final_state,
            hash(&u32_bytes(&pooled.tokens)),
            hash(&u32_bytes(&ondemand.tokens)),
        ),
    )?;
    Ok(status)
}
