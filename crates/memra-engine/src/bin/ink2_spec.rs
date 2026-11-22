//! Inkling 2 Mini speculative decoding on one GPU: the DSpark drafter (`inkling2_dspark`)
//! proposing, the trunk's route verifying (`inkling2_spec`).
//!
//! usage: ink2-spec --tiny-gate
//!          the GPU drafter on the tiny fixture against the pinned f64 torch-port oracle
//!        ink2-spec CKPT_DIR --prompts FILE.jsonl [--gen N] [--ks 1,2,...] [--check-plain]
//!                  [--sample T --seed S] [--crosscheck] [--class declared|float]
//!                  [--chunk N] [--context N] [--out FILE] [--device D] [--limit N]
//!   --prompts      JSON lines {"id", "kind", "ids": [token ids]} (a rendered chat template)
//!   --check-plain  greedy decode without drafting first; every --ks run must equal it token for
//!                  token (the run-spec self-consistency gate); exit 1 on a mismatch
//!   --sample T     probabilistic drafting at temperature T (draft samples, ratio test)
//!   --crosscheck   the first prompt's first block against the f64 reference drafter on CPU
//!   --red-no-restore  the gate's red arm: a partial accept keeps the rejected rows' history
//!   --red-tap-slot I:S  red arm: drafter tap I reads the input of slot S (the plan's: 5, 29, 47)
//!   --bench N      component costs (median of N): plain step, verify per row count, draft per k,
//!                  prefix commit

use memra_engine::inkling2_dspark::{CkptDraft, DraftGpu, Shared};
use memra_engine::inkling2_gpu::{Ckpt, Dev, Head, Route};
use memra_engine::inkling2_spec::{
    Sampling, SpecArms, bench_costs, generate_arm, plain_greedy, plain_greedy_timed, tap_slots,
};
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::{ExpertActivationMode, ModelPlan};
use memra_reference::inkling2_drafter as refd;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    if let Err(e) = run() {
        eprintln!("ink2-spec: {e}");
        std::process::exit(1);
    }
}

#[derive(Default)]
struct Args {
    ckpt: Option<PathBuf>,
    tiny: bool,
    prompts: Option<PathBuf>,
    gen_tokens: usize,
    ks: Vec<usize>,
    check_plain: bool,
    sample: Option<f64>,
    seed: u64,
    crosscheck: bool,
    per_position: Option<PathBuf>,
    bench: usize,
    profile_rows: usize,
    class: String,
    /// The drafter's output projection: `shared` (the class's unembed) or `bf16` (the
    /// checkpoint's own BF16 rows, loaded beside an NVFP4 unembed).
    draft_head: String,
    chunk: usize,
    context: usize,
    out: Option<PathBuf>,
    device: usize,
    limit: usize,
    red_no_restore: bool,
    red_tap: Option<(usize, u32)>,
}

fn parse() -> Result<Args, String> {
    let mut a = Args {
        gen_tokens: 128,
        ks: vec![7],
        class: "declared".into(),
        draft_head: "shared".into(),
        chunk: 512,
        context: 4096,
        seed: 1,
        limit: usize::MAX,
        ..Args::default()
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().ok_or(format!("{arg} needs a value"));
        let num = |v: String| v.parse::<usize>().map_err(|e| format!("{e}"));
        match arg.as_str() {
            "--tiny-gate" => a.tiny = true,
            "--prompts" => a.prompts = Some(PathBuf::from(val()?)),
            "--gen" => a.gen_tokens = num(val()?)?,
            "--ks" => {
                a.ks = val()?
                    .split(',')
                    .map(|v| num(v.into()))
                    .collect::<Result<_, _>>()?
            }
            "--check-plain" => a.check_plain = true,
            "--sample" => a.sample = Some(val()?.parse().map_err(|e| format!("{e}"))?),
            "--seed" => a.seed = num(val()?)? as u64,
            "--crosscheck" => a.crosscheck = true,
            "--per-position" => a.per_position = Some(PathBuf::from(val()?)),
            "--bench" => a.bench = num(val()?)?,
            "--profile-rows" => a.profile_rows = num(val()?)?,
            "--red-no-restore" => a.red_no_restore = true,
            "--red-tap-slot" => {
                let v = val()?;
                let (i, s) = v.split_once(':').ok_or("--red-tap-slot INDEX:SLOT")?;
                a.red_tap = Some((num(i.into())?, num(s.into())? as u32));
            }
            "--class" => a.class = val()?,
            "--draft-head" => a.draft_head = val()?,
            "--chunk" => a.chunk = num(val()?)?,
            "--context" => a.context = num(val()?)?,
            "--out" => a.out = Some(PathBuf::from(val()?)),
            "--device" => a.device = num(val()?)?,
            "--limit" => a.limit = num(val()?)?,
            s if !s.starts_with("--") && a.ckpt.is_none() => a.ckpt = Some(PathBuf::from(s)),
            _ => return Err(format!("unknown argument {arg}")),
        }
    }
    Ok(a)
}

fn run() -> Result<(), String> {
    let a = parse()?;
    if a.tiny {
        return tiny_gate(&a);
    }
    let ckpt = a
        .ckpt
        .clone()
        .ok_or("usage: ink2-spec CKPT_DIR --prompts FILE | --tiny-gate")?;
    let read =
        |name: &str| std::fs::read_to_string(ckpt.join(name)).map_err(|e| format!("{name}: {e}"));
    let mut config = ModelConfig::from_hf(&HfConfig::try_parse(&read("config.json")?)?);
    config.attach_quant_groups(&read("hf_quant_config.json")?)?;
    let pack = memra_gguf::model_packs::for_config(&config).ok_or("no pack matches this config")?;
    let default = pack
        .compile_plan(&config)
        .map_err(|e| format!("plan: {e:?}"))?;
    let plan: ModelPlan = if a.class == "nvfp4-rig" {
        // the family's default class: its overlay (the checkpoint's sibling) binds by hash, and
        // the drafter shares its NVFP4 unembed
        memra_gguf::model_packs::inkling2::nvfp4_rig_plan(&config)
            .map_err(|e| format!("plan: {e:?}"))?
    } else if a.class == "float" {
        let n = default
            .inkling2
            .as_ref()
            .ok_or("not inkling2")?
            .numerics
            .experts
            .len();
        memra_gguf::model_packs::inkling2::plan_with_expert_activation(
            &config,
            &vec![ExpertActivationMode::Float; n],
        )
        .map_err(|e| format!("plan: {e:?}"))?
    } else {
        default
    };
    let g = refd::DraftGeometry::of_plan(&plan)?;
    let stop: Vec<u32> = {
        let generation: serde_json::Value = serde_json::from_str(&read("generation_config.json")?)
            .map_err(|e| format!("generation_config.json: {e}"))?;
        match &generation["eos_token_id"] {
            serde_json::Value::Number(n) => vec![n.as_u64().unwrap_or(0) as u32],
            serde_json::Value::Array(v) => v
                .iter()
                .filter_map(|x| x.as_u64())
                .map(|x| x as u32)
                .collect(),
            _ => Vec::new(),
        }
    };
    let prompts_text = std::fs::read_to_string(a.prompts.as_ref().ok_or("--prompts FILE")?)
        .map_err(|e| format!("prompts: {e}"))?;
    let mut prompts = Vec::new();
    for line in prompts_text.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value =
            serde_json::from_str(line).map_err(|e| format!("prompt line: {e}"))?;
        let ids: Vec<u32> = v["ids"]
            .as_array()
            .ok_or("a prompt without ids")?
            .iter()
            .map(|x| x.as_u64().unwrap_or(0) as u32)
            .collect();
        prompts.push((
            v["id"].as_str().unwrap_or("?").to_string(),
            v["kind"].as_str().unwrap_or("?").to_string(),
            ids,
        ));
    }
    prompts.truncate(a.limit);
    let ck = Ckpt::open(&ckpt, pack, &config, &plan)?;
    let dev = Dev::new(a.device)?;
    let mut route = Route::load(dev, &ck, &plan, a.context, a.chunk)?;
    match a.draft_head.as_str() {
        "shared" => {}
        "bf16" => {
            // the checkpoint's BF16 rows, read through the shipped class (no overlay)
            let shipped = pack
                .compile_plan(&config)
                .map_err(|e| format!("plan: {e:?}"))?;
            let ck0 = Ckpt::open(&ckpt, pack, &config, &shipped)?;
            let rows = ck0.f32s(
                "model.llm.unembed.weight",
                &[plan.vocab_size as usize, plan.hidden_size as usize],
            )?;
            let bytes: Vec<u8> = rows
                .iter()
                .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
                .collect();
            route.set_draft_head(&bytes)?;
        }
        other => return Err(format!("--draft-head is shared or bf16, not {other}")),
    }
    let aux = ck
        .auxiliary
        .iter()
        .find(|(_, m)| m.raw("fuse_weight").is_some())
        .ok_or("no drafter file among the checkpoint's auxiliary files")?;
    let shared = Shared::of_route(&route)?;
    let mut drafter = DraftGpu::load(
        &route.dev,
        g.clone(),
        &CkptDraft(&aux.1),
        shared,
        a.context,
        a.chunk,
    )?;
    drafter.charge(&mut route.account);
    eprintln!("{}", route.account.describe());
    eprintln!(
        "loaded: route {:.3} GiB weights, drafter {:.3} GiB; class {}",
        route.weight_bytes as f64 / (1u64 << 30) as f64,
        drafter.bytes as f64 / (1u64 << 30) as f64,
        memra_engine::inkling2_gpu::numeric_class(&plan)?
    );
    let mut out: Box<dyn Write> = match &a.out {
        Some(p) => Box::new(std::fs::File::create(p).map_err(|e| format!("{}: {e}", p.display()))?),
        None => Box::new(std::io::stdout()),
    };
    if a.crosscheck {
        crosscheck(&mut route, &mut drafter, &aux.1, &ck, &g, &prompts[0].2)?;
    }
    if a.profile_rows > 0 {
        memra_engine::inkling2_spec::profile_verify(&mut route, &prompts[0].2, a.profile_rows, 10)?;
        return Ok(());
    }
    if a.bench > 0 {
        for (id, kind, ids) in &prompts {
            let c = bench_costs(&mut route, &mut drafter, ids, g.rows, a.bench)?;
            writeln!(
                out,
                "{}",
                serde_json::json!({"id": id, "kind": kind, "bench": {"reps": a.bench,
                    "prompt_tokens": ids.len(), "plain_step_ms": c.plain_step_ms,
                    "verify_ms_by_rows": c.verify_ms, "draft_ms_by_k": c.draft_ms,
                    "commit_rows_ms": c.commit_rows_ms}})
            )
            .map_err(|e| e.to_string())?;
            eprintln!(
                "bench {id}: plain {:.2} ms, verify {:?} ms, draft {:?} ms, commit {:.3} ms",
                c.plain_step_ms, c.verify_ms, c.draft_ms, c.commit_rows_ms
            );
        }
        return Ok(());
    }
    if let Some(dir) = &a.per_position {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for (id, kind, ids) in &prompts {
            per_position(
                &mut route,
                &mut drafter,
                &g,
                dir,
                id,
                kind,
                ids,
                a.gen_tokens,
                &stop,
                SpecArms {
                    skip_restore: false,
                    tap_slot: a.red_tap,
                },
            )?;
        }
        return Ok(());
    }
    let mut failed = 0;
    for (id, kind, ids) in &prompts {
        let plain = if a.check_plain {
            let (p, prefill_s, decode_s) = match a.sample {
                Some(t) => memra_engine::inkling2_spec::plain_sampled_timed(
                    &mut route,
                    ids,
                    a.gen_tokens,
                    &stop,
                    t,
                    a.seed,
                )?,
                None => plain_greedy_timed(&mut route, ids, a.gen_tokens, &stop)?,
            };
            writeln!(
                out,
                "{}",
                serde_json::json!({"id": id, "kind": kind, "mode": "plain", "prompt_tokens": ids.len(),
                    "generated": p.len(), "prefill_s": prefill_s, "decode_s": decode_s,
                    "decode_tok_s": (p.len() - 1) as f64 / decode_s, "tokens": p})
            )
            .map_err(|e| e.to_string())?;
            Some(p)
        } else {
            None
        };
        for &k in &a.ks {
            let sampling = match a.sample {
                Some(t) => Sampling::Sample {
                    temperature: t,
                    seed: a.seed,
                },
                None => Sampling::Greedy,
            };
            let (toks, st) = generate_arm(
                &mut route,
                &mut drafter,
                ids,
                a.gen_tokens,
                k,
                sampling,
                &stop,
                SpecArms {
                    skip_restore: a.red_no_restore,
                    tap_slot: a.red_tap,
                },
            )?;
            // a sampled request has no token-for-token reference
            let equal = plain
                .as_ref()
                .filter(|_| a.sample.is_none())
                .map(|p| *p == toks);
            let first_diff = plain.as_ref().and_then(|p| {
                p.iter()
                    .zip(&toks)
                    .position(|(x, y)| x != y)
                    .or(if p.len() != toks.len() {
                        Some(p.len().min(toks.len()))
                    } else {
                        None
                    })
            });
            if equal == Some(false) {
                failed += 1;
            }
            writeln!(
                out,
                "{}",
                serde_json::json!({"id": id, "kind": kind, "mode": if a.sample.is_some() { "sample" } else { "greedy" },
                    "temperature": a.sample, "k": k, "prompt_tokens": ids.len(), "generated": st.generated,
                    "steps": st.steps, "proposed": st.proposed, "accepted": st.accepted,
                    "acceptance_length": st.acceptance_length(), "tokens_per_step": st.tokens_per_step(),
                    "hist": st.hist, "k_hist": st.k_hist, "partial_commits": st.partial_commits, "plain_steps": st.plain_steps, "prefill_s": st.prefill_s, "decode_s": st.decode_s,
                    "decode_tok_s": (st.generated.max(1) - 1) as f64 / st.decode_s,
                    "equal_plain": equal, "first_diff": first_diff, "tokens": toks})
            )
            .map_err(|e| e.to_string())?;
            out.flush().map_err(|e| e.to_string())?;
            eprintln!(
                "{id} [{kind}] k={k}: accept len {:.3}, tokens/step {:.3}, {} steps, plain-equal {:?}",
                st.acceptance_length(),
                st.tokens_per_step(),
                st.steps,
                equal
            );
        }
    }
    if failed > 0 {
        return Err(format!(
            "{failed} speculative runs differ from plain greedy decode"
        ));
    }
    Ok(())
}

/// Teacher-forced proposals at every generated position: the plain greedy sequence of the
/// prompt, run through the route once (rows independent of the chunking), each row committed to
/// the drafter in order and, after row `P - 1`, seven proposals for the anchor `S[P]`. Writes
/// `<id>.json` (sequence, route argmax per row, proposals per position) and `<id>.taps.f32`
/// (`[positions, taps, hidden]` little-endian f32, the drafter's inputs) for the CPU arms.
#[allow(clippy::too_many_arguments)] // allow: one request's whole configuration
fn per_position(
    route: &mut Route,
    drafter: &mut DraftGpu,
    g: &refd::DraftGeometry,
    dir: &std::path::Path,
    id: &str,
    kind: &str,
    prompt: &[u32],
    n: usize,
    stop: &[u32],
    arms: SpecArms,
) -> Result<(), String> {
    let generated = plain_greedy(route, prompt, n, stop)?;
    let mut seq = prompt.to_vec();
    seq.extend_from_slice(&generated);
    let slots = tap_slots(route, arms)?;
    route.set_capture(&slots)?;
    route.reset()?;
    drafter.reset();
    let shared = Shared::of_route(route)?;
    let (h, nt, chunk) = (g.hidden, g.taps, route.chunk);
    let k = g.rows - 1;
    let mut argmax = Vec::new();
    let mut props: Vec<Vec<u32>> = Vec::new();
    let mut taps_out: Vec<u8> = Vec::with_capacity(seq.len() * nt * h * 4);
    let mut at = 0;
    for part in seq.chunks(chunk) {
        let o = route.forward(part, Head::AllArgmax, None)?;
        argmax.extend(o.argmax);
        let cap = route.captured().ok_or("capture")?;
        let all = route.dev.down(cap, 0, nt * chunk * h)?;
        for r in 0..part.len() {
            for t in 0..nt {
                for v in &all[(t * chunk + r) * h..(t * chunk + r + 1) * h] {
                    taps_out.extend_from_slice(&v.to_le_bytes());
                }
            }
            drafter.commit(&route.dev, cap, chunk, r, 1)?;
            let p = at + r + 1;
            if p >= prompt.len() && p < seq.len() {
                props.push(drafter.propose(&route.dev, shared, seq[p], k, None)?);
            }
        }
        at += part.len();
    }
    route.set_capture(&[])?;
    std::fs::write(dir.join(format!("{id}.taps.f32")), &taps_out).map_err(|e| e.to_string())?;
    let json = serde_json::json!({"id": id, "kind": kind, "prompt_tokens": prompt.len(), "seq": seq,
        "argmax": argmax, "first_anchor": prompt.len(), "proposals": props,
        "taps": nt, "hidden": h, "numeric_class": memra_engine::inkling2_gpu::numeric_class(route.plan())?});
    std::fs::write(dir.join(format!("{id}.json")), json.to_string()).map_err(|e| e.to_string())?;
    eprintln!(
        "per-position {id}: {} positions, {} proposals",
        seq.len(),
        props.len()
    );
    Ok(())
}

/// The GPU drafter against the f64 reference on the checkpoint: the first prompt's taps (every
/// row, from the route's capture), committed on both, then one greedy block.
fn crosscheck(
    route: &mut Route,
    drafter: &mut DraftGpu,
    aux: &memra_gguf::safetensors::StModel,
    ck: &Ckpt,
    g: &refd::DraftGeometry,
    prompt: &[u32],
) -> Result<(), String> {
    let slots = tap_slots(route, SpecArms::default())?;
    route.set_capture(&slots)?;
    route.reset()?;
    drafter.reset();
    let (h, nt, chunk) = (g.hidden, g.taps, route.chunk);
    let mut host_taps: Vec<f64> = Vec::new(); // [pos][tap][hidden]
    let mut anchor = 0;
    for (i, part) in prompt.chunks(chunk).enumerate() {
        let last = (i + 1) * chunk >= prompt.len();
        let o = route.forward(part, if last { Head::LastArgmax } else { Head::None }, None)?;
        let cap = route.captured().ok_or("capture")?;
        drafter.commit(&route.dev, cap, chunk, 0, part.len())?;
        let all = route.dev.down(cap, 0, nt * chunk * h)?;
        for r in 0..part.len() {
            for t in 0..nt {
                host_taps.extend(
                    all[(t * chunk + r) * h..(t * chunk + r + 1) * h]
                        .iter()
                        .map(|v| f64::from(*v)),
                );
            }
        }
        if last {
            anchor = *o.argmax.last().ok_or("argmax")?;
        }
    }
    let shared = Shared::of_route(route)?;
    let k = g.rows - 1;
    let gpu_tokens = drafter.propose(&route.dev, shared, anchor, k, None)?;
    let gpu_logits = drafter.logits(&route.dev, k)?;
    route.set_capture(&[])?;
    // the reference on CPU
    let w = refd::DraftWeights::load(g, &CkptDraft(aux))?;
    let mut cpu = refd::Drafter::new(g.clone(), w);
    let tw = nt * h;
    for pos in 0..prompt.len() {
        cpu.commit(&host_taps[pos * tw..(pos + 1) * tw]);
    }
    let f32s = |name: &str, shape: &[usize]| ck.f32s(name, shape);
    let embed = f32s("model.llm.embed.weight", &[g.padded_vocab, h])?;
    let embed_norm = f32s("model.llm.embed_norm.weight", &[h])?;
    let mut head = f32s("model.llm.unembed.weight", &[g.padded_vocab, h])?;
    head.truncate(g.vocab * h);
    let sh = refd::TrunkShared {
        embed: &embed,
        embed_norm: &embed_norm,
        head: &head,
    };
    let feature = &host_taps[(prompt.len() - 1) * tw..prompt.len() * tw];
    // teacher-forced on the GPU's proposals, so a near tie cannot fork the two chains
    let forced = gpu_tokens.clone();
    let prop = cpu.propose(
        anchor,
        feature,
        &sh,
        &mut |_, l| refd::argmax(l),
        Some(&forced),
    );
    let mut worst: f64 = 0.0;
    for (gv, cv) in gpu_logits.iter().zip(&prop.logits) {
        worst = worst.max((f64::from(*gv) - cv).abs() / (cv.abs() + 1.0));
    }
    let agree = prop
        .tokens
        .iter()
        .zip(&gpu_tokens)
        .filter(|(a, b)| a == b)
        .count();
    eprintln!(
        "crosscheck: prompt {} tokens, anchor {anchor}; GPU proposals {gpu_tokens:?}, reference argmax on the same chain {:?} ({agree}/{k} equal); logits max |gpu - ref| / (|ref| + 1) = {worst:.3e}",
        prompt.len(),
        prop.tokens
    );
    println!(
        "{}",
        serde_json::json!({"crosscheck": {"prompt_tokens": prompt.len(), "anchor": anchor, "gpu": gpu_tokens,
            "reference": prop.tokens, "agree": agree, "max_rel_logit_err": worst}})
    );
    Ok(())
}

/// The GPU drafter on the tiny fixture against the pinned f64 oracle: context K/V through the
/// commits, the block rows, the head's feature, greedy proposals, and the teacher-forced chain.
fn tiny_gate(a: &Args) -> Result<(), String> {
    let g = refd::tiny_geometry();
    let image = refd::StImage::parse(refd::TINY_FIXTURE)?;
    let oracle = refd::StImage::parse(refd::TINY_ORACLE)?;
    let dev = Dev::new(a.device)?;
    let h = g.hidden;
    let bf = |v: Vec<f32>| -> Vec<u8> {
        v.iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    };
    let embed = dev.up(&bf(image.bf16("trunk.embed", &[g.padded_vocab, h])?))?;
    let embed_norm = dev.up(&image.bf16("trunk.embed_norm", &[h])?)?;
    let head = dev.up(&bf(image.bf16("trunk.head", &[g.vocab, h])?))?;
    let st = dev.stream().clone();
    let ptr = |s: &cudarc::driver::CudaSlice<u8>| {
        use cudarc::driver::DevicePtr;
        s.device_ptr(&st).0
    };
    let ptrf = |s: &cudarc::driver::CudaSlice<f32>| {
        use cudarc::driver::DevicePtr;
        s.device_ptr(&st).0
    };
    let shared = Shared {
        embed: ptr(&embed),
        embed_norm: ptrf(&embed_norm),
        head: ptr(&head),
        head_nv: None,
    };
    let total = 13;
    let taps = image.bf16("taps", &[total, g.taps * h])?;
    // [pos][tap][hidden] -> [tap][pos][hidden], the route's capture layout
    let mut laid = vec![0f32; total * g.taps * h];
    for pos in 0..total {
        for t in 0..g.taps {
            laid[(t * total + pos) * h..(t * total + pos + 1) * h]
                .copy_from_slice(&taps[pos * g.taps * h + t * h..pos * g.taps * h + (t + 1) * h]);
        }
    }
    let taps_dev = dev.up(&laid)?;
    let anchors = oracle_i64(&image, "anchors")?;
    let forced = oracle_i64(&image, "forced")?;
    let worst = |got: &[f32], want: &[f64]| -> f64 {
        got.iter()
            .zip(want)
            .map(|(x, y)| (f64::from(*x) - y).abs() / (y.abs() + 1.0))
            .fold(0.0, f64::max)
    };
    const BAND: f64 = 2e-5;
    let mut ok = true;
    // commits of up to 16 rows (local rings of 32 rows: no wrap) and of 3 rows (rings of 8:
    // 13 committed positions wrap them)
    for chunk in [16usize, 3] {
        let mut drafter = DraftGpu::load(&dev, g.clone(), &image, shared, 64, chunk)?;
        for (s, &p) in refd::TINY_SCENARIOS.iter().enumerate() {
            drafter.reset();
            // commits of 3 rows and a remainder: the commit split must not matter
            let mut at = 0;
            while at < p {
                let n = 3.min(p - at).min(chunk);
                drafter.commit(&dev, &taps_dev, total, at, n)?;
                at += n;
            }
            let k = g.rows - 1;
            let anchor = anchors[s] as u32;
            let toks = drafter.propose(&dev, shared, anchor, k, None)?;
            let logits = drafter.logits(&dev, k)?;
            let (hidden, feature) = drafter.hidden(&dev)?;
            let want_tokens: Vec<u32> = oracle_i64(&oracle, &format!("s{s}.tokens"))?
                .iter()
                .map(|v| *v as u32)
                .collect();
            let e_l = worst(&logits, &oracle.f64s(&format!("s{s}.logits"))?);
            let e_h = worst(&hidden, &oracle.f64s(&format!("s{s}.hidden"))?);
            let e_f = worst(&feature, &oracle.f64s(&format!("s{s}.f_rnn"))?);
            let chain: Vec<u32> = forced[s * k..(s + 1) * k]
                .iter()
                .map(|v| *v as u32)
                .collect();
            let mut pick = |i: usize, _: &[f32]| chain[i];
            drafter.propose(&dev, shared, anchor, k, Some(&mut pick))?;
            let e_c = worst(
                &drafter.logits(&dev, k)?,
                &oracle.f64s(&format!("s{s}.forced_logits"))?,
            );
            let pass =
                toks == want_tokens && e_l <= BAND && e_h <= BAND && e_f <= BAND && e_c <= BAND;
            ok &= pass;
            println!(
                "ink2-dspark-tiny chunk {chunk} scenario {s} (P={p}): proposals {toks:?} (oracle {want_tokens:?}); max |gpu - f64| / (|f64| + 1): logits {e_l:.2e}, block rows {e_h:.2e}, feature {e_f:.2e}, forced chain {e_c:.2e} (band {BAND:.0e}) {}",
                if pass { "PASS" } else { "FAIL" }
            );
        }
        // two requests on one drafter: scenario 2's state stays committed while scenario 0 runs
        // in a second state; each one's proposals are unchanged
        let mut other = drafter.new_state(&dev)?;
        drafter.swap_state(&mut other)?;
        drafter.commit(&dev, &taps_dev, total, 0, 1)?;
        let t0 = drafter.propose(&dev, shared, anchors[0] as u32, g.rows - 1, None)?;
        drafter.swap_state(&mut other)?;
        let t2 = drafter.propose(&dev, shared, anchors[2] as u32, g.rows - 1, None)?;
        let w0: Vec<u32> = oracle_i64(&oracle, "s0.tokens")?
            .iter()
            .map(|v| *v as u32)
            .collect();
        let w2: Vec<u32> = oracle_i64(&oracle, "s2.tokens")?
            .iter()
            .map(|v| *v as u32)
            .collect();
        let two = t0 == w0 && t2 == w2;
        ok &= two;
        println!(
            "ink2-dspark-tiny chunk {chunk} two request states: {t0:?} / {t2:?} {}",
            if two { "PASS" } else { "FAIL" }
        );
    }
    if !ok {
        return Err("the GPU drafter does not match the oracle".into());
    }
    Ok(())
}

fn oracle_i64(image: &refd::StImage<'_>, name: &str) -> Result<Vec<i64>, String> {
    image.i64s(name)
}
