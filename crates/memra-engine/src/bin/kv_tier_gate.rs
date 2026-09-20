//! Native eager baseline capture, NOT active/prefix tier qualification.
//! Run GPU cases ONLY through tools/tier-battery.py (canonical rig lock + telemetry).
//! No tier restore is simulated: active/prefix refuse before model or CUDA initialization.
#[path = "kv_tier_gate/capture_contract.rs"]
mod capture_contract;
#[path = "kv_tier_gate/cli.rs"]
mod cli;

use memra_engine::{Engine, forward::argmax, hybrid::HybridModel};
use memra_gguf::{GgufFile, model_plan::ModelPlan};
use memra_kv::Cache;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::Path,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const GENERATE: usize = 128;

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn file_hash(path: &Path) -> Result<String> {
    let mut input = File::open(path)?;
    let mut buf = vec![0; 1024 * 1024];
    let mut digest = Sha256::new();
    loop {
        let n = input.read(&mut buf)?;
        if n == 0 {
            break;
        }
        digest.update(&buf[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|x| x.to_bits().to_le_bytes())
        .collect()
}
fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Semantic state only: valid native K/V bytes, counters, current conv/SSM, logits, hidden.
/// Spare SSM ping-pong storage is scratch, not continuation state. Never hash allocator slack.
/// A new state plane is unsupported until explicitly covered here.
fn capture(
    e: &Engine,
    cache: &Cache,
    plan: &ModelPlan,
    logits: &[f32],
    hidden: &[f32],
    out: &Path,
    name: &str,
) -> Result<String> {
    cache.ensure_usable("kv-tier-gate capture")?;
    if cache.latent.iter().any(Option::is_some)
        || cache.tp_kv.iter().any(Option::is_some)
        || cache.glm5_tp_recur.iter().any(Option::is_some)
        || cache.glm5_tp_latent_peer.iter().any(Option::is_some)
        || cache.dflash_taps.is_some()
        || cache.hc_taps.is_some()
        || cache.glm5_decode_graph.is_some()
        || cache.glm5_tp_sym_graph.is_some()
        || cache.qwen_prime_graph.is_some()
        || cache.last_logits_dev.is_some()
    {
        return Err(
            "REFUSED: state contains unbound latent/rank/draft/graph/parked-logit planes".into(),
        );
    }
    e.stream().synchronize()?;
    let mut manifest = String::from("plane\tvalid_bytes\tsha256\n");
    let mut record = |role: &str, bytes: &[u8]| {
        manifest.push_str(&format!("{role}\t{}\t{}\n", bytes.len(), hash(bytes)));
    };
    record("position-u64le", &(cache.pos as u64).to_le_bytes());
    record("max-context-u64le", &(cache.max_ctx as u64).to_le_bytes());
    if cache.kv.len() != cache.recur.len() {
        return Err("state vector length mismatch".into());
    }
    capture_contract::validate_plan(plan, cache.kv.len())?;
    for (i, (kv, recur)) in cache.kv.iter().zip(&cache.recur).enumerate() {
        record(
            &format!("layer-{i}-presence"),
            &[u8::from(kv.is_some()), u8::from(recur.is_some())],
        );
        let geometry = kv.as_ref().map(|kv| capture_contract::KvGeometry {
            len: kv.len,
            ring: kv.ring.is_some(),
            base: kv.base_d.is_some(),
            key_width: kv.kv_dim_k,
            value_width: kv.kv_dim_v,
            key_row: kv.k_tok_bytes,
            value_row: kv.v_tok_bytes,
            key_allocation: kv.k.len(),
            value_allocation: kv.v.len(),
        });
        let plane = capture_contract::classify(plan, i, cache.pos, geometry, recur.is_some())?;
        if plane == capture_contract::Plane::AbsentUnexecutedMtp {
            record(&format!("layer-{i}-absent-unexecuted-mtp"), &[]);
            continue;
        }
        if let Some(kv) = kv {
            record(
                &format!("layer-{i}-geometry-u64le"),
                &[
                    kv.len,
                    kv.kv_dim_k,
                    kv.kv_dim_v,
                    kv.k_tok_bytes,
                    kv.v_tok_bytes,
                ]
                .iter()
                .flat_map(|n| (*n as u64).to_le_bytes())
                .collect::<Vec<_>>(),
            );
            let k_len = kv
                .len
                .checked_mul(kv.k_tok_bytes)
                .ok_or("K extent overflow")?;
            let v_len = kv
                .len
                .checked_mul(kv.v_tok_bytes)
                .ok_or("V extent overflow")?;
            if k_len > kv.k.len() || v_len > kv.v.len() {
                return Err("KV extent exceeds allocation".into());
            }
            record(
                &format!("layer-{i}-key-q8_0"),
                &e.dtoh_u8_view(&kv.k.slice(..k_len))?,
            );
            record(
                &format!("layer-{i}-value-q5_1"),
                &e.dtoh_u8_view(&kv.v.slice(..v_len))?,
            );
            record(
                &format!("layer-{i}-len-device-i32le"),
                &e.dtoh_i32(&kv.len_d)?
                    .iter()
                    .flat_map(|n| n.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
        }
        if let Some(recur) = recur {
            record(
                &format!("layer-{i}-conv-f32le"),
                &f32_bytes(&e.dtoh(&recur.conv_state)?),
            );
            record(
                &format!("layer-{i}-ssm-f32le"),
                &f32_bytes(&e.dtoh(&recur.ssm_state)?),
            );
        }
    }
    record("last-logits-f32le", &f32_bytes(logits));
    record("last-hidden-f32le", &f32_bytes(hidden));
    fs::write(out.join(format!("{name}-state.tsv")), &manifest)?;
    Ok(hash(manifest.as_bytes()))
}

fn baseline(args: &cli::Args) -> Result<()> {
    // A fixed process is not permission to select alternate numerical/parallel programs.
    // Inspect NAMES only; never expose environment values in a receipt.
    for (key, _) in std::env::vars_os() {
        let key = key.to_string_lossy();
        if key.starts_with("MEMRA_")
            && !["MEMRA_NVCC", "MEMRA_CUDA_ARCH", "MEMRA_GPU_LOCK"].contains(&key.as_ref())
        {
            return Err(format!(
                "REFUSED: runtime override {key} must be unset for this naked eager baseline"
            )
            .into());
        }
    }
    let artifact_hash = file_hash(&args.artifact)?;
    let binary_hash = file_hash(&std::env::current_exe()?)?;
    let g = GgufFile::open(&args.artifact)?;
    let tokenizer = Tokenizer::from_gguf(&g)?;
    let e = Engine::new(0)?;
    // Same trunk-only loader as run-gen; optional MTP is deliberately not a scored program.
    let model = HybridModel::load_without_mtp(&e, &g)?;
    capture_contract::validate_plan(&model.plan, model.cfg.n_layer as usize)?;
    if args.context > model.plan.context_length as usize {
        return Err("requested context exceeds model plan".into());
    }
    let plan = format!("{:?}", model.plan);
    fs::write(args.out.join("plan.debug"), &plan)?;
    // Raw-token probe, not a chat-template or quality gate. Tokenizer bytes are artifact-bound.
    let seed = tokenizer.encode(
        "The quick brown fox jumps over the lazy dog. Explain why the sky is blue.\n",
        false,
    );
    if seed.is_empty() || seed.iter().any(|&t| t >= model.cfg.n_vocab) {
        return Err("invalid tokenizer seed".into());
    }
    let prompt: Vec<u32> = seed
        .iter()
        .copied()
        .cycle()
        .take(args.context - GENERATE)
        .collect();
    let prompt_bytes = u32_bytes(&prompt);
    fs::write(args.out.join("prompt.u32le"), &prompt_bytes)?;
    fs::write(
        args.out.join("identity.txt"),
        format!(
            "artifact_sha256={artifact_hash}\nbinary_sha256={binary_hash}\nplan_debug_sha256={}\nprompt_sha256={}\nprogram=native-decode_step_h-tokenwise-trunk-no-mtp\nmode=raw-token-no-chat-template\ncontext={}\nprompt_tokens={}\ngenerate={GENERATE}\nrequested_tiers={}\nengaged_tier_bytes=0\nscope=baseline-only; no tier qualification or performance claim\n",
            hash(plan.as_bytes()),
            hash(&prompt_bytes),
            args.context,
            prompt.len(),
            args.tiers
        ),
    )?;
    let mut cache = memra_engine::pp::new_cache(&e, &model.cfg, args.context)?;
    let mut last = None;
    for (i, &token) in prompt.iter().enumerate() {
        last = Some(model.decode_step_h(&e, token, &mut cache)?);
        if (i + 1).is_multiple_of(256) {
            eprintln!("baseline prompt committed={}", cache.pos);
        }
    }
    let (mut logits, mut hidden) = last.ok_or("empty prompt")?;
    let prefix_hash = capture(
        &e,
        &cache,
        &model.plan,
        &logits,
        &e.dtoh(&hidden)?,
        &args.out,
        "prefix",
    )?;
    let mut rows = File::create(args.out.join("logits.tsv"))?;
    writeln!(rows, "committed\tlogits_f32le_sha256")?;
    let mut tokens = Vec::with_capacity(GENERATE);
    for _ in 0..GENERATE {
        if logits.len() != model.cfg.n_vocab as usize || logits.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite or wrong-sized logits".into());
        }
        writeln!(rows, "{}\t{}", cache.pos, hash(&f32_bytes(&logits)))?;
        let token = argmax(&logits) as u32;
        tokens.push(token);
        // Fixed length diagnostic: even EOS is fed back, reaching the promised committed context.
        (logits, hidden) = model.decode_step_h(&e, token, &mut cache)?;
    }
    if cache.pos != args.context {
        return Err("gate did not reach promised committed context".into());
    }
    if logits.len() != model.cfg.n_vocab as usize || logits.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite or wrong-sized final logits".into());
    }
    let final_hash = capture(
        &e,
        &cache,
        &model.plan,
        &logits,
        &e.dtoh(&hidden)?,
        &args.out,
        "final",
    )?;
    writeln!(rows, "{}\t{}", cache.pos, hash(&f32_bytes(&logits)))?;
    rows.sync_all()?;
    let tokens = u32_bytes(&tokens);
    fs::write(args.out.join("tokens.u32le"), &tokens)?;
    let ids: Vec<u32> = tokens
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    fs::write(args.out.join("output.txt"), tokenizer.decode(&ids))?;
    fs::write(args.out.join("final-logits.f32le"), f32_bytes(&logits))?;
    fs::write(
        args.out.join("BASELINE.txt"),
        format!(
            "BASELINE_CAPTURED\ncommitted={}\nprefix_state_manifest_sha256={prefix_hash}\nfinal_state_manifest_sha256={final_hash}\ntokens_sha256={}\nlogit_rows_sha256={}\nactive_engaged=false\nprefix_engaged=false\n",
            cache.pos,
            hash(&tokens),
            file_hash(&args.out.join("logits.tsv"))?
        ),
    )?;
    println!(
        "BASELINE_CAPTURED committed={} generated={GENERATE}; tiers NOT engaged",
        cache.pos
    );
    Ok(())
}

fn run() -> Result<()> {
    let args = cli::parse(std::env::args().skip(1))?;
    fs::create_dir(&args.out)?; // Immutable receipt namespace; never overwrite an earlier attempt.
    let result = if args.case != "baseline" {
        Err(format!("REFUSED: {} requires a native CUDA materializer + scheduler binding with nonzero demote/reload engagement; CPU fixtures and HostPrefix patch do not provide that binding", args.case).into())
    } else {
        baseline(&args)
    };
    if let Err(error) = &result {
        fs::write(args.out.join("REFUSED.txt"), format!("{error}\n"))?;
    }
    result
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{}", cli::diagnostic(&error.to_string()));
        std::process::exit(2);
    }
}
