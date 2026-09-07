//! TP-2 box battery probe (lane/glm5-tp2-battery, 2026-08-31).
//!
//! ENGINE-LEVEL twin instrument for the TP-2 box window: the serving worker REFUSES
//! `MEMRA_GLM5_TP` by design (serving wiring — per-session TP admission/rollback — is the
//! lane's named increment 6, deliberately NOT this window), so every TP arm runs at the
//! engine level through the card3-probe program class (`prime_cache` + `decode_step`), and
//! the PP-3 plain arm runs through the SAME binary so the comparison is one instrument.
//! A separate served PP-3 boot ties this instrument to the banked 35.41 tok/s baseline
//! (the instrument-offset receipt).
//!
//! The arm (plain single-card / PP-3 / TP-2 / red) is chosen entirely by the ENV the
//! wrapper script sets (MEMRA_GLM5_TP / MEMRA_PP_* / MEMRA_MOE_* / MEMRA_GLM5_TP_GATE_RED);
//! this binary reads none of those itself — one binary, env-selected arms, per the
//! comparability requirement.
//!
//! Modes (BOXP_MODE):
//!   tape   (default) exactness rows, NO timing printed: per prompt — prime, dump the
//!          last-prime-token logits bytes (<tag>.prime.f32, f32 LE), greedy decode
//!          BOXP_MAX_NEW steps, dump the tape (<tag>.ids / <tag>.txt) and the first
//!          BOXP_LOGIT_STEPS decode-step logits (<tag>.step<i>.f32). Byte/band compare
//!          happens offline against the twin arm's dumps.
//!   spec   COMPOSED spec rows (lane/glm5-composition, spec x TP): per prompt — a
//!          [`Glm5SpecSession`] burst loop at BOXP_SPEC_K (default 3) drafts/verifies to
//!          BOXP_MAX_NEW; one JSONL row per prompt with spec tok/s, rounds, drafted,
//!          accepted, acc-rate and tok/cyc ((accepted + rounds) / rounds — the flip_check
//!          receipt shape). BOXP_SAMPLED=1 appends the vendor-default sampled twin on the
//!          first prompt (session-owned Philox sampler — the SERVING sampler, unlike the
//!          timed mode's host instrument RNG). Needs a draft source (MEMRA_GLM5_DFLASH)
//!          and, on a TP-armed boot, MEMRA_GLM5_SPEC_TP=1 (the composition flag; the
//!          session refuses without it — run the refusal once as the OFF-arm receipt).
//!   timed  pricing rows (only under the window's TIMING-IN-FLIGHT marker): per prompt —
//!          prime wall, per-step decode walls, decode tok/s over steps 2..N (the streamed
//!          (ct-1)/(t_last-t_first) estimator shape), TTFT proxy = prime + step1. One
//!          JSONL row per prompt on stdout. BOXP_SAMPLED=1 appends ONE vendor-default
//!          sampled row (temperature 1.0, top_p 0.95 — the artifact's
//!          generation_config.json — seeded host sampler, seed printed) on the first
//!          prompt, per the never-serve-greedy law's shape-twin requirement.
//!
//! Greedy rows stop at EOS (BOXP_EOS, default the glm5 triple 154820,154827,154829) or
//! BOXP_MAX_NEW; finish is recorded so the 128-token floor can exclude short rows by name
//! (the 3way measurement-trap guard). Loop-law screening happens on the banked .txt files.
//!
//! BOXP_FORCE_DIR (tape mode): TEACHER-FORCED decode — the rig gate's shape. Decode-step
//! inputs follow the reference arm's banked <tag>.ids from that dir, so a legal near-tie
//! flip in the PRIME logits cannot fork the tape and every step's logits are compared on
//! identical inputs. The probe's own argmax choices are banked as <tag>.own.ids; with
//! byte-identical logits they must equal the forced stream, and any mismatch is located
//! by index in the row.

use memra_engine::Engine;
use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_gguf::source::SafetensorsSource;
use std::io::Write;
use std::time::Instant;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn dump_f32(path: &std::path::Path, v: &[f32]) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    for x in v {
        f.write_all(&x.to_le_bytes())?;
    }
    f.flush()
}

/// xorshift64* — deterministic, seed printed in the row. Instrument RNG, not the serving
/// sampler; the sampled row is a traffic-SHAPE twin, never a serving-sampler parity claim.
struct Rng(u64);
impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545F4914F6CDD1D);
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// temperature + top-p nucleus over host logits (vendor defaults on the wrapper).
fn sample_top_p(logits: &[f32], temp: f32, top_p: f32, rng: &mut Rng) -> usize {
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let exps: Vec<f64> = idx
        .iter()
        .map(|&i| (((logits[i] - mx) / temp) as f64).exp())
        .collect();
    let z: f64 = exps.iter().sum();
    let mut cum = 0.0;
    let mut cut = exps.len();
    for (n, e) in exps.iter().enumerate() {
        cum += e / z;
        if cum >= top_p as f64 {
            cut = n + 1;
            break;
        }
    }
    let zc: f64 = exps[..cut].iter().sum();
    let draw = rng.next_f64() * zc;
    let mut acc = 0.0;
    for n in 0..cut {
        acc += exps[n];
        if draw <= acc {
            return idx[n];
        }
    }
    idx[cut - 1]
}

fn sha16(bytes: &[u8]) -> String {
    // FNV-1a 128-ish via two 64-bit lanes — receipt fingerprint only (byte identity is
    // decided by full-file compare offline, never by this fingerprint).
    let mut a: u64 = 0xcbf29ce484222325;
    let mut b: u64 = 0x9e3779b97f4a7c15;
    for &x in bytes {
        a = (a ^ x as u64).wrapping_mul(0x100000001b3);
        b = (b ^ a).wrapping_mul(0xff51afd7ed558ccd);
    }
    format!("{:08x}{:08x}", (a >> 32) as u32, (b >> 32) as u32)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = args
        .next()
        .expect("usage: glm5-tp2-box-probe <model_dir> <prompts_dir> <out_dir>");
    let prompts_dir = args.next().expect("prompts_dir");
    let out_dir = std::path::PathBuf::from(args.next().expect("out_dir"));
    std::fs::create_dir_all(&out_dir)?;

    let mode = std::env::var("BOXP_MODE").unwrap_or_else(|_| "tape".into());
    let max_new = env_usize("BOXP_MAX_NEW", 200);
    let logit_steps = env_usize("BOXP_LOGIT_STEPS", 8);
    let sampled = std::env::var("BOXP_SAMPLED").as_deref() == Ok("1");
    let seed: u64 = std::env::var("BOXP_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20260831);
    let temp: f32 = std::env::var("BOXP_TEMP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let top_p: f32 = std::env::var("BOXP_TOP_P")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.95);
    let eos: Vec<u32> = std::env::var("BOXP_EOS")
        .unwrap_or_else(|_| "154820,154827,154829".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let force_dir = std::env::var("BOXP_FORCE_DIR")
        .ok()
        .map(std::path::PathBuf::from);
    if force_dir.is_some() && mode != "tape" {
        return Err(
            "BOXP_FORCE_DIR is a tape-mode knob (the gate shape); timed rows free-run".into(),
        );
    }

    eprintln!(
        "[tp2-probe] mode={mode} max_new={max_new} logit_steps={logit_steps} sampled={sampled} \
         seed={seed} temp={temp} top_p={top_p} eos={eos:?} \
         MEMRA_GLM5_TP={:?} MEMRA_PP_STAGES={:?} MEMRA_MOE_RESIDENT={:?} MEMRA_MOE_SLOTS={:?} \
         MEMRA_BF16_MMV={:?} GATE_RED={:?}",
        std::env::var("MEMRA_GLM5_TP").ok(),
        std::env::var("MEMRA_PP_STAGES").ok(),
        std::env::var("MEMRA_MOE_RESIDENT").ok(),
        std::env::var("MEMRA_MOE_SLOTS").ok(),
        std::env::var("MEMRA_BF16_MMV").ok(),
        std::env::var("MEMRA_GLM5_TP_GATE_RED").ok(),
    );

    let load_t0 = Instant::now();
    let e = Engine::new(0)?;
    eprintln!("[tp2-probe] GPU0: {}", e.ctx().name()?);
    eprintln!(
        "[tp2-probe] engine up at {:.1}s",
        load_t0.elapsed().as_secs_f64()
    );
    let src = SafetensorsSource::open(std::path::Path::new(&model_dir))?;
    eprintln!(
        "[tp2-probe] source open at {:.1}s",
        load_t0.elapsed().as_secs_f64()
    );
    let model = HybridModel::load_from_source(&e, &src)?;
    let load_s = load_t0.elapsed().as_secs_f64();
    eprintln!(
        "[tp2-probe] loaded in {load_s:.1}s: n_layer={} n_embd={} n_vocab={} mtp_loaded={}",
        model.cfg.n_layer,
        model.cfg.n_embd,
        model.cfg.n_vocab,
        model.mtp.is_some(),
    );
    let (free, total) = e.ctx().mem_get_info()?;
    eprintln!(
        "[tp2-probe] vram-post-load dev-root: free={:.2} GiB / total={:.2} GiB",
        free as f64 / (1 << 30) as f64,
        total as f64 / (1 << 30) as f64
    );
    let tok = memra_tokenizer::Tokenizer::from_hf_dir(std::path::Path::new(&model_dir))?;

    let mut prompt_files: Vec<std::path::PathBuf> = std::fs::read_dir(&prompts_dir)?
        .filter_map(|d| d.ok().map(|d| d.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    prompt_files.sort();
    if prompt_files.is_empty() {
        return Err("no .txt prompts in prompts_dir".into());
    }

    let rows_path = out_dir.join("rows.jsonl");
    let mut rows = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&rows_path)?;

    for (pi, pf) in prompt_files.iter().enumerate() {
        let tag = pf.file_stem().unwrap().to_string_lossy().to_string();
        let text = std::fs::read_to_string(pf)?;
        let rendered = tok.apply_chat_template(&[("user", text.trim())], true);
        let ids = tok.encode(&rendered, true);
        eprintln!(
            "[tp2-probe] {tag}: {} chars -> {} tokens",
            text.len(),
            ids.len()
        );

        // Sub-rows for this prompt: always the greedy row; in timed mode the first prompt
        // also carries the vendor-default sampled twin when asked.
        let arms: &[&str] = if (mode == "timed" || mode == "spec") && sampled && pi == 0 {
            &["greedy", "vendor"]
        } else {
            &["greedy"]
        };
        if mode == "spec" {
            let k = env_usize("BOXP_SPEC_K", 3);
            for &arm in arms {
                let max_ctx = ids.len() + max_new + k + 16;
                let sampling = (arm == "vendor").then_some(memra_engine::spec::SpecSampling {
                    temp,
                    seed: seed ^ 0x5ec5_0000 ^ pi as u64,
                    top_k: 0,
                    top_p,
                    min_p: 0.0,
                    penalty_last_n: 0,
                    penalty_repeat: 1.0,
                    penalty_freq: 0.0,
                    penalty_present: 0.0,
                });
                let t0 = Instant::now();
                let mut sess = model.glm5_spec_session_new(&e, &ids, max_ctx, sampling)?;
                let prime_s = t0.elapsed().as_secs_f64();
                let mut out: Vec<u32> = Vec::with_capacity(max_new);
                let (mut drafted, mut accepted) = (0usize, 0usize);
                let g0 = Instant::now();
                while out.len() < max_new && !sess.finished() {
                    let (burst, d, a) = model.glm5_spec_session_burst(
                        &e,
                        &mut sess,
                        max_new - out.len(),
                        k,
                        &eos,
                    )?;
                    if burst.is_empty() {
                        break; // ctx guard tripped with nothing new — never spin
                    }
                    out.extend(burst);
                    drafted += d;
                    accepted += a;
                }
                let gen_wall = g0.elapsed().as_secs_f64();
                // EOS CLAMP + honest finish (#82 review): a burst commits WHOLE rounds, so
                // an EOS-terminated round can carry up to K post-EOS tokens and a
                // max_new-bounded round can overshoot. The plain/timed arms break AT the
                // EOS token; clamp here so the tape, the token count and the sha describe
                // the same stream the serving surface would emit, and label the finish
                // from what the CLAMPED tape actually contains.
                let eos_at = out.iter().position(|t| eos.contains(t));
                let finish = if let Some(i) = eos_at {
                    out.truncate(i + 1);
                    "stop"
                } else if out.len() >= max_new {
                    out.truncate(max_new);
                    "length"
                } else if sess.finished() {
                    // No EOS in the emitted stream and the session says done = the ctx
                    // guard tripped; never labelled "stop".
                    "ctx"
                } else {
                    "length"
                };
                let rounds = sess.rounds;
                // The wall covers every token the rounds produced, including any
                // post-EOS/overshoot tail the clamp dropped; `surplus` names it so a row
                // is never read as a clean tokens/wall ratio when it is not.
                let surplus = (accepted + rounds).saturating_sub(out.len().saturating_sub(1));
                let tok_s = if gen_wall > 0.0 {
                    out.len() as f64 / gen_wall
                } else {
                    0.0
                };
                let tok_cyc = if rounds > 0 {
                    (accepted + rounds) as f64 / rounds as f64
                } else {
                    0.0
                };
                let acc_rate = if drafted > 0 {
                    accepted as f64 / drafted as f64
                } else {
                    0.0
                };
                let dec = tok.decode(&out);
                let suffix = if arm == "vendor" { "-vendor" } else { "" };
                std::fs::write(out_dir.join(format!("{tag}{suffix}.spec.txt")), &dec)?;
                std::fs::write(
                    out_dir.join(format!("{tag}{suffix}.spec.ids")),
                    out.iter()
                        .map(|t| t.to_string())
                        .collect::<Vec<_>>()
                        .join(" "),
                )?;
                let row = format!(
                    "{{\"tag\":\"{tag}\",\"mode\":\"spec\",\"arm\":\"{arm}\",\"k\":{k},\
                     \"prompt_tokens\":{},\"out_tokens\":{},\"prime_s\":{prime_s:.4},\
                     \"gen_wall_s\":{gen_wall:.4},\"spec_tok_s\":{tok_s:.3},\
                     \"rounds\":{rounds},\"drafted\":{drafted},\"accepted\":{accepted},\
                     \"acc_rate\":{acc_rate:.4},\"tok_cyc\":{tok_cyc:.4},\
                     \"surplus_dropped\":{surplus},\
                     \"finish\":\"{finish}\",\"seed\":{seed},\"tape_sha16\":\"{}\"}}",
                    ids.len(),
                    out.len(),
                    sha16(dec.as_bytes()),
                );
                writeln!(rows, "{row}")?;
                rows.flush()?;
                println!("{row}");
            }
            continue;
        }
        for &arm in arms {
            let max_ctx = ids.len() + max_new + 16;
            let mut cache =
                memra_engine::cache::Cache::new_planned(&e, &model.cfg, &model.plan, max_ctx)?;
            let t0 = Instant::now();
            let (logits0, _seed_t, _hiddens) = model.prime_cache(&e, &ids, &mut cache, 0)?;
            let prime_s = t0.elapsed().as_secs_f64();

            if mode == "tape" && arm == "greedy" {
                dump_f32(&out_dir.join(format!("{tag}.prime.f32")), &logits0)?;
            }

            let mut rng = Rng(seed ^ (pi as u64) << 1 | 1);
            let pick = |ll: &[f32], rng: &mut Rng| -> u32 {
                if arm == "vendor" {
                    sample_top_p(ll, temp, top_p, rng) as u32
                } else {
                    argmax(ll) as u32
                }
            };

            // Teacher-forced reference stream (gate shape): inputs follow the reference
            // arm's tape; own choices are banked separately.
            let forced: Option<Vec<u32>> = match &force_dir {
                Some(d) => {
                    let raw = std::fs::read_to_string(d.join(format!("{tag}.ids")))
                        .map_err(|e| format!("BOXP_FORCE_DIR: no {tag}.ids: {e}"))?;
                    Some(
                        raw.split_whitespace()
                            .filter_map(|s| s.parse().ok())
                            .collect(),
                    )
                }
                None => None,
            };
            let cap = forced.as_ref().map(|f| f.len()).unwrap_or(max_new);

            let mut tape: Vec<u32> = Vec::with_capacity(cap); // inputs actually fed
            let mut own: Vec<u32> = Vec::with_capacity(cap); // this arm's own choices
            let mut step_walls: Vec<f64> = Vec::with_capacity(cap);
            let own0 = pick(&logits0, &mut rng);
            own.push(own0);
            let first = forced.as_ref().map(|f| f[0]).unwrap_or(own0);
            tape.push(first);
            let mut finish = "length";
            if forced.is_none() && eos.contains(&first) {
                finish = "stop";
            } else {
                while tape.len() < cap {
                    let st = Instant::now();
                    let ll = model.decode_step(&e, *tape.last().unwrap(), &mut cache)?;
                    step_walls.push(st.elapsed().as_secs_f64());
                    if mode == "tape" && arm == "greedy" && step_walls.len() <= logit_steps {
                        dump_f32(
                            &out_dir.join(format!("{tag}.step{}.f32", step_walls.len())),
                            &ll,
                        )?;
                    }
                    let t = pick(&ll, &mut rng);
                    own.push(t);
                    let fed = forced.as_ref().map(|f| f[tape.len()]).unwrap_or(t);
                    tape.push(fed);
                    if forced.is_none() && eos.contains(&t) {
                        finish = "stop";
                        break;
                    }
                }
            }
            if let Some(f) = &forced {
                let div = own.iter().zip(f.iter()).position(|(a, b)| a != b);
                eprintln!(
                    "[tp2-probe] {tag}: FORCED run, own-vs-forced first divergence = {:?} \
                     (None = tape-identical under forcing)",
                    div
                );
                finish = "forced";
            }

            // Banked tape = this arm's OWN choices (under forcing, banking the fed stream
            // would compare the reference to itself — vacuous by construction).
            let bank: &[u32] = if forced.is_some() { &own } else { &tape };
            let dec = tok.decode(bank);
            let suffix = if arm == "vendor" { "-vendor" } else { "" };
            std::fs::write(out_dir.join(format!("{tag}{suffix}.txt")), &dec)?;
            std::fs::write(
                out_dir.join(format!("{tag}{suffix}.ids")),
                tape.iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
            )?;

            if mode == "timed" {
                // decode tok/s over the walls AFTER the first emitted token — the streamed
                // (ct-1)/(t_last-t_first) estimator shape the banked baselines use.
                let n_steps = step_walls.len();
                let gen_wall: f64 = step_walls.iter().sum();
                let tok_s = if n_steps > 0 {
                    n_steps as f64 / gen_wall
                } else {
                    0.0
                };
                let ttft = prime_s + step_walls.first().copied().unwrap_or(0.0);
                let row = format!(
                    "{{\"tag\":\"{tag}\",\"arm\":\"{arm}\",\"prompt_tokens\":{},\"out_tokens\":{},\
                     \"prime_s\":{prime_s:.4},\"ttft_s\":{ttft:.4},\"gen_wall_s\":{gen_wall:.4},\
                     \"decode_tok_s\":{tok_s:.3},\"finish\":\"{finish}\",\"seed\":{seed},\
                     \"tape_sha16\":\"{}\"}}",
                    ids.len(),
                    tape.len(),
                    sha16(dec.as_bytes()),
                );
                writeln!(rows, "{row}")?;
                rows.flush()?;
                println!("{row}");
            } else {
                eprintln!(
                    "[tp2-probe] {tag}{suffix}: {} tokens finish={finish} sha16={}",
                    tape.len(),
                    sha16(dec.as_bytes())
                );
            }
        }
    }
    // Non-vacuity engagement receipt on the real artifact: TP arms must have dispatched
    // peer-owned expert slots (the rig gate's R3 lesson — a stream can legally route
    // root-only for a while; the CLAIM requires proven peer engagement).
    let peer_slots = memra_engine::glm5_tp::GLM5_EP_PEER_SLOT_DISPATCHES
        .load(std::sync::atomic::Ordering::Relaxed);
    eprintln!("[tp2-probe] ep-peer-slot-dispatches={peer_slots}");
    eprintln!("[tp2-probe] done: {}", rows_path.display());
    Ok(())
}
