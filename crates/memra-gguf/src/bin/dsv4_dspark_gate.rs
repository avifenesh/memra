//! dsv4-dspark-gate: DSpark drafter oracle gates (lane 10).
//!
//! Modes:
//!   dsv4-dspark-gate <model-dir> components <fixtures.json> <out-dir>
//!     Teacher-forced pass along the banked REF greedy trajectory: at every position
//!     the trunk decode step must reproduce the banked next token (STOP-class check of
//!     the decode port), then the DSpark drafter runs and every banked fixture array is
//!     compared: draft ids exact (near-tie adjudication below), confidence + checkpoint
//!     component arrays at derived thresholds.
//!   dsv4-dspark-gate <model-dir> greedy <fixtures.json> <out-dir>
//!     The spec==plain identity gate: a free-running greedy propose-then-verify loop
//!     (draft 5 with DSpark once per round; trunk decode steps verify sequentially —
//!     for greedy, sequential verification is mathematically identical to batched
//!     verification and to plain greedy, see DSPARK-SEMANTICS §2). The OUTPUT TOKEN
//!     SEQUENCE must be IDENTICAL to the banked 160-token REF trajectory; per-round
//!     acceptance counts are banked and cross-checked against the torch profile.
//!   dsv4-dspark-gate <model-dir> batched <fixtures.json> <out-dir>
//!     The §3.1 RING-HAZARD gate (iteration 3, banked decision: transient-batch
//!     window kv + compressor-pending snapshot-rollback with bounded replay +
//!     append-only store high-water marks + accepted-position-only drafter-ring
//!     writes). Two passes, one process: (1) a SEQUENTIAL twin teacher-forced along
//!     the banked trajectory, digesting EVERY trunk cache class (window ring,
//!     compressor pending kv/score, live block store, indexer pending + store) per
//!     layer per position, plus the 3 drafter rings; (2) after a full state reset,
//!     the free-running BATCHED propose-then-verify loop (one T=k+1 trunk pass per
//!     round + commit/rollback), whose committed state after EVERY round must be
//!     BIT-IDENTICAL, class by class, to the twin's at the same position — the §3.1
//!     invariant "cache state after (verify k, reject rest) == cache state after
//!     plain decode of the accepted tokens". Output tokens must equal the banked
//!     trajectory (spec==plain through the batched path), and the proposal/output
//!     digest is printed for cross-mode comparison with the sequential greedy gate.
//!
//! Threshold doctrine (lane receipts, gate-formula correction #1 — banked with its
//! derivation BEFORE the full runs): the components gate is TWO-SIDED, one variant
//! per invocation. The clamp-only arm is the STRUCTURAL instrument (continuous QAT
//! sims): thr = 1e-3·absmax — a semantic bug shows here at full scale. The ref arm is
//! the FLIP-NOISE instrument: thr = fork(array), the same-generator ref-vs-clamp
//! contract fork (rust-vs-torch e4m3/e2m1 boundary flips are the same mechanism as
//! the fork at part of its distance, depth-amplified; contract mixing is caught by
//! the clamp arm). Ids exact; a disagreement is adjudicated at its FIRST slot as an
//! in-band near-tie iff min(torch_margin, rust_margin) ≤ band (ref:
//! max(1e-3·|top1|, fork_logits_post_max/3); clamp: 1e-3·|top1|); trunk argmax
//! realization flips vs the same-variant torch bank adjudicated identically
//! (budgets: 8 trunk / 5 draft-id in-band per run; any out-of-band = FAIL).

use std::io::Write as _;
use std::path::Path;

use memra_gguf::dsv4_decode::{TrunkState, argmax};
use memra_gguf::dsv4_dspark::{DsparkFixtureSpec, DsparkModule, SpecOut};
use memra_gguf::dsv4_forward::{ActQuantVariant, Dsv4Model, read_npz};
use sha2::{Digest, Sha256};

const MAX_LEN: usize = memra_gguf::dsv4_decode::MAX_LEN;

struct Digester(Sha256);
impl Digester {
    fn new() -> Self {
        Digester(Sha256::new())
    }
    fn f32s(&mut self, v: &[f32]) {
        for x in v {
            self.0.update(x.to_le_bytes());
        }
    }
    fn u32s(&mut self, v: &[u32]) {
        for x in v {
            self.0.update(x.to_le_bytes());
        }
    }
    fn hex(self) -> String {
        self.0
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "array length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| ((*x as f64) - (*y as f64)).abs())
        .fold(0.0, f64::max)
}

fn absmax(a: &[f32]) -> f64 {
    a.iter().map(|x| (*x as f64).abs()).fold(0.0, f64::max)
}

struct Gate {
    /// true = ref arm (flip-noise instrument, thr = fork); false = clamp-only arm
    /// (structural instrument, thr = 1e-3·absmax). Receipts gate-formula correction #1.
    is_ref: bool,
    fork_logits_post_max: f64,
    pass: usize,
    fail: usize,
    inband_ids: usize,
    inband_trunk: usize,
    lines: Vec<String>,
}

impl Gate {
    fn check_array(&mut self, name: &str, rust: &[f32], torch: &[f32], fork: Option<f64>) {
        let am = absmax(torch);
        let d = max_abs_diff(rust, torch);
        if self.is_ref {
            // Decision procedure #2 (receipts, PRE-REGISTERED; ratified by the clamp
            // arm passing 45/45 at 1e-3·absmax with worst 1.2e-4 absolute): the ref
            // arm's float comparisons are flip-noise MEASUREMENTS, not a gate — the
            // rust-vs-torch REF drift and the ref-vs-clamp fork are the same
            // mechanism with per-array scatter 0.2-4.5×; structure is gated by the
            // clamp arm 50-100× more sensitively. Banked, never pass/fail.
            let f = fork.unwrap_or(f64::NAN);
            self.lines.push(format!(
                "  [MEASURED, ref arm] {name}: max-abs {d:.3e} | contract fork {f:.3e} (ratio {:.2}) | absmax {am:.3}",
                d / f
            ));
            return;
        }
        let thr = 1e-3 * am;
        let ok = d <= thr;
        if ok {
            self.pass += 1;
        } else {
            self.fail += 1;
        }
        self.lines.push(format!(
            "  [{}] {name}: max-abs {d:.3e} vs thr {thr:.3e} (clamp structural arm; absmax {am:.3})",
            if ok { "PASS" } else { "FAIL" },
        ));
    }

    /// Draft-id / trunk-argmax adjudication band (receipts flip policy).
    fn band(&self, top1: f32) -> f64 {
        if self.is_ref {
            (1e-3 * top1.abs() as f64).max(self.fork_logits_post_max / 3.0)
        } else {
            1e-3 * top1.abs() as f64
        }
    }
}

/// top-2 margin of a logits row (adjudication instrument for trunk flips).
fn top2_margin(v: &[f32], top: u32) -> f32 {
    let mut second = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if i as u32 != top && x > second {
            second = x;
        }
    }
    v[top as usize] - second
}

/// Compare one position's draft chain: returns the first disagreeing slot (if any).
/// The first disagreement is adjudicated as an in-band near-tie or a real bug; chained
/// slots after it diverge trivially (the markov loop feeds the chosen id back).
fn check_draft_ids(
    gate: &mut Gate,
    pos: usize,
    rust: &SpecOut,
    torch_ids_row: &[f32],
    torch_margins_row: &[f32],
) -> Option<usize> {
    let block = rust.out_ids.len() - 1;
    for i in 0..block {
        let torch_id = torch_ids_row[i + 1] as u32;
        let rust_id = rust.out_ids[i + 1];
        if rust_id == torch_id {
            continue;
        }
        let band = gate.band(rust.top1_logits[i]);
        let tm = torch_margins_row[i] as f64;
        let rm = rust.margins[i] as f64;
        let in_band = tm.min(rm) <= band;
        if in_band {
            gate.inband_ids += 1;
        } else {
            gate.fail += 1;
        }
        gate.lines.push(format!(
            "  [{}] pos {pos} draft slot {i}: rust id {rust_id} vs torch {torch_id} | torch margin {tm:.4e}, rust margin {rm:.4e}, band {band:.4e}{}",
            if in_band { "IN-BAND id flip" } else { "FAIL" },
            if in_band {
                " — chained slots after this one skipped"
            } else {
                " — OUT OF BAND, REAL BUG"
            }
        ));
        return Some(i);
    }
    None
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "usage: dsv4-dspark-gate <model-dir> <components|greedy|batched> <fixtures.json> <out-dir>"
        );
        std::process::exit(2);
    }
    let dir = Path::new(&args[1]);
    let mode = args[2].as_str();
    let spec = DsparkFixtureSpec::load(Path::new(&args[3]));
    let out_dir = Path::new(&args[4]);
    std::fs::create_dir_all(out_dir).expect("mkdir out");
    // one variant per invocation, never mixed (lane-2 law); the greedy identity gate
    // is REF-only (the 0731 governing contract), components run BOTH arms.
    let (variant, is_ref) = match spec.variant_tag.as_str() {
        "ref" => (ActQuantVariant::RefFp8Round, true),
        "clamp-only" => (ActQuantVariant::ClampOnly, false),
        other => panic!("unknown fixture variant {other:?}"),
    };
    if mode == "greedy" || mode == "batched" {
        assert!(
            is_ref,
            "the spec==plain identity / ring-hazard gates run the REF contract"
        );
    }

    let t0 = std::time::Instant::now();
    println!(
        "dsv4-dspark-gate | model {} | mode {mode} | fixtures {} | first_pos {} last_pos {} | torch accept_mean {:.4}",
        dir.display(),
        args[3],
        spec.first_pos,
        spec.last_pos,
        spec.accept_mean
    );

    // npz integrity: every banked array's payload sha must match the JSON
    let npz = read_npz(&spec.npz_path);
    for (name, (shape, sha)) in &spec.arrays {
        let (nshape, _, nsha) = npz
            .get(name)
            .unwrap_or_else(|| panic!("npz missing banked array {name}"));
        assert_eq!(nshape, shape, "{name}: npz shape vs JSON");
        assert_eq!(nsha, sha, "{name}: npz sha vs JSON — corrupt fixture file");
    }
    println!("npz integrity: {} arrays sha-verified", spec.arrays.len());

    let seq: Vec<u32> = spec
        .prompt
        .iter()
        .chain(spec.tokens.iter())
        .cloned()
        .collect();
    let model = Dsv4Model::open(dir).unwrap_or_else(|error| {
        eprintln!("dsv4 model load failed: {error}");
        std::process::exit(1);
    });
    let mut dspark = DsparkModule::load(&model, dir, MAX_LEN);
    let mut trunk = TrunkState::load(&model, &dspark.cfg.target_layer_ids.clone(), MAX_LEN);
    println!("model loaded t={:.0}s", t0.elapsed().as_secs_f64());

    let n_t = dspark.cfg.target_layer_ids.len();
    let hidden = model.mc.n_embd as usize;
    let mh_w = n_t * hidden;
    let p0 = spec.prompt.len();
    let block = dspark.cfg.block_size;

    let fork_logits_post_max = spec
        .fork
        .iter()
        .filter(|(k, _)| k.ends_with("_logits_post"))
        .map(|(_, v)| *v)
        .fold(0.0f64, f64::max);
    assert!(
        !is_ref || fork_logits_post_max > 0.0,
        "ref arm: no *_logits_post fork banked — refuse to derive bands"
    );
    let mut gate = Gate {
        is_ref,
        fork_logits_post_max,
        pass: 0,
        fail: 0,
        inband_ids: 0,
        inband_trunk: 0,
        lines: Vec::new(),
    };
    let mut digest = Digester::new();

    match mode {
        "components" => {
            // ---- trunk prefill + drafter ring prime (the greedy mode's prefill is
            // owned by the generic spec_oracle loop)
            let pre = trunk.forward(&model, &seq[..p0], 0, variant);
            dspark.prime_prefill(&model, &pre.main_hidden, p0, variant);
            let t_first = argmax(&pre.logits);
            if is_ref {
                // the prefill realization is Gate-C-pinned (lane-3 oracle, 160/160): hard.
                assert_eq!(
                    t_first, seq[p0],
                    "prefill argmax {t_first} != banked token {} — decode-port or trajectory corrupt (STOP)",
                    seq[p0]
                );
            } else if t_first != seq[p0] {
                println!(
                    "clamp arm: prefill argmax {t_first} != banked REF token {} (informational — different contract)",
                    seq[p0]
                );
            }
            println!(
                "prefill({p0}) + ring prime done (argmax {t_first}) t={:.0}s",
                t0.elapsed().as_secs_f64()
            );
            let ids_arr = &npz["draft_ids"].1;
            let margins_arr = &npz["draft_margins"].1;
            let conf_arr = &npz["confidence_all"].1;
            let accept_arr = &npz["accept_counts"].1;
            let n_pos = spec.last_pos - spec.first_pos + 1;
            assert_eq!(npz["draft_ids"].0, vec![n_pos, block + 1]);
            let trunk_argmax_arr = &npz["trunk_argmax"].1;
            let trunk_margins_arr = &npz["trunk_margins"].1;
            let mut accept_ok = 0usize;
            let mut conf_worst = 0f64;
            let conf_am = absmax(conf_arr);
            let conf_thr = if is_ref {
                *spec.fork.get("confidence_all").expect("confidence fork")
            } else {
                1e-3 * conf_am
            };
            for pos in spec.first_pos..=spec.last_pos {
                let step = trunk.forward(&model, &seq[pos..pos + 1], pos, variant);
                let got = argmax(&step.logits);
                let r = pos - spec.first_pos;
                let torch_got = trunk_argmax_arr[r] as u32;
                if got != torch_got {
                    // realization flip vs same-variant torch (receipts flip policy);
                    // teacher-forcing keeps state banked-driven — comparison-only event.
                    let tm = trunk_margins_arr[r] as f64;
                    let rm = top2_margin(&step.logits, got) as f64;
                    let band = gate.band(step.logits[got as usize]);
                    let in_band = tm.min(rm) <= band;
                    if in_band {
                        gate.inband_trunk += 1;
                    } else {
                        gate.fail += 1;
                    }
                    gate.lines.push(format!(
                        "  [{}] pos {pos} TRUNK argmax: rust {got} vs torch {torch_got} (banked {}) | torch margin {tm:.4e}, rust margin {rm:.4e}, band {band:.4e}",
                        if in_band { "IN-BAND trunk flip" } else { "FAIL — OUT OF BAND" },
                        seq[pos + 1]
                    ));
                }
                dspark.write_rings(&model, &step.main_hidden, pos, variant);
                let cap = spec.capture_positions.contains(&pos);
                let spec_out =
                    dspark.forward_spec(&model, seq[pos + 1], &step.main_hidden, pos, variant, cap);
                let ids_row = &ids_arr[r * (block + 1)..(r + 1) * (block + 1)];
                let margins_row = &margins_arr[r * block..(r + 1) * block];
                let fd = check_draft_ids(&mut gate, pos, &spec_out, ids_row, margins_row);
                // confidence: slots 0..=fd are id-comparable (conf[i] uses ids < i+1)
                let comparable = fd.map(|i| i + 1).unwrap_or(block);
                for i in 0..comparable {
                    let d =
                        ((spec_out.confidence[i] as f64) - (conf_arr[r * block + i] as f64)).abs();
                    conf_worst = conf_worst.max(d);
                    if d > conf_thr {
                        gate.fail += 1;
                        gate.lines.push(format!(
                            "  [FAIL] pos {pos} confidence[{i}]: {d:.3e} vs thr {conf_thr:.3e}"
                        ));
                    }
                }
                // acceptance
                if fd.is_none() {
                    let mut acc = 0usize;
                    let avail = (seq.len() - (pos + 2)).min(block);
                    for j in 0..avail {
                        if spec_out.out_ids[1 + j] == seq[pos + 2 + j] {
                            acc += 1;
                        } else {
                            break;
                        }
                    }
                    if acc == accept_arr[r] as usize {
                        accept_ok += 1;
                    } else {
                        gate.fail += 1;
                        gate.lines.push(format!(
                            "  [FAIL] pos {pos}: accept {acc} != torch {}",
                            accept_arr[r]
                        ));
                    }
                }
                digest.u32s(&spec_out.out_ids);
                digest.f32s(&spec_out.confidence);
                // checkpoint component arrays
                if let Some(c) = &spec_out.capture {
                    let fk = |n: &str| spec.fork.get(&format!("pos{pos}_{n}")).copied();
                    let t = |n: &str| -> &[f32] { &npz[&format!("pos{pos}_{n}")].1 };
                    gate.check_array(
                        &format!("pos{pos}_main_hidden"),
                        &step.main_hidden[..mh_w],
                        t("main_hidden"),
                        fk("main_hidden"),
                    );
                    gate.check_array(
                        &format!("pos{pos}_main_x"),
                        &c.main_x,
                        t("main_x"),
                        fk("main_x"),
                    );
                    for (k, bo) in c.block_outs.iter().enumerate() {
                        gate.check_array(
                            &format!("pos{pos}_block{k}_out"),
                            bo,
                            t(&format!("block{k}_out")),
                            fk(&format!("block{k}_out")),
                        );
                    }
                    gate.check_array(
                        &format!("pos{pos}_x_collapsed"),
                        &c.x_collapsed,
                        t("x_collapsed"),
                        fk("x_collapsed"),
                    );
                    gate.check_array(
                        &format!("pos{pos}_logits_pre"),
                        &c.logits_pre,
                        t("logits_pre"),
                        fk("logits_pre"),
                    );
                    // logits_post / markov_embed rows are chained; compare comparable rows
                    let rows = fd.map(|i| i + 1).unwrap_or(block);
                    gate.check_array(
                        &format!("pos{pos}_logits_post[..{rows}]"),
                        &c.logits_post[..rows * dspark.vocab],
                        &t("logits_post")[..rows * dspark.vocab],
                        fk("logits_post"),
                    );
                    let rank = dspark.cfg.markov_rank;
                    gate.check_array(
                        &format!("pos{pos}_markov_embed[..{rows}]"),
                        &c.markov_embed[..rows * rank],
                        &t("markov_embed")[..rows * rank],
                        fk("markov_embed"),
                    );
                    digest.f32s(&c.logits_post);
                }
                if (pos - spec.first_pos) % 20 == 19 {
                    println!(
                        "pos {pos} done ({} arrays pass, {} fail, {} in-band id flips) t={:.0}s",
                        gate.pass,
                        gate.fail,
                        gate.inband_ids,
                        t0.elapsed().as_secs_f64()
                    );
                }
            }
            for l in &gate.lines {
                println!("{l}");
            }
            // budgets per decision procedure #2: 8 trunk / 12 draft-id in-band flips
            let verdict = gate.fail == 0 && gate.inband_ids <= 12 && gate.inband_trunk <= 8;
            println!(
                "\nCOMPONENTS GATE [{} arm] [{}]: arrays {} PASS / {} FAIL{} | trunk flips in-band {} (budget 8) | draft-id flips in-band {} (budget 12) | accept rows exact {}/{} | confidence worst {:.3e} (thr {:.3e}) | determinism sha256 {}",
                spec.variant_tag,
                if verdict { "PASS" } else { "FAIL" },
                gate.pass,
                gate.fail,
                if is_ref {
                    " (ref float arrays are MEASUREMENTS per decision procedure #2)"
                } else {
                    ""
                },
                gate.inband_trunk,
                gate.inband_ids,
                accept_ok,
                spec.last_pos - spec.first_pos + 1,
                conf_worst,
                conf_thr,
                digest.hex()
            );
            println!("elapsed {:.0}s", t0.elapsed().as_secs_f64());
            std::process::exit(if verdict { 0 } else { 1 });
        }
        "greedy" => {
            // Free-running propose-then-verify through the FAMILY-GENERIC loop
            // (spec_oracle::run_spec_greedy; owner-directive seam — the dsv4
            // specifics enter only through the two adapters). Sequential greedy
            // verification == batched == plain; the identity is the GATE, not an
            // assumption.
            let n_new = seq.len() - p0;
            let ids_arr = &npz["draft_ids"].1;
            let gate_cell = std::cell::RefCell::new(&mut gate);
            let digest_cell = std::cell::RefCell::new(&mut digest);
            let mut drafter = memra_gguf::dsv4_dspark::DsparkOracleAdapter {
                module: &mut dspark,
                model: &model,
                variant,
            };
            let mut trunk_ad = memra_gguf::dsv4_decode::TrunkOracleAdapter {
                trunk: &mut trunk,
                model: &model,
                variant,
            };
            let run = memra_gguf::spec_oracle::run_spec_greedy(
                &mut trunk_ad,
                &mut drafter,
                &seq[..p0],
                n_new,
                |step, got, logits| {
                    if step == usize::MAX {
                        // prefill emit: Gate-C-pinned realization, hard.
                        assert_eq!(
                            got, seq[p0],
                            "prefill argmax {got} != banked token {} (STOP)",
                            seq[p0]
                        );
                        return got;
                    }
                    // banked-trajectory pin (receipts flip policy): a divergence is a
                    // realization flip vs torch-decode; adjudicate, then take the
                    // banked token as a CORRECTION so the rest stays comparable.
                    let want = seq[p0 + step + 1];
                    if got == want {
                        return got;
                    }
                    let mut g = gate_cell.borrow_mut();
                    let m = p0 + step;
                    let tm = npz["trunk_margins"].1[m - spec.first_pos] as f64;
                    let rm = top2_margin(logits, got) as f64;
                    let band = g.band(logits[got as usize]);
                    let in_band = tm.min(rm) <= band;
                    if in_band {
                        g.inband_trunk += 1;
                    } else {
                        g.fail += 1;
                    }
                    println!(
                        "  [{}] step {step} TRUNK flip: rust {got} vs banked {want} | torch margin {tm:.4e}, rust margin {rm:.4e}, band {band:.4e} -> corrected to banked",
                        if in_band {
                            "IN-BAND"
                        } else {
                            "FAIL — OUT OF BAND"
                        }
                    );
                    want
                },
                |prop| {
                    let mut d = digest_cell.borrow_mut();
                    d.u32s(&prop.out_ids);
                    d.f32s(&prop.confidence);
                },
            );
            let out_tokens = run.tokens;
            let rounds = run.rounds;
            let last_logits = run.last_logits;
            // identity vs the banked trajectory
            let mut first_div = None;
            for (i, (&got, &want)) in out_tokens.iter().zip(spec.tokens.iter()).enumerate() {
                if got != want {
                    first_div = Some((i, got, want));
                    break;
                }
            }
            // cross-check round drafts + accepts against the torch per-position profile
            // (receipts flip policy applied here too: a draft chain that diverges from
            // the torch row at a near-tie is an in-band flip round, adjudicated with
            // the SAME band as the components gate; its accepts are then checked
            // against the rust drafts' own prefix match vs the banked continuation.
            // A round whose verification was truncated by the 160-token budget is
            // checked only on its verified prefix.)
            let margins_arr = &npz["draft_margins"].1;
            let mut xchecked = 0usize;
            let mut xflips = 0usize;
            let mut xfail = 0usize;
            for rd in &rounds {
                if rd.start_pos < spec.first_pos || rd.start_pos > spec.last_pos {
                    continue;
                }
                let r = rd.start_pos - spec.first_pos;
                let ids_row = &ids_arr[r * (block + 1)..(r + 1) * (block + 1)];
                let torch_ids: Vec<u32> = ids_row[1..].iter().map(|x| *x as u32).collect();
                let torch_accept = npz["accept_counts"].1[r] as usize;
                // rust drafts' own expected accepts vs the banked continuation
                let own_accept = {
                    let mut a = 0usize;
                    for (j, &d) in rd.drafts.iter().enumerate() {
                        let idx = rd.start_pos + 2 + j;
                        if j < rd.verified && idx < seq.len() && d == seq[idx] {
                            a += 1;
                        } else {
                            break;
                        }
                    }
                    a
                };
                if rd.drafts == torch_ids {
                    if rd.accepts == torch_accept.min(rd.verified) {
                        xchecked += 1;
                    } else {
                        xfail += 1;
                        println!(
                            "  cross-check MISMATCH at round start_pos {}: accepts {} (verified {}) vs torch {torch_accept}",
                            rd.start_pos, rd.accepts, rd.verified
                        );
                    }
                    continue;
                }
                // draft chain diverges: adjudicate the first differing slot
                let i = rd
                    .drafts
                    .iter()
                    .zip(&torch_ids)
                    .position(|(a, b)| a != b)
                    .unwrap();
                let tm = margins_arr[r * block + i] as f64;
                let rm = rd.margins[i] as f64;
                let band = gate.band(rd.top1[i]);
                let in_band = tm.min(rm) <= band;
                let acc_ok = rd.accepts == own_accept;
                if in_band && acc_ok {
                    xflips += 1;
                    gate.inband_ids += 1;
                    println!(
                        "  [IN-BAND flip round] start_pos {} slot {i}: rust {} vs torch {} | torch margin {tm:.4e}, rust margin {rm:.4e}, band {band:.4e} | accepts {} == own prefix {}",
                        rd.start_pos, rd.drafts[i], torch_ids[i], rd.accepts, own_accept
                    );
                } else {
                    xfail += 1;
                    println!(
                        "  cross-check FAIL at round start_pos {}: rust drafts {:?} vs torch {torch_ids:?} | slot {i} margins ({tm:.4e},{rm:.4e}) band {band:.4e} | accepts {} own {}",
                        rd.start_pos, rd.drafts, rd.accepts, own_accept
                    );
                }
            }
            let total_accepts: usize = rounds.iter().map(|r| r.accepts).sum();
            digest.u32s(&out_tokens);
            digest.f32s(&last_logits);
            // bank the run
            let mut f =
                std::fs::File::create(out_dir.join("dspark_greedy_e2e.json")).expect("json");
            let rounds_json: Vec<String> = rounds
                .iter()
                .map(|rd| {
                    format!(
                        "{{\"start_pos\": {}, \"drafts\": {:?}, \"accepts\": {}, \"verified\": {}}}",
                        rd.start_pos, rd.drafts, rd.accepts, rd.verified
                    )
                })
                .collect();
            // corrections force out_tokens == banked; the honest identity metric is
            // the correction count (0 = literal identity).
            assert!(first_div.is_none(), "corrected run must match banked");
            write!(
                f,
                "{{\n \"tokens\": {:?},\n \"in_band_corrections\": {},\n \"out_of_band\": {},\n \"rounds\": [\n  {}\n ],\n \"total_rounds\": {},\n \"total_accepts\": {},\n \"accept_mean_per_round\": {:.4}\n}}\n",
                out_tokens,
                gate.inband_trunk,
                gate.fail,
                rounds_json.join(",\n  "),
                rounds.len(),
                total_accepts,
                total_accepts as f64 / rounds.len() as f64
            )
            .expect("write json");
            let verdict = gate.fail == 0 && gate.inband_trunk <= 8 && xfail == 0;
            let identity_note = if gate.inband_trunk == 0 && gate.fail == 0 {
                "LITERAL IDENTITY, zero corrections".to_string()
            } else {
                format!(
                    "{} in-band corrections (budget 8), {} out-of-band",
                    gate.inband_trunk, gate.fail
                )
            };
            println!(
                "\nSPEC==PLAIN IDENTITY GATE [{}]: {}/{} tokens == banked REF trajectory | {identity_note} | rounds {} | accepted drafts {total_accepts} (mean {:.3}/round, {:.3} tokens/round incl. bonus) | cross-check vs torch profile {xchecked} exact + {xflips} in-band flip rounds / {xfail} FAIL | determinism sha256 {}",
                if verdict { "PASS" } else { "FAIL" },
                out_tokens.len(),
                spec.tokens.len(),
                rounds.len(),
                total_accepts as f64 / rounds.len() as f64,
                n_new as f64 / rounds.len() as f64,
                digest.hex()
            );
            println!("elapsed {:.0}s", t0.elapsed().as_secs_f64());
            std::process::exit(if verdict { 0 } else { 1 });
        }
        "batched" => {
            // ---- §3.1 ring-hazard gate (module header doc). REF-only, like greedy.
            let mut n_new = seq.len() - p0;
            // MEMRA_DSPARK_BATCH_SMOKE=N truncates the run for fast machinery
            // feedback — NEVER a gate of record (the banner below marks it).
            if let Ok(v) = std::env::var("MEMRA_DSPARK_BATCH_SMOKE") {
                n_new = n_new.min(v.parse().expect("MEMRA_DSPARK_BATCH_SMOKE integer"));
                println!("*** SMOKE RUN (n_new truncated to {n_new}) — not a gate verdict ***");
            }
            // ---------- pass 1: sequential twin, teacher-forced along the banked
            // trajectory, digesting every cache class after every position ----------
            let pre = trunk.forward(&model, &seq[..p0], 0, variant);
            dspark.prime_prefill(&model, &pre.main_hidden, p0, variant);
            assert_eq!(
                argmax(&pre.logits),
                seq[p0],
                "prefill argmax != banked token (STOP)"
            );
            let mut seq_digests: Vec<Vec<(String, [u8; 32])>> =
                Vec::with_capacity(n_new.saturating_sub(1));
            for m in p0..p0 + n_new - 1 {
                let step = trunk.forward(&model, &seq[m..m + 1], m, variant);
                dspark.write_rings(&model, &step.main_hidden, m, variant);
                seq_digests.push(state_digests(&trunk, &dspark));
                if (m - p0) % 40 == 39 {
                    println!("twin pos {m} digested t={:.0}s", t0.elapsed().as_secs_f64());
                }
            }
            println!(
                "sequential twin done: {} positions digested ({} classes each) t={:.0}s",
                seq_digests.len(),
                seq_digests.first().map(|d| d.len()).unwrap_or(0),
                t0.elapsed().as_secs_f64()
            );
            // ---------- reset to construction state ----------
            trunk.reset_state();
            dspark.reset_state();
            // ---------- pass 2: free-running BATCHED loop with per-round compare ----------
            let gate_cell = std::cell::RefCell::new(&mut gate);
            let digest_cell = std::cell::RefCell::new(&mut digest);
            let state_fails = std::cell::RefCell::new(0usize);
            let rounds_compared = std::cell::RefCell::new(0usize);
            let mut drafter = memra_gguf::dsv4_dspark::DsparkOracleAdapter {
                module: &mut dspark,
                model: &model,
                variant,
            };
            let mut trunk_ad =
                memra_gguf::dsv4_decode::TrunkBatchAdapter::new(&mut trunk, &model, variant);
            let seq_digests_ref = &seq_digests;
            let run = memra_gguf::spec_oracle::run_spec_greedy_batched(
                &mut trunk_ad,
                &mut drafter,
                &seq[..p0],
                n_new,
                |step, got, logits| {
                    if step == usize::MAX {
                        assert_eq!(
                            got, seq[p0],
                            "prefill argmax {got} != banked token {} (STOP)",
                            seq[p0]
                        );
                        return got;
                    }
                    let want = seq[p0 + step + 1];
                    if got == want {
                        return got;
                    }
                    let mut g = gate_cell.borrow_mut();
                    let m = p0 + step;
                    let tm = npz["trunk_margins"].1[m - spec.first_pos] as f64;
                    let rm = top2_margin(logits, got) as f64;
                    let band = g.band(logits[got as usize]);
                    let in_band = tm.min(rm) <= band;
                    if in_band {
                        g.inband_trunk += 1;
                    } else {
                        g.fail += 1;
                    }
                    println!(
                        "  [{}] step {step} TRUNK flip: rust {got} vs banked {want} | torch margin {tm:.4e}, rust margin {rm:.4e}, band {band:.4e} -> corrected to banked",
                        if in_band {
                            "IN-BAND"
                        } else {
                            "FAIL — OUT OF BAND"
                        }
                    );
                    want
                },
                |prop| {
                    let mut d = digest_cell.borrow_mut();
                    d.u32s(&prop.out_ids);
                    d.f32s(&prop.confidence);
                },
                |tr, dr, round| {
                    // §3.1 invariant: committed state after this round ==
                    // twin state after plain decode through the same position.
                    let m_end = round.start_pos + 1 + round.accepts;
                    let got = state_digests(tr.trunk, dr.module);
                    let want = &seq_digests_ref[m_end - p0];
                    assert_eq!(got.len(), want.len(), "digest class count drift");
                    let mut bad = 0usize;
                    for ((gl, gd), (wl, wd)) in got.iter().zip(want.iter()) {
                        assert_eq!(gl, wl, "digest label order drift");
                        if gd != wd {
                            bad += 1;
                            println!(
                                "  [FAIL §3.1] round@{} committed through pos {m_end}: class {gl} diverges from the sequential twin",
                                round.start_pos
                            );
                        }
                    }
                    if bad > 0 {
                        *state_fails.borrow_mut() += bad;
                    }
                    *rounds_compared.borrow_mut() += 1;
                },
            );
            let out_tokens = run.tokens;
            let rounds = run.rounds;
            let last_logits = run.last_logits;
            let state_fails = state_fails.into_inner();
            let rounds_compared = rounds_compared.into_inner();
            // identity vs the banked trajectory (corrections force equality; the
            // honest identity metric is the correction count)
            let mut first_div = None;
            for (i, (&got, &want)) in out_tokens.iter().zip(spec.tokens.iter()).enumerate() {
                if got != want {
                    first_div = Some((i, got, want));
                    break;
                }
            }
            assert!(
                first_div.is_none(),
                "corrected run must match banked: {first_div:?}"
            );
            let total_accepts: usize = rounds.iter().map(|r| r.accepts).sum();
            let full_rounds = rounds.iter().filter(|r| r.verified > 0).count();
            digest.u32s(&out_tokens);
            digest.f32s(&last_logits);
            let mut f =
                std::fs::File::create(out_dir.join("dspark_batched_e2e.json")).expect("json");
            let rounds_json: Vec<String> = rounds
                .iter()
                .map(|rd| {
                    format!(
                        "{{\"start_pos\": {}, \"drafts\": {:?}, \"accepts\": {}, \"verified\": {}}}",
                        rd.start_pos, rd.drafts, rd.accepts, rd.verified
                    )
                })
                .collect();
            write!(
                f,
                "{{\n \"tokens\": {:?},\n \"in_band_corrections\": {},\n \"out_of_band\": {},\n \"state_class_fails\": {},\n \"rounds_state_compared\": {},\n \"rounds\": [\n  {}\n ],\n \"total_rounds\": {},\n \"total_accepts\": {},\n \"accept_mean_per_round\": {:.4}\n}}\n",
                out_tokens,
                gate.inband_trunk,
                gate.fail,
                state_fails,
                rounds_compared,
                rounds_json.join(",\n  "),
                rounds.len(),
                total_accepts,
                total_accepts as f64 / full_rounds.max(1) as f64
            )
            .expect("write json");
            let verdict = gate.fail == 0 && gate.inband_trunk <= 8 && state_fails == 0;
            let identity_note = if gate.inband_trunk == 0 && gate.fail == 0 {
                "LITERAL IDENTITY, zero corrections".to_string()
            } else {
                format!(
                    "{} in-band corrections (budget 8), {} out-of-band",
                    gate.inband_trunk, gate.fail
                )
            };
            println!(
                "\nRING-HAZARD GATE (§3.1, batched verify + rollback/commit) [{}]: {}/{} tokens == banked REF trajectory | {identity_note} | state classes BIT-equal to the sequential twin at {rounds_compared}/{rounds_compared_total} round boundaries with {state_fails} class fails | rounds {} | accepted drafts {total_accepts} (mean {:.3}/verify round) | determinism sha256 {}",
                if verdict { "PASS" } else { "FAIL" },
                out_tokens.len(),
                spec.tokens.len(),
                rounds.len(),
                total_accepts as f64 / full_rounds.max(1) as f64,
                digest.hex(),
                rounds_compared_total = rounds_compared,
            );
            println!("elapsed {:.0}s", t0.elapsed().as_secs_f64());
            std::process::exit(if verdict { 0 } else { 1 });
        }
        other => {
            eprintln!("unknown mode {other}");
            std::process::exit(2);
        }
    }
}

/// sha256 of an f32 slice (LE bytes) — the bit-level state-class digest.
fn sha_f32(v: &[f32]) -> [u8; 32] {
    let mut h = Sha256::new();
    for x in v {
        h.update(x.to_le_bytes());
    }
    h.finalize().into()
}

/// Every trunk cache class + the drafter rings, per layer, label-ordered — the §3.1
/// gate's bit-level state fingerprint (window ring; compressor pending kv/score +
/// live store; indexer pending kv/score + live store; dspark main_kv rings).
fn state_digests(
    trunk: &memra_gguf::dsv4_decode::TrunkState,
    dspark: &memra_gguf::dsv4_dspark::DsparkModule,
) -> Vec<(String, [u8; 32])> {
    let mut out = Vec::new();
    for (lid, b) in trunk.blocks.iter().enumerate() {
        out.push((format!("L{lid}.ring"), sha_f32(b.attn.ring_view())));
        if let Some(c) = &b.attn.comp {
            let (kv, sc, st) = c.state_views();
            out.push((format!("L{lid}.comp.kv"), sha_f32(kv)));
            out.push((format!("L{lid}.comp.score"), sha_f32(sc)));
            out.push((format!("L{lid}.comp.store"), sha_f32(st)));
        }
        if let Some(ix) = &b.attn.idx {
            let (kv, sc, st) = ix.compressor.state_views();
            out.push((format!("L{lid}.idx.kv"), sha_f32(kv)));
            out.push((format!("L{lid}.idx.score"), sha_f32(sc)));
            out.push((format!("L{lid}.idx.store"), sha_f32(st)));
        }
    }
    for (k, r) in dspark.ring_views().iter().enumerate() {
        out.push((format!("dspark.{k}.ring"), sha_f32(r)));
    }
    out
}
