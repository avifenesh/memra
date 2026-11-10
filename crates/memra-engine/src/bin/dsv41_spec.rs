//! DeepSeek-V4.1-Flash DSpark drafting: the gates and measurements of the draft-and-verify
//! decode loop over `dsv41_gpu::decode` and `dsv41_gpu::draft`.
//!
//! usage: dsv41-spec CKPT_DIR --plan FILE --out DIR [--token-map FILE] [--devices 0,1]
//!                   [--streamed] [--cache-slots N] [--cache-reserve-mb N] [--eos ID]
//!                   [--layers N --resident] [--no-draft] [--copy-keys] [--batch-max N]
//!   MEMRA_FULL_PREC=1        required: the route has one numeric program, the checkpoint's
//!   MEMRA_DSV41_TOKEN_MAP=F  the Engram compressed token map (raw i32), when --token-map is absent
//!   --streamed               stream every layer's weights per forward (slow; default is the
//!                            serving route's one-card residency: the layers' own weights and the
//!                            draft head on the card, the routed experts in pinned host memory
//!                            behind the device cache)
//!   --layers N --resident    a shortened trunk with every weight on the card (the routed experts
//!                            too): what a row costs where no expert is fetched. It has no draft
//!                            head (the head's target layers are not all run): pass --no-draft
//!   --no-draft               do not load the draft head (verify-rows and rows-time jobs only)
//!   --copy-keys              decode calls copy the ring and the compressed rows into the call's
//!                            own key rows (the reference arm of the keyed attention entry);
//!                            verify-rows and batch-rows write `hashes.txt`, equal between the arms
//!
//! The plan lists jobs, one per line, all run in one process (the weights load once):
//!   verify-rows ID PROMPT.i32 FORCE.i32
//!       The byte gate of the several-token decode call. Arm A feeds the forced tokens one decode
//!       call at a time. Arm B feeds them as verification blocks of 2..6 rows; every second block
//!       has wrong tokens past a kept prefix and is committed at that prefix (a rejection). Every
//!       kept row's logits and draft-head inputs must equal arm A's row of the same position byte
//!       for byte, whatever was rejected before it.
//!   draft-tape ID TAPE_DIR
//!       The draft head on a vendor tape's own inputs (`export_tape.py`): seed the rings from the
//!       prompt's main_hidden, then per step write the step's ring row and draft from its token.
//!       Writes memra's tensors next to the tape's names for `draft_cmp.py`. Every step's draft
//!       runs again untraced (on one rank its picks chain on the device) and must name the
//!       same tokens and confidences. Then the last step's draft is timed, with its launches
//!       by stage and kernel and each kernel's own time (a card sync after every launch).
//!   gen ID PROMPT.i32 N [DEPTH [TAU]]
//!       Plain greedy generation of N tokens, then drafted greedy generation of the same prompt:
//!       the tokens and the logits behind every token must be identical, and the device
//!       reduction of every row (`Runner::argmax_rows`) must name the host argmax's token. Prints acceptance and
//!       the time of a one-token step, a draft and a verification step. DEPTH caps the drafts
//!       verified per step (default: the block); TAU verifies only the leading drafts whose
//!       confidence sigmoid is at least TAU.
//!   rows-time ID PROMPT.i32 FORCE.i32 [REPS]
//!       What a decode call costs by its rows. After the prefill, REPS one-token calls (the plain
//!       step), then for 2..8 rows the same block REPS times, dropped whole after each run: the
//!       first run of a block fetches its experts where they live off the card, the later ones
//!       find them (the rows' compute alone). Each width runs twice: a block of different tokens
//!       (its rows route to up to rows x top-k experts a layer) and a block of one token repeated
//!       (its rows route alike), which separates the experts a block reads from its other
//!       per-row work.
//!   batch-rows ID PROMPT.i32 FORCE.i32 [PROMPT.i32 FORCE.i32 ...]
//!       The byte gate of a decode call over several sequences. Arm A runs every sequence alone
//!       (its prompt, then its forced tokens one call at a time). Arm B prefills the sequences
//!       and then steps them together, one batched call per step (eight rows at most a call; a
//!       sequence leaves the batch when its forced tokens end, so the widths mix; --batch-max N
//!       raises the eight: wider calls run the GEMMs' general bodies). Every row's
//!       logits and taps must equal arm A's row of the same sequence and step, and the device
//!       argmax of the rows must name the host argmax's tokens.
//!   batch-time ID REPS PROMPT.i32 FORCE.i32 [PROMPT.i32 FORCE.i32 ...]
//!       What a decode step costs by the sequences it carries: for 1, 2, 4 and 8 sequences, REPS
//!       batched calls (forward and the rows' logits), and the same steps with every sequence's
//!       call run alone through each block (one forward of that many calls); also the batched
//!       call with the rows' device argmax (a served greedy step) and with no head, and past
//!       8 sequences the step as calls of 8 rows (`chunk8`) against the one wide call. Then the
//!       launches of two steps by layer and kernel for one sequence and for the widest batch,
//!       and each kernel's own time (a card sync after every launch).
//! Every job prints one JSON line and writes its files under --out/ID.

use memra_engine::dsv41_gpu::{
    self as v, Buf, Geo,
    decode::{Call, Out, Runner, Seq, TOPK, argmax, lse, top_k},
    draft::{DraftSeq, DraftW, accept_walk},
};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("dsv41-spec: {e}");
        std::process::exit(1);
    }
}

fn read_i32(f: &Path) -> Result<Vec<u32>, String> {
    let raw = std::fs::read(f).map_err(|e| format!("{}: {e}", f.display()))?;
    if raw.len() % 4 != 0 {
        return Err(format!("{}: not an i32 array", f.display()));
    }
    Ok(raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
        .collect())
}

fn write(dir: &Path, name: &str, b: &[u8]) -> Result<(), String> {
    std::fs::write(dir.join(name), b).map_err(|e| format!("{}/{name}: {e}", dir.display()))
}

fn f32_le(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i32_le(x: &[i32]) -> Vec<u8> {
    x.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// FNV-1a over the bytes of a logits row.
fn fnv(x: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in x {
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn fnv_bytes(x: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in x {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// One sequence of a batch job: its prompt and its forced tokens.
type SeqIn = (Vec<u32>, Vec<u32>);

enum Job {
    VerifyRows {
        id: String,
        prompt: Vec<u32>,
        force: Vec<u32>,
    },
    DraftTape {
        id: String,
        dir: PathBuf,
    },
    Gen {
        id: String,
        prompt: Vec<u32>,
        n: usize,
        depth: Option<usize>,
        tau: Option<f32>,
    },
    RowsTime {
        id: String,
        prompt: Vec<u32>,
        force: Vec<u32>,
        reps: usize,
    },
    BatchRows {
        id: String,
        seqs: Vec<SeqIn>,
    },
    BatchTime {
        id: String,
        reps: usize,
        seqs: Vec<SeqIn>,
    },
}

impl Job {
    fn id(&self) -> &str {
        match self {
            Job::VerifyRows { id, .. }
            | Job::DraftTape { id, .. }
            | Job::Gen { id, .. }
            | Job::RowsTime { id, .. }
            | Job::BatchRows { id, .. }
            | Job::BatchTime { id, .. } => id,
        }
    }
    /// the positions the job reaches
    fn positions(&self) -> Result<usize, String> {
        Ok(match self {
            Job::VerifyRows { prompt, force, .. } => prompt.len() + force.len(),
            Job::DraftTape { dir, .. } => {
                let pos = read_i32(&dir.join("pos.i32")).unwrap_or_default();
                let p = read_i32(&dir.join("prompt.i32"))?;
                pos.iter()
                    .map(|&x| x as usize)
                    .max()
                    .unwrap_or(0)
                    .max(p.len())
                    + 1
            }
            Job::Gen { prompt, n, .. } => prompt.len() + n,
            Job::RowsTime { prompt, force, .. } => prompt.len() + force.len(),
            Job::BatchRows { seqs, .. } | Job::BatchTime { seqs, .. } => seqs
                .iter()
                .map(|(p, f)| p.len() + f.len())
                .max()
                .unwrap_or(0),
        })
    }
}

fn run() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let ckpt = PathBuf::from(
        a.get(1)
            .ok_or("usage: dsv41-spec CKPT_DIR --plan FILE --out DIR ...")?,
    );
    let flag = |k: &str| {
        a.iter()
            .position(|x| x == k)
            .and_then(|i| a.get(i + 1))
            .cloned()
    };
    let num = |k: &str| -> Result<Option<usize>, String> {
        flag(k)
            .map(|x| x.parse().map_err(|_| format!("{k} {x:?}: not a count")))
            .transpose()
    };
    if std::env::var("MEMRA_FULL_PREC").as_deref() != Ok("1") {
        return Err("set MEMRA_FULL_PREC=1: the deepseek_v41 route runs only the checkpoint's own numeric program".into());
    }
    let tmap = flag("--token-map")
        .or_else(|| std::env::var("MEMRA_DSV41_TOKEN_MAP").ok())
        .ok_or("the Engram hash needs the compressed token map: --token-map FILE or MEMRA_DSV41_TOKEN_MAP")?;
    let plan = flag("--plan").ok_or("--plan FILE lists the jobs")?;
    let out = PathBuf::from(flag("--out").ok_or("--out DIR")?);
    let devices: Vec<usize> = match flag("--devices") {
        Some(x) => x
            .split(',')
            .map(|d| d.trim().parse().map_err(|_| format!("--devices {x:?}")))
            .collect::<Result<_, _>>()?,
        None => vec![0],
    };
    let cfg_path = ckpt.join("config.json");
    let mc =
        memra_gguf::config::ModelConfig::from_config_json(&cfg_path).map_err(|e| e.to_string())?;
    let g = Geo::from_config(&mc)?;
    let eos: u32 = match flag("--eos") {
        Some(x) => x.parse().map_err(|_| format!("--eos {x:?}"))?,
        None => {
            let txt = std::fs::read_to_string(&cfg_path).map_err(|e| e.to_string())?;
            let j: serde_json::Value = serde_json::from_str(&txt).map_err(|e| e.to_string())?;
            j["eos_token_id"]
                .as_u64()
                .ok_or("config.json has no eos_token_id: pass --eos")? as u32
        }
    };
    let trunk =
        g.compress_ratios.len() - mc.dsv41.as_ref().unwrap().num_nextn_predict_layers as usize;
    let n_layers = num("--layers")?.unwrap_or(trunk).min(trunk);
    let no_draft = a.iter().any(|x| x == "--no-draft");

    let pairs = |f: &[&str]| -> Result<Vec<SeqIn>, String> {
        f.chunks(2)
            .map(|c| Ok((read_i32(Path::new(c[0]))?, read_i32(Path::new(c[1]))?)))
            .collect()
    };
    let mut jobs = Vec::new();
    let txt = std::fs::read_to_string(&plan).map_err(|e| format!("{plan}: {e}"))?;
    for line in txt
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        let bad = || format!("{plan}: {line:?}: not a job");
        jobs.push(match (f[0], f.len()) {
            ("verify-rows", 4) => Job::VerifyRows {
                id: f[1].into(),
                prompt: read_i32(Path::new(f[2]))?,
                force: read_i32(Path::new(f[3]))?,
            },
            ("draft-tape", 3) => Job::DraftTape {
                id: f[1].into(),
                dir: PathBuf::from(f[2]),
            },
            ("gen", 4..=6) => Job::Gen {
                id: f[1].into(),
                prompt: read_i32(Path::new(f[2]))?,
                n: f[3].parse().map_err(|_| bad())?,
                depth: f.get(4).map(|x| x.parse().map_err(|_| bad())).transpose()?,
                tau: f.get(5).map(|x| x.parse().map_err(|_| bad())).transpose()?,
            },
            ("rows-time", 4..=5) => Job::RowsTime {
                id: f[1].into(),
                prompt: read_i32(Path::new(f[2]))?,
                force: read_i32(Path::new(f[3]))?,
                reps: f
                    .get(4)
                    .map(|x| x.parse().map_err(|_| bad()))
                    .transpose()?
                    .unwrap_or(8),
            },
            ("batch-rows", n) if n >= 4 && n % 2 == 0 => Job::BatchRows {
                id: f[1].into(),
                seqs: pairs(&f[2..])?,
            },
            ("batch-time", n) if n >= 5 && n % 2 == 1 => Job::BatchTime {
                id: f[1].into(),
                reps: f[2].parse().map_err(|_| bad())?,
                seqs: pairs(&f[3..])?,
            },
            _ => return Err(bad()),
        });
    }
    if jobs.is_empty() {
        return Err(format!("{plan}: no jobs"));
    }
    let block = mc.dsv41.as_ref().unwrap().dspark_block_size as usize;
    let mut max_pos = 1024;
    for j in &jobs {
        max_pos = max_pos.max(j.positions()? + block + 16);
    }

    let t0 = Instant::now();
    let stm = memra_gguf::safetensors::StModel::open(&ckpt).map_err(|e| e.to_string())?;
    let ec =
        v::engram::EngramConsts::derive(&g, v::engram::token_map_from_file(Path::new(&tmap))?)?;
    let mut rn = Runner::new(
        &stm,
        g.clone(),
        Some(ec),
        &devices,
        max_pos,
        n_layers,
        0,
        8192,
    )?;
    if a.iter().any(|x| x == "--copy-keys") {
        for c in rn.grp.cx.iter_mut() {
            c.keyed = false;
        }
    }
    let dw: Option<DraftW>;
    let load = |rn: &mut Runner| -> Result<Option<DraftW>, String> {
        if no_draft {
            Ok(None)
        } else {
            DraftW::load(rn, &mc).map(Some)
        }
    };
    if a.iter().any(|x| x == "--resident") {
        // every weight of the (shortened) trunk on the card
        let rep = rn.load_resident(n_layers, 8 << 30)?;
        eprintln!(
            "dsv41-spec: {} layers resident in {:.1}s",
            rep.layers, rep.load_s
        );
        dw = load(&mut rn)?;
    } else if a.iter().any(|x| x == "--streamed") {
        dw = load(&mut rn)?;
    } else {
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
        let ctx = rn.grp.cx[0].ops.st.context().clone();
        let host = std::sync::Arc::new(v::experts::HostExperts::fill(
            &stm, &g, n_layers, &ctx, threads,
        )?);
        eprintln!(
            "dsv41-spec: routed experts of {n_layers} layers in pinned host memory: {} bytes, filled in {:.1}s on {threads} threads",
            host.bytes, host.fill_s
        );
        let resident = rn.experts_off_card(host)?;
        // the draft head goes on the card before the cache takes what is left
        dw = load(&mut rn)?;
        let reserve = num("--cache-reserve-mb")?.unwrap_or(16384) as u64 * (1 << 20);
        let slots = rn.grow_caches(reserve, num("--cache-slots")?.unwrap_or(usize::MAX))?;
        eprintln!(
            "dsv41-spec: resident trunk weights {resident:?} bytes per rank; expert cache {slots:?} slots per rank, {reserve} bytes reserved"
        );
    }
    eprintln!(
        "dsv41-spec: {} job(s), {n_layers} layers + {} draft stages, {} rank(s) on devices {devices:?}, up in {:.1}s",
        jobs.len(),
        dw.as_ref().map_or(0, |d| d.cfg.stages),
        rn.grp.n(),
        t0.elapsed().as_secs_f64()
    );
    let bmax = num("--batch-max")?.unwrap_or(BATCH_ROWS).max(1);
    let need_dw = || {
        dw.as_ref()
            .ok_or("the job needs the draft head (no --no-draft)".to_string())
    };

    let mut failed = 0;
    for j in &jobs {
        let dir = out.join(j.id());
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let t = Instant::now();
        let r = match j {
            Job::VerifyRows { id, prompt, force } => verify_rows(&mut rn, id, prompt, force, &dir),
            Job::DraftTape { id, dir: tape } => {
                need_dw().and_then(|dw| draft_tape(&mut rn, dw, id, tape, &dir))
            }
            Job::Gen {
                id,
                prompt,
                n,
                depth,
                tau,
            } => need_dw()
                .and_then(|dw| generate(&mut rn, dw, id, prompt, *n, eos, *depth, *tau, &dir)),
            Job::RowsTime {
                id,
                prompt,
                force,
                reps,
            } => rows_time(&mut rn, id, prompt, force, *reps),
            Job::BatchRows { id, seqs } => batch_rows(&mut rn, id, seqs, bmax, &dir),
            Job::BatchTime { id, reps, seqs } => batch_time(&mut rn, id, *reps, seqs),
        };
        match r {
            Ok((pass, line)) => {
                if !pass {
                    failed += 1;
                }
                println!("{line}");
                write(&dir, "result.json", line.as_bytes())?;
            }
            Err(e) => {
                failed += 1;
                println!(
                    "{}",
                    serde_json::json!({"id": j.id(), "pass": false, "error": e})
                );
            }
        }
        eprintln!(
            "dsv41-spec: {} done in {:.1}s",
            j.id(),
            t.elapsed().as_secs_f64()
        );
    }
    let (cs, slots) = rn.cache_stats();
    eprintln!("dsv41-spec: expert cache {cs:?}, slots {slots:?}");
    println!("SPEC-DONE jobs {} failed {failed}", jobs.len());
    if failed > 0 {
        return Err(format!("{failed} job(s) failed"));
    }
    Ok(())
}

/// The first tap row of a call as rank 0 holds it (bf16 bytes of `rows` rows from row `r0`);
/// empty when the runner keeps no taps.
fn tap_bytes(rn: &Runner, o: &Out, r0: usize, rows: usize) -> Result<Vec<u8>, String> {
    if o.taps.is_empty() {
        return Ok(Vec::new());
    }
    let st = rn.grp.cx[0].ops.st.clone();
    let all = o.taps[0].host(&st)?;
    let w = all.len() / o.tap_rows;
    let a = (r0 - o.tap_row0) * w;
    Ok(all[a..a + rows * w].to_vec())
}

fn verify_rows(
    rn: &mut Runner,
    id: &str,
    prompt: &[u32],
    force: &[u32],
    dir: &Path,
) -> Result<(bool, String), String> {
    let v_ = rn.g.vocab;
    let n = force.len();
    if n < 8 {
        return Err("verify-rows wants at least 8 forced tokens".into());
    }
    // arm A: the prompt, then every forced token but the last as its own decode call
    let mut seqs = vec![rn.seq_with_capacity(prompt.len() + n + 16)];
    let mut calls = vec![Call {
        seq: 0,
        tokens: prompt.to_vec(),
    }];
    for &t in &force[..n - 1] {
        calls.push(Call {
            seq: 0,
            tokens: vec![t],
        });
    }
    let t0 = Instant::now();
    let outs = rn.forward(&mut seqs, &calls)?;
    let mut a_hash = Vec::with_capacity(n);
    let mut a_tap = Vec::with_capacity(n);
    let mut a_arg = Vec::with_capacity(n);
    for o in &outs {
        let lg = rn.logits(o, o.s - 1, 1)?;
        a_hash.push(fnv(&lg));
        a_arg.push(argmax(&lg) as u32);
        a_tap.push(fnv_bytes(&tap_bytes(rn, o, o.s - 1, 1)?));
    }
    drop(outs);
    let a_s = t0.elapsed().as_secs_f64();
    // arm B: the prompt alone, then blocks
    let mut seqs = vec![rn.seq_with_capacity(prompt.len() + n + 16)];
    let outs = rn.forward(
        &mut seqs,
        &[Call {
            seq: 0,
            tokens: prompt.to_vec(),
        }],
    )?;
    let lg = rn.logits(&outs[0], outs[0].s - 1, 1)?;
    let mut rows = 1usize;
    let mut equal = usize::from(fnv(&lg) == a_hash[0]);
    let mut taps_equal =
        usize::from(fnv_bytes(&tap_bytes(rn, &outs[0], outs[0].s - 1, 1)?) == a_tap[0]);
    let mut first_bad: Option<String> = None;
    let (mut blocks, mut rejected_rows) = (0usize, 0usize);
    let mut k = 0usize; // the next forced token to feed (arm A's step k + 1 scores it)
    let t1 = Instant::now();
    let mut log = String::new();
    while k < n - 1 {
        let width = (2 + blocks % 5).min(n - 1 - k);
        // every second block rejects: wrong tokens past a kept prefix
        let keep = if blocks % 2 == 1 && width > 1 {
            1 + (blocks / 2) % (width - 1)
        } else {
            width
        };
        let mut toks: Vec<u32> = force[k..k + width].to_vec();
        for t in toks.iter_mut().skip(keep) {
            *t = (*t + 1) % v_ as u32;
        }
        let outs = rn.forward(
            &mut seqs,
            &[Call {
                seq: 0,
                tokens: toks,
            }],
        )?;
        let o = &outs[0];
        let lg = rn.logits(o, 0, width)?;
        let tb = tap_bytes(rn, o, 0, width)?;
        let tw = tb.len() / width;
        for i in 0..keep {
            let step = k + 1 + i;
            let h = fnv(&lg[i * v_..(i + 1) * v_]);
            rows += 1;
            if h == a_hash[step] {
                equal += 1;
            } else if first_bad.is_none() {
                first_bad = Some(format!(
                    "step {step} (block {blocks}, row {i} of {width}, position {}): argmax {} vs {}",
                    o.sp + i,
                    argmax(&lg[i * v_..(i + 1) * v_]),
                    a_arg[step]
                ));
            }
            if fnv_bytes(&tb[i * tw..(i + 1) * tw]) == a_tap[step] {
                taps_equal += 1;
            }
        }
        if width > 1 {
            rn.commit(&mut seqs[0], keep)?;
        }
        log.push_str(&format!("{blocks}\t{}\t{width}\t{keep}\n", o.sp));
        rejected_rows += width - keep;
        k += keep;
        blocks += 1;
    }
    let b_s = t1.elapsed().as_secs_f64();
    write(dir, "blocks.tsv", log.as_bytes())?;
    // every one-token step's logits and taps, for a comparison across binaries or arms
    let hashes: String = a_hash
        .iter()
        .zip(&a_tap)
        .map(|(h, t)| format!("{h:016x} {t:016x}\n"))
        .collect();
    write(dir, "hashes.txt", hashes.as_bytes())?;
    let pass = equal == rows && taps_equal == rows;
    Ok((
        pass,
        serde_json::json!({
            "job": "verify-rows", "id": id, "pass": pass, "prompt": prompt.len(), "steps": n,
            "rows": rows, "logit_rows_equal": equal, "tap_rows_equal": taps_equal,
            "blocks": blocks, "rejected_rows": rejected_rows, "first_bad": first_bad,
            "arm_a_s": a_s, "arm_b_s": b_s,
        })
        .to_string(),
    ))
}

/// The most rows a batched call carries unless `--batch-max` says otherwise (the small-M GEMM
/// bodies' limit).
const BATCH_ROWS: usize = 8;

/// The byte gate of the decode call over several sequences (see the module doc).
fn batch_rows(
    rn: &mut Runner,
    id: &str,
    input: &[SeqIn],
    bmax: usize,
    dir: &Path,
) -> Result<(bool, String), String> {
    let v_ = rn.g.vocab;
    let ns = input.len();
    if input.iter().any(|(p, f)| p.is_empty() || f.len() < 2) {
        return Err("batch-rows wants a prompt and at least 2 forced tokens per sequence".into());
    }
    let cap = |p: &Vec<u32>, f: &Vec<u32>| p.len() + f.len() + 16;
    // arm A: every sequence alone
    let mut a_hash: Vec<Vec<u64>> = Vec::with_capacity(ns);
    let mut a_tap: Vec<Vec<u64>> = Vec::with_capacity(ns);
    let ta = Instant::now();
    for (p, f) in input {
        let mut seqs = vec![rn.seq_with_capacity(cap(p, f))];
        let mut calls = vec![Call {
            seq: 0,
            tokens: p.clone(),
        }];
        for &t in &f[..f.len() - 1] {
            calls.push(Call {
                seq: 0,
                tokens: vec![t],
            });
        }
        let outs = rn.forward(&mut seqs, &calls)?;
        let (mut hs, mut ts) = (Vec::new(), Vec::new());
        for o in &outs {
            hs.push(fnv(&rn.logits(o, o.s - 1, 1)?));
            ts.push(fnv_bytes(&tap_bytes(rn, o, o.s - 1, 1)?));
        }
        a_hash.push(hs);
        a_tap.push(ts);
    }
    let a_s = ta.elapsed().as_secs_f64();
    let mut hashes = String::new();
    for (hs, ts) in a_hash.iter().zip(&a_tap) {
        for (h, t) in hs.iter().zip(ts) {
            hashes.push_str(&format!("{h:016x} {t:016x}\n"));
        }
    }
    write(dir, "hashes.txt", hashes.as_bytes())?;
    // arm B: the prompts (each call alone), then the sequences together
    let tb = Instant::now();
    let mut seqs: Vec<Seq> = input
        .iter()
        .map(|(p, f)| rn.seq_with_capacity(cap(p, f)))
        .collect();
    let calls: Vec<Call> = input
        .iter()
        .enumerate()
        .map(|(i, (p, _))| Call {
            seq: i,
            tokens: p.clone(),
        })
        .collect();
    let outs = rn.forward(&mut seqs, &calls)?;
    let (mut rows_n, mut equal, mut taps_equal, mut dev_equal) = (0usize, 0usize, 0usize, 0usize);
    for (i, o) in outs.iter().enumerate() {
        rows_n += 1;
        equal += usize::from(fnv(&rn.logits(o, o.s - 1, 1)?) == a_hash[i][0]);
        taps_equal += usize::from(fnv_bytes(&tap_bytes(rn, o, o.s - 1, 1)?) == a_tap[i][0]);
        dev_equal += 1;
    }
    drop(outs);
    let mut first_bad: Option<String> = None;
    let mut widths = vec![0usize; bmax + 1];
    let steps = input.iter().map(|(_, f)| f.len()).max().unwrap_or(0);
    for k in 1..steps {
        let live: Vec<(usize, u32)> = input
            .iter()
            .enumerate()
            .filter(|(_, (_, f))| k < f.len())
            .map(|(i, (_, f))| (i, f[k - 1]))
            .collect();
        for chunk in live.chunks(bmax) {
            let o = rn.forward_batch(&mut seqs, chunk)?;
            let n = chunk.len();
            widths[n] += 1;
            let lg = rn.logits(&o, 0, n)?;
            let tb_ = tap_bytes(rn, &o, 0, n)?;
            let tw = tb_.len() / n;
            let dev = rn.argmax_rows(&o, 0, n)?;
            for (r, &(si, _)) in chunk.iter().enumerate() {
                let row = &lg[r * v_..(r + 1) * v_];
                rows_n += 1;
                if fnv(row) == a_hash[si][k] {
                    equal += 1;
                } else if first_bad.is_none() {
                    first_bad = Some(format!(
                        "sequence {si} step {k} (row {r} of {n}, position {})",
                        seqs[si].len() - 1
                    ));
                }
                taps_equal += usize::from(fnv_bytes(&tb_[r * tw..(r + 1) * tw]) == a_tap[si][k]);
                dev_equal += usize::from(dev[r] == argmax(row) as u32);
            }
        }
    }
    let b_s = tb.elapsed().as_secs_f64();
    let pass = equal == rows_n && taps_equal == rows_n && dev_equal == rows_n;
    Ok((
        pass,
        serde_json::json!({
            "job": "batch-rows", "id": id, "pass": pass, "sequences": ns,
            "prompts": input.iter().map(|(p, _)| p.len()).collect::<Vec<_>>(),
            "steps": input.iter().map(|(_, f)| f.len()).collect::<Vec<_>>(),
            "rows": rows_n, "logit_rows_equal": equal, "tap_rows_equal": taps_equal,
            "device_argmax_equal": dev_equal, "calls_by_width": widths, "first_bad": first_bad,
            "arm_a_s": a_s, "arm_b_s": b_s,
        })
        .to_string(),
    ))
}

/// What a decode step costs by the sequences it carries (see the module doc).
fn batch_time(
    rn: &mut Runner,
    id: &str,
    reps: usize,
    input: &[SeqIn],
) -> Result<(bool, String), String> {
    let reps = reps.max(4);
    if input.iter().any(|(p, f)| p.is_empty() || f.len() < reps) {
        return Err(format!(
            "batch-time wants {reps} forced tokens per sequence"
        ));
    }
    let med = |x: &[f64]| -> f64 {
        let mut v = x.to_vec();
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    let mut out = serde_json::Map::new();
    for w in [1usize, 2, 4, 8, 16, 32] {
        if w > input.len() {
            break;
        }
        // batched: one call and the logits of every row; alone: each sequence its own call;
        // argmax: one call and the device argmax of every row (a served greedy step); fwd: one
        // call and a wait for the card, no head; chunk8 (past 8 sequences): the step as calls
        // of 8 rows, each with its device argmax, against the one wide call of `argmax`
        for mode in ["batched", "alone", "argmax", "fwd", "chunk8"] {
            if mode == "chunk8" && w <= BATCH_ROWS {
                continue;
            }
            let batched = mode != "alone";
            let mut seqs: Vec<Seq> = input[..w]
                .iter()
                .map(|(p, f)| rn.seq_with_capacity(p.len() + f.len() + 16))
                .collect();
            let calls: Vec<Call> = input[..w]
                .iter()
                .enumerate()
                .map(|(i, (p, _))| Call {
                    seq: i,
                    tokens: p.clone(),
                })
                .collect();
            drop(rn.forward(&mut seqs, &calls)?);
            let mut ms = Vec::with_capacity(reps);
            for k in 0..reps {
                let t = Instant::now();
                if batched {
                    let rows: Vec<(usize, u32)> = input[..w]
                        .iter()
                        .enumerate()
                        .map(|(i, (_, f))| (i, f[k]))
                        .collect();
                    if mode == "chunk8" {
                        for c in rows.chunks(BATCH_ROWS) {
                            let o = rn.forward_batch(&mut seqs, c)?;
                            rn.argmax_rows(&o, 0, c.len())?;
                        }
                    } else {
                        let o = rn.forward_batch(&mut seqs, &rows)?;
                        match mode {
                            "argmax" => drop(rn.argmax_rows(&o, 0, w)?),
                            "fwd" => rn.grp.each(|_, cx| cx.ops.sync()).map(|_| ())?,
                            _ => drop(rn.logits(&o, 0, w)?),
                        }
                    }
                } else {
                    let calls: Vec<Call> = input[..w]
                        .iter()
                        .enumerate()
                        .map(|(i, (_, f))| Call {
                            seq: i,
                            tokens: vec![f[k]],
                        })
                        .collect();
                    for o in &rn.forward(&mut seqs, &calls)? {
                        rn.logits(o, 0, 1)?;
                    }
                }
                ms.push(t.elapsed().as_secs_f64() * 1e3);
            }
            // the first two steps warm the shapes
            let key = format!("{mode}_{w}_ms");
            out.insert(key, serde_json::json!(med(&ms[2..])));
        }
    }
    // how many experts a layer the widest batch reads: the router's selection of four more steps
    let w = [32usize, 16, 8, 4, 2, 1]
        .into_iter()
        .find(|&w| w <= input.len())
        .unwrap_or(1);
    // the launches of two consecutive batched steps (both parities of a ratio-2 pair), by layer
    // and kernel, for one sequence and for the widest batch: what grows with the sequences.
    // Then the same two steps with a wait for the card after every launch: each kernel's own cost
    for ws in [1usize, w] {
        let mut seqs: Vec<Seq> = input[..ws]
            .iter()
            .map(|(p, f)| rn.seq_with_capacity(p.len() + f.len() + 16))
            .collect();
        let calls: Vec<Call> = input[..ws]
            .iter()
            .enumerate()
            .map(|(i, (p, _))| Call {
                seq: i,
                tokens: p.clone(),
            })
            .collect();
        drop(rn.forward(&mut seqs, &calls)?);
        let step = |rn: &mut Runner, seqs: &mut [Seq], k: usize| -> Result<(), String> {
            let rows: Vec<(usize, u32)> = input[..ws]
                .iter()
                .enumerate()
                .map(|(i, (_, f))| (i, f[k]))
                .collect();
            let o = rn.forward_batch(seqs, &rows)?;
            rn.argmax_rows(&o, 0, ws).map(|_| ())
        };
        step(rn, &mut seqs, 0)?;
        step(rn, &mut seqs, 1)?;
        for (timed, k0) in [(false, 2usize), (true, 4)] {
            if input[..ws].iter().any(|(_, f)| f.len() < k0 + 2) {
                break;
            }
            memra_engine::dsv41_gpu::tally::start(timed);
            let r = step(rn, &mut seqs, k0).and_then(|_| step(rn, &mut seqs, k0 + 1));
            let t = memra_engine::dsv41_gpu::tally::take();
            r?;
            out.insert(
                format!(
                    "{}_{ws}_two_steps",
                    if timed { "launch_ms" } else { "launches" }
                ),
                tally_json(&t),
            );
        }
        if ws == w {
            break;
        }
    }
    {
        let mut seqs: Vec<Seq> = input[..w]
            .iter()
            .map(|(p, f)| rn.seq_with_capacity(p.len() + f.len() + 16))
            .collect();
        let calls: Vec<Call> = input[..w]
            .iter()
            .enumerate()
            .map(|(i, (p, _))| Call {
                seq: i,
                tokens: p.clone(),
            })
            .collect();
        drop(rn.forward(&mut seqs, &calls)?);
        rn.grp.trace = true;
        for c in rn.grp.cx.iter_mut() {
            c.trace = true;
        }
        let topk = rn.g.topk;
        let mut counts: Vec<usize> = Vec::new();
        let mut res = Ok(());
        for k in 0..4 {
            let rows: Vec<(usize, u32)> = input[..w]
                .iter()
                .enumerate()
                .map(|(i, (_, f))| (i, f[k]))
                .collect();
            match rn.forward_batch(&mut seqs, &rows) {
                Ok(_) => {
                    for layer in rn.last_routes.chunks(w * topk) {
                        let mut ids = layer.to_vec();
                        ids.sort_unstable();
                        ids.dedup();
                        counts.push(ids.len());
                    }
                }
                Err(e) => {
                    res = Err(e);
                    break;
                }
            }
        }
        rn.grp.trace = false;
        for c in rn.grp.cx.iter_mut() {
            c.trace = false;
        }
        res?;
        let nlay = counts.len().max(1) as f64;
        out.insert("routes_sequences".into(), w.into());
        out.insert("routes_pairs_per_layer".into(), (w * topk).into());
        out.insert(
            "distinct_experts_per_layer_mean".into(),
            serde_json::json!(counts.iter().sum::<usize>() as f64 / nlay),
        );
        out.insert(
            "distinct_experts_per_layer_min_max".into(),
            serde_json::json!([counts.iter().min(), counts.iter().max()]),
        );
    }
    out.insert("job".into(), "batch-time".into());
    out.insert("id".into(), id.into());
    out.insert("pass".into(), true.into());
    out.insert("reps".into(), reps.into());
    out.insert(
        "prompts".into(),
        serde_json::json!(input.iter().map(|(p, _)| p.len()).collect::<Vec<_>>()),
    );
    Ok((true, serde_json::Value::Object(out).to_string()))
}

/// What a decode call costs by its rows (see the module doc).
fn rows_time(
    rn: &mut Runner,
    id: &str,
    prompt: &[u32],
    force: &[u32],
    reps: usize,
) -> Result<(bool, String), String> {
    let reps = reps.max(2);
    if force.len() < reps + 8 {
        return Err(format!(
            "rows-time wants at least {} forced tokens",
            reps + 8
        ));
    }
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
    let med = |x: &[f64]| -> f64 {
        let mut v = x.to_vec();
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    let mut seqs = vec![rn.seq_with_capacity(prompt.len() + force.len() + 16)];
    let t = Instant::now();
    let outs = rn.forward(
        &mut seqs,
        &[Call {
            seq: 0,
            tokens: prompt.to_vec(),
        }],
    )?;
    rn.logits(&outs[0], outs[0].s - 1, 1)?;
    let prefill_ms = ms(t);
    drop(outs);
    // the plain step: REPS one-token calls
    let mut one = Vec::with_capacity(reps);
    for &tok in &force[..reps] {
        let t = Instant::now();
        let outs = rn.forward(
            &mut seqs,
            &[Call {
                seq: 0,
                tokens: vec![tok],
            }],
        )?;
        rn.logits(&outs[0], 0, 1)?;
        one.push(ms(t));
    }
    // blocks of 2..8 rows at the same position, dropped whole after each run: different
    // tokens, then one token repeated
    let mut first = Vec::new();
    let mut warm = Vec::new();
    let mut same_first = Vec::new();
    let mut same_warm = Vec::new();
    for k in 2..=8usize {
        for same in [false, true] {
            let toks = if same {
                vec![force[reps]; k]
            } else {
                force[reps..reps + k].to_vec()
            };
            let mut runs = Vec::with_capacity(reps);
            for _ in 0..reps {
                let t = Instant::now();
                let outs = rn.forward(
                    &mut seqs,
                    &[Call {
                        seq: 0,
                        tokens: toks.clone(),
                    }],
                )?;
                rn.logits(&outs[0], 0, k)?;
                runs.push(ms(t));
                drop(outs);
                rn.discard(&mut seqs[0])?;
            }
            if same {
                same_first.push(runs[0]);
                same_warm.push(med(&runs[1..]));
            } else {
                first.push(runs[0]);
                warm.push(med(&runs[1..]));
            }
        }
    }
    Ok((
        true,
        serde_json::json!({
            "job": "rows-time", "id": id, "pass": true, "prompt": prompt.len(), "reps": reps,
            "prefill_ms": prefill_ms,
            "one_token_ms": one, "one_token_ms_median": med(&one),
            "rows_2_to_8_first_ms": first, "rows_2_to_8_repeat_ms_median": warm,
            "same_token_rows_2_to_8_first_ms": same_first,
            "same_token_rows_2_to_8_repeat_ms_median": same_warm,
        })
        .to_string(),
    ))
}

/// A launch tally as JSON: totals, `[launches, ms]` per layer and per kernel, and every
/// `layer:kernel` entry as `[launches, microseconds]`.
fn tally_json(t: &memra_engine::dsv41_gpu::tally::Table) -> serde_json::Value {
    let mut by = serde_json::Map::new();
    let mut layers: std::collections::BTreeMap<usize, (u64, u64)> = Default::default();
    let mut kernels: std::collections::BTreeMap<&str, (u64, u64)> = Default::default();
    for ((li, name), (c, ns)) in t {
        let l = if *li == usize::MAX {
            "out".to_string()
        } else {
            format!("L{li}")
        };
        let us = (*ns as f64 / 1e3).round();
        by.insert(format!("{l}:{name}"), serde_json::json!([c, us]));
        let e = layers.entry(*li).or_insert((0, 0));
        (e.0, e.1) = (e.0 + c, e.1 + ns);
        let e = kernels.entry(name.as_str()).or_insert((0, 0));
        (e.0, e.1) = (e.0 + c, e.1 + ns);
    }
    let ms = |ns: u64| (ns as f64 / 1e4).round() / 1e2;
    let per_layer: Vec<(u64, f64)> = layers.values().map(|&(c, ns)| (c, ms(ns))).collect();
    let per_kernel: serde_json::Map<String, serde_json::Value> = kernels
        .iter()
        .map(|(k, &(c, ns))| (k.to_string(), serde_json::json!([c, ms(ns)])))
        .collect();
    serde_json::json!({
        "total": t.values().map(|v| v.0).sum::<u64>(),
        "total_ms": ms(t.values().map(|v| v.1).sum::<u64>()),
        "per_layer": per_layer,
        "per_kernel": per_kernel,
        "by": by,
    })
}

/// A call's taps given outright (a tape's main_hidden rows) on every rank.
fn taped_out(rn: &mut Runner, bytes: &[u8], sp: usize, rows: usize) -> Result<Out, String> {
    let taps = rn.grp.each(|_, cx| Buf::from_bytes(&cx.ops.st, bytes))?;
    Ok(Out {
        h: Vec::new(),
        pre: Vec::new(),
        sp,
        s: rows,
        taps,
        tap_row0: 0,
        tap_rows: rows,
        batch: Vec::new(),
    })
}

fn draft_tape(
    rn: &mut Runner,
    dw: &DraftW,
    id: &str,
    tape: &Path,
    dir: &Path,
) -> Result<(bool, String), String> {
    let g = rn.g.clone();
    let (d, win, v_) = (g.dim, g.window, g.vocab);
    let nt = dw.cfg.targets.len();
    let rowb = nt * d * 2;
    let rd =
        |n: &str| std::fs::read(tape.join(n)).map_err(|e| format!("{}/{n}: {e}", tape.display()));
    let prompt = read_i32(&tape.join("prompt.i32"))?;
    let pmh = rd("prompt_mh.bf16")?;
    if pmh.len() != prompt.len() * rowb {
        return Err(format!(
            "prompt_mh.bf16: {} bytes for {} positions",
            pmh.len(),
            prompt.len()
        ));
    }
    let toks = read_i32(&tape.join("tok_in.i32")).unwrap_or_default();
    let pos = read_i32(&tape.join("pos.i32")).unwrap_or_default();
    let mh = if toks.is_empty() {
        Vec::new()
    } else {
        rd("mh.bf16")?
    };
    let want: Vec<u32> = if toks.is_empty() {
        Vec::new()
    } else {
        read_i32(&tape.join("out_ids.i32"))?
    };
    let meta: serde_json::Value =
        serde_json::from_slice(&rd("meta.json")?).map_err(|e| e.to_string())?;
    let full_steps = meta["full_steps"].as_u64().unwrap_or(0) as usize;
    let mut ds = dw.seq(rn);
    // seed: the prompt's last `window` rows
    let s = prompt.len();
    let r0 = s - s.min(win);
    let o = taped_out(rn, &pmh[r0 * rowb..], r0, s - r0)?;
    rn.draft_commit(dw, &mut ds, &o, 0, s - r0, false)?;
    let mut ring0 = Vec::new();
    for si in 0..dw.cfg.stages {
        ring0.extend(ds.ring(rn, si)?);
    }
    write(dir, "ring0.bf16", &ring0)?;
    let b = dw.cfg.block;
    let st_n = dw.cfg.stages;
    let mut main_x = Vec::new();
    let mut hs: Vec<Vec<u8>> = vec![Vec::new(); st_n];
    let mut pres: Vec<Vec<u8>> = vec![Vec::new(); st_n];
    let mut ain: Vec<Vec<u8>> = vec![Vec::new(); st_n];
    let mut aout: Vec<Vec<u8>> = vec![Vec::new(); st_n];
    let mut fout: Vec<Vec<u8>> = vec![Vec::new(); st_n];
    let mut routes: Vec<Vec<u8>> = vec![Vec::new(); st_n];
    let (mut head_x, mut out_ids, mut conf) = (Vec::new(), Vec::new(), Vec::new());
    let (mut lses, mut top_ids, mut top_vals) = (Vec::new(), Vec::new(), Vec::new());
    let (mut full, mut full_bias) = (Vec::new(), Vec::new());
    let mut same_ids = 0usize;
    let mut same_blocks = 0usize;
    let mut untraced_equal = 0usize;
    let t0 = Instant::now();
    for (k, (&tok, &p)) in toks.iter().zip(&pos).enumerate() {
        let p = p as usize;
        let o = taped_out(rn, &mh[k * rowb..(k + 1) * rowb], p, 1)?;
        let mx = rn
            .draft_commit(dw, &mut ds, &o, 0, 1, true)?
            .ok_or("main_x not returned")?;
        main_x.extend(mx);
        let dr = rn.draft(dw, &mut ds, tok, p + 1, true)?;
        // the same draft untraced: on one rank its picks chain on the device, and must name the
        // traced draft's tokens and confidences
        let plain = rn.draft(dw, &mut ds, tok, p + 1, false)?;
        untraced_equal += usize::from(plain.ids == dr.ids && plain.conf == dr.conf);
        let t = dr.trace.as_ref().ok_or("no trace")?;
        for si in 0..st_n {
            hs[si].extend(&t.h[si]);
            pres[si].extend(f32_le(&t.pre[si]));
            ain[si].extend(&t.attn_in[si]);
            aout[si].extend(&t.attn_out[si]);
            fout[si].extend(&t.ffn_out[si]);
            routes[si].extend(i32_le(&t.route_sel[si]));
        }
        head_x.extend(&t.head_x);
        out_ids.extend(dr.ids.iter().map(|&x| x as i32));
        conf.extend(&dr.conf);
        for i in 0..b {
            let row = &t.logits[i * v_..(i + 1) * v_];
            lses.push(lse(row) as f32);
            for (ti, tv) in top_k(row) {
                top_ids.push(ti as i32);
                top_vals.push(tv);
            }
        }
        if k < full_steps {
            full.extend(&t.logits);
            full_bias.extend(&t.bias);
        }
        let w = &want[k * b..(k + 1) * b];
        let eq = dr.ids.iter().zip(w).filter(|(a, b)| a == b).count();
        same_ids += eq;
        same_blocks += usize::from(eq == b);
    }
    let secs = t0.elapsed().as_secs_f64();
    // what a draft costs: the last step's untraced draft again (a draft leaves the rings as they
    // are), its launches, and each kernel's own time with a card sync after every launch
    let mut timing = serde_json::Map::new();
    if let (Some(&tok), Some(&p)) = (toks.last(), pos.last()) {
        let p = p as usize + 1;
        let mut ms = Vec::with_capacity(16);
        for _ in 0..16 {
            let t = Instant::now();
            rn.draft(dw, &mut ds, tok, p, false)?;
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        ms.sort_by(|a, b| a.total_cmp(b));
        timing.insert("draft_ms_median".into(), ms[ms.len() / 2].into());
        timing.insert("draft_ms_min".into(), ms[0].into());
        for timed in [false, true] {
            memra_engine::dsv41_gpu::tally::start(timed);
            let r = rn
                .draft(dw, &mut ds, tok, p, false)
                .and_then(|_| rn.draft(dw, &mut ds, tok, p, false));
            let t = memra_engine::dsv41_gpu::tally::take();
            r?;
            timing.insert(
                if timed {
                    "draft_launch_ms_two"
                } else {
                    "draft_launches_two"
                }
                .into(),
                tally_json(&t),
            );
        }
    }
    let mut ring1 = Vec::new();
    for si in 0..st_n {
        ring1.extend(ds.ring(rn, si)?);
    }
    write(dir, "ring1.bf16", &ring1)?;
    let n = toks.len();
    if n > 0 {
        write(dir, "main_x.bf16", &main_x)?;
        for si in 0..st_n {
            write(dir, &format!("h{si}.bf16"), &hs[si])?;
            write(dir, &format!("pre{si}.f32"), &pres[si])?;
            write(dir, &format!("attn_in{si}.bf16"), &ain[si])?;
            write(dir, &format!("attn_out{si}.bf16"), &aout[si])?;
            write(dir, &format!("ffn_out{si}.bf16"), &fout[si])?;
            write(dir, &format!("route_i{si}.i32"), &routes[si])?;
        }
        write(dir, "head_x.bf16", &head_x)?;
        write(dir, "out_ids.i32", &i32_le(&out_ids))?;
        write(dir, "conf.f32", &f32_le(&conf))?;
        write(dir, "lse.f32", &f32_le(&lses))?;
        write(dir, "top_ids.i32", &i32_le(&top_ids))?;
        write(dir, "top_vals.f32", &f32_le(&top_vals))?;
        if full_steps > 0 {
            write(dir, "logits_full.f32", &f32_le(&full))?;
            write(dir, "bias_full.f32", &f32_le(&full_bias))?;
        }
    }
    debug_assert_eq!(TOPK, 32);
    let pass = untraced_equal == n;
    Ok((
        pass,
        serde_json::json!({
            "job": "draft-tape", "id": id, "pass": pass, "prompt": s, "steps": n,
            "draft_tokens": n * b, "draft_tokens_equal_vendor": same_ids,
            "blocks_equal_vendor": same_blocks, "untraced_drafts_equal": untraced_equal,
            "secs": secs, "timing": timing,
        })
        .to_string(),
    ))
}

/// Drafts to verify under a confidence floor: the leading rows whose sigmoid is at least `tau`.
fn confident(conf: &[f32], tau: f32) -> usize {
    let tl = (tau / (1.0 - tau)).ln();
    conf.iter().take_while(|&&c| c >= tl).count()
}

struct Round {
    pos: usize,
    verified: usize,
    accepted: usize,
    draft_ms: f64,
    verify_ms: f64,
    commit_ms: f64,
    conf: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn generate(
    rn: &mut Runner,
    dw: &DraftW,
    id: &str,
    prompt: &[u32],
    n_new: usize,
    eos: u32,
    depth: Option<usize>,
    tau: Option<f32>,
    dir: &Path,
) -> Result<(bool, String), String> {
    let v_ = rn.g.vocab;
    let cap = prompt.len() + n_new + dw.cfg.block + 16;
    // plain greedy: one token per decode call
    let mut seqs: Vec<Seq> = vec![rn.seq_with_capacity(cap)];
    let t0 = Instant::now();
    let outs = rn.forward(
        &mut seqs,
        &[Call {
            seq: 0,
            tokens: prompt.to_vec(),
        }],
    )?;
    let lg = rn.logits(&outs[0], outs[0].s - 1, 1)?;
    let prefill_s = t0.elapsed().as_secs_f64();
    drop(outs);
    let mut p_tok = vec![argmax(&lg) as u32];
    let mut p_hash = vec![fnv(&lg)];
    let mut p_ms = Vec::new();
    let (mut dev_rows, mut dev_equal) = (0usize, 0usize);
    while p_tok.len() < n_new && *p_tok.last().unwrap() != eos {
        let t = Instant::now();
        let outs = rn.forward(
            &mut seqs,
            &[Call {
                seq: 0,
                tokens: vec![*p_tok.last().unwrap()],
            }],
        )?;
        let lg = rn.logits(&outs[0], 0, 1)?;
        p_ms.push(t.elapsed().as_secs_f64() * 1e3);
        p_tok.push(argmax(&lg) as u32);
        p_hash.push(fnv(&lg));
        // the device reduction of the same row must name the same token
        dev_rows += 1;
        dev_equal += usize::from(rn.argmax_rows(&outs[0], 0, 1)? == [argmax(&lg) as u32]);
    }
    drop(seqs);
    let plain_s: f64 = p_ms.iter().sum::<f64>() / 1e3;

    // drafted greedy
    let mut seqs: Vec<Seq> = vec![rn.seq_with_capacity(cap)];
    let mut ds: DraftSeq = dw.seq(rn);
    let outs = rn.forward(
        &mut seqs,
        &[Call {
            seq: 0,
            tokens: prompt.to_vec(),
        }],
    )?;
    let o = &outs[0];
    let lg = rn.logits(o, o.s - 1, 1)?;
    rn.draft_commit(dw, &mut ds, o, o.tap_row0, o.tap_rows, false)?;
    drop(outs);
    let mut s_tok: Vec<u32> = Vec::new();
    let mut s_hash: Vec<u64> = Vec::new();
    let mut t_tok = argmax(&lg) as u32;
    let mut t_hash = fnv(&lg);
    let mut rounds: Vec<Round> = Vec::new();
    let ts = Instant::now();
    loop {
        s_tok.push(t_tok);
        s_hash.push(t_hash);
        if s_tok.len() >= n_new || t_tok == eos {
            break;
        }
        let m0 = seqs[0].len();
        let td = Instant::now();
        let dr = rn.draft(dw, &mut ds, t_tok, m0, false)?;
        let draft_ms = td.elapsed().as_secs_f64() * 1e3;
        let mut k = depth.unwrap_or(dr.ids.len()).min(dr.ids.len());
        if let Some(tau) = tau {
            k = k.min(confident(&dr.conf, tau));
        }
        // leave room for the token after the accepted drafts
        k = k.min(n_new - s_tok.len() - 1);
        let mut batch = vec![t_tok];
        batch.extend_from_slice(&dr.ids[..k]);
        let tv = Instant::now();
        let outs = rn.forward(
            &mut seqs,
            &[Call {
                seq: 0,
                tokens: batch,
            }],
        )?;
        let o = &outs[0];
        let lg = rn.logits(o, 0, k + 1)?;
        let verify_ms = tv.elapsed().as_secs_f64() * 1e3;
        let targets: Vec<u32> = (0..=k)
            .map(|i| argmax(&lg[i * v_..(i + 1) * v_]) as u32)
            .collect();
        dev_rows += k + 1;
        let dev = rn.argmax_rows(o, 0, k + 1)?;
        dev_equal += dev.iter().zip(&targets).filter(|(a, b)| a == b).count();
        let (mut c_d, mut t_next) = accept_walk(&targets, &dr.ids[..k]);
        // an accepted end-of-sequence ends the run there
        if let Some(e) = dr.ids[..c_d].iter().position(|&x| x == eos) {
            c_d = e;
            t_next = eos;
        }
        let tc = Instant::now();
        if k > 0 {
            rn.commit(&mut seqs[0], c_d + 1)?;
        }
        rn.draft_commit(dw, &mut ds, o, 0, c_d + 1, false)?;
        rn.grp.sync_all()?;
        let commit_ms = tc.elapsed().as_secs_f64() * 1e3;
        for i in 0..c_d {
            s_tok.push(dr.ids[i]);
            s_hash.push(fnv(&lg[i * v_..(i + 1) * v_]));
        }
        t_tok = t_next;
        t_hash = fnv(&lg[c_d * v_..(c_d + 1) * v_]);
        rounds.push(Round {
            pos: m0,
            verified: k,
            accepted: c_d,
            draft_ms,
            verify_ms,
            commit_ms,
            conf: dr.conf.clone(),
        });
    }
    let spec_s = ts.elapsed().as_secs_f64();
    let same_tok = s_tok == p_tok;
    let same_logits = s_hash == p_hash;
    let pass = same_tok && same_logits && dev_equal == dev_rows;
    let first_diff = s_tok
        .iter()
        .zip(&p_tok)
        .position(|(a, b)| a != b)
        .or((s_tok.len() != p_tok.len()).then_some(s_tok.len().min(p_tok.len())));
    let mut tsv = String::from("pos\tverified\taccepted\tdraft_ms\tverify_ms\tcommit_ms\tconf\n");
    for r in &rounds {
        let conf: Vec<String> = r.conf.iter().map(|c| format!("{c:.4}")).collect();
        tsv.push_str(&format!(
            "{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{}\n",
            r.pos,
            r.verified,
            r.accepted,
            r.draft_ms,
            r.verify_ms,
            r.commit_ms,
            conf.join(",")
        ));
    }
    write(dir, "rounds.tsv", tsv.as_bytes())?;
    write(
        dir,
        "plain.i32",
        &i32_le(&p_tok.iter().map(|&x| x as i32).collect::<Vec<_>>()),
    )?;
    write(
        dir,
        "spec.i32",
        &i32_le(&s_tok.iter().map(|&x| x as i32).collect::<Vec<_>>()),
    )?;
    write(
        dir,
        "plain_step_ms.f32",
        &f32_le(&p_ms.iter().map(|&x| x as f32).collect::<Vec<_>>()),
    )?;
    let med = |mut x: Vec<f64>| -> f64 {
        if x.is_empty() {
            return 0.0;
        }
        x.sort_by(|a, b| a.total_cmp(b));
        x[x.len() / 2]
    };
    let nr = rounds.len().max(1) as f64;
    let acc: usize = rounds.iter().map(|r| r.accepted).sum();
    let ver: usize = rounds.iter().map(|r| r.verified).sum();
    let mut hist = vec![0usize; dw.cfg.block + 1];
    for r in &rounds {
        hist[r.accepted] += 1;
    }
    let by_rows = |rows: usize| {
        med(rounds
            .iter()
            .filter(|r| r.verified + 1 == rows)
            .map(|r| r.verify_ms)
            .collect())
    };
    Ok((
        pass,
        serde_json::json!({
            "job": "gen", "id": id, "pass": pass, "prompt": prompt.len(),
            "tokens": p_tok.len(), "spec_tokens": s_tok.len(),
            "tokens_identical": same_tok, "logits_identical": same_logits, "first_diff": first_diff,
            "device_argmax_rows": dev_rows, "device_argmax_equal": dev_equal,
            "depth": depth, "tau": tau,
            "rounds": rounds.len(), "drafts_verified": ver, "drafts_accepted": acc,
            "accepted_per_round": acc as f64 / nr,
            "tokens_per_round": (s_tok.len() as f64 - 1.0) / nr,
            "accept_hist": hist,
            "prefill_s": prefill_s,
            "plain_step_ms_median": med(p_ms.clone()),
            "plain_tok_s": (p_tok.len() as f64 - 1.0) / plain_s.max(1e-9),
            "draft_ms_median": med(rounds.iter().map(|r| r.draft_ms).collect()),
            "verify_ms_median": med(rounds.iter().map(|r| r.verify_ms).collect()),
            "verify_ms_median_by_rows": (1..=dw.cfg.block + 1).map(by_rows).collect::<Vec<_>>(),
            "commit_ms_median": med(rounds.iter().map(|r| r.commit_ms).collect()),
            "spec_tok_s": (s_tok.len() as f64 - 1.0) / spec_s.max(1e-9),
            "speedup": ((s_tok.len() as f64 - 1.0) / spec_s.max(1e-9))
                / ((p_tok.len() as f64 - 1.0) / plain_s.max(1e-9)),
        })
        .to_string(),
    ))
}
