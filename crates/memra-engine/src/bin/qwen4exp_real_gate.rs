//! qwen4_exp REAL-CHECKPOINT gate — the fleet-box arm of the eager ladder (phase 4/8
//! REAL-CHECKPOINT item deferred from GPU-EAGER.md).
//!
//! Runs the GPU eager path against a REAL qwen4_exp artifact dir (the pinned BF16 export
//! or the per-expert NVFP4 mint) and gates it against banked transformers goldens:
//!
//! - `--goldens <dump>`: per-layer wide-stream parity — prefill the goldens prompt with
//!   capture and compare EVERY post-layer wide state + the exit mixer + final logits
//!   against the transformers bf16 forward (hidden-goldens.pt, dumped to raw f32 bins by
//!   research/qwen4exp-bringup-20260829/gpu-eager/prep-real-gate.py). The gate REPORTS
//!   the measured envelope per layer (bf16-forward vs f32-eager is an accumulation
//!   comparison, not an exactness one); thresholds only classify.
//! - `--prompts <tsv>`: greedy-continuation divergence — 64-token argmax chains per
//!   prompt vs greedy-goldens.json; the deliverable is the FIRST-DIVERGENCE STEP per
//!   prompt (bf16 vs f32 argmax chains are expected to fork somewhere), not 100% match.
//! - `--compare-logits <bin>`: cross-arm probe-logit comparison (NVFP4 arm vs the BF16
//!   arm's saved logits): per-row top-1, top-20 overlap, KL(ref‖candidate).
//! - `--decode-timing <n>`: n self-fed argmax decode steps after the goldens prefill,
//!   wall-clock per step. UNTUNED EAGER NUMBER, correctness-arm residency — never a
//!   perf claim (stated in the receipt header too).
//!
//! Loader knobs: `--host-bf16-banks` (BF16 expert banks host-resident, per-routed-expert
//! upload — the 360 GB export's f32 banks ≈ 483 GB and cannot be device-resident) and
//! `--indexer-norm-raw` (the (1+w)-fold two-arm probe for the SEMANTICS.md VERIFY on the
//! indexer layernorms).
//!
//! Usage: qwen4exp_real_gate <ckpt_dir> <out_dir> --label <label> [flags above]

use memra_engine::Engine;
use memra_engine::qwen4exp_gpu::{LoadOptions, Qwen4ExpGpu, read_checkpoint_with};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (index, &value) in row.iter().enumerate() {
        if value > row[best] {
            best = index;
        }
    }
    best
}

#[derive(Default)]
struct RowStats {
    max_abs: f32,
    max_rel: f32,
    mean_abs: f64,
    ref_absmax: f32,
}

/// Elementwise envelope: max_abs, max_rel (denominator max(1, |ref|) — the
/// modelplan_reference_gate convention), mean_abs, and the reference magnitude the
/// abs numbers read against.
fn compare(reference: &[f32], candidate: &[f32]) -> RowStats {
    let mut s = RowStats::default();
    for (&r, &c) in reference.iter().zip(candidate) {
        let abs = (r - c).abs();
        s.max_abs = s.max_abs.max(abs);
        s.max_rel = s.max_rel.max(abs / r.abs().max(1.0));
        s.mean_abs += abs as f64;
        s.ref_absmax = s.ref_absmax.max(r.abs());
    }
    s.mean_abs /= reference.len().max(1) as f64;
    s
}

/// Top-k indices by value desc (ties index-asc — deterministic).
fn top_k(row: &[f32], k: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..row.len()).collect();
    indices.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
    indices.truncate(k);
    indices
}

/// KL(P‖Q) with P = softmax(reference), Q = softmax(candidate), f64 throughout.
fn kl_divergence(reference: &[f32], candidate: &[f32]) -> f64 {
    let log_softmax = |row: &[f32]| -> Vec<f64> {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let mut sum = 0.0f64;
        for &v in row {
            sum += ((v as f64) - max).exp();
        }
        let log_z = max + sum.ln();
        row.iter().map(|&v| v as f64 - log_z).collect()
    };
    let lp = log_softmax(reference);
    let lq = log_softmax(candidate);
    lp.iter().zip(&lq).map(|(&a, &b)| a.exp() * (a - b)).sum()
}

fn nvidia_smi() -> String {
    std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader",
        ])
        .output()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .trim()
                .replace('\n', " | ")
        })
        .unwrap_or_else(|e| format!("nvidia-smi unavailable: {e}"))
}

fn read_f32_bin(path: &Path, expect_len: usize) -> Res<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() != expect_len * 4 {
        return Err(format!(
            "{}: {} bytes, expected {} f32",
            path.display(),
            bytes.len(),
            expect_len
        )
        .into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn write_f32_bin(path: &Path, data: &[f32]) -> Res<()> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for &v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, out)?;
    Ok(())
}

struct GoldenRecord {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

struct Goldens {
    input_ids: Vec<u32>,
    records: std::collections::BTreeMap<String, GoldenRecord>,
}

fn read_goldens(dir: &Path) -> Res<Goldens> {
    let input_ids = std::fs::read_to_string(dir.join("input-ids.txt"))?
        .split_whitespace()
        .map(|t| t.parse::<u32>())
        .collect::<Result<Vec<_>, _>>()?;
    let mut records = std::collections::BTreeMap::new();
    for line in std::fs::read_to_string(dir.join("manifest.tsv"))?.lines() {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 4 {
            return Err(format!("manifest.tsv: malformed line {line:?}").into());
        }
        let (rows, cols): (usize, usize) = (fields[1].parse()?, fields[2].parse()?);
        let data = read_f32_bin(&dir.join(fields[3]), rows * cols)?;
        records.insert(fields[0].to_string(), GoldenRecord { rows, cols, data });
    }
    Ok(Goldens { input_ids, records })
}

struct Prompt {
    index: usize,
    ids: Vec<u32>,
    golden: Vec<u32>,
}

/// One own-gen corpus prompt: `index \t class \t ids-csv` (the rank-corpus pack; the
/// class column is what the DRAFT-REGIME coverage law is reported against).
struct CorpusPrompt {
    index: usize,
    class: String,
    ids: Vec<u32>,
}

fn read_corpus_prompts(path: &Path) -> Res<Vec<CorpusPrompt>> {
    let mut prompts = Vec::new();
    for line in std::fs::read_to_string(path)?.lines() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 3 {
            return Err(format!("corpus prompts: malformed line {line:?}").into());
        }
        prompts.push(CorpusPrompt {
            index: fields[0].parse()?,
            class: fields[1].to_string(),
            ids: fields[2]
                .split(',')
                .map(|t| t.parse::<u32>().map_err(Into::into))
                .collect::<Res<Vec<u32>>>()?,
        });
    }
    if prompts.is_empty() {
        return Err("corpus prompts: file is empty".into());
    }
    Ok(prompts)
}

/// Read a ranks sidecar: `id` or `id\tcount` per line, RANK ORDER (most frequent first).
fn read_ranks(path: &Path) -> Res<Vec<u32>> {
    let mut ids = Vec::new();
    for line in std::fs::read_to_string(path)?.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let first = line.split('\t').next().unwrap_or(line);
        ids.push(first.parse::<u32>()?);
    }
    if ids.is_empty() {
        return Err("ranks file: no ids".into());
    }
    Ok(ids)
}

fn read_prompts(path: &Path) -> Res<Vec<Prompt>> {
    let mut prompts = Vec::new();
    for line in std::fs::read_to_string(path)?.lines() {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 3 {
            return Err(format!("prompts.tsv: malformed line {line:?}").into());
        }
        let parse_csv = |s: &str| -> Res<Vec<u32>> {
            s.split(',')
                .map(|t| t.parse::<u32>().map_err(Into::into))
                .collect()
        };
        prompts.push(Prompt {
            index: fields[0].parse()?,
            ids: parse_csv(fields[1])?,
            golden: parse_csv(fields[2])?,
        });
    }
    Ok(prompts)
}

fn ms(from: Instant) -> f64 {
    from.elapsed().as_secs_f64() * 1e3
}

fn csv(ids: &[u32]) -> String {
    ids.iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

// ---------------------------------------------------------------- host-side probe
//
// The 262k HOST-lever instrument. `prof_section` brackets each section with a device sync,
// so it prices host wall time honestly but is blind to the three things a host lever is
// actually about. All three are read from /proc and `sched_getcpu`, so there is no new
// dependency and nothing is inserted into the measured path:
//
//  - **VOLUNTARY context switches are the blocking-sync counter.** A CUDA wait that SPINS
//    burns host cycles and parks nothing, so it shows zero voluntary switches; a wait that
//    BLOCKS parks the thread and shows one switch per wait plus a wake latency. So
//    `vol_cs/step` is the spin-vs-block audit and the event-wait-bubble census stated as a
//    number, instead of a claim read off a flag default.
//  - **NONVOLUNTARY switches are preemption** — the launch thread losing the CPU to another
//    runnable task. On a SHARED box that is the other agent's run, which makes this the
//    field that says whether a timing was taken against contention. A timing row whose
//    nonvoluntary count jumps is suspect on its face, independent of its spread.
//  - **`sched_getcpu` sampled per step counts MIGRATIONS**, which is the sticky-thread
//    question as a measurement rather than an assumption.
//
// Deliberately NOT sampled per step: the per-thread /proc/<tid>/status parse. Four threads x
// a file read x 65 ms steps is ~0.1% and it would land inside the very number it is
// measuring; the census is taken at round boundaries and `sched_getcpu` (a vDSO read, tens
// of nanoseconds) is what rides every step.
#[derive(Clone, Default, Debug)]
struct HostCensus {
    threads: usize,
    vol_cs: u64,
    nonvol_cs: u64,
    /// Per-thread `comm` + last-run CPU, for the "which thread is where" table.
    per_thread: Vec<(String, u64, u64, i64)>,
}

/// Sum voluntary/nonvoluntary context switches over EVERY thread of this process.
/// Per-thread rather than /proc/self/status, because /proc/self/status reports the main
/// thread only and the CUDA driver's own worker threads are exactly where a blocking wait
/// would park.
fn host_census() -> HostCensus {
    let mut out = HostCensus::default();
    let Ok(dir) = std::fs::read_dir("/proc/self/task") else {
        return out;
    };
    for entry in dir.flatten() {
        let base = entry.path();
        let Ok(status) = std::fs::read_to_string(base.join("status")) else {
            continue;
        };
        let comm = std::fs::read_to_string(base.join("comm"))
            .unwrap_or_default()
            .trim()
            .to_string();
        let field = |key: &str| -> u64 {
            status
                .lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.rsplit(char::is_whitespace).next())
                .and_then(|v| v.parse().ok())
                .unwrap_or(0)
        };
        let vol = field("voluntary_ctxt_switches:");
        let nonvol = field("nonvoluntary_ctxt_switches:");
        // Field 39 of /proc/<tid>/stat is `processor`. Everything before the LAST ')' is
        // pid + comm, and comm can itself contain spaces and parens — so split there, not
        // on whitespace from the start. After the ')' the first token is field 3 (state),
        // which puts `processor` at post-split index 36.
        let cpu = std::fs::read_to_string(base.join("stat"))
            .ok()
            .and_then(|s| {
                let tail = &s[s.rfind(')')? + 1..];
                tail.split_whitespace().nth(36)?.parse::<i64>().ok()
            })
            .unwrap_or(-1);
        out.threads += 1;
        out.vol_cs += vol;
        out.nonvol_cs += nonvol;
        out.per_thread.push((comm, vol, nonvol, cpu));
    }
    out.per_thread.sort();
    out
}

/// CPU the CALLING thread is running on right now (vDSO, no syscall on x86_64).
fn self_cpu() -> i32 {
    // SAFETY: sched_getcpu takes no arguments, writes nothing, and cannot fail in a way
    // that matters here (-1 on an unsupported platform, which reads as "unknown").
    unsafe { libc::sched_getcpu() }
}

/// Per-step host jitter, which is the number the median hides. A 48-vCPU box shares its
/// cores; the median of a round can be clean while a handful of steps are 3x the median
/// because the launch thread was preempted or a page fault landed on the n-gram table. So
/// the lane reports the tail and the outlier COUNT, not just the middle.
struct Jitter {
    n: usize,
    mean: f64,
    median: f64,
    stddev: f64,
    cv_pct: f64,
    p99: f64,
    max: f64,
    /// Steps exceeding 1.5x the median — the "something else ran" count.
    outliers_15x: usize,
}

fn jitter(samples: &[f64]) -> Jitter {
    if samples.is_empty() {
        return Jitter {
            n: 0,
            mean: 0.0,
            median: 0.0,
            stddev: 0.0,
            cv_pct: 0.0,
            p99: 0.0,
            max: 0.0,
            outliers_15x: 0,
        };
    }
    let mut s = samples.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    let mean = s.iter().sum::<f64>() / n as f64;
    let median = s[n / 2];
    let var = s.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
    let stddev = var.sqrt();
    Jitter {
        n,
        mean,
        median,
        stddev,
        cv_pct: 100.0 * stddev / mean.max(1e-12),
        p99: s[((n as f64 * 0.99) as usize).min(n - 1)],
        max: s[n - 1],
        outliers_15x: s.iter().filter(|&&v| v > 1.5 * median).count(),
    }
}

impl Jitter {
    fn receipt(&self) -> String {
        format!(
            "n={} mean={:.2} med={:.2} sd={:.3} cv={:.2}% p99={:.2} max={:.2} outliers_1.5x={}",
            self.n,
            self.mean,
            self.median,
            self.stddev,
            self.cv_pct,
            self.p99,
            self.max,
            self.outliers_15x
        )
    }
}

/// Advisory lock that serialises MEASUREMENT on a shared box without serialising the box, in two
/// modes: `Exclusive` around a timed block, `Shared` around work that must not race a timed block
/// but does not need protecting itself.
///
/// ## Why not `flock <lock> -c '<whole cell>'`
///
/// The lane's original convention wrapped a whole queue cell. At these depths that puts 11-80
/// minutes of PREFILL inside the exclusive window, during which no sibling may measure, to protect
/// a prefill wall the cell is not claiming. Held around the timed rounds instead, the exclusive
/// window per cell drops from ~30 minutes to ~2.
///
/// ## Why the obvious fix to THAT is also wrong, and what replaces it
///
/// The natural repair — "hold `LOCK_SH` across load+prefill and upgrade to `LOCK_EX` for the timed
/// block" — does not work, and the reason is worth keeping. `LOCK_EX` blocks until NO holder
/// remains, shared ones included, so a 25-minute prefill holding `LOCK_SH` blocks every exclusive
/// waiter for 25 minutes: the 30-minute window is rebuilt, with the stall moved onto whoever wants
/// to measure. Upgrading `LOCK_SH` to `LOCK_EX` on one fd is not atomic on Linux either — the
/// shared lock is dropped first — so two upgraders can interleave or deadlock.
///
/// So the shared lock is held **per prefill CHUNK**, released and re-acquired at every chunk
/// boundary. A waiting `LOCK_EX` gets in at the next boundary, and the prefill's following `LOCK_SH`
/// then blocks until the timed block finishes. **Prefills yield to measurements instead of racing
/// them**, the worst-case wait for an exclusive window is ONE CHUNK (~11 s at chunk 2048 with
/// `idxsel`), and nothing runs unlocked. The cost is two `flock` syscalls per ~11 seconds of work.
///
/// ## And it makes the discipline checkable rather than promised
///
/// `# measure-lock what=<block> mode=<ex|sh> path=<lock> waited_s=<n>` lands in the receipt. Today
/// "was this timing serialised?" is answerable only by reading the launcher that produced it — and
/// this lane has already discarded two banked arms for exactly that reason (`r2ab131k-off-1`, timed
/// with no lock file present; then a pair timed while three processes computed at once), each caught
/// by a sibling agent reading `ps`/`ls /tmp` rather than by the receipt. A receipt that cannot state
/// whether it was serialised cannot be read, which is the same defect as one that could not state
/// its own cache arm.
///
/// `waited_s` is the second signal: a long wait proves a sibling was holding the lock, and a zero
/// wait during a known battery is itself worth a second look.
///
/// Failure is HARD in both modes. A lock that cannot be opened or taken returns `Err` rather than
/// warning and proceeding, because "proceed unserialised" is precisely the invalid measurement the
/// flag exists to prevent, and a diagnostic written to end a silent failure must not itself be
/// silent.
#[derive(Copy, Clone, PartialEq, Eq)]
enum LockMode {
    /// Timed block: nothing else may compute.
    Exclusive,
    /// Untimed work that must not race a timed block. Yields to an exclusive waiter at the next
    /// release point, which for a prefill is the next chunk boundary.
    Shared,
}

impl LockMode {
    fn flag(self) -> libc::c_int {
        match self {
            Self::Exclusive => libc::LOCK_EX,
            Self::Shared => libc::LOCK_SH,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Exclusive => "ex",
            Self::Shared => "sh",
        }
    }
}

struct MeasureLock {
    _file: std::fs::File,
    path: String,
    mode: LockMode,
    waited_s: f64,
}

impl MeasureLock {
    fn acquire(path: &str, mode: LockMode) -> Res<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| format!("measure lock {path}: {e}"))?;
        let fd = std::os::unix::io::AsRawFd::as_raw_fd(&file);
        let t0 = Instant::now();
        // POLLED rather than blocking, and that is the whole point. A plain blocking `flock` turns
        // one specific mistake into a silent hang with no output: **nesting this lock inside an
        // outer `flock(1)` wrapper on the same path self-deadlocks.** flock locks are per
        // open-file-description, so taking LOCK_EX on a new fd while the PARENT process holds
        // LOCK_SH on the same file waits on one's own parent, forever. During a 25-80 minute
        // prefill that is indistinguishable from a slow box, which is exactly the class of failure
        // this lane keeps paying for — the diagnostic written to end a silent failure must not
        // itself be silent.
        //
        // So: retry non-blockingly, announce the wait every 60 s so a stall is VISIBLE in the log
        // while it is happening, and hard-fail at a deadline naming the self-deadlock as the first
        // suspect. `MEMRA_Q4E_MEASURE_LOCK_TIMEOUT_S` (default 7200) is generous on purpose — a
        // legitimate wait under the whole-invocation protocol is a sibling's entire cell.
        let timeout_s: f64 = std::env::var("MEMRA_Q4E_MEASURE_LOCK_TIMEOUT_S")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(7200.0);
        let mut announced = 0u64;
        loop {
            // SAFETY: `fd` is owned by `file` and outlives the call; flock only sets an advisory
            // lock on it. LOCK_NB makes it return EWOULDBLOCK instead of parking.
            let rc = unsafe { libc::flock(fd, mode.flag() | libc::LOCK_NB) };
            if rc == 0 {
                break;
            }
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() != Some(libc::EWOULDBLOCK) {
                return Err(
                    format!("measure lock {path}: flock {} failed: {err}", mode.name()).into(),
                );
            }
            let waited = t0.elapsed().as_secs_f64();
            if waited >= timeout_s {
                return Err(format!(
                    "measure lock {path}: flock {} still unavailable after {waited:.0}s \
                     (MEMRA_Q4E_MEASURE_LOCK_TIMEOUT_S={timeout_s:.0}). FIRST SUSPECT: this run is \
                     nested inside an outer `flock` wrapper on the SAME path — flock locks are \
                     per open-file-description, so this process is waiting on its own parent and \
                     will wait forever. Use EITHER the outer wrapper OR \
                     MEMRA_Q4E_MEASURE_LOCK, never both.",
                    mode.name()
                )
                .into());
            }
            if waited as u64 / 60 > announced {
                announced = waited as u64 / 60;
                println!(
                    "# measure-lock-waiting\tmode={}\tpath={path}\twaited_s={waited:.0}\t\
                     note=if nested inside an outer flock on this path, this will never return",
                    mode.name()
                );
                std::io::stdout().flush().ok();
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
        Ok(Self {
            _file: file,
            path: path.to_string(),
            mode,
            waited_s: t0.elapsed().as_secs_f64(),
        })
    }

    /// Convenience for the caller that only has an `Option<&str>` config: `None` means the flag is
    /// unset and there is no lock, which is the default and changes nothing.
    fn maybe(path: Option<&str>, mode: LockMode) -> Res<Option<Self>> {
        match path {
            Some(p) => Ok(Some(Self::acquire(p, mode)?)),
            None => Ok(None),
        }
    }

    fn receipt(&self, what: &str) -> String {
        format!(
            "# measure-lock\twhat={what}\tmode={}\tpath={}\twaited_s={:.1}\n",
            self.mode.name(),
            self.path,
            self.waited_s
        )
    }
}

impl Drop for MeasureLock {
    fn drop(&mut self) {
        let fd = std::os::unix::io::AsRawFd::as_raw_fd(&self._file);
        // SAFETY: same fd, still owned by `self`; LOCK_UN releases the advisory lock. Closing the
        // file would release it anyway — this is explicit so the release point is the end of the
        // guarded block rather than wherever the fd happens to drop, which is what makes the
        // per-chunk yield land at the chunk boundary.
        unsafe { libc::flock(fd, libc::LOCK_UN) };
    }
}

struct Args {
    ckpt: PathBuf,
    out: PathBuf,
    label: String,
    goldens: Option<PathBuf>,
    prompts: Option<PathBuf>,
    compare_logits: Option<PathBuf>,
    decode_timing: usize,
    profile: usize,
    profiler_window: bool,
    /// Interleaved A/B of one optimization seam: (reps, steps per arm).
    ab_moe: Option<(usize, usize)>,
    /// Which seam the A/B moves: "moe" (grouped expert matvec) or "hc" (fused gates).
    ab_seam: String,
    max_new: usize,
    opts: LoadOptions,
    /// Load the TP2 shard on device 1; decode-timing and `--ab-seam tp2` route decode
    /// through `decode_step_tp2`.
    tp2: bool,
    /// Free the trunk f32 originals whose bf16 twins serve (yarn follow-up 3).
    trunk_diet: bool,
    /// TP2-vs-single-card per-step logits gate over n decode steps (requires --tp2 +
    /// --goldens): tolerance class (split-GEMV/expert-half/join reduction reorders).
    tp2_gate: usize,
    /// TP2-PREFILL output gate: chunked TP2 prefill + decode vs the single-card
    /// chunked prefill + decode, same fed tokens; modelplan-class envelope.
    tp2_prefill_gate: usize,
    /// Measure the TP2 class gate's bands WITHOUT barring on them (the calibration run
    /// that sets the constants; calibrate-downward law - never widen a band to pass).
    tp2_class_calibrate: bool,
    /// Seam decode-row gate over n steps (with --ab-seam): OFF vs ON per-step logits
    /// envelope + KL + argmax — the decode-shape numeric instrument.
    seam_gate: usize,
    /// Draft-forward parity vs the HOST reference MTP twin over the goldens probe rows
    /// (requires --goldens; implies --mtp): batched pass + single-step chain, per-row
    /// argmax + envelope + KL. The mtp-spec lane's real-checkpoint deliverable-2 gate.
    draft_gate: bool,
    /// Verify-row BIT gate over n decode steps (requires --goldens; implies --mtp):
    /// plain t==1 decode rows vs verify chunks of spec_k+1 fed the SAME tokens — every
    /// value must be bit-identical (the spec byte-identity contract, row level).
    verify_bit: usize,
    /// DEEP-SEEDED verify-row bit gate: seed both states from a `--ladder-ids` corpus
    /// prefix of this many tokens instead of from `goldens.input_ids`, so the comparison
    /// actually crosses the QSA selection horizon. Row count comes from
    /// `--verify-bit-gate` (default 24).
    verify_bit_deep: usize,
    /// REWIND-row bit gate (mtp11): n plain rows re-derived through chunk+partial-
    /// rewind cycles per keep in 1..=spec_k, every committed row bit-compared.
    rewind_bit: usize,
    /// Replay an exact (t,keep) round pattern across rewinds (mtp11 diagnosis).
    rewind_replay: Option<PathBuf>,
    /// Spec byte-identity gate over n tokens per prompt (requires --prompts; implies
    /// --mtp): plain greedy chain vs spec_generate chain, token for token.
    spec_gate: usize,
    /// Interleaved plain-vs-spec A/B: (reps, tokens per arm). Requires --goldens.
    spec_ab: Option<(usize, usize)>,
    /// Spec K for the spec instruments (default 4).
    spec_k: usize,
    /// K ladder, e.g. "1,2,3,4,6,8" — one spec run per K over --spec-ab's token count
    /// (or 128). Requires --goldens.
    spec_ladder: Vec<usize>,
    /// ONE vendor-default SAMPLED spec run (temp 1.0 / top_p 0.95 / top_k 20, fixed
    /// seed) with the spec-engagement receipt (rounds with accepts > 0). Requires
    /// --goldens.
    spec_sampled: bool,
    /// Section-profile ONE spec run at this K (prof_section timers over the whole spec
    /// loop — draft + verify sections mixed; sync-bounded, shares are the signal).
    spec_profile: usize,
    /// OWN-GEN rank corpus (DRAFT-REGIME law 1): generate WITH this model over the
    /// real-shaped prompt pack, count only the EMITTED tokens, write the ranks sidecar +
    /// the coverage table. Runs with the FULL draft head (never a trimmed one).
    owngen: Option<PathBuf>,
    /// Greedy tokens per corpus prompt (the byte-identity instrument; capped to bound
    /// greedy-loop damage per the loop law).
    owngen_greedy: usize,
    /// Vendor-default SAMPLED tokens per corpus prompt per seed (the serving shape).
    owngen_sampled: usize,
    /// Distinct sampled seeds per corpus prompt.
    owngen_seeds: usize,
    /// Where the ranks sidecar lands (`id\tcount` per line, rank order).
    owngen_out: Option<PathBuf>,
    /// End-of-turn ids: a generation is counted up to and INCLUDING the first one
    /// (post-EOS continuation is off-distribution and would pollute the corpus).
    owngen_eos: Vec<u32>,
    /// SKIP corpus prompts longer than this (0 = no cap). Not a preference: a prompt's
    /// PREFILL transients scale with its length and both the trunk and the draft prefill
    /// it, so past some length the pair does not fit this artifact's post-load headroom at
    /// all — measured, not assumed (see the receipt line and PROFILE-6 provenance).
    owngen_max_prompt: usize,
    /// Generate at most this many NEW prompts per invocation, then stop (0 = no limit).
    /// Bounded chunks in a FRESH process are the documented recipe (DRAFT-REGIME's
    /// frspec-owngen --limit) and they also hand every chunk a clean device allocator.
    owngen_limit: usize,
    /// Resume ledger for the corpus: every finished generation's COUNTED ids appended as
    /// `index\tclass\tarm\tseed\thit_eos\tids-csv`. Rows already present are counted
    /// from the file instead of regenerated, so a crash or a bounded chunk costs nothing.
    owngen_corpus_out: Option<PathBuf>,
    /// FR-Spec draft-head trim: ranks sidecar to arm the draft head with.
    draft_trim: Option<PathBuf>,
    /// Trim width (top-N of the ranks file; 0 = every id in the file).
    draft_trim_n: usize,
    /// Interleaved A/B of the draft head: full-vocab arm vs trimmed arm at `spec_k`
    /// (reps, tokens per arm). Requires --draft-trim.
    trim_ab: Option<(usize, usize)>,
    /// Trim-WIDTH sweep: one spec run per N (rebuilding the trim each time) plus the
    /// full-vocab control, at `spec_k` over `--spec-ab`'s token count. The accept table
    /// the trim width is chosen from. Requires --draft-trim.
    trim_sweep: Vec<usize>,
    /// Interleaved A/B of the VERIFY scan-chain segment graphs (`set_verify_graphs`):
    /// (reps, tokens per arm) at `spec_k`. Chains must be identical across arms — a graph
    /// replay is bit-identical to the eager chain by construction.
    vgraph_ab: Option<(usize, usize)>,
    /// Interleaved A/B of the DEFERRED round readback (mtp11, `SpecOpts::defer`):
    /// (reps, tokens per arm) at `spec_k`, arms host / defer (+ defer-guard-sync when
    /// the p-min guard is armed). Chains AND admission counters must be identical
    /// across arms per rep — the deferred round is the same picks by construction.
    defer_ab: Option<(usize, usize)>,
    router_ab: Option<(usize, usize)>,
    /// Card-1 draft placement (mtp10): load the MTP block + a private lm-head copy on
    /// device 1 (`load_from_dir_dev1`); the spec loop P2P-crosses the wide seed rows and
    /// the receipts carry the measured crossing cost. Unlocks agentic-length prompts
    /// (the co-resident placement OOMs past ~400 prompt tokens — PROFILE-6 finding 2).
    mtp_dev1: bool,
    /// Bounded spec-admission options (all default OFF — flags law): p-min draft guard,
    /// adaptive per-round K (accepted+1), rolling dyn-K decay. Applied to every spec
    /// instrument in the run and printed in every receipt header.
    spec_opts: memra_engine::qwen4exp_gpu::SpecOpts,
    /// Per-round trace runs over EVERY prompt in --prompts: n tokens each at spec_k,
    /// greedy, with a plain byte-identity twin per prompt. The decay-diagnosis
    /// instrument (accept-vs-position, fork margins, carrier drift).
    spec_trace: usize,
    /// Long-context affordability ladder (yarn lane): ascending fill depths in tokens.
    ladder: Vec<usize>,
    /// Pre-tokenized REAL text ids (whitespace/comma separated), long enough for the
    /// deepest rung plus the per-rung decode continuations that stay in context.
    ladder_ids: Option<PathBuf>,
    ladder_chunk: usize,
    ladder_decode: usize,
    /// KV placement arm: QSA KV caches on card 1 (UVA P2P reads), trunk stays card 0.
    ladder_kv_dev1: bool,
    /// Run the (non-spec) ladder on the TP2 route: TP2-native state (halves at
    /// capacity, single-card KV stubbed), chunked TP2 prefill, TP2 decode timing.
    ladder_tp2: bool,
    /// Spec arm: per rung, a FRESH spec_generate_ext (chunked co-prefill + ring-bounded
    /// wide stash) at this K, under the CLI admission policy (--spec-pmin/--spec-adapt).
    /// Spec-at-depth SHAPE suffix: a prompts.tsv whose FIRST row's ids are appended to
    /// the end of the deep corpus fill, so the fed sequence is [deep document][chat-template
    /// turn] — the real long-agentic shape. Total fill stays == rung, so the VRAM and
    /// depth rows remain comparable with the raw arm.
    ladder_spec_shape: Option<PathBuf>,
    ladder_spec: Option<usize>,
    /// Run the spec ladder SAMPLED (vendor defaults, this seed) instead of greedy.
    ladder_spec_sampled: Option<u64>,
    /// WITHIN-PREFILL interleaved decode A/B over one seam name (262k host-lever lane).
    /// After each rung's normal timing row, alternate the seam OFF/ON across timed decode
    /// rounds on the SAME state and report per-arm medians.
    ///
    /// Why this exists rather than one process per arm: at 262,144 the prefill is ~25-80
    /// minutes and the lever under test is DECODE-ONLY, so the per-arm-process protocol
    /// spends 6 prefills (2.6-8 h of card time) to measure 36 decode steps, and it puts box
    /// clock drift between the arms — the exact defect the interleaved protocol exists to
    /// remove. Sharing one prefill makes the arms differ in the seam and in nothing else.
    ///
    /// SOUNDNESS BOUND, and it is the caller's to respect: a seam is eligible only if its
    /// state is rebuildable from the token history, so that arming it mid-run cannot leave
    /// stale state behind. `plecache` qualifies (a cold or stale id cache is rebuilt by
    /// longest-common-prefix compare, and its own oracle covers exactly that transition).
    /// A seam that mutates the KV cache or a device mirror in an arm-dependent way does NOT
    /// qualify and must use per-arm processes.
    ladder_ab_seam: Option<String>,
    /// Interleaved reps per arm (x3 default, escalating to x5 on anomaly).
    ladder_ab_rounds: usize,
    /// Timed steps per arm per rep, after `LADDER_AB_WARM` excluded warmup steps.
    ladder_ab_steps: usize,
    /// Host-side probe: per-thread context-switch census at round boundaries and
    /// launch-thread CPU sampled every decode step (migration count + host jitter tail).
    host_probe: bool,
    /// Serialise TIMED blocks only against this lock path (see `MeasureLock`). Also read from
    /// `MEMRA_Q4E_MEASURE_LOCK` so a queue can set it once for every cell.
    measure_lock: Option<String>,
}

fn parse_args() -> Res<Args> {
    let mut it = std::env::args().skip(1);
    let usage = "usage: qwen4exp_real_gate <ckpt_dir> <out_dir> --label <label> \
                 [--goldens <dump_dir>] [--prompts <prompts.tsv>] [--compare-logits <bin>] \
                 [--decode-timing <n>] [--profile <n>] [--profiler-window] \
                 [--ab-moe <reps>x<steps>] [--ab-seam moe|hc|trunk|ws|graph|selv2|hcmicro|selv3|gdnstep|gdnfuse|projstack|hcdiet|gufuse|routerb16] [--max-new <n>] [--host-bf16-banks] \
                 [--indexer-norm-raw] [--mtp] [--draft-gate] [--verify-bit-gate <n>] [--verify-bit-deep <fill>] \
                 [--rewind-bit-gate <n>] [--spec-gate <n>] [--spec-ab <reps>x<tokens>] [--spec-k <k>] \
                 [--spec-ladder <k1,k2,..>] [--spec-sampled] [--vmt on|off] \
                 [--spec-profile <k>] \
                 [--owngen <corpus-prompts.tsv> --owngen-out <ranks.txt> \
                  [--owngen-greedy <n>] [--owngen-sampled <n>] [--owngen-seeds <n>] \
                  [--owngen-eos <id,id>] [--owngen-corpus-out <corpus-ids.tsv>] [--owngen-limit <n>] [--owngen-max-prompt <n>]] \
                 [--draft-trim <ranks.txt>] [--draft-trim-n <N>] [--trim-ab <reps>x<tokens>] \
                 [--trim-sweep <n1,n2,..>] [--vgraph-ab <reps>x<tokens>] \
                 [--mtp-dev1] [--spec-pmin <p>] [--spec-adapt <k_lo>] \
                 [--spec-dynk <window>,<thr>,<k_floor>] [--spec-trace <tokens>] \
                 [--spec-defer] [--spec-defer-guard-sync] [--defer-ab <reps>x<tokens>] \
                 [--ladder-ab-seam <seam> [--ladder-ab-rounds <n>] [--ladder-ab-steps <n>]] \
                 [--host-probe] [--measure-lock <path>]";
    let ckpt = PathBuf::from(it.next().ok_or(usage)?);
    let out = PathBuf::from(it.next().ok_or(usage)?);
    let mut args = Args {
        ckpt,
        out,
        label: String::new(),
        goldens: None,
        prompts: None,
        compare_logits: None,
        decode_timing: 0,
        profile: 0,
        profiler_window: false,
        ab_moe: None,
        ab_seam: "moe".to_string(),
        max_new: 64,
        opts: LoadOptions::default(),
        tp2: false,
        trunk_diet: false,
        tp2_gate: 0,
        tp2_prefill_gate: 0,
        tp2_class_calibrate: false,
        ladder_spec_shape: None,
        seam_gate: 0,
        draft_gate: false,
        verify_bit: 0,
        verify_bit_deep: 0,
        rewind_bit: 0,
        rewind_replay: None,
        spec_gate: 0,
        spec_ab: None,
        spec_k: 4,
        spec_ladder: Vec::new(),
        spec_sampled: false,
        spec_profile: 0,
        owngen: None,
        owngen_greedy: 256,
        owngen_sampled: 0,
        owngen_seeds: 1,
        owngen_out: None,
        owngen_eos: vec![248046, 248044],
        owngen_max_prompt: 0,
        owngen_limit: 0,
        owngen_corpus_out: None,
        draft_trim: None,
        draft_trim_n: 0,
        trim_ab: None,
        trim_sweep: Vec::new(),
        vgraph_ab: None,
        defer_ab: None,
        router_ab: None,
        mtp_dev1: false,
        spec_opts: memra_engine::qwen4exp_gpu::SpecOpts::default(),
        spec_trace: 0,
        ladder: Vec::new(),
        ladder_ids: None,
        ladder_chunk: 8192,
        ladder_decode: 96,
        ladder_kv_dev1: false,
        ladder_tp2: false,
        ladder_spec: None,
        ladder_spec_sampled: None,
        ladder_ab_seam: None,
        ladder_ab_rounds: 3,
        ladder_ab_steps: 16,
        host_probe: false,
        measure_lock: std::env::var("MEMRA_Q4E_MEASURE_LOCK")
            .ok()
            .filter(|v| !v.is_empty()),
    };
    while let Some(flag) = it.next() {
        let mut value = |name: &str| -> Res<String> {
            it.next()
                .ok_or_else(|| format!("{name} needs a value").into())
        };
        match flag.as_str() {
            "--label" => args.label = value("--label")?,
            "--goldens" => args.goldens = Some(PathBuf::from(value("--goldens")?)),
            "--prompts" => args.prompts = Some(PathBuf::from(value("--prompts")?)),
            "--compare-logits" => {
                args.compare_logits = Some(PathBuf::from(value("--compare-logits")?))
            }
            "--decode-timing" => args.decode_timing = value("--decode-timing")?.parse()?,
            "--profile" => args.profile = value("--profile")?.parse()?,
            // cuProfilerStart/Stop around the decode-timing steps (skipping the first 2),
            // so `nsys profile --capture-range=cudaProfilerApi` counts EXACTLY the warm
            // decode window's kernel launches and memcpys.
            "--profiler-window" => args.profiler_window = true,
            // "<reps>x<steps>": per rep, run BOTH arms (per-expert then grouped) with a
            // fresh state + probe prefill + 4 warmup steps each — the interleaved-x5 law.
            "--ab-moe" => {
                let spec = value("--ab-moe")?;
                let (reps, steps) = spec
                    .split_once('x')
                    .ok_or("--ab-moe wants <reps>x<steps>")?;
                args.ab_moe = Some((reps.parse()?, steps.parse()?));
            }
            "--ab-seam" => args.ab_seam = value("--ab-seam")?,
            "--max-new" => args.max_new = value("--max-new")?.parse()?,
            "--tp2" => args.tp2 = true,
            "--trunk-diet" => args.trunk_diet = true,
            "--tp2-prefill-gate" => {
                args.tp2_prefill_gate = value("--tp2-prefill-gate")?.parse()?;
            }
            "--tp2-gate" => {
                args.tp2 = true;
                args.tp2_gate = value("--tp2-gate")?.parse()?;
            }
            "--seam-gate" => args.seam_gate = value("--seam-gate")?.parse()?,
            "--host-bf16-banks" => args.opts.host_bf16_banks = true,
            "--indexer-norm-raw" => args.opts.indexer_norm_raw = true,
            "--mtp" => args.opts.load_mtp = true,
            "--draft-gate" => {
                args.opts.load_mtp = true;
                args.draft_gate = true;
            }
            "--rewind-bit-replay" => {
                args.opts.load_mtp = true;
                args.rewind_replay = Some(PathBuf::from(value("--rewind-bit-replay")?));
            }
            "--rewind-bit-gate" => {
                args.opts.load_mtp = true;
                args.rewind_bit = value("--rewind-bit-gate")?.parse()?;
            }
            "--verify-bit-deep" => {
                args.verify_bit_deep = value("--verify-bit-deep")?.parse()?;
            }
            "--verify-bit-gate" => {
                args.opts.load_mtp = true;
                args.verify_bit = value("--verify-bit-gate")?.parse()?;
            }
            "--spec-gate" => {
                args.opts.load_mtp = true;
                args.spec_gate = value("--spec-gate")?.parse()?;
            }
            "--spec-ab" => {
                args.opts.load_mtp = true;
                let spec = value("--spec-ab")?;
                let (reps, toks) = spec
                    .split_once('x')
                    .ok_or("--spec-ab wants <reps>x<tokens>")?;
                args.spec_ab = Some((reps.parse()?, toks.parse()?));
            }
            "--spec-k" => args.spec_k = value("--spec-k")?.parse()?,
            "--spec-ladder" => {
                args.opts.load_mtp = true;
                args.spec_ladder = value("--spec-ladder")?
                    .split(',')
                    .map(|s| s.parse::<usize>())
                    .collect::<Result<_, _>>()?;
            }
            "--spec-sampled" => {
                args.opts.load_mtp = true;
                args.spec_sampled = true;
            }
            "--spec-profile" => {
                args.opts.load_mtp = true;
                args.spec_profile = value("--spec-profile")?.parse()?;
            }
            "--owngen" => {
                args.opts.load_mtp = true;
                args.owngen = Some(PathBuf::from(value("--owngen")?));
            }
            "--owngen-greedy" => args.owngen_greedy = value("--owngen-greedy")?.parse()?,
            "--owngen-sampled" => args.owngen_sampled = value("--owngen-sampled")?.parse()?,
            "--owngen-seeds" => args.owngen_seeds = value("--owngen-seeds")?.parse()?,
            "--owngen-out" => args.owngen_out = Some(PathBuf::from(value("--owngen-out")?)),
            "--owngen-limit" => args.owngen_limit = value("--owngen-limit")?.parse()?,
            "--owngen-max-prompt" => {
                args.owngen_max_prompt = value("--owngen-max-prompt")?.parse()?
            }
            "--owngen-corpus-out" => {
                args.owngen_corpus_out = Some(PathBuf::from(value("--owngen-corpus-out")?))
            }
            "--owngen-eos" => {
                args.owngen_eos = value("--owngen-eos")?
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse::<u32>())
                    .collect::<Result<_, _>>()?;
            }
            "--draft-trim" => {
                args.opts.load_mtp = true;
                args.draft_trim = Some(PathBuf::from(value("--draft-trim")?));
            }
            "--draft-trim-n" => args.draft_trim_n = value("--draft-trim-n")?.parse()?,
            "--vgraph-ab" => {
                args.opts.load_mtp = true;
                let spec = value("--vgraph-ab")?;
                let (reps, toks) = spec
                    .split_once('x')
                    .ok_or("--vgraph-ab wants <reps>x<tokens>")?;
                args.vgraph_ab = Some((reps.parse()?, toks.parse()?));
            }
            "--trim-sweep" => {
                args.opts.load_mtp = true;
                args.trim_sweep = value("--trim-sweep")?
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse::<usize>())
                    .collect::<Result<_, _>>()?;
            }
            "--trim-ab" => {
                args.opts.load_mtp = true;
                let spec = value("--trim-ab")?;
                let (reps, toks) = spec
                    .split_once('x')
                    .ok_or("--trim-ab wants <reps>x<tokens>")?;
                args.trim_ab = Some((reps.parse()?, toks.parse()?));
            }
            "--vmt" => match value("--vmt")?.as_str() {
                "on" => memra_engine::qwen4exp_gpu::set_verify_mt(true),
                "off" => memra_engine::qwen4exp_gpu::set_verify_mt(false),
                other => return Err(format!("--vmt {other}: want on|off").into()),
            },
            "--mtp-dev1" => {
                args.opts.load_mtp = true;
                args.mtp_dev1 = true;
            }
            "--spec-pmin" => args.spec_opts.pmin = value("--spec-pmin")?.parse()?,
            "--spec-defer" => {
                args.opts.load_mtp = true;
                args.spec_opts.defer = true;
            }
            "--spec-defer-guard-sync" => {
                args.opts.load_mtp = true;
                args.spec_opts.defer = true;
                args.spec_opts.defer_guard_sync = true;
            }
            "--router-ab" => {
                let spec = value("--router-ab")?;
                let (reps, toks) = spec
                    .split_once('x')
                    .ok_or("--router-ab wants <reps>x<tokens>")?;
                args.router_ab = Some((reps.parse()?, toks.parse()?));
            }
            "--defer-ab" => {
                args.opts.load_mtp = true;
                let spec = value("--defer-ab")?;
                let (reps, toks) = spec
                    .split_once('x')
                    .ok_or("--defer-ab wants <reps>x<tokens>")?;
                args.defer_ab = Some((reps.parse()?, toks.parse()?));
            }
            "--spec-adapt" => args.spec_opts.adapt_k_lo = Some(value("--spec-adapt")?.parse()?),
            "--spec-dynk" => {
                let spec = value("--spec-dynk")?;
                let parts: Vec<&str> = spec.split(',').collect();
                if parts.len() != 3 {
                    return Err("--spec-dynk wants <window>,<thr>,<k_floor>".into());
                }
                args.spec_opts.dynk = Some(memra_engine::qwen4exp_gpu::DynKCfg {
                    window: parts[0].parse()?,
                    thr: parts[1].parse()?,
                    k_floor: parts[2].parse()?,
                });
            }
            "--spec-trace" => {
                args.opts.load_mtp = true;
                args.spec_trace = value("--spec-trace")?.parse()?;
            }
            "--ladder" => {
                args.ladder = value("--ladder")?
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse::<usize>())
                    .collect::<Result<_, _>>()?;
            }
            "--ladder-ids" => args.ladder_ids = Some(PathBuf::from(value("--ladder-ids")?)),
            "--ladder-chunk" => args.ladder_chunk = value("--ladder-chunk")?.parse()?,
            "--ladder-decode" => args.ladder_decode = value("--ladder-decode")?.parse()?,
            "--ladder-kv-dev1" => args.ladder_kv_dev1 = true,
            "--tp2-class-calibrate" => args.tp2_class_calibrate = true,
            "--ladder-tp2" => args.ladder_tp2 = true,
            "--ladder-spec" => {
                args.opts.load_mtp = true;
                args.ladder_spec = Some(value("--ladder-spec")?.parse()?);
            }
            "--ladder-spec-shape" => {
                args.ladder_spec_shape = Some(PathBuf::from(value("--ladder-spec-shape")?))
            }
            "--ladder-spec-sampled" => {
                args.ladder_spec_sampled = Some(value("--ladder-spec-sampled")?.parse()?);
            }
            "--ladder-ab-seam" => args.ladder_ab_seam = Some(value("--ladder-ab-seam")?),
            "--ladder-ab-rounds" => args.ladder_ab_rounds = value("--ladder-ab-rounds")?.parse()?,
            "--ladder-ab-steps" => args.ladder_ab_steps = value("--ladder-ab-steps")?.parse()?,
            "--host-probe" => args.host_probe = true,
            "--measure-lock" => args.measure_lock = Some(value("--measure-lock")?),
            other => return Err(format!("unknown flag {other}\n{usage}").into()),
        }
    }
    if args.label.is_empty() {
        return Err(usage.into());
    }
    if args.owngen.is_some() != args.owngen_out.is_some() {
        return Err("--owngen and --owngen-out go together".into());
    }
    if args.trim_ab.is_some() && args.draft_trim.is_none() {
        return Err("--trim-ab needs --draft-trim (the ranks artifact is the ON arm)".into());
    }
    if !args.trim_sweep.is_empty() && args.draft_trim.is_none() {
        return Err("--trim-sweep needs --draft-trim (the ranks artifact it slices)".into());
    }
    if args.mtp_dev1 && args.tp2 {
        return Err("--mtp-dev1 and --tp2 both claim device 1; pick one route".into());
    }
    // Validate the A/B seam name HERE, before anything is loaded. The lane's own lesson: a
    // typo'd seam name that only surfaces after the rung's prefill costs 25-80 minutes of
    // card time and banks nothing (LADDER §5, the ascending ladder cannot bank partial
    // rungs). `set_seam` is the same name table `MEMRA_Q4E_SEAMS` uses, so the check and the
    // action cannot drift apart.
    if let Some(seam) = args.ladder_ab_seam.as_deref() {
        if args.ladder.is_empty() {
            return Err("--ladder-ab-seam is a --ladder instrument".into());
        }
        if !memra_engine::qwen4exp_gpu::seam_exists(seam) {
            return Err(format!(
                "--ladder-ab-seam {seam:?}: not a MEMRA_Q4E_SEAMS name (checked against the \
                 same table apply_env_seams uses)"
            )
            .into());
        }
        if args.ladder_ab_rounds < 2 || args.ladder_ab_steps < 4 {
            return Err("--ladder-ab-seam needs >=2 rounds and >=4 steps per arm".into());
        }
    }
    if !args.ladder.is_empty() {
        if args.ladder_ids.is_none() {
            return Err("--ladder needs --ladder-ids (pre-tokenized real text)".into());
        }
        if args.tp2 && !args.ladder_tp2 {
            return Err(
                "--ladder is a single-card-route instrument (kv-dev1 is the \
                        two-card arm); add --ladder-tp2 for the TP2-route ladder"
                    .into(),
            );
        }
        if args.ladder_tp2 {
            if !args.tp2 {
                return Err("--ladder-tp2 requires --tp2 (the shard)".into());
            }
            if args.ladder_kv_dev1 {
                return Err("--ladder-tp2 and --ladder-kv-dev1 are different routes".into());
            }
            if args.ladder_spec.is_some() {
                return Err("--ladder-tp2 has no spec arm (spec at depth is single-card)".into());
            }
        }
        if !args.ladder.windows(2).all(|w| w[0] < w[1]) {
            return Err("--ladder rungs must ascend (partial results bank in order)".into());
        }
    }
    Ok(args)
}

fn main() -> Res<()> {
    let args = parse_args()?;
    std::fs::create_dir_all(&args.out)?;
    let executable = std::fs::read(std::env::current_exe()?)?;
    let sha256 = Sha256::digest(&executable)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let mut header = format!(
        "# qwen4exp_real_gate\tlabel={}\tbinary_sha256={sha256}\tckpt={}\thost_bf16_banks={}\tindexer_norm_raw={}\tmtp_dev1={}\tspec_opts={:?}\n\
         # timing lines are UNTUNED EAGER wall clocks under correctness-arm residency — NOT perf claims\n",
        args.label,
        args.ckpt.display(),
        args.opts.host_bf16_banks,
        args.opts.indexer_norm_raw,
        args.mtp_dev1,
        args.spec_opts,
    );

    // Gate instrumentation: force not-yet-default seams for correctness runs
    // (MEMRA_Q4E_SEAMS, flags law — receipts precede the default flip).
    memra_engine::qwen4exp_gpu::apply_env_seams();
    // Reference-parity comparisons (hidden/greedy vs the BANKED transformers goldens) run
    // the f32 exactness-instrument cache arms regardless of the flipped serving defaults
    // (kvq ON / idxq q8): a quantized cache leaks cross-config drift into same-config
    // gates (tiny-gate receipt 2026-08-31). --ab-seam/--seam-gate arms and an explicit
    // MEMRA_Q4E_SEAMS entry still set their own state and win over this pin.
    //
    // SCOPED to the runs that actually make a golden comparison. It used to be
    // UNCONDITIONAL, which its own comment did not say and which quietly made this binary
    // an f32-only instrument: the round-2 ladder, the spec-at-depth cells and the TP2 gates
    // are not reference-parity runs, and every one of them silently measured the f32 cache
    // while reporting itself as running the ship defaults. It cost a full 100k ladder rung
    // (~9 min of prefill) before a state-alloc probe against MEMRA_Q4E_SEAMS=kvq showed
    // 91,475 MiB against the default arm's 96,243 MiB — a 4,768 MiB difference that the
    // default was supposed to be delivering and was not.
    //
    // A golden comparison is exactly: --goldens (hidden gate, verify-bit, draft/tp2 gates
    // that read goldens.input_ids) or --prompts (greedy first-divergence against banked
    // golden token chains). Everything else — the ladder, decode timing, the A/B seams —
    // is same-config or self-consistency and MUST see the serving defaults, or it is
    // measuring an arm nobody ships.
    let seams_env = std::env::var("MEMRA_Q4E_SEAMS").unwrap_or_default();
    let golden_comparison = args.goldens.is_some() || args.prompts.is_some();
    if golden_comparison {
        if !seams_env.contains("kvq") {
            memra_engine::qwen4exp_gpu::set_kv_quant(false);
        }
        if !seams_env.contains("idxq") {
            memra_engine::qwen4exp_gpu::set_idxq("f32");
        }
    }
    // EVERY receipt records the cache arm it actually ran. Without this line a receipt
    // cannot be read: a ladder rung that measured the f32 cache and a ladder rung that
    // measured the ship default produce identical-looking headers, and the round-2 ladder
    // was written up as "kvq ship defaults" on exactly that ambiguity. `golden_pin` names
    // WHY the arm is what it is, so an f32 row under --goldens reads as intended rather
    // than as a bug.
    header.push_str(&format!(
        "# cache\tkv_quant={}\tidxq={}\tgolden_pin={}\tseams_env={}\n",
        if memra_engine::qwen4exp_gpu::kv_quant_is_on() {
            "q8_0/q5_1"
        } else {
            "f32"
        },
        memra_engine::qwen4exp_gpu::idxq_mode_name(),
        golden_comparison,
        if seams_env.is_empty() {
            "(unset)"
        } else {
            seams_env.as_str()
        },
    ));
    print!("{header}");
    let engine = Engine::new(0)?;
    // Card-1 draft placement (mtp10): a second engine + P2P for the wide-row crossings.
    let draft_engine: Option<Engine> = if args.mtp_dev1 {
        let e1 = Engine::new(1)?;
        memra_engine::qwen4exp_gpu::tp2_enable_p2p(&engine, &e1)?;
        Some(e1)
    } else {
        None
    };
    println!("# vram\tpost-engine\t{}", nvidia_smi());
    let t_load = Instant::now();
    let mut tp2: Option<(Engine, memra_engine::qwen4exp_gpu::Tp2Shard)> = None;
    let mut draft_ref = None;
    let mut model = if args.tp2 {
        let e1 = Engine::new(1)?;
        let (m, shard) = Qwen4ExpGpu::load_from_dir_tp2(&engine, &e1, &args.ckpt, args.opts)?;
        tp2 = Some((e1, shard));
        m
    } else if args.draft_gate {
        // One checkpoint read serves both sides: clone the mtp rows + banks for the
        // HOST reference twin before the engine consumes the checkpoint.
        let ck = read_checkpoint_with(&args.ckpt, args.opts)?;
        draft_ref = Some(ck.mtp_reference_weights()?);
        Qwen4ExpGpu::from_loaded_checkpoint_dual(&engine, draft_engine.as_ref(), ck)?
    } else if let Some(e1) = draft_engine.as_ref() {
        Qwen4ExpGpu::load_from_dir_dev1(&engine, e1, &args.ckpt, args.opts)?
    } else {
        Qwen4ExpGpu::load_from_dir_with(&engine, &args.ckpt, args.opts)?
    };
    // The DRAFT engine: card 1 under --mtp-dev1, else the trunk's own card. Every draft
    // state / spec call below presents this one.
    let de: &Engine = draft_engine.as_ref().unwrap_or(&engine);
    if args.profile > 0 && args.tp2 {
        return Err("--profile is a single-card instrument; drop --tp2".into());
    }
    let load_s = t_load.elapsed().as_secs_f64();
    let vram_load = nvidia_smi();
    println!("# load\t{load_s:.1}s\n# vram\tpost-load\t{vram_load}");
    // Trunk f32 diet (yarn follow-up 3): free the f32 originals whose bf16 twins serve
    // under the ship seams; the receipt is the before/after VRAM pair above/below.
    if args.trunk_diet {
        let freed = model.trunk_f32_diet(&engine)?;
        println!(
            "# trunk-diet\tfreed={:.1}MiB\n# vram\tpost-diet\t{}",
            freed as f64 / (1024.0 * 1024.0),
            nvidia_smi()
        );
    }
    let vocab = model.plan.vocab_size as usize;

    // ---------------------------------------------------------------- own-gen rank corpus
    // DRAFT-REGIME law 1: ranks are a distribution artifact of THIS artifact, derived from
    // ITS OWN generations. Prompts are input only; the counted distribution is the emitted
    // tokens. The draft head is FULL-VOCAB here by construction (the trim is armed after
    // this block), so the corpus can never be biased by a trim it is about to define.
    if let (Some(path), Some(out_path)) = (args.owngen.as_ref(), args.owngen_out.as_ref()) {
        let corpus = read_corpus_prompts(path)?;
        let k = args.spec_k;
        let cut = |ids: &[u32]| -> (usize, bool) {
            match ids.iter().position(|t| args.owngen_eos.contains(t)) {
                // Count the end-of-turn token itself once — the draft must be able to
                // propose it — then stop: post-EOS continuation is off-distribution.
                Some(p) => (p + 1, true),
                None => (ids.len(), false),
            }
        };
        let mut counts = vec![0u64; vocab];
        let mut per_class: std::collections::BTreeMap<String, Vec<u64>> =
            std::collections::BTreeMap::new();
        let mut receipt = header.clone();
        receipt.push_str(&format!(
            "# owngen: own-generated rank corpus, k={k}, FULL-VOCAB draft head, greedy={} sampled={}x{} eos={:?}\n\
             # provenance: prompts are REAL-SHAPED (goldens continuations + chat-template renders \
             composed on the box); the counted distribution is this artifact's OWN emitted tokens\n\
             prompt\tclass\tarm\tseed\tprompt_tokens\tgen_tokens\tcounted\thit_eos\taccept_rate\tmean_accept_len\tms_per_token\n",
            args.owngen_greedy, args.owngen_sampled, args.owngen_seeds, args.owngen_eos
        ));
        let t_corpus = Instant::now();
        let mut arms: Vec<(&str, Option<u64>, usize)> = Vec::new();
        if args.owngen_greedy > 0 {
            arms.push(("greedy", None, args.owngen_greedy));
        }
        for s in 0..args.owngen_seeds {
            if args.owngen_sampled > 0 {
                arms.push(("sampled", Some(0x5eed_0000 + s as u64), args.owngen_sampled));
            }
        }
        // Capacity is PER GENERATION, deliberately, after measuring both alternatives.
        // Uniform (max-over-pack) capacity was tried and is WRONG here: a trunk state + a
        // draft state + their workspaces at the longest prompt is ~2 GiB against ~2.6 GiB
        // of post-load headroom, so making every generation pay the maximum put a single
        // generation on the edge — a CLEAN process then OOM'd on its FIRST generation once
        // the longest prompt ran first. Fragmentation across many differently-sized cycles
        // is real but it is the chunking's job (`--owngen-limit`, a fresh process and so a
        // clean allocator every N prompts), not the capacity's.
        let cap_for = |prompt_len: usize, toks: usize| prompt_len + toks + k + 4;
        // Resume ledger (the frspec-owngen --corpus-out pattern): every finished
        // generation's COUNTED ids are appended, so a crash or a bounded chunk costs
        // nothing and the raw corpus itself is a bankable artifact. Rows already present
        // are counted from the file and NOT regenerated.
        let mut done: std::collections::BTreeSet<(usize, String)> =
            std::collections::BTreeSet::new();
        let mut resumed_rows = 0usize;
        if let Some(path) = args.owngen_corpus_out.as_ref() {
            if path.exists() {
                for line in std::fs::read_to_string(path)?.lines() {
                    let f: Vec<&str> = line.split('\t').collect();
                    if f.len() != 6 || line.starts_with('#') {
                        continue;
                    }
                    let (index, class, key): (usize, &str, String) =
                        (f[0].parse()?, f[1], format!("{}:{}", f[2], f[3]));
                    let class_counts = per_class
                        .entry(class.to_string())
                        .or_insert_with(|| vec![0u64; vocab]);
                    for tok in f[5].split(',').filter(|s| !s.is_empty()) {
                        let tok: u32 = tok.parse()?;
                        counts[tok as usize] += 1;
                        class_counts[tok as usize] += 1;
                    }
                    done.insert((index, key));
                    resumed_rows += 1;
                }
                println!(
                    "# owngen resume: {resumed_rows} generations already banked in {}",
                    path.display()
                );
            }
        }
        receipt.push_str(&format!(
            "# state_capacity=PER-GENERATION (prompt+tokens+k+4; uniform max-over-pack was \
             measured WORSE — one generation at the pack maximum does not fit the ~2.6 GiB \
             headroom)\tmax_cap={}\tresumed_rows={resumed_rows}\tlimit={}\n",
            cap_for(
                corpus.iter().map(|p| p.ids.len()).max().unwrap_or(0),
                arms.iter().map(|&(_, _, t)| t).max().unwrap_or(0)
            ),
            args.owngen_limit
        ));
        // LONGEST PROMPT FIRST: the big allocations land while the heap is clean, so the
        // shorter ones fit in the holes they leave rather than the reverse. Counts are
        // order-independent and the ledger keys on prompt INDEX, so ordering changes
        // neither the artifact nor resumability.
        let mut order: Vec<&CorpusPrompt> = corpus.iter().collect();
        order.sort_by(|a, b| b.ids.len().cmp(&a.ids.len()).then(a.index.cmp(&b.index)));
        let mut generated = 0usize;
        let mut skipped_long: Vec<(usize, usize)> = Vec::new();
        for p in order {
            if args.owngen_max_prompt > 0 && p.ids.len() > args.owngen_max_prompt {
                skipped_long.push((p.index, p.ids.len()));
                continue;
            }
            if args.owngen_limit > 0 && generated >= args.owngen_limit {
                println!(
                    "# owngen limit {} reached — rerun the same command to continue",
                    args.owngen_limit
                );
                break;
            }
            let mut did_any = false;
            for (arm, seed, toks) in &arms {
                let seed_s = seed.map(|s| format!("{s:#x}")).unwrap_or("-".into());
                if done.contains(&(p.index, format!("{arm}:{seed_s}"))) {
                    continue;
                }
                did_any = true;
                let cap = cap_for(p.ids.len(), *toks);
                let mut ss = model.alloc_state(&engine, cap)?;
                let mut ds = model.mtp_state(de, cap)?;
                let sampler = seed.map(|seed| memra_engine::qwen4exp_gpu::SpecSamplerCfg {
                    temperature: 1.0,
                    top_p: 0.95,
                    top_k: 20,
                    seed,
                });
                let report = model.spec_generate_ext(
                    &engine,
                    de,
                    &p.ids,
                    *toks,
                    k,
                    &mut ss,
                    &mut ds,
                    sampler,
                    args.spec_opts,
                    None,
                )?;
                let (n_counted, hit_eos) = cut(&report.tokens);
                let class_counts = per_class
                    .entry(p.class.clone())
                    .or_insert_with(|| vec![0u64; vocab]);
                for &t in &report.tokens[..n_counted] {
                    counts[t as usize] += 1;
                    class_counts[t as usize] += 1;
                }
                if let Some(path) = args.owngen_corpus_out.as_ref() {
                    use std::io::Write as _;
                    let mut f = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(path)?;
                    writeln!(
                        f,
                        "{}\t{}\t{arm}\t{seed_s}\t{hit_eos}\t{}",
                        p.index,
                        p.class,
                        csv(&report.tokens[..n_counted])
                    )?;
                }
                let per_tok =
                    (report.total_ms - report.prefill_ms) / report.tokens.len().max(1) as f64;
                receipt.push_str(&format!(
                    "{}\t{}\t{arm}\t{seed_s}\t{}\t{}\t{n_counted}\t{hit_eos}\t{:.3}\t{:.2}\t{per_tok:.2}\n",
                    p.index,
                    p.class,
                    p.ids.len(),
                    report.tokens.len(),
                    report.accept_rate(),
                    report.mean_accept_len(),
                ));
            }
            if !did_any {
                continue;
            }
            generated += 1;
            // VRAM per prompt, in the RECEIPT: the corpus OOM'd twice, and a growth trend
            // across generations (vs a flat line) is the difference between fragmentation
            // and a leak. Guessing between those cost two runs; measuring it costs a line.
            let vram = nvidia_smi();
            receipt.push_str(&format!(
                "# vram\tafter_prompt={}\tgenerated_this_run={generated}\t{vram}\n",
                p.index
            ));
            println!(
                "# owngen prompt {} ({}) done — {:.0}s elapsed, generated={generated}, vram {vram}",
                p.index,
                p.class,
                t_corpus.elapsed().as_secs_f64()
            );
        }
        if !skipped_long.is_empty() {
            // A NAMED coverage gap, not a silent one: these classes are under-represented
            // in the ranks and the trim can therefore under-propose on long contexts.
            receipt.push_str(&format!(
                "# skipped_over_max_prompt={}\tmax_prompt={}\tindices_lens={:?}\n",
                skipped_long.len(),
                args.owngen_max_prompt,
                skipped_long
            ));
            println!(
                "# owngen: SKIPPED {} prompts over --owngen-max-prompt {} — a named coverage gap: {:?}",
                skipped_long.len(),
                args.owngen_max_prompt,
                skipped_long
            );
        }
        // Rank order: count desc, id asc on ties (deterministic artifact).
        let mut ranked: Vec<u32> = (0..vocab as u32)
            .filter(|&i| counts[i as usize] > 0)
            .collect();
        ranked.sort_by(|&a, &b| counts[b as usize].cmp(&counts[a as usize]).then(a.cmp(&b)));
        let total: u64 = counts.iter().sum();
        receipt.push_str(&format!(
            "# corpus\ttotal_counted_tokens={total}\tdistinct_ids={}\tvocab={vocab}\tseconds={:.0}\n",
            ranked.len(),
            t_corpus.elapsed().as_secs_f64()
        ));
        // Coverage ladder: the top-N prefix's share of the counted distribution, globally
        // and per CLASS (the coverage law: a trim wins the cells it covers, loses the
        // cells it does not). Law 1's corpus floor is >= 4x topN counted tokens.
        let ladder: Vec<usize> = [1024usize, 2048, 4096, 8192, 16384, 32768, 65536]
            .into_iter()
            .filter(|&n| n <= vocab)
            .collect();
        receipt.push_str("# coverage table (share of counted tokens inside the global top-N)\n");
        receipt.push_str("topN\tglobal_coverage\tfloor_4x_tokens\tfloor_met");
        for class in per_class.keys() {
            receipt.push_str(&format!("\tcov_{class}"));
        }
        receipt.push('\n');
        for &n in &ladder {
            let take = n.min(ranked.len());
            let covered: u64 = ranked[..take].iter().map(|&i| counts[i as usize]).sum();
            receipt.push_str(&format!(
                "{n}\t{:.5}\t{}\t{}",
                covered as f64 / total.max(1) as f64,
                4 * n,
                total >= 4 * n as u64
            ));
            for cc in per_class.values() {
                let ctotal: u64 = cc.iter().sum();
                let ccov: u64 = ranked[..take].iter().map(|&i| cc[i as usize]).sum();
                receipt.push_str(&format!("\t{:.5}", ccov as f64 / ctotal.max(1) as f64));
            }
            receipt.push('\n');
        }
        let mut sidecar = String::new();
        sidecar.push_str(&format!(
            "# qwen4_exp own-gen ranks\tbinary_sha256={sha256}\tckpt={}\tcounted_tokens={total}\tdistinct={}\n\
             # rank order: count desc, id asc on ties. Format: id<TAB>count\n",
            args.ckpt.display(),
            ranked.len()
        ));
        for &id in &ranked {
            sidecar.push_str(&format!("{id}\t{}\n", counts[id as usize]));
        }
        std::fs::write(out_path, &sidecar)?;
        receipt.push_str(&format!("# ranks_sidecar\t{}\n", out_path.display()));
        let rpath = args.out.join(format!("owngen-{}.tsv", args.label));
        std::fs::write(&rpath, &receipt)?;
        println!("{receipt}\n# owngen receipt: {}", rpath.display());
    }

    // ---------------------------------------------------------------- FR-Spec draft trim
    // Default OFF (full-vocab draft head): a trim is a per-model rank artifact, never an
    // inferred default. Armed only when a ranks file is named. Exactness is untouched by
    // construction (the verify chunk stays full-vocab); acceptance is the measured axis.
    let mut trim_ids: Vec<u32> = Vec::new();
    let mut trim_all: Vec<u32> = Vec::new();
    if let Some(path) = args.draft_trim.as_ref() {
        trim_all = read_ranks(path)?;
        let n = if args.draft_trim_n == 0 {
            trim_all.len()
        } else {
            args.draft_trim_n.min(trim_all.len())
        };
        trim_ids = trim_all[..n].to_vec();
        let t_trim = Instant::now();
        model.build_draft_trim(de, &trim_ids)?;
        println!(
            "# draft-trim\tranks={}\tavailable={}\tarmed_n={n}\tbuild_s={:.1}\tvram\t{}",
            path.display(),
            trim_all.len(),
            t_trim.elapsed().as_secs_f64(),
            nvidia_smi()
        );
    }

    // Deferred-chain embed table (mtp11): armed only when a defer instrument asks
    // (flags law), AFTER the trim block so a trimmed run gets the trim-rank table.
    if args.spec_opts.defer || args.defer_ab.is_some() {
        let t_arm = Instant::now();
        model.arm_spec_devchain(de)?;
        println!(
            "# spec-defer\tdefault_defer={}\tguard_sync={}\tarm_s={:.1}\tvram\t{}",
            args.spec_opts.defer,
            args.spec_opts.defer_guard_sync,
            t_arm.elapsed().as_secs_f64(),
            nvidia_smi()
        );
    }

    // ---------------------------------------------------------------- trim-width sweep
    // The accept table the width is CHOSEN from (DRAFT-REGIME law 3: the verdict metric is
    // end-to-end tok/s; acceptance is the diagnostic for why). One run per N plus the
    // full-vocab control, same prompt, same K — ordering signal, so the interleaved
    // `--trim-ab` at the chosen N remains the claim.
    if !args.trim_sweep.is_empty() {
        let toks = args.spec_ab.map(|(_, t)| t).unwrap_or(256);
        let k = args.spec_k;
        let sweep_prompt: Vec<u32> = match args.prompts.as_ref() {
            Some(path) => read_prompts(path)?
                .first()
                .map(|p| p.ids.clone())
                .ok_or("empty prompts file")?,
            None => return Err("--trim-sweep needs --prompts (real prompts, perf-row law)".into()),
        };
        let mut receipt = header.clone();
        receipt.push_str(&format!(
            "# trim-sweep k={k} tokens={toks} ranks_available={}: one run per trim width plus the \
             full-vocab control (not interleaved — the width-choice table; --trim-ab is the claim)\n\
             draft_head_rows\ttokens\tms_per_token\ttok_per_s\taccept_rate\tmean_accept_len\tdraft_ms_share\thist\tchain_matches_control\n",
            trim_all.len()
        ));
        let mut control_chain: Vec<u32> = Vec::new();
        let mut widths: Vec<Option<usize>> = vec![None];
        widths.extend(args.trim_sweep.iter().map(|&n| Some(n.min(trim_all.len()))));
        let mut all_match = true;
        for width in widths {
            match width {
                None => model.set_draft_trim(false),
                Some(n) => model.build_draft_trim(de, &trim_all[..n])?,
            }
            let rows = model.draft_logits_width();
            let cap = sweep_prompt.len() + toks + k + 4;
            let mut ss = model.alloc_state(&engine, cap)?;
            let mut ds = model.mtp_state(de, cap)?;
            let report = model.spec_generate_ext(
                &engine,
                de,
                &sweep_prompt,
                toks,
                k,
                &mut ss,
                &mut ds,
                None,
                args.spec_opts,
                None,
            )?;
            let decode_ms = report.total_ms - report.prefill_ms;
            let per_tok = decode_ms / report.tokens.len().max(1) as f64;
            let matches = if control_chain.is_empty() {
                control_chain = report.tokens.clone();
                true
            } else {
                control_chain == report.tokens
            };
            all_match &= matches;
            receipt.push_str(&format!(
                "{rows}\t{}\t{per_tok:.2}\t{:.2}\t{:.3}\t{:.2}\t{:.2}\t{}\t{matches}\n",
                report.tokens.len(),
                1e3 / per_tok,
                report.accept_rate(),
                report.mean_accept_len(),
                report.draft_ms / decode_ms,
                report
                    .accept_hist
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            ));
            println!(
                "# trim-sweep rows={rows}: {:.2} tok/s accept={:.3} len={:.2} chain_matches_control={matches}",
                1e3 / per_tok,
                report.accept_rate(),
                report.mean_accept_len()
            );
        }
        receipt.push_str(&format!(
            "# verdict\tevery_width_chain_equals_the_full_vocab_control={all_match}\n# control_chain\t{}\n",
            csv(&control_chain)
        ));
        let path = args.out.join(format!("trim-sweep-k{k}-{}.tsv", args.label));
        std::fs::write(&path, &receipt)?;
        println!("{receipt}\n# trim-sweep receipt: {}", path.display());
        if !all_match {
            eprintln!("trim-sweep: a trimmed width moved the committed chain — the exactness law");
            std::process::exit(1);
        }
        // Restore the width the run was asked to arm.
        if !trim_ids.is_empty() {
            model.build_draft_trim(de, &trim_ids)?;
        }
    }

    // ---------------------------------------------------------------- hidden goldens
    if let Some(dir) = args.goldens.as_ref() {
        let goldens = read_goldens(dir)?;
        let t = goldens.input_ids.len();
        let mut receipt = header.clone();
        receipt.push_str(&format!(
            "# load_seconds={load_s:.1}\n# vram\tpost-load\t{vram_load}\n# probe_tokens={t}\tids={:?}\n",
            goldens.input_ids
        ));
        let profile_warmup = if args.profile > 0 { 4 } else { 0 };
        let mut state = model.alloc_state(
            &engine,
            t + args.decode_timing + profile_warmup + args.profile + 1,
        )?;
        let t_prefill = Instant::now();
        let (logits, capture) = model.prefill_captured(&engine, &goldens.input_ids, &mut state)?;
        let prefill_s = t_prefill.elapsed().as_secs_f64();
        receipt.push_str(&format!(
            "# prefill_seconds={prefill_s:.2}\n# vram\tpost-prefill\t{}\n",
            nvidia_smi()
        ));
        receipt.push_str("record\trows\tcols\tmax_abs\tmax_rel\tmean_abs\tref_absmax\n");
        let compare_record = |name: &str, ours: &[f32], receipt: &mut String| -> Res<()> {
            let golden = goldens
                .records
                .get(name)
                .ok_or_else(|| format!("goldens dump is missing record {name}"))?;
            if golden.data.len() != ours.len() {
                return Err(format!(
                    "{name}: golden {}x{} vs ours {} values",
                    golden.rows,
                    golden.cols,
                    ours.len()
                )
                .into());
            }
            let s = compare(&golden.data, ours);
            receipt.push_str(&format!(
                "{name}\t{}\t{}\t{:.3e}\t{:.3e}\t{:.3e}\t{:.3e}\n",
                golden.rows, golden.cols, s.max_abs, s.max_rel, s.mean_abs, s.ref_absmax
            ));
            Ok(())
        };
        for (index, wide) in capture.layer_wide.iter().enumerate() {
            compare_record(&format!("layer{index}"), wide, &mut receipt)?;
        }
        compare_record("exit_mixer", &capture.exit_mixed, &mut receipt)?;
        compare_record("logits", &logits, &mut receipt)?;
        // Per-row logits: argmax agreement + envelope (the row the greedy chain reads).
        let golden_logits = goldens
            .records
            .get("logits")
            .ok_or("goldens dump is missing logits")?;
        let mut argmax_matches = 0usize;
        receipt
            .push_str("logits_row\tmax_abs\tmax_rel\tref_absmax\tgolden_top1\tour_top1\tmatch\n");
        for row in 0..t {
            let g = &golden_logits.data[row * vocab..(row + 1) * vocab];
            let o = &logits[row * vocab..(row + 1) * vocab];
            let s = compare(g, o);
            let (gt, ot) = (argmax(g), argmax(o));
            argmax_matches += usize::from(gt == ot);
            receipt.push_str(&format!(
                "{row}\t{:.3e}\t{:.3e}\t{:.3e}\t{gt}\t{ot}\t{}\n",
                s.max_abs,
                s.max_rel,
                s.ref_absmax,
                gt == ot
            ));
        }
        receipt.push_str(&format!(
            "# logits_argmax_agreement\t{argmax_matches}/{t}\n"
        ));
        let logits_path = args.out.join(format!("probe-logits-{}.bin", args.label));
        write_f32_bin(&logits_path, &logits)?;
        receipt.push_str(&format!(
            "# probe_logits\t{}\trows={t}\tvocab={vocab}\n",
            logits_path.display()
        ));
        // Draft-forward parity (mtp-spec lane deliverable 2): the engine draft — one
        // BATCHED pass (the replay shape) and a fresh single-step chain (the K-step
        // decode shape) — vs the HOST reference MTP twin, both fed the ENGINE's own
        // trunk wide state (pure draft-program comparison, pos_off = 0 to match the
        // twin's row-indexed positions). Same-bytes f32 comparison => the modelplan
        // policy applies (max_rel <= 0.01 + argmax), unlike the bf16-goldens envelope.
        if args.draft_gate {
            let weights = draft_ref.as_ref().ok_or("draft-gate lost its host twin")?;
            let trunk_wide = capture
                .layer_wide
                .last()
                .ok_or("prefill capture has no trunk wide state")?;
            let wide = trunk_wide.len() / t;
            let t_ref = Instant::now();
            let reference = memra_reference::execute_mtp_standalone(
                &model.plan,
                weights,
                &goldens.input_ids,
                trunk_wide,
            )?;
            let host_s = t_ref.elapsed().as_secs_f64();
            let mtp_ref = reference
                .first()
                .ok_or("host twin produced no MTP output")?;
            let wide_dev = de.htod(trunk_wide)?;
            let mut dreceipt = header.clone();
            dreceipt.push_str(&format!(
                "# draft-gate: engine draft vs host reference MTP twin, {t} probe rows, host_twin_seconds={host_s:.1}\n\
                 phase\trow\tmax_abs\tmax_rel\tref_absmax\tkl\ttop1_ref\ttop1_ours\tmatch\n"
            ));
            let mut worst = (0.0f32, 0.0f32);
            let mut matches = 0usize;
            let mut rows_total = 0usize;
            {
                let mut check_rows = |phase: &str, ours: &[f32]| {
                    for row in 0..t {
                        let r = &mtp_ref.logits[row * vocab..(row + 1) * vocab];
                        let o = &ours[row * vocab..(row + 1) * vocab];
                        let s = compare(r, o);
                        let kl = kl_divergence(r, o);
                        let (rt, ot) = (argmax(r), argmax(o));
                        matches += usize::from(rt == ot);
                        rows_total += 1;
                        worst.0 = worst.0.max(s.max_abs);
                        worst.1 = worst.1.max(s.max_rel);
                        dreceipt.push_str(&format!(
                            "{phase}\t{row}\t{:.3e}\t{:.3e}\t{:.3e}\t{kl:.5}\t{rt}\t{ot}\t{}\n",
                            s.max_abs,
                            s.max_rel,
                            s.ref_absmax,
                            rt == ot
                        ));
                    }
                };
                // Batched pass (the replay shape) + the carrier (the K>1 seed).
                let mut ds = model.mtp_state(de, t + 1)?;
                let (ld, cd) = model.mtp_draft_forward(
                    de,
                    &goldens.input_ids,
                    &wide_dev,
                    0,
                    &mut ds,
                    0,
                    true,
                )?;
                let ours = de.dtoh_view(&ld.slice(0..t * vocab))?;
                check_rows("batched", &ours);
                let carrier = de.dtoh_view(&cd.slice(0..t * wide))?;
                let cs = compare(&mtp_ref.hidden, &carrier);
                model.mtp_recycle(&mut ds, ld, cd);
                // Single-step chain (the K-step decode shape) on a fresh draft state.
                let mut ds2 = model.mtp_state(de, t + 1)?;
                let mut step_rows = vec![0.0f32; t * vocab];
                for row in 0..t {
                    let (ld, cd) = model.mtp_draft_forward(
                        de,
                        &goldens.input_ids[row..row + 1],
                        &wide_dev,
                        row,
                        &mut ds2,
                        0,
                        true,
                    )?;
                    step_rows[row * vocab..(row + 1) * vocab]
                        .copy_from_slice(&de.dtoh_view(&ld.slice(0..vocab))?);
                    model.mtp_recycle(&mut ds2, ld, cd);
                }
                check_rows("step", &step_rows);
                dreceipt.push_str(&format!(
                    "# carrier\tmax_abs={:.3e}\tmax_rel={:.3e}\tref_absmax={:.3e}\n",
                    cs.max_abs, cs.max_rel, cs.ref_absmax
                ));
            }
            let pass = matches == rows_total && worst.1 <= 0.01;
            dreceipt.push_str(&format!(
                "# verdict\trows={rows_total}\targmax_matches={matches}\tworst_abs={:.3e}\tworst_rel={:.3e}\tpolicy=max_rel<=0.01+argmax\tpass={pass}\n",
                worst.0, worst.1
            ));
            let path = args.out.join(format!("draft-gate-{}.tsv", args.label));
            std::fs::write(&path, &dreceipt)?;
            println!("{dreceipt}\n# draft-gate receipt: {}", path.display());
            if !pass {
                eprintln!("draft-gate FAILED (receipt {})", path.display());
                std::process::exit(1);
            }
        }
        // Decode-timing signal: self-fed argmax steps (also exercises incremental decode
        // on the real state). Untuned eager number by header law.
        let mut next = argmax(&logits[(t - 1) * vocab..t * vocab]) as u32;
        let mut unprofiled_mean_ms = 0.0f64;
        if args.decode_timing > 0 {
            let mut continuation = vec![next];
            let mut step_ms = Vec::with_capacity(args.decode_timing);
            for step in 0..args.decode_timing {
                if args.profiler_window && step == 2.min(args.decode_timing - 1) {
                    unsafe { cudarc::driver::sys::cuProfilerStart() };
                    receipt.push_str(&format!("# profiler_window\tstart_step={step}\n"));
                }
                let t_step = Instant::now();
                let row = match tp2.as_ref() {
                    Some((e1, shard)) if args.tp2 => {
                        model.decode_step_tp2(&engine, e1, shard, next, &mut state)?
                    }
                    _ => model.decode_step(&engine, next, &mut state)?,
                };
                step_ms.push(ms(t_step));
                next = argmax(&row) as u32;
                continuation.push(next);
            }
            if args.profiler_window {
                unsafe { cudarc::driver::sys::cuProfilerStop() };
            }
            let mut sorted = step_ms.clone();
            sorted.sort_by(f64::total_cmp);
            // Warm mean excludes the first TWO post-prefill steps: step 0 is the
            // allocator/cuBLASLt warmup (round-1 exclusion), step 1 carries the one-time
            // decode-graph captures (item 2b). Median/p90 stay over all steps.
            let warm = &step_ms[2.min(step_ms.len() - 1)..];
            let mean = warm.iter().sum::<f64>() / warm.len() as f64;
            unprofiled_mean_ms = mean;
            receipt.push_str(&format!(
                "# decode_timing\tsteps={}\tmean_ms={mean:.1}\tmedian_ms={:.1}\tp90_ms={:.1}\ttok_per_s_untuned={:.2}\n# decode_continuation_ids\t{}\n# vram\tpost-decode\t{}\n",
                step_ms.len(),
                sorted[sorted.len() / 2],
                sorted[(sorted.len() * 9) / 10],
                1e3 / mean,
                csv(&continuation),
                nvidia_smi()
            ));
        }
        // Per-section wall profile: sync-bounded sections over warm self-fed decode steps
        // (first token and warmup steps excluded). The sync boundaries distort the step
        // total, so shares are the signal; the unprofiled mean above is the absolute.
        if args.profile > 0 {
            for _ in 0..profile_warmup {
                let row = model.decode_step(&engine, next, &mut state)?;
                next = argmax(&row) as u32;
            }
            memra_engine::qwen4exp_gpu::prof::enable();
            let t_prof = Instant::now();
            for _ in 0..args.profile {
                let row = model.decode_step(&engine, next, &mut state)?;
                next = argmax(&row) as u32;
            }
            let profiled_wall_ms = ms(t_prof);
            let mut rows = memra_engine::qwen4exp_gpu::prof::take();
            rows.sort_by(|a, b| b.1.total_cmp(&a.1));
            let attributed_ms: f64 = rows.iter().map(|r| r.1 * 1e3).sum();
            let steps = args.profile as f64;
            let mut prof_receipt = header.clone();
            prof_receipt.push_str(&format!(
                "# profile\tsteps={}\tt_kv_start={}\tprofiled_wall_ms_per_token={:.2}\tattributed_ms_per_token={:.2}\tunprofiled_mean_ms_per_token={:.2}\n",
                args.profile,
                state.position() - args.profile,
                profiled_wall_ms / steps,
                attributed_ms / steps,
                unprofiled_mean_ms,
            ));
            prof_receipt
                .push_str("section\tcalls_per_token\ttotal_ms\tms_per_token\tpct_of_attributed\n");
            for (name, seconds, calls) in &rows {
                prof_receipt.push_str(&format!(
                    "{name}\t{:.1}\t{:.1}\t{:.3}\t{:.1}\n",
                    *calls as f64 / steps,
                    seconds * 1e3,
                    seconds * 1e3 / steps,
                    seconds * 1e3 / (attributed_ms / 100.0),
                ));
            }
            let path = args.out.join(format!("profile-{}.tsv", args.label));
            std::fs::write(&path, &prof_receipt)?;
            println!("{prof_receipt}\n# profile receipt: {}", path.display());
        }
        // Interleaved A/B of an optimization seam: per rep BOTH arms run back to back,
        // each from a fresh state + probe prefill, 4 warmup steps excluded — the
        // box-clock-drift law (interleaved x5). `--ab-seam` picks which seam moves; the
        // other stays at its shipped default so each row isolates one change.
        if let Some((reps, steps)) = args.ab_moe {
            let (arm_off_name, arm_on_name) = match args.ab_seam.as_str() {
                "moe" => ("per_expert", "sel_grouped"),
                "hc" => ("hc_unfused", "hc_fused"),
                "trunk" => ("trunk_f32", "trunk_bf16"),
                "ws" => ("alloc_per_step", "step_workspace"),
                "graph" => ("eager_ws", "decode_graphs"),
                "selv2" => ("sel_v1", "sel_v2"),
                "hcmicro" => ("hc_item3", "hc_micro"),
                "selv3" => ("sel_v2", "sel_v3"),
                "gdnstep" => ("scan_naive", "scan_step"),
                "gdnfuse" => ("norm_gate_chain", "norm_gate_fused"),
                "projstack" => ("proj_per_mat", "proj_stack"),
                "hcdiet" => ("hc_fused_chain", "hc_diet"),
                "gufuse" => ("gate_up_silu_chain", "gu_silu_fused"),
                "routerb16" => ("router_f32", "router_bf16"),
                "routerdev" => ("router_host", "router_dev"),
                "idxcache" => ("idx_host_cache", "idx_dev_cache"),
                // NOTE: `idxsel` engages only on rows past the 2,048-token selection
                // horizon, so an --ab-moe probe-prefill row is a structural ZERO for it.
                // Its perf instrument is the LADDER at depth; this entry exists for
                // --seam-gate and vocabulary completeness.
                "idxsel" => ("idx_sel_host", "idx_sel_dev"),
                "plecache" => ("ple_ids_rebuild", "ple_ids_cached"),
                "devtwin" => ("devtwin_off", "devtwin_on"),
                "kvq" => ("kv_f32", "kv_q8q5"),
                "idxq" => ("idx_f32", "idx_q8"),
                "tp2" => ("single_card", "tp2"),
                other => {
                    return Err(format!(
                        "--ab-seam {other}: want moe|hc|trunk|ws|graph|selv2|hcmicro|selv3|\
                         gdnstep|gdnfuse|projstack|hcdiet|gufuse|routerb16|routerdev|idxcache|\
                         devtwin|idxsel|plecache|kvq|idxq|tp2"
                    )
                    .into());
                }
            };
            let mut ab = header.clone();
            ab.push_str(&format!(
                "# ab-{}: fresh state + probe prefill per arm, 4 warmup steps excluded per arm\n\
                 rep\tarm\tsteps\tmean_ms\tmedian_ms\ttok_per_s\n",
                args.ab_seam
            ));
            let mut arm_means: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
            let mut rep0_ids: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
            for rep in 0..reps {
                for (arm, arm_on) in [(0usize, false), (1usize, true)] {
                    match args.ab_seam.as_str() {
                        "hc" => memra_engine::qwen4exp_gpu::set_hc_fused_gate(arm_on),
                        "trunk" => memra_engine::qwen4exp_gpu::set_trunk_bf16(arm_on),
                        "ws" => memra_engine::qwen4exp_gpu::set_step_ws(arm_on),
                        "graph" => memra_engine::qwen4exp_gpu::set_decode_graphs(arm_on),
                        "selv2" => memra_engine::qwen4exp_gpu::set_sel_v2(arm_on),
                        "hcmicro" => memra_engine::qwen4exp_gpu::set_hc_micro(arm_on),
                        "selv3" => memra_engine::qwen4exp_gpu::set_sel_v3(arm_on),
                        "gdnstep" => memra_engine::qwen4exp_gpu::set_gdn_step(arm_on),
                        "gdnfuse" => memra_engine::qwen4exp_gpu::set_gdn_fuse(arm_on),
                        "projstack" => memra_engine::qwen4exp_gpu::set_proj_stack(arm_on),
                        "hcdiet" => memra_engine::qwen4exp_gpu::set_hc_diet(arm_on),
                        "gufuse" => memra_engine::qwen4exp_gpu::set_sel_gufuse(arm_on),
                        "routerb16" => memra_engine::qwen4exp_gpu::set_router_bf16(arm_on),
                        "routerdev" => memra_engine::qwen4exp_gpu::set_router_dev(arm_on),
                        "idxcache" => memra_engine::qwen4exp_gpu::set_idx_cache(arm_on),
                        "idxsel" => memra_engine::qwen4exp_gpu::set_idx_sel(arm_on),
                        "plecache" => memra_engine::qwen4exp_gpu::set_ple_cache(arm_on),
                        // The devtwin STACK as one arm (both host twins move together —
                        // the combined-verdict row).
                        "devtwin" => {
                            memra_engine::qwen4exp_gpu::set_router_dev(arm_on);
                            memra_engine::qwen4exp_gpu::set_idx_cache(arm_on);
                        }
                        // Per-STATE latched formats: the setter only matters at the
                        // fresh alloc below, which is exactly per arm per rep.
                        "kvq" => memra_engine::qwen4exp_gpu::set_kv_quant(arm_on),
                        "idxq" => {
                            memra_engine::qwen4exp_gpu::set_idxq(if arm_on { "q8" } else { "f32" })
                        }
                        "tp2" => {} // routed at the decode call, not a global seam
                        _ => memra_engine::qwen4exp_gpu::set_moe_sel_path(arm_on),
                    }
                    let mut st = model.alloc_state(&engine, goldens.input_ids.len() + steps + 6)?;
                    let logits = model.prefill(&engine, &goldens.input_ids, &mut st)?;
                    let mut next = argmax(&logits[(goldens.input_ids.len() - 1) * vocab..]) as u32;
                    // `--ab-seam tp2` toggles the ROUTE; any other seam under `--tp2`
                    // measures BOTH arms on the TP2 route (the deployment config).
                    let tp2_arm =
                        (args.ab_seam == "tp2" && arm_on) || (args.tp2 && args.ab_seam != "tp2");
                    let step_once = |next: u32, st: &mut _| -> Res<Vec<f32>> {
                        match tp2.as_ref() {
                            Some((e1, shard)) if tp2_arm => {
                                model.decode_step_tp2(&engine, e1, shard, next, st)
                            }
                            _ => model.decode_step(&engine, next, st),
                        }
                    };
                    for _ in 0..4 {
                        let row = step_once(next, &mut st)?;
                        next = argmax(&row) as u32;
                    }
                    let mut step_ms = Vec::with_capacity(steps);
                    for _ in 0..steps {
                        let t_step = Instant::now();
                        let row = step_once(next, &mut st)?;
                        step_ms.push(ms(t_step));
                        next = argmax(&row) as u32;
                        if rep == 0 {
                            rep0_ids[arm].push(next);
                        }
                    }
                    let mut sorted = step_ms.clone();
                    sorted.sort_by(f64::total_cmp);
                    let mean = step_ms.iter().sum::<f64>() / step_ms.len().max(1) as f64;
                    arm_means[arm].push(mean);
                    ab.push_str(&format!(
                        "{rep}\t{}\t{steps}\t{mean:.2}\t{:.2}\t{:.2}\n",
                        if arm_on { arm_on_name } else { arm_off_name },
                        sorted[sorted.len() / 2],
                        1e3 / mean,
                    ));
                }
            }
            memra_engine::qwen4exp_gpu::set_moe_sel_path(true);
            memra_engine::qwen4exp_gpu::set_hc_fused_gate(true);
            memra_engine::qwen4exp_gpu::set_trunk_bf16(true);
            memra_engine::qwen4exp_gpu::set_step_ws(true);
            memra_engine::qwen4exp_gpu::set_decode_graphs(true);
            memra_engine::qwen4exp_gpu::set_sel_v2(true);
            memra_engine::qwen4exp_gpu::set_hc_micro(true);
            memra_engine::qwen4exp_gpu::set_sel_v3(memra_engine::qwen4exp_gpu::SEL_V3_DEFAULT);
            memra_engine::qwen4exp_gpu::set_gdn_step(memra_engine::qwen4exp_gpu::GDN_STEP_DEFAULT);
            memra_engine::qwen4exp_gpu::set_gdn_fuse(memra_engine::qwen4exp_gpu::GDN_FUSE_DEFAULT);
            memra_engine::qwen4exp_gpu::set_proj_stack(
                memra_engine::qwen4exp_gpu::PROJ_STACK_DEFAULT,
            );
            memra_engine::qwen4exp_gpu::set_hc_diet(memra_engine::qwen4exp_gpu::HC_DIET_DEFAULT);
            memra_engine::qwen4exp_gpu::set_sel_gufuse(
                memra_engine::qwen4exp_gpu::SEL_GUFUSE_DEFAULT,
            );
            memra_engine::qwen4exp_gpu::set_router_bf16(
                memra_engine::qwen4exp_gpu::ROUTER_B16_DEFAULT,
            );
            memra_engine::qwen4exp_gpu::set_router_dev(
                memra_engine::qwen4exp_gpu::ROUTER_DEV_DEFAULT,
            );
            memra_engine::qwen4exp_gpu::set_idx_cache(
                memra_engine::qwen4exp_gpu::IDX_CACHE_DEFAULT,
            );
            for (arm, name) in [(0usize, arm_off_name), (1usize, arm_on_name)] {
                let v = &arm_means[arm];
                let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
                let min = v.iter().copied().fold(f64::INFINITY, f64::min);
                let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                ab.push_str(&format!(
                    "# arm {name}\tmean_of_means_ms={mean:.2}\tmin={min:.2}\tmax={max:.2}\ttok_per_s={:.2}\n",
                    1e3 / mean
                ));
            }
            // Informational: rep-0 greedy chains per arm (argmax ties may fork — the
            // accumulation class; correctness is the gates' job, not this line's).
            let first_div = rep0_ids[0]
                .iter()
                .zip(&rep0_ids[1])
                .position(|(a, b)| a != b)
                .map(|p| p as i64)
                .unwrap_or(-1);
            ab.push_str(&format!(
                "# rep0_arm_chain_first_divergence\t{first_div}\n# rep0_{arm_off_name}\t{}\n# rep0_{arm_on_name}\t{}\n",
                csv(&rep0_ids[0]),
                csv(&rep0_ids[1]),
            ));
            let path = args
                .out
                .join(format!("ab-{}-{}.tsv", args.ab_seam, args.label));
            std::fs::write(&path, &ab)?;
            println!("{ab}\n# ab-moe receipt: {}", path.display());
        }
        // Seam DECODE-row gate (--seam-gate <n> with --ab-seam <name>): two states,
        // identical prefill, the SAME fed token every step (the OFF arm's argmax
        // chain), per-step logits envelope + KL + argmax between the seam's OFF and ON
        // arms. This is the decode-shape numeric instrument the prefill-shaped goldens
        // battery cannot provide (e.g. W4A4 engages only at t == 1). REPORTING gate for
        // real-error seams: the verdict line prints the envelope; acceptance is the
        // lane's call, not an exit code.
        if args.seam_gate > 0 {
            let set_seam = |on: bool| -> Res<()> {
                match args.ab_seam.as_str() {
                    "moe" => memra_engine::qwen4exp_gpu::set_moe_sel_path(on),
                    "hc" => memra_engine::qwen4exp_gpu::set_hc_fused_gate(on),
                    "trunk" => memra_engine::qwen4exp_gpu::set_trunk_bf16(on),
                    "ws" => memra_engine::qwen4exp_gpu::set_step_ws(on),
                    "graph" => memra_engine::qwen4exp_gpu::set_decode_graphs(on),
                    "selv2" => memra_engine::qwen4exp_gpu::set_sel_v2(on),
                    "hcmicro" => memra_engine::qwen4exp_gpu::set_hc_micro(on),
                    "selv3" => memra_engine::qwen4exp_gpu::set_sel_v3(on),
                    "gdnstep" => memra_engine::qwen4exp_gpu::set_gdn_step(on),
                    "gdnfuse" => memra_engine::qwen4exp_gpu::set_gdn_fuse(on),
                    "projstack" => memra_engine::qwen4exp_gpu::set_proj_stack(on),
                    "hcdiet" => memra_engine::qwen4exp_gpu::set_hc_diet(on),
                    "gufuse" => memra_engine::qwen4exp_gpu::set_sel_gufuse(on),
                    "routerb16" => memra_engine::qwen4exp_gpu::set_router_bf16(on),
                    "routerdev" => memra_engine::qwen4exp_gpu::set_router_dev(on),
                    "idxcache" => memra_engine::qwen4exp_gpu::set_idx_cache(on),
                    "idxsel" => memra_engine::qwen4exp_gpu::set_idx_sel(on),
                    "plecache" => memra_engine::qwen4exp_gpu::set_ple_cache(on),
                    "devtwin" => {
                        memra_engine::qwen4exp_gpu::set_router_dev(on);
                        memra_engine::qwen4exp_gpu::set_idx_cache(on);
                    }
                    // kvq/idxq latch per STATE at alloc: the two states below allocate
                    // under opposite settings, so the per-step re-sets are no-ops for
                    // them (the state carries its format) — exactly the instrument.
                    "kvq" => memra_engine::qwen4exp_gpu::set_kv_quant(on),
                    "idxq" => memra_engine::qwen4exp_gpu::set_idxq(if on { "q8" } else { "f32" }),
                    other => return Err(format!("--seam-gate: seam {other} unsupported").into()),
                }
                Ok(())
            };
            let mut receipt = header.clone();
            receipt.push_str(&format!(
                "# seam-gate ({0}): OFF arm vs ON arm, same fed tokens (OFF argmax chain)\n\
                 step\tfed\tmax_abs\tmax_rel\tref_absmax\tkl_off_on\ttop1_off\ttop1_on\tmatch\n",
                args.ab_seam
            ));
            set_seam(false)?;
            let mut sa = model.alloc_state(&engine, t + args.seam_gate + 1)?;
            let la = model.prefill(&engine, &goldens.input_ids, &mut sa)?;
            set_seam(true)?;
            let mut sb = model.alloc_state(&engine, t + args.seam_gate + 1)?;
            let _ = model.prefill(&engine, &goldens.input_ids, &mut sb)?;
            let mut next = argmax(&la[(t - 1) * vocab..]) as u32;
            let mut worst = (0.0f32, 0.0f32);
            let mut worst_kl = 0.0f64;
            let mut matches = 0usize;
            for step in 0..args.seam_gate {
                set_seam(false)?;
                let ra = model.decode_step(&engine, next, &mut sa)?;
                set_seam(true)?;
                let rb = model.decode_step(&engine, next, &mut sb)?;
                let s = compare(&ra, &rb);
                let kl = kl_divergence(&ra, &rb);
                let (ta, tb) = (argmax(&ra), argmax(&rb));
                matches += usize::from(ta == tb);
                worst.0 = worst.0.max(s.max_abs);
                worst.1 = worst.1.max(s.max_rel);
                worst_kl = worst_kl.max(kl);
                receipt.push_str(&format!(
                    "{step}\t{next}\t{:.3e}\t{:.3e}\t{:.3e}\t{kl:.5}\t{ta}\t{tb}\t{}\n",
                    s.max_abs,
                    s.max_rel,
                    s.ref_absmax,
                    ta == tb
                ));
                next = ta as u32;
            }
            set_seam(false)?;
            receipt.push_str(&format!(
                "# verdict\tsteps={}\targmax_matches={matches}\tworst_abs={:.3e}\tworst_rel={:.3e}\tworst_kl={worst_kl:.5}\n",
                args.seam_gate, worst.0, worst.1
            ));
            let path = args
                .out
                .join(format!("seam-gate-{}-{}.tsv", args.ab_seam, args.label));
            std::fs::write(&path, &receipt)?;
            println!("{receipt}\n# seam-gate receipt: {}", path.display());
        }
        // TP2-vs-single-card per-step logits gate: two states, identical prefill, the
        // SAME fed token every step (the single-card argmax chain), row-by-row envelope.
        // Tolerance class, stated: TP2 splits the mixer out-projections into row halves,
        // the expert combine into per-card slot chains, and the join adds the partials —
        // three reduction reorders vs the single-card chain. Policy: per-row max_rel <=
        // 0.01 (denominator max(1,|ref|)) AND argmax match — the modelplan gate class.
        if args.tp2_gate > 0 {
            let (e1, shard) = tp2.as_ref().ok_or("--tp2-gate requires --tp2")?;
            let mut receipt = header.clone();
            receipt.push_str(
                "# tp2-gate: single-card vs TP2, same fed tokens (single-card argmax chain)\n\
                 step\tfed\tmax_abs\tmax_rel\tref_absmax\ttop1_single\ttop1_tp2\tmatch\n",
            );
            let mut sa = model.alloc_state(&engine, t + args.tp2_gate + 1)?;
            let la = model.prefill(&engine, &goldens.input_ids, &mut sa)?;
            let mut sb = model.alloc_state(&engine, t + args.tp2_gate + 1)?;
            let _ = model.prefill(&engine, &goldens.input_ids, &mut sb)?;
            let mut next = argmax(&la[(t - 1) * vocab..]) as u32;
            let mut worst = (0.0f32, 0.0f32);
            let mut matches = 0usize;
            for step in 0..args.tp2_gate {
                let ra = model.decode_step(&engine, next, &mut sa)?;
                let rb = model.decode_step_tp2(&engine, e1, shard, next, &mut sb)?;
                let s = compare(&ra, &rb);
                let (ta, tb) = (argmax(&ra), argmax(&rb));
                matches += usize::from(ta == tb);
                worst.0 = worst.0.max(s.max_abs);
                worst.1 = worst.1.max(s.max_rel);
                receipt.push_str(&format!(
                    "{step}\t{next}\t{:.3e}\t{:.3e}\t{:.3e}\t{ta}\t{tb}\t{}\n",
                    s.max_abs,
                    s.max_rel,
                    s.ref_absmax,
                    ta == tb
                ));
                next = ta as u32;
            }
            let pass = matches == args.tp2_gate && worst.1 <= 0.01;
            receipt.push_str(&format!(
                "# verdict\tsteps={}\targmax_matches={matches}\tworst_abs={:.3e}\tworst_rel={:.3e}\tpolicy=max_rel<=0.01+argmax\tpass={pass}\n",
                args.tp2_gate, worst.0, worst.1
            ));
            let path = args.out.join(format!("tp2-gate-{}.tsv", args.label));
            std::fs::write(&path, &receipt)?;
            println!("{receipt}\n# tp2-gate receipt: {}", path.display());
            if !pass {
                eprintln!("tp2-gate FAILED (receipt {})", path.display());
                std::process::exit(1);
            }
        }
        // ---- TP2-PREFILL CLASS gate (tp2-prefill lane) -----------------------------
        //
        // TWO REGIMES, because TP2-vs-single-card is two different numeric questions and
        // one bar for both is either too loose to catch a defect or too tight to pass:
        //
        //   PRIME (t >= 2): a documented NEAR-TIE BAND class. Batched GEMM widths select
        //   shape-dependent K-reduction splits (cuBLASLt m-variance), so a half-width
        //   sharded projection legally differs by ulps from the full tensor for the same
        //   logical row — the same class the glm5 TP-2 lane calibrated
        //   (research/glm53-flash-bringup-20260827/tp2-20260831, `glm5_tp_gate.rs`) and
        //   the class `Engine::linear`'s documented m-dependence names. On top of that,
        //   qwen4_exp TP2 splits the MoE by expert half and joins the halves, which is a
        //   reduction REORDER of the single-card combine. Bar: calibrated band + greedy
        //   TAPE identity, with reds required to land orders louder.
        //
        //   DECODE (t == 1): measured, then barred at what the measurement supports. The
        //   glm5 lane got BYTE identity at t=1 because its program was
        //   column-parallel-over-gather; ours is NOT — the existing `--tp2-gate` receipt
        //   is "24/24 argmax, worst rel 3.0e-5", i.e. the expert-half join already puts
        //   qwen4_exp's t=1 rows in the near-tie class too. So this gate does NOT claim
        //   decode byte-identity; it reports `decode_byte_identical` as a measured field
        //   and bars decode on its own (tighter) band. Byte identity at t=1 would be a
        //   FINDING here, not the bar.
        //
        // The previous policy at this flag was a flat `max_rel <= 0.01` for every row.
        // That was never calibrated against anything: it is ~50x looser than the prime
        // band below and ~300x looser than the decode band, so it would have passed a
        // genuinely wrong program whose error happened to be small. Replaced, and the
        // receipt states the policy so an old row is never compared to a new one.
        //
        // CALIBRATION LAW (cert-lines / calibrate-downward): the constants below are set
        // from THIS gate's own measurements on this artifact and card class, at 10x the
        // measured green worst, and the reds must clear RED_FLOOR. Run
        // `--tp2-class-calibrate` to measure without barring, read the receipt, then set
        // the constants from it. Never widen a band to make a red arm's twin pass.
        //
        // MEASURED CALIBRATION (round 2, the lane box, 2x RTX PRO 6000 Blackwell Server
        // Edition 600 W, artifact q48fn-nvfp4, f32 exactness-instrument caches — this gate
        // takes --goldens, so the golden pin selects f32, which is the right arm because a
        // band on the TP2 expert-half split must isolate the split from cache-quant noise;
        // the receipt's `# cache` line states the arm) — the receipt that
        // sets these numbers is `tp2-prefill-gate-cal-green2.tsv`, banked in
        // round2-box-receipts/. Both bands are this gate's OWN measured green worst x10, in
        // this gate's OWN metric, on this artifact and card class. The prior constants were
        // 10x a BORROWED green worst expressed in a DIFFERENT metric, which is why the first
        // calibration run could not read a number off them.
        //
        // Measured green worst, `--tp2-class-calibrate`, 19 rows, argmax 19/19, tape OK:
        //   prime  (10 all-rows + 1 chunked): 3.815e-6 .. 1.383e-5, worst 1.383e-5
        //   decode (8 x t==1):                6.557e-6 .. 1.574e-5, worst 1.574e-5
        if args.tp2_prefill_gate > 0 {
            /// Prime (t >= 2) near-tie band: 10x the measured green worst 1.383e-5.
            /// TIGHTER than the 2e-4 placeholder it replaces (calibrate DOWNWARD, never up).
            const TP2_PRIME_BAND: f32 = 1.4e-4;
            /// Decode (t == 1) band: 10x the measured green worst 1.574e-5.
            ///
            /// FINDING, and the reason this is not "tighter than prime" as the placeholder's
            /// comment asserted: on THIS program decode's green worst (1.574e-5) is slightly
            /// LARGER than prime's (1.383e-5). The placeholder reasoned that a t==1 row has
            /// no batched-GEMM width variance and so must be tighter; measured, the
            /// expert-half join REORDER alone already puts t==1 in the same order, and the
            /// batched width variance the prose expected to dominate does not. The band
            /// follows the measurement, not the prediction. (It is also 10x tighter than the
            /// 3e-4 placeholder, so the correction moves the bar the safe way.)
            const TP2_DECODE_BAND: f32 = 1.6e-4;
            /// A red arm must land at least here, or break the tape, or it is not a red:
            /// it means the band ABSORBS a wrong program instead of distinguishing it.
            /// 1e-3 is ~64x the measured green worst and ~6x either band, and the three RED
            /// arms' receipts show where they actually land.
            const TP2_RED_FLOOR: f32 = 1e-3;

            let (e1, shard) = tp2.as_ref().ok_or("--tp2-prefill-gate requires --tp2")?;
            let chunk = args.ladder_chunk.max(1).min(t.max(1));
            let red = std::env::var("MEMRA_Q4E_TP2_GATE_RED").unwrap_or_else(|_| "none".into());
            let is_red = !(red.is_empty() || red == "none" || red == "0");
            let calibrate = args.tp2_class_calibrate;
            let (peer0, home0, both0, touch0) =
                memra_engine::qwen4exp_gpu::tp2_expert_split_stats();
            let mut receipt = header.clone();
            receipt.push_str(&format!(
                "# tp2-prefill-class-gate: single-card vs TP2, same fed tokens\n\
                 # regimes: PRIME t>={t} (all rows, one full-head forward per route) + \
                 CHUNKED prefill last row (chunk={chunk}) + DECODE t==1 x {}\n\
                 # policy: prime max_rel<={TP2_PRIME_BAND:.1e}, decode max_rel<=\
                 {TP2_DECODE_BAND:.1e}, greedy TAPE identity, peer-engagement non-vacuity\
                 ; red arms must exceed {TP2_RED_FLOOR:.1e} or break the tape\n\
                 # metric: max_rel = max|a-b|/max(|a|,1.0) — the same compare() units as \
                 --tp2-gate's 3.0e-5 and as the borrowed glm5 band; elem_rel (floor 1e-6) \
                 is a DIAGNOSTIC column, never the bar\n\
                 # prime regime runs the single-card side on the GROUPED MoE executor \
                 (set_prefill_grouped_all) so the comparison isolates the TP2 expert-half \
                 split instead of straddling it and the grouped-vs-per-expert difference\n\
                 # red_arm={red}\tcalibrate_only={calibrate}\n\
                 regime\trow\tfed\tmax_abs\tmax_rel\telem_rel\tref_absmax\tbit_equal\t\
                 top1_single\ttop1_tp2\ttape_ok\n",
                args.tp2_prefill_gate
            ));

            // BAND METRIC: `compare()`, i.e. `max |a-b| / max(|a|, 1.0)`.
            //
            // This is the SAME metric as `--tp2-gate`'s receipted 3.0e-5, as every other
            // hidden/greedy/seam receipt in this lane, and as the glm5 TP-2 lane's 4.85e-5
            // green worst that the placeholder bands were borrowed from. That matters more
            // than it looks: the band constants below are numbers, and a number is only a
            // band if the gate computes it in the same units.
            //
            // What was here before floored the denominator at 1e-6 instead of 1.0, and its
            // comment had the sign of its own effect backwards — it claimed a near-zero
            // logit "cannot flatter the comparison", when in fact a near-zero denominator
            // catastrophically PENALIZES. Over a 248,320-wide vocab there are always logits
            // near zero, so the measured "worst rel" was 2.865e4 on a row whose worst
            // ABSOLUTE difference was 3.975e0 and whose top-1 matched. Checking 2.865e4
            // against a band derived from a floor-1.0 measurement is a category error, not
            // a loose bar, and it is the reason this calibration run could not simply read
            // a number off the receipt.
            //
            // The 1e-6-floored quantity is still reported, as `worst_elemrel`, because it
            // does say something real about near-zero-logit behavior. It is a diagnostic
            // column and never the bar.
            let max_rel_of = |a: &[f32], b: &[f32]| -> (f32, f32, f32) {
                let mut mr = 0.0f32;
                let mut ma = 0.0f32;
                let mut er = 0.0f32;
                for (x, y) in a.iter().zip(b.iter()) {
                    let d = (x - y).abs();
                    ma = ma.max(d);
                    mr = mr.max(d / x.abs().max(1.0));
                    er = er.max(d / x.abs().max(1e-6));
                }
                (ma, mr, er)
            };
            let bit_equal = |a: &[f32], b: &[f32]| -> bool {
                a.len() == b.len()
                    && a.iter()
                        .zip(b.iter())
                        .all(|(x, y)| x.to_bits() == y.to_bits())
            };
            let mut prime_worst = 0.0f32;
            let mut decode_worst = 0.0f32;
            let mut worst_elemrel = 0.0f32;
            let mut decode_all_bits = true;
            let mut tape_ok = true;
            let mut rows = 0usize;
            let mut argmax_matches = 0usize;
            let mut record = |regime: &str,
                              label: i64,
                              fed: u32,
                              ra: &[f32],
                              rb: &[f32],
                              receipt: &mut String,
                              rows: &mut usize,
                              argmax_matches: &mut usize,
                              tape_ok: &mut bool,
                              worst_elemrel: &mut f32|
             -> f32 {
                let (ma, mr, er) = max_rel_of(ra, rb);
                let (ta, tb) = (argmax(ra), argmax(rb));
                let bits = bit_equal(ra, rb);
                let refmax = ra.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                *rows += 1;
                *argmax_matches += usize::from(ta == tb);
                *tape_ok &= ta == tb;
                *worst_elemrel = worst_elemrel.max(er);
                receipt.push_str(&format!(
                    "{regime}\t{label}\t{fed}\t{ma:.3e}\t{mr:.3e}\t{er:.3e}\t{refmax:.3e}\t\
                     {bits}\t{ta}\t{tb}\t{}\n",
                    ta == tb
                ));
                mr
            };

            // ---- PRIME regime: ONE full-head forward per route over the whole probe,
            // comparing EVERY row. This is the t>=2 statement; the old gate compared only
            // the chunked prefill's last row, which is a t==1-shaped read of a t>=2
            // program and could not have seen an interior-row defect at all.
            {
                // Both sides on the GROUPED executor: TP2's tp2_moe_rows is grouped by
                // construction, so without this the prime rows measure the executor
                // difference (2e-3..4e-3 here) instead of the TP2 split (1.4e-5) and any
                // band read off them would be ~100x too loose. Scoped to this block and
                // restored immediately — the flag must not leak into the chunked/decode
                // regimes below, which run the serving program on both sides already.
                memra_engine::qwen4exp_gpu::set_prefill_grouped_all(true);
                let mut pa = model.alloc_state(&engine, t + 2)?;
                let la_all = model.prefill(&engine, &goldens.input_ids, &mut pa);
                memra_engine::qwen4exp_gpu::set_prefill_grouped_all(false);
                let la_all = la_all?;
                let mut pb = model.alloc_state_tp2(&engine, e1, shard, t + 2, t)?;
                let lb_all = memra_engine::qwen4exp_gpu::Qwen4ExpGpu::forward_tp2(
                    &model,
                    &engine,
                    e1,
                    shard,
                    &goldens.input_ids,
                    &mut pb,
                    memra_engine::qwen4exp_gpu::HeadMode::All,
                )?;
                if la_all.len() != lb_all.len() {
                    return Err(format!(
                        "tp2 prime: single-card produced {} logits, TP2 {} (head mismatch)",
                        la_all.len(),
                        lb_all.len()
                    )
                    .into());
                }
                for row in 0..t {
                    let mr = record(
                        "prime",
                        row as i64,
                        goldens.input_ids[row],
                        &la_all[row * vocab..(row + 1) * vocab],
                        &lb_all[row * vocab..(row + 1) * vocab],
                        &mut receipt,
                        &mut rows,
                        &mut argmax_matches,
                        &mut tape_ok,
                        &mut worst_elemrel,
                    );
                    prime_worst = prime_worst.max(mr);
                }
            }

            // ---- CHUNKED prefill (the deep-fill program the ladder actually runs) ----
            let capn = t + args.tp2_prefill_gate + 2;
            let mut sa = model.alloc_state_reserve(&engine, capn, chunk, None)?;
            let la = model.prefill_extend(&engine, &goldens.input_ids, &mut sa, chunk)?;
            let mut sb = model.alloc_state_tp2(&engine, e1, shard, capn, chunk)?;
            let lb =
                model.prefill_extend_tp2(&engine, e1, shard, &goldens.input_ids, &mut sb, chunk)?;
            let mr = record(
                "chunked",
                -1,
                0,
                &la[(la.len() - vocab)..],
                &lb[(lb.len() - vocab)..],
                &mut receipt,
                &mut rows,
                &mut argmax_matches,
                &mut tape_ok,
                &mut worst_elemrel,
            );
            prime_worst = prime_worst.max(mr);

            // ---- DECODE regime: t==1 rows fed the single-card argmax chain -----------
            let mut next = argmax(&la[(la.len() - vocab)..]) as u32;
            for step in 0..args.tp2_prefill_gate {
                let ra = model.decode_step(&engine, next, &mut sa)?;
                let rb = model.decode_step_tp2(&engine, e1, shard, next, &mut sb)?;
                decode_all_bits &= bit_equal(&ra, &rb);
                let ta = argmax(&ra);
                let mr = record(
                    "decode",
                    step as i64,
                    next,
                    &ra,
                    &rb,
                    &mut receipt,
                    &mut rows,
                    &mut argmax_matches,
                    &mut tape_ok,
                    &mut worst_elemrel,
                );
                decode_worst = decode_worst.max(mr);
                next = ta as u32;
            }

            // ---- non-vacuity: the peer card must actually have been dispatched work ---
            let (peer1, home1, both1, touch1) =
                memra_engine::qwen4exp_gpu::tp2_expert_split_stats();
            let (peer, home, both, touch) =
                (peer1 - peer0, home1 - home0, both1 - both0, touch1 - touch0);
            let engaged = peer > 0;
            // The per-rank fractions the glm5 lane could only DERIVE, measured here.
            let peer_byte_frac = if peer + home > 0 {
                peer as f64 / (peer + home) as f64
            } else {
                0.0
            };
            let both_frac = if touch > 0 {
                both as f64 / touch as f64
            } else {
                0.0
            };
            receipt.push_str(&format!(
                "# expert-split\tpeer_slots={peer}\thome_slots={home}\t\
                 peer_slot_fraction={peer_byte_frac:.4}\tlayer_tokens={touch}\t\
                 both_card_rows={both}\tboth_card_fraction={both_frac:.4}\tengaged={engaged}\n"
            ));

            let green = prime_worst <= TP2_PRIME_BAND
                && decode_worst <= TP2_DECODE_BAND
                && tape_ok
                && engaged;
            let loud = prime_worst > TP2_RED_FLOOR || decode_worst > TP2_RED_FLOOR || !tape_ok;
            let pass = if calibrate {
                true
            } else if is_red {
                // A red PASSES by being LOUD. It must also have engaged the peer, or the
                // arm proved nothing (a red that never routed a peer expert is a no-op).
                loud && engaged
            } else {
                green
            };
            receipt.push_str(&format!(
                "# verdict\trows={rows}\targmax_matches={argmax_matches}\t\
                 prime_worst_rel={prime_worst:.3e}\tprime_band={TP2_PRIME_BAND:.1e}\t\
                 decode_worst_rel={decode_worst:.3e}\tdecode_band={TP2_DECODE_BAND:.1e}\t\
                 decode_byte_identical={decode_all_bits}\tworst_elemrel={worst_elemrel:.3e}\t\
                 tape_ok={tape_ok}\t\
                 peer_engaged={engaged}\tred_arm={red}\tred_floor={TP2_RED_FLOOR:.1e}\t\
                 loud={loud}\tcalibrate_only={calibrate}\tpass={pass}\n"
            ));
            let path = args
                .out
                .join(format!("tp2-prefill-gate-{}.tsv", args.label));
            std::fs::write(&path, &receipt)?;
            println!("{receipt}\n# tp2-prefill-gate receipt: {}", path.display());
            if !pass {
                eprintln!("tp2-prefill-gate FAILED (receipt {})", path.display());
                std::process::exit(1);
            }
        }
        // ---- verify-row BIT gate (mtp-spec): plain t==1 decode rows vs verify chunks
        // fed the SAME tokens — every logit bit-identical, the row-level statement of
        // the spec byte-identity law.
        if args.verify_bit > 0 {
            let k1 = args.spec_k + 1;
            let mut receipt = header.clone();
            receipt.push_str(&format!(
                "# verify-bit-gate: plain t==1 rows vs t=={k1} chunk rows, same fed tokens, n={}\n\
                 row\tabs_pos\tchunk\tbit_identical\n",
                args.verify_bit
            ));
            let mut sa = model.alloc_state(&engine, t + args.verify_bit + 2)?;
            let la = model.prefill(&engine, &goldens.input_ids, &mut sa)?;
            let mut fed: Vec<u32> = vec![argmax(&la[(t - 1) * vocab..]) as u32];
            let mut plain_rows: Vec<Vec<f32>> = Vec::with_capacity(args.verify_bit);
            for step in 0..args.verify_bit {
                let row = model.decode_step(&engine, fed[step], &mut sa)?;
                fed.push(argmax(&row) as u32);
                plain_rows.push(row);
            }
            let mut sb = model.alloc_state(&engine, t + args.verify_bit + k1 + 2)?;
            model.spec_arm(&engine, &mut sb, k1)?;
            let _ = model.prefill(&engine, &goldens.input_ids, &mut sb)?;
            let mut mismatched_rows = 0usize;
            let mut rows_done = 0usize;
            let mut chunk_id = 0usize;
            while rows_done < args.verify_bit {
                let tlen = k1.min(args.verify_bit - rows_done);
                let chunk = &fed[rows_done..rows_done + tlen];
                let rows: Vec<f32> = if tlen == 1 {
                    model.decode_step(&engine, chunk[0], &mut sb)?
                } else {
                    model.prefill(&engine, chunk, &mut sb)?
                };
                for r in 0..tlen {
                    let same = rows[r * vocab..(r + 1) * vocab]
                        .iter()
                        .zip(&plain_rows[rows_done + r])
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                    if !same {
                        mismatched_rows += 1;
                    }
                    receipt.push_str(&format!(
                        "{}\t{}\t{chunk_id}\t{same}\n",
                        rows_done + r,
                        t + rows_done + r
                    ));
                }
                rows_done += tlen;
                chunk_id += 1;
            }
            let pass = mismatched_rows == 0;
            receipt.push_str(&format!(
                "# verdict\trows={}\tmismatched={mismatched_rows}\tpolicy=bit-identity\tpass={pass}\n",
                args.verify_bit
            ));
            let path = args.out.join(format!("verify-bit-gate-{}.tsv", args.label));
            std::fs::write(&path, &receipt)?;
            println!("{receipt}\n# verify-bit receipt: {}", path.display());
            if !pass {
                eprintln!("verify-bit-gate FAILED (receipt {})", path.display());
                std::process::exit(1);
            }
        }
        // ---- REWIND-row BIT gate, REPLAY mode (mtp11): re-walk an exact recorded
        // round pattern — lines of "t,keep" (t = drafts+1, keep = a+1 from a spec
        // trace) — over the FIRST --prompts prompt's plain chain, bit-comparing every
        // committed row. Reproduces a failing spec run's trunk walk deterministically
        // and reports the FIRST corrupt row, which names the round that broke state.
        if let Some(replay_path) = args.rewind_replay.as_ref() {
            let pattern: Vec<(usize, usize)> = std::fs::read_to_string(replay_path)?
                .lines()
                .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
                .map(|l| {
                    let (a, b) = l.trim().split_once(',').ok_or("replay line wants t,keep")?;
                    Ok((a.parse::<usize>()?, b.parse::<usize>()?))
                })
                .collect::<Res<_>>()?;
            let ids: Vec<u32> = match args.prompts.as_ref() {
                Some(path) => read_prompts(path)?
                    .first()
                    .map(|p| p.ids.clone())
                    .ok_or("empty prompts file")?,
                None => goldens.input_ids.clone(),
            };
            let n: usize = pattern.iter().map(|&(_, keep)| keep).sum();
            let tmax = pattern.iter().map(|&(t, _)| t).max().unwrap_or(1);
            let pt = ids.len();
            let mut receipt = header.clone();
            receipt.push_str(&format!(
                "# rewind-bit-replay: pattern={} rounds, {n} rows, prompt_len={pt}\n\
                 round\tt\tkeep\trow\tbit_identical\n",
                pattern.len()
            ));
            let mut sa = model.alloc_state(&engine, pt + n + 2)?;
            let la = model.prefill(&engine, &ids, &mut sa)?;
            let mut fed: Vec<u32> = vec![argmax(&la[(pt - 1) * vocab..]) as u32];
            let mut plain_rows: Vec<Vec<f32>> = Vec::with_capacity(n);
            for step in 0..n {
                let row = model.decode_step(&engine, fed[step], &mut sa)?;
                fed.push(argmax(&row) as u32);
                plain_rows.push(row);
            }
            let mut sc = model.alloc_state(&engine, pt + n + tmax + 2)?;
            model.spec_arm(&engine, &mut sc, tmax.max(args.spec_k + 1))?;
            let _ = model.prefill(&engine, &ids, &mut sc)?;
            let mut rows_done = 0usize;
            let mut mismatched = 0usize;
            let mut first_bad: i64 = -1;
            let mut first_bad_round: i64 = -1;
            for (ri, &(t_r, keep_r)) in pattern.iter().enumerate() {
                if rows_done >= n {
                    break;
                }
                let tlen = t_r.min(n - rows_done);
                let keep_r = keep_r.min(tlen);
                let chunk = &fed[rows_done..rows_done + tlen];
                let rows: Vec<f32> = if tlen == 1 {
                    model.decode_step(&engine, chunk[0], &mut sc)?
                } else {
                    model.prefill(&engine, chunk, &mut sc)?
                };
                for r in 0..keep_r {
                    let same = rows[r * vocab..(r + 1) * vocab]
                        .iter()
                        .zip(&plain_rows[rows_done + r])
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                    if !same {
                        mismatched += 1;
                        if first_bad < 0 {
                            first_bad = (rows_done + r) as i64;
                            first_bad_round = ri as i64;
                        }
                    }
                    receipt.push_str(&format!(
                        "{ri}\t{tlen}\t{keep_r}\t{}\t{same}\n",
                        rows_done + r
                    ));
                }
                if tlen > 1 && keep_r < tlen {
                    model.verify_rewind(&engine, &mut sc, keep_r)?;
                }
                rows_done += keep_r;
            }
            let pass = mismatched == 0;
            receipt.push_str(&format!(
                "# verdict\trows={rows_done}\tmismatched={mismatched}\tfirst_bad_row={first_bad}\tfirst_bad_round={first_bad_round}\tpass={pass}\n"
            ));
            let path = args
                .out
                .join(format!("rewind-bit-replay-{}.tsv", args.label));
            std::fs::write(&path, &receipt)?;
            println!(
                "# rewind-bit-replay: rows={rows_done} mismatched={mismatched} first_bad_row={first_bad} first_bad_round={first_bad_round}\n# receipt: {}",
                path.display()
            );
            if !pass {
                eprintln!("rewind-bit-replay FAILED (receipt {})", path.display());
            }
        }
        // ---- REWIND-row BIT gate (mtp11): the verify-bit statement EXTENDED across
        // the partial-accept rewind. The spec loop's trunk walk is chunk -> rewind ->
        // re-feed; verify-bit never rewinds, which is exactly how a rewind-path
        // corruption stayed latent through every 64-token green gate (found by the
        // mtp11 256-token battery: K=5 diverges at gen 157, K=2 clean). One sub-gate
        // per keep in 1..=k: chunk t=k+1 plain-chain rows, verify_rewind(keep),
        // re-feed the dropped tokens at the SAME positions; every COMMITTED row must
        // be bit-identical to the plain t==1 chain. First corrupt (keep, position)
        // lands in the receipt.
        if args.rewind_bit > 0 {
            let k1 = args.spec_k + 1;
            let n = args.rewind_bit;
            let mut receipt = header.clone();
            receipt.push_str(&format!(
                "# rewind-bit-gate: plain t==1 rows vs chunk(t={k1})+rewind(keep)+refeed rows, n={n}\n\
                 keep\trow\tabs_pos\tchunk\tbit_identical\n"
            ));
            let mut sa = model.alloc_state(&engine, t + n + 2)?;
            let la = model.prefill(&engine, &goldens.input_ids, &mut sa)?;
            let mut fed: Vec<u32> = vec![argmax(&la[(t - 1) * vocab..]) as u32];
            let mut plain_rows: Vec<Vec<f32>> = Vec::with_capacity(n);
            for step in 0..n {
                let row = model.decode_step(&engine, fed[step], &mut sa)?;
                fed.push(argmax(&row) as u32);
                plain_rows.push(row);
            }
            let mut pass = true;
            for keep in 1..k1 {
                let mut sc = model.alloc_state(&engine, t + n + k1 + 2)?;
                model.spec_arm(&engine, &mut sc, k1)?;
                let _ = model.prefill(&engine, &goldens.input_ids, &mut sc)?;
                let mut rows_done = 0usize;
                let mut chunk_id = 0usize;
                let mut mismatched = 0usize;
                let mut first_bad: i64 = -1;
                while rows_done < n {
                    let tlen = k1.min(n - rows_done);
                    let chunk = &fed[rows_done..rows_done + tlen];
                    let rows: Vec<f32> = if tlen == 1 {
                        model.decode_step(&engine, chunk[0], &mut sc)?
                    } else {
                        model.prefill(&engine, chunk, &mut sc)?
                    };
                    let kept = keep.min(tlen);
                    for r in 0..kept {
                        let same = rows[r * vocab..(r + 1) * vocab]
                            .iter()
                            .zip(&plain_rows[rows_done + r])
                            .all(|(a, b)| a.to_bits() == b.to_bits());
                        if !same {
                            mismatched += 1;
                            if first_bad < 0 {
                                first_bad = (rows_done + r) as i64;
                            }
                        }
                        receipt.push_str(&format!(
                            "{keep}\t{}\t{}\t{chunk_id}\t{same}\n",
                            rows_done + r,
                            t + rows_done + r
                        ));
                    }
                    if tlen > 1 && kept < tlen {
                        model.verify_rewind(&engine, &mut sc, kept)?;
                    }
                    rows_done += kept;
                    chunk_id += 1;
                }
                pass &= mismatched == 0;
                receipt.push_str(&format!(
                    "# keep={keep}\trows={rows_done}\tmismatched={mismatched}\tfirst_bad_row={first_bad}\n"
                ));
                println!(
                    "# rewind-bit keep={keep}: {rows_done} rows, mismatched={mismatched}, first_bad_row={first_bad}"
                );
            }
            receipt.push_str(&format!(
                "# verdict\tpolicy=bit-identity-across-rewind\tpass={pass}\n"
            ));
            let path = args.out.join(format!("rewind-bit-gate-{}.tsv", args.label));
            std::fs::write(&path, &receipt)?;
            println!("# rewind-bit receipt: {}", path.display());
            if !pass {
                eprintln!("rewind-bit-gate FAILED (receipt {})", path.display());
                std::process::exit(1);
            }
        }

        // Spec measurement prompt: a REAL prompt when --prompts is given (the goldens
        // probe's greedy chain degenerates into repetition — the greedy-loop law bans
        // it from perf rows; measured in the mtp2 battery, chains banked there).
        let spec_prompt: Vec<u32> = match args.prompts.as_ref() {
            Some(path) => read_prompts(path)?
                .first()
                .map(|p| p.ids.clone())
                .ok_or("empty prompts file")?,
            None => goldens.input_ids.clone(),
        };

        // ---- interleaved plain-vs-spec A/B (the box-clock-drift law) + rep-0
        // byte-identity check.
        if let Some((reps, toks)) = args.spec_ab {
            let k = args.spec_k;
            let mut ab = header.clone();
            ab.push_str(&format!(
                "# spec-ab k={k}: plain greedy decode vs spec_generate, fresh states per arm per rep\n\
                 rep\tarm\ttokens\tms_per_token\ttok_per_s\taccept_rate\tmean_accept_len\n"
            ));
            let mut arm_means: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
            let mut rep0: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
            let mut hist_total: Vec<u64> = vec![0; k + 1];
            for rep in 0..reps {
                // plain arm
                let mut ps = model.alloc_state(&engine, spec_prompt.len() + toks + 4)?;
                let pl = model.prefill(&engine, &spec_prompt, &mut ps)?;
                let mut next = argmax(&pl[(spec_prompt.len() - 1) * vocab..]) as u32;
                let mut chain = vec![next];
                let mut step_ms: Vec<f64> = Vec::with_capacity(toks);
                for _ in 1..toks {
                    let t_step = Instant::now();
                    let row = model.decode_step(&engine, next, &mut ps)?;
                    step_ms.push(ms(t_step));
                    next = argmax(&row) as u32;
                    chain.push(next);
                }
                let warm = &step_ms[2.min(step_ms.len() - 1)..];
                let mean = warm.iter().sum::<f64>() / warm.len() as f64;
                arm_means[0].push(mean);
                ab.push_str(&format!(
                    "{rep}\tplain\t{toks}\t{mean:.2}\t{:.2}\t-\t-\n",
                    1e3 / mean
                ));
                if rep == 0 {
                    rep0[0] = chain;
                }
                // spec arm. `--profiler-window` here brackets rep 0's WHOLE spec run
                // (prefill excluded is not possible at this seam — nsys post-filters by
                // kernel name; the plain twin comes from `--decode-timing
                // --profiler-window` in a separate capture).
                let mut ss = model.alloc_state(&engine, spec_prompt.len() + toks + k + 4)?;
                let mut ds = model.mtp_state(de, spec_prompt.len() + toks + k + 4)?;
                if args.profiler_window && rep == 0 {
                    unsafe { cudarc::driver::sys::cuProfilerStart() };
                }
                let report = model.spec_generate_ext(
                    &engine,
                    de,
                    &spec_prompt,
                    toks,
                    k,
                    &mut ss,
                    &mut ds,
                    None,
                    args.spec_opts,
                    None,
                )?;
                if args.profiler_window && rep == 0 {
                    unsafe { cudarc::driver::sys::cuProfilerStop() };
                }
                let decode_ms = report.total_ms - report.prefill_ms;
                let per_tok = decode_ms / report.tokens.len().max(1) as f64;
                arm_means[1].push(per_tok);
                for (a, &c) in hist_total.iter_mut().zip(&report.accept_hist) {
                    *a += c;
                }
                ab.push_str(&format!(
                    "{rep}\tspec\t{}\t{per_tok:.2}\t{:.2}\t{:.3}\t{:.2}\n",
                    report.tokens.len(),
                    1e3 / per_tok,
                    report.accept_rate(),
                    report.mean_accept_len()
                ));
                // The round-cost identity table (owner item 1): where a spec round's
                // milliseconds sit vs the plain step the same rep just measured.
                let r = report.rounds.max(1) as f64;
                ab.push_str(&format!(
                    "# round-cost\trep={rep}\trounds={}\tchain_ms={:.2}\treplay_ms={:.2}\tverify_ms={:.2}\tcross_ms={:.3}\tdraft_prefill_ms={:.1}\tzero_draft={}\tguard_stops={}\tplain_steps={}\tk_decays={:?}\n",
                    report.rounds,
                    report.chain_ms / r,
                    report.replay_ms / r,
                    report.verify_ms / r,
                    report.cross_ms / r,
                    report.draft_prefill_ms,
                    report.zero_draft_rounds,
                    report.guard_stops,
                    report.plain_steps,
                    report.k_decays,
                ));
                if rep == 0 {
                    rep0[1] = report.tokens.clone();
                }
            }
            for (arm, name) in [(0usize, "plain"), (1usize, "spec")] {
                let v = &arm_means[arm];
                let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
                let min = v.iter().copied().fold(f64::INFINITY, f64::min);
                let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                ab.push_str(&format!(
                    "# arm {name}\tmean_of_means_ms={mean:.2}\tmin={min:.2}\tmax={max:.2}\ttok_per_s={:.2}\n",
                    1e3 / mean
                ));
            }
            let first_div = rep0[0]
                .iter()
                .zip(&rep0[1])
                .position(|(a, b)| a != b)
                .map(|p| p as i64)
                .unwrap_or(-1);
            ab.push_str(&format!(
                "# rep0_plain_vs_spec_first_divergence\t{first_div}\n# accept_hist_total\t{}\n# rep0_plain\t{}\n# rep0_spec\t{}\n",
                hist_total
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                csv(&rep0[0]),
                csv(&rep0[1]),
            ));
            let path = args.out.join(format!("ab-spec-k{k}-{}.tsv", args.label));
            std::fs::write(&path, &ab)?;
            println!("{ab}\n# spec-ab receipt: {}", path.display());
            if first_div >= 0 {
                eprintln!(
                    "spec-ab: rep0 spec chain DIVERGED from plain at {first_div} — the exactness law"
                );
                std::process::exit(1);
            }
        }

        // ---- K ladder (D3/D5): one spec run per K, chains banked.
        if !args.spec_ladder.is_empty() {
            let toks = args.spec_ab.map(|(_, t)| t).unwrap_or(128);
            let mut receipt = header.clone();
            receipt.push_str(
                "# spec-ladder: one run per K (not interleaved — ordering signal only; the A/B is the claim)\n\
                 k\ttokens\tms_per_token\ttok_per_s\taccept_rate\tmean_accept_len\tdraft_ms_share\tverify_ms_share\thist\tchain_prefix\n",
            );
            for &k in &args.spec_ladder {
                let mut ss = model.alloc_state(&engine, spec_prompt.len() + toks + k + 4)?;
                let mut ds = model.mtp_state(de, spec_prompt.len() + toks + k + 4)?;
                let report = model.spec_generate_ext(
                    &engine,
                    de,
                    &spec_prompt,
                    toks,
                    k,
                    &mut ss,
                    &mut ds,
                    None,
                    args.spec_opts,
                    None,
                )?;
                let decode_ms = report.total_ms - report.prefill_ms;
                let per_tok = decode_ms / report.tokens.len().max(1) as f64;
                receipt.push_str(&format!(
                    "{k}\t{}\t{per_tok:.2}\t{:.2}\t{:.3}\t{:.2}\t{:.2}\t{:.2}\t{}\t{}\n",
                    report.tokens.len(),
                    1e3 / per_tok,
                    report.accept_rate(),
                    report.mean_accept_len(),
                    report.draft_ms / decode_ms,
                    report.verify_ms / decode_ms,
                    report
                        .accept_hist
                        .iter()
                        .map(|v| v.to_string())
                        .collect::<Vec<_>>()
                        .join(","),
                    csv(&report.tokens[..report.tokens.len().min(32)]),
                ));
                let r = report.rounds.max(1) as f64;
                receipt.push_str(&format!(
                    "# round-cost\tk={k}\trounds={}\tchain_ms={:.2}\treplay_ms={:.2}\tverify_ms={:.2}\tcross_ms={:.3}\tzero_draft={}\tguard_stops={}\n",
                    report.rounds,
                    report.chain_ms / r,
                    report.replay_ms / r,
                    report.verify_ms / r,
                    report.cross_ms / r,
                    report.zero_draft_rounds,
                    report.guard_stops,
                ));
            }
            let path = args.out.join(format!("spec-ladder-{}.tsv", args.label));
            std::fs::write(&path, &receipt)?;
            println!("{receipt}\n# spec-ladder receipt: {}", path.display());
        }

        // ---- spec section profile: prof_section timers over one spec run (draft +
        // verify sections together; sync-bounded — SHARES are the signal).
        if args.spec_profile > 0 {
            let k = args.spec_profile;
            let toks = 64usize;
            let mut ss = model.alloc_state(&engine, spec_prompt.len() + toks + k + 4)?;
            let mut ds = model.mtp_state(de, spec_prompt.len() + toks + k + 4)?;
            memra_engine::qwen4exp_gpu::prof::enable();
            let report = model.spec_generate_ext(
                &engine,
                de,
                &spec_prompt,
                toks,
                k,
                &mut ss,
                &mut ds,
                None,
                args.spec_opts,
                None,
            )?;
            let mut rows = memra_engine::qwen4exp_gpu::prof::take();
            rows.sort_by(|a, b| b.1.total_cmp(&a.1));
            let attributed_ms: f64 = rows.iter().map(|r| r.1 * 1e3).sum();
            let rounds = report.rounds.max(1) as f64;
            let mut receipt = header.clone();
            receipt.push_str(&format!(
                "# spec-profile k={k} tokens={} rounds={} total_ms={:.1} draft_ms={:.1} verify_ms={:.1} attributed_ms={:.1}\n\
                 section\tcalls_per_round\ttotal_ms\tms_per_round\tpct_of_attributed\n",
                report.tokens.len(),
                report.rounds,
                report.total_ms,
                report.draft_ms,
                report.verify_ms,
                attributed_ms,
            ));
            for (name, seconds, calls) in &rows {
                receipt.push_str(&format!(
                    "{name}\t{:.1}\t{:.1}\t{:.3}\t{:.1}\n",
                    *calls as f64 / rounds,
                    seconds * 1e3,
                    seconds * 1e3 / rounds,
                    seconds * 1e3 / (attributed_ms / 100.0),
                ));
            }
            let path = args
                .out
                .join(format!("spec-profile-k{k}-{}.tsv", args.label));
            std::fs::write(&path, &receipt)?;
            println!("{receipt}\n# spec-profile receipt: {}", path.display());
        }

        // ---- vendor-default SAMPLED spec run (the serving law's probe shape) with the
        // spec-engagement receipt.
        if args.spec_sampled {
            let toks = args.spec_ab.map(|(_, t)| t).unwrap_or(128);
            let k = args.spec_k;
            let cfg = memra_engine::qwen4exp_gpu::SpecSamplerCfg {
                temperature: 1.0,
                top_p: 0.95,
                top_k: 20,
                seed: 0x5eed_cafe,
            };
            let mut ss = model.alloc_state(&engine, spec_prompt.len() + toks + k + 4)?;
            let mut ds = model.mtp_state(de, spec_prompt.len() + toks + k + 4)?;
            let report = model.spec_generate_ext(
                &engine,
                de,
                &spec_prompt,
                toks,
                k,
                &mut ss,
                &mut ds,
                Some(cfg),
                args.spec_opts,
                None,
            )?;
            let engaged_rounds: u64 = report.accept_hist.iter().skip(1).sum();
            let decode_ms = report.total_ms - report.prefill_ms;
            let per_tok = decode_ms / report.tokens.len().max(1) as f64;
            let mut receipt = header.clone();
            receipt.push_str(&format!(
                "# spec-sampled: vendor defaults temp=1.0 top_p=0.95 top_k=20 seed=0x5eedcafe k={k}\n\
                 # SPEC-ENGAGEMENT\trounds={}\trounds_with_accepts={engaged_rounds}\taccepted={}\tdrafted={}\taccept_hist={}\n\
                 # timing\ttokens={}\tms_per_token={per_tok:.2}\ttok_per_s={:.2}\n\
                 # ids\t{}\n",
                report.rounds,
                report.accepted,
                report.drafted,
                report
                    .accept_hist
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                report.tokens.len(),
                1e3 / per_tok,
                csv(&report.tokens),
            ));
            let path = args
                .out
                .join(format!("spec-sampled-k{k}-{}.tsv", args.label));
            std::fs::write(&path, &receipt)?;
            println!("{receipt}\n# spec-sampled receipt: {}", path.display());
            if engaged_rounds == 0 {
                eprintln!(
                    "spec-sampled: ZERO rounds with accepts — spec not engaged under the sampled shape"
                );
                std::process::exit(1);
            }
        }

        let path = args.out.join(format!("hidden-gate-{}.tsv", args.label));
        std::fs::write(&path, &receipt)?;
        println!("{receipt}\n# hidden receipt: {}", path.display());
    }

    // ---------------------------------------------------------------- trim A/B
    // Interleaved (box-clock-drift law) full-vocab-head vs trimmed-head spec at spec_k on
    // the REAL prompt. The trim cannot move the OUTPUT (the verify chunk is full-vocab),
    // so the arms' chains must be token-for-token identical — asserted, hard fail. What it
    // CAN move is acceptance, which is why both arms carry accept rate + mean length.
    if let Some((reps, toks)) = args.trim_ab {
        let ab_prompt: Vec<u32> = match args.prompts.as_ref() {
            Some(path) => read_prompts(path)?
                .first()
                .map(|p| p.ids.clone())
                .ok_or("empty prompts file")?,
            None => return Err("--trim-ab needs --prompts (real prompts, perf-row law)".into()),
        };
        let k = args.spec_k;
        let n = trim_ids.len();
        let mut ab = header.clone();
        ab.push_str(&format!(
            "# trim-ab k={k} trim_n={n} vocab={vocab}: full-vocab draft head vs FR-Spec trimmed head, \
             interleaved, fresh states per arm per rep\n\
             rep\tarm\tdraft_head_rows\ttokens\tms_per_token\ttok_per_s\taccept_rate\tmean_accept_len\tdraft_ms_share\thist\n"
        ));
        let mut arm_means: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        let mut arm_accept: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        let mut arm_len: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        let mut rep0: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
        for rep in 0..reps {
            for (slot, on) in [(0usize, false), (1usize, true)] {
                model.set_draft_trim(on);
                let width = model.draft_logits_width();
                let cap = ab_prompt.len() + toks + k + 4;
                let mut ss = model.alloc_state(&engine, cap)?;
                let mut ds = model.mtp_state(de, cap)?;
                let report = model.spec_generate_ext(
                    &engine,
                    de,
                    &ab_prompt,
                    toks,
                    k,
                    &mut ss,
                    &mut ds,
                    None,
                    args.spec_opts,
                    None,
                )?;
                let decode_ms = report.total_ms - report.prefill_ms;
                let per_tok = decode_ms / report.tokens.len().max(1) as f64;
                arm_means[slot].push(per_tok);
                arm_accept[slot].push(report.accept_rate());
                arm_len[slot].push(report.mean_accept_len());
                ab.push_str(&format!(
                    "{rep}\t{}\t{width}\t{}\t{per_tok:.2}\t{:.2}\t{:.3}\t{:.2}\t{:.2}\t{}\n",
                    if on { "trim" } else { "full" },
                    report.tokens.len(),
                    1e3 / per_tok,
                    report.accept_rate(),
                    report.mean_accept_len(),
                    report.draft_ms / decode_ms,
                    report
                        .accept_hist
                        .iter()
                        .map(|v| v.to_string())
                        .collect::<Vec<_>>()
                        .join(","),
                ));
                if rep == 0 {
                    rep0[slot] = report.tokens.clone();
                }
            }
        }
        for (slot, name) in [(0usize, "full"), (1usize, "trim")] {
            let v = &arm_means[slot];
            let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
            let acc = arm_accept[slot].iter().sum::<f64>() / arm_accept[slot].len().max(1) as f64;
            let len = arm_len[slot].iter().sum::<f64>() / arm_len[slot].len().max(1) as f64;
            ab.push_str(&format!(
                "# arm {name}\tmean_of_means_ms={mean:.2}\tmin={:.2}\tmax={:.2}\ttok_per_s={:.2}\tmean_accept_rate={acc:.3}\tmean_accept_len={len:.2}\n",
                v.iter().copied().fold(f64::INFINITY, f64::min),
                v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                1e3 / mean,
            ));
        }
        let full_mean = arm_means[0].iter().sum::<f64>() / arm_means[0].len().max(1) as f64;
        let trim_mean = arm_means[1].iter().sum::<f64>() / arm_means[1].len().max(1) as f64;
        let first_div = rep0[0]
            .iter()
            .zip(&rep0[1])
            .position(|(a, b)| a != b)
            .map(|p| p as i64)
            .unwrap_or(if rep0[0].len() == rep0[1].len() {
                -1
            } else {
                rep0[0].len().min(rep0[1].len()) as i64
            });
        ab.push_str(&format!(
            "# speedup_trim_over_full\t{:.4}\n# rep0_full_vs_trim_first_divergence\t{first_div}\n# rep0_full\t{}\n# rep0_trim\t{}\n",
            full_mean / trim_mean,
            csv(&rep0[0]),
            csv(&rep0[1]),
        ));
        let path = args
            .out
            .join(format!("ab-trim-k{k}-n{n}-{}.tsv", args.label));
        std::fs::write(&path, &ab)?;
        println!("{ab}\n# trim-ab receipt: {}", path.display());
        if first_div >= 0 {
            eprintln!(
                "trim-ab: the trimmed arm's chain DIVERGED from the full-head arm at {first_div} — \
                 the trim must not move the committed output"
            );
            std::process::exit(1);
        }
        // Leave the trim armed for whatever instruments follow.
        model.set_draft_trim(true);
    }

    // ------------------------------------------------- verify scan-chain graph A/B
    // The verify chunk's per-GDN-layer scan run is a serially DEPENDENT launch chain, the
    // one case the trunk's decode-graph receipt (+1.3% for 2,400 launches removed, mostly
    // overlapped) does not cover. Interleaved, at whatever draft config is live, on the
    // real prompt. A replay is bit-identical to the eager chain by construction, so the
    // arms' chains must match — hard fail otherwise.
    if let Some((reps, toks)) = args.vgraph_ab {
        let ab_prompt: Vec<u32> = match args.prompts.as_ref() {
            Some(path) => read_prompts(path)?
                .first()
                .map(|p| p.ids.clone())
                .ok_or("empty prompts file")?,
            None => return Err("--vgraph-ab needs --prompts (real prompts, perf-row law)".into()),
        };
        let k = args.spec_k;
        let mut ab = header.clone();
        ab.push_str(&format!(
            "# vgraph-ab k={k} draft_head_rows={}: verify scan-chain segment graphs OFF vs ON, \
             interleaved, fresh states per arm per rep\n\
             rep\tarm\ttokens\tms_per_token\ttok_per_s\taccept_rate\tmean_accept_len\tverify_ms_share\n",
            model.draft_logits_width()
        ));
        let mut arm_means: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        let mut rep0: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
        for rep in 0..reps {
            for (slot, on) in [(0usize, false), (1usize, true)] {
                memra_engine::qwen4exp_gpu::set_verify_graphs(on);
                let cap = ab_prompt.len() + toks + k + 4;
                let mut ss = model.alloc_state(&engine, cap)?;
                let mut ds = model.mtp_state(de, cap)?;
                let report = model.spec_generate_ext(
                    &engine,
                    de,
                    &ab_prompt,
                    toks,
                    k,
                    &mut ss,
                    &mut ds,
                    None,
                    args.spec_opts,
                    None,
                )?;
                let decode_ms = report.total_ms - report.prefill_ms;
                let per_tok = decode_ms / report.tokens.len().max(1) as f64;
                arm_means[slot].push(per_tok);
                ab.push_str(&format!(
                    "{rep}\t{}\t{}\t{per_tok:.2}\t{:.2}\t{:.3}\t{:.2}\t{:.2}\n",
                    if on { "vgraph" } else { "eager" },
                    report.tokens.len(),
                    1e3 / per_tok,
                    report.accept_rate(),
                    report.mean_accept_len(),
                    report.verify_ms / decode_ms,
                ));
                if rep == 0 {
                    rep0[slot] = report.tokens.clone();
                }
            }
        }
        for (slot, name) in [(0usize, "eager"), (1usize, "vgraph")] {
            let v = &arm_means[slot];
            let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
            ab.push_str(&format!(
                "# arm {name}\tmean_of_means_ms={mean:.2}\tmin={:.2}\tmax={:.2}\ttok_per_s={:.2}\n",
                v.iter().copied().fold(f64::INFINITY, f64::min),
                v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                1e3 / mean,
            ));
        }
        let eager_mean = arm_means[0].iter().sum::<f64>() / arm_means[0].len().max(1) as f64;
        let graph_mean = arm_means[1].iter().sum::<f64>() / arm_means[1].len().max(1) as f64;
        let first_div = rep0[0]
            .iter()
            .zip(&rep0[1])
            .position(|(a, b)| a != b)
            .map(|p| p as i64)
            .unwrap_or(if rep0[0].len() == rep0[1].len() {
                -1
            } else {
                rep0[0].len().min(rep0[1].len()) as i64
            });
        ab.push_str(&format!(
            "# speedup_vgraph_over_eager\t{:.4}\n# rep0_eager_vs_vgraph_first_divergence\t{first_div}\n",
            eager_mean / graph_mean
        ));
        let path = args.out.join(format!("ab-vgraph-k{k}-{}.tsv", args.label));
        std::fs::write(&path, &ab)?;
        println!("{ab}\n# vgraph-ab receipt: {}", path.display());
        if first_div >= 0 {
            eprintln!(
                "vgraph-ab: the graph arm's chain DIVERGED from the eager arm at {first_div} — a \
                 replay must be bit-identical to the chain it captured"
            );
            std::process::exit(1);
        }
        // Leave the seam at its shipped default for whatever follows.
        memra_engine::qwen4exp_gpu::set_verify_graphs(false);
    }

    // ---------------------------------------------------------------- defer-ab
    // Interleaved A/B of the DEFERRED round readback (mtp11): host chain vs the
    // deferred chain (+ the sequential-guard sub-arm when pmin is armed — the
    // guard-forces-a-readback measurement the owner asked for). Every arm must
    // produce the SAME chain and the SAME admission counters per rep: the deferred
    // round is the same picks by construction, and this harness hard-fails on any
    // divergence rather than tolerating it.
    if let Some((reps, toks)) = args.defer_ab {
        let ab_prompt: Vec<u32> = match args.prompts.as_ref() {
            Some(path) => read_prompts(path)?
                .first()
                .map(|p| p.ids.clone())
                .ok_or("empty prompts file")?,
            None => return Err("--defer-ab needs --prompts (real prompts, perf-row law)".into()),
        };
        // K ladder in ONE model load (--spec-ladder; model load dominates box wall).
        let ks: Vec<usize> = if args.spec_ladder.is_empty() {
            vec![args.spec_k]
        } else {
            args.spec_ladder.clone()
        };
        let mut any_diverged = false;
        for &k in &ks {
            let guard = args.spec_opts.pmin > 0.0;
            let mut arms: Vec<(&str, memra_engine::qwen4exp_gpu::SpecOpts)> = Vec::new();
            let host_opts = memra_engine::qwen4exp_gpu::SpecOpts {
                defer: false,
                defer_guard_sync: false,
                ..args.spec_opts
            };
            let defer_opts = memra_engine::qwen4exp_gpu::SpecOpts {
                defer: true,
                defer_guard_sync: false,
                ..args.spec_opts
            };
            arms.push(("host", host_opts));
            arms.push(("defer", defer_opts));
            if guard {
                arms.push((
                    "defer-gsync",
                    memra_engine::qwen4exp_gpu::SpecOpts {
                        defer: true,
                        defer_guard_sync: true,
                        ..args.spec_opts
                    },
                ));
            }
            let mut ab = header.clone();
            ab.push_str(&format!(
            "# defer-ab k={k} draft_head_rows={} pmin={} adapt={:?}: host vs deferred round \
             readback, interleaved, fresh states per arm per rep\n\
             rep\tarm\ttokens\tms_per_token\ttok_per_s\taccept_rate\tmean_accept_len\tchain_ms_per_round\tverify_ms_per_round\tguard_stops\tzero_draft\n",
            model.draft_logits_width(),
            args.spec_opts.pmin,
            args.spec_opts.adapt_k_lo,
        ));
            // Protocol (owner amendment 2026-08-30, fleet-wide): interleaved x`reps`
            // rounds per arm by DEFAULT (the battery passes 3); escalate to x5 ONLY
            // on anomaly — (a) an arm's within-arm relative spread of the decision
            // median exceeds 0.5% (spread = (max-min)/median over that arm's reps),
            // or (b) a pair's verdict |median_arm - median_host| sits within 2x the
            // pair's POOLED spread ((spread_abs_host + spread_abs_arm)/2). The
            // escalation extends the AFFECTED pair (host rides along — interleaving
            // is the law), the receipt names which rule fired, and every arm reports
            // its spread so x`reps` sufficiency is itself receipted.
            const ESCALATE_CAP: usize = 5;
            let median = |v: &[f64]| -> f64 {
                let mut s = v.to_vec();
                s.sort_by(f64::total_cmp);
                s[s.len() / 2]
            };
            let spread_abs = |v: &[f64]| -> f64 {
                v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                    - v.iter().copied().fold(f64::INFINITY, f64::min)
            };
            let mut arm_means: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
            let mut diverged = false;
            let mut escal_notes: Vec<String> = Vec::new();
            let mut active: Vec<usize> = (0..arms.len()).collect();
            let mut rep = 0usize;
            let mut target = reps;
            loop {
                let mut rep_tokens: Vec<(usize, Vec<u32>)> = Vec::new();
                let mut rep_counters: Vec<(usize, u64, u64, usize, usize)> = Vec::new();
                for &slot in &active {
                    let (name, opts) = &arms[slot];
                    let cap = ab_prompt.len() + toks + k + 4;
                    let mut ss = model.alloc_state(&engine, cap)?;
                    let mut ds = model.mtp_state(de, cap)?;
                    let report = model.spec_generate_ext(
                        &engine, de, &ab_prompt, toks, k, &mut ss, &mut ds, None, *opts, None,
                    )?;
                    let decode_ms = report.total_ms - report.prefill_ms;
                    let per_tok = decode_ms / report.tokens.len().max(1) as f64;
                    arm_means[slot].push(per_tok);
                    let r = report.rounds.max(1) as f64;
                    ab.push_str(&format!(
                    "{rep}\t{name}\t{}\t{per_tok:.2}\t{:.2}\t{:.3}\t{:.2}\t{:.2}\t{:.2}\t{}\t{}\n",
                    report.tokens.len(),
                    1e3 / per_tok,
                    report.accept_rate(),
                    report.mean_accept_len(),
                    report.chain_ms / r,
                    report.verify_ms / r,
                    report.guard_stops,
                    report.zero_draft_rounds,
                ));
                    rep_tokens.push((slot, report.tokens.clone()));
                    rep_counters.push((
                        report.rounds,
                        report.drafted,
                        report.accepted,
                        report.guard_stops,
                        report.zero_draft_rounds,
                    ));
                }
                // Identity per round vs host (active[0] is always slot 0 = host).
                for i in 1..rep_tokens.len() {
                    let name = arms[rep_tokens[i].0].0;
                    if rep_tokens[i].1 != rep_tokens[0].1 {
                        eprintln!(
                            "defer-ab rep {rep}: arm {name} chain DIVERGED from host — the \
                             deferred round must be the same picks by construction",
                        );
                        diverged = true;
                    }
                    if rep_counters[i] != rep_counters[0] {
                        eprintln!(
                            "defer-ab rep {rep}: arm {name} counters {:?} != host {:?} — the \
                             deferred guard must stop where the host guard stopped",
                            rep_counters[i], rep_counters[0]
                        );
                        diverged = true;
                    }
                }
                rep += 1;
                if rep == reps && reps < ESCALATE_CAP {
                    // The x`reps` boundary: apply the two anomaly rules per pair.
                    let m0 = median(&arm_means[0]);
                    let s0 = spread_abs(&arm_means[0]);
                    let mut affected: Vec<usize> = Vec::new();
                    for slot in 1..arms.len() {
                        let mi = median(&arm_means[slot]);
                        let si = spread_abs(&arm_means[slot]);
                        let rule_a = s0 / m0 > 0.005 || si / mi > 0.005;
                        let rule_b = (mi - m0).abs() < 2.0 * ((s0 + si) / 2.0);
                        if rule_a || rule_b {
                            affected.push(slot);
                            escal_notes.push(format!(
                                "# escalation\tpair=host-vs-{}\trule={}\treps {reps}->{ESCALATE_CAP}\tspread_rel_host={:.4}%\tspread_rel_arm={:.4}%\tverdict_ms={:.3}\tpooled_spread_ms={:.3}\n",
                                arms[slot].0,
                                match (rule_a, rule_b) {
                                    (true, true) => "a+b",
                                    (true, false) => "a",
                                    _ => "b",
                                },
                                100.0 * s0 / m0,
                                100.0 * si / mi,
                                (mi - m0).abs(),
                                (s0 + si) / 2.0,
                            ));
                        }
                    }
                    if affected.is_empty() {
                        break;
                    }
                    active = std::iter::once(0).chain(affected).collect();
                    target = ESCALATE_CAP;
                    continue;
                }
                if rep >= target {
                    break;
                }
            }
            for (slot, (name, _)) in arms.iter().enumerate() {
                let v = &arm_means[slot];
                let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
                let med = median(v);
                let sa = spread_abs(v);
                ab.push_str(&format!(
                "# arm {name}\treps={}\tmedian_ms={med:.3}\tspread_abs_ms={sa:.3}\tspread_rel={:.4}%\tmean_ms={mean:.3}\ttok_per_s={:.2}\n",
                v.len(),
                100.0 * sa / med,
                1e3 / med,
            ));
            }
            for note in &escal_notes {
                ab.push_str(note);
            }
            if escal_notes.is_empty() {
                ab.push_str(&format!(
                    "# protocol\tdefault_reps={reps}\tescalation=none (x{reps} sufficient by both rules)\n"
                ));
            }
            let host_med = median(&arm_means[0]);
            let defer_med = median(&arm_means[1]);
            ab.push_str(&format!(
                "# speedup_defer_over_host\t{:.4}\t(decision medians)\n# identity\t{}\n",
                host_med / defer_med,
                if diverged { "FAIL" } else { "PASS" }
            ));
            let path = args.out.join(format!("ab-defer-k{k}-{}.tsv", args.label));
            std::fs::write(&path, &ab)?;
            println!("{ab}\n# defer-ab receipt: {}", path.display());
            any_diverged |= diverged;
        }
        if any_diverged {
            std::process::exit(1);
        }
    }

    // ---------------------------------------------------------------- router-ab
    // Interleaved A/B of the DEVICE MoE router (devtwin lane) under the SPEC loop:
    // router_host vs router_dev per rep, fresh states, the x`reps`+escalation protocol.
    // Divergence policy: the device route is set+order EXACT vs the host twin (audit/
    // oracle), but weights may differ within the documented ULP, so chain forks are
    // REPORTED (first index, informational — the accumulation-class posture of the
    // trunk/routerb16 seams), never tolerated silently: byte-identity LAW gates
    // (spec-gate/verify-bit) run per arm separately.
    if let Some((reps, toks)) = args.router_ab {
        let ab_prompt: Vec<u32> = match args.prompts.as_ref() {
            Some(path) => read_prompts(path)?
                .first()
                .map(|p| p.ids.clone())
                .ok_or("empty prompts file")?,
            None => return Err("--router-ab needs --prompts (real prompts, perf-row law)".into()),
        };
        let ks: Vec<usize> = if args.spec_ladder.is_empty() {
            vec![args.spec_k]
        } else {
            args.spec_ladder.clone()
        };
        // Which devtwin seam moves: routerdev (default), idxcache, or BOTH as one arm
        // ("devtwin" — the combined-stack verdict); picked by --ab-seam.
        let seam = match args.ab_seam.as_str() {
            "idxcache" => "idxcache",
            "idxsel" => "idxsel",
            "plecache" => "plecache",
            "devtwin" => "devtwin",
            "kvq" => "kvq",
            "idxq" => "idxq",
            _ => "routerdev",
        };
        let set_devtwin = |on: bool| match seam {
            "idxcache" => memra_engine::qwen4exp_gpu::set_idx_cache(on),
            "idxsel" => memra_engine::qwen4exp_gpu::set_idx_sel(on),
            "plecache" => memra_engine::qwen4exp_gpu::set_ple_cache(on),
            "devtwin" => {
                memra_engine::qwen4exp_gpu::set_router_dev(on);
                memra_engine::qwen4exp_gpu::set_idx_cache(on);
            }
            // Per-STATE latched: takes effect at the fresh per-arm allocs below.
            "kvq" => memra_engine::qwen4exp_gpu::set_kv_quant(on),
            "idxq" => memra_engine::qwen4exp_gpu::set_idxq(if on { "q8" } else { "f32" }),
            _ => memra_engine::qwen4exp_gpu::set_router_dev(on),
        };
        for &k in &ks {
            let arms: [(String, bool); 2] =
                [(format!("{seam}_off"), false), (format!("{seam}_on"), true)];
            let mut ab = header.clone();
            ab.push_str(&format!(
                "# router-ab seam={seam} k={k} pmin={} adapt={:?}: host twin vs device \
                 seam, interleaved, fresh states per arm per rep\n\
                 rep\tarm\ttokens\tms_per_token\ttok_per_s\taccept_rate\tmean_accept_len\tchain_ms_per_round\tverify_ms_per_round\tguard_stops\tzero_draft\tfirst_div_vs_host\n",
                args.spec_opts.pmin, args.spec_opts.adapt_k_lo,
            ));
            const ESCALATE_CAP: usize = 5;
            let median = |v: &[f64]| -> f64 {
                let mut s = v.to_vec();
                s.sort_by(f64::total_cmp);
                s[s.len() / 2]
            };
            let spread_abs = |v: &[f64]| -> f64 {
                v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                    - v.iter().copied().fold(f64::INFINITY, f64::min)
            };
            let mut arm_means: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
            let mut escal_notes: Vec<String> = Vec::new();
            let mut worst_first_div: i64 = -1;
            let mut rep = 0usize;
            let mut target = reps;
            loop {
                let mut host_chain: Vec<u32> = Vec::new();
                for (slot, (name, on)) in arms.iter().enumerate() {
                    set_devtwin(*on);
                    let cap = ab_prompt.len() + toks + k + 4;
                    let mut ss = model.alloc_state(&engine, cap)?;
                    let mut ds = model.mtp_state(de, cap)?;
                    let report = model.spec_generate_ext(
                        &engine,
                        de,
                        &ab_prompt,
                        toks,
                        k,
                        &mut ss,
                        &mut ds,
                        None,
                        args.spec_opts,
                        None,
                    )?;
                    let decode_ms = report.total_ms - report.prefill_ms;
                    let per_tok = decode_ms / report.tokens.len().max(1) as f64;
                    arm_means[slot].push(per_tok);
                    let r = report.rounds.max(1) as f64;
                    let first_div: i64 = if slot == 0 {
                        host_chain = report.tokens.clone();
                        -1
                    } else {
                        report
                            .tokens
                            .iter()
                            .zip(&host_chain)
                            .position(|(a, b)| a != b)
                            .map(|p| p as i64)
                            .unwrap_or(if report.tokens.len() == host_chain.len() {
                                -1
                            } else {
                                report.tokens.len().min(host_chain.len()) as i64
                            })
                    };
                    if first_div >= 0 && (worst_first_div < 0 || first_div < worst_first_div) {
                        worst_first_div = first_div;
                    }
                    ab.push_str(&format!(
                        "{rep}\t{name}\t{}\t{per_tok:.2}\t{:.2}\t{:.3}\t{:.2}\t{:.2}\t{:.2}\t{}\t{}\t{first_div}\n",
                        report.tokens.len(),
                        1e3 / per_tok,
                        report.accept_rate(),
                        report.mean_accept_len(),
                        report.chain_ms / r,
                        report.verify_ms / r,
                        report.guard_stops,
                        report.zero_draft_rounds,
                    ));
                }
                rep += 1;
                if rep == reps && reps < ESCALATE_CAP {
                    let m0 = median(&arm_means[0]);
                    let s0 = spread_abs(&arm_means[0]);
                    let m1 = median(&arm_means[1]);
                    let s1 = spread_abs(&arm_means[1]);
                    let rule_a = s0 / m0 > 0.005 || s1 / m1 > 0.005;
                    let rule_b = (m1 - m0).abs() < 2.0 * ((s0 + s1) / 2.0);
                    if rule_a || rule_b {
                        escal_notes.push(format!(
                            "# escalation\tpair=host-vs-dev\trule={}\treps {reps}->{ESCALATE_CAP}\tspread_rel_host={:.4}%\tspread_rel_arm={:.4}%\tverdict_ms={:.3}\tpooled_spread_ms={:.3}\n",
                            match (rule_a, rule_b) {
                                (true, true) => "a+b",
                                (true, false) => "a",
                                _ => "b",
                            },
                            100.0 * s0 / m0,
                            100.0 * s1 / m1,
                            (m1 - m0).abs(),
                            (s0 + s1) / 2.0,
                        ));
                        target = ESCALATE_CAP;
                        continue;
                    }
                    break;
                }
                if rep >= target {
                    break;
                }
            }
            memra_engine::qwen4exp_gpu::set_router_dev(
                memra_engine::qwen4exp_gpu::ROUTER_DEV_DEFAULT,
            );
            memra_engine::qwen4exp_gpu::set_idx_cache(
                memra_engine::qwen4exp_gpu::IDX_CACHE_DEFAULT,
            );
            for (slot, (name, _)) in arms.iter().enumerate() {
                let v = &arm_means[slot];
                let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
                let med = median(v);
                let sa = spread_abs(v);
                ab.push_str(&format!(
                    "# arm {name}\treps={}\tmedian_ms={med:.3}\tspread_abs_ms={sa:.3}\tspread_rel={:.4}%\tmean_ms={mean:.3}\ttok_per_s={:.2}\n",
                    v.len(),
                    100.0 * sa / med,
                    1e3 / med,
                ));
            }
            for note in &escal_notes {
                ab.push_str(note);
            }
            if escal_notes.is_empty() {
                ab.push_str(&format!(
                    "# protocol\tdefault_reps={reps}\tescalation=none (x{reps} sufficient by both rules)\n"
                ));
            }
            let host_med = median(&arm_means[0]);
            let dev_med = median(&arm_means[1]);
            ab.push_str(&format!(
                "# speedup_dev_over_host\t{:.4}\t(decision medians)\n# earliest_chain_divergence\t{worst_first_div}\n",
                host_med / dev_med,
            ));
            let path = args.out.join(format!("ab-{seam}-k{k}-{}.tsv", args.label));
            std::fs::write(&path, &ab)?;
            println!("{ab}\n# router-ab receipt: {}", path.display());
        }
    }

    // ---------------------------------------------------------------- greedy prompts
    if let Some(path) = args.prompts.as_ref() {
        let prompts = read_prompts(path)?;
        let mut receipt = header.clone();
        receipt.push_str(
            "prompt\tprompt_tokens\tgolden_tokens\tfirst_divergence_step\tmatched_prefix\tprefill_s\tmean_decode_ms\tmedian_decode_ms\tour_ids\tgolden_ids\n",
        );
        for prompt in &prompts {
            let n_new = args.max_new.min(prompt.golden.len()).max(1);
            let mut state = model.alloc_state(&engine, prompt.ids.len() + n_new + 1)?;
            let t_prefill = Instant::now();
            let logits = model.prefill(&engine, &prompt.ids, &mut state)?;
            let prefill_s = t_prefill.elapsed().as_secs_f64();
            let mut ours: Vec<u32> = vec![argmax(&logits[(prompt.ids.len() - 1) * vocab..]) as u32];
            let mut step_ms = Vec::with_capacity(n_new.saturating_sub(1));
            for j in 1..n_new {
                let t_step = Instant::now();
                let row = model.decode_step(&engine, ours[j - 1], &mut state)?;
                step_ms.push(ms(t_step));
                ours.push(argmax(&row) as u32);
            }
            let first_div = ours
                .iter()
                .zip(&prompt.golden)
                .position(|(a, b)| a != b)
                .map(|p| p as i64)
                .unwrap_or(-1);
            let matched = if first_div < 0 {
                ours.len()
            } else {
                first_div as usize
            };
            let mut sorted = step_ms.clone();
            sorted.sort_by(f64::total_cmp);
            let mean = step_ms.iter().sum::<f64>() / step_ms.len().max(1) as f64;
            let median = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
            receipt.push_str(&format!(
                "{}\t{}\t{}\t{first_div}\t{matched}\t{prefill_s:.2}\t{mean:.1}\t{median:.1}\t{}\t{}\n",
                prompt.index,
                prompt.ids.len(),
                prompt.golden.len(),
                csv(&ours),
                csv(&prompt.golden)
            ));
            println!(
                "# greedy prompt {}: first_divergence={first_div} matched={matched}/{} prefill={prefill_s:.2}s mean_decode={mean:.1}ms",
                prompt.index,
                prompt.golden.len()
            );
        }
        receipt.push_str(&format!("# vram\tpost-greedy\t{}\n", nvidia_smi()));
        let path = args.out.join(format!("greedy-gate-{}.tsv", args.label));
        std::fs::write(&path, &receipt)?;
        println!("# greedy receipt: {}", path.display());
    }

    // ---------------------------------------------------------------- spec byte-identity
    // (mtp-spec exactness law): plain greedy chain vs spec_generate chain per prompt.
    if args.spec_gate > 0 {
        let path_in = args
            .prompts
            .as_ref()
            .ok_or("--spec-gate needs --prompts (real prompts, greedy law)")?;
        let prompts = read_prompts(path_in)?;
        let k = args.spec_k;
        let n_new = args.spec_gate;
        let mut receipt = header.clone();
        receipt.push_str(&format!(
            "# spec-gate k={k} tokens={n_new}: plain greedy vs spec chains, byte identity\n\
             prompt\tfirst_divergence\taccept_rate\tmean_accept_len\thist\tspec_ms_per_token\tplain_ms_per_token\tpass\n"
        ));
        let mut all_pass = true;
        for prompt in &prompts {
            let mut ps = model.alloc_state(&engine, prompt.ids.len() + n_new + 4)?;
            let t_plain = Instant::now();
            let logits = model.prefill(&engine, &prompt.ids, &mut ps)?;
            let plain_prefill_ms = ms(t_plain);
            let mut next = argmax(&logits[(prompt.ids.len() - 1) * vocab..]) as u32;
            let mut plain = vec![next];
            let t_dec = Instant::now();
            for _ in 1..n_new {
                let row = model.decode_step(&engine, next, &mut ps)?;
                next = argmax(&row) as u32;
                plain.push(next);
            }
            let plain_ms = ms(t_dec) / (n_new - 1).max(1) as f64;
            let _ = plain_prefill_ms;
            let mut ss = model.alloc_state(&engine, prompt.ids.len() + n_new + k + 4)?;
            let mut ds = model.mtp_state(de, prompt.ids.len() + n_new + k + 4)?;
            let report = model.spec_generate_ext(
                &engine,
                de,
                &prompt.ids,
                n_new,
                k,
                &mut ss,
                &mut ds,
                None,
                args.spec_opts,
                None,
            )?;
            let first_div = plain
                .iter()
                .zip(&report.tokens)
                .position(|(a, b)| a != b)
                .map(|p| p as i64)
                .unwrap_or(if plain.len() == report.tokens.len() {
                    -1
                } else {
                    plain.len().min(report.tokens.len()) as i64
                });
            let pass = first_div < 0;
            all_pass &= pass;
            let spec_ms = (report.total_ms - report.prefill_ms) / report.tokens.len().max(1) as f64;
            receipt.push_str(&format!(
                "{}\t{first_div}\t{:.3}\t{:.2}\t{}\t{spec_ms:.2}\t{plain_ms:.2}\t{pass}\n",
                prompt.index,
                report.accept_rate(),
                report.mean_accept_len(),
                report
                    .accept_hist
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            ));
            let r = report.rounds.max(1) as f64;
            receipt.push_str(&format!(
                "# round-cost\tprompt={}\trounds={}\tchain_ms={:.2}\treplay_ms={:.2}\tverify_ms={:.2}\tcross_ms={:.3}\tzero_draft={}\tguard_stops={}\n",
                prompt.index,
                report.rounds,
                report.chain_ms / r,
                report.replay_ms / r,
                report.verify_ms / r,
                report.cross_ms / r,
                report.zero_draft_rounds,
                report.guard_stops,
            ));
            println!(
                "# spec-gate prompt {}: first_divergence={first_div} accept_rate={:.3} mean_len={:.2} spec={spec_ms:.1}ms plain={plain_ms:.1}ms",
                prompt.index,
                report.accept_rate(),
                report.mean_accept_len()
            );
        }
        receipt.push_str(&format!(
            "# verdict\tpolicy=byte-identity\tpass={all_pass}\n"
        ));
        let path = args.out.join(format!("spec-gate-k{k}-{}.tsv", args.label));
        std::fs::write(&path, &receipt)?;
        println!("# spec-gate receipt: {}", path.display());
        if !all_pass {
            eprintln!("spec-gate FAILED (receipt {})", path.display());
            std::process::exit(1);
        }
    }

    // ---------------------------------------------------------------- spec trace
    // The decay-diagnosis instrument (mtp10): per-round accept position, fork margins,
    // and carrier drift over EVERY prompt in --prompts, with a plain byte-identity twin
    // per prompt (trace mode must be the same program — asserted, hard fail).
    if args.spec_trace > 0 {
        let path_in = args
            .prompts
            .as_ref()
            .ok_or("--spec-trace needs --prompts (real prompts, greedy law)")?;
        let prompts = read_prompts(path_in)?;
        let k = args.spec_k;
        let n_new = args.spec_trace;
        let mut receipt = header.clone();
        receipt.push_str(&format!(
            "# spec-trace k={k} tokens={n_new}: per-round records; carrier_rel_l2[j] = draft seed for chunk row j vs the trunk's TRUE wide row (chain step j+1's input)\n\
             prompt\tround\tgen_pos\tbase\tk_round\ta\tdrafts\ttargets\td_top1\td_top2\td_tgt_logit\td_tgt_rank\tt_top1\tt_top2\tt_draft_logit\tt_entropy\tcarrier_rel_l2\tcarrier_cos\n"
        ));
        for prompt in &prompts {
            // Plain twin (the byte-identity assert for the traced program).
            let mut plain = Vec::with_capacity(n_new);
            {
                let mut ps = model.alloc_state(&engine, prompt.ids.len() + n_new + 4)?;
                let logits = model.prefill(&engine, &prompt.ids, &mut ps)?;
                let mut next = argmax(&logits[(prompt.ids.len() - 1) * vocab..]) as u32;
                plain.push(next);
                for _ in 1..n_new {
                    let row = model.decode_step(&engine, next, &mut ps)?;
                    next = argmax(&row) as u32;
                    plain.push(next);
                }
            }
            let mut ss = model.alloc_state(&engine, prompt.ids.len() + n_new + k + 4)?;
            let mut ds = model.mtp_state(de, prompt.ids.len() + n_new + k + 4)?;
            let mut rounds: Vec<memra_engine::qwen4exp_gpu::SpecTraceRound> = Vec::new();
            let report = model.spec_generate_ext(
                &engine,
                de,
                &prompt.ids,
                n_new,
                k,
                &mut ss,
                &mut ds,
                None,
                args.spec_opts,
                Some(&mut rounds),
            )?;
            let first_div = plain
                .iter()
                .zip(&report.tokens)
                .position(|(a, b)| a != b)
                .map(|p| p as i64)
                .unwrap_or(-1);
            for r in &rounds {
                receipt.push_str(&format!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.4}\t{}\t{}\n",
                    prompt.index,
                    r.round,
                    r.gen_pos,
                    r.base,
                    r.k,
                    r.a,
                    csv(&r.drafts),
                    csv(&r.targets),
                    r.draft_top1,
                    r.draft_top2,
                    r.draft_tgt_logit,
                    r.draft_tgt_rank,
                    r.target_top1,
                    r.target_top2,
                    r.target_draft_logit,
                    r.target_entropy,
                    r.carrier_rel_l2
                        .iter()
                        .map(|v| format!("{v:.4}"))
                        .collect::<Vec<_>>()
                        .join(","),
                    r.carrier_cos
                        .iter()
                        .map(|v| format!("{v:.5}"))
                        .collect::<Vec<_>>()
                        .join(","),
                ));
            }
            receipt.push_str(&format!(
                "# prompt {}\tfirst_divergence={first_div}\taccept={:.3}\tmean_len={:.2}\trounds={}\tzero_draft={}\tguard_stops={}\tcross_ms={:.2}\n",
                prompt.index,
                report.accept_rate(),
                report.mean_accept_len(),
                report.rounds,
                report.zero_draft_rounds,
                report.guard_stops,
                report.cross_ms,
            ));
            println!(
                "# spec-trace prompt {}: first_divergence={first_div} accept={:.3} rounds={}",
                prompt.index,
                report.accept_rate(),
                report.rounds
            );
            if first_div >= 0 {
                let path = args.out.join(format!("spec-trace-k{k}-{}.tsv", args.label));
                std::fs::write(&path, &receipt)?;
                eprintln!(
                    "spec-trace: traced chain DIVERGED from plain at {first_div} — trace mode \
                     must be the same program (receipt {})",
                    path.display()
                );
                std::process::exit(1);
            }
        }
        let path = args.out.join(format!("spec-trace-k{k}-{}.tsv", args.label));
        std::fs::write(&path, &receipt)?;
        println!("# spec-trace receipt: {}", path.display());
    }

    // ------------------------------------------------- DEEP-SEEDED verify-row BIT gate
    // The 262k-depth gate item (LADDER.md §5a): the shallow `--verify-bit-gate` seeds both
    // states from `goldens.input_ids` — 10 tokens — so it never crosses the 2,048-token QSA
    // selection horizon and cannot say anything about a deep-context path. The COMPARISON
    // needs no oracle (plain t==1 rows vs verify chunk rows fed the same tokens, same
    // config, bit-identity), so the only missing piece was a deep seed. This variant seeds
    // both states from the `--ladder-ids` corpus prefix at `--verify-bit-deep <fill>`,
    // chunked exactly like the ladder.
    //
    // Why it has a depth CEILING that is lower than the ladder's, stated rather than
    // discovered later: the gate is a TWO-STATE instrument by construction (a plain state
    // and a spec-armed state must hold the same prefix simultaneously), so it pays the
    // per-token state cost TWICE. At the measured 11.08 KiB/token that is ~2.9 GiB for a
    // 131,072 pair and ~5.7 GiB for a 262,144 pair against ~7.9 GiB free after the trunk —
    // so 131,072 is comfortable, 262,144 is at the edge, and a deeper pair does not exist.
    if args.verify_bit_deep > 0 {
        let path = args
            .ladder_ids
            .as_ref()
            .ok_or("--verify-bit-deep needs --ladder-ids (real pre-tokenized corpus)")?;
        let ids_text = std::fs::read_to_string(path)?;
        let ids: Vec<u32> = ids_text
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<u32>())
            .collect::<Result<_, _>>()?;
        let fill = args.verify_bit_deep;
        let n = if args.verify_bit > 0 {
            args.verify_bit
        } else {
            24
        };
        let k1 = args.spec_k + 1;
        let chunk = args.ladder_chunk.max(1);
        if ids.len() < fill {
            return Err(format!(
                "--verify-bit-deep {fill}: --ladder-ids has only {} tokens",
                ids.len()
            )
            .into());
        }
        let seed = &ids[..fill];
        let mut receipt = header.clone();
        receipt.push_str(&format!(
            "# verify-bit-deep: plain t==1 rows vs t=={k1} chunk rows, same fed tokens, \
             SEEDED FROM A {fill}-TOKEN CORPUS PREFIX (chunk {chunk}), n={n}\n\
             row\tabs_pos\tchunk\tbit_identical\n"
        ));
        let t_seed = Instant::now();
        let mut sa = model.alloc_state_reserve(&engine, fill + n + 2, chunk, None)?;
        let mut last = Vec::new();
        for piece in seed.chunks(chunk) {
            last = model.prefill_extend(&engine, piece, &mut sa, chunk)?;
        }
        println!(
            "# verify-bit-deep\tplain-seed-s={:.1}\t{}",
            t_seed.elapsed().as_secs_f64(),
            nvidia_smi()
        );
        let mut fed: Vec<u32> = vec![argmax(&last) as u32];
        let mut plain_rows: Vec<Vec<f32>> = Vec::with_capacity(n);
        for step in 0..n {
            let row = model.decode_step(&engine, fed[step], &mut sa)?;
            fed.push(argmax(&row) as u32);
            plain_rows.push(row);
        }
        // The plain state is dropped BEFORE the spec-armed one is allocated wherever that
        // is possible — but it is not: the rows have to be compared against a live second
        // walk, so both prefixes coexist. That is the two-state cost named above, and it is
        // why this gate reports its own VRAM line.
        let t_spec = Instant::now();
        let mut sb = model.alloc_state_reserve(&engine, fill + n + k1 + 2, chunk, None)?;
        model.spec_arm(&engine, &mut sb, k1)?;
        for piece in seed.chunks(chunk) {
            let _ = model.prefill_extend(&engine, piece, &mut sb, chunk)?;
        }
        println!(
            "# verify-bit-deep\tspec-seed-s={:.1}\t{}",
            t_spec.elapsed().as_secs_f64(),
            nvidia_smi()
        );
        let mut mismatched_rows = 0usize;
        let mut rows_done = 0usize;
        let mut chunk_id = 0usize;
        while rows_done < n {
            let tlen = k1.min(n - rows_done);
            let piece = &fed[rows_done..rows_done + tlen];
            let rows: Vec<f32> = if tlen == 1 {
                model.decode_step(&engine, piece[0], &mut sb)?
            } else {
                model.prefill(&engine, piece, &mut sb)?
            };
            for r in 0..tlen {
                let same = rows[r * vocab..(r + 1) * vocab]
                    .iter()
                    .zip(&plain_rows[rows_done + r])
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                if !same {
                    mismatched_rows += 1;
                }
                receipt.push_str(&format!(
                    "{}\t{}\t{chunk_id}\t{same}\n",
                    rows_done + r,
                    fill + rows_done + r
                ));
            }
            rows_done += tlen;
            chunk_id += 1;
        }
        let pass = mismatched_rows == 0;
        receipt.push_str(&format!(
            "# verdict\tfill={fill}\trows={n}\tmismatched={mismatched_rows}\t\
             policy=bit-identity\tpass={pass}\t{}\n",
            nvidia_smi()
        ));
        let out = args.out.join(format!("verify-bit-deep-{}.tsv", args.label));
        std::fs::write(&out, &receipt)?;
        println!("{receipt}\n# verify-bit-deep receipt: {}", out.display());
        if !pass {
            eprintln!("verify-bit-deep FAILED (receipt {})", out.display());
            std::process::exit(1);
        }
    }

    // ---------------------------------------------------------------- long-context ladder
    // The yarn affordability cell (owner question: "can we afford full context on two
    // cards"): ascending fill depths over REAL pre-tokenized text, chunked prefill with
    // wall clocks + per-card VRAM at every rung, then a timed greedy decode whose
    // continuation STAYS in context (real serving shape: generation interleaved with the
    // document). Rows flush per rung so a reclaimed box still banks partial results.
    if !args.ladder.is_empty() {
        let ids_text = std::fs::read_to_string(args.ladder_ids.as_ref().unwrap())?;
        let ids: Vec<u32> = ids_text
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<u32>())
            .collect::<Result<_, _>>()?;
        let max_rung = *args.ladder.iter().max().unwrap();
        // Capacity must cover the WORST case of the timing loop, which is FIVE rounds, not
        // three. The loop runs `round_steps = max(ladder_decode/3, 8)` steps per round and
        // ESCALATES to 5 rounds when the within-arm spread exceeds 0.5%, so it can consume
        // `5/3 x ladder_decode` positions — plus `--profile` steps after the rung.
        //
        // The old `ladder_decode + 2` assumed the x3 path and it is how the at-depth idxsel
        // audit cell died: `--ladder-decode 600` reserved 602 positions, the audit's restored
        // host work made the rounds noisy, the instrument correctly escalated to x5, and the
        // 1,000th decode step hit "state capacity exceeded" after 755 s of prefill. The
        // failure mode is the nastiest shape available — the escalation exists precisely for
        // noisy arms, so the reservation was wrong exactly when the extra rounds were needed,
        // and it destroyed the rung it was measuring instead of shortening it.
        // Read once: whether prefill chunks take the shared lock (see the chunk site).
        let prefill_lock_shared =
            std::env::var("MEMRA_Q4E_MEASURE_LOCK_PREFILL").as_deref() != Ok("off");
        let round_steps_max = (args.ladder_decode / 3).max(8);
        // The interleaved A/B consumes positions on TOP of the timing loop, and it has its own
        // escalation (rounds + 2). Worst case per rung: 2 arms x (rounds+2) reps x (2 warmup +
        // steps), plus one untimed arming step and `--profile` profiled steps for EACH of the
        // two arms in the per-arm section profile. Reserved here rather than discovered at the
        // last step of a 25-80 minute prefill — the same defect the comment above records, and
        // adding an instrument without extending the reservation is exactly how it recurs.
        let ab_steps = match args.ladder_ab_seam.as_deref() {
            Some(_) => {
                let reps = args.ladder_ab_rounds + 2;
                2 * reps * (2 + args.ladder_ab_steps) + 2 * (1 + args.profile)
            }
            None => 0,
        };
        let cap =
            max_rung + args.ladder.len() * (round_steps_max * 5 + args.profile + 2 + ab_steps) + 64;
        if ids.len() < cap {
            return Err(format!(
                "--ladder-ids has {} tokens; the ladder needs >= {cap} (deepest rung + \
                 in-context continuations)",
                ids.len()
            )
            .into());
        }
        let kv_e1: Option<Engine> = if args.ladder_kv_dev1 {
            let e1 = Engine::new(1)?;
            memra_engine::qwen4exp_gpu::tp2_enable_p2p(&engine, &e1)?;
            Some(e1)
        } else {
            None
        };
        let host_rss_mib = || -> u64 {
            std::fs::read_to_string("/proc/self/status")
                .ok()
                .and_then(|s| {
                    s.lines().find(|l| l.starts_with("VmRSS:")).and_then(|l| {
                        l.split_whitespace()
                            .nth(1)
                            .and_then(|v| v.parse::<u64>().ok())
                    })
                })
                .map(|kb| kb / 1024)
                .unwrap_or(0)
        };
        let receipt_path = args.out.join(format!("ladder-{}.tsv", args.label));
        let mut receipt = header.clone();
        receipt.push_str(&format!(
            "# ladder\trungs={:?}\tchunk={}\tdecode={}\tkv_dev1={}\tspec={:?}\tids={}\tcap={cap}\n\
             # vram\tpre-ladder\t{}\n",
            args.ladder,
            args.ladder_chunk,
            args.ladder_decode,
            args.ladder_kv_dev1,
            args.ladder_spec,
            ids.len(),
            nvidia_smi()
        ));
        if let Some(spec_k) = args.ladder_spec {
            // ---- SPEC arm: per rung a FRESH pair of states and one spec_generate_ext
            // (chunked co-prefill + ring-bounded wide stash) under the CLI admission
            // policy. Timing sub-rounds derive from the per-round wall samples of the
            // ONE generation (x3 protocol; a fresh 1M prefill per timing round is
            // prohibitive — stated, spreads named per rung).
            receipt.push_str(
                "rung\tprefill_s\tdraft_prefill_s\tdecode_ms_per_tok\ttok_per_s\taccept\t\
                 rounds\tzero_draft\tsub_rounds\tspread_pct\tlooped\tcross_mb\tvram\t\
                 continuation_ids\n",
            );
            std::fs::write(&receipt_path, &receipt)?;
            let sampler =
                args.ladder_spec_sampled
                    .map(|seed| memra_engine::qwen4exp_gpu::SpecSamplerCfg {
                        temperature: 1.0,
                        top_p: 0.95,
                        top_k: 20,
                        seed,
                    });
            // Shape suffix (spec-at-depth per shape): the LAST tokens of the fed sequence are
            // the chat-template render, so the model sees [deep document][task turn] — what a
            // long agentic request actually looks like. The corpus is trimmed by exactly the
            // suffix length, so total fill == rung and every depth/VRAM row stays comparable
            // with the raw arm. Refused loudly if the suffix does not fit the shallowest rung,
            // because a silently truncated task turn would measure a different prompt than the
            // one named in the receipt.
            let shape_ids: Vec<u32> = match args.ladder_spec_shape.as_deref() {
                Some(path) => {
                    let ps = read_prompts(path)?;
                    let first = ps
                        .first()
                        .ok_or("--ladder-spec-shape: the prompts file has no rows")?;
                    if first.ids.is_empty() {
                        return Err("--ladder-spec-shape: the first row has no ids".into());
                    }
                    let min_rung = *args.ladder.iter().min().unwrap();
                    if first.ids.len() + 16 > min_rung {
                        return Err(format!(
                            "--ladder-spec-shape: suffix is {} tokens but the shallowest rung \
                             is {min_rung}; a truncated task turn is a different prompt",
                            first.ids.len()
                        )
                        .into());
                    }
                    first.ids.clone()
                }
                None => Vec::new(),
            };
            receipt.push_str(&format!(
                "# spec-shape\tsource={}\tsuffix_tokens={}\n",
                args.ladder_spec_shape
                    .as_deref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "(none: raw corpus)".into()),
                shape_ids.len()
            ));
            std::fs::write(&receipt_path, &receipt)?;
            for &rung in &args.ladder {
                let cap_r = rung + args.ladder_decode + spec_k + 8;
                let mut st = model.alloc_state_reserve(
                    &engine,
                    cap_r,
                    args.ladder_chunk.max(1),
                    kv_e1.as_ref(),
                )?;
                let mut ds = model.mtp_state(de, cap_r)?;
                let mut o = args.spec_opts;
                o.prefill_chunk = Some(args.ladder_chunk.max(1));
                o.wide_ring = Some(4 * args.ladder_chunk.max(spec_k + 2));
                // Fed sequence: corpus head trimmed by the suffix, then the shape turn.
                let fed: Vec<u32> = if shape_ids.is_empty() {
                    ids[..rung].to_vec()
                } else {
                    let head = rung - shape_ids.len();
                    let mut v = Vec::with_capacity(rung);
                    v.extend_from_slice(&ids[..head]);
                    v.extend_from_slice(&shape_ids);
                    v
                };
                let report = model.spec_generate_ext(
                    &engine,
                    de,
                    &fed,
                    args.ladder_decode,
                    spec_k,
                    &mut st,
                    &mut ds,
                    sampler,
                    o,
                    None,
                )?;
                let decode_ms = report.total_ms - report.prefill_ms - report.draft_prefill_ms;
                let per_tok = decode_ms / report.tokens.len().max(1) as f64;
                // x3 sub-rounds from the round-wall samples (token-equal thirds).
                let mut sub_rates: Vec<f64> = Vec::new();
                if let (Some(first), Some(last)) =
                    (report.round_wall.first(), report.round_wall.last())
                {
                    let n_thirds = 3usize;
                    let tok_span = last.0.saturating_sub(first.0).max(1);
                    let mut prev = *first;
                    for i in 1..=n_thirds {
                        let target = first.0 + tok_span * i / n_thirds;
                        if let Some(&s) = report.round_wall.iter().find(|(tk, _)| *tk >= target) {
                            let toks = s.0.saturating_sub(prev.0).max(1);
                            sub_rates.push((s.1 - prev.1) / toks as f64);
                            prev = s;
                        }
                    }
                }
                let spread_pct = if sub_rates.len() >= 2 {
                    let (mn, mx) = sub_rates
                        .iter()
                        .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
                    100.0 * (mx - mn) / sub_rates[sub_rates.len() / 2].max(1e-9)
                } else {
                    f64::NAN
                };
                let looped = {
                    let tail: Vec<u32> = report.tokens.iter().rev().take(24).copied().collect();
                    (1..=8usize).any(|cycle| {
                        tail.len() >= 2 * cycle
                            && tail
                                .windows(cycle)
                                .step_by(cycle)
                                .all(|w| w == &tail[..cycle])
                    })
                };
                let row = format!(
                    "{rung}\t{:.1}\t{:.1}\t{per_tok:.1}\t{:.2}\t{:.3}\t{}\t{}\tsub_ms={:?}\t{spread_pct:.2}\t{looped}\t{:.1}\t{}\t{}\n",
                    report.prefill_ms / 1e3,
                    report.draft_prefill_ms / 1e3,
                    1e3 / per_tok,
                    report.accept_rate(),
                    report.rounds,
                    report.zero_draft_rounds,
                    sub_rates
                        .iter()
                        .map(|v| (v * 10.0).round() / 10.0)
                        .collect::<Vec<_>>(),
                    report.cross_bytes as f64 / 1e6,
                    nvidia_smi(),
                    csv(&report.tokens)
                );
                receipt.push_str(&row);
                std::fs::write(&receipt_path, &receipt)?;
                println!("# ladder-spec-rung\t{row}");
            }
            println!("# ladder receipt: {}", receipt_path.display());
        } else {
            receipt.push_str(
            "rung\tprefill_seg_s\tprefill_cum_s\tchunks\tdecode_warm_mean_ms\tdecode_median_ms\t\
             decode_p90_ms\ttok_per_s\ttiming_rounds\tlooped\thost_rss_mib\tvram\t\
             continuation_ids\n",
        );
            std::fs::write(&receipt_path, &receipt)?;
            let tp2_route = if args.ladder_tp2 { tp2.as_ref() } else { None };
            let mut state = match tp2_route {
                Some((e1, shard)) => {
                    // TP2-native state: halves at capacity, single-card KV stubbed —
                    // the 1M route (yarn round 2).
                    model.alloc_state_tp2(&engine, e1, shard, cap, args.ladder_chunk.max(1))?
                }
                None => model.alloc_state_reserve(
                    &engine,
                    cap,
                    args.ladder_chunk.max(1),
                    kv_e1.as_ref(),
                )?,
            };
            println!("# ladder\tstate-allocated\t{}", nvidia_smi());
            let mut pos = 0usize;
            let mut consumed = 0usize;
            let mut cum_prefill = 0f64;
            for &rung in &args.ladder {
                if rung <= pos {
                    return Err(format!("ladder rung {rung} <= current fill {pos}").into());
                }
                let need = rung - pos;
                let seg = &ids[consumed..consumed + need];
                consumed += need;
                let t_seg = Instant::now();
                let mut chunks = 0usize;
                let mut last_logits = Vec::new();
                for piece in seg.chunks(args.ladder_chunk) {
                    // SHARED lock, per CHUNK. A prefill is not timed and does not need
                    // protecting, but it must not race a sibling's timed block — and it cannot
                    // simply hold the lock for its whole 11-80 minutes, because LOCK_EX waits
                    // for shared holders too and that rebuilds the 30-minute exclusive window.
                    // Taking and RELEASING it per chunk makes the prefill yield: a waiting
                    // exclusive gets in at this boundary, and the next chunk blocks until the
                    // timed block is done. Worst-case wait for a measurement is one chunk.
                    //
                    // `MEMRA_Q4E_MEASURE_LOCK_PREFILL=off` skips it. That exists because per-chunk
                    // yielding only bounds the wait when EVERY participant is fine-grained: a
                    // sibling holding LOCK_EX across a whole cell starves a per-chunk LOCK_SH
                    // acquirer completely, measured at 960 s on this lane with BOTH cards at 0%
                    // utilisation. When that is the situation, an unlocked prefill plus EXCLUSIVE
                    // timed blocks keeps the timed rows valid — which is the part that matters —
                    // instead of trading them for an idle box. Default stays `sh`.
                    let chunk_lock = if prefill_lock_shared {
                        MeasureLock::maybe(args.measure_lock.as_deref(), LockMode::Shared)?
                    } else {
                        None
                    };
                    // First chunk of each segment: section profile (sync-bounded —
                    // shares, not absolutes; where the prefill wall goes at this depth).
                    let profiled = args.profile > 0 && chunks == 0;
                    if profiled {
                        memra_engine::qwen4exp_gpu::prof::enable();
                    }
                    last_logits = match tp2_route {
                        Some((e1, shard)) => model.prefill_extend_tp2(
                            &engine,
                            e1,
                            shard,
                            piece,
                            &mut state,
                            args.ladder_chunk,
                        )?,
                        None => {
                            model.prefill_extend(&engine, piece, &mut state, args.ladder_chunk)?
                        }
                    };
                    if profiled {
                        let mut rows = memra_engine::qwen4exp_gpu::prof::take();
                        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
                        let total: f64 = rows.iter().map(|r| r.1).sum();
                        receipt.push_str(&format!(
                            "# prefill-profile\trung={rung}\tfirst_chunk_t={}\n",
                            piece.len()
                        ));
                        for (name, secs, calls) in rows.iter().take(10) {
                            receipt.push_str(&format!(
                                "# prefill-profile\t{name}\t{:.1}ms\t{:.1}%\tcalls={calls}\n",
                                secs * 1e3,
                                100.0 * secs / total.max(1e-12)
                            ));
                        }
                        std::fs::write(&receipt_path, &receipt)?;
                    }
                    pos += piece.len();
                    chunks += 1;
                    // Release at the chunk boundary: this is the yield point.
                    drop(chunk_lock);
                    if chunks % 8 == 0 || pos == rung {
                        println!(
                            "# ladder-progress\tfill={pos}\telapsed_s={:.1}\t{}",
                            t_seg.elapsed().as_secs_f64(),
                            nvidia_smi()
                        );
                    }
                }
                let seg_s = t_seg.elapsed().as_secs_f64();
                cum_prefill += seg_s;
                let mut next = argmax(&last_logits) as u32;
                let mut continuation = vec![next];
                // x3 timing rounds per arm (fleet protocol 2026-08-30); escalate to x5 when
                // the within-arm relative spread of the round medians exceeds 0.5%. Rounds
                // are consecutive decode segments on the SAME fill (a fresh prefill per
                // round is prohibitive at these depths — stated here, spreads named below).
                let round_steps = (args.ladder_decode / 3).max(8);
                // Serialise ONLY the timed rounds. The prefill above ran unlocked on purpose:
                // it is 11-80 minutes and its wall is not the number this loop measures, so
                // holding the box's measurement lock across it would block every other agent
                // for the whole cell to protect a figure nobody is claiming.
                let rung_lock =
                    MeasureLock::maybe(args.measure_lock.as_deref(), LockMode::Exclusive)?;
                if let Some(l) = rung_lock.as_ref() {
                    receipt.push_str(&l.receipt(&format!("rung-timing-{rung}")));
                    std::fs::write(&receipt_path, &receipt)?;
                }
                let mut round_medians: Vec<f64> = Vec::new();
                let mut all_warm: Vec<f64> = Vec::new();
                let mut total_rounds = 3usize;
                let mut escalated = false;
                let mut r = 0usize;
                while r < total_rounds {
                    let mut step_ms = Vec::with_capacity(round_steps);
                    for step in 0..round_steps {
                        let t_step = Instant::now();
                        let row = match tp2_route {
                            Some((e1, shard)) => {
                                model.decode_step_tp2(&engine, e1, shard, next, &mut state)?
                            }
                            None => model.decode_step(&engine, next, &mut state)?,
                        };
                        // Round 0 steps 0-1: allocator warmup + graph capture (excluded).
                        if r > 0 || step >= 2 {
                            step_ms.push(ms(t_step));
                        }
                        pos += 1;
                        next = argmax(&row) as u32;
                        continuation.push(next);
                    }
                    all_warm.extend_from_slice(&step_ms);
                    let mut s = step_ms.clone();
                    s.sort_by(f64::total_cmp);
                    round_medians.push(s[s.len() / 2]);
                    r += 1;
                    if r == 3 && total_rounds == 3 {
                        let (mn, mx) = round_medians
                            .iter()
                            .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
                        let mid = round_medians[1].max(1e-9);
                        if (mx - mn) / mid > 0.005 {
                            total_rounds = 5;
                            escalated = true;
                        }
                    }
                }
                drop(rung_lock);
                let mut sorted = all_warm.clone();
                sorted.sort_by(f64::total_cmp);
                let mean = all_warm.iter().sum::<f64>() / all_warm.len().max(1) as f64;
                let (mn, mx) = round_medians
                    .iter()
                    .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
                let spread_pct =
                    100.0 * (mx - mn) / round_medians[round_medians.len() / 2].max(1e-9);
                // Greedy-loop flag (greedy law: loops are an artifact, never a finding):
                // any cycle of length 1..=8 covering the last 24 tokens.
                let looped = {
                    let tail: Vec<u32> = continuation.iter().rev().take(24).copied().collect();
                    (1..=8usize).any(|cycle| {
                        tail.len() >= 2 * cycle
                            && tail
                                .windows(cycle)
                                .step_by(cycle)
                                .all(|w| w == &tail[..cycle])
                    })
                };
                let row = format!(
                    "{rung}\t{seg_s:.1}\t{cum_prefill:.1}\t{chunks}\t{mean:.1}\t{:.1}\t{:.1}\t{:.2}\trounds={}x{round_steps} medians={:?} spread={spread_pct:.2}%{}\t{looped}\t{}\t{}\t{}\n",
                    sorted[sorted.len() / 2],
                    sorted[(sorted.len() * 9) / 10],
                    1e3 / mean,
                    round_medians.len(),
                    round_medians
                        .iter()
                        .map(|v| (v * 10.0).round() / 10.0)
                        .collect::<Vec<_>>(),
                    if escalated { " (escalated x5)" } else { "" },
                    host_rss_mib(),
                    nvidia_smi(),
                    csv(&continuation)
                );
                receipt.push_str(&row);
                std::fs::write(&receipt_path, &receipt)?;
                println!("# ladder-rung\t{row}");
                // The jitter tail of the SAME samples the row above medianed. Appended as its
                // own line rather than as row columns, because the row's column list is read
                // by every banked parser in this lane (append-only discipline).
                {
                    let j = jitter(&all_warm);
                    let line = format!(
                        "# ladder-jitter\trung={rung}\tarm=rung-default\t{}\n",
                        j.receipt()
                    );
                    receipt.push_str(&line);
                    std::fs::write(&receipt_path, &receipt)?;
                    print!("{line}");
                }

                // ---- WITHIN-PREFILL interleaved decode A/B over one seam (host-lever lane).
                // The arms share this rung's prefill and this state, alternate A/B/A/B... , and
                // flip which arm leads on odd reps so any lead-position bias cancels rather
                // than accumulating. Escalation follows the fleet protocol: x3, and x5 when
                // either arm's own spread exceeds 0.5% or the verdict sits inside 2x the
                // pooled spread (a verdict smaller than the instrument's noise is not a
                // verdict).
                if let Some(seam) = args.ladder_ab_seam.as_deref() {
                    // Exact save/restore. Re-running apply_env_seams would NOT restore a seam
                    // absent from MEMRA_Q4E_SEAMS, and the run would carry on from whichever
                    // arm ran last — a silent arm flip, which is the failure this lane has
                    // already been bitten by once (the golden pin, PROFILE-10 §4).
                    let seam_entry = memra_engine::qwen4exp_gpu::seam_state(seam);
                    let warm = 2usize; // arming a seam can pay a one-time rebuild; never timed.
                    let mut med: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
                    let mut all: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
                    let cs0 = host_census();
                    let mut cpu_marks: Vec<i32> = Vec::new();
                    let mut reps = args.ladder_ab_rounds;
                    let mut ab_escalated = false;
                    let mut rep = 0usize;
                    while rep < reps {
                        // Lead flip: rep 0 runs OFF,ON; rep 1 runs ON,OFF; ...
                        let order: [bool; 2] = if rep % 2 == 0 {
                            [false, true]
                        } else {
                            [true, false]
                        };
                        let rep_lock =
                            MeasureLock::maybe(args.measure_lock.as_deref(), LockMode::Exclusive)?;
                        if let Some(l) = rep_lock.as_ref() {
                            receipt.push_str(&l.receipt(&format!("ab-rep-{rung}-{seam}-{rep}")));
                            std::fs::write(&receipt_path, &receipt)?;
                        }
                        for &arm_on in &order {
                            if !memra_engine::qwen4exp_gpu::set_seam(seam, arm_on, None) {
                                return Err(
                                    format!("--ladder-ab-seam {seam}: set_seam refused").into()
                                );
                            }
                            let mut step_ms = Vec::with_capacity(args.ladder_ab_steps);
                            for step in 0..(warm + args.ladder_ab_steps) {
                                let t_step = Instant::now();
                                let logits = model.decode_step(&engine, next, &mut state)?;
                                let dt = ms(t_step);
                                if step >= warm {
                                    step_ms.push(dt);
                                }
                                if args.host_probe {
                                    cpu_marks.push(self_cpu());
                                }
                                pos += 1;
                                next = argmax(&logits) as u32;
                                continuation.push(next);
                            }
                            let idx = usize::from(arm_on);
                            all[idx].extend_from_slice(&step_ms);
                            let mut s = step_ms.clone();
                            s.sort_by(f64::total_cmp);
                            med[idx].push(s[s.len() / 2]);
                        }
                        // Released BETWEEN reps: both arms of an interleaved pair must be
                        // taken under one lock (that is what makes them comparable), but a
                        // whole escalating A/B must not monopolise the box.
                        drop(rep_lock);
                        rep += 1;
                        if rep == args.ladder_ab_rounds && !ab_escalated {
                            let spread = |v: &Vec<f64>| {
                                let (mn, mx) = v
                                    .iter()
                                    .fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
                                (mx - mn) / mx.max(1e-9)
                            };
                            let mid = |v: &mut Vec<f64>| {
                                v.sort_by(f64::total_cmp);
                                v[v.len() / 2]
                            };
                            let (so, sn) = (spread(&med[0]), spread(&med[1]));
                            let (mo, mn_) = (mid(&mut med[0].clone()), mid(&mut med[1].clone()));
                            let verdict = (mo - mn_).abs() / mo.max(1e-9);
                            if so > 0.005 || sn > 0.005 || verdict < 2.0 * so.max(sn) {
                                reps = args.ladder_ab_rounds + 2;
                                ab_escalated = true;
                            }
                        }
                    }
                    let cs1 = host_census();
                    let migrations = cpu_marks.windows(2).filter(|w| w[0] != w[1]).count();
                    let distinct = {
                        let mut v = cpu_marks.clone();
                        v.sort_unstable();
                        v.dedup();
                        v.len()
                    };
                    let summary = |idx: usize, name: &str| -> (f64, String) {
                        let mut m = med[idx].clone();
                        m.sort_by(f64::total_cmp);
                        let mid = m[m.len() / 2];
                        let (lo, hi) = m
                            .iter()
                            .fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
                        let j = jitter(&all[idx]);
                        (
                            mid,
                            format!(
                                "# ladder-ab\trung={rung}\tseam={seam}\tarm={name}\t\
                                 median_ms={mid:.2}\ttok_per_s={:.2}\treps={}\t\
                                 medians={:?}\tspread={:.2}%\t{}\n",
                                1e3 / mid,
                                m.len(),
                                med[idx]
                                    .iter()
                                    .map(|v| (v * 100.0).round() / 100.0)
                                    .collect::<Vec<_>>(),
                                100.0 * (hi - lo) / hi.max(1e-9),
                                j.receipt()
                            ),
                        )
                    };
                    let (off_ms, off_line) = summary(0, "off");
                    let (on_ms, on_line) = summary(1, "on");
                    let verdict = format!(
                        "# ladder-ab-verdict\trung={rung}\tseam={seam}\toff_ms={off_ms:.2}\t\
                         on_ms={on_ms:.2}\tspeedup={:.4}x\tdelta_pct={:.2}%\treps_per_arm={}{}\t\
                         host_probe={}\tvol_cs={}\tnonvol_cs={}\tthreads={}\t\
                         launch_cpu_migrations={migrations}\tlaunch_cpus_seen={distinct}\n",
                        off_ms / on_ms.max(1e-9),
                        100.0 * (off_ms - on_ms) / off_ms.max(1e-9),
                        med[0].len(),
                        if ab_escalated { " (escalated)" } else { "" },
                        args.host_probe,
                        cs1.vol_cs.saturating_sub(cs0.vol_cs),
                        cs1.nonvol_cs.saturating_sub(cs0.nonvol_cs),
                        cs1.threads,
                    );
                    receipt.push_str(&off_line);
                    receipt.push_str(&on_line);
                    receipt.push_str(&verdict);
                    if args.host_probe {
                        for (comm, vol, nonvol, cpu) in &cs1.per_thread {
                            receipt.push_str(&format!(
                                "# host-thread\trung={rung}\tcomm={comm}\tvol_cs={vol}\t\
                                 nonvol_cs={nonvol}\tlast_cpu={cpu}\n"
                            ));
                        }
                    }
                    std::fs::write(&receipt_path, &receipt)?;
                    print!("{off_line}{on_line}{verdict}");

                    // A SECTION PROFILE PER ARM at this depth — the before/after host profile
                    // the lever is judged on. Sync-bounded, so shares not absolutes, and it
                    // runs after the timed reps so it cannot contaminate them.
                    if args.profile > 0 {
                        for (arm_on, name) in [(false, "off"), (true, "on")] {
                            memra_engine::qwen4exp_gpu::set_seam(seam, arm_on, None);
                            // One untimed step first: arming pays a one-time cache rebuild,
                            // and attributing that to the section would misprice the arm.
                            let logits = model.decode_step(&engine, next, &mut state)?;
                            pos += 1;
                            next = argmax(&logits) as u32;
                            continuation.push(next);
                            memra_engine::qwen4exp_gpu::prof::enable();
                            for _ in 0..args.profile {
                                let logits = model.decode_step(&engine, next, &mut state)?;
                                pos += 1;
                                next = argmax(&logits) as u32;
                                continuation.push(next);
                            }
                            let mut rows = memra_engine::qwen4exp_gpu::prof::take();
                            rows.sort_by(|a, b| b.1.total_cmp(&a.1));
                            let total: f64 = rows.iter().map(|r| r.1).sum();
                            receipt.push_str(&format!(
                                "# ab-profile\trung={rung}\tseam={seam}\tarm={name}\tsteps={}\n",
                                args.profile
                            ));
                            for (sect, secs, calls) in rows.iter().take(12) {
                                receipt.push_str(&format!(
                                    "# ab-profile\trung={rung}\tarm={name}\t{sect}\t{:.1}ms\t\
                                     {:.1}%\tcalls={calls}\n",
                                    secs * 1e3 / args.profile as f64,
                                    100.0 * secs / total.max(1e-12)
                                ));
                            }
                            std::fs::write(&receipt_path, &receipt)?;
                        }
                    }

                    // Restore, loudly. Stating the restored arm in the receipt is what makes
                    // every number AFTER this block readable.
                    if let Some(was) = seam_entry {
                        memra_engine::qwen4exp_gpu::set_seam(seam, was, None);
                        let line = format!(
                            "# ladder-ab-restore\tseam={seam}\tarm={}\n",
                            if was { "on" } else { "off" }
                        );
                        receipt.push_str(&line);
                        std::fs::write(&receipt_path, &receipt)?;
                        print!("{line}");
                    }
                    let _ = (&cs0, &cs1);
                }
                // A profiled decode step per rung (sync-bounded shares; the timing rows
                // above are the absolutes): where the host indexer starts to bite.
                if args.profile > 0 {
                    memra_engine::qwen4exp_gpu::prof::enable();
                    for _ in 0..args.profile {
                        let row = model.decode_step(&engine, next, &mut state)?;
                        pos += 1;
                        next = argmax(&row) as u32;
                        continuation.push(next);
                    }
                    let mut rows = memra_engine::qwen4exp_gpu::prof::take();
                    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
                    let total: f64 = rows.iter().map(|r| r.1).sum();
                    receipt.push_str(&format!("# profile\trung={rung}\tsteps={}\n", args.profile));
                    for (name, secs, calls) in rows.iter().take(12) {
                        receipt.push_str(&format!(
                            "# profile\t{name}\t{:.1}ms\t{:.1}%\tcalls={calls}\n",
                            secs * 1e3 / args.profile as f64,
                            100.0 * secs / total
                        ));
                    }
                    std::fs::write(&receipt_path, &receipt)?;
                }
            }
            println!("# ladder receipt: {}", receipt_path.display());
        }
    }

    // ---------------------------------------------------------------- cross-arm logits
    if let Some(other) = args.compare_logits.as_ref() {
        let ours_path = args.out.join(format!("probe-logits-{}.bin", args.label));
        let ours_bytes = std::fs::metadata(&ours_path)?.len() as usize / 4;
        let rows = ours_bytes / vocab;
        let ours = read_f32_bin(&ours_path, rows * vocab)?;
        let reference = read_f32_bin(other, rows * vocab)?;
        let mut receipt = header.clone();
        receipt.push_str(&format!(
            "# reference={}\tcandidate={}\trows={rows}\tvocab={vocab}\n\
             row\ttop1_ref\ttop1_ours\ttop1_match\ttop20_overlap\tkl_ref_ours\tmax_abs\n",
            other.display(),
            ours_path.display()
        ));
        for row in 0..rows {
            let r = &reference[row * vocab..(row + 1) * vocab];
            let o = &ours[row * vocab..(row + 1) * vocab];
            let (rt, ot) = (argmax(r), argmax(o));
            let top_r = top_k(r, 20);
            let top_o = top_k(o, 20);
            let overlap = top_r.iter().filter(|i| top_o.contains(i)).count();
            let kl = kl_divergence(r, o);
            let s = compare(r, o);
            receipt.push_str(&format!(
                "{row}\t{rt}\t{ot}\t{}\t{overlap}/20\t{kl:.5}\t{:.3e}\n",
                rt == ot,
                s.max_abs
            ));
        }
        let path = args.out.join(format!("logits-compare-{}.tsv", args.label));
        std::fs::write(&path, &receipt)?;
        println!("{receipt}\n# logits-compare receipt: {}", path.display());
    }

    // Router-audit receipt line (MEMRA_Q4E_ROUTER_AUDIT=1 + routerdev): every device
    // route in the run was hard-compared against the host twin; rows=0 means the audit
    // never engaged (seam off or no device-routed forward ran) — a run claiming audit
    // coverage must show rows > 0 (loud-failures law: assert outcomes, not liveness).
    {
        let (rows, worst_ulp) = memra_engine::qwen4exp_gpu::route_audit_stats();
        if std::env::var("MEMRA_Q4E_ROUTER_AUDIT").as_deref() == Ok("1") {
            println!("# router-audit\trows={rows}\tworst_w_ulp={worst_ulp}");
        }
    }
    // idxq-audit receipt line (MEMRA_Q4E_IDXQ_AUDIT=1 + idxq=q8/bf16): every scored
    // row's selection recomputed from the f32 twin cache; rows=0 means the audit never
    // engaged (no scored row past the horizon, or idxq=f32) — a flip-rate claim needs
    // rows > 0 (loud-failures law).
    {
        let (rows, flipped, blocks) = memra_engine::qwen4exp_gpu::idxq_audit_stats();
        if std::env::var("MEMRA_Q4E_IDXQ_AUDIT").as_deref() == Ok("1") {
            println!(
                "# idxq-audit\tscored_rows={rows}\tflipped_rows={flipped}\tsymdiff_blocks={blocks}"
            );
        }
    }
    // idxsel-audit receipt line (MEMRA_Q4E_IDXSEL_AUDIT=1 + idxsel): every device indexer
    // selection recomputed on the host from the SAME score slab and hard-compared on ids
    // AND emitted order. rows=0 means the audit never engaged (seam off, or no row past
    // the 2,048-token selection horizon) — a coverage claim needs rows > 0, and
    // `deepest_blocks` is what makes it an AT-DEPTH audit rather than a shallow one
    // (loud-failures law: assert the outcome, not liveness).
    {
        let (rows, mismatched, deepest) = memra_engine::qwen4exp_gpu::idx_sel_audit_stats();
        if std::env::var("MEMRA_Q4E_IDXSEL_AUDIT").as_deref() == Ok("1") {
            println!(
                "# idxsel-audit\trows={rows}\tmismatched={mismatched}\tdeepest_blocks={deepest}"
            );
        }
    }
    // plecache-audit receipt line (MEMRA_Q4E_PLECACHE_AUDIT=1 + plecache): the chunk's cached
    // n-gram ids recomputed from the FULL twin and hard-compared. rows=0 means the audit never
    // engaged; `deepest_fill` is the history length it actually reached, which is what
    // separates an at-depth audit from a shallow one.
    {
        let (rows, mismatched, deepest) = memra_engine::qwen4exp_gpu::ple_cache_audit_stats();
        if std::env::var("MEMRA_Q4E_PLECACHE_AUDIT").as_deref() == Ok("1") {
            println!(
                "# plecache-audit\trows={rows}\tmismatched={mismatched}\tdeepest_fill={deepest}"
            );
        }
    }
    std::io::stdout().flush()?;
    Ok(())
}
