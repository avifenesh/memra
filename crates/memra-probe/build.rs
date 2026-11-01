// Phase-0 spine: compile a hand-written .cu kernel to an sm_120a (or sm_121a) fatbin via nvcc at
// build time.
// Proves the cargo -> build.rs -> nvcc -> fatbin -> cudarc-load path that the whole engine rests on.
use std::path::PathBuf;
use std::process::Command;

/// Same resolution policy as `memra-engine/build.rs::resolve_nvcc`, in compact form:
/// explicit intent (`MEMRA_NVCC`, then `CUDA_HOME`/`CUDA_PATH`/`CUDA_ROOT`) wins
/// unvalidated; otherwise every candidate on `PATH` + `/usr/local/cuda*` is asked its
/// release and the NEWEST wins. Newest-wins is not cosmetic: the 5090 dev rig carries a
/// distro `/usr/bin/nvcc` at CUDA 12.4 that predates sm_120 and would fail
/// `compute_120a` outright if a first-hit-on-PATH resolver picked it.
///
/// Deliberately duplicated rather than shared — a `build-dependencies` helper crate would
/// have to be published alongside memra-engine, and `include!("../..")` crosses the crate
/// directory boundary `cargo package` enforces.
fn resolve_nvcc() -> String {
    println!("cargo:rerun-if-env-changed=MEMRA_NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    if let Ok(p) = std::env::var("MEMRA_NVCC") {
        return p;
    }
    for var in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"] {
        if let Ok(root) = std::env::var(var) {
            let cand = PathBuf::from(root).join("bin/nvcc");
            if cand.is_file() {
                return cand.to_string_lossy().into_owned();
            }
        }
    }
    let ver = |p: &std::path::Path| -> Option<(u32, u32)> {
        let out = Command::new(p).arg("--version").output().ok()?;
        if !out.status.success() {
            return None;
        }
        let s = String::from_utf8_lossy(&out.stdout);
        let rel = s
            .split("release ")
            .nth(1)?
            .split(',')
            .next()?
            .trim()
            .to_string();
        let mut it = rel.split('.');
        let maj = it.next()?.parse::<u32>().ok()?;
        Some((
            maj,
            it.next().and_then(|m| m.parse::<u32>().ok()).unwrap_or(0),
        ))
    };
    let mut cands: Vec<PathBuf> = Vec::new();
    if let Ok(path) = std::env::var("PATH") {
        cands.extend(
            path.split(':')
                .filter(|d| !d.is_empty())
                .map(|d| PathBuf::from(d).join("nvcc")),
        );
    }
    cands.push(PathBuf::from("/usr/local/cuda/bin/nvcc"));
    if let Ok(rd) = std::fs::read_dir("/usr/local") {
        for ent in rd.flatten() {
            if ent.file_name().to_string_lossy().starts_with("cuda-") {
                cands.push(ent.path().join("bin/nvcc"));
            }
        }
    }
    let mut ranked: Vec<((u32, u32), PathBuf)> = Vec::new();
    let mut seen: Vec<PathBuf> = Vec::new();
    for c in cands.into_iter().filter(|c| c.is_file()) {
        let abs = c.canonicalize().unwrap_or(c);
        if seen.contains(&abs) {
            continue;
        }
        seen.push(abs.clone());
        if let Some(v) = ver(&abs) {
            ranked.push((v, abs));
        }
    }
    ranked.sort_by_key(|r| std::cmp::Reverse(r.0));
    match ranked.first() {
        Some(((maj, min), p)) => {
            println!("cargo:warning=nvcc CUDA {maj}.{min} at {}", p.display());
            p.to_string_lossy().into_owned()
        }
        None => "/usr/local/cuda-13.1/bin/nvcc".to_string(),
    }
}

/// The probe's SASS target, same contract as memra-engine's `MEMRA_CUDA_ARCH`
/// (`src/cuda_arch.rs` there, duplicated for the reason `resolve_nvcc` gives): explicit
/// `MEMRA_CUDA_ARCH` wins, else `nvidia-smi` compute capability 12.1 (GB10, DGX Spark) builds
/// 121a and everything else 120a. The probe's kernels are sm_120-family only, so only those two
/// targets are accepted. Arch-specific SASS loads on one capability, which is why 12.1 cannot
/// reuse the 120a fatbin.
fn probe_target() -> String {
    println!("cargo:rerun-if-env-changed=MEMRA_CUDA_ARCH");
    let target = std::env::var("MEMRA_CUDA_ARCH").unwrap_or_else(|_| {
        let cap = Command::new("nvidia-smi")
            .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.lines().next().map(|l| l.trim().to_string()));
        match cap.as_deref() {
            Some("12.1") => "121a",
            _ => "120a",
        }
        .to_string()
    });
    assert!(
        matches!(target.as_str(), "120a" | "121a"),
        "memra-probe builds only the sm_120 family: MEMRA_CUDA_ARCH must be 120a or 121a"
    );
    target
}

fn main() {
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let cu = "src/kernels.cu";
    println!("cargo:rerun-if-changed={cu}");

    let fatbin = out.join("kernels.fatbin");
    // CRITICAL: -gencode arch=compute_120a,code=sm_120a (or the 121a twin) is the only form that
    // assembles FP4/FP8 block-scale mma on sm_120 (the bare -arch=sm_120a shortcut misroutes to
    // compute_120).
    let target = probe_target();
    let gencode = format!("arch=compute_{target},code=sm_{target}");
    let nvcc = resolve_nvcc();
    let status = Command::new(&nvcc)
        .args([
            "-gencode",
            &gencode,
            "-O3",
            "--fatbin",
            "-o",
            fatbin.to_str().unwrap(),
            cu,
        ])
        .status()
        .expect("failed to spawn nvcc");
    assert!(status.success(), "nvcc fatbin build failed");
    println!("cargo:rustc-env=MEMRA_FATBIN={}", fatbin.display());
    println!("cargo:rustc-env=MEMRA_PROBE_CUDA_TARGET={target}");
}
