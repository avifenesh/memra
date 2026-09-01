//! CPU affinity for the GPU worker thread — `MEMRA_WORKER_AFFINITY`, default OFF.
//!
//! WHY THIS EXISTS (lane/glm5-host-audit, 2026-09-01; engine-wide, not one family's).
//! Every served family's decode tick runs on the ONE `memra-gpu-worker` thread spawned in
//! `worker::spawn`. On a 96-core / 12-CCD EPYC 9654 that thread was measured unpinned and
//! MIGRATING: a 45 s procfs sample of a live serving boot put it on 11 distinct CPUs across
//! 3 L3 domains, with 5.6 involuntary preemptions/second, while 184 tokio runtime threads
//! shared the same 192 logical CPUs. Each cross-CCX move throws away a 32 MB L3 the decode
//! tick had warm. This flag is the seam that pins it; the arms that price it are in
//! `research/glm53-flash-bringup-20260827/host-audit-20260901/LANE.md`.
//!
//! DEFAULT IS OFF, BY DESIGN, WITH ITS REASONS STATED (house law: new flags are ON or OFF by
//! design and written, never an accident of implementation order):
//!   * A CPU mask is MACHINE-SPECIFIC. The right mask on this EPYC (one 8-core CCX) is wrong
//!     on box12's Core Ultra 9 285K and wrong again on a cpuset-restricted container, and a
//!     capacity-keyed default would be the exact shape of the Q8RP incident: a pin proven on
//!     one card/host class silently applied to another (LAW card-keyed-defaults-need-full-pins).
//!   * Pinning REDUCES the CPU available to the worker. On a box whose cpuset is already narrow,
//!     an unconditional CCX pin is a throughput regression, not a win.
//!   * The mask is INHERITED by threads the worker creates after the call — including the CUDA
//!     driver's helper threads (measured: three otherwise-unnamed threads inherit the worker's
//!     `comm`, so they are its children). Confining the driver's helpers alongside the worker is
//!     the POINT on a big-core box, and a hazard on a small mask. That is a per-host judgement.
//! It becomes a default only with per-host-class receipts, the same bar every other pin carries.
//!
//! NUMERICS: setting a CPU mask cannot change arithmetic — no kernel selection, no reduction
//! order, no tolerance depends on it. It is asserted anyway (24-step byte identity ON vs OFF on
//! two families' suites), because "obviously non-numeric" is how a numeric-class door ships
//! unmeasured.
//!
//! FAILURE IS LOUD AND THE RECEIPT IS A READBACK. A `sched_setaffinity` that succeeds can still
//! leave a NARROWER mask than requested (an outer cpuset clamps it), and one that fails with
//! EPERM leaves the thread exactly where it was. Both look identical to a caller that trusts its
//! own request. So the announce line reports `effective=` from `sched_getaffinity` after the
//! call, never the request, and a mismatch says so in the same line (LAW loud-failures-fail-quietly
//! / ab-arm-identity-not-liveness: an arm must announce what it actually did).

use std::collections::BTreeSet;

/// Where the CCX/CCD map is read from. Injected so the parser is testable without a fixture
/// kernel; production always passes `"/sys/devices/system/cpu"`.
pub const SYSFS_CPU: &str = "/sys/devices/system/cpu";

const ENV_WORKER_AFFINITY: &str = "MEMRA_WORKER_AFFINITY";

/// What the operator asked for, validated but not yet applied.
///
/// `SelfCcx` and `Ccx(_)` stay symbolic until the worker thread applies them: "the CCX I am on"
/// is only meaningful on that thread, and resolving it at parse time (on `main`) would pin the
/// worker to whichever domain the *startup* thread happened to land on — a silently wrong mask
/// that would still announce success.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AffinitySpec {
    /// No syscall is made at all. Byte-for-byte the shipped scheduling behaviour.
    Off,
    /// An explicit logical-CPU list, e.g. `8-15,104-111`.
    Cpus(Vec<usize>),
    /// The L3 domain (CCX) the worker thread is running on when it applies the mask.
    SelfCcx,
    /// The Nth L3 domain in sysfs discovery order, 0-based.
    Ccx(usize),
}

/// One host's L3 (CCX/CCD) map, read from sysfs — never assumed from a core count.
///
/// A hardcoded "8 cores per CCX" rule is wrong on the next SKU and wrong today under a cpuset,
/// so the domains come from `cache/index3/shared_cpu_list` and the flag's `ccx` forms are defined
/// in terms of what the kernel reports.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Topology {
    /// Discovery-ordered L3 domains, each the sorted logical CPUs sharing one L3.
    pub domains: Vec<Vec<usize>>,
    /// Every online CPU sysfs showed us.
    pub online: BTreeSet<usize>,
}

impl Topology {
    /// Read the L3 map. A host with no `index3` (no L3, or a sysfs the container hides) yields
    /// an EMPTY domain list rather than an error: the explicit-cpulist form still works there,
    /// and only the `ccx` forms need domains.
    pub fn read(root: &str) -> Self {
        let mut cpus: Vec<usize> = Vec::new();
        if let Ok(entries) = std::fs::read_dir(root) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if let Some(Ok(idx)) = name.strip_prefix("cpu").map(str::parse::<usize>) {
                    cpus.push(idx);
                }
            }
        }
        cpus.sort_unstable();
        let mut domains: Vec<Vec<usize>> = Vec::new();
        let mut seen: Vec<String> = Vec::new();
        let mut online = BTreeSet::new();
        for cpu in cpus {
            online.insert(cpu);
            let path = format!("{root}/cpu{cpu}/cache/index3/shared_cpu_list");
            let Ok(raw) = std::fs::read_to_string(&path) else {
                continue;
            };
            let key = raw.trim().to_string();
            if key.is_empty() {
                continue;
            }
            if !seen.iter().any(|k| k == &key) {
                let mut members = parse_cpu_list(&key).unwrap_or_default();
                members.sort_unstable();
                seen.push(key);
                domains.push(members);
            }
        }
        Topology { domains, online }
    }

    fn domain_of(&self, cpu: usize) -> Option<usize> {
        self.domains.iter().position(|d| d.contains(&cpu))
    }
}

/// Parse a Linux-style CPU list: `"8-15,104-111"`, `"3"`, `"0-3,7"`. Rejects anything else.
///
/// Strict on purpose. A typo'd mask that parses to something plausible is worse than a refusal:
/// the operator gets a server that reports pinned-arm throughput while running the OFF arm.
pub fn parse_cpu_list(raw: &str) -> Result<Vec<usize>, String> {
    let mut out = BTreeSet::new();
    for part in raw.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            return Err(format!("empty range in cpu list {raw:?}"));
        }
        match part.split_once('-') {
            Some((lo, hi)) => {
                let lo: usize = lo
                    .trim()
                    .parse()
                    .map_err(|_| format!("bad range start {lo:?} in {raw:?}"))?;
                let hi: usize = hi
                    .trim()
                    .parse()
                    .map_err(|_| format!("bad range end {hi:?} in {raw:?}"))?;
                if hi < lo {
                    return Err(format!("descending range {part:?} in {raw:?}"));
                }
                for cpu in lo..=hi {
                    out.insert(cpu);
                }
            }
            None => {
                let cpu: usize = part
                    .parse()
                    .map_err(|_| format!("bad cpu {part:?} in {raw:?}"))?;
                out.insert(cpu);
            }
        }
    }
    if out.is_empty() {
        return Err(format!("cpu list {raw:?} selects no cpus"));
    }
    Ok(out.into_iter().collect())
}

/// Render a CPU set back to a compact Linux-style list, for the announce line.
pub fn render_cpu_list(cpus: &[usize]) -> String {
    let mut out = String::new();
    let mut idx = 0;
    while idx < cpus.len() {
        let start = cpus[idx];
        let mut end = start;
        while idx + 1 < cpus.len() && cpus[idx + 1] == end + 1 {
            idx += 1;
            end = cpus[idx];
        }
        if !out.is_empty() {
            out.push(',');
        }
        if start == end {
            out.push_str(&start.to_string());
        } else {
            out.push_str(&format!("{start}-{end}"));
        }
        idx += 1;
    }
    out
}

/// Validate the operator's value against this host. Returns `Off` when the flag is unset or
/// explicitly disabled; `Err` when it is set to something this host cannot honour.
///
/// A MALFORMED OR UNSATISFIABLE VALUE IS A STARTUP ERROR, NOT A WARNING. The alternative — accept
/// it, log a line, serve unpinned — is the exception-that-absorbs-the-regression shape: an
/// operator who mistyped `MEMRA_WORKER_AFFINITY=8-15,1O4-111` would get an unpinned server whose
/// receipts are filed under the pinned arm.
pub fn parse_affinity(raw: Option<&str>, topo: &Topology) -> Result<AffinitySpec, String> {
    let Some(raw) = raw else {
        return Ok(AffinitySpec::Off);
    };
    let value = raw.trim();
    match value.to_ascii_lowercase().as_str() {
        "" | "0" | "off" | "false" | "no" => return Ok(AffinitySpec::Off),
        "ccx" | "1" | "on" | "self" => return Ok(AffinitySpec::SelfCcx),
        _ => {}
    }
    if let Some(n) = value.to_ascii_lowercase().strip_prefix("ccx:") {
        let idx: usize = n.trim().parse().map_err(|_| {
            format!("{ENV_WORKER_AFFINITY}={value:?}: 'ccx:N' needs a number, got {n:?}")
        })?;
        if topo.domains.is_empty() {
            return Err(format!(
                "{ENV_WORKER_AFFINITY}={value:?}: this host exposes no L3 (index3) map in \
                 {SYSFS_CPU}, so 'ccx' forms cannot be resolved — use an explicit cpu list"
            ));
        }
        if idx >= topo.domains.len() {
            return Err(format!(
                "{ENV_WORKER_AFFINITY}={value:?}: this host has {} L3 domain(s) (0..{}), \
                 so domain {idx} does not exist",
                topo.domains.len(),
                topo.domains.len() - 1
            ));
        }
        return Ok(AffinitySpec::Ccx(idx));
    }
    let cpus = parse_cpu_list(value).map_err(|why| format!("{ENV_WORKER_AFFINITY}: {why}"))?;
    if !topo.online.is_empty() {
        let missing: Vec<usize> = cpus
            .iter()
            .copied()
            .filter(|c| !topo.online.contains(c))
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "{ENV_WORKER_AFFINITY}={value:?}: cpu(s) {} are not present on this host \
                 (sysfs shows {} cpus)",
                render_cpu_list(&missing),
                topo.online.len()
            ));
        }
    }
    Ok(AffinitySpec::Cpus(cpus))
}

/// Read + validate the flag from the environment. Called ONCE, on the startup thread, so a bad
/// value fails the boot instead of being discovered in a receipt.
pub fn worker_affinity_spec() -> Result<AffinitySpec, String> {
    let raw = std::env::var(ENV_WORKER_AFFINITY).ok();
    let topo = Topology::read(SYSFS_CPU);
    parse_affinity(raw.as_deref(), &topo)
}

/// The mask actually installed, as the kernel reports it back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Applied {
    pub requested: Vec<usize>,
    pub effective: Vec<usize>,
    pub l3_domains: usize,
}

/// Apply the spec TO THE CALLING THREAD and return the readback. Never panics; a refusal is an
/// `Err` the caller logs loudly and continues serving on (an unpinned server serves; a dead one
/// does not).
///
/// `Off` returns `Ok(None)` WITHOUT making any syscall — the OFF arm must be the shipped bytes
/// and the shipped syscall trace, not "the same mask, set explicitly".
#[cfg(target_os = "linux")]
pub fn apply(spec: &AffinitySpec) -> Result<Option<Applied>, String> {
    let topo = Topology::read(SYSFS_CPU);
    let requested: Vec<usize> = match spec {
        AffinitySpec::Off => return Ok(None),
        AffinitySpec::Cpus(cpus) => cpus.clone(),
        AffinitySpec::Ccx(idx) => topo
            .domains
            .get(*idx)
            .cloned()
            .ok_or_else(|| format!("L3 domain {idx} vanished between validation and apply"))?,
        AffinitySpec::SelfCcx => {
            let cpu = current_cpu()?;
            let idx = topo.domain_of(cpu).ok_or_else(|| {
                format!("cpu {cpu} is in no L3 domain sysfs reports — cannot resolve 'ccx'")
            })?;
            topo.domains[idx].clone()
        }
    };

    // SAFETY: `set` is a zeroed cpu_set_t we own for the duration of the call; CPU_SET only
    // writes inside it, and every index is bounds-checked against CPU_SETSIZE first. The pid
    // argument 0 means "the calling thread" for sched_setaffinity (thread, not process — that
    // distinction is the whole point: pinning the process would drag 184 tokio threads with it).
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &cpu in &requested {
            if cpu >= libc::CPU_SETSIZE as usize {
                return Err(format!(
                    "cpu {cpu} is beyond CPU_SETSIZE ({})",
                    libc::CPU_SETSIZE
                ));
            }
            libc::CPU_SET(cpu, &mut set);
        }
        if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
            let err = std::io::Error::last_os_error();
            return Err(format!(
                "sched_setaffinity({}) failed: {err} — the thread stays where it was, \
                 so this boot is an OFF-arm boot and must not be filed as a pinned row",
                render_cpu_list(&requested)
            ));
        }
    }

    // READ BACK. A success return does not mean the mask is what we asked for: an outer cpuset
    // narrows it silently, and that narrower mask is what the receipts must carry.
    let effective = read_effective()?;
    let l3_domains = {
        let mut doms = BTreeSet::new();
        for &cpu in &effective {
            if let Some(d) = topo.domain_of(cpu) {
                doms.insert(d);
            }
        }
        doms.len()
    };
    Ok(Some(Applied {
        requested,
        effective,
        l3_domains,
    }))
}

#[cfg(not(target_os = "linux"))]
pub fn apply(spec: &AffinitySpec) -> Result<Option<Applied>, String> {
    match spec {
        AffinitySpec::Off => Ok(None),
        _ => Err("CPU affinity is only implemented on Linux".to_string()),
    }
}

#[cfg(target_os = "linux")]
fn read_effective() -> Result<Vec<usize>, String> {
    // SAFETY: same contract as above; `set` is ours, and CPU_ISSET only reads it.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return Err(format!(
                "sched_getaffinity failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        let mut out = Vec::new();
        for cpu in 0..libc::CPU_SETSIZE as usize {
            if libc::CPU_ISSET(cpu, &set) {
                out.push(cpu);
            }
        }
        Ok(out)
    }
}

#[cfg(target_os = "linux")]
fn current_cpu() -> Result<usize, String> {
    // SAFETY: sched_getcpu takes no arguments and only returns an int.
    let cpu = unsafe { libc::sched_getcpu() };
    if cpu < 0 {
        return Err(format!(
            "sched_getcpu failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(cpu as usize)
}

/// Apply on the worker thread and print the ONE announce line the A/B gate greps for.
///
/// BOTH ARMS ANNOUNCE, following the `[gpu-watch] disabled (MEMRA_GPU_WATCH=0)` precedent in
/// `health.rs`. An arm whose identity is "the absence of a line" is indistinguishable from an
/// arm whose announce regressed (LAW ab-arm-identity-not-liveness, and the sibling law that
/// the diagnostic written to end a silent failure is itself silent). So the OFF arm prints
/// `off`, the ON arm prints the kernel readback, and a REFUSAL prints what the boot now IS —
/// three states, three distinguishable lines, none of them inferred from silence.
pub fn apply_and_announce(spec: &AffinitySpec) {
    match apply(spec) {
        Ok(None) => {
            eprintln!("[worker-affinity] off (MEMRA_WORKER_AFFINITY unset or =0)");
        }
        Ok(Some(applied)) => {
            let req = render_cpu_list(&applied.requested);
            let eff = render_cpu_list(&applied.effective);
            let clamped = if applied.requested == applied.effective {
                ""
            } else {
                " CLAMPED-BY-OUTER-CPUSET"
            };
            eprintln!(
                "[worker-affinity] engaged request={req} effective={eff} cpus={} \
                 l3_domains={}{clamped}",
                applied.effective.len(),
                applied.l3_domains
            );
        }
        Err(why) => {
            // Loud, and it says what the boot now IS rather than only what failed.
            eprintln!("[worker-affinity] REFUSED — {why}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn epyc_9654() -> Topology {
        // The real Box B map, read from sysfs 2026-08-31: 12 L3 domains, each 8 cores plus
        // their 8 SMT siblings (`0-7,96-103`, `8-15,104-111`, ...), one NUMA node.
        let domains = (0..12)
            .map(|k| {
                let mut v: Vec<usize> = (8 * k..8 * k + 8).collect();
                v.extend(96 + 8 * k..96 + 8 * k + 8);
                v
            })
            .collect::<Vec<_>>();
        Topology {
            domains,
            online: (0..192).collect(),
        }
    }

    #[test]
    fn unset_and_disabled_values_make_no_syscall() {
        let t = epyc_9654();
        for raw in [
            None,
            Some(""),
            Some("0"),
            Some("off"),
            Some("  OFF "),
            Some("no"),
        ] {
            assert_eq!(
                parse_affinity(raw, &t).unwrap(),
                AffinitySpec::Off,
                "{raw:?} must be OFF — the default is OFF by design"
            );
        }
        // And OFF really does skip the syscall: apply returns None, not an installed mask.
        assert_eq!(apply(&AffinitySpec::Off).unwrap(), None);
    }

    #[test]
    fn ccx_forms_resolve_against_the_hosts_own_l3_map() {
        let t = epyc_9654();
        assert_eq!(
            parse_affinity(Some("ccx"), &t).unwrap(),
            AffinitySpec::SelfCcx
        );
        assert_eq!(
            parse_affinity(Some("on"), &t).unwrap(),
            AffinitySpec::SelfCcx
        );
        assert_eq!(
            parse_affinity(Some("ccx:3"), &t).unwrap(),
            AffinitySpec::Ccx(3)
        );
        // Past the end of THIS host's map is a refusal, not a clamp: a clamp would pin to a
        // domain the operator did not name and announce success.
        let err = parse_affinity(Some("ccx:12"), &t).unwrap_err();
        assert!(err.contains("12 L3 domain(s)"), "{err}");
        assert!(err.contains("does not exist"), "{err}");
    }

    #[test]
    fn ccx_is_refused_when_sysfs_shows_no_l3_map() {
        // A container that hides the cache tree must not silently fall back to "whole host".
        let blind = Topology {
            domains: vec![],
            online: (0..8).collect(),
        };
        let err = parse_affinity(Some("ccx:0"), &blind).unwrap_err();
        assert!(err.contains("no L3"), "{err}");
        assert!(err.contains("explicit cpu list"), "{err}");
    }

    #[test]
    fn explicit_cpu_lists_parse_and_round_trip() {
        assert_eq!(parse_cpu_list("3").unwrap(), vec![3]);
        assert_eq!(parse_cpu_list("0-3,7").unwrap(), vec![0, 1, 2, 3, 7]);
        // Overlapping and unsorted input normalises; the announce line stays canonical.
        assert_eq!(parse_cpu_list("7,0-3,2").unwrap(), vec![0, 1, 2, 3, 7]);
        assert_eq!(render_cpu_list(&[8, 9, 10, 11, 12, 13, 14, 15]), "8-15");
        assert_eq!(
            render_cpu_list(&parse_cpu_list("8-15,104-111").unwrap()),
            "8-15,104-111"
        );
        assert_eq!(render_cpu_list(&[1, 3, 4, 5, 9]), "1,3-5,9");
        assert_eq!(render_cpu_list(&[]), "");
    }

    #[test]
    fn malformed_values_are_startup_errors_never_a_silent_off() {
        let t = epyc_9654();
        // The motivating typo: a letter O for a zero. It must NOT parse, and it must NOT
        // degrade to OFF — that is how a pinned-arm receipt gets written by an unpinned boot.
        for bad in ["8-15,1O4-111", "ccx:x", "15-8", "", ","] {
            if bad.is_empty() {
                continue; // "" is the documented OFF spelling, covered above
            }
            let got = parse_affinity(Some(bad), &t);
            assert!(got.is_err(), "{bad:?} must be refused, got {got:?}");
        }
    }

    #[test]
    fn cpus_absent_from_this_host_are_refused() {
        let t = epyc_9654();
        let err = parse_affinity(Some("190-200"), &t).unwrap_err();
        assert!(err.contains("not present on this host"), "{err}");
        assert!(err.contains("192 cpus"), "{err}");
    }

    /// The readback contract, exercised for real: pin this test thread to the CPUs the kernel
    /// already allows it, and assert the announce data comes from sched_getaffinity rather than
    /// from the request. Uses the CURRENT mask so it passes inside any cpuset.
    #[cfg(target_os = "linux")]
    #[test]
    fn apply_reports_the_kernel_readback_not_the_request() {
        let current = read_effective().expect("sched_getaffinity must work");
        assert!(!current.is_empty());
        let applied = apply(&AffinitySpec::Cpus(current.clone()))
            .expect("pinning to the mask we already have cannot fail")
            .expect("a non-Off spec installs a mask");
        assert_eq!(
            applied.effective, current,
            "effective comes from the kernel"
        );
        assert_eq!(applied.requested, current);
    }

    /// `ccx` resolves on the CALLING thread — the reason the spec stays symbolic until apply.
    #[cfg(target_os = "linux")]
    #[test]
    fn self_ccx_resolves_on_the_calling_thread_or_says_why_not() {
        match apply(&AffinitySpec::SelfCcx) {
            Ok(Some(applied)) => {
                assert!(!applied.effective.is_empty());
                // One CCX is one L3 domain by construction, unless an outer cpuset cut it.
                assert!(applied.l3_domains <= 1 || applied.requested != applied.effective);
            }
            // A host with no index3 in sysfs, or a cpuset that forbids the widen: both are
            // legitimate here and must be reported, not panicked on.
            Ok(None) => panic!("SelfCcx must never be a silent no-op"),
            Err(why) => assert!(
                why.contains("L3") || why.contains("sched_") || why.contains("cpu"),
                "an unexplained refusal is the failure mode this lane exists to remove: {why}"
            ),
        }
    }
}
