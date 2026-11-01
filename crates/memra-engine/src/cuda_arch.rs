//! CUDA build-target policy, shared by `build.rs` (through `#[path]`) and the engine's
//! startup guard, so the target the fatbins are compiled for and the device check that admits
//! them come from one table.
//!
//! Every build has two names:
//!
//! - The TARGET is the exact `-gencode arch=compute_<t>,code=sm_<t>` value
//!   (`MEMRA_BUILT_CUDA_TARGET`). The fatbins carry arch-specific SASS and no PTX, and `a` SASS
//!   loads only on the one compute capability it names, so the startup guard keys on the target.
//! - The FAMILY is the policy key (`MEMRA_BUILT_CUDA_ARCH`): every kernel choice, stub switch and
//!   arch-keyed default reads it. `121a` (GB10, DGX Spark) compiles the same sources with the same
//!   block-scale MMA kinds as `120a` (RTX 50-series, RTX PRO 6000), so it belongs to the `120a`
//!   family. Only the compiled SASS differs.
//!
//! The accepted target set itself lives in the `matches!` assert in `build.rs`, which
//! `tools/arch-matrix-census.sh` parses.
#![allow(dead_code)] // build.rs uses the mapping half, the engine the guard half.

/// The build target `MEMRA_CUDA_ARCH` auto-detection picks for a `nvidia-smi compute_cap`
/// string. `None` for a capability no target covers; the caller decides the fallback.
pub fn target_for_compute_cap(cap: &str) -> Option<&'static str> {
    match cap.trim() {
        "12.0" => Some("120a"),
        "12.1" => Some("121a"),
        "10.0" => Some("100a"),
        "9.0" => Some("90a"),
        "8.9" => Some("89"),
        _ => None,
    }
}

/// The policy family of a build target. `121a` runs the `120a` program; every other target is
/// its own family.
pub fn family(target: &str) -> &str {
    match target {
        "121a" => "120a",
        t => t,
    }
}

/// Whether a binary built for `target` can load its fatbins on a device of compute capability
/// `major.minor`. Arch-specific SASS is not forward compatible across minor versions: sm_120a
/// SASS does not load on a 12.1 GB10, and sm_121a SASS does not load on a 12.0 card.
pub fn target_runs_on(target: &str, major: i32, minor: i32) -> bool {
    matches!(
        (target, major, minor),
        ("120a", 12, 0) | ("121a", 12, 1) | ("100a", 10, 0) | ("90a", 9, 0) | ("89", 8, 9)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spark_capability_gets_its_own_target_in_the_120a_family() {
        assert_eq!(target_for_compute_cap("12.0"), Some("120a"));
        assert_eq!(target_for_compute_cap("12.1"), Some("121a"));
        assert_eq!(target_for_compute_cap(" 12.1\n"), Some("121a"));
        assert_eq!(target_for_compute_cap("10.0"), Some("100a"));
        assert_eq!(target_for_compute_cap("9.0"), Some("90a"));
        assert_eq!(target_for_compute_cap("8.9"), Some("89"));
        assert_eq!(target_for_compute_cap("10.3"), None);
        assert_eq!(family("121a"), "120a");
        for t in ["120a", "100a", "90a", "89"] {
            assert_eq!(family(t), t);
        }
    }

    #[test]
    fn baked_policy_arch_is_the_family_of_the_baked_target() {
        // build.rs writes both env vars; a drift here means policy reads a different arch than
        // the one the fatbins were compiled for.
        assert_eq!(crate::BUILT_CUDA_ARCH, family(crate::BUILT_CUDA_TARGET));
    }

    #[test]
    fn guard_admits_each_target_only_on_its_own_capability() {
        // The pre-split guard let a 120a build through on 12.1, where its fatbins cannot load.
        assert!(!target_runs_on("120a", 12, 1));
        assert!(target_runs_on("120a", 12, 0));
        assert!(target_runs_on("121a", 12, 1));
        assert!(!target_runs_on("121a", 12, 0));
        assert!(target_runs_on("100a", 10, 0));
        assert!(!target_runs_on("100a", 12, 0));
        assert!(target_runs_on("90a", 9, 0));
        assert!(target_runs_on("89", 8, 9));
        // Every capability auto-detection maps to a target is admitted by that target.
        for cap in ["12.0", "12.1", "10.0", "9.0", "8.9"] {
            let (maj, min) = cap.split_once('.').unwrap();
            let t = target_for_compute_cap(cap).unwrap();
            assert!(
                target_runs_on(t, maj.parse().unwrap(), min.parse().unwrap()),
                "{cap}"
            );
        }
    }
}
