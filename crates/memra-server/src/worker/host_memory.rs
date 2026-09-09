//! Fresh host/cgroup headroom admission immediately before startup pinning.
use std::path::{Path, PathBuf};

const MARGIN: usize = 32 << 30;

fn read(path: impl AsRef<Path>) -> Result<String, String> {
    let path = path.as_ref();
    std::fs::read_to_string(path)
        .map_err(|e| format!("host arena headroom {}: {e}", path.display()))
}

fn number(text: &str) -> Result<usize, String> {
    text.trim()
        .parse()
        .map_err(|e| format!("invalid cgroup memory value {text:?}: {e}"))
}

fn require(budget: usize, available: usize, detail: &str) -> Result<(), String> {
    if budget
        .checked_add(MARGIN)
        .is_none_or(|need| need > available)
    {
        return Err(format!(
            "startup pinned arena headroom refused: budget={budget} margin={MARGIN} available={available} {detail}"
        ));
    }
    Ok(())
}

fn unescape(path: &str) -> PathBuf {
    PathBuf::from(
        path.replace("\\040", " ")
            .replace("\\011", "\t")
            .replace("\\012", "\n")
            .replace("\\134", "\\"),
    )
}

pub(super) fn check_headroom(budget: usize) -> Result<(), String> {
    check_headroom_from_contents(
        budget,
        &read("/proc/meminfo")?,
        &read("/proc/self/cgroup")?,
        &read("/proc/self/mountinfo")?,
        |path| match std::fs::read_to_string(path) {
            Ok(text) => Ok(Some(text)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(format!("host arena headroom {}: {e}", path.display())),
        },
    )
}

// The admission unit consumes proc contents and an injected cgroup-file source;
// fixtures exercise exactly the runtime parser without touching /proc or sysfs.
fn check_headroom_from_contents(
    budget: usize,
    meminfo: &str,
    groups: &str,
    mounts: &str,
    mut file: impl FnMut(&Path) -> Result<Option<String>, String>,
) -> Result<(), String> {
    let available = super::meminfo_available_bytes(meminfo)
        .ok_or("startup pinned arena headroom: missing MemAvailable")?;
    require(budget, available, "MemAvailable")?;
    let mut checked = false;
    for line in mounts.lines() {
        let Some((left, right)) = line.split_once(" - ") else {
            continue;
        };
        let right: Vec<_> = right.split_whitespace().collect();
        let v2 = right.first() == Some(&"cgroup2");
        let v1 = right.first() == Some(&"cgroup")
            && right
                .get(2)
                .is_some_and(|s| s.split(',').any(|s| s == "memory"));
        if !v1 && !v2 {
            continue;
        }
        let group = groups
            .lines()
            .find_map(|l| {
                let mut fields = l.splitn(3, ':');
                fields.next()?;
                let controllers = fields.next()?;
                let path = fields.next()?;
                ((v2 && controllers.is_empty())
                    || (v1 && controllers.split(',').any(|c| c == "memory")))
                .then_some(path)
            })
            .ok_or("startup pinned arena headroom: memory cgroup membership missing")?;
        let left: Vec<_> = left.split_whitespace().collect();
        if left.len() < 5 {
            return Err("invalid cgroup mountinfo".into());
        }
        let root = unescape(left[3]);
        let mount = unescape(left[4]);
        let group = Path::new(group);
        // A cgroup namespace reports its root as / even when mountinfo still
        // names the host-side subtree. Both refer to this visible mount root.
        let relative = if group == Path::new("/") {
            Path::new("")
        } else {
            group
                .strip_prefix(&root)
                .map_err(|e| format!("cgroup membership outside visible mount: {e}"))?
        };
        let mut path = mount.join(relative);
        loop {
            let limit_path = path.join(if v2 {
                "memory.max"
            } else {
                "memory.limit_in_bytes"
            });
            if let Some(limit_text) = file(&limit_path)? {
                let usage_path = path.join(if v2 {
                    "memory.current"
                } else {
                    "memory.usage_in_bytes"
                });
                let usage = number(&file(&usage_path)?.ok_or_else(|| {
                    format!("host arena headroom: missing {}", usage_path.display())
                })?)?;
                let limit = if limit_text.trim() == "max" {
                    usize::MAX
                } else {
                    number(&limit_text)?
                };
                let remaining = limit.saturating_sub(usage);
                let detail = format!(
                    "cgroup={} limit={limit} usage={usage} MemAvailable={available}",
                    path.display()
                );
                require(budget, remaining, &detail)?;
                eprintln!(
                    "[prefix-host DEBUG] arena headroom: budget={budget} margin={MARGIN} {detail}"
                );
                checked = true;
            } else if v2 && path == mount && (root == Path::new("/") || group == Path::new("/")) {
                // The unified hierarchy root has no memory.max (and may have
                // no memory.current). It is unlimited; MemAvailable gates it.
                checked = true;
            } else if v2 {
                let controllers = file(&path.join("cgroup.controllers"))?
                    .ok_or("host arena headroom: missing cgroup.controllers")?;
                if controllers.split_whitespace().any(|c| c == "memory") {
                    return Err(format!(
                        "host arena headroom: missing {}",
                        limit_path.display()
                    ));
                }
                // Hybrid v2 hierarchies may have no memory controller. Keep
                // walking ancestors and the separate v1 memory mount, if any.
                checked = true;
            } else {
                return Err(format!(
                    "host arena headroom: missing {}",
                    limit_path.display()
                ));
            }
            if path == mount {
                break;
            }
            if !path.pop() || !path.starts_with(&mount) {
                return Err("invalid cgroup ancestor path".into());
            }
        }
    }
    if !checked {
        return Err("startup pinned arena headroom: no readable memory cgroup limit/usage".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn host_arena_headroom_requires_full_budget_and_margin() {
        assert!(require(4096, MARGIN + 4096, "fixture").is_ok());
        assert!(
            require(4096, MARGIN + 4095, "fixture")
                .unwrap_err()
                .contains("budget=4096")
        );
        assert!(require(usize::MAX, usize::MAX, "fixture").is_err());
    }

    const MEMINFO: &str = "MemAvailable: 1786991172 kB\n";
    const V2: &str = "31 20 0:28 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n";
    const V1: &str = "32 20 0:29 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n";

    fn fixture(
        budget: usize,
        groups: &str,
        mounts: &str,
        files: &[(&str, &str)],
    ) -> Result<(), String> {
        check_headroom_from_contents(budget, MEMINFO, groups, mounts, |path| {
            Ok(files
                .iter()
                .find(|(name, _)| Path::new(name) == path)
                .map(|(_, text)| (*text).into()))
        })
    }

    #[test]
    fn host_arena_v2_root_without_limit_or_usage() {
        assert!(fixture(288 << 30, "0::/\n", V2, &[]).is_ok());
        assert!(
            fixture(1786991172 * 1024 - MARGIN + 1, "0::/\n", V2, &[])
                .unwrap_err()
                .contains("MemAvailable")
        );
    }

    #[test]
    fn host_arena_v2_namespace_root_reads_mount_root() {
        let mounts = "31 20 0:28 /docker/id /sys/fs/cgroup rw - cgroup2 cgroup rw\n";
        let files = [
            ("/sys/fs/cgroup/memory.max", "68719476736"),
            ("/sys/fs/cgroup/memory.current", "0"),
        ];
        assert!(fixture(1 << 30, "0::/\n", mounts, &files).is_ok());
        assert!(fixture(33 << 30, "0::/\n", mounts, &files).is_err());
        assert!(fixture(288 << 30, "0::/\n", mounts, &[]).is_ok());
    }

    #[test]
    fn host_arena_v2_nested_numeric_and_max() {
        for limit in ["1028733796352", "max"] {
            let files = [
                ("/sys/fs/cgroup/docker/id/memory.max", limit),
                ("/sys/fs/cgroup/docker/id/memory.current", "1073741824"),
                ("/sys/fs/cgroup/docker/memory.max", "max"),
                ("/sys/fs/cgroup/docker/memory.current", "1073741824"),
            ];
            assert!(fixture(288 << 30, "0::/docker/id\n", V2, &files).is_ok());
        }
    }

    #[test]
    fn host_arena_v2_nested_enforces_ancestor_and_requires_usage() {
        let files = [
            ("/sys/fs/cgroup/docker/id/memory.max", "max"),
            ("/sys/fs/cgroup/docker/id/memory.current", "0"),
            ("/sys/fs/cgroup/docker/memory.max", "68719476736"),
            ("/sys/fs/cgroup/docker/memory.current", "1073741824"),
        ];
        assert!(
            fixture(32 << 30, "0::/docker/id\n", V2, &files)
                .unwrap_err()
                .contains("cgroup=/sys/fs/cgroup/docker")
        );
        assert!(
            fixture(1 << 30, "0::/docker/id\n", V2, &files[..1])
                .unwrap_err()
                .contains("memory.current")
        );
    }

    #[test]
    fn host_arena_v1_docker_serving_shape() {
        let mounts = "32 20 0:29 /docker/id /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n";
        let files = [
            (
                "/sys/fs/cgroup/memory/memory.limit_in_bytes",
                "1028733796352",
            ),
            ("/sys/fs/cgroup/memory/memory.usage_in_bytes", "1073741824"),
        ];
        assert_eq!(
            super::super::meminfo_available_bytes(MEMINFO),
            Some(1829878960128)
        );
        assert!(fixture(288 << 30, "5:memory:/docker/id\n", mounts, &files).is_ok());
        assert!(
            fixture(
                1028733796352 - 1073741824 - MARGIN + 1,
                "5:memory:/docker/id\n",
                mounts,
                &files
            )
            .is_err()
        );
    }

    #[test]
    fn host_arena_hybrid_v2_without_memory_controller() {
        let files = [
            ("/sys/fs/cgroup/docker/id/cgroup.controllers", "cpu io pids"),
            ("/sys/fs/cgroup/docker/cgroup.controllers", "cpu io pids"),
            (
                "/sys/fs/cgroup/memory/docker/id/memory.limit_in_bytes",
                "1028733796352",
            ),
            ("/sys/fs/cgroup/memory/docker/id/memory.usage_in_bytes", "0"),
            (
                "/sys/fs/cgroup/memory/docker/memory.limit_in_bytes",
                "1028733796352",
            ),
            ("/sys/fs/cgroup/memory/docker/memory.usage_in_bytes", "0"),
            (
                "/sys/fs/cgroup/memory/memory.limit_in_bytes",
                "1028733796352",
            ),
            ("/sys/fs/cgroup/memory/memory.usage_in_bytes", "0"),
        ];
        let mounts = format!("{V2}{V1}");
        let groups = "0::/docker/id\n5:memory:/docker/id\n";
        assert!(fixture(288 << 30, groups, &mounts, &files).is_ok());
        assert!(fixture(1028733796352 - MARGIN + 1, groups, &mounts, &files).is_err());
        assert!(fixture(288 << 30, "0::/docker/id\n", V2, &files[..2]).is_ok());
    }
}
