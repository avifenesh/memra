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
    let available = super::meminfo_available_bytes(&read("/proc/meminfo")?)
        .ok_or("startup pinned arena headroom: missing MemAvailable")?;
    require(budget, available, "MemAvailable")?;
    let groups = read("/proc/self/cgroup")?;
    let mounts = read("/proc/self/mountinfo")?;
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
        let relative = group
            .strip_prefix(&root)
            .map_err(|e| format!("cgroup membership outside visible mount: {e}"))?;
        let mut path = mount.join(relative);
        loop {
            let limit_path = path.join(if v2 {
                "memory.max"
            } else {
                "memory.limit_in_bytes"
            });
            // Hybrid systems can expose a v2 hierarchy without a memory controller.
            if v1 || limit_path.exists() {
                let limit_text = read(&limit_path)?;
                let usage = number(&read(path.join(if v2 {
                    "memory.current"
                } else {
                    "memory.usage_in_bytes"
                }))?)?;
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
}
