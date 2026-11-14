#!/usr/bin/env python3
"""memra#1307 diagnostic lines (stderr only; no decision reads them), applied to both cell arms'
trees so the arms differ only in the change under test:

- [diag1307] dspark-pool: one line per pooled DFlash2 verify forward the lockstep plans (a
  non-deferred lead round): live rounds, members taken, rows, the headroom the width compared,
  the raw driver free and MemAvailable at that moment, the pool's cached bytes, whether the
  memory cap stopped the walk (with the need it refused), what the members taken need beyond one
  serial round (extra_kib, the figure the width compares), and one checkpoint column's bytes.
- [diag1307] draft-bracket: one line per draft-graph capture bracket that captured: the raw driver
  free, MemAvailable and pool-cached bytes at its entry and end, and both deltas (raw, and
  MemAvailable, and free memory plus page cache plus reclaimable slab, the head's delta reading),
  in KiB, signed.

- with --deep (arm D only, never an arm whose rows are compared): [diag1307] bracket-dump, for the
  first four brackets that captured, at the entry and the end: this process's device memory as the
  driver accounts it (`nvidia-smi --query-compute-apps`, MiB), the /proc/meminfo fields a reading
  could use and the process's own resident memory. The nvidia-smi call runs inside the production
  bracket, so arm D's own high-water line is not evidence; its draft-bracket deltas are taken
  inside the dump calls and are.

usage: diag1307.py TREE [--deep]   (TREE/crates/memra-engine/src/{dflash,spec}.rs are patched in place)
"""
import sys
from pathlib import Path

tree = Path(sys.argv[1])
deep = "--deep" in sys.argv[2:]
DEEP_FN = r'''        static DIAG1307_PRINTED: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        let diag1307_dump = || -> String {
            if DIAG1307_PRINTED.load(std::sync::atomic::Ordering::Relaxed) >= 4 {
                return String::new();
            }
            let pid = std::process::id().to_string();
            let smi = std::process::Command::new("nvidia-smi")
                .args(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .and_then(|t| {
                    t.lines()
                        .find(|l| l.split(',').next().map(str::trim) == Some(pid.as_str()))
                        .map(|l| l.split(',').nth(1).unwrap_or("").trim().to_string())
                })
                .unwrap_or_else(|| "?".to_string());
            let mut s = format!("smi_used_mib={smi}");
            let pick = |text: &str, keys: &[&str], s: &mut String| {
                for l in text.lines() {
                    if keys.iter().any(|k| l.starts_with(k)) {
                        let mut it = l.split_whitespace();
                        s.push_str(&format!(" {}{}", it.next().unwrap_or(""), it.next().unwrap_or("")));
                    }
                }
            };
            pick(
                &std::fs::read_to_string("/proc/meminfo").unwrap_or_default(),
                &[
                    "MemFree:", "MemAvailable:", "Buffers:", "Cached:", "SwapCached:", "Active(anon):",
                    "Inactive(anon):", "Active(file):", "Inactive(file):", "Unevictable:", "SwapFree:",
                    "Dirty:", "Writeback:", "AnonPages:", "Mapped:", "Shmem:", "KReclaimable:",
                    "SReclaimable:", "SUnreclaim:", "KernelStack:", "PageTables:",
                ],
                &mut s,
            );
            pick(
                &std::fs::read_to_string("/proc/self/status").unwrap_or_default(),
                &["VmRSS:", "RssAnon:", "RssFile:", "RssShmem:"],
                &mut s,
            );
            s
        };
        let diag1307_d0 = diag1307_dump();
'''
DEEP_END = r'''            if !diag1307_d0.is_empty() {
                eprintln!("[diag1307] bracket-dump entry {diag1307_d0}");
                eprintln!("[diag1307] bracket-dump end {}", diag1307_dump());
                DIAG1307_PRINTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
'''
dfl = tree / "crates/memra-engine/src/dflash.rs"
spec = tree / "crates/memra-engine/src/spec.rs"


def sub(text, old, new, what):
    n = text.count(old)
    if n != 1:
        sys.exit(f"diag1307: anchor {what!r} found {n} times")
    return text.replace(old, new)


s = dfl.read_text()
if "[diag1307]" not in s:
    s = sub(
        s,
        "                    let headroom = crate::spec::pooled_verify_headroom(e);\n",
        "                    let headroom = crate::spec::pooled_verify_headroom(e);\n"
        "                    let mut diag_capped: Option<usize> = None;\n",
        "headroom",
    )
    s = sub(
        s,
        "                        if s2 - x2 > headroom {\n",
        "                        if s2 - x2 > headroom {\n"
        "                            diag_capped = Some(s2 - x2);\n",
        "cap branch",
    )
    s = sub(
        s,
        "                        (rows, sum, max) = (rows + p.vt, s2, x2);\n"
        "                        members.push(m);\n"
        "                        m += 1;\n"
        "                    }\n",
        "                        (rows, sum, max) = (rows + p.vt, s2, x2);\n"
        "                        members.push(m);\n"
        "                        m += 1;\n"
        "                    }\n"
        "                    {\n"
        "                        let live = preps.iter().filter(|p| p.is_some()).count();\n"
        "                        let raw = e.ctx().mem_get_info().map(|x| x.0).unwrap_or(0);\n"
        "                        let avail = crate::unified_memory::host_mem_available_bytes()"
        ".unwrap_or(0);\n"
        "                        eprintln!(\n"
        "                            \"[diag1307] dspark-pool lead={j} live={live} members={} rows={rows} \\\n"
        "                             headroom_mb={} raw_free_mb={} mem_available_mb={} cached_mb={} \\\n"
        "                             capped_extra_mb={:?} extra_kib={} col_kib={}\",\n"
        "                            members.len(),\n"
        "                            headroom >> 20,\n"
        "                            raw >> 20,\n"
        "                            avail >> 20,\n"
        "                            e.pool_cached_bytes() >> 20,\n"
        "                            diag_capped.map(|b| b >> 20),\n"
        "                            (sum - max) >> 10,\n"
        "                            col_bytes >> 10,\n"
        "                        );\n"
        "                    }\n",
        "walk end",
    )
    dfl.write_text(s)

s = spec.read_text()
if "[diag1307]" not in s:
    s = sub(
        s,
        "        // Peak instrument for the same bracket: the CAPTURE-TIME peak (warmup transients +\n",
        (DEEP_FN if deep else "")
        + "        let diag1307_fpr = || -> usize {\n"
        "            let m = std::fs::read_to_string(\"/proc/meminfo\").unwrap_or_default();\n"
        "            [\"MemFree:\", \"Buffers:\", \"Cached:\", \"SReclaimable:\"]\n"
        "                .iter()\n"
        "                .map(|k| {\n"
        "                    m.lines()\n"
        "                        .find(|l| l.starts_with(k))\n"
        "                        .and_then(|l| l.split_whitespace().nth(1))\n"
        "                        .and_then(|v| v.parse::<usize>().ok())\n"
        "                        .unwrap_or(0)\n"
        "                        * 1024\n"
        "                })\n"
        "                .sum()\n"
        "        };\n"
        "        let diag1307_0 = (\n"
        "            e.ctx().mem_get_info().map(|x| x.0).unwrap_or(0),\n"
        "            crate::unified_memory::host_mem_available_bytes().unwrap_or(0),\n"
        "            e.pool_cached_bytes(),\n"
        "            diag1307_fpr(),\n"
        "        );\n"
        "        // Peak instrument for the same bracket: the CAPTURE-TIME peak (warmup transients +\n",
        "bracket entry",
    )
    s = sub(
        s,
        "        // FAILURE IS AN OBSERVATION TOO: a capture that OOM'd at entry-effective E proved\n",
        "        if captured_now {\n"
        "            let raw1 = e.ctx().mem_get_info().map(|x| x.0).unwrap_or(0);\n"
        "            let av1 = crate::unified_memory::host_mem_available_bytes().unwrap_or(0);\n"
        "            let c1 = e.pool_cached_bytes();\n"
        "            let f1 = diag1307_fpr();\n"
        "            let (raw0, av0, c0, f0) = diag1307_0;\n"
        "            let d = |a0: usize, a1: usize| ((a0 + c0) as i64 - (a1 + c1) as i64) / 1024;\n"
        "            eprintln!(\n"
        "                \"[diag1307] draft-bracket raw0_mb={} avail0_mb={} cached0_mb={} raw1_mb={} \\\n"
        "                 avail1_mb={} cached1_mb={} raw_delta_kib={} avail_delta_kib={} \\\n"
        "                 fpr_delta_kib={}\",\n"
        "                raw0 >> 20,\n"
        "                av0 >> 20,\n"
        "                c0 >> 20,\n"
        "                raw1 >> 20,\n"
        "                av1 >> 20,\n"
        "                c1 >> 20,\n"
        "                d(raw0, raw1),\n"
        "                d(av0, av1),\n"
        "                d(f0, f1),\n"
        "            );\n"
        + (DEEP_END if deep else "")
        + "        }\n"
        "        // FAILURE IS AN OBSERVATION TOO: a capture that OOM'd at entry-effective E proved\n",
        "bracket end",
    )
    spec.write_text(s)
print("diag1307 applied")
