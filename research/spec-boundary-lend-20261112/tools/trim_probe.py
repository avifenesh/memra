#!/usr/bin/env python3
"""trim_probe.py IN_WORKER_RS OUT_WORKER_RS: memra#1221 part 2's leak-check probe, the same patch
on main's and the head's worker.rs. The stock server has no admin listener, so this probe drives
the existing TrimPools path (the engine half of `/admin/trim`: every evictable pool dropped, the
device pools trimmed back to the driver) from a file: when the file named by PROBE_TRIM_FILE
exists at a tick top, it is removed and one trim runs. It prints the pool's reserved and used
bytes and the prefix cache's charged bytes once at the first tick (the boot baseline) and after
every trim. Measurement only: no other line of the worker changes."""
import sys

src = open(sys.argv[1]).read()


def sub(old, new, count=1):
    global src
    assert src.count(old) == count, (old[:80], src.count(old))
    src = src.replace(old, new)


sub(
    """        for tx in pending_trims.drain(..) {
            synchronize_model_devices(&engine, &loaded, "admin-trim-before-drop");""",
    """        // memra#1221 part 2 leak-check probe (tools/trim_probe.py).
        {
            static BASELINE: std::sync::Once = std::sync::Once::new();
            BASELINE.call_once(|| {
                let (r, u) = engine.pool_reserved_used();
                eprintln!(
                    "[probe-trim] baseline pool_reserved={r} pool_used={u} prefix_total_bytes={}",
                    px.total_bytes
                );
            });
            if let Some(path) = std::env::var_os("PROBE_TRIM_FILE")
                && std::path::Path::new(&path).exists()
            {
                let _ = std::fs::remove_file(&path);
                let (tx, _rx) = tokio::sync::oneshot::channel();
                pending_trims.push(tx);
                eprintln!("[probe-trim] requested");
            }
        }
        for tx in pending_trims.drain(..) {
            synchronize_model_devices(&engine, &loaded, "admin-trim-before-drop");""",
)
sub(
    """                report.prefix_entries
            );
            let _ = tx.send(report);
        }""",
    """                report.prefix_entries
            );
            let _ = tx.send(report);
            synchronize_model_devices(&engine, &loaded, "probe-trim-after");
            let (r, u) = engine.pool_reserved_used();
            eprintln!(
                "[probe-trim] after trim pool_reserved={r} pool_used={u} prefix_total_bytes={} \\
                 prefix_entries={}",
                px.total_bytes,
                px.entries.values().map(Vec::len).sum::<usize>()
            );
        }""",
)
open(sys.argv[2], "w").write(src)
