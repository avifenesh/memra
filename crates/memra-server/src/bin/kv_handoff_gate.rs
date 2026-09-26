//! B2 measurement seam (lane/spill-f-20260919, `M1-PREREG.md` B2 amendment): the stock server
//! plus one SIGUSR1-triggered host-tier handoff export, so the handoff's storage cost can be
//! measured with the in-tree engine alone. No env read and no HTTP route; the stock
//! `memra-server` binary is unchanged.
//!
//! SIGUSR1 calls `HostHandoffHandle::export(force = false)` and prints
//! `[handoff-gate] export ok <report json>` or `[handoff-gate] export refused: <reason>`.
//! The hook task drops every handle on the drain shutdown signal: the GPU worker exits only
//! when all of its command senders are gone. Send SIGUSR1 only after the
//! `[handoff-gate] armed` line; before the handler exists the signal's default action ends
//! the process.

use memra_server::{RuntimeHandles, ServerWiring, serve_with};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        eprintln!("[handoff-gate] FATAL: takes no arguments (got {args:?})");
        std::process::exit(2);
    }
    let wiring = ServerWiring::stock().on_ready(|handles: RuntimeHandles| {
        tokio::spawn(export_on_sigusr1(handles));
    });
    serve_with(wiring).await
}

async fn export_on_sigusr1(handles: RuntimeHandles) {
    use tokio::signal::unix::{SignalKind, signal};
    let RuntimeHandles {
        trim,
        metadata_reload,
        purge,
        kv_handoff,
        mut shutdown,
    } = handles;
    drop((trim, metadata_reload, purge));
    let mut usr1 = match signal(SignalKind::user_defined1()) {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("[handoff-gate] cannot install the SIGUSR1 handler: {err}");
            return;
        }
    };
    eprintln!("[handoff-gate] armed: SIGUSR1 exports the host tier");
    loop {
        tokio::select! {
            got = usr1.recv() => {
                if got.is_none() {
                    break;
                }
                match kv_handoff.export(false).await {
                    Ok(report) => eprintln!("[handoff-gate] export ok {report}"),
                    Err(reason) => eprintln!("[handoff-gate] export refused: {reason}"),
                }
            }
            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow() {
                    break;
                }
            }
        }
    }
    drop(kv_handoff);
    eprintln!("[handoff-gate] handles dropped on drain");
}
