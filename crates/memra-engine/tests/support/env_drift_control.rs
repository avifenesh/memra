//! Linux-only, external-debugger rendezvous for the fixed `MEMRA_FAST=0 -> 1 -> 0`
//! experiment. Import this module from a probe and call `rendezvous("mutate")`
//! after retaining the object under test, then `rendezvous("restore")` before
//! checking recovery. Run the probe directly under `native_env_controller.py`.
//!
//! Assumptions: a trusted, single process (threads are allowed), ordinary libc
//! exec-time environment storage, no setenv/putenv/unsetenv/clearenv or Rust env
//! mutation anywhere in the process, and no other debugger or SIGCONT sender.
//! The controller waits for Linux's complete thread-group stop and checks every
//! task before changing one byte through /proc. This is Linux debugger behavior,
//! not a portable language-level environment mutation API. The helper keeps no
//! CStr, slice, or borrowed environment reference across the stop; it reacquires
//! getenv and copies bytes with volatile reads after resumption.
//!
//! SIGSTOP stops CPU threads, not already-submitted device work. Callers must
//! arrange an idle boundary themselves. This module calls no GPU APIs and does
//! not establish CUDA safety, authorization, or qualification. A missing parent
//! controller leaves SIGSTOP pending indefinitely: always use its bounded CLI.

#[cfg(target_os = "linux")]
pub fn rendezvous(phase: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::{self, Write};
    use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

    static STATE: AtomicU8 = AtomicU8::new(0);
    static ADDRESS: AtomicUsize = AtomicUsize::new(0);

    fn error(message: &str) -> io::Error {
        io::Error::other(message)
    }

    fn snapshot(expected: u8) -> Result<usize, io::Error> {
        // The owned String is dropped before we obtain the raw pointer.
        match std::env::var("MEMRA_FAST") {
            Ok(value) if value.as_bytes() == [expected] => {}
            _ => return Err(error("MEMRA_FAST differs from the expected 0/1 value")),
        }
        // SAFETY: getenv returns a NUL-terminated string under the documented
        // no-environment-mutator assumption. Only inspect the terminator after
        // proving the first byte is nonzero. No reference escapes this scope.
        unsafe {
            let pointer = libc::getenv(c"MEMRA_FAST".as_ptr()).cast::<u8>();
            if pointer.is_null() || pointer.read_volatile() != expected {
                return Err(error("MEMRA_FAST getenv value is not the expected 0/1"));
            }
            if pointer.add(1).read_volatile() != 0 {
                return Err(error(
                    "MEMRA_FAST must have exactly one byte followed by NUL",
                ));
            }
            Ok(pointer as usize)
        }
    }

    let (before, after, state) = match phase {
        "mutate" => (b'0', b'1', 0),
        "restore" => (b'1', b'0', 2),
        _ => return Err(error("environment control phase must be mutate or restore").into()),
    };
    // Reject concurrent, repeated and out-of-order calls. An error permanently
    // poisons this sequence: a probe must propagate it rather than retry.
    STATE
        .compare_exchange(state, state + 1, Ordering::SeqCst, Ordering::SeqCst)
        .map_err(|_| error("environment control calls must be exactly mutate then restore"))?;
    let address = snapshot(before)?;
    if phase == "restore" && address != ADDRESS.load(Ordering::SeqCst) {
        return Err(error("MEMRA_FAST getenv address changed before restore").into());
    }
    {
        let stdout = io::stdout();
        let mut output = stdout.lock();
        // The leading newline makes this standalone even after a partial log or
        // a test-harness prefix. Release the stdout lock before stopping.
        writeln!(
            output,
            "\nMEMRA_ENV_CONTROL\t{phase}\t{}\t{address}\t{}\t{}",
            std::process::id(),
            char::from(before),
            char::from(after)
        )?;
        output.flush()?;
    }
    // SAFETY: SIGSTOP is uncatchable on Linux and requests a thread-group stop.
    // The owning parent must observe completion before touching the known byte.
    if unsafe { libc::raise(libc::SIGSTOP) } != 0 {
        return Err(io::Error::last_os_error().into());
    }
    if snapshot(after)? != address {
        return Err(error("MEMRA_FAST getenv address changed after resume").into());
    }
    ADDRESS.store(address, Ordering::SeqCst);
    STATE.store(state + 2, Ordering::SeqCst);
    Ok(())
}

#[cfg(not(target_os = "linux"))]
pub fn rendezvous(_phase: &str) -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "environment drift rendezvous requires Linux /proc and an owning controller",
    )
    .into())
}
