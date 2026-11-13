#!/usr/bin/env python3
"""arms_1221b.py IN_WORKER_RS OUT_WORKER_RS ARM: the measurement arms of memra#1221 part 2.

Every arm carries part 1's trace probe (probe_i782.py: pool reserved/used at the publisher's entry
and exit, per plane allocation time and pool growth, per copy launch time; MEMRA_TICK_TRACE=1).
  B  the publisher as it is (the control).
  Z  the floor of a publication that does not copy: the publisher returns right after its entry
     line, so no plane is allocated or copied and no entry is inserted.
  O  the copies on the engine's copy stream: the allocations stay where they are (the scheduler
     thread, the compute stream), each copy waits for an event recorded on the compute stream
     after its plane's allocation and runs on `engine.copy_stream`; the compute stream waits for
     the last copy's event at the next tick's top (and before a same-tick retirement), so the
     copies overlap burst 2. Measurement only: nothing reads the entry before that join in a
     one-request cell.
"""
import subprocess
import sys

inp, out, arm = sys.argv[1], sys.argv[2], sys.argv[3]
here = __file__.rsplit("/", 1)[0]
subprocess.run([sys.executable, f"{here}/probe_i782.py", inp, out], check=True)
src = open(out).read()


def sub(old, new, count=1):
    global src
    assert src.count(old) == count, (arm, old[:80], src.count(old))
    src = src.replace(old, new)


if arm == "B":
    pass
elif arm == "Z":
    sub(
        """            "[specpub-probe] enter why={why} pos={pos} reserved={r} used={u} cached={}",
            r.saturating_sub(u)
        );
    }""",
        """            "[specpub-probe] enter why={why} pos={pos} reserved={r} used={u} cached={}",
            r.saturating_sub(u)
        );
    }
    // memra#1221b arm Z: the floor of a publication that does not copy.
    if std::hint::black_box(true) {
        if tick_trace_enabled() {
            eprintln!("[specpub-probe] arm-Z return why={why} pos={pos}");
        }
        return;
    }""",
    )
elif arm == "O":
    sub(
        """    let t = Instant::now();
    let ok = bytes == 0 || engine.copy_u8_into(dst, 0, src, bytes).is_ok();""",
        """    let t = Instant::now();
    let ok = bytes == 0 || probe_side_copy(engine, dst, src, bytes).is_ok();""",
    )
    sub(
        "/// memra#1221 probe (arm I): [`prefix_plane_alloc`] with a trace line.",
        """thread_local! {
    /// memra#1221b arm O: the last side-stream copy's completion event, joined by the compute
    /// stream at the next tick's top.
    static PROBE_SIDE_DONE: std::cell::RefCell<Option<cudarc::driver::CudaEvent>> =
        const { std::cell::RefCell::new(None) };
}

/// memra#1221b arm O: one plane's copy on the engine's copy stream, after the compute stream's
/// work so far (the plane's allocation included).
fn probe_side_copy(
    engine: &Engine,
    dst: &mut CudaSlice<u8>,
    src: &CudaSlice<u8>,
    bytes: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let ready = engine.stream().record_event(None)?;
    engine.copy_stream.wait(&ready)?;
    let mut view = dst
        .try_slice_mut(0..bytes)
        .ok_or("side copy destination range")?;
    engine
        .copy_stream
        .memcpy_dtod(&src.slice(0..bytes), &mut view)?;
    let done = engine.copy_stream.record_event(None)?;
    PROBE_SIDE_DONE.with(|c| *c.borrow_mut() = Some(done));
    Ok(())
}

/// memra#1221b arm O: the compute stream waits for the side copies (no-op when none is pending).
fn probe_side_join(engine: &Engine) {
    PROBE_SIDE_DONE.with(|c| {
        if let Some(ev) = c.borrow_mut().take() {
            let t = Instant::now();
            let ok = engine.stream().wait(&ev).is_ok();
            if tick_trace_enabled() {
                eprintln!(
                    "[specpub-probe] side-join ok={ok} us={}",
                    t.elapsed().as_micros()
                );
            }
        }
    });
}

/// memra#1221 probe (arm I): [`prefix_plane_alloc`] with a trace line.""",
    )
    sub(
        """            for s in active.iter_mut() {
                // A saved prime is not a committed decode/park boundary. Its adapter
                // owns the partial captures until finalization succeeds.
                if s.prime_service.pending {
                    continue;
                }
                drain_mtp_prefix_captures(&engine, &mut px, &mut hpx, s);
""",
        """            probe_side_join(&engine);
            for s in active.iter_mut() {
                // A saved prime is not a committed decode/park boundary. Its adapter
                // owns the partial captures until finalization succeeds.
                if s.prime_service.pending {
                    continue;
                }
                drain_mtp_prefix_captures(&engine, &mut px, &mut hpx, s);
""",
    )
    sub(
        """                drain_glm5_prefix_capture(&engine, &mut px, &mut hpx, s);
            }
        }
        // retire finished sessions""",
        """                drain_glm5_prefix_capture(&engine, &mut px, &mut hpx, s);
            }
        }
        probe_side_join(&engine);
        // retire finished sessions""",
    )
else:
    sys.exit(f"unknown arm {arm}")
open(out, "w").write(src)
