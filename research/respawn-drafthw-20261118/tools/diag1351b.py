"""Write spec.rs with [diag1351b] lines for an arm: graph pool (reserved, used) and the free readings
at the draft-state bracket's entry (every bracket) and end (every landed capture)."""
import sys, pathlib
src = pathlib.Path(sys.argv[1]).read_text()
def diag(where, indent):
    body = f'''{{
    let gattr = |attr: cudarc::driver::sys::CUgraphMem_attribute| -> u64 {{
        let mut bytes: u64 = 0;
        if let Ok(dev) = cudarc::driver::result::device::get(e.ctx().ordinal() as i32) {{
            let _ = unsafe {{
                cudarc::driver::sys::cuDeviceGetGraphMemAttribute(
                    dev,
                    attr,
                    &mut bytes as *mut u64 as *mut std::ffi::c_void,
                )
            }};
        }}
        bytes
    }};
    let (praw, _) = e.ctx().mem_get_info().unwrap_or((0, 0));
    let (pres, pused) = e.pool_reserved_used();
    eprintln!(
        "[diag1351b] {where} raw={{}} delta={{:?}} pool_res={{}} pool_used={{}} graph_res={{}} graph_used={{}}",
        praw >> 20,
        crate::unified_memory::delta_free_bytes(e.unified_memory(), praw).map(|d| d >> 20),
        pres >> 20,
        pused >> 20,
        gattr(cudarc::driver::sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT) >> 20,
        gattr(cudarc::driver::sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT) >> 20,
    );
}}
'''
    return "".join(indent + l if l else l for l in body.splitlines(True))
a = "        let (cap_eff0, cap_meas0) = match e.ctx().mem_get_info() {"
assert src.count(a) == 1
src = src.replace(a, diag("entry", "        ") + a)
for b in ["            let parked_delta = draft_state_parked_delta(meas0, f1, e.pool_cached_bytes());\n",
          "            let parked_delta = draft_state_parked_delta(\n"]:
    if src.count(b) == 1:
        src = src.replace(b, diag("end", "            ") + b); break
else:
    raise SystemExit("no end anchor")
pathlib.Path(sys.argv[2]).write_text(src)
print("ok", src.count("[diag1351b]"))
