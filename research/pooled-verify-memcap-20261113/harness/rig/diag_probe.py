#!/usr/bin/env python3
"""diag_probe.py TREE: memra#1266 diagnostic arm D (never for landing). Adds `[p1266]` log lines
to a main tree in place: driver free / pool reserved / used / graph-mem readings at the admission
gate, around newcomers' serial steps, at the pooled cohort's entry and after its bursts begin, at
each pooled verify call while driver free is under 1 GiB, at the graph-replay suspension, on every
failed engine allocation (size, track_caller site) and every engine allocation of 64 MiB or more,
at each failing stage of the pooled verify forward, and at the step OOM. Logging only: no device
work, no control-flow change."""
import sys

root = sys.argv[1].rstrip("/")


def patch(rel, subs):
    path = f"{root}/{rel}"
    src = open(path).read()
    for old, new in subs:
        n = src.count(old)
        assert n == 1, (rel, old[:90], n)
        src = src.replace(old, new)
    open(path, "w").write(src)


MEM_FN = '''    /// memra#1266 diag probe: one reading of the device and its pools, MiB.
    pub fn p1266_mem(&self) -> String {
        let free = self.ctx().mem_get_info().map(|(f, _)| f).unwrap_or(0);
        let (r, u) = self.pool_reserved_used();
        format!(
            "drv_free={}MiB pool_res={}MiB pool_used={}MiB pool_cached={}MiB graph_mem={}MiB",
            free >> 20,
            r >> 20,
            u >> 20,
            r.saturating_sub(u) >> 20,
            self.device_graph_mem_reserved() >> 20
        )
    }

    #[track_caller]
    fn p1266_alloc_note(&self, bytes: usize) {
        if bytes >= 64 << 20 {
            let loc = std::panic::Location::caller();
            eprintln!(
                "[p1266] big-alloc bytes={bytes} at {}:{} {}",
                loc.file(),
                loc.line(),
                self.p1266_mem()
            );
        }
    }

    #[track_caller]
    fn p1266_alloc_fail(&self, bytes: usize, err: &dyn std::fmt::Debug) {
        let loc = std::panic::Location::caller();
        eprintln!(
            "[p1266] alloc-fail bytes={bytes} at {}:{} err={err:?} {}",
            loc.file(),
            loc.line(),
            self.p1266_mem()
        );
    }

    pub fn pool_trim_to_zero(&self) -> usize {'''


def alloc(expr, ty_bytes, binding="let s"):
    return f'''{binding} = {{
            self.p1266_alloc_note({ty_bytes});
            match {expr} {{
                Ok(v) => v,
                Err(err) => {{
                    self.p1266_alloc_fail({ty_bytes}, &err);
                    return Err(err.into());
                }}
            }}
        }};'''


patch(
    "crates/memra-engine/src/lib.rs",
    [
        ("    pub fn pool_trim_to_zero(&self) -> usize {", MEM_FN),
        (
            "let mut s = unsafe { self.gpu.stream().alloc::<T>(n)? };",
            alloc("unsafe { self.gpu.stream().alloc::<T>(n) }", "n * std::mem::size_of::<T>()", "let mut s"),
        ),
        ("let s = self.gpu.stream().alloc_zeros::<u8>(n)?;", alloc("self.gpu.stream().alloc_zeros::<u8>(n)", "n")),
        ("let s = unsafe { self.gpu.stream().alloc::<f32>(n)? };", alloc("unsafe { self.gpu.stream().alloc::<f32>(n) }", "n * 4")),
        ("let s = unsafe { self.gpu.stream().alloc::<u8>(n)? };", alloc("unsafe { self.gpu.stream().alloc::<u8>(n) }", "n")),
        ("let s = self.gpu.stream().alloc_zeros::<f32>(n)?;", alloc("self.gpu.stream().alloc_zeros::<f32>(n)", "n * 4")),
    ],
)


def stage(call, name):
    return (
        f"{call}.map_err(|err| {{ eprintln!(\"[p1266] pooled-verify-fail stage={name} il={{il}} rows={{r_total}} err={{err}} {{}}\", e.p1266_mem()); err }})?"
    )


patch(
    "crates/memra-engine/src/spec.rs",
    [
        (
            "Mixer::Linear(_) => self.pooled_linear_layer(e, il, &x, r_total, segs)?,",
            "Mixer::Linear(_) => " + stage("self.pooled_linear_layer(e, il, &x, r_total, segs)", "linear") + ",",
        ),
        (
            "self.pooled_fa_layer(e, il, &x, r_total, &pos, segs)?",
            stage("self.pooled_fa_layer(e, il, &x, r_total, &pos, segs)", "fa"),
        ),
        (
            "self.pooled_dflash_taps(e, il, &x, segs)?;",
            stage("self.pooled_dflash_taps(e, il, &x, segs)", "taps") + ";",
        ),
        (
            "        let mut hn = vbuf(e, r_total * n_embd)?;",
            "        let il = usize::MAX;\n        let mut hn = "
            + stage("vbuf(e, r_total * n_embd)", "hn")
            + ";",
        ),
        (
            "let logits = e.matmul(&self.output, &hn, r_total)?;",
            "let logits = " + stage("e.matmul(&self.output, &hn, r_total)", "lm_head") + ";",
        ),
        (
            "let mut bursts: Vec<Option<SpecBurst<'_, '_, '_>>> = Vec::with_capacity(n);",
            "eprintln!(\"[p1266] cohort-engine enter n={n} {}\", e.p1266_mem());\n"
            "        let mut bursts: Vec<Option<SpecBurst<'_, '_, '_>>> = Vec::with_capacity(n);",
        ),
        (
            "            // Plain riders: this round's rows, after each rider's emission step.",
            "            {\n                static FIRST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);\n"
            "                let free = e.ctx().mem_get_info().map(|(f, _)| f).unwrap_or(0);\n"
            "                if free < (1 << 30) || FIRST.swap(false, std::sync::atomic::Ordering::Relaxed) {\n"
            "                    eprintln!(\"[p1266] cohort-round pre-verify preps={} {}\", preps.iter().flatten().count(), e.p1266_mem());\n"
            "                }\n            }\n"
            "            // Plain riders: this round's rows, after each rider's emission step.",
        ),
        (
            "rs.graph_guard_noted = true;",
            "rs.graph_guard_noted = true;\n            eprintln!(\"[p1266] graph-suspend {}\", e.p1266_mem());",
        ),
    ],
)

patch(
    "crates/memra-server/src/worker.rs",
    [
        (
            "let step_result = if let Some((r, _, _)) = pooled_step {",
            "let p1266_new = pooled_step.is_none() && active[i].generated.is_empty();\n"
            "                    if p1266_new {\n"
            "                        eprintln!(\"[p1266] serial-step before slot={i} {}\", engine.p1266_mem());\n"
            "                    }\n"
            "                    let step_result = if let Some((r, _, _)) = pooled_step {",
        ),
        (
            "let step_elapsed_ms =",
            "if p1266_new {\n"
            "                        eprintln!(\"[p1266] serial-step after slot={i} ok={} generated={} {}\", step_result.is_ok(), active[i].generated.len(), engine.p1266_mem());\n"
            "                    }\n"
            "                    let step_elapsed_ms =",
        ),
        (
            "let fits_full = headroom.sufficient(required);",
            "eprintln!(\n"
            "                        \"[p1266] admit-gate model={model_key} prompt={prompt_len} cost={}MiB reserve={}MiB required={}MiB headroom_free={}MiB pending_booked={}MiB px_total={}MiB px_evictable={}MiB restore_rows={} sufficient={} {}\",\n"
            "                        cost >> 20, reserve >> 20, required >> 20, headroom.limiting_free_bytes() >> 20, pending_booked >> 20,\n"
            "                        px.total_bytes >> 20, px.evictable_bytes() >> 20,\n"
            "                        admission_restore.as_ref().map_or(0, |p| p.rows), headroom.sufficient(required), engine.p1266_mem()\n"
            "                    );\n"
            "                    let fits_full = headroom.sufficient(required);",
        ),
        (
            "\"[admit-oom] step OOM NOT parked (model {}, retries \\",
            "\"[p1266] step-oom px_total={}MiB px_evictable={}MiB {}\", px.total_bytes >> 20, px.evictable_bytes() >> 20, engine.p1266_mem());\n"
            "                                eprintln!(\n"
            "                                    \"[admit-oom] step OOM NOT parked (model {}, retries \\",
        ),
        (
            "let model_name = active[cohort[0]].model.clone();",
            "let model_name = active[cohort[0]].model.clone();\n"
            "    eprintln!(\"[p1266] cohort-worker enter members={cohort:?} riders={riders:?} {}\", engine.p1266_mem());",
        ),
        (
            "    let outcomes = {\n        let mut reqs: Vec<memra_engine::spec::PooledBurstReq<'_>> = Vec::new();",
            "    eprintln!(\"[p1266] cohort-worker preambles-done {}\", engine.p1266_mem());\n"
            "    let outcomes = {\n        let mut reqs: Vec<memra_engine::spec::PooledBurstReq<'_>> = Vec::new();",
        ),
    ],
)
print("diag probe applied to", root)
