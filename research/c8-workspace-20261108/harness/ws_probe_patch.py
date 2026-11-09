#!/usr/bin/env python3
"""Measurement-only patch: log the device pool's peak and retained growth around every outermost
prime call (single prime_cache, the walker's prime_chunk, the concat prime_cache_batch).

  python3 ws_probe_patch.py WORKTREE        apply (idempotent)
  python3 ws_probe_patch.py WORKTREE --undo remove

Line per call:
  [ws-probe] kind=K rows=R members=M per=[..] pos0=P used0=B peak=B retained=B slab0=B slab1=B
peak = USED_MEM_HIGH during the call minus USED_MEM_CURRENT at its start (the pool watermark is
reset at the start); retained = USED_MEM_CURRENT after minus before. Only the outermost call on the
thread probes (a depth counter), so nested prime_chunk calls under prime_cache do not reset the
outer watermark. Never landed: the probe binary is a measurement build.
"""
import sys
from pathlib import Path

wt = Path(sys.argv[1])
undo = "--undo" in sys.argv
f = wt / "crates/memra-engine/src/hybrid_forward.rs"
src = f.read_text()
MARK = "// WS-PROBE (measurement build)"

helpers = f'''
{MARK}
thread_local! {{
    static WS_PROBE_DEPTH: std::cell::Cell<usize> = const {{ std::cell::Cell::new(0) }};
}}

{MARK}
struct WsProbe {{
    outer: bool,
    used0: usize,
    slab0: usize,
}}

{MARK}
impl HybridModel {{
    fn ws_probe_begin(&self, e: &Engine) -> WsProbe {{
        let depth = WS_PROBE_DEPTH.with(|d| {{
            let v = d.get();
            d.set(v + 1);
            v
        }});
        if depth > 0 {{
            return WsProbe {{ outer: false, used0: 0, slab0: 0 }};
        }}
        let _ = e.pool_high_water_reset();
        let (_res, used0) = e.pool_reserved_used();
        WsProbe {{ outer: true, used0, slab0: self.prime_slab_bytes(e) }}
    }}

    fn ws_probe_end(&self, e: &Engine, p: WsProbe, kind: &str, per: &[usize], pos0: usize) {{
        WS_PROBE_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
        if !p.outer {{
            return;
        }}
        let _ = e.stream().synchronize();
        let (_res_high, used_high) = e.pool_high_water_reset();
        let (_res1, used1) = e.pool_reserved_used();
        let slab1 = self.prime_slab_bytes(e);
        eprintln!(
            "[ws-probe] kind={{kind}} rows={{}} members={{}} per={{:?}} pos0={{pos0}} used0={{}} peak={{}} \\
             retained={{}} slab0={{}} slab1={{}}",
            per.iter().sum::<usize>(),
            per.len(),
            per,
            p.used0,
            used_high.saturating_sub(p.used0),
            used1 as i64 - p.used0 as i64,
            p.slab0,
            slab1,
        );
    }}
}}
'''

edits = [
    # single prime_cache entry
    (
        "        let events_before = crate::progress::events();\n"
        "        let a4_before = self.a4_prime_receipt_begin();\n"
        "        let out = self.prime_cache_overlaid_inner(e, tokens, cache, queued_after, overlay);\n",
        "        let events_before = crate::progress::events();\n"
        "        let a4_before = self.a4_prime_receipt_begin();\n"
        f"        let ws_pos0 = cache.pos; {MARK}\n"
        f"        let ws = self.ws_probe_begin(e); {MARK}\n"
        "        let out = self.prime_cache_overlaid_inner(e, tokens, cache, queued_after, overlay);\n"
        f"        self.ws_probe_end(e, ws, \"prime\", &[tokens.len()], ws_pos0); {MARK}\n",
    ),
    # concat batch entry
    (
        "        let events_before = crate::progress::events();\n"
        "        let a4_before = self.a4_prime_receipt_begin();\n"
        "        let out = self.prime_cache_batch_inner(e, prompts, caches);\n",
        "        let events_before = crate::progress::events();\n"
        "        let a4_before = self.a4_prime_receipt_begin();\n"
        f"        let ws_per: Vec<usize> = prompts.iter().map(|p| p.len()).collect(); {MARK}\n"
        f"        let ws_pos0 = caches.iter().map(|c| c.pos).max().unwrap_or(0); {MARK}\n"
        f"        let ws = self.ws_probe_begin(e); {MARK}\n"
        "        let out = self.prime_cache_batch_inner(e, prompts, caches);\n"
        f"        self.ws_probe_end(e, ws, \"batch\", &ws_per, ws_pos0); {MARK}\n",
    ),
]

chunk_sig = (
    "    pub(crate) fn prime_chunk(\n"
    "        &self,\n"
    "        e: &Engine,\n"
    "        tokens: &[u32],\n"
    "        cache: &mut Cache,\n"
    "        seq_end: usize,\n"
    "        chunk_off: usize,\n"
    "        overlay: Option<&crate::vision::EmbedOverlay>,\n"
    "    ) -> Result<(Vec<f32>, CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {\n"
)
chunk_new = (
    chunk_sig
    + f"        let ws_pos0 = cache.pos; {MARK}\n"
    + f"        let ws = self.ws_probe_begin(e); {MARK}\n"
    + f"        let out = self.prime_chunk_probed(e, tokens, cache, seq_end, chunk_off, overlay); {MARK}\n"
    + f"        self.ws_probe_end(e, ws, \"chunk\", &[tokens.len()], ws_pos0); {MARK}\n"
    + f"        out {MARK}\n"
    + f"    }} {MARK}\n"
    + f"    #[allow(clippy::type_complexity)] {MARK}\n"
    + chunk_sig.replace("pub(crate) fn prime_chunk(", "fn prime_chunk_probed(")
)

if undo:
    lines = [l for l in src.split("\n") if MARK not in l]
    out = "\n".join(lines)
    # restore the chunk wrapper (a probed twin header line has no marker)
    out = out.replace(chunk_sig.replace("pub(crate) fn prime_chunk(", "fn prime_chunk_probed("), "")
    f.write_text(out)
    print("undone; verify with git diff")
    sys.exit(0)

if MARK in src:
    print("already applied")
    sys.exit(0)
for old, new in edits:
    assert src.count(old) == 1, ("anchor count", old[:80], src.count(old))
    src = src.replace(old, new)
assert src.count(chunk_sig) == 1, "prime_chunk anchor"
src = src.replace(chunk_sig, chunk_new)
# helpers at the end of the file (outside any test module is fine: an impl block anywhere)
src = src.rstrip("\n") + "\n" + helpers
f.write_text(src)
print("applied")
