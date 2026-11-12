#!/usr/bin/env bash
# Runs ON the dev desk (memra#1211). Red arm 1: the slot probe's storage (three zero-rate folds
# into the realized slot (n, K=3), no mark map read), which the two cost-model tests must fail.
# Red arm 2: the tick's stall sweep taken out, which the wiring test must fail. Red arm 3: the
# mark names the tick's count only (no admission marks), which the spread test must fail. Red arms
# 4-6 (the hold): no held admission, a held wave on a fresh time-to-live, and the demotion sites
# without the session's stall count; the hold test or the wiring test must fail each. Red arms
# 7-8 (the refresh): a refresh that creates a mark, and the tick refresh taken out. Red arms
# 9-12 (review): a refresh that keeps a count whose mark is gone, a hold on the earliest expiry,
# a demotion check that never rewrites the count, a late mark on a wave no verdict reads. Then green.
set -u
D=$HOME/.cache/memra-1211
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$D/target
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_NVCC$\|^MEMRA_CUDA_ARCH$'); do unset "$v"; done
cd "$D/tree" || exit 1
W=crates/memra-server/src/worker.rs
cp "$W" "$D/worker.green.rs"
TESTS="spec_cost_stall_mark spec_cost_stall_hold spec_cost_stall_refresh spec_press_history_marks_at_four spec_stall_mark_is_wired"
run() {
  local name=$1
  echo "== $name: $(date -Is)"
  for t in $TESTS; do
    nice -n 15 cargo test --release -p memra-server --lib -j 14 "$t" >> "$D/redgreen-$name.log" 2>&1
    echo "$name $t rc=$?"
  done
  grep -E "^test .*(ok|FAILED)$|panicked at|test result" "$D/redgreen-$name.log" | head -n 30
}
rm -f "$D"/redgreen-*.log
python3 - "$W" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
a = """    fn stalled_at(&self, n: usize, now: Instant) -> bool {
        self.stalled.get(&n).is_some_and(|until| now < *until)
    }"""
b = """    fn mark_stall(&mut self, n: usize, now: Instant) -> bool {
        self.mark_stall_until(n, now + SPEC_COST_REALIZED_TTL, now)
    }"""
assert s.count(a) == 1 and s.count(b) == 1
s = s.replace(a, """    fn stalled_at(&self, _n: usize, _now: Instant) -> bool {
        false
    }""")
s = s.replace(b, """    fn mark_stall(&mut self, n: usize, now: Instant) -> bool {
        // RED ARM: the slot probe's storage, three zero-rate samples folded into (n, K=3).
        for _ in 0..3 {
            let slot = self.realized.entry((n, 3)).or_insert_with(|| RealizedRate::new(0.0));
            let a = SPEC_COST_REALIZED_ALPHA;
            if slot.samples > 0 {
                let d = 0.0 - slot.ema;
                slot.ema += a * d;
                slot.var = (1.0 - a) * (slot.var + a * d * d);
            }
            slot.samples = slot.samples.saturating_add(1);
            slot.last = now;
        }
        true
    }""")
p.write_text(s)
PY
run red-slot
cp "$D/worker.green.rs" "$W"
python3 - "$W" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
a = "                    spec_cost_mark_stall(&s.model, n_tick, waiting);\n"
assert s.count(a) == 1
p.write_text(s.replace(a, "                    let _ = (n_tick, waiting);\n"))
PY
run red-wiring
cp "$D/worker.green.rs" "$W"
python3 - "$W" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
a = "    let mark = if entry.wave_marks > 0 {"
assert s.count(a) == 1
p.write_text(s.replace(a, "    let mark = if false && entry.wave_marks > 0 {"))
PY
run red-ntick
cp "$D/worker.green.rs" "$W"
python3 - "$W" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
a = "        WaveMark::Held(entry.wave_hold.filter(|until| Instant::now() < *until)?)\n"
assert s.count(a) == 1
p.write_text(s.replace(a, "        return None\n"))
PY
run red-nohold
cp "$D/worker.green.rs" "$W"
python3 - "$W" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
a = "        let slot = self.stalled.entry(n).or_insert(until);\n        *slot = (*slot).max(until);\n"
assert s.count(a) == 1
p.write_text(s.replace(a, "        let _ = until;\n        self.stalled.insert(n, now + SPEC_COST_REALIZED_TTL);\n"))
PY
run red-freshttl
cp "$D/worker.green.rs" "$W"
python3 - "$W" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
a = "} else if !spec_demote_due_session(s, n_live) {"
assert s.count(a) == 4  # three demotion sites, then the wiring test's own literal
p.write_text(s.replace(a, "} else if !spec_demote_due(&s.model, s.spec_k, n_live) {", 3))
PY
run red-nodemotekey
cp "$D/worker.green.rs" "$W"
python3 - "$W" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
a = "        if !self.stalled_at(n, now) {\n            return false;\n        }\n        self.stalled.insert(n, now + SPEC_COST_REALIZED_TTL);"
assert s.count(a) == 1
p.write_text(s.replace(a, "        self.stalled.insert(n, now + SPEC_COST_REALIZED_TTL);"))
PY
run red-refreshcreates
cp "$D/worker.green.rs" "$W"
python3 - "$W" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
a = "                    if spec_cost_refresh_stall(&s.model, &mut s.spec_stall_n) {\n"
assert s.count(a) == 1
p.write_text(s.replace(a, "                    if { let _ = &mut s.spec_stall_n; false } {\n"))
PY
run red-norefreshwire
cp "$D/worker.green.rs" "$W"
python3 - "$W" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
a = "    if !live {\n        *key = None;\n    }\n    live\n"
assert s.count(a) == 1
p.write_text(s.replace(a, "    live\n"))
PY
run red-staledrop
cp "$D/worker.green.rs" "$W"
python3 - "$W" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
a = "Some(until) => hold = Some(hold.map_or(until, |h| h.max(until))),"
assert s.count(a) == 1
p.write_text(s.replace(a, "Some(until) => hold = Some(hold.map_or(until, |h| h.min(until))),"))
PY
run red-holdmin
cp "$D/worker.green.rs" "$W"
python3 - "$W" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
a = "    s.spec_stall_n = (due && spec_cost_stalled(&s.model, n_live)).then_some(n_live);\n"
assert s.count(a) == 1
p.write_text(s.replace(a, "    if due && spec_cost_stalled(&s.model, n_live) {\n        s.spec_stall_n = Some(n_live);\n    }\n"))
PY
run red-stickycount
cp "$D/worker.green.rs" "$W"
python3 - "$W" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
a = "    if spec.is_some() && req_spec_k_replay.is_none() {\n        // The request's stall mark follows it to the wave this second verdict reads.\n        if let Some(mark) = stall_wave {\n            spec_cost_mark_wave(&req.model, projected_wave, mark);\n        }\n"
assert s.count(a) == 1
p.write_text(s.replace(a, "    if let Some(mark) = stall_wave {\n        spec_cost_mark_wave(&req.model, projected_wave, mark);\n    }\n    if spec.is_some() && req_spec_k_replay.is_none() {\n"))
PY
run red-latemark
cp "$D/worker.green.rs" "$W"
run green
echo "REDGREEN-DONE"
