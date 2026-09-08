"""Diagnostic source overlay only. Never applied to the shipping worktree.

Forces affordable valid plans on small fixtures, then corrupts one committed source
or consumes its carrier. No new runtime environment variables or shipping flags.
"""
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('source', type=Path)
a = ap.parse_args()
p = a.source / 'crates/memra-server/src/worker.rs'
s = p.read_text()
s = s.replace('let reclaim_prefix_pin = if !headroom.sufficient(required) {', 'let reclaim_prefix_pin = if true {')
anchor = s.index('if let Some((rows, next_cost)) = restored.zip(planned_cost)')
end = s.index('let eager_arm', anchor)
block = s[anchor:end]
assert block.count('&& !headroom.sufficient(required)') == 1
block = block.replace('&& !headroom.sufficient(required)', '// Qualification overlay: force a valid affordable plan.')
s = s[:anchor] + block + s[end:]
anchor = s.index('    if let Some(plan) = admission_restore {', s.index('fn admit('))
anchor += len('    if let Some(plan) = admission_restore {')
start = s.index('fn admit(')
prompt_at = s.index('    let prompt = req', start)
s = s[:prompt_at] + s[prompt_at:].replace('    let prompt = req', '    let mut prompt = req', 1)
anchor += 4
s = s[:anchor] + '''
        // Qualification overlay: mutate only after the real plan has committed.
        let fault = req.cache_ns.rsplit('\\x1f').next().unwrap_or("");
        if let Some(i) = px.id_index(&plan.pin) {
            let en = &mut px.entries.get_mut(&plan.pin.key).unwrap()[i];
            match fault {
                "missing-tail" => en.dspark_draft = None,
                "short-tail" => {
                    let tail = en.dspark_draft.as_mut().unwrap();
                    tail.rows -= 1;
                    tail.base += 1;
                },
                "invalid-source" => en.pos -= 1,
                "missing-logits" => en.last_logits.clear(),
                "short-suffix" => prompt.truncate(plan.rows + 1),
                "lost-source" => {
                    px.unpin(&plan.pin);
                    px.evict_all();
                },
                _ => {},
            }
        }
        eprintln!("[retained-fault] committed source fault={fault}");
''' + s[anchor:]
anchor = s.index('match lm.model.dspark_spec_session_from_restored(')
end = s.index(') {', anchor)
s = s[:end] + ''').and_then(|sess| {
                        if req.cache_ns.ends_with("consumed-carrier") {
                            drop(sess);
                            Err("injected failure after carrier consumption".into())
                        } else { Ok(sess) }
                    }''' + s[end:]
needle = "let (pool_reserved, pool_used) = engine.pool_reserved_used();"
assert s.count(needle) == 1
s = s.replace(needle, needle + "\n            eprintln!(\"[retained-pool] active={} parked_dspark={} pinned_bytes={} pool_used={pool_used}\", active.len(), dspark_reuse.values().map(Vec::len).sum::<usize>(), px.pinned_bytes());")
p.write_text(s)
