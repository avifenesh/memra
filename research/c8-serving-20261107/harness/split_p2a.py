"""Revert or re-apply the seed-batch (P2a) hunks in worker.rs, leaving the head-of-line hunks.

  python3 split_p2a.py revert|apply WORKER_RS
'revert' writes the P2a-free file and saves the full file next to it as .full; 'apply' restores .full.
"""
import shutil
import sys

mode, path = sys.argv[1], sys.argv[2]
if mode == 'apply':
    shutil.copy(path + '.full', path)
    sys.exit(0)

s = open(path).read()
shutil.copy(path, path + '.full')

# 1. the take cap in the candidate loop
a = """                        // A grid-aligned seed inside the prompt (memra#602) stops this row on
                        // the boundary, as `prefill_tick` stops a single prime; the seed then
                        // publishes after the batch call below.
                        let take = batch_take_to_seed(ql, take, s.fed.len(), s.seed_at);
"""
assert s.count(a) == 1, 'take cap'
s = s.replace(a, '')
# 2. the exclusion predicate
b = """                            // The grid-aligned seed boundary (memra#602) joins the batch: its
                            // row's take ends on the boundary (`batch_take_to_seed`). A boundary
                            // closer than the prime floor leaves `take < min_t` and the single
                            // path serves it, as before.
                            && take >= min_t"""
assert s.count(b) == 1, 'predicate'
s = s.replace(b, """                            // the grid-aligned seed boundary (memra#602) is the same class
                            // when it stops inside the prompt; a seed at the prompt end is not
                            // a stop and keeps the concat prime.
                            && !seed_boundary_inside_prompt(
                                s.seed_at,
                                s.fed.len() + s.prefill_queue.len(),
                            )
                            && take >= min_t""")
# 3. the post-batch seed publish
start = s.index('                                // GRID-ALIGNED SEED (memra#602): a row whose take ended on its')
end = s.index('                                s.prefill_done = s.prefill_queue.is_empty();', start)
s = s[:start] + s[end:]
open(path, 'w').write(s)
print('reverted P2a hunks (helper and its test stay; unused helper only warns)')
