#!/usr/bin/env python3
"""Single-line mutants of the partial reclaim and its booking: each must fail a test.

  python3 mutants.py WORKTREE TARGET_DIR LOG
Restores worker.rs after every mutant and checks its sha256 at the end.
"""
import hashlib, os, subprocess, sys

wt, target, log = sys.argv[1], sys.argv[2], sys.argv[3]
only = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
path = os.path.join(wt, "crates/memra-server/src/worker.rs")
orig = open(path).read()
orig_sha = hashlib.sha256(orig.encode()).hexdigest()

M = [
    ("door_armed dropped", "    !door_armed && !unified && primary_reading && budget_above_floor\n",
     "    !unified && primary_reading && budget_above_floor\n"),
    ("unified dropped", "    !door_armed && !unified && primary_reading && budget_above_floor\n",
     "    !door_armed && primary_reading && budget_above_floor\n"),
    ("primary dropped", "    !door_armed && !unified && primary_reading && budget_above_floor\n",
     "    !door_armed && !unified && budget_above_floor\n"),
    ("floor dropped", "    !door_armed && !unified && primary_reading && budget_above_floor\n",
     "    !door_armed && !unified && primary_reading\n"),
    ("floor >=", "} => *bytes > entry_bytes.saturating_mul(PREFIX_CACHE_DEFAULT_ENTRIES),",
     "} => *bytes >= entry_bytes.saturating_mul(PREFIX_CACHE_DEFAULT_ENTRIES),"),
    ("configured true", "            Self::Configured { .. } => false,\n", "            Self::Configured { .. } => true,\n"),
    ("target keeps all", "    resident.saturating_sub(need)\n", "    resident\n"),
    ("own seed always", "    if prompt_len >= PREFIX_CACHE_MIN_TOKENS {\n        entry_bytes_at_prompt\n",
     "    if true {\n        entry_bytes_at_prompt\n"),
    ("own seed > floor", "    if prompt_len >= PREFIX_CACHE_MIN_TOKENS {\n        entry_bytes_at_prompt\n",
     "    if prompt_len > PREFIX_CACHE_MIN_TOKENS {\n        entry_bytes_at_prompt\n"),
    ("booking seeds uncapped", "        prime_owed.saturating_add(seed_booking_cap(pending_seed_uncapped, budget, resident))\n",
     "        prime_owed.saturating_add(pending_seed_uncapped)\n"),
    ("booking without prime", "        prime_owed.saturating_add(seed_booking_cap(pending_seed_uncapped, budget, resident))\n",
     "        seed_booking_cap(pending_seed_uncapped, budget, resident)\n"),
    ("disarm ignores prime", "    armed && prime_owed == 0 && pending_seed_uncapped == 0\n",
     "    armed && pending_seed_uncapped == 0\n"),
    ("disarm ignores seeds", "    armed && prime_owed == 0 && pending_seed_uncapped == 0\n",
     "    armed && prime_owed == 0\n"),
    ("disarm unarmed", "    armed && prime_owed == 0 && pending_seed_uncapped == 0\n",
     "    prime_owed == 0 && pending_seed_uncapped == 0\n"),
    ("rest when covered", "    let rest = !covered && kept_bytes > 0;\n", "    let rest = kept_bytes > 0;\n"),
    ("rest without kept", "    let rest = !covered && kept_bytes > 0;\n", "    let rest = !covered;\n"),
    ("rest never runs", "    if rest {\n        flush_rest();\n    }\n", "    if rest {}\n"),
    ("booking not joined", "                let pending_booked = pending_booked.saturating_add(reclaim_booked);\n",
     "                let pending_booked = pending_booked;\n"),
    ("seed term door only", "                let pending_seed_uncapped = if admit_memory_cfg.armed || reclaim_booking {\n",
     "                let pending_seed_uncapped = if admit_memory_cfg.armed {\n"),
    ("disarm skipped", "                        reclaim_booking = false;\n                        eprintln!(\n                            \"[admit-oom] reclaim booking off",
     "                        eprintln!(\n                            \"[admit-oom] reclaim booking off"),
    ("need double-books", "                                    .saturating_sub(reclaim_booked)\n", ""),
    ("owed not added", "                                    .saturating_add(owed),\n", "                                    ,\n"),
    ("own seed not added", "                                    .saturating_add(reclaim_arrival_seed_bytes(\n",
     "                                    .saturating_sub(reclaim_arrival_seed_bytes(\n"),
    ("pass is the flush", "                                (px.evict_to_bytes(target).0, 0, 0)\n",
     "                                (px.evict_all(), 0, 0)\n"),
    ("no fallback flush", "                                evicted_prefix += px.evict_all();\n",
     "                                evicted_prefix += 0;\n"),
    ("booking never stored", "                                reclaim_booking = true;\n", ""),
    ("owed without seeds", "        .saturating_add(pending_seed_bytes(active))\n}", "\n}"),
]

TESTS = ["partial_reclaim", "reclaim_booking", "pending_seed_is_armed_only"]
env = dict(os.environ, CARGO_TARGET_DIR=target, CARGO_BUILD_JOBS="12", CUDA_VISIBLE_DEVICES="")
out = open(log, "w")
killed = 0
try:
    for name, old, new in M:
        if only and name not in only:
            continue
        n = orig.count(old)
        if n != 1:
            out.write(f"SKIP {name}: pattern count {n}\n"); out.flush(); continue
        open(path, "w").write(orig.replace(old, new))
        b = subprocess.run(["nice", "-n", "10", "cargo", "test", "-p", "memra-server", "--lib", "--no-run",
                            "--message-format=short"], cwd=wt, env=env, capture_output=True, text=True)
        if b.returncode != 0:
            err = [l for l in b.stderr.splitlines() if l.startswith("error")][:2]
            out.write(f"KILLED(build) {name}: {err}\n"); out.flush(); killed += 1; continue
        exe = [l for l in b.stderr.splitlines() if "Executable unittests src/lib.rs" in l][0].split("(")[1].rstrip(")")
        failed = []
        for t in TESTS:
            r = subprocess.run([exe, t], cwd=wt, env=env, capture_output=True, text=True)
            if r.returncode != 0:
                failed += [l.split()[1] for l in r.stdout.splitlines() if l.startswith("test ") and l.endswith("FAILED")]
        verdict = "KILLED" if failed else "SURVIVED"
        killed += bool(failed)
        out.write(f"{verdict} {name}: {failed}\n"); out.flush()
finally:
    open(path, "w").write(orig)
assert hashlib.sha256(open(path).read().encode()).hexdigest() == orig_sha
out.write(f"killed {killed} of {len(M)}; worker.rs restored {orig_sha[:16]}\n")
out.close()
