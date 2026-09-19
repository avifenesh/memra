#!/usr/bin/env python3
"""One-time reproducible v1→v2 patch rewrite in deleted scratch files, never runtime."""
from pathlib import Path
import difflib
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = '020d2047'
FILES = ['crates/memra-server/src/worker.rs', 'crates/memra-server/src/admit_memory.rs', 'crates/memra-server/src/worker/host_glm.rs']
# v1 is retained in Git history, not overwritten measurement evidence.
v1 = subprocess.check_output(['git', 'show', '40f34e47:research/spill-b-20260919/HOSTPREFIX-PATCH.diff'], cwd=ROOT)
with tempfile.TemporaryDirectory(prefix='spill-b-patch-v2-') as directory:
    scratch = Path(directory)
    originals = {}
    for name in FILES:
        originals[name] = subprocess.check_output(['git','show',f'{BASE}:{name}'],cwd=ROOT).decode()
        p = scratch / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(originals[name])
    subprocess.run(['git','apply','-'], input=v1, cwd=scratch, check=True)
    p = scratch / FILES[0]
    s = p.read_text()
    s = s.replace('crate::admit_memory::reserve_tier_image(tier.governor.clone(), &request)',
                  'crate::admit_memory::reserve_tier_image(tier.governor.clone(), program, &request)')
    s = s.replace('if e._tier_charge.is_none() || e._tier_identity.lease(program, generation).is_err() {',
        '''if e._tier_charge.as_ref().and_then(|c| c.program()) != Some(program)
                || e._tier_identity.lease(program, generation).is_err()
            {''')
    needle = '        // Narrow first slice: native plain KV + recurrent continuation.'
    s = s.replace(needle, '''        if entry.model_generation.as_ref().is_none_or(|g| !Arc::ptr_eq(g, generation)) {
            return Err("tier model generation mismatch".into());
        }
'''+needle)
    # Charge shape metadata before publishing identity. Metadata allocation remains
    # bounded synchronous scratch, and its allocator overhead requires a rig audit.
    old = '''        entry
            ._tier_identity
            .bind(bundle, &entry.toks, program, &layout, generation.clone())
            .map_err(|e| format!("tier image identity refused: {e:?}"))?;
        entry._tier_metadata_charge =
            self.tier_charge(&entry.pool_key, 0, metadata.len() as u64, false)?;
        entry._tier_metadata = metadata;'''
    new = '''        let metadata_charge =
            self.tier_charge(&entry.pool_key, 0, metadata.capacity() as u64, false)?;
        entry._tier_identity
            .bind(bundle, &entry.toks, program, &layout, generation.clone())
            .map_err(|e| format!("tier image identity refused: {e:?}"))?;
        entry._tier_metadata_charge = metadata_charge;
        entry._tier_metadata = metadata;'''
    assert old in s
    s = s.replace(old, new)
    p.write_text(s)
    p = scratch / FILES[1]
    s = p.read_text().replace('    request: &memra_engine::cache::tiered::BudgetRequest,',
        '    program: &memra_engine::cache::record::ProgramIdentity,\n    request: &memra_engine::cache::tiered::BudgetRequest,')
    s = s.replace('ResidentCharge::reserve(governor, request)', 'ResidentCharge::reserve_for_program(governor, program, request)')
    p.write_text(s)
    subprocess.run(['rustfmt','--edition','2024','--config','skip_children=true',*[str(scratch / n) for n in FILES]], check=True)
    text = '''From 020d2047 Mon Sep 17 00:00:00 2001
From: WP-B <wp-b@invalid>
Subject: [PATCH v2] WIP host-prefix program-bound shared residency leases

REVIEW ONLY. UNAPPLIED, server/engine UNBUILT. Bootstrap stays tier=None.
Base: integration 020d2047. Requires day-4 memra-kv program-bound guard.
No new environment read or Cargo amendment. GPU and native binding gates pending.

'''
    for name in FILES:
        after = (scratch / name).read_text()
        text += f'diff --git a/{name} b/{name}\n'
        text += ''.join(difflib.unified_diff(originals[name].splitlines(True), after.splitlines(True), fromfile=f'a/{name}', tofile=f'b/{name}'))
    # Blank context lines are valid without the context space and pass diff --check.
    text = text.replace('\n \n', '\n\n')
    (HERE / 'HOSTPREFIX-PATCH.diff').write_text(text)
