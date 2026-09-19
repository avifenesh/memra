#!/usr/bin/env python3
"""Generate the deliberately PARTIAL mask-guard patch without editing runtime files.
Full BankedResidency/UniformLease dispatch migration is blocked; see PATCH-REVIEW.md.
"""
from pathlib import Path
import difflib
import subprocess

root = Path(__file__).resolve().parents[2]
changes = {}
p = Path("crates/memra-engine/src/hybrid.rs")
old = (root / p).read_text()
needle = "impl MoeWeights {\n"
assert old.count(needle) == 1
helper = '''impl MoeWeights {
    /// Validate original router positions before layout lookup, cache admission,
    /// page hints, or staging. A masked position never acquires backing bytes.
    pub(crate) fn require_active_expert(&self, original: usize) -> std::io::Result<()> {
        let count = self.gate_exps.n_expert;
        if self.up_exps.n_expert != count || self.down_exps.n_expert != count
            || self.active_experts.as_ref().is_some_and(|mask| mask.len() != count)
        {
            return Err(std::io::Error::other("inconsistent expert bank geometry/mask"));
        }
        if original >= count || original > u16::MAX as usize {
            return Err(std::io::Error::other("original expert id out of dispatch range"));
        }
        if self.active_experts.as_ref().is_some_and(|mask| !mask[original]) {
            return Err(std::io::Error::other("masked original expert cannot be staged"));
        }
        // v2: validate authoritative metadata before expert_layout or source slicing.
        // CPU-proven extent rule; no native BankSource/SLRU installation implied.
        for exps in [&self.gate_exps, &self.up_exps, &self.down_exps] {
            if exps.layouts.as_ref().is_some_and(|v| v.len() != count)
                || exps.tiers.as_ref().is_some_and(|v| v.len() != count)
                || exps.macros.as_ref().is_some_and(|v| v.len() != count)
            {
                return Err(std::io::Error::other("inconsistent expert source metadata"));
            }
            // expert_layout computes e * stride for uniform sources; prove no overflow first.
            if exps.layouts.is_none() && original.checked_mul(exps.expert_stride).is_none() {
                return Err(std::io::Error::other("expert source offset overflow"));
            }
            let layout = exps.expert_layout(original);
            let available = exps.tiers.as_ref().map_or_else(
                || exps.bytes.len(), |tiers| tiers[original].len());
            memra_tier::bank::validate_bank_source_extent(
                layout.offset as u64, layout.len as u64, available as u64, exps.tiers.is_some()
            ).map_err(|err| std::io::Error::other(format!("invalid expert source extent: {err:?}")))?;
        }
        Ok(())
    }
'''
changes[p] = (old, old.replace(needle, helper))
p = Path("crates/memra-engine/src/hybrid_forward.rs")
old = (root / p).read_text()
new = old
for name in ("moe_cached_gemm_q8", "moe_cached_gemm", "moe_frozen_gemm", "moe_profile_admit_expert", "moe_prefetch_expert", "moe_prefetch_disk_expert"):
    start = new.index(f"    fn {name}(")
    body = new.index('        use crate::moe_cache::', start)
    new = new[:body] + '        m.require_active_expert(ex)?;\n' + new[body:]
needle = '''                let ex = ex as usize;
                // PER-EXPERT SLAB READ'''
assert new.count(needle) == 1
new = new.replace(needle, '''                let ex = ex as usize;
                m.require_active_expert(ex)?;
                // PER-EXPERT SLAB READ''')
# Grouped top-k MUST remain unchanged; validate before any selected id is indexed.
start = new.index('    pub(crate) fn moe_ffn_grouped(')
pos = new.index('        crate::moesd::record_host_routes', start)
new = new[:pos] + '''        for &original in &sel_all {
            m.require_active_expert(original as usize)?;
        }
''' + new[pos:]
# Invalid mixed resident state refuses; never switch its numeric program to a staged arm.
start = new.index('    pub(crate) fn moe_ffn_grouped(')
pos = new.index('        let slab_local = m', start)
new = new[:pos] + '''        if m.dev_exps.is_some() && !m.has_uniform_expert_layout() {
            return Err(std::io::Error::other("mixed expert bank cannot own a uniform resident slab").into());
        }
''' + new[pos:]
changes[p] = (old,new)
patch = []
for p, (old,new) in changes.items():
    formatted = subprocess.run(['rustfmt','--edition','2024','--emit','stdout'],input=new,text=True,capture_output=True,check=True).stdout
    patch.extend(difflib.unified_diff(old.splitlines(keepends=True),formatted.splitlines(keepends=True),fromfile='a/'+str(p),tofile='b/'+str(p)))
(root / 'research/spill-c-20260919/HY3-DISPATCH-PATCH.diff').write_text(''.join('\n' if line == ' \n' else line for line in patch))
print('Generated PARTIAL guard patch only; runtime sources unchanged')
