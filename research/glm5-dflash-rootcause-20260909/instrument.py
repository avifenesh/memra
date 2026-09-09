#!/usr/bin/env python3
"""Apply research-only captures to a disposable source checkout, never serving defaults."""
from pathlib import Path
import sys
root=Path(sys.argv[1])
def edit(path, old, new):
 p=root/path; s=p.read_text(); assert s.count(old)==1,(path,old,s.count(old)); p.write_text(s.replace(old,new))
edit('crates/memra-server/src/worker.rs', 'fn spec_k_pin() -> Option<usize> {', '''fn spec_k_pin() -> Option<usize> {
    // Research-only serial controller, pinned before each HTTP request.
    let value = std::fs::read_to_string("/root/dflash-rootcause-control/k").expect("rootcause K control");
    return if value.trim() == "auto" { None } else { Some(value.trim().parse().expect("rootcause K")) };
    #[allow(unreachable_code)]
''')
edit('crates/memra-engine/src/glm_spec.rs', '        let n_vocab = self.output.out_features();\n        let n_embd = self.cfg.n_embd as usize;\n        // The MTP block', '''        let rootcause_start = std::time::Instant::now();
        let n_vocab = self.output.out_features();
        let n_embd = self.cfg.n_embd as usize;
        // The MTP block''')
edit('crates/memra-engine/src/glm_spec.rs', '        // ---- 3. verify: one t=K+1 walk over the trunk ----', '''        e.stream().synchronize()?;
        let rootcause_verify = std::time::Instant::now();
        // ---- 3. verify: one t=K+1 walk over the trunk ----''')
edit('crates/memra-engine/src/glm_spec.rs', '        bump!(verify);\n        plap!(first_verify_ms);', '''        e.stream().synchronize()?;
        let rootcause_verify_ms = rootcause_verify.elapsed().as_secs_f64()*1000.0;
        bump!(verify);
        plap!(first_verify_ms);''')
edit('crates/memra-engine/src/glm_spec.rs', '                    let vam = eh.dtoh_u32(&vam_d)?;', '''                    let vam = eh.dtoh_u32(&vam_d)?;
                    eprintln!("[rootcause-greedy] drafts={:?} argmax={:?}", drafts, &vam[..drafts.len()]);''')
edit('crates/memra-engine/src/glm_spec.rs', '        Ok((round_tokens, drafts.len()))', '''        e.stream().synchronize()?;
        eprintln!("[rootcause-round] round={} k={} drafted={} accepted={} verify_ms={:.6} wall_ms={:.6}", sess.rounds, k, drafts.len(), j, rootcause_verify_ms, rootcause_start.elapsed().as_secs_f64()*1000.0);
        Ok((round_tokens, drafts.len()))''')
edit('crates/memra-engine/src/dflash.rs', '    let m = rejection_accept_len(&pj[..nq], &qj[..nq], &us);', '''    let m = rejection_accept_len(&pj[..nq], &qj[..nq], &us);
    if std::fs::read_to_string("/root/dflash-rootcause-control/mode").unwrap_or_default().trim() == "agreement" {
        let mut am = e.alloc_u32_zeroed(nq)?;
        let mut sampled = Vec::new();
        for r in 0..nq {
            e.argmax_token_device_col(p_src, r, n_vocab, &mut am, r)?;
            let mut pb = e.zeros(n_vocab)?;
            // Independent shadow stream. Never advances session sctr or uctr.
            e.gumbel_perturb_filtered_col(p_src, r, &mut pb, n_vocab,
                sp.seed ^ 0xD1A6_2026_0909_u64, (*uctr).wrapping_add(r as u32), sp.temp, &pmx, &pth, r)?;
            let td = e.argmax_token_device(&pb, n_vocab)?;
            sampled.push(e.dtoh_u32_one(&td)?);
        }
        let av = e.dtoh_u32(&am)?;
        eprintln!("[rootcause-agreement] drafts={:?} argmax={:?} sampled={:?} p={:?} q={:?} u={:?} accepted={}", ids, av, sampled, pj, qj, us, m);
    }''')
