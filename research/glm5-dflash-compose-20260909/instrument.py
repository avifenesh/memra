#!/usr/bin/env python3
"""Apply research-only captures to a disposable source checkout, never serving defaults."""
from pathlib import Path
import sys
root=Path(sys.argv[1])
def edit(path, old, new):
 p=root/path; s=p.read_text(); assert s.count(old)==1,(path,old,s.count(old)); p.write_text(s.replace(old,new))
edit('crates/memra-server/src/worker.rs', '    let _ = s.tx.send(Event::TokenSnapshot(s.generated.clone()));', '''    if memra_engine::COMPOSE_FLAGS.load(std::sync::atomic::Ordering::Relaxed) & 24 != 0 {
        eprintln!("[compose-output-ids] tokens={:?}", s.generated);
    }
    eprintln!("[compose-dispatch] flags={} kda={}", memra_engine::COMPOSE_FLAGS.load(std::sync::atomic::Ordering::Relaxed), memra_engine::kda::GLM5_VERIFY_E4M3_FUSED6_DISPATCHES.load(std::sync::atomic::Ordering::Relaxed));
    let _ = s.tx.send(Event::TokenSnapshot(s.generated.clone()));''')
edit('crates/memra-engine/src/lib.rs', 'const GDN_K2_DYNAMIC_SHARED_BYTES', '''pub static COMPOSE_FLAGS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
const GDN_K2_DYNAMIC_SHARED_BYTES''')
edit('crates/memra-engine/src/kda.rs', 'std::env::var("MEMRA_GLM5_VERIFY_E4M3_FUSED6").as_deref() == Ok("1")', '(crate::COMPOSE_FLAGS.load(std::sync::atomic::Ordering::Relaxed) & 1 != 0)')
edit('crates/memra-engine/src/glm_spec.rs', '''    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_GLM5_SPEC_CAUSAL_PMIN").as_deref() == Ok("1"))''', '''    crate::COMPOSE_FLAGS.load(std::sync::atomic::Ordering::Relaxed) & 4 != 0''')
edit('crates/memra-engine/src/mla_ffi.rs', '''        static SPLITKV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let splitkv = *SPLITKV
            .get_or_init(|| std::env::var("MEMRA_GLM5_MLA_VERIFY_SPLITKV").as_deref() == Ok("1"));''', '''        let splitkv = crate::COMPOSE_FLAGS.load(std::sync::atomic::Ordering::Relaxed) & 2 != 0;''')
edit('crates/memra-engine/src/mla_ffi.rs', '''                return ck("attn_gathered_verify_splitkv", rc);''', '''                ck("attn_gathered_verify_splitkv", rc)?;
                if crate::COMPOSE_FLAGS.load(std::sync::atomic::Ordering::Relaxed) & 8 != 0 {
                    let candidate = self.dtoh(o_lat)?;
                    let mut reference = self.zeros(t_q * n_head * kv_rank)?;
                    crate::COMPOSE_FLAGS.fetch_and(!2, std::sync::atomic::Ordering::Relaxed);
                    let result = self.mla_attn_gathered(q_lat, q_pe, cache, idx, &mut reference, n_head, kv_rank, d_rope, t_q, n_slots, scale);
                    crate::COMPOSE_FLAGS.fetch_or(2, std::sync::atomic::Ordering::Relaxed);
                    result?;
                    let reference = self.dtoh(&reference)?;
                    let mut changed = 0usize;
                    let mut max_ratio = 0f32;
                    let mut flips = 0usize;
                    for (a,b) in candidate.chunks(kv_rank).zip(reference.chunks(kv_rank)) {
                        let peak = b.iter().fold(0f32, |m,x| m.max(x.abs()));
                        let band = 1e-5 + 1e-4 * peak;
                        let argmax = |v: &[f32]| v.iter().enumerate().max_by(|x,y| x.1.total_cmp(y.1)).unwrap().0;
                        flips += usize::from(argmax(a) != argmax(b));
                        for (x,y) in a.iter().zip(b) {
                            assert!(x.is_finite() && y.is_finite());
                            changed += usize::from(x.to_bits() != y.to_bits());
                            max_ratio = max_ratio.max((x-y).abs()/band);
                        }
                    }
                    eprintln!("[compose-mla-oracle] t={} rows={} changed={} flips={} max_ratio={}", t_q, t_q*n_head, changed, flips, max_ratio);
                    assert!(flips == 0 && max_ratio <= 1.0, "MLA numerical contract failed");
                }
                return Ok(());''')
edit('crates/memra-server/src/worker.rs', 'fn spec_k_pin() -> Option<usize> {', '''fn spec_k_pin() -> Option<usize> {
    // Research-only serial controller, pinned before each HTTP request.
    let flags: u64 = std::fs::read_to_string("/root/dflash-compose-control/flags").expect("compose flags").trim().parse().expect("flags integer");
    memra_engine::COMPOSE_FLAGS.store(flags, std::sync::atomic::Ordering::Relaxed);
    let value = std::fs::read_to_string("/root/dflash-compose-control/k").expect("rootcause K control");
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
        if crate::COMPOSE_FLAGS.load(std::sync::atomic::Ordering::Relaxed) & 24 != 0 {
            eprintln!("[compose-tape] tokens={:?}", round_tokens);
        }
        eprintln!("[rootcause-round] round={} k={} drafted={} accepted={} verify_ms={:.6} wall_ms={:.6}", sess.rounds, k, drafts.len(), j, rootcause_verify_ms, rootcause_start.elapsed().as_secs_f64()*1000.0);
        Ok((round_tokens, drafts.len()))''')
edit('crates/memra-engine/src/dflash.rs', '    let m = rejection_accept_len(&pj[..nq], &qj[..nq], &us);', '''    let m = rejection_accept_len(&pj[..nq], &qj[..nq], &us);
    if std::fs::read_to_string("/root/dflash-compose-control/mode").unwrap_or_default().trim() == "agreement" {
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
