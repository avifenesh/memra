"""Apply offline observer/dispatch only to a disposable, pinned source snapshot."""
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
lib = root / "crates/memra-engine/src/lib.rs"
s = lib.read_text()
start = s.index("    pub fn fa_prefill_view_ws(")
point = s.index("        // pass 2: the bf16-workspace prefill twin", start)
insert = r'''
        // OFFLINE FA2 CHECKPOINT ONLY. Not a serving door or shipped source.
        if std::env::var("FA2_PROBE_ACTIVE").as_deref() == Ok("1") {
            assert!(live.is_none(), "diagnostic may not execute inside capture");
            assert_eq!((head_dim, n_head, n_head_kv, t, causal), (256, 24, 4, 1024, true));
            if let Ok(dir) = std::env::var("FA2_PROBE_DUMP") {
                static DUMPED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                if !DUMPED.swap(true, std::sync::atomic::Ordering::SeqCst) {
                    let qh = self.gpu.stream().clone_dtoh(q)?;
                    let kh = self.gpu.stream().clone_dtoh(kw)?;
                    let vh = self.gpu.stream().clone_dtoh(vw)?;
                    std::fs::create_dir_all(&dir)?;
                    std::fs::write(format!("{dir}/q.f32"), qh.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())?;
                    std::fs::write(format!("{dir}/k.bf16"), &kh[..k_ws_bytes])?;
                    std::fs::write(format!("{dir}/v.bf16"), &vh[..v_ws_bytes])?;
                    std::fs::write(format!("{dir}/shape.json"), format!("{{\"rows\":{t},\"depth\":{t_kv}}}"))?;
                }
            }
            if let Ok(name) = std::env::var("FA2_PROBE_KERNEL") {
                let path = std::env::var("FA2_PROBE_FATBIN")?;
                let module = self.gpu.ctx.load_module(cudarc::nvrtc::Ptx::from_binary(std::fs::read(path)?))?;
                let f = module.load_function(&name)?;
                let cfg = LaunchConfig { grid_dim: (((t * 6).div_ceil(64)) as u32, 4, 1), block_dim: (32,4,1), shared_mem_bytes: 32768 };
                let stream = self.gpu.stream();
                let (ti, tkvi) = (t as i32, t_kv as i32);
                let mut b = stream.launch_builder(&f);
                b.arg(q).arg(&*kw).arg(&*vw).arg(o).arg(&ti).arg(&tkvi);
                unsafe { b.launch(cfg)?; }
                stream.synchronize()?;
                eprintln!("fa2_probe kernel={name} rows={t} depth={t_kv}");
                return Ok(());
            }
        }
'''
assert "OFFLINE FA2 CHECKPOINT ONLY" not in s
lib.write_text(s[:point] + insert + s[point:])
(root / "crates/memra-engine/src/bin/fa2_real_chunk_probe.rs").write_text(
    pathlib.Path(__file__).with_name("real_chunk_probe.rs").read_text())
cargo = root / "crates/memra-engine/Cargo.toml"
cargo.write_text(cargo.read_text() + '\n[[bin]]\nname = "fa2-real-chunk-probe"\npath = "src/bin/fa2_real_chunk_probe.rs"\n')
