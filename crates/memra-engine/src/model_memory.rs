//! Loaded ownership, independent of architecture names and ambient TP flags.
use crate::Engine;
use crate::cache::{Cache, LatentKvLayer};
use crate::glm5_tp::Glm5TpRt;
use crate::hybrid::{Ffn, HybridModel, Mixer, StepTpKvDeviceAdmission};
use crate::model_memory_plan::{
    DeviceBytes, f32_bytes, kda_elements, kda_workspace_row_bytes, missing_f32_bytes, mla_elements,
    mla_score_bytes, peer_workspace_bytes, unique_owners,
};

impl HybridModel {
    fn ownership_layers(&self) -> impl Iterator<Item = (&Mixer, &Ffn)> {
        self.layers
            .iter()
            .map(|layer| (&layer.mixer, &layer.ffn))
            .chain(
                self.mtp
                    .iter()
                    .chain(&self.mtp_extra)
                    .map(|head| (&head.mixer, &head.ffn)),
            )
    }

    pub(crate) fn glm_tp_runtimes(&self) -> Vec<&Glm5TpRt> {
        let mut runtimes = Vec::new();
        for (mixer, ffn) in self.ownership_layers() {
            match mixer {
                Mixer::Kda(kda) => {
                    if let Some(tp) = &kda.tp {
                        runtimes.push(tp.rt.as_ref());
                    }
                }
                Mixer::Mla(mla) => {
                    if let Some(tp) = &mla.tp {
                        runtimes.push(tp.rt.as_ref());
                    }
                }
                _ => {}
            }
            if let Ffn::Moe(moe) = ffn {
                // An expert-only runtime need not appear in an attention sidecar.
                if let Some(ep) = &moe.glm5_ep {
                    runtimes.push(ep.rt.as_ref());
                }
                if let Some(tp) = &moe.glm5_tp_split {
                    runtimes.push(tp.rt.as_ref());
                }
            }
        }
        unique_owners(runtimes)
    }

    fn step_runtimes(&self) -> Vec<&crate::tp::TpE4m3HostBounce> {
        let mut runtimes = Vec::new();
        for (mixer, ffn) in self.ownership_layers() {
            if let Mixer::Full(attention) = mixer
                && let Some(tp) = &attention.step_tp_qkv
            {
                runtimes.push(tp.runtime.as_ref());
            }
            if let Ffn::Moe(moe) = ffn {
                if let Some(ep) = &moe.step_ep {
                    runtimes.push(ep.runtime.as_ref());
                }
                if let Some(tp) = &moe.step_tp {
                    runtimes.push(tp.runtime.as_ref());
                }
            }
        }
        unique_owners(runtimes)
    }

    /// Every engine owning model execution state, once by Engine identity. Distinct
    /// engines on one ordinal have independent streams and must each be synchronized.
    /// Physical default-pool statistics and trim may subsequently collapse ordinals.
    /// This only observes already-loaded PP/TP runtimes; it never initializes a device.
    pub fn owned_engines<'a>(&'a self, primary: &'a Engine) -> Vec<&'a Engine> {
        let mut owners = vec![primary];
        if let Some(pp) = crate::pp::PpNRt::initialized() {
            for stage in 0..pp.n_stages() {
                owners.push(pp.engine(stage, primary));
            }
        }
        for rt in self.step_runtimes() {
            for rank in 0..rt.devices().len() {
                if let Some(engine) = rt.rank_engine(rank) {
                    owners.push(engine);
                }
            }
        }
        for rt in self.glm_tp_runtimes() {
            owners.extend(rt.peers.iter());
        }
        unique_owners(owners)
    }

    /// Additional conservative prefill workspace on each distinct GLM peer engine.
    /// `prompt_rows` is the full prompt/context depth (including a restored prefix),
    /// as in HyperPrimeWorkspaceShape::admission_bytes, never the context capacity.
    /// The worker owns the prefill-workspace switch and its separate transient reserve.
    ///
    /// GLM peers run replicated MLA/indexer projections and grouped expert staging in
    /// mla_tp_attn_cached / hyper_ffn_branch. Give each peer the SAME conservative
    /// unsharded prime envelope as the root; do not divide by TP degree. This includes
    /// the context-dependent k-pool score plane and root-only tail hiddens as slack.
    /// Add the KDA core's named allocations, not itemized in the generic coefficient.
    /// Shared runtime handles charge once; independent same-ordinal peers add physically.
    /// This is a budgeting bound, not GPU qualification or an exact peak-VRAM claim.
    pub fn peer_prefill_workspace_bytes(
        &self,
        prompt_rows: usize,
    ) -> Result<Vec<StepTpKvDeviceAdmission>, String> {
        let owners = unique_owners(
            self.glm_tp_runtimes()
                .into_iter()
                .flat_map(|rt| rt.peers.iter()),
        );
        if owners.is_empty() {
            return Ok(Vec::new());
        }
        let prompt_rows = prompt_rows.max(1);
        let (mut primary_workspace, call_rows) =
            if let Some(shape) = self.hyper_prime_workspace_shape() {
                (
                    shape.admission_bytes(prompt_rows),
                    crate::hybrid_forward::hyper_prime_call_rows(
                        prompt_rows,
                        shape.n_layers,
                        shape.gdn_grid,
                    ),
                )
            } else {
                let shape = self
                    .prime_workspace_shape()
                    .ok_or("missing prime workspace shape")?;
                (
                    shape.admission_bytes(prompt_rows),
                    crate::hybrid_forward::prime_chunk_tokens(prompt_rows, shape.n_layers)
                        .max(1)
                        .min(prompt_rows),
                )
            };
        // The non-hyper shape itemizes dense/GDN slabs, not MLA. Its GLM peer
        // still runs query/attention and replicated context-dependent scores.
        if self.hyper.is_none() {
            let mut mla_workspace = 0;
            for (mixer, _) in self.ownership_layers() {
                if let Mixer::Mla(mla) = mixer
                    && let Some(tp) = &mla.tp
                {
                    let g = mla.geom;
                    let query_width = g
                        .d_nope
                        .checked_add(g.d_rope)
                        .and_then(|v| v.checked_add(g.d_v))
                        .and_then(|v| v.checked_add(g.kv_rank))
                        .and_then(|v| v.checked_mul(tp.full_heads))
                        .and_then(|v| v.checked_mul(call_rows))
                        .ok_or("MLA peer workspace overflow")?;
                    let score = match &mla.index {
                        Some(index) => mla_score_bytes(call_rows, prompt_rows, index.geom.pool)?,
                        None => 0,
                    };
                    mla_workspace = mla_workspace.max(
                        f32_bytes(query_width)?
                            .checked_add(score)
                            .ok_or("MLA peer workspace overflow")?,
                    );
                }
            }
            primary_workspace = primary_workspace
                .checked_add(mla_workspace)
                .ok_or("MLA peer workspace overflow")?;
        }
        let mut extra_row_bytes = 0;
        for (mixer, _) in self.ownership_layers() {
            if let Mixer::Kda(kda) = mixer
                && kda.tp.is_some()
            {
                extra_row_bytes =
                    extra_row_bytes.max(kda_workspace_row_bytes(kda.heads(), kda.head_dim())?);
            }
        }
        let budget = peer_workspace_bytes(primary_workspace, extra_row_bytes, call_rows, 0)?;
        let mut charges = DeviceBytes::default();
        for owner in owners {
            charges.add(owner.ctx().ordinal(), budget)?;
        }
        Ok(charges
            .0
            .into_iter()
            .map(|(device, bytes)| StepTpKvDeviceAdmission { device, bytes })
            .collect())
    }

    /// Additional state still to be allocated on distributed ranks, grouped by physical
    /// device. Zero-byte entries retain participation in the caller's per-device reserve.
    ///
    /// Base cache accounting already covers the canonical Step shadow, full KDA shadow,
    /// and root MLA plane (including root pool keys); those are not charged again here.
    /// A present cache supplies its actual capacity and index layout, rather than re-reading
    /// an env-dependent layout decision. KDA adds conv + both recurrent planes on every
    /// rank. MLA adds peer latent/index rows, len_d and missing lazy pool keys only.
    ///
    /// These are persistent cache obligations, not prompt-dependent scratch. The caller
    /// must retain its workspace/transient reserve on EVERY returned device, even at zero
    /// bytes, and apply peer_prefill_workspace_bytes with the actual prompt length. Capacity alone cannot size prompt-dependent workspace correctly.
    pub fn unmaterialized_state_bytes(
        &self,
        cache: Option<&Cache>,
        capacity: usize,
    ) -> Result<Vec<StepTpKvDeviceAdmission>, String> {
        let capacity = cache.map_or(capacity, |cache| cache.max_ctx);
        let mut charges = DeviceBytes::default();
        for charge in self.step_tp_unmaterialized_kv_bytes(cache, capacity)? {
            charges.add(charge.device, charge.bytes)?;
        }
        for rt in self.step_runtimes() {
            for &device in rt.devices() {
                charges.add(device, 0)?;
            }
        }
        for rt in self.glm_tp_runtimes() {
            for device in rt.devices() {
                charges.add(device, 0)?;
            }
        }
        for (il, (mixer, _)) in self.ownership_layers().enumerate() {
            match mixer {
                Mixer::Kda(kda) if kda.tp.is_some() => {
                    let rt = &kda.tp.as_ref().unwrap().rt;
                    let required = kda_elements(kda.heads(), kda.head_dim(), kda.conv_kernel())?;
                    let planes = match cache {
                        Some(cache) => cache
                            .glm5_tp_recur
                            .get(il)
                            .ok_or_else(|| {
                                format!("GLM TP admission: missing KDA cache slot {il}")
                            })?
                            .as_deref(),
                        None => None,
                    };
                    if let Some(planes) = planes
                        && planes.len() != rt.ranks()
                    {
                        return Err(format!(
                            "GLM TP admission: KDA layer {il} has {} state ranks, expected {}",
                            planes.len(),
                            rt.ranks()
                        ));
                    }
                    for (rank, device) in std::iter::once(rt.root_dev)
                        .chain(rt.peers.iter().map(|engine| engine.ctx().ordinal()))
                        .enumerate()
                    {
                        let live = planes.map(|p| {
                            [
                                p[rank].conv_state.len(),
                                p[rank].ssm_state.len(),
                                p[rank].ssm_state_alt.len(),
                            ]
                        });
                        if let Some(live) = live
                            && live != required
                        {
                            return Err(format!(
                                "GLM TP admission: KDA layer {il} rank {rank} state shape {live:?} != {required:?}"
                            ));
                        }
                        charges.add(
                            device,
                            missing_f32_bytes(
                                &required,
                                live.as_ref().map_or(&[], |x| x.as_slice()),
                            )?,
                        )?;
                    }
                }
                Mixer::Mla(mla) if mla.tp.is_some() => {
                    let rt = &mla.tp.as_ref().unwrap().rt;
                    let (canonical, peers) = match cache {
                        Some(cache) => (
                            Some(cache.latent.get(il).and_then(Option::as_ref).ok_or_else(
                                || format!("GLM TP admission: missing canonical MLA layer {il}"),
                            )?),
                            cache
                                .glm5_tp_latent_peer
                                .get(il)
                                .ok_or_else(|| {
                                    format!("GLM TP admission: missing MLA peer cache slot {il}")
                                })?
                                .as_deref(),
                        ),
                        None => (None, None),
                    };
                    if let Some(peers) = peers
                        && peers.len() != rt.peers.len()
                    {
                        return Err(format!(
                            "GLM TP admission: MLA layer {il} has {} peer states, expected {}",
                            peers.len(),
                            rt.peers.len()
                        ));
                    }
                    let pool = mla.index.as_ref().map_or(0, |index| index.geom.pool);
                    let index_width = mla
                        .index
                        .as_ref()
                        .map_or(0, |index| index.geom.head_dim * 2);
                    let index_rows = canonical.map_or_else(
                        || crate::cache::index_ring_rows(capacity).unwrap_or(capacity),
                        |canonical| canonical.index_ring_rows.unwrap_or(capacity),
                    );
                    let mut required =
                        mla_elements(capacity, mla.geom.latent_dim, index_width, index_rows, pool)?;
                    if let Some(canonical) = canonical {
                        // ensure_mla_peer_latent clones allocation lengths, not logical len.
                        required[0] = canonical.rows.len();
                        required[1] = canonical.index_rows.as_ref().map_or(0, |rows| rows.len());
                    }
                    for (rank, engine) in rt.peers.iter().enumerate() {
                        let live = peers.map(|p| latent_elements(&p[rank]));
                        charges.add(
                            engine.ctx().ordinal(),
                            missing_f32_bytes(
                                &required,
                                live.as_ref().map_or(&[], |x| x.as_slice()),
                            )?,
                        )?;
                    }
                }
                _ => {}
            }
        }
        Ok(charges
            .0
            .into_iter()
            .map(|(device, bytes)| StepTpKvDeviceAdmission { device, bytes })
            .collect())
    }
}

fn latent_elements(layer: &LatentKvLayer) -> [usize; 4] {
    [
        layer.rows.len(),
        layer.index_rows.as_ref().map_or(0, |rows| rows.len()),
        layer.len_d.len(),
        layer.index_pool_keys.as_ref().map_or(0, |keys| keys.len()),
    ]
}

#[cfg(test)]
mod native_tests {
    use super::*;
    use std::sync::Arc;

    /// Native-only ownership/admission/allocator gate. No model support promotion or
    /// numerical qualification. Run on a designated non-serving rig, with a clean env:
    /// Use tools/qualify-model-device-memory.py inside the coordinator's per-card
    /// lock wrapper: same-device stage locks one physical GPU; pair stage locks two.
    /// Record source SHA, binary hash, CUDA/card identity and full output with the receipt.
    fn fixture(same_device: bool) -> (Engine, HybridModel, Arc<Glm5TpRt>) {
        let path = std::env::temp_dir().join(format!(
            "memra-544-{}-{same_device}.gguf",
            std::process::id()
        ));
        memra_gguf::micro_gguf::write_glm_dsa_micro(&path, 544).unwrap();
        use sha2::Digest;
        let bytes = std::fs::read(&path).unwrap();
        eprintln!(
            "[model-memory-544] fixture=glm-dsa-micro seed=544 bytes={} sha256={:x}",
            bytes.len(),
            sha2::Sha256::digest(&bytes)
        );
        let source = memra_gguf::GgufFile::open(&path).unwrap();
        let primary = Engine::new(0).expect("native CUDA device 0 is required; never skip");
        let mut model = HybridModel::load(&primary, &source).unwrap();
        std::fs::remove_file(path).unwrap();
        let rt = Arc::new(if same_device {
            Glm5TpRt::new_gate_same_device(0, 2).unwrap()
        } else {
            Glm5TpRt::new(&[0, 1]).expect("two native CUDA devices are required; never skip")
        });
        model.layers = std::mem::take(&mut model.layers)
            .into_iter()
            .map(|mut layer| {
                layer.mixer = match layer.mixer {
                    Mixer::Mla(mla) => {
                        Mixer::Mla(crate::glm5_tp::shard_mla_layer(&primary, &rt, mla).unwrap())
                    }
                    _ => panic!("the micro fixture must contain MLA layers"),
                };
                layer
            })
            .collect();
        (primary, model, rt)
    }

    #[test]
    #[ignore = "native CUDA required; run under the provided one-card exclusive lock"]
    fn glm_same_ordinal_owners_are_not_lost() {
        let (primary, model, rt) = fixture(true);
        let owners = model.owned_engines(&primary);
        assert!(owners.iter().any(|e| std::ptr::eq(*e, &primary)));
        assert_eq!(
            owners
                .iter()
                .filter(|e| std::ptr::eq(**e, &rt.peers[0]))
                .count(),
            1
        );
        assert_eq!(owners.iter().filter(|e| e.ctx().ordinal() == 0).count(), 2);
        for owner in owners {
            owner.stream().synchronize().unwrap();
        }
    }

    #[test]
    #[ignore = "native CUDA pair required; run under the provided two-card exclusive locks"]
    fn glm_peer_admission_materialization_trim_and_refill() {
        let (primary, model, rt) = fixture(false);
        let capacity = 32;
        let mut cache = Cache::new_planned(&primary, &model.cfg, &model.plan, capacity).unwrap();
        let owners = model.owned_engines(&primary);
        assert!(owners.iter().any(|e| std::ptr::eq(*e, &rt.peers[0])));
        assert_eq!(
            owners
                .iter()
                .filter(|e| std::ptr::eq(**e, &rt.peers[0]))
                .count(),
            1,
            "shared MLA runtime across layers must enumerate once"
        );
        assert_eq!(model.devices(), [0, 1]);
        // Historical failure in the SAME loaded fixture: Step-only accounting is empty.
        let old = model
            .step_tp_unmaterialized_kv_bytes(Some(&cache), capacity)
            .unwrap();
        assert!(
            old.is_empty(),
            "red control must actually exercise the old omission"
        );
        let cold = model.unmaterialized_state_bytes(None, capacity).unwrap();
        assert_eq!(
            cold,
            model
                .unmaterialized_state_bytes(Some(&cache), capacity)
                .unwrap()
        );
        assert_eq!(
            cold.iter().find(|c| c.device == 0).unwrap().bytes,
            0,
            "canonical MLA is already in the base cache budget"
        );
        let cold_peer = cold.iter().find(|c| c.device == 1).unwrap().bytes;
        assert!(
            cold_peer > 0,
            "old Step-only helper cannot satisfy this assertion"
        );
        let mut materialized = 0;
        let mut missing_keys = 0;
        for (il, layer) in model.layers.iter().enumerate() {
            let Mixer::Mla(mla) = &layer.mixer else {
                unreachable!()
            };
            crate::glm5_tp::ensure_mla_peer_latent(
                &rt,
                cache.latent[il].as_ref().unwrap(),
                &mut cache.glm5_tp_latent_peer[il],
            )
            .unwrap();
            let peer = &mut cache.glm5_tp_latent_peer[il].as_mut().unwrap()[0];
            materialized += latent_elements(peer).iter().sum::<usize>() * 4;
            let keys = mla.index.as_ref().map_or(0, |index| {
                (capacity / index.geom.pool * index.geom.head_dim).max(1)
            });
            missing_keys += keys * 4;
        }
        let partial = model
            .unmaterialized_state_bytes(Some(&cache), capacity)
            .unwrap();
        assert_eq!(
            partial.iter().find(|c| c.device == 1).unwrap().bytes,
            missing_keys
        );
        assert_eq!(
            cold_peer,
            materialized + missing_keys,
            "charge must match actual allocations"
        );
        for (il, layer) in model.layers.iter().enumerate() {
            let Mixer::Mla(mla) = &layer.mixer else {
                unreachable!()
            };
            if let Some(index) = &mla.index {
                cache.glm5_tp_latent_peer[il].as_mut().unwrap()[0].index_pool_keys = Some(
                    rt.peers[0]
                        .zeros((capacity / index.geom.pool * index.geom.head_dim).max(1))
                        .unwrap(),
                );
            }
        }
        let warm = model
            .unmaterialized_state_bytes(Some(&cache), capacity)
            .unwrap();
        assert_eq!(
            warm,
            [
                StepTpKvDeviceAdmission {
                    device: 0,
                    bytes: 0
                },
                StepTpKvDeviceAdmission {
                    device: 1,
                    bytes: 0
                }
            ]
        );

        let peer = &rt.peers[0];
        let sentinel = peer.htod(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        for owner in &owners {
            owner.stream().synchronize().unwrap();
        }
        for owner in &owners {
            owner.pool_trim_to_zero();
        }
        let scratch = peer.zeros(16 * 1024 * 1024).unwrap();
        peer.stream().synchronize().unwrap();
        drop(scratch);
        for owner in &owners {
            owner.stream().synchronize().unwrap();
        }
        let before = peer.pool_reserved_used();
        assert!(
            before.0 > before.1,
            "peer-only cached allocation is required for a non-vacuous gate"
        );
        // RED: trimming only primary cannot reclaim the other device's pool.
        primary.pool_trim_to_zero();
        assert_eq!(peer.pool_reserved_used(), before);
        let mut reclaimed = 0;
        for owner in &owners {
            if owner.ctx().ordinal() == 1 {
                reclaimed += owner.pool_trim_to_zero();
            }
        }
        let after = peer.pool_reserved_used();
        assert!(reclaimed > 0);
        assert!(after.0 < before.0);
        assert_eq!(
            after.1, before.1,
            "trim must preserve live state/weights/source owners"
        );
        assert_eq!(
            peer.dtoh_view(&sentinel.slice(..)).unwrap(),
            [1.0, 2.0, 3.0, 4.0]
        );
        let refill = peer.zeros(16 * 1024 * 1024).unwrap();
        peer.stream().synchronize().unwrap();
        assert_eq!(peer.dtoh_view(&refill.slice(0..4)).unwrap(), [0.0; 4]);
        assert_eq!(
            model
                .unmaterialized_state_bytes(Some(&cache), capacity)
                .unwrap(),
            warm
        );
        eprintln!(
            "[model-memory-544] old_step_entries={} cold_peer_bytes={cold_peer} partial_key_bytes={missing_keys} before={before:?} after={after:?} reclaimed={reclaimed} refill=PASS",
            old.len()
        );
        for owner in owners {
            owner.stream().synchronize().unwrap();
        }
    }
    /// Two KDA and two genuinely indexed MLA layers: cold, partial and warm state
    /// obligations must agree with the allocations owned by the real cache. No weights
    /// are downloaded and no model/support/performance claim is made by this gate.
    #[test]
    #[ignore = "requires a native CUDA pair under the coordinator's exact two-card leases"]
    fn glm_indexed_mla_kda_state_materialization() {
        use crate::hybrid_forward::IndexerPlanes;
        use crate::model_memory_fixture::{CAPACITY, FixtureSource, KDA_LAYERS, MLA_LAYERS};
        use std::mem::size_of;

        let source = FixtureSource::new();
        let (tensors, bytes, digest) = source.identity();
        eprintln!(
            "[model-memory-544-indexed] fixture=glm5-next-memory-tp2-v1 tensors={tensors} bytes={bytes} sha256={digest} capacity={CAPACITY}"
        );
        let primary = Engine::new(0).expect("native device 0 required; never skip");
        let mut model = HybridModel::load_from_source(&primary, &source).unwrap();
        assert_eq!(model.plan, source.plan);
        let rt = Arc::new(Glm5TpRt::new(&[0, 1]).expect("native device 1 required; never skip"));
        model.layers = std::mem::take(&mut model.layers)
            .into_iter()
            .map(|mut layer| {
                layer.mixer = match layer.mixer {
                    Mixer::Kda(kda) => {
                        Mixer::Kda(crate::glm5_tp::shard_kda_layer(&primary, &rt, kda).unwrap())
                    }
                    Mixer::Mla(mla) => {
                        assert!(
                            mla.index.is_some(),
                            "fixture must load a real k-pool indexer"
                        );
                        Mixer::Mla(crate::glm5_tp::shard_mla_layer(&primary, &rt, mla).unwrap())
                    }
                    _ => panic!("unexpected fixture mixer"),
                };
                layer
            })
            .collect();
        for il in KDA_LAYERS {
            let Mixer::Kda(kda) = &model.layers[il].mixer else {
                panic!("missing KDA")
            };
            assert_eq!(kda.heads(), 1, "outer layer must be a real TP2 shard");
            assert_eq!(kda.tp.as_ref().unwrap().peers.len(), 1);
        }
        let peer = &rt.peers[0];
        let mut cache = Cache::new_planned(&primary, &model.cfg, &model.plan, CAPACITY).unwrap();
        for il in MLA_LAYERS {
            let plane = cache.latent[il].as_ref().unwrap();
            assert!(plane.index_rows.is_some());
            assert!(plane.index_width > 0);
            assert!(
                plane.index_ring_rows.is_some(),
                "fixture must exercise a physical tail ring"
            );
        }
        let owners = model.owned_engines(&primary);
        assert_eq!(owners.len(), 2);
        let fence = || {
            for owner in &owners {
                owner.stream().synchronize().unwrap();
            }
        };
        // Measure cache-owned allocations independently of the admission formula.
        let allocated = |cache: &Cache| -> [usize; 2] {
            let mut bytes = [0; 2];
            for ranks in cache.glm5_tp_recur.iter().flatten() {
                assert_eq!(ranks.len(), 2);
                for (rank, plane) in ranks.iter().enumerate() {
                    bytes[rank] += (plane.conv_state.len()
                        + plane.ssm_state.len()
                        + plane.ssm_state_alt.len())
                        * size_of::<f32>();
                }
            }
            for peers in cache.glm5_tp_latent_peer.iter().flatten() {
                assert_eq!(peers.len(), 1);
                let plane = &peers[0];
                bytes[1] += (plane.rows.len()
                    + plane.index_rows.as_ref().map_or(0, |rows| rows.len())
                    + plane.index_pool_keys.as_ref().map_or(0, |keys| keys.len()))
                    * size_of::<f32>()
                    + plane.len_d.len() * size_of::<i32>();
            }
            bytes
        };
        let remaining = |cache: Option<&Cache>| -> [usize; 2] {
            let charges = model.unmaterialized_state_bytes(cache, CAPACITY).unwrap();
            assert_eq!(charges.iter().map(|c| c.device).collect::<Vec<_>>(), [0, 1]);
            [charges[0].bytes, charges[1].bytes]
        };
        fence();
        let cold = remaining(None);
        assert_eq!(cold, remaining(Some(&cache)));
        assert_eq!(allocated(&cache), [0, 0]);
        assert!(
            cold[0] > 0 && cold[1] > cold[0],
            "KDA ranks AND indexed peer state must be nonzero"
        );
        let check = |cache: &Cache, label: &str| -> [usize; 2] {
            fence();
            let live = allocated(cache);
            let pending = remaining(Some(cache));
            for rank in 0..2 {
                assert_eq!(
                    live[rank] + pending[rank],
                    cold[rank],
                    "{label}: rank {rank} new real allocations must exactly discharge its prediction"
                );
            }
            eprintln!(
                "[model-memory-544-indexed] phase={label} cold={cold:?} allocated={live:?} remaining={pending:?}"
            );
            pending
        };
        let allocate_kda = |cache: &mut Cache, il: usize| {
            let Mixer::Kda(kda) = &model.layers[il].mixer else {
                panic!("missing KDA")
            };
            crate::glm5_tp::ensure_kda_tp_state(&primary, &rt, kda, cache, il).unwrap();
        };
        let key_bytes = |il: usize| {
            let Mixer::Mla(mla) = &model.layers[il].mixer else {
                panic!("missing MLA")
            };
            let index = mla.tp.as_ref().unwrap().peers[0]
                .index
                .as_ref()
                .expect("real peer indexer");
            (CAPACITY / index.geom.pool * index.geom.head_dim).max(1) * size_of::<f32>()
        };
        let keys = [key_bytes(MLA_LAYERS[0]), key_bytes(MLA_LAYERS[1])];
        assert!(
            keys.iter().all(|&bytes| bytes > 0),
            "zero key bytes cannot qualify indexed MLA"
        );

        allocate_kda(&mut cache, KDA_LAYERS[0]);
        for il in MLA_LAYERS {
            crate::glm5_tp::ensure_mla_peer_latent(
                &rt,
                cache.latent[il].as_ref().unwrap(),
                &mut cache.glm5_tp_latent_peer[il],
            )
            .unwrap();
        }
        let partial_ranks = check(&cache, "partial-ranks-and-keys");
        assert!(partial_ranks[0] > 0 && partial_ranks[0] < cold[0]);
        assert_eq!(
            partial_ranks[1],
            partial_ranks[0] + keys.iter().sum::<usize>()
        );

        // Use the real indexer append/build path, not a test-side keys=zeros assignment.
        let build_keys = |cache: &mut Cache, il: usize, slot: usize| {
            let Mixer::Mla(mla) = &model.layers[il].mixer else {
                panic!("missing MLA")
            };
            let shard = &mla.tp.as_ref().unwrap().peers[0];
            let indexer = shard.index.as_ref().unwrap();
            let t = indexer.geom.pool;
            let h = peer
                .htod(
                    &(0..t * model.cfg.n_embd as usize)
                        .map(|i| (i % 17) as f32 * 0.01 - 0.08)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let q = peer.zeros(t * shard.wq_a.out_features()).unwrap();
            let plane = &mut cache.glm5_tp_latent_peer[il].as_mut().unwrap()[0];
            let (indices, width) = HybridModel::mla_kpool_indices(
                peer,
                indexer,
                &h,
                &q,
                IndexerPlanes {
                    state: plane.index_rows.as_mut().unwrap(),
                    pool_keys: &mut plane.index_pool_keys,
                    ready: &mut plane.index_pools_ready,
                    state_ring_rows: plane.index_ring_rows.unwrap_or(0),
                    capacity_tokens: CAPACITY,
                },
                t,
                slot,
            )
            .unwrap();
            assert!(width > 0 && indices.len() == t * width);
            assert_eq!(plane.index_pools_ready, (slot + t) / indexer.geom.pool);
            let keys = plane
                .index_pool_keys
                .as_ref()
                .expect("native indexer must allocate keys");
            assert_eq!(keys.len() * size_of::<f32>(), key_bytes(il));
            assert!(
                peer.dtoh_view(&keys.slice(..indexer.geom.head_dim))
                    .unwrap()
                    .iter()
                    .all(|v| v.is_finite())
            );
        };
        build_keys(&mut cache, MLA_LAYERS[0], 0);
        let partial_keys = check(&cache, "partial-keys");
        assert_eq!(partial_keys[0], partial_ranks[0]);
        assert_eq!(partial_keys[1], partial_keys[0] + keys[1]);
        assert!(partial_keys[1] > partial_keys[0]);

        allocate_kda(&mut cache, KDA_LAYERS[1]);
        assert_eq!(check(&cache, "keys-only"), [0, keys[1]]);
        build_keys(&mut cache, MLA_LAYERS[1], 0);
        assert_eq!(check(&cache, "warm"), [0, 0]);
        for il in MLA_LAYERS {
            let Mixer::Mla(mla) = &model.layers[il].mixer else {
                unreachable!()
            };
            build_keys(&mut cache, il, mla.index.as_ref().unwrap().geom.pool);
        }
        assert_eq!(check(&cache, "warm-append"), [0, 0]);
        eprintln!(
            "[model-memory-544-indexed] PASS kda_layers=2 indexed_mla_layers=2 ranks=2 cold_primary={} cold_peer={} partial_primary={} partial_peer={} partial_key_bytes={} warm_primary=0 warm_peer=0",
            cold[0], cold[1], partial_ranks[0], partial_ranks[1], keys[1]
        );
        drop(cache);
        fence();
    }
}
