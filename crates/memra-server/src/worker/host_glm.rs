//! Process-lifetime host images of the current model-owned prefix planes.
//! Streams retain their CUDA context, never the demoted device allocation. No
//! process-global TP runtime is consulted: root KDA and peer MLA have different
//! rank indexing, already represented by Glm5TpPrefixShards.
use super::*;
use cudarc::driver::CudaStream;
use memra_engine::cache::LatentPlaneSnapshot;
use memra_engine::glm5_tp::Glm5TpPrefixShards;

type Stream = Arc<CudaStream>;
type HostRecur = Vec<Option<Vec<(Plane, Plane)>>>;

pub(super) fn read_f32(p: &CudaSlice<f32>) -> Result<Vec<f32>, String> {
    let out = p.stream().clone_dtoh(p).map_err(|e| e.to_string())?;
    p.stream().synchronize().map_err(|e| e.to_string())?;
    Ok(out)
}

struct Plane {
    data: memra_engine::PinnedHostBuf,
    stream: Stream,
}
impl Plane {
    fn down(p: &CudaSlice<f32>, planes: &mut HostPlaneLeases) -> Result<Self, String> {
        let mut data = planes.take(p.len() * 4)?;
        data.copy_from_device_f32(p)
            .map_err(|e| format!("pinned GLM copy failed: {e}"))?;
        Ok(Self {
            data,
            stream: p.stream().clone(),
        })
    }
    fn up(&self) -> Result<CudaSlice<f32>, String> {
        self.data
            .to_device_f32(&self.stream)
            .map_err(|e| e.to_string())
    }
}
struct Latent {
    rows: Plane,
    width: usize,
    len: usize,
    index_width: usize,
    index_pool: usize,
    index_tail: Option<Plane>,
    index_pool_keys: Option<Plane>,
    index_pools_ready: usize,
}
impl Latent {
    fn down(p: &LatentPlaneSnapshot, planes: &mut HostPlaneLeases) -> Result<Self, String> {
        Ok(Self {
            rows: Plane::down(&p.rows, planes)?,
            width: p.width,
            len: p.len,
            index_width: p.index_width,
            index_pool: p.index_pool,
            index_tail: p
                .index_tail
                .as_ref()
                .map(|p| Plane::down(p, planes))
                .transpose()?,
            index_pool_keys: p
                .index_pool_keys
                .as_ref()
                .map(|p| Plane::down(p, planes))
                .transpose()?,
            index_pools_ready: p.index_pools_ready,
        })
    }
    fn up(&self) -> Result<LatentPlaneSnapshot, String> {
        Ok(LatentPlaneSnapshot {
            rows: self.rows.up()?,
            width: self.width,
            len: self.len,
            index_width: self.index_width,
            index_pool: self.index_pool,
            index_tail: self.index_tail.as_ref().map(Plane::up).transpose()?,
            index_pool_keys: self.index_pool_keys.as_ref().map(Plane::up).transpose()?,
            index_pools_ready: self.index_pools_ready,
        })
    }
}
struct Tp {
    recur: HostRecur,
    latent_peer: Vec<Option<Vec<Latent>>>,
    bytes: usize,
}
struct Draft {
    layers: Vec<(Plane, Plane)>,
    base: usize,
    rows: usize,
    len: usize,
    row_bytes: usize,
    floor: usize,
}
/// All GLM latent/recurrent/TP/draft plane bytes are cacheable pinned RAM.
/// Only token IDs, logits and small entry metadata remain ordinary host values.
pub(super) struct HostGlmState {
    pool_key: PoolKey,
    root_context: Arc<cudarc::driver::CudaContext>,
    latent: Vec<Option<Latent>>,
    tp: Option<Tp>,
    conv: Vec<Option<Plane>>,
    ssm: Vec<Option<Plane>>,
    draft: Option<Draft>,
}
impl HostGlmState {
    pub(super) fn plane_sizes(e: &PrefixEntry) -> Vec<usize> {
        fn latent(p: &LatentPlaneSnapshot, out: &mut Vec<usize>) {
            out.push(p.rows.len() * 4);
            out.extend(
                p.index_tail
                    .iter()
                    .chain(&p.index_pool_keys)
                    .map(|p| p.len() * 4),
            );
        }
        let mut out = Vec::new();
        for p in e.latent.iter().flatten() {
            latent(p, &mut out);
        }
        if let Some(tp) = &e.tp {
            for (c, s) in tp.recur.iter().flatten().flatten() {
                out.extend([c.len() * 4, s.len() * 4]);
            }
            for p in tp.latent_peer.iter().flatten().flatten() {
                latent(p, &mut out);
            }
        }
        out.extend(e.conv.iter().chain(&e.ssm).flatten().map(|p| p.len() * 4));
        if let Some(d) = &e.dspark_draft {
            for (k, v) in &d.layers {
                out.extend([k.len() * 4, v.len() * 4]);
            }
        }
        out
    }
    pub(super) fn down(
        engine: &Engine,
        e: &PrefixEntry,
        planes: &mut HostPlaneLeases,
    ) -> Result<Self, String> {
        let latent = e
            .latent
            .iter()
            .map(|p| p.as_ref().map(|p| Latent::down(p, planes)).transpose())
            .collect::<Result<_, _>>()?;
        let tp =
            e.tp.as_ref()
                .map(|tp| -> Result<Tp, String> {
                    let recur = tp
                        .recur
                        .iter()
                        .map(|ps| {
                            ps.as_ref()
                                .map(|ps| {
                                    ps.iter()
                                        .map(|(c, s)| {
                                            Ok((Plane::down(c, planes)?, Plane::down(s, planes)?))
                                        })
                                        .collect::<Result<Vec<_>, String>>()
                                })
                                .transpose()
                        })
                        .collect::<Result<_, _>>()?;
                    let latent_peer = tp
                        .latent_peer
                        .iter()
                        .map(|ps| {
                            ps.as_ref()
                                .map(|ps| {
                                    ps.iter()
                                        .map(|p| Latent::down(p, planes))
                                        .collect::<Result<Vec<_>, _>>()
                                })
                                .transpose()
                        })
                        .collect::<Result<_, _>>()?;
                    Ok(Tp {
                        recur,
                        latent_peer,
                        bytes: tp.bytes,
                    })
                })
                .transpose()?;
        Ok(Self {
            root_context: engine.ctx().clone(),
            pool_key: e.pool_key.clone(),
            latent,
            tp,
            conv: e
                .conv
                .iter()
                .map(|p| p.as_ref().map(|p| Plane::down(p, planes)).transpose())
                .collect::<Result<_, _>>()?,
            ssm: e
                .ssm
                .iter()
                .map(|p| p.as_ref().map(|p| Plane::down(p, planes)).transpose())
                .collect::<Result<_, _>>()?,
            draft: e
                .dspark_draft
                .as_ref()
                .map(|d| -> Result<Draft, String> {
                    Ok(Draft {
                        layers: d
                            .layers
                            .iter()
                            .map(|(k, v)| Ok((Plane::down(k, planes)?, Plane::down(v, planes)?)))
                            .collect::<Result<_, String>>()?,
                        base: d.base,
                        rows: d.rows,
                        len: d.len,
                        row_bytes: d.row_bytes,
                        floor: d.floor,
                    })
                })
                .transpose()?,
        })
    }
    pub(super) fn validate_owner(&self, engine: &Engine, key: &PoolKey) -> Result<(), String> {
        if &self.pool_key != key {
            return Err("host GLM model/namespace identity mismatch".into());
        }
        if !Arc::ptr_eq(&self.root_context, engine.ctx()) {
            return Err("host GLM entry belongs to a different worker CUDA context".into());
        }
        Ok(())
    }
    pub(super) fn state_bytes(&self) -> usize {
        fn latent_bytes(p: &Latent) -> usize {
            p.rows.data.len()
                + p.index_tail.as_ref().map_or(0, |p| p.data.len())
                + p.index_pool_keys.as_ref().map_or(0, |p| p.data.len())
        }
        let primary: usize = self.latent.iter().flatten().map(latent_bytes).sum();
        primary
            + self
                .conv
                .iter()
                .chain(&self.ssm)
                .flatten()
                .map(|p| p.data.len())
                .sum::<usize>()
            + self.draft.as_ref().map_or(0, |d| {
                d.layers
                    .iter()
                    .map(|(k, v)| k.data.len() + v.data.len())
                    .sum()
            })
            + self.tp.as_ref().map_or(0, |tp| {
                tp.recur
                    .iter()
                    .flatten()
                    .flatten()
                    .map(|(c, s)| c.data.len() + s.data.len())
                    .sum::<usize>()
                    + tp.latent_peer
                        .iter()
                        .flatten()
                        .flatten()
                        .map(latent_bytes)
                        .sum::<usize>()
            })
    }
    pub(super) fn latent_up(&self) -> Result<Vec<Option<LatentPlaneSnapshot>>, String> {
        self.latent
            .iter()
            .map(|p| p.as_ref().map(Latent::up).transpose())
            .collect()
    }
    pub(super) fn tp_up(&self) -> Result<Option<Glm5TpPrefixShards>, String> {
        self.tp
            .as_ref()
            .map(|tp| {
                let recur = tp
                    .recur
                    .iter()
                    .map(|ps| {
                        ps.as_ref()
                            .map(|ps| {
                                ps.iter()
                                    .map(|(c, s)| Ok((c.up()?, s.up()?)))
                                    .collect::<Result<Vec<_>, String>>()
                            })
                            .transpose()
                    })
                    .collect::<Result<_, _>>()?;
                let latent_peer = tp
                    .latent_peer
                    .iter()
                    .map(|ps| {
                        ps.as_ref()
                            .map(|ps| ps.iter().map(Latent::up).collect::<Result<Vec<_>, _>>())
                            .transpose()
                    })
                    .collect::<Result<_, _>>()?;
                Ok(Glm5TpPrefixShards {
                    recur,
                    latent_peer,
                    bytes: tp.bytes,
                })
            })
            .transpose()
    }
    pub(super) fn conv_up(&self) -> Result<Vec<Option<CudaSlice<f32>>>, String> {
        self.conv
            .iter()
            .map(|p| p.as_ref().map(Plane::up).transpose())
            .collect()
    }
    pub(super) fn ssm_up(&self) -> Result<Vec<Option<CudaSlice<f32>>>, String> {
        self.ssm
            .iter()
            .map(|p| p.as_ref().map(Plane::up).transpose())
            .collect()
    }
    pub(super) fn draft_up(&self) -> Result<Option<memra_engine::dflash::DflashKvTail>, String> {
        self.draft
            .as_ref()
            .map(|d| {
                Ok(memra_engine::dflash::DflashKvTail {
                    layers: d
                        .layers
                        .iter()
                        .map(|(k, v)| Ok((k.up()?, v.up()?)))
                        .collect::<Result<_, String>>()?,
                    base: d.base,
                    rows: d.rows,
                    len: d.len,
                    row_bytes: d.row_bytes,
                    floor: d.floor,
                })
            })
            .transpose()
    }
}

/// Host-tier digest covers namespace, shape, rank/device identity, all logical
/// planes, boundary logits and draft tail. It is separate from the legacy split
/// oracle, whose domain intentionally excludes TP/draft state.
pub(super) fn digest(e: &PrefixEntry) -> Result<String, String> {
    fn words(h: &mut Sha256, p: &CudaSlice<f32>) -> Result<(), String> {
        digest_usize(h, p.ordinal());
        digest_usize(h, p.len());
        for start in (0..p.len()).step_by(1 << 20) {
            let end = (start + (1 << 20)).min(p.len());
            let data = p
                .stream()
                .clone_dtoh(&p.slice(start..end))
                .map_err(|e| e.to_string())?;
            p.stream().synchronize().map_err(|e| e.to_string())?;
            for x in data {
                h.update(x.to_bits().to_le_bytes())
            }
        }
        Ok(())
    }
    fn bytes(h: &mut Sha256, p: &CudaSlice<u8>) -> Result<(), String> {
        digest_usize(h, p.ordinal());
        digest_usize(h, p.len());
        let data = p.stream().clone_dtoh(p).map_err(|e| e.to_string())?;
        p.stream().synchronize().map_err(|e| e.to_string())?;
        h.update(data);
        Ok(())
    }
    fn latent(h: &mut Sha256, p: &LatentPlaneSnapshot) -> Result<(), String> {
        for x in [
            p.width,
            p.len,
            p.index_width,
            p.index_pool,
            p.index_pools_ready,
        ] {
            digest_usize(h, x)
        }
        words(h, &p.rows)?;
        for p in [&p.index_tail, &p.index_pool_keys] {
            match p {
                Some(p) => {
                    h.update([1]);
                    words(h, p)?
                }
                None => h.update([0]),
            }
        }
        Ok(())
    }
    fn plane(h: &mut Sha256, p: &PrefixPlane) -> Result<(), String> {
        for x in [p.len, p.k_tok_bytes, p.v_tok_bytes] {
            digest_usize(h, x)
        }
        bytes(h, &p.k)?;
        bytes(h, &p.v)
    }
    let mut h = Sha256::new();
    h.update(b"memra-host-prefix-state-v1");
    for s in [&e.pool_key.0, &e.pool_key.1] {
        digest_usize(&mut h, s.len());
        h.update(s.as_bytes())
    }
    for x in [e.layout_version as usize, e.pos, e.bytes, e.toks.len()] {
        digest_usize(&mut h, x)
    }
    for t in &e.toks {
        h.update(t.to_le_bytes())
    }
    digest_f32_plane(&mut h, &e.last_logits);
    digest_f32_plane(&mut h, &e.last_h);
    digest_usize(&mut h, e.kv.len());
    for p in &e.kv {
        match p {
            Some(p) => {
                h.update([1]);
                plane(&mut h, p)?
            }
            None => h.update([0]),
        }
    }
    for ps in [&e.conv, &e.ssm] {
        digest_usize(&mut h, ps.len());
        for p in ps {
            match p {
                Some(p) => {
                    h.update([1]);
                    words(&mut h, p)?
                }
                None => h.update([0]),
            }
        }
    }
    digest_usize(&mut h, e.latent.len());
    for p in &e.latent {
        match p {
            Some(p) => {
                h.update([1]);
                latent(&mut h, p)?
            }
            None => h.update([0]),
        }
    }
    match &e.tp {
        Some(tp) => {
            h.update([1]);
            digest_usize(&mut h, tp.bytes);
            digest_usize(&mut h, tp.recur.len());
            for ps in &tp.recur {
                match ps {
                    Some(ps) => {
                        h.update([1]);
                        digest_usize(&mut h, ps.len());
                        for (c, s) in ps {
                            words(&mut h, c)?;
                            words(&mut h, s)?
                        }
                    }
                    None => h.update([0]),
                }
            }
            digest_usize(&mut h, tp.latent_peer.len());
            for ps in &tp.latent_peer {
                match ps {
                    Some(ps) => {
                        h.update([1]);
                        digest_usize(&mut h, ps.len());
                        for p in ps {
                            latent(&mut h, p)?
                        }
                    }
                    None => h.update([0]),
                }
            }
        }
        None => h.update([0]),
    }
    match &e.draft {
        Some(p) => {
            h.update([1]);
            plane(&mut h, p)?
        }
        None => h.update([0]),
    }
    match &e.dspark_draft {
        Some(t) => {
            h.update([1]);
            for x in [t.base, t.rows, t.len, t.row_bytes, t.floor, t.layers.len()] {
                digest_usize(&mut h, x)
            }
            for (k, v) in &t.layers {
                words(&mut h, k)?;
                words(&mut h, v)?
            }
        }
        None => h.update([0]),
    }
    Ok(format!("{:x}", h.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn latent(e: &Engine, base: f32) -> LatentPlaneSnapshot {
        LatentPlaneSnapshot {
            rows: e
                .htod(&(0..256).map(|i| base + i as f32).collect::<Vec<_>>())
                .unwrap(),
            width: 2,
            len: 128,
            index_width: 2,
            index_pool: 2,
            index_pools_ready: 64,
            index_tail: None,
            index_pool_keys: Some(
                e.htod(&(0..64).map(|i| base + i as f32).collect::<Vec<_>>())
                    .unwrap(),
            ),
        }
    }
    fn entry(root: &Engine, peer: &Engine) -> PrefixEntry {
        let tp = Glm5TpPrefixShards {
            recur: vec![Some(vec![
                (
                    root.htod(&[10.0, 11.0]).unwrap(),
                    root.htod(&[12.0, 13.0]).unwrap(),
                ),
                (
                    peer.htod(&[20.0, 21.0]).unwrap(),
                    peer.htod(&[22.0, 23.0]).unwrap(),
                ),
            ])],
            latent_peer: vec![Some(vec![latent(peer, 30.0)])],
            bytes: 1312,
        };
        PrefixEntry {
            layout_version: PREFIX_ENTRY_LAYOUT_VERSION,
            pool_key: ("model-a".into(), "tenant-a\u{1f}ns".into()),
            toks: (0..128).collect(),
            kv: vec![Some(PrefixPlane {
                k: root.ctx().default_stream().clone_htod(&[71u8; 8]).unwrap(),
                v: root.ctx().default_stream().clone_htod(&[81u8; 8]).unwrap(),
                len: 8,
                k_tok_bytes: 1,
                v_tok_bytes: 1,
            })],
            conv: vec![Some(peer.htod(&[40.0]).unwrap())],
            ssm: vec![Some(peer.htod(&[41.0]).unwrap())],
            latent: vec![Some(latent(root, 50.0))],
            tp: Some(tp),
            pos: 128,
            last_logits: vec![0.25, 0.75],
            draft: None,
            dspark_draft: Some(memra_engine::dflash::DflashKvTail {
                layers: vec![(
                    peer.htod(&[60.0, 61.0]).unwrap(),
                    peer.htod(&[62.0, 63.0]).unwrap(),
                )],
                base: 0,
                rows: 2,
                len: 2,
                row_bytes: 4,
                floor: 0,
            }),
            last_h: vec![0.125],
            bytes: 2632,
            last_use: Instant::now(),
            id: 0,
            segment: PrefixSegment::Probation,
            pins: 0,
        }
    }

    /// Requires two devices. It executes the ACTUAL server demote/promote
    /// constructors over root+peer KDA, peer latent and cross-device draft state.
    #[test]
    #[ignore = "requires two CUDA devices; run the host-cache qualification gate explicitly"]
    fn host_glm_two_device_roundtrip_ownership_and_namespace() {
        let root = Engine::new(0).expect("host GLM gate requires device0");
        let peer = Engine::new(1).expect("host GLM gate requires device1");
        unsafe { std::env::set_var("MEMRA_GLM5_TP_KV_HOST", "1") };
        let src = entry(&root, &peer);
        let want = digest(&src).unwrap();
        let mut pool = HostPrefixCache::new(1 << 20);
        super::host_memory::check_headroom(pool.budget).unwrap();
        let start = Instant::now();
        pool.arena =
            Some(memra_engine::PinnedHostArena::reserve(root.ctx().clone(), pool.budget).unwrap());
        eprintln!(
            "[prefix-host DEBUG] arena startup: reserve_ms={:.3} capacity={} leased=0 free={}",
            start.elapsed().as_secs_f64() * 1000.0,
            pool.budget,
            pool.budget
        );
        pool.model_generations
            .insert("model-a".into(), Arc::new(()));
        let mut host = host_entry_from_device(&root, &mut pool, &src, Some(want.clone())).unwrap();
        drop(src);
        // Host images retain streams/contexts, not source CUDA buffers.
        let restored = device_entry_from_host(&root, &host).unwrap();
        assert_eq!(digest(&restored).unwrap(), want);
        let tp = restored.tp.as_ref().unwrap();
        assert_eq!(tp.recur[0].as_ref().unwrap().len(), 2);
        assert_eq!(tp.recur[0].as_ref().unwrap()[0].0.ordinal(), 0);
        assert_eq!(tp.recur[0].as_ref().unwrap()[1].0.ordinal(), 1);
        assert_eq!(tp.latent_peer[0].as_ref().unwrap()[0].rows.ordinal(), 1);
        assert_eq!(restored.conv[0].as_ref().unwrap().ordinal(), 1);
        assert_eq!(
            restored.dspark_draft.as_ref().unwrap().layers[0]
                .0
                .ordinal(),
            1
        );
        assert!(
            host.glm
                .as_ref()
                .unwrap()
                .validate_owner(&peer, &host.pool_key)
                .is_err()
        );
        // A mutation of rank1, namespace or boundary logits must be observable.
        host.glm.as_mut().unwrap().tp.as_mut().unwrap().recur[0]
            .as_mut()
            .unwrap()[1]
            .1
            .data
            .as_mut_slice()[0] ^= 1;
        let bad = device_entry_from_host(&root, &host).unwrap();
        assert_ne!(digest(&bad).unwrap(), want);
        drop(bad);
        host.glm.as_mut().unwrap().tp.as_mut().unwrap().recur[0]
            .as_mut()
            .unwrap()[1]
            .1
            .data
            .as_mut_slice()[0] ^= 1;
        host.pool_key.1 = "tenant-b".into();
        assert!(device_entry_from_host(&root, &host).is_err());
        host.pool_key.1 = "tenant-a\u{1f}ns".into();
        host.last_logits[0] += 1.0;
        assert_ne!(
            digest(&device_entry_from_host(&root, &host).unwrap()).unwrap(),
            want
        );
        // Missing owner must fail after partial restoration without publishing
        // an entry. The original valid host image remains intact and retryable.
        host.last_logits[0] -= 1.0;
        let data = std::mem::replace(
            &mut host.glm.as_mut().unwrap().ssm[0].as_mut().unwrap().data,
            memra_engine::PinnedHostBuf::new(3).unwrap(),
        );
        assert!(device_entry_from_host(&root, &host).is_err());
        host.glm.as_mut().unwrap().ssm[0].as_mut().unwrap().data = data;
        assert_eq!(
            digest(&device_entry_from_host(&root, &host).unwrap()).unwrap(),
            want
        );
        // Actual promote hook: unrelated namespace cannot probe this entry.
        let key = host.pool_key.clone();
        let prompt = host.toks.clone();
        let mut px = PrefixCache::default();
        assert!(pool.insert(&key, host));
        assert!(
            host_promote_prefix_hit(
                &root,
                &mut px,
                &mut pool,
                &("model-a".into(), "tenant-b".into()),
                &prompt,
                0
            )
            .is_none()
        );
        assert_eq!(pool.promotions, 0);
        assert!(host_promote_prefix_hit(&root, &mut px, &mut pool, &key, &prompt, 0).is_some());
        assert_eq!(pool.promotions, 1);
        // A new artifact/model instance under the SAME name invalidates the old
        // host image before allocating, independently of VERIFY.
        pool.model_generations.insert(key.0.clone(), Arc::new(()));
        let mut empty = PrefixCache::default();
        assert!(host_promote_prefix_hit(&root, &mut empty, &mut pool, &key, &prompt, 0).is_none());
        assert_eq!(pool.n_entries(), 0);
        assert_eq!(empty.n_entries(), 0);
        // Corruption must be rejected by the real hook, not merely hash unequal.
        let fresh = entry(&root, &peer);
        let want = digest(&fresh).unwrap();
        let mut damaged = host_entry_from_device(&root, &mut pool, &fresh, Some(want)).unwrap();
        damaged.glm.as_mut().unwrap().tp.as_mut().unwrap().recur[0]
            .as_mut()
            .unwrap()[1]
            .1
            .data
            .as_mut_slice()[0] ^= 1;
        assert!(pool.insert(&key, damaged));
        assert!(host_promote_prefix_hit(&root, &mut empty, &mut pool, &key, &prompt, 0).is_none());
        assert_eq!(pool.digest_mismatches, 1);
        assert_eq!(pool.n_entries(), 0);
        assert_eq!(empty.n_entries(), 0);
        // Actual snapshot preflight must demote its capacity victim before
        // reserving a new snapshot. The leased case must not evict or copy.
        let before = entry(&root, &peer);
        let expected = digest(&before).unwrap();
        let size = before.bytes;
        let mut preflight = PrefixCache::default();
        preflight.insert_with_budget(&key, before, "preflight-fixture", size);
        let lease = preflight.pin_n(&key, 0, 1).unwrap();
        let demotions = pool.demotions;
        {
            let mut sink = |dead| host_demote_prefix_entry(&root, &mut pool, dead);
            assert!(!preflight.prepare_snapshot(size, size, false, Some(&mut sink)));
        }
        assert_eq!(pool.demotions, demotions);
        let mut lease = Some(lease);
        retire_prefix_pin(&mut preflight, &mut lease);
        {
            let mut sink = |dead| host_demote_prefix_entry(&root, &mut pool, dead);
            assert!(preflight.prepare_snapshot(size, size, false, Some(&mut sink)));
        }
        assert_eq!(preflight.total_bytes, 0);
        assert_eq!(pool.demotions, demotions + 1);
        assert_eq!(pool.total_bytes, size);
        let (index, _) =
            host_promote_prefix_hit(&root, &mut preflight, &mut pool, &key, &prompt, 0).unwrap();
        assert_eq!(digest(&preflight.entries[&key][index]).unwrap(), expected);
        eprintln!(
            "host-glm: prepare_snapshot reserved bytes, respected lease, demoted and promoted real state PASS"
        );
        // Evict every image, poison the full backing, then reuse exactly that
        // storage. Every owner plane must overwrite poison before publication.
        pool.entries.clear();
        pool.lru.clear();
        pool.total_bytes = 0;
        pool.tenant_bytes.clear();
        let arena = pool.arena.as_ref().unwrap().clone();
        assert_eq!(arena.bytes(), (pool.budget, 0, pool.budget));
        let mut poison = arena.try_reserve_planes(&[pool.budget]).unwrap();
        poison[0].as_mut_slice().fill(0xa5);
        drop(poison);
        let recycled = entry(&root, &peer);
        let want = digest(&recycled).unwrap();
        let host = host_entry_from_device(&root, &mut pool, &recycled, Some(want.clone())).unwrap();
        assert_eq!(arena.bytes().1, recycled.bytes);
        assert_eq!(
            digest(&device_entry_from_host(&root, &host).unwrap()).unwrap(),
            want
        );
        drop(host);
        assert_eq!(arena.bytes(), (pool.budget, 0, pool.budget));
        eprintln!(
            "host-glm: startup victim + recycled victim + full plane overwrite + unchanged digest PASS"
        );
        // Typed pinned helpers reject partial floats and safely handle zero rows.
        let zero = root.htod(&[]).unwrap();
        let pinned = memra_engine::PinnedHostBuf::from_device_f32(&zero).unwrap();
        assert!(pinned.to_device_f32(zero.stream()).unwrap().is_empty());
        assert!(
            memra_engine::PinnedHostBuf::new(3)
                .unwrap()
                .to_device_f32(zero.stream())
                .is_err()
        );
        eprintln!(
            "host-glm: actual promote, generation/tenant isolation, rank corruption, pinned zero/error and partial restore PASS"
        );
    }
}
