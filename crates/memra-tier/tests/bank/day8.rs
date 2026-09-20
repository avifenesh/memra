//! Recorded native host-bank demands replayed through the CPU SLRU and fake
//! transfer lifetime. This is NOT a reconstruction of GPU cache hits/routing,
//! nor numerical execution of the model. The eight-byte fake payload is only a
//! lifetime probe; policy admission uses the original recorded expert byte size.
use super::*;

fn replay(case: &str) {
    use sha2::{Digest as _, Sha256};
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../research/spill-c-20260919/rented-5090-20260919/day8");
    let log = std::fs::read_to_string(root.join(case).join("command.log")).unwrap();
    let capture: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join(case).join("command.capture.json")).unwrap(),
    )
    .unwrap();
    let hash: String = Sha256::digest(log.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(hash, capture["raw_log"]["sha256"].as_str().unwrap());
    let mut policy = SlruPolicy::new(&[(860160, 16)]).unwrap();
    let mut ids = BTreeMap::new();
    let mut names = BTreeMap::new();
    let mut sizes = BTreeMap::new();
    let mut seen = std::collections::BTreeSet::new();
    let (mut decisions, mut evictions, mut rereads, mut transfers) = (0, 0, 0, 0);
    for line in log
        .lines()
        .filter_map(|line| line.strip_prefix("[expert-host-slru] "))
    {
        let fields: BTreeMap<_, _> = line
            .split_whitespace()
            .map(|s| s.split_once('=').unwrap())
            .collect();
        let key = fields["key"];
        let bytes = fields["bytes"].parse::<u64>().unwrap();
        let expected_slot = fields["slot"].parse::<usize>().unwrap();
        let expected_hit = fields["hit"].parse::<bool>().unwrap();
        let next = ids.len() as u32;
        let id = ids
            .entry(key.to_owned())
            .or_insert_with(|| bank_id(next, &layout(u64::from(next), 13, bytes - 4)))
            .clone();
        names.insert(id.clone(), key.to_owned());
        assert_eq!(*sizes.entry(key.to_owned()).or_insert(bytes), bytes);
        let hit = policy.hit(&id);
        assert_eq!(hit, expected_hit, "{case} decision {decisions}");
        if hit {
            assert_eq!(policy.resident(&id), Some(expected_slot));
            assert_eq!(fields["victim"], "-");
        } else {
            rereads += usize::from(seen.contains(key));
            let decision = policy.reserve(&id, bytes, &[]).unwrap().unwrap();
            assert_eq!(decision.slot, expected_slot, "{case} decision {decisions}");
            let victim = decision
                .evicted
                .as_ref()
                .map(|old| names.get(old).unwrap().as_str())
                .unwrap_or("-");
            assert_eq!(victim, fields["victim"], "{case} decision {decisions}");
            evictions += usize::from(decision.evicted.is_some());
            let transfer = device_rows::FrozenUpload::new(next as u8, 8);
            // Only a consumer-ready fake completion can precede residency publication.
            transfer.finish(true);
            assert_eq!(policy.publish(&id).unwrap(), expected_slot);
            transfers += 1;
        }
        seen.insert(key.to_owned());
        decisions += 1;
    }
    assert!(decisions > 0 && evictions > 0 && rereads > 0);
    let verdict: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join(format!("{case}-verdict.json"))).unwrap())
            .unwrap();
    let row = &verdict["cells"][0];
    assert_eq!(row["host_demands"].as_u64().unwrap(), decisions as u64);
    assert_eq!(row["host_evictions"].as_u64().unwrap(), evictions as u64);
    assert_eq!(row["rereads"].as_u64().unwrap(), rereads as u64);
    assert_eq!(row["physical_reads"].as_u64().unwrap(), transfers as u64);
    println!(
        "{case}: {decisions} decisions, {evictions} evictions, {rereads} rereads, {transfers} fake transfers"
    );
}

#[test]
fn banked_residency_8g_gen_pressure() {
    replay("8g-gen-on");
}
#[test]
fn banked_residency_8g_spec_pressure() {
    replay("8g-spec-on");
}
#[test]
fn banked_residency_4g_gen_pressure() {
    replay("4g-gen-on");
}
#[test]
fn banked_residency_4g_spec_pressure() {
    replay("4g-spec-on");
}
