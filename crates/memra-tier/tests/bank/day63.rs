//! Day 63 (`research/spill-c-20260919/DAY63.md` section 1): the bank stage clock's split is log only. Every write of
//! a day-63 field is an addition of a bracket's nanoseconds, the retire side's three calls still add to `retire_ns`,
//! and the line keeps the day-40 fields first, in their order.
const RESIDENCY: &str = include_str!("../../src/bank/residency.rs");

#[test]
fn the_split_only_adds_nanoseconds() {
    let code = &RESIDENCY[..RESIDENCY.find("#[cfg(test)]").unwrap_or(RESIDENCY.len())];
    for field in [
        "stage_lookup_ns",
        "stage_cache_ns",
        "stage_charge_ns",
        "publish_output_ns",
        "publish_policy_ns",
        "host_use_ns",
        "retire_only_ns",
        "ack_ns",
        "ack_release_ns",
    ] {
        let writes = code.matches(&format!("c.{field} ")).count();
        let adds = code.matches(&format!("c.{field} += ns")).count();
        assert_eq!(writes, 1, "{field}: one write");
        assert_eq!(adds, 1, "{field}: the write adds a bracket");
    }
    assert_eq!(code.matches("c.retire_ns += ns").count(), 3);
    let line = &code[code.find("pub fn line(&self)").unwrap()..];
    let format = &line[..line.find(")\n").unwrap()];
    assert!(format.contains(
        "stages={} stage_ns={} alloc_ns={} steps={} step_ns={} verified={} verify_ns={} publish_ns={} retire_ns={} collect_ns={} "
    ));
}
