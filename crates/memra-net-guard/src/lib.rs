//! memra-net-guard: SSRF address-range classification for server-initiated outbound
//! fetches (memra#533: `image_url` accepts an `http(s)` URL, gated behind
//! `MEMRA_FETCH_URLS`).
//!
//! Pure host logic, zero I/O and zero async runtime dependency, deliberately: the fetch
//! path in `memra-server` resolves a hostname to one or more addresses (once for the
//! initial request, again on every redirect hop, because each hop is a fresh connection),
//! then asks THIS classifier about the RESOLVED address, never the hostname string. A
//! hostname says nothing about where a DNS answer or a redirect's `Location` header
//! actually points, and that is the whole DNS-rebinding class of bypass: checking the
//! string instead of the address is how it survives a filter that looks correct.
//!
//! The ranges below cover what memra#533 named: loopback, RFC1918 private, link-local
//! (also where the AWS/GCP/Azure/DigitalOcean cloud metadata endpoint
//! `169.254.169.254` lives), and IPv6 unique-local. They also cover the ranges every SSRF
//! bypass survey lists next to them: carrier-grade NAT (100.64.0.0/10, where Alibaba
//! Cloud's metadata endpoint `100.100.100.200` lives), multicast, unspecified, broadcast,
//! the IETF documentation/benchmarking ranges, and IPv4-mapped IPv6 embeddings of any of
//! the above (`::ffff:127.0.0.1` reaches the same host as `127.0.0.1`; a filter that only
//! looks at `Ipv4Addr` misses it entirely).

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

/// Why an address was refused. `as_str` is the wire-visible token in the 400's
/// `image_url_blocked` detail: stable, so a caller building a dashboard can match on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockReason {
    Loopback,
    Private,
    LinkLocal,
    CarrierGradeNat,
    UniqueLocal,
    Multicast,
    Unspecified,
    Broadcast,
    Documentation,
    Benchmarking,
}

impl BlockReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            BlockReason::Loopback => "loopback",
            BlockReason::Private => "private",
            BlockReason::LinkLocal => "link_local",
            BlockReason::CarrierGradeNat => "carrier_grade_nat",
            BlockReason::UniqueLocal => "unique_local",
            BlockReason::Multicast => "multicast",
            BlockReason::Unspecified => "unspecified",
            BlockReason::Broadcast => "broadcast",
            BlockReason::Documentation => "documentation",
            BlockReason::Benchmarking => "benchmarking",
        }
    }
}

impl std::fmt::Display for BlockReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Classify a RESOLVED address. `None` means the address is fetch-eligible on address
/// range alone (the caller's host allowlist, if any, is a separate, later decision).
pub fn classify_ip(ip: IpAddr) -> Option<BlockReason> {
    match ip {
        IpAddr::V4(v4) => classify_v4(v4),
        IpAddr::V6(v6) => classify_v6(v6),
    }
}

/// Convenience predicate over `classify_ip`.
pub fn is_blocked(ip: IpAddr) -> bool {
    classify_ip(ip).is_some()
}

fn classify_v4(ip: Ipv4Addr) -> Option<BlockReason> {
    if ip.is_loopback() {
        return Some(BlockReason::Loopback);
    }
    if ip.is_unspecified() {
        return Some(BlockReason::Unspecified);
    }
    if ip.is_broadcast() {
        return Some(BlockReason::Broadcast);
    }
    if ip.is_multicast() {
        return Some(BlockReason::Multicast);
    }
    // 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16 (std's own RFC1918 predicate).
    if ip.is_private() {
        return Some(BlockReason::Private);
    }
    // 169.254.0.0/16: link-local, and the range every major cloud's metadata endpoint
    // (169.254.169.254) is carved out of.
    if ip.is_link_local() {
        return Some(BlockReason::LinkLocal);
    }
    let o = ip.octets();
    // 100.64.0.0/10: RFC 6598 shared/carrier-grade NAT space. Alibaba Cloud's metadata
    // endpoint (100.100.100.200) lives here, outside RFC1918 and outside link-local.
    if o[0] == 100 && (64..=127).contains(&o[1]) {
        return Some(BlockReason::CarrierGradeNat);
    }
    // IETF documentation/protocol ranges: never a real production destination, and a
    // frequent typo-adjacent target (192.0.2.0/24 TEST-NET-1, 198.51.100.0/24
    // TEST-NET-2, 203.0.113.0/24 TEST-NET-3, 192.0.0.0/24 IETF protocol assignments).
    if (o[0] == 192 && o[1] == 0 && (o[2] == 0 || o[2] == 2))
        || (o[0] == 198 && o[1] == 51 && o[2] == 100)
        || (o[0] == 203 && o[1] == 0 && o[2] == 113)
    {
        return Some(BlockReason::Documentation);
    }
    // 198.18.0.0/15: benchmarking, never a legitimate fetch target.
    if o[0] == 198 && (o[1] == 18 || o[1] == 19) {
        return Some(BlockReason::Benchmarking);
    }
    None
}

fn classify_v6(ip: Ipv6Addr) -> Option<BlockReason> {
    // ::ffff:a.b.c.d: an IPv4-mapped IPv6 address reaches the identical host as the
    // embedded IPv4 address. Unwrap and recurse so `::ffff:127.0.0.1` classifies exactly
    // like `127.0.0.1`; checking only `Ipv6Addr`-shaped predicates on this form misses it.
    if let Some(v4) = ip.to_ipv4_mapped() {
        return classify_v4(v4);
    }
    if ip.is_loopback() {
        return Some(BlockReason::Loopback);
    }
    if ip.is_unspecified() {
        return Some(BlockReason::Unspecified);
    }
    if ip.is_multicast() {
        return Some(BlockReason::Multicast);
    }
    let seg = ip.segments();
    // fc00::/7: unique local (the IPv6 analog of RFC1918 private space).
    if (seg[0] & 0xfe00) == 0xfc00 {
        return Some(BlockReason::UniqueLocal);
    }
    // fe80::/10: link-local.
    if (seg[0] & 0xffc0) == 0xfe80 {
        return Some(BlockReason::LinkLocal);
    }
    // 64:ff9b::/96: the well-known NAT64 prefix; the low 32 bits carry an embedded IPv4
    // address reaching that same host, same bypass shape as the mapped form above.
    if seg[0] == 0x0064 && seg[1] == 0xff9b && seg[2] == 0 && seg[3] == 0 && seg[4] == 0 {
        let v4 = Ipv4Addr::new(
            (seg[6] >> 8) as u8,
            (seg[6] & 0xff) as u8,
            (seg[7] >> 8) as u8,
            (seg[7] & 0xff) as u8,
        );
        return classify_v4(v4);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, Ipv6Addr};
    use std::str::FromStr;

    fn v4(s: &str) -> IpAddr {
        IpAddr::V4(Ipv4Addr::from_str(s).unwrap())
    }
    fn v6(s: &str) -> IpAddr {
        IpAddr::V6(Ipv6Addr::from_str(s).unwrap())
    }

    #[test]
    fn v4_loopback_blocked() {
        assert_eq!(classify_ip(v4("127.0.0.1")), Some(BlockReason::Loopback));
        assert_eq!(
            classify_ip(v4("127.255.255.255")),
            Some(BlockReason::Loopback)
        );
    }

    #[test]
    fn v4_rfc1918_private_ranges_blocked() {
        assert_eq!(classify_ip(v4("10.0.0.1")), Some(BlockReason::Private));
        assert_eq!(classify_ip(v4("172.16.0.1")), Some(BlockReason::Private));
        assert_eq!(
            classify_ip(v4("172.31.255.254")),
            Some(BlockReason::Private)
        );
        assert_eq!(classify_ip(v4("192.168.1.1")), Some(BlockReason::Private));
        // 172.32.0.0 is just outside the /12 and must NOT be blocked by this rule.
        assert_eq!(classify_ip(v4("172.32.0.1")), None);
    }

    #[test]
    fn v4_link_local_and_cloud_metadata_blocked() {
        assert_eq!(classify_ip(v4("169.254.1.1")), Some(BlockReason::LinkLocal));
        // AWS/GCP/Azure/DigitalOcean metadata endpoint, the exact address memra#533
        // names as a metadata range, and it lives in 169.254.0.0/16.
        assert_eq!(
            classify_ip(v4("169.254.169.254")),
            Some(BlockReason::LinkLocal)
        );
    }

    #[test]
    fn v4_carrier_grade_nat_and_alibaba_metadata_blocked() {
        assert_eq!(
            classify_ip(v4("100.64.0.1")),
            Some(BlockReason::CarrierGradeNat)
        );
        // Alibaba Cloud's metadata endpoint sits in 100.64.0.0/10, outside RFC1918 and
        // outside link-local: the range memra#533 asks for by name ("metadata ranges").
        assert_eq!(
            classify_ip(v4("100.100.100.200")),
            Some(BlockReason::CarrierGradeNat)
        );
        // just outside the /10 on both sides.
        assert_eq!(classify_ip(v4("100.63.255.255")), None);
        assert_eq!(classify_ip(v4("100.128.0.0")), None);
    }

    #[test]
    fn v4_unspecified_broadcast_multicast_blocked() {
        assert_eq!(classify_ip(v4("0.0.0.0")), Some(BlockReason::Unspecified));
        assert_eq!(
            classify_ip(v4("255.255.255.255")),
            Some(BlockReason::Broadcast)
        );
        assert_eq!(classify_ip(v4("224.0.0.1")), Some(BlockReason::Multicast));
    }

    #[test]
    fn v4_documentation_and_benchmarking_blocked() {
        assert_eq!(
            classify_ip(v4("192.0.2.1")),
            Some(BlockReason::Documentation)
        );
        assert_eq!(
            classify_ip(v4("198.51.100.1")),
            Some(BlockReason::Documentation)
        );
        assert_eq!(
            classify_ip(v4("203.0.113.1")),
            Some(BlockReason::Documentation)
        );
        assert_eq!(
            classify_ip(v4("192.0.0.1")),
            Some(BlockReason::Documentation)
        );
        assert_eq!(
            classify_ip(v4("198.18.0.1")),
            Some(BlockReason::Benchmarking)
        );
        assert_eq!(
            classify_ip(v4("198.19.255.255")),
            Some(BlockReason::Benchmarking)
        );
    }

    #[test]
    fn v4_public_addresses_not_blocked() {
        assert_eq!(classify_ip(v4("8.8.8.8")), None);
        assert_eq!(classify_ip(v4("1.1.1.1")), None);
        assert_eq!(classify_ip(v4("93.184.216.34")), None);
    }

    #[test]
    fn v6_loopback_and_unspecified_blocked() {
        assert_eq!(classify_ip(v6("::1")), Some(BlockReason::Loopback));
        assert_eq!(classify_ip(v6("::")), Some(BlockReason::Unspecified));
    }

    #[test]
    fn v6_unique_local_and_link_local_blocked() {
        assert_eq!(classify_ip(v6("fc00::1")), Some(BlockReason::UniqueLocal));
        assert_eq!(
            classify_ip(v6("fd12:3456:789a::1")),
            Some(BlockReason::UniqueLocal)
        );
        assert_eq!(classify_ip(v6("fe80::1")), Some(BlockReason::LinkLocal));
    }

    #[test]
    fn v6_multicast_blocked() {
        assert_eq!(classify_ip(v6("ff02::1")), Some(BlockReason::Multicast));
    }

    /// IPv4-mapped IPv6 reaches the identical host as the embedded IPv4 address, the
    /// bypass a filter that only inspects `Ipv6Addr`-shaped predicates misses (memra#533
    /// asks explicitly for redirect and DNS-rebinding coverage; a mapped-address literal
    /// is the same class of "the string looks like an address filter should reject but
    /// isn't the shape the filter checked").
    #[test]
    fn v6_ipv4_mapped_embeds_the_v4_verdict() {
        assert_eq!(
            classify_ip(v6("::ffff:127.0.0.1")),
            Some(BlockReason::Loopback)
        );
        assert_eq!(
            classify_ip(v6("::ffff:10.0.0.1")),
            Some(BlockReason::Private)
        );
        assert_eq!(
            classify_ip(v6("::ffff:169.254.169.254")),
            Some(BlockReason::LinkLocal)
        );
        // a mapped PUBLIC v4 address stays unblocked.
        assert_eq!(classify_ip(v6("::ffff:8.8.8.8")), None);
    }

    /// The well-known NAT64 prefix (64:ff9b::/96) embeds an IPv4 address in its low 32
    /// bits, the IPv6 analog of the mapped-address bypass above.
    #[test]
    fn v6_nat64_prefix_embeds_the_v4_verdict() {
        assert_eq!(
            classify_ip(v6("64:ff9b::127.0.0.1")),
            Some(BlockReason::Loopback)
        );
        assert_eq!(classify_ip(v6("64:ff9b::8.8.8.8")), None);
    }

    #[test]
    fn v6_public_addresses_not_blocked() {
        // Cloudflare and Google public DNS, IPv6.
        assert_eq!(classify_ip(v6("2606:4700:4700::1111")), None);
        assert_eq!(classify_ip(v6("2001:4860:4860::8888")), None);
    }

    #[test]
    fn is_blocked_matches_classify_ip() {
        assert!(is_blocked(v4("127.0.0.1")));
        assert!(!is_blocked(v4("8.8.8.8")));
    }

    #[test]
    fn block_reason_as_str_is_stable_and_lowercase_snake_case() {
        for reason in [
            BlockReason::Loopback,
            BlockReason::Private,
            BlockReason::LinkLocal,
            BlockReason::CarrierGradeNat,
            BlockReason::UniqueLocal,
            BlockReason::Multicast,
            BlockReason::Unspecified,
            BlockReason::Broadcast,
            BlockReason::Documentation,
            BlockReason::Benchmarking,
        ] {
            let s = reason.as_str();
            assert!(s.chars().all(|c| c.is_ascii_lowercase() || c == '_'));
            assert_eq!(reason.to_string(), s);
        }
    }
}
