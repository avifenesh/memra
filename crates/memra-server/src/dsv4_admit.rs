//! Memory admission for the DSv4 serving route (memra#503).
//!
//! The route serves one request at a time, and before this module a request reached its
//! allocations with only the host-C4 budget check in front of it: a session too large for a
//! peer card, or one arriving while a co-tenant held the memory, found out from a CUDA
//! out-of-memory error halfway through allocating its state. The central worker's memory door
//! (`admit_memory`) does not see this route, so the route charges itself here, with the same
//! decision rule.
//!
//! A request's charge is per owning DEVICE: stages sharing a card sum, each card is checked
//! against its own effective free reading (driver free plus the async pool's mapped, unused
//! bytes), and a pinned host charge (active host C4) is checked against `MemAvailable`. The
//! order is `admit_memory::decide`'s: fit now; else yield what can be yielded (on the host, the
//! parked-prefix tier, never an entry this request would restore from; the device has nothing
//! demotable, since the only device state the route holds between requests is its weights);
//! else wait inside the defer budget; else refuse 429 with `Retry-After`. A session larger than
//! the route's boot ceiling on some card can never fit whatever the wait, and refuses at once as
//! a context-length error naming the largest session that fits.
//!
//! The defer budget is `MEMRA_ADMIT_DEFER_BUDGET_MS`, clamped to half the stall bound: the
//! route reads BUSY while a request defers and publishes no forward progress, so an unclamped
//! wait would read as a hung route.

use crate::admit_memory::{self, MemoryVerdict, Tiers};

/// How often a deferred request re-reads memory.
pub(crate) const DEFER_POLL_MS: u64 = 50;

/// One owning device's share of a request's charge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DeviceNeed {
    pub dev: usize,
    /// Device bytes the request will allocate on this card, every stage on it summed.
    pub need: u64,
    /// Effective free on this card measured at boot with no request resident: the most the
    /// route can ever offer a request there while its co-tenants stay as they were.
    pub ceiling: u64,
}

/// What one request claims before it allocates anything.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SessionNeed {
    pub devices: Vec<DeviceNeed>,
    /// Pinned host bytes (the active host-C4 history), 0 when the tier is off.
    pub host: u64,
}

/// Fold per-stage bytes onto their devices, in first-seen device order. `ceiling` is per stage
/// too, and stages on one card read the same card, so the device keeps its first reading.
pub(crate) fn per_device(devs: &[usize], need: &[u64], ceiling: &[u64]) -> Vec<DeviceNeed> {
    let mut out: Vec<DeviceNeed> = Vec::new();
    for ((&dev, &n), &c) in devs.iter().zip(need).zip(ceiling) {
        match out.iter_mut().find(|d| d.dev == dev) {
            Some(d) => d.need = d.need.saturating_add(n),
            None => out.push(DeviceNeed {
                dev,
                need: n,
                ceiling: c,
            }),
        }
    }
    out
}

/// The live readings and actions the admission loop needs, behind a seam so the loop runs on
/// the CPU against a script.
pub(crate) trait MemoryProbe {
    /// Effective free bytes on each of `devs`, read after fencing the route's streams there.
    fn device_free(&mut self, devs: &[usize]) -> Result<Vec<u64>, String>;
    /// `(available, reclaimable)`: host bytes a pinned allocation can claim now, and the parked
    /// bytes an eviction may return. `None` when the host reading is unavailable, which skips the
    /// host check (the C4 budget already bounded the charge).
    fn host(&mut self) -> Option<(u64, u64)>;
    /// Evict parked entries worth at least `bytes`, returning the bytes evicted.
    fn reclaim_host(&mut self, bytes: u64) -> u64;
    /// The client is gone.
    fn cancelled(&self) -> bool;
    fn elapsed_ms(&self) -> u64;
    fn wait(&mut self, ms: u64);
}

/// Which tier was short, and by how much, at a defer or a refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Shortfall {
    Device { dev: usize, need: u64, free: u64 },
    Host { need: u64, available: u64 },
}

impl Shortfall {
    pub(crate) fn short_by(self) -> u64 {
        match self {
            Shortfall::Device { need, free, .. } => need.saturating_sub(free),
            Shortfall::Host { need, available } => need.saturating_sub(available),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Admission {
    /// Fits. `deferred` is the first shortfall when the request had to wait for it.
    Admit {
        waited_ms: u64,
        reclaimed: u64,
        deferred: Option<Shortfall>,
    },
    /// The client left while the request waited.
    Cancelled { waited_ms: u64, short: Shortfall },
    /// Larger than the card holds with the route idle; no wait can admit it.
    NeverFits { dev: usize, need: u64, ceiling: u64 },
    /// Still short when the defer budget ran out.
    Refuse {
        short: Shortfall,
        waited_ms: u64,
        reclaimed: u64,
    },
}

impl Admission {
    pub(crate) fn verdict(&self) -> &'static str {
        match self {
            Admission::Admit { .. } => MemoryVerdict::Admit.as_str(),
            Admission::Cancelled { .. } => "cancelled",
            Admission::NeverFits { .. } => "never-fits",
            Admission::Refuse { .. } => MemoryVerdict::Refuse { short_by: 0 }.as_str(),
        }
    }
}

/// The defer budget the route may spend: the configured budget, and never more than half the
/// stall bound (a configured 0, the worker's "unbounded", is the half-bound here: this route
/// cannot wait unboundedly without reading hung).
pub(crate) fn defer_budget_ms(configured_ms: u64, stall_ms: u64) -> u64 {
    let bound = (stall_ms / 2).max(1);
    if configured_ms == 0 {
        bound
    } else {
        configured_ms.min(bound)
    }
}

/// Decide one request. Device first on every card (no demotable bytes), then the host tier with
/// the parked bytes as its yield, then time. A device shortfall never evicts parked host state:
/// the eviction could not buy the admission.
pub(crate) fn admit_session(
    need: &SessionNeed,
    probe: &mut impl MemoryProbe,
    budget_ms: u64,
) -> Result<Admission, String> {
    let budget_ms = budget_ms.max(1);
    let devs: Vec<usize> = need.devices.iter().map(|d| d.dev).collect();
    let mut reclaimed = 0u64;
    let mut deferred: Option<Shortfall> = None;
    loop {
        let waited_ms = probe.elapsed_ms();
        let free = probe.device_free(&devs)?;
        if free.len() != devs.len() {
            return Err(format!(
                "dsv4 memory probe read {} devices for {}",
                free.len(),
                devs.len()
            ));
        }
        let mut short: Option<Shortfall> = None;
        let mut refusing = false;
        for (d, &f) in need.devices.iter().zip(&free) {
            let ceiling = d.ceiling.max(f);
            if d.need > ceiling {
                return Ok(Admission::NeverFits {
                    dev: d.dev,
                    need: d.need,
                    ceiling,
                });
            }
            let tiers = Tiers {
                device_free_bytes: f,
                demotable_device_bytes: 0,
                host_free_bytes: 0,
            };
            let verdict = admit_memory::decide(d.need, &tiers, waited_ms, budget_ms);
            if let MemoryVerdict::Defer { short_by } | MemoryVerdict::Refuse { short_by } = verdict
            {
                refusing |= matches!(verdict, MemoryVerdict::Refuse { .. });
                if short.is_none_or(|s| short_by > s.short_by()) {
                    short = Some(Shortfall::Device {
                        dev: d.dev,
                        need: d.need,
                        free: f,
                    });
                }
            }
        }
        if short.is_none()
            && need.host > 0
            && let Some((available, reclaimable)) = probe.host()
        {
            let tiers = Tiers {
                device_free_bytes: available,
                demotable_device_bytes: reclaimable,
                // an evicted parked entry needs nowhere to land
                host_free_bytes: u64::MAX,
            };
            match admit_memory::decide(need.host, &tiers, waited_ms, budget_ms) {
                MemoryVerdict::Admit => {}
                MemoryVerdict::DemoteThenAdmit { demote_bytes } => {
                    let got = probe.reclaim_host(demote_bytes);
                    reclaimed = reclaimed.saturating_add(got);
                    if got > 0 {
                        // re-read every tier after the eviction rather than assume its effect
                        continue;
                    }
                    short = Some(Shortfall::Host {
                        need: need.host,
                        available,
                    });
                }
                verdict @ (MemoryVerdict::Defer { .. } | MemoryVerdict::Refuse { .. }) => {
                    refusing = matches!(verdict, MemoryVerdict::Refuse { .. });
                    short = Some(Shortfall::Host {
                        need: need.host,
                        available,
                    });
                }
            }
        }
        let Some(short) = short else {
            return Ok(Admission::Admit {
                waited_ms,
                reclaimed,
                deferred,
            });
        };
        deferred.get_or_insert(short);
        if refusing || waited_ms >= budget_ms {
            return Ok(Admission::Refuse {
                short,
                waited_ms,
                reclaimed,
            });
        }
        if probe.cancelled() {
            return Ok(Admission::Cancelled { waited_ms, short });
        }
        probe.wait(DEFER_POLL_MS.min(budget_ms - waited_ms));
    }
}

/// The largest session capacity in `1..=max` whose charge fits every card's ceiling, by
/// bisection over a charge that grows with capacity. 0 when not even one token fits.
pub(crate) fn largest_fitting_capacity(max: usize, mut fits: impl FnMut(usize) -> bool) -> usize {
    let (mut lo, mut hi) = (0usize, max);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

fn device_list(need: &SessionNeed, f: impl Fn(&DeviceNeed) -> u64) -> String {
    need.devices
        .iter()
        .map(|d| format!("{}:{}", d.dev, f(d)))
        .collect::<Vec<_>>()
        .join(",")
}

/// Everything one DSv4 `[admit-mem]` receipt line says.
pub(crate) struct AdmissionLine<'a> {
    pub request_id: &'a str,
    pub model: &'a str,
    pub capacity: usize,
    pub spec: bool,
    pub need: &'a SessionNeed,
    pub outcome: &'a Admission,
    pub retry_after_s: Option<u64>,
}

/// One grep-stable line, `[admit-mem]`-prefixed like the worker's, all fields `key=value`, with
/// `route=dsv4-thread` telling the two apart. Device lists read `dev:bytes`.
pub(crate) fn admission_line(line: &AdmissionLine<'_>) -> String {
    let (waited_ms, reclaimed, short) = match *line.outcome {
        Admission::Admit {
            waited_ms,
            reclaimed,
            deferred,
        } => (waited_ms, reclaimed, deferred),
        Admission::Cancelled { waited_ms, short } => (waited_ms, 0, Some(short)),
        Admission::NeverFits { dev, need, ceiling } => (
            0,
            0,
            Some(Shortfall::Device {
                dev,
                need,
                free: ceiling,
            }),
        ),
        Admission::Refuse {
            short,
            waited_ms,
            reclaimed,
        } => (waited_ms, reclaimed, Some(short)),
    };
    let short = match short {
        None => "-".to_string(),
        Some(s @ Shortfall::Device { dev, .. }) => format!("dev{dev}:{}", s.short_by()),
        Some(s @ Shortfall::Host { .. }) => format!("host:{}", s.short_by()),
    };
    format!(
        "[admit-mem] id={} model={:?} route=dsv4-thread verdict={} capacity={} spec={} \
         need={} ceiling={} host_need={} short={} waited_ms={} reclaimed={} retry_after_s={}",
        line.request_id,
        line.model,
        line.outcome.verdict(),
        line.capacity,
        line.spec,
        device_list(line.need, |d| d.need),
        device_list(line.need, |d| d.ceiling),
        line.need.host,
        short,
        waited_ms,
        reclaimed,
        line.retry_after_s
            .map_or("-".to_string(), |v| v.to_string()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scripted probe: `free[i]` is the device reading on poll `i` (the last repeats), each
    /// wait advances the clock, and a cancel lands at a chosen poll.
    struct Script {
        free: Vec<Vec<u64>>,
        host: Option<(u64, u64)>,
        evictable: Vec<u64>,
        cancel_at_poll: Option<usize>,
        polls: usize,
        clock_ms: u64,
        evicted: u64,
    }

    impl Script {
        fn new(free: Vec<Vec<u64>>) -> Self {
            Script {
                free,
                host: None,
                evictable: Vec::new(),
                cancel_at_poll: None,
                polls: 0,
                clock_ms: 0,
                evicted: 0,
            }
        }
    }

    impl MemoryProbe for Script {
        fn device_free(&mut self, devs: &[usize]) -> Result<Vec<u64>, String> {
            let row = &self.free[self.polls.min(self.free.len() - 1)];
            self.polls += 1;
            assert_eq!(row.len(), devs.len());
            Ok(row.clone())
        }
        fn host(&mut self) -> Option<(u64, u64)> {
            self.host
                .map(|(avail, _)| (avail, self.evictable.iter().sum()))
        }
        fn reclaim_host(&mut self, bytes: u64) -> u64 {
            let mut got = 0;
            while got < bytes && !self.evictable.is_empty() {
                got += self.evictable.remove(0);
            }
            if let Some((avail, _)) = self.host.as_mut() {
                *avail += got;
            }
            self.evicted += got;
            got
        }
        fn cancelled(&self) -> bool {
            self.cancel_at_poll.is_some_and(|p| self.polls >= p)
        }
        fn elapsed_ms(&self) -> u64 {
            self.clock_ms
        }
        fn wait(&mut self, ms: u64) {
            self.clock_ms += ms;
        }
    }

    fn two_cards(need0: u64, need1: u64, ceiling: u64) -> SessionNeed {
        SessionNeed {
            devices: vec![
                DeviceNeed {
                    dev: 0,
                    need: need0,
                    ceiling,
                },
                DeviceNeed {
                    dev: 1,
                    need: need1,
                    ceiling,
                },
            ],
            host: 0,
        }
    }

    #[test]
    fn stages_fold_onto_their_devices() {
        let d = per_device(&[3, 5, 3], &[10, 20, 5], &[100, 200, 999]);
        assert_eq!(
            d,
            vec![
                DeviceNeed {
                    dev: 3,
                    need: 15,
                    ceiling: 100
                },
                DeviceNeed {
                    dev: 5,
                    need: 20,
                    ceiling: 200
                },
            ]
        );
    }

    #[test]
    fn a_session_that_fits_every_card_admits_without_waiting() {
        let mut p = Script::new(vec![vec![100, 100]]);
        let out = admit_session(&two_cards(60, 90, 100), &mut p, 1000).unwrap();
        assert_eq!(
            out,
            Admission::Admit {
                waited_ms: 0,
                reclaimed: 0,
                deferred: None
            }
        );
        assert_eq!(p.polls, 1);
    }

    #[test]
    fn a_session_above_a_cards_ceiling_never_fits_and_does_not_wait() {
        // overload: the peer card could never hold this session, however long it waits
        let mut p = Script::new(vec![vec![100, 100]]);
        let out = admit_session(&two_cards(60, 101, 100), &mut p, 1000).unwrap();
        assert_eq!(
            out,
            Admission::NeverFits {
                dev: 1,
                need: 101,
                ceiling: 100
            }
        );
        assert_eq!((p.polls, p.clock_ms), (1, 0));
    }

    #[test]
    fn a_short_peer_card_defers_then_refuses_at_the_budget() {
        // card 0 is fine; the peer is held by a co-tenant for the whole budget
        let mut p = Script::new(vec![vec![100, 40]]);
        let out = admit_session(&two_cards(60, 90, 100), &mut p, 200).unwrap();
        let short = Shortfall::Device {
            dev: 1,
            need: 90,
            free: 40,
        };
        assert_eq!(
            out,
            Admission::Refuse {
                short,
                waited_ms: 200,
                reclaimed: 0
            }
        );
        assert_eq!(short.short_by(), 50);
        assert_eq!(p.polls, 5, "polled at 0, 50, 100, 150 and 200 ms");
    }

    #[test]
    fn memory_freed_mid_defer_admits_the_waiting_request() {
        // recovery: the co-tenant leaves on the third reading
        let mut p = Script::new(vec![vec![100, 40], vec![100, 40], vec![100, 95]]);
        let out = admit_session(&two_cards(60, 90, 100), &mut p, 8000).unwrap();
        assert_eq!(
            out,
            Admission::Admit {
                waited_ms: 100,
                reclaimed: 0,
                deferred: Some(Shortfall::Device {
                    dev: 1,
                    need: 90,
                    free: 40
                }),
            }
        );
    }

    #[test]
    fn a_client_that_leaves_mid_defer_is_cancelled_not_refused() {
        let mut p = Script::new(vec![vec![10, 10]]);
        p.cancel_at_poll = Some(3);
        let out = admit_session(&two_cards(60, 90, 100), &mut p, 8000).unwrap();
        assert!(
            matches!(out, Admission::Cancelled { waited_ms: 100, .. }),
            "{out:?}"
        );
    }

    #[test]
    fn a_host_shortfall_evicts_parked_entries_and_admits() {
        let mut need = two_cards(10, 10, 100);
        need.host = 50;
        let mut p = Script::new(vec![vec![100, 100]]);
        p.host = Some((20, 0));
        p.evictable = vec![10, 15, 40];
        let out = admit_session(&need, &mut p, 8000).unwrap();
        assert_eq!(
            out,
            Admission::Admit {
                waited_ms: 0,
                reclaimed: 65,
                deferred: None
            }
        );
        assert_eq!(
            p.evicted, 65,
            "evicted up to the shortfall of 30, whole entries"
        );
    }

    #[test]
    fn a_device_shortfall_never_evicts_parked_host_state() {
        let mut need = two_cards(10, 90, 100);
        need.host = 50;
        let mut p = Script::new(vec![vec![100, 40]]);
        p.host = Some((20, 0));
        p.evictable = vec![100];
        let out = admit_session(&need, &mut p, 100).unwrap();
        assert!(matches!(out, Admission::Refuse { .. }), "{out:?}");
        assert_eq!(p.evicted, 0);
    }

    #[test]
    fn a_host_shortfall_past_the_parked_tier_defers_then_refuses_without_evicting() {
        let mut need = two_cards(10, 10, 100);
        need.host = 500;
        let mut p = Script::new(vec![vec![100, 100]]);
        p.host = Some((20, 0));
        p.evictable = vec![30];
        let out = admit_session(&need, &mut p, 100).unwrap();
        assert!(
            matches!(
                out,
                Admission::Refuse {
                    short: Shortfall::Host { need: 500, .. },
                    reclaimed: 0,
                    ..
                }
            ),
            "{out:?}"
        );
        assert_eq!(
            p.evicted, 0,
            "an eviction that cannot buy the admission is not paid"
        );
    }

    #[test]
    fn the_budget_is_clamped_under_the_stall_bound() {
        assert_eq!(defer_budget_ms(8_000, 120_000), 8_000);
        assert_eq!(
            defer_budget_ms(0, 120_000),
            60_000,
            "0 is the half-bound here"
        );
        assert_eq!(defer_budget_ms(600_000, 120_000), 60_000);
        assert_eq!(defer_budget_ms(8_000, 1_000), 500);
    }

    #[test]
    fn bisection_finds_the_largest_fitting_capacity() {
        assert_eq!(largest_fitting_capacity(1 << 20, |c| c * 3 <= 3000), 1000);
        assert_eq!(largest_fitting_capacity(64, |_| true), 64);
        assert_eq!(largest_fitting_capacity(64, |_| false), 0);
    }

    #[test]
    fn the_receipt_line_is_key_value_and_names_the_short_card() {
        let need = two_cards(60, 90, 100);
        let outcome = Admission::Refuse {
            short: Shortfall::Device {
                dev: 1,
                need: 90,
                free: 40,
            },
            waited_ms: 200,
            reclaimed: 0,
        };
        let s = admission_line(&AdmissionLine {
            request_id: "r1",
            model: "ds",
            capacity: 4096,
            spec: true,
            need: &need,
            outcome: &outcome,
            retry_after_s: Some(5),
        });
        assert!(s.starts_with("[admit-mem] "), "{s}");
        for kv in [
            "route=dsv4-thread",
            "verdict=refuse",
            "capacity=4096",
            "need=0:60,1:90",
            "ceiling=0:100,1:100",
            "short=dev1:50",
            "waited_ms=200",
            "retry_after_s=5",
        ] {
            assert!(s.contains(kv), "{kv} missing from {s}");
        }
    }
}
