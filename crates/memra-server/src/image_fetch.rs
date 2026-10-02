//! memra#533: `image_url` accepts base64 data URIs only; most OpenAI SDK examples and
//! agent stacks pass `http(s)` URLs instead. This module fetches those URLs
//! server-side, bounded and SSRF-fail-closed, then hands the caller a base64 data URI:
//! the ONE shape every vision content-walker in `lib.rs` already decodes and admits. The
//! walkers themselves (`content_to_text_vision`, `content_to_text_vision_step`) are
//! UNCHANGED: they still refuse anything that is not `data:`, so this module's whole job
//! is to turn an admitted `http(s)` URL into that shape before the walker ever sees it.
//!
//! Default OFF. `MEMRA_FETCH_URLS=1` arms it; the name is pinned by
//! `docs/decisions/VISION-LANE.md` and `memra-engine`'s own `decode_data_uri` error
//! message, both written before this lane. Disabled: an `http(s)` `image_url` reaches
//! the walker exactly as before and gets the existing
//! "image_url must be a base64 data URI (http(s) fetch is disabled)" 400.
//!
//! SSRF posture (memra#533): fail closed on loopback, RFC1918/CGNAT private, link-local
//! (which is where every major cloud's metadata endpoint lives), IPv6 unique-local, and
//! their documented-bypass cousins (IPv4-mapped IPv6, NAT64). The address-range half
//! lives in `memra-net-guard`, unit-tested there with no I/O. This module's own job is
//! getting the RIGHT ADDRESS in front of that classifier at the right time:
//!
//!   * A literal IP host (`http://127.0.0.1/x`) never reaches a DNS resolver at all:
//!     hyper-util's connector special-cases an IP-shaped host and skips resolution
//!     entirely (verified against hyper-util 0.1.21's `HttpConnector::call_async`). So
//!     the literal-IP case is checked HERE, before the first request is ever sent, not
//!     inferred from a resolver error that will never fire.
//!   * A hostname is checked at the moment it resolves, via a custom `reqwest::dns::Resolve`
//!     that classifies every candidate address and refuses to hand back a blocked one.
//!     Resolve-then-check, never check-then-resolve, so a DNS answer that arrives after
//!     the check (rebinding) is still caught: there is no earlier answer to become stale.
//!   * A redirect is a NEW request through the same client, so both checks above run
//!     again on every hop; the redirect policy additionally caps hop count and refuses a
//!     scheme change (a redirect to `file://` or `data:` is refused before it can be
//!     followed).
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::http::StatusCode;
use base64::Engine;
use futures_util::StreamExt;

use crate::error_response_coded;

/// The flag name is pinned by `docs/decisions/VISION-LANE.md` ("http(s) URL fetch gated
/// behind MEMRA_FETCH_URLS=1") and by `memra_engine::vision_pre::decode_data_uri`'s error
/// message, both written before this module existed. Read fresh every call (not cached
/// behind a `OnceLock` like the vision-family switches): tests toggle it per case, and a
/// per-request env read costs nothing next to the network fetch it gates.
pub(crate) fn fetch_urls_enabled() -> bool {
    std::env::var("MEMRA_FETCH_URLS").as_deref() == Ok("1")
}

/// Comma-separated exact hostnames (case-insensitive) that may resolve to an address the
/// classifier would otherwise refuse: memra#533's "unless allowlisted". This bypasses
/// the ADDRESS check only; scheme and redirect-count limits still apply, and it does
/// nothing when `MEMRA_FETCH_URLS` itself is off.
fn allowed_hosts() -> Vec<String> {
    std::env::var("MEMRA_FETCH_URLS_ALLOWED_HOSTS")
        .ok()
        .map(|raw| {
            raw.split(',')
                .map(|h| h.trim().to_ascii_lowercase())
                .filter(|h| !h.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// Per-image raw byte cap. Reuses `memra_engine::vision_pre::IMG_MAX_RAW_BYTES` (12 MiB)
/// rather than a second number: that constant is already the per-image budget the 192
/// MiB body ceiling was itemized against (`lib.rs:170`), and a fetched image occupies the
/// identical downstream slot a decoded data-URI payload does.
const FETCH_MAX_BYTES: usize = memra_engine::vision_pre::IMG_MAX_RAW_BYTES;
const FETCH_MAX_REDIRECTS: usize = 5;
const FETCH_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const FETCH_TOTAL_TIMEOUT: Duration = Duration::from_secs(10);
/// Deadline for the WHOLE per-request fetch pass, every image included (PR #904 review):
/// `VISION_MAX_IMAGES` (8) sequential fetches at `FETCH_TOTAL_TIMEOUT` (10 s) each could
/// otherwise hold a request open for up to 80 s. 20 s bounds the worst case well under
/// that while still completing a normal one- or two-image request; a slower fetch pass
/// aborts with a named 400 instead of running out the individual per-image timeouts one
/// at a time.
const FETCH_URLS_TOTAL_BUDGET: Duration = Duration::from_secs(20);
/// Content-Type allowlist. The walkers below this module sniff actual image bytes
/// (PNG/JPEG/etc. magic) and refuse an unrecognized format on their own; this check exists
/// so a redirect to an HTML error page or an arbitrary blob fails fast with a clear typed
/// error instead of spending a decode attempt on it.
const FETCH_ALLOWED_CONTENT_TYPES: &[&str] = &[
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/gif",
];

/// Marker error carried through reqwest's DNS-resolver and redirect-policy error paths so
/// `fetch_remote_image` can tell an SSRF refusal apart from an ordinary network failure by
/// walking the error's `source()` chain (`std::error::Error::source`), rather than by
/// matching on `reqwest::Error`'s private internal `Kind`, which is not part of its public
/// API surface.
#[derive(Debug)]
struct SsrfBlocked(String);

impl std::fmt::Display for SsrfBlocked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ssrf_blocked: {}", self.0)
    }
}

impl std::error::Error for SsrfBlocked {}

fn find_ssrf_blocked(err: &(dyn std::error::Error + 'static)) -> Option<String> {
    if let Some(b) = err.downcast_ref::<SsrfBlocked>() {
        return Some(b.0.clone());
    }
    err.source().and_then(find_ssrf_blocked)
}

/// Named failure classes (memra#533: "Errors are named 400s ... never a hung request").
#[derive(Debug)]
pub(crate) enum FetchError {
    /// The host, a resolved address, or a redirect target is on a refused range (or the
    /// guard itself refused: too many redirects, a scheme change, or every resolved
    /// address blocked).
    Blocked(String),
    /// The response (by `Content-Length` or by the actual byte count while streaming)
    /// exceeds `FETCH_MAX_BYTES`.
    TooLarge(usize),
    /// Anything else that keeps this image from reaching the caller: connect/read
    /// timeout, DNS failure, non-2xx status, TLS failure, or a Content-Type outside the
    /// allowlist.
    Unreachable(String),
    /// More `image_url` parts than `VISION_MAX_IMAGES` (PR #904 review): checked
    /// BEFORE any fetch starts, not after, so a small request body can never make this
    /// module fetch an unbounded number of images. Carries no `image_url_*` code: this is
    /// the exact "too many images" shape the vision content walkers already produce for
    /// `data:` images (`lib.rs` `VISION_MAX_IMAGES` checks), so a caller sees the identical
    /// error whether the images were inline or fetched.
    TooMany(usize),
    /// Fetched base64 would exceed the original request body budget.
    RequestTooLarge,
}

impl FetchError {
    fn code(&self) -> Option<&'static str> {
        match self {
            FetchError::Blocked(_) => Some("image_url_blocked"),
            FetchError::TooLarge(_) | FetchError::RequestTooLarge => Some("image_url_too_large"),
            FetchError::Unreachable(_) => Some("image_url_unreachable"),
            FetchError::TooMany(_) => None,
        }
    }

    fn message(&self) -> String {
        match self {
            FetchError::Blocked(detail) => format!("image_url fetch refused: {detail}"),
            FetchError::TooLarge(n) => format!(
                "image_url response is {n} bytes, over the {FETCH_MAX_BYTES}-byte per-image cap"
            ),
            FetchError::Unreachable(detail) => format!("image_url fetch failed: {detail}"),
            FetchError::RequestTooLarge => format!(
                "expanded image request exceeds the {}-byte request budget",
                crate::MAX_BODY_BYTES
            ),
            FetchError::TooMany(n) => {
                let max = crate::VISION_MAX_IMAGES;
                format!("too many images (max {max}): {n} image_url parts requested")
            }
        }
    }

    /// Every named failure here is a 400: the caller sent an unfetchable or refused URL,
    /// never a fault this server owns (memra#533 point 4: a hung request is the one
    /// shape this module must never produce, and it never produces a 5xx either).
    pub(crate) fn into_response(self) -> axum::response::Response {
        let code = self.code();
        let message = self.message();
        error_response_coded(
            StatusCode::BAD_REQUEST,
            &message,
            "invalid_request_error",
            Some("messages"),
            code,
        )
    }
}

struct GuardedResolver {
    allowed: Arc<Vec<String>>,
}

impl reqwest::dns::Resolve for GuardedResolver {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let allowed = self.allowed.clone();
        let host = name.as_str().to_string();
        Box::pin(async move {
            let addrs: Vec<std::net::SocketAddr> = tokio::net::lookup_host((host.as_str(), 0))
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?
                .collect();
            filter_resolved(addrs, &allowed, &host)
                .map(|kept| Box::new(kept.into_iter()) as reqwest::dns::Addrs)
                .map_err(|detail| {
                    Box::new(SsrfBlocked(detail)) as Box<dyn std::error::Error + Send + Sync>
                })
        })
    }
}

/// Resolve-then-check: classify whatever addresses THIS call was just handed, never a
/// cached or earlier answer, and never the hostname string. There is nothing here to
/// compare a new lookup against, on purpose. That absence is what catches DNS rebinding.
/// A hostname that answers with a public address on lookup N and a private one on lookup
/// N+1 is judged fresh both times: this function has no memory of lookup N when it runs
/// for N+1, so a rebound answer is refused exactly like a first-time private answer would
/// be, never grandfathered in because an earlier check on the "same" host passed.
fn filter_resolved(
    addrs: Vec<std::net::SocketAddr>,
    allowed: &[String],
    host: &str,
) -> Result<Vec<std::net::SocketAddr>, String> {
    if host_is_allowed(allowed, host) {
        if addrs.is_empty() {
            return Err(format!("{host}: no addresses resolved"));
        }
        return Ok(addrs);
    }
    let filtered: Vec<std::net::SocketAddr> = addrs
        .into_iter()
        .filter(|a| memra_net_guard::classify_ip(a.ip()).is_none())
        .collect();
    if filtered.is_empty() {
        return Err(format!(
            "{host}: every resolved address is on a refused range"
        ));
    }
    Ok(filtered)
}

fn host_is_allowed(allowed: &[String], host: &str) -> bool {
    allowed.iter().any(|h| h.eq_ignore_ascii_case(host))
}

/// A literal IP host bypasses `reqwest::dns::Resolve` entirely (hyper-util's
/// `HttpConnector::call_async` special-cases an address-shaped host via
/// `dns::SocketAddrs::try_parse` and connects directly, never calling the injected
/// resolver). Checked explicitly on the initial URL before the first `send()`, and again
/// on every redirect target inside the redirect policy below; both are the only two
/// places a literal IP can appear.
///
/// Uses `Url::host()`, not `Url::host_str()`: for an IPv6 literal `host_str()` returns
/// the bracketed form (`"[::1]"`), which fails `str::parse::<IpAddr>()` outright and would
/// let every IPv6 literal straight past this check. `host()` returns the address already
/// typed, with no bracket-stripping to get wrong.
fn check_literal_ip_host(url: &reqwest::Url, allowed: &[String]) -> Result<(), FetchError> {
    match url.host() {
        // A hostname, not a literal, resolved later, and checked there
        // (`GuardedResolver::resolve`) against the addresses that resolution actually
        // returns. Nothing to check on the string itself.
        Some(url::Host::Domain(_)) => Ok(()),
        Some(url::Host::Ipv4(ip)) => check_literal_ip(IpAddr::V4(ip), allowed),
        Some(url::Host::Ipv6(ip)) => check_literal_ip(IpAddr::V6(ip), allowed),
        None => Err(FetchError::Unreachable("url has no host".into())),
    }
}

fn check_literal_ip(ip: IpAddr, allowed: &[String]) -> Result<(), FetchError> {
    if host_is_allowed(allowed, &ip.to_string()) {
        return Ok(());
    }
    Err(FetchError::Blocked(format!(
        "literal IP {ip} requires an explicit host allowlist entry"
    )))
}

fn redirect_policy(allowed: Arc<Vec<String>>) -> reqwest::redirect::Policy {
    reqwest::redirect::Policy::custom(move |attempt| {
        // reqwest includes the initial URL in previous; it is not a redirect.
        if attempt.previous().len() > FETCH_MAX_REDIRECTS {
            return attempt.error(SsrfBlocked(format!(
                "more than {FETCH_MAX_REDIRECTS} redirects"
            )));
        }
        let scheme_ok = {
            let next = attempt.url();
            next.scheme() == "http" || next.scheme() == "https"
        };
        if !scheme_ok {
            let scheme = attempt.url().scheme().to_string();
            return attempt.error(SsrfBlocked(format!(
                "redirect to non-http(s) scheme {scheme:?}"
            )));
        }
        let literal_check = {
            let next = attempt.url();
            check_literal_ip_host(next, &allowed)
        };
        if let Err(FetchError::Blocked(detail)) = literal_check {
            return attempt.error(SsrfBlocked(detail));
        }
        attempt.follow()
    })
}

fn build_client(allowed: Arc<Vec<String>>) -> Result<reqwest::Client, FetchError> {
    reqwest::Client::builder()
        .dns_resolver(Arc::new(GuardedResolver {
            allowed: allowed.clone(),
        }))
        .redirect(redirect_policy(allowed))
        .connect_timeout(FETCH_CONNECT_TIMEOUT)
        .timeout(FETCH_TOTAL_TIMEOUT)
        // Every hop the caller might reach is already covered by our own resolver and
        // redirect policy; a system proxy would route around both, so it stays off.
        .no_proxy()
        .build()
        .map_err(|e| FetchError::Unreachable(format!("client build: {e}")))
}

/// Fetch one `http(s)` URL, bounded and guarded, returning raw bytes plus the response's
/// `Content-Type` (validated against `FETCH_ALLOWED_CONTENT_TYPES`).
async fn fetch_remote_image(url: &str, max_bytes: usize) -> Result<(Vec<u8>, String), FetchError> {
    let parsed = reqwest::Url::parse(url)
        .map_err(|e| FetchError::Unreachable(format!("malformed url: {e}")))?;
    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        return Err(FetchError::Blocked(format!(
            "scheme {:?} is not http(s)",
            parsed.scheme()
        )));
    }
    let allowed = Arc::new(allowed_hosts());
    // The literal-IP case never reaches the resolver at all (see the doc comment on
    // `check_literal_ip_host`), so it is checked here, on the INITIAL url, before the
    // client is even built.
    check_literal_ip_host(&parsed, &allowed)?;
    let client = build_client(allowed)?;
    let resp = match client.get(parsed).send().await {
        Ok(r) => r,
        Err(err) => {
            let boxed: &dyn std::error::Error = &err;
            if let Some(detail) = find_ssrf_blocked(boxed) {
                return Err(FetchError::Blocked(detail));
            }
            if err.is_timeout() {
                return Err(FetchError::Unreachable(format!("timed out: {err}")));
            }
            return Err(FetchError::Unreachable(err.to_string()));
        }
    };
    // reqwest can decline a non-HTTP Location before invoking the custom policy.
    // Preserve the same blocked classification for that un-followed redirect.
    if resp.status().is_redirection()
        && let Some(location) = resp
            .headers()
            .get(reqwest::header::LOCATION)
            .and_then(|v| v.to_str().ok())
        && let Ok(next) = resp.url().join(location)
        && !matches!(next.scheme(), "http" | "https")
    {
        return Err(FetchError::Blocked(format!(
            "redirect to non-http(s) scheme {:?}",
            next.scheme()
        )));
    }
    if !resp.status().is_success() {
        return Err(FetchError::Unreachable(format!(
            "http status {}",
            resp.status()
        )));
    }
    if let Some(len) = resp.content_length()
        && len as usize > max_bytes
    {
        return Err(size_error(len as usize, max_bytes));
    }
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.split(';').next().unwrap_or(v).trim().to_ascii_lowercase())
        .unwrap_or_default();
    if !FETCH_ALLOWED_CONTENT_TYPES.contains(&content_type.as_str()) {
        return Err(FetchError::Unreachable(format!(
            "unsupported content-type {content_type:?} (expected one of {FETCH_ALLOWED_CONTENT_TYPES:?})"
        )));
    }
    // The Content-Length check above is advisory (a server can omit or lie about it); the
    // real cap is enforced here, byte by byte, so a body larger than declared is aborted
    // mid-stream rather than fully downloaded first.
    let mut buf: Vec<u8> = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| FetchError::Unreachable(format!("body read: {e}")))?;
        if buf.len().saturating_add(chunk.len()) > max_bytes {
            return Err(size_error(buf.len().saturating_add(chunk.len()), max_bytes));
        }
        buf.extend_from_slice(&chunk);
    }
    Ok((buf, content_type))
}

fn size_error(bytes: usize, limit: usize) -> FetchError {
    if limit < FETCH_MAX_BYTES {
        FetchError::RequestTooLarge
    } else {
        FetchError::TooLarge(bytes)
    }
}

/// The received JSON byte count is a conservative starting budget. Replacing a URL
/// subtracts its decoded length (never more than its wire length) and adds the data
/// URI's exact ASCII length. Text, inline images and other request fields keep their
/// original charge. Each fetch is capped before allocation, including base64 growth.
struct WireBudget {
    used: usize,
    max: usize,
}
impl WireBudget {
    fn raw_limit(&self, old_url_bytes: usize) -> Result<usize, FetchError> {
        let remaining = self
            .max
            .checked_sub(self.used.saturating_sub(old_url_bytes))
            .and_then(|n| n.checked_sub("data:image/jpeg;base64,".len()))
            .ok_or(FetchError::RequestTooLarge)?;
        let raw = (remaining / 4) * 3;
        if raw == 0 {
            return Err(FetchError::RequestTooLarge);
        }
        Ok(raw.min(FETCH_MAX_BYTES))
    }
    fn replace(&mut self, old_url_bytes: usize, new_bytes: usize) -> Result<(), FetchError> {
        let next = self
            .used
            .saturating_sub(old_url_bytes)
            .saturating_add(new_bytes);
        if next > self.max {
            return Err(FetchError::RequestTooLarge);
        }
        self.used = next;
        Ok(())
    }
}

/// One content part's `image_url` string, if the part's type is `image_url` and the value
/// takes either OpenAI shape (a bare string, or `{"url": "..."}`). Read-only: both the
/// fetch-and-rewrite pass uses this for both supported content shapes.
fn image_url_str(part: &serde_json::Value) -> Option<&str> {
    if part.get("type").and_then(|t| t.as_str()) != Some("image_url") {
        return None;
    }
    let image_url = part.get("image_url")?;
    match image_url {
        serde_json::Value::String(s) => Some(s.as_str()),
        serde_json::Value::Object(_) => image_url.get("url").and_then(|u| u.as_str()),
        _ => None,
    }
}

fn is_fetchable_url(url: &str) -> bool {
    url.get(..7)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("http://"))
        || url
            .get(..8)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("https://"))
}

/// Count all image parts before any fetch, including inline images and malformed
/// image values. Mixing data URIs with remote URLs must not evade the total cap.
pub(crate) fn count_image_urls(messages: &[crate::ChatMessage]) -> usize {
    messages
        .iter()
        .filter_map(|msg| match &msg.content {
            serde_json::Value::Array(parts) => Some(parts.as_slice()),
            _ => None,
        })
        .flatten()
        .filter(|part| part.get("type").and_then(|t| t.as_str()) == Some("image_url"))
        .count()
}

/// Rewrite every `http(s)` `image_url`/`image_url.url` in `content` to a base64 data URI
/// in place, fetching each through `fetch_remote_image`. `data:` URIs and non-string
/// `image_url` shapes pass through untouched (the walkers below already handle a
/// malformed shape with their own named error). Only fires when `fetch_urls_enabled()`;
/// callers check that once, up front, and skip this entirely when it is false so an
/// `http(s)` URL reaches the walker's existing "http(s) fetch is disabled" 400 unchanged.
async fn resolve_content_image_urls(
    content: &mut serde_json::Value,
    budget: &mut WireBudget,
) -> Result<(), FetchError> {
    let serde_json::Value::Array(parts) = content else {
        return Ok(());
    };
    for part in parts.iter_mut() {
        let Some(url) = image_url_str(part) else {
            continue;
        };
        if !is_fetchable_url(url) {
            continue;
        }
        let url_owned = url.to_string();
        let limit = budget.raw_limit(url_owned.len())?;
        let (bytes, content_type) = fetch_remote_image(&url_owned, limit).await?;
        let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let data_uri = format!("data:{content_type};base64,{encoded}");
        budget.replace(url_owned.len(), data_uri.len())?;
        let image_url = part
            .get_mut("image_url")
            .expect("image_url_str just confirmed this part carries an image_url value");
        if image_url.is_string() {
            *image_url = serde_json::Value::String(data_uri);
        } else {
            image_url["url"] = serde_json::Value::String(data_uri);
        }
    }
    Ok(())
}

/// Runs once per chat-completions request, before any vision content walker sees the
/// message list. No-op (and no allocation beyond the flag read) when
/// `fetch_urls_enabled()` is false, so the disabled path is byte-identical to before this
/// module existed.
///
/// Two bounds sit above the per-image ones in `fetch_remote_image` (PR #904 review): the
/// TOTAL image count never exceeds `VISION_MAX_IMAGES`, checked before any fetch
/// starts, and the WHOLE pass carries one overall deadline (`FETCH_URLS_TOTAL_BUDGET`)
/// (also capped by the caller deadline), so a slow multi-image request cannot run out
/// `VISION_MAX_IMAGES` individual timeouts back to back.
pub(crate) async fn resolve_remote_image_urls_in_messages(
    messages: &mut [crate::ChatMessage],
    wire_bytes: usize,
    remaining: Duration,
) -> Result<(), FetchError> {
    if !fetch_urls_enabled() {
        return Ok(());
    }
    let requested = count_image_urls(messages);
    if requested > crate::VISION_MAX_IMAGES {
        return Err(FetchError::TooMany(requested));
    }
    let mut budget = WireBudget {
        used: wire_bytes,
        max: crate::MAX_BODY_BYTES,
    };
    let timeout = FETCH_URLS_TOTAL_BUDGET.min(remaining);
    match tokio::time::timeout(timeout, async {
        for msg in messages.iter_mut() {
            resolve_content_image_urls(&mut msg.content, &mut budget).await?;
        }
        Ok(())
    })
    .await
    {
        Ok(result) => result,
        Err(_) => Err(FetchError::Unreachable(format!(
            "image_url fetch pass exceeded the {}s overall deadline for {requested} image(s)",
            timeout.as_secs_f64()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    static FETCH_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn fetch_urls_enabled_defaults_off() {
        let _lock = FETCH_ENV_LOCK.lock().unwrap();
        // SAFETY (test-only): no other test in this process touches this exact var, and
        // std::env mutation in a `#[test]` is the established pattern this crate's other
        // flag tests use (see `vision_placement_gate_tests`).
        unsafe {
            std::env::remove_var("MEMRA_FETCH_URLS");
        }
        assert!(!fetch_urls_enabled());
        unsafe {
            std::env::set_var("MEMRA_FETCH_URLS", "1");
        }
        assert!(fetch_urls_enabled());
        unsafe {
            std::env::set_var("MEMRA_FETCH_URLS", "0");
        }
        assert!(!fetch_urls_enabled());
        unsafe {
            std::env::remove_var("MEMRA_FETCH_URLS");
        }
    }

    #[test]
    fn allowed_hosts_parses_csv_case_insensitively() {
        let _lock = FETCH_ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var(
                "MEMRA_FETCH_URLS_ALLOWED_HOSTS",
                " Example.com, other.test ,,",
            );
        }
        let hosts = allowed_hosts();
        assert_eq!(
            hosts,
            vec!["example.com".to_string(), "other.test".to_string()]
        );
        assert!(host_is_allowed(&hosts, "EXAMPLE.COM"));
        assert!(!host_is_allowed(&hosts, "evil.example.com"));
        unsafe {
            std::env::remove_var("MEMRA_FETCH_URLS_ALLOWED_HOSTS");
        }
    }

    /// The literal-IP pre-check: every named block reason from memra#533 (loopback,
    /// private, link-local/metadata) refuses on the URL string alone, before any network
    /// call: this is the half of the guard that a DNS-resolver hook can never reach,
    /// because hyper-util skips the resolver for an address-shaped host.
    #[test]
    fn literal_ip_pre_check_refuses_each_named_range() {
        let none: Vec<String> = Vec::new();
        for (url, should_block) in [
            ("http://127.0.0.1/x", true),
            ("http://10.0.0.5/x", true),
            ("http://172.16.0.5/x", true),
            ("http://192.168.1.5/x", true),
            ("http://169.254.169.254/latest/meta-data/", true), // cloud metadata
            ("http://[::1]/x", true),
            ("http://[fc00::1]/x", true),
            ("http://8.8.8.8/x", true),
            ("http://example.com/x", false), // hostname: not this check's job
        ] {
            let parsed = reqwest::Url::parse(url).unwrap();
            let result = check_literal_ip_host(&parsed, &none);
            assert_eq!(
                result.is_err(),
                should_block,
                "url {url:?} expected blocked={should_block}, got {result:?}"
            );
        }
    }

    #[test]
    fn literal_ip_pre_check_respects_allowlist() {
        let allowed = vec!["127.0.0.1".to_string()];
        assert!(
            check_literal_ip_host(
                &reqwest::Url::parse("http://127.0.0.1/x").unwrap(),
                &allowed
            )
            .is_ok()
        );
        // a DIFFERENT blocked address is not covered by an allowlist entry for another host.
        assert!(
            check_literal_ip_host(&reqwest::Url::parse("http://10.0.0.1/x").unwrap(), &allowed)
                .is_err()
        );
    }

    #[test]
    fn fetch_error_maps_to_named_400_codes() {
        assert_eq!(
            FetchError::Blocked("x".into()).code(),
            Some("image_url_blocked")
        );
        assert_eq!(FetchError::TooLarge(1).code(), Some("image_url_too_large"));
        assert_eq!(
            FetchError::Unreachable("x".into()).code(),
            Some("image_url_unreachable")
        );
        for err in [
            FetchError::Blocked("x".into()),
            FetchError::TooLarge(1),
            FetchError::RequestTooLarge,
            FetchError::Unreachable("x".into()),
        ] {
            assert_eq!(err.into_response().status(), StatusCode::BAD_REQUEST);
        }
    }

    /// The DNS-rebinding shape: the SAME hostname resolves to a public address on one
    /// lookup and a private address on the next. `filter_resolved` has no memory of the
    /// earlier lookup: each call judges only the addresses it was just handed, so the
    /// second lookup is refused exactly as a first-time private answer would be, never
    /// grandfathered in because the "same" host passed before.
    #[test]
    fn filter_resolved_catches_a_rebound_answer_with_no_memory_of_the_prior_lookup() {
        use std::net::{SocketAddr, SocketAddrV4};
        let none: Vec<String> = Vec::new();
        let public_lookup = vec![SocketAddr::V4(SocketAddrV4::new(
            "8.8.8.8".parse().unwrap(),
            0,
        ))];
        let rebound_lookup = vec![SocketAddr::V4(SocketAddrV4::new(
            "127.0.0.1".parse().unwrap(),
            0,
        ))];
        assert!(filter_resolved(public_lookup, &none, "attacker.example").is_ok());
        assert!(filter_resolved(rebound_lookup, &none, "attacker.example").is_err());
    }

    #[test]
    fn filter_resolved_drops_only_the_blocked_addresses_in_a_mixed_answer() {
        use std::net::{SocketAddr, SocketAddrV4};
        let none: Vec<String> = Vec::new();
        let mixed = vec![
            SocketAddr::V4(SocketAddrV4::new("127.0.0.1".parse().unwrap(), 0)),
            SocketAddr::V4(SocketAddrV4::new("8.8.8.8".parse().unwrap(), 0)),
        ];
        let kept = filter_resolved(mixed, &none, "mixed.example").unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].ip().to_string(), "8.8.8.8");
    }

    /// A redirect that would land on a refused range is refused by the policy, not
    /// silently followed. Exercised at the policy level (no network) so it runs without
    /// a fixture server.
    #[test]
    fn redirect_policy_refuses_a_redirect_to_a_blocked_literal_ip() {
        // reqwest's `Attempt` has no public constructor outside the crate, so this is
        // exercised end-to-end via `check_literal_ip_host` (the exact function the policy
        // calls on every hop) rather than by constructing an `Attempt` directly.
        let none: Vec<String> = Vec::new();
        let next = reqwest::Url::parse("http://169.254.169.254/latest/meta-data/").unwrap();
        assert!(check_literal_ip_host(&next, &none).is_err());
    }

    fn msg(content: serde_json::Value) -> crate::ChatMessage {
        crate::ChatMessage {
            role: "user".to_string(),
            content,
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
            reasoning: None,
        }
    }

    /// Inline and remote images share one request-wide count across message boundaries.
    #[test]
    fn count_image_urls_counts_inline_and_remote_parts_across_all_messages() {
        let messages = vec![
            msg(json!([
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "http://example.com/a.png"}},
                {"type": "image_url", "image_url": "https://example.com/b.png"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ])),
            msg(json!([
                {"type": "image_url", "image_url": {"url": "http://example.com/c.png"}},
            ])),
            msg(json!("plain string content, no parts at all")),
        ];
        assert_eq!(count_image_urls(&messages), 4);
    }

    #[test]
    fn count_image_urls_is_zero_with_no_image_parts() {
        let messages = vec![msg(json!([{"type": "text", "text": "hi"}]))];
        assert_eq!(count_image_urls(&messages), 0);
    }

    /// The over-cap error carries no `image_url_*` code (PR #904 review): it is the exact
    /// "too many images" shape `lib.rs`'s vision content walkers already produce for
    /// `data:` images, so a caller sees an identical error either way.
    #[test]
    fn too_many_error_has_no_image_url_code_and_names_the_cap() {
        let err = FetchError::TooMany(9);
        assert_eq!(err.code(), None);
        let msg = err.message();
        assert!(msg.contains("too many images"));
        assert!(msg.contains(&crate::VISION_MAX_IMAGES.to_string()));
    }
    #[test]
    fn expanded_request_budget_accounts_for_base64_and_prior_images() {
        let mut budget = WireBudget { used: 60, max: 100 };
        let old_url = 20;
        let limit = budget.raw_limit(old_url).unwrap();
        let encode = |n| {
            format!(
                "data:image/jpeg;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(vec![0; n])
            )
        };
        let fits = encode(limit);
        let too_big = encode(limit + 1);
        assert!(budget.used - old_url + fits.len() <= budget.max);
        assert!(budget.used - old_url + too_big.len() > budget.max);
        assert!(matches!(
            budget.replace(old_url, too_big.len()),
            Err(FetchError::RequestTooLarge)
        ));
        assert_eq!(
            budget.used, 60,
            "a refused replacement must not change the budget"
        );
        budget.replace(old_url, fits.len()).unwrap();
        assert!(
            matches!(budget.raw_limit(old_url), Err(FetchError::RequestTooLarge)),
            "the next image cannot reuse bytes spent by the first"
        );
    }

    #[test]
    fn per_image_cap_and_total_request_cap_have_the_same_named_error() {
        let roomy = WireBudget {
            used: 100,
            max: crate::MAX_BODY_BYTES,
        };
        assert_eq!(roomy.raw_limit(20).unwrap(), FETCH_MAX_BYTES);
        assert!(matches!(
            size_error(FETCH_MAX_BYTES + 1, FETCH_MAX_BYTES),
            FetchError::TooLarge(_)
        ));
        assert!(matches!(size_error(101, 100), FetchError::RequestTooLarge));
        assert_eq!(
            FetchError::RequestTooLarge.code(),
            Some("image_url_too_large")
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // allow: isolate the process-global fetch switch while awaiting the preflight refusal
    async fn mixed_inline_and_remote_images_refuse_before_any_fetch() {
        let _lock = FETCH_ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("MEMRA_FETCH_URLS", "1");
        }
        let mut images = vec![
            json!({"type":"image_url", "image_url":"data:image/png;base64,AAAA"});
            crate::VISION_MAX_IMAGES
        ];
        images.push(json!({"type":"image_url", "image_url":"http://127.0.0.1/never-connect"}));
        let mut messages = vec![msg(json!(images))];
        let result =
            resolve_remote_image_urls_in_messages(&mut messages, 1024, Duration::from_secs(20))
                .await;
        unsafe {
            std::env::remove_var("MEMRA_FETCH_URLS");
        }
        assert!(matches!(result, Err(FetchError::TooMany(n)) if n == crate::VISION_MAX_IMAGES + 1));
    }

    #[test]
    fn literal_aliases_require_explicit_allowlisting() {
        for input in [
            "http://2130706433/x",
            "http://0x7f000001/x",
            "http://[::ffff:127.0.0.1]/x",
            "http://8.8.8.8/x",
        ] {
            let url = reqwest::Url::parse(input).unwrap();
            assert!(
                matches!(
                    check_literal_ip_host(&url, &[]),
                    Err(FetchError::Blocked(_))
                ),
                "{input}"
            );
        }
        assert!(
            check_literal_ip_host(
                &reqwest::Url::parse("http://8.8.8.8/x").unwrap(),
                &["8.8.8.8".into()]
            )
            .is_ok()
        );
        assert!(is_fetchable_url("HTTP://example.com/a.png"));
        assert!(is_fetchable_url("HTTPS://example.com/a.png"));
    }
    /// Runs only against the controlled loopback fixture in image-url-fetch-gate.py.
    /// This proves transport/rewriting bytes, not a model's vision implementation.
    #[tokio::test]
    #[ignore = "requires the controlled image URL HTTP fixture"]
    #[allow(clippy::await_holding_lock)] // allow: isolate fetch env while the one explicit fixture test awaits HTTP
    async fn controlled_http_fixture_rewrites_exact_image_bytes() {
        let _lock = FETCH_ENV_LOCK.lock().unwrap();
        let root = std::env::var("IMAGE_FETCH_FIXTURE_URL").expect("fixture URL required");
        assert!(root.starts_with("http://127.0.0.1:"));
        let dir = std::path::PathBuf::from(
            std::env::var("IMAGE_FETCH_FIXTURE_DIR").expect("fixture directory required"),
        );
        unsafe {
            std::env::set_var("MEMRA_FETCH_URLS", "1");
            std::env::set_var("MEMRA_FETCH_URLS_ALLOWED_HOSTS", "127.0.0.1");
        }
        for (colour, suffix) in [
            ("red", "/red.png"),
            ("red", "/chain/5"),
            ("blue", "/blue.png"),
        ] {
            let bytes = std::fs::read(dir.join(format!("{colour}.png"))).unwrap();
            let expected = format!(
                "data:image/png;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(bytes)
            );
            let mut messages = vec![msg(json!([
                {"type":"text", "text":"unchanged"},
                {"type":"image_url", "image_url":{"url":format!("{root}{suffix}"), "detail":"low"}}
            ]))];
            resolve_remote_image_urls_in_messages(&mut messages, 1024, Duration::from_secs(20))
                .await
                .unwrap();
            assert_eq!(messages[0].content[1]["image_url"]["url"], expected);
            assert_eq!(messages[0].content[1]["image_url"]["detail"], "low");
            assert_eq!(messages[0].content[0]["text"], "unchanged");
        }
        unsafe {
            std::env::remove_var("MEMRA_FETCH_URLS");
            std::env::remove_var("MEMRA_FETCH_URLS_ALLOWED_HOSTS");
        }
        println!(
            "IMAGE_FETCH_FIXTURE_PASS exact red/blue bytes, five redirects, metadata retained"
        );
    }
}
