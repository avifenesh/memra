//! Default-off provisioning surface for generic tenant/key operations.
//!
//! The listener is separate from the public completion router, is loopback-only, and
//! accepts exactly one bearer read from a mode-0600/0640 token file. Audit records are
//! committed before an authorized handler is allowed to mutate durable state.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::net::{SocketAddr, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    Json, Router,
    extract::{Path as AxumPath, Request, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::json;

use super::{auth, capture, ledger};

const AUDIT_FORMAT: &str = "memra.admin-audit.v1";

#[derive(Clone)]
struct AdminState {
    token: Arc<str>,
    keys_path: Arc<PathBuf>,
    ledger: ledger::Ledger,
    budgets: ledger::TenantBudgets,
    keys_lock: Arc<Mutex<()>>,
    audit: AuditLog,
    /// None when MEMRA_CAPTURE_DIR is off; the capture endpoints then answer 503 so a
    /// provisioning caller can tell "not configured" from "cleared".
    capture: Option<capture::CaptureStore>,
    /// Worker command channel for /admin/trim (deploy-headroom lane, 2026-08-27).
    /// None until `router()` receives it — the admin config is built before the
    /// worker spawns; the route answers 503 in that window.
    cmd_tx: Option<std::sync::mpsc::Sender<super::worker::Cmd>>,
}

pub(crate) struct Config {
    addr: SocketAddr,
    state: AdminState,
}

impl Config {
    pub(crate) fn from_env(
        request_ledger: Option<&ledger::Ledger>,
        key_store: Option<&auth::KeyStore>,
        capture: Option<&capture::CaptureStore>,
    ) -> Result<Option<Self>, String> {
        let addr = optional_env("MEMRA_ADMIN_ADDR")?;
        let token_file = optional_env("MEMRA_ADMIN_TOKEN_FILE")?;
        let Some(addr) = addr else {
            if token_file.is_some() {
                return Err("MEMRA_ADMIN_TOKEN_FILE is set but MEMRA_ADMIN_ADDR is off".into());
            }
            return Ok(None);
        };
        if addr.is_empty() {
            return Err("MEMRA_ADMIN_ADDR must not be empty".into());
        }
        let addr = resolve_loopback(&addr)?;
        let token_file = token_file.ok_or_else(|| {
            "MEMRA_ADMIN_ADDR requires MEMRA_ADMIN_TOKEN_FILE; admin auth is never optional"
                .to_string()
        })?;
        if token_file.is_empty() {
            return Err("MEMRA_ADMIN_TOKEN_FILE must not be empty".into());
        }
        let token = read_token_file(Path::new(&token_file))?;
        let ledger = request_ledger
            .cloned()
            .ok_or_else(|| "MEMRA_ADMIN_ADDR requires MEMRA_REQUEST_LEDGER".to_string())?;
        let budgets = ledger.budgets().ok_or_else(|| {
            "MEMRA_ADMIN_ADDR requires MEMRA_TENANT_BUDGETS for credit/balance endpoints"
                .to_string()
        })?;
        let keys_path = key_store
            .and_then(auth::KeyStore::file_path)
            .ok_or_else(|| {
                "MEMRA_ADMIN_ADDR requires file-backed MEMRA_API_KEYS (inline keys cannot be provisioned)"
                    .to_string()
            })?
            .to_path_buf();
        let audit = AuditLog::open(&ledger.admin_audit_path())?;
        Ok(Some(Self {
            addr,
            state: AdminState {
                token: Arc::from(token),
                keys_path: Arc::new(keys_path),
                ledger,
                budgets,
                keys_lock: Arc::new(Mutex::new(())),
                audit,
                capture: capture.cloned(),
                cmd_tx: None,
            },
        }))
    }

    pub(crate) fn addr(&self) -> SocketAddr {
        self.addr
    }

    pub(crate) fn router(&self, cmd_tx: std::sync::mpsc::Sender<super::worker::Cmd>) -> Router {
        let mut state = self.state.clone();
        state.cmd_tx = Some(cmd_tx);
        Router::new()
            .route("/admin/trim", post(trim_pools))
            .route("/admin/keys", post(create_key))
            .route("/admin/keys/revoke", post(revoke_key))
            .route("/admin/tenants/:tenant/usage", get(tenant_usage))
            .route("/admin/tenants/:tenant/credit", post(credit_tenant))
            .route("/admin/tenants/:tenant/debit", post(debit_tenant))
            .route("/admin/tenants/:tenant/balance", get(tenant_balance))
            .route(
                "/admin/tenants/:tenant/admission",
                get(tenant_admission).post(set_admission),
            )
            .route("/admin/readiness", get(readiness))
            .route(
                "/admin/tenants/:tenant/capture",
                get(capture_status).post(set_capture),
            )
            .fallback(not_found)
            .layer(middleware::from_fn_with_state(
                state.clone(),
                authenticate_and_audit,
            ))
            .with_state(state)
    }
}

async fn authenticate_and_audit(
    State(state): State<AdminState>,
    request: Request,
    next: Next,
) -> Response {
    let authorization = authorize(request.headers(), &state.token);
    let audit_result = state.audit.record(
        request.method().as_str(),
        audit_route(request.uri().path()),
        if authorization.is_ok() {
            "authorized"
        } else {
            "denied"
        },
    );
    if let Err(err) = audit_result {
        eprintln!("[admin] ERROR: audit commit failed: {err}");
        return admin_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "admin audit log is unavailable",
            "admin_audit_unavailable",
        );
    }
    if let Err(status) = authorization {
        return match status {
            StatusCode::UNAUTHORIZED => admin_error(
                status,
                "admin bearer token is required",
                "admin_auth_required",
            ),
            _ => admin_error(
                StatusCode::FORBIDDEN,
                "admin bearer token is invalid",
                "admin_auth_forbidden",
            ),
        };
    }
    next.run(request).await
}

fn authorize(headers: &HeaderMap, expected: &str) -> Result<(), StatusCode> {
    let Some(value) = headers.get("authorization") else {
        return Err(StatusCode::UNAUTHORIZED);
    };
    let Some(candidate) = value
        .to_str()
        .ok()
        .and_then(|value| value.strip_prefix("Bearer "))
    else {
        return Err(StatusCode::FORBIDDEN);
    };
    if auth::constant_time_secret_eq(candidate, expected) {
        Ok(())
    } else {
        Err(StatusCode::FORBIDDEN)
    }
}

#[derive(Deserialize)]
struct CreateKeyRequest {
    tenant: String,
    #[serde(default)]
    lane: Option<String>,
    #[serde(default)]
    rate_limit: Option<usize>,
}

async fn create_key(
    State(state): State<AdminState>,
    Json(request): Json<CreateKeyRequest>,
) -> Response {
    if !auth::tenant_is_valid(&request.tenant) {
        return admin_error(
            StatusCode::BAD_REQUEST,
            "tenant must match [A-Za-z0-9_-]+",
            "invalid_tenant",
        );
    }
    let lane = match request.lane.as_deref() {
        None => auth::LaneClass::Interactive,
        Some(value) => match auth::LaneClass::parse(value) {
            Some(lane) => lane,
            None => {
                return admin_error(
                    StatusCode::BAD_REQUEST,
                    "lane must be interactive or batch",
                    "invalid_lane",
                );
            }
        },
    };
    if request.rate_limit == Some(0) {
        return admin_error(
            StatusCode::BAD_REQUEST,
            "rate_limit must be a positive integer when present",
            "invalid_rate_limit",
        );
    }
    // In budget-enabled mode a key for an absent tenant would be valid auth but
    // unmetered inference. Refuse it before mutating the keyring; the console
    // performs the same all-origins check before requesting this endpoint.
    match state.budgets.balance(&request.tenant) {
        Ok(Some(balance)) if balance.admission_mode != ledger::AdmissionMode::Blocked => {}
        Ok(Some(_)) => {
            return admin_error(
                StatusCode::CONFLICT,
                "tenant admission is blocked by provisioning policy",
                "tenant_admission_blocked",
            );
        }
        Ok(None) => {
            return admin_error(
                StatusCode::CONFLICT,
                "tenant is not enrolled for prepaid billing",
                "tenant_not_enrolled",
            );
        }
        Err(err) => {
            eprintln!("[admin] key creation budget check failed: {err}");
            return admin_budget_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "tenant budget accounting is unavailable",
                "tenant_budget_unavailable",
                state.budgets.health(),
            );
        }
    }
    let _keys = match state.keys_lock.lock() {
        Ok(lock) => lock,
        Err(_) => {
            return admin_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "keyring mutation lock is unavailable",
                "keyring_unavailable",
            );
        }
    };
    let key = match auth::gen_key(&state.keys_path, &request.tenant, lane, request.rate_limit) {
        Ok(key) => key,
        Err(err) => {
            eprintln!(
                "[admin] key creation failed for tenant {:?}: {err}",
                request.tenant
            );
            return admin_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "could not persist the API key",
                "keyring_unavailable",
            );
        }
    };
    let prefix = display_prefix(&key).expect("gen_key always emits a 48-hex secret");
    Json(json!({
        "key": key,
        "prefix": prefix,
        "tenant": request.tenant,
        "lane": lane.as_str(),
        "rate_limit": request.rate_limit,
    }))
    .into_response()
}

#[derive(Deserialize)]
struct RevokeKeyRequest {
    prefix: String,
}

async fn revoke_key(
    State(state): State<AdminState>,
    Json(request): Json<RevokeKeyRequest>,
) -> Response {
    if request.prefix.is_empty()
        || request.prefix.len() > 256
        || !request
            .prefix
            .bytes()
            .all(|byte| (0x21..=0x7e).contains(&byte))
    {
        return admin_error(
            StatusCode::BAD_REQUEST,
            "prefix must be 1..256 printable ASCII characters",
            "invalid_prefix",
        );
    }
    let _keys = match state.keys_lock.lock() {
        Ok(lock) => lock,
        Err(_) => {
            return admin_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "keyring mutation lock is unavailable",
                "keyring_unavailable",
            );
        }
    };
    match auth::revoke_key(&state.keys_path, &request.prefix) {
        Ok(prefix) => Json(json!({ "prefix": prefix, "revoked": true })).into_response(),
        Err(err) if err.contains("no key matches") || err.contains("already revoked") => {
            admin_error(StatusCode::NOT_FOUND, &err, "key_not_found")
        }
        Err(err) if err.contains(" keys match") => {
            admin_error(StatusCode::CONFLICT, &err, "ambiguous_prefix")
        }
        Err(err) => {
            eprintln!("[admin] key revocation failed: {err}");
            admin_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "could not persist key revocation",
                "keyring_unavailable",
            )
        }
    }
}

async fn tenant_usage(
    State(state): State<AdminState>,
    AxumPath(tenant): AxumPath<String>,
) -> Response {
    if !auth::tenant_is_valid(&tenant) {
        return admin_error(
            StatusCode::BAD_REQUEST,
            "tenant must match [A-Za-z0-9_-]+",
            "invalid_tenant",
        );
    }
    match state.ledger.tenant_usage(&tenant) {
        Ok(usage) => Json(usage).into_response(),
        Err(err) => {
            eprintln!("[admin] tenant usage read failed: {err}");
            admin_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "request ledger is unavailable",
                "request_ledger_unavailable",
            )
        }
    }
}

#[derive(Deserialize)]
struct CreditRequest {
    amount_micro: i64,
    idempotency_key: String,
}

async fn credit_tenant(
    State(state): State<AdminState>,
    AxumPath(tenant): AxumPath<String>,
    Json(request): Json<CreditRequest>,
) -> Response {
    match state
        .budgets
        .credit(&tenant, request.amount_micro, &request.idempotency_key)
    {
        Ok(result) => Json(result).into_response(),
        Err(ledger::CreditError::Invalid(message)) => {
            admin_error(StatusCode::BAD_REQUEST, &message, "invalid_credit")
        }
        Err(ledger::CreditError::Conflict(message)) => {
            admin_error(StatusCode::CONFLICT, &message, "idempotency_conflict")
        }
        Err(ledger::CreditError::Unavailable(err)) => {
            eprintln!("[admin] tenant credit failed: {err}");
            admin_budget_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "tenant budget accounting is unavailable",
                "tenant_budget_unavailable",
                state.budgets.health(),
            )
        }
    }
}

#[derive(Deserialize)]
struct DebitRequest {
    amount_micro: i64,
    idempotency_key: String,
    reason: String,
}

async fn debit_tenant(
    State(state): State<AdminState>,
    AxumPath(tenant): AxumPath<String>,
    Json(request): Json<DebitRequest>,
) -> Response {
    match state.budgets.admin_debit(
        &tenant,
        request.amount_micro,
        &request.idempotency_key,
        &request.reason,
    ) {
        Ok(result) => Json(result).into_response(),
        Err(ledger::CreditError::Invalid(message)) => {
            admin_error(StatusCode::BAD_REQUEST, &message, "invalid_debit")
        }
        Err(ledger::CreditError::Conflict(message)) => {
            admin_error(StatusCode::CONFLICT, &message, "idempotency_conflict")
        }
        Err(ledger::CreditError::Unavailable(err)) => {
            eprintln!("[admin] tenant debit failed: {err}");
            admin_budget_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "tenant budget accounting is unavailable",
                "tenant_budget_unavailable",
                state.budgets.health(),
            )
        }
    }
}

/// Authenticated billing-readiness probe used by the console's cross-origin
/// router gate. A public /readyz only proves that the inference worker is alive;
/// this endpoint additionally proves that the budget journal is available, so a
/// route cannot be advertised as metered while the accounting source is down.
async fn readiness(State(state): State<AdminState>) -> Response {
    let health = state.budgets.health();
    let ready = health.source_available;
    let body = json!({
        "ready": ready,
        "budget_source_available": health.source_available,
        "budget_source_reload_consecutive": health.source_reload_consecutive,
        "admission_modes": true,
    });
    if ready {
        (StatusCode::OK, Json(body)).into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response()
    }
}

async fn tenant_balance(
    State(state): State<AdminState>,
    AxumPath(tenant): AxumPath<String>,
) -> Response {
    if !auth::tenant_is_valid(&tenant) {
        return admin_error(
            StatusCode::BAD_REQUEST,
            "tenant must match [A-Za-z0-9_-]+",
            "invalid_tenant",
        );
    }
    match state.budgets.balance(&tenant) {
        Ok(balance) => {
            let health = state.budgets.health();
            match balance {
                Some(balance) => Json(json!({
                    "limited": true,
                    "tenant": balance.tenant,
                    "currency": balance.currency,
                    "balance_micro": balance.balance_micro,
                    "admission_mode": balance.admission_mode,
                    "budget_source_reload_failed": health.source_reload_failed,
                    "budget_source_reload_consecutive": health.source_reload_consecutive,
                    "budget_source_available": health.source_available,
                }))
                .into_response(),
                None => Json(json!({
                    "limited": false,
                    "tenant": tenant,
                    "currency": "USD",
                    "balance_micro": null,
                    "budget_source_reload_failed": health.source_reload_failed,
                    "budget_source_reload_consecutive": health.source_reload_consecutive,
                    "budget_source_available": health.source_available,
                }))
                .into_response(),
            }
        }
        Err(err) => {
            eprintln!("[admin] tenant balance failed: {err}");
            admin_budget_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "tenant budget accounting is unavailable",
                "tenant_budget_unavailable",
                state.budgets.health(),
            )
        }
    }
}

#[derive(Deserialize)]
struct AdmissionRequest {
    expected_mode: ledger::AdmissionMode,
    mode: ledger::AdmissionMode,
}

async fn tenant_admission(
    State(state): State<AdminState>,
    AxumPath(tenant): AxumPath<String>,
) -> Response {
    if !auth::tenant_is_valid(&tenant) {
        return admin_error(
            StatusCode::BAD_REQUEST,
            "tenant must match [A-Za-z0-9_-]+",
            "invalid_tenant",
        );
    }
    match state.budgets.balance(&tenant) {
        Ok(Some(balance)) => Json(json!({
            "tenant": tenant,
            "mode": balance.admission_mode,
        }))
        .into_response(),
        Ok(None) => admin_error(
            StatusCode::NOT_FOUND,
            "tenant is not enrolled for prepaid billing",
            "tenant_not_enrolled",
        ),
        Err(err) => {
            eprintln!("[admin] tenant admission read failed: {err}");
            admin_budget_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "tenant admission state is unavailable",
                "tenant_budget_unavailable",
                state.budgets.health(),
            )
        }
    }
}

async fn set_admission(
    State(state): State<AdminState>,
    AxumPath(tenant): AxumPath<String>,
    Json(request): Json<AdmissionRequest>,
) -> Response {
    if !auth::tenant_is_valid(&tenant) {
        return admin_error(
            StatusCode::BAD_REQUEST,
            "tenant must match [A-Za-z0-9_-]+",
            "invalid_tenant",
        );
    }
    match state
        .budgets
        .set_admission_mode(&tenant, request.expected_mode, request.mode)
    {
        Ok(result) => Json(json!({
            "tenant": result.tenant,
            "mode": result.admission_mode,
        }))
        .into_response(),
        Err(ledger::AdmissionError::InvalidTenant(message)) => {
            admin_error(StatusCode::BAD_REQUEST, &message, "invalid_tenant")
        }
        Err(ledger::AdmissionError::Unenrolled(message)) => {
            admin_error(StatusCode::NOT_FOUND, &message, "tenant_not_enrolled")
        }
        Err(ledger::AdmissionError::Conflict(message)) => {
            admin_error(StatusCode::CONFLICT, &message, "admission_mode_conflict")
        }
        Err(ledger::AdmissionError::Unavailable(err)) => {
            eprintln!("[admin] tenant admission update failed: {err}");
            admin_budget_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "tenant admission state is unavailable",
                "tenant_budget_unavailable",
                state.budgets.health(),
            )
        }
    }
}

#[derive(Deserialize)]
struct CaptureRequest {
    mode: String,
}

/// Set or clear a tenant's capture mark. `mode` is `trial`, `consent`, or `off`. The
/// caller (the provisioning system) owns the meaning of the marks; this endpoint only
/// refuses vocabulary it does not know, so a typo can never silently retain data.
async fn set_capture(
    State(state): State<AdminState>,
    AxumPath(tenant): AxumPath<String>,
    Json(request): Json<CaptureRequest>,
) -> Response {
    if !auth::tenant_is_valid(&tenant) {
        return admin_error(
            StatusCode::BAD_REQUEST,
            "tenant must match [A-Za-z0-9_-]+",
            "invalid_tenant",
        );
    }
    let Some(store) = state.capture.as_ref() else {
        return admin_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "request capture is not configured on this server (MEMRA_CAPTURE_DIR is off)",
            "capture_not_configured",
        );
    };
    let Ok(mode) = capture::parse_mode_request(&request.mode) else {
        return admin_error(
            StatusCode::BAD_REQUEST,
            "mode must be trial, consent, or off",
            "invalid_capture_mode",
        );
    };
    match store.set_mode(&tenant, mode) {
        Ok(()) => Json(store.status(&tenant)).into_response(),
        Err(err) => {
            eprintln!("[admin] capture mode update failed for tenant {tenant:?}: {err}");
            admin_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "could not persist the capture mark",
                "capture_unavailable",
            )
        }
    }
}

async fn capture_status(
    State(state): State<AdminState>,
    AxumPath(tenant): AxumPath<String>,
) -> Response {
    if !auth::tenant_is_valid(&tenant) {
        return admin_error(
            StatusCode::BAD_REQUEST,
            "tenant must match [A-Za-z0-9_-]+",
            "invalid_tenant",
        );
    }
    let Some(store) = state.capture.as_ref() else {
        return admin_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "request capture is not configured on this server (MEMRA_CAPTURE_DIR is off)",
            "capture_not_configured",
        );
    };
    Json(store.status(&tenant)).into_response()
}

async fn not_found() -> Response {
    admin_error(
        StatusCode::NOT_FOUND,
        "admin endpoint not found",
        "admin_not_found",
    )
}

fn admin_error(status: StatusCode, message: &str, code: &str) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": message,
                "type": "admin_error",
                "param": null,
                "code": code,
            }
        })),
    )
        .into_response()
}

fn admin_budget_error(
    status: StatusCode,
    message: &str,
    code: &str,
    health: ledger::BudgetHealth,
) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": message,
                "type": "admin_error",
                "param": null,
                "code": code,
            },
            "budget_source_reload_failed": health.source_reload_failed,
            "budget_source_reload_consecutive": health.source_reload_consecutive,
            "budget_source_available": health.source_available,
        })),
    )
        .into_response()
}

fn display_prefix(key: &str) -> Option<String> {
    let secret_start = key.len().checked_sub(48)?;
    let secret_end = secret_start.checked_add(12)?;
    Some(format!(
        "{}{}",
        &key[..secret_start],
        &key[secret_start..secret_end]
    ))
}

fn optional_env(name: &str) -> Result<Option<String>, String> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!("{name} must be valid UTF-8")),
    }
}

fn resolve_loopback(addr: &str) -> Result<SocketAddr, String> {
    let addresses: Vec<SocketAddr> = addr
        .to_socket_addrs()
        .map_err(|e| format!("MEMRA_ADMIN_ADDR={addr:?} cannot be resolved: {e}"))?
        .collect();
    let first = addresses
        .first()
        .copied()
        .ok_or_else(|| format!("MEMRA_ADMIN_ADDR={addr:?} resolved to no socket addresses"))?;
    if addresses
        .iter()
        .any(|address| !address.ip().to_canonical().is_loopback())
    {
        return Err(format!(
            "MEMRA_ADMIN_ADDR={addr:?} must resolve only to loopback addresses"
        ));
    }
    // Bind this exact verified address rather than resolving the hostname a second time.
    Ok(first)
}

/// POST /admin/trim — drop every evictable cross-request pool in the worker (KV
/// reuse, spec/dspark resume, prefix cache) and return the freed entry counts.
/// The deploy-headroom caller: serve-deploy frees VRAM for a green slot beside a
/// warm blue. In-flight sessions and pinned prefix leases are untouched.
async fn trim_pools(State(state): State<AdminState>) -> Response {
    let Some(cmd_tx) = state.cmd_tx.as_ref() else {
        return service_unavailable("worker command channel not wired yet");
    };
    let (tx, rx) = tokio::sync::oneshot::channel();
    if cmd_tx.send(super::worker::Cmd::TrimPools(tx)).is_err() {
        return service_unavailable("worker is down");
    }
    match tokio::time::timeout(std::time::Duration::from_secs(30), rx).await {
        Ok(Ok(report)) => (StatusCode::OK, Json(json!(report))).into_response(),
        _ => service_unavailable("worker did not answer the trim within 30s"),
    }
}

fn service_unavailable(msg: &str) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({"error": {"message": msg, "type": "admin_error"}})),
    )
        .into_response()
}

fn audit_route(path: &str) -> &'static str {
    let segments: Vec<&str> = path
        .trim_matches('/')
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect();
    match segments.as_slice() {
        ["admin", "trim"] => "/admin/trim",
        ["admin", "keys"] => "/admin/keys",
        ["admin", "keys", "revoke"] => "/admin/keys/revoke",
        ["admin", "tenants", _, "usage"] => "/admin/tenants/{tenant}/usage",
        ["admin", "tenants", _, "credit"] => "/admin/tenants/{tenant}/credit",
        ["admin", "tenants", _, "debit"] => "/admin/tenants/{tenant}/debit",
        ["admin", "tenants", _, "balance"] => "/admin/tenants/{tenant}/balance",
        ["admin", "tenants", _, "admission"] => "/admin/tenants/{tenant}/admission",
        ["admin", "readiness"] => "/admin/readiness",
        ["admin", "tenants", _, "capture"] => "/admin/tenants/{tenant}/capture",
        _ => "/admin/*",
    }
}

fn read_token_file(path: &Path) -> Result<String, String> {
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt as _;

    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NOFOLLOW);
    let file = options
        .open(path)
        .map_err(|e| format!("open MEMRA_ADMIN_TOKEN_FILE {}: {e}", path.display()))?;
    validate_private_file(&file, path, "admin token file")?;
    let mut token = String::new();
    use std::io::Read as _;
    (&file)
        .read_to_string(&mut token)
        .map_err(|e| format!("read MEMRA_ADMIN_TOKEN_FILE {}: {e}", path.display()))?;
    let token = token.trim_end_matches(['\r', '\n']).to_string();
    if token.is_empty() || !token.bytes().all(|byte| (0x21..=0x7e).contains(&byte)) {
        return Err(
            "MEMRA_ADMIN_TOKEN_FILE must contain one non-empty printable ASCII bearer token".into(),
        );
    }
    Ok(token)
}

#[derive(Clone)]
struct AuditLog {
    inner: Arc<AuditInner>,
}

struct AuditInner {
    path: PathBuf,
    file: Mutex<File>,
}

impl AuditLog {
    fn open(path: &Path) -> Result<Self, String> {
        #[cfg(unix)]
        use std::os::unix::fs::OpenOptionsExt as _;

        let creating = !path.exists();
        let mut options = OpenOptions::new();
        options.create(true).append(true);
        #[cfg(unix)]
        options.mode(0o640).custom_flags(libc::O_NOFOLLOW);
        let file = options
            .open(path)
            .map_err(|e| format!("open admin audit log {}: {e}", path.display()))?;
        validate_private_file(&file, path, "admin audit log")?;
        if creating {
            sync_parent(path, "admin audit log")?;
        }
        Ok(Self {
            inner: Arc::new(AuditInner {
                path: path.to_path_buf(),
                file: Mutex::new(file),
            }),
        })
    }

    fn record(&self, method: &str, path: &str, authorization: &str) -> Result<(), String> {
        let mut row = serde_json::to_vec(&json!({
            "format": AUDIT_FORMAT,
            "unix_ms": unix_ms(),
            "method": method,
            "path": path,
            "authorization": authorization,
        }))
        .map_err(|e| format!("serialize admin audit row: {e}"))?;
        row.push(b'\n');
        let mut file = self
            .inner
            .file
            .lock()
            .map_err(|_| "admin audit writer mutex is poisoned".to_string())?;
        file.write_all(&row)
            .and_then(|()| file.sync_data())
            .map_err(|e| format!("append admin audit log {}: {e}", self.inner.path.display()))
    }
}

fn sync_parent(path: &Path, label: &str) -> Result<(), String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|e| format!("sync {label} directory {}: {e}", parent.display()))
}

fn validate_private_file(file: &File, path: &Path, label: &str) -> Result<(), String> {
    let metadata = file
        .metadata()
        .map_err(|e| format!("stat {label} {}: {e}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!("{label} {} is not a regular file", path.display()));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let mode = metadata.permissions().mode() & 0o777;
        if mode & 0o137 != 0 {
            return Err(format!(
                "{label} {} must have 0600 or 0640-class permissions; found {mode:04o}",
                path.display(),
            ));
        }
    }
    Ok(())
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admin_auth_distinguishes_missing_and_invalid_bearers() {
        let headers = HeaderMap::new();
        assert_eq!(
            authorize(&headers, "admin-secret"),
            Err(StatusCode::UNAUTHORIZED)
        );

        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer wrong-secret".parse().unwrap());
        assert_eq!(
            authorize(&headers, "admin-secret"),
            Err(StatusCode::FORBIDDEN)
        );
        headers.insert("authorization", "Basic admin-secret".parse().unwrap());
        assert_eq!(
            authorize(&headers, "admin-secret"),
            Err(StatusCode::FORBIDDEN)
        );
        headers.insert("authorization", "Bearer admin-secret".parse().unwrap());
        assert_eq!(authorize(&headers, "admin-secret"), Ok(()));
    }

    #[tokio::test]
    async fn admin_router_returns_401_and_403_and_audits_without_tokens() {
        use tower::ServiceExt as _;

        let dir = std::env::temp_dir().join(format!(
            "memra-admin-auth-{}-{}",
            std::process::id(),
            unix_ms(),
        ));
        std::fs::create_dir(&dir).unwrap();
        let keys_path = dir.join("keys.toml");
        let budget_path = dir.join("budgets.toml");
        std::fs::write(&keys_path, "").unwrap();
        std::fs::write(
            &budget_path,
            "[[budgets]]\ntenant = \"acme\"\ncurrency = \"USD\"\nbalance_micro = 0\n",
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            for path in [&keys_path, &budget_path] {
                std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o640)).unwrap();
            }
        }
        let request_path = dir.join("requests.jsonl");
        let metadata = super::super::OpenRouterModelMetadata {
            pricing: super::super::OpenRouterPricing {
                prompt: Some("0.000001".into()),
                cached_prompt: Some("0.0000001".into()),
                completion: Some("0.000002".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        let ledger =
            ledger::Ledger::for_test_with_budgets(&request_path, &budget_path, "m", &metadata);
        let audit_path = ledger.admin_audit_path();
        let config = Config {
            addr: "127.0.0.1:8005".parse().unwrap(),
            state: AdminState {
                token: Arc::from("admin-secret"),
                keys_path: Arc::new(keys_path),
                budgets: ledger.budgets().unwrap(),
                audit: AuditLog::open(&audit_path).unwrap(),
                ledger,
                keys_lock: Arc::new(Mutex::new(())),
                capture: None,
                cmd_tx: None,
            },
        };
        let router = config.router(std::sync::mpsc::channel().0);
        let request = || {
            axum::http::Request::builder()
                .uri("/admin/tenants/acme/balance")
                .body(axum::body::Body::empty())
                .unwrap()
        };
        assert_eq!(
            router.clone().oneshot(request()).await.unwrap().status(),
            StatusCode::UNAUTHORIZED,
        );
        let mut wrong = request();
        wrong
            .headers_mut()
            .insert("authorization", "Bearer wrong-secret".parse().unwrap());
        assert_eq!(
            router.clone().oneshot(wrong).await.unwrap().status(),
            StatusCode::FORBIDDEN,
        );
        let injected = axum::http::Request::builder()
            .uri("/admin/do-not-log-this-secret")
            .body(axum::body::Body::empty())
            .unwrap();
        assert_eq!(
            router.clone().oneshot(injected).await.unwrap().status(),
            StatusCode::UNAUTHORIZED,
        );
        let mut valid = request();
        valid
            .headers_mut()
            .insert("authorization", "Bearer admin-secret".parse().unwrap());
        let valid = router.clone().oneshot(valid).await.unwrap();
        assert_eq!(valid.status(), StatusCode::OK);
        let valid_body = axum::body::to_bytes(valid.into_body(), usize::MAX)
            .await
            .unwrap();
        let valid_body: serde_json::Value = serde_json::from_slice(&valid_body).unwrap();
        assert_eq!(valid_body["budget_source_reload_failed"], 0);
        assert_eq!(valid_body["budget_source_reload_consecutive"], 0);
        assert_eq!(valid_body["budget_source_available"], true);
        assert_eq!(valid_body["admission_mode"], "prepaid");
        let readiness = axum::http::Request::builder()
            .uri("/admin/readiness")
            .header("authorization", "Bearer admin-secret")
            .body(axum::body::Body::empty())
            .unwrap();
        let readiness = router.clone().oneshot(readiness).await.unwrap();
        assert_eq!(readiness.status(), StatusCode::OK);
        let readiness_body = axum::body::to_bytes(readiness.into_body(), usize::MAX)
            .await
            .unwrap();
        let readiness_body: serde_json::Value = serde_json::from_slice(&readiness_body).unwrap();
        assert_eq!(readiness_body["ready"], true);
        assert_eq!(readiness_body["admission_modes"], true);

        let admission_post = |body: &str| {
            axum::http::Request::builder()
                .method("POST")
                .uri("/admin/tenants/acme/admission")
                .header("authorization", "Bearer admin-secret")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body.to_string()))
                .unwrap()
        };
        let body_of = |response: axum::response::Response| async {
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()
        };
        let metered = router
            .clone()
            .oneshot(admission_post(
                r#"{"expected_mode":"prepaid","mode":"metered"}"#,
            ))
            .await
            .unwrap();
        assert_eq!(metered.status(), StatusCode::OK);
        assert_eq!(body_of(metered).await["mode"], "metered");
        let converged_replay = router
            .clone()
            .oneshot(admission_post(
                r#"{"expected_mode":"prepaid","mode":"metered"}"#,
            ))
            .await
            .unwrap();
        assert_eq!(converged_replay.status(), StatusCode::OK);
        assert_eq!(body_of(converged_replay).await["mode"], "metered");
        let admission_get = axum::http::Request::builder()
            .uri("/admin/tenants/acme/admission")
            .header("authorization", "Bearer admin-secret")
            .body(axum::body::Body::empty())
            .unwrap();
        let admission_get = router.clone().oneshot(admission_get).await.unwrap();
        assert_eq!(body_of(admission_get).await["mode"], "metered");
        let stale = router
            .clone()
            .oneshot(admission_post(
                r#"{"expected_mode":"prepaid","mode":"blocked"}"#,
            ))
            .await
            .unwrap();
        assert_eq!(stale.status(), StatusCode::CONFLICT);
        let locked = router
            .clone()
            .oneshot(admission_post(
                r#"{"expected_mode":"metered","mode":"prepaid_locked"}"#,
            ))
            .await
            .unwrap();
        assert_eq!(locked.status(), StatusCode::OK);
        assert_eq!(body_of(locked).await["mode"], "prepaid_locked");
        let downgrade = router
            .clone()
            .oneshot(admission_post(
                r#"{"expected_mode":"prepaid_locked","mode":"metered"}"#,
            ))
            .await
            .unwrap();
        assert_eq!(downgrade.status(), StatusCode::CONFLICT);
        let unenrolled = axum::http::Request::builder()
            .method("POST")
            .uri("/admin/tenants/ghost/admission")
            .header("authorization", "Bearer admin-secret")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"expected_mode":"prepaid","mode":"metered"}"#,
            ))
            .unwrap();
        let unenrolled = router.clone().oneshot(unenrolled).await.unwrap();
        assert_eq!(unenrolled.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            body_of(unenrolled).await["error"]["code"],
            "tenant_not_enrolled"
        );
        let invalid_tenant = axum::http::Request::builder()
            .method("POST")
            .uri("/admin/tenants/bad%20tenant/admission")
            .header("authorization", "Bearer admin-secret")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"expected_mode":"prepaid","mode":"metered"}"#,
            ))
            .unwrap();
        let invalid_tenant = router.clone().oneshot(invalid_tenant).await.unwrap();
        assert_eq!(invalid_tenant.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            body_of(invalid_tenant).await["error"]["code"],
            "invalid_tenant"
        );
        let mut balance_after = request();
        balance_after
            .headers_mut()
            .insert("authorization", "Bearer admin-secret".parse().unwrap());
        let balance_after = router.clone().oneshot(balance_after).await.unwrap();
        assert_eq!(
            body_of(balance_after).await["admission_mode"],
            "prepaid_locked"
        );
        let audit = std::fs::read_to_string(&audit_path).unwrap();
        assert_eq!(audit.lines().count(), 14);
        assert!(!audit.contains("admin-secret"));
        assert!(!audit.contains("wrong-secret"));
        assert!(!audit.contains("do-not-log-this-secret"));
        assert!(audit.contains("/admin/tenants/{tenant}/balance"));
        assert!(audit.contains("/admin/tenants/{tenant}/admission"));
        assert!(audit.contains("/admin/readiness"));
        assert!(audit.contains("/admin/*"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn admin_debit_endpoint_debits_audits_and_refuses_a_missing_reason() {
        use tower::ServiceExt as _;

        let dir = std::env::temp_dir().join(format!(
            "memra-admin-debit-{}-{}",
            std::process::id(),
            unix_ms(),
        ));
        std::fs::create_dir(&dir).unwrap();
        let keys_path = dir.join("keys.toml");
        let budget_path = dir.join("budgets.toml");
        std::fs::write(&keys_path, "").unwrap();
        std::fs::write(
            &budget_path,
            "[[budgets]]\ntenant = \"acme\"\ncurrency = \"USD\"\nbalance_micro = 100\n",
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            for path in [&keys_path, &budget_path] {
                std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o640)).unwrap();
            }
        }
        let request_path = dir.join("requests.jsonl");
        let metadata = super::super::OpenRouterModelMetadata {
            pricing: super::super::OpenRouterPricing {
                prompt: Some("0.000001".into()),
                cached_prompt: Some("0.0000001".into()),
                completion: Some("0.000002".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        let ledger =
            ledger::Ledger::for_test_with_budgets(&request_path, &budget_path, "m", &metadata);
        let audit_path = ledger.admin_audit_path();
        let config = Config {
            addr: "127.0.0.1:8005".parse().unwrap(),
            state: AdminState {
                token: Arc::from("admin-secret"),
                keys_path: Arc::new(keys_path),
                budgets: ledger.budgets().unwrap(),
                audit: AuditLog::open(&audit_path).unwrap(),
                ledger,
                keys_lock: Arc::new(Mutex::new(())),
                capture: None,
                cmd_tx: None,
            },
        };
        let router = config.router(std::sync::mpsc::channel().0);
        let post = |body: &str| {
            axum::http::Request::builder()
                .method("POST")
                .uri("/admin/tenants/acme/debit")
                .header("authorization", "Bearer admin-secret")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body.to_string()))
                .unwrap()
        };
        let ok = router
            .clone()
            .oneshot(post(
                r#"{"amount_micro":40,"idempotency_key":"mig-1","reason":"DE spend migration"}"#,
            ))
            .await
            .unwrap();
        assert_eq!(ok.status(), StatusCode::OK);
        let body = axum::body::to_bytes(ok.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["balance_micro"], 60);
        let bad = router
            .clone()
            .oneshot(post(
                r#"{"amount_micro":40,"idempotency_key":"mig-2","reason":"  "}"#,
            ))
            .await
            .unwrap();
        assert_eq!(bad.status(), StatusCode::BAD_REQUEST);
        let conflict = router
            .oneshot(post(
                r#"{"amount_micro":41,"idempotency_key":"mig-1","reason":"DE spend migration"}"#,
            ))
            .await
            .unwrap();
        assert_eq!(conflict.status(), StatusCode::CONFLICT);
        let audit = std::fs::read_to_string(&audit_path).unwrap();
        assert!(audit.contains("/admin/tenants/{tenant}/debit"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn admin_capture_endpoint_sets_clears_and_rejects_unknown_modes() {
        use tower::ServiceExt as _;

        let dir = std::env::temp_dir().join(format!(
            "memra-admin-capture-{}-{}",
            std::process::id(),
            unix_ms(),
        ));
        std::fs::create_dir(&dir).unwrap();
        let keys_path = dir.join("keys.toml");
        let budget_path = dir.join("budgets.toml");
        std::fs::write(&keys_path, "").unwrap();
        std::fs::write(
            &budget_path,
            "[[budgets]]\ntenant = \"acme\"\ncurrency = \"USD\"\nbalance_micro = 0\n",
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            for path in [&keys_path, &budget_path] {
                std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o640)).unwrap();
            }
        }
        let request_path = dir.join("requests.jsonl");
        let metadata = super::super::OpenRouterModelMetadata {
            pricing: super::super::OpenRouterPricing {
                prompt: Some("0.000001".into()),
                cached_prompt: Some("0.0000001".into()),
                completion: Some("0.000002".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        let ledger =
            ledger::Ledger::for_test_with_budgets(&request_path, &budget_path, "m", &metadata);
        let audit_path = ledger.admin_audit_path();
        let store = capture::CaptureStore::open(&dir.join("capture")).unwrap();
        let config = Config {
            addr: "127.0.0.1:8005".parse().unwrap(),
            state: AdminState {
                token: Arc::from("admin-secret"),
                keys_path: Arc::new(keys_path),
                budgets: ledger.budgets().unwrap(),
                audit: AuditLog::open(&audit_path).unwrap(),
                ledger,
                keys_lock: Arc::new(Mutex::new(())),
                capture: Some(store.clone()),
                cmd_tx: None,
            },
        };
        let router = config.router(std::sync::mpsc::channel().0);
        let call = |method: &str, body: Option<&str>| {
            let mut builder = axum::http::Request::builder()
                .method(method)
                .uri("/admin/tenants/acme/capture")
                .header("authorization", "Bearer admin-secret");
            if body.is_some() {
                builder = builder.header("content-type", "application/json");
            }
            builder
                .body(match body {
                    Some(body) => axum::body::Body::from(body.to_string()),
                    None => axum::body::Body::empty(),
                })
                .unwrap()
        };
        let body_of = |response: axum::response::Response| async {
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()
        };

        let set = router
            .clone()
            .oneshot(call("POST", Some(r#"{"mode":"trial"}"#)))
            .await
            .unwrap();
        assert_eq!(set.status(), StatusCode::OK);
        assert_eq!(body_of(set).await["mode"], "trial");
        assert_eq!(
            store.mode("acme"),
            Some(capture::CaptureMode::Trial),
            "the admin write must reach the live store the serving path reads",
        );

        let status = router.clone().oneshot(call("GET", None)).await.unwrap();
        assert_eq!(body_of(status).await["mode"], "trial");

        let cleared = router
            .clone()
            .oneshot(call("POST", Some(r#"{"mode":"off"}"#)))
            .await
            .unwrap();
        assert_eq!(body_of(cleared).await["mode"], "off");
        assert_eq!(store.mode("acme"), None);

        let unknown = router
            .clone()
            .oneshot(call("POST", Some(r#"{"mode":"paid"}"#)))
            .await
            .unwrap();
        assert_eq!(unknown.status(), StatusCode::BAD_REQUEST);
        assert_eq!(store.mode("acme"), None, "a typo must never mark a tenant");

        let audit = std::fs::read_to_string(&audit_path).unwrap();
        assert!(audit.contains("/admin/tenants/{tenant}/capture"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn admin_capture_endpoint_answers_503_when_not_configured() {
        use tower::ServiceExt as _;

        let dir = std::env::temp_dir().join(format!(
            "memra-admin-nocapture-{}-{}",
            std::process::id(),
            unix_ms(),
        ));
        std::fs::create_dir(&dir).unwrap();
        let keys_path = dir.join("keys.toml");
        let budget_path = dir.join("budgets.toml");
        std::fs::write(&keys_path, "").unwrap();
        std::fs::write(
            &budget_path,
            "[[budgets]]\ntenant = \"acme\"\ncurrency = \"USD\"\nbalance_micro = 0\n",
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            for path in [&keys_path, &budget_path] {
                std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o640)).unwrap();
            }
        }
        let request_path = dir.join("requests.jsonl");
        let metadata = super::super::OpenRouterModelMetadata {
            pricing: super::super::OpenRouterPricing {
                prompt: Some("0.000001".into()),
                cached_prompt: Some("0.0000001".into()),
                completion: Some("0.000002".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        let ledger =
            ledger::Ledger::for_test_with_budgets(&request_path, &budget_path, "m", &metadata);
        let audit_path = ledger.admin_audit_path();
        let config = Config {
            addr: "127.0.0.1:8005".parse().unwrap(),
            state: AdminState {
                token: Arc::from("admin-secret"),
                keys_path: Arc::new(keys_path),
                budgets: ledger.budgets().unwrap(),
                audit: AuditLog::open(&audit_path).unwrap(),
                ledger,
                keys_lock: Arc::new(Mutex::new(())),
                capture: None,
                cmd_tx: None,
            },
        };
        let ghost = axum::http::Request::builder()
            .method("POST")
            .uri("/admin/keys")
            .header("authorization", "Bearer admin-secret")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"tenant":"ghost","lane":"interactive"}"#,
            ))
            .unwrap();
        let ghost_response = config
            .router(std::sync::mpsc::channel().0)
            .oneshot(ghost)
            .await
            .unwrap();
        assert_eq!(ghost_response.status(), StatusCode::CONFLICT);
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/admin/tenants/acme/capture")
            .header("authorization", "Bearer admin-secret")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(r#"{"mode":"trial"}"#))
            .unwrap();
        let response = config
            .router(std::sync::mpsc::channel().0)
            .oneshot(request)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn generated_key_display_prefix_contains_no_full_secret() {
        let key = "mk-tenant-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        assert_eq!(
            display_prefix(key).as_deref(),
            Some("mk-tenant-aaaaaaaaaaaa")
        );
    }

    #[test]
    fn admin_listener_must_be_loopback() {
        assert!(resolve_loopback("127.0.0.1:8005").is_ok());
        assert!(resolve_loopback("[::1]:8005").is_ok());
        assert!(resolve_loopback("0.0.0.0:8005").is_err());
        assert!(resolve_loopback("[::]:8005").is_err());
    }

    #[tokio::test]
    async fn admin_key_create_is_visible_to_the_hot_reloading_keystore() {
        use tower::ServiceExt as _;

        let dir = std::env::temp_dir().join(format!(
            "memra-admin-key-{}-{}",
            std::process::id(),
            unix_ms(),
        ));
        std::fs::create_dir(&dir).unwrap();
        let keys_path = dir.join("keys.toml");
        let budget_path = dir.join("budgets.toml");
        std::fs::write(&keys_path, "").unwrap();
        std::fs::write(
            &budget_path,
            "[[budgets]]\ntenant = \"acme\"\ncurrency = \"USD\"\nbalance_micro = 0\n",
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            for path in [&keys_path, &budget_path] {
                std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o640)).unwrap();
            }
        }
        let store = auth::KeyStore::from_spec(keys_path.to_str().unwrap())
            .unwrap()
            .with_poll(std::time::Duration::ZERO);
        let request_path = dir.join("requests.jsonl");
        let metadata = super::super::OpenRouterModelMetadata {
            pricing: super::super::OpenRouterPricing {
                prompt: Some("0.000001".into()),
                cached_prompt: Some("0.0000001".into()),
                completion: Some("0.000002".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        let ledger =
            ledger::Ledger::for_test_with_budgets(&request_path, &budget_path, "m", &metadata);
        let audit_path = ledger.admin_audit_path();
        let config = Config {
            addr: "127.0.0.1:8005".parse().unwrap(),
            state: AdminState {
                token: Arc::from("admin-secret"),
                keys_path: Arc::new(keys_path.clone()),
                budgets: ledger.budgets().unwrap(),
                audit: AuditLog::open(&audit_path).unwrap(),
                ledger,
                keys_lock: Arc::new(Mutex::new(())),
                capture: None,
                cmd_tx: None,
            },
        };
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/admin/keys")
            .header("authorization", "Bearer admin-secret")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"tenant":"acme","lane":"interactive","rate_limit":2}"#,
            ))
            .unwrap();
        let response = config
            .router(std::sync::mpsc::channel().0)
            .oneshot(request)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let key = body["key"].as_str().unwrap();
        assert_eq!(body["prefix"], display_prefix(key).unwrap());

        // Force an unmistakable mtime edge even on coarse filesystems, then prove the live
        // store (created before the admin call) authenticates the newly appended hash.
        let file = File::options().write(true).open(&keys_path).unwrap();
        file.set_modified(SystemTime::now() + std::time::Duration::from_secs(2))
            .unwrap();
        drop(file);
        let tenant = store.lookup(key).unwrap();
        assert_eq!(tenant.tenant, "acme");
        assert_eq!(tenant.rate_limit, Some(2));
        let audit = std::fs::read_to_string(audit_path).unwrap();
        assert_eq!(audit.lines().count(), 1);
        assert!(!audit.contains(key));
        assert!(!audit.contains("admin-secret"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
