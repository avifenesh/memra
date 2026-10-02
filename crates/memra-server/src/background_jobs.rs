//! Buffered chat/text delivery over the existing JobStore and receipt seams.

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::sync::{Notify, watch};

use crate::metering::{JobRecord, JobStatus};
use crate::{AppState, Envelope, InflightGuard, RateLimit};

/// Cancellation and terminal publication are separate signals. The watch value
/// survives a publication before the HTTP cancel handler begins waiting.
#[derive(Clone)]
pub(crate) struct Control {
    pub(crate) cancel: Arc<Notify>,
    terminal: watch::Receiver<bool>,
}

impl Control {
    pub(crate) fn new() -> (Self, watch::Sender<bool>) {
        let (tx, terminal) = watch::channel(false);
        (
            Self {
                cancel: Arc::new(Notify::new()),
                terminal,
            },
            tx,
        )
    }

    pub(crate) fn notify_one(&self) {
        self.cancel.notify_one();
    }

    #[cfg(test)]
    pub(crate) async fn notified(&self) {
        self.cancel.notified().await;
    }

    pub(crate) async fn wait_terminal(&self) {
        let mut terminal = self.terminal.clone();
        let _ = terminal.wait_for(|done| *done).await;
    }
}

pub(crate) fn default_background() -> Value {
    Value::Bool(false)
}

pub(crate) fn validate(value: &Value, stream: bool) -> Result<(), &'static str> {
    let background = match value {
        Value::Bool(value) => *value,
        // Serde's default for an omitted Value is null. Use a dedicated false
        // default so an explicit null is distinguishable and refused.
        _ => return Err("background must be a boolean"),
    };
    if background && !crate::responses_api::background_door_open() {
        return Err("background delivery is disabled; MEMRA_BACKGROUND_RESPONSES is off");
    }
    if background && stream {
        return Err("background and stream cannot both be true");
    }
    Ok(())
}

/// Complete results retain the submitting dialect's original response body.
/// Non-terminal states carry only identity/status because no result exists yet.
fn record_response(id: &str, record: JobRecord) -> Response {
    if let Some(body) = record.output {
        return crate::with_request_id(id, axum::Json(body).into_response());
    }
    let mut body = json!({
        "id": id,
        "status": crate::responses_api::job_status_str(record.status),
    });
    if record.status == JobStatus::Failed {
        body["error"] = json!({
            "message": record.error.unwrap_or_else(|| "background job failed".into()),
            "type": "server_error",
            "code": "background_job_failed",
        });
    }
    crate::with_request_id(id, axum::Json(body).into_response())
}

pub(crate) async fn poll_admitted(
    State(st): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    crate::responses_api::poll_record(st, headers, id, record_response).await
}

pub(crate) async fn cancel_admitted(
    State(st): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    crate::responses_api::cancel_record(st, headers, id, record_response).await
}

#[allow(clippy::too_many_arguments)] // the admitted request's existing owned resources
pub(crate) async fn submit(
    st: AppState,
    tenant: String,
    env: Envelope,
    model: String,
    chat: bool,
    rx: crate::worker::EventReceiver,
    receipt: Option<Box<dyn crate::metering::Receipt>>,
    guard: InflightGuard,
    rl: RateLimit,
    parser: Option<crate::toolcall::ToolStreamParser>,
    stop_strings: Vec<String>,
) -> Response {
    let key = crate::responses_api::background_store_key(&tenant, &env.id);
    if let Err(err) = st.job_store.put(&key, JobRecord::queued()) {
        drop(rx);
        drop(guard);
        let code = match err {
            crate::metering::JobStoreError::CapacityExceeded => {
                "background_job_store_capacity_exceeded"
            }
            _ => "background_job_store_unavailable",
        };
        return rl.attach(crate::ledger_rejected(
            receipt,
            crate::error_response_coded(
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "the background job store cannot admit this request right now",
                "server_error",
                None,
                Some(code),
            ),
            code,
            &env.id,
        ));
    }
    let immediate = rl.attach(record_response(&env.id, JobRecord::queued()));
    let (control, terminal) = Control::new();
    st.background_cancel
        .lock()
        .unwrap()
        .insert(key.clone(), control.clone());
    tokio::spawn(async move {
        let _ = st.job_store.put(
            &key,
            JobRecord {
                status: JobStatus::InProgress,
                output: None,
                error: None,
            },
        );
        let (publication, mut receipt) =
            crate::job_publication::Publication::new(st.job_store.clone(), key.clone(), receipt);
        if let Err(error) = publication.initial_reservation() {
            publication.fail_storage(error);
            crate::responses_api::finalize_terminal_job(
                &crate::job_publication::PublishingStore(publication),
                &key,
                JobRecord {
                    status: JobStatus::Failed,
                    output: None,
                    error: Some(
                        "background output could not be buffered by the configured store".into(),
                    ),
                },
            );
            let _ = terminal.send(true);
            st.background_cancel.lock().unwrap().remove(&key);
            drop(guard);
            return;
        }
        let response = crate::collect_blocking_response(
            rx,
            model,
            chat,
            stop_strings,
            parser,
            env.clone(),
            &mut receipt,
            None,
            Some(control.cancel),
            Some(&publication),
        )
        .await;
        let http_status = response.status();
        let parsed =
            match axum::body::to_bytes(response.into_body(), publication.body_limit()).await {
                Ok(bytes) => serde_json::from_slice::<Value>(&bytes).ok(),
                Err(_) => None,
            };
        let record = match parsed {
            Some(mut body) if body.is_object() => {
                let code = body
                    .get("error")
                    .and_then(|e| e.get("code"))
                    .and_then(Value::as_str);
                let failed = !http_status.is_success()
                    || body.get("error").is_some_and(|error| !error.is_null());
                let finish = body
                    .get("choices")
                    .and_then(|v| v.get(0))
                    .and_then(|v| v.get("finish_reason"))
                    .and_then(Value::as_str)
                    .or_else(|| {
                        body.get("stop_reason")
                            .and_then(Value::as_str)
                            .map(crate::stop_reason_to_finish)
                    });
                let status = if code == Some("cancelled") {
                    JobStatus::Cancelled
                } else if failed {
                    JobStatus::Failed
                } else if finish == Some("length") {
                    JobStatus::Incomplete
                } else {
                    JobStatus::Completed
                };
                body["id"] = json!(env.id);
                body["status"] = json!(crate::responses_api::job_status_str(status));
                JobRecord {
                    status,
                    output: Some(body),
                    error: None,
                }
            }
            _ => JobRecord {
                status: JobStatus::Failed,
                output: None,
                error: Some("background completion could not be encoded".into()),
            },
        };
        let guarded = crate::job_publication::PublishingStore(publication);
        crate::responses_api::finalize_terminal_job(&guarded, &key, record);
        let _ = terminal.send(true);
        st.background_cancel.lock().unwrap().remove(&key);
        drop(guard);
    });
    immediate
}
