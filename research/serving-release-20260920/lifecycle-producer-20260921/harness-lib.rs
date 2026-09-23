#![deny(warnings)]
use std::sync::Arc;
use std::time::Duration;
use axum::{body::Body, extract::{Request as AxumRequest, State}, Extension, http::{header::CONTENT_TYPE, StatusCode}, middleware::Next, response::Response, Router};
use futures_core::Stream as _;
use tower::ServiceExt as _;
mod ttft;
pub fn production_middleware_router() -> Router { Router::new().layer(axum::middleware::from_fn(ttft_request_start)) }
#[derive(Clone, Default)]
struct TtftRequestTrace(Option<Arc<ttft::Trace>>);

fn is_sse_data_frame(bytes: &[u8]) -> bool {
    bytes
        .windows(b"data:".len())
        .any(|window| window == b"data:")
}

enum HttpTraceStage {
    PendingResponse,
    Body,
    Disarmed,
}

struct HttpTraceLifetime {
    trace: Arc<ttft::Trace>,
    stage: HttpTraceStage,
}

impl Drop for HttpTraceLifetime {
    fn drop(&mut self) {
        match self.stage {
            HttpTraceStage::PendingResponse => self.trace.mark_http_pending_drop(),
            HttpTraceStage::Body => self.trace.mark_http_body_drop(),
            HttpTraceStage::Disarmed => {}
        }
    }
}

async fn ttft_request_start(req: AxumRequest, next: Next) -> Response {
    let trace = ttft::start(req.uri().path());
    ttft_request_start_with_trace(req, next, trace).await
}

async fn ttft_request_start_with_trace(
    mut req: AxumRequest,
    next: Next,
    trace: Option<Arc<ttft::Trace>>,
) -> Response {
    if let Some(trace) = trace.as_ref()
        && let Some(key) = req.headers().get("x-memra-trace-id")
    {
        // The diagnostic accepts only a bounded opaque correlation key. It never
        // supplies the server request ID or participates in request authority.
        trace.bind_client_trace_key(key.to_str().unwrap_or(""));
    }
    req.extensions_mut().insert(TtftRequestTrace(trace.clone()));
    // Queue admission can await the worker before there is any response body.
    // Keep a guard across that await so a pre-header disconnect is observable.
    let lifetime = trace.map(|trace| HttpTraceLifetime {
        trace,
        stage: HttpTraceStage::PendingResponse,
    });
    let response = next.run(req).await;
    let Some(mut lifetime) = lifetime else {
        return response;
    };
    let is_sse = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("text/event-stream"));
    if !is_sse {
        // Preserve non-SSE response transport unchanged. Returning an error body
        // normally is not a client-disconnect observation.
        lifetime.stage = HttpTraceStage::Disarmed;
        return response;
    }

    lifetime.stage = HttpTraceStage::Body;
    let trace = lifetime.trace.clone();
    // Stamp the first serialized application data frame as Hyper polls it. Axum's
    // keepalive comments can precede a long prefill, so non-data frames do not count.
    let (parts, body) = response.into_parts();
    let mut body = Box::pin(body.into_data_stream());
    let stream = async_stream::stream! {
        let _lifetime = lifetime;
        let mut failed = false;
        while let Some(frame) =
            std::future::poll_fn(|cx| body.as_mut().poll_next(cx)).await
        {
            failed |= frame.is_err();
            if frame
                .as_ref()
                .is_ok_and(|bytes| is_sse_data_frame(bytes))
            {
                trace.mark_first_sse_byte();
            }
            yield frame;
        }
        if !failed {
            trace.mark_http_body_eof();
        }
    };
    Response::from_parts(parts, Body::from_stream(stream))
}

#[cfg(test)]
mod request_lifecycle_http_tests {
    use super::*;
    use axum::body::{Bytes, HttpBody as _};

    const KEY: &str = "0123456789abcdef0123456789abcdef";

    #[derive(Clone, Copy)]
    enum FixtureBody {
        PendingResponse,
        PendingBody,
        PrefixThenPending,
        Complete,
        Error,
        Json,
    }

    #[derive(Clone)]
    struct FixtureState {
        body: FixtureBody,
        entered: Arc<tokio::sync::Notify>,
    }

    async fn fixture_handler(
        State(state): State<FixtureState>,
        Extension(trace): Extension<TtftRequestTrace>,
    ) -> Response {
        let trace = trace
            .0
            .expect("the actual middleware must attach its trace");
        trace.bind_request("server-generated-for-test", "fixture");
        trace.bind_worker(7, "shared_gpu_worker");
        trace.mark_queued();
        if matches!(state.body, FixtureBody::PendingResponse) {
            state.entered.notify_one();
            return std::future::pending().await;
        }
        if matches!(state.body, FixtureBody::Json) {
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from("{\"error\":\"fixture\"}"))
                .unwrap();
        }
        // These are HTTP lifetime controls with a fake backend, not model or
        // worker-phase qualification. The production middleware/body is exercised.
        trace.mark_first_decode();
        let body = match state.body {
            FixtureBody::PendingBody => Body::from_stream(async_stream::stream! {
                std::future::pending::<()>().await;
                yield Ok::<Bytes, std::io::Error>(Bytes::new());
            }),
            FixtureBody::PrefixThenPending => Body::from_stream(async_stream::stream! {
                yield Ok::<Bytes, std::io::Error>(Bytes::from_static(b"data: prefix\n\n"));
                std::future::pending::<()>().await;
            }),
            FixtureBody::Complete => Body::from("data: [DONE]\n\n"),
            FixtureBody::Error => Body::from_stream(async_stream::stream! {
                yield Ok::<Bytes, std::io::Error>(Bytes::from_static(b"data: prefix\n\n"));
                yield Err(std::io::Error::other("fixture body error"));
            }),
            FixtureBody::PendingResponse | FixtureBody::Json => unreachable!(),
        };
        Response::builder()
            .header(CONTENT_TYPE, "text/event-stream")
            .body(body)
            .unwrap()
    }

    fn fixture(
        body: FixtureBody,
    ) -> (
        Arc<ttft::Trace>,
        ttft::LifecycleObserver,
        Arc<tokio::sync::Notify>,
        Router,
    ) {
        let trace = ttft::Trace::for_test("/v1/completions");
        let observer = trace.observe_for_test();
        let entered = Arc::new(tokio::sync::Notify::new());
        let middleware_trace = trace.clone();
        let app = Router::new()
            .route("/v1/completions", axum::routing::post(fixture_handler))
            .with_state(FixtureState {
                body,
                entered: entered.clone(),
            })
            .layer(axum::middleware::from_fn(
                move |req: AxumRequest, next: Next| {
                    ttft_request_start_with_trace(req, next, Some(middleware_trace.clone()))
                },
            ));
        (trace, observer, entered, app)
    }

    fn request(key: &str) -> AxumRequest {
        AxumRequest::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("x-memra-trace-id", key)
            .body(Body::empty())
            .unwrap()
    }

    fn events(observer: &ttft::LifecycleObserver) -> Vec<serde_json::Value> {
        observer
            .json_lines()
            .iter()
            .map(|line| serde_json::from_str(line).expect("valid lifecycle JSON"))
            .collect()
    }

    fn event_count(rows: &[serde_json::Value], name: &str) -> usize {
        rows.iter().filter(|row| row["event"] == name).count()
    }

    #[tokio::test]
    async fn queued_preheader_drop_retains_client_and_server_identity() {
        let (_trace, observer, entered, app) = fixture(FixtureBody::PendingResponse);
        let task = tokio::spawn(app.oneshot(request(KEY)));
        tokio::time::timeout(Duration::from_secs(5), entered.notified())
            .await
            .expect("handler reached its actual pre-response await");
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        let rows = events(&observer);
        let pending = rows
            .iter()
            .find(|row| row["event"] == "http_pending_drop")
            .unwrap();
        assert_eq!(pending["client_trace_key"], KEY);
        assert_eq!(pending["request_id"], "server-generated-for-test");
        assert_eq!(pending["phase"], "queued");
        assert_eq!(event_count(&rows, "http_pending_drop"), 1);
        assert_eq!(event_count(&rows, "http_body_drop"), 0);
        assert_eq!(event_count(&rows, "http_body_eof"), 0);
        assert_eq!(event_count(&rows, "retired"), 0);
    }

    #[tokio::test]
    async fn never_polled_sse_body_still_records_drop() {
        let (_trace, observer, _, app) = fixture(FixtureBody::PendingBody);
        let response = app.oneshot(request(KEY)).await.unwrap();
        drop(response);
        let rows = events(&observer);
        assert_eq!(event_count(&rows, "http_body_drop"), 1);
        assert_eq!(event_count(&rows, "http_pending_drop"), 0);
        assert_eq!(event_count(&rows, "http_body_eof"), 0);
    }

    #[tokio::test]
    async fn dropping_a_stream_after_data_is_not_eof_or_retirement() {
        let (_trace, observer, _, app) = fixture(FixtureBody::PrefixThenPending);
        let response = app.oneshot(request(KEY)).await.unwrap();
        let mut stream = Box::pin(response.into_body().into_data_stream());
        let bytes = std::future::poll_fn(|cx| stream.as_mut().poll_next(cx))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(bytes.as_ref(), b"data: prefix\n\n");
        drop(stream);
        let rows = events(&observer);
        assert_eq!(event_count(&rows, "http_body_drop"), 1);
        assert_eq!(event_count(&rows, "http_body_eof"), 0);
        assert_eq!(event_count(&rows, "retired"), 0);
    }

    #[tokio::test]
    async fn complete_sse_body_records_eof_before_drop() {
        let (_trace, observer, _, app) = fixture(FixtureBody::Complete);
        let response = app.oneshot(request(KEY)).await.unwrap();
        let bytes = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        assert_eq!(bytes.as_ref(), b"data: [DONE]\n\n");
        let rows = events(&observer);
        let eof = rows
            .iter()
            .position(|row| row["event"] == "http_body_eof")
            .unwrap();
        let drop = rows
            .iter()
            .position(|row| row["event"] == "http_body_drop")
            .unwrap();
        assert!(eof < drop);
        assert_eq!(event_count(&rows, "http_body_eof"), 1);
        assert_eq!(event_count(&rows, "http_body_drop"), 1);
    }

    #[tokio::test]
    async fn body_error_followed_by_stream_end_never_records_normal_eof() {
        let (_trace, observer, _, app) = fixture(FixtureBody::Error);
        let response = app.oneshot(request(KEY)).await.unwrap();
        let mut stream = Box::pin(response.into_body().into_data_stream());
        assert!(
            std::future::poll_fn(|cx| stream.as_mut().poll_next(cx))
                .await
                .unwrap()
                .is_ok()
        );
        assert!(
            std::future::poll_fn(|cx| stream.as_mut().poll_next(cx))
                .await
                .unwrap()
                .is_err()
        );
        assert!(
            std::future::poll_fn(|cx| stream.as_mut().poll_next(cx))
                .await
                .is_none()
        );
        drop(stream);
        let rows = events(&observer);
        assert_eq!(event_count(&rows, "http_body_drop"), 1);
        assert_eq!(event_count(&rows, "http_body_eof"), 0);
    }

    #[tokio::test]
    async fn non_sse_error_transport_is_unchanged_and_not_a_disconnect() {
        for key in [KEY, "INVALID-PRIVATE-HEADER-VALUE"] {
            let (_trace, observer, _, app) = fixture(FixtureBody::Json);
            let response = app.oneshot(request(key)).await.unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert_eq!(response.headers()[CONTENT_TYPE], "application/json");
            assert_eq!(response.body().size_hint().exact(), Some(19));
            let bytes = axum::body::to_bytes(response.into_body(), 1024)
                .await
                .unwrap();
            assert_eq!(bytes.as_ref(), b"{\"error\":\"fixture\"}");
            let rows = events(&observer);
            assert_eq!(event_count(&rows, "http_pending_drop"), 0);
            assert_eq!(event_count(&rows, "http_body_drop"), 0);
            assert_eq!(event_count(&rows, "http_body_eof"), 0);
            assert!(
                observer
                    .json_lines()
                    .iter()
                    .all(|line| !line.contains("INVALID-PRIVATE-HEADER-VALUE"))
            );
        }
    }
}

