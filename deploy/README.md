# Deployment

This directory holds one thing: `systemd/`, the example units that document the engine's
supervision contract — `Type=notify` with `READY=1` after the models load and the socket binds,
`WATCHDOG=1` only while inference is live, `STOPPING=1` and `EXTEND_TIMEOUT_USEC` so a drain is
not SIGKILLed, and exit 70 for an unrecoverable GPU distinguished from exit 1 for bad config.
Copy them; do not symlink. Every path is site-specific, and the value is in the directive
choices, each commented with the failure it prevents.

The target hardware shape is a 2x RTX PRO 6000 Blackwell pair. Serving uses PP-2 when a model
spans both cards and independent GPU placement when it fits on one.

**Gateway, TLS, key management, trial and channel-worker runbooks are not here.** Which provider
hosts which role, what it costs and how the public edge is wired are deployment facts, and they
live in darklanes. This repo documents the engine and the shape it runs on, never which machine
is serving.

Historical note: receipts under `research/` cite `deploy/gateway/…`, `deploy/glue/…` and
`deploy/runpod/…`. Those paths were correct on the dates those receipts were written and the
records are left as written — the material they name now lives in darklanes under
`ops/memra-deploy/`.
