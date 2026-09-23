"""Pure queue-overload recovery predicate over a trusted program and raw captures.

``required`` is the overload_recovery cell from serving_manifest.required_cells,
using independently trusted manifest bytes and release scope. ``program`` must be
frozen before capture. The caller binds its exact payloads, roles, denominator,
model/route, owned listener and process identity to the actual run; changing both
the program and captures is outside this predicate's provenance boundary. Source,
binary, lease and artifact binding also belong to the caller. No I/O is performed.

Program shape (inclusive usage ranges; timestamps are monotonic nanoseconds)::

    {"cell_id": "tiny/overload_recovery", "scope": <required scope>,
     "server_identity": {"pid": 100, "start_identity": "<boot UUID>:1234"},
     "mode": "overload_then_recovery",
     "requests": [
       {"id": "peer", "role": "peer", "model": "gate", "wire": "chat_json",
        "path": "/v1/chat/completions", "payload": <frozen request JSON>,
        "prompt_tokens": {"min": 1, "max": 32},
        "completion_tokens": {"min": 1, "max": 64}},
       {"id": "shed", "role": "refused", "model": "gate", "wire": "chat_json",
        "path": "/v1/chat/completions", "payload": <frozen request JSON>,
        "error_code": "shed_queue"},
       {"id": "retry", "role": "recovery", <same fields as peer>}]}

At least one request per role is required; every planned request is accounted.
Raw attempts have id, server_identity, method, path, started_ns, finished_ns,
status, body bytes, response headers as [name,value] pairs, and transport_error
(None or absent on success). Health captures additionally use unique IDs, GET
/health and raw worker.generation. Uninterrupted before/after health brackets are
required for EVERY attempt, including refusals, and all observed generations must
agree even across the gap before recovery. Peers and recovery use completed_group's
full wire, nonempty output and usage checks. Summary passed/skip fields never count.

Each refusal must fit strictly inside a clean peer's client capture interval;
all recovery requests start after all peer/refusal captures finish. These are
client observations, not server scheduling, simultaneous GPU work or a last-write
timestamp. Buffered response reads can overlap without concurrent server work.

The supported source contract is deliberately queue-specific: lib.rs
reserve_pending_admit_with_ceiling emits 429 rate_limit_error with shed_queue,
shed_deadline or shed_queue_wait; retry_contract_response emits decimal seconds
clamped to 1..60 and matching retry-after-ms. The frozen request names ONE code,
not an observed-code allowlist. Quota rate_limit_exceeded and 503 overloaded
(also used for worker unavailability) do not establish this queue scenario.
This checks the retry header contract, not the unobserved queue estimate, queue
depth, billing ledger or obedience to the suggested retry delay. Success returns
derived evidence, not a whole-release qualification or a timing benchmark.
"""

from serving_completion import _keys, _same
from serving_lifecycle import account_generations
from serving_policy import evaluate_scenario
from serving_release import WIRE_VALIDATORS, account_attempts, json_object, require


_REQUIREMENTS = {"typed_refusal": True, "retry_headers": True,
                 "peer_completion": True, "recovery_completion": True}
_QUEUE_CODES = ("shed_queue", "shed_deadline", "shed_queue_wait")


def _queue_retry(headers):
    require(isinstance(headers, list), "queue refusal lacks raw response headers")
    selected = {}
    for pair in headers:
        require(isinstance(pair, (list, tuple)) and len(pair) == 2
                and all(isinstance(v, str) for v in pair), "invalid raw response header")
        name, value = pair[0].lower(), pair[1]
        if name in ("retry-after", "retry-after-ms", "x-should-retry"):
            require(name not in selected, "duplicate retry contract header")
            selected[name] = value
    require(set(selected) == {"retry-after", "retry-after-ms"},
            "missing or contradictory queue retry contract")
    seconds = selected["retry-after"]
    require(1 <= len(seconds) <= 2 and seconds.isascii() and seconds.isdecimal(),
            "queue Retry-After must be decimal seconds")
    value = int(seconds)
    require(1 <= value <= 60 and seconds == str(value)
            and selected["retry-after-ms"] == str(value * 1000),
            "queue retry headers disagree or exceed the source clamp")
    return value


def evaluate_overload_recovery_cell(required, program, *, attempts, health_samples):
    """Return derived evidence or raise ServingGateError; never mutate inputs."""
    _keys(required, {"id", "scope", "scenario", "requirements"}, "required recovery cell")
    require(required["scenario"] == "overload_recovery"
            and _same(required["requirements"], _REQUIREMENTS), "unknown recovery requirements")
    scope = required["scope"]
    _keys(scope, {"id", "model", "route", "profile"}, "required scope")
    require(all(isinstance(v, str) and v.strip() for v in scope.values())
            and required["id"] == scope["id"] + "/overload_recovery", "invalid required scope")
    _keys(program, {"cell_id", "scope", "server_identity", "mode", "requests"}, "recovery program")
    require(program["cell_id"] == required["id"] and _same(program["scope"], scope)
            and program["mode"] == "overload_then_recovery", "recovery program scope or mode differs")
    requests = program["requests"]
    require(isinstance(requests, list) and 3 <= len(requests) <= 256,
            "recovery program needs 3..256 requests")
    by_role = {"peer": [], "refused": [], "recovery": []}
    planned = {}
    schedule = []
    completion_schedule = []
    for request in requests:
        require(isinstance(request, dict) and isinstance(request.get("role"), str)
                and request["role"] in by_role, "unknown recovery request role")
        role = request["role"]
        names = {"id", "role", "model", "wire", "path", "payload"}
        names |= {"error_code"} if role == "refused" else {"prompt_tokens", "completion_tokens"}
        _keys(request, names, "recovery request")
        request_id = request["id"]
        require(isinstance(request_id, str) and request_id.strip() and request_id not in planned,
                "empty or duplicate planned request ID")
        wire = request["wire"]
        require(isinstance(wire, str) and wire in WIRE_VALIDATORS, "unknown response wire")
        path = "/v1/chat/completions" if wire.startswith("chat_") else "/v1/completions"
        payload = request["payload"]
        require(request["model"] == scope["model"] and request["path"] == path
                and isinstance(payload, dict) and payload.get("model") == scope["model"]
                and type(payload.get("stream")) is bool
                and payload["stream"] == wire.endswith("_sse"), "request route/model/stream differs")
        row = {k: request[k] for k in ("id", "model", "wire")}
        if role == "refused":
            require(request["error_code"] in _QUEUE_CODES, "program must name a source queue refusal")
        else:
            completion_schedule.append({**row, **{k: request[k] for k in
                                                  ("prompt_tokens", "completion_tokens")}})
        schedule.append(row)
        planned[request_id] = request
        by_role[role].append(request_id)
    require(all(by_role.values()), "peer, refused and recovery denominators must all be nonempty")

    # Account the ENTIRE program first. Refusals are not dropped to obtain a
    # survivor-only completion group; each has its own checks below.
    accounting = account_attempts(schedule, attempts)
    generations = account_generations(accounting, attempts, health_samples)
    require(_same(generations["server_identity"], program["server_identity"])
            and generations["respawns_observed"] == 0
            and generations["generation_affected_requests"] == 0,
            "server or worker generation changed during overload/recovery")
    observed = {row["id"]: row for row in attempts}
    wire_results = {row["id"]: row for row in accounting["requests"]}
    for attempt in attempts:
        require(_same(attempt.get("server_identity"), program["server_identity"])
                and attempt.get("method") == "POST"
                and attempt.get("path") == planned[attempt["id"]]["path"],
                "request observation owner or HTTP route differs")
        require(attempt["started_ns"] < attempt["finished_ns"], "empty request capture interval")
    for sample in health_samples:
        require(isinstance(sample.get("id"), str) and sample["id"] not in planned,
                "invalid health ID or health and request IDs overlap")
        require(sample.get("method") == "GET" and sample.get("status") == 200
                and json_object(sample["body"]).get("status") == "ok",
                "overload/recovery requires healthy worker brackets")

    # Reuse the existing full completion policy for both successful roles. It
    # also checks health IDs, owned identity, complete /health captures and ranges.
    clean_attempts = [row for row in attempts if planned[row["id"]]["role"] != "refused"]
    clean_accounting = account_attempts(completion_schedule, clean_attempts)
    clean_generations = account_generations(clean_accounting, clean_attempts, health_samples)
    completed = evaluate_scenario(
        {"scenario": "completed_group", "server_identity": program["server_identity"],
         "requests": completion_schedule}, accounting=clean_accounting,
        generations=clean_generations, attempts=clean_attempts, health_samples=health_samples)["completed"]

    refused = []
    for request_id in by_role["refused"]:
        attempt, result = observed[request_id], wire_results[request_id]
        require(result["outcome"] == "refused" and result["status"] == 429
                and attempt.get("transport_error") is None, "planned queue refusal was not complete HTTP 429")
        payload = json_object(attempt["body"])
        _keys(payload, {"error"}, "queue refusal body")
        error = payload["error"]
        _keys(error, {"message", "type", "param", "code"}, "queue refusal error")
        require(error["type"] == "rate_limit_error" and error["param"] is None
                and error["code"] == planned[request_id]["error_code"]
                and isinstance(error["message"], str) and error["message"].strip(),
                "refusal does not match the frozen queue error")
        retry = _queue_retry(attempt.get("headers"))
        peers = [key for key in by_role["peer"]
                 if observed[key]["started_ns"] < attempt["started_ns"]
                 and attempt["finished_ns"] < observed[key]["finished_ns"]]
        require(peers, "queue refusal is not spanned by a clean completed peer")
        refused.append({"id": request_id, "status": 429, "error_code": error["code"],
                        "retry_after_s": retry, "retry_after_ms": retry * 1000,
                        "started_ns": attempt["started_ns"], "finished_ns": attempt["finished_ns"],
                        "spanning_peer_ids": peers})
    pressure_finished = max(observed[key]["finished_ns"] for key in
                            by_role["peer"] + by_role["refused"])
    require(all(observed[key]["started_ns"] > pressure_finished for key in by_role["recovery"]),
            "recovery did not start after all pressure attempts completed")
    return {"id": required["id"], "scope": dict(scope), "scenario": "overload_recovery",
            "server_identity": dict(generations["server_identity"]),
            "planned": accounting["attempted"], "planned_by_role": by_role,
            "wire_counts": dict(accounting["counts"]), "refused": refused,
            "completed": [{**row, "role": planned[row["id"]]["role"]} for row in completed],
            "generation_observations": generations, "pressure_finished_ns": pressure_finished,
            "timing_scope": "client capture intervals; no server execution or throughput claim"}
