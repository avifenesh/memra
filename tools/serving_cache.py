"""Cold/seed/restored native request parity and cache accounting predicates.

The independently frozen program supplies three serial native_json requests:
``cold`` runs the extended prompt in an isolated cache salt, ``seed`` publishes
the shorter prefix in a second salt, and ``restore`` runs the extended prompt in
that second salt. Cold and restore differ only in cache_salt. Exact generated
token arrays, text and stop reason must agree; a cache-hit counter alone is not
correctness evidence. The restored cached count must cover the whole seed prompt.
Global prompt/cached/computed totals cover every reuse tier. Prefix-cache counters
are recorded separately: a continuation-pool hit bypasses the prefix probe, so its
cached tokens must not be required to appear in prefix_cache_hit_tokens. This cell
does not claim which cache tier supplied the restored state.

Inputs are trusted program/manifest bindings plus raw observations, as in
serving_completion. The release layer must bind their provenance, exclusive server,
source, binary, model, numeric environment and card lease. This predicate supplies
no such provenance and makes no whole-release or performance claim.
"""

from serving_completion import _keys, _same
from serving_lifecycle import account_generations
from serving_policy import evaluate_scenario
from serving_release import account_attempts, json_object, require


_GLOBAL_COUNTERS = ("prompt_tokens_in", "cached_tokens_in", "computed_tokens_in")
_PREFIX_DIAGNOSTICS = ("prefix_cache_hit_tokens", "prefix_cache_hits", "prefix_cache_misses")
_COUNTERS = _GLOBAL_COUNTERS + _PREFIX_DIAGNOSTICS


def evaluate_cache_cell(required, program, *, attempts, health_samples, metrics_samples):
    _keys(required, {"id", "scope", "scenario", "requirements"}, "required cache cell")
    require(required["scenario"] == "cache_restore" and _same(required["requirements"],
            {"cold_cached_tokens": 0, "restore_cache_hit": True, "same_program_exact_output": True}),
            "unknown cache requirements")
    scope = required["scope"]
    _keys(scope, {"id", "model", "route", "profile"}, "required scope")
    require(all(isinstance(v, str) and v for v in scope.values())
            and required["id"] == scope["id"] + "/cache_restore", "invalid required scope")
    _keys(program, {"cell_id", "scope", "server_identity", "mode", "requests"}, "cache program")
    require(program["cell_id"] == required["id"] and _same(program["scope"], scope)
            and program["mode"] == "serial", "cache program scope or scheduling differs")
    requests = program["requests"]
    require(isinstance(requests, list) and len(requests) == 3, "cache program requires three requests")
    planned = []
    for request, role in zip(requests, ("cold", "seed", "restore")):
        _keys(request, {"id", "role", "model", "wire", "path", "payload",
                        "prompt_tokens", "completion_tokens"}, "cache request")
        require(request["role"] == role and request["model"] == scope["model"]
                and request["wire"] == "native_json" and request["path"] == "/v1/completions",
                "cache request role/model/wire differs")
        payload = request["payload"]
        require(isinstance(payload, dict) and payload.get("model") == scope["model"]
                and payload.get("stream") is False and "prompt" not in payload
                and "messages" not in payload, "cache request needs unambiguous native IDs")
        ids = payload.get("prompt_ids")
        require(isinstance(ids, list) and ids and
                all(type(t) is int and 0 <= t <= 0xffffffff for t in ids), "invalid cache prompt IDs")
        require(isinstance(payload.get("cache_salt"), str) and payload["cache_salt"],
                "cache request needs an explicit salt")
        require(type(payload.get("temperature")) in (int, float) and payload["temperature"] == 0,
                "cache parity program must be greedy")
        planned.append({k: request[k] for k in
                        ("id", "model", "wire", "prompt_tokens", "completion_tokens")})
    cold, seed, restored = (r["payload"] for r in requests)
    require(cold["cache_salt"] != seed["cache_salt"] == restored["cache_salt"],
            "cold and restored cache namespaces are not isolated")
    require(_same({k: v for k, v in cold.items() if k != "cache_salt"},
                  {k: v for k, v in restored.items() if k != "cache_salt"}),
            "cold and restored programs differ beyond cache namespace")
    prefix = seed["prompt_ids"]
    require(len(restored["prompt_ids"]) > len(prefix)
            and restored["prompt_ids"][:len(prefix)] == prefix, "restore is not a strict seed extension")

    accounting = account_attempts(planned, attempts)
    generations = account_generations(accounting, attempts, health_samples)
    evaluated = evaluate_scenario({"scenario": "completed_group", "server_identity": program["server_identity"],
                                   "requests": planned}, accounting=accounting, generations=generations,
                                  attempts=attempts, health_samples=health_samples)
    observed = {r["id"]: r for r in attempts}
    ordered = [observed[r["id"]] for r in requests]
    require(all(a["finished_ns"] <= b["started_ns"] for a, b in zip(ordered, ordered[1:])),
            "cache captures were not serial in the required order")
    require(all(r.get("method") == "POST" and r.get("path") == "/v1/completions" for r in ordered),
            "cache observation route differs")
    responses = {r["id"]: r["response"] for r in accounting["requests"]}
    cold_result, seed_result, restored_result = [responses[r["id"]] for r in requests]
    for request, response in zip(requests, (cold_result, seed_result, restored_result)):
        require(response["usage"]["prompt_tokens"] == len(request["payload"]["prompt_ids"]),
                "cache usage differs from explicit prompt length")
    require(cold_result["usage"]["cached_tokens"] == seed_result["usage"]["cached_tokens"] == 0,
            "a required cold request reused a prefix")
    require(restored_result["usage"]["cached_tokens"] >= len(prefix),
            "restored request did not reuse the whole seed prefix")
    for field in ("tokens", "content", "stop_reason"):
        require(_same(cold_result[field], restored_result[field]), "restored output differs from cold control")
    require(cold_result["usage"]["completion_tokens"] == restored_result["usage"]["completion_tokens"],
            "restored output length differs")

    require(isinstance(metrics_samples, list) and len(metrics_samples) == 2,
            "cache accounting needs before and after metrics")
    metrics = []
    for sample in metrics_samples:
        require(isinstance(sample, dict) and sample.get("server_identity") == program["server_identity"]
                and sample.get("method") == "GET" and sample.get("path") == "/metrics"
                and type(sample.get("status")) is int and sample["status"] == 200
                and sample.get("transport_error") is None, "incomplete or foreign metrics sample")
        start, end = sample.get("started_ns"), sample.get("finished_ns")
        require(type(start) is int and type(end) is int and 0 <= start <= end,
                "invalid metrics interval")
        value = json_object(sample.get("body"))
        require(all(type(value.get(k)) is int and value[k] >= 0 for k in _COUNTERS),
                "missing or invalid cache counters")
        metrics.append(value)
    require(metrics_samples[0]["finished_ns"] <= ordered[0]["started_ns"]
            and metrics_samples[1]["started_ns"] >= ordered[-1]["finished_ns"],
            "metrics do not bracket all cache requests")
    prompt = sum(r["usage"]["prompt_tokens"] for r in responses.values())
    cached = sum(r["usage"]["cached_tokens"] for r in responses.values())
    expected = {"prompt_tokens_in": prompt, "cached_tokens_in": cached,
                "computed_tokens_in": prompt - cached}
    delta = {k: metrics[1][k] - metrics[0][k] for k in _COUNTERS}
    require(all(v >= 0 for v in delta.values()), "cache counters reset during capture")
    require({k: delta[k] for k in _GLOBAL_COUNTERS} == expected,
            "global cache metrics differ from the full request denominator")
    return {"id": required["id"], "scope": dict(scope), "scenario": "cache_restore",
            "completed": evaluated["completed"], "restored_prefix_tokens": restored_result["usage"]["cached_tokens"],
            "seed_prompt_tokens": len(prefix), "metrics_delta": expected,
            "prefix_diagnostics_delta": {k: delta[k] for k in _PREFIX_DIAGNOSTICS},
            "cache_tier_attribution": None}
