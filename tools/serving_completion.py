"""Required completion scenarios evaluated from raw wire and health observations.

``required`` is one cell returned by serving_manifest.required_cells using trusted
manifest bytes and release scope. ``program`` is the independently frozen request
program, not a config or oracle selected by the captured run. The release binding
layer must prove that this exact program was sent by the owned collector. This
pure predicate neither establishes that provenance nor qualifies a whole release.

Supported scenarios: short_prompt, long_prompt, wire_completion and
concurrent_completion. Other required scenarios refuse. Observed overlap means
overlapping client request intervals, not simultaneous GPU execution or throughput.
Short-prompt answer oracles compare the complete content string exactly (including
whitespace); reasoning is parsed but is not substituted for the answer.
"""

from serving_lifecycle import account_generations
from serving_policy import evaluate_scenario
from serving_release import WIRE_VALIDATORS, account_attempts, require


_REQUIREMENTS = {
    "short_prompt": {"prompt_tokens_min": 1, "prompt_tokens_max": 32,
                     "temperature": 0, "exact_answer_oracle": True},
    "long_prompt": {"prompt_tokens_min": 512, "nonempty_output": True},
    "wire_completion": {"wires": ["chat_json", "chat_sse", "native_json", "native_sse"],
                        "terminal_usage": True},
    "concurrent_completion": {"minimum_requests": 3, "observed_overlap": True,
                              "all_attempts_accounted": True},
}


def _keys(value, names, label):
    require(isinstance(value, dict) and set(value) == set(names),
            "unknown or missing " + label + " fields")


def _same(left, right):
    """Policy values retain their JSON types; bool is not an integer threshold."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def evaluate_completion_cell(required, program, *, attempts, health_samples):
    """Return derived observations; refuse missing, failed or substituted work.

    A program has cell_id, scope, server_identity, mode and requests. Each request
    has id/model/wire/path/payload and inclusive prompt_tokens/completion_tokens
    ranges. Short-prompt requests additionally need answer_oracle, a nonempty exact
    answer string supplied by the reviewed program. All responses must pass their
    full wire parser and unchanged-generation health brackets before predicates run.
    """
    _keys(required, {"id", "scope", "scenario", "requirements"}, "required cell")
    scenario = required["scenario"]
    require(isinstance(scenario, str) and scenario in _REQUIREMENTS,
            "unsupported completion scenario")
    require(_same(required["requirements"], _REQUIREMENTS[scenario]),
            "unknown completion requirements")
    scope = required["scope"]
    _keys(scope, {"id", "model", "route", "profile"}, "required scope")
    require(all(isinstance(v, str) and v for v in scope.values())
            and required["id"] == scope["id"] + "/" + scenario, "invalid required scope")
    _keys(program, {"cell_id", "scope", "server_identity", "mode", "requests"}, "program")
    require(program["cell_id"] == required["id"] and _same(program["scope"], scope),
            "program differs from required scope")
    require(program["mode"] in ("serial", "concurrent"), "unknown request scheduling mode")
    requests = program["requests"]
    require(isinstance(requests, list) and 1 <= len(requests) <= 256,
            "completion program must have 1..256 requests")
    schedule = []
    for request in requests:
        names = {"id", "model", "wire", "path", "payload", "prompt_tokens", "completion_tokens"}
        if scenario == "short_prompt":
            names.add("answer_oracle")
        _keys(request, names, "request program")
        wire = request["wire"]
        require(isinstance(wire, str) and wire in WIRE_VALIDATORS, "unknown response wire")
        path = "/v1/chat/completions" if wire.startswith("chat_") else "/v1/completions"
        payload = request["payload"]
        require(request["model"] == scope["model"] and request["path"] == path
                and isinstance(payload, dict) and payload.get("model") == scope["model"]
                and type(payload.get("stream")) is bool
                and payload["stream"] == wire.endswith("_sse"), "request route/model/stream differs")
        if scenario == "short_prompt":
            require(type(payload.get("temperature")) in (int, float)
                    and payload["temperature"] == 0, "short prompt must use temperature zero")
            require(isinstance(request["answer_oracle"], str) and request["answer_oracle"],
                    "missing independent exact-answer oracle")
        schedule.append({k: request[k] for k in
                         ("id", "model", "wire", "prompt_tokens", "completion_tokens")})

    # Recompute everything from raw observations; receipt-supplied verdicts or
    # survivor-only summaries are not an input to this API.
    accounting = account_attempts(schedule, attempts)
    generations = account_generations(accounting, attempts, health_samples)
    evaluated = evaluate_scenario(
        {"scenario": "completed_group", "server_identity": program["server_identity"],
         "requests": schedule}, accounting=accounting, generations=generations,
        attempts=attempts, health_samples=health_samples)
    by_request = {request["id"]: request for request in requests}
    for attempt in attempts:
        require(attempt.get("method") == "POST"
                and attempt.get("path") == by_request[attempt["id"]]["path"],
                "observed HTTP route differs from program")
    completed = evaluated["completed"]
    peak = None
    if scenario == "short_prompt":
        require(all(1 <= row["prompt_tokens"] <= 32 for row in completed),
                "short prompt usage is outside required range")
        for row in accounting["requests"]:
            require(row["response"]["content"] == by_request[row["id"]]["answer_oracle"],
                    "short prompt differs from exact-answer oracle")
    elif scenario == "long_prompt":
        require(all(row["prompt_tokens"] >= 512 for row in completed),
                "long prompt did not meet required input length")
    elif scenario == "wire_completion":
        require({r["wire"] for r in requests} == set(WIRE_VALIDATORS),
                "required wire contract is missing")
    else:
        require(program["mode"] == "concurrent" and len(completed) >= 3,
                "concurrency requires at least three planned concurrent completions")
        events = []
        for row in completed:
            require(row["started_ns"] < row["finished_ns"], "empty concurrency interval")
            events.extend([(row["started_ns"], 1), (row["finished_ns"], -1)])
        count, peak = 0, 0
        # End before start at equal timestamps: touching intervals do not overlap.
        for _, delta in sorted(events):
            count += delta
            peak = max(peak, count)
        require(peak >= 3, "fewer than three client request intervals overlap")
    return {"id": required["id"], "scope": dict(scope), "scenario": scenario,
            "server_identity": evaluated["server_identity"], "planned": evaluated["planned"],
            "completed": completed, "peak_client_intervals": peak}
