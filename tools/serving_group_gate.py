"""Bind one reviewed completion/cache program to its complete captured group.

The required cell, program, capture digest, launch plan and artifact identities
come from the external release binding layer. None is selected by the capture.
The reader checks all sealed bytes, launch/identity/cleanup and raw accounting;
this bridge additionally binds exact request payloads/order and applies the
existing scenario predicate. It cannot choose a successful subset of a capture.

This is not a whole-release verifier. Required-suite coverage, trusted source,
build/numeric/hardware provenance and GPU lease validation remain outside it.
Cancellation, drain, overload and worker-failure orchestration need their own
capture contracts and are explicitly unsupported here.
"""

from serving_cache import evaluate_cache_cell
from serving_completion import _keys, _same, evaluate_completion_cell
from serving_evidence import read_group_capture
from serving_release import require


_COMPLETION = frozenset({"short_prompt", "long_prompt", "wire_completion", "concurrent_completion"})
_REQUEST_FIELDS = ("id", "model", "wire", "path", "payload")


def evaluate_group_capture(required, program, capture_bytes, evidence_reader, *,
                           expected_capture_sha256, expected_plan, expected_identities):
    """Return a bound cell result, never a release-qualification verdict."""
    _keys(required, {"id", "scope", "scenario", "requirements"}, "required cell")
    scenario = required["scenario"]
    require(isinstance(scenario, str) and scenario in _COMPLETION | {"cache_restore"},
            "scenario has no supported group-capture contract")
    _keys(program, {"cell_id", "scope", "server_identity", "mode", "requests"}, "program")
    require(program["cell_id"] == required["id"] and _same(program["scope"], required["scope"]),
            "program differs from required cell")
    requests = program["requests"]
    require(isinstance(requests, list) and requests and all(
        isinstance(r, dict) and all(k in r for k in _REQUEST_FIELDS) for r in requests),
        "reviewed request program is missing its sent fields")
    require(isinstance(expected_plan, dict) and isinstance(expected_plan.get("groups"), list)
            and len(expected_plan["groups"]) == 1,
            "a cell must account for its entire single-group capture")

    decoded = read_group_capture(capture_bytes, evidence_reader,
        expected_capture_sha256=expected_capture_sha256, expected_plan=expected_plan,
        expected_identities=expected_identities)
    group = decoded["groups"][0]
    require(group["id"] == required["id"] and group["plan"]["mode"] == program["mode"],
            "captured cell or scheduling mode differs from reviewed program")
    require(_same(program["server_identity"], decoded["server_identity"]),
            "program owner differs from captured process")
    schedule = [{k: r[k] for k in _REQUEST_FIELDS} for r in requests]
    require(_same(group["plan"]["requests"], schedule),
            "sent request payloads or order differ from reviewed program")

    args = {"attempts": group["attempts"], "health_samples": group["health_samples"]}
    if scenario == "cache_restore":
        require(group["plan"].get("metrics") is True and "metrics_samples" in group,
                "cache cell requires captured metrics brackets")
        result = evaluate_cache_cell(required, program, metrics_samples=group["metrics_samples"], **args)
    else:
        result = evaluate_completion_cell(required, program, **args)
    return {"cell": result, "capture_sha256": decoded["capture_sha256"],
            "verified_payloads": decoded["verified_payloads"], "qualification": False}
