"""Replay the entire pressure/recovery capture against its reviewed program.

The caller supplies the externally bound program, capture digest, launch plan and
artifact identities. The first group concurrently issues every peer/refusal;
the second serially issues every recovery request, in program order. The reader
checks all bytes, lifecycle, group chronology and accounting before the existing
overload predicate checks actual queue refusals and completed peers/recovery.

This bridge does not authenticate a release source/build/lease or qualify a
release. Those bindings and required-suite coverage belong to the outer verifier.
"""

from serving_completion import _keys, _same
from serving_evidence import read_group_capture
from serving_recovery import evaluate_overload_recovery_cell
from serving_release import require


_SENT = ("id", "model", "wire", "path", "payload")


def evaluate_overload_capture(required, program, capture_bytes, evidence_reader, *,
                              expected_capture_sha256, expected_plan, expected_identities):
    """Return a bound cell result, retaining the complete request denominator."""
    _keys(required, {"id", "scope", "scenario", "requirements"}, "required overload cell")
    require(required["scenario"] == "overload_recovery" and type(required["id"]) is str
            and bool(required["id"]), "unsupported overload capture scenario or cell ID")
    _keys(program, {"cell_id", "scope", "server_identity", "mode", "requests"}, "overload program")
    require(program["cell_id"] == required["id"] and _same(program["scope"], required["scope"])
            and program["mode"] == "overload_then_recovery", "overload program differs from required cell")
    requests = program["requests"]
    require(type(requests) is list and 3 <= len(requests) <= 256 and all(
        type(r) is dict and r.get("role") in ("peer", "refused", "recovery")
        and all(k in r for k in _SENT) for r in requests), "invalid overload request program")
    require(all(any(r["role"] == role for r in requests) for role in ("peer", "refused", "recovery")),
            "pressure and recovery roles must all be present")
    groups = [
        {"id": required["id"] + "/pressure", "mode": "concurrent",
         "requests": [{k: r[k] for k in _SENT} for r in requests if r["role"] != "recovery"]},
        {"id": required["id"] + "/recovery", "mode": "serial",
         "requests": [{k: r[k] for k in _SENT} for r in requests if r["role"] == "recovery"]},
    ]
    require(type(expected_plan) is dict and _same(expected_plan.get("groups"), groups),
            "capture must contain exactly the complete pressure and recovery schedules")
    decoded = read_group_capture(capture_bytes, evidence_reader,
        expected_capture_sha256=expected_capture_sha256, expected_plan=expected_plan,
        expected_identities=expected_identities)
    require(_same(decoded["server_identity"], program["server_identity"]),
            "overload program owner differs from captured process")
    attempts = [row for group in decoded["groups"] for row in group["attempts"]]
    health = [row for group in decoded["groups"] for row in group["health_samples"]]
    result = evaluate_overload_recovery_cell(required, program, attempts=attempts, health_samples=health)
    return {"cell": result, "capture_sha256": decoded["capture_sha256"],
            "verified_payloads": decoded["verified_payloads"], "qualification": False}
