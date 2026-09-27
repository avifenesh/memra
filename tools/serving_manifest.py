"""Required serving coverage from reviewed policy, independent of receipt claims.

The caller supplies exact tracked manifest bytes and the externally required
model/route scope (from the release roster and reviewed execution program). A run
cannot choose a smaller scope or replace this manifest with its own policy. This
function only validates coverage, never the scenario outcomes or qualification.
"""

import re

from serving_release import json_object, require


_NAME = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_SCENARIOS = frozenset({"short_prompt", "long_prompt", "wire_completion", "cache_restore",
                        "concurrent_completion", "overload_recovery", "cancel_queued",
                        "cancel_prime", "cancel_decode", "drain", "worker_failure_recovery"})


def required_cells(manifest_bytes, expected_scopes):
    """Expand fixed scenario IDs for every required immutable model/route scope."""
    manifest = json_object(manifest_bytes)
    require(set(manifest) == {"schema", "profiles"}
            and manifest["schema"] == "memra-required-serving-cells-v1", "unknown serving manifest")
    profiles = manifest["profiles"]
    require(isinstance(profiles, dict) and profiles, "serving profiles are missing")
    for name, cells in profiles.items():
        require(isinstance(name, str) and _NAME.fullmatch(name), "invalid serving profile name")
        require(isinstance(cells, list) and cells, "empty required scenario list")
        ids = []
        for cell in cells:
            require(isinstance(cell, dict) and set(cell) == {"id", "requirements"}
                    and isinstance(cell["id"], str) and cell["id"] in _SCENARIOS
                    and isinstance(cell["requirements"], dict)
                    and cell["requirements"], "invalid required scenario declaration")
            ids.append(cell["id"])
        require(len(ids) == len(set(ids)) and set(ids) == _SCENARIOS,
                "serving profile must contain every required scenario exactly once")
    require(isinstance(expected_scopes, list) and expected_scopes, "required model/route scopes are missing")
    scopes, result = set(), []
    for scope in expected_scopes:
        require(isinstance(scope, dict) and set(scope) == {"id", "model", "route", "profile"},
                "invalid required model/route scope")
        require(isinstance(scope["id"], str) and _NAME.fullmatch(scope["id"])
                and scope["id"] not in scopes, "invalid or duplicate scope id")
        require(all(isinstance(scope[k], str) and scope[k] for k in ("model", "route", "profile"))
                and scope["profile"] in profiles, "unknown model/route profile")
        scopes.add(scope["id"])
        for declaration in profiles[scope["profile"]]:
            result.append({"id": scope["id"] + "/" + declaration["id"],
                           "scope": dict(scope), "scenario": declaration["id"],
                           "requirements": dict(declaration["requirements"])})
    return result


def validate_cell_coverage(manifest_bytes, expected_scopes, observed_cells):
    """Refuse skipped, missing, duplicate or substituted cells before predicates run.

    Evidence/body/lease fields are validated by the scenario and release binding
    layers. This returns authoritative requirements, not a passing verdict.
    """
    expected = {c["id"]: c for c in required_cells(manifest_bytes, expected_scopes)}
    require(isinstance(observed_cells, list), "observed serving cells must be an array")
    found = {}
    for cell in observed_cells:
        require(isinstance(cell, dict) and isinstance(cell.get("id"), str)
                and cell["id"] in expected and cell["id"] not in found,
                "unknown or duplicate observed serving cell")
        require(cell.get("scope") == expected[cell["id"]]["scope"]
                and cell.get("scenario") == expected[cell["id"]]["scenario"],
                "observed model/route/scenario differs from required scope")
        require("skip" not in cell and "skipped" not in cell and cell.get("state") == "captured",
                "required serving cell was not captured")
        found[cell["id"]] = cell
    require(found.keys() == expected.keys(), "missing required serving cells")
    return list(expected.values())
