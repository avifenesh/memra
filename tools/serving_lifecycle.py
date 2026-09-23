"""Derive worker-generation accounting from captured health responses."""

from serving_release import json_object, require


def account_generations(accounting, attempts, health_samples):
    """Exclude requests bracketed by different worker generations from clean timing.

    Health samples bracket the *observation interval*, so affected does not assert
    that the particular request caused or survived a panic. This deliberately
    conservative count stays separate from the raw wire outcome. The runner must
    pin these observations to its owned server process and recorded port before
    this analysis can be included in a qualified serving cell.
    """
    require(isinstance(health_samples, list) and len(health_samples) >= 2,
            "worker accounting needs before and after health samples")
    samples, identity = [], None
    previous_start, previous_generation = -1, -1
    for sample in health_samples:
        require(isinstance(sample, dict), "invalid health observation")
        start, end = sample.get("started_ns"), sample.get("finished_ns")
        require(type(start) is int and type(end) is int and 0 <= start <= end
                and start > previous_start, "invalid or unordered health timestamps")
        owner = sample.get("server_identity")
        require(isinstance(owner, dict) and set(owner) == {"pid", "start_identity"}
                and type(owner["pid"]) is int and owner["pid"] > 1
                and isinstance(owner["start_identity"], str) and owner["start_identity"],
                "health observation lacks owned-server identity")
        if identity is None:
            identity = owner
        require(owner == identity, "server process changed during generation census")
        status = sample.get("status")
        require(type(status) is int and status in (200, 503), "invalid health response status")
        require(isinstance(sample.get("body"), bytes), "health body must be raw bytes")
        payload = json_object(sample["body"])
        worker = payload.get("worker")
        require(isinstance(worker, dict), "health body lacks worker state")
        generation = worker.get("generation")
        require(type(generation) is int and generation >= previous_generation and generation >= 0,
                "invalid or decreasing worker generation")
        samples.append((start, end, generation))
        previous_start, previous_generation = start, generation

    require(isinstance(attempts, list) and isinstance(accounting, dict), "invalid attempt accounting")
    results = accounting.get("requests")
    require(isinstance(results, list), "missing wire results")
    result_by_id = {}
    for result in results:
        require(isinstance(result, dict) and isinstance(result.get("id"), str)
                and result["id"] not in result_by_id, "duplicate or invalid wire result")
        result_by_id[result["id"]] = result
    seen, observations = set(), []
    for attempt in attempts:
        require(isinstance(attempt, dict) and isinstance(attempt.get("id"), str),
                "invalid attempt id")
        request_id = attempt["id"]
        require(request_id not in seen and request_id in result_by_id,
                "duplicate or unknown generation-accounted request")
        seen.add(request_id)
        start, end = attempt.get("started_ns"), attempt.get("finished_ns")
        require(type(start) is int and type(end) is int and 0 <= start <= end,
                "invalid request timestamps")
        result = result_by_id[request_id]
        require(type(result.get("wall_ns")) is int and result["wall_ns"] == end - start,
                "generation and wire timings disagree")
        before = [s for s in samples if s[1] <= start]
        after = [s for s in samples if s[0] >= end]
        require(before and after, "health samples do not bracket every request")
        generation_before, generation_after = before[-1][2], after[0][2]
        require(generation_before <= generation_after, "inverted generation bracket")
        affected = generation_before != generation_after
        observations.append({"id": request_id, "generation_before": generation_before,
                             "generation_after": generation_after,
                             "generation_affected": affected,
                             "wire_outcome": result["outcome"],
                             "clean_latency_ns": result["wall_ns"] if
                             result["outcome"] == "clean_success" and not affected else None})
    require(seen == result_by_id.keys() and type(accounting.get("attempted")) is int
            and accounting["attempted"] == len(attempts), "generation census omitted requests")
    return {"server_identity": dict(identity),
            "respawns_observed": samples[-1][2] - samples[0][2],
            "generation_affected_requests": sum(o["generation_affected"] for o in observations),
            "requests": observations,
            "clean_latency_ns": [o["clean_latency_ns"] for o in observations
                                 if o["clean_latency_ns"] is not None]}
