#!/usr/bin/env python3
"""Deterministic byte-cache policy simulation for the SLRU acceptance target.

This is not a latency or throughput benchmark.  It replays explicit request-key/entry-byte
sequences through the shipped v0.82.0 LRU and SLRU policy semantics.  The measured model entry
sizes and configured byte budgets come from ``traffic_model.lock.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MIB = 1024 * 1024
EVICTCHURN_ENTRY_BYTES = 68_313_600
EVICTCHURN_BUDGET_BYTES = 782 * MIB


@dataclass(frozen=True)
class Request:
    key: str
    entry_bytes: int
    traffic_class: str
    model: str
    logical_session: int | None


@dataclass
class Entry:
    entry_bytes: int
    last_use: int
    segment: str


class LruCache:
    name = "lru"

    def __init__(self, budget_bytes: int) -> None:
        self.budget_bytes = budget_bytes
        self.entries: dict[str, Entry] = {}
        self.total_bytes = 0

    def access(self, request: Request, clock: int) -> dict:
        if request.key in self.entries:
            self.entries[request.key].last_use = clock
            return {"hit": True, "evictions": [], "promotions": 0, "demotions": 0}
        if request.entry_bytes > self.budget_bytes:
            return {"hit": False, "evictions": [], "promotions": 0, "demotions": 0,
                    "refused": True}
        self.entries[request.key] = Entry(request.entry_bytes, clock, "lru")
        self.total_bytes += request.entry_bytes
        evictions: list[dict] = []
        while self.total_bytes > self.budget_bytes:
            victim_key, victim = min(
                self.entries.items(), key=lambda item: (item[1].last_use, item[0])
            )
            del self.entries[victim_key]
            self.total_bytes -= victim.entry_bytes
            evictions.append(
                {"key": victim_key, "entry_bytes": victim.entry_bytes, "segment": "lru"}
            )
        return {"hit": False, "evictions": evictions, "promotions": 0, "demotions": 0}

    def state(self) -> dict:
        return {
            "entries": len(self.entries),
            "bytes": self.total_bytes,
            "probation_entries": None,
            "probation_bytes": None,
            "protected_entries": None,
            "protected_bytes": None,
        }


class SlruCache:
    name = "slru"

    def __init__(self, budget_bytes: int, protected_pct: int) -> None:
        self.budget_bytes = budget_bytes
        self.protected_pct = protected_pct
        self.protected_target_bytes = (
            (budget_bytes // 100) * protected_pct
            + ((budget_bytes % 100) * protected_pct) // 100
        )
        self.entries: dict[str, Entry] = {}
        self.total_bytes = 0
        self.probation_bytes = 0
        self.protected_bytes = 0

    def _oldest(self, segment: str) -> tuple[str, Entry] | None:
        candidates = [item for item in self.entries.items() if item[1].segment == segment]
        return min(candidates, key=lambda item: (item[1].last_use, item[0])) if candidates else None

    def _rebalance_protected(self) -> int:
        demotions = 0
        while self.protected_bytes > self.protected_target_bytes:
            victim = self._oldest("protected")
            if victim is None:
                break
            _, entry = victim
            entry.segment = "probation"
            self.protected_bytes -= entry.entry_bytes
            self.probation_bytes += entry.entry_bytes
            demotions += 1
        return demotions

    def access(self, request: Request, clock: int) -> dict:
        existing = self.entries.get(request.key)
        if existing is not None:
            promotions = 0
            if existing.segment == "probation":
                existing.segment = "protected"
                self.probation_bytes -= existing.entry_bytes
                self.protected_bytes += existing.entry_bytes
                promotions = 1
            existing.last_use = clock
            demotions = self._rebalance_protected()
            return {
                "hit": True,
                "evictions": [],
                "promotions": promotions,
                "demotions": demotions,
            }
        if request.entry_bytes > self.budget_bytes:
            return {"hit": False, "evictions": [], "promotions": 0, "demotions": 0,
                    "refused": True}
        self.entries[request.key] = Entry(request.entry_bytes, clock, "probation")
        self.total_bytes += request.entry_bytes
        self.probation_bytes += request.entry_bytes
        evictions: list[dict] = []
        while self.total_bytes > self.budget_bytes:
            victim = self._oldest("probation")
            if victim is None:
                raise AssertionError("SLRU exceeded its byte budget without a probation victim")
            victim_key, entry = victim
            del self.entries[victim_key]
            self.total_bytes -= entry.entry_bytes
            self.probation_bytes -= entry.entry_bytes
            evictions.append(
                {"key": victim_key, "entry_bytes": entry.entry_bytes,
                 "segment": "probation"}
            )
        return {"hit": False, "evictions": evictions, "promotions": 0, "demotions": 0}

    def state(self) -> dict:
        probation_entries = sum(entry.segment == "probation" for entry in self.entries.values())
        protected_entries = sum(entry.segment == "protected" for entry in self.entries.values())
        assert self.total_bytes == self.probation_bytes + self.protected_bytes
        assert self.total_bytes <= self.budget_bytes
        return {
            "entries": len(self.entries),
            "bytes": self.total_bytes,
            "probation_entries": probation_entries,
            "probation_bytes": self.probation_bytes,
            "protected_entries": protected_entries,
            "protected_bytes": self.protected_bytes,
        }


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def reuse_gap_summary(requests: list[Request]) -> dict:
    last: dict[str, int] = {}
    gaps: list[int] = []
    for index, request in enumerate(requests):
        if request.key in last:
            gaps.append(index - last[request.key])
        last[request.key] = index
    return {
        "n": len(gaps),
        "p50_requests": percentile(gaps, 0.50),
        "p90_requests": percentile(gaps, 0.90),
        "p99_requests": percentile(gaps, 0.99),
        "max_requests": max(gaps) if gaps else None,
    }


def run_trace(
    policy: LruCache | SlruCache,
    requests: list[Request],
    *,
    start_clock: int = 1,
) -> dict:
    seen: set[str] = set()
    ever_hit: set[str] = set()
    returning_keys = {request.key for request in requests if request.traffic_class == "returning"}
    last_eviction: dict[str, dict] = {}
    counters = {
        "requests": 0,
        "hits": 0,
        "misses": 0,
        "returning_requests": 0,
        "returning_hits": 0,
        "scan_requests": 0,
        "scan_hits": 0,
        "reuse_opportunities": 0,
        "reuse_hits": 0,
        "thrash_misses": 0,
        "pre_first_hit_thrash_misses": 0,
        "post_first_hit_misses": 0,
        "direct_scan_caused_post_first_hit_misses": 0,
        "evictions": 0,
        "returning_evictions": 0,
        "proven_returning_evictions": 0,
        "scan_triggered_evictions": 0,
        "promotions": 0,
        "demotions": 0,
        "refusals": 0,
    }
    for clock, request in enumerate(requests, start=start_clock):
        seen_before = request.key in seen
        hit_before = request.key in ever_hit
        prior_eviction = last_eviction.get(request.key)
        result = policy.access(request, clock)
        hit = bool(result["hit"])
        counters["requests"] += 1
        counters["hits" if hit else "misses"] += 1
        if request.traffic_class == "returning":
            counters["returning_requests"] += 1
            counters["returning_hits"] += int(hit)
        else:
            counters["scan_requests"] += 1
            counters["scan_hits"] += int(hit)
        if seen_before:
            counters["reuse_opportunities"] += 1
            counters["reuse_hits"] += int(hit)
            if not hit:
                counters["thrash_misses"] += 1
                if hit_before:
                    counters["post_first_hit_misses"] += 1
                    if prior_eviction and prior_eviction["trigger_class"] == "scan":
                        counters["direct_scan_caused_post_first_hit_misses"] += 1
                else:
                    counters["pre_first_hit_thrash_misses"] += 1
        if hit:
            ever_hit.add(request.key)
        counters["promotions"] += int(result.get("promotions", 0))
        counters["demotions"] += int(result.get("demotions", 0))
        counters["refusals"] += int(bool(result.get("refused", False)))
        for eviction in result["evictions"]:
            victim_key = str(eviction["key"])
            counters["evictions"] += 1
            counters["scan_triggered_evictions"] += int(request.traffic_class == "scan")
            if victim_key in returning_keys:
                counters["returning_evictions"] += 1
                counters["proven_returning_evictions"] += int(victim_key in ever_hit)
            last_eviction[victim_key] = {
                "clock": clock,
                "trigger_class": request.traffic_class,
                "victim_segment": eviction["segment"],
                "victim_had_hit": victim_key in ever_hit,
            }
        seen.add(request.key)
        state = policy.state()
        if state["bytes"] > policy.budget_bytes:
            raise AssertionError("policy exceeded configured byte budget")
    state = policy.state()
    return {
        **counters,
        "hit_rate": rate(counters["hits"], counters["requests"]),
        "returning_hit_rate": rate(counters["returning_hits"], counters["returning_requests"]),
        "reuse_hit_rate": rate(counters["reuse_hits"], counters["reuse_opportunities"]),
        "final_state": state,
        "reuse_gap": reuse_gap_summary(requests),
    }


def policy_for(name: str, budget_bytes: int, protected_pct: int) -> LruCache | SlruCache:
    if name == "lru":
        return LruCache(budget_bytes)
    if name == "slru":
        return SlruCache(budget_bytes, protected_pct)
    raise ValueError(name)


def variant_hot_keys(variant: str, sessions: int, sizes: dict[str, int]) -> list[Request]:
    requests: list[Request] = []
    models = ("q27", "q35") if variant == "paired" else (variant,)
    for session in range(sessions):
        for model in models:
            requests.append(
                Request(
                    key=f"{variant}:hot:s{session}:{model}",
                    entry_bytes=sizes[model],
                    traffic_class="returning",
                    model=model,
                    logical_session=session,
                )
            )
    return requests


def hot_scan_schedule(
    *,
    variant: str,
    sessions: int,
    requests_n: int,
    returning_fraction: float,
    alpha: float,
    seed: int,
    sizes: dict[str, int],
) -> list[Request]:
    hot_keys = variant_hot_keys(variant, sessions, sizes)
    returning_n = round(requests_n * returning_fraction)
    scan_n = requests_n - returning_n
    if returning_n < len(hot_keys):
        raise ValueError("returning request count must cover every hot key once")
    rng = random.Random(seed)
    labels = [True] * returning_n + [False] * scan_n
    rng.shuffle(labels)
    model_divisor = 2 if variant == "paired" else 1
    weights = [
        1.0 / (((request.logical_session or 0) + 1) ** alpha) / model_divisor
        for request in hot_keys
    ]
    hot_values = hot_keys + rng.choices(hot_keys, weights=weights, k=returning_n - len(hot_keys))
    rng.shuffle(hot_values)
    hot_iter = iter(hot_values)
    scan_values: list[Request] = []
    scan_models = ("q27", "q35") if variant == "paired" else (variant,)
    for index in range(scan_n):
        model = scan_models[index % len(scan_models)]
        scan_values.append(
            Request(
                key=f"{variant}:scan:{seed}:{index}:{model}",
                entry_bytes=sizes[model],
                traffic_class="scan",
                model=model,
                logical_session=None,
            )
        )
    rng.shuffle(scan_values)
    scan_iter = iter(scan_values)
    return [next(hot_iter) if is_hot else next(scan_iter) for is_hot in labels]


def evictchurn_hot_schedule() -> list[Request]:
    working_set = 40
    requests_n = 160
    hot_n = 8
    hot_requests = 128
    cold = list(range(hot_n, working_set))
    labels = [True] * hot_requests + [False] * (requests_n - hot_requests)
    rng = random.Random(3407)
    rng.shuffle(labels)
    weights = [1.0 / (rank + 1) for rank in range(hot_n)]
    hot_values = list(range(hot_n)) + rng.choices(
        list(range(hot_n)), weights=weights, k=hot_requests - hot_n
    )
    rng.shuffle(hot_values)
    hot_iter = iter(hot_values)
    cold_iter = iter(cold[index % len(cold)] for index in range(requests_n - hot_requests))
    ids = [next(hot_iter) if is_hot else next(cold_iter) for is_hot in labels]
    return [
        Request(
            key=f"evict:{prefix_id}",
            entry_bytes=EVICTCHURN_ENTRY_BYTES,
            traffic_class="returning" if prefix_id < hot_n else "scan",
            model="q35-264-token",
            logical_session=prefix_id if prefix_id < hot_n else None,
        )
        for prefix_id in ids
    ]


def evictchurn_validation(protected_pct: int) -> dict:
    expected = {
        "round_robin": {
            "lru": {"hits": 0, "evictions": 68},
            "slru": {"hits": 0, "evictions": 68},
        },
        "hotset": {
            "lru": {"hits": 107, "evictions": 41, "thrash_misses": 13},
            "slru": {
                "hits": 115,
                "evictions": 33,
                "thrash_misses": 5,
                "post_first_hit_misses": 0,
            },
        },
        "sequential_scan": {
            "lru": {"hits": 0, "evictions": 28},
            "slru": {"hits": 0, "evictions": 28},
        },
    }
    round_robin = [
        Request(f"evict:{index % 40}", EVICTCHURN_ENTRY_BYTES, "returning",
                "q35-264-token", index % 40)
        for index in range(80)
    ]
    sequential = [
        Request(f"evict:{index}", EVICTCHURN_ENTRY_BYTES, "scan", "q35-264-token", None)
        for index in range(40)
    ]
    schedules = {
        "round_robin": round_robin,
        "hotset": evictchurn_hot_schedule(),
        "sequential_scan": sequential,
    }
    observed: dict[str, dict] = {}
    failures: list[str] = []
    for schedule_name, schedule in schedules.items():
        observed[schedule_name] = {}
        for policy_name in ("lru", "slru"):
            metrics = run_trace(
                policy_for(policy_name, EVICTCHURN_BUDGET_BYTES, protected_pct), schedule
            )
            observed[schedule_name][policy_name] = {
                field: metrics[field] for field in expected[schedule_name][policy_name]
            }
            if observed[schedule_name][policy_name] != expected[schedule_name][policy_name]:
                failures.append(
                    f"{schedule_name}/{policy_name}: expected "
                    f"{expected[schedule_name][policy_name]}, observed "
                    f"{observed[schedule_name][policy_name]}"
                )
    return {
        "kind": "validation",
        "schema": "memra.slrutarget.simulation.v1",
        "name": "reproduce committed evictchurn and slrucache summaries",
        "expected": expected,
        "observed": observed,
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }


def simulation_row(
    *,
    scenario: str,
    variant: str,
    budget_mib: int,
    policy_name: str,
    protected_pct: int,
    requests: list[Request],
    seed: int | None,
    parameters: dict,
    policy: LruCache | SlruCache | None = None,
    start_clock: int = 1,
) -> dict:
    budget_bytes = budget_mib * MIB
    chosen = policy or policy_for(policy_name, budget_bytes, protected_pct)
    metrics = run_trace(chosen, requests, start_clock=start_clock)
    return {
        "kind": "simulation",
        "schema": "memra.slrutarget.simulation.v1",
        "scenario": scenario,
        "variant": variant,
        "budget_mib": budget_mib,
        "budget_bytes": budget_bytes,
        "policy": policy_name,
        "protected_pct": protected_pct if policy_name == "slru" else None,
        "seed": seed,
        "parameters": parameters,
        "metrics": metrics,
    }


def stationary_cycle_rows(
    variant: str, budget_mib: int, sizes: dict[str, int], protected_pct: int
) -> Iterable[dict]:
    unit_bytes = sizes["q27"] + sizes["q35"] if variant == "paired" else sizes[variant]
    capacity = (budget_mib * MIB) // unit_bytes
    for sessions in sorted({max(1, capacity - 1), capacity, capacity + 1}):
        one_cycle = variant_hot_keys(variant, sessions, sizes)
        requests = one_cycle * 3
        for policy_name in ("lru", "slru"):
            yield simulation_row(
                scenario="stationary_cycle",
                variant=variant,
                budget_mib=budget_mib,
                policy_name=policy_name,
                protected_pct=protected_pct,
                requests=requests,
                seed=None,
                parameters={"logical_sessions": sessions, "cycles": 3,
                            "floor_capacity_sessions": capacity},
            )


def warm_protected(
    policy: LruCache | SlruCache,
    requests: list[Request],
    start_clock: int = 1,
) -> int:
    clock = start_clock
    for request in requests:
        policy.access(request, clock)
        clock += 1
        policy.access(request, clock)
        clock += 1
    return clock


def turnover_rows(
    variant: str, budget_mib: int, sizes: dict[str, int], protected_pct: int
) -> Iterable[dict]:
    budget_bytes = budget_mib * MIB
    protected_target = (budget_bytes // 100) * protected_pct + (
        (budget_bytes % 100) * protected_pct
    ) // 100
    unit_bytes = sizes["q27"] + sizes["q35"] if variant == "paired" else sizes[variant]
    old_sessions = protected_target // unit_bytes
    old_requests = variant_hot_keys(variant, old_sessions, sizes)
    old_bytes = old_sessions * unit_bytes
    residual_sessions = (budget_bytes - old_bytes) // unit_bytes
    new_sessions = residual_sessions + 1
    new_one_cycle = [
        Request(
            key=request.key.replace(":hot:", ":new-hot:"),
            entry_bytes=request.entry_bytes,
            traffic_class="returning",
            model=request.model,
            logical_session=request.logical_session,
        )
        for request in variant_hot_keys(variant, new_sessions, sizes)
    ]
    requests = new_one_cycle * 4
    for policy_name in ("lru", "slru"):
        policy = policy_for(policy_name, budget_bytes, protected_pct)
        measurement_clock = warm_protected(policy, old_requests)
        yield simulation_row(
            scenario="hotset_turnover_cycle",
            variant=variant,
            budget_mib=budget_mib,
            policy_name=policy_name,
            protected_pct=protected_pct,
            requests=requests,
            seed=None,
            parameters={
                "old_protected_logical_sessions": old_sessions,
                "old_protected_bytes": old_bytes,
                "new_logical_sessions": new_sessions,
                "new_working_set_bytes": new_sessions * unit_bytes,
                "residual_floor_capacity_sessions": residual_sessions,
                "cycles": 4,
                "phase_boundary": "old demonstrated-reuse cohort goes idle; disjoint new cohort cycles",
            },
            policy=policy,
            start_clock=measurement_clock,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()
    lock = json.loads(args.model.read_text(encoding="utf-8"))
    sizes = {key: int(value) for key, value in lock["entry_bytes"].items()}
    protected_pct = int(lock["protected_pct"])
    protocol = {
        "kind": "protocol",
        "schema": "memra.slrutarget.simulation.v1",
        "runtime_base": lock["runtime_base"],
        "runtime_tag": lock["runtime_tag"],
        "cachesize_source": lock["cachesize_source"],
        "entry_bytes": sizes,
        "budgets_mib": lock["budgets_mib"],
        "protected_pct": protected_pct,
        "policy_semantics": {
            "lru": "global byte-budgeted timestamp LRU; every fitting miss admits",
            "slru": "new entry probation; hit promotes; protected overflow demotes its LRU; capacity evicts probation LRU",
        },
        "scope": "policy behavior only; no latency, throughput, GPU, or live-traffic claim",
    }
    print(json.dumps(protocol, sort_keys=True), flush=True)
    validation = evictchurn_validation(protected_pct)
    print(json.dumps(validation, sort_keys=True), flush=True)
    if validation["verdict"] != "PASS":
        return 1

    for budget_mib in lock["budgets_mib"]:
        budget_bytes = int(budget_mib) * MIB
        protected_target = (budget_bytes // 100) * protected_pct + (
            (budget_bytes % 100) * protected_pct
        ) // 100
        capacity = {
            "kind": "capacity",
            "schema": "memra.slrutarget.simulation.v1",
            "budget_mib": int(budget_mib),
            "budget_bytes": budget_bytes,
            "protected_target_bytes": protected_target,
            "q27_entries": budget_bytes // sizes["q27"],
            "q35_entries": budget_bytes // sizes["q35"],
            "paired_sessions": budget_bytes // (sizes["q27"] + sizes["q35"]),
            "q27_protected_entries": protected_target // sizes["q27"],
            "q35_protected_entries": protected_target // sizes["q35"],
            "paired_protected_sessions": protected_target // (sizes["q27"] + sizes["q35"]),
        }
        print(json.dumps(capacity, sort_keys=True), flush=True)

    primary = lock["primary"]
    seeds = range(int(primary["seeds"]["start"]),
                  int(primary["seeds"]["start"]) + int(primary["seeds"]["count"]))
    for variant in primary["variants"]:
        for budget_mib in lock["budgets_mib"]:
            for seed in seeds:
                requests = hot_scan_schedule(
                    variant=variant,
                    sessions=int(primary["logical_sessions"]),
                    requests_n=int(primary["requests"]),
                    returning_fraction=float(primary["returning_fraction"]),
                    alpha=float(primary["zipf_alpha"]),
                    seed=seed,
                    sizes=sizes,
                )
                parameters = {
                    "logical_sessions": int(primary["logical_sessions"]),
                    "requests": int(primary["requests"]),
                    "returning_fraction": float(primary["returning_fraction"]),
                    "scan_fraction": 1.0 - float(primary["returning_fraction"]),
                    "zipf_alpha": float(primary["zipf_alpha"]),
                }
                for policy_name in ("lru", "slru"):
                    row = simulation_row(
                        scenario="primary_hot_scan",
                        variant=variant,
                        budget_mib=int(budget_mib),
                        policy_name=policy_name,
                        protected_pct=protected_pct,
                        requests=requests,
                        seed=seed,
                        parameters=parameters,
                    )
                    print(json.dumps(row, sort_keys=True), flush=True)

    sensitivity = lock["sensitivity"]
    sensitivity_seeds = range(
        int(sensitivity["seeds"]["start"]),
        int(sensitivity["seeds"]["start"]) + int(sensitivity["seeds"]["count"]),
    )
    for variant in sensitivity["variants"]:
        for budget_mib in lock["budgets_mib"]:
            for sessions in sensitivity["logical_sessions"]:
                for returning_fraction in sensitivity["returning_fractions"]:
                    for alpha in sensitivity["zipf_alpha"]:
                        for seed in sensitivity_seeds:
                            requests = hot_scan_schedule(
                                variant=variant,
                                sessions=int(sessions),
                                requests_n=int(primary["requests"]),
                                returning_fraction=float(returning_fraction),
                                alpha=float(alpha),
                                seed=seed,
                                sizes=sizes,
                            )
                            parameters = {
                                "logical_sessions": int(sessions),
                                "requests": int(primary["requests"]),
                                "returning_fraction": float(returning_fraction),
                                "scan_fraction": 1.0 - float(returning_fraction),
                                "zipf_alpha": float(alpha),
                            }
                            for policy_name in ("lru", "slru"):
                                row = simulation_row(
                                    scenario="sensitivity_hot_scan",
                                    variant=variant,
                                    budget_mib=int(budget_mib),
                                    policy_name=policy_name,
                                    protected_pct=protected_pct,
                                    requests=requests,
                                    seed=seed,
                                    parameters=parameters,
                                )
                                print(json.dumps(row, sort_keys=True), flush=True)

    for variant in sensitivity["variants"]:
        for budget_mib in lock["budgets_mib"]:
            for row in stationary_cycle_rows(variant, int(budget_mib), sizes, protected_pct):
                print(json.dumps(row, sort_keys=True), flush=True)
            for row in turnover_rows(variant, int(budget_mib), sizes, protected_pct):
                print(json.dumps(row, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
