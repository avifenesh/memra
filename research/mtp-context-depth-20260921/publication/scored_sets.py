"""Apply whole-set exclusions while retaining the original scoring function."""


def matched_report(groups, declared, observed, summarize, annotate=False):
    if declared != observed:
        raise ValueError("Excluded sets must exactly match independently observed repetition")
    if len(groups) != 10 or [g["cycle"] for g in groups] != list(range(10)):
        raise ValueError("The complete original ten-set schedule is required")
    arms = {"calibrated", "native", "context", "learned", "measured"}
    if any(len(g["records"]) != 5 or {r["arm"] for r in g["records"]} != arms
           or any(r["turns"] != 8 for r in g["records"]) for g in groups):
        raise ValueError("Every recorded matched set must retain all five eight-turn policies")
    if not set(declared) <= {str(g["cycle"]) for g in groups}:
        raise ValueError("An exclusion names a set outside the original schedule")
    included = [g for g in groups if str(g["cycle"]) not in declared]
    if not included:
        raise ValueError("No complete unflagged matched set remains")
    report = summarize(included, 10)
    if annotate:
        report.update(
            recorded_sets=10, recorded_runs=50, recorded_turns=400,
            original_schedule_completed=True,
            excluded_cycles=sorted(int(c) for c in declared),
        )
    return report
