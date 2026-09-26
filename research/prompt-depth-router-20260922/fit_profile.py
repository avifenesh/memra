"""Fit a request-class lookup from matched whole-request measurements."""
import math


def fit_profile(global_k, trials, group_ids):
    if len(group_ids) < 2 or len(set(group_ids)) != len(group_ids):
        raise ValueError("need distinct calibration conversations")
    profile, fitting = {}, {}
    for kind in ("prose", "code", "numeric"):
        values_by_k = trials[kind]
        if global_k not in values_by_k:
            raise ValueError("global control is missing")
        for values in values_by_k.values():
            if len(values) != len(group_ids):
                raise ValueError("unmatched calibration coverage")
            for tokens, seconds in values:
                if tokens < 0 or not math.isfinite(seconds) or seconds < 0 or (tokens > 0) != (seconds > 0):
                    raise ValueError("invalid request cost")
        baseline = values_by_k[global_k]
        eligible, details = [global_k], {}
        for k, values in values_by_k.items():
            if any((a[0] > 0) != (b[0] > 0) for a, b in zip(values, baseline)):
                raise ValueError("class coverage differs between depths")
            confirmations = sum(
                seconds > 0 and reference[1] > 0
                and tokens / seconds > reference[0] / reference[1]
                for (tokens, seconds), reference in zip(values, baseline)
            )
            tokens, seconds = map(sum, zip(*values))
            details[k] = {
                "tokens": tokens, "seconds": seconds,
                "rate": tokens / seconds if seconds else None,
                "confirmations": confirmations,
            }
            if confirmations >= 2:
                eligible.append(k)
        profile[kind] = min(eligible, key=lambda k: (-(details[k]["rate"] or 0), k))
        fitting[kind] = details
    profile["fallback"] = global_k
    return profile, fitting
