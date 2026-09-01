#!/usr/bin/env python3
import json
import hashlib
import pathlib
import re
import statistics
import sys


def require_match(pattern: str, text: str, label: str) -> re.Match[str]:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        raise SystemExit(f"{label}: missing required pattern: {pattern}")
    return match


def expected_form(label: str) -> str | None:
    if label.endswith("-tile") or label == "naked-debug":
        return "TILE"
    if label.endswith("-sk"):
        return "SK"
    return None


out = pathlib.Path(sys.argv[1])
logs = sorted(out.glob("[0-9][0-9]-*.log"))
if len(logs) != 12:
    raise SystemExit(f"expected 12 arm logs, found {len(logs)}")

rows = []
for log in logs:
    label = log.stem.split("-", 1)[1]
    text = log.read_text(errors="replace")
    stats = require_match(
        r"\[gemma-spec\] rounds=(\d+) drafted=(\d+) accepted=(\d+)"
        r"\s+accept-rate=([0-9.]+) tok/round=([0-9.]+)",
        text,
        label,
    )
    perf = require_match(
        r"plain: ([0-9.]+) tok/s \| spec: ([0-9.]+) tok/s .*"
        r"stream agreement (\d+)/(\d+)",
        text,
        label,
    )
    plain_tokens = require_match(r"^plain tokens: (\[.*\])$", text, label).group(1)
    spec_tokens = require_match(r"^spec tokens: (\[.*\])$", text, label).group(1)
    if plain_tokens != spec_tokens:
        raise SystemExit(f"{label}: spec token stream differs from plain")

    agreement = int(perf.group(3))
    total = int(perf.group(4))
    if agreement != 128 or total != 128:
        raise SystemExit(f"{label}: expected stream agreement 128/128, got {agreement}/{total}")

    selected = sorted(set(re.findall(r"\[mmq-sk\].* -> (SK|TILE)$", text, re.MULTILINE)))
    expected = expected_form(label)
    if expected is not None and selected != [expected]:
        raise SystemExit(f"{label}: expected selector {expected}, observed {selected}")

    rows.append(
        {
            "label": label,
            "rounds": int(stats.group(1)),
            "drafted": int(stats.group(2)),
            "accepted": int(stats.group(3)),
            "accept_rate": float(stats.group(4)),
            "tok_per_round": float(stats.group(5)),
            "plain_tok_s": float(perf.group(1)),
            "spec_tok_s": float(perf.group(2)),
            "agreement": f"{agreement}/{total}",
            "selector": selected[0] if selected else None,
            "plain_tokens": plain_tokens,
            "plain_tokens_sha256": hashlib.sha256(plain_tokens.encode()).hexdigest(),
        }
    )

form_tokens = {}
for form in ("tile", "sk"):
    form_rows = [row for row in rows if row["label"].endswith(f"-{form}")]
    if len(form_rows) != 5:
        raise SystemExit(f"{form}: expected five repetitions")
    acceptance = {(row["drafted"], row["accepted"]) for row in form_rows}
    if len(acceptance) != 1:
        raise SystemExit(f"{form}: acceptance is not deterministic: {sorted(acceptance)}")
    token_streams = {row["plain_tokens"] for row in form_rows}
    if len(token_streams) != 1:
        raise SystemExit(f"{form}: plain token stream is not deterministic")
    form_tokens[form] = next(iter(token_streams))

tile_rows = [row for row in rows if row["label"].endswith("-tile")]
sk_rows = [row for row in rows if row["label"].endswith("-sk")]
defaults = {row["label"]: row for row in rows if row["label"].startswith("naked-")}
if set(defaults) != {"naked-debug", "naked-clean"}:
    raise SystemExit(f"missing default boots: {sorted(defaults)}")
tile_acceptance = (tile_rows[0]["drafted"], tile_rows[0]["accepted"])
for label, row in defaults.items():
    if (row["drafted"], row["accepted"]) != tile_acceptance:
        raise SystemExit(f"{label}: default acceptance does not reproduce TILE")
    if row["plain_tokens"] != form_tokens["tile"]:
        raise SystemExit(f"{label}: default token stream does not reproduce TILE")

summary = {
    "schema": "memra.mmq-deterministic.v1",
    "result": "PASS",
    "cross_form_token_identity": form_tokens["tile"] == form_tokens["sk"],
    "cross_form_divergence_proves_timing_selection_unsafe":
        form_tokens["tile"] != form_tokens["sk"],
    "tile": {
        "n": len(tile_rows),
        "drafted": tile_rows[0]["drafted"],
        "accepted": tile_rows[0]["accepted"],
        "accept_rate": tile_rows[0]["accept_rate"],
        "plain_tokens_sha256": tile_rows[0]["plain_tokens_sha256"],
        "plain_tok_s_median": statistics.median(row["plain_tok_s"] for row in tile_rows),
        "spec_tok_s_median": statistics.median(row["spec_tok_s"] for row in tile_rows),
    },
    "sk": {
        "n": len(sk_rows),
        "drafted": sk_rows[0]["drafted"],
        "accepted": sk_rows[0]["accepted"],
        "accept_rate": sk_rows[0]["accept_rate"],
        "plain_tokens_sha256": sk_rows[0]["plain_tokens_sha256"],
        "plain_tok_s_median": statistics.median(row["plain_tok_s"] for row in sk_rows),
        "spec_tok_s_median": statistics.median(row["spec_tok_s"] for row in sk_rows),
    },
    "default_debug_selector": defaults["naked-debug"]["selector"],
    "default_debug_matches_tile": defaults["naked-debug"]["plain_tokens"] == form_tokens["tile"],
    "default_clean_matches_tile": defaults["naked-clean"]["plain_tokens"] == form_tokens["tile"],
    "rows": [
        {key: value for key, value in row.items() if key != "plain_tokens"}
        for row in rows
    ],
}
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
with (out / "summary.tsv").open("w") as handle:
    fields = [
        "label",
        "rounds",
        "drafted",
        "accepted",
        "accept_rate",
        "plain_tok_s",
        "spec_tok_s",
        "agreement",
        "selector",
    ]
    handle.write("\t".join(fields) + "\n")
    for row in rows:
        handle.write("\t".join(str(row[field]) for field in fields) + "\n")
print(json.dumps({
    key: summary[key]
    for key in (
        "result",
        "cross_form_token_identity",
        "cross_form_divergence_proves_timing_selection_unsafe",
        "tile",
        "sk",
        "default_debug_selector",
        "default_debug_matches_tile",
        "default_clean_matches_tile",
    )
}))
