#!/usr/bin/env python3
"""Regenerate the perf surfaces from research/tune-data/current-board.json.

Generated surfaces:
    - docs/MODELS.md PERF-MODELS block      (supported-models table, from "supported_models")
    - docs/PERFORMANCE.md PERF-DATE / PERF-PLAIN / PERF-SPEC blocks  (full 5090 boards)
    - docs/PERFORMANCE.md PERF-H100 block   (full H100 board, from "h100_board")

The comparison SVG cards (docs/perf-card*.svg) were RETIRED 2026-08-09 (owner call):
memra is its own sm_120 engine, not a llama.cpp bypass — reference numbers stay in the
boards as regression anchors, but no scoreboard-style artifact is published.

Usage:
    tools/update-perf-board.py          # regenerate all surfaces in place
    tools/update-perf-board.py --check  # exit 1 if any surface would change (for a pre-push hook / CI)

The board JSON is the single source of truth for published numbers. Never hand-edit the
generated regions in docs/MODELS.md or docs/PERFORMANCE.md (marked with PERF-*:START/END
comments); edit research/tune-data/current-board.json and rerun this script instead. The README
is a concise entry point and carries no generated performance surface.

Note: 5090 ratios are computed from the display-rounded tok/s values stored in the board JSON,
not from raw unrounded measurements. This can shift a borderline ratio by ~0.01x versus a value
computed from full-precision logs (e.g. 162/155 rounds to 1.05x here even where the underlying
raw measurement was 1.04x) — store more decimal places in current-board.json if a row is close
to the bold_ratio_threshold and this matters. H100 rows instead carry an explicit receipted
"ratio" field (the board publishes the ratio measured from full-precision logs, which can
differ by 0.01x from the rounded-e2e quotient); rows render sorted by that ratio, descending.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOARD_PATH = ROOT / "research" / "tune-data" / "current-board.json"
MODELS_PATH = ROOT / "docs" / "MODELS.md"
PERFORMANCE_PATH = ROOT / "docs" / "PERFORMANCE.md"
MODEL_CARD_PATHS = {
    "Qwen3.5-9B": "qwen35-9b.md",
    "Qwen3.8-27B": "qwen38-27b.md",
    "Qwen3.6-27B": "qwen36-27b.md",
    "Qwen3.6-35B-A3B": "qwen36-35b-a3b.md",
    "Gemma-4 26B-A4B": "gemma4-26b-a4b.md",
    "Gemma-4 31B": "gemma4-31b.md",
    "Gemma-4 E4B": "gemma4-e4b.md",
    "Gemma-4 12B": "gemma4-12b.md",
    "Ornith-1.0-9B": "ornith10-9b.md",
    "Ornith-1.0-35B": "ornith10-35b.md",
    "Ornith-1.5-35B-A3B": "ornith15-35b-a3b.md",
    "Qwen-AgentWorld-35B-A3B": "qwen-agentworld-35b-a3b.md",
    "Step-3.7-Flash 196B-A11B": "step37-flash.md",
}


def load_board():
    return json.loads(BOARD_PATH.read_text())


def fmt_ratio(memra, llama, threshold):
    ratio = memra / llama
    text = f"{ratio:.2f}x"
    return f"**{text}**" if ratio >= threshold else text


def render_date_block(board):
    return (
        f"Measured {board['updated']} on the tracked measuring rig ({board['rig']}, {board['protocol']}) "
        "against llama.cpp built on the same machine, same exact prompts, both engines "
        "re-baselined the same day. The llama.cpp columns are frozen reference points "
        "recorded through 2026-08-03, when head-to-head benching stopped (owner call) — "
        "regression anchors, not a live scoreboard. Boards move with the tuning campaign — "
        "`research/tune-data/rig5090.jsonl` is the running record; the generated boards "
        "in this document are refreshed with every board-moving merge."
    )


def render_plain_table(board):
    threshold = board["bold_ratio_threshold"]
    lines = ["| Model | memra plain | llama.cpp plain | Ratio |", "|---|---|---|---|"]
    for row in board["plain_decode"]["rows"]:
        ratio = fmt_ratio(row["memra"], row["llama"], threshold)
        lines.append(f"| {row['model']} | {row['memra']} | {row['llama']} | {ratio} |")
    return "\n".join(lines)


def render_spec_table(board):
    threshold = board["bold_ratio_threshold"]
    lines = ["| Model | memra spec | llama.cpp spec-best | Ratio |", "|---|---|---|---|"]
    for row in board["speculative"]["rows"]:
        memra_cells = " / ".join(str(v) for v in row["memra"])
        llama_cells = " / ".join(str(v) for v in row["llama"])
        ratios = " / ".join(
            fmt_ratio(b, l, threshold) for b, l in zip(row["memra"], row["llama"])
        )
        lines.append(f"| {row['model']} | {memra_cells} | {llama_cells} | {ratios} |")
    return "\n".join(lines)


def render_models_block(board):
    lines = [
        "| Model | Class | Quant | Drafter | Supported since |",
        "|---|---|---|---|---|",
    ]
    for row in board["supported_models"]:
        lines.append(
            f"| {row['model']} | {row['class']} | {row['quants']} | {row['drafter']} "
            f"| {row['since']} |"
        )
    return "\n".join(lines)


def h100_rows_sorted(board):
    return sorted(board["h100_board"]["rows"], key=lambda r: r["ratio"], reverse=True)


def render_h100_block(board):
    h = board["h100_board"]
    lines = [
        f"Measured {h['updated']} on {h['rig']} against {h['competitor']} — {h['protocol']}",
        "",
        f"| Model | memra e2e | {h['competitor']} e2e (artifact) | Ratio |",
        "|---|---:|---:|---:|",
    ]
    for row in h100_rows_sorted(board):
        ratio = row["ratio"]
        memra = f"**{row['memra_e2e']}**" if ratio > 1.0 else str(row["memra_e2e"])
        vllm = f"**{row['vllm_e2e']}**" if ratio < 1.0 else str(row["vllm_e2e"])
        ratio_txt = f"{ratio:.2f}x"
        if ratio > 1.0:
            ratio_txt = f"**{ratio_txt}**"
        lines.append(
            f"| {row['model']} | {memra} | {vllm} ({row['vllm_artifact']}) | {ratio_txt} |"
        )
    return "\n".join(lines)


def replace_block(text, tag, body, path):
    pattern = re.compile(
        rf"(<!-- {tag}:START[^>]*-->\n).*?(\n<!-- {tag}:END -->)", re.DOTALL
    )
    if not pattern.search(text):
        raise SystemExit(f"marker block {tag} not found in {path.name}")
    return pattern.sub(lambda m: m.group(1) + body + m.group(2), text)


def render_models_doc(board, original):
    return replace_block(text=original, tag="PERF-MODELS", body=render_models_block(board), path=MODELS_PATH)


def render_performance_doc(board, original):
    text = original
    text = replace_block(text, "PERF-DATE", render_date_block(board), PERFORMANCE_PATH)
    text = replace_block(text, "PERF-PLAIN", render_plain_table(board), PERFORMANCE_PATH)
    text = replace_block(text, "PERF-SPEC", render_spec_table(board), PERFORMANCE_PATH)
    text = replace_block(text, "PERF-H100", render_h100_block(board), PERFORMANCE_PATH)
    return text


def main():
    check_only = "--check" in sys.argv
    board = load_board()

    board_models = {row["model"] for row in board["supported_models"]}
    missing_mappings = sorted(board_models - MODEL_CARD_PATHS.keys())
    stale_mappings = sorted(MODEL_CARD_PATHS.keys() - board_models)
    missing_files = sorted(
        model for model, filename in MODEL_CARD_PATHS.items()
        if not (ROOT / "docs" / "models" / filename).is_file()
    )
    if missing_mappings or stale_mappings or missing_files:
        raise SystemExit(
            "model cards are out of sync: "
            f"missing mappings={missing_mappings}, stale mappings={stale_mappings}, "
            f"missing files={missing_files}"
        )

    original_perf = PERFORMANCE_PATH.read_text()
    original_models = MODELS_PATH.read_text()
    new_perf = render_performance_doc(board, original_perf)
    new_models = render_models_doc(board, original_models)

    changed = new_perf != original_perf or new_models != original_models

    if check_only:
        if changed:
            print("perf board is stale — run tools/update-perf-board.py and commit the result")
            sys.exit(1)
        print("perf board is up to date")
        return

    PERFORMANCE_PATH.write_text(new_perf)
    MODELS_PATH.write_text(new_models)
    if changed:
        print("regenerated docs/MODELS.md + docs/PERFORMANCE.md perf tables")
    else:
        print("no changes (already up to date)")


if __name__ == "__main__":
    main()
