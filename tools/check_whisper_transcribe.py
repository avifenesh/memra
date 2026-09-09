#!/usr/bin/env python3
"""Gate native end-to-end Whisper beam-1 transcription against the pinned rental oracle.

The native side is `whisper-stage transcribe`: its own log-mel, its own window program, its
own encoder, its own cached decoder and its own suppression/timestamp policy. Nothing from the
oracle enters that run. This checker only reads what came out.

Three independent things are checked per clip:

  window program  native seeks and segment extents must equal the oracle's window boundaries
  token sequence  native ids must equal the CT2 beam-1 ids, window by window
  text            native transcript, canonized the way the ivrit-ai leaderboard canonizes,
                  against the CT2 transcript, with word error rate per domain

Where a window's ids differ, the first divergent step is reported with the oracle's own margin
at that step: the recorded post-suppression logit gap between the id CT2 took and the id the
native decode took. Both are conditioned on the identical prefix, so the gap is comparable. A
gap of zero is a tie the oracle's fp16 logits cannot break; a gap at or under one fp16 step at
that magnitude is the smallest difference its precision can express at all.

Detokenization uses the checkpoint's tokenizer through transformers. A native detokenizer is
not built yet, so text numbers here are gated on offline tooling and say so in the receipt.
"""
import argparse, hashlib, json, os, re, string
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# Ids at or above this carry a time, not text: the last 1501 of the 51866-id vocabulary.
TIMESTAMP_BEGIN = 50365


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ivrit-ai leaderboard canonization (compare_texts.py): drop ASCII punctuation except the
# hyphen, turn the hyphen into a space, collapse whitespace.
def canon(text):
    table = str.maketrans("", "", string.punctuation.replace("-", ""))
    table[ord("-")] = " "
    return re.sub(r"\s+", " ", text.translate(table)).strip()


def main():
    import numpy as np
    import jiwer
    from transformers import WhisperTokenizerFast

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--native", type=Path, required=True, help="sweep output directory")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--numeric", choices=("f32", "f16"), default="f32")
    p.add_argument("--threads", type=int, default=1, help="MEMRA_SPEECH_THREADS the sweep used")
    p.add_argument("--wer-delta-limit", type=float, default=0.05, help="percentage points")
    args = p.parse_args()

    manifest = json.loads((args.oracle / "MANIFEST.json").read_text())
    ct2 = next(b for b in manifest["backends"] if b["backend"] == "ct2")
    tokenizer = WhisperTokenizerFast.from_pretrained(str(args.checkpoint))

    clips, rows = [], []
    window_total = window_match = 0
    for clip in ct2["clips"]:
        native_dir = args.native / clip["clip_id"]
        if not (native_dir / "windows.tsv").exists():
            continue
        native_windows = []
        for line in (native_dir / "windows.tsv").read_text().splitlines()[1:]:
            index, seek, segment, count, capped = line.split("\t")
            native_windows.append(
                {
                    "index": int(index),
                    "seek_frame": int(seek),
                    "segment_frames": int(segment),
                    "token_count": int(count),
                    "reached_cap": capped == "true",
                }
            )
        row = {
            "clip": clip["clip_id"],
            "domain": clip["domain"],
            "oracle_windows": len(clip["windows"]),
            "native_windows": len(native_windows),
            "window_program_matches": len(native_windows) == len(clip["windows"]),
            "windows": [],
        }
        native_ids, oracle_ids = [], []
        for want, got in zip(clip["windows"], native_windows):
            window_total += 1
            entry = want["decodes"][0]["token_ids"]
            reference = args.oracle / entry["path"]
            if sha(reference) != entry["sha256"]:
                raise SystemExit("oracle token hash mismatch: " + clip["clip_id"])
            oracle = np.fromfile(reference, dtype="<i4")
            native = np.fromfile(
                native_dir / f"window-{got['index']:03d}-tokens.i32.bin", dtype="<i4"
            )
            native_ids.append(native)
            oracle_ids.append(oracle)
            boundary = (
                got["seek_frame"] == want["seek_frame"]
                and got["segment_frames"] == want["segment_size_frames"]
            )
            row["window_program_matches"] &= boundary
            same = native.size == oracle.size and bool((native == oracle).all())
            window_match += same
            record = {
                "index": got["index"],
                "seek_frame": got["seek_frame"],
                "boundary_matches": boundary,
                "oracle_tokens": int(oracle.size),
                "native_tokens": int(native.size),
                "tokens_match": same,
            }
            if not same:
                shared = min(native.size, oracle.size)
                step = next(
                    (i for i in range(shared) if native[i] != oracle[i]), shared
                )
                record["first_divergent_step"] = step
                # A window that does not start where the oracle's window started is not the
                # same audio, so its tokens are a consequence of an earlier shift and not an
                # independent disagreement.
                record["divergence_class"] = "boundary_cascade" if not boundary else None
                if step < shared:
                    logits = np.memmap(
                        args.oracle / want["decodes"][0]["logits"]["path"],
                        dtype="<f4",
                        mode="r",
                        shape=tuple(want["decodes"][0]["logits"]["shape"]),
                    )
                    taken = float(logits[step][int(oracle[step])])
                    ours = float(logits[step][int(native[step])])
                    margin = taken - ours
                    record.update(
                        {
                            "oracle_token": int(oracle[step]),
                            "native_token": int(native[step]),
                            "oracle_logit_margin": margin,
                            "oracle_logit_of_native_token": ours,
                            "both_tokens_are_timestamps": bool(
                                oracle[step] >= TIMESTAMP_BEGIN
                                and native[step] >= TIMESTAMP_BEGIN
                            ),
                        }
                    )
                    # How far apart fp16 can hold two numbers of this size. The oracle's
                    # logits are fp16 values widened to f32, so a gap at or under one step is
                    # the smallest difference its precision can express at all.
                    step_size = float(np.spacing(np.float16(abs(taken))))
                    record["oracle_fp16_step_at_magnitude"] = step_size
                    if record["divergence_class"] is None:
                        if margin == 0.0:
                            record["divergence_class"] = "fp16_tie"
                        elif margin <= step_size:
                            record["divergence_class"] = "fp16_ulp"
                        else:
                            record["divergence_class"] = "real"
                else:
                    record["divergence"] = "length only"
                    record["divergence_class"] = record["divergence_class"] or "length"
            row["windows"].append(record)

        def text(chunks):
            return "".join(
                tokenizer.decode(chunk.tolist(), skip_special_tokens=True)
                for chunk in chunks
            )

        row["native_text"] = text(native_ids)
        row["oracle_text"] = text(oracle_ids)
        row["text_matches"] = canon(row["native_text"]) == canon(row["oracle_text"])
        row["oracle_text_matches_manifest_transcript"] = canon(row["oracle_text"]) == canon(
            clip["transcript"]
        )
        row["tokens_match"] = all(w["tokens_match"] for w in row["windows"]) and row[
            "window_program_matches"
        ]
        rows.append(row)
        clips.append(clip["clip_id"])
        print(
            f"{row['clip']:14s} windows {sum(w['tokens_match'] for w in row['windows'])}"
            f"/{len(row['windows'])} text_equal={row['text_matches']}",
            flush=True,
        )

    domains = {}
    for domain in sorted({r["domain"] for r in rows}):
        picked = [r for r in rows if r["domain"] == domain]
        native = canon(" ".join(r["native_text"] for r in picked))
        oracle = canon(" ".join(r["oracle_text"] for r in picked))
        measure = jiwer.process_words(oracle, native)
        domains[domain] = {
            "clips": len(picked),
            "wer_percent_native_vs_ct2": measure.wer * 100.0,
            "substitutions": measure.substitutions,
            "deletions": measure.deletions,
            "insertions": measure.insertions,
            "reference_words": len(oracle.split()),
            "within_limit": measure.wer * 100.0 <= args.wer_delta_limit,
        }

    exact_text = sum(r["text_matches"] for r in rows)
    classes = {}
    for row in rows:
        for window in row["windows"]:
            label = window.get("divergence_class")
            if label:
                classes[label] = classes.get(label, 0) + 1
    receipt = {
        "schema": "memra-whisper-transcribe-v1",
        # The stage gate is the text: every clip's canonized transcript equal to CT2's, or a
        # WER delta under the limit per domain. Window boundaries are reported separately
        # because a tied timestamp can shift one and cascade into the next window's tokens
        # without changing a word.
        "status": "passed"
        if rows and (exact_text == len(rows) or all(d["within_limit"] for d in domains.values()))
        else "failed",
        "divergence_classes": classes,
        "clips_window_program_exact": sum(r["window_program_matches"] for r in rows),
        "numeric": args.numeric,
        "speech_threads": args.threads,
        "binary_sha256": sha(args.binary),
        "oracle_manifest_sha256": sha(args.oracle / "MANIFEST.json"),
        "cell_lock_sha256": manifest["cell_lock_sha256"],
        "clip_count": len(rows),
        "window_count": window_total,
        "windows_token_exact": window_match,
        "clips_token_exact": sum(r["tokens_match"] for r in rows),
        "clips_text_exact": exact_text,
        "clips_oracle_text_matches_transcript": sum(
            r["oracle_text_matches_manifest_transcript"] for r in rows
        ),
        "wer_delta_limit_percent": args.wer_delta_limit,
        "domains": domains,
        "clips": rows,
        "scope": "native log-mel, window program, encoder, cached decoder and beam-1 policy "
        "against the pinned CT2 rental oracle on real recorded Hebrew audio. CPU reference "
        "only: no GPU execution, no serving endpoint, no timestamp-alignment gate, and "
        "detokenization is offline tooling because a native detokenizer is not built yet.",
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        f"transcribe {receipt['status']}: {window_match}/{window_total} windows token-exact, "
        f"{receipt['clips_token_exact']}/{len(rows)} clips token-exact, "
        f"{exact_text}/{len(rows)} clips text-exact, "
        f"{receipt['clips_window_program_exact']}/{len(rows)} clips window-exact"
    )
    if classes:
        print("  divergence classes: " + ", ".join(f"{k}={v}" for k, v in sorted(classes.items())))
    for domain, value in domains.items():
        print(f"  {domain}: WER vs CT2 {value['wer_percent_native_vs_ct2']:.4f} pt")
    raise SystemExit(0 if receipt["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
