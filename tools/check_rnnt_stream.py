#!/usr/bin/env python3
"""Gate the native streaming session against the pinned NeMo session reference.

Not a stage gate: this is audio in, tokens out, with the encoder caches and the predictor
hypothesis carried across 26 chunks. It compares the running token sequence after *every*
chunk, so a session that reaches the right final answer by a wrong path fails, and it compares
the transcript the engine's own vocabulary produced against the reference's.
"""
import argparse, hashlib, json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    args = p.parse_args()

    manifest = json.loads((args.oracle / "MANIFEST.json").read_text())
    rows = []
    passed = True
    for chunk in manifest["chunks"]:
        index = chunk["index"]
        path = args.native / f"chunk-{index:03d}-tokens.txt"
        if not path.exists():
            raise SystemExit(f"native session has no chunk {index}")
        native = [int(t) for t in path.read_text().split()]
        want = [int(t) for t in chunk["tokens_so_far"]]
        same = native == want
        passed &= same
        rows.append(
            {
                "index": index,
                "reference_tokens": len(want),
                "native_tokens": len(native),
                "partial_matches": same,
            }
        )
        if not same:
            first = next(
                (
                    i
                    for i in range(min(len(native), len(want)))
                    if native[i] != want[i]
                ),
                min(len(native), len(want)),
            )
            rows[-1]["first_divergent_index"] = first
            print(f"chunk {index:3d} DIFFERS at token {first}", flush=True)

    engine_transcript = None
    transcript_matches = None
    path = args.native / "transcript.txt"
    if path.exists():
        engine_transcript = path.read_text()
        transcript_matches = engine_transcript == manifest["final_transcript"]
        passed &= transcript_matches
        print(
            f"engine transcript {'equal' if transcript_matches else 'DIFFERS'}: "
            f"{engine_transcript!r} against {manifest['final_transcript']!r}"
        )

    native_final = [int(t) for t in (args.native / "tokens.txt").read_text().split()]
    reference_final = [int(t) for t in manifest["final_tokens"]]
    final_equal = native_final == reference_final
    passed &= final_equal

    receipt = {
        "schema": "memra-rnnt-stream-v1",
        "status": "passed" if passed else "failed",
        "chunks": len(rows),
        "chunks_partial_exact": sum(r["partial_matches"] for r in rows),
        "final_tokens_equal": final_equal,
        "final_token_count": len(reference_final),
        "final_tokens": reference_final,
        "final_transcript": manifest["final_transcript"],
        "engine_transcript": engine_transcript,
        "engine_transcript_matches": transcript_matches,
        "language": manifest["language"],
        "att_context_size": manifest["att_context_size"],
        "samples": manifest["samples"],
        "binary_sha256": sha(args.binary),
        "archive_sha256": manifest["archive_sha256"],
        "oracle_manifest_sha256": sha(args.oracle / "MANIFEST.json"),
        "reference": {"torch": manifest["torch"], "nemo": manifest["nemo"]},
        "rows": rows,
        "scope": "native streaming session, audio to token ids over the [56,0] arm, against a "
        "pinned CPU FP32 NeMo session. Partial output is compared at every chunk. No GPU, no "
        "endpoint, no admission or concurrency behaviour.",
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(
        f"rnnt stream {receipt['status']}: {receipt['chunks_partial_exact']}/{len(rows)} "
        f"chunk partials exact, final {'equal' if final_equal else 'DIFFERS'}, "
        f"{len(reference_final)} tokens"
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
