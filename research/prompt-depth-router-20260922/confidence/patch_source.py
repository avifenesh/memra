"""Enable confidence cuts only for a fixed-depth MTP research control."""

import argparse
import hashlib
import json
from pathlib import Path


BASE_SPEC_SHA256 = "d862deb7f919f172ba01277d84e1ad7e1da4614c74d3328430d3ef341a5cf8ca"
OLD = """        if depth_study.is_some()
            && (stream_active || p_min != 0.0 || d_vocab != n_vocab || mtp.d2t.is_some())
        {
            return Err("MTP depth study requires full vocabulary, confidence cuts off, and host round boundaries".into());
        }
"""
NEW = """        // A fixed depth remains a genuine K ceiling when p-min shortens the
        // proposed chain. Contextual depth learning stays separate: its reward
        // assumes an uncensored K action and cannot interpret a confidence cut.
        let fixed_depth_only = depth_study.as_ref().is_some_and(|(policy, _)| {
            matches!(&**policy, MtpDepthExperiment::Learned(learner) if learner.context().is_none())
        });
        if depth_study.is_some()
            && (stream_active
                || (p_min != 0.0 && !fixed_depth_only)
                || d_vocab != n_vocab
                || mtp.d2t.is_some())
        {
            return Err("MTP depth study requires full vocabulary and host round boundaries; confidence cuts require fixed depth".into());
        }
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    path = args.repo / "crates/memra-engine/src/spec.rs"
    before = path.read_bytes()
    if sha(before) != BASE_SPEC_SHA256:
        raise ValueError("research runtime spec.rs differs from the sealed source")
    text = before.decode()
    if text.count(OLD) != 1:
        raise ValueError("fixed-depth guard changed or is ambiguous")
    after = text.replace(OLD, NEW).encode()
    path.write_bytes(after)
    args.receipt.write_text(json.dumps({
        "path": str(path.relative_to(args.repo)),
        "before_sha256": sha(before),
        "after_sha256": sha(after),
        "patcher_sha256": sha(Path(__file__).read_bytes()),
        "scope": "fixed-depth native research driver only; contextual depth stays guarded",
    }, indent=2) + "\n")
    print(f"FIXED_CONFIDENCE_SOURCE {sha(after)}")


if __name__ == "__main__":
    main()
