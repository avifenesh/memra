"""Add request routing to the exact runtime from the preceding engine comparison.

Only the two research drivers and their shared I/O module change. Model math,
sampling, KV reuse, span calibration and fixed-depth control remain the pinned
program. The source identity is checked before extracting or editing anything.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

SOURCE_SHA256 = "943d165b80f268ffacc63e78191c669ed4bda562b3151d45758876321e3f6dc8"
SOURCE_REVISION = "5450580fe2e5eb5c34cd17452f472ed368034f41"
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"source anchor is not unique: {before[:100]!r}")
    return text.replace(before, after)


def prepare(source, out):
    if sha(source) != SOURCE_SHA256:
        raise ValueError("unrecognized native source archive")
    out.mkdir(parents=True, exist_ok=False)
    with tarfile.open(source) as archive:
        seen = set()
        for member in archive.getmembers():
            name = member.name.rstrip("/")
            path = PurePosixPath(name)
            if (
                path.is_absolute() or ".." in path.parts or str(path) != name
                or name in seen or not (member.isfile() or member.isdir())
            ):
                raise ValueError("noncanonical or unsafe source member")
            seen.add(name)
        archive.extractall(out, filter="data")

    common = out / "crates/memra-engine/src/bin/depth_study_io"
    common_module = common / "mod.rs"
    common_module.write_text(
        "pub mod router;\npub mod request_routing;\n" + common_module.read_text()
    )
    for name in ["router.rs", "request_routing.rs"]:
        shutil.copyfile(HERE / name, common / name)

    for family, file, enum, cap, destination, indent in [
        ("qwen", "mtp_depth_study.rs", "MtpDepthExperiment", "cap", "output", "        "),
        ("gemma", "gemma_depth_study.rs", "GemmaDepthExperiment", "K", "out", "            "),
    ]:
        path = common.parent / file
        text = path.read_text()
        factory = (
            "    let initial_experiment = experiment(arm, cap, &tok, model.output.out_features(), priors)?;"
            if family == "qwen" else
            "    let initial_policy = policy(arm, &tok, model.output.out_features(), priors)?;"
        )
        initial_name = "initial_experiment" if family == "qwen" else "initial_policy"
        call = (
            "experiment(&initial_arm, cap, &tok, model.output.out_features(), priors)?"
            if family == "qwen" else
            "policy(&initial_arm, &tok, model.output.out_features(), priors)?"
        )
        text = replace_once(text, factory, f"""    let routing = depth_study_io::request_routing::RequestRouting::parse(arm, {cap} as u8)?;
    let initial_arm = routing.map(|r| format!("fixed:{{}}", r.initial_k()))
        .unwrap_or_else(|| arm.to_owned());
    let {initial_name} = {call};
    let mut routing_log = std::fs::File::create_new({destination}.join("routing.tsv"))?;
    writeln!(routing_log, "turn\\tsource\\tkind\\tk\\trouting_ns")?;""")
        marker = indent + "let resumed = cached_tokens > 0;"
        text = replace_once(text, marker, marker + f"""
{indent}let request_selection = routing.map(|r| r.select(turn, user, {cap} as u8)).transpose()?;""")
        fixed = (
            f"{enum}::Learned(memra_engine::learned_depth::LearnedDepth::fixed("
            "usize::from(selection.decision.k) + 1)?)"
        )
        if family == "qwen":
            marker = "        if let Some(MtpDepthExperiment::Learned(policy)) = &mut sess.depth_experiment {"
            text = replace_once(text, marker, f"""        if let Some(selection) = request_selection {{
            sess.depth_experiment = Some({fixed});
        }}
""" + marker)
        else:
            marker = "            if let Some(mut p) = carried.clone() {"
            text = replace_once(text, marker, f"""            let next_policy = if let Some(selection) = request_selection {{
                Some({fixed})
            }} else {{
                carried.clone()
            }};
            if let Some(mut p) = next_policy {{""")
        marker = indent + f'std::fs::write({destination}.join(format!("turn-{{turn}}.user.txt")), user)?;'
        text = replace_once(
            text, marker, marker + "\n" + indent
            + "depth_study_io::request_routing::write_record(&mut routing_log, turn, request_selection)?;"
        )
        path.write_text(text)

    # Keep the proven serialization, artifact checks, GPU contention monitor,
    # token-count reconstruction and KV audit in the existing study runner.
    runner = out / "research/mtp-context-depth-20260921/run_study.py"
    text = runner.read_text()
    text = replace_once(text, "    a = p.parse_args()", """    p.add_argument('--request-depths', type=Path, help='Frozen prompt/schedule arm map')
    p.add_argument('--runtime-binding', type=Path, required=True)
    a = p.parse_args()""")
    text = replace_once(text, "    if a.schedule:\n", """    request_depths = json.loads(a.request_depths.read_text()) if a.request_depths else {}
    if set(request_depths) & (reserved | set(fixed_depths)):
        raise ValueError('Request-depth labels collide with controls')
    for key, value in request_depths.items():
        if not re.fullmatch(r'[a-z][a-z0-9_]*', key) or not isinstance(value, str):
            raise ValueError('Invalid request-depth label')
        if not re.fullmatch(r'(prompt:(?:[0-9]+,){3}[0-9]+|schedule:(?:[0-9]+,){7}[0-9]+)', value):
            raise ValueError('Invalid request-depth policy')
        if any(not 1 <= int(k) <= maximum for k in value.split(':', 1)[1].split(',')):
            raise ValueError('Request depth exceeds native cap')
    binding = json.loads(a.runtime_binding.read_text())
    if binding['parent_runtime_archive_sha256'] != '943d165b80f268ffacc63e78191c669ed4bda562b3151d45758876321e3f6dc8':
        raise ValueError('Unrecognized native runtime parent')
    for name, expected in binding['files'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or digest(a.runtime_binding.parent / path) != expected:
            raise ValueError('Prepared native source binding changed')
    if a.schedule:
""")
    text = replace_once(text, "if not set(order) <= reserved | set(fixed_depths):",
                        "if not set(order) <= reserved | set(fixed_depths) | set(request_depths):")
    text = replace_once(text, "        'source_commit': a.source_commit,",
                        """        'source_recipe_commit': a.source_commit,
        'runtime_binding_sha256': digest(a.runtime_binding),
        'runtime_binding': binding,
        'request_depths': request_depths,""")
    text = replace_once(text,
                        "            runtime_arm = f'fixed:{fixed_depths[arm]}' if arm in fixed_depths else arm",
                        "            runtime_arm = request_depths.get(arm, f'fixed:{fixed_depths[arm]}' if arm in fixed_depths else arm)")
    text = replace_once(text, "'calibration_only': True", "'rates_only': True")
    runner.write_text(text)

    touched = [
        "crates/memra-engine/src/bin/mtp_depth_study.rs",
        "crates/memra-engine/src/bin/gemma_depth_study.rs",
        "crates/memra-engine/src/bin/depth_study_io/mod.rs",
        "crates/memra-engine/src/bin/depth_study_io/router.rs",
        "crates/memra-engine/src/bin/depth_study_io/request_routing.rs",
        "research/mtp-context-depth-20260921/run_study.py",
    ]
    receipt = {
        "parent_runtime_archive_sha256": SOURCE_SHA256,
        "parent_runtime_revision": SOURCE_REVISION,
        "preparer_sha256": sha(Path(__file__)),
        "files": {name: sha(out / name) for name in touched},
        "scope": "research request drivers, routing and orchestration; numerical engine remains the pinned parent",
    }
    (out / "REQUEST-ROUTING-SOURCE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_archive, args.out), indent=2))


if __name__ == "__main__":
    main()
