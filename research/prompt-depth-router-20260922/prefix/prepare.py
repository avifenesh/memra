"""Prepare two native, independent-request prompt-prefix research drivers."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from prepare_native import prepare, replace_once  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source, out):
    parent = prepare(source, out)
    bins = out / "crates/memra-engine/src/bin"
    common = bins / "prefix_study_io"
    shutil.copytree(bins / "depth_study_io", common)
    module = common / "mod.rs"
    text = module.read_text().replace("pub mod request_routing;\n", "")
    text = "pub mod prefix_policy;\npub mod native_prefix;\n" + text
    module.write_text(text)
    (common / "request_routing.rs").unlink()
    shutil.copyfile(HERE / "prefix_policy.rs", common / "prefix_policy.rs")
    shutil.copyfile(HERE / "native_io.rs", common / "native_prefix.rs")
    touched = [str(p.relative_to(out)) for p in common.iterdir() if p.is_file()]
    for family, original, destination, cap, outvar, indent in [
        ("qwen", "mtp_depth_study.rs", "qwen_prefix_study.rs", "cap", "output", "        "),
        ("gemma", "gemma_depth_study.rs", "gemma_prefix_study.rs", "K", "out", "            "),
    ]:
        text = (bins / original).read_text().replace("depth_study_io", "prefix_study_io")
        text = replace_once(
            text, "    // CPU-only preparation: count the actual checkpoint tokenizer and template.",
            """    if args.len() == 3 && args[1] == "count-server" {
        let gguf = GgufFile::open(&args[2])?;
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        return prefix_study_io::native_prefix::count_server(&tokenizer);
    }
    // CPU-only preparation: count the actual checkpoint tokenizer and template.""")
        text = replace_once(
            text,
            '    let (gate, cold_reference, priors) = prefix_study_io::extra_args(&args)?;\n'
            '    if cold_reference && temperature != 0.0 {\n'
            '        return Err("cold reference is a greedy correctness diagnostic only".into());\n'
            '    }',
            '    let (gate, _, priors) = prefix_study_io::extra_args(&args)?;\n'
            '    // Each matrix entry is an independent request, with a fresh native cache.\n'
            '    let cold_reference = true;')
        text = replace_once(
            text,
            f"    let routing = prefix_study_io::request_routing::RequestRouting::parse(arm, {cap} as u8)?;",
            "    let routing = prefix_study_io::native_prefix::Routing::parse(arm)?;\n"
            "    let prefix_shape = prefix_study_io::native_prefix::TemplateShape::new(&tok)?;")
        text = replace_once(
            text,
            '    writeln!(routing_log, "turn\\tsource\\tkind\\tk\\trouting_ns")?;',
            '    writeln!(routing_log, "turn\\tsource\\tbudget\\tkind\\tk\\ttokens_read\\tuser_tokens\\tinput_tokens\\tdecoded_bytes\\tinspected_bytes\\theader_tokens\\tcore_ns\\trouting_ns")?;')
        text = replace_once(
            text, indent + 'history.push(Turn {\n' + indent + '    role: "user".into(),',
            indent + "history.clear();\n" + indent + 'history.push(Turn {\n'
            + indent + '    role: "user".into(),')
        text = replace_once(
            text,
            indent + f"let request_selection = routing.map(|r| r.select(turn, user, {cap} as u8)).transpose()?;",
            indent + "let request_selection = routing.map(|r| r.select(turn, &tok, &prompt, &prefix_shape)).transpose()?;")
        text = text.replace("if let Some(selection) = request_selection", "if let Some(selection) = request_selection.as_ref()")
        text = text.replace("selection.decision.k", "selection.k")
        text = text.replace("policy.begin_request(user);", 'policy.begin_request("");')
        if family == "qwen":
            text = replace_once(
                text,
                "    while warm_start.elapsed() < Duration::from_secs(10) {\n"
                "        let mut warm = model.new_session(&e, warm_prompt.len() + 256)?;",
                "    let mut warm_iteration = 0usize;\n"
                "    while warm_start.elapsed() < Duration::from_secs(10) || warm_iteration < 3 {\n"
                "        let warm_k = 2 + warm_iteration % 3;\n"
                "        warm_iteration += 1;\n"
                "        let mut warm = model.new_session(&e, warm_prompt.len() + 256)?;\n"
                "        warm.depth_experiment = Some(MtpDepthExperiment::Learned(\n"
                "            memra_engine::learned_depth::LearnedDepth::fixed(warm_k + 1)?));")
        else:
            text = replace_once(
                text,
                "    while warm_started.elapsed() < Duration::from_secs(10) {",
                "    let mut warm_iteration = 0usize;\n"
                "    while warm_started.elapsed() < Duration::from_secs(10) || warm_iteration < 3 {\n"
                "        let warm_k = 2 + warm_iteration % 3;\n"
                "        warm_iteration += 1;")
            text = replace_once(
                text,
                "        if temperature == 0.0 {\n"
                "            model.gemma_spec_session_burst(&e, &mut draft, &mut s, 32, K, &eos)?;",
                "        s.set_depth_experiment(GemmaDepthExperiment::Learned(\n"
                "            memra_engine::learned_depth::LearnedDepth::fixed(warm_k + 1)?))?;\n"
                "        if temperature == 0.0 {\n"
                "            model.gemma_spec_session_burst(&e, &mut draft, &mut s, 32, K, &eos)?;")
        text = replace_once(
            text,
            indent + "prefix_study_io::request_routing::write_record(&mut routing_log, turn, request_selection)?;",
            indent + "prefix_study_io::native_prefix::Recorder {\n"
            + indent + f"    log: &mut routing_log, out: {outvar},\n"
            + indent + "}.write(\n"
            + indent + "    turn, request_selection.as_ref(), &tok, &prompt, &prefix_shape,\n"
            + indent + '    arm.strip_prefix("fixed:").and_then(|k| k.parse().ok()).unwrap_or(3),\n'
            + indent + ")?;")
        path = bins / destination
        path.write_text(text)
        touched.append(str(path.relative_to(out)))
    cargo = out / "crates/memra-engine/Cargo.toml"
    with cargo.open("a") as stream:
        for family in ("qwen", "gemma"):
            stream.write(f'\n[[bin]]\nname = "{family}-prefix-study"\npath = "src/bin/{family}_prefix_study.rs"\n')
    touched.append(str(cargo.relative_to(out)))
    receipt = {
        "parent": parent,
        "preparer_sha256": sha(Path(__file__)),
        "files": {name: sha(out / name) for name in sorted(touched)},
        "scope": "independent native requests; one prefix forecast per request; frozen model math and heads",
    }
    (out / "PREFIX-ROUTING-SOURCE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source_archive, args.out), indent=2))
