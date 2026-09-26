"""Probe pinned independent judge access and order consistency before training."""

import argparse
import hashlib
import json
from pathlib import Path

import judge_bedrock
import prose_packets


QUERY = "Write one short English sentence greeting the reader."
CHECKLIST = [
    "Does the response greet the reader in English?",
    "Is the response a short sentence without unrelated content?",
]
GOOD = "Hello, and welcome!"
BAD = "The number is 42."
TEMPLATE_SHA = "ccd57bd8c4c73f4f83cf8963ef3c2697c1c7b9e907ead91e0d0512cca4ae7a11"


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def preflight(config_path, template_path, credential_file, out):
    config = json.loads(config_path.read_text())
    judge_bedrock.validate_config(config)
    if (
        sha(template_path) != TEMPLATE_SHA
        or TEMPLATE_SHA != judge_bedrock.TEMPLATE_SHA
    ):
        raise ValueError("independent judge template changed")
    price = judge_bedrock.price_quote(config)
    token = judge_bedrock.credential(credential_file)
    template = template_path.read_text()
    out.mkdir(exist_ok=True)
    receipts = []
    for order in (0, 1):
        a, b = (GOOD, BAD) if order == 0 else (BAD, GOOD)
        prompt = prose_packets.render(
            template, QUERY, CHECKLIST, a, b,
        )
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
        packet = {"prompt_sha256": prompt_sha}
        response_path = out / f"response-{order}.json"
        if response_path.exists():
            raw = json.loads(response_path.read_text())
        else:
            raw = judge_bedrock.call(prompt, config, token)
            raw["prompt_sha256"] = prompt_sha
            save(response_path, raw)
        parsed = judge_bedrock.parse_response(
            raw, config, packet, order,
        )
        if parsed["choice"] not in (
            ("A++", "A+") if order == 0 else ("B++", "B+")
        ):
            raise ValueError("independent judge reversed-order pilot disagrees")
        receipts.append({
            **parsed, "order": order,
            "raw_response_sha256": sha(response_path),
        })
    usage = {
        "input_tokens": sum(item["input_tokens"] for item in receipts),
        "output_tokens": sum(item["output_tokens"] for item in receipts),
    }
    actual_usd = (
        usage["input_tokens"]
        * price["global_standard"]["input"]["usd_per_million"]
        + usage["output_tokens"]
        * price["global_standard"]["output"]["usd_per_million"]
    ) / 1_000_000
    result = {
        "schema": 1,
        "status": "independent-judge-template-and-order-qualified",
        "model_id": config["model_id"],
        "config_sha256": sha(config_path),
        "template_sha256": sha(template_path),
        "price_quote": price,
        "usage": usage,
        "quoted_spend_usd": actual_usd,
        "receipts": receipts,
    }
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != result:
            raise ValueError("resumed judge preflight receipt changed")
    else:
        save(manifest_path, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "template", "credential-file", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = preflight(
        args.config.resolve(), args.template.resolve(),
        args.credential_file.resolve(), args.out.resolve(),
    )
    print(json.dumps({
        "status": result["status"],
        "model_id": result["model_id"],
        "quoted_spend_usd": result["quoted_spend_usd"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
