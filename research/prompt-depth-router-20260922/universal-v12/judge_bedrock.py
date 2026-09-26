"""Run frozen WildBench-style packets through an independent Bedrock judge."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import urllib.parse
import urllib.request


CHOICES = {"A++", "A+", "A=B", "B+", "B++"}
FIELDS = {
    "analysis of A", "analysis of B",
    "reason of A=B", "reason of A>B", "reason of B>A",
    "choice",
}
TEMPLATE_SHA = "ccd57bd8c4c73f4f83cf8963ef3c2697c1c7b9e907ead91e0d0512cca4ae7a11"
PRICING_URL = (
    "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/"
    "AmazonBedrockFoundationModels/current/us-east-1/index.json"
)


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


def jsonl(path):
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip()
    ]


def credential(path):
    values = dict(
        line.split("=", 1) for line in path.read_text().splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    )
    value = values.get("AWS_BEARER_TOKEN_BEDROCK")
    if not value:
        raise ValueError("explicit Bedrock credential file is unavailable")
    return value


def quoted_usd(price, input_tokens, output_tokens):
    return (
        input_tokens
        * price["global_standard"]["input"]["usd_per_million"]
        + output_tokens
        * price["global_standard"]["output"]["usd_per_million"]
    ) / 1_000_000


def validate_config(config):
    if (
        config["schema"] != 1
        or config["provider"] != "amazon-bedrock-converse"
        or config["model_id"] != "global.anthropic.claude-sonnet-5"
        or config["region"] != "us-east-1"
        or config["template_sha256"] != TEMPLATE_SHA
        or config["spend_basis"] != "live_global_standard_quote"
        or config["max_output_tokens"] != 4096
        or config["max_prompt_bytes"] != 120000
        or config["total_usd_cap"] <= 0
        or config["input_usd_per_million_budget"] <= 0
        or config["output_usd_per_million_budget"] <= 0
    ):
        raise ValueError("independent prose judge configuration differs")


def price_quote(config):
    with urllib.request.urlopen(PRICING_URL, timeout=20) as response:
        index = json.load(response)
    prices = {}
    for kind in ("input", "output"):
        matching = []
        suffix = f"_{kind}_tokens_global_standard-Units"
        for sku, product in index["products"].items():
            attributes = product.get("attributes", {})
            if (
                attributes.get("servicename")
                != "Claude Sonnet 5 (Amazon Bedrock Edition)"
                or attributes.get("regionCode") != config["region"]
                or not attributes.get("usagetype", "").endswith(suffix)
            ):
                continue
            for term in index["terms"]["OnDemand"][sku].values():
                for dimension in term["priceDimensions"].values():
                    if dimension["unit"] == "1M tokens":
                        matching.append((
                            sku, float(dimension["pricePerUnit"]["USD"]),
                        ))
        if (
            len(matching) != 1
            or not math.isfinite(matching[0][1])
            or matching[0][1] <= 0
            or matching[0][1] > config[
                f"{kind}_usd_per_million_budget"
            ]
        ):
            raise ValueError("live Bedrock judge price exceeds budget rate")
        prices[kind] = {
            "sku": matching[0][0],
            "usd_per_million": matching[0][1],
        }
    return {
        "schema": 1,
        "source": PRICING_URL,
        "publication_date": index["publicationDate"],
        "model_id": config["model_id"],
        "region": config["region"],
        "global_standard": prices,
    }


def parse_response(raw, config, packet, index):
    if raw["prompt_sha256"] != packet["prompt_sha256"]:
        raise ValueError("Bedrock judge response belongs to another packet")
    body = raw["body"]
    blocks = body.get("output", {}).get("message", {}).get("content", [])
    text = "".join(
        item.get("text", "") for item in blocks
        if isinstance(item, dict)
    )
    if body.get("stopReason") != "end_turn":
        raise ValueError("Bedrock judge did not finish")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("Bedrock judge did not return strict JSON") from error
    if (
        not isinstance(parsed, dict)
        or set(parsed) != FIELDS
        or parsed["choice"] not in CHOICES
        or any(
            not isinstance(parsed[key], str)
            for key in FIELDS
        )
    ):
        raise ValueError("Bedrock judge output differs from frozen template")
    usage = body.get("usage", {})
    input_tokens = usage.get("inputTokens")
    output_tokens = usage.get("outputTokens")
    if (
        not isinstance(input_tokens, int)
        or not isinstance(output_tokens, int)
        or input_tokens <= 0
        or output_tokens <= 0
        or not raw.get("request_id")
    ):
        raise ValueError("Bedrock judge usage or request ID is missing")
    return {
        "packet_index": index,
        "prompt_sha256": packet["prompt_sha256"],
        "model_id": config["model_id"],
        "provider_response_id": raw["request_id"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "choice": parsed["choice"],
    }


def call(prompt, config, token):
    model = urllib.parse.quote(config["model_id"], safe="")
    url = (
        f"https://bedrock-runtime.{config['region']}"
        f".amazonaws.com/model/{model}/converse"
    )
    body = {
        "system": [{
            "text": (
                "Evaluate the two responses using the supplied checklist. "
                "Return only strict JSON matching the requested fields, "
                "without a code fence or trailing comma."
            ),
        }],
        "messages": [{
            "role": "user",
            "content": [{"text": prompt}],
        }],
        "inferenceConfig": {
            "maxTokens": config["max_output_tokens"],
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return {
            "request_id": response.headers.get("x-amzn-requestid"),
            "body": json.load(response),
        }


def run(packets_dir, config_path, credential_file, out,
        prior_manifest_path=None):
    config = json.loads(config_path.read_text())
    validate_config(config)
    manifest = json.loads((packets_dir / "manifest.json").read_text())
    packet_path = packets_dir / "packets.jsonl"
    if (
        manifest["schema"] != 1
        or manifest["template_sha256"] != TEMPLATE_SHA
        or sha(packet_path) != manifest["packets_sha256"]
    ):
        raise ValueError("independent judge packet source differs")
    packets = jsonl(packet_path)
    if len(packets) != manifest["packet_count"]:
        raise ValueError("independent judge packet count differs")
    live_price = price_quote(config)
    prior_usage = {"input_tokens": 0, "output_tokens": 0}
    prior_spend = 0.0
    prior_sha = None
    if manifest["phase"] == "final":
        if prior_manifest_path is None:
            raise ValueError("final judge lacks validation budget receipt")
        prior = json.loads(prior_manifest_path.read_text())
        if (
            prior["status"] != "complete"
            or prior["config_sha256"] != sha(config_path)
            or prior["model_id"] != config["model_id"]
            or prior["region"] != config["region"]
        ):
            raise ValueError("final judge budget lineage differs")
        prior_usage = prior["usage"]
        prior_spend = prior["cumulative_quoted_spend_usd"]
        if (
            not isinstance(prior_spend, (int, float))
            or not math.isfinite(prior_spend)
            or prior_spend < 0
            or prior_spend > config["total_usd_cap"]
        ):
            raise ValueError("validation judge spend receipt differs")
        prior_sha = sha(prior_manifest_path)
    elif manifest["phase"] != "validation" or (
        prior_manifest_path is not None
    ):
        raise ValueError("validation judge budget phase differs")
    out.mkdir(exist_ok=True)
    pricing_path = out / "pricing.json"
    if pricing_path.exists():
        pinned_price = json.loads(pricing_path.read_text())
        if pinned_price["global_standard"] != live_price["global_standard"]:
            raise ValueError("Bedrock judge price changed during this phase")
    else:
        save(pricing_path, live_price)
    results_path = out / "results.jsonl"
    completed = jsonl(results_path)
    if len(completed) > len(packets):
        raise ValueError("independent judge has excess results")
    responses = out / "responses"
    responses.mkdir(exist_ok=True)
    for index, item in enumerate(completed):
        response_path = responses / f"{index:05d}.json"
        if (
            item["packet_index"] != index
            or item["prompt_sha256"] != packets[index]["prompt_sha256"]
            or item["model_id"] != config["model_id"]
            or item["choice"] not in CHOICES
            or sha(response_path)
            != item["raw_response_sha256"]
        ):
            raise ValueError("resumed Bedrock judgment differs")
        observed = parse_response(
            json.loads(response_path.read_text()),
            config, packets[index], index,
        )
        if any(item[key] != value for key, value in observed.items()):
            raise ValueError("resumed Bedrock choice changed from provider body")
    if (out / "manifest.json").exists():
        prior = json.loads((out / "manifest.json").read_text())
        if (
            prior["status"] != "complete"
            or prior["packets_sha256"] != sha(packet_path)
            or prior["config_sha256"] != sha(config_path)
            or prior["results_sha256"] != sha(results_path)
            or prior["prior_judge_manifest_sha256"] != prior_sha
            or prior["pricing_sha256"] != sha(pricing_path)
            or len(completed) != len(packets)
        ):
            raise ValueError("completed Bedrock judgment changed")
        return
    input_tokens = sum(row["input_tokens"] for row in completed)
    output_tokens = sum(row["output_tokens"] for row in completed)
    token = credential(credential_file)
    for index in range(len(completed), len(packets)):
        if index and index % 100 == 0:
            refreshed = price_quote(config)
            if (
                refreshed["global_standard"]
                != live_price["global_standard"]
            ):
                raise ValueError(
                    "live Bedrock judge price changed during batch"
                )
        packet = packets[index]
        prompt = packet["prompt"]
        if hashlib.sha256(prompt.encode()).hexdigest() != (
            packet["prompt_sha256"]
        ):
            raise ValueError("independent judge packet text changed")
        byte_count = len(prompt.encode())
        if byte_count > config["max_prompt_bytes"]:
            raise ValueError("independent judge prompt exceeds byte limit")
        if prior_spend + quoted_usd(
            live_price,
            input_tokens + byte_count,
            output_tokens + config["max_output_tokens"],
        ) > config["total_usd_cap"]:
            raise ValueError("Bedrock judgment budget ceiling reached")
        response_path = responses / f"{index:05d}.json"
        if response_path.exists():
            raw = json.loads(response_path.read_text())
        else:
            raw = call(prompt, config, token)
            raw["prompt_sha256"] = packet["prompt_sha256"]
            save(response_path, raw)
        row = parse_response(raw, config, packet, index)
        row["raw_response_sha256"] = sha(response_path)
        with results_path.open("a") as output:
            output.write(json.dumps(row, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        input_tokens += row["input_tokens"]
        output_tokens += row["output_tokens"]
        print(json.dumps({
            "judged": index + 1,
            "total": len(packets),
            "cumulative_quoted_spend_usd": prior_spend + quoted_usd(
                live_price,
                input_tokens, output_tokens,
            ),
        }), flush=True)
    save(out / "manifest.json", {
        "schema": 1,
        "status": "complete",
        "packets_sha256": sha(packet_path),
        "config_sha256": sha(config_path),
        "results_sha256": sha(results_path),
        "model_id": config["model_id"],
        "region": config["region"],
        "pricing_sha256": sha(pricing_path),
        "pricing": live_price["global_standard"],
        "prior_judge_manifest_sha256": prior_sha,
        "requests": len(packets),
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "cumulative_usage": {
            key: prior_usage[key] + value
            for key, value in {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }.items()
        },
        "cumulative_quoted_spend_usd": prior_spend + quoted_usd(
            live_price,
            input_tokens, output_tokens,
        ),
    })


def main():
    parser = argparse.ArgumentParser()
    for name in ("packets", "config", "credential-file", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prior-manifest", type=Path)
    args = parser.parse_args()
    run(
        args.packets.resolve(), args.config.resolve(),
        args.credential_file.resolve(), args.out.resolve(),
        args.prior_manifest.resolve() if args.prior_manifest else None,
    )


if __name__ == "__main__":
    main()
