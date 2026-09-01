#!/usr/bin/env python3
"""Live prepaid-budget/admin smoke without persisting either bearer secret."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import tomllib
import urllib.error
import urllib.request
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def request_json(
    url: str,
    *,
    body: dict[str, Any] | None = None,
    bearer: str | None = None,
) -> tuple[int, dict[str, Any]]:
    headers = {"Content-Type": "application/json"}
    if bearer is not None:
        headers["Authorization"] = f"Bearer {bearer}"
    encoded = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=encoded, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            status = response.status
            payload = response.read()
    except urllib.error.HTTPError as error:
        status = error.code
        payload = error.read()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{url} returned non-JSON HTTP {status}") from error
    require(isinstance(parsed, dict), f"{url} returned a non-object JSON body")
    return status, parsed


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{path} line {line_number} is invalid JSON") from error
        require(isinstance(row, dict), f"{path} line {line_number} is not an object")
        rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-base", required=True)
    parser.add_argument("--admin-base", required=True)
    parser.add_argument("--admin-token-file", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    admin_token = args.admin_token_file.read_text().strip()
    require(admin_token, "admin token fixture is empty")
    balance_url = f"{args.admin_base}/admin/tenants/smoke/balance"

    missing_status, missing_body = request_json(balance_url)
    require(missing_status == 401, f"missing admin bearer returned {missing_status}, want 401")
    require(
        missing_body.get("error", {}).get("code") == "admin_auth_required",
        "missing admin bearer returned the wrong error code",
    )
    wrong_status, wrong_body = request_json(balance_url, bearer="wrong-admin-token")
    require(wrong_status == 403, f"wrong admin bearer returned {wrong_status}, want 403")
    require(
        wrong_body.get("error", {}).get("code") == "admin_auth_forbidden",
        "wrong admin bearer returned the wrong error code",
    )

    create_status, create_body = request_json(
        f"{args.admin_base}/admin/keys",
        bearer=admin_token,
        body={"tenant": "smoke", "lane": "interactive", "rate_limit": 1},
    )
    require(create_status == 200, f"admin key create returned HTTP {create_status}")
    key = create_body.get("key")
    prefix = create_body.get("prefix")
    require(isinstance(key, str) and key.startswith("mk-smoke-"), "key create omitted plaintext")
    require(isinstance(prefix, str) and key.startswith(prefix), "key create returned bad prefix")

    # KeyStore was constructed before this append. Cross the production two-second poll
    # interval, then prove the public router sees the new hash without a restart.
    time.sleep(2.2)
    completion_body = {
        "model": "smoke",
        "prompt": "Budget enforcement smoke.",
        "max_tokens": 1,
        "temperature": 0,
    }
    oversized_body = dict(completion_body, max_tokens=2)
    oversized_status, oversized_response = request_json(
        f"{args.public_base}/v1/completions",
        bearer=key,
        body=oversized_body,
    )
    require(oversized_status == 402, f"over-budget reservation returned {oversized_status}, want 402")
    require(
        oversized_response.get("error", {}).get("code") == "insufficient_balance",
        "over-budget reservation returned the wrong error code",
    )
    first_status, first_body = request_json(
        f"{args.public_base}/v1/completions",
        bearer=key,
        body=completion_body,
    )
    require(first_status == 200, f"positive-balance request returned HTTP {first_status}")
    second_status, second_body = request_json(
        f"{args.public_base}/v1/completions",
        bearer=key,
        body=completion_body,
    )
    require(second_status == 402, f"exhausted request returned HTTP {second_status}, want 402")
    second_error = second_body.get("error", {})
    require(second_error.get("type") == "insufficient_balance", "402 has wrong error type")
    require(second_error.get("code") == "insufficient_balance", "402 has wrong error code")

    balance_status, balance_body = request_json(balance_url, bearer=admin_token)
    require(balance_status == 200, f"admin balance returned HTTP {balance_status}")
    usage_status, usage_body = request_json(
        f"{args.admin_base}/admin/tenants/smoke/usage",
        bearer=admin_token,
    )
    require(usage_status == 200, f"admin usage returned HTTP {usage_status}")

    ledger_path = args.state_dir / "requests.jsonl"
    journal_path = args.state_dir / "requests.jsonl.tenant-budget-journal.jsonl"
    snapshot_path = args.state_dir / "requests.jsonl.tenant-budget-snapshot.toml"
    audit_path = args.state_dir / "requests.jsonl.admin-audit.jsonl"
    keyring_path = args.state_dir / "keys.toml"
    ledger_rows = read_jsonl(ledger_path)
    completed = [row for row in ledger_rows if row.get("outcome") == "completed"]
    rejected = [row for row in ledger_rows if row.get("outcome") == "rejected"]
    require(len(completed) == 1, f"request ledger has {len(completed)} completed rows, want 1")
    require(len(rejected) == 2, f"request ledger has {len(rejected)} rejected rows, want 2")
    completed_row = completed[0]
    require(completed_row.get("request_id") == first_body.get("id"), "HTTP/ledger request id drift")
    for rejected_row in rejected:
        require(rejected_row.get("http_status") == 402, "rejected ledger row is not HTTP 402")
        require(rejected_row.get("error_code") == "insufficient_balance", "ledger 402 code drift")
        require(rejected_row.get("budget") is None, "rejected request unexpectedly carries a debit")

    exact_cost = completed_row.get("cost_usd", {}).get("total")
    require(isinstance(exact_cost, str), "completed ledger row omitted exact decimal cost")
    expected_debit = int(
        (Decimal(exact_cost) * Decimal(1_000_000)).to_integral_value(rounding=ROUND_CEILING)
    )
    debit_micro = completed_row.get("budget", {}).get("debit_micro")
    reserved_micro = completed_row.get("budget", {}).get("reserved_micro")
    require(debit_micro == expected_debit, "ledger debit is not ceil(exact USD * 1e6)")
    require(reserved_micro == 10_000, "admission did not reserve the one-cent worst case")
    require(expected_debit <= reserved_micro, "terminal debit exceeded its admission hold")
    require(expected_debit == 10_000, "smoke tariff did not exactly exhaust one cent")
    expected_balance = 10_000 - expected_debit
    require(balance_body.get("limited") is True, "admin balance lost prepaid enrollment")
    require(balance_body.get("currency") == "USD", "admin balance currency drift")
    require(balance_body.get("balance_micro") == expected_balance, "admin balance/debit mismatch")
    require(expected_balance == 0, "one-cent tenant was not exactly exhausted")

    journal_rows = read_jsonl(journal_path)
    require(len(journal_rows) == 1, f"budget journal has {len(journal_rows)} rows, want 1")
    journal = journal_rows[0]
    require(journal.get("kind") == "debit", "budget journal row is not a debit")
    require(journal.get("request_id") == completed_row.get("request_id"), "journal id drift")
    require(journal.get("amount_micro") == expected_debit, "journal/ledger debit mismatch")
    require(journal.get("exact_cost_usd") == exact_cost, "journal exact-cost drift")
    require(journal.get("balance_after_micro") == expected_balance, "journal balance drift")

    snapshot = tomllib.loads(snapshot_path.read_text())
    snapshot_rows = [
        row for row in snapshot.get("balances", []) if row.get("tenant") == "smoke"
    ]
    require(len(snapshot_rows) == 1, "balance snapshot omitted or duplicated the smoke tenant")
    require(snapshot_rows[0].get("currency") == "USD", "balance snapshot currency drift")
    require(
        snapshot_rows[0].get("balance_micro") == expected_balance,
        "balance snapshot did not compact the terminal debit",
    )

    days = usage_body.get("days")
    require(isinstance(days, list) and len(days) == 1, "usage did not return one UTC day bucket")
    day = days[0]
    require(day.get("requests") == 1, "usage bucket request count drift")
    require(day.get("cost_usd") == exact_cost, "usage/ledger exact cost mismatch")
    require(day.get("debited_micro") == expected_debit, "usage/ledger debit mismatch")
    ledger_usage = completed_row.get("usage", {})
    response_usage = first_body.get("usage", {})
    require(
        day.get("prompt_tokens") == ledger_usage.get("prompt_tokens") == response_usage.get("prompt_tokens"),
        "prompt-token accounting drift",
    )
    require(
        day.get("completion_tokens")
        == ledger_usage.get("completion_tokens")
        == response_usage.get("completion_tokens"),
        "completion-token accounting drift",
    )

    keyring_text = keyring_path.read_text()
    require(key not in keyring_text, "keyring persisted the plaintext API key")
    keyring = tomllib.loads(keyring_text)
    entries = [entry for entry in keyring.get("keys", []) if entry.get("prefix") == prefix]
    require(len(entries) == 1, "keyring omitted or duplicated the generated prefix")
    require(
        entries[0].get("sha256") == hashlib.sha256(key.encode()).hexdigest(),
        "keyring hash does not match the returned one-time key",
    )

    audit_rows = read_jsonl(audit_path)
    require(len(audit_rows) == 5, f"admin audit has {len(audit_rows)} rows, want 5")
    require(
        [row.get("authorization") for row in audit_rows]
        == ["denied", "denied", "authorized", "authorized", "authorized"],
        "admin audit authorization sequence drift",
    )
    audit_text = audit_path.read_text()
    require(key not in audit_text, "admin audit contains the generated key")
    require(admin_token not in audit_text, "admin audit contains the admin bearer")
    require("wrong-admin-token" not in audit_text, "admin audit contains the rejected bearer")

    write_json(args.out / "admin-auth.json", {
        "missing": {"status": missing_status, "body": missing_body},
        "wrong": {"status": wrong_status, "body": wrong_body},
    })
    write_json(args.out / "reservation-rejected-response.json", oversized_response)
    write_json(args.out / "first-response.json", first_body)
    write_json(args.out / "exhausted-response.json", second_body)
    write_json(args.out / "balance.json", balance_body)
    write_json(args.out / "usage.json", usage_body)
    write_json(args.out / "budget-smoke-summary.json", {
        "verdict": "PASS",
        "initial_balance_micro": 10_000,
        "exact_cost_usd": exact_cost,
        "debit_micro": expected_debit,
        "reserved_micro": reserved_micro,
        "balance_micro": expected_balance,
        "key_prefix": prefix,
        "request_id": completed_row.get("request_id"),
        "reservation_refusal_status": oversized_status,
        "admin_audit_rows": len(audit_rows),
        "rounding": "ceil_positive_request_to_micro_usd",
    })
    for source, destination in [
        (ledger_path, "requests.jsonl"),
        (journal_path, "tenant-budget-journal.jsonl"),
        (snapshot_path, "tenant-budget-snapshot.toml"),
        (audit_path, "admin-audit.jsonl"),
        (args.state_dir / "budgets.toml", "budget-source.toml"),
        (args.state_dir / "models.toml", "model-metadata.toml"),
    ]:
        require(source.is_file(), f"missing smoke artifact {source}")
        shutil.copy2(source, args.out / destination)

    # Never print or write the two bearer values. The prefix is the intended revoke handle.
    print(
        "budget-smoke: PASS "
        f"prefix={prefix} exact_cost_usd={exact_cost} debit_micro={expected_debit} "
        f"balance_micro={expected_balance} second_status={second_status}"
    )


if __name__ == "__main__":
    main()
