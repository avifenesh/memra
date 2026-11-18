#!/usr/bin/env python3
"""Unit tests for the fail-closed public-boundary policy mechanics."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).with_name("check-public-boundary.py")
SPEC = importlib.util.spec_from_file_location("public_boundary", SCRIPT)
assert SPEC and SPEC.loader
boundary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = boundary
SPEC.loader.exec_module(boundary)


def entry(
    path: str = "deploy/gateway/unit",
    digest: str = "a" * 64,
    rules: list[str] | None = None,
) -> dict:
    return {
        "path": path,
        "sha256": digest,
        "category": "private_path",
        "rules": ["test"] if rules is None else rules,
        "reason": "test",
    }


class AllowlistTests(unittest.TestCase):
    def write_lines(self, *entries: dict) -> Path:
        handle = tempfile.NamedTemporaryFile("w", delete=False)
        self.addCleanup(Path(handle.name).unlink, missing_ok=True)
        with handle:
            for item in entries:
                handle.write(json.dumps(item) + "\n")
        return Path(handle.name)

    def test_valid_entry_loads(self) -> None:
        item = entry()
        loaded = boundary.load_allowlist(self.write_lines(item))
        self.assertEqual(loaded[(item["path"], item["sha256"])], item)

    def test_duplicate_key_is_rejected(self) -> None:
        item = entry()
        with self.assertRaisesRegex(SystemExit, "duplicate allowlist key"):
            boundary.load_allowlist(self.write_lines(item, item))

    def test_malformed_digest_is_rejected(self) -> None:
        with self.assertRaisesRegex(SystemExit, "malformed sha256"):
            boundary.load_allowlist(self.write_lines(entry(digest="not-a-digest")))

    def test_entry_without_rules_is_rejected(self) -> None:
        """An entry naming no rule is an exemption from EVERY rule — the 2026-08-19 defect.

        Keyed on (path, sha256) alone, grandfathering a recon journal for
        `production_endpoint` also exempted a provider capacity-block id in the same bytes from a
        rule added months later, permanently and silently. There is deliberately no
        legacy-tolerant read: the allowlist ships beside the scanner, so an entry without a
        scope is a hand edit that has to be rejected rather than interpreted.
        """
        unscoped = entry()
        del unscoped["rules"]
        with self.assertRaisesRegex(SystemExit, "missing required keys: rules"):
            boundary.load_allowlist(self.write_lines(unscoped))

    def test_empty_rules_list_is_rejected(self) -> None:
        with self.assertRaisesRegex(SystemExit, "non-empty list"):
            boundary.load_allowlist(self.write_lines(entry(rules=[])))

    def test_unremediated_must_name_rules_the_entry_grandfathers(self) -> None:
        bad = entry(rules=["test"])
        bad["unremediated"] = ["some_other_rule"]
        with self.assertRaisesRegex(SystemExit, "does not grandfather"):
            boundary.load_allowlist(self.write_lines(bad))

    def test_malformed_expiry_is_rejected(self) -> None:
        bad = entry()
        bad["expires"] = "next month"
        with self.assertRaisesRegex(SystemExit, "ISO date"):
            boundary.load_allowlist(self.write_lines(bad))

    def test_empty_lane_is_rejected(self) -> None:
        bad = entry()
        bad["lane"] = ""
        with self.assertRaisesRegex(SystemExit, "non-empty string"):
            boundary.load_allowlist(self.write_lines(bad))

    def test_severity4_private_path_entry_demands_expiry_and_lane(self) -> None:
        """A parked private SURFACE is a dated owner decision, not a permanent fact.

        Before 2026-08-29 the severity-4 whole-file grandfathers (ledger, admin, darklane,
        lanes) were re-blessed hash-bump after hash-bump — four lanes in one week — and
        nothing ever asked when the parking ends.
        """
        policy = boundary.Policy(
            secret_patterns={}, secret_sources={},
            secret_union=boundary.re.compile(r"(?!x)x"), secret_groups={},
            private_paths={"test": "deploy/gateway/**"}, bypass_paths=[],
            severity={"test": 4},
        )
        undated = entry()
        key = (undated["path"], undated["sha256"])
        with self.assertRaisesRegex(SystemExit, "lack 'expires'"):
            boundary.enforce_expiry_metadata({key: undated}, policy)
        dated = entry()
        dated["expires"] = "2999-01-01"
        dated["lane"] = "some-lane"
        boundary.enforce_expiry_metadata({key: dated}, policy)

    def test_expiry_demand_stops_below_severity4_and_outside_private_path(self) -> None:
        sev3 = boundary.Policy(
            secret_patterns={}, secret_sources={},
            secret_union=boundary.re.compile(r"(?!x)x"), secret_groups={},
            private_paths={"test": "deploy/gateway/**"}, bypass_paths=[],
            severity={"test": 3},
        )
        self.assertFalse(boundary.needs_expiry(entry(), sev3))
        # A severity-4 SECRET-PATTERN pin never demands a date — its safety case lives in
        # the reason text; only whole parked surfaces (private_path rules) do.
        secret_rule = boundary.Policy(
            secret_patterns={}, secret_sources={},
            secret_union=boundary.re.compile(r"(?!x)x"), secret_groups={},
            private_paths={}, bypass_paths=[], severity={"test": 5},
        )
        self.assertFalse(boundary.needs_expiry(entry(), secret_rule))
        # An unranked rule only exists in hand-built test policies; the demand keys on the
        # rank the policy explicitly declares, never on DEFAULT_SEVERITY.
        unranked = boundary.Policy(
            secret_patterns={}, secret_sources={},
            secret_union=boundary.re.compile(r"(?!x)x"), secret_groups={},
            private_paths={"test": "deploy/gateway/**"}, bypass_paths=[],
        )
        self.assertFalse(boundary.needs_expiry(entry(), unranked))

    def test_shipped_allowlist_parks_no_private_surfaces(self) -> None:
        """The end state the expiry machinery was built to force, reached 2026-08-29.

        The severity-4 whole-file grandfathers (ledger, admin, darklane, lanes) are
        GONE: the business tier migrated out (engine-billing-extraction-20260829) and
        lane QoS was reclassified engine-open. The rules stay as tripwires; any entry
        for them reappearing here demands a dated owner decision again.
        """
        policy = boundary.load_policy(boundary.POLICY_PATH)
        allowlist = boundary.load_allowlist(boundary.ALLOWLIST_PATH)
        boundary.enforce_expiry_metadata(allowlist, policy)
        dated = [e["path"] for e in allowlist.values() if boundary.needs_expiry(e, policy)]
        self.assertEqual(
            dated, [], "a private SURFACE is parked in the public repo again"
        )
        # The tripwires themselves must stay declared, or deleting the files also
        # deleted the fence that keeps them out.
        for rule in ("server_ledger", "server_admin", "server_capture"):
            self.assertIn(rule, policy.private_paths)
            self.assertGreaterEqual(policy.severity.get(rule, 0), 4)

    def test_shipped_allowlist_loads_and_is_fully_rule_scoped(self) -> None:
        """The migration's own invariant: no entry in the repo grandfathers everything."""
        allowlist = boundary.load_allowlist(boundary.ALLOWLIST_PATH)
        self.assertTrue(allowlist)
        policy = boundary.load_policy(boundary.POLICY_PATH)
        known = set(policy.secret_sources) | set(policy.private_paths)
        for (path, _digest), item in allowlist.items():
            with self.subTest(path=path):
                self.assertTrue(item["rules"], "entry grandfathers no named rule")
                self.assertEqual(set(item["rules"]) - known, set())

    def test_every_policy_rule_is_ranked(self) -> None:
        """A rule with no severity rank sorts by an implicit default, and a default is how a
        severity-5 finding ends up buried inside a class of noise."""
        policy = boundary.load_policy(boundary.POLICY_PATH)
        named = set(policy.secret_sources) | set(policy.private_paths)
        self.assertEqual(named - set(policy.severity), set())
        self.assertEqual(set(policy.severity) - named, set())
        # The ranks the report ordering depends on: account disclosure must outrank the
        # authorship-noise class it hid inside, and build provenance must sort below both.
        self.assertGreater(policy.rank("aws_account_id"), policy.rank("personal_email"))
        self.assertGreater(policy.rank("aws_resource_id"), policy.rank("personal_email"))
        self.assertGreater(policy.rank("personal_email"), policy.rank("live_fingerprint"))

    def test_public_policy_contains_no_literal_private_values(self) -> None:
        text = boundary.POLICY_PATH.read_text()
        self.assertNotRegex(text, r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")
        self.assertNotRegex(text, r"\b0x[0-9a-fA-F]{40}\b")

    def test_structural_secret_patterns_match_synthetic_values(self) -> None:
        policy = boundary.load_policy(boundary.POLICY_PATH)
        joined = "".join
        samples = {
            # Keep policy test fixtures structural without making this tracked test file a
            # scanner violation itself. Joining fragments across source lines still exercises
            # each exact runtime shape without placing a private-shaped literal in this file.
            "production_endpoint": joined(
                ("public_api=https://api.example", ".ai/v1")
            ),
            "payout_wallet": joined(
                ("wallet=0x11111111111111111111", "11111111111111111111")
            ),
            "rented_ipv4": joined(
                ("ssh endpoint root@203.0.113", ".42:22000")
            ),
            "runpod_pod_id": joined(
                ("runpod pod_id=abcdef1", "2345678")
            ),
            # a NAKED id, no keyword anywhere near — the shape the keyword rule missed.
            "runpod_pod_id_bare": joined(
                ("| u8f2n1qxl", "4mo3p | RTX 5090 |")
            ),
            "vast_serve_host": joined(
                ("ssh -p 12345 root@ssh7.vast", ".ai")
            ),
            "serve_data_root": joined(
                ("/data/memra-", "prod-20260814")
            ),
            "live_fingerprint": joined(
                ("binary=memra-01234567", "89abcdef")
            ),
        }
        for name, sample in samples.items():
            with self.subTest(name=name):
                self.assertRegex(sample, policy.secret_patterns[name])

    def test_host_destination_and_cloud_inventory_shapes_are_detected(self) -> None:
        """Shapes that reached pushed lane branches while every rule above stayed silent.

        Two review ticks found them: a bench-box transfer target written with no keyword on
        the line for `rented_ipv4` to anchor on, and a rented-rig inventory quoted through
        the resource and contract ids rather than through an instance id. Both go through the
        combined union, because that — not the per-rule dict — is what the gate evaluates.
        Fragments are joined at runtime so this tracked test file is not itself a violation.
        """
        policy = boundary.load_policy(boundary.POLICY_PATH)
        joined = "".join
        # The prefix and the hex body are concatenated at runtime for the same reason the
        # samples above are: a whole `<prefix>-<17 hex>` literal would make this tracked test
        # file its own `aws_resource_id` violation.
        hexbody = "0f1e2d3c4b5a69788"
        expected = {
            "ssh_destination": joined(
                ("- Box flow: `git push ubuntu@203.0.113", ".114:/srv/memra-src lane/x`")
            ),
            "aws_resource_id": "  launched into subnet-" + hexbody,
            "provider_contract_id": joined(
                ("Box: vast.ai contract 9", "999999 — 4x RTX PRO 6000 96GB (sm_120)")
            ),
        }
        for name, line in expected.items():
            with self.subTest(name=name):
                hits = boundary.scan_secret_bytes(
                    line.encode(), policy.secret_union, policy.secret_groups
                )
                self.assertEqual([hit[0] for hit in hits], [name])

        # Every prefix in the family, so a rig receipt that quotes an image, a volume or a
        # capacity block instead of an instance id cannot slip through the next time.
        for prefix in ("ami", "subnet", "vpc", "vol", "eni", "snap", "cr"):
            with self.subTest(prefix=prefix):
                hits = boundary.scan_secret_bytes(
                    (f"capacity block {prefix}-" + hexbody).encode(),
                    policy.secret_union,
                    policy.secret_groups,
                )
                self.assertEqual([hit[0] for hit in hits], ["aws_resource_id"])

        # The exact class the gate AS DEPLOYED could not see. `aws_resource_id` was written
        # 2026-08-18 and never pushed, so on every published ref the only prefixes any rule
        # matched were `i-` and `sg-`: the capacity-block id that motivated the 2026-08-19
        # public-ref audit was undetectable on any branch at any time, and so was every
        # `subnet-` and `ami-` id. This asserts the whole family through the real policy union
        # AND through the per-rule pass, since that is what the scanner now runs.
        for prefix in ("cr", "subnet", "ami"):
            with self.subTest(deployed_gap=prefix):
                blob = f"# provisioned {prefix}-{hexbody} in zone 2a\n".encode()
                for label, patterns in (
                    ("union", None),
                    ("per-rule", policy.secret_patterns),
                ):
                    hits = boundary.scan_secret_bytes(
                        blob, policy.secret_union, policy.secret_groups, patterns
                    )
                    self.assertEqual(
                        [hit[0] for hit in hits],
                        ["aws_resource_id"],
                        f"{prefix}- undetected via {label} pass",
                    )
                # And it survives the full evaluate path, which is what the gate calls.
                violation = boundary.evaluate_content(
                    policy, f"research/lane/RECEIPTS-{prefix}.md", blob
                )
                assert violation is not None
                self.assertIn("aws_resource_id", violation.rules)
                self.assertGreaterEqual(violation.severity, 4)

        # Non-routable and label-shaped addresses must stay silent: a gate that fires on a
        # quickstart's loopback line or on an `ncu` run label is a gate that gets bypassed.
        quiet = (
            joined(("ADDR=127.0.0", ".1:8080")),
            joined(("host 192.168", ".1.50")),
            joined(("downkernel-baseline@10.0", ".0.1")),
        )
        for line in quiet:
            with self.subTest(line=line):
                self.assertEqual(
                    boundary.scan_secret_bytes(
                        line.encode(), policy.secret_union, policy.secret_groups
                    ),
                    [],
                )

    def test_bare_pod_id_rule_rejects_the_measured_false_positive_shapes(self) -> None:
        # The bare-id rule is structural, so its NEGATIVE space is the contract: every
        # shape below was a measured in-tree false positive of naive [a-z0-9]{14}
        # (24,744 unique tokens) and must stay out. None of these literals matches any
        # policy rule, so they are safe to write plainly here.
        policy = boundary.load_policy(boundary.POLICY_PATH)
        rule = policy.secret_patterns["runpod_pod_id_bare"]
        for benign in (
            "administration",  # 14-letter English word: no digits
            "quantification",
            "0x7236bd755390",  # profiler hex address
            "0x1.ebfce50fac4f3p+1",  # C hexfloat literal (after-dot exclusion)
            "4096x1536x4096",  # GEMM shape
            "128x128x8warps",  # kernel tile shape (digit-x-digit exclusion)
            "\\u2014including",  # \\uXXXX escape in saved JSON/HTML
            "\\u003cformatted",
            "20260802release",  # blocky date+word: no digit->letter->digit interleave
        ):
            with self.subTest(benign=benign):
                self.assertNotRegex(benign, rule)
        # and interleaved random base36 ids (the pod-id shape) DO match bare:
        for hot in ("".join(("3yrauhltt9", "zana")), "".join(("hq1628lvek", "28gn"))):
            with self.subTest(hot=hot):
                self.assertRegex(hot, rule)

    # --- CREDENTIAL MATERIAL (main @ 274f378faf, folded into the v0.94.0 candidate) ---
    #
    # Every rule the gate carried before these described an IDENTIFIER. These describe a
    # SECRET, which is the one question a public-boundary gate exists to answer, so they get
    # their own end-to-end assertions rather than a line in the synthetic-samples dict: each
    # fixture goes through the real policy union, the real per-rule pass AND `evaluate_content`,
    # because that is what the gate calls. Fragments are joined at runtime for the same reason
    # as every fixture above — a whole credential-shaped literal would make this tracked test
    # file its own violation, which is exactly the trap it is testing for.
    CREDENTIAL_FIXTURES = {
        # mk-<tenant>-<48 hex>: what `auth.rs gen_key` mints. A live credential.
        "minted_api_key": ("Authorization: Bearer mk-acme-", "0123456789abcdef" * 3),
        # mk-<tenant>-<12 hex>: the revocation prefix. Authenticates nothing; names a customer.
        "api_key_prefix": ("revoked mk-acme-", "0123456789ab"),
        # tg-<32+ hex>: hand-minted box key from the bring-up runbook. No tenant in the string.
        "hand_minted_key": ("KEY=tg-", "0123456789abcdef" * 2),
        # `openssl rand -hex 32` behind an admin/metrics keyword.
        "admin_token": ("metrics-token: ", "0123456789abcdef" * 4),
        # A cloudflared credentials file. The token IS the tunnel's authority.
        "connector_credentials": ('{"Tunnel', 'Secret": "<value>"}'),
        "provider_token": ("ghp_", "A" * 30),
        "provider_secret_assignment": ("CLOUDFLARE_API_TOKEN=", "v" * 24),
    }
    # The six that authenticate something must sit at the ceiling: a published secret cannot
    # sort below the identifiers whose whole harm is making a future secret leak worse.
    CREDENTIAL_AT_CEILING = frozenset(CREDENTIAL_FIXTURES) - {"api_key_prefix"}

    def test_commit_scan_uses_the_checkout_scans_raw_byte_prefilter(self) -> None:
        """Commit/ref mode and the checkout scan must judge the same candidate set.

        The checkout scan asks `git grep --text -P` over raw bytes first; commit mode decoded
        with errors="ignore" and scanned everything, so bytes glued across an undecodable byte
        spelled a needle that exists in no byte view. 2026-09-20: two gzipped trace logs were
        refused by the pre-push hook and then called stale pins by `check`/`verify-allowlist`.
        """
        import gzip

        policy = boundary.Policy(
            secret_patterns={"alpha": boundary.re.compile("alpha needle")},
            secret_sources={"alpha": "alpha needle"},
            secret_union=boundary.re.compile("(?P<rule_0>alpha needle)"),
            secret_groups={"rule_0": "alpha"},
            private_paths={},
            bypass_paths=[],
        )
        glued = b"alpha \xffneedle\n"
        # The text scan alone would report it (ignore-decoding glues the needle together) ...
        self.assertEqual(
            [h[0] for h in boundary.scan_secret_bytes(glued, policy.secret_union, policy.secret_groups)],
            ["alpha"],
        )
        # ... but the raw bytes never contained it, so neither scan reports it.
        self.assertFalse(boundary.raw_bytes_prefilter(glued, policy.secret_sources))
        self.assertIsNone(boundary.evaluate_content(policy, "research/lane/NOTES.md", glued))
        text = b"# trace log\n[expert-host-slru] key=34:1:7 bytes=450560 slot=0 hit=false\n"
        self.assertIsNone(
            boundary.evaluate_content(policy, "research/lane/trace.log.gz", gzip.compress(text * 64))
        )
        # A real needle, even inside a binary-looking blob, is still a finding (CommitBlobTests).
        violation = boundary.evaluate_content(policy, "research/lane/capture.bin", b"\0alpha needle\n")
        assert violation is not None
        self.assertEqual(violation.rules, ("alpha",))
        # Every shipped rule compiles as a bytes pattern, so the prefilter never has to fail open.
        shipped = boundary.load_policy(boundary.POLICY_PATH)
        for name, source in shipped.secret_sources.items():
            with self.subTest(rule=name):
                boundary.re.compile(source.encode("utf-8"))

    def test_credential_material_rules_fire_end_to_end(self) -> None:
        policy = boundary.load_policy(boundary.POLICY_PATH)
        for name, parts in self.CREDENTIAL_FIXTURES.items():
            sample = "".join(parts)
            with self.subTest(rule=name):
                self.assertIn(name, policy.secret_patterns, "rule missing from policy")
                blob = f"# captured setup log\n{sample}\n".encode()
                for label, patterns in (
                    ("union", None),
                    ("per-rule", policy.secret_patterns),
                ):
                    hits = boundary.scan_secret_bytes(
                        blob, policy.secret_union, policy.secret_groups, patterns
                    )
                    self.assertIn(
                        name,
                        [hit[0] for hit in hits],
                        f"{name} undetected via the {label} pass",
                    )
                # And through the path the gate actually calls.
                violation = boundary.evaluate_content(
                    policy, f"research/lane/CAPTURE-{name}.log", blob
                )
                assert violation is not None
                self.assertIn(name, violation.rules)
                self.assertGreaterEqual(violation.severity, 4)

    def test_credential_rules_outrank_the_identifier_classes(self) -> None:
        """A credential ranked with the service surface is a credential nobody reads first.

        The [severity] table is mandatory, so a new credential rule cannot land unranked — but
        it can land at 3 beside `production_endpoint`, and then the worst line in a
        several-hundred-finding report sorts into the middle of it.
        """
        policy = boundary.load_policy(boundary.POLICY_PATH)
        for name in self.CREDENTIAL_AT_CEILING:
            with self.subTest(rule=name):
                self.assertEqual(policy.rank(name), boundary.SEVERITY_CEILING)
        # The mk- split exists to carry different severities; assert the split is real in the
        # table and not just in the two regexes.
        self.assertGreater(
            policy.rank("minted_api_key"), policy.rank("api_key_prefix")
        )
        self.assertGreater(
            policy.rank("api_key_prefix"), policy.rank("production_endpoint")
        )

    def test_the_two_mk_rules_cannot_both_match_one_string(self) -> None:
        """The claim the mk- split rests on, verified in both directions.

        `[0-9a-f]{12}\\b` cannot end inside a 48-hex run because hex digits are word
        characters, so a minted key is never also reported as its own prefix and a prefix is
        never mistaken for a live credential. If that ever stopped holding, every minted-key
        finding would arrive carrying a second, wrong, lower-severity label.
        """
        policy = boundary.load_policy(boundary.POLICY_PATH)
        minted = "".join(self.CREDENTIAL_FIXTURES["minted_api_key"])
        prefix = "".join(self.CREDENTIAL_FIXTURES["api_key_prefix"])
        self.assertRegex(minted, policy.secret_patterns["minted_api_key"])
        self.assertNotRegex(minted, policy.secret_patterns["api_key_prefix"])
        self.assertRegex(prefix, policy.secret_patterns["api_key_prefix"])
        self.assertNotRegex(prefix, policy.secret_patterns["minted_api_key"])

    def test_prose_about_keys_and_bare_hashes_stay_quiet(self) -> None:
        """The negative space. A gate that fires on documentation is a gate that gets bypassed.

        `admin_token` is keyword-anchored precisely because a bare 64-hex string is a sha256 and
        this repo is full of them — the commit shas, blob digests and allowlist pins in every
        receipt must not each become a credential finding.
        """
        policy = boundary.load_policy(boundary.POLICY_PATH)
        quiet = (
            "Rotate the api key with `POST /admin/keys` and record only the prefix.",
            "sha256=" + "0123456789abcdef" * 4,
            "resumed from checkpoint " + "0123456789abcdef" * 4,
            "set MEMRA_METRICS_TOKEN in the unit file, never in the repo",
            "mk-acme-" + "0123456789",  # too short to be either mk- shape
        )
        credential_rules = set(self.CREDENTIAL_FIXTURES)
        for line in quiet:
            with self.subTest(line=line[:48]):
                hits = boundary.scan_secret_bytes(
                    line.encode(),
                    policy.secret_union,
                    policy.secret_groups,
                    policy.secret_patterns,
                )
                self.assertEqual(
                    credential_rules.intersection(hit[0] for hit in hits), set()
                )

    def test_no_shipped_allowlist_entry_shadows_a_credential_rule(self) -> None:
        """A credential grant has to be its own named grant, on bytes somebody reviewed.

        This is the live instance the fold surfaced: `ONLIST-LIVE.md` was pinned for `onlist`
        and `production_endpoint`, and under the old (path, sha256) keying that pin silently
        absorbed the `api_key_prefix` finding on line 102 — live customer key prefixes with
        per-key request counts, exempted by an entry granted for something else. Rule scoping
        turned it back into a violation that had to be named. Assert the shape that keeps it
        named: every entry covering a credential rule says so out loud, and the count matches
        what the fold reviewed.
        """
        allowlist = boundary.load_allowlist(boundary.ALLOWLIST_PATH)
        credential_rules = set(self.CREDENTIAL_FIXTURES)
        granted = [
            item
            for item in allowlist.values()
            if credential_rules.intersection(item["rules"])
        ]
        # 13 rows folded from main. Was 15: the ONLIST-LIVE.md entry (api_key_prefix —
        # live customer key prefixes) was REMEDIATED 2026-08-21 — the file migrated to
        # darklanes (research/memra-boundary-migration-20260821/) with the other 65
        # unremediated entries and its grant was pruned. The docs/SERVING.md revoke example was
        # then rewritten to a non-key-shaped placeholder and its grant pruned 2026-08-24. A new
        # credential grant is a new decision: bump this count only with reviewed bytes named here.
        # 24 since memra#1198 (gzip blobs judged by their text): eleven source snapshots carry
        # `crates/memra-server/src/auth.rs`, whose credential-shaped matches are, string for
        # string, the six unit-test fixtures of the in-tree auth.rs grant (checked when granted):
        # the eight `runtime-source*.tar.gz` under research/mtp-context-depth-20260921/,
        # research/mtp-continuing-session-20260921/ and research/prompt-depth-router-20260922/, and
        # the three `joint-v8/*/native-data.tar.gz` there, through the `runtime-source-joint-v*`
        # tarball nested in each.
        snapshots = [
            i
            for i in granted
            if "runtime-source" in i["path"] or i["path"].endswith("/native-data.tar.gz")
        ]
        self.assertEqual(len(snapshots), 11)
        self.assertEqual(len(granted), 24)
        for item in granted:
            with self.subTest(path=item["path"]):
                # `rules` is a whitelist, never a wildcard: a credential grant that did not
                # name the rule would not be an exemption at all.
                self.assertTrue(credential_rules.intersection(item["rules"]))
        # Exactly one blob carries a minted key (the auth.rs unit-test fixtures); the rest are
        # revocation prefixes. A second minted-key grant is a new decision, not a reseed.
        minted = [
            i for i in granted if "minted_api_key" in i["rules"] and i not in snapshots
        ]
        self.assertEqual([i["path"] for i in minted], ["crates/memra-server/src/auth.rs"])


class CheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = boundary.Policy(
            secret_patterns={},
            secret_sources={},
            secret_union=boundary.re.compile(r"(?!x)x"),
            secret_groups={},
            private_paths={},
            bypass_paths=[],
        )

    def run_check(
        self, allowlist: dict, violations: list, **kwargs: object
    ) -> tuple[int, str]:
        output = io.StringIO()
        with (
            mock.patch.object(boundary, "load_allowlist", return_value=allowlist),
            mock.patch.object(boundary, "evaluate", return_value=violations),
            contextlib.redirect_stdout(output),
        ):
            rc = boundary.cmd_check(self.policy, **kwargs)
        return rc, output.getvalue()

    def test_grandfathered_live_entry_passes(self) -> None:
        violation = boundary.Violation("deploy/gateway/unit", "a" * 64, "private_path", "test")
        key = (violation.path, violation.sha256)
        rc, output = self.run_check({key: entry()}, [violation])
        self.assertEqual(rc, 0)
        self.assertIn("1 grandfathered, 0 new", output)

    def test_stale_entry_fails_closed(self) -> None:
        stale = entry()
        key = (stale["path"], stale["sha256"])
        rc, output = self.run_check({key: stale}, [])
        self.assertEqual(rc, 1)
        self.assertIn("stale allowlist entries", output)

    def test_expired_grandfather_resurfaces_the_finding(self) -> None:
        """Past its date an entry exempts nothing; the report names the parking record."""
        violation = boundary.Violation("deploy/gateway/unit", "a" * 64, "private_path", "test")
        parked = entry()
        parked["expires"] = "2000-01-01"
        parked["lane"] = "some-lane"
        key = (violation.path, violation.sha256)
        rc, output = self.run_check({key: parked}, [violation])
        self.assertEqual(rc, 1)
        self.assertIn("EXPIRED grandfathers", output)
        self.assertIn("lane: some-lane", output)

    def test_future_dated_grandfather_still_exempts(self) -> None:
        violation = boundary.Violation("deploy/gateway/unit", "a" * 64, "private_path", "test")
        parked = entry()
        parked["expires"] = "2999-01-01"
        parked["lane"] = "some-lane"
        key = (violation.path, violation.sha256)
        rc, output = self.run_check({key: parked}, [violation])
        self.assertEqual(rc, 0)
        self.assertIn("1 grandfathered, 0 new", output)

    def test_summary_only_reports_counts_without_paths(self) -> None:
        """A public Actions log must not publish where the unremediated leaks are.

        The default output names the file and the line, which is the right thing locally and
        the wrong thing in a job whose log anyone can read. --summary-only still fails the
        job; it just refuses to say where to look.
        """
        violations = [
            boundary.Violation(
                "research/lane-a/RECEIPTS.md", "b" * 64, "secret_pattern",
                "aws_resource_id (first hit line 118)",
            ),
            boundary.Violation(
                "research/lane-b/SUMMARY.md", "c" * 64, "secret_pattern",
                "aws_resource_id,ssh_destination (first hit line 5)",
            ),
        ]
        rc, output = self.run_check({}, violations, summary_only=True)
        self.assertEqual(rc, 1)
        self.assertIn("0 grandfathered, 2 new", output)
        self.assertIn("2  aws_resource_id", output)
        self.assertIn("1  ssh_destination", output)
        for leaky in ("research/lane-a/RECEIPTS.md", "research/lane-b/SUMMARY.md", "118"):
            self.assertNotIn(leaky, output)
        # Same violations, default output: the paths are exactly what a maintainer needs.
        _, verbose = self.run_check({}, violations)
        self.assertIn("research/lane-a/RECEIPTS.md", verbose)

    def test_secret_scan_reports_every_rule_not_just_the_earliest(self) -> None:
        """Defect 2, 2026-08-19: first-match-only buried the worst finding in the repo.

        `wiki-055.txt` carries `personal_email` at offset 367 and a cloud account id at 556, and
        was therefore reported as `personal_email` — one line inside 56 hits of a class a
        reviewer correctly triages as authorship noise, with the account id, the
        AdministratorAccess assertion and the identity principal beside it named nowhere. One hit is
        enough to enforce; triage needs all of them.
        """
        handle = tempfile.NamedTemporaryFile("w", delete=False)
        self.addCleanup(Path(handle.name).unlink, missing_ok=True)
        with handle:
            handle.write("safe\nalpha needle\nsafe\nbeta needle\nalpha needle again\n")
        matcher = boundary.re.compile(r"(?P<alpha>alpha)|(?P<beta>beta)")
        hits = boundary.scan_secrets(
            Path(handle.name), matcher, {"alpha": "alpha", "beta": "beta"}
        )
        self.assertEqual(
            [(name, line) for name, line, _ in hits], [("alpha", 2), ("beta", 4)]
        )

    def test_overlapping_rules_on_one_span_are_both_reported(self) -> None:
        """Alternation reports one branch per position, so the per-rule pass is not optional.

        Two rules matching the SAME span is the case the union structurally cannot report no
        matter how it is iterated — whichever alternative is listed first wins and the other is
        invisible. Passing the per-rule dict is what makes the report complete.
        """
        matcher = boundary.re.compile(r"(?P<wide>need\w+)|(?P<narrow>needle)")
        groups = {"wide": "wide", "narrow": "narrow"}
        patterns = {
            "wide": boundary.re.compile(r"need\w+"),
            "narrow": boundary.re.compile(r"needle"),
        }
        union_only = boundary.scan_secret_bytes(b"one needle\n", matcher, groups)
        self.assertEqual([name for name, _, _ in union_only], ["wide"])
        both = boundary.scan_secret_bytes(b"one needle\n", matcher, groups, patterns)
        self.assertEqual(sorted(name for name, _, _ in both), ["narrow", "wide"])

    def test_multi_rule_blob_reports_all_rules_worst_first(self) -> None:
        """The report has to name every rule AND lead with the worst one.

        Ordering is half the fix: a complete list still hides its own worst line if it is
        ordered by scan offset, which is the order that put `personal_email` first.
        """
        policy = boundary.Policy(
            secret_patterns={
                "low": boundary.re.compile("noise"),
                "high": boundary.re.compile("account"),
            },
            secret_sources={"low": "noise", "high": "account"},
            secret_union=boundary.re.compile("(?P<rule_0>noise)|(?P<rule_1>account)"),
            secret_groups={"rule_0": "low", "rule_1": "high"},
            private_paths={},
            bypass_paths=[],
            severity={"low": 1, "high": 5},
        )
        # `noise` matches first by offset; `account` is the finding that matters.
        v = boundary.evaluate_content(policy, "research/lane/NOTES.md", b"noise\naccount\n")
        assert v is not None
        self.assertEqual(v.rules, ("high", "low"))
        self.assertEqual(v.severity, 5)
        self.assertTrue(v.detail.startswith("high,low"))
        self.assertEqual(v.lines, {"low": 1, "high": 2})

    def test_report_is_ordered_by_severity_not_scan_order(self) -> None:
        policy = boundary.Policy(
            secret_patterns={},
            secret_sources={},
            secret_union=boundary.re.compile(r"(?!x)x"),
            secret_groups={},
            private_paths={},
            bypass_paths=[],
            severity={"noise": 1, "account": 5},
        )
        noisy = boundary.Violation(
            "aaa-first-by-path.md", "b" * 64, "secret_pattern", "noise",
            rules=("noise",), severity=1,
        )
        worst = boundary.Violation(
            "zzz-last-by-path.md", "c" * 64, "secret_pattern", "account",
            rules=("account",), severity=5,
        )
        output = io.StringIO()
        with (
            mock.patch.object(boundary, "load_allowlist", return_value={}),
            mock.patch.object(boundary, "evaluate", return_value=[noisy, worst]),
            contextlib.redirect_stdout(output),
        ):
            boundary.cmd_check(policy)
        printed = output.getvalue()
        self.assertLess(
            printed.index("zzz-last-by-path.md"), printed.index("aaa-first-by-path.md")
        )
        self.assertIn("[NEW sev5]", printed)

    def test_allowlisted_for_one_rule_still_trips_another(self) -> None:
        """Defect 1, 2026-08-19: an exemption for rule A was an exemption for every rule.

        The live instance this reproduces: `recon-journal-v2.jsonl` was grandfathered as
        `production_endpoint` and the same bytes carried a provider capacity-block id at offset
        9,616, which was therefore exempt from `aws_resource_id` — a rule written specifically
        to catch it — and appeared in no report at all.
        """
        policy = boundary.Policy(
            secret_patterns={},
            secret_sources={},
            secret_union=boundary.re.compile(r"(?!x)x"),
            secret_groups={},
            private_paths={},
            bypass_paths=[],
            severity={"production_endpoint": 3, "aws_resource_id": 4},
        )
        violation = boundary.Violation(
            "research/model-selection/recon-journal-v2.jsonl",
            "d" * 64,
            "secret_pattern",
            "aws_resource_id,production_endpoint (first hit line 5)",
            rules=("aws_resource_id", "production_endpoint"),
            severity=4,
            lines={"production_endpoint": 5, "aws_resource_id": 6},
        )
        allowlist = {
            (violation.path, violation.sha256): entry(
                path=violation.path, digest="d" * 64, rules=["production_endpoint"]
            )
        }
        output = io.StringIO()
        with (
            mock.patch.object(boundary, "load_allowlist", return_value=allowlist),
            mock.patch.object(boundary, "evaluate", return_value=[violation]),
            contextlib.redirect_stdout(output),
        ):
            rc = boundary.cmd_check(policy)
        printed = output.getvalue()
        self.assertEqual(rc, 1)
        self.assertIn("0 grandfathered, 1 new", printed)
        # Reported for the rule it is NOT exempt for, and only that one, with the partial
        # exemption stated so the reader knows why it was invisible before.
        self.assertIn("aws_resource_id (first hit line 6)", printed)
        self.assertIn("[allowlisted only for: production_endpoint]", printed)
        self.assertIn("allowlisted for a DIFFERENT rule", printed)
        # Naming both rules clears it, and that is the only thing that does.
        allowlist[(violation.path, violation.sha256)]["rules"] = [
            "production_endpoint",
            "aws_resource_id",
        ]
        output = io.StringIO()
        with (
            mock.patch.object(boundary, "load_allowlist", return_value=allowlist),
            mock.patch.object(boundary, "evaluate", return_value=[violation]),
            contextlib.redirect_stdout(output),
        ):
            rc = boundary.cmd_check(policy)
        self.assertEqual(rc, 0)
        self.assertIn("1 grandfathered, 0 new", output.getvalue())

    def test_pinned_undecided_findings_are_reported_on_a_passing_run(self) -> None:
        """A pinned finding is a decision deferred, and a deferral only stays honest if loud.

        Green-and-quiet is how the last leak hid inside a 578-entry allowlist, so the count is
        printed even when the gate passes.
        """
        policy = boundary.Policy(
            secret_patterns={},
            secret_sources={},
            secret_union=boundary.re.compile(r"(?!x)x"),
            secret_groups={},
            private_paths={},
            bypass_paths=[],
            severity={"aws_resource_id": 4, "production_endpoint": 3},
        )
        violation = boundary.Violation(
            "research/lane/journal.jsonl", "f" * 64, "secret_pattern",
            "aws_resource_id,production_endpoint",
            rules=("aws_resource_id", "production_endpoint"),
        )
        pinned = entry(
            path=violation.path,
            digest="f" * 64,
            rules=["aws_resource_id", "production_endpoint"],
        )
        pinned["unremediated"] = ["aws_resource_id"]
        output = io.StringIO()
        with (
            mock.patch.object(
                boundary,
                "load_allowlist",
                return_value={(violation.path, violation.sha256): pinned},
            ),
            mock.patch.object(boundary, "evaluate", return_value=[violation]),
            contextlib.redirect_stdout(output),
        ):
            rc = boundary.cmd_check(policy)
        printed = output.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("1 grandfathered, 0 new", printed)
        self.assertIn("1 allowlisted findings are marked unremediated", printed)
        self.assertIn("worst sev4", printed)
        self.assertIn("aws_resource_id=1", printed)
        # Only the undecided rule is outstanding — the long-standing pin is not double-counted.
        self.assertNotIn("production_endpoint=1", printed)

    def test_remediating_the_granted_rule_makes_the_entry_stale(self) -> None:
        """The grant expires with the finding it was granted for.

        Under (path, sha256) an entry stayed live while the bytes tripped ANY rule, so fixing
        the rule it was granted for left the exemption in place still covering everything else.
        """
        still_live = boundary.Violation(
            "research/lane/NOTES.md", "e" * 64, "secret_pattern", "other_rule",
            rules=("other_rule",),
        )
        # The entry was granted for two rules; only one still matches these bytes.
        granted = entry(
            path=still_live.path,
            digest="e" * 64,
            rules=["other_rule", "remediated_rule"],
        )
        rc, printed = self.run_check(
            {(still_live.path, still_live.sha256): granted}, [still_live]
        )
        self.assertEqual(rc, 1)
        self.assertIn("stale allowlist entries", printed)
        self.assertIn("rules no longer match: remediated_rule", printed)


class CommitBlobTests(unittest.TestCase):
    def setUp(self) -> None:
        # `ignore_cleanup_errors`: these suites build a REAL git repo in the tempdir, and git may
        # touch `.git` from a background task while rmtree is walking it — which surfaced on CI as
        # `OSError: [Errno 39] Directory not empty: '.git'` raised from TemporaryDirectory cleanup
        # AFTER every assertion had already passed, reddening the build for nothing. The two
        # `config` lines below remove the cause (no auto-gc, no background maintenance in a
        # throwaway repo); the flag makes the whole class of cleanup race non-fatal, because a
        # leaked temp dir is never a reason to fail a policy gate.
        self.tempdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Boundary Test")
        self.git("config", "user.email", "boundary@example.invalid")
        self.git("config", "gc.auto", "0")
        self.git("config", "maintenance.auto", "false")
        self.policy = boundary.Policy(
            secret_patterns={"alpha": boundary.re.compile("alpha needle")},
            secret_sources={"alpha": "alpha needle"},
            secret_union=boundary.re.compile("(?P<rule_0>alpha needle)"),
            secret_groups={"rule_0": "alpha"},
            private_paths={},
            bypass_paths=[],
        )

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def write(self, path: str, data: bytes) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def symlink(self, path: str, target: str) -> None:
        link = self.root / path
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)

    def commit(self, path: str, data: bytes, message: str) -> str:
        self.write(path, data)
        self.git("add", "-f", path)
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")

    def test_secret_prefilter_treats_binary_file_as_text(self) -> None:
        self.write("capture.bin", b"\0alpha needle\n")
        self.git("add", "capture.bin")
        with mock.patch.object(boundary, "ROOT", self.root):
            candidates = boundary.secret_candidate_files(self.policy.secret_sources)
        self.assertIn("capture.bin", candidates)

    def test_checkout_scan_reads_symlink_blob_without_following_absolute_target(self) -> None:
        """Checkout mode scans the published link text, not a machine-local target.

        The live failure was a tracked receipt link into `/root`: pathlib followed it while
        evaluating `is_file()` and Python 3.12 raised PermissionError on the hosted runner.
        """
        self.symlink("receipt-link", "/root/alpha needle")
        self.git("add", "receipt-link")
        self.git("commit", "-q", "-m", "absolute receipt link")
        with mock.patch.object(boundary, "ROOT", self.root):
            violations = boundary.evaluate(self.policy)
        self.assertEqual(
            [(violation.path, violation.detail) for violation in violations],
            [("receipt-link", "alpha (first hit line 1)")],
        )

    def test_commit_scan_reads_binary_blob_instead_of_checkout(self) -> None:
        bad_commit = self.commit("capture.bin", b"\0alpha needle\n", "binary evidence")
        clean_commit = self.commit("capture.bin", b"\0safe\n", "clean checkout")
        with mock.patch.object(boundary, "ROOT", self.root):
            violations = boundary.evaluate_commits(
                self.policy, [clean_commit, bad_commit]
            )
        self.assertEqual(
            [(violation.path, violation.detail) for violation in violations],
            [("capture.bin", "alpha (first hit line 1)")],
        )

    # --- memra#1198: a gzip blob is judged by its decompressed text ---------------------------

    @staticmethod
    def chance_gzip() -> tuple[bytes, str]:
        """A gzip blob whose compressed bytes spell a four-letter run its text cannot hold: the
        text is digits and newlines only, so any letters are the deflate stream's by chance."""
        import gzip

        payload = "\n".join(str(i * 7919 % 100003) for i in range(40000)).encode()
        blob = gzip.compress(payload, mtime=0)
        match = boundary.re.search(rb"[A-Za-z]{4}", blob[10:])
        assert match is not None, "no letter run in 100 KB of deflate bytes"
        return blob, match.group().decode()

    def chance_policy(self, needle: str) -> "boundary.Policy":
        return boundary.Policy(
            secret_patterns={"chance": boundary.re.compile(needle)},
            secret_sources={"chance": needle},
            secret_union=boundary.re.compile(f"(?P<rule_0>{needle})"),
            secret_groups={"rule_0": "chance"},
            private_paths={},
            bypass_paths=[],
        )

    def test_gzip_bytes_that_match_by_chance_are_not_a_finding(self) -> None:
        """memra#1198: the compressed bytes match the rule, the text does not. The gate used to
        refuse such a receipt; recompressing at another level made it pass."""
        blob, needle = self.chance_gzip()
        policy = self.chance_policy(needle)
        self.assertTrue(boundary.raw_bytes_prefilter(blob, policy.secret_sources))
        self.assertIsNone(boundary.evaluate_content(policy, "research/lane/rows.jsonl.gz", blob))
        bad = self.commit("research/lane/rows.jsonl.gz", blob, "compressed receipt")
        with mock.patch.object(boundary, "ROOT", self.root):
            self.assertEqual(boundary.evaluate(policy), [])
            self.assertEqual(boundary.evaluate_commits(policy, [bad]), [])
            refs = boundary.published_refs("refs/heads/**")
            self.assertEqual(boundary.evaluate_refs(policy, refs)[0], [])

    def test_a_match_in_a_gzip_blobs_text_is_a_finding(self) -> None:
        """memra#1198: the other half, a real match inside a compressed receipt. Its compressed
        bytes do not spell it, so a scan of the bytes never saw it."""
        import gzip

        text = b"# trace\n" + b"row 1 ok\n" * 4000 + b"the alpha needle line\n"
        blob = gzip.compress(text, mtime=0)
        self.assertFalse(boundary.raw_bytes_prefilter(blob, self.policy.secret_sources))
        violation = boundary.evaluate_content(self.policy, "research/lane/trace.log.gz", blob)
        assert violation is not None
        self.assertEqual(violation.rules, ("alpha",))
        self.assertEqual(violation.lines, {"alpha": 4002})
        bad = self.commit("research/lane/trace.log.gz", blob, "compressed trace")
        with mock.patch.object(boundary, "ROOT", self.root):
            self.assertEqual(
                [v.path for v in boundary.evaluate(self.policy)], ["research/lane/trace.log.gz"]
            )
            self.assertEqual(
                [v.path for v in boundary.evaluate_commits(self.policy, [bad])],
                ["research/lane/trace.log.gz"],
            )
            refs = boundary.published_refs("refs/heads/**")
            self.assertEqual(
                [v.path for v in boundary.evaluate_refs(self.policy, refs)[0]],
                ["research/lane/trace.log.gz"],
            )

    def test_a_tar_gz_is_judged_by_its_text_members(self) -> None:
        """A tar's regular files are the members; binary content (a NUL in its first 8000 bytes,
        git's own test) is left out, as tensor and float dumps would match by chance too."""
        import gzip
        import io
        import tarfile

        def tar_gz(members: dict[str, bytes]) -> bytes:
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tar:
                for name, body in members.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(body)
                    tar.addfile(info, io.BytesIO(body))
            return gzip.compress(buf.getvalue(), mtime=0)

        quiet = tar_gz({"weights.f32": b"\0\0\x80?alpha needle", "notes.md": b"clean\n"})
        self.assertIsNone(boundary.evaluate_content(self.policy, "research/lane/r.tar.gz", quiet))
        loud = tar_gz({"notes.md": b"clean\n", "src/run.log": b"x\nthe alpha needle\n"})
        violation = boundary.evaluate_content(self.policy, "research/lane/r.tar.gz", loud)
        assert violation is not None
        self.assertEqual(violation.lines, {"alpha": 2})
        self.assertEqual(
            boundary.scan_members(self.policy, boundary.gzip_members(loud) or [])[0][2],
            "src/run.log: the alpha needle",
        )

    def test_a_gzip_magic_that_does_not_decompress_is_judged_raw(self) -> None:
        """A damaged archive is never a blind spot: it keeps the raw-byte scan."""
        blob = b"\x1f\x8b\x08\x00garbage alpha needle\n"
        self.assertIsNone(boundary.gzip_members(blob))
        violation = boundary.evaluate_content(self.policy, "research/lane/bad.log.gz", blob)
        assert violation is not None
        self.assertEqual(violation.rules, ("alpha",))

    # --- memra#1198 review round 1 ---------------------------------------------------------

    def scans(self, path: str) -> list[list[str]]:
        """The paths each of the three scans reports for the tree committed so far."""
        with mock.patch.object(boundary, "ROOT", self.root):
            head = self.git("rev-parse", "HEAD")
            checkout = [v.path for v in boundary.evaluate(self.policy)]
            commits = [v.path for v in boundary.evaluate_commits(self.policy, [head])]
            refs = boundary.published_refs("refs/heads/**")
            at_refs = [v.path for v in boundary.evaluate_refs(self.policy, refs)[0]]
        return [checkout, commits, at_refs]

    def test_the_ref_scan_judges_each_blob_once_and_names_every_carrier(self) -> None:
        """Review blocker: the same gzip blob on several refs is read and judged once."""
        import gzip

        blob = gzip.compress(b"row\nthe alpha needle\n", mtime=0)
        self.commit("research/lane/trace.log.gz", blob, "receipt")
        default = self.git("rev-parse", "--abbrev-ref", "HEAD")
        for branch in ("lane/a", "lane/b"):
            self.git("branch", branch)
        calls = []
        real = boundary.gzip_members

        def counting(data, *args, **kwargs):
            if not args and not kwargs:
                calls.append(len(data))
            return real(data, *args, **kwargs)

        with mock.patch.object(boundary, "ROOT", self.root), mock.patch.object(
            boundary, "gzip_members", counting
        ):
            refs = boundary.published_refs("refs/heads/**")
            violations, carriers = boundary.evaluate_refs(self.policy, refs)
        self.assertEqual(len(refs), 3)
        self.assertEqual(len(calls), 1, "one blob, judged once")
        self.assertEqual([v.path for v in violations], ["research/lane/trace.log.gz"])
        (carried,) = carriers.values()
        self.assertEqual(carried, {default, "lane/a", "lane/b"})

    def test_a_gzip_headers_name_and_comment_are_scanned(self) -> None:
        """Review regression: main scanned the raw bytes, header fields included."""
        import gzip
        import io

        for field in ("filename", "comment"):
            with self.subTest(field=field):
                buf = io.BytesIO()
                kwargs = {"filename": "the alpha needle.log"} if field == "filename" else {}
                with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, **kwargs) as fh:
                    fh.write(b"clean rows\n")
                blob = buf.getvalue()
                if field == "comment":
                    # Python writes no FCOMMENT: set the flag and splice the field in by hand.
                    blob = blob[:3] + bytes([blob[3] | 0x10]) + blob[4:10] + b"the alpha needle\0" + blob[10:]
                members = boundary.gzip_members(blob)
                assert members is not None
                self.assertIn(boundary.GZIP_HEADER_MEMBER, [name for name, _ in members])
                violation = boundary.evaluate_content(self.policy, "research/lane/h.log.gz", blob)
                assert violation is not None
                self.assertEqual(violation.rules, ("alpha",))

    @staticmethod
    def tar_bytes(members: dict[str, bytes]) -> bytes:
        import io
        import tarfile

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for name, body in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(body)
                tar.addfile(info, io.BytesIO(body))
        return buf.getvalue()

    def test_a_gzip_inside_a_tar_gz_is_opened(self) -> None:
        """Review: a nested archive was dropped by the binary probe. It is opened up to three
        levels, and its text members are judged like the outer ones."""
        import gzip

        inner = gzip.compress(self.tar_bytes({"run.log": b"x\nthe alpha needle\n"}), mtime=0)
        outer = gzip.compress(self.tar_bytes({"receipts/inner.tar.gz": inner}), mtime=0)
        members = boundary.gzip_members(outer)
        assert members is not None
        self.assertEqual([name for name, _ in members], ["receipts/inner.tar.gz/run.log"])
        self.assertIsNotNone(boundary.evaluate_content(self.policy, "research/lane/o.tar.gz", outer))

    def test_a_damaged_tar_and_a_tars_tail_are_judged_as_text(self) -> None:
        """Review: a tar that does not parse was skipped whole, and bytes after a tar's end were
        never read. Both are judged as text now, whatever they hold."""
        import gzip

        tar = self.tar_bytes({"notes.md": b"clean\n"})
        tail = gzip.compress(tar + b"\0trailing \0alpha needle\n", mtime=0)
        violation = boundary.evaluate_content(self.policy, "research/lane/t.tar.gz", tail)
        assert violation is not None
        self.assertEqual(
            [name for name, _ in boundary.gzip_members(tail) or []],
            ["notes.md", boundary.TAR_TAIL_MEMBER],
        )
        # A ustar magic at its offset over a header that does not parse.
        damaged = bytearray(tar)
        damaged[148:156] = b"garbage!"  # the header checksum
        damaged[600:620] = b"\0the alpha needle\0\0\0"
        blob = gzip.compress(bytes(damaged), mtime=0)
        self.assertIsNotNone(boundary.evaluate_content(self.policy, "research/lane/d.tar.gz", blob))
        self.assertIsNone(
            boundary.evaluate_content(self.policy, "research/lane/ok.tar.gz", gzip.compress(tar))
        )

    def test_a_gzip_over_the_scan_cap_is_a_finding(self) -> None:
        """Review: no decompressed-size cap. Past it the blob is a finding, never a pass."""
        import gzip

        blob = gzip.compress(b"0" * 5000, mtime=0)
        with mock.patch.object(boundary, "SCAN_CAP_BYTES", 4096):
            violation = boundary.evaluate_content(self.policy, "research/lane/big.log.gz", blob)
        assert violation is not None
        self.assertEqual(violation.rules, (boundary.SCAN_CAP_RULE,))
        self.assertIsNone(boundary.evaluate_content(self.policy, "research/lane/big.log.gz", blob))

    def test_a_gzip_named_blob_that_is_not_gzip_keeps_the_raw_scan_in_every_mode(self) -> None:
        """Review: the raw fallback of a gzip-named non-gzip, through the three scans."""
        self.commit("research/lane/plain.log.gz", b"not gzip: the alpha needle\n", "misnamed")
        self.assertEqual(self.scans("research/lane/plain.log.gz"), [["research/lane/plain.log.gz"]] * 3)

    def test_the_checkout_scan_finds_a_gzip_by_its_magic(self) -> None:
        """Review nit: a gzip blob under another name is opened by the checkout scan too."""
        import gzip

        blob = gzip.compress(b"row\nthe alpha needle\n", mtime=0)
        self.assertFalse(boundary.raw_bytes_prefilter(blob, self.policy.secret_sources))
        self.commit("research/lane/trace.bin", blob, "unnamed gzip")
        checkout, commits, _ = self.scans("research/lane/trace.bin")
        self.assertEqual(checkout, ["research/lane/trace.bin"])
        self.assertEqual(commits, ["research/lane/trace.bin"])

    # --- memra#1198 review round 2 (several from the reviewer's probes) ----------------------

    def refs_scan(self, policy: "boundary.Policy | None" = None):
        with mock.patch.object(boundary, "ROOT", self.root):
            refs = boundary.published_refs("refs/heads/**")
            return boundary.evaluate_refs(policy or self.policy, refs)

    def test_one_gzip_blob_at_two_paths_is_two_verdicts(self) -> None:
        """Review blocker: verdicts keyed by the oid alone reported the last path only."""
        import gzip

        blob = gzip.compress(b"row\nthe alpha needle\n", mtime=0)
        self.write("research/a/trace.log.gz", blob)
        self.write("research/b/trace.log.gz", blob)
        self.git("add", "-f", ".")
        self.git("commit", "-q", "-m", "two copies")
        violations, _ = self.refs_scan()
        self.assertEqual(
            sorted(v.path for v in violations),
            ["research/a/trace.log.gz", "research/b/trace.log.gz"],
        )

    def test_a_gzip_renamed_on_a_branch_names_each_paths_carriers(self) -> None:
        """Review blocker: an old path on an old ref must not hide behind an allowlisted new one."""
        import gzip

        blob = gzip.compress(b"row\nthe alpha needle\n", mtime=0)
        self.commit("research/a/trace.log.gz", blob, "receipt")
        default = self.git("rev-parse", "--abbrev-ref", "HEAD")
        self.git("checkout", "-q", "-b", "lane/x")
        self.git("rm", "-q", "research/a/trace.log.gz")
        self.write("research/b/trace.log.gz", blob)
        self.git("add", "-f", "research/b/trace.log.gz")
        self.git("commit", "-q", "-m", "rename")
        self.git("checkout", "-q", default)
        _, carriers = self.refs_scan()
        self.assertEqual(
            {key[0]: refs for key, refs in carriers.items()},
            {"research/a/trace.log.gz": {default}, "research/b/trace.log.gz": {"lane/x"}},
        )

    def test_a_private_path_sharing_a_gzips_oid_keeps_its_violation(self) -> None:
        """Review blocker: the public copy's verdict stood in for the private path's."""
        import gzip

        blob = gzip.compress(b"clean rows\n", mtime=0)
        self.write("private/x.log.gz", blob)
        self.write("research/x.log.gz", blob)
        self.git("add", "-f", ".")
        self.git("commit", "-q", "-m", "copies")
        policy = boundary.Policy(
            secret_patterns=self.policy.secret_patterns,
            secret_sources=self.policy.secret_sources,
            secret_union=self.policy.secret_union,
            secret_groups=self.policy.secret_groups,
            private_paths={"priv": "private/**"},
            bypass_paths=[],
        )
        violations, _ = self.refs_scan(policy)
        self.assertEqual(
            [(v.path, v.category) for v in violations], [("private/x.log.gz", "private_path")]
        )

    def test_the_cap_counts_a_tars_bytes_once(self) -> None:
        """Review: the tar's members were charged on top of the tar's own decompressed bytes."""
        import gzip

        # A 6,000-byte member, then a small gzip whose decompression draws on what is left.
        tarred = self.tar_bytes(
            {"run.log": b"x" * 6000, "rows.log.gz": gzip.compress(b"clean\n" * 50, mtime=0)}
        )
        self.assertEqual(len(tarred), 10240)
        blob = gzip.compress(tarred, mtime=0)
        with mock.patch.object(boundary, "SCAN_CAP_BYTES", 15000):
            self.assertIsNone(boundary.evaluate_content(self.policy, "research/l/c.tar.gz", blob))
        with mock.patch.object(boundary, "SCAN_CAP_BYTES", 10000):
            violation = boundary.evaluate_content(self.policy, "research/l/c.tar.gz", blob)
        assert violation is not None
        self.assertEqual(violation.rules, (boundary.SCAN_CAP_RULE,))

    def test_a_sparse_tar_member_is_charged_against_the_cap(self) -> None:
        """A sparse member expands past its stored bytes, so it is charged before it is read."""
        import gzip
        import tarfile

        blob = gzip.compress(self.tar_bytes({"run.log": b"x" * 6000}), mtime=0)
        with mock.patch.object(boundary, "SCAN_CAP_BYTES", 12000), mock.patch.object(
            tarfile.TarInfo, "issparse", lambda _info: True
        ):
            violation = boundary.evaluate_content(self.policy, "research/l/s.tar.gz", blob)
        assert violation is not None
        self.assertEqual(violation.rules, (boundary.SCAN_CAP_RULE,))

    def test_an_over_cap_gzip_is_a_finding_in_every_scan(self) -> None:
        """Review: the batch path's over-cap verdict was dropped by no test."""
        import gzip

        self.commit("research/l/big.log.gz", gzip.compress(b"0" * 5000, mtime=0), "big")
        with mock.patch.object(boundary, "SCAN_CAP_BYTES", 4096):
            self.assertEqual(self.scans("research/l/big.log.gz"), [["research/l/big.log.gz"]] * 3)

    def test_a_gzip_headers_extra_field_is_scanned(self) -> None:
        import gzip

        body = gzip.compress(b"clean rows\n", mtime=0)
        extra = b"the alpha needle"
        blob = (
            body[:3]
            + bytes([body[3] | 0x04])
            + body[4:10]
            + len(extra).to_bytes(2, "little")
            + extra
            + body[10:]
        )
        self.assertIsNotNone(boundary.evaluate_content(self.policy, "research/l/e.log.gz", blob))

    def test_three_nested_tar_gz_levels_are_opened(self) -> None:
        import gzip

        level = self.tar_bytes({"run.log": b"x\nthe alpha needle\n"})
        for i in range(3):
            level = self.tar_bytes({f"l{i}.tar.gz": gzip.compress(level, mtime=0)})
        blob = gzip.compress(level, mtime=0)
        self.assertIsNotNone(boundary.evaluate_content(self.policy, "research/l/n.tar.gz", blob))
        self.assertIn(
            "l2.tar.gz/l1.tar.gz/l0.tar.gz/run.log",
            [name for name, _ in boundary.gzip_members(blob) or []],
        )

    def test_a_gzip_past_the_nesting_limit_is_judged_by_its_bytes(self) -> None:
        """Review round 3: GZIP_NEST_DEPTH is 7, the outer tar.gz's tar and three nested tar.gz
        levels. A gzip in the innermost tar is the eighth container: it is kept as its bytes."""
        import gzip

        level = self.tar_bytes({"run.log.gz": gzip.compress(b"x\nthe alpha needle\n", mtime=0)})
        for i in range(3):
            level = self.tar_bytes({f"l{i}.tar.gz": gzip.compress(level, mtime=0)})
        members = dict(boundary.gzip_members(gzip.compress(level, mtime=0)) or [])
        body = members["l2.tar.gz/l1.tar.gz/l0.tar.gz/run.log.gz"]
        self.assertEqual(body[:2], boundary.GZIP_MAGIC)

    def test_a_nested_gzip_that_does_not_decompress_is_judged_by_its_bytes(self) -> None:
        """Review round 3: a top-level gzip that does not decompress keeps the raw scan; inside a
        tar.gz the same blob was dropped as binary (its header holds a NUL)."""
        import gzip

        inner = gzip.compress(b"the alpha needle\n" + b"pad row\n" * 400, compresslevel=0, mtime=0)
        broken = inner[: len(inner) // 2]
        self.assertIsNotNone(boundary.evaluate_content(self.policy, "research/l/top.log.gz", broken))
        outer = gzip.compress(self.tar_bytes({"x.log.gz": broken}), mtime=0)
        self.assertIsNotNone(boundary.evaluate_content(self.policy, "research/l/o.tar.gz", outer))

    def test_nested_plain_tars_stay_near_the_cap(self) -> None:
        """Review round 3: each plain tar opened from a tar member is a copy no decompression
        charged, so seven nested tars held about one copy of the cap per level. Charged, the scan
        stops past the cap with a finding, and its peak stays near the cap."""
        import gzip
        import tracemalloc

        cap = 1 << 20
        # A leaf of about 40% of the cap: one charged copy fits, the second does not, so the scan
        # stops only when each copy is also subtracted from the budget (review round 4).
        level = self.tar_bytes({"run.log": b"row of clean text\n" * ((410 << 10) // 18)})
        for i in range(6):
            level = self.tar_bytes({f"l{i}.tar": level})
        blob = gzip.compress(level, mtime=0)
        with mock.patch.object(boundary, "SCAN_CAP_BYTES", cap):
            tracemalloc.start()
            try:
                violation = boundary.evaluate_content(self.policy, "research/l/t.tar.gz", blob)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
        assert violation is not None
        self.assertEqual(violation.rules, (boundary.SCAN_CAP_RULE,))
        self.assertLess(peak, 3 * cap)

    def test_every_member_of_a_multi_member_gzip_is_read(self) -> None:
        import gzip

        blob = gzip.compress(b"clean\n", mtime=0) + gzip.compress(b"the alpha needle\n", mtime=0)
        self.assertIsNotNone(boundary.evaluate_content(self.policy, "research/l/m.log.gz", blob))

    def test_a_gzip_in_a_gzip_a_tar_in_a_tar_gz_and_a_v7_tar_are_opened(self) -> None:
        """Review nits: all three were dropped by the binary probe."""
        import gzip

        twice = gzip.compress(gzip.compress(b"row\nthe alpha needle\n", mtime=0), mtime=0)
        self.assertIsNotNone(boundary.evaluate_content(self.policy, "research/l/x.log.gz.gz", twice))
        inner = self.tar_bytes({"run.log": b"x\nthe alpha needle\n"})
        outer = gzip.compress(self.tar_bytes({"inner.tar": inner}), mtime=0)
        self.assertIsNotNone(boundary.evaluate_content(self.policy, "research/l/o.tar.gz", outer))
        raw = bytearray(self.tar_bytes({"run.log": b"x\nthe alpha needle\n"}))
        raw[257:265] = b"\0" * 8  # no ustar magic or version: a v7 header
        header = raw[:512]
        header[148:156] = b" " * 8
        raw[148:156] = b"%06o\0 " % sum(header)
        v7 = gzip.compress(bytes(raw), mtime=0)
        self.assertIsNotNone(boundary.evaluate_content(self.policy, "research/l/v7.tar.gz", v7))
        self.assertFalse(boundary.looks_like_tar(b"plain text\n" * 100))

    def test_the_scratch_dir_outside_a_repository_is_the_systems(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as outside, mock.patch.object(
            boundary, "ROOT", boundary.Path(outside)
        ):
            self.assertIsNone(boundary.scratch_dir())

    def test_ref_scan_finds_a_blob_the_checkout_no_longer_carries(self) -> None:
        """The gap that let two lanes stay unexamined: nothing re-reads a pushed branch.

        The checkout-based prefilter only ever sees the working tree, and the range check only
        sees blobs a push introduces, so a rule added after the push never looks again. A blob
        that survives only on another ref has to be found through the ref.
        """
        self.commit("receipts.md", b"clean\n", "clean default branch")
        default = self.git("rev-parse", "--abbrev-ref", "HEAD")
        self.git("checkout", "-q", "-b", "lane/stale")
        self.commit("receipts.md", b"alpha needle\n", "lane evidence")
        self.git("checkout", "-q", default)

        with mock.patch.object(boundary, "ROOT", self.root):
            # The checkout is clean, which is exactly why the existing modes stayed quiet.
            self.assertEqual(
                boundary.secret_candidate_files(self.policy.secret_sources), set()
            )
            refs = boundary.published_refs("refs/heads/**")
            violations, carriers = boundary.evaluate_refs(self.policy, refs)

        self.assertEqual({boundary.short_ref(r) for r in refs}, {default, "lane/stale"})
        self.assertEqual(
            [(v.path, v.detail) for v in violations],
            [("receipts.md", "alpha (first hit line 1)")],
        )
        # Reported with the ref to go clean up, and deduplicated by content across refs.
        self.assertEqual(
            carriers[(violations[0].path, violations[0].sha256)], {"lane/stale"}
        )

    def test_ref_scan_deduplicates_shared_history_across_refs(self) -> None:
        commit = self.commit("receipts.md", b"alpha needle\n", "shared evidence")
        self.git("branch", "lane/one", commit)
        self.git("branch", "lane/two", commit)
        with mock.patch.object(boundary, "ROOT", self.root):
            refs = boundary.published_refs("refs/heads/**")
            violations, carriers = boundary.evaluate_refs(self.policy, refs)
        self.assertEqual(len(refs), 3)
        self.assertEqual(len(violations), 1)  # one blob, not one per ref
        self.assertEqual(
            carriers[(violations[0].path, violations[0].sha256)],
            {self.git("rev-parse", "--abbrev-ref", "HEAD"), "lane/one", "lane/two"},
        )

    def test_ref_scan_covers_tags_not_only_heads(self) -> None:
        """Tags were out of scope until 2026-08-19 and carried MORE violations than heads.

        The old default glob was `refs/remotes/origin/**`, which cannot match `refs/tags/*`, and
        the workflow fetched `--no-tags` on top of that — so the tag namespace was reported clean
        by a scan that never opened it. Tags are also the worst place for a leak: immutable
        release markers are what forks and consumers pull, so detection is the only lever.
        """
        self.commit("receipts.md", b"clean\n", "clean default branch")
        default = self.git("rev-parse", "--abbrev-ref", "HEAD")
        tagged = self.commit("receipts.md", b"alpha needle\n", "tagged evidence")
        self.git("tag", "v9.9.9", tagged)
        self.git("reset", "-q", "--hard", "HEAD~1")

        with mock.patch.object(boundary, "ROOT", self.root):
            # Heads-only: the blob survives on a tag alone, so a heads glob proves nothing.
            heads_only = boundary.published_refs("refs/heads/**")
            found, _ = boundary.evaluate_refs(self.policy, heads_only)
            self.assertEqual(found, [])
            both = boundary.published_refs("refs/heads/**,refs/tags/**")
            violations, carriers = boundary.evaluate_refs(self.policy, both)

        self.assertEqual({boundary.short_ref(r) for r in both}, {default, "tags/v9.9.9"})
        self.assertEqual([v.path for v in violations], ["receipts.md"])
        # The tag namespace is kept in the carrier name: a tag is not a branch to remediate.
        self.assertEqual(
            carriers[(violations[0].path, violations[0].sha256)], {"tags/v9.9.9"}
        )
        spread = boundary.ref_namespaces(both)
        self.assertEqual((spread["heads"], spread["tags"]), (1, 1))

    def test_default_ref_glob_spans_heads_tags_and_pull_refs(self) -> None:
        self.assertIn("refs/tags/", boundary.DEFAULT_REF_GLOB)
        self.assertIn("origin-pull", boundary.DEFAULT_REF_GLOB)
        self.assertIn("refs/remotes/origin/", boundary.DEFAULT_REF_GLOB)
        spread = boundary.ref_namespaces(
            [
                "refs/remotes/origin/main",
                "refs/tags/v0.1.0",
                "refs/remotes/origin-pull/7",
            ]
        )
        self.assertEqual(
            (spread["heads"], spread["tags"], spread["pull"], spread["other"]), (1, 1, 1, 0)
        )

    def test_ref_scan_with_no_matching_ref_fails_closed(self) -> None:
        """"Nothing matched" must not be indistinguishable from "nothing found"."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rc = boundary.cmd_check(self.policy, refs=[])
        self.assertEqual(rc, 1)
        self.assertIn("no refs matched", output.getvalue())

    def test_ref_scan_reports_a_private_path_only_an_old_ref_carries(self) -> None:
        """memra#1356: git ls-tree refused the ':(glob)' pathspec, so the ref scan never found a
        private path by its glob. A clean private file is found on every ref that carries it,
        one verdict per version, and a public ref without it stays clean."""
        self.policy.private_paths["private"] = "deploy/private/**"
        self.commit("receipts.md", b"clean\n", "clean default branch")
        default = self.git("rev-parse", "--abbrev-ref", "HEAD")
        self.git("checkout", "-q", "-b", "lane/old")
        self.commit("deploy/private/unit/host.service", b"clean v1\n", "private v1")
        self.git("branch", "lane/older")
        self.commit("deploy/private/unit/host.service", b"clean v2\n", "private v2")
        self.git("checkout", "-q", default)
        with mock.patch.object(boundary, "ROOT", self.root):
            self.assertEqual(
                boundary.evaluate_refs(self.policy, boundary.published_refs(f"refs/heads/{default}"))[0],
                [],
            )
            violations, carriers = boundary.evaluate_refs(
                self.policy, boundary.published_refs("refs/heads/**")
            )
        self.assertEqual(
            sorted((v.path, v.category) for v in violations),
            [("deploy/private/unit/host.service", "private_path")] * 2,
        )
        self.assertEqual(
            sorted(sorted(carriers[(v.path, v.sha256)]) for v in violations),
            [["lane/old"], ["lane/older"]],
        )

    def test_a_failed_ls_tree_stops_the_ref_scan(self) -> None:
        """memra#1356: a refused `git ls-tree` returned [] and hid the bug; it now raises."""
        self.commit("receipts.md", b"clean\n", "clean")
        with mock.patch.object(boundary, "ROOT", self.root), self.assertRaises(RuntimeError):
            boundary.tree_candidates_at_refs(self.policy, ["refs/heads/no-such-ref"])

    def test_the_ref_scan_takes_each_paths_glob_verdict_once(self) -> None:
        """memra#1356: refs repeat their paths, so the private-glob verdict is taken once per path,
        not once per (ref, path); per entry, it made the ls-tree pass about 9 times slower."""
        self.policy.private_paths["private"] = "deploy/private/**"
        self.commit("a.md", b"clean\n", "one")
        self.commit("deploy/private/b.service", b"clean\n", "two")
        for name in ("lane/x", "lane/y", "lane/z"):
            self.git("branch", name)
        calls: list = []
        real = boundary.glob_matcher

        class Counted:
            def __init__(self, pattern) -> None:
                self.pattern = pattern

            def match(self, path: str):
                calls.append(path)
                return self.pattern.match(path)

        with mock.patch.object(boundary, "ROOT", self.root), mock.patch.object(
            boundary, "glob_matcher", lambda globs: Counted(real(globs))
        ):
            groups = boundary.tree_candidates_at_refs(
                self.policy, boundary.published_refs("refs/heads/**")
            )
        self.assertEqual(sorted(path for path, _ in groups), ["deploy/private/b.service"])
        self.assertEqual(len(next(iter(groups.values()))), 4)
        self.assertEqual(sorted(calls), ["a.md", "deploy/private/b.service"])

    def test_the_private_glob_regex_agrees_with_matches_glob(self) -> None:
        """Review of memra#1356: the ref scan's compiled regex and the checkout scan's
        `matches_glob` give the same verdict, on the real policy's globs and near-misses of
        each (a suffix, a parent directory, a sibling directory, a prefix)."""
        globs = list(boundary.load_policy(boundary.POLICY_PATH).private_paths.values())
        globs += ["research/*.log", "deploy/a*/b/**", "x?y/**"]
        paths = ["", "a", "deploy/", "research/a/b.log", "xzy/q", "xy/q", "deploy/ab/b/c"]
        for glob in globs:
            stem = glob.rstrip("*")
            paths += [
                glob, stem, stem + "x", stem + "/x", stem.rstrip("/") + "X/y",
                "a/" + glob, glob + ".orig", stem[:-1], stem + "a/b/c.service",
            ]
        regex = boundary.glob_matcher(globs)
        assert regex is not None
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual(
                    regex.match(path) is not None,
                    any(boundary.matches_glob(path, glob) for glob in globs),
                )
        self.assertIsNone(boundary.glob_matcher([]))

    def test_exact_private_paths_and_their_near_misses_match_in_both_scans(self) -> None:
        """Review of memra#1356: 13 of the 14 production rules are exact paths. The checkout scan
        and the ref scan report the same private paths, and no near-miss."""
        self.policy.private_paths.update({"priv": "tools/priv.sh", "dir": "deploy/private/**"})
        for path in (
            "tools/priv.sh", "tools/priv.sh.orig", "a/tools/priv.sh", "tools/priv.s",
            "deploy/private/u.service", "deploy/privateX/u.service", "deploy/private.md",
        ):
            self.write(path, b"clean\n")
        self.git("add", "-f", ".")
        self.git("commit", "-q", "-m", "near misses")
        with mock.patch.object(boundary, "ROOT", self.root):
            checkout = sorted(v.path for v in boundary.evaluate(self.policy))
            refs, _ = boundary.evaluate_refs(self.policy, boundary.published_refs("refs/heads/**"))
        self.assertEqual(checkout, ["deploy/private/u.service", "tools/priv.sh"])
        self.assertEqual(sorted(v.path for v in refs), checkout)

    def test_a_symlink_at_a_private_path_is_a_ref_scan_finding(self) -> None:
        """Review of memra#1356: a symlink (mode 120000) is a blob whose content is its target, and
        under a private glob it is a private path in the ref scan as in the checkout scan."""
        self.policy.private_paths["private"] = "deploy/private/**"
        self.symlink("deploy/private/link", "/etc/passwd")
        self.git("add", "deploy/private/link")
        self.git("commit", "-q", "-m", "a link at a private path")
        with mock.patch.object(boundary, "ROOT", self.root):
            checkout = [(v.path, v.category) for v in boundary.evaluate(self.policy)]
            refs, _ = boundary.evaluate_refs(self.policy, boundary.published_refs("refs/heads/**"))
        self.assertEqual(checkout, [("deploy/private/link", "private_path")])
        self.assertEqual([(v.path, v.category) for v in refs], checkout)

    def test_commit_scan_applies_private_path_rules(self) -> None:
        self.policy.private_paths["private"] = "private/**"
        commit = self.commit("private/evidence.txt", b"safe\n", "private path")
        with mock.patch.object(boundary, "ROOT", self.root):
            violations = boundary.evaluate_commits(self.policy, [commit])
        self.assertEqual(
            [(violation.path, violation.category) for violation in violations],
            [("private/evidence.txt", "private_path")],
        )


class VerifyAllowlistTests(unittest.TestCase):
    """Teeth for `check-public-boundary.py verify-allowlist`.

    Until 2026-08-23 this mode had NO automated caller anywhere — not `ci.yml`, not
    `boundary-refs.yml`, not the pre-push hook — and no test in this file touched `cmd_verify`.
    The only evidence it had ever passed was two transcript lines in
    `research/public-boundary-20260814/PROGRESS.md`. That is a gate governing what the public
    repository is permitted to publish, with 685 pinned entries, running only when a human
    remembered.

    The allowlist is the half of the boundary policy that decides what is EXEMPT, so it is the
    half that fails silently: an entry whose file was edited, deleted or remediated keeps sitting
    in the list looking like a decision someone made, while `check` never consults it again for
    bytes that no longer exist. `verify-allowlist` is what notices; nothing was running it.

    These arms drive the REAL `cmd_verify` against a REAL throwaway tree — real `git ls-files`,
    real `evaluate()`, a real allowlist file on disk — via the `mock.patch.object(boundary,
    "ROOT", ...)` idiom `CommitBlobTests` above already uses. Nothing about the verify logic is
    reimplemented here, or these tests would stay green while the gate rotted.
    """

    def setUp(self) -> None:
        # `ignore_cleanup_errors`: these suites build a REAL git repo in the tempdir, and git may
        # touch `.git` from a background task while rmtree is walking it — which surfaced on CI as
        # `OSError: [Errno 39] Directory not empty: '.git'` raised from TemporaryDirectory cleanup
        # AFTER every assertion had already passed, reddening the build for nothing. The two
        # `config` lines below remove the cause (no auto-gc, no background maintenance in a
        # throwaway repo); the flag makes the whole class of cleanup race non-fatal, because a
        # leaked temp dir is never a reason to fail a policy gate.
        self.tempdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Boundary Test")
        self.git("config", "user.email", "boundary@example.invalid")
        self.git("config", "gc.auto", "0")
        self.git("config", "maintenance.auto", "false")
        self.allowlist_path = self.root / "tools" / "public-boundary-allowlist.jsonl"
        self.allowlist_path.parent.mkdir(parents=True, exist_ok=True)
        # Two independent rules on distinct needles, so a rule-scoped grant can have exactly one
        # of its rules remediated while the bytes still trip the other.
        self.policy = boundary.Policy(
            secret_patterns={
                "alpha": boundary.re.compile("alpha needle"),
                "beta": boundary.re.compile("beta needle"),
            },
            secret_sources={"alpha": "alpha needle", "beta": "beta needle"},
            secret_union=boundary.re.compile(
                "(?P<rule_0>alpha needle)|(?P<rule_1>beta needle)"
            ),
            secret_groups={"rule_0": "alpha", "rule_1": "beta"},
            private_paths={},
            bypass_paths=[],
        )

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def commit(self, path: str, data: bytes) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.git("add", "-f", path)
        self.git("commit", "-q", "-m", f"write {path}")

    def pin(self, path: str, rules: list[str]) -> dict:
        """The allowlist row for `path` AS IT CURRENTLY IS ON DISK.

        Hashing the live bytes rather than accepting a caller-supplied digest is what makes the
        corrupt-and-restore arm honest: the "clean" state is pinned to real content, so a later
        edit has to be what breaks it.
        """
        digest = boundary.hashlib.sha256((self.root / path).read_bytes()).hexdigest()
        return {
            "path": path,
            "sha256": digest,
            "category": "secret_pattern",
            "rules": rules,
            "reason": "pinned by the verify fixture",
        }

    def write_allowlist(self, *entries: dict) -> None:
        lines = ["# fixture allowlist"]
        lines += [json.dumps(item, sort_keys=True) for item in entries]
        self.allowlist_path.write_text("\n".join(lines) + "\n")

    def run_verify(self, prune: bool = False) -> tuple[int, str]:
        output = io.StringIO()
        with (
            mock.patch.object(boundary, "ROOT", self.root),
            mock.patch.object(boundary, "ALLOWLIST_PATH", self.allowlist_path),
            contextlib.redirect_stdout(output),
        ):
            rc = boundary.cmd_verify(self.policy, prune)
        return rc, output.getvalue()

    def test_drift_reads_only_pinned_paths_but_check_still_finds_new_violations(self) -> None:
        self.commit("deploy/pinned.txt", b"alpha needle\n")
        self.commit("deploy/new.txt", b"beta needle\n")
        self.write_allowlist(self.pin("deploy/pinned.txt", ["alpha"]))
        with mock.patch.object(boundary, "worktree_blob_bytes", wraps=boundary.worktree_blob_bytes) as reads:
            rc, output = self.run_verify()
        self.assertEqual(rc, 0, output)
        self.assertEqual([call.args[0].relative_to(self.root).as_posix() for call in reads.call_args_list],
                         ["deploy/pinned.txt"])
        full_output = io.StringIO()
        with mock.patch.object(boundary, "ROOT", self.root), \
                mock.patch.object(boundary, "ALLOWLIST_PATH", self.allowlist_path), \
                contextlib.redirect_stdout(full_output):
            self.assertEqual(boundary.cmd_check(self.policy), 1)
        self.assertIn("deploy/new.txt", full_output.getvalue())

    def test_pinned_path_is_literal_not_a_git_glob(self) -> None:
        self.commit("deploy/a[1].txt", b"alpha needle\n")
        self.commit("deploy/a1.txt", b"beta needle\n")
        self.write_allowlist(self.pin("deploy/a[1].txt", ["alpha"]))
        rc, output = self.run_verify()
        self.assertEqual(rc, 0, output)
        self.assertIn("1 allowlist entries", output)

    def test_git_magic_prefix_is_a_literal_tracked_filename(self) -> None:
        literal = ":(glob)deploy/pinned.txt"
        target = self.root / literal
        target.parent.mkdir(parents=True)
        target.write_bytes(b"alpha needle\n")
        self.git("--literal-pathspecs", "add", literal)
        self.git("commit", "-q", "-m", "literal magic-looking filename")
        self.commit("deploy/pinned.txt", b"beta needle\n")
        self.write_allowlist(self.pin(literal, ["alpha"]))
        rc, output = self.run_verify()
        self.assertEqual(rc, 0, output)
        self.assertIn("1 allowlist entries", output)

    def mirror_origin(self, pull: int = 1, fetched_pull: int | None = None) -> None:
        """Make this repo look like a clone that mirrors origin, as boundary-refs.yml fetches it:
        the repo is its own `origin` remote, each branch is at refs/remotes/origin/*, and `pull`
        PR heads (refs/pull/N/head, which `git ls-remote origin` lists) of which `fetched_pull`
        are at refs/remotes/origin-pull/*. The tags are the repo's own."""
        if "origin" not in self.git("remote").split():
            self.git("remote", "add", "origin", str(self.root))
        for branch in self.git("for-each-ref", "--format=%(refname:short)", "refs/heads").split():
            self.git("update-ref", f"refs/remotes/origin/{branch}", f"refs/heads/{branch}")
        head = self.git("rev-parse", "HEAD")
        for n in range(1, pull + 1):
            self.git("update-ref", f"refs/pull/{n}/head", head)
            if n <= (pull if fetched_pull is None else fetched_pull):
                self.git("update-ref", f"refs/remotes/origin-pull/{n}", head)

    def historical(self, path: str, data: bytes, rules: list[str]) -> dict:
        """A `refs_only` row for `data` at `path`: a version the checkout no longer carries."""
        return {
            "path": path,
            "sha256": boundary.hashlib.sha256(data).hexdigest(),
            "category": "secret_pattern",
            "rules": rules,
            "reason": "a reviewed historical version",
            "scope": "refs_only",
        }

    def run_check(
        self, refs: list[str] | None = None, commits: list[str] | None = None
    ) -> tuple[int, str]:
        output = io.StringIO()
        with (
            mock.patch.object(boundary, "ROOT", self.root),
            mock.patch.object(boundary, "ALLOWLIST_PATH", self.allowlist_path),
            contextlib.redirect_stdout(output),
        ):
            rc = boundary.cmd_check(self.policy, commits=commits, refs=refs)
        return rc, output.getvalue()

    def assert_refs_only_section(self) -> None:
        """The file's refs_only rows sit together, last, under REFS_ONLY_SECTION, given once."""
        lines = self.allowlist_path.read_text().splitlines()
        section = list(boundary.REFS_ONLY_SECTION)
        self.assertEqual(lines.count(section[0]), 1, lines)
        at = lines.index(section[0])
        self.assertEqual(lines[at : at + len(section)], section)
        rows = [json.loads(ln) for ln in lines if ln.startswith("{")]
        scoped = [ln.startswith("{") and json.loads(ln).get("scope") == "refs_only" for ln in lines]
        self.assertTrue(all(scoped[at + len(section) :]), lines)
        self.assertFalse(any(scoped[:at]), lines)
        self.assertTrue(any(row.get("scope") == "refs_only" for row in rows))

    def tag_old_version(self) -> bytes:
        """`deploy/old.txt` trips alpha at tag v1 and is clean in the checkout."""
        old = b"alpha needle\n"
        self.commit("deploy/old.txt", old)
        self.git("tag", "v1")
        self.commit("deploy/old.txt", b"clean\n")
        return old

    def test_a_refs_only_entry_exempts_the_ref_scan_and_never_the_tree(self) -> None:
        """A reviewed historical version is grandfathered at the refs that carry it. The checkout
        scan neither calls it stale nor lets it cover the same bytes coming back into the tree."""
        old = self.tag_old_version()
        with mock.patch.object(boundary, "ROOT", self.root):
            tags = boundary.published_refs("refs/tags/**")
        self.write_allowlist()
        self.assertEqual(self.run_check(tags)[0], 1)
        self.write_allowlist(self.historical("deploy/old.txt", old, ["alpha"]))
        rc, output = self.run_check(tags)
        self.assertEqual(rc, 0, output)
        self.assertIn("(1 grandfathered, 0 new)", output)
        rc, output = self.run_check()
        self.assertEqual(rc, 0, output)
        self.commit("deploy/old.txt", old)
        rc, output = self.run_check()
        self.assertEqual(rc, 1, output)
        self.assertIn("deploy/old.txt", output)

    def test_a_commit_that_brings_back_refs_only_bytes_is_a_finding(self) -> None:
        """Review of memra#1368: the pre-push hook's commit scan, like the checkout scan, never
        honours a refs_only entry. The same bytes pushed again are a new finding."""
        old = self.tag_old_version()
        self.write_allowlist(self.historical("deploy/old.txt", old, ["alpha"]))
        self.commit("deploy/old.txt", old)
        rc, output = self.run_check(commits=[self.git("rev-parse", "HEAD")])
        self.assertEqual(rc, 1, output)
        self.assertIn("deploy/old.txt", output)

    def test_a_version_only_on_a_local_branch_is_at_no_published_ref(self) -> None:
        """Review of memra#1368: a local head is not published. Only origin's heads (as
        refs/remotes/origin/*), its PR heads and the tags count."""
        old = b"alpha needle\n"
        self.commit("deploy/old.txt", old)
        self.git("branch", "x")
        self.commit("deploy/old.txt", b"clean\n")
        self.git("tag", "v1")
        self.write_allowlist(self.historical("deploy/old.txt", old, ["alpha"]))
        rc, output = self.run_verify()
        self.assertEqual(rc, 1, output)
        self.assertIn("at no published ref", output)

    def test_verify_with_no_published_ref_fails_closed(self) -> None:
        """Review of memra#1368: no published ref here means no refs_only entry was judged."""
        old = b"alpha needle\n"
        self.commit("deploy/old.txt", old)
        self.commit("deploy/old.txt", b"clean\n")
        self.write_allowlist(self.historical("deploy/old.txt", old, ["alpha"]))
        rc, output = self.run_verify()
        self.assertEqual(rc, 1, output)
        self.assertIn("no refs matched", output)

    def test_prune_keeps_refs_only_entries_unless_the_refs_mirror_origin(self) -> None:
        """Review of memra#1368: a clone without origin's PR heads drifts every refs_only entry
        those refs carry, so --prune there keeps them and fails; a clone with origin's heads and
        PR heads prunes a drifted one."""
        self.tag_old_version()
        never = self.historical("deploy/old.txt", b"alpha needle, never pushed\n", ["alpha"])
        self.write_allowlist(never)
        rc, output = self.run_verify(prune=True)
        self.assertEqual(rc, 1, output)
        self.assertIn("kept 1 refs_only entries unpruned", output)
        with mock.patch.object(boundary, "ROOT", self.root):
            self.assertEqual(len(boundary.load_allowlist(self.allowlist_path)), 1)
        # Review of memra#1368 round 2: a plain clone (origin's branches and the tags, no PR
        # heads) keeps it too.
        self.mirror_origin(pull=0)
        self.write_allowlist(never)
        rc, output = self.run_verify(prune=True)
        self.assertEqual(rc, 1, output)
        self.assertIn("kept 1 refs_only entries unpruned: the published refs here lack one of", output)
        # One PR head fetched of two that origin lists: the shape holds, the counts do not.
        self.mirror_origin(pull=2, fetched_pull=1)
        rc, output = self.run_verify(prune=True)
        self.assertEqual(rc, 1, output)
        self.assertIn("(pull heads: 1 here, 2 on origin)", output)
        self.mirror_origin(pull=2)
        rc, output = self.run_verify(prune=True)
        self.assertEqual(rc, 0, output)
        with mock.patch.object(boundary, "ROOT", self.root):
            self.assertEqual(boundary.load_allowlist(self.allowlist_path), {})

    def test_prune_without_an_origin_remote_keeps_refs_only_entries(self) -> None:
        """Review of memra#1368 round 2: the refs' counts are compared with `git ls-remote
        origin`; when that cannot run, the set is not known to be whole."""
        self.tag_old_version()
        self.mirror_origin()
        # Removing the remote would drop refs/remotes/origin/* with it; point it nowhere instead.
        self.git("remote", "set-url", "origin", str(self.root / "no-such-repo"))
        self.write_allowlist(self.historical("deploy/old.txt", b"never pushed\n", ["alpha"]))
        rc, output = self.run_verify(prune=True)
        self.assertEqual(rc, 1, output)
        self.assertIn("`git ls-remote origin` failed", output)

    def test_prune_on_a_plain_clone_still_prunes_checkout_entries(self) -> None:
        """Review of memra#1368 round 2: on a plain clone a refs_only entry that did NOT drift is
        not "kept unpruned", and a drifted checkout entry still drops, with rc 0."""
        old = self.tag_old_version()
        self.commit("deploy/gone.txt", b"beta needle\n")
        gone = self.pin("deploy/gone.txt", ["beta"])
        self.commit("deploy/gone.txt", b"clean\n")
        self.mirror_origin(pull=0)
        kept = self.historical("deploy/old.txt", old, ["alpha"])
        self.write_allowlist(kept, gone)
        rc, output = self.run_verify(prune=True)
        self.assertEqual(rc, 0, output)
        self.assertIn("dropped 1 entries", output)
        self.assertNotIn("unpruned", output)
        with mock.patch.object(boundary, "ROOT", self.root):
            self.assertEqual(list(boundary.load_allowlist(self.allowlist_path)), [("deploy/old.txt", kept["sha256"])])

    def test_the_ref_set_shape_needs_each_published_namespace(self) -> None:
        """Review of memra#1368 round 2: origin's branches, its PR heads and the tags, each."""
        whole = ["refs/remotes/origin/main", "refs/remotes/origin-pull/7", "refs/tags/v1"]
        self.assertTrue(boundary.ref_set_looks_complete(whole))
        for missing in whole:
            with self.subTest(missing=missing):
                self.assertFalse(
                    boundary.ref_set_looks_complete([ref for ref in whole if ref != missing])
                )

    def test_remote_ref_shortfall_counts_each_namespace(self) -> None:
        listing = "\n".join(
            f"{'0' * 40}\t{name}"
            for name in (
                "HEAD",
                "refs/heads/main",
                "refs/heads/lane",
                "refs/tags/v1",
                "refs/tags/v1^{}",
                "refs/pull/1/head",
                "refs/pull/1/merge",
                "refs/pull/2/head",
            )
        )
        refs = [
            "refs/remotes/origin/HEAD",
            "refs/remotes/origin/main",
            "refs/remotes/origin/lane",
            "refs/remotes/origin-pull/1",
            "refs/remotes/origin-pull/2",
            "refs/tags/v1",
        ]
        self.assertEqual(boundary.remote_ref_shortfall(refs, listing), [])
        self.assertEqual(
            boundary.remote_ref_shortfall(
                [ref for ref in refs if ref not in ("refs/remotes/origin/lane", "refs/tags/v1")], listing
            ),
            ["heads: 1 here, 2 on origin", "tags: 0 here, 1 on origin"],
        )
        self.assertEqual(
            boundary.remote_ref_shortfall(refs[:2] + refs[3:], listing),
            ["heads: 1 here, 2 on origin"],
        )
        self.assertEqual(
            boundary.remote_ref_shortfall(refs[:4] + refs[5:], listing),
            ["pull heads: 1 here, 2 on origin"],
        )

    def test_verify_fails_on_a_secret_inside_a_refs_only_private_version(self) -> None:
        """Review of memra#1368: a private path's violation names its globs only, so the ref scan
        would exempt a secret inside a listed private version unseen. Verify runs the secret
        rules on every refs_only version and fails on a hit the entry does not name."""
        self.policy.private_paths["priv"] = "deploy/private/**"
        clean = b"a reviewed private version\n"
        secret = b"a private version with an alpha needle\n"
        self.commit("deploy/private/u.txt", clean)
        self.git("tag", "v1")
        self.commit("deploy/private/u.txt", secret)
        self.git("tag", "v2")
        self.git("rm", "-q", "deploy/private/u.txt")
        self.git("commit", "-q", "-m", "moved out")
        rows = [
            {**self.historical("deploy/private/u.txt", data, ["priv"]), "category": "private_path"}
            for data in (clean, secret)
        ]
        self.write_allowlist(*rows)
        rc, output = self.run_verify()
        self.assertEqual(rc, 1, output)
        self.assertIn("1 allowlist entries no longer match", output)
        self.assertIn(
            f"sha256={rows[1]['sha256'][:12]}…  (rules the entry does not name: alpha)",
            output,
        )
        # Review of memra#1368 round 2: --prune cannot fix an unnamed secret rule, so it fails
        # too, even on refs that mirror origin, and keeps the entry.
        self.mirror_origin()
        rc, output = self.run_verify(prune=True)
        self.assertEqual(rc, 1, output)
        self.assertNotIn("unpruned", output)
        with mock.patch.object(boundary, "ROOT", self.root):
            self.assertEqual(len(boundary.load_allowlist(self.allowlist_path)), 2)
        # An entry that names the secret rule too (a reviewed placeholder) holds.
        self.write_allowlist(rows[0], {**rows[1], "rules": ["alpha", "priv"]})
        rc, output = self.run_verify()
        self.assertEqual(rc, 0, output)

    def test_prune_keeps_the_section_comment_over_refs_only_entries(self) -> None:
        """Review of memra#1368: a rewrite puts the refs_only rows, under their comment, last."""
        old = self.tag_old_version()
        self.mirror_origin()
        self.commit("deploy/live.txt", b"beta needle\n")
        live = self.pin("deploy/live.txt", ["beta"])
        self.commit("deploy/gone.txt", b"beta needle too\n")
        gone = self.pin("deploy/gone.txt", ["beta"])
        self.commit("deploy/gone.txt", b"clean\n")
        kept = self.historical("deploy/old.txt", old, ["alpha"])
        lines = ["# fixture allowlist", *boundary.REFS_ONLY_SECTION]
        lines += [json.dumps(row, sort_keys=True) for row in (kept, live, gone)]
        self.allowlist_path.write_text("\n".join(lines) + "\n")
        rc, output = self.run_verify(prune=True)
        self.assertEqual(rc, 0, output)
        self.assertIn("dropped 1 entries", output)
        self.assert_refs_only_section()
        self.assertEqual(
            self.allowlist_path.read_text().splitlines()[:2],
            ["# fixture allowlist", json.dumps(live, sort_keys=True)],
        )

    def test_a_non_utf8_path_reads_at_a_ref(self) -> None:
        """Review of memra#1368: git paths are bytes; one that is not UTF-8 must not stop the
        `<ref>:<path>` passes."""
        name = os.fsdecode(b"deploy/\xff.txt")
        self.commit(name, b"alpha needle\n")
        with mock.patch.object(boundary, "ROOT", self.root):
            oids = boundary.oids_at([f"HEAD:{name}"])
            blobs = boundary.blobs_at([f"HEAD:{name}"])
        self.assertEqual(len(oids), 1)
        self.assertEqual(list(blobs.values()), [b"alpha needle\n"])

    def test_verify_holds_a_refs_only_entry_to_a_published_ref(self) -> None:
        old = self.tag_old_version()
        self.write_allowlist(self.historical("deploy/old.txt", old, ["alpha"]))
        rc, output = self.run_verify()
        self.assertEqual(rc, 0, output)
        self.assertIn("1 refs_only entries all pin versions at published refs", output)
        never = b"alpha needle, never pushed\n"
        self.write_allowlist(self.historical("deploy/old.txt", never, ["alpha"]))
        rc, output = self.run_verify()
        self.assertEqual(rc, 1, output)
        self.assertIn("at no published ref", output)
        self.write_allowlist(self.historical("deploy/old.txt", old, ["alpha", "beta"]))
        rc, output = self.run_verify()
        self.assertEqual(rc, 1, output)
        self.assertIn("rules no longer match: beta", output)

    def test_a_shallow_clone_leaves_refs_only_entries_to_the_full_clone(self) -> None:
        """ci.yml's boundary job is shallow: it says it did not judge them; --prune keeps them."""
        self.tag_old_version()
        self.commit("deploy/pinned.txt", b"alpha needle\n")
        stale = self.pin("deploy/pinned.txt", ["alpha"])
        self.commit("deploy/pinned.txt", b"clean\n")
        kept = self.historical("deploy/old.txt", b"alpha needle, never pushed\n", ["alpha"])
        self.write_allowlist(stale, kept)
        with mock.patch.object(boundary, "is_shallow", lambda: True):
            rc, output = self.run_verify(prune=True)
        self.assertEqual(rc, 0, output)
        self.assertIn("1 refs_only entries not checked here", output)
        with mock.patch.object(boundary, "ROOT", self.root):
            self.assertEqual(
                list(boundary.load_allowlist(self.allowlist_path)),
                [("deploy/old.txt", kept["sha256"])],
            )

    def test_a_refs_only_scope_is_the_only_scope(self) -> None:
        entry = {**self.historical("deploy/old.txt", b"x", ["alpha"]), "scope": "refs"}
        self.write_allowlist(entry)
        with self.assertRaises(SystemExit):
            boundary.load_allowlist(self.allowlist_path)

    def test_a_refs_only_private_path_entry_carries_no_expiry(self) -> None:
        """A version on immutable refs parks nothing, so the severity-4 date demand skips it."""
        policy = boundary.Policy(
            secret_patterns={}, secret_sources={}, secret_union=boundary.re.compile("(?!)"),
            secret_groups={}, private_paths={"priv": "private/**"}, bypass_paths=[],
            severity={"priv": 4},
        )
        entry = {**self.historical("private/a.md", b"x", ["priv"]), "category": "private_path"}
        self.assertTrue(boundary.needs_expiry({**entry, "scope": None}, policy))
        self.assertFalse(boundary.needs_expiry(entry, policy))

    def test_seed_keeps_refs_only_entries(self) -> None:
        """A reseed rebuilds the checkout's entries; it cannot regenerate a historical one."""
        old = self.tag_old_version()
        self.commit("deploy/live.txt", b"beta needle\n")
        kept = self.historical("deploy/old.txt", old, ["alpha"])
        self.write_allowlist(kept)
        with (
            mock.patch.object(boundary, "ROOT", self.root),
            mock.patch.object(boundary, "ALLOWLIST_PATH", self.allowlist_path),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(boundary.cmd_seed(self.policy, force=True), 0)
            entries = boundary.load_allowlist(self.allowlist_path)
        self.assertEqual(entries[("deploy/old.txt", kept["sha256"])], kept)
        self.assertEqual(sorted(path for path, _ in entries), ["deploy/live.txt", "deploy/old.txt"])
        self.assert_refs_only_section()

    def test_boundary_refs_runs_verify_allowlist(self) -> None:
        """The full clone that holds every published ref is where refs_only entries are held.
        Review of memra#1368: the scan and the drift check run in one step, each failing the step
        without skipping the other, and an allowlist or scanner change on a pull request runs the
        drift check. Round 2: the drift check runs first (a timeout kill in the slow scan must not
        take it along), and the step body is RUN, as GitHub runs it (`bash -e`), with a stub
        python3 that records each call."""
        workflow = (Path(__file__).resolve().parent.parent / ".github/workflows/boundary-refs.yml")
        text = workflow.read_text()
        steps = text.split("\n      - name: ")
        step = next(s for s in steps if s.startswith("Public-boundary ref scan and allowlist drift"))
        script = textwrap.dedent(step.split("run: |\n", 1)[1])
        code = ["\n".join(ln for ln in s.splitlines() if not ln.strip().startswith("#")) for s in steps]
        self.assertEqual(sum("verify-allowlist" in c for c in code), 1)
        stub = self.root / "stub"
        stub.mkdir()
        (stub / "python3").write_text(
            '#!/bin/sh\necho "$*" >> "$CALLS"\n'
            'case "$*" in *verify-allowlist*) exit "$V_RC";; *) exit "$C_RC";; esac\n'
        )
        (stub / "python3").chmod(0o755)
        (stub / "step.sh").write_text(script)
        calls = stub / "calls"
        verify = "tools/check-public-boundary.py verify-allowlist"
        scan = (
            "tools/check-public-boundary.py check --refs "
            "refs/remotes/origin/**,refs/remotes/origin-pull/**,refs/tags/** --summary-only"
        )
        for event in ("schedule", "pull_request", "workflow_dispatch"):
            for v_rc in (0, 1):
                for c_rc in (0, 1):
                    with self.subTest(event=event, v_rc=v_rc, c_rc=c_rc):
                        calls.write_text("")
                        env = {
                            **os.environ,
                            "PATH": f"{stub}:{os.environ['PATH']}",
                            "CALLS": str(calls),
                            "EVENT": event,
                            "REF_GLOB": "",
                            "V_RC": str(v_rc),
                            "C_RC": str(c_rc),
                        }
                        run = subprocess.run(
                            ["bash", "--noprofile", "--norc", "-eo", "pipefail", str(stub / "step.sh")],
                            env=env,
                            capture_output=True,
                            text=True,
                        )
                        made = calls.read_text().splitlines()
                        if event == "pull_request":
                            self.assertEqual(made, [verify])
                            self.assertEqual(run.returncode, v_rc, run.stdout)
                        else:
                            self.assertEqual(made, [verify, scan])
                            self.assertEqual(run.returncode, int(bool(v_rc or c_rc)), run.stdout)
        on = text.split("\njobs:", 1)[0]
        pr = on.split("\n  pull_request:\n", 1)[1].split("\n  workflow_dispatch:", 1)[0]
        listed = [ln.strip() for ln in pr.splitlines() if ln.strip().startswith("- ")]
        self.assertEqual(
            sorted(listed),
            sorted(
                f'- "{path}"'
                for path in (
                    "tools/public-boundary-allowlist.jsonl",
                    "tools/public-boundary-policy.toml",
                    "tools/check-public-boundary.py",
                    ".github/workflows/boundary-refs.yml",
                )
            ),
        )

    def test_empty_allowlist_never_requests_a_whole_tree_grep(self) -> None:
        self.commit("deploy/new.txt", b"alpha needle\n")
        self.write_allowlist()
        real_run = boundary.subprocess.run
        with mock.patch.object(boundary.subprocess, "run", wraps=real_run) as calls:
            rc, output = self.run_verify()
        self.assertEqual(rc, 0, output)
        self.assertIn("0 allowlist entries", output)
        self.assertFalse(any("grep" in call.args[0] for call in calls.call_args_list))

    def test_multiple_hashes_for_one_path_read_once_and_stale_hash_still_fails(self) -> None:
        self.commit("deploy/pinned.txt", b"alpha needle\n")
        live = self.pin("deploy/pinned.txt", ["alpha"])
        stale = {**live, "sha256": "f" * 64}
        self.write_allowlist(live, stale)
        with mock.patch.object(boundary, "worktree_blob_bytes", wraps=boundary.worktree_blob_bytes) as reads:
            rc, output = self.run_verify()
        self.assertEqual(rc, 1, output)
        self.assertEqual(reads.call_count, 1)
        self.assertIn("1 allowlist entries no longer match", output)

    def test_matching_untracked_file_cannot_keep_an_exemption_live(self) -> None:
        self.commit("deploy/pinned.txt", b"alpha needle\n")
        self.write_allowlist(self.pin("deploy/pinned.txt", ["alpha"]))
        self.git("rm", "--cached", "deploy/pinned.txt")
        self.assertTrue((self.root / "deploy/pinned.txt").is_file())
        rc, output = self.run_verify()
        self.assertEqual(rc, 1, output)
        self.assertIn("no live violation", output)

    def test_matching_untracked_symlink_cannot_keep_an_exemption_live(self) -> None:
        path = "deploy/link"
        link = self.root / path
        link.parent.mkdir(parents=True)
        link.symlink_to("alpha needle")
        self.git("add", path)
        self.git("commit", "-q", "-m", "tracked symlink")
        digest = boundary.hashlib.sha256(b"alpha needle").hexdigest()
        self.write_allowlist({"path": path, "sha256": digest, "category": "secret_pattern",
                              "rules": ["alpha"], "reason": "fixture link text"})
        self.git("rm", "--cached", path)
        self.assertTrue(link.is_symlink())
        rc, output = self.run_verify()
        self.assertEqual(rc, 1, output)

    def test_broken_symlink_pin_is_verified_from_link_text_not_target(self) -> None:
        path = "deploy/link"
        link = self.root / path
        link.parent.mkdir(parents=True)
        link.symlink_to("alpha needle")
        self.git("add", path)
        self.git("commit", "-q", "-m", "pin symlink")
        digest = boundary.hashlib.sha256(b"alpha needle").hexdigest()
        self.write_allowlist({"path": path, "sha256": digest, "category": "secret_pattern",
                              "rules": ["alpha"], "reason": "fixture link text"})
        rc, output = self.run_verify()
        self.assertEqual(rc, 0, output)
        link.unlink()
        link.symlink_to("beta needle")
        rc, output = self.run_verify()
        self.assertEqual(rc, 1, output)

    def test_clean_allowlist_passes_and_prints_its_count(self) -> None:
        """A green run must SAY how many entries it verified.

        Same idiom as the pre-push flags arm that landed today: a gate that is silent when it
        passes is indistinguishable from a gate that has stopped running, and this one had in
        fact never run in an automated context at all.
        """
        self.commit("deploy/evidence.txt", b"alpha needle\n")
        self.write_allowlist(self.pin("deploy/evidence.txt", ["alpha"]))
        rc, output = self.run_verify()
        self.assertEqual(rc, 0, output)
        self.assertIn("1 allowlist entries all pin live tracked files", output)

    def test_edited_file_makes_its_entry_stale_then_restoring_it_passes_again(self) -> None:
        """CORRUPT -> red, RESTORE -> green, in one arm.

        Both directions matter and neither alone is evidence: an arm that only proves the red
        would be satisfied by a verify that refuses everything, and an arm that only proves the
        green would be satisfied by one that refuses nothing.
        """
        clean = b"alpha needle\n"
        self.commit("deploy/evidence.txt", clean)
        self.write_allowlist(self.pin("deploy/evidence.txt", ["alpha"]))
        self.assertEqual(self.run_verify()[0], 0)

        # CORRUPT: same rule still fires, but the bytes — and so the pinned digest — moved.
        self.commit("deploy/evidence.txt", b"alpha needle, now with a second line\n")
        rc, output = self.run_verify()
        self.assertEqual(rc, 1, output)
        self.assertIn("no longer match tracked files", output)
        self.assertIn("[DRIFT]", output)
        self.assertIn("no live violation for these bytes", output)

        # RESTORE.
        self.commit("deploy/evidence.txt", clean)
        rc, output = self.run_verify()
        self.assertEqual(rc, 0, output)
        self.assertIn("1 allowlist entries all pin live tracked files", output)

    def test_deleted_file_fails_closed(self) -> None:
        """An entry pointing at nothing is the quiet half of allowlist rot.

        `check` never reports it — there are no bytes to match — so without this mode a grant can
        outlive its file indefinitely and nobody learns the list has stopped describing reality.
        """
        self.commit("deploy/evidence.txt", b"alpha needle\n")
        self.write_allowlist(self.pin("deploy/evidence.txt", ["alpha"]))
        self.git("rm", "-q", "deploy/evidence.txt")
        self.git("commit", "-q", "-m", "remediate by deletion")
        rc, output = self.run_verify()
        self.assertEqual(rc, 1, output)
        self.assertIn("[DRIFT]", output)

    def test_remediated_rule_expires_that_grant_and_is_named(self) -> None:
        """Rule-scoped drift: bytes still trip `alpha`, so only the dead `beta` grant expires."""
        self.commit("deploy/evidence.txt", b"alpha needle\nbeta needle\n")
        self.write_allowlist(self.pin("deploy/evidence.txt", ["alpha", "beta"]))
        self.assertEqual(self.run_verify()[0], 0)

        # Drop `beta` from the policy: the entry's grant for it is now a grant for nothing.
        del self.policy.secret_patterns["beta"]
        del self.policy.secret_sources["beta"]
        self.policy.secret_union = boundary.re.compile("(?P<rule_0>alpha needle)")
        self.policy.secret_groups = {"rule_0": "alpha"}
        rc, output = self.run_verify()
        self.assertEqual(rc, 1, output)
        self.assertIn("rules no longer match: beta", output)

    def test_ci_invokes_verify_allowlist(self) -> None:
        """The wiring, anchored on the INVOCATION rather than any mention of it.

        A bare substring search over `ci.yml` would be satisfied by the rationale COMMENT that
        sits above the step — the exact defect found today in the flags-guard fixture, whose
        `grep tools/check-flags.sh` stayed green through a deliberate unwiring because the hook's
        own comment names the script three times. So comment lines are stripped before the
        search, and the pattern requires the mode argument on the same line as the script.

        Comment-stripped by hand rather than via a YAML parser on purpose: PyYAML is not
        guaranteed on every interpreter this suite runs under, and a wiring assertion that can
        fail for a missing dependency is a wiring assertion that will be deleted.
        """
        workflow = boundary.ROOT / ".github" / "workflows" / "ci.yml"
        self.assertTrue(workflow.is_file(), f"missing {workflow}")
        code = "\n".join(
            line
            for line in workflow.read_text().splitlines()
            if not line.lstrip().startswith("#")
        )
        self.assertRegex(
            code,
            r"check-public-boundary\.py\s+verify-allowlist",
            "ci.yml has no live `check-public-boundary.py verify-allowlist` invocation — the "
            "allowlist gate is unwired and would go back to running only when a human "
            "remembered (it had no caller at all until 2026-08-23)",
        )


class UnstatablePathTests(unittest.TestCase):
    """TEETH for the unstat-able-tracked-path arm (2026-09-01).

    `evaluate()` calls `full.is_file()`, which STATS, and a stat can RAISE — a tracked
    symlink whose target sits on another machine under a directory this user cannot
    traverse gives PermissionError. That killed the whole gate in CI: it died on
    `.../mv-battery-20260831/receipts/c3/off-1 -> /root/out-mv/c2/off-1` and therefore
    scanned NOTHING after it, while the same tree passed on a workstation whose Python
    returned False instead of raising. A gate that governs what the public repository may
    publish must not be silenced by one unreadable path, and must not silently skip it
    either.

    Both halves are proven here: the scan SURVIVES the raise, and it REPORTS the path.
    """

    def setUp(self) -> None:
        self.policy = boundary.Policy(
            secret_patterns={},
            secret_sources={},
            secret_union=boundary.re.compile(r"(?!x)x"),
            secret_groups={},
            private_paths={},
            bypass_paths=[],
        )

    def _evaluate_with(self, tracked, raiser):
        def fake_is_file(self):  # noqa: ANN001
            if self.name == raiser:
                raise PermissionError(13, "Permission denied")
            # Every other tracked path in these fixtures is a real file as far as the scan
            # is concerned; returning the on-disk answer would skip them for not existing
            # and the "did the scan continue" assertion could never fail.
            return True

        with (
            mock.patch.object(boundary, "tracked_files", return_value=tracked),
            mock.patch.object(boundary, "secret_candidate_files", return_value=set()),
            mock.patch.object(boundary.Path, "is_file", fake_is_file),
        ):
            return boundary.evaluate(self.policy)

    def test_unstatable_path_is_reported_not_raised(self) -> None:
        violations = self._evaluate_with(["a/unreadable", "README.md"], "unreadable")
        paths = [v.path for v in violations]
        self.assertIn(
            "a/unreadable",
            paths,
            "an unstat-able tracked path must be REPORTED — a silent skip lets an "
            "unreadable tracked path hide anything at all",
        )
        v = next(v for v in violations if v.path == "a/unreadable")
        self.assertEqual(v.category, "unstatable_path")
        self.assertIn("PermissionError", v.detail)

    def test_scan_continues_past_the_unstatable_path(self) -> None:
        # The regression that mattered: the raise used to abort the loop, so every file
        # AFTER the bad one went unscanned. Order it first and prove the rest still ran.
        with mock.patch.object(boundary, "worktree_blob_bytes", return_value=b""):
            self.policy.private_paths = {"deploy_unit": "later/*"}
            violations = self._evaluate_with(
                ["a/unreadable", "later/thing"], "unreadable"
            )
        paths = [v.path for v in violations]
        self.assertIn("a/unreadable", paths)
        self.assertIn(
            "later/thing",
            paths,
            "the scan stopped at the unstat-able path — every later tracked file would go "
            "unchecked, which is how the gate reported success while scanning nothing",
        )

    def test_no_tracked_symlink_escapes_the_repo(self) -> None:
        """The CAUSE, not just the symptom: an absolute symlink target is unresolvable on
        every machine but the one that wrote it, so it is never a receipt and it publishes
        a foreign box's filesystem layout. Six mv-battery links pointed at `/root/out-mv/`
        and one requal link at `/opt/scratch/...` while their real targets sat beside them
        in-repo."""
        listing = subprocess.run(
            ["git", "ls-files", "-s"],
            cwd=boundary.ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        offenders = []
        for line in listing:
            meta, _, rel = line.partition("\t")
            if not meta.startswith("120000"):
                continue
            target = (boundary.ROOT / rel).readlink()
            if target.is_absolute() or not (boundary.ROOT / rel).exists():
                offenders.append(f"{rel} -> {target}")
        self.assertEqual(
            offenders,
            [],
            "tracked symlink(s) are absolute or dangling: "
            + "; ".join(offenders)
            + " — retarget them repo-relative at their in-repo twin",
        )


if __name__ == "__main__":
    unittest.main()
