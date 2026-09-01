#!/usr/bin/env python3
"""Unit tests for the fail-closed public-boundary policy mechanics."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
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
        self.assertEqual(len(granted), 13)
        for item in granted:
            with self.subTest(path=item["path"]):
                # `rules` is a whitelist, never a wildcard: a credential grant that did not
                # name the rule would not be an exemption at all.
                self.assertTrue(credential_rules.intersection(item["rules"]))
        # Exactly one blob carries a minted key (the auth.rs unit-test fixtures); the rest are
        # revocation prefixes. A second minted-key grant is a new decision, not a reseed.
        minted = [i for i in granted if "minted_api_key" in i["rules"]]
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
    `research/public-boundary-20260814/PROGRESS.md`. That is a gate governing what a public MIT
    repo is permitted to publish, with 685 pinned entries, running only when a human remembered.

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
    returned False instead of raising. A gate that governs what a public MIT repo may
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
