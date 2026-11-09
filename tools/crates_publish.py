#!/usr/bin/env python3
"""crates_publish.py: publish workspace crates to crates.io, each one visible in the index before the next.

usage: crates_publish.py --version X.Y.Z [--dry-run] CRATE...
       crates_publish.py --version X.Y.Z --dry-run --workflow .github/workflows/publish.yml

.github/workflows/publish.yml calls it once per crate, in the topological order tools/workspace-publish-census.sh
checks. For each crate:

- skip-if-live: a version the crates.io API already has is not uploaded again (a rerun finishes a partial publish);
- `cargo publish -p CRATE --locked`; a 429 waits out the new-crate window (620 s) and retries, up to 6 attempts;
- then it polls the sparse index cargo resolves against (https://index.crates.io) until that exact version is
  there, and fails with a named error after INDEX_WAIT_S. A skipped (already live) crate is polled the same way.

Why the poll: cargo waits only 60 s after an upload (`publish.timeout` is still unstable, `-Zpublish-timeout`,
nightly only) and then exits 0 with a warning. On the v0.138.0 and v0.139.0 tags memra-engine was still missing
from the index the CDN served when memra-server resolved `memra-engine = "=X"` seconds later, and the publish
stopped half way. A dependent that still fails that way (or with "no matching package named", when cargo's copy
has no entry for a new crate at all) waits until the index lists the named crate here; when it already did (the
index answered here but not to cargo), it backs off LAG_RETRY_S, growing per retry, and retries. The whole lag,
index waits included, stays within INDEX_WAIT_S of the first miss; a 429 in between starts a new window.

--dry-run uploads nothing and never runs cargo: it reports which versions are live, checks each live one in the
index and says which it would publish; a crate whose state the API does not answer fails it. The workflow's
dry-run dispatch runs it over the publish list (--workflow parses the same `for crate in ... ; do` list the census
reads). CPU contract: tools/test_crates_publish.py (a fake registry; CRATES_PUBLISH_REGISTRY and the *_S
variables are its seams).
"""
import argparse
import http.client
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

USER_AGENT = "memra-publish-workflow (github.com/avifenesh/memra)"
# The sparse index answers with `cache-control: public,max-age=600` (measured 2026-11-09), so a copy the CDN
# cached before the upload can be served for up to 600 s after it. Twice that also covers the index job's own lag.
INDEX_WAIT_S = 1200
POLL_S = 10
HEARTBEAT_S = 60  # a long wait says it is still waiting
LAG_RETRY_S = 30  # cargo missed a dependency the index already lists here: back off 30 s, 60 s, 90 s, ...
RATE_LIMIT_WAIT_S = 620  # crates.io's new-crate burst window refills at one crate per 10 minutes
ATTEMPTS = 6
API_TRIES = 3
MISSING_DEP = re.compile(r'failed to select a version for the requirement `([A-Za-z0-9_-]+) = "=([^"]+)"`')
# cargo says this instead when the index copy it got has no entry for the crate at all (its first release; the
# index CDN caches a 404 too). The pinned version of a workspace crate is the one being published.
MISSING_CRATE = re.compile(r"no matching package named `(memra-[A-Za-z0-9_-]+)` found")
ALREADY_UPLOADED = re.compile(r"crate version `[^`]+` is already uploaded|already exists on crates\.io index")


class PublishError(RuntimeError):
    pass


def log(message):
    print(message, flush=True)  # Actions buffers a pipe: a warning before a 10-minute sleep must show before it


def seconds(name, default):
    return float(os.environ.get(name, default))


def registry():
    """(api base, index base). CRATES_PUBLISH_REGISTRY is the test's seam: one fake serving both."""
    fake = os.environ.get("CRATES_PUBLISH_REGISTRY")
    if fake:
        return fake.rstrip("/") + "/api/v1/crates", fake.rstrip("/") + "/index"
    return "https://crates.io/api/v1/crates", "https://index.crates.io"


def index_path(name):
    """The sparse index file path cargo reads for a crate (the registry index format)."""
    name = name.lower()
    if len(name) <= 2:
        return f"{len(name)}/{name}"
    if len(name) == 3:
        return f"3/{name[0]}/{name}"
    return f"{name[:2]}/{name[2:4]}/{name}"


def http_get(url):
    """(status, body). An HTTP error status is an answer; a network failure raises OSError."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as err:
        return err.code, ""
    except http.client.HTTPException as err:  # IncompleteRead, BadStatusLine: a failed read like any other
        raise OSError(f"{type(err).__name__}: {err}") from err


def api_live(name, version, get=http_get, sleep=time.sleep):
    """True or False when the crates.io API answers for NAME@VERSION; None when it does not (a network failure or
    an error status, after API_TRIES tries)."""
    api, _ = registry()
    for attempt in range(API_TRIES):
        if attempt:
            sleep(seconds("CRATES_PUBLISH_API_RETRY_S", 2))
        try:
            status, body = get(f"{api}/{name}/{version}")
        except OSError:
            continue
        if status == 404:
            return False
        if status != 200:
            continue
        try:
            return json.loads(body).get("version", {}).get("num") == version
        except (ValueError, AttributeError):
            continue
    return None


def index_versions(name, get=http_get):
    """The versions the sparse index lists for NAME (empty when the crate has none)."""
    _, index = registry()
    status, body = get(f"{index}/{index_path(name)}")
    if status == 404:
        return set()
    if status != 200:
        raise OSError(f"the index answered HTTP {status}")
    found = set()
    for line in body.splitlines():
        if line.strip():
            found.add(json.loads(line)["vers"])
    return found


def wait_visible(name, version, timeout=None, poll=None, get=http_get, clock=time.monotonic, sleep=time.sleep,
                 log=log):
    """Poll the index until it lists NAME@VERSION; return the seconds waited. Raises PublishError at TIMEOUT."""
    timeout = seconds("CRATES_PUBLISH_INDEX_WAIT_S", INDEX_WAIT_S) if timeout is None else timeout
    poll = seconds("CRATES_PUBLISH_POLL_S", POLL_S) if poll is None else poll
    _, index = registry()
    url = f"{index}/{index_path(name)}"
    start = clock()
    said = start
    last = "not listed yet"
    while True:
        try:
            if version in index_versions(name, get):
                waited = clock() - start
                log(f"{name}@{version} is in the crates.io index ({waited:.0f} s)")
                return waited
            last = "not listed yet"
        except (OSError, ValueError, KeyError) as err:  # a failed read is retried within the same bound
            last = f"index read failed: {err}"
        now = clock()
        if now - start >= timeout:
            raise PublishError(f"{name}@{version} is not in the crates.io index after {timeout:.0f} s ({last}; {url}). "
                               "A crate that depends on it cannot publish until it is listed. Rerun the publish on "
                               "the tag once the index lists it (docs/RELEASING.md, \"Re-running the publish on a "
                               "tag\"); live crates are skipped.")
        if now - said >= HEARTBEAT_S:
            log(f"waiting for {name}@{version} in the crates.io index: {now - start:.0f} s of {timeout:.0f} s ({last})")
            said = now
        sleep(poll)


def cargo_publish(name):
    """Run `cargo publish -p NAME --locked`, echoing its output; return (exit code, output)."""
    proc = subprocess.Popen(["cargo", "publish", "-p", name, "--locked"], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line)
    return proc.wait(), "".join(lines)


def publish(name, version, run=cargo_publish, wait=wait_visible, sleep=time.sleep, clock=time.monotonic, log=log):
    """Upload NAME@VERSION, retrying a 429 and a dependency cargo cannot resolve yet."""
    rate_limited = lag_retries = 0
    lag_since = None
    while True:
        code, out = run(name)
        if code == 0:
            return "published"
        if ALREADY_UPLOADED.search(out):
            log(f"{name}@{version} is already on crates.io")
            return "live"
        if "429 Too Many Requests" in out:
            lag_since, lag_retries = None, 0  # cargo got past resolution: a later miss starts a new window
            rate_limited += 1
            if rate_limited >= ATTEMPTS:
                raise PublishError(f"{name} still rate-limited after {ATTEMPTS} attempts")
            wait_s = seconds("CRATES_PUBLISH_RATE_WAIT_S", RATE_LIMIT_WAIT_S)
            log(f"::warning::{name} rate-limited (attempt {rate_limited}/{ATTEMPTS}); waiting {wait_s:.0f} s for "
                "the new-crate window")
            sleep(wait_s)
            continue
        missing = MISSING_DEP.search(out)
        if missing and missing[2] == version:
            dep = missing[1]
        elif MISSING_CRATE.search(out):
            dep = MISSING_CRATE.search(out)[1]
        else:
            raise PublishError(f"{name} failed for a reason other than a rate limit or index lag; see the log above")
        bound = seconds("CRATES_PUBLISH_INDEX_WAIT_S", INDEX_WAIT_S)
        if lag_since is None:
            lag_since = clock()
        elif clock() - lag_since >= bound:
            raise PublishError(f"cargo still cannot resolve {dep}@{version} for {name} after {bound:.0f} s, although "
                               "the crates.io index lists it here. Rerun the publish on the tag; live crates are "
                               "skipped.")
        lag_retries += 1
        log(f"::warning::{name} could not resolve {dep}@{version} (retry {lag_retries}); waiting for the index to "
            "list it")
        # the index wait shares the window: the whole lag stays within INDEX_WAIT_S of the first miss
        waited = wait(dep, version, timeout=max(bound - (clock() - lag_since), 0))
        if waited < seconds("CRATES_PUBLISH_POLL_S", POLL_S):
            # The index already listed it here, so cargo was served an older copy: give that copy time to expire.
            back_off = min(seconds("CRATES_PUBLISH_LAG_RETRY_S", LAG_RETRY_S) * lag_retries,
                           max(bound - (clock() - lag_since), 0))
            log(f"{dep}@{version} is listed here but not to cargo; retrying {name} in {back_off:.0f} s")
            sleep(back_off)


def workflow_crates(path):
    """The publish list, parsed the way tools/workspace-publish-census.sh parses it."""
    text = Path(path).read_text()
    start = text.index("for crate in")
    end = text.index("; do", start)
    return text[start + len("for crate in"):end].replace("\\", " ").split()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--version", required=True)
    ap.add_argument("--dry-run", action="store_true", help="upload nothing; report and check live versions")
    ap.add_argument("--workflow", type=Path, help="take the crate list from this workflow's publish loop")
    ap.add_argument("crates", nargs="*")
    a = ap.parse_args(argv)
    crates = list(a.crates)
    if a.workflow:
        if crates or not a.dry_run:
            ap.error("--workflow lists the crates for a --dry-run, without CRATE arguments")
        crates = workflow_crates(a.workflow)
    if not crates:
        ap.error("no crate named")
    try:
        for name in crates:
            live = api_live(name, a.version)
            if live is None and a.dry_run:
                raise PublishError(f"the crates.io API did not answer for {name}@{a.version}; a dry run cannot "
                                   "tell whether it is live")
            if live:
                log(f"{name}@{a.version} already on crates.io; skipping the upload")
            elif a.dry_run:
                log(f"{name}@{a.version} is not on crates.io; a publish would upload it")
                continue
            else:  # not live, or the API did not answer: crates.io itself refuses a version it has
                if live is None:
                    log(f"::warning::the crates.io API did not answer for {name}@{a.version}; trying the upload")
                if publish(name, a.version) == "published":
                    log(f"{name}@{a.version} uploaded")
            wait_visible(name, a.version)
    except PublishError as err:
        log(f"::error::{err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
