#!/usr/bin/env python3
"""Daily per-device regression: main's head on this device, results under research/, one
issue per failing device.

Once a day a user timer runs this on each device that has it installed. It runs the
device's full cell set on origin/main (tools/cell_evidence.py: battery, serve-smoke and a
perf A/B against the previous tested head), commits the raw record to the device's
results branch `regression/<device>` under research/daily-regression/<device>/, and keeps
one GitHub issue per device: opened or updated with the failing cells while the device
fails, closed with a comment when it passes again. A head that was already tested is
skipped. Fixes then land per hardware instead of every push rerunning every battery.

    python3 tools/daily-regression.py install --device rtx5090 --at 05:30   # units, not enabled
    python3 tools/daily-regression.py run --device rtx5090 --bootstrap      # what the timer runs
    python3 tools/daily-regression.py run --device rtx5090 --no-publish     # local only

The timer runs `run --bootstrap`: it fetches origin/main and re-executes this script from a
detached checkout of that head, so the runner is always main's version.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
UNIT_DIR = Path.home() / ".config/systemd/user"
INSTALL_DIR = Path.home() / ".local/share/memra-daily"
LABEL = "daily-regression"
RESULTS = "research/daily-regression"


def log(message):
    stamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] {message}", flush=True)


def git(repo, *args, check=True, env=None):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=env)
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])}: {result.stderr.strip()}")
    return result.stdout.strip()


def work_dir(device):
    return Path(os.environ.get("MEMRA_DAILY_WORK", Path.home() / ".cache/memra-daily" / device))


# --------------------------------------------------------------------------- install

def install(args):
    """Copy the runner and the unit templates; never enable the timer."""
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    UNIT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), INSTALL_DIR / "daily-regression.py")
    for name in ("memra-daily-regression@.service", "memra-daily-regression@.timer"):
        shutil.copy2(HERE / "daily-regression" / name, UNIT_DIR / name)
    instance = f"memra-daily-regression@{args.device}"
    if args.at:
        dropin = UNIT_DIR / f"{instance}.timer.d"
        dropin.mkdir(exist_ok=True)
        (dropin / "at.conf").write_text(f"[Timer]\nOnCalendar=\nOnCalendar=*-*-* {args.at}:00\n")
    env_file = Path.home() / ".config/memra-daily" / f"{args.device}.env"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    if not env_file.exists():
        env_file.write_text(f"# Overrides for {instance}.service\nMEMRA_REPO={args.repo.resolve()}\n")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    print(f"installed {instance} (timer NOT enabled)\n"
          f"  repo override: {env_file}\n"
          f"  enable:  systemctl --user enable --now {instance}.timer\n"
          f"  one run: systemctl --user start {instance}.service\n"
          f"  logs:    journalctl --user -u {instance}.service")
    return 0


# ------------------------------------------------------------------------- bootstrap

def bootstrap(args):
    repo = args.repo
    git(repo, "fetch", "-q", "origin", "main")
    head = git(repo, "rev-parse", "origin/main")
    tools = work_dir(args.device) / "tools-checkout"
    git(repo, "worktree", "remove", "--force", str(tools), check=False)
    shutil.rmtree(tools, ignore_errors=True)
    git(repo, "worktree", "prune")
    tools.parent.mkdir(parents=True, exist_ok=True)
    git(repo, "worktree", "add", "--detach", str(tools), head)
    command = [sys.executable, str(tools / "tools/daily-regression.py"), "run", "--repo", str(repo),
               "--device", args.device, "--head", head]
    command += ["--force"] if args.force else []
    command += ["--no-publish"] if args.no_publish else []
    command += ["--accept-regression", args.accept_regression] if args.accept_regression else []
    log(f"bootstrap: running main's runner at {head[:12]}")
    try:
        return subprocess.run(command).returncode
    finally:
        git(repo, "worktree", "remove", "--force", str(tools), check=False)


# ------------------------------------------------------------------------------- run

def failing_cells(record):
    cells = list(record.get("failures", []))
    for kind, entry in record.get("kinds", {}).items():
        if entry.get("exit_code") not in (None, 0) and not any(c.startswith(kind) for c in cells):
            cells.append(f"{kind}: exited {entry['exit_code']}")
    return cells


def summarize(record, head, device, target_name, run_dir):
    perf = record.get("kinds", {}).get("perf", {})
    return {"date": dt.date.today().isoformat(), "commit": head, "device": device, "target": target_name,
            "status": record["status"], "failing": failing_cells(record),
            "perf": perf.get("verdicts") or perf.get("absolute"), "perf_base": perf.get("base_commit"),
            "record": str(run_dir)}


def publish(repo, device, run_dir, summary, tools, pusher=None):
    """Commit the run to regression/<device> (an orphan results branch) and push it.

    The push runs from `pusher`, a full checkout of main's head (the runner's own checkout
    by default), so the pre-push hook runs its censuses on a real tree and checks the
    results commit; the orphan results worktree has no tools/ to run a hook from."""
    branch = f"regression/{device}"
    results = work_dir(device) / "results"
    git(repo, "worktree", "remove", "--force", str(results), check=False)
    shutil.rmtree(results, ignore_errors=True)
    # A worktree directory deleted outside git leaves a registration that blocks the branch.
    git(repo, "worktree", "prune")
    local = f"daily-results/{device}"
    git(repo, "branch", "-D", local, check=False)
    if git(repo, "ls-remote", "--heads", "origin", branch):
        git(repo, "fetch", "-q", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}")
        git(repo, "worktree", "add", "-b", local, str(results), f"refs/remotes/origin/{branch}")
    else:
        # The results branch holds only research/daily-regression/; it shares no tree with main.
        git(repo, "worktree", "add", "--orphan", "-b", local, str(results))
    dest = results / RESULTS / device / f"{summary['date']}-{summary['commit'][:12]}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(run_dir, dest)
    (dest / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    history = results / RESULTS / device / "history.tsv"
    with history.open("a") as out:
        out.write("\t".join([summary["date"], summary["commit"], summary["status"],
                             "; ".join(summary["failing"]) or "-"]) + "\n")
    git(results, "add", "-A", RESULTS)
    git(results, "commit", "-qm", f"data(regression): {device} {summary['status']} at {summary['commit'][:12]}")
    sha = git(results, "rev-parse", "HEAD")
    commits = work_dir(device) / "publish-commits.txt"
    commits.write_text(sha + "\n")
    boundary = subprocess.run([sys.executable, str(tools / "check-public-boundary.py"), "check",
                               "--commits-file", str(commits)], cwd=results)
    if boundary.returncode != 0:
        raise RuntimeError("public-boundary check refused the results commit; not pushed")
    env = {**os.environ, "MEMRA_RELEASE_QUALIFICATION_MODE": "development"}
    git(pusher or HERE.parent, "push", "-q", "origin", f"{sha}:refs/heads/{branch}", env=env)
    git(repo, "worktree", "remove", "--force", str(results), check=False)
    git(repo, "branch", "-D", local, check=False)
    return f"https://github.com/{repo_slug(repo)}/tree/{sha}/{RESULTS}/{device}/{dest.name}"


def repo_slug(repo):
    url = git(repo, "remote", "get-url", "origin")
    return url.removesuffix(".git").split("github.com")[-1].lstrip(":/")


def update_issue(repo, device, target_name, summary, link):
    """One issue per device: open/update while failing, close when passing."""
    slug = repo_slug(repo)
    title = f"Daily regression: {device} ({target_name})"
    found = subprocess.run(["gh", "issue", "list", "--repo", slug, "--state", "all", "--label", LABEL,
                            "--search", f'in:title "{title}"', "--json", "number,state,title"],
                           capture_output=True, text=True, check=True)
    issue = next((i for i in json.loads(found.stdout) if i["title"] == title), None)
    if summary["status"] == "pass":
        if issue and issue["state"] == "OPEN":
            subprocess.run(["gh", "issue", "close", str(issue["number"]), "--repo", slug, "--comment",
                            f"Passing on {device} at {summary['commit'][:12]} ({summary['date']}): {link}"], check=True)
        return issue["number"] if issue else None
    rows = "\n".join(f"- {cell}" for cell in summary["failing"]) or "- (no cell named; see the record)"
    body = (f"`{device}` ({target_name}) fails main's daily regression.\n\n"
            f"Latest failing head: {summary['commit']} ({summary['date']})\n\n"
            f"Failing cells:\n{rows}\n\nRaw record: {link}\n\n"
            "This issue is updated by tools/daily-regression.py on every failing day and closed when the "
            "device passes again. Fix per hardware: the cells above name what broke on this device. "
            "A reviewed, intended perf change is accepted on the device with "
            f"`python3 tools/daily-regression.py run --device {device} --bootstrap --force "
            "--accept-regression '<why>'`.")
    subprocess.run(["gh", "label", "create", LABEL, "--repo", slug, "--color", "B60205",
                    "--description", "Opened by the per-device daily regression run"], capture_output=True)
    if issue is None:
        out = subprocess.run(["gh", "issue", "create", "--repo", slug, "--title", title, "--label", LABEL,
                              "--body", body], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    number = str(issue["number"])
    if issue["state"] != "OPEN":
        subprocess.run(["gh", "issue", "reopen", number, "--repo", slug], check=True)
    subprocess.run(["gh", "issue", "edit", number, "--repo", slug, "--body", body], check=True)
    subprocess.run(["gh", "issue", "comment", number, "--repo", slug, "--body",
                    f"Still failing at {summary['commit'][:12]} ({summary['date']}): "
                    + "; ".join(summary["failing"])], check=True)
    return number


def prune_bins(work, keep):
    bins = work / "bins"
    if not bins.is_dir():
        return
    for entry in bins.iterdir():
        if entry.is_dir() and not any(entry.name.startswith(k) for k in keep):
            shutil.rmtree(entry)


def run(args):
    sys.path.insert(0, str(HERE))
    import cell_evidence as ce
    import change_impact as ci
    repo = args.repo
    head = args.head or git(repo, "rev-parse", "origin/main")
    work = work_dir(args.device)
    work.mkdir(parents=True, exist_ok=True)
    state_path = work / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get("last_commit") == head and not args.force:
        log(f"main is still {head[:12]}, tested {state.get('last_date')} ({state.get('last_status')}); nothing to run")
        return 0
    try:
        arch = ci.arch_table_at(repo, head)
        rules = ci.rules_at(repo, head, arch)
        target = ci.device_target(rules, arch, args.device)
    except ci.ImpactError as error:
        log(f"daily regression: main {head[:12]} cannot be classified for {args.device}: {error}")
        return 2
    target_name = ci.target_name(rules, target)
    # The perf baseline is the last head that PASSED here, so a regression stays failing
    # (and its issue open) until it is fixed or accepted, instead of passing the next day
    # against its own slow self.
    base = state.get("last_pass")
    run_dir = work / "runs" / f"{dt.date.today().isoformat()}-{head[:12]}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    log(f"{args.device} ({target_name}): main {head[:12]}, perf base {base[:12] if base else 'none'}")
    try:
        record = ce.produce(repo, head, args.device, ["battery", "serving", "perf"], run_dir,
                            base=base, work=work, accept_regression=args.accept_regression, log=log,
                            reference_perf=base is None)
    except ce.DeviceBusy as error:
        log(f"{args.device}: {error}; not a regression, the next timer run retries")
        shutil.rmtree(run_dir, ignore_errors=True)
        return 3
    except (ce.EvidenceError, ci.ImpactError, OSError, subprocess.SubprocessError) as error:
        run_dir.mkdir(parents=True, exist_ok=True)
        record = {"status": "fail", "failures": [f"run: {error}"], "kinds": {}}
        (run_dir / "record.json").write_text(json.dumps(record, indent=1) + "\n")
    summary = summarize(record, head, args.device, target_name, run_dir)
    log(f"result: {summary['status']} {'; '.join(summary['failing'])}")
    if not args.no_publish:
        # Publish and update the issue before marking the head tested: if either fails, the
        # next run retries this head instead of silently skipping a failing device.
        link = publish(repo, args.device, run_dir, summary, HERE)
        log(f"published {link}")
        issue = update_issue(repo, args.device, target_name, summary, link)
        if issue:
            log(f"issue: {issue}")
    else:
        log(f"not published (--no-publish); record at {run_dir}")
    if record.get("kinds"):
        # A run that produced a record counts as tested; a build or setup failure retries tomorrow.
        state.update(last_commit=head, last_date=summary["date"], last_status=summary["status"])
        if record["status"] == "pass":
            state["last_pass"] = head
        state_path.write_text(json.dumps(state, indent=1) + "\n")
        prune_bins(work, [head, state.get("last_pass") or head])
    return 0 if record["status"] == "pass" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("install")
    p.add_argument("--device", required=True)
    p.add_argument("--repo", type=Path, default=HERE.parent)
    p.add_argument("--at", help="local HH:MM for this device's timer (default from the unit: 05:30)")
    p = sub.add_parser("run")
    p.add_argument("--device", required=True)
    p.add_argument("--repo", type=Path, default=Path(os.environ.get("MEMRA_REPO", HERE.parent)))
    p.add_argument("--head", help="tested commit (default: origin/main)")
    p.add_argument("--bootstrap", action="store_true", help="fetch main and run main's runner")
    p.add_argument("--force", action="store_true", help="rerun a head that was already tested")
    p.add_argument("--no-publish", action="store_true", help="keep the record local: no branch push, no issue")
    p.add_argument("--accept-regression", help="record a reviewed perf regression (with why) so this head "
                                                 "becomes the new baseline")
    args = parser.parse_args()
    try:
        if args.mode == "install":
            return install(args)
        return bootstrap(args) if args.bootstrap else run(args)
    except (RuntimeError, subprocess.CalledProcessError, OSError) as error:
        log(f"daily regression: FAILED to run: {error}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
