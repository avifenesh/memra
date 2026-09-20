#!/usr/bin/env bash
# Prepare the dedicated compiler sandbox on disposable GitHub-hosted runners.
# Ubuntu restricts unprofiled user namespaces even when bwrap is installed.
set -euo pipefail

[[ ${GITHUB_ACTIONS:-} == true && ${RUNNER_ENVIRONMENT:-} == github-hosted ]] || {
    echo 'release sandbox setup is only for disposable GitHub-hosted runners' >&2
    exit 1
}
: "${GITHUB_PATH:?GitHub runner PATH file required}"

sudo apt-get update -qq
sudo apt-get install -y bubblewrap

# A distinct root-owned executable avoids replacing a distribution or another
# application's AppArmor profile. The producer records these exact bwrap bytes.
sandbox_dir=/usr/local/libexec/memra-release-sandbox
sudo install -d -o root -g root -m 0755 "$sandbox_dir"
sudo install -o root -g root -m 0755 /usr/bin/bwrap "$sandbox_dir/bwrap"

restriction=/proc/sys/kernel/apparmor_restrict_unprivileged_userns
if [[ -r $restriction && $(< "$restriction") == 1 ]]; then
    # Grant userns to this one sandbox launcher, not a host-wide sysctl waiver.
    # The compiler still runs in bwrap's private namespaces with --cap-drop ALL.
    sudo tee /etc/apparmor.d/memra-release-bwrap >/dev/null <<'PROFILE'
abi <abi/4.0>,
include <tunables/global>
profile memra-release-bwrap /usr/local/libexec/memra-release-sandbox/bwrap flags=(unconfined) {
    userns,
}
PROFILE
    sudo apparmor_parser -r /etc/apparmor.d/memra-release-bwrap
    [[ $(< "$restriction") == 1 ]]
fi

printf '%s\n' "$sandbox_dir" >> "$GITHUB_PATH"
"$sandbox_dir/bwrap" --version
sha256sum /usr/bin/bwrap "$sandbox_dir/bwrap"
