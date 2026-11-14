"""Landing 17: 1 Hz log of AC online, platform profile and the CPU caps (ppt_pl1_spl, ppt_pl2_sppt) during a timed
5090 cell. Any second off AC voids the take (coordinator, 2026-11-13): the guard writes VOID_FILE and stops the
cell's process group (its pgid read from PGID_FILE). Runs until killed.
usage: ac_guard17.py VOID_FILE PGID_FILE   (TSV on stdout: HH:MM:SS ac profile pl1 pl2)"""
import glob
import os
import signal
import sys
import time

void_file, pgid_file = sys.argv[1], sys.argv[2]
ac = sorted(glob.glob("/sys/class/power_supply/A*/online"))[0]
arm = "/sys/class/firmware-attributes/asus-armoury/attributes"


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return "?"


while True:
    online = read(ac)
    profile = read("/sys/firmware/acpi/platform_profile")
    pl1 = read(f"{arm}/ppt_pl1_spl/current_value")
    pl2 = read(f"{arm}/ppt_pl2_sppt/current_value")
    stamp = time.strftime("%H:%M:%S")
    print(f"{stamp}\t{online}\t{profile}\t{pl1}\t{pl2}", flush=True)
    if online != "1" and not os.path.exists(void_file):
        with open(void_file, "w") as f:
            f.write(f"{stamp} AC offline: take void\n")
        try:
            pgid = int(read(pgid_file))
            os.killpg(pgid, signal.SIGTERM)
        except (ValueError, OSError):
            pass
    time.sleep(1)
