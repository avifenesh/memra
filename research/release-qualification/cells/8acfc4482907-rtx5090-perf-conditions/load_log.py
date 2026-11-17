"""Every 30 s: HH:MM:SS <tab> 1, 5 and 15 minute load <tab> card temperature. Runs until killed."""
import subprocess
import time

while True:
    with open("/proc/loadavg") as f:
        l1, l5, l15 = f.read().split()[:3]
    temp = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    print(f"{time.strftime('%H:%M:%S')}\t{l1}\t{l5}\t{l15}\t{temp}", flush=True)
    time.sleep(30)
