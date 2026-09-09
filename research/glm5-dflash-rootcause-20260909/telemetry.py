#!/usr/bin/env python3
import subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent
pid=int((R/'raw/pid').read_text())
with (R/'raw/telemetry.csv').open('w') as out:
 p=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,uuid,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem','--format=csv','--loop-ms=250'],stdout=out,stderr=subprocess.STDOUT)
 try:
  while Path(f'/proc/{pid}').exists():time.sleep(1)
 finally:p.terminate();p.wait()
