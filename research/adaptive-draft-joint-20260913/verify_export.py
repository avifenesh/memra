"""Run the repository flag census on a Windows-origin export without CRLF shell errors."""
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[2]
temporary=root/'tools/.adaptive-check-flags.sh'
try:
    temporary.write_bytes((root/'tools/check-flags.sh').read_bytes().replace(b'\r\n',b'\n'))
    result=subprocess.run(['bash',str(temporary)],cwd=root)
finally:
    temporary.unlink(missing_ok=True)
raise SystemExit(result.returncode)
