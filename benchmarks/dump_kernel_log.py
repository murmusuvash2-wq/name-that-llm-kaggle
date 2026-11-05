"""Flatten a Kaggle kernel log into readable text.

Kaggle's log file is JSON lines with the actual message under "data"; grepping it directly finds nothing
because the payload is escaped inside a JSON string. This prints the messages in order so a failure is
readable, and it is a separate file rather than an inline heredoc so it cannot be mangled by YAML.
"""
import glob
import json
import sys

pattern = sys.argv[1] if len(sys.argv) > 1 else "out/*.log"
files = sorted(glob.glob(pattern))
if not files:
    print(f"(no log matching {pattern})")
    sys.exit(0)
for f in files:
    print(f"===== {f} =====")
    for line in open(f, errors="replace"):
        line = line.strip().lstrip(",")
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            print(line)
            continue
        msg = d.get("data", "")
        if isinstance(msg, str) and msg.strip():
            sys.stdout.write(msg if msg.endswith("\n") else msg + "\n")
