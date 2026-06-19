"""
generate_test_data.py
---------------------
Exports the `reviews` list from data.py into tests/experimentation/test_data.json
so that locustfile.py can load it at startup without importing the module at
request time.

data.py uses JSON-style boolean literals (true/false/null) which are not valid
Python, so we read the file as raw text, extract the JSON array, and parse it
with json.loads() rather than importing the module.

Usage:
    python3 generate_test_data.py

Output:
    test_data.json   (gitignored)
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(HERE, "data.py")
OUT_PATH = os.path.join(HERE, "test_data.json")

# Read the raw source and strip the Python variable assignment
with open(SRC_PATH, encoding="utf-8") as f:
    raw = f.read()

# The file is structured as:
#   reviews = [
#     { ... JSON-style dicts ... }
#   ]
# followed by generator code that produces `samples`.
# We only want the `reviews` list (the static array at the top).
# Find the first '[' and the matching closing ']' for the reviews assignment.
start = raw.index("[")

# Walk forward to find the balanced closing bracket for the top-level array.
depth = 0
end = start
for i, ch in enumerate(raw[start:], start):
    if ch == "[":
        depth += 1
    elif ch == "]":
        depth -= 1
        if depth == 0:
            end = i
            break

json_text = raw[start : end + 1]

try:
    reviews = json.loads(json_text)
except json.JSONDecodeError as exc:
    print(f"[generate_test_data] ERROR: failed to parse reviews array: {exc}", flush=True)
    raise

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(reviews, f, ensure_ascii=False, indent=2)

print(f"[generate_test_data] Wrote {len(reviews)} samples → {OUT_PATH}")
