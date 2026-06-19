"""
locustfile.py — ReviewSentinel /predict load test
==================================================

Driven by run_experiments.sh for headless execution.  Can also be opened in
the Locust web UI:

    locust -f locustfile.py --host http://localhost:30080

Scenarios A, B, C are controlled entirely by CLI flags passed from the shell
script.  The SCENARIO env var optionally sets which payload composition to use:

    A / B : 100 % short payloads   (include_explanation=False, include_similar_reviews=False)
    C     : 50 % long payloads     (include_explanation=True,  include_similar_reviews=True)

Error labels emitted by this file match what analyze_results.py expects:
    HTTP_502_BAD_GATEWAY
    HTTP_503_MODEL_NOT_LOADED
    HTTP_504_GATEWAY_TIMEOUT
    HTTP_5XX_OTHER
    HTTP_4XX
    CONNECTION_ERROR
"""

import json
import os
import random
import sys

from locust import HttpUser, task, between, events
from locust.exception import RescheduleTask

# ---------------------------------------------------------------------------
# Load test data
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "test_data.json")

if not os.path.exists(DATA_PATH):
    print(
        f"[locustfile] ERROR: {DATA_PATH} not found.\n"
        "Run: python3 generate_test_data.py",
        file=sys.stderr,
    )
    sys.exit(1)

with open(DATA_PATH, encoding="utf-8") as f:
    _ALL_REVIEWS: list = json.load(f)

# Partition by payload "weight":
#   short  – both include_* flags are False  (fast inference, no extras)
#   long   – at least one include_* flag is True  (SHAP or similar-review lookup)
_SHORT = [r for r in _ALL_REVIEWS if not r["include_explanation"] and not r["include_similar_reviews"]]
_LONG  = [r for r in _ALL_REVIEWS if r["include_explanation"] or r["include_similar_reviews"]]

# If the data file has no examples in one category, fall back to all reviews
if not _SHORT:
    _SHORT = _ALL_REVIEWS
if not _LONG:
    _LONG = _ALL_REVIEWS

# Scenario C uses a 50/50 mix; A and B use short-only payloads.
SCENARIO = os.environ.get("SCENARIO", "A").upper()

print(
    f"[locustfile] Loaded {len(_ALL_REVIEWS)} samples "
    f"({len(_SHORT)} short, {len(_LONG)} long). SCENARIO={SCENARIO}"
)


def _pick_payload() -> dict:
    """Return a single review payload dict according to the active scenario."""
    if SCENARIO == "C" and random.random() < 0.50:
        base = random.choice(_LONG)
    else:
        base = random.choice(_SHORT)

    # Return a copy so the original list is never mutated
    return {
        "title":                  base["title"],
        "text":                   base["text"],
        "include_explanation":    base["include_explanation"],
        "include_similar_reviews": base["include_similar_reviews"],
        "similar_reviews_count":  base.get("similar_reviews_count", 5),
    }


# ---------------------------------------------------------------------------
# Custom failure labelling
# ---------------------------------------------------------------------------
def _label_response(response) -> str | None:
    """
    Return a human-readable failure label, or None if the request succeeded.
    Matches the names expected by analyze_results.py.
    """
    if response.status_code == 200:
        return None
    if response.status_code == 502:
        return "HTTP_502_BAD_GATEWAY"
    if response.status_code == 503:
        return "HTTP_503_MODEL_NOT_LOADED"
    if response.status_code == 504:
        return "HTTP_504_GATEWAY_TIMEOUT"
    if 500 <= response.status_code < 600:
        return f"HTTP_5XX_OTHER_{response.status_code}"
    return f"HTTP_{response.status_code}"


# ---------------------------------------------------------------------------
# Locust user
# ---------------------------------------------------------------------------
class ReviewSentinelUser(HttpUser):
    """
    Simulates a client posting to /predict.

    wait_time is intentionally short (0.1–0.5 s) so that at 500 concurrent
    users we saturate the thread-pool well before the nginx 60 s timeout.
    """

    wait_time = between(0.1, 0.5)

    @task
    def predict(self):
        payload = _pick_payload()

        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            # Give up after 65 s — just above nginx's default 60 s proxy timeout
            timeout=65,
        ) as resp:
            label = _label_response(resp)
            if label is None:
                resp.success()
            else:
                resp.failure(label)
