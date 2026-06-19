"""
shapes.py — Optional precision LoadTestShape classes for ReviewSentinel
=======================================================================

Usage (instead of using --users / --spawn-rate CLI flags):

    locust -f locustfile.py --host http://localhost:30080 --headless \
           --run-time 4m --shape-class FlashCrowdShape

    # or pipe it through run_experiments.sh with LOCUST_SHAPE=FlashCrowd

The shapes mirror the three scenarios defined in PLAN.md:

    FlashCrowdShape   → Scenario A  (500 users at spawn 500/s, hold 30 s, then stop)
    GradualRampShape  → Scenario B  (ramp 10 u/s up to 500, hold until total 4 m)
    VariablePayloadShape → Scenario C  (same ramp as B; payload mix is in locustfile.py)
"""

from locust import LoadTestShape


class FlashCrowdShape(LoadTestShape):
    """
    Scenario A — Flash Crowd
    Spike to 500 users in ~1 s, hold for 30 s, then stop.
    """

    spawn_rate   = 500   # users/second
    peak_users   = 500
    hold_seconds = 30

    def tick(self):
        run_time = self.get_run_time()
        if run_time < self.hold_seconds:
            return (self.peak_users, self.spawn_rate)
        return None   # signal Locust to stop


class GradualRampShape(LoadTestShape):
    """
    Scenario B — Gradual Ramp-up
    Add 10 users/s until 500, hold until 4 minutes total.
    """

    spawn_rate      = 10     # users/second
    peak_users      = 500
    total_duration  = 4 * 60  # 240 s

    def tick(self):
        run_time = self.get_run_time()
        if run_time >= self.total_duration:
            return None
        current = min(int(run_time * self.spawn_rate), self.peak_users)
        return (max(current, 1), self.spawn_rate)


class VariablePayloadShape(GradualRampShape):
    """
    Scenario C — Variable Payload Lengths
    Same ramp as Scenario B; the 50 % long-payload mix is set via
    SCENARIO=C in locustfile.py's _pick_payload().
    Inherit GradualRampShape entirely — no tick() override needed.
    """
    pass
