"""
analyze_results.py — KPI extraction from Locust CSV output
===========================================================

Reads the *_stats.csv and *_stats_history.csv files produced by
run_experiments.sh and writes results/KPI_SUMMARY.md.

Usage:
    python3 analyze_results.py --results-dir results/

KPIs extracted per scenario
---------------------------
1. P95 / P99 latency          — from stats_history.csv (percentile columns)
2. Throughput (RPS)           — from stats_history.csv (Requests/s)
3. Error rate (%)             — from *_stats.csv (Failure Count / Request Count)
4. Concurrency threshold      — first sample timestamp where P95 > 5000 ms
5. Error breakdown            — from *_failures.csv (grouped by name / failure label)
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(path: str) -> list[dict]:
    """Return a list of dicts from a CSV file, or [] if not found."""
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _int(val, default=0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Per-scenario analysis
# ---------------------------------------------------------------------------

def analyse_scenario(results_dir: str, scenario: str) -> dict:
    """
    Returns a dict with KPI values for scenario A, B, or C.
    Keys:
        p95_ms, p99_ms        — peak percentile latency
        avg_rps               — mean RPS over the whole run
        peak_rps              — highest single sample RPS
        total_requests        — total request count
        total_failures        — total failure count
        error_rate_pct        — overall error %
        concurrency_threshold — timestamp (s) where P95 first > 5000 ms, or None
        error_breakdown       — dict {label: count}
        missing_files         — list of files that were expected but not found
    """
    prefix = os.path.join(results_dir, f"scenario_{scenario}")
    stats_file       = f"{prefix}_stats_stats.csv"
    history_file     = f"{prefix}_stats_stats_history.csv"
    failures_file    = f"{prefix}_stats_failures.csv"

    missing = [f for f in [stats_file, history_file] if not os.path.exists(f)]

    stats_rows   = _read_csv(stats_file)
    history_rows = _read_csv(history_file)
    failure_rows = _read_csv(failures_file)

    # ---- aggregate row from stats_stats.csv --------------------------------
    agg = next((r for r in stats_rows if r.get("Name") == "Aggregated"), {})
    total_requests = _int(agg.get("Request Count", 0))
    total_failures = _int(agg.get("Failure Count", 0))
    error_rate_pct = (
        (total_failures / total_requests * 100) if total_requests else 0.0
    )

    # ---- history (time-series) ----------------------------------------
    # Locust history CSV headers (Locust ≥ 2.20):
    #   Timestamp, User count, Type, Name, Requests/s, Failures/s,
    #   50%ile (ms), 66%ile (ms), 75%ile (ms), 80%ile (ms), 90%ile (ms),
    #   95%ile (ms), 98%ile (ms), 99%ile (ms), 99.9%ile (ms), 100%ile (ms),
    #   Average (ms), Min (ms), Max (ms)
    agg_history = [r for r in history_rows if r.get("Name") == "Aggregated"]

    rps_values  = [_float(r.get("Requests/s", 0)) for r in agg_history]
    p95_values  = [_float(r.get("95%ile (ms)", 0)) for r in agg_history]
    p99_values  = [_float(r.get("99%ile (ms)", 0)) for r in agg_history]
    timestamps  = [_float(r.get("Timestamp", 0)) for r in agg_history]

    avg_rps  = (sum(rps_values) / len(rps_values)) if rps_values else 0.0
    peak_rps = max(rps_values, default=0.0)
    peak_p95 = max(p95_values, default=0.0)
    peak_p99 = max(p99_values, default=0.0)

    # Concurrency threshold — first sample where P95 > 5000 ms
    threshold_ts = None
    for ts, p95 in zip(timestamps, p95_values):
        if p95 > 5000:
            threshold_ts = ts
            break

    # ---- error breakdown from failures CSV --------------------------------
    error_breakdown: dict[str, int] = defaultdict(int)
    for row in failure_rows:
        label = row.get("Name", "UNKNOWN")
        count = _int(row.get("Occurrences", 0))
        error_breakdown[label] += count

    return {
        "p95_ms":                peak_p95,
        "p99_ms":                peak_p99,
        "avg_rps":               avg_rps,
        "peak_rps":              peak_rps,
        "total_requests":        total_requests,
        "total_failures":        total_failures,
        "error_rate_pct":        error_rate_pct,
        "concurrency_threshold": threshold_ts,
        "error_breakdown":       dict(error_breakdown),
        "missing_files":         missing,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def build_summary(results: dict[str, dict]) -> str:
    """Return a Markdown KPI summary string."""
    lines = [
        "# ReviewSentinel `/predict` Load Test — KPI Summary",
        "",
        "Generated by `analyze_results.py`.",
        "",
    ]

    # --- KPI table ---------------------------------------------------------
    lines += [
        "## KPI Overview",
        "",
        "| Scenario | Peak P95 (ms) | Peak P99 (ms) | Avg RPS | Peak RPS | Error Rate | P95 > 5 s threshold |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in ["A", "B", "C"]:
        r = results.get(s)
        if r is None:
            lines.append(f"| {s} | — | — | — | — | — | — |")
            continue
        threshold = (
            f"{r['concurrency_threshold']:.0f} s into run"
            if r["concurrency_threshold"] is not None
            else "never"
        )
        lines.append(
            f"| {s} "
            f"| {r['p95_ms']:.0f} "
            f"| {r['p99_ms']:.0f} "
            f"| {r['avg_rps']:.1f} "
            f"| {r['peak_rps']:.1f} "
            f"| {r['error_rate_pct']:.2f}% "
            f"| {threshold} |"
        )

    lines += [""]

    # --- Error breakdown per scenario --------------------------------------
    lines += ["## Error Breakdown", ""]
    for s in ["A", "B", "C"]:
        r = results.get(s)
        if r is None:
            continue
        lines.append(f"### Scenario {s}")
        lines.append("")
        if not r["error_breakdown"]:
            lines.append("*No failures recorded.*")
        else:
            lines.append("| Label | Count |")
            lines.append("|---|---|")
            for label, count in sorted(r["error_breakdown"].items(), key=lambda x: -x[1]):
                lines.append(f"| `{label}` | {count} |")
        lines.append("")

    # --- Request totals ----------------------------------------------------
    lines += ["## Request Totals", ""]
    lines += [
        "| Scenario | Total Requests | Total Failures |",
        "|---|---|---|",
    ]
    for s in ["A", "B", "C"]:
        r = results.get(s)
        if r is None:
            lines.append(f"| {s} | — | — |")
        else:
            lines.append(f"| {s} | {r['total_requests']} | {r['total_failures']} |")
    lines.append("")

    # --- Missing files warning ---------------------------------------------
    missing_all = []
    for s in ["A", "B", "C"]:
        r = results.get(s)
        if r and r["missing_files"]:
            missing_all.extend(r["missing_files"])
    if missing_all:
        lines += [
            "## ⚠️ Missing Result Files",
            "",
            "The following files were expected but not found (scenario may not have run):",
            "",
        ]
        for f in missing_all:
            lines.append(f"- `{f}`")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse Locust CSV output and produce KPI_SUMMARY.md.")
    parser.add_argument("--results-dir", default="results", help="Directory containing Locust CSVs.")
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        print(f"[analyze_results] ERROR: {results_dir!r} is not a directory.", file=sys.stderr)
        sys.exit(1)

    results = {}
    for scenario in ["A", "B", "C"]:
        print(f"[analyze_results] Processing scenario {scenario} ...")
        results[scenario] = analyse_scenario(results_dir, scenario)

    summary = build_summary(results)
    out_path = os.path.join(results_dir, "KPI_SUMMARY.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"[analyze_results] KPI_SUMMARY.md written → {out_path}")

    # Print quick summary to stdout
    print()
    print(summary)


if __name__ == "__main__":
    main()
