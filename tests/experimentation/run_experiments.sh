#!/usr/bin/env bash
# run_experiments.sh — ReviewSentinel /predict load test driver
# =============================================================
# Usage:
#   ./run_experiments.sh                          # default host
#   LOCUST_HOST=http://<vm-ip>:30080 ./run_experiments.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HOST="${LOCUST_HOST:-http://localhost:8000}"
RESULTS_DIR="$HERE/results"
LOCUST_BIN="${VIRTUAL_ENV:-$HERE/.venv}/bin/locust"

mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Pre-flight: /health must report model_loaded=true
# ---------------------------------------------------------------------------
echo "[preflight] Checking $HOST/health ..."

HEALTH_JSON=$(curl -sf --max-time 10 "$HOST/health" || true)
if [ -z "$HEALTH_JSON" ]; then
    echo "[preflight] ERROR: /health endpoint unreachable at $HOST" >&2
    exit 1
fi

MODEL_LOADED=$(echo "$HEALTH_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(str(d.get('model_loaded', False)).lower())" 2>/dev/null || echo "false")

if [ "$MODEL_LOADED" != "true" ]; then
    echo "[preflight] ERROR: model_loaded=false. Train or wait for self-healing first." >&2
    echo "  Health response: $HEALTH_JSON" >&2
    exit 1
fi

echo "[preflight] model_loaded=true. Proceeding."

# ---------------------------------------------------------------------------
# Verify test data exists
# ---------------------------------------------------------------------------
if [ ! -f "$HERE/test_data.json" ]; then
    echo "[setup] test_data.json not found — generating ..."
    python3 generate_test_data.py
fi

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
run_scenario() {
    local name="$1"      # e.g. "A"
    local label="$2"     # e.g. "Flash Crowd"
    local users="$3"
    local spawn="$4"
    local runtime="$5"
    local scenario_env="${6:-$name}"

    echo ""
    echo "========================================================"
    echo " Scenario $name — $label"
    echo "   users=$users  spawn-rate=$spawn/s  run-time=$runtime"
    echo "   SCENARIO=$scenario_env"
    echo "========================================================"

    SCENARIO="$scenario_env" \
    "$LOCUST_BIN" \
        -f "$HERE/locustfile.py" \
        --host "$HOST" \
        --headless \
        --users "$users" \
        --spawn-rate "$spawn" \
        --run-time "$runtime" \
        --csv "$RESULTS_DIR/scenario_${name}_stats" \
        --csv-full-history \
        --html "$RESULTS_DIR/scenario_${name}_report.html" \
        --loglevel WARNING \
        2>&1 | tee "$RESULTS_DIR/scenario_${name}.log"

    echo "[scenario $name] Done. CSVs in $RESULTS_DIR/scenario_${name}_stats*.csv"
}

# ---------------------------------------------------------------------------
# Scenario A — Flash Crowd
#   500 users, spawn 500/s (≈1 s ramp), hold 30 s
# ---------------------------------------------------------------------------
run_scenario "A" "Flash Crowd"    500 500 "30s" "A"

# ---------------------------------------------------------------------------
# Scenario B — Gradual Ramp-up
#   500 users, spawn 10/s (≈50 s ramp), total 4 m
# ---------------------------------------------------------------------------
run_scenario "B" "Gradual Ramp"   500  10 "4m"  "B"

# ---------------------------------------------------------------------------
# Scenario C — Variable Payload Lengths
#   Same ramp as B, but 50 % of requests use long payloads
# ---------------------------------------------------------------------------
run_scenario "C" "Variable Payload" 500 10 "4m" "C"

# ---------------------------------------------------------------------------
# KPI extraction
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo " Extracting KPIs ..."
echo "========================================================"
python3 "$HERE/analyze_results.py" --results-dir "$RESULTS_DIR"

echo ""
echo "All done. KPI summary: $RESULTS_DIR/KPI_SUMMARY.md"
