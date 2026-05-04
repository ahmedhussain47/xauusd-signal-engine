#!/bin/bash
# RunPod overnight run — executes all notebooks in order.
# Usage: bash run_all.sh
# Estimated time: 3-6 hours depending on GPU/CPU speed.

set -e  # stop on first error

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "  Equity Forecasting — Full Pipeline Run"
echo "  $(date)"
echo "  Working dir: $REPO_DIR"
echo "========================================"

run_notebook() {
    local nb="$1"
    local label="$2"
    echo ""
    echo ">>> [$label] Starting: $nb"
    echo "    $(date)"
    jupyter nbconvert \
        --to notebook \
        --execute \
        --inplace \
        --ExecutePreprocessor.timeout=14400 \
        --ExecutePreprocessor.kernel_name=python3 \
        "$REPO_DIR/$nb" \
        2>&1 | tee "$LOG_DIR/$(basename $nb .ipynb).log"
    echo "    Done: $(date)"
}

# Step 1: Fetch raw price data for all 160 tickers
run_notebook "notebooks/01_data_collection.ipynb"       "1/5 Data Collection"

# Step 2: Build the prepared panel (returns, features)
run_notebook "notebooks/02_prepare_forecasting_data.ipynb" "2/5 Prepare Data"

# Step 3: Run all models (Classical, N-BEATS, PatchTST, Chronos, Kronos, Embeddings)
run_notebook "notebooks/03_run_new_models_complete.ipynb"  "3/5 Run Models"

# Step 4: Barrier classifier (BarrierRF + BarrierXGB, 9-core parallel)
run_notebook "notebooks/barrier_classifier.ipynb"          "4/5 Barrier Classifier"

# Step 5: Final analysis, RankIC, plots, comparison table
run_notebook "notebooks/04_final_analysis.ipynb"           "5/5 Final Analysis"

echo ""
echo "========================================"
echo "  ALL DONE — $(date)"
echo "  Results in: $REPO_DIR/results/new_models/"
echo "  Logs in:    $LOG_DIR/"
echo "========================================"
