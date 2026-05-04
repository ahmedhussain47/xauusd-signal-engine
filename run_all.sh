#!/bin/bash
# RunPod overnight run — executes all notebooks in order.
# Usage:
#   bash run_all.sh          → full run (all 5 notebooks)
#   bash run_all.sh --skip-data  → skip notebooks 01 & 02 (data already downloaded)

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"

SKIP_DATA=false
if [[ "$1" == "--skip-data" ]]; then
    SKIP_DATA=true
fi

echo "========================================"
echo "  Equity Forecasting — Full Pipeline Run"
echo "  $(date)"
echo "  Working dir: $REPO_DIR"
echo "  Skip data:   $SKIP_DATA"
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

if [[ "$SKIP_DATA" == false ]]; then
    run_notebook "notebooks/01_data_collection.ipynb"          "1/5 Data Collection"
    run_notebook "notebooks/02_prepare_forecasting_data.ipynb" "2/5 Prepare Data"
else
    echo ""
    echo ">>> Skipping notebooks 01 & 02 (--skip-data flag set)"
fi

run_notebook "notebooks/03_run_new_models_complete.ipynb"  "3/5 Run Models"
run_notebook "notebooks/barrier_classifier.ipynb"          "4/5 Barrier Classifier"
run_notebook "notebooks/04_final_analysis.ipynb"           "5/5 Final Analysis"

echo ""
echo "========================================"
echo "  ALL DONE — $(date)"
echo "  Results in: $REPO_DIR/results/new_models/"
echo "  Logs in:    $LOG_DIR/"
echo "========================================"
