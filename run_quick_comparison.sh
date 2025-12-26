#!/bin/bash

###############################################################################
# Quick Comparison: Baseline vs Full Adaptive
#
# This script runs a quick comparison between:
# 1. Baseline (no adaptive scheduling)
# 2. Full adaptive scheduling (all three dimensions)
#
# Usage: bash run_quick_comparison.sh [dataset] [gpu]
#   dataset: tiktok, baby, or sports (default: sports)
#   gpu: GPU ID to use (default: 0)
#
# Example: bash run_quick_comparison.sh sports 0
###############################################################################

# Parse arguments
DATASET=${1:-sports}
GPU=${2:-1}

# Experiment configuration
EPOCHS=50
BATCH_SIZE=1024
LEARNING_RATE=1e-3
SAMPLING_STEPS=5
GEN_TOPK=5
REBUILD_K=1
FLIP_PROB=0.15

# Create results directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="./results_quick_${DATASET}_${TIMESTAMP}"
mkdir -p ${RESULTS_DIR}

# Log file
LOGFILE="${RESULTS_DIR}/experiment_log.txt"

# Helper function for logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a ${LOGFILE}
}

# Helper function to run experiment
run_experiment() {
    local exp_name=$1
    local use_adaptive=$2

    log "=================================================="
    log "Running: ${exp_name}"
    log "=================================================="

    # Create experiment subdirectory
    EXP_DIR="${RESULTS_DIR}/${exp_name}"
    mkdir -p ${EXP_DIR}

    # Run training (use 0/1 for boolean values to avoid argparse bool parsing issues)
    python Main.py \
        --data ${DATASET} \
        --gpu ${GPU} \
        --epoch ${EPOCHS} \
        --batch ${BATCH_SIZE} \
        --lr ${LEARNING_RATE} \
        --sampling_steps ${SAMPLING_STEPS} \
        --gen_topk ${GEN_TOPK} \
        --rebuild_k ${REBUILD_K} \
        --use_adaptive_flip ${use_adaptive} \
        --flip_prob ${FLIP_PROB} \
        --use_activity_adaptive ${use_adaptive} \
        --use_epoch_adaptive ${use_adaptive} \
        --use_popularity_adaptive ${use_adaptive} \
        2>&1 | tee ${EXP_DIR}/training.log

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log "✓ ${exp_name} completed successfully"
    else
        log "✗ ${exp_name} failed"
    fi
    log ""
}

###############################################################################
# Main Experiments
###############################################################################

log "=================================================="
log "Quick Comparison: Baseline vs Full Adaptive"
log "=================================================="
log "Dataset: ${DATASET}"
log "GPU: ${GPU}"
log "Flip Probability: ${FLIP_PROB}"
log "Results Directory: ${RESULTS_DIR}"
log "=================================================="
log ""

# Experiment 1: Baseline (use 0 for False)
run_experiment "baseline" 0

# Experiment 2: Full Adaptive (use 1 for True)
run_experiment "full_adaptive" 1

###############################################################################
# Compare Results
###############################################################################

log "=================================================="
log "Comparison Complete!"
log "=================================================="
log ""
log "Results saved to: ${RESULTS_DIR}"
log ""
log "View logs:"
log "  Baseline: ${RESULTS_DIR}/baseline/training.log"
log "  Full Adaptive: ${RESULTS_DIR}/full_adaptive/training.log"
log ""
log "=================================================="
