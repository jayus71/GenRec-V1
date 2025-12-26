#!/bin/bash

###############################################################################
# Run Experiments on All Datasets
#
# This script runs key experiments (baseline vs full adaptive) on all datasets:
# - TikTok
# - Baby
# - Sports
#
# Usage: bash run_all_datasets.sh [gpu]
#   gpu: GPU ID to use (default: 0)
#
# Example: bash run_all_datasets.sh 0
###############################################################################

# Parse arguments
GPU=${1:-0}

# Create master results directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_RESULTS_DIR="./results_all_datasets_${TIMESTAMP}"
mkdir -p ${MASTER_RESULTS_DIR}

# Log file
LOGFILE="${MASTER_RESULTS_DIR}/master_log.txt"

# Helper function for logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a ${LOGFILE}
}

log "=================================================="
log "Running Experiments on All Datasets"
log "=================================================="
log "GPU: ${GPU}"
log "Results Directory: ${MASTER_RESULTS_DIR}"
log "=================================================="
log ""

# Datasets to test
DATASETS=("tiktok" "baby" "sports")

# Run experiments for each dataset
for dataset in "${DATASETS[@]}"; do
    log "=================================================="
    log "Starting experiments on ${dataset} dataset"
    log "=================================================="

    # Create dataset-specific directory
    DATASET_DIR="${MASTER_RESULTS_DIR}/${dataset}"
    mkdir -p ${DATASET_DIR}

    # Run baseline (use 0 for False to avoid argparse bool parsing issues)
    log "Running baseline on ${dataset}..."
    python Main.py \
        --data ${dataset} \
        --gpu ${GPU} \
        --epoch 50 \
        --batch 1024 \
        --lr 1e-3 \
        --sampling_steps 5 \
        --gen_topk 5 \
        --rebuild_k 1 \
        --use_adaptive_flip 0 \
        2>&1 | tee ${DATASET_DIR}/baseline.log

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log "✓ Baseline on ${dataset} completed"
    else
        log "✗ Baseline on ${dataset} failed"
    fi

    # Run full adaptive (use 1 for True to avoid argparse bool parsing issues)
    log "Running full adaptive on ${dataset}..."
    python Main.py \
        --data ${dataset} \
        --gpu ${GPU} \
        --epoch 50 \
        --batch 1024 \
        --lr 1e-3 \
        --sampling_steps 5 \
        --gen_topk 5 \
        --rebuild_k 1 \
        --use_adaptive_flip 1 \
        --flip_prob 0.15 \
        --use_activity_adaptive 1 \
        --use_epoch_adaptive 1 \
        --use_popularity_adaptive 1 \
        2>&1 | tee ${DATASET_DIR}/full_adaptive.log

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log "✓ Full adaptive on ${dataset} completed"
    else
        log "✗ Full adaptive on ${dataset} failed"
    fi

    log ""
done

###############################################################################
# Create Summary
###############################################################################

log "=================================================="
log "All Dataset Experiments Completed!"
log "=================================================="
log ""
log "Results Location: ${MASTER_RESULTS_DIR}"
log ""
log "Dataset Results:"
for dataset in "${DATASETS[@]}"; do
    log "  ${dataset}:"
    log "    Baseline: ${MASTER_RESULTS_DIR}/${dataset}/baseline.log"
    log "    Full Adaptive: ${MASTER_RESULTS_DIR}/${dataset}/full_adaptive.log"
done
log ""
log "=================================================="
