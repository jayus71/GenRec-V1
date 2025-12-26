#!/bin/bash

###############################################################################
# Adaptive Flip Scheduler Ablation Study
#
# This script runs comprehensive experiments comparing:
# 1. Baseline (no adaptive scheduling)
# 2. Activity-only adaptive scheduling
# 3. Epoch-only adaptive scheduling
# 4. Popularity-only adaptive scheduling
# 5. Full adaptive scheduling (all three dimensions)
#
# Usage: bash run_ablation_experiments.sh [dataset] [gpu]
#   dataset: tiktok, baby, or sports (default: sports)
#   gpu: GPU ID to use (default: 0)
#
# Example: bash run_ablation_experiments.sh sports 0
###############################################################################

# Parse arguments
DATASET=${1:-sports}
GPU=${2:-0}

# Experiment configuration
EPOCHS=50
BATCH_SIZE=1024
LEARNING_RATE=1e-3
SAMPLING_STEPS=5
GEN_TOPK=5
REBUILD_K=1

# Create results directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="./results_ablation_${DATASET}_${TIMESTAMP}"
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
    local use_activity=$3
    local use_epoch=$4
    local use_popularity=$5
    local flip_prob=$6

    log "=================================================="
    log "Starting Experiment: ${exp_name}"
    log "=================================================="
    log "Configuration:"
    log "  Dataset: ${DATASET}"
    log "  GPU: ${GPU}"
    log "  Adaptive Flip: ${use_adaptive}"
    log "  Activity Adaptive: ${use_activity}"
    log "  Epoch Adaptive: ${use_epoch}"
    log "  Popularity Adaptive: ${use_popularity}"
    log "  Base Flip Prob: ${flip_prob}"
    log "  Epochs: ${EPOCHS}"
    log "  Batch Size: ${BATCH_SIZE}"
    log "  Learning Rate: ${LEARNING_RATE}"

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
        --flip_prob ${flip_prob} \
        --use_activity_adaptive ${use_activity} \
        --use_epoch_adaptive ${use_epoch} \
        --use_popularity_adaptive ${use_popularity} \
        2>&1 | tee ${EXP_DIR}/training.log

    # Check if training succeeded
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log "✓ Experiment ${exp_name} completed successfully"
    else
        log "✗ Experiment ${exp_name} failed"
    fi

    log ""
}

###############################################################################
# Main Experiment Suite
###############################################################################

log "=================================================="
log "Adaptive Flip Scheduler Ablation Study"
log "=================================================="
log "Dataset: ${DATASET}"
log "GPU: ${GPU}"
log "Results Directory: ${RESULTS_DIR}"
log "=================================================="
log ""

# Save experiment configuration
cat > ${RESULTS_DIR}/config.txt <<EOF
Experiment Configuration
========================
Dataset: ${DATASET}
GPU: ${GPU}
Epochs: ${EPOCHS}
Batch Size: ${BATCH_SIZE}
Learning Rate: ${LEARNING_RATE}
Sampling Steps: ${SAMPLING_STEPS}
Gen TopK: ${GEN_TOPK}
Rebuild K: ${REBUILD_K}
Timestamp: ${TIMESTAMP}
EOF

# ###############################################################################
# # Experiment 1: Baseline (No Adaptive Scheduling)
# ###############################################################################
# run_experiment \
#     "exp1_baseline" \
#     0 \
#     0 \
#     0 \
#     0 \
#     0.15

# ###############################################################################
# # Experiment 2: Activity-Only Adaptive Scheduling
# ###############################################################################
# run_experiment \
#     "exp2_activity_only" \
#     1 \
#     1 \
#     0 \
#     0 \
#     0.15

# ###############################################################################
# # Experiment 3: Epoch-Only Adaptive Scheduling
# ###############################################################################
# run_experiment \
#     "exp3_epoch_only" \
#     1 \
#     0 \
#     1 \
#     0 \
#     0.15

# ###############################################################################
# # Experiment 4: Popularity-Only Adaptive Scheduling
# ###############################################################################
# run_experiment \
#     "exp4_popularity_only" \
#     1 \
#     0 \
#     0 \
#     1 \
#     0.15

# ###############################################################################
# # Experiment 5: Activity + Epoch Adaptive Scheduling
# ###############################################################################
# run_experiment \
#     "exp5_activity_epoch" \
#     1 \
#     1 \
#     1 \
#     0 \
#     0.15

# ###############################################################################
# # Experiment 6: Activity + Popularity Adaptive Scheduling
# ###############################################################################
# run_experiment \
#     "exp6_activity_popularity" \
#     1 \
#     1 \
#     0 \
#     1 \
#     0.15

###############################################################################
# Experiment 7: Epoch + Popularity Adaptive Scheduling
###############################################################################
run_experiment \
    "exp7_epoch_popularity" \
    1 \
    0 \
    1 \
    1 \
    0.15

###############################################################################
# Experiment 8: Full Adaptive Scheduling (All Three Dimensions)
###############################################################################
run_experiment \
    "exp8_full_adaptive" \
    1 \
    1 \
    1 \
    1 \
    0.15

###############################################################################
# Experiment 9: Full Adaptive with Lower Base Flip Prob (0.10)
###############################################################################
run_experiment \
    "exp9_full_adaptive_flip0.10" \
    1 \
    1 \
    1 \
    1 \
    0.10

###############################################################################
# Experiment 10: Full Adaptive with Higher Base Flip Prob (0.20)
###############################################################################
run_experiment \
    "exp10_full_adaptive_flip0.20" \
    1 \
    1 \
    1 \
    1 \
    0.20

###############################################################################
# Extract and Compare Results
###############################################################################

log "=================================================="
log "Extracting Results"
log "=================================================="

# Create results summary
SUMMARY_FILE="${RESULTS_DIR}/results_summary.txt"

cat > ${SUMMARY_FILE} <<EOF
Adaptive Flip Scheduler Ablation Study Results
===============================================
Dataset: ${DATASET}
Date: ${TIMESTAMP}

Experiment Configurations:
--------------------------
1. Baseline (no adaptive)
2. Activity-only adaptive
3. Epoch-only adaptive
4. Popularity-only adaptive
5. Activity + Epoch adaptive
6. Activity + Popularity adaptive
7. Epoch + Popularity adaptive
8. Full adaptive (baseline flip_prob=0.15)
9. Full adaptive (flip_prob=0.10)
10. Full adaptive (flip_prob=0.20)

Results:
--------
EOF

# Extract best results from each experiment
for exp_dir in ${RESULTS_DIR}/exp*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename $exp_dir)
        log "Extracting results from ${exp_name}..."

        # Look for best epoch results in training log
        if [ -f "${exp_dir}/training.log" ]; then
            echo "" >> ${SUMMARY_FILE}
            echo "${exp_name}:" >> ${SUMMARY_FILE}
            echo "----------------------------------------" >> ${SUMMARY_FILE}

            # Extract best epoch line (adjust pattern based on your actual log format)
            grep -i "best\|Best" ${exp_dir}/training.log | tail -5 >> ${SUMMARY_FILE} 2>/dev/null || echo "  No results found" >> ${SUMMARY_FILE}
        fi
    fi
done

log "Results summary saved to: ${SUMMARY_FILE}"

###############################################################################
# Create Comparison Script
###############################################################################

# Create Python script to parse and compare results
cat > ${RESULTS_DIR}/compare_results.py <<'EOF'
"""
Parse and compare experimental results
Usage: python compare_results.py
"""

import os
import re
import pandas as pd
from pathlib import Path

def parse_training_log(log_file):
    """Extract best metrics from training log"""
    metrics = {
        'recall@20': None,
        'ndcg@20': None,
        'precision@20': None,
        'best_epoch': None
    }

    try:
        with open(log_file, 'r') as f:
            content = f.read()

            # Look for best epoch results (adjust patterns based on your log format)
            # Example patterns - modify based on actual output
            recall_match = re.search(r'[Bb]est.*[Rr]ecall@?20[:\s]+([0-9.]+)', content)
            ndcg_match = re.search(r'[Bb]est.*[Nn][Dd][Cc][Gg]@?20[:\s]+([0-9.]+)', content)
            precision_match = re.search(r'[Bb]est.*[Pp]recision@?20[:\s]+([0-9.]+)', content)
            epoch_match = re.search(r'[Bb]est [Ee]poch[:\s]+([0-9]+)', content)

            if recall_match:
                metrics['recall@20'] = float(recall_match.group(1))
            if ndcg_match:
                metrics['ndcg@20'] = float(ndcg_match.group(1))
            if precision_match:
                metrics['precision@20'] = float(precision_match.group(1))
            if epoch_match:
                metrics['best_epoch'] = int(epoch_match.group(1))

    except Exception as e:
        print(f"Error parsing {log_file}: {e}")

    return metrics

def main():
    results_dir = Path('.')

    experiments = {
        'exp1_baseline': 'Baseline',
        'exp2_activity_only': 'Activity Only',
        'exp3_epoch_only': 'Epoch Only',
        'exp4_popularity_only': 'Popularity Only',
        'exp5_activity_epoch': 'Activity + Epoch',
        'exp6_activity_popularity': 'Activity + Popularity',
        'exp7_epoch_popularity': 'Epoch + Popularity',
        'exp8_full_adaptive': 'Full Adaptive (0.15)',
        'exp9_full_adaptive_flip0.10': 'Full Adaptive (0.10)',
        'exp10_full_adaptive_flip0.20': 'Full Adaptive (0.20)',
    }

    results = []

    for exp_id, exp_name in experiments.items():
        log_file = results_dir / exp_id / 'training.log'

        if log_file.exists():
            metrics = parse_training_log(log_file)
            results.append({
                'Experiment': exp_name,
                'Recall@20': metrics['recall@20'],
                'NDCG@20': metrics['ndcg@20'],
                'Precision@20': metrics['precision@20'],
                'Best Epoch': metrics['best_epoch']
            })
        else:
            print(f"Warning: {log_file} not found")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    df.to_csv('comparison_results.csv', index=False)
    print("\n" + "="*80)
    print("Experimental Results Comparison")
    print("="*80)
    print(df.to_string(index=False))
    print("\n" + "="*80)

    # Calculate improvements over baseline
    if len(df) > 0 and df.loc[0, 'Experiment'] == 'Baseline':
        baseline_recall = df.loc[0, 'Recall@20']
        baseline_ndcg = df.loc[0, 'NDCG@20']
        baseline_precision = df.loc[0, 'Precision@20']

        if baseline_recall:
            print("\nImprovements over Baseline:")
            print("-"*80)
            for idx, row in df.iterrows():
                if idx == 0:
                    continue
                if row['Recall@20']:
                    recall_imp = (row['Recall@20'] - baseline_recall) / baseline_recall * 100
                    ndcg_imp = (row['NDCG@20'] - baseline_ndcg) / baseline_ndcg * 100 if row['NDCG@20'] else 0
                    prec_imp = (row['Precision@20'] - baseline_precision) / baseline_precision * 100 if row['Precision@20'] else 0

                    print(f"{row['Experiment']:30s}: "
                          f"Recall {recall_imp:+.2f}%, "
                          f"NDCG {ndcg_imp:+.2f}%, "
                          f"Precision {prec_imp:+.2f}%")

    print("\nResults saved to comparison_results.csv")

if __name__ == '__main__':
    main()
EOF

log "Comparison script created: ${RESULTS_DIR}/compare_results.py"

###############################################################################
# Final Summary
###############################################################################

log "=================================================="
log "All Experiments Completed!"
log "=================================================="
log ""
log "Results Location: ${RESULTS_DIR}"
log ""
log "To view results summary:"
log "  cat ${SUMMARY_FILE}"
log ""
log "To compare results (requires pandas):"
log "  cd ${RESULTS_DIR}"
log "  python compare_results.py"
log ""
log "Individual experiment logs:"
for exp_dir in ${RESULTS_DIR}/exp*; do
    if [ -d "$exp_dir" ]; then
        log "  $(basename $exp_dir)/training.log"
    fi
done
log ""
log "=================================================="

# Make the script executable
chmod +x ${RESULTS_DIR}/compare_results.py

log "Experiment suite finished successfully!"
