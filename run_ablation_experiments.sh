#!/bin/bash

# ============================================================================
# GenRec Ablation Study: DDIM + Importance Sampling
# ============================================================================
# This script runs ablation experiments to compare:
# 1. Baseline (original GenRec)
# 2. +DDIM (DDIM accelerated sampling)
# 3. +Importance Sampling (timestep importance sampling)
# 4. +Both (DDIM + Importance Sampling)
# ============================================================================

# Configuration
DATASET="sports"  # or "tiktok"
GPU_ID="1"
EPOCHS=50
BATCH_SIZE=1024

# Create results directory
RESULTS_DIR="./ablation_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

echo "============================================"
echo "  GenRec Ablation Experiments"
echo "  Dataset: $DATASET"
echo "  Results will be saved to: $RESULTS_DIR"
echo "============================================"
echo ""

# ============================================================================
# Experiment 1: Baseline
# ============================================================================
echo "[1/4] Running Baseline (Original GenRec)..."
python Main.py \
    --data $DATASET \
    --gpu $GPU_ID \
    --epoch $EPOCHS \
    --batch $BATCH_SIZE \
    --use_ddim false \
    --importance_sampling false \
    --steps 5 \
    --sampling_steps 5 \
    > $RESULTS_DIR/baseline.log 2>&1

echo "  ✓ Baseline completed. Log saved to $RESULTS_DIR/baseline.log"
echo ""

# ============================================================================
# Experiment 2: +DDIM
# ============================================================================
echo "[2/4] Running +DDIM (2-step accelerated sampling)..."
python Main.py \
    --data $DATASET \
    --gpu $GPU_ID \
    --epoch $EPOCHS \
    --batch $BATCH_SIZE \
    --use_ddim true \
    --ddim_steps 2 \
    --importance_sampling false \
    --steps 5 \
    --sampling_steps 5 \
    > $RESULTS_DIR/ddim.log 2>&1

echo "  ✓ +DDIM completed. Log saved to $RESULTS_DIR/ddim.log"
echo ""

# ============================================================================
# Experiment 3: +Importance Sampling
# ============================================================================
echo "[3/4] Running +Importance Sampling..."
python Main.py \
    --data $DATASET \
    --gpu $GPU_ID \
    --epoch $EPOCHS \
    --batch $BATCH_SIZE \
    --use_ddim false \
    --importance_sampling true \
    --loss_momentum 0.9 \
    --steps 5 \
    --sampling_steps 5 \
    > $RESULTS_DIR/importance.log 2>&1

echo "  ✓ +Importance Sampling completed. Log saved to $RESULTS_DIR/importance.log"
echo ""

# ============================================================================
# Experiment 4: +DDIM +Importance Sampling
# ============================================================================
echo "[4/4] Running +DDIM +Importance Sampling..."
python Main.py \
    --data $DATASET \
    --gpu $GPU_ID \
    --epoch $EPOCHS \
    --batch $BATCH_SIZE \
    --use_ddim true \
    --ddim_steps 2 \
    --importance_sampling true \
    --loss_momentum 0.9 \
    --steps 5 \
    --sampling_steps 5 \
    > $RESULTS_DIR/both.log 2>&1

echo "  ✓ +DDIM +Importance completed. Log saved to $RESULTS_DIR/both.log"
echo ""

# ============================================================================
# Extract Results
# ============================================================================
echo "============================================"
echo "  Extracting Results..."
echo "============================================"

# Create results summary file
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
echo "GenRec Ablation Study Results" > $SUMMARY_FILE
echo "==============================" >> $SUMMARY_FILE
echo "Dataset: $DATASET" >> $SUMMARY_FILE
echo "Epochs: $EPOCHS" >> $SUMMARY_FILE
echo "Batch Size: $BATCH_SIZE" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# Function to extract metrics from log file
extract_metrics() {
    local log_file=$1
    local exp_name=$2

    echo "[$exp_name]" >> $SUMMARY_FILE

    # Extract best Recall@20 and NDCG@20
    best_recall=$(grep -oP "recall@20:\s+\K[\d.]+" $log_file | sort -rn | head -1)
    best_ndcg=$(grep -oP "ndcg@20:\s+\K[\d.]+" $log_file | sort -rn | head -1)

    # Extract training time (if available)
    train_time=$(grep -oP "Total training time:\s+\K[\d.]+" $log_file | tail -1)

    echo "  Best Recall@20: ${best_recall:-N/A}" >> $SUMMARY_FILE
    echo "  Best NDCG@20:   ${best_ndcg:-N/A}" >> $SUMMARY_FILE
    echo "  Training Time:  ${train_time:-N/A}s" >> $SUMMARY_FILE
    echo "" >> $SUMMARY_FILE
}

# Extract metrics for each experiment
extract_metrics "$RESULTS_DIR/baseline.log" "Baseline"
extract_metrics "$RESULTS_DIR/ddim.log" "+DDIM"
extract_metrics "$RESULTS_DIR/importance.log" "+Importance Sampling"
extract_metrics "$RESULTS_DIR/both.log" "+DDIM +Importance"

# Display summary
echo ""
cat $SUMMARY_FILE
echo ""

echo "============================================"
echo "  All Experiments Completed!"
echo "  Results saved to: $RESULTS_DIR"
echo "============================================"
