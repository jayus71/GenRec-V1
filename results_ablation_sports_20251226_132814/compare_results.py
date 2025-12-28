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
