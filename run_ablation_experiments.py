#!/usr/bin/env python3
"""
GenRec 消融实验脚本
对比 DDIM 和重要性采样的效果
"""

import os
import subprocess
import time
from datetime import datetime
import json
import re

# 实验配置
EXPERIMENTS = {
    "baseline": {
        "name": "Baseline (原始GenRec)",
        "args": {
            "use_ddim": "false",
            "importance_sampling": "false",
            "steps": 5,
            "sampling_steps": 5
        }
    },
    "ddim": {
        "name": "+DDIM (2步加速采样)",
        "args": {
            "use_ddim": "true",
            "ddim_steps": 2,
            "importance_sampling": "false",
            "steps": 5,
            "sampling_steps": 5
        }
    },
    "ddim_3steps": {
        "name": "+DDIM (3步采样)",
        "args": {
            "use_ddim": "true",
            "ddim_steps": 3,
            "importance_sampling": "false",
            "steps": 5,
            "sampling_steps": 5
        }
    },
    "importance": {
        "name": "+重要性采样",
        "args": {
            "use_ddim": "false",
            "importance_sampling": "true",
            "loss_momentum": 0.9,
            "steps": 5,
            "sampling_steps": 5
        }
    },
    "both": {
        "name": "+DDIM +重要性采样",
        "args": {
            "use_ddim": "true",
            "ddim_steps": 2,
            "importance_sampling": "true",
            "loss_momentum": 0.9,
            "steps": 5,
            "sampling_steps": 5
        }
    }
}

# 基础配置
BASE_CONFIG = {
    "data": "allrecipes",  # 或 "tiktok"
    "gpu": "0",
    "epoch": 50,
    "batch": 1024,
    "lr": 1e-3,
    "topk": 20
}


def run_experiment(exp_name, exp_config, results_dir):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"  运行实验: {exp_config['name']}")
    print(f"{'='*60}")

    # 构建命令行参数
    cmd = ["python", "Main.py"]

    # 添加基础配置
    for key, value in BASE_CONFIG.items():
        cmd.extend([f"--{key}", str(value)])

    # 添加实验特定配置
    for key, value in exp_config["args"].items():
        cmd.extend([f"--{key}", str(value)])

    # 日志文件路径
    log_file = os.path.join(results_dir, f"{exp_name}.log")

    print(f"  命令: {' '.join(cmd)}")
    print(f"  日志: {log_file}\n")

    # 运行实验
    start_time = time.time()
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # 实时输出并保存到文件
        for line in process.stdout:
            print(f"  {line.rstrip()}")
            f.write(line)
            f.flush()

        process.wait()

    elapsed_time = time.time() - start_time

    if process.returncode == 0:
        print(f"\n  ✓ 实验完成 (耗时: {elapsed_time:.1f}秒)")
    else:
        print(f"\n  ✗ 实验失败 (返回码: {process.returncode})")

    return elapsed_time, log_file


def extract_metrics(log_file):
    """从日志文件中提取指标"""
    metrics = {
        "best_recall": None,
        "best_ndcg": None,
        "best_precision": None,
        "best_epoch": None,
        "training_time": None
    }

    if not os.path.exists(log_file):
        return metrics

    with open(log_file, "r") as f:
        content = f.read()

        # 提取最终结果行: Best epoch :  11  , Recall :  0.0991  , NDCG :  0.0442  , Precision 0.0052  , Time :  1234.5
        # 注意: Precision 后面没有冒号
        best_pattern = r"Best epoch\s*:\s*(\d+)\s*,\s*Recall\s*:\s*([\d.]+)\s*,\s*NDCG\s*:\s*([\d.]+)\s*,\s*Precision\s*([\d.]+)(?:\s*,\s*Time\s*:\s*([\d.]+))?"
        match = re.search(best_pattern, content)

        if match:
            metrics["best_epoch"] = int(match.group(1))
            metrics["best_recall"] = float(match.group(2))
            metrics["best_ndcg"] = float(match.group(3))
            metrics["best_precision"] = float(match.group(4))
            if match.group(5):  # Time is optional (old logs may not have it)
                metrics["training_time"] = float(match.group(5))

    return metrics


def print_summary(results, results_dir):
    """打印和保存实验总结"""
    print(f"\n\n{'='*100}")
    print("  实验结果总结")
    print(f"{'='*100}\n")

    # 准备表格数据
    table_data = []
    baseline_recall = None
    baseline_ndcg = None
    baseline_precision = None
    baseline_time = None

    for exp_name, exp_info in results.items():
        metrics = exp_info["metrics"]

        # 记录 baseline
        if exp_name == "baseline":
            baseline_recall = metrics["best_recall"]
            baseline_ndcg = metrics["best_ndcg"]
            baseline_precision = metrics["best_precision"]
            baseline_time = metrics["training_time"] or exp_info["elapsed_time"]

        # 计算改进幅度
        recall_improve = ""
        ndcg_improve = ""
        precision_improve = ""
        time_improve = ""

        if baseline_recall and metrics["best_recall"]:
            improve = ((metrics["best_recall"] - baseline_recall) / baseline_recall) * 100
            recall_improve = f"({improve:+.2f}%)"

        if baseline_ndcg and metrics["best_ndcg"]:
            improve = ((metrics["best_ndcg"] - baseline_ndcg) / baseline_ndcg) * 100
            ndcg_improve = f"({improve:+.2f}%)"

        if baseline_precision and metrics["best_precision"]:
            improve = ((metrics["best_precision"] - baseline_precision) / baseline_precision) * 100
            precision_improve = f"({improve:+.2f}%)"

        current_time = metrics["training_time"] or exp_info["elapsed_time"]
        if baseline_time and current_time:
            improve = ((current_time - baseline_time) / baseline_time) * 100
            time_improve = f"({improve:+.2f}%)"

        table_data.append({
            "name": exp_info["config"]["name"],
            "recall": metrics["best_recall"] or "N/A",
            "recall_improve": recall_improve,
            "ndcg": metrics["best_ndcg"] or "N/A",
            "ndcg_improve": ndcg_improve,
            "precision": metrics["best_precision"] or "N/A",
            "precision_improve": precision_improve,
            "time": current_time,
            "time_improve": time_improve
        })

    # 打印表格
    print(f"{'实验配置':<25} {'Recall@20':<18} {'NDCG@20':<18} {'Precision@20':<18} {'时间(秒)':<18}")
    print("-" * 100)

    for row in table_data:
        recall_str = f"{row['recall']:.4f} {row['recall_improve']}" if isinstance(row['recall'], float) else row['recall']
        ndcg_str = f"{row['ndcg']:.4f} {row['ndcg_improve']}" if isinstance(row['ndcg'], float) else row['ndcg']
        precision_str = f"{row['precision']:.4f} {row['precision_improve']}" if isinstance(row['precision'], float) else row['precision']
        time_str = f"{row['time']:.1f} {row['time_improve']}" if row['time'] else "N/A"

        print(f"{row['name']:<25} {recall_str:<18} {ndcg_str:<18} {precision_str:<18} {time_str:<18}")

    print("")

    # 保存到 JSON
    summary_file = os.path.join(results_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✓ 详细结果已保存到: {summary_file}\n")

    # 保存到文本文件
    summary_txt = os.path.join(results_dir, "summary.txt")
    with open(summary_txt, "w", encoding='utf-8') as f:
        f.write("GenRec 消融实验结果\n")
        f.write("=" * 100 + "\n")
        f.write(f"数据集: {BASE_CONFIG['data']}\n")
        f.write(f"训练轮数: {BASE_CONFIG['epoch']}\n")
        f.write(f"批次大小: {BASE_CONFIG['batch']}\n")
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"{'实验配置':<25} {'Recall@20':<18} {'NDCG@20':<18} {'Precision@20':<18} {'时间(秒)':<18}\n")
        f.write("-" * 100 + "\n")

        for row in table_data:
            recall_str = f"{row['recall']:.4f} {row['recall_improve']}" if isinstance(row['recall'], float) else row['recall']
            ndcg_str = f"{row['ndcg']:.4f} {row['ndcg_improve']}" if isinstance(row['ndcg'], float) else row['ndcg']
            precision_str = f"{row['precision']:.4f} {row['precision_improve']}" if isinstance(row['precision'], float) else row['precision']
            time_str = f"{row['time']:.1f} {row['time_improve']}" if row['time'] else "N/A"

            f.write(f"{row['name']:<25} {recall_str:<18} {ndcg_str:<18} {precision_str:<18} {time_str:<18}\n")

    print(f"✓ 文本总结已保存到: {summary_txt}\n")


def main():
    """主函数"""
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"./ablation_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("  GenRec 消融实验")
    print("=" * 80)
    print(f"  数据集: {BASE_CONFIG['data']}")
    print(f"  训练轮数: {BASE_CONFIG['epoch']}")
    print(f"  批次大小: {BASE_CONFIG['batch']}")
    print(f"  实验数量: {len(EXPERIMENTS)}")
    print(f"  结果目录: {results_dir}")
    print("=" * 80)

    # 运行所有实验
    results = {}
    for i, (exp_name, exp_config) in enumerate(EXPERIMENTS.items(), 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] {exp_config['name']}")

        elapsed_time, log_file = run_experiment(exp_name, exp_config, results_dir)

        # 提取指标
        metrics = extract_metrics(log_file)

        results[exp_name] = {
            "config": exp_config,
            "elapsed_time": elapsed_time,
            "log_file": log_file,
            "metrics": metrics
        }

    # 打印总结
    print_summary(results, results_dir)

    print("=" * 80)
    print("  所有实验完成!")
    print(f"  结果保存在: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
