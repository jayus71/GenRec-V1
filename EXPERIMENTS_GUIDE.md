# 实验脚本指南

本指南解释如何使用实验脚本来评估自适应翻转调度器。

## 可用脚本

### 1. `run_ablation_experiments.sh` - 完整消融实验

**目的**：运行包含 10 种不同配置的综合消融实验。

**包含的实验**：
1. 基线（无自适应调度）
2. 仅活跃度自适应调度
3. 仅 Epoch 自适应调度
4. 仅流行度自适应调度
5. 活跃度 + Epoch 自适应调度
6. 活跃度 + 流行度自适应调度
7. Epoch + 流行度自适应调度
8. 完全自适应调度（flip_prob=0.15）
9. 完全自适应调度（flip_prob=0.10）
10. 完全自适应调度（flip_prob=0.20）

**使用方法**：
```bash
# 设置可执行权限（仅首次）
chmod +x run_ablation_experiments.sh

# 在 Sports 数据集上使用 GPU 0 运行
bash run_ablation_experiments.sh sports 0

# 在 TikTok 数据集上使用 GPU 1 运行
bash run_ablation_experiments.sh tiktok 1

# 在 Baby 数据集上使用 GPU 0 运行（默认）
bash run_ablation_experiments.sh baby
```

**输出**：
- 创建 `results_ablation_[dataset]_[timestamp]/` 目录
- 每个实验都有自己的子目录和 `training.log` 文件
- `results_summary.txt`：所有结果的摘要
- `compare_results.py`：用于解析和比较指标的 Python 脚本

**时间估计**：约 8-15 小时完成 10 个实验（取决于数据集大小）

---

### 2. `run_quick_comparison.sh` - 快速基线 vs 自适应对比

**目的**：在基线和完全自适应调度之间进行快速比较。

**包含的实验**：
1. 基线（无自适应调度）
2. 完全自适应调度（启用所有三个维度）

**使用方法**：
```bash
# 设置可执行权限（仅首次）
chmod +x run_quick_comparison.sh

# 在 Sports 数据集上使用 GPU 0 运行
bash run_quick_comparison.sh sports 0

# 在 TikTok 数据集上运行
bash run_quick_comparison.sh tiktok 0
```

**输出**：
- 创建 `results_quick_[dataset]_[timestamp]/` 目录
- `baseline/training.log`：基线结果
- `full_adaptive/training.log`：完全自适应结果

**时间估计**：约 1.5-3 小时完成 2 个实验

---

### 3. `run_all_datasets.sh` - 跨数据集评估

**目的**：在所有三个数据集（TikTok、Baby、Sports）上运行基线 vs 完全自适应。

**包含的实验**：
- TikTok、Baby、Sports 上的基线
- TikTok、Baby、Sports 上的完全自适应

**使用方法**：
```bash
# 设置可执行权限（仅首次）
chmod +x run_all_datasets.sh

# 在 GPU 0 上运行
bash run_all_datasets.sh 0

# 在 GPU 1 上运行
bash run_all_datasets.sh 1
```

**输出**：
- 创建 `results_all_datasets_[timestamp]/` 目录
- 子目录：`tiktok/`、`baby/`、`sports/`
- 每个目录包含 `baseline.log` 和 `full_adaptive.log`

**时间估计**：约 4-9 小时完成 6 个实验（每个数据集 2 个）

---

## 分步使用指南

### 初始设置

1. **设置脚本可执行权限**：
```bash
cd /media/main/hongbo/python_projects/GenRec-V1
chmod +x run_ablation_experiments.sh
chmod +x run_quick_comparison.sh
chmod +x run_all_datasets.sh
```

2. **验证数据集可用性**：
```bash
ls Datasets/tiktok/
ls Datasets/baby/
ls Datasets/sports/
```

3. **解压 Baby 数据集图像**（如果尚未完成）：
```bash
cd Datasets/baby
unzip image_feat.npy.zip
cd ../..
```

### 运行实验

#### 快速测试（首先推荐）

从快速比较开始验证一切正常：

```bash
# 在 Sports 数据集上进行快速测试
bash run_quick_comparison.sh sports 0
```

监控输出是否有错误。结果将保存在 `results_quick_sports_[timestamp]/` 中。

#### 完整消融实验

快速测试成功后，运行完整的消融实验：

```bash
# 在 Sports 数据集上进行完整消融实验
bash run_ablation_experiments.sh sports 0
```

这将需要几个小时。你可以监控进度：
```bash
# 在另一个终端中查看日志
tail -f results_ablation_sports_*/experiment_log.txt
```

#### 跨数据集评估

在所有数据集上进行评估：

```bash
# 在所有数据集上运行
bash run_all_datasets.sh 0
```

### 分析结果

#### 选项 1：查看摘要（文本）

```bash
# 查看结果摘要
cat results_ablation_sports_*/results_summary.txt
```

#### 选项 2：使用 Python 比较（推荐）

```bash
# 导航到结果目录
cd results_ablation_sports_*/

# 运行比较脚本（需要 pandas）
python compare_results.py

# 查看 CSV 输出
cat comparison_results.csv
```

#### 选项 3：手动分析

从每个实验中提取最佳结果：

```bash
# 查找最佳 epoch 结果
grep -r "Best" results_ablation_sports_*/exp*/training.log

# 提取特定指标
grep -r "Recall@20" results_ablation_sports_*/exp*/training.log
grep -r "NDCG@20" results_ablation_sports_*/exp*/training.log
```

---

## 推荐的实验流程

### 阶段 1：快速验证（1-3 小时）
```bash
# 在 Sports 数据集上测试
bash run_quick_comparison.sh sports 0
```

**目标**：验证自适应调度相比基线有所改进。

### 阶段 2：组件消融（8-15 小时）
```bash
# 在 Sports 数据集上进行完整消融
bash run_ablation_experiments.sh sports 0
```

**目标**：了解哪些组件对改进贡献最大。

### 阶段 3：跨数据集验证（4-9 小时）
```bash
# 在所有数据集上测试
bash run_all_datasets.sh 0
```

**目标**：验证改进在各数据集上的泛化性。

### 阶段 4：超参数调优（可选）

手动运行不同翻转概率的实验：
```bash
python Main.py --data sports --gpu 0 --epoch 50 \
  --use_adaptive_flip True --flip_prob 0.12 \
  --use_activity_adaptive True --use_epoch_adaptive True --use_popularity_adaptive True

python Main.py --data sports --gpu 0 --epoch 50 \
  --use_adaptive_flip True --flip_prob 0.18 \
  --use_activity_adaptive True --use_epoch_adaptive True --use_popularity_adaptive True
```

---

## 预期结果

基于设计，你应该观察到：

### Sport 数据集（高度倾斜）
- **基线 Recall@20**：约 0.040-0.050
- **完全自适应 Recall@20**：预期改进 +5-15%
- **优势**：由于稀疏用户占比高（63.60%），改进最大

### Baby 数据集（中度倾斜）
- **基线 Recall@20**：约 0.045-0.055
- **完全自适应 Recall@20**：预期改进 +3-10%

### TikTok 数据集（最密集）
- **基线 Recall@20**：约 0.050-0.060
- **完全自适应 Recall@20**：预期改进 +2-8%

### 组件贡献（预期）
- **仅活跃度**：+3-8% 改进
- **仅 Epoch**：+2-5% 改进
- **仅流行度**：+1-3% 改进
- **完全（三者都有）**：+5-15% 改进（协同效应）

---

## 故障排查

### 脚本无法运行
```bash
# 检查权限
ls -l run_*.sh

# 设置可执行权限
chmod +x run_ablation_experiments.sh
chmod +x run_quick_comparison.sh
chmod +x run_all_datasets.sh
```

### GPU 内存不足
在脚本中减小批量大小：
```bash
# 编辑脚本
nano run_quick_comparison.sh

# 将 BATCH_SIZE=1024 改为 BATCH_SIZE=512
```

### 数据集未找到
```bash
# 验证数据集存在
ls Datasets/sports/

# 检查所需文件
ls Datasets/sports/trnMat.pkl
ls Datasets/sports/tstMat.pkl
ls Datasets/sports/image_feat.npy
ls Datasets/sports/text_feat.npy
```

### Baby 数据集图像特征
```bash
# 如果看到 baby 数据集的 "image_feat.npy not found"
cd Datasets/baby/
unzip image_feat.npy.zip
cd ../..
```

### Python 比较脚本失败
```bash
# 如果需要安装 pandas
pip install pandas

# 或使用 conda
conda install pandas
```

---

## 输出目录结构

运行 `run_ablation_experiments.sh sports 0` 后：

```
results_ablation_sports_20250124_143022/
├── config.txt                    # 实验配置
├── experiment_log.txt            # 主日志文件
├── results_summary.txt           # 结果的文本摘要
├── compare_results.py            # 比较脚本
├── comparison_results.csv        # 解析的结果（运行 compare_results.py 后）
├── exp1_baseline/
│   └── training.log
├── exp2_activity_only/
│   └── training.log
├── exp3_epoch_only/
│   └── training.log
├── exp4_popularity_only/
│   └── training.log
├── exp5_activity_epoch/
│   └── training.log
├── exp6_activity_popularity/
│   └── training.log
├── exp7_epoch_popularity/
│   └── training.log
├── exp8_full_adaptive/
│   └── training.log
├── exp9_full_adaptive_flip0.10/
│   └── training.log
└── exp10_full_adaptive_flip0.20/
    └── training.log
```

---

## 高效实验的技巧

1. **从小处着手**：先使用 `run_quick_comparison.sh` 验证设置

2. **监控进度**：使用 `tail -f` 实时观察日志

3. **过夜运行**：完整的消融实验可能需要 8-15 小时

4. **保存结果**：结果目录带有时间戳，因此可以运行多个实验

5. **GPU 利用率**：如果有多个 GPU，可以并行运行不同的数据集：
   ```bash
   # 终端 1
   bash run_ablation_experiments.sh sports 0 &

   # 终端 2
   bash run_ablation_experiments.sh tiktok 1 &

   # 终端 3
   bash run_ablation_experiments.sh baby 2 &
   ```

6. **跨运行比较**：保留所有结果目录以比较不同的配置

---

## 结果解读

### 关键指标

- **Recall@20**：Top-20 推荐中相关物品的比例
  - 越高越好
  - 对稀疏用户最重要的指标

- **NDCG@20**：归一化折扣累积增益（在 20）
  - 考虑排序质量
  - 越高越好

- **Precision@20**：Top-20 推荐的精确度
  - 越高越好
  - 补充召回率

### 关注要点

1. **基线 vs 完全自适应**：应该显示明显改进
2. **组件贡献**：哪个维度最重要
3. **翻转概率敏感性**：flip_prob 对结果的影响有多大
4. **数据集依赖性**：改进是否因数据集特征而异

### 报告结果

报告结果时应包括：
- 数据集统计信息（用户/物品数量、稀疏度）
- 最佳配置和超参数
- 比较表（基线 vs 自适应）
- 消融实验结果（组件贡献）
- 统计显著性（如果有多次运行）

---

## 联系方式

有关这些脚本的问题或疑问，请参阅：
- `ADAPTIVE_FLIP_SCHEDULER.md`：调度器文档
- `IMPLEMENTATION_SUMMARY.md`：技术实现细节
- 主仓库 README 了解一般设置问题
