# 自适应翻转概率调度器

## 概述

本文档描述了为 GenRec-V1 实现的**自适应翻转概率调度器**，基于 Sport 数据集的统计特征设计，适用于所有数据集（TikTok、Baby、Sports）。

## 动机

Sport 数据集呈现高度倾斜的用户活跃度分布：
- **63.60%** 的用户仅有 3-5 次交互（极度稀疏）
- **24.52%** 的用户有 5-10 次交互（中度稀疏）
- **9.85%** 的用户有 10-20 次交互（相对密集）
- **2.03%** 的用户有 20+ 次交互（非常密集）

对所有用户使用固定的翻转概率是次优的，因为：
1. **稀疏用户**需要更高的翻转概率来探索多样化的兴趣
2. **密集用户**需要更低的翻转概率来保持精确性
3. 训练动态随 epoch 变化（探索 → 利用）

## 设计

自适应调度器使用**三维调度**：

### 1. 基于用户活跃度的调度

用户根据交互次数被分为 5 组：

| 组别 | 交互范围 | 百分比（Sport） | 翻转乘数 |
|-------|---------|----------------|---------|
| 极低 | 0-5 | 63.60% | **1.5x** |
| 低 | 5-10 | 24.52% | **1.2x** |
| 中等 | 10-20 | 9.85% | **1.0x**（基线） |
| 高 | 20-50 | 1.93% | **0.7x** |
| 极高 | 50+ | 0.10% | **0.5x** |

**原理：**
- 稀疏用户获得提升的翻转概率以生成更多样化的兴趣
- 密集用户获得降低的翻转概率以避免噪声

### 2. 基于 Epoch 的调度

翻转概率在训练 epoch 中动态变化：

| 训练阶段 | Epoch 进度 | Epoch 乘数 | 策略 |
|---------|-----------|-----------|------|
| 早期 | 0-20% | **1.3x** | 高度探索 |
| 中期 | 20-60% | **1.3x → 0.8x** | 逐步过渡 |
| 后期 | 60-100% | **0.8x → 0.5x** | 利用为主 |

**原理：**
- 早期 epoch：高翻转用于探索和兴趣发现
- 中期 epoch：逐步衰减平衡探索和利用
- 后期 epoch：低翻转用于精确推荐

### 3. 基于物品流行度的调度（可选）

物品根据流行度分类：

| 流行度 | 条件 | 乘数 |
|-------|------|------|
| 长尾 | 低于中位数 | **1.2x** |
| 中等流行度 | 中位数到 75% | **1.0x** |
| 热门 | 高于 75% | **0.8x** |

**原理：**
- 长尾物品获得更高的翻转概率以增加曝光
- 热门物品获得更低的翻转概率（已经充分展示）

### 组合翻转概率

用户-物品对的最终翻转概率为：

```
final_flip_prob = base_flip_prob × activity_multiplier × epoch_multiplier × item_multiplier
```

限制在范围内：`[0.01, 0.95]`

## 使用方法

### 使用自适应调度器的基本训练（推荐）

```bash
# 在 Sport 数据集上使用自适应翻转调度训练
python Main.py --data sports --use_adaptive_flip True --flip_prob 0.15

# 在 TikTok 数据集上训练
python Main.py --data tiktok --use_adaptive_flip True --flip_prob 0.15

# 在 Baby 数据集上训练
python Main.py --data baby --use_adaptive_flip True --flip_prob 0.15
```

### 自定义自适应调度组件

```bash
# 仅启用基于用户活跃度的调度
python Main.py --data sports \
  --use_adaptive_flip True \
  --use_activity_adaptive True \
  --use_epoch_adaptive False \
  --use_popularity_adaptive False

# 仅启用基于 epoch 的调度
python Main.py --data sports \
  --use_adaptive_flip True \
  --use_activity_adaptive False \
  --use_epoch_adaptive True \
  --use_popularity_adaptive False

# 启用所有三个维度（默认）
python Main.py --data sports \
  --use_adaptive_flip True \
  --use_activity_adaptive True \
  --use_epoch_adaptive True \
  --use_popularity_adaptive True
```

### 禁用自适应调度（基线）

```bash
# 使用固定翻转概率（原始实现）
python Main.py --data sports --use_adaptive_flip False
```

### 调整基础翻转概率

```bash
# 使用更高的基础翻转概率
python Main.py --data sports --use_adaptive_flip True --flip_prob 0.20

# 使用更低的基础翻转概率
python Main.py --data sports --use_adaptive_flip True --flip_prob 0.10
```

## 参数

所有参数在 `Params.py` 中定义：

| 参数 | 类型 | 默认值 | 描述 |
|-----|------|-------|------|
| `--use_adaptive_flip` | bool | True | 启用/禁用自适应翻转调度器 |
| `--flip_prob` | float | 0.15 | 基础翻转概率 |
| `--use_activity_adaptive` | bool | True | 启用基于用户活跃度的调度 |
| `--use_epoch_adaptive` | bool | True | 启用基于 epoch 的调度 |
| `--use_popularity_adaptive` | bool | True | 启用基于物品流行度的调度 |

## 实现细节

### 文件结构

```
GenRec-V1/
├── adaptive_flip_scheduler.py    # 调度器实现
├── Model.py                       # 带调度器支持的 FlipInterestDiffusion
├── Main.py                        # 集成调度器的训练循环
├── Params.py                      # 超参数定义
└── ADAPTIVE_FLIP_SCHEDULER.md    # 本文档
```

### 核心类和方法

**`AdaptiveFlipScheduler`**（在 `adaptive_flip_scheduler.py` 中）：
- `get_user_flip_probabilities(user_ids, current_epoch)`：返回每个用户的翻转概率
- `get_item_flip_probabilities(item_ids)`：返回基于物品的乘数
- `get_adaptive_flip_matrix(user_ids, item_ids, current_epoch)`：组合自适应概率
- `print_epoch_stats(current_epoch)`：打印调度统计信息

**`FlipInterestDiffusion`**（在 `Model.py` 中）：
- 修改 `__init__()` 以接受 `adaptive_scheduler` 参数
- 修改 `q_sample()` 以使用自适应翻转概率
- 修改 `training_losses()` 以接受 `user_ids` 和 `current_epoch`

**`Coach.trainEpoch()`**（在 `Main.py` 中）：
- 在 `prepareModel()` 中初始化调度器
- 在 epoch 开始时打印调度器统计信息
- 将 user_ids 和 current_epoch 传递给扩散训练

## 预期输出

启用自适应调度器训练时，你会看到如下输出：

```
=== AdaptiveFlipScheduler Initialized ===
Total users: 35598
Activity group distribution:
  extreme_low: 22639 users (63.60%)
  low: 8728 users (24.52%)
  medium: 3505 users (9.85%)
  high: 686 users (1.93%)
  extreme_high: 40 users (0.11%)
Base flip probability: 0.15
Item popularity median: 6.00
Item popularity 75%: 12.00
✓ Adaptive flip scheduler initialized with base_flip_prob=0.15

=== Epoch 0 Flip Probability Schedule ===
Epoch multiplier: 1.300
Activity-based flip probabilities:
  extreme_low    : 0.2925 (22639 users, 63.60%)
  low            : 0.2340 ( 8728 users, 24.52%)
  medium         : 0.1950 ( 3505 users,  9.85%)
  high           : 0.1365 (  686 users,  1.93%)
  extreme_high   : 0.0975 (   40 users,  0.11%)

=== Epoch 25 Flip Probability Schedule ===
Epoch multiplier: 0.925
Activity-based flip probabilities:
  extreme_low    : 0.2081 (22639 users, 63.60%)
  low            : 0.1665 ( 8728 users, 24.52%)
  medium         : 0.1388 ( 3505 users,  9.85%)
  high           : 0.0971 (  686 users,  1.93%)
  extreme_high   : 0.0694 (   40 users,  0.11%)

=== Epoch 49 Flip Probability Schedule ===
Epoch multiplier: 0.520
Activity-based flip probabilities:
  extreme_low    : 0.1170 (22639 users, 63.60%)
  low            : 0.0936 ( 8728 users, 24.52%)
  medium         : 0.0780 ( 3505 users,  9.85%)
  high           : 0.0546 (  686 users,  1.93%)
  extreme_high   : 0.0390 (   40 users,  0.11%)
```

## 实验建议

### 消融实验

在 Sport 数据集上比较以下配置：

1. **基线（固定翻转）**：
   ```bash
   python Main.py --data sports --use_adaptive_flip False
   ```

2. **仅活跃度自适应**：
   ```bash
   python Main.py --data sports --use_adaptive_flip True \
     --use_activity_adaptive True --use_epoch_adaptive False --use_popularity_adaptive False
   ```

3. **仅 Epoch 自适应**：
   ```bash
   python Main.py --data sports --use_adaptive_flip True \
     --use_activity_adaptive False --use_epoch_adaptive True --use_popularity_adaptive False
   ```

4. **完全自适应（推荐）**：
   ```bash
   python Main.py --data sports --use_adaptive_flip True \
     --use_activity_adaptive True --use_epoch_adaptive True --use_popularity_adaptive True
   ```

### 超参数调优

建议尝试的基础翻转概率：
- `--flip_prob 0.10`：保守（较少探索）
- `--flip_prob 0.15`：**默认（平衡）**
- `--flip_prob 0.20`：激进（更多探索）

## 预期改进

基于设计原理，自适应调度器应该：

1. **改进稀疏用户的 Recall@20**（63.60% 的用户）
   - 更高的翻转概率生成更多样化的兴趣
   - 缓解冷启动问题

2. **改进所有用户的 NDCG@20**
   - 基于 epoch 的调度平衡探索和利用
   - 后期利用改进排序质量

3. **更好的长尾物品覆盖**
   - 基于物品流行度的调度提升长尾曝光
   - 减少流行度偏差

4. **更快的收敛**
   - 自适应调度与训练动态对齐
   - 早期探索，后期利用

## 引用

如果你在研究中使用此自适应翻转调度器，请引用：

```bibtex
@inproceedings{he2025flip,
  title={Flip is Better than Noise: Unbiased Interest Generation for Multimedia Recommendation},
  author={He, Yue and Xie, Jingxi and Li, Fengling and Zhu, Lei and Li, Jingjing},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={6298--6306},
  year={2025}
}
```

## 联系方式

有关自适应翻转调度器的问题或疑问，请在仓库中提交 issue。
