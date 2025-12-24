import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.sparse import coo_matrix
    import scipy.sparse as sp
    return coo_matrix, np, pickle, plt, sp


@app.cell
def _(coo_matrix, np, pickle, sp):
    # 加载 TikTok 数据集的训练和测试交互矩阵
    predir = './Datasets/tiktok/'
    trnfile = predir + 'trnMat.pkl'
    tstfile = predir + 'tstMat.pkl'

    def loadOneFile(filename):
        with open(filename, 'rb') as fs:
            ret = (pickle.load(fs) != 0).astype(np.float32)
        if type(ret) != coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret

    trnMat = loadOneFile(trnfile)
    tstMat = loadOneFile(tstfile)

    print(f"训练集矩阵形状 (用户数, 物品数): {trnMat.shape}")
    print(f"测试集矩阵形状 (用户数, 物品数): {tstMat.shape}")
    print(f"训练集交互数量: {trnMat.nnz}")
    print(f"测试集交互数量: {tstMat.nnz}")
    return trnMat, tstMat


@app.cell
def _(np, trnMat):
    # 统计每个用户的交互物品数量（训练集）
    user_interactions_trn = np.array(trnMat.sum(axis=1)).flatten()

    print("\n=== 训练集用户交互统计 ===")
    print(f"平均交互数: {user_interactions_trn.mean():.2f}")
    print(f"中位数交互数: {np.median(user_interactions_trn):.2f}")
    print(f"最小交互数: {user_interactions_trn.min()}")
    print(f"最大交互数: {user_interactions_trn.max()}")
    print(f"标准差: {user_interactions_trn.std():.2f}")
    return (user_interactions_trn,)


@app.cell
def _(np, tstMat):
    # 统计每个用户的交互物品数量（测试集）
    user_interactions_tst = np.array(tstMat.sum(axis=1)).flatten()

    print("\n=== 测试集用户交互统计 ===")
    print(f"平均交互数: {user_interactions_tst.mean():.2f}")
    print(f"中位数交互数: {np.median(user_interactions_tst):.2f}")
    print(f"最小交互数: {user_interactions_tst.min()}")
    print(f"最大交互数: {user_interactions_tst.max()}")
    print(f"标准差: {user_interactions_tst.std():.2f}")
    return


@app.cell
def _(np, trnMat, tstMat):
    # 合并训练集和测试集统计总体交互
    total_interactions = np.array((trnMat + tstMat).sum(axis=1)).flatten()

    print("\n=== 总体用户交互统计 (训练集 + 测试集) ===")
    print(f"平均交互数: {total_interactions.mean():.2f}")
    print(f"中位数交互数: {np.median(total_interactions):.2f}")
    print(f"最小交互数: {total_interactions.min()}")
    print(f"最大交互数: {total_interactions.max()}")
    print(f"标准差: {total_interactions.std():.2f}")
    return (total_interactions,)


@app.cell
def _(np, user_interactions_trn):
    # 活跃度分析（基于训练集）
    # 定义活跃度等级
    percentiles = [25, 50, 75, 90, 95, 99]

    print("\n=== 用户活跃度分位数分析 (训练集) ===")
    for p in percentiles:
        val = np.percentile(user_interactions_trn, p)
        print(f"{p}% 分位数: {val:.2f}")

    # 统计不同活跃度区间的用户数量
    bins = [0, 5, 10, 20, 50, 100, float('inf')]
    labels = ['极低活跃(0-5)', '低活跃(5-10)', '中活跃(10-20)',
              '高活跃(20-50)', '极高活跃(50-100)', '超高活跃(100+)']

    print("\n=== 用户活跃度分布 ===")
    for i in range(len(bins)-1):
        count = np.sum((user_interactions_trn > bins[i]) & (user_interactions_trn <= bins[i+1]))
        percentage = count / len(user_interactions_trn) * 100
        print(f"{labels[i]}: {count} 用户 ({percentage:.2f}%)")
    return


@app.cell
def _(np, total_interactions, user_interactions_trn):
    # 找出最活跃和最不活跃的用户
    print("\n=== 极端用户分析 ===")

    # 最不活跃的用户（训练集交互数为0或极少）
    inactive_threshold = 3
    inactive_users = np.where(user_interactions_trn <= inactive_threshold)[0]
    print(f"\n训练集交互数 <= {inactive_threshold} 的用户数: {len(inactive_users)} ({len(inactive_users)/len(user_interactions_trn)*100:.2f}%)")

    # 最活跃的用户（训练集交互数超过均值+1个标准差）
    active_threshold = user_interactions_trn.mean() + user_interactions_trn.std()
    active_users = np.where(user_interactions_trn >= active_threshold)[0]
    print(f"训练集交互数 >= {active_threshold:.2f} 的活跃用户数: {len(active_users)} ({len(active_users)/len(user_interactions_trn)*100:.2f}%)")

    # Top 10 最活跃用户
    top_k = 10
    top_users_idx = np.argsort(user_interactions_trn)[-top_k:][::-1]
    print(f"\n=== Top {top_k} 最活跃用户 (训练集) ===")
    for rank, user_idx in enumerate(top_users_idx, 1):
        trn_count = user_interactions_trn[user_idx]
        total_count = total_interactions[user_idx]
        print(f"#{rank} 用户ID {user_idx}: 训练集交互 {trn_count:.0f}, 总交互 {total_count:.0f}")

    # Bottom 10 最不活跃用户（但有交互的）
    bottom_users_idx = np.argsort(user_interactions_trn)[:top_k]
    print(f"\n=== Bottom {top_k} 最不活跃用户 (训练集) ===")
    for rank, user_idx in enumerate(bottom_users_idx, 1):
        trn_count = user_interactions_trn[user_idx]
        total_count = total_interactions[user_idx]
        print(f"#{rank} 用户ID {user_idx}: 训练集交互 {trn_count:.0f}, 总交互 {total_count:.0f}")
    return


@app.cell
def _(np, plt, user_interactions_trn):
    # 可视化：用户交互数量分布
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 直方图 - 完整分布
    axes[0, 0].hist(user_interactions_trn, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('交互物品数量')
    axes[0, 0].set_ylabel('用户数量')
    axes[0, 0].set_title('用户交互数量分布 (完整)')
    axes[0, 0].grid(True, alpha=0.3)

    # 直方图 - 截断版本（只看前95%）
    threshold_95 = np.percentile(user_interactions_trn, 95)
    filtered_interactions = user_interactions_trn[user_interactions_trn <= threshold_95]
    axes[0, 1].hist(filtered_interactions, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('交互物品数量')
    axes[0, 1].set_ylabel('用户数量')
    axes[0, 1].set_title(f'用户交互数量分布 (截断至95%分位数: {threshold_95:.0f})')
    axes[0, 1].grid(True, alpha=0.3)

    # 箱线图
    axes[1, 0].boxplot(user_interactions_trn, vert=True)
    axes[1, 0].set_ylabel('交互物品数量')
    axes[1, 0].set_title('用户交互数量箱线图')
    axes[1, 0].grid(True, alpha=0.3)

    # 累积分布函数 (CDF)
    sorted_interactions = np.sort(user_interactions_trn)
    cumulative = np.arange(1, len(sorted_interactions) + 1) / len(sorted_interactions)
    axes[1, 1].plot(sorted_interactions, cumulative, linewidth=2)
    axes[1, 1].set_xlabel('交互物品数量')
    axes[1, 1].set_ylabel('累积概率')
    axes[1, 1].set_title('用户交互数量累积分布函数 (CDF)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tiktok_user_interactions_analysis.png', dpi=150, bbox_inches='tight')
    print("图表已保存至: tiktok_user_interactions_analysis.png")

    fig
    return


@app.cell
def _(np, trnMat):
    # 物品交互统计（被多少用户交互过）
    item_interactions = np.array(trnMat.sum(axis=0)).flatten()

    print("\n=== 物品交互统计 (训练集) ===")
    print(f"平均被交互次数: {item_interactions.mean():.2f}")
    print(f"中位数被交互次数: {np.median(item_interactions):.2f}")
    print(f"最少被交互次数: {item_interactions.min()}")
    print(f"最多被交互次数: {item_interactions.max()}")
    print(f"标准差: {item_interactions.std():.2f}")

    # 冷启动物品统计
    cold_items = np.sum(item_interactions == 0)
    print(f"\n冷启动物品数（无交互）: {cold_items} ({cold_items/len(item_interactions)*100:.2f}%)")
    return


@app.cell
def _(trnMat):
    # 稀疏度分析
    total_possible_interactions = trnMat.shape[0] * trnMat.shape[1]
    actual_interactions = trnMat.nnz
    sparsity = 1 - (actual_interactions / total_possible_interactions)

    print("\n=== 矩阵稀疏度分析 ===")
    print(f"可能的交互总数: {total_possible_interactions:,}")
    print(f"实际交互数: {actual_interactions:,}")
    print(f"稀疏度: {sparsity*100:.4f}%")
    print(f"密度: {(1-sparsity)*100:.4f}%")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(coo_matrix, np, pickle, sp):
    def _():
        # 加载 Sports 数据集的训练和测试交互矩阵
        sports_predir = './Datasets/sports/'
        sports_trnfile = sports_predir + 'trnMat.pkl'
        sports_tstfile = sports_predir + 'tstMat.pkl'

        def _sports_load_one_file(filename):
            with open(filename, 'rb') as fs:
                ret = (pickle.load(fs) != 0).astype(np.float32)
            if type(ret) != coo_matrix:
                ret = sp.coo_matrix(ret)
            return ret

        sports_trnMat = _sports_load_one_file(sports_trnfile)
        sports_tstMat = _sports_load_one_file(sports_tstfile)

        print(f"训练集矩阵形状 (用户数, 物品数): {sports_trnMat.shape}")
        print(f"测试集矩阵形状 (用户数, 物品数): {sports_tstMat.shape}")
        print(f"训练集交互数量: {sports_trnMat.nnz}")
        print(f"测试集交互数量: {sports_tstMat.nnz}")

        # 用户交互统计
        sports_user_interactions_trn = np.array(sports_trnMat.sum(axis=1)).flatten()
        sports_user_interactions_tst = np.array(sports_tstMat.sum(axis=1)).flatten()
        sports_total_interactions = np.array((sports_trnMat + sports_tstMat).sum(axis=1)).flatten()

        print("\n=== Sports 训练集用户交互统计 ===")
        print(f"平均交互数: {sports_user_interactions_trn.mean():.2f}")
        print(f"中位数交互数: {np.median(sports_user_interactions_trn):.2f}")
        print(f"最小交互数: {sports_user_interactions_trn.min()}")
        print(f"最大交互数: {sports_user_interactions_trn.max()}")
        print(f"标准差: {sports_user_interactions_trn.std():.2f}")

        print("\n=== Sports 测试集用户交互统计 ===")
        print(f"平均交互数: {sports_user_interactions_tst.mean():.2f}")
        print(f"中位数交互数: {np.median(sports_user_interactions_tst):.2f}")
        print(f"最小交互数: {sports_user_interactions_tst.min()}")
        print(f"最大交互数: {sports_user_interactions_tst.max()}")
        print(f"标准差: {sports_user_interactions_tst.std():.2f}")

        print("\n=== Sports 总体用户交互统计 (训练集 + 测试集) ===")
        print(f"平均交互数: {sports_total_interactions.mean():.2f}")
        print(f"中位数交互数: {np.median(sports_total_interactions):.2f}")
        print(f"最小交互数: {sports_total_interactions.min()}")
        print(f"最大交互数: {sports_total_interactions.max()}")
        print(f"标准差: {sports_total_interactions.std():.2f}")

        # 活跃度分位数
        sports_percentiles = [25, 50, 75, 90, 95, 99]
        print("\n=== Sports 用户活跃度分位数分析 (训练集) ===")
        for p in sports_percentiles:
            val = np.percentile(sports_user_interactions_trn, p)
            print(f"{p}% 分位数: {val:.2f}")

        # 活跃度区间分布
        sports_bins = [0, 5, 10, 20, 50, 100, float('inf')]
        sports_labels = ['极低活跃(0-5)', '低活跃(5-10)', '中活跃(10-20)',
                         '高活跃(20-50)', '极高活跃(50-100)', '超高活跃(100+)']
        print("\n=== Sports 用户活跃度分布 ===")
        for i in range(len(sports_bins)-1):
            count = np.sum((sports_user_interactions_trn > sports_bins[i]) & (sports_user_interactions_trn <= sports_bins[i+1]))
            percentage = count / len(sports_user_interactions_trn) * 100
            print(f"{sports_labels[i]}: {count} 用户 ({percentage:.2f}%)")

        # 极端用户分析
        sports_inactive_threshold = 3
        sports_inactive_users = np.where(sports_user_interactions_trn <= sports_inactive_threshold)[0]
        print("\n=== Sports 极端用户分析 ===")
        print(f"训练集交互数 <= {sports_inactive_threshold} 的用户数: {len(sports_inactive_users)} ({len(sports_inactive_users)/len(sports_user_interactions_trn)*100:.2f}%)")

        sports_active_threshold = sports_user_interactions_trn.mean() + sports_user_interactions_trn.std()
        sports_active_users = np.where(sports_user_interactions_trn >= sports_active_threshold)[0]
        print(f"训练集交互数 >= {sports_active_threshold:.2f} 的活跃用户数: {len(sports_active_users)} ({len(sports_active_users)/len(sports_user_interactions_trn)*100:.2f}%)")

        # Top/Bottom 用户
        sports_top_k = 10
        sports_top_users_idx = np.argsort(sports_user_interactions_trn)[-sports_top_k:][::-1]
        print(f"\n=== Sports Top {sports_top_k} 最活跃用户 (训练集) ===")
        for rank, user_idx in enumerate(sports_top_users_idx, 1):
            trn_count = sports_user_interactions_trn[user_idx]
            total_count = sports_total_interactions[user_idx]
            print(f"#{rank} 用户ID {user_idx}: 训练集交互 {trn_count:.0f}, 总交互 {total_count:.0f}")

        sports_bottom_users_idx = np.argsort(sports_user_interactions_trn)[:sports_top_k]
        print(f"\n=== Sports Bottom {sports_top_k} 最不活跃用户 (训练集) ===")
        for rank, user_idx in enumerate(sports_bottom_users_idx, 1):
            trn_count = sports_user_interactions_trn[user_idx]
            total_count = sports_total_interactions[user_idx]
        return print(f"#{rank} 用户ID {user_idx}: 训练集交互 {trn_count:.0f}, 总交互 {total_count:.0f}")


    _()
    return


@app.cell
def _(coo_matrix, np, pickle, plt, sp):
    def _():
        # 可视化：Sports 用户交互数量分布
        import matplotlib
        matplotlib.use('Agg')

        # 在函数内加载 Sports 训练集矩阵并计算用户交互数量
        _sports_predir = './Datasets/sports/'
        _sports_trnfile = _sports_predir + 'trnMat.pkl'

        with open(_sports_trnfile, 'rb') as _fs:
            _ret = (pickle.load(_fs) != 0).astype(np.float32)
        if type(_ret) != coo_matrix:
            _ret = sp.coo_matrix(_ret)
        _sports_trnMat = _ret
        _sports_user_interactions_trn = np.array(_sports_trnMat.sum(axis=1)).flatten()

        sports_fig, sports_axes = plt.subplots(2, 2, figsize=(14, 10))

        # 直方图 - 完整分布
        sports_axes[0, 0].hist(_sports_user_interactions_trn, bins=50, edgecolor='black', alpha=0.7)
        sports_axes[0, 0].set_xlabel('交互物品数量')
        sports_axes[0, 0].set_ylabel('用户数量')
        sports_axes[0, 0].set_title('Sports 用户交互数量分布 (完整)')
        sports_axes[0, 0].grid(True, alpha=0.3)

        # 直方图 - 截断版本（95%分位数）
        sports_threshold_95 = np.percentile(_sports_user_interactions_trn, 95)
        sports_filtered_interactions = _sports_user_interactions_trn[_sports_user_interactions_trn <= sports_threshold_95]
        sports_axes[0, 1].hist(sports_filtered_interactions, bins=50, edgecolor='black', alpha=0.7, color='orange')
        sports_axes[0, 1].set_xlabel('交互物品数量')
        sports_axes[0, 1].set_ylabel('用户数量')
        sports_axes[0, 1].set_title(f'Sports 用户交互数量分布 (截断至95%分位数: {sports_threshold_95:.0f})')
        sports_axes[0, 1].grid(True, alpha=0.3)

        # 箱线图
        sports_axes[1, 0].boxplot(_sports_user_interactions_trn, vert=True)
        sports_axes[1, 0].set_ylabel('交互物品数量')
        sports_axes[1, 0].set_title('Sports 用户交互数量箱线图')
        sports_axes[1, 0].grid(True, alpha=0.3)

        # 累积分布函数 (CDF)
        sports_sorted_interactions = np.sort(_sports_user_interactions_trn)
        sports_cumulative = np.arange(1, len(sports_sorted_interactions) + 1) / len(sports_sorted_interactions)
        sports_axes[1, 1].plot(sports_sorted_interactions, sports_cumulative, linewidth=2)
        sports_axes[1, 1].set_xlabel('交互物品数量')
        sports_axes[1, 1].set_ylabel('累积概率')
        sports_axes[1, 1].set_title('Sports 用户交互数量累积分布函数 (CDF)')
        sports_axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('sports_user_interactions_analysis.png', dpi=150, bbox_inches='tight')
        print("图表已保存至: sports_user_interactions_analysis.png")
        return plt.gca()


    _()
    return


@app.cell
def _(coo_matrix, np, pickle, sp):
    # 物品交互与稀疏度分析（Sports）
    _sports_predir = './Datasets/sports/'
    _sports_trnfile = _sports_predir + 'trnMat.pkl'

    with open(_sports_trnfile, 'rb') as _fs:
        _ret = (pickle.load(_fs) != 0).astype(np.float32)
    if type(_ret) != coo_matrix:
        _ret = sp.coo_matrix(_ret)
    _sports_trnMat = _ret

    _sports_item_interactions = np.array(_sports_trnMat.sum(axis=0)).flatten()

    print("\n=== Sports 物品交互统计 (训练集) ===")
    print(f"平均被交互次数: {_sports_item_interactions.mean():.2f}")
    print(f"中位数被交互次数: {np.median(_sports_item_interactions):.2f}")
    print(f"最少被交互次数: {_sports_item_interactions.min()}")
    print(f"最多被交互次数: {_sports_item_interactions.max()}")
    print(f"标准差: {_sports_item_interactions.std():.2f}")

    _sports_cold_items = np.sum(_sports_item_interactions == 0)
    print(f"\n冷启动物品数（无交互）: {_sports_cold_items} ({_sports_cold_items/len(_sports_item_interactions)*100:.2f}%)")

    _sports_total_possible_interactions = _sports_trnMat.shape[0] * _sports_trnMat.shape[1]
    _sports_actual_interactions = _sports_trnMat.nnz
    _sports_sparsity = 1 - (_sports_actual_interactions / _sports_total_possible_interactions)

    print("\n=== Sports 矩阵稀疏度分析 ===")
    print(f"可能的交互总数: {_sports_total_possible_interactions:,}")
    print(f"实际交互数: {_sports_actual_interactions:,}")
    print(f"稀疏度: {_sports_sparsity*100:.4f}%")
    print(f"密度: {(1-_sports_sparsity)*100:.4f}%")
    return


@app.cell
def _(mo):
    mo.md(f"""
    # Sports 数据集交互分析概览

    本分析复用 TikTok 数据集的分析流程，针对 Sports 数据集给出：
    - 基础规模：训练/测试矩阵形状与交互数量
    - 用户维度统计：训练/测试/总体的均值、中位数、极值和标准差
    - 活跃度分析：分位数、分段分布、极端用户（低活跃与高活跃）
    - Top/Bottom 用户列表（训练集）
    - 可视化图表：直方图（完整与95%截断）、箱线图、CDF
    - 物品维度统计：平均被交互次数、冷启动物品占比
    - 稀疏度分析：可能交互 vs 实际交互，稀疏度/密度

    已将图表保存: sports_user_interactions_analysis.png
    """)
    return


if __name__ == "__main__":
    app.run()
