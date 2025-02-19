import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import scipy.stats as stats
import matplotlib.cm as cm

def evaluator(smiles, labels, preds, save_path):
    """
    数据分析函数，绘制散点图和线性回归直线，以及分类分析热力图
    :param smiles: list of SMILES (化学反应的 SMILES 表示)
    :param labels: list of true labels (真实标签)
    :param preds: list of predictions (模型预测值)
    :param save_path: 保存图像的路径
    """
    # 数据预处理：提取反应物类型和试剂类型
    reactants_types = []
    reagents_types = []

    reactant_classes = set()
    reagent_classes = set()

    for smile in smiles:
        reactants, reagents, products = smile.split(">>")
        reactant_types = tuple(sorted(reactants.split()))
        reagent_types = tuple(sorted(reagents.split()))
        reactants_types.append(reactant_types)
        reagents_types.append(reagent_types)

        reactant_classes.update(reactant_types)
        reagent_classes.update(reagent_types)

    # 创建颜色映射
    reactant_color_map = {reactant: color for reactant, color in zip(reactant_classes, cm.tab20.colors)}
    reagent_color_map = {reagent: color for reagent, color in zip(reagent_classes, cm.tab20.colors)}

    # 绘制图一：回归分析散点图 + 线性回归直线
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Label")
    ax.set_ylabel("Prediction")
    ax.set_title("Regression Analysis")

    # 计算 Pearson-R, 斯皮尔曼相关系数, 拟合优度 R2
    pearson_r = np.corrcoef(labels, preds)[0, 1]
    spearman_r, _ = stats.spearmanr(labels, preds)
    r2 = r2_score(labels, preds)

    # 绘制散点图，并根据反应物种类和试剂种类设置颜色和形状
    for i in range(len(labels)):
        reactant_color = reactant_color_map.get(reactants_types[i], '#000000')  # 默认黑色
        reagent_color = reagent_color_map.get(reagents_types[i], '#000000')  # 默认黑色
        ax.scatter(labels[i], preds[i], color=reactant_color, marker='o', alpha=0.5)
        ax.scatter(labels[i], preds[i], color=reagent_color, marker='^', alpha=0.5)

    # 计算回归线的系数
    m, b = np.polyfit(labels, preds, 1)
    ax.plot([0, 1], [m * 0 + b, m * 1 + b], color="black", lw=2)

    # 显示回归方程、相关系数和拟合优度
    regression_text = f"y = {m:.2f}x + {b:.2f}\nPearson-R = {pearson_r:.2f}\nSpearman-R = {spearman_r:.2f}\nR2 = {r2:.2f}"
    ax.text(0.05, 0.9, regression_text, transform=ax.transAxes, fontsize=12, verticalalignment='top')

    # 保存图像
    plt.savefig(f"{save_path}/regression_analysis.png")
    plt.close()

    # 数据分析图二：高低反应产率的分类热力图
    thresholds = np.arange(0.1, 1.0, 0.1)
    heatmaps = []

    for threshold in thresholds:
        # 按照阈值区分高产率和低产率反应
        high_low = ['high' if label > threshold else 'low' for label in labels]
        predictions_classified = ['high' if pred > threshold else 'low' for pred in preds]

        # 计算混淆矩阵：假阳性、真阳性、假阴性、真阴性
        tp = sum(1 for i in range(len(high_low)) if high_low[i] == 'high' and predictions_classified[i] == 'high')
        tn = sum(1 for i in range(len(high_low)) if high_low[i] == 'low' and predictions_classified[i] == 'low')
        fp = sum(1 for i in range(len(high_low)) if high_low[i] == 'low' and predictions_classified[i] == 'high')
        fn = sum(1 for i in range(len(high_low)) if high_low[i] == 'high' and predictions_classified[i] == 'low')

        # 计算假阳性率、真阳性率、假阴性率和真阴性率
        total = len(labels)
        confusion_matrix = np.array([[tp/total, fp/total],
                                     [fn/total, tn/total]])

        heatmaps.append(confusion_matrix)

    # 绘制分类分析热力图
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        sns.heatmap(heatmaps[i], annot=True, fmt=".2f", cmap='Blues', ax=ax, cbar=True,
                    xticklabels=["Predicted High", "Predicted Low"], yticklabels=["True High", "True Low"])
        ax.set_title(f"Threshold = {thresholds[i]:.1f}")

    fig.suptitle("Classification Analysis", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    # 保存热力图图像
    plt.savefig(f"{save_path}/classification_analysis.png")
    plt.close()
