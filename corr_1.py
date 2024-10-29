import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

font_path = "微软雅黑.ttf"
font = FontProperties(fname=font_path)

inventory_data = pd.read_csv("附件1.csv", encoding="gbk")
sales_data = pd.read_csv("附件2.csv", encoding="gbk")
category_info = pd.read_csv("附件5.csv", encoding="gbk")

inventory_merged = inventory_data.merge(category_info, on="品类", how="left")
sales_merged = sales_data.merge(category_info, on="品类", how="left")

plt.rcParams["axes.unicode_minus"] = False
sns.set(style="whitegrid")


def plot_correlation_heatmap(data, title):
    data = data.dropna(axis=0, how="all").dropna(axis=1, how="all")

    data.index = data.index.astype(str).str.strip()
    data.columns = data.columns.astype(str).str.strip()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        data,
        annot=True,
        cmap="RdBu_r",
        fmt=".2f",
        linewidths=0.5,
        center=0,
        vmin=-1,
        vmax=1,
    )
    plt.title(title, fontsize=14, fontproperties=font)

    plt.xlabel("品类", fontsize=12, fontproperties=font)
    plt.ylabel("品类", fontsize=12, fontproperties=font)
    plt.xticks(rotation=45, ha="right", fontproperties=font)
    plt.yticks(rotation=0, fontproperties=font)

    plt.tight_layout()
    plt.show()


# 计算并绘制每个高级品类下的库存量相关性热力图
for high_category, group in inventory_merged.groupby("高级品类"):
    inventory_pivot = group.pivot_table(index="月份", columns="品类", values="库存量")
    corr_matrix_inventory = inventory_pivot.corr()
    plot_correlation_heatmap(corr_matrix_inventory, f"{high_category} - 月库存量相关性")

# 计算并绘制每个高级品类下的销量相关性热力图
for high_category, group in sales_merged.groupby("高级品类"):
    sales_pivot = group.pivot_table(index="日期", columns="品类", values="销量")
    corr_matrix_sales = sales_pivot.corr()
    plot_correlation_heatmap(corr_matrix_sales, f"{high_category} - 销量相关性")
