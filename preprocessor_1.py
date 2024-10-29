import pandas as pd

# 读取销售数据与高级品类信息
sales_data = pd.read_csv("附件2.csv", encoding="gbk")
category_info = pd.read_csv("附件5.csv", encoding="gbk")

# 转换日期格式
sales_data["日期"] = pd.to_datetime(sales_data["日期"], format="%Y/%m/%d")

# 合并数据
sales_data_with_info = pd.merge(sales_data, category_info, how="left", on="品类")

# 过滤出22年7月到9月和23年4月到6月的数据
filtered_data = sales_data_with_info[
    (sales_data_with_info["日期"].dt.year == 2022)
    & (sales_data_with_info["日期"].dt.month >= 7)
    & (sales_data_with_info["日期"].dt.month <= 9)
    | (sales_data_with_info["日期"].dt.year == 2023)
    & (sales_data_with_info["日期"].dt.month >= 4)
    & (sales_data_with_info["日期"].dt.month <= 6)
]

# 创建数据透视表以便计算缺失情况和填补
sales_pivoted = filtered_data.pivot_table(index="日期", columns="品类", values="销量")

# 计算缺失值情况
missing_counts = sales_pivoted.isnull().sum()

# 时间序列插值（缺失数据量较少）
low_missing_categories = missing_counts[missing_counts <= 18].index  # 10% 阈值
sales_pivoted[low_missing_categories] = sales_pivoted[
    low_missing_categories
].interpolate(method="linear", limit_direction="both")

# 基于相关品类的加权填补（缺失数据量中等）
mid_missing_categories = missing_counts[
    (missing_counts > 18) & (missing_counts <= 90)
].index  # 10%-50% 阈值

# 计算相关性矩阵
correlation_matrix = sales_pivoted.corr()

# 加权填补
for category in mid_missing_categories:
    top_correlated = (
        correlation_matrix[category].drop(category).sort_values(ascending=False).head(5)
    )
    weights = top_correlated / top_correlated.sum()
    missing_indices = sales_pivoted[category].isnull()
    sales_pivoted.loc[missing_indices, category] = (
        sales_pivoted[top_correlated.index] * weights
    ).sum(axis=1)

# 移动均值填补（缺失数据量较多）
high_missing_categories = missing_counts[missing_counts > 90].index  # 超过50%阈值
sales_pivoted[high_missing_categories] = sales_pivoted[high_missing_categories].fillna(
    sales_pivoted[high_missing_categories].rolling(window=7, min_periods=1).mean()
)

# 输出结果文件
sales_pivoted.to_csv("填补完缺失值的销量数据.csv", encoding="gbk")
