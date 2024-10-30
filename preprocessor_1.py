import pandas as pd
from sklearn.preprocessing import StandardScaler

# 读取销售数据与高级品类信息
sales_data = pd.read_csv("附件2.csv", encoding="gbk")
category_info = pd.read_csv("附件5.csv", encoding="gbk")

# 转换日期格式
sales_data["日期"] = pd.to_datetime(sales_data["日期"], format="%Y/%m/%d")

# 合并数据
sales_data_with_info = pd.merge(sales_data, category_info, how="left", on="品类")

# 过滤出22年7月到9月和23年4月到6月的数据
filtered_data = sales_data_with_info[
    (
        (sales_data_with_info["日期"].dt.year == 2022)
        & (sales_data_with_info["日期"].dt.month >= 7)
        & (sales_data_with_info["日期"].dt.month <= 9)
    )
    | (
        (sales_data_with_info["日期"].dt.year == 2023)
        & (sales_data_with_info["日期"].dt.month >= 4)
        & (sales_data_with_info["日期"].dt.month <= 6)
    )
]

# 创建数据透视表以便计算缺失情况和填补
sales_pivoted = filtered_data.pivot_table(index="日期", columns="品类", values="销量")

# 标准化数据
scaler = StandardScaler()
sales_standardized = pd.DataFrame(
    scaler.fit_transform(sales_pivoted),
    index=sales_pivoted.index,
    columns=sales_pivoted.columns,
)

# 计算缺失值情况
missing_counts = sales_standardized.isnull().sum()

# 时间序列插值（缺失数据量较少）
low_missing_categories = missing_counts[missing_counts <= 18].index  # 10% 阈值
sales_standardized[low_missing_categories] = sales_standardized[
    low_missing_categories
].interpolate(method="linear", limit_direction="both")

# 基于相关品类的加权填补（缺失数据量中等）
mid_missing_categories = missing_counts[
    (missing_counts > 18) & (missing_counts <= 90)
].index  # 10%-50% 阈值

# 计算相关性矩阵
correlation_matrix = sales_standardized.corr()

# 加权填补
for category in mid_missing_categories:
    top_correlated = (
        correlation_matrix[category].drop(category).sort_values(ascending=False).head(5)
    )
    weights = top_correlated / top_correlated.sum()
    missing_indices = sales_standardized[category].isnull()
    sales_standardized.loc[missing_indices, category] = (
        sales_standardized[top_correlated.index] * weights
    ).sum(axis=1)

# 移动均值填补（缺失数据量较多）
high_missing_categories = missing_counts[missing_counts > 90].index  # 超过50%阈值
sales_standardized[high_missing_categories] = sales_standardized[
    high_missing_categories
].fillna(
    sales_standardized[high_missing_categories].rolling(window=7, min_periods=1).mean()
)

# 进一步填补残余缺失值
# 再次进行线性插值尝试填补
sales_standardized = sales_standardized.interpolate(
    method="linear", limit_direction="both"
)

# 对于仍然未填补的缺失值，使用高级品类的均值填补
for category in sales_standardized.columns[sales_standardized.isnull().any()]:
    high_category = category_info.loc[
        category_info["品类"] == category, "高级品类"
    ].values[0]
    related_categories = category_info[category_info["高级品类"] == high_category][
        "品类"
    ]
    related_data = sales_standardized[related_categories].mean(axis=1)
    sales_standardized[category].fillna(related_data, inplace=True)

# 反标准化处理
sales_filled = pd.DataFrame(
    scaler.inverse_transform(sales_standardized),
    index=sales_standardized.index,
    columns=sales_standardized.columns,
)

# 对填补后的数据进行非负处理，确保没有负值
sales_filled[sales_filled < 0] = 0

# 对销量数据取整
sales_filled = sales_filled.round().astype(int)

# 将数据从宽格式转换回长格式，准备保存
sales_long_format = sales_filled.reset_index().melt(
    id_vars=["日期"], var_name="品类", value_name="销量"
)

# 输出结果文件
sales_long_format.to_csv("填补完缺失值的销量数据.csv", encoding="gbk", index=False)

# 检查是否还有缺失值
remaining_missing = sales_long_format["销量"].isnull().sum()

if remaining_missing == 0:
    print("数据填补完成，没有剩余缺失值。")
else:
    print(f"数据填补未完全，仍有 {remaining_missing} 个缺失值。")
