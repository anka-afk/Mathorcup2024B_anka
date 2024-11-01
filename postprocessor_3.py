import pandas as pd

# 读取CSV文件
df = pd.read_csv("分配结果3.csv")

# 创建新的数据结构
result_data = {"category": [], "warehouse1": [], "warehouse2": [], "warehouse3": []}

# 遍历每一行
for index, row in df.iterrows():
    # 获取品类名
    category = row["品类"]

    # 找出该行中值为33.3%的仓库
    warehouses = []
    for col in df.columns:
        if col.startswith("仓库_") and row[col] == "33.3%":
            # 从列名中提取仓库编号
            warehouse_num = col.split("_")[1]
            warehouses.append(warehouse_num)

    # 确保有3个仓库（如果不足3个，用空字符串填充）
    warehouses.extend([""] * (3 - len(warehouses)))

    # 添加到结果数据中
    result_data["category"].append(category)
    result_data["warehouse1"].append(warehouses[0])
    result_data["warehouse2"].append(warehouses[1])
    result_data["warehouse3"].append(warehouses[2])

# 创建新的DataFrame
result_df = pd.DataFrame(result_data)

# 保存到新的CSV文件
result_df.to_csv("warehouse_categories.csv", index=False)
