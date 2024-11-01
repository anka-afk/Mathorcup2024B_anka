import pandas as pd

# 读取文件
file_path = "预测结果.csv"
data = pd.read_csv(file_path, encoding="gbk", header=None)

# 创建列名
columns = ["品类"]
# 添加日期列名：7月1日到9月30日
for month, days in [(7, 31), (8, 31), (9, 30)]:
    for day in range(1, days + 1):
        columns.append(f"{month}月{day}日")

# 设置列名
data.columns = columns

# 将所有负数值替换为0，并对数值取整
for column in data.columns[1:]:  # 跳过'品类'列
    data[column] = data[column].apply(lambda x: round(max(x, 0)))

# 保存处理后的数据
output_path = "预测结果_非负处理.csv"  # 输出文件名
data.to_csv(output_path, encoding="gbk", index=False)
print(f"处理完成，结果已保存到 {output_path}")
