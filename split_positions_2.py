import pandas as pd
import numpy as np
from platypus import NSGAII, Problem, Integer, Solution

# 加载数据
inventory_data = pd.read_csv("月库存量预测结果.csv", encoding="gbk")
sales_data = pd.read_csv("预测结果_非负处理.csv", encoding="gbk")
warehouse_data = pd.read_csv("附件3.csv", encoding="gbk")
correlation_data = pd.read_csv("附件4.csv", encoding="gbk")
category_info = pd.read_csv("附件5.csv", encoding="gbk")

# 数据预处理
inventory_max = (
    inventory_data.groupby("品类")["库存量"].max().reset_index(name="最大库存量")
)
sales_max = (
    sales_data.melt(id_vars=["品类"], var_name="日期", value_name="销量")
    .groupby("品类")["销量"]
    .max()
    .reset_index(name="最大销量")
)
category_capacity = pd.merge(inventory_max, sales_max, on="品类")

categories = category_capacity["品类"].tolist()
warehouses = warehouse_data["仓库"].tolist()
cost_dict = warehouse_data.set_index("仓库")["仓租日成本"].to_dict()
warehouse_capacity = warehouse_data.set_index("仓库")["仓容上限"].to_dict()
warehouse_production = warehouse_data.set_index("仓库")["产能上限"].to_dict()
correlation_matrix = correlation_data.set_index(["品类1", "品类2"])["关联度"].to_dict()

# 定义问题维度
num_categories = len(categories)
num_warehouses = len(warehouses)


# 定义目标函数
def objective_function(vars):
    # vars 包含每个品类的分配仓库
    allocation = vars[:num_categories]
    # 计算各优化目标
    # 总仓租成本
    total_rent_cost = sum(
        cost_dict[warehouses[allocation[i]]] for i in range(num_categories)
    )

    # 最小仓容利用率（取最低值以确保平衡）
    inventory_utilization = min(
        sum(
            category_capacity.set_index("品类").loc[categories[i], "最大库存量"]
            for i in range(num_categories)
            if allocation[i] == j
        )
        / warehouse_capacity[warehouses[j]]
        for j in range(num_warehouses)
    )

    # 最小产能利用率（同上）
    production_utilization = min(
        sum(
            category_capacity.set_index("品类").loc[categories[i], "最大销量"]
            for i in range(num_categories)
            if allocation[i] == j
        )
        / warehouse_production[warehouses[j]]
        for j in range(num_warehouses)
    )

    # 总关联度
    total_correlation = sum(
        correlation_matrix.get((categories[i], categories[k]), 0)
        for i in range(num_categories)
        for k in range(num_categories)
        if allocation[i] == allocation[k]
    )

    # 返回各目标值：分别为总仓租成本、负的最小仓容利用率、负的最小产能利用率、负的总关联度
    return [
        total_rent_cost,
        -inventory_utilization,
        -production_utilization,
        -total_correlation,
    ]


# 定义优化问题
problem = Problem(num_categories, 4)  # 4个目标
problem.types[:] = [
    Integer(0, num_warehouses - 1) for _ in range(num_categories)
]  # 每个品类只能分配一个仓库
problem.directions[:] = [
    Problem.MINIMIZE,
    Problem.MAXIMIZE,
    Problem.MAXIMIZE,
    Problem.MAXIMIZE,
]
problem.function = objective_function

# 使用NSGA-II求解
algorithm = NSGAII(problem)
algorithm.run(10000)  # 设置迭代次数

# 提取最优解并展示
best_solutions = []
for solution in algorithm.result:
    allocation = solution.variables[:num_categories]
    objectives = solution.objectives
    best_solutions.append((allocation, objectives))

# 打印最优解中的各优化目标值和分配方案
for i, (allocation, objectives) in enumerate(best_solutions):
    print(f"解 {i + 1}:")
    print("仓租成本:", objectives[0])
    print("最小仓容利用率:", -objectives[1])
    print("最小产能利用率:", -objectives[2])
    print("总品类关联度:", -objectives[3])
    print("分配方案:", allocation)
    print("-" * 50)

# 保存最优解为CSV
solution_df = pd.DataFrame(
    [
        {"品类": categories[i], "仓库": warehouses[allocation[i]]}
        for allocation, _ in best_solutions
        for i in range(num_categories)
    ]
)
solution_df.to_csv("多目标优化分配方案.csv", index=False, encoding="gbk")

print("最优分配方案已保存到 '多目标优化分配方案.csv'")
