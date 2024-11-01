import pandas as pd
import numpy as np
from scipy.optimize import NonlinearConstraint
import random
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


# 首先定义所有辅助函数
def analyze_allocation(allocation):
    # 初始化仓库统计信息
    warehouse_stats = {
        w: {"已用库存": 0, "已用产能": 0, "品类列表": []} for w in warehouses
    }

    # 分析每个品类的分配情况
    for category, warehouse_allocations in allocation.items():
        inventory = category_capacity[category_capacity["品类"] == category][
            "最大库存量"
        ].values[0]
        sales = category_capacity[category_capacity["品类"] == category][
            "最大销量"
        ].values[0]

        for warehouse, fraction in warehouse_allocations.items():
            # 累计仓库使用情况
            warehouse_stats[warehouse]["已用库存"] += inventory * fraction
            warehouse_stats[warehouse]["已用产能"] += sales * fraction
            warehouse_stats[warehouse]["品类列表"].append(
                f"{category}({fraction*100:.1f}%)"
            )

    # 打印分析结果
    print("\n仓库使用情况分析:")
    for warehouse, stats in warehouse_stats.items():
        if len(stats["品类列表"]) > 0:  # 只显示有分配的仓库
            print(f"\n仓库 {warehouse}:")
            print(
                f"  库存使用: {stats['已用库存']:.2f}/{warehouse_capacity[warehouse]:.2f} "
                f"(利用率: {stats['已用库存']/warehouse_capacity[warehouse]*100:.2f}%)"
            )
            print(
                f"  产能使用: {stats['已用产能']:.2f}/{warehouse_production[warehouse]:.2f} "
                f"(利用率: {stats['已用产能']/warehouse_production[warehouse]*100:.2f}%)"
            )
            print(f"  分配品类: {', '.join(stats['品类列表'])}")

    return warehouse_stats


def plot_warehouse_utilization(warehouse_stats):
    # 筛选出已使用的仓库
    active_warehouses = {
        w: stats for w, stats in warehouse_stats.items() if len(stats["品类列表"]) > 0
    }

    # 准备数据
    warehouses = list(active_warehouses.keys())
    capacity_usage = [
        stats["已用库存"] / warehouse_capacity[w] * 100
        for w, stats in active_warehouses.items()
    ]
    production_usage = [
        stats["已用产能"] / warehouse_production[w] * 100
        for w, stats in active_warehouses.items()
    ]

    # 创建图形
    plt.figure(figsize=(12, 6))
    x = np.arange(len(warehouses))
    width = 0.35

    # 绘制柱状图
    plt.bar(x - width / 2, capacity_usage, width, label="仓容利用率")
    plt.bar(x + width / 2, production_usage, width, label="产能利用率")

    plt.xlabel("仓库编号")
    plt.ylabel("利用率 (%)")
    plt.title("各仓库利用率分布")
    plt.xticks(x, warehouses, rotation=45)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def sensitivity_analysis(base_solution):
    global cost_dict, warehouse_production
    original_cost_dict = cost_dict.copy()
    original_production = warehouse_production.copy()

    # 成本敏感性分析
    cost_variations = np.linspace(0.5, 1.5, 10)  # 成本变化范围：50%~150%
    cost_objectives = []

    for factor in cost_variations:
        temp_cost_dict = {k: v * factor for k, v in original_cost_dict.items()}
        cost_dict = temp_cost_dict
        obj_value = objective_function(base_solution)
        cost_objectives.append(obj_value)

    # 恢复原始cost_dict
    cost_dict = original_cost_dict

    # 产能敏感性分析
    capacity_variations = np.linspace(0.5, 1.5, 10)  # 产能变化范围：50%~150%
    capacity_objectives = []

    for factor in capacity_variations:
        temp_production = {k: v * factor for k, v in original_production.items()}
        warehouse_production = temp_production
        obj_value = objective_function(base_solution)
        capacity_objectives.append(obj_value)

    # 恢复原始warehouse_production
    warehouse_production = original_production

    # 绘制敏感性分析图
    plt.figure(figsize=(12, 5))

    # 成本敏感性
    plt.subplot(1, 2, 1)
    plt.plot(cost_variations * 100, cost_objectives, "b-o")
    plt.xlabel("成本变化百分比 (%)")
    plt.ylabel("目标函数值")
    plt.title("仓租成本敏感性分析")
    plt.grid(True)

    # 产能敏感性
    plt.subplot(1, 2, 2)
    plt.plot(capacity_variations * 100, capacity_objectives, "r-o")
    plt.xlabel("产能变化百分比 (%)")
    plt.ylabel("目标函数值")
    plt.title("仓库产能敏感性分析")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# 读取数据
inventory_data = pd.read_csv("月库存量预测结果.csv", encoding="gbk")
sales_data = pd.read_csv("预测结果_非负处理.csv", encoding="gbk")
warehouse_data = pd.read_csv("附件3.csv", encoding="gbk")
correlation_data = pd.read_csv("附件4.csv", encoding="gbk")
category_info = pd.read_csv("附件5.csv", encoding="gbk")

# 确保category_info使用品类作为索引
category_info = category_info.set_index("品类")

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

# 获取品类和仓库列表
categories = category_capacity["品类"].tolist()[:350]
warehouses = warehouse_data["仓库"].tolist()[:140]

# 仓库相关信息
cost_dict = warehouse_data.set_index("仓库")["仓租日成本"].to_dict()
warehouse_capacity = warehouse_data.set_index("仓库")["仓容上限"].to_dict()
warehouse_production = warehouse_data.set_index("仓库")["产能上限"].to_dict()
correlation_matrix = correlation_data.set_index(["品类1", "品类2"])["关联度"].to_dict()


# 目标函数定义
def objective_function(allocation):
    warehouse_load = {w: {"库存": 0, "销量": 0} for w in warehouses}
    total_cost = 0
    total_correlation = 0
    total_similarity = 0

    # 计算每个仓库的负载
    for category, warehouse_allocations in allocation.items():
        inventory = category_capacity[category_capacity["品类"] == category][
            "最大库存量"
        ].values[0]
        sales = category_capacity[category_capacity["品类"] == category][
            "最大销量"
        ].values[0]

        # 遍历该品类分配的所有仓库
        for warehouse, fraction in warehouse_allocations.items():
            # 计算负载
            warehouse_load[warehouse]["库存"] += inventory * fraction
            warehouse_load[warehouse]["销量"] += sales * fraction
            total_cost += cost_dict[warehouse] * fraction

            # 计算品类关联度和相似度
            for other_category, other_allocations in allocation.items():
                if category >= other_category:
                    continue

                # 如果两个品类共用同一个仓库
                if warehouse in other_allocations:
                    # 计算关联度
                    correlation = correlation_matrix.get((category, other_category), 0)
                    total_correlation += (
                        correlation * fraction * other_allocations[warehouse]
                    )

                    # 检查高级品类相似度
                    try:
                        if (
                            category_info.at[category, "高级品类"]
                            == category_info.at[other_category, "高级品类"]
                        ):
                            total_similarity += fraction * other_allocations[warehouse]
                    except KeyError:
                        # 如果找不到品类信息，跳过相似度计算
                        continue

    # 计算仓容和产能利用率
    min_capacity_usage = min(
        warehouse_load[w]["库存"] / warehouse_capacity[w]
        for w in warehouses
        if warehouse_capacity[w] > 0
    )
    min_production_usage = min(
        warehouse_load[w]["销量"] / warehouse_production[w]
        for w in warehouses
        if warehouse_production[w] > 0
    )

    # 标准化各项指标
    max_possible_cost = sum(max(cost_dict.values()) for _ in range(len(categories)))
    normalized_cost = total_cost / max_possible_cost

    max_possible_correlation = sum(
        max(correlation_matrix.values())
        for _ in range(len(categories) * (len(categories) - 1) // 2)
    )
    normalized_correlation = (
        total_correlation / max_possible_correlation
        if max_possible_correlation > 0
        else 0
    )

    # 直接使用百分比作为标准化值
    normalized_capacity = min_capacity_usage
    normalized_production = min_production_usage

    max_possible_similarity = len(categories) * (len(categories) - 1) / 2
    normalized_similarity = (
        total_similarity / max_possible_similarity if max_possible_similarity > 0 else 0
    )

    # 设定权重
    w1, w2, w3, w4, w5 = 0.2, 0.2, 0.2, 1, 0.2

    # 计算目标值
    objective_value = (
        w1 * normalized_cost
        - w2 * normalized_capacity
        - w3 * normalized_production
        - w4 * normalized_correlation
        - w5 * normalized_similarity
    )

    return objective_value


# 设置约束条件
def combined_constraints(x):
    violations = []
    warehouse_loads = {w: {"库存": 0, "销量": 0} for w in warehouses}

    for i, category in enumerate(categories):
        # 获取分配的仓库（排除-1值）
        warehouses_assigned = [warehouses[int(w)] for w in x[i] if w >= 0]
        if not warehouses_assigned:
            continue

        # 按比例分配需求
        inventory = category_capacity.loc[
            category_capacity["品类"] == category, "最大库存量"
        ].values[0]
        sales = category_capacity.loc[
            category_capacity["品类"] == category, "最大销量"
        ].values[0]

        per_warehouse_inventory = inventory / len(warehouses_assigned)
        per_warehouse_sales = sales / len(warehouses_assigned)

        for warehouse in warehouses_assigned:
            warehouse_loads[warehouse]["库存"] += per_warehouse_inventory
            warehouse_loads[warehouse]["销量"] += per_warehouse_sales

    # 检查约束条件
    for warehouse in warehouses:
        violations.append(
            warehouse_capacity[warehouse] - warehouse_loads[warehouse]["库存"]
        )
        violations.append(
            warehouse_production[warehouse] - warehouse_loads[warehouse]["销量"]
        )

    return np.array(violations)


# 定义贪心算法类
class GreedyAllocator:
    def __init__(
        self,
        categories,
        warehouses,
        category_capacity,
        warehouse_capacity,
        warehouse_production,
        cost_dict,
        correlation_matrix,
    ):
        self.categories = categories
        self.warehouses = warehouses
        self.category_capacity = category_capacity
        self.warehouse_capacity = warehouse_capacity
        self.warehouse_production = warehouse_production
        self.cost_dict = cost_dict
        self.correlation_matrix = correlation_matrix
        self.allocation = {category: {} for category in categories}

    def allocate(self):
        # 初始化仓库负载
        warehouse_loads = {w: {"库存": 0, "销量": 0} for w in self.warehouses}

        # 按库存量降序排序品类
        sorted_categories = self.category_capacity.sort_values(
            "最大库存量", ascending=False
        )

        for _, row in sorted_categories.iterrows():
            category = row["品类"]
            if category not in self.categories:
                continue

            inventory = row["最大库存量"]
            sales = row["最大销量"]

            assigned_warehouses = 0
            fraction = 1 / 3  # 每个仓库分配三分之一

            # 为每个品类分配最多3个仓库
            for warehouse in self.warehouses:
                if assigned_warehouses >= 3:
                    break

                # 检查容量约束
                if (
                    warehouse_loads[warehouse]["库存"] + inventory * fraction
                    <= self.warehouse_capacity[warehouse]
                    and warehouse_loads[warehouse]["销量"] + sales * fraction
                    <= self.warehouse_production[warehouse]
                ):

                    # 记录分配结果
                    self.allocation[category][warehouse] = fraction
                    warehouse_loads[warehouse]["库存"] += inventory * fraction
                    warehouse_loads[warehouse]["销量"] += sales * fraction
                    assigned_warehouses += 1

        return self.allocation


# PSO算法类
class PSOOptimizer:
    def __init__(
        self,
        n_particles,
        n_iterations,
        categories,
        warehouses,
        category_capacity,
        warehouse_capacity,
        warehouse_production,
        cost_dict,
        correlation_matrix,
    ):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.categories = categories
        self.warehouses = warehouses
        self.category_capacity = category_capacity
        self.warehouse_capacity = warehouse_capacity
        self.warehouse_production = warehouse_production
        self.cost_dict = cost_dict
        self.correlation_matrix = correlation_matrix

    def initialize_particles(self, greedy_solution):
        particles = []
        velocities = []

        # 第一个粒子使用贪心解
        particles.append(greedy_solution)

        # 其他粒子随机初始化
        for _ in range(1, self.n_particles):
            particle = {category: {} for category in self.categories}
            for category in self.categories:
                # 随机选择1-3个仓库
                n_warehouses = random.randint(1, 3)
                selected_warehouses = random.sample(self.warehouses, n_warehouses)
                fraction = 1.0 / n_warehouses
                for warehouse in selected_warehouses:
                    particle[category][warehouse] = fraction
            particles.append(particle)

        # 初化速度
        for _ in range(self.n_particles):
            velocity = {
                category: {
                    warehouse: random.uniform(-0.1, 0.1)
                    for warehouse in self.warehouses
                }
                for category in self.categories
            }
            velocities.append(velocity)

        return particles, velocities

    def optimize(self, greedy_solution):
        particles, velocities = self.initialize_particles(greedy_solution)

        # 初始化个体最优和全局最优
        p_best = particles.copy()
        p_best_fitness = [objective_function(p) for p in p_best]
        g_best = p_best[np.argmin(p_best_fitness)].copy()
        g_best_fitness = min(p_best_fitness)

        # PSO参数
        w = 0.9  # 惯性权重
        c1 = 2.0  # 个体学习因子
        c2 = 2.0  # 社会学习因子

        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                # 更新速度和位置
                for category in self.categories:
                    for warehouse in self.warehouses:
                        r1, r2 = random.random(), random.random()

                        # 更新速度
                        if warehouse in particles[i][category]:
                            current_pos = particles[i][category][warehouse]
                        else:
                            current_pos = 0

                        if warehouse in p_best[i][category]:
                            p_best_pos = p_best[i][category][warehouse]
                        else:
                            p_best_pos = 0

                        if warehouse in g_best[category]:
                            g_best_pos = g_best[category][warehouse]
                        else:
                            g_best_pos = 0

                        velocities[i][category][warehouse] = (
                            w * velocities[i][category][warehouse]
                            + c1 * r1 * (p_best_pos - current_pos)
                            + c2 * r2 * (g_best_pos - current_pos)
                        )

                        # 更新位置
                        new_pos = current_pos + velocities[i][category][warehouse]
                        if new_pos > 0:
                            particles[i][category][warehouse] = new_pos
                        else:
                            if warehouse in particles[i][category]:
                                del particles[i][category][warehouse]

                # 归一化分配比例
                for category in self.categories:
                    total = sum(particles[i][category].values())
                    if total > 0:
                        for warehouse in particles[i][category]:
                            particles[i][category][warehouse] /= total

                # 评估新位置
                fitness = objective_function(particles[i])
                if fitness < p_best_fitness[i]:
                    p_best[i] = particles[i].copy()
                    p_best_fitness[i] = fitness

                    if fitness < g_best_fitness:
                        g_best = particles[i].copy()
                        g_best_fitness = fitness

            # 态整惯性权重
            w = max(0.4, w * 0.99)

            if iteration % 10 == 0:
                print(f"迭代 {iteration}, 当前最优值: {g_best_fitness}")

        return g_best, g_best_fitness


# 主程序修改
# 1. 先用贪心算法获得初始解
allocator = GreedyAllocator(
    categories=categories,
    warehouses=warehouses,
    category_capacity=category_capacity,
    warehouse_capacity=warehouse_capacity,
    warehouse_production=warehouse_production,
    cost_dict=cost_dict,
    correlation_matrix=correlation_matrix,
)

greedy_solution = allocator.allocate()
greedy_fitness = objective_function(greedy_solution)
print("贪心算法目标值：", greedy_fitness)

# 2. 用PSO进一步优化
pso = PSOOptimizer(
    n_particles=50,
    n_iterations=100,
    categories=categories,
    warehouses=warehouses,
    category_capacity=category_capacity,
    warehouse_capacity=warehouse_capacity,
    warehouse_production=warehouse_production,
    cost_dict=cost_dict,
    correlation_matrix=correlation_matrix,
)

best_solution, best_fitness = pso.optimize(greedy_solution)
print("PSO优化后目标值：", best_fitness)

# 打印分配结果
print("\n最终分配结果：")
for category, warehouse_allocations in best_solution.items():
    print(f"品类 {category} 分配到以下仓库：")
    for warehouse, fraction in warehouse_allocations.items():
        print(f"  - 仓库 {warehouse}: {fraction*100:.1f}%")

# 创建结果数据框
result_data = []
for category, warehouse_allocations in best_solution.items():
    row = {"品类": category}
    for warehouse, fraction in warehouse_allocations.items():
        row[f"仓库_{warehouse}"] = f"{fraction*100:.1f}%"
    result_data.append(row)

result_df = pd.DataFrame(result_data)

# 保存为 CSV 文件
result_df.to_csv("分配结果3.csv", index=False, encoding="utf-8-sig")
print("\n结果已保存为分配结果3.csv")

# 调用分析函数并生成图表
print("\n详细分配分析：")
warehouse_stats = analyze_allocation(best_solution)

# 生成可视化图表
plot_warehouse_utilization(warehouse_stats)
sensitivity_analysis(best_solution)
