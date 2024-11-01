import pandas as pd
import numpy as np
from scipy.optimize import NonlinearConstraint
import random
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


# 读取数据
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

    # 计算每个仓库的库存量和销量
    for i, category in enumerate(categories):
        warehouse = warehouses[int(allocation[i])]
        demand_inventory = category_capacity[category_capacity["品类"] == category][
            "最大库存量"
        ].values[0]
        demand_sales = category_capacity[category_capacity["品类"] == category][
            "最大销量"
        ].values[0]

        warehouse_load[warehouse]["库存"] += demand_inventory
        warehouse_load[warehouse]["销量"] += demand_sales
        total_cost += cost_dict[warehouse]

        # 计算品类关联度
        for j in range(i + 1, len(categories)):
            related_category = categories[j]
            related_warehouse = warehouses[int(allocation[j])]
            if warehouse == related_warehouse:
                total_correlation += correlation_matrix.get(
                    (category, related_category), 0
                )

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

    # 标准化处理
    # 1. 成本标准化（除以最大可能成本）
    max_possible_cost = sum(max(cost_dict.values()) for _ in range(len(categories)))
    normalized_cost = total_cost / max_possible_cost  # 值域[0,1]

    # 2. 仓容利用率已经是百分比，值域[0,1]
    normalized_capacity = min_capacity_usage

    # 3. 产能利用率已经是百分比，值域[0,1]
    normalized_production = min_production_usage

    # 4. 关联度标准化（除以最大可能关联度）
    max_possible_correlation = sum(
        max(correlation_matrix.values())
        for _ in range(len(categories) * (len(categories) - 1) // 2)
    )
    normalized_correlation = (
        total_correlation / max_possible_correlation
        if max_possible_correlation > 0
        else 0
    )  # 值域[0,1]

    # 设置权重（现在所有指标都在[0,1]范围内）
    w1, w2, w3, w4 = 1, 0.25, 0.25, 0.25  # 可以根据实际需求调整权重

    # 计算目标值（所有项都在相同尺度下）
    objective_value = (
        w1 * normalized_cost  # 最小化成本
        - w2 * normalized_capacity  # 最大化最小仓容利用率
        - w3 * normalized_production  # 最大化最小产能利用率
        - w4 * normalized_correlation  # 最大化关联度
    )

    return objective_value


# 设置约束条件
def combined_constraints(x):
    violations = []
    warehouse_loads = {w: {"库存": 0, "销量": 0} for w in warehouses}

    # 计算每个仓库的总库存和总销量
    for i, category in enumerate(categories):
        warehouse = warehouses[int(x[i])]
        inventory = category_capacity.loc[
            category_capacity["品类"] == category, "最大库存量"
        ].values[0]
        sales = category_capacity.loc[
            category_capacity["品类"] == category, "最大销量"
        ].values[0]

        warehouse_loads[warehouse]["库存"] += inventory
        warehouse_loads[warehouse]["销量"] += sales

    # 添加仓容约束
    for warehouse in warehouses:
        violations.append(
            warehouse_capacity[warehouse] - warehouse_loads[warehouse]["库存"]
        )

    # 添加产能约束
    for warehouse in warehouses:
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

    def allocate(self):
        # 初始化仓库当前负载
        warehouse_loads = {w: {"库存": 0, "销量": 0} for w in self.warehouses}
        # 初始化分配结果
        allocation = {}

        # 按照库存量降序排序品类
        sorted_categories = self.category_capacity.sort_values(
            "最大库存量", ascending=False
        )

        for _, row in sorted_categories.iterrows():
            category = row["品类"]
            if category not in self.categories:
                continue

            inventory = row["最大库存量"]
            sales = row["最大销量"]

            best_warehouse = None
            best_score = float("inf")

            # 为当前品类找到最佳仓库
            for warehouse in self.warehouses:
                # 检查容量约束
                new_inventory = warehouse_loads[warehouse]["库存"] + inventory
                new_sales = warehouse_loads[warehouse]["销量"] + sales

                if (
                    new_inventory > self.warehouse_capacity[warehouse]
                    or new_sales > self.warehouse_production[warehouse]
                ):
                    continue

                # 计算评分（考虑成本、利用率和关联度）
                cost_score = self.cost_dict[warehouse]
                capacity_score = new_inventory / self.warehouse_capacity[warehouse]
                production_score = new_sales / self.warehouse_production[warehouse]

                # 计算与已分配品类的关联度
                correlation_score = 0
                for allocated_category, allocated_warehouse in allocation.items():
                    if allocated_warehouse == warehouse:
                        correlation_score += self.correlation_matrix.get(
                            (category, allocated_category), 0
                        )

                # 综合评分（可以调整权重）
                total_score = (
                    cost_score * 0.4
                    - capacity_score * 0.2
                    - production_score * 0.2
                    - correlation_score * 0.2
                )

                if total_score < best_score:
                    best_score = total_score
                    best_warehouse = warehouse

            if best_warehouse is not None:
                allocation[category] = best_warehouse
                warehouse_loads[best_warehouse]["库存"] += inventory
                warehouse_loads[best_warehouse]["销量"] += sales
            else:
                print(f"警告：无法为品类 {category} 找到合适的仓库")

        # 转换为与原代码相同的输出格式
        result = np.zeros(len(self.categories))
        for i, category in enumerate(self.categories):
            if category in allocation:
                result[i] = self.warehouses.index(allocation[category])

        return result


# 添加PSO算法类
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

        # 问题维度
        self.n_dimensions = len(categories)
        # 速度限制
        self.v_max = len(warehouses) - 1
        self.v_min = -self.v_max

    def initialize_particles(self, greedy_solution):
        # 使用贪心解作为第一个粒子
        particles = np.zeros((self.n_particles, self.n_dimensions))
        velocities = np.zeros((self.n_particles, self.n_dimensions))

        # 第一个粒子使用贪心解
        particles[0] = greedy_solution

        # 其他粒子���贪心解附近随机初始化
        for i in range(1, self.n_particles):
            # 80%概率保持贪心解，20%概率随机选择其他仓库
            mask = np.random.random(self.n_dimensions) > 0.8
            particles[i] = greedy_solution.copy()
            particles[i][mask] = np.random.randint(
                0, len(self.warehouses), size=sum(mask)
            )

        # 初始化速度
        velocities = np.random.uniform(
            self.v_min, self.v_max, (self.n_particles, self.n_dimensions)
        )

        return particles, velocities

    def optimize(self, greedy_solution):
        # 初始化子
        particles, velocities = self.initialize_particles(greedy_solution)

        # 初始化个体最优和全局最优
        p_best = particles.copy()
        p_best_fitness = np.array([objective_function(p) for p in p_best])
        g_best = p_best[np.argmin(p_best_fitness)].copy()
        g_best_fitness = np.min(p_best_fitness)

        # 迭代优化
        w = 0.9  # 惯性权重
        c1 = 2.0  # 个体学习因子
        c2 = 2.0  # 社会学习因子

        for iteration in range(self.n_iterations):
            # 更新速度和位置
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(2)

                # 更新速度
                velocities[i] = (
                    w * velocities[i]
                    + c1 * r1 * (p_best[i] - particles[i])
                    + c2 * r2 * (g_best - particles[i])
                )

                # 限制速度范围
                velocities[i] = np.clip(velocities[i], self.v_min, self.v_max)

                # 更新位置
                particles[i] = particles[i] + velocities[i]
                # 四舍五入到最近的整数并确保在有效范围内
                particles[i] = np.clip(
                    np.round(particles[i]), 0, len(self.warehouses) - 1
                )

                # 检查束条件
                if all(combined_constraints(particles[i]) >= 0):
                    # 计算新位置的适应度
                    fitness = objective_function(particles[i])

                    # 更新个体最优
                    if fitness < p_best_fitness[i]:
                        p_best[i] = particles[i].copy()
                        p_best_fitness[i] = fitness

                        # 更新全局最优
                        if fitness < g_best_fitness:
                            g_best = particles[i].copy()
                            g_best_fitness = fitness

            # 动态调整惯性权重
            w = max(0.4, w * 0.99)

            if iteration % 10 == 0:
                print(f"迭代 {iteration}, 当前最优值: {g_best_fitness}")

        return g_best, g_best_fitness


# 主程序
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
    n_particles=500,  # 粒子数量
    n_iterations=10000,  # 迭代次数
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

# 使用最终的最优解进行结果分析
allocation = best_solution
for i, category in enumerate(categories):
    warehouse = warehouses[int(allocation[i])]
    print(f"品类 {category} 分配到 仓库 {warehouse}")

print("优化目标值：", best_fitness)


# 在获得优化结果后分析
def analyze_results(allocation):
    # 初始化每个仓库的统计信息
    warehouse_stats = {
        w: {
            "品类列表": [],
            "已用库存": 0,
            "库存上限": warehouse_capacity[w],
            "已用产能": 0,
            "产能上限": warehouse_production[w],
            "库存利用率": 0,
            "产能利用率": 0,
            "仓租日成本": cost_dict[w],  # 添加仓租日成本
        }
        for w in warehouses
    }

    # 计算每个品类的分配情况
    for i, category in enumerate(categories):
        warehouse = warehouses[int(allocation[i])]
        inventory = category_capacity.loc[
            category_capacity["品类"] == category, "最大库存量"
        ].values[0]
        sales = category_capacity.loc[
            category_capacity["品类"] == category, "最大销量"
        ].values[0]

        # 更新仓库统计信息
        warehouse_stats[warehouse]["品类列表"].append(category)
        warehouse_stats[warehouse]["已用库存"] += inventory
        warehouse_stats[warehouse]["已用产能"] += sales

    # 计算利用率
    for w in warehouses:
        if warehouse_stats[w]["库存上限"] > 0:
            warehouse_stats[w]["库存利用率"] = (
                warehouse_stats[w]["已用库存"] / warehouse_stats[w]["库存上限"] * 100
            )
        if warehouse_stats[w]["产能上限"] > 0:
            warehouse_stats[w]["产能利用率"] = (
                warehouse_stats[w]["已用产能"] / warehouse_stats[w]["产能上限"] * 100
            )

    # 打印结果
    print("\n=== 仓库分配情况分析 ===")
    for w in warehouses:
        print(f"\n仓库 {w} 情况：")
        print(f"仓租日成本：{warehouse_stats[w]['仓租日成本']:.2f}")  # 新增这行
        print(f"包含品类数量：{len(warehouse_stats[w]['品类列表'])}")
        print(f"包含品类：{', '.join(warehouse_stats[w]['品类列表'])}")
        print(
            f"库存使用：{warehouse_stats[w]['已用库存']:.2f} / {warehouse_stats[w]['库存上限']:.2f} "
            f"(利用率: {warehouse_stats[w]['库存利用率']:.2f}%)"
        )
        print(
            f"产能使用：{warehouse_stats[w]['已用产能']:.2f} / {warehouse_stats[w]['产能上限']:.2f} "
            f"(利用率: {warehouse_stats[w]['产能利用率']:.2f}%)"
        )

    # 打印总体统计
    total_inventory_usage = sum(stats["已用库存"] for stats in warehouse_stats.values())
    total_inventory_capacity = sum(
        stats["库存上限"] for stats in warehouse_stats.values()
    )
    total_production_usage = sum(
        stats["已用产能"] for stats in warehouse_stats.values()
    )
    total_production_capacity = sum(
        stats["产能上限"] for stats in warehouse_stats.values()
    )

    print("\n=== 总体统计 ===")
    print(f"总库存利用率：{(total_inventory_usage/total_inventory_capacity*100):.2f}%")
    print(
        f"总产能利用率：{(total_production_usage/total_production_capacity*100):.2f}%"
    )

    return warehouse_stats


# 在打印优化结果后调用分析函数
warehouse_stats = analyze_results(best_solution)


def plot_warehouse_utilization(warehouse_stats):
    # 筛选出已使用的仓库
    active_warehouses = {
        w: stats for w, stats in warehouse_stats.items() if len(stats["品类列表"]) > 0
    }

    # 准备数据
    warehouses = list(active_warehouses.keys())
    capacity_usage = [stats["库存利用率"] for stats in active_warehouses.values()]
    production_usage = [stats["产能利用率"] for stats in active_warehouses.values()]

    # 创建图形
    plt.figure(figsize=(12, 6))
    x = np.arange(len(warehouses))
    width = 0.35

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


plot_warehouse_utilization(warehouse_stats)
sensitivity_analysis(best_solution)


def save_allocation_results(
    allocation, categories, warehouses, output_file="allocation_results.csv"
):
    """
    保存分配结果到CSV文件

    Args:
        allocation: PSO算法得到的最优解
        categories: 品类列表
        warehouses: 仓库列表
        output_file: 输出文件名
    """
    # 创建结果数据框
    results = pd.DataFrame(
        {
            "category": categories,
            "warehouse": [
                warehouses[int(allocation[i])] for i in range(len(categories))
            ],
        }
    )

    # 按品类序号排序
    results["category_num"] = results["category"].str.extract("(\d+)").astype(int)
    results = results.sort_values("category_num")
    results = results.drop("category_num", axis=1)

    # 保存到CSV文件
    results.to_csv(output_file, index=False, encoding="gbk")
    print(f"分配结果已保存到 {output_file}")


# 在获得最优解后调用保存函数
save_allocation_results(best_solution, categories, warehouses)
