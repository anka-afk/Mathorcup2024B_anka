import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# 读取附件数据
inventory_data = pd.read_csv("月库存量预测结果.csv", encoding="gbk")
sales_data = pd.read_csv("预测结果_非负处理.csv", encoding="gbk")
warehouse_data = pd.read_csv("附件3.csv", encoding="gbk")
correlation_data = pd.read_csv("附件4.csv", encoding="gbk")
category_info = pd.read_csv("附件5.csv", encoding="gbk")

# 计算库存量的最大值
inventory_max = inventory_data.groupby("品类")["库存量"].max().reset_index()
inventory_max.columns = ["品类", "最大库存量"]

# 计算销量的最大值
sales_data_melted = sales_data.melt(
    id_vars=["品类"], var_name="日期", value_name="销量"
)
sales_max = sales_data_melted.groupby("品类")["销量"].max().reset_index()
sales_max.columns = ["品类", "最大销量"]

# 合并数据
category_capacity = pd.merge(inventory_max, sales_max, on="品类")

# 提取品类、仓库和目标函数数据
categories = category_capacity["品类"].tolist()
warehouses = warehouse_data["仓库"].tolist()
cost_dict = warehouse_data.set_index("仓库")["仓租日成本"].to_dict()

# 关联度矩阵
correlation_matrix = correlation_data.pivot(
    index="品类1", columns="品类2", values="关联度"
).fillna(0)

# 初始化参数
num_particles = 100  # 增加粒子数量以提升解的多样性
num_iterations = 200  # 增加迭代次数以细化搜索
w = 0.5  # 调整惯性权重
c1 = 2.0  # 增强个体学习因子
c2 = 2.0  # 增强社会学习因子
temperature = 100  # 模拟退火初始温度
cooling_rate = 0.95  # 温度下降系数


# 粒子类
class Particle:
    def __init__(self):
        # 每个粒子的分配方案（每个品类随机分配一个仓库）
        self.position = {cat: random.choice(warehouses) for cat in categories}
        self.velocity = {cat: 0 for cat in categories}
        self.best_position = self.position.copy()
        self.best_fitness = self.evaluate(self.position)

    def evaluate(self, position):
        # 计算总仓储成本
        total_cost = sum(cost_dict[ware] for ware in position.values())

        # 计算总关联度
        total_correlation = 0
        for cat1 in categories:
            for cat2 in categories:
                if cat1 != cat2 and position[cat1] == position[cat2]:
                    try:
                        correlation = correlation_matrix.at[cat1, cat2]
                    except KeyError:
                        correlation = 0
                    total_correlation += correlation

        return total_cost, -total_correlation  # 费用最小化，关联度最大化

    def update_velocity(self, global_best_position):
        for cat in categories:
            r1 = random.random()
            r2 = random.random()
            self.velocity[cat] = (
                w * self.velocity[cat]
                + c1 * r1 * (self.best_position[cat] != self.position[cat])
                + c2 * r2 * (global_best_position[cat] != self.position[cat])
            )

    def update_position(self):
        for cat in categories:
            if random.random() < self.velocity[cat]:
                self.position[cat] = random.choice(warehouses)

    def simulated_annealing(self, current_fitness):
        # 模拟退火优化：尝试改变某些位置，并接受或拒绝新的位置
        new_position = self.position.copy()
        for cat in random.sample(categories, k=len(categories) // 10):
            new_position[cat] = random.choice(warehouses)
        new_fitness = self.evaluate(new_position)
        delta_fitness = new_fitness[0] - current_fitness[0]
        # 通过概率接受更差的解以跳出局部最优
        if delta_fitness < 0 or random.random() < np.exp(-delta_fitness / temperature):
            self.position = new_position
            self.best_fitness = new_fitness


# 初始化粒子群
particles = [Particle() for _ in range(num_particles)]
pareto_front = []

from mpl_toolkits.mplot3d import Axes3D

# 设置实时动态三维可视化
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# 迭代优化
for iteration in range(num_iterations):
    print(f"Iteration {iteration + 1}/{num_iterations}")
    for particle in particles:
        if pareto_front:
            global_best_position = min(pareto_front, key=lambda x: x[1])[2]
        else:
            global_best_position = particle.position

        # 更新粒子速度和位置
        particle.update_velocity(global_best_position)
        particle.update_position()

        # 模拟退火优化
        particle.simulated_annealing(particle.best_fitness)

        # 计算适应度并更新帕累托前沿
        fitness = particle.evaluate(particle.position)
        position_copy = particle.position.copy()

        # 检查帕累托更新条件
        if not any((fitness[0] <= f[0] and fitness[1] <= f[1]) for f in pareto_front):
            pareto_front.append((fitness[0], fitness[1], position_copy))
            if len(pareto_front) > num_particles:
                pareto_front = sorted(pareto_front, key=lambda x: x[0])[:num_particles]

    # 动态更新帕累托前沿的三维图表
    ax.clear()
    costs, correlations = zip(*[(p[0], -p[1]) for p in pareto_front])
    warehouse_utilization = [
        len(set(p[2].values())) for p in pareto_front
    ]  # 每个解的仓库利用率
    ax.scatter(costs, correlations, warehouse_utilization, color="blue")
    ax.set_xlabel("Total Cost")
    ax.set_ylabel("Total Correlation")
    ax.set_zlabel("Warehouse Utilization")
    ax.set_title(f"Pareto Front at Iteration {iteration + 1}")
    plt.draw()
    plt.pause(0.01)

    # 降低模拟退火的温度
    temperature *= cooling_rate

plt.ioff()
plt.show()

# 提取最终分仓方案（选择一个帕累托最优解）
best_solution = min(pareto_front, key=lambda x: x[0])  # 最小成本方案
final_allocation = best_solution[2]

# 输出最终分仓方案表格
result_df = pd.DataFrame(list(final_allocation.items()), columns=["品类", "仓库"])
result_df.to_csv("final_allocation.csv", index=False, encoding="utf-8-sig")
print("Final allocation saved to final_allocation.csv")
