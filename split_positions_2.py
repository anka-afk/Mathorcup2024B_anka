import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial

# 读取数据
inventory_data = pd.read_csv("月库存量预测结果.csv", encoding="gbk")
sales_data = pd.read_csv("预测结果_非负处理.csv", encoding="gbk")
warehouse_data = pd.read_csv("附件3.csv", encoding="gbk")
correlation_data = pd.read_csv("附件4.csv", encoding="gbk")
category_info = pd.read_csv("附件5.csv", encoding="gbk")

# 数据预处理
inventory_max = inventory_data.groupby("品类")["库存量"].max().reset_index()
inventory_max.columns = ["品类", "最大库存量"]
sales_max = (
    sales_data.melt(id_vars=["品类"], var_name="日期", value_name="销量")
    .groupby("品类")["销量"]
    .max()
    .reset_index()
)
sales_max.columns = ["品类", "最大销量"]
category_capacity = pd.merge(inventory_max, sales_max, on="品类")

# 变量提取
categories = category_capacity["品类"].tolist()
warehouses = warehouse_data["仓库"].tolist()
cost_dict = warehouse_data.set_index("仓库")["仓租日成本"].to_dict()
warehouse_capacity = warehouse_data.set_index("仓库")["仓容上限"].to_dict()
warehouse_production = warehouse_data.set_index("仓库")["产能上限"].to_dict()
correlation_matrix = correlation_data.set_index(["品类1", "品类2"])["关联度"].to_dict()

# 参数初始化
num_particles = 100
num_iterations = 100
w = 0.5
c1 = 2.0
c2 = 2.0
temperature = 50
cooling_rate = 0.98


# 粒子类定义
class Particle:
    def __init__(self):
        self.position = {cat: random.choice(warehouses) for cat in categories}
        self.velocity = {cat: 0 for cat in categories}
        self.best_position = self.position.copy()
        self.category_inventory = {
            cat: category_capacity.set_index("品类").at[cat, "最大库存量"]
            for cat in categories
        }
        self.category_sales = {
            cat: category_capacity.set_index("品类").at[cat, "最大销量"]
            for cat in categories
        }
        self.best_fitness = self.evaluate(self.position)

    def evaluate(self, position):
        # 使用set减少重复计算
        used_warehouses = set(position.values())
        total_cost = sum(cost_dict[ware] for ware in used_warehouses)

        # 优化关联度计算
        total_correlation = 0
        for cat1, ware1 in position.items():
            for cat2, ware2 in position.items():
                if cat1 < cat2 and ware1 == ware2:  # 只计算一半的组合
                    total_correlation += correlation_matrix.get((cat1, cat2), 0)

        # 优化仓库使用计算
        warehouse_inventory = {ware: 0 for ware in used_warehouses}
        warehouse_sales = {ware: 0 for ware in used_warehouses}

        for cat, ware in position.items():
            warehouse_inventory[ware] += self.category_inventory[cat]
            warehouse_sales[ware] += self.category_sales[cat]

        utilization_penalty = 0
        utilization_reward = 0

        for ware in used_warehouses:
            utilization_ratio = warehouse_inventory[ware] / warehouse_capacity[ware]
            production_ratio = warehouse_sales[ware] / warehouse_production[ware]

            if utilization_ratio > 1:
                utilization_penalty += (utilization_ratio - 1) * 1000
            elif utilization_ratio >= 0.8:
                utilization_reward += utilization_ratio * 50

            if production_ratio > 1:
                utilization_penalty += (production_ratio - 1) * 1000
            elif production_ratio >= 0.8:
                utilization_reward += production_ratio * 50

        return total_cost + utilization_penalty - utilization_reward, -total_correlation

    def update_velocity(self, global_best_position):
        for cat in categories:
            r1, r2 = random.random(), random.random()
            self.velocity[cat] = (
                w * self.velocity[cat]
                + c1 * r1 * (self.best_position[cat] != self.position[cat])
                + c2 * r2 * (global_best_position[cat] != self.position[cat])
            )

    def update_position(self):
        for cat in categories:
            if random.random() < self.velocity[cat]:
                self.position[cat] = random.choice(warehouses)
        # 位置更新后检查约束
        self.apply_constraints()

    def apply_constraints(self):
        warehouse_inventory = {ware: 0 for ware in warehouses}
        warehouse_sales = {ware: 0 for ware in warehouses}

        # 计算每个仓库的实际库存和销量
        for cat, ware in self.position.items():
            warehouse_inventory[ware] += self.category_inventory[cat]
            warehouse_sales[ware] += self.category_sales[cat]

        # 逐仓库检查约束条件，调整分配以确保约束
        for ware in warehouses:
            while (
                warehouse_inventory[ware] > warehouse_capacity[ware]
                or warehouse_sales[ware] > warehouse_production[ware]
            ):
                # 随机选择一个超出约束的品类重新分配到新的仓库
                over_capacity_cats = [
                    cat for cat, w in self.position.items() if w == ware
                ]
                if over_capacity_cats:
                    cat_to_move = random.choice(over_capacity_cats)
                    new_ware = random.choice([w for w in warehouses if w != ware])

                    # 更新位置及库存、销量记录
                    self.position[cat_to_move] = new_ware
                    warehouse_inventory[ware] -= self.category_inventory[cat_to_move]
                    warehouse_sales[ware] -= self.category_sales[cat_to_move]
                    warehouse_inventory[new_ware] += self.category_inventory[
                        cat_to_move
                    ]
                    warehouse_sales[new_ware] += self.category_sales[cat_to_move]

    def simulated_annealing(self, current_fitness):
        new_position = self.position.copy()
        for cat in random.sample(categories, k=len(categories) // 10):
            new_position[cat] = random.choice(warehouses)
        new_fitness = self.evaluate(new_position)
        delta_fitness = new_fitness[0] - current_fitness[0]
        if delta_fitness < 0 or random.random() < np.exp(-delta_fitness / temperature):
            self.position = new_position
            self.best_fitness = new_fitness


# 添加并行计算辅助函数
def process_particle(particle, global_best_position, temperature):
    particle.update_velocity(global_best_position)
    particle.update_position()
    particle.simulated_annealing(particle.best_fitness)
    fitness = particle.evaluate(particle.position)
    return (fitness, particle.position.copy(), particle)


# 修改主循环，添加并行处理
if __name__ == "__main__":
    # 初始化进程池
    num_cores = mp.cpu_count() - 1  # 留一个核心给系统
    pool = mp.Pool(processes=num_cores)

    # 初始化粒子群
    particles = [Particle() for _ in range(num_particles)]
    pareto_front = []

    # 设置动态可视化
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    visualization_interval = 5
    for iteration in range(num_iterations):
        # 确定全局最优位置
        if pareto_front:
            global_best_position = min(pareto_front, key=lambda x: x[0])[2]
        else:
            global_best_position = particles[0].position

        # 并行处理所有粒子
        process_func = partial(
            process_particle,
            global_best_position=global_best_position,
            temperature=temperature,
        )

        # 并行执行粒子更新
        results = pool.map(process_func, particles)

        # 更新粒子群和Pareto前沿
        new_particles = []
        for fitness, position, particle in results:
            new_particles.append(particle)
            if not any(
                (fitness[0] <= f[0] and fitness[1] <= f[1]) for f in pareto_front
            ):
                pareto_front.append((fitness[0], fitness[1], position))
                if len(pareto_front) > num_particles:
                    pareto_front = sorted(pareto_front, key=lambda x: x[0])[
                        :num_particles
                    ]

        particles = new_particles

        # 可视化更新
        if iteration % visualization_interval == 0:
            ax.clear()
            costs, correlations = zip(*[(p[0], -p[1]) for p in pareto_front])
            warehouse_utilization = [len(set(p[2].values())) for p in pareto_front]
            ax.scatter(costs, correlations, warehouse_utilization, color="blue")
            ax.set_xlabel("Total Cost")
            ax.set_ylabel("Total Correlation")
            ax.set_zlabel("Warehouse Utilization")
            ax.set_title(f"Pareto Front at Iteration {iteration + 1}")
            plt.draw()
            plt.pause(0.01)

        temperature *= cooling_rate

    pool.close()
    pool.join()

    plt.ioff()
    plt.show()

    # 提取最终分仓方案
    best_solution = min(pareto_front, key=lambda x: x[0])
    final_allocation = best_solution[2]

    result_df = pd.DataFrame(list(final_allocation.items()), columns=["品类", "仓库"])
    result_df.to_csv("final_allocation.csv", index=False, encoding="utf-8-sig")
    print("Final allocation saved to final_allocation.csv")
