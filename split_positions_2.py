import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 数据准备（与前面代码一致）
inventory_data = pd.read_csv("月库存量预测结果.csv", encoding="gbk")
sales_data = pd.read_csv("预测结果_非负处理.csv", encoding="gbk")
warehouse_data = pd.read_csv("附件3.csv", encoding="gbk")
correlation_data = pd.read_csv("附件4.csv", encoding="gbk")
category_info = pd.read_csv("附件5.csv", encoding="gbk")

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

# 遗传算法参数
population_size = 100
generations = 200
mutation_rate = 0.01


# 初始化种群
def initialize_population():
    population = []
    for _ in range(population_size):
        individual = {j: random.choice(warehouses) for j in categories}
        population.append(individual)
    return population


# 适应度函数
def fitness(individual):
    storage_utilizations = []
    production_utilizations = []
    total_cost, correlation_score = 0, 0

    for i in warehouses:
        # 计算仓库 i 的总库存和总销量
        total_inventory = sum(
            category_capacity.loc[category_capacity["品类"] == j, "最大库存量"].values[
                0
            ]
            for j, w in individual.items()
            if w == i
        )
        total_sales = sum(
            category_capacity.loc[category_capacity["品类"] == j, "最大销量"].values[0]
            for j, w in individual.items()
            if w == i
        )

        # 仅对使用中的仓库计算利用率
        if total_inventory > 0 and total_inventory <= warehouse_capacity[i]:
            storage_utilizations.append(total_inventory / warehouse_capacity[i])

        if total_sales > 0 and total_sales <= warehouse_production[i]:
            production_utilizations.append(total_sales / warehouse_production[i])

        # 累加租金成本
        total_cost += sum(cost_dict[i] for j, w in individual.items() if w == i)

        # 计算关联度得分
        for j in categories:
            for k in categories:
                if j != k and individual[j] == i and individual[k] == i:
                    correlation_score += correlation_matrix.get((j, k), 0)

    # 找到使用中的仓库的最小仓库利用率和最小产能利用率
    min_storage_utilization = min(storage_utilizations) if storage_utilizations else 0
    min_production_utilization = (
        min(production_utilizations) if production_utilizations else 0
    )

    # 加权求和
    w1, w2, w3, w4 = 1, 1, 1, 1
    return (
        w1 * min_storage_utilization
        + w2 * min_production_utilization
        - w3 * total_cost
        + w4 * correlation_score
    )


# 选择、交叉和变异
def select(population):
    return random.choices(population, weights=[fitness(ind) for ind in population], k=2)


def crossover(parent1, parent2):
    child = {}
    for j in categories:
        child[j] = parent1[j] if random.random() < 0.5 else parent2[j]
    return child


def mutate(individual):
    for j in categories:
        if random.random() < mutation_rate:
            individual[j] = random.choice(warehouses)


# 初始化绘图
fig, ax = plt.subplots()
ax.set_xlim(0, generations)
ax.set_ylim(-1000, 1000)  # 根据实际适应度值范围调整
ax.set_xlabel("Generation")
ax.set_ylabel("Best Fitness Score")
(line,) = ax.plot([], [], lw=2)
best_fitness_scores = []


# 更新函数，用于动画
def update(frame):
    global population
    new_population = []
    for _ in range(population_size):
        parent1, parent2 = select(population)
        child = crossover(parent1, parent2)
        mutate(child)
        new_population.append(child)
    population = new_population

    # 记录当前代的最佳适应度值
    best_individual = max(population, key=fitness)
    best_fitness = fitness(best_individual)
    best_fitness_scores.append(best_fitness)

    # 更新绘图数据
    line.set_data(range(len(best_fitness_scores)), best_fitness_scores)
    ax.set_title(f"Generation {frame + 1}, Best Fitness: {best_fitness:.2f}")
    return (line,)


# 动态显示
population = initialize_population()
ani = FuncAnimation(fig, update, frames=generations, blit=True, repeat=False)

plt.show()

# 找到最优解并保存结果
best_individual = max(population, key=fitness)
result_df = pd.DataFrame(
    [(category, warehouse) for category, warehouse in best_individual.items()],
    columns=["品类", "仓库"],
)
result_df.to_csv("最终分配结果.csv", index=False, encoding="gbk")

print("最优分配方案已保存到 '最终分配结果.csv'")
