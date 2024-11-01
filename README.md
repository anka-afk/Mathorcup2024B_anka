# 电商品类分仓优化系统

## 项目概述

本项目是一个电商品类分仓优化系统，旨在解决大规模电商仓储网络中的货量预测和品类分仓规划问题。系统包含三个主要模块：

- 货量预测模块：预测 350 个品类未来 3 个月的库存量和销量
- 一品一仓规划模块：基于预测结果进行单仓库品类分配
- 一品多仓规划模块：支持每个品类最多分配到 3 个仓库

## 功能特点

### 1. 货量预测模块

- 基于时间序列模型进行月度库存量预测
- 使用深度学习模型进行日销量预测
- 支持考虑季节性和趋势因素
- 提供预测可视化功能

### 2. 一品一仓规划模块

- 基于混合整数规划模型进行优化
- 考虑仓容上限和产能上限约束
- 优化仓库使用成本
- 平衡仓容和产能利用率

### 3. 一品多仓规划模块

- 支持单品类最多分配到 3 个仓库
- 考虑品类关联度优化
- 支持同件型、同高级品类集中存储
- 多目标优化平衡各项业务指标

## 项目结构

```
├── data/
│   ├── 附件1.csv         # 历史库存数据
│   ├── 附件2.csv         # 历史销量数据
│   ├── 附件3.csv         # 仓库信息
│   ├── 附件4.csv         # 品类关联度
│   └── 附件5.csv         # 品类信息
├── src/
│   ├── inventory_forecast.py    # 库存预测模块
│   ├── sales_forecast.py        # 销量预测模块
│   ├── single_warehouse.py      # 一品一仓规划
│   └── multi_warehouse.py       # 一品多仓规划
├── output/
│   ├── 月库存量预测结果.csv
│   ├── 日销量预测结果.csv
│   ├── 一品一仓分仓方案.csv
│   └── 一品多仓分仓方案.csv
└── requirements.txt
```

## 技术特点

1. 预测方法：

   - 使用 Prophet 模型进行时间序列预测
   - LSTM 深度学习网络预测销量
   - 考虑季节性和趋势因素
   - 集成多模型预测结果

2. 优化算法：

   - 混合整数线性规划(MILP)
   - 遗传算法优化
   - 多目标优化

3. 约束处理：
   - 仓容约束
   - 产能约束
   - 品类关联度约束
   - 分仓数量约束

## 使用方法

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 运行预测模块：

```bash
python src/inventory_forecast.py
python src/sales_forecast.py
```

3. 运行规划模块：

```bash
python src/single_warehouse.py
python src/multi_warehouse.py
```

## 主要依赖

- pandas
- numpy
- prophet
- tensorflow
- pulp
- matplotlib
- seaborn

## 结果评估

系统输出以下关键指标：

- 预测准确率
- 仓容利用率
- 产能利用率
- 总仓租成本
- 品类关联度得分

## 待优化项

- [ ] 提升预测准确率
- [ ] 优化求解速度
- [ ] 增加更多业务约束
- [ ] 支持动态调整参数

## 作者

anka-afk

## 许可证

MIT License
