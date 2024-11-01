# 赛道B：电商品类货量预测及品类分仓规划

电商企业在各区域的商品存储主要由多个仓库组成的仓群承担。其中存储的商品主要按照属性（品类、件型等）进行划分和打标，便于进行库存管理。图1是一个简化的示意图，商品品类各异，件数众多，必须将这些商品分散到各个仓库存储。品类分仓规划决定各商品存放在哪些仓库问题，合理的品类分仓规划对提升每个仓的管理效率、降低总体仓储成本至关重要。

准确的仓储货量预测是品类分仓规划的重要依据，对于准确的预测结果能够预见性地决定未来的仓储资源使用决策，以提前规划仓储资源，减少冗余场地的投入。一般来说，该场景需要预测两个目标，分别为库存量和销量。其中，库存量为该品类在全部仓库所需存放的总库存，分仓结果中受到仓库的仓容限制；销量为该品类在全部仓库所需打包出库的总量，分仓结果中受到产能限制。

在得到未来各品类的预测货量后，各个品类的分仓规划是供应链规划者的重要研究问题。若将品类集中存放在数量较少的仓库中，则将超过该仓的仓容及产能上限，造成履约问题；若同一品类分在多个仓库中，则会显著增加仓库数量，增大品类库存的管理难度及总成本。此场景需考虑的上限包括两个，分别为仓容上限和产能上限，其中仓容上限为某仓库可以存放的最高库存量，产能上限为某仓库一天可以出库的最高销量。

另外，若将相似的品类（使用品类关联度衡量相似性）放在同一个仓库中，同一订单中的商品更可能集中出货，可以在实际履约中减少包裹数量，从而降低履约成本。

合理的品类分仓方案，应该同时考虑仓群的复杂度及单仓仓容及产能约束，给出最优的分仓结果需综合考虑以下指标：

1. **仓容利用率**：单仓总库存/仓容上限；
2. **产能利用率**：单仓总出库量/产能上限；
3. **总仓租成本**：使用仓库的仓租成本之和；
4. **品类分仓数**：单品类存放的仓库数量；
5. **品类关联度**：存放在同一仓库的所有品类之间的关联度之和。

现有一个仓储网络，包含140个仓库以及350种品类，附件1及附件2分别为各品类的历史库存量及销量，附件3为不同仓库相关信息（仓租日成本、仓容上限、产能上限），附件4为不同品类之间的关联度（表中未出现的品类组合关联度设为0），附件5为不同品类的相关信息（品类编码、件型）。基于以上数据，请完成以下问题。

## 初赛问题

### 问题1

建立货量预测模型，对该仓储网络350个品类未来3个月（7-9月）每个月的库存量及销量进行预测，其中库存量根据历史每月数据预测月均库存量即可，填写表1的预测结果并放在正文中，并将完整结果填写在`result`表格文件中的“月库存预测结果”的表单中；销量需给出未来每天的预测结果，填写表2的预测结果并放在正文中，并将完整结果填写在`result`表格文件中的“日销量预测结果”的表单中。

#### 表1：月库存量预测结果

|          | 7月库存量 | 8月库存量 | 9月库存量 |
|----------|-----------|-----------|-----------|
| category1   |           |           |           |
| category31  |           |           |           |
| category61  |           |           |           |
| category91  |           |           |           |
| category121 |           |           |           |
| category151 |           |           |           |
| category181 |           |           |           |
| category211 |           |           |           |
| category241 |           |           |           |
| category271 |           |           |           |
| category301 |           |           |           |
| category331 |           |           |           |

#### 表2：日销量预测结果

|          | 7.1 | 7.11 | 7.21 | 7.31 | 8.11 | 8.21 | 8.31 | 9.11 | 9.21 |
|----------|------|------|------|------|------|------|------|------|------|
| category1   |      |      |      |      |      |      |      |      |      |
| category31  |      |      |      |      |      |      |      |      |      |
| category61  |      |      |      |      |      |      |      |      |      |
| category91  |      |      |      |      |      |      |      |      |      |
| category121 |      |      |      |      |      |      |      |      |      |
| category151 |      |      |      |      |      |      |      |      |      |
| category181 |      |      |      |      |      |      |      |      |      |
| category211 |      |      |      |      |      |      |      |      |      |
| category241 |      |      |      |      |      |      |      |      |      |
| category271 |      |      |      |      |      |      |      |      |      |
| category301 |      |      |      |      |      |      |      |      |      |
| category331 |      |      |      |      |      |      |      |      |      |

### 问题2

假设当前限定每个品类只能放在一个仓库中，即一品一仓，各品类之间请基于问题1的预测结果建立规划模型，综合考虑多个业务目标，求得品类的分仓方案，包括：应使用哪些仓库，使用的仓库需存放哪些品类的库存。填写表3的分仓结果并放在正文中，并将完整品类分仓结果填写在`result`表格中的“一品一仓分仓方案”表单中。

#### 表3：“一品一仓”分仓方案

| warehouse |           |
|-----------|-----------|
| category1   |           |
| category31  |           |
| category61  |           |
| category91  |           |
| category121 |           |
| category151 |           |
| category181 |           |
| category211 |           |
| category241 |           |
| category271 |           |
| category301 |           |
| category331 |           |

### 问题3

现在为每个品类按照件型及高级品类进行打标（如附件5），并放开一品一仓假设，即允许一个品类存放于多个仓库，但同一品类存放的仓库数量不能超过3个，并希望同件型、同高级品类尽量放在一个仓库中。假设同一品类在不同仓库之间分布的库存量比例及出库量比例相同，当前业务的首要目标是最大品类关联度，同时兼顾其他指标。请基于问题1的预测结果建立规划模型，求得新的品类分仓方案，并分析不同方案中各业务指标的表现。填写表4的分仓结果并放在正文中，并将完整品类分仓结果填写在`result`表格中的“一品多仓分仓方案”表单中。

#### 表4：“一品多仓”分仓方案

| warehouse | warehouse | warehouse |
|-----------|-----------|-----------|
| category1   |           |           |
| category31  |           |           |
| category61  |           |           |
| category91  |           |           |
| category121 |           |           |
| category151 |           |           |
| category181 |           |           |
| category211 |           |           |
| category241 |           |           |
| category271 |           |           |
| category301 |           |           |
| category331 |           |           |

*注：提交论文时，请将`result`结果文件表作为计算结果提交，提交时不要改动`result`表格的格式。*

# 部分数据一览

附件1.csv:

| 品类          | 月份       | 库存量     |
| ----------- | -------- | ------- |
| category225 | 2023/6/1 | 4676058 |
| category84  | 2023/1/1 | 4421974 |
| category21  | 2023/1/1 | 4411095 |
| category84  | 2022/7/1 | 3689222 |
| category84  | 2023/2/1 | 3431261 |
|             |          |         |
每个品类共有9条数据项

### 数据特征信息

- **库存量统计**:
  - 总记录数: 3150
  - 平均库存量: 119,272.8
  - 库存量标准差: 347,302.4
  - 库存量范围: 最小值 1，最大值 4,676,058

附件2.csv:

### 数据概览

| 品类        | 日期         | 销量     |
|-------------|--------------|----------|
| category84  | 2023/6/18    | 141914   |
| category21  | 2022/8/31    | 130556   |
| category84  | 2023/6/1     | 115928   |
| category225 | 2023/6/1     | 110234   |
| category21  | 2022/8/29    | 110020   |

### 数据特征信息

- **销量统计**:
  - 总记录数: 59396
  - 平均销量: 1290.24
  - 销量标准差: 4344.24
  - 销量范围: 最小值 0，最大值 141,914

附件3.csv:

### 数据概览

| 仓库          | 仓容上限   | 产能上限   | 仓租日成本     |
|---------------|------------|------------|----------------|
| warehouse1    | 618887     | 20310      | 1826.33       |
| warehouse2    | 1200847    | 41117      | 7073.34       |
| warehouse3    | 781957     | 33857      | 6018.78       |
| warehouse4    | 1112506    | 35571      | 11591.78      |
| warehouse5    | 42989      | 61         | 172.46        |

### 数据特征信息

- **仓容上限**:
  - 总记录数: 140
  - 平均值: 1,314,409
  - 标准差: 2,118,503
  - 范围: 最小值 1,200，最大值 13,500,000

- **产能上限**:
  - 平均值: 31,271.78
  - 范围: 最小值 4，最大值 445,991

- **仓租日成本**:
  - 平均日成本: 7,466.10
  - 范围: 最小值 6.95，最大值 107,298.07

附件4.csv:

### 数据概览

| 品类1         | 品类2         | 关联度   |
|---------------|---------------|----------|
| category157   | category195   | 103      |
| category157   | category226   | 7448     |
| category157   | category60    | 45       |
| category157   | category279   | 119      |
| category195   | category226   | 3139     |

### 数据特征信息

- **关联度**:
  - 总记录数: 3164
  - 平均关联度: 593.40
  - 标准差: 2908.91
  - 范围: 最小值 11，最大值 74,630

附件5.csv:

### 数据概览

| 品类          | 高级品类           | 件型  |
| ----------- | -------------- | --- |
| category165 | high_category1 | B   |
| category204 | high_category2 | A   |
| category134 | high_category3 | A   |
| category110 | high_category4 | A   |
| category199 | high_category5 | B   |

### 数据特征信息

- **品类**:
  - 总记录数: 350
  - 独特值数量: 350
  
- **高级品类**:
  - 独特值数量: 45
  - 出现最多的高级品类: high_category3 (出现57次)
  
- **件型**:
  - 类别数量: 3 (A, B, C)
  - 出现最多的件型: C (出现171次)


