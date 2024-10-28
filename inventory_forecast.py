import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("附件1.csv", encoding="gbk")
data["月份"] = pd.to_datetime(data["月份"])
data.set_index("月份", inplace=True)
data = data.sort_index()


category = "category97"
category_data = data[data["品类"] == category]["库存量"]

train_data = category_data["2023-01":"2023-06"]

last_year = category_data["2022-07":"2022-09"]
last_year_growth = last_year.pct_change().dropna().values  # 去年同期的月变化率


# 定义带同期趋势约束的损失函数
def custom_loss(params):
    alpha, beta = params
    model = ExponentialSmoothing(
        train_data,
        trend="mul",  # 使用乘法趋势而不是加法趋势
        damped_trend=True,  # 添加阻尼因子来避免趋势过度延伸
        seasonal=None,
    ).fit(smoothing_level=alpha, smoothing_trend=beta)

    # 进行预测并将结果转换为数组
    predictions = model.forecast(steps=3).values  # 使用.values获取numpy数组

    # 计算 MSE 作为基本损失
    mse_loss = np.mean((predictions - train_data.values[-1]) ** 2)

    # 计算同期趋势损失
    trend_loss = 0
    for i in range(len(predictions) - 1):
        pred_growth = (predictions[i + 1] - predictions[i]) / predictions[i]
        trend_loss += (pred_growth - last_year_growth[i]) ** 2

    # 调整权重
    total_loss = 40000000000 * trend_loss + mse_loss  # 可以调整权重来平衡两种损失
    return total_loss


# 初始参数
initial_params = [0.5, 0.5]

# 优化 Holt-Winters 参数
result = minimize(custom_loss, initial_params, bounds=[(0, 1), (0, 1)])
alpha_opt, beta_opt = result.x

# 使用最佳参数进行最终预测
final_model = ExponentialSmoothing(
    train_data,
    trend="mul",  # 使用乘法趋势
    damped_trend=True,  # 添加阻尼因子
    seasonal=None,
).fit(smoothing_level=alpha_opt, smoothing_trend=beta_opt)

# 修正预测日期为 7 月 1 日、8 月 1 日和 9 月 1 日
forecast = pd.Series(
    final_model.forecast(steps=3).values,
    index=pd.date_range(start="2023-07-01", periods=3, freq="MS"),
)

# 设置pandas显示格式，避免科学计数法
pd.set_option("display.float_format", lambda x: "%.2f" % x)

# 设置matplotlib中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

# 输出预测结果
print("预测的 2023 年 7-9 月库存量：")
print(forecast)

# 可视化预测结果与历史数据，并绘制去年同期数据
plt.figure(figsize=(10, 6))
plt.plot(train_data.index, train_data.values, label="历史库存量")
plt.plot(forecast.index, forecast.values, label="预测库存量", linestyle="--")
plt.plot(
    last_year.index.shift(12, freq="M"),
    last_year.values,
    label="去年同期库存量",
    linestyle=":",
    color="gray",
)
plt.title(f"{category} 库存量预测 (Holt-Winters + 同期趋势)")
plt.xlabel("日期")
plt.ylabel("库存量")
plt.legend()
plt.show()
