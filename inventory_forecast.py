import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import matplotlib.pyplot as plt

# 可调节参数配置
PARAMS = {
    "smoothing_level": 0.8,  # 指数平滑级别系数
    "smoothing_trend": 0.5,  # 趋势平滑系数
    "alpha": 0.6,  # 历史数据与预测值的加权系数
    "window_size": 6,  # 滚动窗口大小（月）
    "forecast_months": 3,  # 预测未来月数
    "step_size": 1,  # 每次滚动的步长（月）
    "last_year_period": ("2022-07", "2022-09"),  # 去年同期数据
    "train_period": ("2023-01", "2023-06"),  # 训练数据期间
}

# 图表配置
PLOT_CONFIG = {
    "figure_size": (10, 6),
    "font_family": "SimHei",
}

data = pd.read_csv("附件1.csv", encoding="gbk")
data["月份"] = pd.to_datetime(data["月份"])
data.set_index("月份", inplace=True)
data = data.sort_index()

forecast_results = {}

for category, group in data.groupby("品类"):
    group = group.resample("MS").sum()

    # 获取去年同期和训练数据
    last_year = group.loc[
        f"{PARAMS['last_year_period'][0]}" :f"{PARAMS['last_year_period'][1]}"
    ]
    train_data = group.loc[
        f"{PARAMS['train_period'][0]}" :f"{PARAMS['train_period'][1]}"
    ]

    # 预测未来3个月
    model = ExponentialSmoothing(
        train_data["库存量"],
        trend="add",
        seasonal=None,
        damped_trend=True,
        initialization_method="estimated",
    )
    model_fit = model.fit(
        smoothing_level=PARAMS["smoothing_level"],
        smoothing_trend=PARAMS["smoothing_trend"],
        optimized=True,
    )

    # 直接预测未来3个月
    forecast = model_fit.forecast(PARAMS["forecast_months"])

    # 确保去年同期数据和预测数据长度相同
    adjusted_forecast = (
        PARAMS["alpha"] * last_year["库存量"].values
        + (1 - PARAMS["alpha"]) * forecast.values
    )

    forecast_dates = pd.to_datetime(["2023-07-01", "2023-08-01", "2023-09-01"])
    adjusted_forecast_series = pd.Series(adjusted_forecast, index=forecast_dates)

    forecast_results[category] = adjusted_forecast_series

    plt.rcParams["font.sans-serif"] = [PLOT_CONFIG["font_family"]]
    plt.rcParams["axes.unicode_minus"] = False

    print(f"预测的 {category} 在2023年7-9月的库存量：")
    print(adjusted_forecast_series)

    plt.figure(figsize=PLOT_CONFIG["figure_size"])
    plt.plot(train_data.index, train_data["库存量"].values, label="历史库存量")

    connection_dates = [train_data.index[-1], adjusted_forecast_series.index[0]]
    connection_values = [
        train_data["库存量"].values[-1],
        adjusted_forecast_series.values[0],
    ]
    plt.plot(connection_dates, connection_values, "k-", linewidth=1)

    plt.plot(
        adjusted_forecast_series.index,
        adjusted_forecast_series.values,
        label="预测库存量",
        linestyle="--",
    )
    plt.plot(
        last_year.index.shift(12, freq="MS"),
        last_year["库存量"].values,
        label="去年同期库存量",
        linestyle=":",
        color="gray",
    )
    plt.title(f"{category} 库存量预测")
    plt.xlabel("日期")
    plt.ylabel("库存量")
    plt.legend()
    plt.show()
