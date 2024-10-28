import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("附件1.csv", encoding="gbk")
data["月份"] = pd.to_datetime(data["月份"])
data.set_index("月份", inplace=True)
data = data.sort_index()

forecast_results = {}

for category, group in data.groupby("品类"):
    group = group.resample("MS").sum()
    last_year = group.loc["2022-07":"2022-09"]
    train_data = group.loc["2023-01":"2023-06"]

    # 使用双指数平滑进行初步预测
    model = ExponentialSmoothing(
        train_data["库存量"],
        trend="add",
        seasonal=None,
        damped_trend=True,
        initialization_method="estimated",
    )
    model_fit = model.fit(smoothing_level=0.8, smoothing_trend=0.5)  # 平滑系数
    forecast = model_fit.forecast(3)

    alpha = 0.6  # 加权系数
    adjusted_forecast = (
        alpha * last_year["库存量"].values + (1 - alpha) * forecast.values
    )

    forecast_dates = pd.to_datetime(["2023-07-01", "2023-08-01", "2023-09-01"])
    adjusted_forecast_series = pd.Series(adjusted_forecast, index=forecast_dates)

    forecast_results[category] = adjusted_forecast_series

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    print(f"预测的 {category} 在2023年7-9月的库存量：")
    print(adjusted_forecast_series)

    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data["库存量"].values, label="历史库存量")
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
