import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import matplotlib.pyplot as plt

# 可调节参数配置
PARAMS = {
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


def find_best_params(train_data, initial_params=None):
    """自动搜索最优参数"""
    if initial_params is None:
        initial_params = {
            "smoothing_level": [0.1, 0.3, 0.5, 0.7, 0.9],
            "smoothing_trend": [0.1, 0.3, 0.5, 0.7, 0.9],
            "damping_trend": [0.8, 0.9, 0.98],  # 添加阻尼因子
        }

    best_aic = float("inf")
    best_params = {}

    for level in initial_params["smoothing_level"]:
        for trend in initial_params["smoothing_trend"]:
            for damp in initial_params["damping_trend"]:
                try:
                    model = ExponentialSmoothing(
                        train_data,
                        trend="add",
                        damped_trend=True,
                        initialization_method="estimated",
                    )
                    fit = model.fit(
                        smoothing_level=level,
                        smoothing_trend=trend,
                        damping_trend=damp,
                        optimized=False,
                    )
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_params = {
                            "smoothing_level": level,
                            "smoothing_trend": trend,
                            "damping_trend": damp,
                        }
                except:
                    continue

    return best_params


def adaptive_alpha(last_year_values, forecast_values):
    """根据去年同期和预测值的差异调整alpha"""
    diff_ratio = np.abs(last_year_values - forecast_values) / (last_year_values + 1e-5)
    dynamic_alpha = np.clip(PARAMS["initial_alpha"] + 0.5 * diff_ratio.mean(), 0, 1)
    return dynamic_alpha


for category, group in data.groupby("品类"):
    group = group.resample("MS").sum()

    # 获取去年同期数据
    last_year = group.loc[
        f"{PARAMS['last_year_period'][0]}" :f"{PARAMS['last_year_period'][1]}"
    ]

    # 获取训练数据
    train_data = group.loc[
        f"{PARAMS['train_period'][0]}" :f"{PARAMS['train_period'][1]}"
    ]
    train_values = train_data["库存量"]

    # 获取最优参数
    best_params = find_best_params(train_values)

    # 使用最优参数训练模型
    model = ExponentialSmoothing(
        train_values,
        trend="add",
        seasonal=None,
        damped_trend=True,
        initialization_method="estimated",
    )

    model_fit = model.fit(
        smoothing_level=best_params["smoothing_level"],
        smoothing_trend=best_params["smoothing_trend"],
        damping_trend=best_params["damping_trend"],
        optimized=False,
    )

    forecast = model_fit.forecast(PARAMS["forecast_months"])

    # 计算自适应alpha值
    adaptive_alpha_value = adaptive_alpha(last_year["库存量"].values, forecast.values)

    # 调整预测值
    adjusted_forecast = (
        adaptive_alpha_value * last_year["库存量"].values
        + (1 - adaptive_alpha_value) * forecast.values
    )

    forecast_dates = pd.to_datetime(["2023-07-01", "2023-08-01", "2023-09-01"])
    adjusted_forecast_series = pd.Series(adjusted_forecast, index=forecast_dates)

    forecast_results[category] = adjusted_forecast_series

    plt.rcParams["font.sans-serif"] = [PLOT_CONFIG["font_family"]]
    plt.rcParams["axes.unicode_minus"] = False

    print(f"预测的 {category} 在2023年7-9月的库存量：")
    print(adjusted_forecast_series)

    plt.figure(figsize=PLOT_CONFIG["figure_size"])
    plt.plot(train_data.index, train_data["库存量"], label="历史库存量")

    connection_dates = [train_data.index[-1], adjusted_forecast_series.index[0]]
    connection_values = [
        train_data["库存量"].iloc[-1],
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
