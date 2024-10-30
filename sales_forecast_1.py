import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import ParameterGrid
from xgboost import XGBRegressor

# 读取数据
sales_data = pd.read_csv("填补完缺失值的销量数据.csv", encoding="gbk")
holidays = pd.DataFrame(
    {
        "holiday": "sales_peak",
        "ds": pd.to_datetime(["2023-01-01", "2023-05-01", "2023-10-01"]),
        "lower_window": -1,
        "upper_window": 1,
    }
)

# 加载关联度数据并处理为特征
correlation_data = pd.read_csv("附件4.csv", encoding="gbk")
category_correlation = correlation_data[correlation_data["品类1"] == "category1"]

# 计算关联度加权特征
sales_data["related_sales"] = 0
for idx, row in category_correlation.iterrows():
    related_category = row["品类2"]
    weight = row["关联度"]
    related_sales = sales_data[sales_data["品类"] == related_category]["销量"].values
    if len(related_sales) > 0:
        sales_data["related_sales"] += weight * related_sales[: len(sales_data)]

# 选择需要预测的品类
category_name = "category1"
category_sales_data = sales_data[sales_data["品类"] == category_name][
    ["日期", "销量", "related_sales"]
].copy()

# 转换日期格式并设置索引，指定频率为日
category_sales_data.columns = ["ds", "y", "related_sales"]
category_sales_data["ds"] = pd.to_datetime(category_sales_data["ds"])
category_sales_data.set_index("ds", inplace=True)
category_sales_data = category_sales_data.asfreq("D")

# 特征工程
category_sales_data["day_of_week"] = category_sales_data.index.dayofweek
category_sales_data["day_of_month"] = category_sales_data.index.day
category_sales_data["month"] = category_sales_data.index.month
category_sales_data["lag_7"] = category_sales_data["y"].shift(7)
category_sales_data["rolling_mean_7"] = (
    category_sales_data["y"].rolling(window=7).mean()
)
category_sales_data["rolling_std_7"] = category_sales_data["y"].rolling(window=7).std()
category_sales_data.dropna(inplace=True)

# Prophet 模型
prophet_data = category_sales_data[["y"]].reset_index()
prophet_data.columns = ["ds", "y"]
prophet_model = Prophet(
    yearly_seasonality=True, weekly_seasonality=True, holidays=holidays
)
prophet_model.fit(prophet_data)
future_dates = prophet_model.make_future_dataframe(periods=92)
prophet_forecast = prophet_model.predict(future_dates)
prophet_forecast_trend = prophet_forecast[["ds", "trend"]].set_index("ds").tail(92)

# SARIMA 模型
sarima_model = SARIMAX(
    category_sales_data["y"],
    exog=category_sales_data[
        [
            "day_of_week",
            "day_of_month",
            "month",
            "lag_7",
            "rolling_mean_7",
            "rolling_std_7",
            "related_sales",
        ]
    ],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
)
sarima_fit = sarima_model.fit(disp=False, maxiter=200)
future_dates_only = future_dates["ds"].tail(92).reset_index(drop=True)
future_features = pd.DataFrame({"ds": future_dates_only})
future_features["day_of_week"] = future_features["ds"].dt.dayofweek
future_features["day_of_month"] = future_features["ds"].dt.day
future_features["month"] = future_features["ds"].dt.month
last_values = category_sales_data.iloc[-7:]["y"]
future_features["lag_7"] = last_values.mean()
future_features["rolling_mean_7"] = last_values.mean()
future_features["rolling_std_7"] = last_values.std()
future_features["related_sales"] = category_sales_data["related_sales"].mean()
future_features.set_index("ds", inplace=True)
forecast_sarima = sarima_fit.get_forecast(steps=92, exog=future_features)
sarima_seasonal = forecast_sarima.predicted_mean

# LSTM 模型
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(category_sales_data[["y"]])
tscv = TimeSeriesSplit(n_splits=5)
lstm_window = 7
best_rmse = float("inf")
best_model = None
for train_index, val_index in tscv.split(scaled_data):
    train, val = scaled_data[train_index], scaled_data[val_index]
    X_train, y_train = [], []
    for i in range(lstm_window, len(train)):
        X_train.append(train[i - lstm_window : i, 0])
        y_train.append(train[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=0)  # 增加 epochs
    X_val, y_val = [], []
    for i in range(lstm_window, len(val)):
        X_val.append(val[i - lstm_window : i, 0])
        y_val.append(val[i, 0])
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
    y_pred = model.predict(X_val)
    y_val = scaler.inverse_transform(y_val.reshape(-1, 1))
    y_pred = scaler.inverse_transform(y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
print(f"Best LSTM model RMSE: {best_rmse}")
inputs = scaled_data[-lstm_window:]
inputs = np.reshape(inputs, (1, lstm_window, 1))
predicted_lstm_scaled = []
for _ in range(92):
    predicted = best_model.predict(inputs)
    predicted_lstm_scaled.append(predicted[0, 0])
    inputs = np.append(inputs[:, 1:, :], np.reshape(predicted, (1, 1, 1)), axis=1)
predicted_lstm = scaler.inverse_transform(
    np.array(predicted_lstm_scaled).reshape(-1, 1)
)

forecast_combined = pd.DataFrame(
    {
        "ds": future_dates_only,
        "Prophet_Trend": prophet_forecast_trend["trend"].values,
        "SARIMA_Seasonal": sarima_seasonal.values,
        "LSTM_Random": predicted_lstm.flatten(),
    }
)

param_grid = {
    "w1": np.arange(0, 1.1, 0.1),
    "w2": np.arange(0, 1.1, 0.1),
    "w3": np.arange(0, 1.1, 0.1),
}
param_grid = [p for p in ParameterGrid(param_grid) if np.isclose(sum(p.values()), 1)]
best_weights = None
best_error = float("inf")

for params in param_grid:
    w1, w2, w3 = params["w1"], params["w2"], params["w3"]
    forecast_combined["weighted_yhat"] = (
        w1 * forecast_combined["Prophet_Trend"]
        + w2 * forecast_combined["SARIMA_Seasonal"]
        + w3 * forecast_combined["LSTM_Random"]
    )
    rmse = np.sqrt(
        mean_squared_error(
            category_sales_data["y"].tail(92), forecast_combined["weighted_yhat"]
        )
    )
    if rmse < best_error:
        best_error = rmse
        best_weights = (w1, w2, w3)

w1, w2, w3 = best_weights
print(f"Best weights: w1={w1}, w2={w2}, w3={w3}")

# 结合最佳权重后的最终预测
forecast_combined["final_yhat"] = (
    w1 * forecast_combined["Prophet_Trend"]
    + w2 * forecast_combined["SARIMA_Seasonal"]
    + w3 * forecast_combined["LSTM_Random"]
)

# 误差修正
train_data = forecast_combined[["Prophet_Trend", "SARIMA_Seasonal", "LSTM_Random"]]
train_target = category_sales_data["y"].tail(len(train_data))
error_model = XGBRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
error_model.fit(train_data, train_target)
forecast_combined["yhat"] = error_model.predict(
    forecast_combined[["Prophet_Trend", "SARIMA_Seasonal", "LSTM_Random"]]
)

# 绘制结果
plt.figure(figsize=(14, 7))
plt.plot(category_sales_data.index, category_sales_data["y"], label="Actual")
plt.plot(
    forecast_combined["ds"], forecast_combined["yhat"], label="Forecast", color="orange"
)
plt.title(f"Sales Forecast vs Actual for {category_name}")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()
