import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# LOAD DATA
# ==============================
def load_data(tower_list):
    data = []
    for tid in tower_list:
        df = pd.read_csv(f"tower_{tid}.csv")
        df["tower_id"] = tid
        data.append(df)
    return pd.concat(data, ignore_index=True)

TRAIN_TOWERS = ["T0", "T1", "T2", "T3"]
TEST_TOWER = "T7"

train_df = load_data(TRAIN_TOWERS)
test_df = load_data([TEST_TOWER])

# ==============================
# SORT
# ==============================
train_df = train_df.sort_values("time")
test_df = test_df.sort_values("time")

# ==============================
# ADD MULTIPLE LAGS (🔥 VERY IMPORTANT)
# ==============================
for lag in [1, 2, 3]:
    train_df[f"lag_{lag}"] = train_df["battery_pct"].shift(lag)
    test_df[f"lag_{lag}"] = test_df["battery_pct"].shift(lag)

train_df = train_df.dropna()
test_df = test_df.dropna()

# ==============================
# FEATURES
# ==============================
features = [
    "time",
    "num_trucks",
    "mesh_links",
    "total_power",
    "lag_1", "lag_2", "lag_3"
]

target = "battery_pct"

X_train = train_df[features].values
y_train = train_df[target].values

X_test = test_df[features].values
y_test = test_df[target].values

# ==============================
# SCALE (IMPORTANT)
# ==============================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# GPR MODEL (IMPROVED KERNEL 🔥)
# ==============================
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel

kernel = (
    ConstantKernel(1.0)
    * Matern(length_scale=5.0, nu=1.5)   # better than RBF for real-world data
    + WhiteKernel(noise_level=1)
)

model = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=5,
    random_state=42
)

print("Using Improved GPR ✅")

# ==============================
# TRAIN
# ==============================
model.fit(X_train, y_train)

# ==============================
# DIRECT PREDICTION (NO RECURSION 🔥)
# ==============================
y_pred, y_std = model.predict(X_test, return_std=True)

# Clip realistic values
y_pred = np.clip(y_pred, 0, 100)

# ==============================
# EVALUATION
# ==============================
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)

print("\n==============================")
print("TEST TOWER:", TEST_TOWER)
print("Mean Absolute Error:", round(mae, 4))
print("==============================")

# ==============================
# PLOT
# ==============================
n = min(200, len(y_pred))
x_axis = test_df["time"].values[:n]

plt.figure(figsize=(12,5))

plt.plot(x_axis, y_test[:n], label="Actual Battery")
plt.plot(x_axis, y_pred[:n], label="Predicted Battery")

plt.fill_between(
    x_axis,
    y_pred[:n] - 2*y_std[:n],
    y_pred[:n] + 2*y_std[:n],
    alpha=0.3,
    label="Confidence Interval"
)

plt.title(f"Improved GPR on {TEST_TOWER}")
plt.xlabel("Time")
plt.ylabel("Battery %")
plt.legend()
plt.grid()

plt.show()