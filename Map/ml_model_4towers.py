import pandas as pd
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

# ==============================
# CONFIG
# ==============================
TRAIN_TOWERS = ["T0", "T1", "T2", "T3"]
TEST_TOWER = "T7"

train_df = load_data(TRAIN_TOWERS)
test_df = load_data([TEST_TOWER])

# ==============================
# SORT BY TIME (IMPORTANT)
# ==============================
train_df = train_df.sort_values("time")
test_df = test_df.sort_values("time")

# ==============================
# ADD TEMPORAL FEATURE (CRITICAL)
# ==============================
train_df["prev_battery"] = train_df["battery_pct"].shift(1)
test_df["prev_battery"] = test_df["battery_pct"].shift(1)

train_df = train_df.dropna()
test_df = test_df.dropna()

# ==============================
# FEATURES & TARGET
# ==============================
features = [
    "time",
    "prev_battery",
    "num_trucks",
    "mesh_links",
    "total_power"
]

target = "battery_pct"

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

# ==============================
# MODEL (TRY XGBOOST FIRST)
# ==============================
try:
    from xgboost import XGBRegressor
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    print("Using XGBoost ✅")

except:
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    print("Using RandomForest (fallback) ⚠️")

# ==============================
# TRAIN
# ==============================
model.fit(X_train, y_train)

# ==============================
# PREDICT (SEQUENTIAL FIX)
# ==============================
y_pred = []

prev = test_df.iloc[0]["prev_battery"]

for i in range(len(X_test)):
    row = X_test.iloc[i].copy()
    row["prev_battery"] = prev

    pred = model.predict([row])[0]
    y_pred.append(pred)

    prev = pred  # feedback loop (IMPORTANT)

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
# VISUALIZATION
# ==============================
plt.figure(figsize=(12,5))

plt.plot(y_test.values[:200], label="Actual Battery")
plt.plot(y_pred[:200], label="Predicted Battery")

plt.title(f"Improved Battery Prediction on {TEST_TOWER}")
plt.xlabel("Time Steps")
plt.ylabel("Battery %")
plt.legend()
plt.grid()

plt.show()