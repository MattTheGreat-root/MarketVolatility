import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("AAPL_2014_2024.csv", index_col=0, parse_dates=True)
df["Return"] = df["Close"].pct_change()
df["Vol_5"] = df["Return"].rolling(5).std()
df["Vol_10"] = df["Return"].rolling(10).std()
df["Vol_20"] = df["Return"].rolling(20).std()
df["TargetVol"] = df["Vol_5"].shift(-1)
df = df.dropna()

X = df[["Return", "Vol_5", "Vol_10", "Vol_20"]]
y = df["TargetVol"]

#time-series split
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

test_df = df.iloc[split:]

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
print("MAE:", mae)

# Baseline comparison
naive_pred = test_df["Vol_5"].shift(1)
naive_mae = mean_absolute_error(test_df["TargetVol"][1:], naive_pred[1:])
print("Naive MAE:", naive_mae)

rolling_pred = test_df["Vol_5"].rolling(5).mean()
roll_mae = mean_absolute_error(test_df["TargetVol"][4:], rolling_pred[4:])
print("Rolling Mean MAE:", roll_mae)

fig, axs = plt.subplots(3, 1, figsize=(10, 15))

#Feature Importance
importances = model.feature_importances_
features = X_train.columns
axs[0].barh(features, importances)
axs[0].set_title("Feature Importance")

# Actual vs Predicted Volatility
axs[1].plot(test_df["TargetVol"].values, label="Actual Volatility")
axs[1].plot(preds, label="Predicted Volatility")
axs[1].legend()
axs[1].set_title("Actual vs Predicted Volatility")

# Prediction Error Distribution
errors = test_df["TargetVol"].values - preds
axs[2].hist(errors, bins=50)
axs[2].set_title("Prediction Error Distribution")

plt.tight_layout()
plt.show()
