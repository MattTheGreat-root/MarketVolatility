import pandas as pd

df = pd.read_csv("AAPL_2014_2024.csv", index_col=0, parse_dates=True)
df["Return"] = df["Close"].pct_change()
df["Vol_5"] = df["Return"].rolling(5).std()
df["Vol_10"] = df["Return"].rolling(10).std()
df["Vol_20"] = df["Return"].rolling(20).std()
df["TargetVol"] = df["Vol_5"].shift(-1)
df = df.dropna()

# Features (X) and target (y)
X = df[["Return", "Vol_5", "Vol_10", "Vol_20"]]
y = df["TargetVol"]

# Time-series split
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]