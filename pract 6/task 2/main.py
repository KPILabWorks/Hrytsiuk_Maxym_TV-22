import pandas as pd
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


def count_steps(group):
    signal = group["smoothed_accel"].dropna()
    peaks, _ = find_peaks(signal, height=10, distance=20)
    return len(peaks)


# 1. Load data
df = pd.read_csv("data/60min.csv")

df["Time (s)"] = df["Time (s)"].astype(float)
df["timestamp"] = pd.to_timedelta(df["Time (s)"], unit='s')
df.set_index("timestamp", inplace=True)

df["smoothed_accel"] = df["Absolute acceleration (m/s^2)"].rolling(window=10, center=True).mean()

df["minute"] = (df["Time (s)"] // 60).astype(int)

step_counts = df.drop(columns="minute").groupby(df["minute"]).apply(count_steps)

features = df.groupby("minute")["Absolute acceleration (m/s^2)"].agg(["mean", "std", "max", "min"])
features.columns = ["mean_acc", "std_acc", "max_acc", "min_acc"]
features["steps"] = step_counts.values

print(features.head())


# 2. Set test and train
X = features.drop(columns=["steps"]) 
y = features["steps"]                

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 3. Train and test model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# 4. Viz results
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.xlabel("Minute index")
plt.ylabel("Steps")
plt.legend()
plt.title("Actual vs Predicted Steps")
plt.show()