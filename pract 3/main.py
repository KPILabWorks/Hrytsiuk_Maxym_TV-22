import pandas as pd
import time

# 1. Prepare Data
df = pd.read_csv("data/household_power_consumption.csv", sep=';', 
                 na_values=['?'], low_memory=False)

df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)

df = df.dropna(subset=["Global_active_power"])
df["Global_active_power"] = df["Global_active_power"].astype(float)

df["Year"] = df["DateTime"].dt.year
df["Month"] = df["DateTime"].dt.month
df["Day"] = df["DateTime"].dt.day

# 2. Aggregation 
def timed_aggregation(column, group_cols):
    start = time.time()
    agg = df.groupby(group_cols)[column].sum().reset_index()
    duration = time.time() - start
    return agg, duration


groupings = {
    "by_day": ["Year", "Month", "Day"],
    "by_month": ["Year", "Month"],
    "by_year": ["Year"]
}

results = {}

for col in ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]:
    results[col] = {}
    for label, group_cols in groupings.items():
        agg, duration = timed_aggregation(col, group_cols)
        results[col][label] = duration

# 3. Results
for col, timings in results.items():
    print(f"\nЧас агрегації для {col}:")
    for level, dur in timings.items():
        print(f"- {level}: {dur:.6f} сек")
