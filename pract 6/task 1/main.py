import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def label_condition(time):
    if time < 300:
        return "Daylight"
    elif time < 600:
        return "Artificial"
    else:
        return "Combined"


df = pd.read_csv("data/light_data.csv")

if 'Condition' not in df.columns:
    df['Condition'] = df['Time (s)'].apply(label_condition)


df['Local_Time'] = df.groupby('Condition')['Time (s)'].transform(lambda x: x - x.min())
print(df.head())

plt.figure(figsize=(12, 6))
for condition, group in df.groupby('Condition'):
    plt.plot(group['Local_Time'], group['Illuminance (lux)'], label=condition)

plt.xlabel("Час (с)")
plt.ylabel("Рівень освітленості (lux)")
plt.title("Зміна рівня освітленості в різних умовах")
plt.legend()
plt.grid(True)
plt.show()
