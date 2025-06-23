import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt


# 1. Load and prepare data
df = pd.read_csv("data/household_power_consumption.csv", sep=';', na_values=["?"], low_memory=False)

df["DateTime"] = pd.to_datetime(df["Date"] + ' ' + df["Time"], dayfirst=True)
df = df.drop(columns=["Date", "Time"])
df = df.dropna(subset=["Global_active_power"])
df["Global_active_power"] = df["Global_active_power"].astype(float)

features = ["Global_active_power", "Voltage", "Global_intensity"]
df_features = df.dropna(subset=features).head(20000)

scaler = StandardScaler()
X = scaler.fit_transform(df_features[features])
# X = scaler.fit_transform(data)

# 2. Hidden Markov Model 
print("=== Початок Hidden Markov Model ===")
hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
hmm.fit(X)
# hmm.fit(X_subset)
print("=== Кінець Hidden Markov Model ===")

hidden_states = hmm.predict(X)


# 3. K-Means
print("=== Початок K-Means ===")
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)
# kmeans_labels = kmeans.fit_predict(X_subset)
print("=== Кінець K-Means ===")


# 4. DBSCAN 
print("=== Початок DBSCAN ===")
dbscan = DBSCAN(eps=0.25, min_samples=5, metric='manhattan')
dbscan_labels = dbscan.fit_predict(X)
# dbscan_labels = dbscan.fit_predict(X_subset)
print("=== Кінець DBSCAN ===")


# 5. Results
plt.figure(figsize=(15, 4))

plt.subplot(4, 1, 1)
plt.plot(df[["Global_active_power"]][:1500], label="Енергоспоживання", color='black')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(hidden_states[:1500], label="HMM", color='blue')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(kmeans_labels[:1500], label="KMeans", color='green')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(dbscan_labels[:1500], label="DBSCAN", color='red')
plt.legend()

plt.tight_layout()
plt.show()