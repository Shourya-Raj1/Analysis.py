
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sns.set_style("whitegrid")

df = pd.read_csv("titanic_cleaned.csv")

features = df[["Age","Fare","FamilySize"]]
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(scaled)

print(df["Cluster"].value_counts())

plt.figure(figsize=(6,4))
sns.scatterplot(data=df,x="Age",y="Fare",hue="Cluster",palette="Set2")
plt.title("K-Means Clusters: Age vs Fare")
plt.tight_layout()
plt.savefig("cluster_scatter.png")
plt.close()

plt.figure(figsize=(6,4))
sns.boxplot(data=df,x="Cluster",y="Fare",palette="Set2")
plt.title("Fare Distribution by Cluster")
plt.tight_layout()
plt.savefig("cluster_boxplot.png")
plt.close()

plt.figure(figsize=(5,4))
df["Cluster"].value_counts().sort_index().plot(kind="bar")
plt.title("Passengers in Each Cluster")
plt.tight_layout()
plt.savefig("cluster_count.png")
plt.close()

print("Week 3 clustering completed.")
