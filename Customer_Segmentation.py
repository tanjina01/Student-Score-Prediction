# ---------------------------------------------------------
# Customer Segmentation using K-Means and DBSCAN
# ---------------------------------------------------------

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN

# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------
df = pd.read_csv("Mall_Customers.csv")
print("✅ Dataset Loaded Successfully!\n")
print(df.head())

# ---------------------------------------------------------
# 2. Basic Info
# ---------------------------------------------------------
print("\n📊 Dataset Info:")
print(df.info())

print("\n📈 Statistical Summary:")
print(df.describe())

# ---------------------------------------------------------
# 3. Select Features for Clustering
# ---------------------------------------------------------
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# ---------------------------------------------------------
# 4. Scale the Features
# ---------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n✅ Data Scaled Successfully!\n")

# ---------------------------------------------------------
# 5. Visualize the Raw Data
# ---------------------------------------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df)
plt.title('Customer Income vs Spending Score (Before Clustering)')
plt.show()

# ---------------------------------------------------------
# 6. Determine Optimal Number of Clusters using Elbow Method
# ---------------------------------------------------------
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,6))
plt.plot(range(1, 11), wcss, marker='o', color='teal')
plt.title('Elbow Method to Determine Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.show()

# ---------------------------------------------------------
# 7. Apply K-Means Clustering (choose k=5 based on elbow)
# ---------------------------------------------------------
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

print("\n✅ K-Means Clustering Applied!\n")
print(df.head())

# ---------------------------------------------------------
# 8. Visualize Clusters
# ---------------------------------------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Cluster', palette='Set2', data=df)
plt.title('Customer Segments (K-Means Clustering)')
plt.legend(title='Cluster')
plt.show()

# ---------------------------------------------------------
# 9. Analyze Average Spending and Income per Cluster
# ---------------------------------------------------------
cluster_summary = df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
print("\n📊 Average Income and Spending per Cluster:")
print(cluster_summary)

# ---------------------------------------------------------
# 10. Bonus: Try DBSCAN Clustering
# ---------------------------------------------------------
dbscan = DBSCAN(eps=1.5, min_samples=5)
db_clusters = dbscan.fit_predict(X_scaled)
df['DBSCAN_Cluster'] = db_clusters

print("\n✅ DBSCAN Clustering Applied!\n")
print(df['DBSCAN_Cluster'].value_counts())

plt.figure(figsize=(8,6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='DBSCAN_Cluster', palette='Set1', data=df)
plt.title('Customer Segments (DBSCAN Clustering)')
plt.legend(title='Cluster')
plt.show()

# ---------------------------------------------------------
# 11. Save Results
# ---------------------------------------------------------
df.to_csv("Clustered_Customers.csv", index=False)
print("\n💾 Clustered data saved as 'Clustered_Customers.csv'!")
