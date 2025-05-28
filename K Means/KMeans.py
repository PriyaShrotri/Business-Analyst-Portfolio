import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the Wine dataset
wine = load_wine()
X = wine.data[:, :2]  # Use the first two features for visualization (Alcohol, Malic acid)

# Visualize the original data (unlabeled)
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c='gray')
plt.xlabel('Alcohol')
plt.ylabel('Malic Acid')
plt.title('Wine Dataset: Malic Acid vs Alcohol (Unlabeled)')
plt.show()

# Fit KMeans with 3 clusters (the wine dataset has 3 classes)
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
kmeans.fit(X)

# Print cluster centers
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# Print inertia (sum of squared distances to closest cluster center)
print(f"Inertia: {kmeans.inertia_:.2f}")

# Assign each point to a cluster and visualize the results
labels = kmeans.labels_
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='red', marker='X', label='Centroids')
plt.xlabel('Alcohol')
plt.ylabel('Malic Acid')
plt.title('K-Means Clustering on Wine Dataset')
plt.legend()
plt.show()
