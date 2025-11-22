import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

np.random.seed(42)  # Set random seed for reproducibility

# 2. Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# View dataset information
print("Dataset feature dimensions:", X.shape)
print("Feature names:", iris.feature_names)
print("Target classes:", iris.target_names)

# Create DataFrame for better data visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in y]

# View first few rows of dataset
print("\nDataset preview:")
print(df.head())

# Data statistical information
print("\nData statistical information:")
print(df.describe())

# 3. Extract first two features for analysis
X_2d = X[:, :2]  # Only take first two dimensions: sepal length and sepal width

# Visualize original data
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Iris Dataset - True Classes')

# 4. Perform clustering using K-means algorithm
# First, we need to find the optimal number of clusters (K)

# Calculate inertia and silhouette score for different K values
inertia = []
silhouette = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_2d)
    inertia.append(kmeans.inertia_)
    if k > 1:  # Silhouette score requires at least 2 clusters
        silhouette.append(silhouette_score(X_2d, kmeans.labels_))

# Visualize the Elbow Method
plt.subplot(1, 2, 2)
plt.plot(k_range, inertia, 'o-', linewidth=2)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualize silhouette scores
plt.figure(figsize=(10, 5))
plt.plot(k_range[:-1], silhouette, 'o-', linewidth=2)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different k Values')
plt.grid(True)
plt.show()

# Find the best k value based on silhouette score
best_k = k_range[np.argmax(silhouette)]
print(f"Based on silhouette score, the optimal number of clusters is: {best_k}")

# Perform K-means clustering with the best k value
kmeans_best = KMeans(n_clusters=best_k, random_state=42)
kmeans_best.fit(X_2d)
y_kmeans = kmeans_best.labels_

# 5. Visualize clustering results
plt.figure(figsize=(12, 10))

# Original data with true labels
plt.subplot(2, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Iris Dataset - True Classes')

# K-means clustering results
plt.subplot(2, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', edgecolor='k', s=50)
centers = kmeans_best.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title(f'K-means Clustering Results (k={best_k})')
plt.legend()

# Compare clustering results with different random states
plt.subplot(2, 2, 3)
kmeans_diff1 = KMeans(n_clusters=best_k, random_state=10)
kmeans_diff1.fit(X_2d)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_diff1.labels_, cmap='viridis', edgecolor='k', s=50)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Results with Different Random Seed (seed=10)')

plt.subplot(2, 2, 4)
kmeans_diff2 = KMeans(n_clusters=best_k, random_state=100)
kmeans_diff2.fit(X_2d)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_diff2.labels_, cmap='viridis', edgecolor='k', s=50)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Results with Different Random Seed (seed=100)')

plt.tight_layout()
plt.show()

# Compare with standardized data
scaler = StandardScaler()
X_2d_scaled = scaler.fit_transform(X_2d)

# Run K-means on standardized data
kmeans_scaled = KMeans(n_clusters=best_k, random_state=42)
kmeans_scaled.fit(X_2d_scaled)
y_kmeans_scaled = kmeans_scaled.labels_

# Visualize clustering results on standardized data
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_2d_scaled[:, 0], X_2d_scaled[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.xlabel('Standardized ' + iris.feature_names[0])
plt.ylabel('Standardized ' + iris.feature_names[1])
plt.title('True Classes on Standardized Data')

plt.subplot(1, 2, 2)
plt.scatter(X_2d_scaled[:, 0], X_2d_scaled[:, 1], c=y_kmeans_scaled, cmap='viridis', edgecolor='k', s=50)
centers_scaled = kmeans_scaled.cluster_centers_
plt.scatter(centers_scaled[:, 0], centers_scaled[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
plt.xlabel('Standardized ' + iris.feature_names[0])
plt.ylabel('Standardized ' + iris.feature_names[1])
plt.title('K-means Clustering Results on Standardized Data')
plt.legend()

plt.tight_layout()
plt.show()

# Calculate clustering accuracy (by comparing with true labels)
from sklearn.metrics import adjusted_rand_score

print(f"\nAdjusted Rand Index for non-standardized data: {adjusted_rand_score(y, y_kmeans):.4f}")
print(f"Adjusted Rand Index for standardized data: {adjusted_rand_score(y, y_kmeans_scaled):.4f}")