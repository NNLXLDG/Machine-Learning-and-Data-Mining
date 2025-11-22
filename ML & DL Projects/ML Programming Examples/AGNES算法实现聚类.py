import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
import pandas as pd
import seaborn as sns

# 设置随机种子以确保结果可复现
np.random.seed(42)

# 1 & 2. 加载鸢尾花数据集并理解数据
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 查看数据集信息
print("Dataset feature dimensions:", X.shape)
print("Feature names:", iris.feature_names)
print("Target classes:", iris.target_names)

# 创建一个DataFrame以便更好地查看数据
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in y]

# 查看数据集的前几行
print("\nDataset preview:")
print(df.head())

# 数据统计信息
print("\nData statistical information:")
print(df.describe())

# 3. 提取前两个特征进行分析（萼片长度和萼片宽度）
X_2d = X[:, :2]

# 可视化原始数据
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Iris Dataset - True Classes')
plt.colorbar(label='Species')
plt.show()

# 4. 使用AGNES算法进行聚类
# 首先，我们创建一个树状图来帮助我们确定最佳聚类数

# 计算层次聚类的链接矩阵
Z = linkage(X_2d, method='ward')  # ward方法试图最小化簇内方差

# 绘制树状图
plt.figure(figsize=(12, 5))
plt.title('Hierarchical Clustering Dendrogram of Iris Dataset')
dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
plt.xlabel('Sample Index')
plt.ylabel('Clustering Distance')
plt.axhline(y=5.5, c='k', linestyle='--', label='Suggested Cut-off Line')  # 可根据树状图调整这个值
plt.legend()
plt.tight_layout()
plt.show()

# 使用轮廓系数来确定最佳聚类数量
silhouette_scores = []
for n_clusters in range(2, 11):
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = clusterer.fit_predict(X_2d)
    silhouette_scores.append(silhouette_score(X_2d, cluster_labels))

plt.figure(figsize=(10, 5))
plt.plot(range(2, 11), silhouette_scores, 'o-', linewidth=2)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.grid(True)
plt.show()

# 找出最佳聚类数量
best_n_clusters = np.argmax(silhouette_scores) + 2  # +2是因为range(2, 11)从2开始
print(f"Based on silhouette score, the optimal number of clusters is: {best_n_clusters}")

# 使用最佳聚类数进行层次聚类
# 我们将尝试不同的连接方法: 'ward', 'complete', 'average','single'
linkage_methods = ['ward', 'complete', 'average','single']
plt.figure(figsize=(15, 10))

for i, method in enumerate(linkage_methods):
    # 使用当前连接方法进行聚类
    agnes = AgglomerativeClustering(n_clusters=best_n_clusters, linkage=method)
    y_agnes = agnes.fit_predict(X_2d)

    # 计算轮廓系数以评估聚类质量
    score = silhouette_score(X_2d, y_agnes)

    # 可视化聚类结果
    plt.subplot(2, 2, i + 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_agnes, cmap='viridis', edgecolor='k', s=50)
    plt.title(f'AGNES Clustering - {method} Linkage Method\nSilhouette Score: {score:.4f}')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])

plt.tight_layout()
plt.show()

# 找出表现最好的连接方法
silhouette_by_method = {}
for method in linkage_methods:
    agnes = AgglomerativeClustering(n_clusters=best_n_clusters, linkage=method)
    y_agnes = agnes.fit_predict(X_2d)
    silhouette_by_method[method] = silhouette_score(X_2d, y_agnes)

best_method = max(silhouette_by_method, key=silhouette_by_method.get)
print(f"The best linkage method is '{best_method}' with a silhouette score of {silhouette_by_method[best_method]:.4f}")

# 使用最佳参数进行最终聚类
best_agnes = AgglomerativeClustering(n_clusters=best_n_clusters, linkage=best_method)
y_best_agnes = best_agnes.fit_predict(X_2d)

# 5. 可视化最终聚类结果
plt.figure(figsize=(12, 5))

# 真实标签
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.title('Iris Dataset - True Classes')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

# AGNES聚类结果
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_best_agnes, cmap='viridis', edgecolor='k', s=50)
plt.title(f'AGNES Clustering Results - {best_method} Linkage Method\nn_clusters={best_n_clusters}')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

plt.tight_layout()
plt.show()

# 尝试使用标准化数据
scaler = StandardScaler()
X_2d_scaled = scaler.fit_transform(X_2d)

# 在标准化数据上应用AGNES
agnes_scaled = AgglomerativeClustering(n_clusters=best_n_clusters, linkage=best_method)
y_agnes_scaled = agnes_scaled.fit_predict(X_2d_scaled)

# 可视化标准化后的结果
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_2d_scaled[:, 0], X_2d_scaled[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.title('True Classes on Standardized Data')
plt.xlabel('Standardized ' + iris.feature_names[0])
plt.ylabel('Standardized ' + iris.feature_names[1])

plt.subplot(1, 2, 2)
plt.scatter(X_2d_scaled[:, 0], X_2d_scaled[:, 1], c=y_agnes_scaled, cmap='viridis', edgecolor='k', s=50)
plt.title(f'AGNES Clustering Results on Standardized Data\n{best_method} Linkage Method, n_clusters={best_n_clusters}')
plt.xlabel('Standardized ' + iris.feature_names[0])
plt.ylabel('Standardized ' + iris.feature_names[1])

plt.tight_layout()
plt.show()

# 计算聚类准确度（通过与真实标签比较）
from sklearn.metrics import adjusted_rand_score

print(f"\nAdjusted Rand Index for non-standardized data: {adjusted_rand_score(y, y_best_agnes):.4f}")
print(f"Adjusted Rand Index for standardized data: {adjusted_rand_score(y, y_agnes_scaled):.4f}")

# 检验算法的确定性
print("\nChecking the Determinacy of AGNES Clustering Algorithm:")
results = []
for i in range(3):
    agnes = AgglomerativeClustering(n_clusters=best_n_clusters, linkage=best_method)
    y_pred = agnes.fit_predict(X_2d)
    results.append(y_pred)

print("Are the results of multiple runs the same:", np.array_equal(results[0], results[1]) and np.array_equal(results[1], results[2]))