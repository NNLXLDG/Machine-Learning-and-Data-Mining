import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import silhouette_score


def region_query(X, point_idx, eps):
    """
    返回X中所有与X[point_idx]距离小于等于eps的点的索引
    """
    # 为了提高速度，计算平方距离
    diff = X - X[point_idx]
    dist2 = np.sum(diff ** 2, axis=1)
    return np.where(dist2 <= eps ** 2)[0].tolist()


def expand_cluster(X, labels, point_idx, cluster_id, eps, min_samples, visited):
    """
    从point_idx开始扩展ID为cluster_id的簇
    """
    # 获取种子邻居
    seeds = region_query(X, point_idx, eps)
    if len(seeds) < min_samples:
        # 暂时标记为噪声
        labels[point_idx] = -1
        return False
    else:
        # 为所有种子点分配cluster_id
        labels[seeds] = cluster_id
        # 确保起始点也在簇中
        labels[point_idx] = cluster_id

        # 处理每个种子点
        i = 0
        while i < len(seeds):
            p = seeds[i]
            if not visited[p]:
                visited[p] = True
                p_neighbors = region_query(X, p, eps)
                if len(p_neighbors) >= min_samples:
                    # 追加新邻居
                    for n in p_neighbors:
                        if labels[n] == -1:
                            labels[n] = cluster_id
                        if labels[n] == 0:
                            labels[n] = cluster_id
                            seeds.append(n)
            i += 1
        return True


def dbscan(X, eps=0.5, min_samples=5):
    """
    一个简单的DBSCAN聚类实现
    返回:
      labels: 形状为(n_samples,)的数组
              每个点的簇标签。噪声被标记为 -1
    """
    n_points = X.shape[0]
    labels = np.zeros(n_points, dtype=int)  # 0表示未分类，-1表示噪声
    visited = np.zeros(n_points, dtype=bool)
    cluster_id = 0

    for i in range(n_points):
        if not visited[i]:
            visited[i] = True
            if expand_cluster(X, labels, i, cluster_id + 1, eps, min_samples, visited):
                cluster_id += 1

    # 如果需要，将簇ID从1...k转换为0...k-1，但保持噪声为 -1
    # 这里我们将标签向下移动1，以便簇从0开始
    labels = np.where(labels > 0, labels - 1, labels)
    return labels


# 2. 加载鸢尾花数据（前两个维度）
iris = datasets.load_iris()
X = iris.data[:, :2]  # 仅取花萼长度和花萼宽度
y_true = iris.target

print("特征矩阵形状:", X.shape)  # 期望(150,2)
# 打印真实标签的类别，用于查看数据集中有多少种不同的真实分类
print("真实标签:", np.unique(y_true))  # [0,1,2]

# 3. 对eps和min_samples进行参数搜索
best_score = -1.0
best_params = None

eps_values = np.linspace(0.1, 1.0, 10)
min_samples_values = range(3, 8)

for eps in eps_values:
    for min_pts in min_samples_values:
        labels = dbscan(X, eps=eps, min_samples=min_pts)
        # 计算真实簇的数量（排除噪声 -1）
        unique = set(labels)
        n_clusters = len(unique) - (1 if -1 in unique else 0)
        if n_clusters >= 2:
            try:
                score = silhouette_score(X, labels)
            except ValueError:
                continue
            if score > best_score:
                best_score = score
                best_params = (eps, min_pts, n_clusters, score)

eps_opt, min_samples_opt, n_clusters_opt, best_score = best_params
print(f"找到的最佳参数:")
print(f"  eps = {eps_opt:.3f}")
print(f"  min_samples = {min_samples_opt}")
print(f"  簇的数量 = {n_clusters_opt}")
print(f"  轮廓系数 = {best_score:.4f}")

# 4. 最终聚类和可视化
labels_opt = dbscan(X, eps=eps_opt, min_samples=min_samples_opt)
unique_labels = sorted(set(labels_opt))

plt.figure(figsize=(6, 5))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # 噪声
        marker = 'x'
        col = 'k'
        label_name = 'noise'
    else:
        marker = 'o'
        label_name = f'cluster {k}'
    pts = X[labels_opt == k]
    plt.scatter(pts[:, 0], pts[:, 1], c=[col], marker=marker, s=50, label=label_name)

plt.title(f"DBSCAN  (eps={eps_opt:.3f}, min_samples={min_samples_opt})")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.legend()
plt.tight_layout()
plt.show()
