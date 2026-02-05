import math
import random
import numpy as np
from sklearn.datasets import load_iris

# =========== 工具函数：度量计算 ===========

def entropy(labels):
    """
    计算标签列表的熵
    """
    if len(labels) == 0:
        return 0
    label_counts = {}
    for lbl in labels:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    ent = 0.0
    total = len(labels)
    for lbl, count in label_counts.items():
        p = count / total
        ent -= p * math.log2(p)
    return ent

def gini(labels):
    """
    计算标签列表的Gini指数
    """
    if len(labels) == 0:
        return 0
    label_counts = {}
    for lbl in labels:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    total = len(labels)
    g = 1.0
    for lbl, count in label_counts.items():
        p = count / total
        g -= p*p
    return g

def split_dataset(X, y, feature_index, threshold):
    """
    根据给定特征和阈值进行二分分割，
    返回 (X_left, y_left), (X_right, y_right)
    """
    X_left, y_left = [], []
    X_right, y_right = [], []
    for i, sample in enumerate(X):
        if sample[feature_index] <= threshold:
            X_left.append(sample)
            y_left.append(y[i])
        else:
            X_right.append(sample)
            y_right.append(y[i])
    return (np.array(X_left), np.array(y_left)), (np.array(X_right), np.array(y_right))


# =========== 决策树节点类 ===========

class TreeNode:
    def __init__(self,
                 feature_index=None,
                 threshold=None,
                 left=None,
                 right=None,
                 leaf=False,
                 class_label=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.leaf = leaf
        self.class_label = class_label


# =========== 手写决策树实现 ===========

class DecisionTree:
    """
    手动实现 ID3, C4.5, CART 三种模式
    criterion='id3'  -> 使用信息增益
    criterion='c4.5' -> 使用信息增益比
    criterion='cart' -> 使用gini指数
    """
    def __init__(self, criterion='id3', max_depth=None, min_samples_split=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        """
        构造决策树
        """
        self.root = self._build_tree(X, y, current_depth=0)

    def _build_tree(self, X, y, current_depth):
        # 若全部样本属于同一类别，或已到达最大深度，或样本数小于阈值 -> 直接叶子节点
        if len(set(y)) == 1 \
           or (self.max_depth is not None and current_depth >= self.max_depth) \
           or (len(y) < self.min_samples_split):
            return TreeNode(leaf=True, class_label=self._majority_class(y))

        # 在当前层选择最优特征与阈值
        best_feature, best_threshold = self._choose_best_split(X, y)

        if best_feature is None:
            # 找不到有效划分，说明可能所有样本都在此处特征值相同
            return TreeNode(leaf=True, class_label=self._majority_class(y))

        # 构建左右子树
        (X_left, y_left), (X_right, y_right) = split_dataset(X, y, best_feature, best_threshold)

        # 如果分割后有一侧为空，则也直接叶子处理
        if len(y_left) == 0 or len(y_right) == 0:
            return TreeNode(leaf=True, class_label=self._majority_class(y))

        left_child = self._build_tree(X_left, y_left, current_depth + 1)
        right_child = self._build_tree(X_right, y_right, current_depth + 1)

        return TreeNode(feature_index=best_feature,
                        threshold=best_threshold,
                        left=left_child,
                        right=right_child,
                        leaf=False)

    def _choose_best_split(self, X, y):
        """
        不同criterion下，选择最优特征与阈值
        对于连续特征，尝试不同的划分点
        """
        best_feature, best_threshold = None, None
        # 对应 ID3/C4.5，用“最大”度量选特征；对 CART，则用“最小”度量
        best_metric = -float('inf') if self.criterion in ('id3', 'c4.5') else float('inf')

        # 父节点度量
        if self.criterion in ('id3', 'c4.5'):
            parent_entropy = entropy(y)
        else:
            parent_gini = gini(y)

        parent_size = len(y)  # 用于计算加权系数

        n_features = X.shape[1]

        for feature_idx in range(n_features):
            # 取该特征所有值并排序，防止重复
            col_values = X[:, feature_idx]
            unique_vals = np.unique(col_values)
            if len(unique_vals) == 1:
                # 该特征只有一个值，无法再进行划分
                continue

            # 针对连续值，用相邻值的均值做候选划分点
            candidate_thresholds = []
            for i in range(len(unique_vals)-1):
                candidate_thresholds.append((unique_vals[i] + unique_vals[i+1]) / 2.0)

            for threshold in candidate_thresholds:
                (X_left, y_left), (X_right, y_right) = split_dataset(X, y, feature_idx, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    # 无效划分：一侧无样本
                    continue

                if self.criterion in ('id3', 'c4.5'):
                    left_entropy = entropy(y_left)
                    right_entropy = entropy(y_right)
                    w_left = len(y_left) / parent_size
                    w_right = len(y_right) / parent_size
                    info_gain = parent_entropy - (w_left * left_entropy + w_right * right_entropy)

                    if self.criterion == 'id3':
                        metric = info_gain
                    else:
                        # c4.5: 计算增益比
                        split_info = 0.0
                        if w_left > 0:
                            split_info -= w_left * math.log2(w_left)
                        if w_right > 0:
                            split_info -= w_right * math.log2(w_right)
                        if split_info == 0:
                            # 避免除0
                            continue
                        gain_ratio = info_gain / split_info
                        metric = gain_ratio

                    # 选 metric 最大的
                    if metric > best_metric:
                        best_metric = metric
                        best_feature = feature_idx
                        best_threshold = threshold

                else:  # CART -> GINI
                    left_g = gini(y_left)
                    right_g = gini(y_right)
                    w_left = len(y_left) / parent_size
                    w_right = len(y_right) / parent_size
                    cart_metric = w_left * left_g + w_right * right_g
                    # 选 gini 最小
                    if cart_metric < best_metric:
                        best_metric = cart_metric
                        best_feature = feature_idx
                        best_threshold = threshold

        return best_feature, best_threshold

    def _majority_class(self, y):
        """
        多数表决
        """
        label_counts = {}
        for lbl in y:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        return max(label_counts.keys(), key=lambda k: label_counts[k])

    def predict_one(self, x):
        """
        对单个样本进行预测
        """
        node = self.root
        while not node.leaf:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.class_label

    def predict(self, X):
        return [self.predict_one(sample) for sample in X]


def evaluate_once(X, y, test_ratio=0.2, criterion='id3'):
    """
    随机打乱数据集并基于给定criterion训练决策树，返回在测试集上的准确率
    """
    n_samples = len(y)
    indices = list(range(n_samples))
    # 这里不设seed，即每次运行都会随机打乱
    random.shuffle(indices)

    split_idx = int(n_samples * (1 - test_ratio))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    dt = DecisionTree(criterion=criterion)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    accuracy = np.mean(np.array(y_pred) == y_test)
    return accuracy


def main():
    # 1. 加载数据
    iris = load_iris()
    X, y = iris.data, iris.target

    # 2. 多次随机打乱测试
    N = 1000  # 重复次数
    criteria = ['id3', 'c4.5', 'cart']
    results = {
        'id3': [],
        'c4.5': [],
        'cart': []
    }

    for i in range(N):
        for c in criteria:
            acc = evaluate_once(X, y, test_ratio=0.2, criterion=c)
            results[c].append(acc)

    # 3. 计算平均准确率
    for c in criteria:
        mean_acc = np.mean(results[c])
        print(f"{c} 在随机打乱测试 {N} 次后的平均准确率： {mean_acc:.4f}")

if __name__ == "__main__":
    main()
