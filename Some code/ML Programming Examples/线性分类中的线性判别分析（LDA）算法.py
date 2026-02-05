import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns

# 手动实现LDA算法
class LDA:
    def fit(self, X, y):
        self.class_means_ = self._compute_class_means(X, y)
        S_w = self._compute_within_class_scatter(X, y, self.class_means_)
        S_b = self._compute_between_class_scatter(X, y, self.class_means_)
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_w).dot(S_b))
        sorted_indices = np.argsort(abs(eigenvalues))[::-1]
        self.components_ = eigenvectors[:, sorted_indices[:1]]

    def _compute_class_means(self, X, y):
        classes = np.unique(y)
        means = {}
        for c in classes:
            means[c] = X[y == c].mean(axis=0)
        return means

    def _compute_within_class_scatter(self, X, y, class_means):
        n_features = X.shape[1]
        S_w = np.zeros((n_features, n_features))
        for c in np.unique(y):
            X_c = X[y == c]
            mean_c = class_means[c]
            S_w += np.sum([np.outer(x - mean_c, x - mean_c) for x in X_c], axis=0)
        return S_w

    def _compute_between_class_scatter(self, X, y, class_means):
        n_samples = X.shape[0]
        overall_mean = np.mean(X, axis=0)
        n_features = X.shape[1]
        S_b = np.zeros((n_features, n_features))
        for c in np.unique(y):
            n_c = X[y == c].shape[0]
            mean_c = class_means[c]
            mean_diff = (mean_c - overall_mean).reshape(n_features, 1)
            S_b += n_c * mean_diff.dot(mean_diff.T)
        return S_b

    def transform(self, X):
        return X.dot(self.components_)

    def predict(self, X):
        X_lda = self.transform(X)
        threshold = (X_lda.min() + X_lda.max()) / 2
        return (X_lda > threshold).astype(int).flatten()

# 加载数据集并预处理
data = datasets.load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用封装的LDA类
lda = LDA()
lda.fit(X_train_scaled, y_train)
y_pred = lda.predict(X_test_scaled)

# 计算混淆矩阵和性能指标
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
