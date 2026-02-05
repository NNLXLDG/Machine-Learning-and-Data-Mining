import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载数据集
boston = fetch_openml(name='boston', version=1)
X, y = boston.data.astype(np.float64), boston.target.astype(np.float64)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化（使用sklearn预处理）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 添加偏置项（全1列）
X_train_scaled = np.hstack([X_train_scaled, np.ones((X_train_scaled.shape[0], 1))])
X_test_scaled = np.hstack([X_test_scaled, np.ones((X_test_scaled.shape[0], 1))])


# 手动实现Ridge回归权重计算
def manual_ridge(X, y, alpha=1.0):

    n_features = X.shape[1]

    # 创建正则化矩阵（排除偏置项）
    I = np.eye(n_features)
    I[-1, -1] = 0  # 最后一个特征（偏置项）不参与正则化

    # 闭式解计算
    XTX = X.T.dot(X)
    regularized_XTX = XTX + alpha * I
    w = np.linalg.pinv(regularized_XTX).dot(X.T).dot(y)

    return w

# 训练模型
alpha = 1.0  #alpha : 正则化强度
w = manual_ridge(X_train_scaled, y_train, alpha)

# 预测函数
def manual_predict(X, weights):
    return X.dot(weights)


# 进行预测
y_pred = manual_predict(X_test_scaled, w)
y_pred_rounded = np.round(y_pred, 2)

# 可视化对比
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, 'b-', label='True Values', marker='o', markersize=5)
plt.plot(y_pred_rounded, 'r--', label='Predicted Values', marker='x', markersize=5)
plt.title('Manual Ridge Regression (Core Algorithm Only)')
plt.xlabel('Sample Index')
plt.ylabel('House Price')
plt.legend()
plt.grid(True)
plt.show()

# 打印权重对比（与sklearn实现对比）
from sklearn.linear_model import Ridge

# sklearn实现用于验证
sk_model = Ridge(alpha=alpha, fit_intercept=False)  # 已手动添加偏置项
sk_model.fit(X_train_scaled, y_train)

print("\n权重对比：")
print("特征名称:", list(boston.feature_names) + ['Bias'])
print("手动实现权重:", np.round(w, 4))
print("Sklearn权重: ", np.round(sk_model.coef_, 4))




