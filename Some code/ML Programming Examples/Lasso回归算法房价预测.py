import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 加载数据集
boston = fetch_openml(name='boston', version=1)
X, y = boston.data.astype(np.float64), boston.target.astype(np.float64)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 自定义 Lasso 回归
# 这个自定义的 Lasso 回归类 CustomLasso 被设计来模拟 sklearn 的 Lasso 回归功能。
# 它通过手动实现梯度下降算法来优化模型参数，并在损失函数中加入 L1 正则化项。
# 自定义 Lasso 回归
class CustomLasso:
    def __init__(self, alpha=0.1, learning_rate=0.01, n_iterations=1000):
        self.alpha = alpha             # L1正则化强度，可以影响特征选择和过拟合程度
        self.learning_rate = learning_rate # 每次参数更新的步长，影响收敛速度
        self.n_iterations = n_iterations   # 模型训练中迭代次数
    def fit(self, X, y):
        # Initialize weights
        self.weights = np.zeros(X.shape[1])  # 初始化权重向量为零
        self.bias = 0                      # 初始化偏置为零
        N = X.shape[0]                     # 样本数量，用于梯度计算
        for _ in range(self.n_iterations):
            y_pred = X.dot(self.weights) + self.bias  # 计算预测值
            residuals = y - y_pred                   # 计算预测残差
            # 最小二乘梯度
            gradient_weights = -(2 / N) * X.T.dot(residuals)  # 线性部分的梯度下降
            gradient_bias = -(2 / N) * np.sum(residuals)      # 偏置的梯度更新
            # L1 正则化
            gradient_weights += self.alpha * np.sign(self.weights)  # 对权重，增加L1正则化影响
            # 更新权重和偏差
            self.weights -= self.learning_rate * gradient_weights   # 更新权重
            self.bias -= self.learning_rate * gradient_bias         # 更新偏置
    def predict(self, X):
        return X.dot(self.weights) + self.bias  # 使用训练后的参数进行预测



# 使用自定义 Lasso 回归
lasso_custom = CustomLasso(alpha=0.01, learning_rate=0.01, n_iterations=1000)
lasso_custom.fit(X_train_scaled, y_train)
# 预测测试集房价，并将预测结果保留2位小数
y_pred = lasso_custom.predict(X_test_scaled)
y_pred_rounded = np.round(y_pred, 2)


# 预测值与真实值的对比图部分
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, label='True Value', linewidth=2)
plt.plot(range(len(y_test)), y_pred_rounded, label='Predicted Value', linewidth=2)
plt.xlabel('Sample Index')
plt.ylabel('House Price')
plt.title('Lasso Regression Prediction vs True House Price')
plt.legend()
plt.show()

# 模型评估
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f'Root - Mean - Squared Error (RMSE): {rmse:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')

