import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform

# 加载数据集
boston = fetch_openml(name='boston', version=1, as_frame=False)
X, y = boston.data.astype(np.float64), boston.target.astype(np.float64)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义模型
elastic_net = ElasticNet()

# 定义参数分布进行随机搜索
param_dist = {
    'alpha': uniform(0.0001, 10),       # alpha从0.1到10的均匀分布
    'l1_ratio': uniform(0, 1)        # l1_ratio从0到1的均匀分布
}

# 使用随机搜索优化参数
random_search = RandomizedSearchCV(elastic_net, param_distributions=param_dist, n_iter=100, cv=5, scoring='neg_mean_squared_error', random_state=42)
random_search.fit(X_train_scaled, y_train)

# 得到最优参数的ElasticNet模型
best_elastic_net = random_search.best_estimator_

# 在测试集上预测
y_pred = best_elastic_net.predict(X_test_scaled)

# 四舍五入保留2位小数
y_pred_rounded = np.round(y_pred, 2)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# 打印最优参数
print("Best Parameters:", random_search.best_params_)

# 绘图比较预测值和真实值
plt.figure(figsize=(14, 7))
plt.plot(range(len(y_test)), y_test, label='True Values', color='blue', marker='o')
plt.plot(range(len(y_pred_rounded)), y_pred_rounded, label='Predicted Values', color='red', linestyle='--', marker='x')
plt.xlabel('Sample Point Index')
plt.ylabel('House Price')
plt.title('Boston Housing Price Prediction: True Values vs Predicted Values')
plt.legend()
plt.show()







