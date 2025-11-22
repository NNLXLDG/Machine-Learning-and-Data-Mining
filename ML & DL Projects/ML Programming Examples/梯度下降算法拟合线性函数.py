import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 随机生成数据
np.random.seed(42)
x = np.random.rand(100)  # 均匀分布生成100个随机数
true_slope = 2.5        # 假设真实斜率是2.5
noise = np.random.normal(0, 0.2, 100)  # 生成高斯噪声
y = true_slope * x + noise  # 得到y = 2.5x + 噪声

# 梯度下降法拟合
def gradient_descent(x, y, learning_rate=0.1, n_iterations=1000):
    m = 0  # 初始斜率
    c = 0  # 初始截距
    N = len(y)

    for _ in range(n_iterations):
        y_pred = m * x + c
        gradient_m = -(2/N) * np.sum((y - y_pred) * x)
        gradient_c = -(2/N) * np.sum(y - y_pred)
        m -= learning_rate * gradient_m
        c -= learning_rate * gradient_c

    return m, c

# 使用梯度下降法拟合数据
slope, intercept = gradient_descent(x, y)

# 将参数四舍五入保留2位小数
slope_rounded = round(slope, 2)
intercept_rounded = round(intercept, 2)

print(f"拟合的斜率: {slope_rounded}")
print(f"拟合的截距: {intercept_rounded}")

# 预测值
y_pred = slope * x + intercept

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y)), y, label='True Values', color='b')
plt.plot(range(len(y)), y_pred, label='Predicted Values', color='r')
plt.xlabel('Sample Index')
plt.ylabel('y - value')
plt.title('Fitting Results of Gradient Descent Method')
plt.legend()
plt.show()


rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)
print(f'Root - Mean - Squared Error (RMSE): {rmse:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')