import numpy as np

x = np.array([67, 89, 90, 92, 95, 95, 97, 99, 100, 120, 134, 45, 124])
y = np.array([113, 154, 174, 177, 186, 180, 193, 195, 204, 235, 274, 97, 240])

n = len(x)

numerator = n * np.sum(x * y) - np.sum(x) * np.sum(y)
denominator = n * np.sum(x ** 2) - np.sum(x) ** 2

a = numerator / denominator
b = (np.sum(y) - a * np.sum(x)) / n

a_rounded = round(a, 2)
b_rounded = round(b, 2)

print(f"线性回归方程系数：a = {a_rounded}, b = {b_rounded}")
print(f"回归方程为：y = {a_rounded}x + {b_rounded}")