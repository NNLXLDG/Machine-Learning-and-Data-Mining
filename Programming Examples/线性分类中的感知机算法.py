from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 加载乳腺癌数据集
data = load_breast_cancer()
X = data.data
y = data.target

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 感知机手动实现
class MyPerceptron:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None

    def _activation_function(self, x):
        # 激活函数：大于等于0输出1，否则输出0
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # 迭代进行模型训练
        for _ in range(self.max_iter):
            for idx, x_i in enumerate(X):
                # 计算线性组合 wx + b
                linear_output = np.dot(x_i, self.weights) + self.bias
                # 应用激活函数
                y_pred = self._activation_function(linear_output)
                # 如果预测错误，则更新权重和偏置
                if y_pred != y[idx]:
                    update = self.learning_rate * (y[idx] - y_pred)
                    self.weights += update * x_i
                    self.bias += update

    def predict(self, X):
        # 返回预测结果
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self._activation_function(linear_output)
        return y_pred


# 使用手动实现的感知机
model = MyPerceptron(learning_rate=0.01, max_iter=1000)
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 生成混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

# 计算准确度和召回率
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f'Model accuracy: {accuracy:.2f}')
print(f'Model recall: {recall:.2f}')

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion matrix of the manually implemented perceptron model')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()