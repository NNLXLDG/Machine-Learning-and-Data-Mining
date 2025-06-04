import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# ---------- 1. One-hot 编码 ----------
def one_hot_encode(y, num_classes=3):
    one_hot = np.zeros((y.shape[0], num_classes))
    for idx, val in enumerate(y):
        one_hot[idx, val] = 1.0
    return one_hot


# ---------- 2. 激活函数及工具 ----------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(a):
    return a * (1.0 - a)


def softmax(z):
    shift_z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shift_z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def initialize_parameters(input_dim, hidden_dim, output_dim=3):
    W1 = np.random.randn(input_dim, hidden_dim) * 0.01
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, output_dim) * 0.01
    b2 = np.zeros((1, output_dim))
    return W1, b1, W2, b2


def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def backward_propagation(X, Y, Z1, A1, A2, W1, W2):
    n_samples = X.shape[0]
    # 输出层误差
    dZ2 = A2 - Y  # (n_samples, 3)
    dW2 = np.dot(A1.T, dZ2) / n_samples
    db2 = np.sum(dZ2, axis=0, keepdims=True) / n_samples

    # 隐藏层误差
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)

    dW1 = np.dot(X.T, dZ1) / n_samples
    db1 = np.sum(dZ1, axis=0, keepdims=True) / n_samples

    return dW1, db1, dW2, db2


def train_nn(X_train, Y_train, hidden_dim=10, epochs=1000, lr=0.1):
    n_samples, input_dim = X_train.shape
    _, output_dim = Y_train.shape

    # 初始化参数
    W1, b1, W2, b2 = initialize_parameters(input_dim, hidden_dim, output_dim)

    for i in range(epochs):
        # 前向传播
        Z1, A1, Z2, A2 = forward_propagation(X_train, W1, b1, W2, b2)

        # 计算损失
        loss = -np.mean(np.sum(Y_train * np.log(A2 + 1e-15), axis=1))

        # 反向传播
        dW1, db1, dW2, db2 = backward_propagation(X_train, Y_train, Z1, A1, A2, W1, W2)

        # 更新
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        # 每隔若干次打印
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Epoch [{i + 1}/{epochs}] Loss: {loss:.4f}")

    return W1, b1, W2, b2


def predict(X, W1, b1, W2, b2):
    _, A1, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    return np.argmax(A2, axis=1)


def confusion_matrix_manual(y_true, y_pred, num_classes=3):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


if __name__ == "__main__":
    # ---------- 3. 加载数据并拆分 ----------
    iris = load_iris()
    X = iris.data  # (150, 4)
    y = iris.target  # (150,)

    # 拆分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # One-hot 编码
    Y_train = one_hot_encode(y_train, num_classes=3)

    # ---------- 4. 训练神经网络 ----------
    # 你可以调节 hidden_dim, epochs, lr 等超参数
    W1, b1, W2, b2 = train_nn(X_train, Y_train,
                              hidden_dim=10,
                              epochs=1000,
                              lr=0.1)

    # ---------- 5. 在测试集上评估 ----------
    y_pred_test = predict(X_test, W1, b1, W2, b2)

    cm = confusion_matrix_manual(y_test, y_pred_test, num_classes=3)
    print("混淆矩阵：\n", cm)

    # 计算准确率
    accuracy = np.mean(y_pred_test == y_test)
    print(f"测试集准确率: {accuracy * 100:.2f}%")

    # 可视化混淆矩阵
    plot_confusion_matrix(cm, iris.target_names, title="Iris Confusion Matrix (Manual BP)")
