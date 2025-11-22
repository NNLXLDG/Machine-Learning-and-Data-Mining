import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import Counter
import seaborn as sns

# Load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Display dataset information
print(f"Dataset shape: {X.shape}")
print(f"Feature names: {feature_names}")
print(f"Target names: {target_names}")


# Custom KNN algorithm
class SimpleKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            # Calculate Euclidean distance
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            # Get labels of these neighbors
            k_nearest_labels = self.y_train[k_indices]
            # Choose most common label
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        return np.array(predictions)


# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train_std = (X_train - X_mean) / X_std
X_test_std = (X_test - X_mean) / X_std

# Test different k values
k_values = [1, 3, 5, 7, 9]
accuracies = []

for k in k_values:
    knn = SimpleKNN(k=k)
    knn.fit(X_train_std, y_train)
    y_pred = knn.predict(X_test_std)
    accuracy = np.mean(y_pred == y_test)
    accuracies.append(accuracy)
    print(f"k = {k}, accuracy = {accuracy:.4f}")

# Find best k value
best_k = k_values[np.argmax(accuracies)]
print(f"\nBest k: {best_k}, accuracy: {max(accuracies):.4f}")

# Use model with best k
best_knn = SimpleKNN(k=best_k)
best_knn.fit(X_train_std, y_train)
y_pred = best_knn.predict(X_test_std)

# Visualization
plt.figure(figsize=(15, 10))

# 1. K values vs accuracy
plt.subplot(2, 2, 1)
plt.plot(k_values, accuracies, marker='o', linestyle='-')
plt.title('K values vs Accuracy')
plt.xlabel('K value')
plt.ylabel('Accuracy')
plt.grid(True)

# 2. Scatter plot (sepal length vs sepal width)
plt.subplot(2, 2, 2)
for i, target_name in enumerate(target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], label=target_name)
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('Sepal Length vs Sepal Width')
plt.legend()

# 3. Scatter plot (petal length vs petal width)
plt.subplot(2, 2, 3)
for i, target_name in enumerate(target_names):
    plt.scatter(X[y == i, 2], X[y == i, 3], label=target_name)
plt.xlabel(feature_names[2])
plt.ylabel(feature_names[3])
plt.title('Petal Length vs Petal Width')
plt.legend()

# 4. Prediction results comparison
plt.subplot(2, 2, 4)
correct = y_pred == y_test
plt.scatter(range(len(y_test)), y_test, c='blue', marker='o', label='Actual')
plt.scatter(range(len(y_pred)), y_pred, c='red', marker='x', label='Predicted')
# Highlight incorrect predictions
for i in range(len(y_test)):
    if not correct[i]:
        plt.plot([i, i], [y_test[i], y_pred[i]], 'k-', alpha=0.3)
plt.title('Prediction Results')
plt.xlabel('Test Sample Index')
plt.ylabel('Class')
plt.yticks([0, 1, 2], target_names)
plt.legend()

plt.tight_layout()
plt.savefig('knn_iris_results.png')
plt.show()

# Confusion matrix visualization
cm = np.zeros((3, 3), dtype=int)
for i in range(len(y_test)):
    cm[y_test[i], y_pred[i]] += 1

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('knn_confusion_matrix.png')
plt.show()
