from sklearn.datasets import load_iris
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

import matplotlib.pyplot as plt
from sklearn import tree

# 加载数据
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 简单查看
df = pd.DataFrame(X, columns=feature_names)
df['label'] = y
print(df.head())
print("类别分布：\n", df['label'].value_counts())


# 1. 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# 2. 设置参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'max_features': ['auto', 'sqrt', 0.5]
}
# 3. GridSearchCV
rfc = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rfc,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
print("最佳参数：", grid_search.best_params_)
print("最佳交叉验证得分：{:.4f}".format(grid_search.best_score_))
# 4. 用最佳参数评估测试集
best_rfc = grid_search.best_estimator_
y_pred = best_rfc.predict(X_test)
print("测试集准确率：{:.4f}".format(accuracy_score(y_test, y_pred)))
print("详细报告：\n", classification_report(y_test, y_pred, target_names=target_names))



# 从已训练好的随机森林中抽取三棵树
estimators = best_rfc.estimators_  # list of DecisionTreeClassifier
# 可视化前 3 棵
for idx in range(3):
    plt.figure(figsize=(12, 8))
    tree.plot_tree(
        estimators[idx],
        feature_names=feature_names,
        class_names=target_names,
        filled=True,
        rounded=True,
        impurity=True,
        fontsize=10
    )
    plt.title(f"Random Forest Tree #{idx+1}")
    plt.show()
