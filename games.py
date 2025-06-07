import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

dataset_path = "online_gaming_behavior_dataset.csv"
dataset = pd.read_csv(dataset_path)

X = dataset.iloc[:, :12]
y = dataset.iloc[:, 12]

# X = pd.get_dummies(X)

print('X:', X)


# if y.dtype == 'object' or isinstance(y[0], str):
#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(y)
#     print("Label encoding mapping:")
#     for i, class_label in enumerate(label_encoder.classes_):
#         print(f"{i} --> {class_label}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

classifiers = [
    SVC(gamma=2, C=1, random_state=42), GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    # DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(max_depth=15, n_estimators=200, max_features='sqrt', random_state=42, min_samples_leaf=1, min_samples_split=5),
    LogisticRegression(max_iter=2)
]

classifiers_names= ["SVM", "Decision Tree","Random Forest", "Logistic Regression"]

for classifier in classifiers:
    model = classifier
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    # y_proba = model.predict_proba(X_test)
    # roc = roc_auc_score(y_test, y_proba, average='weighted', multi_class='ovr')
    y_proba = model.predict_proba(X_test)

    # Compute per-class ROC AUC scores (one-vs-rest)
    roc_auc_per_class = roc_auc_score(y_test, y_proba, average=None, multi_class='ovr')

    # Print AUC for each class
    for i, auc_score in enumerate(roc_auc_per_class):
        print(f"AUC for class {i} ({label_encoder.classes_[i]}): {auc_score:.4f}")
    # print("Accuracy:", accuracy)
    # print("F1 Score:", f1)
    # print("Precision:", precision)
    # print("Recall:", recall)
    # print("ROC:", roc)
    print("Y Pred:", y_pred)
    print("Y Original:", y_test)


# model = RandomForestClassifier(max_depth=5, n_estimators=9, max_features=12, random_state=42)
# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_test)
#
# accuracy = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred, average='weighted')
# precision = precision_score(y_test, y_pred, average='weighted')
# recall = recall_score(y_test, y_pred, average='weighted')
#
# print("Accuracy:", accuracy)
# print("F1 Score:", f1)
# print("Precision:", precision)
# print("Recall:", recall)


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [9, 12, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

# rf = RandomForestClassifier(random_state=42)
#
# grid_search = GridSearchCV(
#     rf, param_grid=param_grid,
#     cv=5, scoring='accuracy', verbose=1, n_jobs=-1
# )
#
# grid_search.fit(X_train, y_train)
#
# print("Best parameters:", grid_search.best_params_)