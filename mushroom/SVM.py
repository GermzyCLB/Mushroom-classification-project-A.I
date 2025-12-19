import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)




# Fetch dataset 
mushroom = fetch_ucirepo(id=73) 


# Correct extraction
X = mushroom.data.features
y = mushroom.data.targets.squeeze()  # make it a Series


# 70 / 15 / 15 stratified split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

categorical_columns = X.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ]
)


svm_pipeline = Pipeline([
    ("onehot", preprocessor),
    ("scaler", StandardScaler(with_mean=False)),  # sparse-safe
    ("svm", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        random_state=42
    ))
])


svm_pipeline.fit(X_train, y_train)


y_pred = svm_pipeline.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")


precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["edible", "poisonous"]
))


cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["edible", "poisonous"]
)

disp.plot(cmap="Blues")
plt.title("SVM Confusion Matrix (Mushroom Dataset)")
plt.show()


labels = ["edible", "poisonous"]
x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(8,5))
plt.bar(x - width, precision, width, label="Precision")
plt.bar(x, recall, width, label="Recall")
plt.bar(x + width, f1, width, label="F1-score")

plt.xticks(x, labels)
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("SVM Performance per Class")
plt.legend()
plt.grid(axis="y")
plt.show()


# Shuffle-Label Baseline

# Shuffle training labels
rng = np.random.default_rng(42)
y_train_shuffled = rng.permutation(y_train)

# Refit the SAME pipeline on shuffled labels
svm_pipeline.fit(X_train, y_train_shuffled)

# Predict on the REAL test set
y_pred_shuffled = svm_pipeline.predict(X_test)

# Metrics
shuffle_accuracy = accuracy_score(y_test, y_pred_shuffled)
shuffle_precision = precision_score(y_test, y_pred_shuffled, average=None)
shuffle_recall = recall_score(y_test, y_pred_shuffled, average=None)
shuffle_f1 = f1_score(y_test, y_pred_shuffled, average=None)

print("\n=== Shuffle-Label Baseline Results ===")
print(f"Accuracy: {shuffle_accuracy:.3f}")

print("\nClassification Report (Shuffled Labels):")
print(classification_report(
    y_test,
    y_pred_shuffled,
    target_names=["edible", "poisonous"]
))

cm_shuffle = confusion_matrix(y_test, y_pred_shuffled)

disp_shuffle = ConfusionMatrixDisplay(
    confusion_matrix=cm_shuffle,
    display_labels=["edible", "poisonous"]
)

disp_shuffle.plot(cmap="Reds")
plt.title("SVM Confusion Matrix (Shuffle-Label Baseline)")
plt.show()


labels = ["edible", "poisonous"]
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, recall, width, label="Real SVM")
plt.bar(x + width/2, shuffle_recall, width, label="Shuffle Baseline")

plt.xticks(x, labels)
plt.ylabel("Recall")
plt.ylim(0, 1)
plt.title("Recall Comparison: Real SVM vs Shuffle-Label Baseline")
plt.legend()
plt.grid(axis="y")
plt.show()

