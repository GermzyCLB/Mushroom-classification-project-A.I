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


