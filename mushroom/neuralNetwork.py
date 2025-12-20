import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1) Load Mushroom Dataset

mushroom = fetch_ucirepo(id=73)

X = mushroom.data.features
y = mushroom.data.targets

# Convert target to Series if needed
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

print("Dataset loaded")
print("X shape:", X.shape)
print("y distribution:\n", y.value_counts())

# 2) One-hot encoding

X_encoded = pd.get_dummies(X, drop_first=False)

# 3) Train / Validation / Test Split (70 / 15 / 15)

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.15,
    stratify=y,
    random_state=42
)

val_fraction = 0.15 / (1 - 0.15)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=val_fraction,
    stratify=y_train_val,
    random_state=42
)

# 4) Majority-class baseline
majority_class = y_train.mode()[0]
y_test_baseline = [majority_class] * len(y_test)

baseline_accuracy = accuracy_score(y_test, y_test_baseline)
print("\nMajority baseline accuracy: ", baseline_accuracy)

# 5) Feature Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 6) Neural Network (MLP classifier)

mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

mlp.fit(X_train_scaled, y_train)

# 7) Validation Performance

y_val_pred = mlp.predict(X_val_scaled)

print("\nValidation Results: ")
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

# 8) Retrain on Train + Validation

X_train_val_scaled = scaler.fit_transform(X_train_val)

final_mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

final_mlp.fit(X_train_val_scaled, y_train_val)

# 9) Test Performance

X_test_scaled = scaler.transform(X_test)
y_test_pred = final_mlp.predict(X_test_scaled)

test_accuracy = accuracy_score(y_test, y_test_pred)

print("\n Test Results: ")
print("Test Accuracy:", test_accuracy)
print(classification_report(y_test, y_test_pred))

# 10) Confusion Matrix

cm = confusion_matrix(y_test, y_test_pred, labels=['e', 'p'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['edible', 'poisonous'])

disp.plot()
plt.title("Neural Network (MLP) - Test Set")
plt.show()

print("\nNeural Network test accuracy:", test_accuracy)
print("Baseline accuracy:", baseline_accuracy)

# 11) Shuffled-Label Baseline

# Shuffle the training labels
y_train_val_shuffled = y_train_val.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Fit the neural network on shuffled labels
mlp_shuffled = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

mlp_shuffled.fit(X_train_val_scaled, y_train_val_shuffled)

# Predict on test set
y_test_shuffled_pred = mlp_shuffled.predict(X_test_scaled)

# Accuracy and classification report
shuffled_accuracy = accuracy_score(y_test, y_test_shuffled_pred)
print("\nShuffled-Label Test Results: ")
print("Test Accuracy (Shuffled Labels):", shuffled_accuracy)
print(classification_report(y_test, y_test_shuffled_pred))

# Confusion Matrix
cm_shuffled = confusion_matrix(y_test, y_test_shuffled_pred, labels=['e', 'p'])
disp_shuffled = ConfusionMatrixDisplay(cm_shuffled, display_labels=['edible', 'poisonous'])
disp_shuffled.plot()
plt.title("Neural Network (MLP) - Test Set with Shuffled Labels")
plt.show()