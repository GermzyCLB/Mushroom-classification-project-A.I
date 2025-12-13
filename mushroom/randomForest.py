import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Loading dataset
# Dataset Link: https://archive.ics.uci.edu/dataset/73/mushroom
 
# fetch dataset
mushroom = fetch_ucirepo(id=73)
 
# data (as pandas dataframes) 
X = mushroom.data.features 
y = mushroom.data.targets 
  
# metadata 
print(mushroom.metadata) 
  
# variable information 
print(mushroom.variables) 

# One-hpt encoding (missing values treated as category)
X_encoded = pd.get_dummies(X, drop_first=False)




# Train/Validate/Test split (70/15/15) on dataset

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_encoded, y,
    test_size = 0.15,
    stratify = y,
    random_state=42
)

val_size = 0.15
remaining = 1 - 0.15
val_fraction = val_size / remaining

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size = val_fraction,
    stratify=y_train_val,
    random_state=42
)




# Majority-class baseline

majority_class = y_train.mode()[0]

y_test_baseline = [majority_class] * len(y_test)

print("Majority baseline accuracy:", accuracy_score(y_test, y_test_baseline))




# Hyperparameter grid

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 'log2']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)




# Grid Search

rf = RandomForestClassifier(
    random_state=42,
    class_weight='balanced'
)

grid = GridSearchCV(
    rf,
    param_grid,
    scoring='f1_macro',
    cv=cv,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\nBest parameters (REAL)", grid.best_params_)
print("Best CV F1-macro (REAL):, grid.best_score_")




# Validation performance
best_rf = grid.best_estimator_
y_val_pred = best_rf.predict(X_val)

print("\nValidation results:")
print(classification_report(y_val, y_val_pred)
