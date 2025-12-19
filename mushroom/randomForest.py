import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import make_scorer, recall_score, f1_score
from sklearn.metrics import precision_score, accuracy_score, roc_auc_score

# Loading dataset
# Dataset Link: https://archive.ics.uci.edu/dataset/73/mushroom
 
# fetch dataset
mushroom = fetch_ucirepo(id=73)
 
# data (as pandas dataframes) 
X = mushroom.data.features 
y = mushroom.data.targets
if isinstance(y, pd.DataFrame):
    if y.shape[1] == 1:
        y = y.iloc[:, 0]
    else:
        raise ValueError(f"Unexpected target DataFrame shaoe: {y.shape}")
        
# Debug: inspect X and y to avoid indexing surprises
print("X type", type(X), "shape:", getattr(X, "shape", None))
print("y type:", type(y), "shape:", getattr(y, "shape", None))
print("y sample values (value_counts):\n", getattr(y, "value_counts", lambda: None)())
print("X sample columns", list(X.columns[:10]))
print("First row of X:\n", X.head(1))

# metadata 
print(mushroom.metadata) 
  
# variable information 
print(mushroom.variables) 

# One-hot encoding (missing values treated as category)
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
print("Best CV F1-macro (REAL):", grid.best_score_)




# Validation performance

best_rf = grid.best_estimator_
y_val_pred = best_rf.predict(X_val)

print("\nValidation results:")
print(classification_report(y_val, y_val_pred))




# Retrain on Train and Validation

final_rf = RandomForestClassifier(
    **grid.best_params_,
    random_state=42,
    class_weight='balanced'
)

final_rf.fit(X_train_val, y_train_val)




# Test performance

y_test_pred = final_rf.predict(X_test)

print("\nTest results:")
print("Accuracy", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

# Confusion Matrix Implementation

cm = confusion_matrix(y_test, y_test_pred, labels=['e', 'p'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['edible', 'poisonous'])

disp.plot()
plt.title("Random Forest - Test Set")
plt.show()
     
os.makedirs("results", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

def evaluate_on_sets(model, X_val, y_val, X_test, y_test):
    yv = model.predict(X_val)
    yt = model.predict(X_test)
    try:
        probs_test = model.predict_proba(X_test)[:,1]
    except Exception:
        probs_test = None
    metrics_val = {
        "accuracy_val": accuracy_score(y_val, yv),
        "precision_val": precision_score(y_val, yv, pos_label='p'),
        "recall_val": recall_score(y_val, yv, pos_label='p'),
        "f1_val": f1_score(y_val, yv, pos_label='p'),
    }
    metrics_test = {
        "accuracy_test": accuracy_score(y_test, yt),
        "precision_test": precision_score(y_test, yt, pos_label='p'),
        "recall_test": recall_score(y_test, yt, pos_label='p'),
        "f1_test": f1_score(y_test, yt, pos_label='p'),
    }
    if probs_test is not None:
        try: 
            metrics_test["roc_auc_test"] = roc_auc_score((y_test == 'p').astype(int), probs_test)
        except Exception:
            metrics_test["roc_auc_test"] = None
    else:
        metrics_test["roc_auc_test"] = None
    return {**metrics_val, **metrics_test}

# Define parameter sets
grid_small = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ['sqrt', 'log2']
}

grid_wider = {
    'n_estimators': [50, 100, 200, 400],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 'log2', 0.5],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}

rand_params = {
    'n_estimators': [50, 100, 200, 400, 800],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.3, 0.5],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}

grid_expanded = {
    'n_estimators': [100, 200, 400, 800],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.3, 0.5],
    'criterion':['gini', 'entropy'],
    'bootstrap': [True, False],
    'max_samples': [None, 0.5, 0.8] # subsampling at tree level
}

refined_params = {
    'n_estimators': [400, 600, 800],
    'max_depth': [None, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 0.5],
    'bootstrap': [True],
    'max_samples': [0.7, 0.9]
}

# scorers
recall_p_scorer = make_scorer(recall_score, pos_label='p')
f1_macro_scorer = 'f1_macro'

experiments = []

def run_grid_search(name, param_grid, scoring, use_random=False, n_iter=30):
    start = time.time()
    if use_random:
        search = RandomizedSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                                    param_distributions=param_grid,
                                    n_iter=n_iter,
                                    scoring=scoring,
                                    cv=cv,
                                    n_jobs=-1,
                                    random_state=42,
                                    verbose=1)
    else:
        search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                              param_grid=param_grid,
                              scoring=scoring,
                              cv=cv,
                              n_jobs=-1,
                              verbose=1)
    search.fit(X_train, y_train)
    duration = time.time() - start
    best = search.best_estimator_
    res = evaluate_on_sets(best, X_val, y_val, X_test, y_test)
    row = {
        "experiment": name,
        "scoring": scoring if isinstance(scoring, str) else "recall_p",
        "best_params": json.dumps(search.best_params_),
        "cv_best_score": float(search.best_score_),
        "duration_s": duration,
        **res
    }
    experiments.append(row)
    # Save intermediate CSV so you don't lose results
    pd.DataFrame(experiments).to_csv("results/ rf_experiments.csv", index=False)
    return search

# 1. Grid (small) optimizing f1_macro
print("Running grid_small optimizing f1_macro...")
g1 = run_grid_search("grid_small_f1", grid_small, f1_macro_scorer, use_random=False)

# 2. Grid (small) optimizing poisonous recall
print("Running grid_small optimizing recall_p...")
g2 = run_grid_search("grid_small_recall", grid_small, recall_p_scorer, use_random=False)

# 3. Randomized (wider) optimizing f1_macro (fast)
print("Running randomized wide optimizing f1_macro...")
r1 = run_grid_search("rand_wide_f1", rand_params, f1_macro_scorer, use_random=True, n_iter=40)

# 4. Randomized (wider) optimizing recall_p
print("Running randomized wide optimizing recall_p...")
r2 = run_grid_search("rand_wide_recall", rand_params, recall_p_scorer, use_random=True, n_iter=40)

# 5. Randomized (expanded) optimizing recall_p (strong RF candidate)
print("Running randomized expanded grid optimizing recall_p...")
r3 = run_grid_search(
    "rand_expanded_recall",
    grid_expanded,
    recall_p_scorer,
    use_random=True,
    n_iter=60
)

# 6. Final refined RF search (fine-grained, limited scope)
print("Running final refined RF search...")
r4 = run_grid_search(
    "rand_refined_recall",
    refined_params,
    recall_p_scorer,
    use_random=True,
    n_iter=25
)

# 7. Shuffle-label baseline: shuffle training labels and run small grid for sanity check
print("Running shuffle-label baseline (sanity check)...")
y_train_shuffled = y_train.sample(frac=1.0, random_state=42)
# Need X_train indices aligned: reset X_train index to match
X_train_shuffled = X_train.reset_index(drop=True)

search_shuffled = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                               param_grid=grid_small, scoring=f1_macro_scorer,
                               cv=cv,
                               n_jobs=-1)
search_shuffled.fit(X_train_shuffled, y_train_shuffled)
best_shuffled = search_shuffled.best_estimator_
res_shuffled = evaluate_on_sets(best_shuffled, X_val, y_val, X_test, y_test)

# Confusion matrix for shuffled-label baseline
y_test_pred_shuffled = best_shuffled.predict(X_test)

cm_shuffled = confusion_matrix(y_test, y_test_pred_shuffled, labels=['e', 'p'])
disp_shuffled = ConfusionMatrixDisplay(
    confusion_matrix=cm_shuffled,
    display_labels=['edible', 'poisonous']
)

disp_shuffled.plot()
plt.title("Random Forest (Shuffled Labels) - Test Set")
plt.tight_layout()
plt.show()


experiments.append({
    "experiment": "shuffle_label",
    "scoring": f1_macro_scorer,
    "best_params": json.dumps(search_shuffled.best_params_),
    "cv_best_score": float(search_shuffled.best_score_),
    "duration_s": None,
    **res_shuffled
})
pd.DataFrame(experiments).to_csv("results/rf_experiments.csv", index=False)

# 8. Additional shuffle-label baselines with different random seeds
print("Running multiple shuffle-label baselines...")

for seed in [1, 7, 21, 99]:
    y_train_shuffled_multi = y_train.sample(frac=1.0, random_state=seed)
    X_train_shuffled_multi = X_train.reset_index(drop=True)
    
    search_shuffled_multi = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        param_grid=grid_small,
        scoring=f1_macro_scorer,
        cv=cv,
        n_jobs=-1
    )
    
    search_shuffled_multi.fit(X_train_shuffled_multi, y_train_shuffled_multi)
    
    best_shuffled_multi = search_shuffled_multi.best_estimator_
    res_multi = evaluate_on_sets(
        best_shuffled_multi, X_val, y_val, X_test, y_test
    )
    
    experiments.append({
        "experiment": f"shuffle_label_seed_{seed}",
        "scoring": f1_macro_scorer,
        "best_params": json.dumps(search_shuffled_multi.best_params_),
        "cv_best_score": float(search_shuffled_multi.best_score_),
        "duration_s": None,
        ** res_multi
    })
pd.DataFrame(experiments).to_csv("results/rf_experiments.csv", index=False)

# 9. Extra shuffle-label seeds (final sanity check)
print("Running extra shuffle-label sanity checks...")

extra_shuffle_seeds = [2, 11, 33, 123]

for seed in extra_shuffle_seeds:
    y_train_shuffled_extra = y_train.sample(frac=1.0, random_state=seed)
    X_train_shuffled_extra = X_train.reset_index(drop=True)
    
    search_extra = GridSearchCV(
        RandomForestClassifier(random_state=seed, class_weight='balanced'),
        param_grid=grid_small,
        scoring=f1_macro_scorer,
        cv=cv,
        n_jobs=-1
    )
    
    search_extra.fit(X_train_shuffled_extra, y_train_shuffled_extra)
    
    best_extra = search_extra.best_estimator_
    res_extra = evaluate_on_sets(
        best_extra, X_val, y_val, X_test, y_test
    )
    
    experiments.append({
        "experiment": f"shuffle_label_seed_{seed}",
        "scoring": f1_macro_scorer,
        "best_params": json.dumps(search_shuffled_multi.best_params_),
        "cv_best_score": float(search_shuffled_multi.best_score_),
        "duration_s": None,
        ** res_extra
    })

pd.DataFrame(experiments).to_csv("results/rf_experiments.csv", index=False)

# Final printout
df_results = pd.DataFrame(experiments)
print("\nALL EXPERIMENTS RESULT SUMMARY:")
print(df_results[["experiment", "cv_best_score", "accuracy_test", "recall_test", "f1_test"]].to_string(index = False))
print("\nSaved experiments to results/rf_experiments.csv")
print("Random Forest test accuracy", accuracy_score(y_test, y_test_pred))

print("\nFinal Random Forest chosen based on highest poisonous recall: ")
best_overall = max(
    experiments,
    key=lambda x: x.get("recall_test", 0)
)
print(best_overall)