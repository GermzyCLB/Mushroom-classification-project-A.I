"""
SVM Classifier Module
Main classifier with hyperparameter tuning and regularization.
"""

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, make_scorer
import numpy as np


def train_svm(X_train, y_train, X_val, y_val, X_test, y_test, use_grid_search=True):
    """
    Train SVM classifier with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
        use_grid_search: Whether to perform hyperparameter tuning
    
    Returns:
        Dictionary containing model, predictions, and metrics
    """
    print("\n" + "=" * 80)
    print("SVM CLASSIFIER (MAIN)")
    print("=" * 80)
    
    poison_f1_scorer = make_scorer(f1_score, pos_label='p')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    if use_grid_search:
        print("Performing hyperparameter tuning with RandomizedSearchCV...")
        
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'linear', 'poly'],
            'class_weight': ['balanced', None]
        }
        
        base_svm = SVC(random_state=42)
        svm_search = RandomizedSearchCV(
            base_svm,
            param_distributions=param_grid,
            n_iter=20,
            cv=cv,
            scoring=poison_f1_scorer,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        svm_search.fit(X_train, y_train.values.ravel())
        svm_model = svm_search.best_estimator_
        
        print(f"Best parameters: {svm_search.best_params_}")
        print(f"Best CV score: {svm_search.best_score_:.4f}")
    else:
        print("Training SVM with default parameters...")
        svm_model = SVC(class_weight='balanced', random_state=42, C=1.0, gamma='scale')
        svm_model.fit(X_train, y_train.values.ravel())
    
    # Validation set evaluation
    y_pred_val = svm_model.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred_val)
    val_f1 = f1_score(y_val, y_pred_val, pos_label='p')
    
    print(f"\nValidation Set Performance:")
    print(f"  Accuracy: {val_acc:.4f}")
    print(f"  Poisonous F1: {val_f1:.4f}")
    
    # Test set evaluation
    y_pred_test = svm_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, pos_label='p')
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Poisonous F1: {test_f1:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(
        svm_model, X_train, y_train.values.ravel(),
        cv=cv, scoring=poison_f1_scorer
    )
    
    print(f"\n5-Fold Cross-Validation:")
    print(f"  Mean F1 (Poisonous): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return {
        'name': 'SVM',
        'model': svm_model,
        'y_pred_test': y_pred_test,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'confusion_matrix': confusion_matrix(y_test, y_pred_test),
        'classification_report': classification_report(y_test, y_pred_test)
    }

