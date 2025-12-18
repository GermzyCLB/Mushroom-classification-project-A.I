"""
Random Forest Classifier Module
Alternative classifier with regularization to prevent overfitting.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, make_scorer


def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train Random Forest classifier with regularization parameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary containing model, predictions, and metrics
    """
    print("\n" + "=" * 80)
    print("RANDOM FOREST CLASSIFIER")
    print("=" * 80)
    
    print("Training Random Forest with regularization parameters...")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train.values.ravel())
    
    # Validation set evaluation
    y_pred_val = rf_model.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred_val)
    val_f1 = f1_score(y_val, y_pred_val, pos_label='p')
    
    print(f"\nValidation Set Performance:")
    print(f"  Accuracy: {val_acc:.4f}")
    print(f"  Poisonous F1: {val_f1:.4f}")
    
    # Test set evaluation
    y_pred_test = rf_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, pos_label='p')
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Poisonous F1: {test_f1:.4f}")
    
    # Cross-validation
    poison_f1_scorer = make_scorer(f1_score, pos_label='p')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        rf_model, X_train, y_train.values.ravel(),
        cv=cv, scoring=poison_f1_scorer
    )
    
    print(f"\n5-Fold Cross-Validation:")
    print(f"  Mean F1 (Poisonous): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return {
        'name': 'Random Forest',
        'model': rf_model,
        'y_pred_test': y_pred_test,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'confusion_matrix': confusion_matrix(y_test, y_pred_test),
        'classification_report': classification_report(y_test, y_pred_test)
    }

