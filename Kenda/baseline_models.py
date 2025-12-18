"""
Baseline Models Module
Implements baseline classifiers for comparison.
"""

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate_baseline_models(X_train, y_train, X_test, y_test):
    """
    Evaluate baseline models for comparison.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
    """
    print("\n" + "=" * 80)
    print("BASELINE MODELS")
    print("=" * 80)
    
    def evaluate_model(model, X_train, y_train, X_test, y_test, name):
        """Helper function to evaluate a baseline model."""
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"\n{name}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        
        return acc
    
    # Majority baseline
    majority = DummyClassifier(strategy="most_frequent")
    evaluate_model(majority, X_train, y_train, X_test, y_test, "Majority Baseline")
    
    # Random baseline
    random = DummyClassifier(strategy="stratified", random_state=42)
    evaluate_model(random, X_train, y_train, X_test, y_test, "Random Baseline")
    
    print("\nBaseline evaluation complete.")

