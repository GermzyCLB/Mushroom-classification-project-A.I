"""
Main Runner Script
Orchestrates all classifiers and reports comprehensive metrics.
"""

import data_loader
import baseline_models
import svm_classifier
import random_forest_classifier
import logistic_regression_classifier
import gradient_boosting_classifier
import naive_bayes_classifier


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_model_comparison(all_results):
    """Print a comprehensive comparison of all models."""
    print_section_header("MODEL COMPARISON SUMMARY")
    
    print("\nPerformance Metrics:")
    print(f"{'Model':<30} {'Test Accuracy':<18} {'Test F1 (Poisonous)':<22} {'CV F1 Mean':<15} {'CV F1 Std':<15}")
    print("-" * 100)
    
    for result in all_results:
        name = result['name']
        test_acc = result['test_accuracy']
        test_f1 = result['test_f1']
        cv_mean = result['cv_mean']
        cv_std = result['cv_std']
        
        print(f"{name:<30} {test_acc:<18.4f} {test_f1:<22.4f} {cv_mean:<15.4f} {cv_std:<15.4f}")
    
    print("\n" + "-" * 100)
    print("Note: SVM is the main classifier. Other models are provided for comparison.")


def print_detailed_results(all_results):
    """Print detailed results for each model."""
    print_section_header("DETAILED RESULTS")
    
    for result in all_results:
        print(f"\n{result['name']} - Detailed Results:")
        print(f"Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"Test F1 (Poisonous): {result['test_f1']:.4f}")
        print(f"5-Fold CV Mean F1: {result['cv_mean']:.4f} (+/- {result['cv_std'] * 2:.4f})")
        print(f"Confusion Matrix:\n{result['confusion_matrix']}")
        print(f"Classification Report:\n{result['classification_report']}")


def main():
    """Main execution function."""
    print_section_header("MUSHROOM CLASSIFICATION PIPELINE")
    print("Main Classifier: SVM")
    print("Alternative Classifiers: Random Forest, Logistic Regression, Gradient Boosting, Naive Bayes")
    
    # Load and prepare data
    print_section_header("DATA PREPARATION")
    data = data_loader.prepare_data()
    
    # Extract data splits
    X_train_dense = data['X_train_dense']
    X_val_dense = data['X_val_dense']
    X_test_dense = data['X_test_dense']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    # Evaluate baseline models
    baseline_models.evaluate_baseline_models(
        X_train_dense, y_train, X_test_dense, y_test
    )
    
    # Train all classifiers
    all_results = []
    
    # Main classifier: SVM
    svm_result = svm_classifier.train_svm(
        X_train_dense, y_train,
        X_val_dense, y_val,
        X_test_dense, y_test,
        use_grid_search=True
    )
    all_results.append(svm_result)
    
    # Alternative classifiers
    rf_result = random_forest_classifier.train_random_forest(
        X_train_dense, y_train,
        X_val_dense, y_val,
        X_test_dense, y_test
    )
    all_results.append(rf_result)
    
    lr_result = logistic_regression_classifier.train_logistic_regression(
        X_train_dense, y_train,
        X_val_dense, y_val,
        X_test_dense, y_test
    )
    all_results.append(lr_result)
    
    gb_result = gradient_boosting_classifier.train_gradient_boosting(
        X_train_dense, y_train,
        X_val_dense, y_val,
        X_test_dense, y_test
    )
    all_results.append(gb_result)
    
    nb_result = naive_bayes_classifier.train_naive_bayes(
        X_train_dense, y_train,
        X_val_dense, y_val,
        X_test_dense, y_test
    )
    all_results.append(nb_result)
    
    # Print comparison and detailed results
    print_model_comparison(all_results)
    print_detailed_results(all_results)
    
    print_section_header("PIPELINE COMPLETE")
    
    return all_results


if __name__ == "__main__":
    results = main()

